//! Page-locked (pinned) host memory for fast device → host transfers.
//!
//! `cudarc` 0.19's `CudaContext::alloc_pinned` always passes
//! `CU_MEMHOSTALLOC_WRITECOMBINED`, which is the **wrong** flag for our
//! result-download path: write-combining bypasses the host CPU cache,
//! which speeds up host writes (good for upload buffers) at the price of
//! making host reads dramatically slower. Our use of pinned memory is
//! the opposite — the device writes the result, the host reads it — so
//! we allocate with the *default* flags (`= 0`) instead, giving us a
//! cacheable page-locked buffer that DMAs at full PCIe bandwidth and
//! also reads back at normal host speed.
//!
//! Switching the result download from a pageable `Vec<u32>` to a default
//! pinned buffer roughly halves the D2H transfer time on the headline
//! `Q = 64, N = 1 M, dim = 768` shape (≈ 256 MB result matrix), which
//! is the single biggest cost in the cached path's wall-clock — see
//! `docs/ADR-001-cuda-backend.md` §Consequences for the breakdown.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{result as drv_result, CudaContext};

use crate::error::{GpuError, GpuResult};

/// Owned page-locked host buffer of `u32`s, allocated with default
/// (cacheable) `cuMemHostAlloc` flags. Suitable as a target for
/// `memcpy_dtoh_async`.
///
/// Holds an `Arc<CudaContext>` so the driver context outlives the
/// allocation.
pub struct PinnedU32Buffer {
    ptr: *mut u32,
    len: usize,
    // Keep the context alive for at least as long as the allocation.
    _ctx: Arc<CudaContext>,
}

// SAFETY: `Send` and `Sync` on a buffer holding a raw `*mut u32` is only
// sound if the type's invariants enforce that nothing else can race with
// the pointer. The conditions justifying the impls below are:
//
// 1. **Address stability.** Memory returned by `cuMemHostAlloc` lives at a process-global virtual
//    address for the lifetime of the buffer; the pointer is never reallocated or remapped, so
//    transferring it across threads carries no aliasing hazard from CUDA's side.
//
// 2. **Context lifetime.** The `Arc<CudaContext>` field keeps the driver context alive at least as
//    long as this buffer, so `Drop` (which calls `cuMemFreeHost` against that context) cannot
//    observe a dangling context after the originating thread terminates.
//
// 3. **Caller-enforced exclusive access during writes.** The producer of the buffer (a
//    `memcpy_dtoh_async` into `as_mut_ptr()` or any kernel that writes through this region) must
//    run to completion — i.e. the stream that wrote into the buffer must be synchronised — *before*
//    any thread accesses `as_slice()` or moves ownership. This is the same precondition that
//    `as_slice` already documents.
//
// 4. **Single-owner Drop.** Rust's borrow checker prevents `Drop::drop` from running while another
//    thread holds `&PinnedU32Buffer` or `&mut PinnedU32Buffer`, because `Drop` takes `&mut self`.
//    There is therefore no UAF risk from `cuMemFreeHost` racing with concurrent `as_slice()` /
//    `as_mut_ptr()` on the same value.
//
// If these conditions are ever weakened (e.g. exposing an interior-mutable
// view that lets two threads write through `as_mut_ptr()` simultaneously,
// or freeing the context out from under the buffer), revisit Send / Sync.
unsafe impl Send for PinnedU32Buffer {}
unsafe impl Sync for PinnedU32Buffer {}

impl PinnedU32Buffer {
    /// Allocate `len` `u32`s of cacheable pinned host memory.
    ///
    /// `len == 0` short-circuits to a safe empty buffer that does **not**
    /// invoke `cuMemHostAlloc(0)`: per the CUDA Driver API spec a zero-byte
    /// request is implementation-defined and may return `NULL`, which the
    /// `debug_assert!(!raw.is_null())` below would catch only in debug
    /// builds. In release the `NULL` would propagate into `from_raw_parts`
    /// / `memcpy_dtoh_async` and trigger UB.
    pub fn new(ctx: &Arc<CudaContext>, len: usize) -> GpuResult<Self> {
        ctx.bind_to_thread().map_err(map_drv)?;
        if len == 0 {
            // `NonNull::<u32>::dangling()` is aligned and non-null, so
            // `as_slice()` (which calls `from_raw_parts(ptr, 0)`) is sound
            // and `Drop` (which is gated on `len > 0`) skips `cuMemFreeHost`.
            return Ok(Self {
                ptr: std::ptr::NonNull::<u32>::dangling().as_ptr(),
                len: 0,
                _ctx: Arc::clone(ctx),
            });
        }
        let num_bytes = len.saturating_mul(std::mem::size_of::<u32>());
        // Flags = 0 → CU_MEMHOSTALLOC_DEFAULT: portable across contexts,
        // cacheable on the host, no device map. This is the right
        // setting for a buffer the host will read back.
        let raw = unsafe { drv_result::malloc_host(num_bytes, 0) }
            .map_err(|e| map_alloc_err("cuMemHostAlloc", num_bytes, e))?;
        debug_assert!(!raw.is_null());
        debug_assert!((raw as usize) % std::mem::align_of::<u32>() == 0);
        Ok(Self {
            ptr: raw as *mut u32,
            len,
            _ctx: Arc::clone(ctx),
        })
    }

    /// Number of `u32`s.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this buffer has zero capacity.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw mutable host pointer for use as a `memcpy_dtoh_async`
    /// destination. The pointer remains valid for the lifetime of
    /// `self`.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u32 {
        self.ptr
    }

    /// Read view of the host data. The caller is responsible for
    /// having synchronised the stream that wrote into this buffer
    /// before calling.
    pub fn as_slice(&self) -> &[u32] {
        // SAFETY: the allocation is `len * sizeof::<u32>()` bytes,
        // properly aligned, and the caller has externally ensured no
        // concurrent in-flight writes (we synchronise the stream
        // before returning the buffer to user code).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for PinnedU32Buffer {
    fn drop(&mut self) {
        // Best-effort: cuMemFreeHost should always succeed for a
        // pointer we got from cuMemHostAlloc; a Drop-time error is
        // benign at process scope. Skip the call entirely when `len == 0`
        // because that branch in `new` never invoked the driver — the
        // pointer is `NonNull::dangling()`, not an allocation.
        if self.len > 0 && !self.ptr.is_null() {
            // SAFETY: ptr came from `cuMemHostAlloc` in `new`. Single-owner
            // Drop, no aliasing &self in flight (enforced by Rust).
            if let Err(e) = unsafe { drv_result::free_host(self.ptr as *mut c_void) } {
                log::warn!("cuMemFreeHost({:p}) failed: {e}", self.ptr);
            }
            self.ptr = std::ptr::null_mut();
            self.len = 0;
        }
    }
}

fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::CpuFallback {
        reason: format!("CUDA driver error: {e}"),
    }
}

/// Classify an allocation-site driver error. `CUDA_ERROR_OUT_OF_MEMORY`
/// is surfaced as [`GpuError::OutOfMemory`] so the caller can release
/// working set or reject the query; all other failures (init, no
/// device, denied permission, …) fall back to CPU.
fn map_alloc_err(op: &'static str, num_bytes: usize, e: cudarc::driver::DriverError) -> GpuError {
    use cudarc::driver::sys::CUresult;
    if e.0 == CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        GpuError::OutOfMemory {
            reason: format!("{op}({num_bytes} bytes): {e}"),
        }
    } else {
        GpuError::CpuFallback {
            reason: format!("{op}({num_bytes} bytes) failed: {e}"),
        }
    }
}

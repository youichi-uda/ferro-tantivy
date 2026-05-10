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
/// allocation. `Send`/`Sync` are safe because the underlying memory is
/// fixed at a process-global virtual address and cuda's host
/// allocations are documented as thread-safe to read/write from any
/// thread that has the same context bound.
pub struct PinnedU32Buffer {
    ptr: *mut u32,
    len: usize,
    // Keep the context alive for at least as long as the allocation.
    _ctx: Arc<CudaContext>,
}

unsafe impl Send for PinnedU32Buffer {}
unsafe impl Sync for PinnedU32Buffer {}

impl PinnedU32Buffer {
    /// Allocate `len` `u32`s of cacheable pinned host memory.
    pub fn new(ctx: &Arc<CudaContext>, len: usize) -> GpuResult<Self> {
        ctx.bind_to_thread().map_err(map_drv)?;
        let num_bytes = len.saturating_mul(std::mem::size_of::<u32>());
        // Flags = 0 → CU_MEMHOSTALLOC_DEFAULT: portable across contexts,
        // cacheable on the host, no device map. This is the right
        // setting for a buffer the host will read back.
        let raw =
            unsafe { drv_result::malloc_host(num_bytes, 0) }.map_err(|e| GpuError::CpuFallback {
                reason: format!("cuMemHostAlloc({num_bytes} bytes, default flags) failed: {e}"),
            })?;
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
        // benign at process scope.
        if !self.ptr.is_null() {
            let _ = unsafe { drv_result::free_host(self.ptr as *mut c_void) };
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

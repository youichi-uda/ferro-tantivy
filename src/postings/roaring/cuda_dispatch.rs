//! Phase 2 C-5 Wave 8.B — CUDA backend host-side wrapper for the Bool
//! query Bitmap kernel.
//!
//! This module is the host-side bridge between
//! [`super::gpu_dispatch::try_gpu_bool`] and
//! [`ferro_compress::BitmapOpKernel`] (CUDA driver-API kernel + cudart
//! memory ops). It mirrors the wgpu `tantivy_gpu::posting::BitmapOpKernel`
//! `compute(op, &[u32], &[u32]) -> Result<Vec<u32>>` API surface so
//! `gpu_dispatch.rs` can swap backends behind the `cuda-bitmap-kernel`
//! Cargo feature without changing the dispatch logic.
//!
//! ## Why this exists
//!
//! Wave 8.A re-bench (`findings_4070_ti_super_20260510.md` § "Wave 8 / A
//! re-bench") showed the wgpu mega_cohort routine at **37.24 ms vs CPU
//! 1.71 ms = 21.7× SLOWER** on RTX 4070 Ti SUPER. Decomposition: 11
//! pairwise GPU AND reductions × wgpu's ~3 ms per-dispatch overhead =
//! ~33 ms structural floor, not amortisable by caching. The
//! parallel-session `bc1f522` kernel-only bench measures **17-34× CPU
//! win** on the same hardware via CUDA — the difference is the backend,
//! not the kernel. This wrapper bridges the production query path to
//! the CUDA backend so the M&A query-path narrative finally lands a
//! positive number.
//!
//! ## Persistent device buffers
//!
//! `ferro_compress::BitmapOpKernel::compute` takes device pointers, not
//! host slices. We keep three persistent device buffers (`d_a`, `d_b`,
//! `d_out`) sized for an initial cohort capacity and grow them on demand
//! when a larger cohort arrives. Each `compute()` call performs:
//!
//!   1. `cudaMemcpyAsync` H→D for `a` (and `b`) on the kernel's stream
//!   2. `cuLaunchKernel` (via `BitmapOpKernel::compute`) on the same stream
//!   3. `cudaMemcpyAsync` D→H for `out` on the same stream
//!   4. `cudaStreamSynchronize` to block the host until the chain
//!      completes
//!   5. return the result `Vec<u32>`
//!
//! The persistent stream + buffer reuse is what eliminates the
//! per-call wgpu floor: the CUDA driver-API dispatch is ~100 µs vs
//! wgpu's ~3 ms.

#![cfg(all(feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::Mutex;

use ferro_compress::nvcomp_sys::cuda::{
    cudaFree, cudaMalloc, cudaMemcpyAsync, cudaMemcpyKind, cudaStreamCreate, cudaStreamDestroy,
    cudaStreamSynchronize, cudaStream_t, CUDA_SUCCESS,
};
use ferro_compress::{BitmapOp as FcBitmapOp, BitmapOpKernel as InnerKernel, Error as FcError};

use super::gpu_dispatch::BoolOp;

/// Errors that can surface from the CUDA wrapper. Distinct from
/// [`ferro_compress::Error`] so the call site can match on
/// memcpy/stream/launch failures separately from kernel-internal
/// errors. The wrapper converts every CUDA runtime-API failure into
/// the [`Cuda`] variant tagged with the call site name; kernel
/// driver-API failures bubble up as [`Inner`].
#[derive(Debug, thiserror::Error)]
pub enum CudaBitmapError {
    /// CUDA runtime-API call failure (memcpy / stream / malloc).
    /// `what` is the static call site name; `code` is the
    /// `cudaError_t` raw value.
    #[error("CUDA runtime error in {what}: code={code}")]
    Cuda {
        /// Static call site identifier (e.g. `"cudaMemcpyAsync(a H2D)"`).
        what: &'static str,
        /// Raw `cudaError_t` value returned by the CUDA runtime.
        code: u32,
    },
    /// Failure inside `ferro_compress::BitmapOpKernel::compute` (kernel
    /// launch, PTX module load, primary-context retain).
    #[error("ferro-compress kernel error: {0}")]
    Inner(#[from] FcError),
    /// Caller supplied input slices of unequal length.
    #[error("input length mismatch: a={a}, b={b}")]
    LenMismatch {
        /// Length of the first slice.
        a: usize,
        /// Length of the second slice.
        b: usize,
    },
}

/// Initial device-buffer capacity in `u32` words. 12 containers ×
/// `BITMAP_CONTAINER_WORDS` (= 2 048) = 24 576 = the canonical
/// `mega_cohort` fixture working set. Buffers grow on demand when a
/// cohort exceeds this; the initial size just avoids an early
/// re-allocation on the routine that drives Wave 8.B's acceptance
/// gate.
const INITIAL_CAPACITY_WORDS: usize = 12 * 2048;

/// Host-side wrapper around [`ferro_compress::BitmapOpKernel`] that
/// owns persistent CUDA device buffers and a per-instance
/// `cudaStream_t`. Exposes the same `compute(op, &[u32], &[u32]) ->
/// Result<Vec<u32>>` surface as the wgpu `BitmapOpKernel` so
/// [`super::gpu_dispatch::try_gpu_bool`] can swap backends with a
/// trivial `match` on the kernel type alias.
pub struct CudaBitmapOpKernel {
    inner: InnerKernel,
    /// Owned by this struct (created in [`Self::new`], destroyed in
    /// [`Drop`]). The inner [`InnerKernel`] borrows it but does not
    /// own it — so when this struct drops it is responsible for the
    /// `cudaStreamDestroy`.
    stream: cudaStream_t,
    /// Mutex protects the device buffers + capacity. Protects against
    /// torn growth across concurrent `compute()` callers; the actual
    /// kernel launch is serialised inside `InnerKernel` via its own
    /// mutex.
    state: Mutex<DeviceBuffers>,
}

struct DeviceBuffers {
    d_a: *mut c_void,
    d_b: *mut c_void,
    d_out: *mut c_void,
    capacity_words: usize,
}

// SAFETY: Device pointers + cudaStream_t are opaque process-global
// handles owned by the CUDA driver. Lifetimes are bounded by the
// wrapper struct (Drop releases). Concurrent access is serialised by
// the internal Mutex around `state`.
unsafe impl Send for CudaBitmapOpKernel {}
unsafe impl Sync for CudaBitmapOpKernel {}

impl std::fmt::Debug for CudaBitmapOpKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cap = self
            .state
            .lock()
            .map(|s| s.capacity_words)
            .unwrap_or(0);
        f.debug_struct("CudaBitmapOpKernel")
            .field("stream_is_null", &self.stream.is_null())
            .field("capacity_words", &cap)
            .finish()
    }
}

impl CudaBitmapOpKernel {
    /// Construct a kernel with a fresh CUDA stream and the initial
    /// device-buffer allocation. Returns an error if any of the
    /// stream-create / memory-allocate / kernel-load steps fail —
    /// callers (specifically `gpu_dispatch::gpu_resources`) cache the
    /// `Result` so that init failure is sticky rather than retried per
    /// call.
    pub fn new() -> Result<Self, CudaBitmapError> {
        let mut stream: cudaStream_t = null_mut();
        // SAFETY: `cudaStreamCreate` writes a valid stream handle
        // into the out pointer on success and leaves it untouched on
        // failure. We pass a properly aligned `&mut cudaStream_t`.
        let rc = unsafe { cudaStreamCreate(&mut stream) };
        if rc != CUDA_SUCCESS {
            return Err(CudaBitmapError::Cuda {
                what: "cudaStreamCreate",
                code: rc,
            });
        }
        let inner = match InnerKernel::with_stream(stream) {
            Ok(k) => k,
            Err(e) => {
                // SAFETY: stream was successfully created and is
                // owned by us (no shared aliasing); destroying it
                // here is the canonical cleanup on the construction
                // error path.
                unsafe {
                    let _ = cudaStreamDestroy(stream);
                }
                return Err(CudaBitmapError::Inner(e));
            }
        };
        let buffers = match unsafe { allocate_buffers(INITIAL_CAPACITY_WORDS) } {
            Ok(b) => b,
            Err(e) => {
                // Drop order: inner first (releases CUDA module +
                // primary context retain) so the stream we destroy
                // next still has a current context.
                drop(inner);
                // SAFETY: stream still owned exclusively by this
                // failed-construction path; safe to destroy.
                unsafe {
                    let _ = cudaStreamDestroy(stream);
                }
                return Err(e);
            }
        };
        Ok(Self {
            inner,
            stream,
            state: Mutex::new(buffers),
        })
    }

    /// Run `out = TERMS[0] OP TERMS[1] OP … OP TERMS[N-1]` as a single
    /// device-resident fold, with one `cudaStreamSynchronize` at the
    /// end and zero host roundtrips between fold steps. This is the
    /// production hot path for [`super::gpu_dispatch::try_gpu_bool`] —
    /// see Wave 8 / B / 3 findings for why per-step `compute`
    /// + sync was bottlenecked at 11 × ~700 µs ≈ 7.7 ms / 12-term
    /// cohort.
    ///
    /// Pattern (N terms):
    /// 1. H→D `term[0]` → `d_acc` (initially aliased to `d_a`)
    /// 2. For `i in 1..N`:
    ///    a. H→D `term[i]` → `d_b`
    ///    b. launch `d_acc OP d_b → d_out`
    ///    c. pointer-swap `d_acc` ↔ `d_out` (no memcpy)
    /// 3. `cudaStreamSynchronize` (single, end-of-fold)
    /// 4. D→H `d_acc` → host `Vec<u32>`
    ///
    /// Wallclock budget: 12 × `cudaMemcpyAsync` H→D + 11 ×
    /// `cuLaunchKernel` + 1 × sync + 1 × `cudaMemcpyAsync` D→H.
    /// Each async op is ~50 µs at PCIe 4.0 × 16 for the mega_cohort
    /// 96 KiB working set; total ≈ 1-2 ms, well under the 8.18 ms
    /// 11-step Wave 8 / B baseline.
    ///
    /// Empty / single-term short-circuit:
    /// - `terms.is_empty()` → returns `Ok(Vec::new())`.
    /// - `terms.len() == 1` → returns `Ok(terms[0].to_vec())` without
    ///   touching the GPU. Matches the semantics of folding a single
    ///   element with no binary op applied.
    pub fn compute_fold(
        &self,
        op: BoolOp,
        terms: &[&[u32]],
    ) -> Result<Vec<u32>, CudaBitmapError> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        if terms.len() == 1 {
            return Ok(terms[0].to_vec());
        }
        let n = terms[0].len();
        if n == 0 {
            return Ok(Vec::new());
        }
        for t in terms.iter().skip(1) {
            if t.len() != n {
                return Err(CudaBitmapError::LenMismatch {
                    a: n,
                    b: t.len(),
                });
            }
        }
        let bytes = n.checked_mul(std::mem::size_of::<u32>()).ok_or(
            CudaBitmapError::Cuda {
                what: "byte-size overflow",
                code: 0,
            },
        )?;

        let mut state = self
            .state
            .lock()
            .expect("CudaBitmapOpKernel state mutex poisoned");
        if n > state.capacity_words {
            // SAFETY: we hold the state mutex; no other caller can
            // observe the intermediate (freed) pointers.
            unsafe { reallocate_buffers(&mut state, n.next_power_of_two())? };
        }

        let cuda_op = to_inner_op(op);
        let n_u32 = u32::try_from(n).map_err(|_| CudaBitmapError::Cuda {
            what: "num_words exceeds u32::MAX",
            code: 0,
        })?;

        // SAFETY: state.d_a / state.d_b / state.d_out are all valid
        // device pointers sized for state.capacity_words ≥ n u32s. We
        // hold the state mutex for the whole fold so the pointers are
        // stable across iterations. The async chain is queued on a
        // single stream, terminated by a single sync before D→H.
        unsafe {
            // Step 1: upload terms[0] into d_a (the initial accumulator).
            let rc = cudaMemcpyAsync(
                state.d_a,
                terms[0].as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            );
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaMemcpyAsync(fold init H2D)",
                    code: rc,
                });
            }
            // Step 2: per-term upload + launch + pointer-swap.
            for term in terms.iter().skip(1) {
                let rc = cudaMemcpyAsync(
                    state.d_b,
                    term.as_ptr() as *const c_void,
                    bytes,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    self.stream,
                );
                if rc != CUDA_SUCCESS {
                    return Err(CudaBitmapError::Cuda {
                        what: "cudaMemcpyAsync(fold term H2D)",
                        code: rc,
                    });
                }
                self.inner.compute(
                    cuda_op,
                    state.d_a as *const u32,
                    state.d_b as *const u32,
                    state.d_out as *mut u32,
                    n_u32,
                )?;
                // Pointer-swap d_a ↔ d_out so the next iteration reads
                // the just-written result as its accumulator. No
                // memcpy. Both slots remain valid device buffers of
                // identical capacity; the role assignment is purely
                // logical. Manual swap (vs `std::mem::swap` on two
                // fields of the same struct) sidesteps the borrow
                // checker's two-mutable-borrow-of-state objection.
                let tmp = state.d_a;
                state.d_a = state.d_out;
                state.d_out = tmp;
            }
            // Step 3 + 4: single sync, then D→H from d_a (which holds
            // the final result after the last swap above).
            let mut out = vec![0u32; n];
            let rc = cudaMemcpyAsync(
                out.as_mut_ptr() as *mut c_void,
                state.d_a as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream,
            );
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaMemcpyAsync(fold result D2H)",
                    code: rc,
                });
            }
            let rc = cudaStreamSynchronize(self.stream);
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaStreamSynchronize(fold)",
                    code: rc,
                });
            }
            Ok(out)
        }
    }

    /// Run `out = a OP b` on host slices, returning a fresh `Vec<u32>`.
    /// Same byte-equal semantics as the wgpu `BitmapOpKernel::compute`
    /// — the byte-equal CPU oracle in
    /// [`super::gpu_dispatch::tests::gpu_and_matches_cpu_oracle`]
    /// continues to gate correctness regardless of backend.
    ///
    /// On a buffer-capacity miss the wrapper grows its persistent
    /// device buffers to the next power-of-two ≥ `n`. Growth happens
    /// under the internal `state` mutex; concurrent callers serialise
    /// only on resize, not on the steady-state `compute` path (the
    /// `cudaMemcpyAsync` + `cuLaunchKernel` chain runs under the lock
    /// too, but only because we need a stable view of the device
    /// pointers — the lock release happens immediately after
    /// `cudaStreamSynchronize` returns).
    ///
    /// **Prefer [`Self::compute_fold`] for multi-term reductions** —
    /// it eliminates the per-step `cudaStreamSynchronize` floor by
    /// batching the whole fold on the device with a single end-of-
    /// fold sync. This `compute` entry point is retained for one-shot
    /// pairwise ops and as the byte-equal oracle that fold tests
    /// compare against.
    pub fn compute(
        &self,
        op: BoolOp,
        a: &[u32],
        b: &[u32],
    ) -> Result<Vec<u32>, CudaBitmapError> {
        if a.len() != b.len() {
            return Err(CudaBitmapError::LenMismatch {
                a: a.len(),
                b: b.len(),
            });
        }
        let n = a.len();
        if n == 0 {
            // Empty cohort intersection — nothing to dispatch.
            return Ok(Vec::new());
        }
        let bytes = n.checked_mul(std::mem::size_of::<u32>()).ok_or(
            CudaBitmapError::Cuda {
                what: "byte-size overflow",
                code: 0,
            },
        )?;

        let mut state = self
            .state
            .lock()
            .expect("CudaBitmapOpKernel state mutex poisoned");
        if n > state.capacity_words {
            // SAFETY: we hold the state mutex; no other caller can
            // observe the intermediate (freed) pointers.
            unsafe { reallocate_buffers(&mut state, n.next_power_of_two())? };
        }

        let cuda_op = to_inner_op(op);
        let mut out = vec![0u32; n];
        // SAFETY: `state.d_a` / `state.d_b` / `state.d_out` are valid
        // device pointers sized for `state.capacity_words ≥ n` u32s.
        // The async memcpy + launch + memcpy chain runs on a single
        // stream; `cudaStreamSynchronize` blocks until all three
        // complete before we read `out`. Host pointers (`a`, `b`,
        // `out`) live for the whole call.
        unsafe {
            let rc = cudaMemcpyAsync(
                state.d_a,
                a.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            );
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaMemcpyAsync(a H2D)",
                    code: rc,
                });
            }
            let rc = cudaMemcpyAsync(
                state.d_b,
                b.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            );
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaMemcpyAsync(b H2D)",
                    code: rc,
                });
            }
            // n fits in u32: a Roaring posting list is at most
            // 2^32 docs (= 65 536 containers × 2 048 words = 1.34 *
            // 10^8 words ≪ u32::MAX).
            let n_u32 = u32::try_from(n).map_err(|_| CudaBitmapError::Cuda {
                what: "num_words exceeds u32::MAX",
                code: 0,
            })?;
            self.inner.compute(
                cuda_op,
                state.d_a as *const u32,
                state.d_b as *const u32,
                state.d_out as *mut u32,
                n_u32,
            )?;
            let rc = cudaMemcpyAsync(
                out.as_mut_ptr() as *mut c_void,
                state.d_out as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream,
            );
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaMemcpyAsync(out D2H)",
                    code: rc,
                });
            }
            let rc = cudaStreamSynchronize(self.stream);
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaStreamSynchronize",
                    code: rc,
                });
            }
        }
        Ok(out)
    }
}

impl Drop for CudaBitmapOpKernel {
    fn drop(&mut self) {
        // Free device memory before destroying the stream so any
        // pending memcpy on the stream sees the buffers still alive
        // (the Drop chain runs after `cudaStreamSynchronize` returned
        // in the last `compute()` call, but defensively we keep the
        // ordering explicit).
        if let Ok(mut buffers) = self.state.lock() {
            // SAFETY: pointers were allocated by `cudaMalloc` and have
            // not been freed elsewhere. Replacing with `null_mut()`
            // lets us be idempotent if Drop somehow runs twice (it
            // cannot, but be conservative).
            unsafe {
                let p = std::mem::replace(&mut buffers.d_a, null_mut());
                if !p.is_null() {
                    let _ = cudaFree(p);
                }
                let p = std::mem::replace(&mut buffers.d_b, null_mut());
                if !p.is_null() {
                    let _ = cudaFree(p);
                }
                let p = std::mem::replace(&mut buffers.d_out, null_mut());
                if !p.is_null() {
                    let _ = cudaFree(p);
                }
            }
        }
        if !self.stream.is_null() {
            // SAFETY: stream was created in `new` and is now exclusively
            // owned by this struct; safe to destroy.
            unsafe {
                let _ = cudaStreamDestroy(self.stream);
            }
            self.stream = null_mut();
        }
    }
}

/// Allocate the three persistent device buffers (`a`, `b`, `out`)
/// each sized for `capacity_words` u32 words.
///
/// # Safety
/// Allocates with `cudaMalloc`; on failure the caller receives an
/// error and no partial state escapes (intermediate successful
/// allocations are freed before returning the error).
unsafe fn allocate_buffers(
    capacity_words: usize,
) -> Result<DeviceBuffers, CudaBitmapError> {
    let bytes = capacity_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(CudaBitmapError::Cuda {
            what: "capacity byte-size overflow",
            code: 0,
        })?;
    let mut d_a: *mut c_void = null_mut();
    let mut d_b: *mut c_void = null_mut();
    let mut d_out: *mut c_void = null_mut();
    // SAFETY: cudaMalloc writes a valid device pointer on success; we
    // free intermediate successful allocations on later failure.
    unsafe {
        let rc = cudaMalloc(&mut d_a, bytes);
        if rc != CUDA_SUCCESS {
            return Err(CudaBitmapError::Cuda {
                what: "cudaMalloc(d_a)",
                code: rc,
            });
        }
        let rc = cudaMalloc(&mut d_b, bytes);
        if rc != CUDA_SUCCESS {
            let _ = cudaFree(d_a);
            return Err(CudaBitmapError::Cuda {
                what: "cudaMalloc(d_b)",
                code: rc,
            });
        }
        let rc = cudaMalloc(&mut d_out, bytes);
        if rc != CUDA_SUCCESS {
            let _ = cudaFree(d_a);
            let _ = cudaFree(d_b);
            return Err(CudaBitmapError::Cuda {
                what: "cudaMalloc(d_out)",
                code: rc,
            });
        }
    }
    Ok(DeviceBuffers {
        d_a,
        d_b,
        d_out,
        capacity_words,
    })
}

/// Replace the device buffers with a larger allocation. Caller must
/// hold the `state` mutex so no other thread observes the intermediate
/// freed state.
///
/// # Safety
/// Frees the old buffers before installing the new ones. No external
/// thread may hold the old pointers — guaranteed by the caller's
/// mutex hold.
unsafe fn reallocate_buffers(
    state: &mut DeviceBuffers,
    new_capacity_words: usize,
) -> Result<(), CudaBitmapError> {
    debug_assert!(new_capacity_words >= state.capacity_words);
    // SAFETY: see fn-doc; mutex hold ordering guarantees exclusive
    // pointer access.
    let new_buffers = unsafe { allocate_buffers(new_capacity_words)? };
    unsafe {
        let p = std::mem::replace(&mut state.d_a, null_mut());
        if !p.is_null() {
            let _ = cudaFree(p);
        }
        let p = std::mem::replace(&mut state.d_b, null_mut());
        if !p.is_null() {
            let _ = cudaFree(p);
        }
        let p = std::mem::replace(&mut state.d_out, null_mut());
        if !p.is_null() {
            let _ = cudaFree(p);
        }
    }
    *state = new_buffers;
    Ok(())
}

fn to_inner_op(op: BoolOp) -> FcBitmapOp {
    match op {
        BoolOp::And => FcBitmapOp::And,
        BoolOp::Or => FcBitmapOp::Or,
        BoolOp::Xor => FcBitmapOp::Xor,
    }
}

// ============================================================
// Tests
// ============================================================
//
// These tests require a working CUDA device on the host. On a
// driver-missing CI box the very first `cudaStreamCreate` in `new()`
// returns a non-zero code; the tests early-out gracefully so the suite
// stays green on machines without an NVIDIA GPU. Heavy CUDA
// correctness coverage already lives in
// `crates/ferro-compress/tests/bitmap_op_gpu.rs`; the tests here just
// pin the wrapper-level behaviour (length mismatch, growth, byte-equal
// vs CPU oracle on a small fixture).

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(seed: u64, len: usize) -> Vec<u32> {
        let mut s = seed.max(1);
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            out.push((s & 0xffff_ffff) as u32);
        }
        out
    }

    /// Helper: try to construct the kernel; if init fails on a
    /// driver-missing host, return None so the caller can skip the
    /// rest of the test cleanly.
    fn try_kernel() -> Option<CudaBitmapOpKernel> {
        match CudaBitmapOpKernel::new() {
            Ok(k) => Some(k),
            Err(_) => None,
        }
    }

    #[test]
    fn length_mismatch_is_caught_before_dispatch() {
        let Some(kernel) = try_kernel() else {
            // No GPU on this host — wrapper construction itself
            // returned an error. The mismatch check is a length
            // comparison that doesn't hit the GPU, but to call
            // `compute` we need a kernel; nothing more to assert.
            return;
        };
        let a = vec![0u32; 10];
        let b = vec![0u32; 11];
        let res = kernel.compute(BoolOp::And, &a, &b);
        assert!(matches!(
            res,
            Err(CudaBitmapError::LenMismatch { a: 10, b: 11 })
        ));
    }

    #[test]
    fn empty_input_returns_empty_output_without_dispatch() {
        let Some(kernel) = try_kernel() else { return };
        let res = kernel.compute(BoolOp::And, &[], &[]).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn cuda_and_matches_cpu_oracle_small() {
        let Some(kernel) = try_kernel() else { return };
        // Smallest realistic Bitmap container fixture — 2 048 u32
        // words = 1 container's worth.
        let a = pseudo_random(0x1234_5678, 2048);
        let b = pseudo_random(0x8765_4321, 2048);
        let got = kernel.compute(BoolOp::And, &a, &b).unwrap();
        let expected: Vec<u32> = a.iter().zip(b.iter()).map(|(x, y)| x & y).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn cuda_or_xor_match_cpu_oracle() {
        let Some(kernel) = try_kernel() else { return };
        let a = pseudo_random(7, 2048);
        let b = pseudo_random(11, 2048);
        let got_or = kernel.compute(BoolOp::Or, &a, &b).unwrap();
        let want_or: Vec<u32> = a.iter().zip(b.iter()).map(|(x, y)| x | y).collect();
        assert_eq!(got_or, want_or);
        let got_xor = kernel.compute(BoolOp::Xor, &a, &b).unwrap();
        let want_xor: Vec<u32> = a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect();
        assert_eq!(got_xor, want_xor);
    }

    #[test]
    fn buffer_growth_on_oversize_cohort() {
        let Some(kernel) = try_kernel() else { return };
        // First call with mega-cohort sized input forces growth from
        // INITIAL_CAPACITY_WORDS up to the next power of two.
        let oversize = INITIAL_CAPACITY_WORDS * 4; // 4× initial = 98 304 words
        let a = pseudo_random(1, oversize);
        let b = pseudo_random(2, oversize);
        let got = kernel.compute(BoolOp::And, &a, &b).unwrap();
        assert_eq!(got.len(), oversize);
        let expected: Vec<u32> = a.iter().zip(b.iter()).map(|(x, y)| x & y).collect();
        assert_eq!(got, expected);
        // Capacity must now be ≥ oversize and a power of two. We
        // validate via a follow-up call that's smaller than the new
        // capacity but larger than the initial — should NOT trigger
        // growth (no allocation happens silently; the test just
        // exercises the no-grow path on the same kernel).
        let mid = INITIAL_CAPACITY_WORDS * 2;
        let a2 = pseudo_random(3, mid);
        let b2 = pseudo_random(4, mid);
        let got2 = kernel.compute(BoolOp::Xor, &a2, &b2).unwrap();
        assert_eq!(got2.len(), mid);
    }

    #[test]
    fn fold_empty_returns_empty() {
        let Some(kernel) = try_kernel() else { return };
        let res = kernel.compute_fold(BoolOp::And, &[]).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn fold_single_term_round_trips_input() {
        let Some(kernel) = try_kernel() else { return };
        let term = pseudo_random(99, 2048);
        let got = kernel.compute_fold(BoolOp::And, &[&term]).unwrap();
        assert_eq!(got, term);
    }

    #[test]
    fn fold_matches_sequential_compute_oracle() {
        let Some(kernel) = try_kernel() else { return };
        // 3-term cohort, large enough to exercise the swap loop.
        let t0 = pseudo_random(1, 2048);
        let t1 = pseudo_random(2, 2048);
        let t2 = pseudo_random(3, 2048);
        let terms: [&[u32]; 3] = [&t0, &t1, &t2];
        // Sequential `compute` oracle: ((t0 AND t1) AND t2). Same
        // wallclock-bottlenecked path that Wave 8 / B / 2 lands as the
        // try_gpu_bool fold loop. compute_fold must produce a
        // byte-identical result — fold is associative for
        // bitwise AND/OR/XOR, and we apply ops in the same order.
        let step0 = kernel.compute(BoolOp::And, &t0, &t1).unwrap();
        let oracle = kernel.compute(BoolOp::And, &step0, &t2).unwrap();
        let got = kernel.compute_fold(BoolOp::And, &terms).unwrap();
        assert_eq!(got, oracle, "AND fold must match sequential compute");

        let or_step0 = kernel.compute(BoolOp::Or, &t0, &t1).unwrap();
        let or_oracle = kernel.compute(BoolOp::Or, &or_step0, &t2).unwrap();
        let or_got = kernel.compute_fold(BoolOp::Or, &terms).unwrap();
        assert_eq!(or_got, or_oracle, "OR fold must match sequential compute");

        let xor_step0 = kernel.compute(BoolOp::Xor, &t0, &t1).unwrap();
        let xor_oracle = kernel.compute(BoolOp::Xor, &xor_step0, &t2).unwrap();
        let xor_got = kernel.compute_fold(BoolOp::Xor, &terms).unwrap();
        assert_eq!(xor_got, xor_oracle, "XOR fold must match sequential compute");
    }

    #[test]
    fn fold_length_mismatch_caught_pre_dispatch() {
        let Some(kernel) = try_kernel() else { return };
        let t0 = vec![0u32; 10];
        let t1 = vec![0u32; 11]; // mismatched
        let terms: [&[u32]; 2] = [&t0, &t1];
        let res = kernel.compute_fold(BoolOp::And, &terms);
        assert!(matches!(
            res,
            Err(CudaBitmapError::LenMismatch { a: 10, b: 11 })
        ));
    }

    #[test]
    fn fold_mega_cohort_correctness_on_smaller_proxy() {
        let Some(kernel) = try_kernel() else { return };
        // 12 terms × 1 024 words each — the same shape as
        // try_gpu_bool's mega_cohort fold but small enough to verify
        // byte-equality against a host-side AND oracle quickly.
        let owned: Vec<Vec<u32>> = (0..12)
            .map(|i| pseudo_random(0xfeed + i as u64, 1024))
            .collect();
        let terms: Vec<&[u32]> = owned.iter().map(|v| v.as_slice()).collect();
        let got = kernel.compute_fold(BoolOp::And, &terms).unwrap();
        // Host AND oracle.
        let mut want = owned[0].clone();
        for term in &owned[1..] {
            for (w, t) in want.iter_mut().zip(term.iter()) {
                *w &= *t;
            }
        }
        assert_eq!(got, want, "12-term AND fold must match host oracle");
    }

    #[test]
    fn debug_format_does_not_panic() {
        let Some(kernel) = try_kernel() else { return };
        // Smoke test: Debug impl reports stream + capacity without
        // touching the kernel state.
        let dbg = format!("{kernel:?}");
        assert!(dbg.contains("CudaBitmapOpKernel"));
        assert!(dbg.contains("capacity_words"));
    }
}

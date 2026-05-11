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
    cudaFree, cudaMalloc, cudaMemcpyAsync, cudaMemcpyKind, cudaMemsetAsync, cudaStreamCreate,
    cudaStreamDestroy, cudaStreamSynchronize, cudaStream_t, CUDA_SUCCESS,
};
use ferro_compress::{
    BitcompDataType, BitcompDeviceCodec, BitmapOp as FcBitmapOp, BitmapOpKernel as InnerKernel,
    Error as FcError,
};

use super::gpu_dispatch::BoolOp;
use super::vram_cht::VramTermEntry;
use super::vram_cht_v3::VramCompressedTermEntry;
use super::BITMAP_CONTAINER_WORDS;
use std::sync::Arc;

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
    /// Phase 2 D-4 v3 — Bitcomp decompress codec sharing this kernel's
    /// stream so decompress→scatter→compute→sync chains on a single
    /// stream. Wrapped in `Mutex` so concurrent
    /// [`Self::compute_fold_v3`] callers serialise on the codec's
    /// internal device singletons. `None` if the codec failed to
    /// construct (e.g. driver-missing host); in that case
    /// `compute_fold_v3` returns `Err(...)` for the whole call.
    decompress_codec: Mutex<Option<BitcompDeviceCodec>>,
    /// Phase 2 D-4 v3 — persistent device decompress workbench buffer.
    /// Sized for the largest term's `uncompressed_bytes`; grows on
    /// demand via [`reallocate_workbench`].
    workbench: Mutex<DecompressWorkbench>,
}

struct DecompressWorkbench {
    d_buf: *mut c_void,
    capacity_bytes: usize,
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
        // Phase 2 D-4 v3 — try to construct a Bitcomp decompress codec
        // sharing this kernel's stream. Failure is non-fatal; v3 just
        // becomes unavailable on this kernel instance and `compute_fold_v3`
        // surfaces an error per call.
        let decompress_codec = BitcompDeviceCodec::with_stream(
            BitcompDataType::Uint32,
            stream,
        )
        .ok();
        Ok(Self {
            inner,
            stream,
            state: Mutex::new(buffers),
            decompress_codec: Mutex::new(decompress_codec),
            workbench: Mutex::new(DecompressWorkbench {
                d_buf: null_mut(),
                capacity_bytes: 0,
            }),
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

    /// Run `out = TERMS[0] OP TERMS[1] OP … OP TERMS[N-1]` on
    /// device-resident terms (Phase 2 D-3 v2 fast path).
    ///
    /// Each term is a [`VramTermEntry`] holding pre-expanded
    /// BitmapContainer-form buckets in device memory under
    /// `(high16, word_offset)` indexing. The function:
    ///
    /// 1. Resizes the persistent kernel buffers to fit
    ///    `union_keys.len() * BITMAP_CONTAINER_WORDS` u32 words if
    ///    needed.
    /// 2. Builds the accumulator (`d_a`) by zero-filling and then
    ///    per-bucket `cudaMemcpyAsync(DeviceToDevice)` from
    ///    `terms[0]`'s cached buckets to the bucket slots dictated by
    ///    `union_keys`. Buckets present in the term but not in
    ///    `union_keys` are skipped (cohort intersection is bounded by
    ///    the union, but a malformed `terms[0]` with extra buckets
    ///    is handled defensively by the binary-search lookup).
    /// 3. For each subsequent term: zero-fill `d_b`, scatter the
    ///    term's buckets into `d_b`, run
    ///    `inner.compute(op, d_a, d_b, d_out, words)`, pointer-swap
    ///    `d_a` ↔ `d_out` so the next iteration consumes the just-
    ///    written result as its accumulator.
    /// 4. After the fold, `cudaMemcpyAsync(DeviceToHost)` from `d_a`
    ///    (post-final-swap) into a fresh host `Vec<u32>` and a single
    ///    end-of-fold `cudaStreamSynchronize`.
    ///
    /// Wallclock target vs [`compute_fold`] (host-slice fold):
    ///
    /// - `compute_fold` (host slices): N × `cudaMemcpyAsync` H→D of
    ///   `union_keys.len() × 8 KiB` per term, plus N - 1 kernel
    ///   launches, plus 1 D→H, plus 1 sync.
    /// - `compute_fold_vram` (device-resident terms): N × per-bucket
    ///   `cudaMemcpyAsync` D→D (only the buckets actually present in
    ///   each term, typically far fewer than `union_keys.len()`),
    ///   plus N - 1 kernel launches, plus 1 D→H, plus 1 sync.
    ///
    /// At cohort scale (12 terms × ~10 non-empty buckets each), the
    /// D→D scatter is O(120) × 8 KiB = ~1 MiB on-device traffic per
    /// query (~1.6 µs at L40S 600 GB/s+ bandwidth), vs ~1 MiB H→D
    /// (~40 µs at PCIe 4.0 ×16). At larger cohort scales (50 terms ×
    /// 100 buckets), the D→D advantage grows to ~24×.
    ///
    /// `union_keys` must be sorted ascending by `high16`.
    /// (`super::gpu_dispatch::union_high16_keys` produces sorted output
    /// from a `BTreeSet`, satisfying the invariant.)
    ///
    /// Empty / single-term short-circuit identical to [`compute_fold`].
    pub fn compute_fold_vram(
        &self,
        op: BoolOp,
        terms: &[Arc<VramTermEntry>],
        union_keys: &[u16],
    ) -> Result<Vec<u32>, CudaBitmapError> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        if union_keys.is_empty() {
            return Ok(Vec::new());
        }
        // Single-term: re-emit the term's own flat layout into a host
        // Vec without going through the kernel. Caller usually short-
        // circuits before us, but defensively handle it.
        if terms.len() == 1 {
            return self.gather_term_to_host(&terms[0], union_keys);
        }
        let words_per_term = union_keys
            .len()
            .checked_mul(BITMAP_CONTAINER_WORDS)
            .ok_or(CudaBitmapError::Cuda {
                what: "words_per_term overflow",
                code: 0,
            })?;
        let bytes_per_term = words_per_term
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaBitmapError::Cuda {
                what: "bytes_per_term overflow",
                code: 0,
            })?;

        let mut state = self
            .state
            .lock()
            .expect("CudaBitmapOpKernel state mutex poisoned");
        if words_per_term > state.capacity_words {
            // SAFETY: we hold the state mutex; no other caller can
            // observe the intermediate (freed) pointers.
            unsafe { reallocate_buffers(&mut state, words_per_term.next_power_of_two())? };
        }

        let cuda_op = to_inner_op(op);
        let n_u32 =
            u32::try_from(words_per_term).map_err(|_| CudaBitmapError::Cuda {
                what: "words_per_term exceeds u32::MAX",
                code: 0,
            })?;

        // SAFETY: every cudaMemset / cudaMemcpyAsync below targets a
        // `*mut c_void` (or `*mut u32` cast to `*mut c_void`) that
        // points to one of the persistent kernel buffers we hold under
        // the state mutex (`d_a`, `d_b`, `d_out`), each sized for
        // `state.capacity_words ≥ words_per_term` u32s. Source
        // pointers are read from the immutable
        // [`VramTermEntry::device_ptr`] (Arc'd term entries hold the
        // device buffer alive for the duration of this call). The whole
        // chain runs on `self.stream`, terminated by a single
        // `cudaStreamSynchronize` before D→H read.
        unsafe {
            // Step 1: scatter terms[0] into d_a (the initial accumulator).
            scatter_term_to_device(
                state.d_a,
                bytes_per_term,
                &terms[0],
                union_keys,
                self.stream,
            )?;
            // Step 2: per-term scatter + kernel + pointer-swap.
            for term in terms.iter().skip(1) {
                scatter_term_to_device(
                    state.d_b,
                    bytes_per_term,
                    term,
                    union_keys,
                    self.stream,
                )?;
                self.inner.compute(
                    cuda_op,
                    state.d_a as *const u32,
                    state.d_b as *const u32,
                    state.d_out as *mut u32,
                    n_u32,
                )?;
                // Pointer-swap d_a ↔ d_out so the next iteration reads
                // the just-written result as its accumulator. Same idiom
                // as [`Self::compute_fold`].
                let tmp = state.d_a;
                state.d_a = state.d_out;
                state.d_out = tmp;
            }
            // Step 3: D→H from d_a (final result after last swap).
            let mut out = vec![0u32; words_per_term];
            let rc = cudaMemcpyAsync(
                out.as_mut_ptr() as *mut c_void,
                state.d_a as *const c_void,
                bytes_per_term,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream,
            );
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaMemcpyAsync(vram fold result D2H)",
                    code: rc,
                });
            }
            let rc = cudaStreamSynchronize(self.stream);
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaStreamSynchronize(vram fold)",
                    code: rc,
                });
            }
            Ok(out)
        }
    }

    /// Phase 2 D-4 v3 — fold device-resident **Bitcomp-compressed**
    /// terms into a Bool-AND/OR/XOR result.
    ///
    /// Same dispatch shape as [`Self::compute_fold_vram`] but each
    /// term is a [`VramCompressedTermEntry`] (stored compressed on
    /// device via [`super::vram_cht_v3`]); this method:
    ///
    /// 1. Resizes the persistent kernel buffers (`d_a` / `d_b` /
    ///    `d_out`) to fit `union_keys.len() * BITMAP_CONTAINER_WORDS`
    ///    u32 words.
    /// 2. Resizes the decompress workbench to fit the largest term's
    ///    `uncompressed_bytes`.
    /// 3. For `terms[0]`: decompresses into the workbench, scatters
    ///    from workbench (using the entry's `bucket_index`) to `d_a`
    ///    in the cohort's `union_keys` layout.
    /// 4. For `terms[i]` (i ≥ 1): same decompress + scatter to `d_b`,
    ///    then `inner.compute(op, d_a, d_b, d_out, words)` and
    ///    pointer-swap `d_a ↔ d_out`.
    /// 5. After the fold, `cudaMemcpyAsync` D→H from `d_a` and a
    ///    single end-of-fold `cudaStreamSynchronize`.
    ///
    /// All decompress + memset + memcpy + kernel launches queue on
    /// the kernel's persistent stream — single sync at end. The
    /// codec's `decompress_one` does its own per-call sync (it reads
    /// `d_uncomp_sizes` back to host to verify the output matches
    /// `expected_uncomp_size`); for v3's hot path this is a stream-
    /// boundary cost worth amortising in a future wave (multi-term
    /// batch decompress).
    ///
    /// Wallclock per-cohort cost (12-term cohort, Phase 2 D level-A
    /// Priority 1 batch decompress — Wave 11+):
    /// - Decompress: 1 batched call + 1 sync ≈ 30-40 µs (was 12 × ~20 µs
    ///   = 240 µs with per-term `decompress_one` loop)
    /// - Scatter (D→D): 12 × ~10 µs ≈ 120 µs
    /// - Kernel + sync: same as v2 (~50 µs)
    /// - Total: ~200-220 µs per cohort warm (was ~400-500 µs)
    ///
    /// vs v2 same cohort: ~350 µs warm. v3 with batch decompress is now
    /// faster than v2 for cohorts ≥ 8 terms while still caching ~3-4×
    /// more terms in the same VRAM budget (Bitcomp ratio on uint32
    /// postings).
    pub fn compute_fold_v3(
        &self,
        op: BoolOp,
        terms: &[Arc<VramCompressedTermEntry>],
        union_keys: &[u16],
    ) -> Result<Vec<u32>, CudaBitmapError> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        if union_keys.is_empty() {
            return Ok(Vec::new());
        }
        // Single-term: decompress to a host vec via the workbench +
        // gather scatter pattern. Caller usually short-circuits before
        // us, but be defensive.
        if terms.len() == 1 {
            return self.gather_v3_term_to_host(&terms[0], union_keys);
        }
        let words_per_term = union_keys
            .len()
            .checked_mul(BITMAP_CONTAINER_WORDS)
            .ok_or(CudaBitmapError::Cuda {
                what: "words_per_term overflow",
                code: 0,
            })?;
        let bytes_per_term = words_per_term
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaBitmapError::Cuda {
                what: "bytes_per_term overflow",
                code: 0,
            })?;
        // Per-term uncompressed sizes and a workbench-offset table.
        // The workbench holds all N terms concatenated so one batched
        // decompress can write the whole cohort in a single launch.
        let mut term_offsets: Vec<usize> = Vec::with_capacity(terms.len());
        let mut total_uncompressed: usize = 0;
        for term in terms {
            term_offsets.push(total_uncompressed);
            total_uncompressed = total_uncompressed
                .checked_add(term.uncompressed_bytes())
                .ok_or(CudaBitmapError::Cuda {
                    what: "workbench offset overflow",
                    code: 0,
                })?;
        }

        let mut state = self
            .state
            .lock()
            .expect("CudaBitmapOpKernel state mutex poisoned");
        if words_per_term > state.capacity_words {
            // SAFETY: we hold the state mutex; no other caller can
            // observe the intermediate (freed) pointers.
            unsafe { reallocate_buffers(&mut state, words_per_term.next_power_of_two())? };
        }

        // Resize the workbench under its own mutex. The batch decompress
        // path requires the workbench to fit ALL terms concatenated, so
        // we grow to `total_uncompressed` (was `max(per_term)` for the
        // serial path).
        {
            let mut workbench = self
                .workbench
                .lock()
                .expect("CudaBitmapOpKernel workbench mutex poisoned");
            if workbench.capacity_bytes < total_uncompressed {
                // SAFETY: we hold the workbench mutex; no other caller
                // can observe the intermediate freed state.
                unsafe { reallocate_workbench(&mut workbench, total_uncompressed)? };
            }
        }

        let cuda_op = to_inner_op(op);
        let n_u32 =
            u32::try_from(words_per_term).map_err(|_| CudaBitmapError::Cuda {
                what: "words_per_term exceeds u32::MAX",
                code: 0,
            })?;

        // SAFETY: device pointers + workbench valid for the whole
        // call; we hold both mutexes.
        unsafe {
            // Step 1: ONE batched decompress writes all N terms into
            // the concatenated workbench at `term_offsets[i]`. Replaces
            // the per-term `decompress_one` loop from the original Wave
            // 9 v3 dispatch — 6-10× wallclock win for typical cohorts
            // (Phase 2 D level-A Priority 1, 2026-05-11).
            self.decompress_batch_into_workbench(terms, &term_offsets)?;

            // Step 2: per-term scatter from the workbench slab to d_a
            // (for terms[0]) or d_b (for terms[1..]) → kernel → swap.
            self.scatter_from_workbench(
                state.d_a,
                bytes_per_term,
                &terms[0],
                term_offsets[0],
                union_keys,
            )?;
            for (i, term) in terms.iter().enumerate().skip(1) {
                self.scatter_from_workbench(
                    state.d_b,
                    bytes_per_term,
                    term,
                    term_offsets[i],
                    union_keys,
                )?;
                self.inner.compute(
                    cuda_op,
                    state.d_a as *const u32,
                    state.d_b as *const u32,
                    state.d_out as *mut u32,
                    n_u32,
                )?;
                let tmp = state.d_a;
                state.d_a = state.d_out;
                state.d_out = tmp;
            }
            // Step 3: D→H from d_a.
            let mut out = vec![0u32; words_per_term];
            let rc = cudaMemcpyAsync(
                out.as_mut_ptr() as *mut c_void,
                state.d_a as *const c_void,
                bytes_per_term,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream,
            );
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaMemcpyAsync(v3 fold result D2H)",
                    code: rc,
                });
            }
            let rc = cudaStreamSynchronize(self.stream);
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaStreamSynchronize(v3 fold)",
                    code: rc,
                });
            }
            Ok(out)
        }
    }

    /// Phase 2 D level-A Priority 1 — multi-term batch decompress into
    /// the workbench. One nvcomp launch + one sync replaces the N-call
    /// per-term loop. Each term lands at `term_offsets[i]` bytes within
    /// the workbench (a flat slab sized for the sum of all terms'
    /// uncompressed bytes).
    ///
    /// Wave Z-6 #3: a term may hold up to
    /// [`super::vram_cht_v3::MAX_CHUNKS_PER_ENTRY`] Bitcomp chunks; we
    /// flatten `M` terms × `N_i` chunks/term into `Σ N_i` independent
    /// nvcomp batched-decompress entries, each landing at
    /// `term_offsets[i] + chunk_byte_offset` (the running prefix sum of
    /// preceding chunks' `uncompressed_bytes` within the term). One
    /// `decompress_batch` call still suffices — the nvcomp API is
    /// per-entry independent.
    ///
    /// # Safety
    /// - Caller has ensured the workbench is sized for the sum of all
    ///   `terms[i].uncompressed_bytes()` and not concurrently grown.
    /// - All chunk pointers (`chunk.d_compressed` for every chunk of
    ///   every term) point to live VRAM owned by the term entries
    ///   (held alive via `Arc`).
    unsafe fn decompress_batch_into_workbench(
        &self,
        terms: &[Arc<VramCompressedTermEntry>],
        term_offsets: &[usize],
    ) -> Result<(), CudaBitmapError> {
        debug_assert_eq!(terms.len(), term_offsets.len());
        let workbench_ptr = {
            let workbench = self
                .workbench
                .lock()
                .expect("CudaBitmapOpKernel workbench mutex poisoned");
            workbench.d_buf
        };
        if workbench_ptr.is_null() {
            return Err(CudaBitmapError::Cuda {
                what: "v3 workbench not allocated",
                code: 0,
            });
        }
        let mut codec_guard = self
            .decompress_codec
            .lock()
            .expect("CudaBitmapOpKernel decompress_codec mutex poisoned");
        let codec = codec_guard.as_mut().ok_or(CudaBitmapError::Cuda {
            what: "v3 decompress codec unavailable",
            code: 0,
        })?;
        // Build the entries vector for the batch API:
        // (d_compressed, comp_size, expected_uncomp_size, d_uncompressed_slot)
        // M × N_i flatten — see method rustdoc.
        let total_chunks: usize = terms.iter().map(|t| t.chunk_count()).sum();
        let mut entries: Vec<(*const c_void, usize, usize, *mut c_void)> =
            Vec::with_capacity(total_chunks);
        for (term, &term_offset) in terms.iter().zip(term_offsets.iter()) {
            let mut chunk_byte_offset: usize = 0;
            for chunk in term.chunks() {
                // SAFETY: workbench_ptr is a single contiguous allocation
                // of ≥ Σ terms[i].uncompressed_bytes(); `term_offset` is
                // the partial-sum start for this term within the slab;
                // `chunk_byte_offset` is the running prefix sum of
                // preceding chunks' uncompressed_bytes within the term
                // and is < term.uncompressed_bytes() by construction.
                let slot = unsafe {
                    (workbench_ptr as *mut u8).add(term_offset + chunk_byte_offset)
                        as *mut c_void
                };
                entries.push((
                    chunk.d_compressed as *const c_void,
                    chunk.compressed_bytes,
                    chunk.uncompressed_bytes,
                    slot,
                ));
                chunk_byte_offset += chunk.uncompressed_bytes;
            }
        }
        // SAFETY: device pointers + workbench slots valid for the call;
        // the codec uses our stream so chained scatter/compute see the
        // decompressed bytes after the post-batch sync inside
        // `decompress_batch`.
        unsafe {
            codec
                .decompress_batch(&entries)
                .map_err(CudaBitmapError::Inner)?;
        }
        drop(codec_guard);
        Ok(())
    }

    /// Phase 2 D level-A Priority 1 — scatter a single term from its
    /// slab within the (pre-decompressed) workbench into `dst` in the
    /// cohort `union_keys` layout. Same shape as the original
    /// [`Self::decompress_then_scatter`] scatter half — the decompress
    /// half is now amortised across the cohort.
    ///
    /// # Safety
    /// - `dst` must point to ≥ `dst_bytes` writable device memory on
    ///   the kernel's stream.
    /// - Caller holds the state and workbench mutexes (or appropriate
    ///   barriers) so the workbench isn't grown under us.
    /// - The workbench has been populated by a preceding
    ///   `decompress_batch_into_workbench` call on the same stream.
    unsafe fn scatter_from_workbench(
        &self,
        dst: *mut c_void,
        dst_bytes: usize,
        term: &Arc<VramCompressedTermEntry>,
        term_offset: usize,
        union_keys: &[u16],
    ) -> Result<(), CudaBitmapError> {
        let workbench_ptr = {
            let workbench = self
                .workbench
                .lock()
                .expect("CudaBitmapOpKernel workbench mutex poisoned");
            workbench.d_buf
        };
        if workbench_ptr.is_null() {
            return Err(CudaBitmapError::Cuda {
                what: "v3 workbench not allocated",
                code: 0,
            });
        }
        // SAFETY: dst sized for dst_bytes; workbench slab sized for
        // term.uncompressed_bytes() at `term_offset`; all on this
        // kernel's stream.
        unsafe {
            let rc = cudaMemsetAsync(dst, 0i32 as std::ffi::c_int, dst_bytes, self.stream);
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaMemsetAsync(v3 dst zero-fill)",
                    code: rc,
                });
            }
            let bucket_bytes = BITMAP_CONTAINER_WORDS * std::mem::size_of::<u32>();
            let term_base = (workbench_ptr as *const u8).add(term_offset);
            for (high16, src_word_off) in term.bucket_index() {
                let Ok(dst_idx) = union_keys.binary_search(high16) else {
                    continue;
                };
                let dst_word_off = dst_idx * BITMAP_CONTAINER_WORDS;
                let dst_ptr =
                    (dst as *mut u8).add(dst_word_off * std::mem::size_of::<u32>());
                let src_ptr =
                    term_base.add((*src_word_off as usize) * std::mem::size_of::<u32>());
                let rc = cudaMemcpyAsync(
                    dst_ptr as *mut c_void,
                    src_ptr as *const c_void,
                    bucket_bytes,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    self.stream,
                );
                if rc != CUDA_SUCCESS {
                    return Err(CudaBitmapError::Cuda {
                        what: "cudaMemcpyAsync(v3 scatter D2D bucket)",
                        code: rc,
                    });
                }
            }
        }
        Ok(())
    }

    /// Internal: decompress `term` into the workbench, then scatter
    /// from workbench to `dst` in the cohort `union_keys` layout.
    /// Uses the same per-bucket DtD memcpy as v2's
    /// [`scatter_term_to_device`].
    ///
    /// **Phase 2 D level-A Priority 1 (2026-05-11)**: superseded on the
    /// `compute_fold_v3` hot path by
    /// [`Self::decompress_batch_into_workbench`] +
    /// [`Self::scatter_from_workbench`], which fuse N per-term
    /// `decompress_one` launches into one batched nvCOMP call. Kept here
    /// as a one-shot reference for the bench harness comparison and as
    /// a fallback path that doesn't depend on the batch API.
    ///
    /// # Safety
    /// - `dst` must point to ≥ `dst_bytes` writable device memory on
    ///   the kernel's stream.
    /// - Caller holds `self.state` and `self.workbench` mutexes (or
    ///   appropriate barriers) so the workbench buffer isn't grown
    ///   under us.
    /// - Caller must hold the `Arc<VramCompressedTermEntry>` alive
    ///   for the duration of the call.
    #[allow(dead_code)]
    pub(super) unsafe fn decompress_then_scatter(
        &self,
        dst: *mut c_void,
        dst_bytes: usize,
        term: &Arc<VramCompressedTermEntry>,
        union_keys: &[u16],
    ) -> Result<(), CudaBitmapError> {
        // 1. Decompress into the workbench (synchronous via codec's
        //    internal stream sync).
        let workbench_ptr = {
            let workbench = self
                .workbench
                .lock()
                .expect("CudaBitmapOpKernel workbench mutex poisoned");
            workbench.d_buf
        };
        if workbench_ptr.is_null() {
            return Err(CudaBitmapError::Cuda {
                what: "v3 workbench not allocated",
                code: 0,
            });
        }
        let mut codec_guard = self
            .decompress_codec
            .lock()
            .expect("CudaBitmapOpKernel decompress_codec mutex poisoned");
        let codec = codec_guard.as_mut().ok_or(CudaBitmapError::Cuda {
            what: "v3 decompress codec unavailable",
            code: 0,
        })?;
        // SAFETY: device buffers are valid; codec uses our stream.
        // Wave Z-6 #3: walk chunks() so multi-chunk entries decompress
        // contiguously starting at `workbench_ptr` (each chunk lands at
        // its `chunk_byte_offset` running prefix sum). For single-chunk
        // entries this degenerates to one `decompress_one` call.
        let mut chunk_byte_offset: usize = 0;
        for chunk in term.chunks() {
            let slot = unsafe {
                (workbench_ptr as *mut u8).add(chunk_byte_offset) as *mut c_void
            };
            codec
                .decompress_one(
                    chunk.d_compressed as *const c_void,
                    chunk.compressed_bytes,
                    slot,
                    chunk.uncompressed_bytes,
                )
                .map_err(CudaBitmapError::Inner)?;
            chunk_byte_offset += chunk.uncompressed_bytes;
        }
        drop(codec_guard);

        // 2. Scatter from workbench to dst (per-bucket DtD memcpy +
        //    zero-fill missing). Same shape as v2 scatter but source
        //    is the workbench, not a cached d_buckets pointer.
        // SAFETY: dst sized for dst_bytes; workbench sized for term
        // uncompressed bytes; both on this kernel's stream.
        unsafe {
            let rc = cudaMemsetAsync(dst, 0i32 as std::ffi::c_int, dst_bytes, self.stream);
            if rc != CUDA_SUCCESS {
                return Err(CudaBitmapError::Cuda {
                    what: "cudaMemsetAsync(v3 dst zero-fill)",
                    code: rc,
                });
            }
            let bucket_bytes = BITMAP_CONTAINER_WORDS * std::mem::size_of::<u32>();
            for (high16, src_word_off) in term.bucket_index() {
                let Ok(dst_idx) = union_keys.binary_search(high16) else {
                    continue;
                };
                let dst_word_off = dst_idx * BITMAP_CONTAINER_WORDS;
                let dst_ptr =
                    (dst as *mut u8).add(dst_word_off * std::mem::size_of::<u32>());
                let src_ptr = (workbench_ptr as *const u8)
                    .add((*src_word_off as usize) * std::mem::size_of::<u32>());
                let rc = cudaMemcpyAsync(
                    dst_ptr as *mut c_void,
                    src_ptr as *const c_void,
                    bucket_bytes,
                    cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    self.stream,
                );
                if rc != CUDA_SUCCESS {
                    return Err(CudaBitmapError::Cuda {
                        what: "cudaMemcpyAsync(v3 scatter D2D bucket)",
                        code: rc,
                    });
                }
            }
        }
        Ok(())
    }

    /// Single-term degenerate path: decompress, then return the
    /// cohort-layout host buffer. Mirrors [`Self::gather_term_to_host`]
    /// but with the v3 decompress step in front.
    fn gather_v3_term_to_host(
        &self,
        term: &Arc<VramCompressedTermEntry>,
        union_keys: &[u16],
    ) -> Result<Vec<u32>, CudaBitmapError> {
        let max_uncompressed = term.uncompressed_bytes();
        {
            let mut workbench = self
                .workbench
                .lock()
                .expect("CudaBitmapOpKernel workbench mutex poisoned");
            if workbench.capacity_bytes < max_uncompressed {
                // SAFETY: workbench mutex held.
                unsafe { reallocate_workbench(&mut workbench, max_uncompressed)? };
            }
        }
        let workbench_ptr = {
            let workbench = self
                .workbench
                .lock()
                .expect("CudaBitmapOpKernel workbench mutex poisoned");
            workbench.d_buf
        };
        if workbench_ptr.is_null() {
            return Err(CudaBitmapError::Cuda {
                what: "v3 workbench not allocated",
                code: 0,
            });
        }
        {
            let mut codec_guard = self
                .decompress_codec
                .lock()
                .expect("CudaBitmapOpKernel decompress_codec mutex poisoned");
            let codec = codec_guard.as_mut().ok_or(CudaBitmapError::Cuda {
                what: "v3 decompress codec unavailable",
                code: 0,
            })?;
            // SAFETY: see decompress_then_scatter. Wave Z-6 #3: walk
            // chunks() so multi-chunk entries decompress into
            // contiguous workbench slots starting at `workbench_ptr`.
            let mut chunk_byte_offset: usize = 0;
            for chunk in term.chunks() {
                let slot = unsafe {
                    (workbench_ptr as *mut u8).add(chunk_byte_offset) as *mut c_void
                };
                unsafe {
                    codec
                        .decompress_one(
                            chunk.d_compressed as *const c_void,
                            chunk.compressed_bytes,
                            slot,
                            chunk.uncompressed_bytes,
                        )
                        .map_err(CudaBitmapError::Inner)?;
                }
                chunk_byte_offset += chunk.uncompressed_bytes;
            }
        }
        // Now gather into host buffer (same shape as
        // gather_term_to_host but workbench is the source).
        let words = union_keys.len() * BITMAP_CONTAINER_WORDS;
        let mut out = vec![0u32; words];
        for (high16, src_off) in term.bucket_index() {
            let Ok(dst_idx) = union_keys.binary_search(high16) else {
                continue;
            };
            let dst_start = dst_idx * BITMAP_CONTAINER_WORDS;
            let dst_end = dst_start + BITMAP_CONTAINER_WORDS;
            // SAFETY: per-bucket sync D→H from workbench.
            unsafe {
                let src_ptr = (workbench_ptr as *const u8)
                    .add((*src_off as usize) * std::mem::size_of::<u32>());
                let dst_ptr = out[dst_start..dst_end].as_mut_ptr() as *mut c_void;
                let rc = ferro_compress::nvcomp_sys::cuda::cudaMemcpy(
                    dst_ptr,
                    src_ptr as *const c_void,
                    BITMAP_CONTAINER_WORDS * std::mem::size_of::<u32>(),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
                if rc != CUDA_SUCCESS {
                    return Err(CudaBitmapError::Cuda {
                        what: "cudaMemcpy(gather v3 single-term D2H)",
                        code: rc,
                    });
                }
            }
        }
        Ok(out)
    }

    /// Single-term degenerate path: gather the term's cached buckets
    /// into a host `Vec<u32>` matching the cohort `union_keys` layout.
    /// Used by [`Self::compute_fold_vram`] when `terms.len() == 1` so
    /// the kernel isn't invoked. Implemented as a synchronous
    /// `cudaMemcpy` D→H to a host staging Vec, with zero-init for
    /// missing buckets.
    fn gather_term_to_host(
        &self,
        term: &Arc<VramTermEntry>,
        union_keys: &[u16],
    ) -> Result<Vec<u32>, CudaBitmapError> {
        let words = union_keys.len() * BITMAP_CONTAINER_WORDS;
        let mut out = vec![0u32; words];
        for (high16, src_off) in term.bucket_index() {
            let Ok(dst_idx) = union_keys.binary_search(high16) else {
                continue;
            };
            let dst_start = dst_idx * BITMAP_CONTAINER_WORDS;
            let dst_end = dst_start + BITMAP_CONTAINER_WORDS;
            // SAFETY: cudaMemcpy synchronously copies BITMAP_CONTAINER_WORDS
            // u32 = 8 KiB from the term's device buffer at offset
            // `src_off * 4` bytes to the corresponding host slice.
            // Bounds: src_off + BITMAP_CONTAINER_WORDS ≤ term.total_words
            // by construction in [`VramCht::build_entry`]; dst_end ≤
            // out.len() because `dst_idx < union_keys.len()`.
            unsafe {
                let src_ptr = (term.device_ptr() as *const u8)
                    .add((*src_off as usize) * std::mem::size_of::<u32>());
                let dst_ptr = out[dst_start..dst_end].as_mut_ptr() as *mut c_void;
                let rc = ferro_compress::nvcomp_sys::cuda::cudaMemcpy(
                    dst_ptr,
                    src_ptr as *const c_void,
                    BITMAP_CONTAINER_WORDS * std::mem::size_of::<u32>(),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                );
                if rc != CUDA_SUCCESS {
                    return Err(CudaBitmapError::Cuda {
                        what: "cudaMemcpy(gather single-term D2H)",
                        code: rc,
                    });
                }
            }
        }
        Ok(out)
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
        // Phase 2 D-4 v3 — free the decompress codec FIRST so its
        // own `cudaFree` calls happen while the stream is still
        // alive. The codec was constructed with `with_stream` =
        // borrowed-stream semantics, so dropping it does NOT destroy
        // the stream (we do that below).
        if let Ok(mut codec_slot) = self.decompress_codec.lock() {
            let _ = codec_slot.take();
        }
        // Free the v3 workbench buffer.
        if let Ok(mut workbench) = self.workbench.lock() {
            // SAFETY: workbench buffer was allocated by `cudaMalloc`
            // in `reallocate_workbench`; replace with null for
            // idempotency.
            unsafe {
                let p = std::mem::replace(&mut workbench.d_buf, null_mut());
                if !p.is_null() {
                    let _ = cudaFree(p);
                }
            }
            workbench.capacity_bytes = 0;
        }
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
/// Phase 2 D-4 v3 — grow the persistent decompress workbench to fit
/// `new_capacity_bytes` of uncompressed term data. Caller must hold
/// the workbench mutex so no other thread observes the intermediate
/// freed state.
///
/// # Safety
/// Frees the old buffer before installing the new one. No external
/// thread may hold the old pointer.
unsafe fn reallocate_workbench(
    workbench: &mut DecompressWorkbench,
    new_capacity_bytes: usize,
) -> Result<(), CudaBitmapError> {
    debug_assert!(new_capacity_bytes >= workbench.capacity_bytes);
    let mut new_buf: *mut c_void = null_mut();
    // SAFETY: cudaMalloc writes a valid device pointer on success.
    let rc = unsafe { cudaMalloc(&mut new_buf, new_capacity_bytes) };
    if rc != CUDA_SUCCESS {
        return Err(CudaBitmapError::Cuda {
            what: "cudaMalloc(v3 workbench)",
            code: rc,
        });
    }
    // SAFETY: caller holds the workbench mutex; old buffer not
    // accessible to other threads.
    unsafe {
        if !workbench.d_buf.is_null() {
            let _ = cudaFree(workbench.d_buf);
        }
    }
    workbench.d_buf = new_buf;
    workbench.capacity_bytes = new_capacity_bytes;
    Ok(())
}

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

/// Scatter a term's cached buckets onto a destination device buffer
/// formatted for the cohort `union_keys` layout (Phase 2 D-3 v2).
///
/// Pre-zero-fills `dst` with [`cudaMemsetAsync`] so buckets present in
/// other cohort members but absent in this term contribute zeros to
/// the bitwise op. Then issues one
/// [`cudaMemcpyAsync`]`(DeviceToDevice)` per `(high16, word_offset)`
/// pair in `term.bucket_index()` whose `high16` is found by binary
/// search in `union_keys`. All ops queue on `stream` and are only
/// guaranteed visible after the caller's
/// [`cudaStreamSynchronize`].
///
/// # Safety
/// - `dst` must point to at least `dst_bytes` of writable device memory
///   on the device that owns `stream`.
/// - `term.device_ptr()` must point to a valid `[u32; total_words]`
///   device allocation owned by `term` (guaranteed by
///   [`VramCht::build_entry`]).
/// - `union_keys` must be sorted ascending (binary-search invariant).
/// - The caller must hold the `Arc<VramTermEntry>` alive until the
///   stream sync completes (otherwise the term's `Drop` could
///   `cudaFree` the source buffer mid-copy).
unsafe fn scatter_term_to_device(
    dst: *mut c_void,
    dst_bytes: usize,
    term: &Arc<VramTermEntry>,
    union_keys: &[u16],
    stream: cudaStream_t,
) -> Result<(), CudaBitmapError> {
    // Zero the destination first so missing buckets contribute zeros.
    let rc = unsafe {
        cudaMemsetAsync(dst, 0i32 as std::ffi::c_int, dst_bytes, stream)
    };
    if rc != CUDA_SUCCESS {
        return Err(CudaBitmapError::Cuda {
            what: "cudaMemsetAsync(scatter zero-fill)",
            code: rc,
        });
    }
    let bucket_bytes = BITMAP_CONTAINER_WORDS * std::mem::size_of::<u32>();
    for (high16, src_word_off) in term.bucket_index() {
        let Ok(dst_idx) = union_keys.binary_search(high16) else {
            // Bucket not in union — skip. Defensive against malformed
            // inputs; correct cohorts produced by union_high16_keys
            // include every term's buckets.
            continue;
        };
        let dst_word_off = dst_idx * BITMAP_CONTAINER_WORDS;
        // SAFETY: byte offsets stay within `dst_bytes` by construction
        // (`dst_idx < union_keys.len()` ⇒ `dst_word_off +
        // BITMAP_CONTAINER_WORDS ≤ words_per_term`). Source: `src_word_off
        // + BITMAP_CONTAINER_WORDS ≤ term.total_words` by [`VramCht::build_entry`]
        // invariant.
        let dst_ptr = unsafe {
            (dst as *mut u8).add(dst_word_off * std::mem::size_of::<u32>())
        };
        let src_ptr = unsafe {
            (term.device_ptr() as *const u8)
                .add((*src_word_off as usize) * std::mem::size_of::<u32>())
        };
        let rc = unsafe {
            cudaMemcpyAsync(
                dst_ptr as *mut c_void,
                src_ptr as *const c_void,
                bucket_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            )
        };
        if rc != CUDA_SUCCESS {
            return Err(CudaBitmapError::Cuda {
                what: "cudaMemcpyAsync(scatter D2D bucket)",
                code: rc,
            });
        }
    }
    Ok(())
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

    // ============================================================
    // VRAM fold tests (Phase 2 D-3 v2)
    // ============================================================

    /// Helper: insert a `RoaringPostings` into a fresh `VramCht` and
    /// return the `Arc<VramTermEntry>`. Returns None if CUDA insert
    /// fails (driver-missing host). `term_hash` is the
    /// disambiguator analogous to the historical `addr` parameter —
    /// tests pass any distinct u64 to get distinct keys.
    fn vram_term_entry(
        rp: &crate::postings::roaring::encoder::RoaringPostings,
        cache: &crate::postings::roaring::vram_cht::VramCht,
        term_hash: u64,
    ) -> Option<Arc<crate::postings::roaring::vram_cht::VramTermEntry>> {
        let key = crate::postings::roaring::cht::ChtKey {
            segment_id: crate::index::SegmentId::generate_random(),
            field: 0,
            term_hash,
        };
        let inserted = cache.insert(key.clone(), rp).ok()?;
        if !inserted {
            return None;
        }
        cache.get(&key)
    }

    #[test]
    fn vram_fold_empty_returns_empty() {
        let Some(kernel) = try_kernel() else { return };
        let union_keys: [u16; 1] = [0];
        let res = kernel
            .compute_fold_vram(BoolOp::And, &[], &union_keys)
            .unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn vram_fold_empty_union_returns_empty() {
        let Some(kernel) = try_kernel() else { return };
        let cache = crate::postings::roaring::vram_cht::VramCht::with_budget(64 * 1024 * 1024);
        let rp = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[1, 2, 3]);
        let Some(t1) = vram_term_entry(&rp, &cache, 0xa1) else { return };
        let Some(t2) = vram_term_entry(&rp, &cache, 0xa2) else { return };
        let res = kernel
            .compute_fold_vram(BoolOp::And, &[t1, t2], &[])
            .unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn vram_fold_single_term_round_trips_layout() {
        let Some(kernel) = try_kernel() else { return };
        let cache = crate::postings::roaring::vram_cht::VramCht::with_budget(64 * 1024 * 1024);
        // Two-bucket term so we cross a high16 boundary in the union.
        let docs: Vec<u32> = (0..3).chain(std::iter::once(65540)).collect();
        let rp = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&docs);
        let Some(term) = vram_term_entry(&rp, &cache, 0xfeed) else { return };
        let union_keys: [u16; 2] = [0, 1];
        let got = kernel
            .compute_fold_vram(BoolOp::And, &[term], &union_keys)
            .unwrap();
        // Build the expected flat layout via the host-side oracle (same
        // shape as flat_buffer_for_term in gpu_dispatch.rs).
        let mut expected = vec![0u32; union_keys.len() * BITMAP_CONTAINER_WORDS];
        for (high16, container) in &rp.containers {
            if container.cardinality() == 0 {
                continue;
            }
            let Ok(idx) = union_keys.binary_search(high16) else {
                continue;
            };
            let start = idx * BITMAP_CONTAINER_WORDS;
            let end = start + BITMAP_CONTAINER_WORDS;
            match container {
                crate::postings::roaring::Container::Bitmap(bm) => {
                    expected[start..end].copy_from_slice(bm.words.as_ref());
                }
                crate::postings::roaring::Container::Array(arr) => {
                    let bm = crate::postings::roaring::BitmapContainer::from_array(arr);
                    expected[start..end].copy_from_slice(bm.words.as_ref());
                }
                crate::postings::roaring::Container::Run(rc) => {
                    let bm = crate::postings::roaring::BitmapContainer::from_run(rc);
                    expected[start..end].copy_from_slice(bm.words.as_ref());
                }
            }
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn vram_fold_matches_compute_fold_oracle() {
        let Some(kernel) = try_kernel() else { return };
        let cache = crate::postings::roaring::vram_cht::VramCht::with_budget(64 * 1024 * 1024);
        // Three terms with overlapping bucket sets — exercises the
        // scatter scatter-zero-fill-and-copy path for buckets present
        // in some terms but not others.
        let t0 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[
            1, 2, 3, 65540, 65541,
        ]);
        let t1 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[
            2, 3, 4, 65541, 65542,
        ]);
        let t2 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[
            3, 4, 5, 65540, 65541,
        ]);
        let Some(v0) = vram_term_entry(&t0, &cache, 0xa1) else { return };
        let Some(v1) = vram_term_entry(&t1, &cache, 0xa2) else { return };
        let Some(v2) = vram_term_entry(&t2, &cache, 0xa3) else { return };

        let term_refs: Vec<&crate::postings::roaring::RoaringPostings> = vec![&t0, &t1, &t2];
        let union_keys =
            crate::postings::roaring::gpu_dispatch::union_high16_keys_for_test(
                &term_refs,
            );

        // Host oracle: flat buffers via flat_buffer_for_term + sequential
        // compute() folds (same as the existing host fold path).
        let host_bufs: Vec<Vec<u32>> = term_refs
            .iter()
            .map(|t| {
                crate::postings::roaring::gpu_dispatch::flat_buffer_for_term_for_test(
                    t,
                    &union_keys,
                )
            })
            .collect();
        let host_refs: Vec<&[u32]> = host_bufs.iter().map(|v| v.as_slice()).collect();
        let oracle = kernel.compute_fold(BoolOp::And, &host_refs).unwrap();

        let vram_terms = vec![v0, v1, v2];
        let got = kernel
            .compute_fold_vram(BoolOp::And, &vram_terms, &union_keys)
            .unwrap();

        assert_eq!(
            got, oracle,
            "VRAM fold result must equal host-fold oracle byte-for-byte"
        );
    }

    #[test]
    fn vram_fold_or_xor_match_oracle() {
        let Some(kernel) = try_kernel() else { return };
        let cache = crate::postings::roaring::vram_cht::VramCht::with_budget(64 * 1024 * 1024);
        let t0 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[1, 2, 3]);
        let t1 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[3, 4, 5]);
        let Some(v0) = vram_term_entry(&t0, &cache, 0xb1) else { return };
        let Some(v1) = vram_term_entry(&t1, &cache, 0xb2) else { return };
        let term_refs: Vec<&crate::postings::roaring::RoaringPostings> = vec![&t0, &t1];
        let union_keys =
            crate::postings::roaring::gpu_dispatch::union_high16_keys_for_test(
                &term_refs,
            );

        for op in [BoolOp::Or, BoolOp::Xor] {
            let host_bufs: Vec<Vec<u32>> = term_refs
                .iter()
                .map(|t| {
                    crate::postings::roaring::gpu_dispatch::flat_buffer_for_term_for_test(
                        t,
                        &union_keys,
                    )
                })
                .collect();
            let host_refs: Vec<&[u32]> = host_bufs.iter().map(|v| v.as_slice()).collect();
            let oracle = kernel.compute_fold(op, &host_refs).unwrap();
            let got = kernel
                .compute_fold_vram(op, &vec![Arc::clone(&v0), Arc::clone(&v1)], &union_keys)
                .unwrap();
            assert_eq!(got, oracle, "{op:?} VRAM fold must match host oracle");
        }
    }

    /// Helper: insert a `RoaringPostings` into a fresh `VramCompressedCht`
    /// and return the `Arc<VramCompressedTermEntry>`. Returns None if
    /// CUDA insert fails (driver-missing host). `term_hash` is the
    /// disambiguator analogous to the historical `addr` parameter.
    fn vram_v3_term_entry(
        rp: &crate::postings::roaring::encoder::RoaringPostings,
        cache: &crate::postings::roaring::vram_cht_v3::VramCompressedCht,
        term_hash: u64,
    ) -> Option<Arc<crate::postings::roaring::vram_cht_v3::VramCompressedTermEntry>> {
        let key = crate::postings::roaring::cht::ChtKey {
            segment_id: crate::index::SegmentId::generate_random(),
            field: 0,
            term_hash,
        };
        let inserted = cache.insert(key.clone(), rp).ok()?;
        if !inserted {
            return None;
        }
        cache.get(&key)
    }

    #[test]
    fn v3_fold_empty_returns_empty() {
        let Some(kernel) = try_kernel() else { return };
        let union_keys: [u16; 1] = [0];
        let res = kernel
            .compute_fold_v3(BoolOp::And, &[], &union_keys)
            .unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn v3_fold_matches_v2_fold_oracle() {
        let Some(kernel) = try_kernel() else { return };
        let cache_v2 =
            crate::postings::roaring::vram_cht::VramCht::with_budget(64 * 1024 * 1024);
        let cache_v3 =
            crate::postings::roaring::vram_cht_v3::VramCompressedCht::with_budget(
                64 * 1024 * 1024,
            )
            .ok();
        let Some(cache_v3) = cache_v3 else { return };
        let t0 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[
            1, 2, 3, 65540, 65541,
        ]);
        let t1 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[
            2, 3, 4, 65541, 65542,
        ]);
        let t2 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[
            3, 4, 5, 65540, 65541,
        ]);
        let Some(v2_t0) = vram_term_entry(&t0, &cache_v2, 0xa1) else { return };
        let Some(v2_t1) = vram_term_entry(&t1, &cache_v2, 0xa2) else { return };
        let Some(v2_t2) = vram_term_entry(&t2, &cache_v2, 0xa3) else { return };
        let Some(v3_t0) = vram_v3_term_entry(&t0, &cache_v3, 0xb1) else { return };
        let Some(v3_t1) = vram_v3_term_entry(&t1, &cache_v3, 0xb2) else { return };
        let Some(v3_t2) = vram_v3_term_entry(&t2, &cache_v3, 0xb3) else { return };

        let term_refs: Vec<&crate::postings::roaring::RoaringPostings> = vec![&t0, &t1, &t2];
        let union_keys =
            crate::postings::roaring::gpu_dispatch::union_high16_keys_for_test(
                &term_refs,
            );

        let v2_terms = vec![v2_t0, v2_t1, v2_t2];
        let v3_terms = vec![v3_t0, v3_t1, v3_t2];

        for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor] {
            let v2_result = kernel
                .compute_fold_vram(op, &v2_terms, &union_keys)
                .expect("v2 fold");
            let v3_result = kernel
                .compute_fold_v3(op, &v3_terms, &union_keys)
                .expect("v3 fold");
            assert_eq!(
                v2_result, v3_result,
                "{op:?} v3 fold (decompressed) must equal v2 fold byte-for-byte"
            );
        }
    }

    #[test]
    fn v3_fold_multi_chunk_matches_single_chunk_oracle() {
        // Wave Z-6 #3 acceptance gate: a cohort containing a
        // multi-chunk Bitcomp entry (≥ 2 chunks of 16 MiB each) must
        // fold byte-identically to the CPU oracle that materialises
        // each term in cohort layout and applies the op per-word.
        // The multi-chunk path exercises the M × N_i flatten in
        // `decompress_batch_into_workbench`; the single-chunk partner
        // covers the degenerate `N_i = 1` case in the same call.
        let Some(kernel) = try_kernel() else { return };
        let cache = crate::postings::roaring::vram_cht_v3::VramCompressedCht::with_budget(
            256 * 1024 * 1024,
        )
        .ok();
        let Some(cache) = cache else { return };

        // Force a 2-chunk admission: BUCKETS_PER_CHUNK * 2 unique
        // high16 values (each with one doc-id), producing a 32-MiB-
        // equivalent uncompressed posting that splits across two
        // Bitcomp chunks. The companion term is a small single-chunk
        // posting that overlaps a few of the multi-chunk term's
        // buckets so AND / OR / XOR all see non-trivial bit patterns.
        let n_multi_buckets =
            crate::postings::roaring::vram_cht_v3::BUCKETS_PER_CHUNK * 2;
        let docs_multi: Vec<u32> =
            (0..n_multi_buckets).map(|i| (i as u32) << 16).collect();
        let rp_multi =
            crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(
                &docs_multi,
            );
        // Single-chunk partner: a handful of doc-ids inside the first
        // bucket of the multi-chunk term plus one bucket past the
        // chunk boundary, so the AND fold has a few set bits and the
        // OR / XOR folds touch multiple high16 buckets.
        let single_high16 = (crate::postings::roaring::vram_cht_v3::BUCKETS_PER_CHUNK
            as u32
            + 1)
            << 16;
        let rp_single =
            crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&[
                0,
                1 << 16,
                single_high16,
            ]);

        let Some(v3_multi) = vram_v3_term_entry(&rp_multi, &cache, 0xc6_03_01)
        else {
            return;
        };
        let Some(v3_single) = vram_v3_term_entry(&rp_single, &cache, 0xc6_03_02)
        else {
            return;
        };

        // Pin the multi-chunk fixture invariant — if admission ever
        // changes such that this test stops exercising N_i ≥ 2, the
        // assertion below fails loud so the test isn't silently
        // degraded into a single-chunk-only run.
        assert!(
            v3_multi.chunk_count() >= 2,
            "test fixture must exercise multi-chunk dispatch (got chunk_count={})",
            v3_multi.chunk_count()
        );
        assert_eq!(v3_single.chunk_count(), 1);

        let term_refs: Vec<&crate::postings::roaring::RoaringPostings> =
            vec![&rp_multi, &rp_single];
        let union_keys =
            crate::postings::roaring::gpu_dispatch::union_high16_keys_for_test(
                &term_refs,
            );
        let v3_terms = vec![v3_multi, v3_single];

        // CPU oracle: materialise each term's flat buffer in cohort
        // layout once, then apply each op per-word.
        let buf_multi =
            crate::postings::roaring::gpu_dispatch::flat_buffer_for_term_for_test(
                &rp_multi,
                &union_keys,
            );
        let buf_single =
            crate::postings::roaring::gpu_dispatch::flat_buffer_for_term_for_test(
                &rp_single,
                &union_keys,
            );

        for op in [BoolOp::And, BoolOp::Or, BoolOp::Xor] {
            let v3_result = kernel
                .compute_fold_v3(op, &v3_terms, &union_keys)
                .expect("multi-chunk v3 fold");
            let expected: Vec<u32> = buf_multi
                .iter()
                .zip(buf_single.iter())
                .map(|(&a, &b)| match op {
                    BoolOp::And => a & b,
                    BoolOp::Or => a | b,
                    BoolOp::Xor => a ^ b,
                })
                .collect();
            assert_eq!(
                v3_result, expected,
                "{op:?} multi-chunk v3 fold must equal CPU oracle byte-for-byte"
            );
        }
    }

    #[test]
    fn v3_fold_single_term_round_trips_layout() {
        let Some(kernel) = try_kernel() else { return };
        let cache = crate::postings::roaring::vram_cht_v3::VramCompressedCht::with_budget(
            64 * 1024 * 1024,
        )
        .ok();
        let Some(cache) = cache else { return };
        let docs: Vec<u32> = (0..3).chain(std::iter::once(65540)).collect();
        let rp = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&docs);
        let Some(term) = vram_v3_term_entry(&rp, &cache, 0xfeed) else { return };
        let union_keys: [u16; 2] = [0, 1];
        let got = kernel
            .compute_fold_v3(BoolOp::And, &[term], &union_keys)
            .unwrap();
        // Compare to host-oracle: build expected flat layout using
        // the same flat_buffer_for_term as v2's oracle test.
        let expected = crate::postings::roaring::gpu_dispatch::flat_buffer_for_term_for_test(
            &rp,
            &union_keys,
        );
        assert_eq!(got, expected);
    }

    #[test]
    fn vram_fold_capacity_growth_on_large_union() {
        let Some(kernel) = try_kernel() else { return };
        let cache = crate::postings::roaring::vram_cht::VramCht::with_budget(1 << 30);
        // Build terms whose union covers many buckets, forcing the
        // kernel buffers to grow past INITIAL_CAPACITY_WORDS.
        // 24 buckets × BITMAP_CONTAINER_WORDS = 49 152 words > initial
        // 24 576 — triggers reallocate_buffers.
        let docs1: Vec<u32> = (0..24).map(|b| b * 65536).collect();
        let docs2: Vec<u32> = (0..24).map(|b| b * 65536 + 1).collect();
        let rp1 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&docs1);
        let rp2 = crate::postings::roaring::encoder::RoaringEncoder::from_doc_ids(&docs2);
        let Some(v1) = vram_term_entry(&rp1, &cache, 0xc1) else { return };
        let Some(v2) = vram_term_entry(&rp2, &cache, 0xc2) else { return };
        let union_keys: Vec<u16> = (0..24).collect();
        let res = kernel
            .compute_fold_vram(BoolOp::Or, &[v1, v2], &union_keys)
            .unwrap();
        assert_eq!(res.len(), 24 * BITMAP_CONTAINER_WORDS);
        // Sanity: at least the doc-id 0 (bit 0) and doc-id 1 (bit 1) of
        // bucket 0 are set after OR.
        assert!(res[0] & 0b11 == 0b11, "OR result must have bits 0 and 1 set in bucket 0");
    }
}

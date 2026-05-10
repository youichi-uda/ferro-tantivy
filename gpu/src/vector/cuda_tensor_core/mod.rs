//! NVIDIA CUDA Tensor Core fast path for binary Hamming distance.
//!
//! This module is gated behind the `cuda-tensor-core` Cargo feature and
//! is **off by default**. When enabled it offers a parallel path for
//! [`BinaryDistanceKernel::compute_batched`](super::binary_distance::BinaryDistanceKernel::compute_batched)
//! that runs on cuBLASLt INT8 IMMA tensor cores instead of the
//! cross-platform WGSL `countOneBits(xor)` shader.
//!
//! ## Algorithm
//!
//! The driver insight (Wave 4 G1 verdict, see
//! `docs/ADR-001-cuda-backend.md`) is that binary Hamming distance has a
//! linear-algebra reformulation:
//!
//! ```text
//!     popcount(q ⊕ d) = popcount(q) + popcount(d) − 2 · ⟨q, d⟩
//! ```
//!
//! With one byte per bit (`q, d ∈ {0, 1}`), `⟨q, d⟩` is an integer dot
//! product that an NVIDIA Ampere/Ada Tensor Core can compute as a dense
//! INT8 matmul orders of magnitude faster than `countOneBits(q ^ d)` on
//! the same hardware. The host pre-computes `popcount(q)` and
//! `popcount(d)` in a single `count_ones()` pass; the device runs:
//!
//! 1. `inner = Q · Dᵀ`  via `cublasLtMatmul` with `CUDA_R_8I` inputs and
//!    `CUDA_R_32I` accumulator (compute type `CUBLAS_COMPUTE_32I`).
//! 2. `out[m, n] = pop_q[m] + pop_d[n] − 2 · inner[m, n]` via a small
//!    NVRTC-compiled element-wise kernel.
//!
//! INT8 with INT32 accumulator is **bit-exact** vs the WGSL/CPU oracle
//! for any `dim_bits ≤ 2^31`, so the parity tests assert byte-equal
//! output across the same `Q × N` grid the WGSL kernel is tested on.
//!
//! ## When this path is taken
//!
//! - `BinaryDistanceKernel::compute_batched` first tries to construct a
//!   [`CudaTensorCoreKernel`]. If construction succeeds (CUDA driver +
//!   cuBLASLt + an NVIDIA device + NVRTC are all available at runtime)
//!   it routes the dispatch through this module.
//! - Construction failure (no CUDA, no NVIDIA device, missing libraries,
//!   container without `/dev/nvidia*`) returns
//!   [`GpuError::CpuFallback`](crate::error::GpuError::CpuFallback) with
//!   a non-fatal reason; the calling code falls back to the WGSL
//!   pipeline.
//! - The construction result is cached (`OnceLock`) for the life of the
//!   `BinaryDistanceKernel`, so the cost is paid once.
//!
//! ## Limits
//!
//! Same as the WGSL batched path:
//! `dim_u32 ≤ BATCHED_MAX_DIM_U32` (= 64 → `dim_bits ≤ 2048`). Larger
//! dimensions return `CpuFallback` so the caller can chunk or use the
//! single-query kernel.

mod kernel;
mod pinned;
mod popcount;

use std::sync::Arc;

use crate::error::{GpuError, GpuResult};
use kernel::{CachedCorpus, CudaGemmRunner};
pub use pinned::PinnedU32Buffer;
use popcount::popcount_per_vec;

/// CUDA Tensor Core kernel for binary Hamming distance.
///
/// One instance owns a cuBLASLt handle, a 4 MiB GEMM workspace, and an
/// NVRTC-compiled correction kernel; reuse across calls amortises all
/// three.
///
/// Construct via [`Self::try_new`] — returns
/// [`GpuError::CpuFallback`](crate::error::GpuError::CpuFallback) when
/// no NVIDIA device is reachable, which is the expected outcome on AMD /
/// Apple / Intel / containers without GPU passthrough.
pub struct CudaTensorCoreKernel {
    inner: Arc<CudaGemmRunner>,
}

impl CudaTensorCoreKernel {
    /// Attempt to build a CUDA Tensor Core kernel against device 0.
    ///
    /// On success the device, the cuBLASLt handle, and the NVRTC kernel
    /// are all loaded and ready. On failure the returned error is the
    /// non-fatal [`GpuError::CpuFallback`](crate::error::GpuError::CpuFallback) variant.
    pub fn try_new() -> GpuResult<Self> {
        let inner = CudaGemmRunner::try_new()?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Compute the row-major `Q × N` Hamming-distance matrix using the
    /// CUDA Tensor Core path. Inputs and outputs use the same calling
    /// contract as
    /// [`BinaryDistanceKernel::compute_batched`](super::binary_distance::BinaryDistanceKernel::compute_batched)
    /// so callers can substitute one for the other transparently.
    pub fn compute_batched(
        &self,
        queries_bits: &[u32],
        corpus_bits: &[u32],
        num_queries: usize,
        num_vecs: usize,
        dim_bits: usize,
    ) -> GpuResult<Vec<u32>> {
        let dim_u32 = dim_bits.div_ceil(32);

        if num_queries == 0 || num_vecs == 0 || dim_bits == 0 {
            return Ok(Vec::new());
        }

        if queries_bits.len() != num_queries * dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("queries_bits.len() == {}", num_queries * dim_u32),
                actual: format!("queries_bits.len() == {}", queries_bits.len()),
            });
        }
        if corpus_bits.len() != num_vecs * dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("corpus_bits.len() == {}", num_vecs * dim_u32),
                actual: format!("corpus_bits.len() == {}", corpus_bits.len()),
            });
        }

        // Host-side prep: just per-row popcounts. Bit unpacking is
        // done on the device by the NVRTC `unpack_bits` kernel so we
        // upload the bit-packed buffer (1 bit/bit) instead of the
        // unpacked one (1 byte/bit).
        let pop_q = popcount_per_vec(queries_bits, num_queries, dim_u32);
        let pop_d = popcount_per_vec(corpus_bits, num_vecs, dim_u32);

        self.inner.run(
            queries_bits,
            corpus_bits,
            &pop_q,
            &pop_d,
            num_queries,
            num_vecs,
            dim_bits,
        )
    }

    /// Upload the corpus + per-row popcount once and return a
    /// [`CudaBinaryCorpus`] handle. Subsequent
    /// [`CudaBinaryCorpus::compute_batched`] calls skip the
    /// upload + unpack + popcount stages — the only PCIe traffic per
    /// query batch is the (small) bit-packed query buffer and the
    /// `Q × N × 4 B` result download.
    ///
    /// This is the production pattern: BBQ corpora are immutable per
    /// segment, so an HNSW search loop over the same segment uploads
    /// the corpus once and re-uses it across many query batches.
    /// Wave 4's reference 6-7× speedup at the N = 1 M shape assumed
    /// exactly this calling pattern.
    pub fn prepare_corpus(
        &self,
        corpus_bits: &[u32],
        num_vecs: usize,
        dim_bits: usize,
    ) -> GpuResult<CudaBinaryCorpus> {
        let dim_u32 = dim_bits.div_ceil(32);
        if num_vecs == 0 || dim_bits == 0 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: "num_vecs > 0 && dim_bits > 0".to_string(),
                actual: format!("num_vecs = {num_vecs}, dim_bits = {dim_bits}"),
            });
        }
        if corpus_bits.len() != num_vecs * dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("corpus_bits.len() == {}", num_vecs * dim_u32),
                actual: format!("corpus_bits.len() == {}", corpus_bits.len()),
            });
        }
        let pop_d = popcount_per_vec(corpus_bits, num_vecs, dim_u32);
        let cached = self
            .inner
            .prepare_corpus(corpus_bits, num_vecs, dim_bits, &pop_d)?;
        Ok(CudaBinaryCorpus { inner: cached })
    }

    /// One-shot end-to-end k-NN search: equivalent to
    /// `prepare_corpus(...).knn_search(...)`. Use the cached form when
    /// running multiple query batches against the same corpus.
    pub fn knn_search(
        &self,
        queries_bits: &[u32],
        corpus_bits: &[u32],
        num_queries: usize,
        num_vecs: usize,
        dim_bits: usize,
        k: usize,
    ) -> GpuResult<Vec<Vec<(u32, u32)>>> {
        if num_queries == 0 || num_vecs == 0 || k == 0 {
            return Ok(vec![Vec::new(); num_queries]);
        }
        let cached = self.prepare_corpus(corpus_bits, num_vecs, dim_bits)?;
        cached.knn_search(queries_bits, num_queries, k)
    }

    /// Allocate a page-locked (pinned) host buffer of `len` `u32`
    /// values, suitable as a destination for
    /// [`CudaBinaryCorpus::compute_batched_into_pinned`]. A pinned
    /// buffer DMAs directly from the device at full PCIe bandwidth,
    /// roughly halving result-download wall-clock at the
    /// `Q = 64, N = 1 M, dim = 768` headline shape (≈ 256 MB result
    /// matrix) compared with the pageable `Vec<u32>` API.
    pub fn alloc_pinned_u32(&self, len: usize) -> GpuResult<PinnedU32Buffer> {
        PinnedU32Buffer::new(self.inner.ctx(), len)
    }
}

/// On-device cached corpus produced by
/// [`CudaTensorCoreKernel::prepare_corpus`]. Owns the unpacked `i8`
/// corpus matrix and the per-row popcount on the GPU; queries against
/// it are short — only the per-call `Q × dim_u32 × 4 B` query upload +
/// `Q × N × 4 B` result download cross PCIe.
pub struct CudaBinaryCorpus {
    inner: CachedCorpus,
}

impl CudaBinaryCorpus {
    /// Number of corpus vectors held on the device.
    pub fn num_vecs(&self) -> usize {
        self.inner.num_vecs
    }

    /// Bit-width the corpus was prepared with. Subsequent query
    /// batches must use the same `dim_bits`.
    pub fn dim_bits(&self) -> usize {
        self.inner.dim_bits
    }

    /// Compute Hamming distances between a batch of queries and the
    /// resident corpus. Output is row-major
    /// `num_queries × num_vecs`, identical in shape and value to
    /// [`CudaTensorCoreKernel::compute_batched`].
    pub fn compute_batched(
        &self,
        queries_bits: &[u32],
        num_queries: usize,
    ) -> GpuResult<Vec<u32>> {
        let dim_u32 = self.inner.dim_bits.div_ceil(32);
        if num_queries == 0 {
            return Ok(Vec::new());
        }
        if queries_bits.len() != num_queries * dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("queries_bits.len() == {}", num_queries * dim_u32),
                actual: format!("queries_bits.len() == {}", queries_bits.len()),
            });
        }
        let pop_q = popcount_per_vec(queries_bits, num_queries, dim_u32);
        self.inner
            .runner
            .run_with_cached_corpus(queries_bits, &pop_q, &self.inner, num_queries)
    }

    /// Pinned-output overload of [`Self::compute_batched`]. The result
    /// matrix is written directly into the user-supplied
    /// [`PinnedU32Buffer`]; only the small popcount + query upload and
    /// the device-side compute need to run before the host can read
    /// `out.as_slice()`. Allocate the buffer once (typically sized to
    /// `max_query_batch * num_vecs`) via
    /// [`CudaTensorCoreKernel::alloc_pinned_u32`] and reuse it across
    /// the search loop.
    ///
    /// The buffer must hold at least `num_queries * num_vecs`
    /// elements; smaller buffers are rejected up-front.
    pub fn compute_batched_into_pinned(
        &self,
        queries_bits: &[u32],
        num_queries: usize,
        out: &mut PinnedU32Buffer,
    ) -> GpuResult<()> {
        let dim_u32 = self.inner.dim_bits.div_ceil(32);
        if num_queries == 0 {
            return Ok(());
        }
        if queries_bits.len() != num_queries * dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("queries_bits.len() == {}", num_queries * dim_u32),
                actual: format!("queries_bits.len() == {}", queries_bits.len()),
            });
        }
        let pop_q = popcount_per_vec(queries_bits, num_queries, dim_u32);
        self.inner.runner.run_with_cached_corpus_into_pinned(
            queries_bits,
            &pop_q,
            &self.inner,
            num_queries,
            out,
        )
    }

    /// End-to-end CUDA k-NN search: compute the Q × N distance matrix
    /// on-device and run the bitonic-merge top-K reducer in CUDA, so
    /// only `Q × K × 8 B` crosses PCIe back regardless of `N`.
    ///
    /// `k` is clamped to `min(k, num_vecs)` and rounded up internally
    /// to the next power of two for the bitonic merge. `k` must
    /// satisfy `1 ≤ k ≤ 1024` (matches
    /// [`crate::vector::binary_distance::TOP_K_MAX_K_PADDED`]).
    ///
    /// Returns one `Vec<(distance, corpus_id)>` per query, sorted
    /// ascending by distance, ties broken by lower id (matches the
    /// WGSL `top_k_select.wgsl` ordering exactly).
    pub fn knn_search(
        &self,
        queries_bits: &[u32],
        num_queries: usize,
        k: usize,
    ) -> GpuResult<Vec<Vec<(u32, u32)>>> {
        if num_queries == 0 || k == 0 {
            return Ok(vec![Vec::new(); num_queries]);
        }
        let dim_u32 = self.inner.dim_bits.div_ceil(32);
        if queries_bits.len() != num_queries * dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("queries_bits.len() == {}", num_queries * dim_u32),
                actual: format!("queries_bits.len() == {}", queries_bits.len()),
            });
        }

        let n = self.inner.num_vecs;
        let k_eff = k.min(n);
        const MAX_K_PADDED: u32 = 1024;
        if (k_eff as u32) > MAX_K_PADDED {
            return Err(GpuError::CpuFallback {
                reason: format!(
                    "knn_search: k = {k_eff} exceeds CUDA top-K shader limit {MAX_K_PADDED}"
                ),
            });
        }
        let k_padded = next_pow2_u32(k_eff as u32).max(2);

        let pop_q = popcount_per_vec(queries_bits, num_queries, dim_u32);
        let (dists, ids) = self.inner.runner.knn_search_with_cached_corpus(
            queries_bits,
            &pop_q,
            &self.inner,
            num_queries,
            k_eff,
            k_padded,
        )?;

        let mut out = Vec::with_capacity(num_queries);
        for q in 0..num_queries {
            let mut row = Vec::with_capacity(k_eff);
            for w in 0..k_eff {
                let idx = q * k_eff + w;
                row.push((dists[idx], ids[idx]));
            }
            out.push(row);
        }
        Ok(out)
    }
}

/// Round `n` up to the next power of two (≥ 1). Mirrors
/// `binary_distance::next_pow2_u32` — kept private here to avoid the
/// WGSL-side dependency creating a public API surface.
fn next_pow2_u32(n: u32) -> u32 {
    if n <= 1 {
        return 1;
    }
    let mut p = 1u32;
    while p < n {
        p <<= 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::binary_distance::{dim_u32_for, hamming_distances_batched_cpu};

    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }
    fn random_u32_vec(n: usize, seed: u64) -> Vec<u32> {
        let mut state = seed;
        (0..n).map(|_| xorshift64(&mut state) as u32).collect()
    }

    #[test]
    fn smoke_cached_corpus_byte_equal() {
        let kernel = match CudaTensorCoreKernel::try_new() {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping CUDA cached-corpus smoke: {e}");
                return;
            }
        };
        let dim_bits = 256;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_queries = 4;
        let num_vecs = 32;
        let queries = random_u32_vec(num_queries * dim_u32, 0xfeed_face_dead_beef);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xc0de_d00d_b00b_5eed);

        let cpu = hamming_distances_batched_cpu(
            &queries, &corpus, num_queries, num_vecs, dim_u32,
        );

        let cached = kernel
            .prepare_corpus(&corpus, num_vecs, dim_bits)
            .expect("prepare_corpus");
        assert_eq!(cached.num_vecs(), num_vecs);
        assert_eq!(cached.dim_bits(), dim_bits);

        // First call.
        let r1 = cached
            .compute_batched(&queries, num_queries)
            .expect("first cached call");
        assert_eq!(r1, cpu, "cached: first call must match CPU oracle");

        // Second call against the same corpus must return identical
        // bytes — proves the device-side state isn't corrupted.
        let r2 = cached
            .compute_batched(&queries, num_queries)
            .expect("second cached call");
        assert_eq!(r2, r1, "cached: repeated call must be byte-equal");
    }

    /// Lightweight smoke test that runs only on hosts with a CUDA
    /// driver. The full parity sweep lives in
    /// `tests/cuda_tensor_core_parity.rs`.
    #[test]
    fn smoke_byte_equal() {
        let kernel = match CudaTensorCoreKernel::try_new() {
            Ok(k) => k,
            Err(e) => {
                eprintln!("skipping CUDA smoke test: {e}");
                return;
            }
        };

        let dim_bits = 256;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_queries = 4;
        let num_vecs = 32;
        let queries = random_u32_vec(num_queries * dim_u32, 0xa1a2a3a4_b1b2b3b4);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xc1c2c3c4_d1d2d3d4);

        let cpu = hamming_distances_batched_cpu(
            &queries, &corpus, num_queries, num_vecs, dim_u32,
        );
        let gpu = kernel
            .compute_batched(&queries, &corpus, num_queries, num_vecs, dim_bits)
            .expect("compute_batched");

        assert_eq!(gpu, cpu, "CUDA path must be byte-equal to CPU oracle");
    }
}

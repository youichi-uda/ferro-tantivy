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
mod popcount;

use std::sync::Arc;

use crate::error::{GpuError, GpuResult};
use kernel::CudaGemmRunner;
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

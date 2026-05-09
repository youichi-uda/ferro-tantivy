//! GPU-accelerated binary (1-bit) Hamming distance for BBQ-style
//! nearest-neighbour search.
//!
//! ## What this is
//!
//! BBQ (Binary Block Quantization) and related schemes compress f32
//! embedding vectors down to 1 bit per dimension by storing the sign
//! (or, in paper-grade RaBitQ, a randomly-rotated sign). Distance
//! between two binary codes is the **Hamming distance** — the number
//! of bit positions where they differ — computed efficiently as
//! `popcount(a XOR b)`.
//!
//! On CPU, modern x86_64 / aarch64 cores execute `popcnt` /
//! `cnt` in a single cycle and roofline at memory bandwidth
//! (~30-50 GB/s on a high-end desktop, ~10-20 GB/s on a laptop). On
//! GPU, the same operation is bandwidth-bound by VRAM read throughput
//! (RTX 4070 Ti SUPER: 800 GB/s peak, ~50-100 GB/s effective for
//! fully-random scatter reads through a kernel like this), giving a
//! theoretical 5-10× over CPU on a single warm batch.
//!
//! ### Throughput estimate
//!
//! For `dim_bits = 768`, `dim_u32 = 24`, `N = 1_000_000` corpus vectors:
//! - corpus size = `N * dim_u32 * 4 B` = ~96 MB
//! - at 50 GB/s effective the GPU reads the corpus in ~2 ms
//! - per-vector work = 24 XOR + 24 popcount + 24 add = trivially overlaps
//!   memory latency, so kernel time ≈ memory time ≈ 2 ms.
//! - on CPU at 40 GB/s, same scan is ~2.4 ms — GPU win is small for a
//!   single query, but **batched-query** workloads (`Q` queries amortise
//!   the corpus read once into shared memory / L2) push the GPU to
//!   10-50× over CPU. Batched-query layout is a Phase 2 B item.
//!
//! ## Scope of this file (Phase 2 A foundation)
//!
//! Phase 2 A delivers:
//! 1. The WGSL shader (`xor_popcount.wgsl`).
//! 2. The Rust kernel wrapper [`BinaryDistanceKernel`] with one query × N
//!    corpus.
//! 3. A CPU reference implementation [`hamming_distance_cpu`] used as
//!    the gold-standard test oracle.
//! 4. Unit tests covering correctness on the CPU-fallback device, plus
//!    edge cases (single u32, identity, all-ones, large dim).
//!
//! Phase 2 A does **not** include:
//! - HNSW integration with binary codes (requires `BinaryDistanceMetric`
//!   threading through `HnswIndex` traversal — separate wave).
//! - Batched-query kernel (`Q` queries × `N` corpus → `Q × N` distances,
//!   shared-memory tiling) — separate wave.
//! - Real-data benchmarks on SIFT1M / GIST1M / DEEP1B — needs dataset
//!   download + ferro-bench-runner integration, GA-prep wave.
//! - Top-K reduction on GPU (we currently return all N distances; a
//!   bitonic / radix top-K keeps results on device for query batching).
//!
//! ## Calling contract
//!
//! - `query_bits`: bit-packed query, length `dim_u32 = ceil(dim_bits/32)`.
//!   Padding of unused trailing bits to zero is the **caller's**
//!   responsibility — the kernel does not mask.
//! - `corpus_bits`: `num_vecs * dim_u32` u32 words, vectors flat-laid.
//! - Returns `Vec<u32>` of length `num_vecs`, each entry in `0..=dim_bits`.
//!
//! The CPU and GPU paths produce **byte-equal** outputs (bitwise
//! `popcount` is exact integer arithmetic — no float drift to worry about).

use bytemuck::{Pod, Zeroable};

use crate::buffer::GpuBuffer;
use crate::device::{BufferUsage, GpuContext, GpuPipelineRaw};
use crate::error::GpuResult;
use crate::kernel::GpuKernel;

const XOR_POPCOUNT_SHADER: &str = include_str!("../shaders/xor_popcount.wgsl");
const WORKGROUP_SIZE: u32 = 64;

/// Distance metric for binary (bit-packed) code-vector search.
///
/// Currently a single variant — Hamming — but kept as an enum so future
/// metrics (Jaccard, weighted Hamming for asymmetric BBQ) can extend it
/// without an API break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryDistanceMetric {
    /// Hamming distance: number of differing bit positions.
    Hamming,
}

/// Uniform buffer layout — must match the `Params` struct in
/// `xor_popcount.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct XorPopcountParams {
    num_vecs: u32,
    dim_u32: u32,
}

/// GPU kernel that computes Hamming distance between one bit-packed
/// query and a batch of bit-packed corpus vectors.
///
/// Dispatch is one thread per corpus vector; each thread streams the
/// shared query and one corpus stripe through XOR + popcount. The
/// shader is bandwidth-bound for typical embedding sizes
/// (`dim_bits` ≥ 256), making this a memory-throughput offload —
/// CPU vs GPU comparison should be done at full corpus sizes
/// (`N` ≥ 100k) to amortise the upload cost.
pub struct BinaryDistanceKernel {
    pipeline: GpuPipelineRaw,
    ctx: GpuContext,
}

impl GpuKernel for BinaryDistanceKernel {
    type Params = BinaryDistanceMetric;
    type Result = Vec<u32>;

    fn compile(ctx: &GpuContext) -> GpuResult<Self> {
        let pipeline = ctx.device().create_pipeline(
            "binary-distance-xor-popcount",
            XOR_POPCOUNT_SHADER,
            "xor_popcount",
        )?;
        Ok(Self {
            pipeline,
            ctx: ctx.clone(),
        })
    }
}

impl BinaryDistanceKernel {
    /// Construct a binary distance kernel; alias for
    /// [`GpuKernel::compile`] kept as a non-trait constructor for
    /// callers that don't want to import the trait.
    pub fn new(ctx: &GpuContext) -> GpuResult<Self> {
        <Self as GpuKernel>::compile(ctx)
    }

    /// Compute Hamming distances between one query and a batch of
    /// corpus vectors.
    ///
    /// # Arguments
    /// - `query_bits`: bit-packed query, `dim_u32` u32 words.
    /// - `corpus_bits`: `num_vecs * dim_u32` u32 words, vectors flat-laid.
    /// - `num_vecs`: number of corpus vectors.
    /// - `dim_bits`: vector dimension in bits. The number of u32 words
    ///   per vector is `dim_u32 = ceil(dim_bits / 32)`. Trailing
    ///   padding bits in the last word **must** be zeroed by the
    ///   caller — the shader does not mask them out.
    ///
    /// # Returns
    /// `Vec<u32>` of length `num_vecs`, each in `0..=dim_bits`.
    ///
    /// # Errors
    /// Returns [`crate::error::GpuError::ColumnTypeMismatch`] if the
    /// supplied buffer lengths do not match `num_vecs * dim_u32`.
    pub fn compute(
        &self,
        query_bits: &[u32],
        corpus_bits: &[u32],
        num_vecs: usize,
        dim_bits: usize,
    ) -> GpuResult<Vec<u32>> {
        let dim_u32 = dim_u32_for(dim_bits);

        if num_vecs == 0 || dim_u32 == 0 {
            return Ok(Vec::new());
        }

        if query_bits.len() != dim_u32 {
            return Err(crate::error::GpuError::ColumnTypeMismatch {
                expected: format!("query_bits.len() == {dim_u32}"),
                actual: format!("query_bits.len() == {}", query_bits.len()),
            });
        }

        if corpus_bits.len() != num_vecs * dim_u32 {
            return Err(crate::error::GpuError::ColumnTypeMismatch {
                expected: format!("corpus_bits.len() == {}", num_vecs * dim_u32),
                actual: format!("corpus_bits.len() == {}", corpus_bits.len()),
            });
        }

        // Upload query
        let query_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "binary-query",
            query_bits.len(),
            BufferUsage::STORAGE,
        )?;
        query_buf.upload(&self.ctx, query_bits)?;

        // Upload corpus
        let corpus_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "binary-corpus",
            corpus_bits.len(),
            BufferUsage::STORAGE,
        )?;
        corpus_buf.upload(&self.ctx, corpus_bits)?;

        // Output distances
        let dist_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "binary-dists",
            num_vecs,
            BufferUsage::STORAGE_READBACK,
        )?;

        // Params uniform
        let params = XorPopcountParams {
            num_vecs: num_vecs as u32,
            dim_u32: dim_u32 as u32,
        };
        let params_buf = GpuBuffer::new::<XorPopcountParams>(
            &self.ctx,
            "binary-params",
            1,
            BufferUsage::UNIFORM,
        )?;
        params_buf.upload(&self.ctx, &[params])?;

        let bind_group = self.ctx.device().create_bind_group(
            &self.pipeline,
            0,
            &[
                query_buf.as_bind_entry(0),
                corpus_buf.as_bind_entry(1),
                dist_buf.as_bind_entry(2),
                params_buf.as_bind_entry(3),
            ],
        )?;

        let num_workgroups = (num_vecs as u32).div_ceil(WORKGROUP_SIZE);
        self.ctx
            .device()
            .dispatch(&self.pipeline, &[bind_group], (num_workgroups, 1, 1))?;

        dist_buf.download(&self.ctx)
    }
}

/// Number of u32 words required to hold `dim_bits` packed bits.
#[inline]
pub fn dim_u32_for(dim_bits: usize) -> usize {
    dim_bits.div_ceil(32)
}

/// CPU reference implementation of Hamming distance over bit-packed
/// vectors. Used as the gold oracle for GPU correctness tests and as a
/// drop-in fallback when no GPU is available.
///
/// `query_bits` must have length `dim_u32`; `corpus_bits` must have
/// length `num_vecs * dim_u32`. Padding bits in the last u32 must be
/// zeroed by the caller (this function does not mask).
///
/// # Panics
/// Panics if `corpus_bits.len() != num_vecs * dim_u32` or
/// `query_bits.len() != dim_u32` — these are programmer errors, not
/// runtime conditions.
pub fn hamming_distance_cpu(
    query_bits: &[u32],
    corpus_bits: &[u32],
    num_vecs: usize,
    dim_u32: usize,
) -> Vec<u32> {
    assert_eq!(
        query_bits.len(),
        dim_u32,
        "query_bits length mismatch (expected {dim_u32}, got {})",
        query_bits.len()
    );
    assert_eq!(
        corpus_bits.len(),
        num_vecs * dim_u32,
        "corpus_bits length mismatch (expected {}, got {})",
        num_vecs * dim_u32,
        corpus_bits.len()
    );

    let mut out = Vec::with_capacity(num_vecs);
    for v in 0..num_vecs {
        let base = v * dim_u32;
        let mut sum: u32 = 0;
        for i in 0..dim_u32 {
            sum = sum.wrapping_add((query_bits[i] ^ corpus_bits[base + i]).count_ones());
        }
        out.push(sum);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 — keeps tests reproducible without a
    /// runtime RNG dependency.
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
        (0..n)
            .map(|_| xorshift64(&mut state) as u32)
            .collect()
    }

    /// Helper: build a kernel against the CPU-fallback device. This always
    /// succeeds, so tests using it never need to be `#[ignore]`'d. The
    /// CPU-fallback `dispatch_cpu_kernel` path runs the Rust reference
    /// (registered in `kernel/cpu_dispatch.rs`) and returns byte-equal
    /// results to the WGSL kernel.
    fn cpu_kernel() -> BinaryDistanceKernel {
        let ctx = GpuContext::cpu_fallback();
        BinaryDistanceKernel::new(&ctx).expect("CPU fallback kernel compile")
    }

    #[test]
    fn single_query_correctness() {
        // dim_bits = 256 → dim_u32 = 8, num_vecs = 64.
        let dim_bits = 256;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_vecs = 64;

        let query = random_u32_vec(dim_u32, 0xdead_beef_cafe_babe);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0x0123_4567_89ab_cdef);

        let cpu = hamming_distance_cpu(&query, &corpus, num_vecs, dim_u32);
        let gpu = cpu_kernel()
            .compute(&query, &corpus, num_vecs, dim_bits)
            .expect("compute");

        assert_eq!(cpu.len(), num_vecs);
        assert_eq!(gpu, cpu, "GPU and CPU must produce byte-equal distances");

        // sanity: distances are bounded by dim_bits
        for &d in &gpu {
            assert!(
                d <= dim_bits as u32,
                "Hamming distance {d} > dim_bits {dim_bits}"
            );
        }
    }

    #[test]
    fn edge_case_small_dim() {
        // dim_bits = 32 (single u32 word).
        let dim_bits = 32;
        let dim_u32 = dim_u32_for(dim_bits);
        assert_eq!(dim_u32, 1);
        let num_vecs = 16;

        let query = vec![0xAAAA_AAAA_u32];
        let corpus: Vec<u32> = (0..num_vecs as u32).collect();

        let cpu = hamming_distance_cpu(&query, &corpus, num_vecs, dim_u32);
        let gpu = cpu_kernel()
            .compute(&query, &corpus, num_vecs, dim_bits)
            .expect("compute");
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn edge_case_large_dim() {
        // dim_bits = 1024 → dim_u32 = 32, num_vecs = 1024 (1 MB corpus).
        let dim_bits = 1024;
        let dim_u32 = dim_u32_for(dim_bits);
        assert_eq!(dim_u32, 32);
        let num_vecs = 1024;

        let query = random_u32_vec(dim_u32, 0xa1b2_c3d4_e5f6_0718);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xfedc_ba98_7654_3210);

        let cpu = hamming_distance_cpu(&query, &corpus, num_vecs, dim_u32);
        let gpu = cpu_kernel()
            .compute(&query, &corpus, num_vecs, dim_bits)
            .expect("compute");
        assert_eq!(gpu, cpu, "large-dim GPU must match CPU");
    }

    #[test]
    fn identity() {
        // Query equal to corpus[0] → dist[0] == 0.
        let dim_bits = 256;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_vecs = 4;

        let mut corpus = random_u32_vec(num_vecs * dim_u32, 0xfeed_face_dead_beef);
        let query: Vec<u32> = corpus[..dim_u32].to_vec();

        // Make corpus[1..] differ from query so we observe non-zero dists too.
        for w in corpus.iter_mut().skip(dim_u32) {
            *w ^= 0x1;
        }

        let gpu = cpu_kernel()
            .compute(&query, &corpus, num_vecs, dim_bits)
            .expect("compute");

        assert_eq!(gpu[0], 0, "dist to self must be 0");
        for &d in &gpu[1..] {
            assert!(d > 0, "perturbed corpus must give non-zero dist (got {d})");
        }
    }

    #[test]
    fn all_ones_vs_all_zeros() {
        // Query all 1s, corpus all 0s → every distance == dim_bits.
        let dim_bits = 256;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_vecs = 8;

        let query = vec![u32::MAX; dim_u32];
        let corpus = vec![0u32; num_vecs * dim_u32];

        let gpu = cpu_kernel()
            .compute(&query, &corpus, num_vecs, dim_bits)
            .expect("compute");
        for &d in &gpu {
            assert_eq!(d, dim_bits as u32);
        }

        // And reverse: all 0 query vs all-1 corpus.
        let query0 = vec![0u32; dim_u32];
        let corpus1 = vec![u32::MAX; num_vecs * dim_u32];
        let gpu = cpu_kernel()
            .compute(&query0, &corpus1, num_vecs, dim_bits)
            .expect("compute");
        for &d in &gpu {
            assert_eq!(d, dim_bits as u32);
        }
    }

    #[test]
    fn empty_corpus_returns_empty() {
        let kernel = cpu_kernel();
        let out = kernel
            .compute(&[0u32; 8], &[], 0, 256)
            .expect("empty compute");
        assert!(out.is_empty());
    }

    #[test]
    fn cpu_reference_is_self_consistent() {
        // Direct CPU-to-CPU sanity (no GPU dependency at all).
        let q = vec![0xFFFF_FFFFu32, 0u32]; // 32 + 0 = 32 ones
        let c = vec![
            0u32, 0u32,                       // dist = 32
            0xFFFF_FFFFu32, 0u32,             // dist = 0
            0u32, 0xFFFF_FFFFu32,             // dist = 64
            0xFFFF_FFFFu32, 0xFFFF_FFFFu32,   // dist = 32
        ];
        let out = hamming_distance_cpu(&q, &c, 4, 2);
        assert_eq!(out, vec![32, 0, 64, 32]);
    }

    #[test]
    fn dim_u32_for_rounds_up() {
        assert_eq!(dim_u32_for(0), 0);
        assert_eq!(dim_u32_for(1), 1);
        assert_eq!(dim_u32_for(31), 1);
        assert_eq!(dim_u32_for(32), 1);
        assert_eq!(dim_u32_for(33), 2);
        assert_eq!(dim_u32_for(768), 24);
        assert_eq!(dim_u32_for(1024), 32);
    }

    #[test]
    fn mismatched_query_len_errors() {
        let kernel = cpu_kernel();
        let q = vec![0u32; 4]; // wrong: dim_bits=256 expects 8
        let c = vec![0u32; 8];
        let err = kernel.compute(&q, &c, 1, 256).unwrap_err();
        // Must be a ColumnTypeMismatch — exact message unimportant.
        assert!(matches!(
            err,
            crate::error::GpuError::ColumnTypeMismatch { .. }
        ));
    }

    #[test]
    fn mismatched_corpus_len_errors() {
        let kernel = cpu_kernel();
        let q = vec![0u32; 8];
        let c = vec![0u32; 7]; // wrong: num_vecs=1 dim_u32=8 expects 8
        let err = kernel.compute(&q, &c, 1, 256).unwrap_err();
        assert!(matches!(
            err,
            crate::error::GpuError::ColumnTypeMismatch { .. }
        ));
    }
}

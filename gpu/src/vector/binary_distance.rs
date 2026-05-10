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
//!   10-50× over CPU.
//!
//! ## Scope of this file (Phase 2 A integration)
//!
//! Phase 2 A integration delivers:
//! 1. The single-query WGSL shader (`xor_popcount.wgsl`) with vec4<u32>
//!    SIMD widening of the inner XOR+popcount loop.
//! 2. A batched multi-query WGSL shader (`xor_popcount_batched.wgsl`)
//!    with workgroup-shared corpus tiling — a single dispatch produces
//!    the full Q × N distance matrix.
//! 3. An on-GPU bitonic top-K shader (`top_k_select.wgsl`) which keeps
//!    only the K smallest distances per query, slashing PCIe readback
//!    from `Q × N × 4 B` to `Q × K × 8 B`.
//! 4. The Rust kernel wrapper [`BinaryDistanceKernel`] exposing
//!    `compute()` (single-query, back-compat), `compute_batched()`
//!    (Q × N matrix), and `knn_search()` (Q × top-K end-to-end).
//! 5. A CPU reference implementation [`hamming_distance_cpu`] used as
//!    the gold-standard test oracle, plus
//!    [`hamming_distances_batched_cpu`] for batched correctness.
//!
//! Phase 2 A integration does **not** include:
//! - HNSW graph traversal with binary codes — that requires
//!   `BinaryDistanceMetric` threaded through `HnswIndex` traversal and
//!   is the Phase 2 A-final wave.
//! - Real-data benchmarks on SIFT1M / GIST1M / DEEP1B — needs dataset
//!   download + ferro-bench-runner integration; GA-prep wave.
//! - Recall@k measurement against a vetted ground-truth set.
//!
//! ## Calling contract
//!
//! - `query_bits` (single): bit-packed query, length `dim_u32 = ceil(dim_bits/32)`.
//!   Padding of unused trailing bits to zero is the **caller's**
//!   responsibility — the kernel does not mask.
//! - `queries_bits` (batched): `num_queries * dim_u32` u32 words, queries
//!   flat-laid.
//! - `corpus_bits`: `num_vecs * dim_u32` u32 words, vectors flat-laid.
//! - `compute()` returns `Vec<u32>` of length `num_vecs`, each entry in
//!   `0..=dim_bits`.
//! - `compute_batched()` returns `Vec<u32>` of length
//!   `num_queries * num_vecs` (row-major).
//! - `knn_search()` returns `Vec<Vec<(u32, u32)>>` (per-query
//!   `(distance, corpus_id)` pairs sorted ascending by distance).
//!
//! The CPU and GPU paths produce **byte-equal** outputs (bitwise
//! `popcount` is exact integer arithmetic — no float drift to worry about).

use bytemuck::{Pod, Zeroable};

use crate::buffer::GpuBuffer;
use crate::device::{BufferUsage, GpuContext, GpuPipelineRaw};
use crate::error::{GpuError, GpuResult};
use crate::kernel::GpuKernel;

const XOR_POPCOUNT_SHADER: &str = include_str!("../shaders/xor_popcount.wgsl");
const XOR_POPCOUNT_BATCHED_SHADER: &str = include_str!("../shaders/xor_popcount_batched.wgsl");
const TOP_K_SELECT_SHADER: &str = include_str!("../shaders/top_k_select.wgsl");

const WORKGROUP_SIZE: u32 = 64;

/// Tile dimensions hard-coded into `xor_popcount_batched.wgsl`.
/// Must match the shader constants exactly.
const BATCHED_TILE_Q: u32 = 8;
/// Number of corpus vectors per workgroup tile in the batched kernel.
const BATCHED_TILE_N: u32 = 8;
/// Maximum `dim_u32` supported by the batched shader (covers
/// `dim_bits` up to 2048).
pub const BATCHED_MAX_DIM_U32: u32 = 64;

/// Maximum `k_padded` (the merge buffer width) supported by the on-GPU
/// top-K shader. Caller-facing `K` may be anything up to this, then is
/// rounded up to the next power of two by `Self::knn_search`.
pub const TOP_K_MAX_K_PADDED: u32 = 1024;
/// Workgroup size used by the top-K shader; mirrored here for the
/// dispatch size calculation.
const TOP_K_WG_SIZE: u32 = 256;

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

/// Uniform buffer layout — must match the `ParamsB` struct in
/// `xor_popcount_batched.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct XorPopcountBatchedParams {
    num_queries: u32,
    num_vecs: u32,
    dim_u32: u32,
    /// `workgroup_id.x` offset, used to split a single Q × N dispatch
    /// into multiple chunks when the logical
    /// `ceil(num_vecs / TILE_N)` exceeds the device's
    /// `max_compute_workgroups_per_dimension` (downlevel default 65535).
    wg_offset_x: u32,
}

/// Maximum number of `workgroup` chunks dispatched along the N
/// dimension in a single `dispatch()` call. Set conservatively to the
/// `downlevel_defaults` value of `max_compute_workgroups_per_dimension`
/// (65535) so the same Rust code works on any wgpu backend without
/// querying the live limit.
const MAX_DISPATCH_X: u32 = 65_535;

/// Uniform buffer layout — must match `ParamsTopK` in
/// `top_k_select.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct TopKParams {
    num_queries: u32,
    num_vecs: u32,
    k: u32,
    k_padded: u32,
}

/// GPU kernel that computes Hamming distance between bit-packed query
/// vectors and bit-packed corpus vectors.
///
/// One instance owns three compiled pipelines:
/// 1. `pipeline_single` — single query × N corpus (vec4 SIMD).
/// 2. `pipeline_batched` — Q queries × N corpus, shared-memory tiled.
/// 3. `pipeline_top_k` — bitonic top-K reduction over a Q × N distance matrix.
///
/// Compilation is one-time at construction; thereafter
/// [`compute`](Self::compute), [`compute_batched`](Self::compute_batched),
/// and [`knn_search`](Self::knn_search) reuse the cached pipelines.
pub struct BinaryDistanceKernel {
    pipeline: GpuPipelineRaw,
    pipeline_batched: GpuPipelineRaw,
    pipeline_top_k: GpuPipelineRaw,
    ctx: GpuContext,
    /// Optional NVIDIA Tensor Core fast path. Initialised once at
    /// construction (if the `cuda-tensor-core` feature is on **and**
    /// the runtime can load libcuda + cuBLASLt + the NVRTC kernel) and
    /// consulted by [`Self::compute_batched`] before the WGSL pipeline.
    /// `None` on AMD / Apple / Intel / CPU-fallback / containers
    /// without GPU passthrough.
    #[cfg(feature = "cuda-tensor-core")]
    cuda_kernel: Option<crate::vector::cuda_tensor_core::CudaTensorCoreKernel>,
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
        let pipeline_batched = ctx.device().create_pipeline(
            "binary-distance-xor-popcount-batched",
            XOR_POPCOUNT_BATCHED_SHADER,
            "xor_popcount_batched",
        )?;
        let pipeline_top_k = ctx.device().create_pipeline(
            "binary-distance-top-k-select",
            TOP_K_SELECT_SHADER,
            "top_k_select",
        )?;
        // Try to spin up the CUDA Tensor Core fast path. Failure here
        // is non-fatal — the WGSL pipeline above is the source of
        // truth, the CUDA path is a runtime optimisation.
        #[cfg(feature = "cuda-tensor-core")]
        let cuda_kernel = if ctx.is_hardware_gpu() {
            match crate::vector::cuda_tensor_core::CudaTensorCoreKernel::try_new() {
                Ok(k) => {
                    log::info!(
                        "binary-distance: CUDA Tensor Core fast path enabled (cuBLASLt INT8 IMMA)"
                    );
                    Some(k)
                }
                Err(e) => {
                    log::debug!(
                        "binary-distance: CUDA Tensor Core path unavailable, using WGSL: {e}"
                    );
                    None
                }
            }
        } else {
            // Honour the caller's explicit CPU-fallback choice.
            None
        };
        Ok(Self {
            pipeline,
            pipeline_batched,
            pipeline_top_k,
            ctx: ctx.clone(),
            #[cfg(feature = "cuda-tensor-core")]
            cuda_kernel,
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

    /// Construct a binary distance kernel with the CUDA Tensor Core
    /// fast path explicitly disabled, even when the
    /// `cuda-tensor-core` feature is enabled and the host has a
    /// reachable NVIDIA device.
    ///
    /// This exists for benchmark harnesses that need to A/B the WGSL
    /// pipeline against the CUDA path on the same kernel instance.
    /// Production callers should use [`Self::new`].
    pub fn new_wgsl_only(ctx: &GpuContext) -> GpuResult<Self> {
        #[cfg_attr(not(feature = "cuda-tensor-core"), allow(unused_mut))]
        let mut kernel = Self::new(ctx)?;
        #[cfg(feature = "cuda-tensor-core")]
        {
            kernel.cuda_kernel = None;
        }
        Ok(kernel)
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
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("query_bits.len() == {dim_u32}"),
                actual: format!("query_bits.len() == {}", query_bits.len()),
            });
        }

        if corpus_bits.len() != num_vecs * dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
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

    /// Compute Hamming distances between a batch of queries and a batch
    /// of corpus vectors.
    ///
    /// One dispatch produces the full Q × N matrix in row-major layout:
    /// `result[q * num_vecs + n]` is the Hamming distance from query `q`
    /// to corpus vector `n`.
    ///
    /// The shader stages each tile of `TILE_Q × dim_u32` query words
    /// and `TILE_N × dim_u32` corpus words into workgroup-shared
    /// memory, so each corpus row is read from VRAM only once per
    /// workgroup column instead of once per query — the per-query VRAM
    /// traffic drops by roughly a factor of `TILE_Q` versus calling
    /// `compute` in a loop.
    ///
    /// # Arguments
    /// - `queries_bits`: `num_queries * dim_u32` u32 words, queries flat-laid.
    /// - `corpus_bits`: `num_vecs * dim_u32` u32 words.
    /// - `num_queries`: count of queries (Q).
    /// - `num_vecs`: count of corpus vectors (N).
    /// - `dim_bits`: vector dimension in bits; `dim_u32 = ceil(dim_bits / 32)`
    ///   must be ≤ [`BATCHED_MAX_DIM_U32`].
    ///
    /// # Returns
    /// `Vec<u32>` of length `num_queries * num_vecs`. Empty `num_queries`
    /// or `num_vecs` returns an empty vector.
    ///
    /// # Errors
    /// - [`GpuError::ColumnTypeMismatch`] on length mismatch.
    /// - [`GpuError::CpuFallback`] if `dim_u32 > BATCHED_MAX_DIM_U32`.
    pub fn compute_batched(
        &self,
        queries_bits: &[u32],
        corpus_bits: &[u32],
        num_queries: usize,
        num_vecs: usize,
        dim_bits: usize,
    ) -> GpuResult<Vec<u32>> {
        let dim_u32 = dim_u32_for(dim_bits);

        if num_queries == 0 || num_vecs == 0 || dim_u32 == 0 {
            return Ok(Vec::new());
        }

        if dim_u32 as u32 > BATCHED_MAX_DIM_U32 {
            return Err(GpuError::CpuFallback {
                reason: format!(
                    "compute_batched: dim_u32 = {dim_u32} exceeds shader limit {BATCHED_MAX_DIM_U32} (dim_bits = {dim_bits})"
                ),
            });
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

        // CUDA Tensor Core fast path — bit-exact with the WGSL kernel
        // (popcount-identity reformulation, INT8 IMMA + INT32 accum).
        // Falls through to WGSL on `CpuFallback` (no NVIDIA, no cuBLASLt
        // algorithm matched, etc.); any other error is propagated.
        #[cfg(feature = "cuda-tensor-core")]
        if let Some(ref cuda) = self.cuda_kernel {
            match cuda.compute_batched(queries_bits, corpus_bits, num_queries, num_vecs, dim_bits) {
                Ok(result) => return Ok(result),
                Err(GpuError::CpuFallback { reason }) => {
                    log::debug!(
                        "binary-distance: CUDA path skipped (Q={num_queries}, N={num_vecs}, dim_bits={dim_bits}): {reason}; falling back to WGSL"
                    );
                }
                Err(other) => return Err(other),
            }
        }

        let queries_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "binary-queries-batched",
            queries_bits.len(),
            BufferUsage::STORAGE,
        )?;
        queries_buf.upload(&self.ctx, queries_bits)?;

        let corpus_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "binary-corpus-batched",
            corpus_bits.len(),
            BufferUsage::STORAGE,
        )?;
        corpus_buf.upload(&self.ctx, corpus_bits)?;

        let dist_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "binary-dists-batched",
            num_queries * num_vecs,
            BufferUsage::STORAGE_READBACK,
        )?;

        self.run_batched_dispatches(
            &queries_buf,
            &corpus_buf,
            &dist_buf,
            num_queries,
            num_vecs,
            dim_u32,
        )?;

        dist_buf.download(&self.ctx)
    }

    /// Run the batched-distance shader with workgroup-count chunking
    /// along the N axis. Each chunk uses its own uniform buffer with
    /// `wg_offset_x` set so the shader can compute the right corpus
    /// stripe. Buffers are reused across chunks.
    fn run_batched_dispatches(
        &self,
        queries_buf: &GpuBuffer,
        corpus_buf: &GpuBuffer,
        dist_buf: &GpuBuffer,
        num_queries: usize,
        num_vecs: usize,
        dim_u32: usize,
    ) -> GpuResult<()> {
        let total_wg_x = (num_vecs as u32).div_ceil(BATCHED_TILE_N);
        let wg_y = (num_queries as u32).div_ceil(BATCHED_TILE_Q);

        let mut wg_x_done: u32 = 0;
        while wg_x_done < total_wg_x {
            let chunk = (total_wg_x - wg_x_done).min(MAX_DISPATCH_X);
            let params = XorPopcountBatchedParams {
                num_queries: num_queries as u32,
                num_vecs: num_vecs as u32,
                dim_u32: dim_u32 as u32,
                wg_offset_x: wg_x_done,
            };
            let params_buf = GpuBuffer::new::<XorPopcountBatchedParams>(
                &self.ctx,
                "binary-params-batched-chunk",
                1,
                BufferUsage::UNIFORM,
            )?;
            params_buf.upload(&self.ctx, &[params])?;

            let bind_group = self.ctx.device().create_bind_group(
                &self.pipeline_batched,
                0,
                &[
                    queries_buf.as_bind_entry(0),
                    corpus_buf.as_bind_entry(1),
                    dist_buf.as_bind_entry(2),
                    params_buf.as_bind_entry(3),
                ],
            )?;

            self.ctx
                .device()
                .dispatch(&self.pipeline_batched, &[bind_group], (chunk, wg_y, 1))?;
            wg_x_done += chunk;
        }
        Ok(())
    }

    /// End-to-end binary k-NN search: Q queries × N corpus → top-K per
    /// query (sorted ascending by Hamming distance).
    ///
    /// This is the production entry point. It chains
    /// [`compute_batched`](Self::compute_batched) (which produces the
    /// full Q × N distance matrix on-device) with the on-GPU bitonic
    /// top-K reducer (`top_k_select.wgsl`). PCIe readback is therefore
    /// only `Q × K × 8 B` instead of `Q × N × 4 B` — for `Q = 64`,
    /// `N = 1_000_000`, `K = 100` that's ~50 KB versus 256 MB.
    ///
    /// `k` is clamped to `min(k, num_vecs)` and rounded up internally
    /// to the next power of two for bitonic merging. `k` must satisfy
    /// `1 ≤ k ≤ TOP_K_MAX_K_PADDED` (= 1024 in this wave).
    ///
    /// # Returns
    /// `Vec<Vec<(distance, corpus_id)>>` outer-indexed by query, each
    /// inner vector having length `min(k, num_vecs)` and sorted
    /// ascending by `distance`.
    ///
    /// # Errors
    /// - [`GpuError::ColumnTypeMismatch`] on slice-length mismatch.
    /// - [`GpuError::CpuFallback`] if `dim_u32 > BATCHED_MAX_DIM_U32`
    ///   or `k > TOP_K_MAX_K_PADDED`.
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

        let k_eff = k.min(num_vecs);
        if (k_eff as u32) > TOP_K_MAX_K_PADDED {
            return Err(GpuError::CpuFallback {
                reason: format!(
                    "knn_search: k = {k_eff} exceeds shader limit {TOP_K_MAX_K_PADDED}"
                ),
            });
        }

        let dim_u32 = dim_u32_for(dim_bits);
        if dim_u32 as u32 > BATCHED_MAX_DIM_U32 {
            return Err(GpuError::CpuFallback {
                reason: format!(
                    "knn_search: dim_u32 = {dim_u32} exceeds shader limit {BATCHED_MAX_DIM_U32}"
                ),
            });
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

        // CUDA Tensor Core fast path. Bit-equivalent to the WGSL
        // pipeline below: same popcount-identity GEMM, same bitonic
        // top-K (CUDA port preserves the WGSL tie-break rule "lower id
        // wins at equal distance"). Falls through to WGSL on
        // `CpuFallback`; any other error propagates.
        #[cfg(feature = "cuda-tensor-core")]
        if let Some(ref cuda) = self.cuda_kernel {
            match cuda.knn_search(
                queries_bits, corpus_bits, num_queries, num_vecs, dim_bits, k_eff,
            ) {
                Ok(result) => return Ok(result),
                Err(GpuError::CpuFallback { reason }) => {
                    log::debug!(
                        "knn_search: CUDA path skipped (Q={num_queries}, N={num_vecs}, dim_bits={dim_bits}, k={k_eff}): {reason}; falling back to WGSL"
                    );
                }
                Err(other) => return Err(other),
            }
        }

        // ── Stage 1: batched distance matrix ──
        let queries_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "knn-queries",
            queries_bits.len(),
            BufferUsage::STORAGE,
        )?;
        queries_buf.upload(&self.ctx, queries_bits)?;

        let corpus_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "knn-corpus",
            corpus_bits.len(),
            BufferUsage::STORAGE,
        )?;
        corpus_buf.upload(&self.ctx, corpus_bits)?;

        // The full Q × N distance buffer is held entirely on-device —
        // it never crosses PCIe.
        let dist_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "knn-dists",
            num_queries * num_vecs,
            BufferUsage::STORAGE,
        )?;

        self.run_batched_dispatches(
            &queries_buf,
            &corpus_buf,
            &dist_buf,
            num_queries,
            num_vecs,
            dim_u32,
        )?;

        // ── Stage 2: on-GPU top-K reduction ──
        let k_padded = next_pow2_u32(k_eff as u32).max(2);
        let top_dists_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "knn-top-dists",
            num_queries * k_eff,
            BufferUsage::STORAGE_READBACK,
        )?;
        let top_ids_buf = GpuBuffer::new::<u32>(
            &self.ctx,
            "knn-top-ids",
            num_queries * k_eff,
            BufferUsage::STORAGE_READBACK,
        )?;
        let top_params = TopKParams {
            num_queries: num_queries as u32,
            num_vecs: num_vecs as u32,
            k: k_eff as u32,
            k_padded,
        };
        let top_params_buf = GpuBuffer::new::<TopKParams>(
            &self.ctx,
            "knn-top-params",
            1,
            BufferUsage::UNIFORM,
        )?;
        top_params_buf.upload(&self.ctx, &[top_params])?;

        let bind_group_top = self.ctx.device().create_bind_group(
            &self.pipeline_top_k,
            0,
            &[
                dist_buf.as_bind_entry(0),
                top_dists_buf.as_bind_entry(1),
                top_ids_buf.as_bind_entry(2),
                top_params_buf.as_bind_entry(3),
            ],
        )?;

        // 1 workgroup per query; workgroup_size handled inside shader.
        let _ = TOP_K_WG_SIZE; // referenced for self-documentation
        self.ctx.device().dispatch(
            &self.pipeline_top_k,
            &[bind_group_top],
            (num_queries as u32, 1, 1),
        )?;

        let dists: Vec<u32> = top_dists_buf.download(&self.ctx)?;
        let ids: Vec<u32> = top_ids_buf.download(&self.ctx)?;

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

/// Round `n` up to the next power of two (≥ 1). Used to pick `k_padded`
/// for the top-K bitonic merge buffer.
#[inline]
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

/// CPU reference for the batched Q × N Hamming-distance matrix. Output
/// is row-major: `result[q * num_vecs + n]`.
///
/// # Panics
/// Panics if length checks on `queries_bits` or `corpus_bits` fail.
pub fn hamming_distances_batched_cpu(
    queries_bits: &[u32],
    corpus_bits: &[u32],
    num_queries: usize,
    num_vecs: usize,
    dim_u32: usize,
) -> Vec<u32> {
    assert_eq!(
        queries_bits.len(),
        num_queries * dim_u32,
        "queries_bits length mismatch"
    );
    assert_eq!(
        corpus_bits.len(),
        num_vecs * dim_u32,
        "corpus_bits length mismatch"
    );
    let mut out = Vec::with_capacity(num_queries * num_vecs);
    for q in 0..num_queries {
        let qbase = q * dim_u32;
        for v in 0..num_vecs {
            let cbase = v * dim_u32;
            let mut sum: u32 = 0;
            for i in 0..dim_u32 {
                sum = sum
                    .wrapping_add((queries_bits[qbase + i] ^ corpus_bits[cbase + i]).count_ones());
            }
            out.push(sum);
        }
    }
    out
}

/// CPU reference for top-K selection over a Q × N distance matrix.
/// Returns Q vectors of `(distance, id)` pairs of length `min(k, num_vecs)`,
/// sorted ascending by distance, ties broken by lower id (matches the
/// shader's deterministic tie-break).
pub fn top_k_select_cpu(
    dists: &[u32],
    num_queries: usize,
    num_vecs: usize,
    k: usize,
) -> Vec<Vec<(u32, u32)>> {
    let k_eff = k.min(num_vecs);
    let mut out = Vec::with_capacity(num_queries);
    for q in 0..num_queries {
        let row_base = q * num_vecs;
        let mut indexed: Vec<(u32, u32)> = (0..num_vecs)
            .map(|i| (dists[row_base + i], i as u32))
            .collect();
        // Stable sort ascending by (dist, id).
        indexed.sort_by_key(|&(d, i)| (d, i));
        indexed.truncate(k_eff);
        out.push(indexed);
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
        (0..n).map(|_| xorshift64(&mut state) as u32).collect()
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
        // dim_bits = 32 (single u32 word; tail-only path).
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
    fn vec4_simd_correctness_non_multiple_of_4() {
        // Exercise the scalar-tail path: dim_u32 = 5 (dim_bits = 160).
        let dim_bits = 160;
        let dim_u32 = dim_u32_for(dim_bits);
        assert_eq!(dim_u32, 5);
        let num_vecs = 32;

        let query = random_u32_vec(dim_u32, 0x1234_5678_90ab_cdef);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0x0fed_cba9_8765_4321);

        let cpu = hamming_distance_cpu(&query, &corpus, num_vecs, dim_u32);
        let gpu = cpu_kernel()
            .compute(&query, &corpus, num_vecs, dim_bits)
            .expect("compute");
        assert_eq!(gpu, cpu, "vec4 + scalar tail path must be exact");
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
    fn next_pow2_works() {
        assert_eq!(next_pow2_u32(0), 1);
        assert_eq!(next_pow2_u32(1), 1);
        assert_eq!(next_pow2_u32(2), 2);
        assert_eq!(next_pow2_u32(3), 4);
        assert_eq!(next_pow2_u32(7), 8);
        assert_eq!(next_pow2_u32(8), 8);
        assert_eq!(next_pow2_u32(100), 128);
        assert_eq!(next_pow2_u32(1024), 1024);
    }

    #[test]
    fn mismatched_query_len_errors() {
        let kernel = cpu_kernel();
        let q = vec![0u32; 4]; // wrong: dim_bits=256 expects 8
        let c = vec![0u32; 8];
        let err = kernel.compute(&q, &c, 1, 256).unwrap_err();
        // Must be a ColumnTypeMismatch — exact message unimportant.
        assert!(matches!(err, GpuError::ColumnTypeMismatch { .. }));
    }

    #[test]
    fn mismatched_corpus_len_errors() {
        let kernel = cpu_kernel();
        let q = vec![0u32; 8];
        let c = vec![0u32; 7]; // wrong: num_vecs=1 dim_u32=8 expects 8
        let err = kernel.compute(&q, &c, 1, 256).unwrap_err();
        assert!(matches!(err, GpuError::ColumnTypeMismatch { .. }));
    }

    // ─── Batched kernel tests ───

    #[test]
    fn batched_correctness() {
        // dim = 256, N = 1024, Q = 8 → 8192 distances.
        let dim_bits = 256;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_queries = 8;
        let num_vecs = 1024;

        let queries = random_u32_vec(num_queries * dim_u32, 0xaaaa_bbbb_cccc_dddd);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0x1111_2222_3333_4444);

        let cpu = hamming_distances_batched_cpu(
            &queries, &corpus, num_queries, num_vecs, dim_u32,
        );
        let gpu = cpu_kernel()
            .compute_batched(&queries, &corpus, num_queries, num_vecs, dim_bits)
            .expect("compute_batched");
        assert_eq!(cpu.len(), num_queries * num_vecs);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn batched_matches_per_query() {
        // The Q × N batched matrix's q-th row must equal the result of
        // calling `compute` with that single query.
        let dim_bits = 768;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_queries = 4;
        let num_vecs = 256;
        let queries = random_u32_vec(num_queries * dim_u32, 0xdead_dead_dead_dead);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xbeef_beef_beef_beef);

        let kernel = cpu_kernel();
        let batched = kernel
            .compute_batched(&queries, &corpus, num_queries, num_vecs, dim_bits)
            .expect("compute_batched");

        for q in 0..num_queries {
            let qslice = &queries[q * dim_u32..(q + 1) * dim_u32];
            let single = kernel
                .compute(qslice, &corpus, num_vecs, dim_bits)
                .expect("compute");
            let row = &batched[q * num_vecs..(q + 1) * num_vecs];
            assert_eq!(row, single.as_slice(), "mismatch at q={q}");
        }
    }

    #[test]
    fn batched_empty_query() {
        let kernel = cpu_kernel();
        let out = kernel
            .compute_batched(&[], &[0u32; 8], 0, 1, 256)
            .expect("empty batched");
        assert!(out.is_empty());
    }

    #[test]
    fn batched_empty_corpus() {
        let kernel = cpu_kernel();
        let out = kernel
            .compute_batched(&[0u32; 8], &[], 1, 0, 256)
            .expect("empty corpus");
        assert!(out.is_empty());
    }

    #[test]
    fn batched_dim_too_large_errors() {
        // dim_u32 > BATCHED_MAX_DIM_U32 (=64) → CpuFallback error.
        let dim_bits = (BATCHED_MAX_DIM_U32 as usize + 1) * 32;
        let dim_u32 = dim_u32_for(dim_bits);
        let kernel = cpu_kernel();
        let q = vec![0u32; dim_u32];
        let c = vec![0u32; dim_u32];
        let err = kernel
            .compute_batched(&q, &c, 1, 1, dim_bits)
            .unwrap_err();
        assert!(matches!(err, GpuError::CpuFallback { .. }));
    }

    // ─── Top-K shader tests ───

    #[test]
    fn top_k_correctness() {
        // dim = 256, N = 1024, Q = 4, K = 10.
        let dim_bits = 256;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_queries = 4;
        let num_vecs = 1024;
        let k = 10;

        let queries = random_u32_vec(num_queries * dim_u32, 0xface_face_face_face);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xbabe_babe_babe_babe);

        // CPU oracle: full distance matrix → CPU top-K.
        let dists_cpu = hamming_distances_batched_cpu(
            &queries, &corpus, num_queries, num_vecs, dim_u32,
        );
        let cpu_topk = top_k_select_cpu(&dists_cpu, num_queries, num_vecs, k);

        let gpu_topk = cpu_kernel()
            .knn_search(&queries, &corpus, num_queries, num_vecs, dim_bits, k)
            .expect("knn_search");
        assert_eq!(gpu_topk.len(), num_queries);
        for q in 0..num_queries {
            assert_eq!(
                gpu_topk[q].len(),
                k,
                "row {q} length mismatch"
            );
            // Check distances are exactly equal.
            let cpu_dists: Vec<u32> = cpu_topk[q].iter().map(|&(d, _)| d).collect();
            let gpu_dists: Vec<u32> = gpu_topk[q].iter().map(|&(d, _)| d).collect();
            assert_eq!(
                gpu_dists, cpu_dists,
                "top-K distances must match exactly for query {q}"
            );
            // Distances must be sorted ascending.
            for w in 1..k {
                assert!(
                    gpu_topk[q][w - 1].0 <= gpu_topk[q][w].0,
                    "top-K must be sorted ascending"
                );
            }
            // Each id must point at the asserted distance.
            for &(d, id) in &gpu_topk[q] {
                let row_dist = dists_cpu[q * num_vecs + id as usize];
                assert_eq!(row_dist, d, "id {id} dist mismatch in query {q}");
            }
        }
    }

    #[test]
    fn top_k_edge_k_equal_one() {
        // K = 1: just the single nearest neighbour per query.
        let dim_bits = 64;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_queries = 3;
        let num_vecs = 16;

        let queries = random_u32_vec(num_queries * dim_u32, 0x9876_5432_10ab_cdef);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xabcd_ef01_2345_6789);

        let dists = hamming_distances_batched_cpu(
            &queries, &corpus, num_queries, num_vecs, dim_u32,
        );
        let cpu = top_k_select_cpu(&dists, num_queries, num_vecs, 1);
        let gpu = cpu_kernel()
            .knn_search(&queries, &corpus, num_queries, num_vecs, dim_bits, 1)
            .expect("knn_search k=1");
        assert_eq!(gpu.len(), num_queries);
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            assert_eq!(g.len(), 1);
            assert_eq!(g[0].0, c[0].0); // nearest distance must match
        }
    }

    #[test]
    fn top_k_edge_k_equal_n() {
        // K = N: full sort.
        let dim_bits = 32;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_queries = 2;
        let num_vecs = 16;
        let k = 16;

        let queries = random_u32_vec(num_queries * dim_u32, 0x1234_5678_9abc_def0);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0x0fed_cba9_8765_4321);

        let gpu = cpu_kernel()
            .knn_search(&queries, &corpus, num_queries, num_vecs, dim_bits, k)
            .expect("knn k=N");
        for row in &gpu {
            assert_eq!(row.len(), k);
            // Distances are sorted ascending.
            for w in 1..k {
                assert!(row[w - 1].0 <= row[w].0);
            }
            // Ids cover the full {0..N} (since K=N).
            let mut ids: Vec<u32> = row.iter().map(|&(_, i)| i).collect();
            ids.sort_unstable();
            for (i, &v) in ids.iter().enumerate() {
                assert_eq!(v, i as u32, "id coverage broken");
            }
        }
    }

    #[test]
    fn top_k_clamps_when_k_exceeds_n() {
        // K > N: should be clamped to N rather than over-allocate.
        let dim_bits = 32;
        let dim_u32 = dim_u32_for(dim_bits);
        let num_queries = 1;
        let num_vecs = 4;
        let k_request = 100;

        let queries = random_u32_vec(num_queries * dim_u32, 0xc0ff_eec0_ffee_c0ff);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xdead_face_dead_face);

        let gpu = cpu_kernel()
            .knn_search(
                &queries,
                &corpus,
                num_queries,
                num_vecs,
                dim_bits,
                k_request,
            )
            .expect("knn k>N");
        assert_eq!(gpu[0].len(), num_vecs);
    }

    #[test]
    fn top_k_empty_query() {
        let kernel = cpu_kernel();
        let out = kernel
            .knn_search(&[], &[0u32; 8], 0, 1, 256, 1)
            .expect("empty query");
        assert!(out.is_empty());
    }

    #[test]
    fn top_k_k_zero_returns_empty_rows() {
        let kernel = cpu_kernel();
        let q = random_u32_vec(8, 0x1);
        let c = random_u32_vec(16, 0x2);
        let out = kernel
            .knn_search(&q, &c, 1, 2, 256, 0)
            .expect("k=0");
        assert_eq!(out.len(), 1);
        assert!(out[0].is_empty());
    }

    #[test]
    fn top_k_k_too_large_errors() {
        let kernel = cpu_kernel();
        let q = vec![0u32; 8];
        let c = vec![0u32; 8 * 4096];
        let err = kernel
            .knn_search(&q, &c, 1, 4096, 256, (TOP_K_MAX_K_PADDED + 1) as usize)
            .unwrap_err();
        assert!(matches!(err, GpuError::CpuFallback { .. }));
    }

    #[test]
    fn top_k_select_cpu_oracle_sorted() {
        // Direct sanity for the CPU oracle.
        let dists = vec![5u32, 2, 8, 1, 4, 7, 3, 6];
        let out = top_k_select_cpu(&dists, 1, 8, 4);
        assert_eq!(out[0], vec![(1, 3), (2, 1), (3, 6), (4, 4)]);
    }
}

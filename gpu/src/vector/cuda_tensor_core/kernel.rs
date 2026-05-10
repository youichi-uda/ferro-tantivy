//! cuBLASLt INT8 GEMM + post-correction kernel for the CUDA Tensor Core
//! fast path.
//!
//! This module implements the two device-side stages that turn
//! per-vector popcounts and an inner-product matmul into a Q × N Hamming
//! distance matrix:
//!
//! 1. **GEMM**: `inner[m, n] = Σ_k Q[m,k] * D[n,k]` with INT8 inputs and
//!    INT32 accumulation. `cublasLtMatmul` selects an IMMA tensor-core
//!    algorithm via the heuristic for any K ≥ 4 (we only call with K =
//!    `dim_bits` ∈ {32, 64, …, 2048}, all multiples of 32).
//! 2. **Correction**: `out[m, n] = pop_q[m] + pop_d[n] − 2 · inner[m, n]`
//!    written by an NVRTC-compiled element-wise kernel. The output is
//!    `u32`, matching the WGSL/CPU oracle.
//!
//! The cuBLASLt call uses the lower-level `result::matmul` because the
//! safe `Matmul<T>` trait in `cudarc` has impls only for `f32`/`f16`/
//! `bf16`. We pass our own typed layouts (`CUDA_R_8I` / `CUDA_R_32I`)
//! and the `CUBLAS_COMPUTE_32I` compute type so the matmul is **bit-exact**
//! — no FP rounding, no TF32 truncation — for our `{0, 1}`-valued
//! inputs.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::cublaslt::{result as blas_result, sys as blas_sys};
use cudarc::driver::{
    sys as drv_sys, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx_with_opts;

use crate::error::{GpuError, GpuResult};

/// Bit-packed + unpacked + popcount corpus, kept resident on the
/// device so successive [`CudaGemmRunner::run_with_cached_corpus`]
/// calls skip the upload + unpack + popcount stages. Build with
/// [`CudaGemmRunner::prepare_corpus`].
pub(super) struct CachedCorpus {
    pub(super) runner: Arc<CudaGemmRunner>,
    pub(super) unpacked_dev: CudaSlice<i8>,
    pub(super) pop_dev: CudaSlice<i32>,
    pub(super) num_vecs: usize,
    pub(super) dim_bits: usize,
}

/// NVRTC sources compiled once at construction.
///
/// 1. `unpack_bits` — read row-major `num_rows × dim_u32` u32 input and
///    write row-major `num_rows × dim_bits` i8 output, with each output
///    byte equal to the matching LSB-first bit. Bits beyond `dim_bits`
///    in the last u32 of each row are dropped (matches the
///    calling-contract assumption that trailing padding bits are zero,
///    which is required by the GEMM-vs-popcount equivalence).
/// 2. `pop_correction` — close the popcount-identity:
///    `out[m, n] = pop_q[m] + pop_d[n] − 2 · inner[m, n]`. The output
///    is `u32` to match the WGSL kernel and the CPU reference exactly.
///
/// Doing the unpack on the device cuts PCIe traffic on the corpus from
/// `8 × num_vecs × dim_bits / 8` bytes (one byte per bit) down to the
/// natural `num_vecs × dim_u32 × 4` bytes (one bit per bit). For
/// `num_vecs = 1 M`, `dim = 768` that's 96 MB instead of 768 MB —
/// large enough to dominate the wall-clock at end-to-end timing.
const KERNEL_CU: &str = r#"
extern "C" __global__ void unpack_bits(
    const unsigned int* __restrict__ packed,
    signed char* __restrict__ out,
    unsigned int num_rows,
    unsigned int dim_u32,
    unsigned int dim_bits) {
    unsigned long long idx =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long) num_rows * (unsigned long long) dim_bits;
    if (idx >= total) {
        return;
    }
    unsigned int row = (unsigned int) (idx / dim_bits);
    unsigned int bit = (unsigned int) (idx - (unsigned long long) row * dim_bits);
    unsigned int word = packed[(unsigned long long) row * dim_u32 + (bit >> 5)];
    out[idx] = (signed char) ((word >> (bit & 31u)) & 1u);
}

extern "C" __global__ void pop_correction(
    const int* __restrict__ inner,
    const int* __restrict__ pop_q,
    const int* __restrict__ pop_d,
    unsigned int* __restrict__ out,
    unsigned int M,
    unsigned int N) {
    unsigned long long idx =
        (unsigned long long) blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long) M * (unsigned long long) N;
    if (idx >= total) {
        return;
    }
    unsigned int m = (unsigned int) (idx / N);
    unsigned int n = (unsigned int) (idx - (unsigned long long) m * N);
    int v = pop_q[m] + pop_d[n] - 2 * inner[idx];
    out[idx] = (unsigned int) v;
}

// Per-query bitonic-merge top-K. Direct port of `top_k_select.wgsl`
// — same algorithm, same tie-break (lower id wins at equal distance),
// same `MAX_K_PADDED = 1024` cap. Block layout: one block per query,
// 256 threads, 16 KB of static shared memory (2 * MAX_K_PADDED u32
// dists + 2 * MAX_K_PADDED u32 ids).
//
// Output `top_k_dists` / `top_k_ids` is row-major Q × K, sorted
// ascending by distance, ties broken by lower id.
__device__ __forceinline__ void __cmpswap(
    unsigned int* d, unsigned int* id,
    unsigned int a, unsigned int b, unsigned int dir) {
    unsigned int da = d[a], db = d[b];
    unsigned int ia = id[a], ib = id[b];
    bool asc = (da > db) || (da == db && ia > ib);
    bool should_swap = (asc && dir == 1u) || (!asc && dir == 0u);
    if (should_swap) {
        d[a] = db;  d[b] = da;
        id[a] = ib; id[b] = ia;
    }
}

extern "C" __global__ void top_k_select(
    const unsigned int* __restrict__ dists,
    unsigned int* __restrict__ top_k_dists,
    unsigned int* __restrict__ top_k_ids,
    unsigned int num_queries,
    unsigned int num_vecs,
    unsigned int k,
    unsigned int k_padded) {
    const unsigned int MAX_K_PADDED = 1024u;
    const unsigned int WG_SIZE = 256u;
    const unsigned int SENTINEL = 0xFFFFFFFFu;
    __shared__ unsigned int buf_dist[2u * MAX_K_PADDED];
    __shared__ unsigned int buf_id[2u * MAX_K_PADDED];

    unsigned int q = blockIdx.x;
    unsigned int tid = threadIdx.x;
    if (q >= num_queries) return;

    unsigned int n = num_vecs;
    unsigned int kp = k_padded;            // power of two, ≤ MAX_K_PADDED
    unsigned int total_len = kp * 2u;
    unsigned int row_base = q * n;

    // Initialize sentinels in both halves.
    for (unsigned int i = tid; i < total_len; i += WG_SIZE) {
        buf_dist[i] = SENTINEL;
        buf_id[i] = 0u;
    }
    __syncthreads();

    // Streaming merge: process N inputs in batches of `kp`. After each
    // batch the lower half holds the running top kp sorted ascending.
    for (unsigned int processed = 0u; processed < n; processed += kp) {
        for (unsigned int j = tid; j < kp; j += WG_SIZE) {
            unsigned int src = processed + j;
            if (src < n) {
                buf_dist[kp + j] = dists[row_base + src];
                buf_id[kp + j] = src;
            } else {
                buf_dist[kp + j] = SENTINEL;
                buf_id[kp + j] = 0u;
            }
        }
        __syncthreads();

        // Bitonic sort 2*kp entries ascending.
        for (unsigned int size = 2u; size <= total_len; size *= 2u) {
            for (unsigned int stride = size / 2u; stride > 0u; stride /= 2u) {
                for (unsigned int idx = tid; idx < total_len; idx += WG_SIZE) {
                    unsigned int pair = idx ^ stride;
                    if (pair > idx) {
                        unsigned int dir = ((idx & size) == 0u) ? 1u : 0u;
                        __cmpswap(buf_dist, buf_id, idx, pair, dir);
                    }
                }
                __syncthreads();
            }
        }
    }

    // Spill the first k entries. Already sorted ascending.
    for (unsigned int w = tid; w < k; w += WG_SIZE) {
        top_k_dists[q * k + w] = buf_dist[w];
        top_k_ids[q * k + w] = buf_id[w];
    }
}
"#;

/// Owns the cuBLASLt handle, NVRTC-compiled correction kernel, and a
/// scratch workspace. One instance is reused across all
/// `compute_batched` invocations.
pub(super) struct CudaGemmRunner {
    handle: blas_sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    workspace_size: usize,
    unpack_func: CudaFunction,
    correction_func: CudaFunction,
    top_k_func: CudaFunction,
    stream: Arc<CudaStream>,
    // Hold the context so the device isn't dropped while we still own
    // the cuBLASLt handle / workspace.
    _ctx: Arc<CudaContext>,
}

impl CudaGemmRunner {
    /// Build a runner against device 0 (or whatever `CUDA_VISIBLE_DEVICES`
    /// selects). Returns `GpuError::CpuFallback` (with a non-fatal reason)
    /// if no CUDA driver / NVIDIA device / cuBLASLt library can be loaded
    /// — callers are expected to fall back to the WGSL kernel on this
    /// outcome.
    pub(super) fn try_new() -> GpuResult<Self> {
        let ctx = CudaContext::new(0).map_err(|e| GpuError::CpuFallback {
            reason: format!("CudaContext::new(0) failed: {e}"),
        })?;
        let stream = ctx.default_stream();

        let handle = blas_result::create_handle().map_err(|e| GpuError::CpuFallback {
            reason: format!("cublasLtCreate failed: {e}"),
        })?;

        // Workspace size mirrors cudarc's `Workspace::new` recommendation
        // (32 MiB on Hopper+, 4 MiB elsewhere). 4 MiB is plenty for our
        // GEMM shapes — typical IMMA algorithms use < 1 MiB.
        let workspace_size = 4 * 1024 * 1024usize;
        let workspace = unsafe { stream.alloc::<u8>(workspace_size) }.map_err(|e| {
            GpuError::CpuFallback {
                reason: format!("workspace alloc {} bytes failed: {e}", workspace_size),
            }
        })?;

        // Compile the unpack + correction kernels via NVRTC. We do
        // this once at construction so per-call latency stays low.
        let ptx = compile_ptx_with_opts(KERNEL_CU, Default::default()).map_err(|e| {
            GpuError::CpuFallback {
                reason: format!("NVRTC compile of cuda_tensor_core kernels failed: {e}"),
            }
        })?;
        let module = ctx.load_module(ptx).map_err(|e| GpuError::CpuFallback {
            reason: format!("load_module(cuda_tensor_core kernels) failed: {e}"),
        })?;
        let unpack_func = module.load_function("unpack_bits").map_err(|e| {
            GpuError::CpuFallback {
                reason: format!("load_function(unpack_bits) failed: {e}"),
            }
        })?;
        let correction_func = module.load_function("pop_correction").map_err(|e| {
            GpuError::CpuFallback {
                reason: format!("load_function(pop_correction) failed: {e}"),
            }
        })?;
        let top_k_func = module.load_function("top_k_select").map_err(|e| {
            GpuError::CpuFallback {
                reason: format!("load_function(top_k_select) failed: {e}"),
            }
        })?;

        Ok(Self {
            handle,
            workspace,
            workspace_size,
            unpack_func,
            correction_func,
            top_k_func,
            stream,
            _ctx: ctx,
        })
    }

    /// Upload bit-packed `inputs` (M rows × dim_u32 words), unpack to
    /// one `i8` per bit on the device, and return the unpacked
    /// `M × dim_bits` row-major `i8` slice on device. Used as a shared
    /// helper by [`Self::run`] (queries + corpus) and
    /// [`Self::prepare_corpus`] (corpus only — kept resident).
    fn unpack_to_device(
        &self,
        bits: &[u32],
        num_rows: usize,
        dim_bits: usize,
    ) -> GpuResult<CudaSlice<i8>> {
        let dim_u32 = dim_bits.div_ceil(32);
        debug_assert_eq!(bits.len(), num_rows * dim_u32);

        let stream = &self.stream;
        let packed_dev = stream.clone_htod(bits).map_err(map_drv)?;
        let mut unpacked_dev =
            unsafe { stream.alloc::<i8>(num_rows * dim_bits) }.map_err(map_drv)?;

        let total = num_rows as u64 * dim_bits as u64;
        let dim_u32_u32 = dim_u32 as u32;
        let dim_bits_u32 = dim_bits as u32;
        let num_rows_u32 = num_rows as u32;
        unsafe {
            stream
                .launch_builder(&self.unpack_func)
                .arg(&packed_dev)
                .arg(&mut unpacked_dev)
                .arg(&num_rows_u32)
                .arg(&dim_u32_u32)
                .arg(&dim_bits_u32)
                .launch(launch_cfg(total))
                .map_err(map_drv)?;
        }
        Ok(unpacked_dev)
    }

    /// Run GEMM + correction and **leave the result on the device** —
    /// returns the row-major `m × n` `u32` distance matrix as a
    /// `CudaSlice<u32>`. Used by both
    /// [`Self::gemm_and_correct`] (which then downloads to host) and
    /// the on-device top-K path in [`Self::knn_search_with_cached_corpus`].
    fn gemm_and_correct_dev(
        &self,
        q_i8_dev: &CudaSlice<i8>,
        d_i8_dev: &CudaSlice<i8>,
        pop_q_dev: &CudaSlice<i32>,
        pop_d_dev: &CudaSlice<i32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> GpuResult<CudaSlice<u32>> {
        let stream = &self.stream;

        let mut inner_dev = unsafe { stream.alloc::<i32>(m * n) }.map_err(map_drv)?;
        let mut out_dev = unsafe { stream.alloc::<u32>(m * n) }.map_err(map_drv)?;

        // GEMM: inner_col[n, m] = Σ_k D[n, k] * Q[m, k]
        //       (= row-major inner[m, n] in the same memory).
        // Layout derivation: see kernel.rs module docs.
        // - A := corpus (D), shape K × N column-major (= storage row-major
        //   N × K), opA = T → op(A) = D in row-major view.
        // - B := queries (Q), shape K × M column-major (= storage row-major
        //   M × K), opB = N → op(B) = Q^T in column-major view.
        // - Output dim: M_p = N (rows), N_p = M (cols), ldc = N.
        {
            let (a_ptr, _record_a) = d_i8_dev.device_ptr(stream);
            let (b_ptr, _record_b) = q_i8_dev.device_ptr(stream);
            let (c_ptr, _record_c) = inner_dev.device_ptr_mut(stream);
            unsafe {
                self.matmul_int8_int32(
                    a_ptr, b_ptr, c_ptr,
                    /* m_p */ n as u64,
                    /* n_p */ m as u64,
                    /* k_p */ k as u64,
                    /* transa */ true,
                    /* transb */ false,
                    /* lda    */ k as i64,
                    /* ldb    */ k as i64,
                    /* ldc    */ n as i64,
                )?;
            }
        }

        let total = m as u64 * n as u64;
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        unsafe {
            stream
                .launch_builder(&self.correction_func)
                .arg(&inner_dev)
                .arg(pop_q_dev)
                .arg(pop_d_dev)
                .arg(&mut out_dev)
                .arg(&m_u32)
                .arg(&n_u32)
                .launch(launch_cfg(total))
                .map_err(map_drv)?;
        }
        Ok(out_dev)
    }

    /// Convenience wrapper around [`Self::gemm_and_correct_dev`] that
    /// downloads the full `m × n` result to host memory and waits for
    /// completion before returning.
    fn gemm_and_correct(
        &self,
        q_i8_dev: &CudaSlice<i8>,
        d_i8_dev: &CudaSlice<i8>,
        pop_q_dev: &CudaSlice<i32>,
        pop_d_dev: &CudaSlice<i32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> GpuResult<Vec<u32>> {
        let out_dev =
            self.gemm_and_correct_dev(q_i8_dev, d_i8_dev, pop_q_dev, pop_d_dev, m, n, k)?;
        let out_host = self.stream.clone_dtoh(&out_dev).map_err(map_drv)?;
        self.stream.synchronize().map_err(map_drv)?;
        Ok(out_host)
    }

    /// End-to-end batched compute: returns row-major `M × N` Hamming
    /// distances given the bit-packed query / corpus / per-row
    /// popcount inputs.
    ///
    /// Uploads the **bit-packed** u32 buffers and unpacks to one i8 per
    /// bit on-device, so PCIe traffic on the corpus is `dim_u32 × 4 B`
    /// per row instead of `dim_bits × 1 B` per row (a 1×, not an 8×,
    /// blow-up vs the WGSL kernel's input size).
    pub(super) fn run(
        &self,
        queries_bits: &[u32],
        corpus_bits: &[u32],
        pop_q: &[i32],
        pop_d: &[i32],
        num_queries: usize,
        num_vecs: usize,
        dim_bits: usize,
    ) -> GpuResult<Vec<u32>> {
        debug_assert_eq!(pop_q.len(), num_queries);
        debug_assert_eq!(pop_d.len(), num_vecs);

        let q_i8_dev = self.unpack_to_device(queries_bits, num_queries, dim_bits)?;
        let d_i8_dev = self.unpack_to_device(corpus_bits, num_vecs, dim_bits)?;
        let pop_q_dev = self.stream.clone_htod(pop_q).map_err(map_drv)?;
        let pop_d_dev = self.stream.clone_htod(pop_d).map_err(map_drv)?;
        self.gemm_and_correct(
            &q_i8_dev, &d_i8_dev, &pop_q_dev, &pop_d_dev, num_queries, num_vecs, dim_bits,
        )
    }

    /// Upload + unpack + popcount the corpus once and return a handle
    /// the caller can re-use across many `compute_batched_with_cached_corpus`
    /// invocations. The handle owns its on-device storage and survives
    /// until dropped.
    pub(super) fn prepare_corpus(
        self: &Arc<Self>,
        corpus_bits: &[u32],
        num_vecs: usize,
        dim_bits: usize,
        pop_d: &[i32],
    ) -> GpuResult<CachedCorpus> {
        debug_assert_eq!(pop_d.len(), num_vecs);
        let unpacked_dev = self.unpack_to_device(corpus_bits, num_vecs, dim_bits)?;
        let pop_dev = self.stream.clone_htod(pop_d).map_err(map_drv)?;
        // Block until the unpack finishes so a follow-up
        // `compute_batched_with_cached_corpus` sees a quiescent corpus.
        self.stream.synchronize().map_err(map_drv)?;
        Ok(CachedCorpus {
            runner: Arc::clone(self),
            unpacked_dev,
            pop_dev,
            num_vecs,
            dim_bits,
        })
    }

    /// Run a query batch against an already-uploaded corpus. Returns
    /// row-major `num_queries × corpus.num_vecs` Hamming distances.
    pub(super) fn run_with_cached_corpus(
        &self,
        queries_bits: &[u32],
        pop_q: &[i32],
        corpus: &CachedCorpus,
        num_queries: usize,
    ) -> GpuResult<Vec<u32>> {
        debug_assert_eq!(pop_q.len(), num_queries);
        let q_i8_dev = self.unpack_to_device(queries_bits, num_queries, corpus.dim_bits)?;
        let pop_q_dev = self.stream.clone_htod(pop_q).map_err(map_drv)?;
        self.gemm_and_correct(
            &q_i8_dev,
            &corpus.unpacked_dev,
            &pop_q_dev,
            &corpus.pop_dev,
            num_queries,
            corpus.num_vecs,
            corpus.dim_bits,
        )
    }

    /// Run an end-to-end k-NN search against a cached corpus —
    /// distance matrix + on-device top-K reduction. Only `Q × K × 8 B`
    /// crosses PCIe back, regardless of N.
    ///
    /// `k_padded` must be a power of two ≤ 1024 (the shader-side
    /// `MAX_K_PADDED`); the caller is responsible for rounding `k` up.
    pub(super) fn knn_search_with_cached_corpus(
        &self,
        queries_bits: &[u32],
        pop_q: &[i32],
        corpus: &CachedCorpus,
        num_queries: usize,
        k: usize,
        k_padded: u32,
    ) -> GpuResult<(Vec<u32>, Vec<u32>)> {
        debug_assert_eq!(pop_q.len(), num_queries);
        let q_i8_dev = self.unpack_to_device(queries_bits, num_queries, corpus.dim_bits)?;
        let pop_q_dev = self.stream.clone_htod(pop_q).map_err(map_drv)?;

        // Stage 1: distance matrix on-device (no host download).
        let dist_dev = self.gemm_and_correct_dev(
            &q_i8_dev,
            &corpus.unpacked_dev,
            &pop_q_dev,
            &corpus.pop_dev,
            num_queries,
            corpus.num_vecs,
            corpus.dim_bits,
        )?;

        // Stage 2: per-query top-K via the CUDA bitonic merge kernel.
        let stream = &self.stream;
        let mut top_dists_dev =
            unsafe { stream.alloc::<u32>(num_queries * k) }.map_err(map_drv)?;
        let mut top_ids_dev =
            unsafe { stream.alloc::<u32>(num_queries * k) }.map_err(map_drv)?;
        let num_queries_u32 = num_queries as u32;
        let num_vecs_u32 = corpus.num_vecs as u32;
        let k_u32 = k as u32;
        // 1 block per query, 256 threads.
        let cfg = LaunchConfig {
            grid_dim: (num_queries_u32.max(1), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0, // shared mem is statically declared in the kernel
        };
        unsafe {
            stream
                .launch_builder(&self.top_k_func)
                .arg(&dist_dev)
                .arg(&mut top_dists_dev)
                .arg(&mut top_ids_dev)
                .arg(&num_queries_u32)
                .arg(&num_vecs_u32)
                .arg(&k_u32)
                .arg(&k_padded)
                .launch(cfg)
                .map_err(map_drv)?;
        }

        let dists_host = stream.clone_dtoh(&top_dists_dev).map_err(map_drv)?;
        let ids_host = stream.clone_dtoh(&top_ids_dev).map_err(map_drv)?;
        stream.synchronize().map_err(map_drv)?;
        Ok((dists_host, ids_host))
    }

    /// Lower-level cuBLASLt INT8 → INT32 matmul.
    ///
    /// This bypasses the `Matmul<T>` trait in `cudarc::cublaslt::safe`
    /// because that trait only ships `Matmul` impls for `f32` (and
    /// optionally `f16`/`bf16`); INT8 GEMM with integer accumulator
    /// requires `CUDA_R_8I` / `CUDA_R_32I` matrix layouts and the
    /// `CUBLAS_COMPUTE_32I` compute type, which the safe trait doesn't
    /// expose.
    ///
    /// # Safety
    /// `a_ptr`, `b_ptr`, `c_ptr` must be live device pointers at least
    /// large enough for the supplied `m_p`, `n_p`, `k_p` and leading
    /// dimensions. The caller is responsible for the lifetime of those
    /// allocations through to `stream.synchronize()`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn matmul_int8_int32(
        &self,
        a_ptr: drv_sys::CUdeviceptr,
        b_ptr: drv_sys::CUdeviceptr,
        c_ptr: drv_sys::CUdeviceptr,
        m_p: u64,
        n_p: u64,
        k_p: u64,
        transa: bool,
        transb: bool,
        lda: i64,
        ldb: i64,
        ldc: i64,
    ) -> GpuResult<()> {
        let i8_t = blas_sys::cudaDataType_t::CUDA_R_8I;
        let i32_t = blas_sys::cudaDataType_t::CUDA_R_32I;
        let compute_t = blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32I;
        let scale_t = i32_t;

        let (a_rows, a_cols) = if transa { (k_p, m_p) } else { (m_p, k_p) };
        let (b_rows, b_cols) = if transb { (n_p, k_p) } else { (k_p, n_p) };

        let a_layout =
            blas_result::create_matrix_layout(i8_t, a_rows, a_cols, lda).map_err(map_blas)?;
        let b_layout =
            blas_result::create_matrix_layout(i8_t, b_rows, b_cols, ldb).map_err(map_blas)?;
        let c_layout =
            blas_result::create_matrix_layout(i32_t, m_p, n_p, ldc).map_err(map_blas)?;

        let matmul_desc =
            blas_result::create_matmul_desc(compute_t, scale_t).map_err(map_blas)?;

        let trans_a_v: i32 = transa as i32;
        let trans_b_v: i32 = transb as i32;
        blas_result::set_matmul_desc_attribute(
            matmul_desc,
            blas_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            (&trans_a_v) as *const i32 as *const c_void,
            std::mem::size_of::<i32>(),
        )
        .map_err(map_blas)?;
        blas_result::set_matmul_desc_attribute(
            matmul_desc,
            blas_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            (&trans_b_v) as *const i32 as *const c_void,
            std::mem::size_of::<i32>(),
        )
        .map_err(map_blas)?;

        let pref = blas_result::create_matmul_pref().map_err(map_blas)?;
        blas_result::set_matmul_pref_attribute(
            pref,
            blas_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&self.workspace_size) as *const usize as *const c_void,
            std::mem::size_of::<usize>(),
        )
        .map_err(map_blas)?;

        let heuristic = blas_result::get_matmul_algo_heuristic(
            self.handle,
            matmul_desc,
            a_layout,
            b_layout,
            c_layout,
            c_layout,
            pref,
        )
        .map_err(|e| GpuError::CpuFallback {
            reason: format!(
                "no cuBLASLt INT8 algorithm matched the requested shape (M={m_p} N={n_p} K={k_p}, transa={transa}, transb={transb}): {e}"
            ),
        })?;

        let alpha: i32 = 1;
        let beta: i32 = 0;
        let (w_ptr, _record_w) = self.workspace.device_ptr(&self.stream);

        let stream_raw = self.stream.cu_stream() as blas_sys::cudaStream_t;
        let res = blas_result::matmul(
            self.handle,
            matmul_desc,
            (&alpha) as *const i32 as *const c_void,
            (&beta) as *const i32 as *const c_void,
            a_ptr as *const c_void,
            a_layout,
            b_ptr as *const c_void,
            b_layout,
            c_ptr as *const c_void,
            c_layout,
            c_ptr as *mut c_void,
            c_layout,
            (&heuristic.algo) as *const _,
            w_ptr as *mut c_void,
            self.workspace_size,
            stream_raw,
        );

        // Always destroy descriptors, even on failure.
        let _ = blas_result::destroy_matmul_pref(pref);
        let _ = blas_result::destroy_matmul_desc(matmul_desc);
        let _ = blas_result::destroy_matrix_layout(a_layout);
        let _ = blas_result::destroy_matrix_layout(b_layout);
        let _ = blas_result::destroy_matrix_layout(c_layout);

        res.map_err(map_blas)
    }
}

impl Drop for CudaGemmRunner {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe {
                let _ = blas_result::destroy_handle(handle);
            }
        }
    }
}

// SAFETY: Send/Sync follow from cudarc's CudaBlasLT pattern: the
// cuBLASLt handle and CudaContext / CudaStream are documented as
// thread-safe so long as the same handle is not used from two threads
// simultaneously. We hold the handle behind a single `&CudaGemmRunner`
// and rely on the calling code to serialise access (the callers are
// the WGSL `BinaryDistanceKernel`, which already holds a `Mutex`-style
// dispatch lock through its `GpuContext`).
unsafe impl Send for CudaGemmRunner {}
unsafe impl Sync for CudaGemmRunner {}

fn launch_cfg(total: u64) -> LaunchConfig {
    const TPB: u32 = 256;
    let blocks = ((total + TPB as u64 - 1) / TPB as u64) as u32;
    LaunchConfig {
        grid_dim: (blocks.max(1), 1, 1),
        block_dim: (TPB, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::CpuFallback {
        reason: format!("CUDA driver error: {e}"),
    }
}

fn map_blas(e: blas_result::CublasError) -> GpuError {
    GpuError::CpuFallback {
        reason: format!("cuBLASLt error: {e}"),
    }
}

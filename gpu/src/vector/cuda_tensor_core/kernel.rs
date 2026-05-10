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

        Ok(Self {
            handle,
            workspace,
            workspace_size,
            unpack_func,
            correction_func,
            stream,
            _ctx: ctx,
        })
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
        let m = num_queries;
        let n = num_vecs;
        let k = dim_bits;
        let dim_u32 = dim_bits.div_ceil(32);

        debug_assert_eq!(queries_bits.len(), m * dim_u32);
        debug_assert_eq!(corpus_bits.len(), n * dim_u32);
        debug_assert_eq!(pop_q.len(), m);
        debug_assert_eq!(pop_d.len(), n);

        let stream = &self.stream;

        // Upload bit-packed inputs (u32) and per-row popcount (i32).
        let q_packed_dev = stream.clone_htod(queries_bits).map_err(map_drv)?;
        let d_packed_dev = stream.clone_htod(corpus_bits).map_err(map_drv)?;
        let pop_q_dev = stream.clone_htod(pop_q).map_err(map_drv)?;
        let pop_d_dev = stream.clone_htod(pop_d).map_err(map_drv)?;

        // Allocate device-side i8 unpacked tensors and reusable
        // intermediates. Total scratch: (m + n) * k bytes for
        // unpacking, m * n * 4 bytes for the inner product, m * n * 4
        // bytes for the final output.
        let mut q_i8_dev = unsafe { stream.alloc::<i8>(m * k) }.map_err(map_drv)?;
        let mut d_i8_dev = unsafe { stream.alloc::<i8>(n * k) }.map_err(map_drv)?;

        // Device-side bit unpack (one byte per bit, ∈ {0, 1}).
        let dim_u32_u32 = dim_u32 as u32;
        let dim_bits_u32 = dim_bits as u32;
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let q_total = m as u64 * k as u64;
        let d_total = n as u64 * k as u64;
        unsafe {
            stream
                .launch_builder(&self.unpack_func)
                .arg(&q_packed_dev)
                .arg(&mut q_i8_dev)
                .arg(&m_u32)
                .arg(&dim_u32_u32)
                .arg(&dim_bits_u32)
                .launch(launch_cfg(q_total))
                .map_err(map_drv)?;
            stream
                .launch_builder(&self.unpack_func)
                .arg(&d_packed_dev)
                .arg(&mut d_i8_dev)
                .arg(&n_u32)
                .arg(&dim_u32_u32)
                .arg(&dim_bits_u32)
                .launch(launch_cfg(d_total))
                .map_err(map_drv)?;
        }

        // Allocate the inner-product matrix and the final output. Both
        // row-major M × N. cuBLASLt writes the inner-product as
        // column-major N × M, which shares memory with row-major M × N.
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
        //
        // The `device_ptr*` guards are dropped at the end of this scope
        // so the subsequent NVRTC kernel launch can take its own
        // immutable borrows of `inner_dev`.
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

        // Element-wise correction: out[idx] = pop_q[m] + pop_d[n] − 2·inner[idx].
        let total = m as u64 * n as u64;
        let cfg = launch_cfg(total);
        unsafe {
            stream
                .launch_builder(&self.correction_func)
                .arg(&inner_dev)
                .arg(&pop_q_dev)
                .arg(&pop_d_dev)
                .arg(&mut out_dev)
                .arg(&m_u32)
                .arg(&n_u32)
                .launch(cfg)
                .map_err(map_drv)?;
        }

        let out_host = stream.clone_dtoh(&out_dev).map_err(map_drv)?;
        stream.synchronize().map_err(map_drv)?;
        Ok(out_host)
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

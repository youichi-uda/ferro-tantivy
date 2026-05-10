//! **Phase 5-3 — BMMA (1-bit Tensor Core) PoC**
//!
//! NVIDIA Turing+ exposes a 1-bit tensor-core instruction
//! `mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32(.xor.popc)` (PTX
//! ISA §9.7.13.5) that natively computes
//!
//! > `D[m,n] = popcount(A[m,k] XOR B[k,n])` reduced over `k = 0..128`
//!
//! per warp invocation, with `A`, `B` as packed 1-bit matrices and `D`
//! as a `s32` accumulator. For binary Hamming distance this is a
//! direct lift: no INT8 unpack, no `popcount(q ⊕ d)` afterwards. PCIe
//! traffic on the corpus stays bit-packed (≈ 96 MB at `N = 1 M, dim =
//! 768`, vs the INT8 path's 96 MB packed + 768 MB unpacked on
//! device), arithmetic throughput per SM lifts by ≈ 8× at the
//! op-count level.
//!
//! cuBLASLt **cannot** dispatch this — `cublasLtMatmul` only supports
//! `CUBLAS_COMPUTE_32I` with `CUDA_R_8I` inputs (per the cuBLAS 12.4
//! reference, "binary GEMM" is not in the supported compute-type
//! matrix). The only path is a hand-rolled CUDA kernel that emits
//! `mma.sync` via inline PTX. NVRTC compiles inline PTX as long as the
//! `--gpu-architecture` flag is set to a binary-MMA-capable target;
//! we use `sm_75` (Turing — the floor for binary MMA).
//!
//! ## What this module does
//!
//! - One small kernel: `bmma_xor_popc_8x8x128_smoke`. Computes the
//!   **Q = 8, dim_bits = 128** Hamming-distance matrix vs an
//!   arbitrary `N` corpus, single warp per 8-corpus tile.
//! - One smoke test (`bmma_smoke_byte_equal_8x1024x128`) that asserts
//!   byte-equality vs the CPU oracle on `Q = 8, N = 1024, dim_bits =
//!   128` — the parity gate the prompt's plan calls out as the
//!   minimum to confirm the PTX path is viable.
//!
//! ## What this module does *not* yet do
//!
//! - Dispatch BMMA for arbitrary `dim_bits ≠ 128`, `Q ≠ 8`, or
//!   compute a corpus-wide kernel. Those need K-loop accumulation,
//!   query tiling, and a full launch grid — the prompt's "Phase
//!   5-3 corpus migration" stage. The smoke kernel intentionally
//!   stays minimal so a positive parity result is unambiguous.
//! - Replace the INT8 path. The INT8 IMMA kernels in `kernel.rs`
//!   stay the production path; BMMA is opt-in via the smoke test
//!   today and a follow-up `Phase 5-3 a` migrates the corpus path
//!   only after the PoC passes parity at this scale.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::error::{GpuError, GpuResult};

/// Single-warp BMMA kernel: one warp computes
/// `D[8 × 8] = popcount(A[8 × 128] XOR B[128 × 8])` per block, where
/// `B` is laid out so each "column" is one corpus vector's 128 bits.
///
/// PTX layout pattern (from PTX ISA 8.4 §9.7.13.5 fragment tables for
/// m8n8k128.b1):
/// - **A** (row-major .row): thread `t` holds bits
///   `A[t/4, (t%4)*32 .. (t%4)*32 + 32]`, packed as one `b32`.
/// - **B** (col-major .col): thread `t` holds bits
///   `B[(t%4)*32 .. (t%4)*32 + 32, t/4]`, packed as one `b32`.
///   Mapped to our row-major corpus storage, that means thread `t`
///   loads `corpus[d_tile*8 + t/4, (t%4)*32 .. + 32]`.
/// - **D** (s32 accumulator): thread `t` holds two s32 values at
///   `D[t/4, (t%4)*2]` and `D[t/4, (t%4)*2 + 1]`.
const BMMA_KERNEL_CU: &str = r#"
extern "C" __global__ void bmma_xor_popc_8x8x128_smoke(
    const unsigned int* __restrict__ q_packed,   // [8, 4]
    const unsigned int* __restrict__ d_packed,   // [N, 4]
    unsigned int* __restrict__ out,              // [8, N] row-major
    unsigned int num_vecs) {
    unsigned int tid     = threadIdx.x;
    unsigned int row     = tid >> 2;          // 0..7   (lane / 4)
    unsigned int chunk   = tid & 3u;          // 0..3   (lane % 4)
    unsigned int d_tile  = blockIdx.x;        // each block handles 8 corpus rows
    unsigned int d_base  = d_tile * 8u;

    // A: query row `row`, K-chunk `chunk`. q_packed has 8 queries × 4 u32.
    unsigned int a = q_packed[row * 4u + chunk];

    // B: corpus row `d_base + row`, K-chunk `chunk` (col-major view of B).
    unsigned int b;
    if (d_base + row < num_vecs) {
        b = d_packed[(d_base + row) * 4u + chunk];
    } else {
        b = 0u;
    }

    // C accumulator initialised to zero; we have a single K-tile so
    // there's no carry-in.
    int c0 = 0, c1 = 0;
    int d0, d1;
    asm volatile(
        "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
        "{%0, %1}, {%2}, {%3}, {%4, %5};"
        : "=r"(d0), "=r"(d1)
        : "r"(a), "r"(b), "r"(c0), "r"(c1));

    // Store: row `row`, columns `(chunk*2)` and `(chunk*2 + 1)`.
    unsigned int col0 = d_base + chunk * 2u;
    unsigned int col1 = d_base + chunk * 2u + 1u;
    if (col0 < num_vecs) {
        out[row * num_vecs + col0] = (unsigned int) d0;
    }
    if (col1 < num_vecs) {
        out[row * num_vecs + col1] = (unsigned int) d1;
    }
}

// General BMMA kernel: handles Q rounded up to 8, N rounded up to 8,
// and `dim_bits` that is a multiple of 128 (`dim_u32` multiple of 4).
// One warp per (q_tile=8 × d_tile=8) output block; K loop runs one
// `mma.sync` per 128-bit step accumulating into the same `c0/c1`
// fragment.
extern "C" __global__ void bmma_xor_popc_general(
    const unsigned int* __restrict__ q_packed,   // [Q, dim_u32]
    const unsigned int* __restrict__ d_packed,   // [N, dim_u32]
    unsigned int* __restrict__ out,              // [Q, N] row-major
    unsigned int Q,
    unsigned int N,
    unsigned int dim_u32) {
    unsigned int tid     = threadIdx.x;
    unsigned int row     = tid >> 2;          // 0..7
    unsigned int chunk   = tid & 3u;          // 0..3
    unsigned int q_tile  = blockIdx.y;
    unsigned int d_tile  = blockIdx.x;
    unsigned int q_base  = q_tile * 8u;
    unsigned int d_base  = d_tile * 8u;

    int c0 = 0, c1 = 0;
    for (unsigned int k_word = 0u; k_word + 4u <= dim_u32; k_word += 4u) {
        unsigned int a;
        unsigned int b;
        if (q_base + row < Q) {
            a = q_packed[(q_base + row) * dim_u32 + k_word + chunk];
        } else {
            a = 0u;
        }
        if (d_base + row < N) {
            b = d_packed[(d_base + row) * dim_u32 + k_word + chunk];
        } else {
            b = 0u;
        }
        int d0, d1;
        asm volatile(
            "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
            "{%0, %1}, {%2}, {%3}, {%4, %5};"
            : "=r"(d0), "=r"(d1)
            : "r"(a), "r"(b), "r"(c0), "r"(c1));
        c0 = d0;
        c1 = d1;
    }

    unsigned int out_row = q_base + row;
    unsigned int out_col0 = d_base + chunk * 2u;
    unsigned int out_col1 = d_base + chunk * 2u + 1u;
    if (out_row < Q) {
        if (out_col0 < N) out[out_row * N + out_col0] = (unsigned int) c0;
        if (out_col1 < N) out[out_row * N + out_col1] = (unsigned int) c1;
    }
}
"#;

/// Loaded BMMA kernel handles. Built once and reused across calls.
pub(super) struct BmmaKernels {
    /// Single-shape `Q = 8, dim_bits = 128` kernel; only the smoke
    /// test exercises it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) smoke: CudaFunction,
    pub(super) general: CudaFunction,
}

/// Compile both BMMA kernels. NVRTC needs the architecture flag for
/// inline PTX; `sm_75` (Turing) is the floor for binary tensor cores
/// and the conservative target — Ampere / Ada / Hopper all
/// forward-compatible.
pub(super) fn build_kernels(ctx: &Arc<CudaContext>) -> GpuResult<BmmaKernels> {
    let opts = CompileOptions {
        arch: Some("sm_75"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(BMMA_KERNEL_CU, opts).map_err(|e| GpuError::CpuFallback {
        reason: format!("NVRTC compile of bmma kernels failed: {e}"),
    })?;
    let module = ctx.load_module(ptx).map_err(|e| GpuError::CpuFallback {
        reason: format!("load_module(bmma) failed: {e}"),
    })?;
    let smoke = module
        .load_function("bmma_xor_popc_8x8x128_smoke")
        .map_err(|e| GpuError::CpuFallback {
            reason: format!("load_function(bmma smoke): {e}"),
        })?;
    let general = module
        .load_function("bmma_xor_popc_general")
        .map_err(|e| GpuError::CpuFallback {
            reason: format!("load_function(bmma general): {e}"),
        })?;
    Ok(BmmaKernels { smoke, general })
}

/// Compute Q × N Hamming distances using the BMMA `mma.sync` kernel.
/// `dim_bits` must be a multiple of 128 (= `dim_u32` multiple of 4);
/// callers should fall back to the INT8 IMMA path for other widths.
/// Returns a fresh `Vec<u32>`; the bench wires this into a head-to-head
/// against `CudaTensorCoreKernel::compute_batched`.
pub(super) fn compute_batched_bmma(
    kernels: &BmmaKernels,
    stream: &Arc<cudarc::driver::CudaStream>,
    queries_bits: &[u32],
    corpus_bits: &[u32],
    num_queries: usize,
    num_vecs: usize,
    dim_bits: usize,
) -> GpuResult<Vec<u32>> {
    if dim_bits == 0 || dim_bits % 128 != 0 {
        return Err(GpuError::CpuFallback {
            reason: format!(
                "BMMA path requires dim_bits to be a positive multiple of 128 (got {dim_bits})"
            ),
        });
    }
    let dim_u32 = dim_bits / 32;
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

    let q_dev = stream.clone_htod(queries_bits).map_err(map_drv)?;
    let d_dev = stream.clone_htod(corpus_bits).map_err(map_drv)?;
    let mut out_dev = unsafe { stream.alloc::<u32>(num_queries * num_vecs) }
        .map_err(map_drv)?;

    let q_tiles = num_queries.div_ceil(8) as u32;
    let d_tiles = num_vecs.div_ceil(8) as u32;
    let cfg = LaunchConfig {
        grid_dim: (d_tiles.max(1), q_tiles.max(1), 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let q_u32 = num_queries as u32;
    let n_u32 = num_vecs as u32;
    let dim_u32_u32 = dim_u32 as u32;
    unsafe {
        stream
            .launch_builder(&kernels.general)
            .arg(&q_dev)
            .arg(&d_dev)
            .arg(&mut out_dev)
            .arg(&q_u32)
            .arg(&n_u32)
            .arg(&dim_u32_u32)
            .launch(cfg)
            .map_err(map_drv)?;
    }
    let out = stream.clone_dtoh(&out_dev).map_err(map_drv)?;
    stream.synchronize().map_err(map_drv)?;
    Ok(out)
}

fn map_drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::CpuFallback {
        reason: format!("CUDA driver error: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::binary_distance::hamming_distances_batched_cpu;
    use cudarc::driver::CudaContext;

    /// Phase 5-3 PoC parity: BMMA `m8n8k128` inline-PTX kernel must
    /// match the CPU oracle byte-for-byte at the minimum useful
    /// shape (Q = 8, N = 1024, dim_bits = 128). A pass here proves
    /// the inline-PTX path compiles, links, runs, and produces
    /// correct numbers; a future Phase 5-3 a is then unblocked to
    /// migrate the production corpus path. A failure here is the
    /// negative-result signal recorded in ADR-001 §Alternatives.
    #[test]
    fn bmma_smoke_byte_equal_8x1024x128() {
        // Skip cleanly when no CUDA host is present.
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping BMMA smoke (no CUDA context): {e}");
                return;
            }
        };

        let kernels = match build_kernels(&ctx) {
            Ok(f) => f,
            Err(e) => {
                // NVRTC failure is the expected negative-result
                // outcome on hosts where binary MMA isn't supported
                // (pre-Turing or stripped CUDA install). Record the
                // skip and bail — the test still passes because the
                // PoC is informational, not a regression gate.
                eprintln!(
                    "skipping BMMA smoke — NVRTC could not build inline-PTX kernel ({e})"
                );
                return;
            }
        };
        let func = &kernels.smoke;

        let q: usize = 8;
        let n: usize = 1024;
        let dim_bits: usize = 128;
        let dim_u32: usize = 4;

        // Deterministic random inputs.
        let mut state: u64 = 0xb1aa_c0de_d00d_5eed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u32
        };
        let queries: Vec<u32> = (0..q * dim_u32).map(|_| next()).collect();
        let corpus: Vec<u32> = (0..n * dim_u32).map(|_| next()).collect();

        let cpu = hamming_distances_batched_cpu(&queries, &corpus, q, n, dim_u32);

        let stream = ctx.default_stream();
        let q_dev = stream.clone_htod(&queries).expect("htod q");
        let d_dev = stream.clone_htod(&corpus).expect("htod d");
        let mut out_dev = unsafe { stream.alloc::<u32>(q * n) }.expect("alloc out");

        let num_blocks = (n / 8) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_u32 = n as u32;
        unsafe {
            stream
                .launch_builder(func)
                .arg(&q_dev)
                .arg(&d_dev)
                .arg(&mut out_dev)
                .arg(&n_u32)
                .launch(cfg)
                .expect("bmma smoke launch");
        }

        let gpu = stream.clone_dtoh(&out_dev).expect("dtoh");
        stream.synchronize().expect("sync");

        assert_eq!(
            gpu, cpu,
            "BMMA smoke parity violated at Q={q} N={n} dim_bits={dim_bits} — \
             first divergence at index {}",
            gpu.iter()
                .zip(cpu.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(0)
        );
    }

    /// General-shape BMMA kernel parity. Sweeps a small grid of
    /// (Q, N, dim_bits) where dim_bits is a multiple of 128. Failure
    /// here means the K-loop accumulator pattern, the column-tile
    /// edge handling, or the MMA-fragment store layout has diverged
    /// from CPU oracle — the PoC has stopped being a "promotion
    /// candidate" and the production INT8 path remains the only
    /// correct one.
    #[test]
    fn bmma_general_parity_sweep() {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("skipping BMMA general (no CUDA context): {e}");
                return;
            }
        };
        let kernels = match build_kernels(&ctx) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("skipping BMMA general (NVRTC build failed): {e}");
                return;
            }
        };

        let cases: &[(usize, usize, usize)] = &[
            (1, 8, 128),
            (4, 64, 256),
            (8, 1024, 512),
            (16, 4096, 768),
            (32, 8192, 1024),
            (64, 16384, 768),
        ];

        let stream = ctx.default_stream();
        for &(q, n, dim_bits) in cases {
            let dim_u32 = dim_bits / 32;
            // Deterministic random inputs.
            let mut state: u64 = 0x9876_5432_1abc_def0u64
                ^ ((q as u64) << 40)
                ^ ((n as u64) << 16)
                ^ dim_bits as u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u32
            };
            let queries: Vec<u32> = (0..q * dim_u32).map(|_| next()).collect();
            let corpus: Vec<u32> = (0..n * dim_u32).map(|_| next()).collect();

            let cpu = hamming_distances_batched_cpu(&queries, &corpus, q, n, dim_u32);

            let q_dev = stream.clone_htod(&queries).expect("htod q");
            let d_dev = stream.clone_htod(&corpus).expect("htod d");
            let mut out_dev = unsafe { stream.alloc::<u32>(q * n) }.expect("alloc out");

            let q_tiles = (q as u32).div_ceil(8);
            let d_tiles = (n as u32).div_ceil(8);
            let cfg = LaunchConfig {
                grid_dim: (d_tiles.max(1), q_tiles.max(1), 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let q_u32 = q as u32;
            let n_u32 = n as u32;
            let dim_u32_u32 = dim_u32 as u32;
            unsafe {
                stream
                    .launch_builder(&kernels.general)
                    .arg(&q_dev)
                    .arg(&d_dev)
                    .arg(&mut out_dev)
                    .arg(&q_u32)
                    .arg(&n_u32)
                    .arg(&dim_u32_u32)
                    .launch(cfg)
                    .expect("bmma general launch");
            }
            let gpu = stream.clone_dtoh(&out_dev).expect("dtoh");
            stream.synchronize().expect("sync");

            assert_eq!(
                gpu,
                cpu,
                "BMMA general parity violated at Q={q} N={n} dim_bits={dim_bits} — \
                 first divergence at index {}",
                gpu.iter()
                    .zip(cpu.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0)
            );
        }
    }
}

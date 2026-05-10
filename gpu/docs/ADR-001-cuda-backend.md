# ADR-001 — CUDA Tensor Core fast path for binary Hamming distance

| Status   | Accepted (2026-05-10) |
|----------|-----------------------|
| Scope    | `tantivy-gpu` only    |
| Driver   | Wave 4 G1 verdict (Inferentia2 sub-project) — L4 PyTorch FP16 matmul recorded **8.3 GP/s @ M=128 N=65k**, vs the existing WGSL `countOneBits(q ^ c)` plateau at ~10M docs/s. WGSL/wgpu cannot reach Tensor Cores by design (cross-platform abstraction). |
| Hardware | RTX 4070 Ti SUPER (local), L4 (g6.xlarge spot) for regression. |

## Context

`BinaryDistanceKernel::compute_batched` runs the WGSL shader
`xor_popcount_batched.wgsl`, which uses `countOneBits(query[i] ^ corpus[i])`
inside a workgroup-tiled inner loop. On NVIDIA hardware this leaves Tensor
Cores idle — the backbone of every Ampere/Ada datacenter GPU — because:

1. wgpu (the cross-platform layer) does not expose `wmma` / `mma.sync` / cuBLAS;
2. WGSL has no hardware-accelerated INT8/FP16 matmul intrinsic;
3. SPIR-V cooperative matrix is only partially supported on the wgpu-side and
   is gated behind unstable feature flags that are not portable.

The Wave 4 PoC (`/home/y1/git/Inferentia2/ferro-knn-inf2/`) confirmed
empirically that the **popcount-identity reformulation**

> `popcount(q ⊕ d) = popcount(q) + popcount(d) − 2·⟨q, d⟩`

turns binary Hamming distance into a dense matmul, which Tensor Cores execute
6-22× faster than `countOneBits(xor)` on the same hardware. INT8 matmul with
FP32 accumulator is **bit-exact** vs the bit-wise oracle (verified in
`g1_int8_bench.py`); FP16 matmul on Ampere/Ada is ±1-3 bits and needs
post-correction.

## Decision

1. Add a parallel CUDA-only fast path under
   `gpu/src/vector/cuda_tensor_core/` that runs whenever
   - the `cuda-tensor-core` Cargo feature is enabled at build time, **and**
   - the runtime can load `libcuda.so` and instantiate cuBLASLt.
2. Use the **`cudarc`** crate (v0.19) with `default-features = false` and the
   feature set `["cuda-12040", "driver", "cublaslt", "nvrtc",
   "dynamic-loading", "std"]`. Rationale:
   - `cudarc` runtime-loads `libcuda.so` / `libcublasLt.so` /
     `libnvrtc.so`, so `cargo build` does not require CUDA on the host —
     critical for CI matrix and macOS / AMD / Apple developer machines.
   - `cublaslt` exposes `cublasLtMatmul` with INT8 GEMM (which we need for
     bit-exactness; see Phase 3 below).
   - `nvrtc` lets us JIT-compile the small post-correction kernel without a
     separate `.cu` build step, so the crate stays a single `cargo` package.
   - Same crate already ships in `candle` and `burn`; mature production use.
3. INT8-unpacked + INT32 accumulator is the **primary path**.
   - On-device `unpack_bits` NVRTC kernel turns each input bit into one
     `i8` byte (∈ {0, 1}) so PCIe traffic stays at one bit per bit
     (96 MB at N = 1 M, dim = 768) instead of 8× that.
   - Inner-product `⟨q, d⟩ ≤ dim_bits` (bounded; INT32 cannot overflow
     for any practical `dim_bits`).
   - cuBLASLt is configured with `CUBLAS_COMPUTE_32I` and
     `CUDA_R_32I` scale type, so accumulation is integer arithmetic
     end-to-end — no FP rounding, no TF32 truncation.
   - Output `result[m, n] = pop_q[m] + pop_d[n] − 2 · ⟨q[m], d[n]⟩` is
     bit-equal to the WGSL/CPU oracle. The Phase 3 parity sweep
     (`tests/cuda_tensor_core_parity.rs`) confirms this across **80
     `(Q, N, dim_bits)` cells** plus four targeted edge tests
     (extremes, identity, small-odd shapes); 0 byte-equality
     violations on RTX 4070 Ti SUPER.
4. FP16 matmul + `±1-bit` host-side repair, and TF32 matmul, are deferred
   fallbacks (Phase 3 follow-up). They are not on the critical path.
5. Runtime detection lives in the new module — `CudaTensorCoreKernel::try_new`
   returns `None` when the driver/library can't be loaded or the device
   isn't NVIDIA. `BinaryDistanceKernel::compute_batched` consults a cached
   `Option<CudaTensorCoreKernel>` and falls back to the WGSL pipeline on
   `None` or any error from the CUDA path.
6. The existing WGSL kernels (`xor_popcount.wgsl`,
   `xor_popcount_batched.wgsl`, `top_k_select.wgsl`) are **not** modified.
   They remain the only path on AMD / Apple / Intel / OpenGL / CPU
   fallback, and the only path when `cuda-tensor-core` is off.

## Alternatives considered

- **`cust` (Rust-CUDA project).** Requires CUDA toolkit at build time,
  which makes the default `cargo build` heavier on contributor machines.
  Its strength is hosting Rust-codegen kernels; we don't need that — cuBLAS
  + NVRTC suffice.
- **Direct `cuda-runtime-sys` / `cublas-sys` bindgen.** Working but no
  runtime loading, no NVRTC, no convenience wrappers around layout
  descriptors. Re-implements the same surface `cudarc` already provides.
- **WGSL with subgroup matrix ops (`wgpu` 24+ unstable).** Not generally
  available on the wgpu adapters we target; no INT8 path; would still
  bottom-out below cuBLAS on dense matmul.

## Consequences

- The crate gains an optional CUDA dependency. `cargo test
  -p tantivy-gpu --no-default-features --features cpu-fallback` and
  `cargo build -p tantivy-gpu` (default features) must continue to work
  without CUDA on the host — `cuda-tensor-core` is **off by default**.
- A new test target (`gpu/tests/cuda_tensor_core_parity.rs`) sweeps
  Q ∈ {1..128}, N ∈ {1..16384}, dim ∈ {128, 256, 512, 768, 1024} and
  asserts byte-equal output vs `hamming_distances_batched_cpu`. The test
  is gated on `#[cfg(feature = "cuda-tensor-core")]` and skips at runtime
  if no NVIDIA device is present.
- A new bench (`gpu/benches/binary_distance.rs` extension) records CUDA
  vs WGSL on the same workload grid; the **Go threshold** is
  CUDA ≥ 3× WGSL at the representative segment shape (N = 100k, Q = 64).
- **Measured on RTX 4070 Ti SUPER (Vulkan WGSL adapter, CUDA 12.4)**:
  | shape (Q × N × dim) | WGSL | CUDA | CUDA / WGSL |
  |---------------------|------|------|-------------|
  | 8 × 10 000 × 768    | 2.56 ms | 0.30 ms | **8.50×** |
  | 64 × 10 000 × 768   | 3.04 ms | 0.49 ms | **6.19×** |
  | 8 × 100 000 × 768   | 7.50 ms | 2.41 ms | **3.12×** |
  | 64 × 100 000 × 768 (assertion gate) | 28.59 ms | 4.22 ms | **6.77×** |
  | 64 × 1 000 000 × 768 (tracking)     | 252.94 ms | 89.35 ms | 2.83× |
  Raw timings live in `gpu/benches/data/cuda_vs_wgsl.json`.
- The N = 1 M shape comes in below 3×. Profile breakdown: corpus PCIe
  upload (96 MB) ~6 ms + on-device unpack ~1 ms + INT8 GEMM ~10 ms +
  correction ~1 ms + result PCIe download (256 MB) ~21 ms ≈ 40 ms — the
  remaining ≈50 ms is serial-stream overhead and pageable-host transfer
  cost. The Wave 4 reference (8.3 GP/s) was measured with a
  **device-resident corpus**, i.e. without the upload leg. To recover
  the full 6-7× at the N = 1 M shape we need a Phase 4 follow-up:
  - Cache `corpus_bits_dev` and the unpacked `i8` corpus on device
    between successive `compute_batched` calls (keyed on a corpus
    fingerprint or an explicit `with_cached_corpus(...)` API).
  - Pinned host memory for the result download.
  - Async stream + double-buffered upload so the next call's upload
    overlaps the previous call's GEMM.
  This is not on the Phase 1 critical path; the Phase 1 deliverable is
  bit-exact parity + ≥ 3× speedup at the production segment shape, both
  of which the Phase 1 implementation meets.
- Future contributors who want to enable the path locally:
  ```
  cargo build -p tantivy-gpu --features cuda-tensor-core
  cargo test  -p tantivy-gpu --features cuda-tensor-core --test cuda_tensor_core_parity
  cargo bench -p tantivy-gpu --features cuda-tensor-core --bench binary_distance
  ```

## Reference

- Wave 4 verdict: `~/git/Inferentia2/ferro-knn-inf2/reports/g1-final-verdict-with-direct-l4.md`
- INT8 reference: `~/git/Inferentia2/ferro-knn-inf2/code/neuron/g1_int8_bench.py`
- L4 multi-query: `~/git/Inferentia2/ferro-knn-inf2/reports/g1_l4_multi_query.json`

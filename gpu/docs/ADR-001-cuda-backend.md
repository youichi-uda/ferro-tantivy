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
- The N = 1 M shape comes in below 3× on the **cold** path (re-uploads
  the corpus on every call). Profile breakdown for the cold call:
  corpus PCIe upload (96 MB) ~6 ms + on-device unpack ~1 ms + INT8
  GEMM ~10 ms + correction ~1 ms + result PCIe download (256 MB)
  ~21 ms ≈ 40 ms — the remaining ≈40 ms is serial-stream overhead and
  pageable-host transfer cost. Wave 4's 8.3 GP/s reference assumed a
  **device-resident corpus**, which is the natural production calling
  pattern (BBQ corpora are immutable per Tantivy segment).
- **Phase 4 — `CudaTensorCoreKernel::prepare_corpus` + `CudaBinaryCorpus::compute_batched`**:
  the corpus is uploaded + unpacked + popcount-summed once and held
  on the device until the handle is dropped. Each subsequent query
  batch only pays the (small) query upload + result download. Measured
  on RTX 4070 Ti SUPER:
  | shape (Q × N × dim) | WGSL | CUDA cold | CUDA cached | cached / WGSL |
  |---------------------|------|-----------|-------------|---------------|
  | 64 × 100 000 × 768  | 22.7 ms | 4.10 ms | 2.29 ms | **9.89×** |
  | 64 × 1 000 000 × 768 | 211.0 ms | 77.9 ms | 60.6 ms | **3.48×** |
  Raw timings: `gpu/benches/data/cuda_cached_vs_wgsl.json`. The cached
  path now clears the 3× Go threshold at the N = 1 M shape.
- Remaining headroom on the cached N = 1 M path (60 ms vs the Wave 4
  reference's ≈8 ms compute envelope) was dominated by the 256 MB
  result download (~21 ms) and serial-stream overhead. Phase 5-1
  closes most of that gap; the rest is staged in 5-2 / 5-3 / 5-4.
- **Phase 5-1 — pinned-output API
  (`compute_batched_into_pinned`)**: a new entry point on
  `CudaTensorCoreKernel` and `CudaBinaryCorpus` writes the `Q × N` `u32`
  distance matrix directly into a user-supplied page-locked host
  buffer, allocated via `CudaTensorCoreKernel::alloc_pinned_u32`. The
  pinned buffer uses `CU_MEMHOSTALLOC_DEFAULT` flags (cacheable; not
  the write-combined flavour exposed by `cudarc::CudaContext::alloc_pinned`,
  which would penalise host reads). This skips both the driver's
  staging-buffer copy and the pinned → `Vec<u32>` host memcpy that the
  `Vec`-returning entry point pays. The `Vec<u32>` API is unchanged —
  for very large outputs (≥ 100 MB) the staging-DMA-into-pageable
  path is still cheaper than DMA-into-pinned + a host memcpy.
  Re-measured on RTX 4070 Ti SUPER:
  | shape (Q × N × dim) | WGSL | CUDA cached `Vec` | CUDA cached pinned | pinned / WGSL |
  |---------------------|------|-------------------|--------------------|---------------|
  | 64 × 100 000 × 768  | 23.66 ms | 2.39 ms | 2.57 ms | **9.20×** |
  | 64 × 1 000 000 × 768 | 250.00 ms | 76.83 ms | **27.41 ms** | **9.12×** |
  Raw timings: `gpu/benches/data/cuda_cached_vs_wgsl.json` (the JSON
  now carries both `cuda_cached_ns` and `cuda_pinned_ns` per shape
  plus a `variants` block describing each). Bit-equivalence is held by
  the new `parity_pinned_buffer_matches_vec_path` parity test (4
  representative shapes; pinned bytes byte-equal to the CPU oracle and
  to the `Vec<u32>` path; buffer reuse across shrinking batches also
  exercised).
- **Phase 5-2 — double-buffered batched pipeline
  (`compute_batches_into_pinned`)**: a new entry point on
  `CudaBinaryCorpus` issues `B` query batches across two internal
  CUDA streams + workspaces, so the next batch's upload + GEMM +
  download overlaps the current batch's compute. Each batch lands in
  its own slice of one large pinned output buffer (offset
  `batch_idx × Q × N`), and the call returns once both pipeline
  streams have synchronised. The per-batch byte-equivalence vs the
  single-batch entry point is asserted in
  `parity_double_buffered_batches_match_single_path` (5 cases
  including odd batch counts so the `i % 2` slot routing flips on the
  final batch).
  Measured on RTX 4070 Ti SUPER at `B = 10` batches:
  | shape | serial pinned | double-buf | speedup |
  |-------|---------------|------------|---------|
  | Q=4  N=100k    | 3.01 ms   | 2.40 ms   | 1.25× |
  | Q=64 N=100k    | 22.05 ms  | 20.11 ms  | 1.10× |
  | Q=64 N=1M      | 271.69 ms | 206.09 ms | **1.32×** |
  Raw timings: `gpu/benches/data/cuda_doublebuf.json`. The gain is
  bounded by how much of the per-batch wall-clock isn't already
  in the GEMM — at the headline `Q = 64, N = 1 M, dim = 768` shape
  GEMM is ≈ 60-70 % of the per-batch budget, leaving only the
  remaining 30-40 % (upload + correction + download) available for
  overlap, so the realistic ceiling is ≈ 1.5×; we land at 1.32×
  steady-state (the first and last batches don't pipeline). For
  workloads with a smaller per-batch compute slot (low Q, low N)
  the same pipeline produces less absolute time saved but a
  similar ratio. There is no Go-threshold assertion on this number
  because pipeline gain is hardware-sensitive (PCIe gen, DRAM
  speed, SM count); the JSON is consumed only for regression
  tracking.
- Phase 5 still on the table: 5-3 BMMA 1-bit tensor-core PoC
  (eliminates the unpack to one-byte-per-bit and lifts arithmetic
  throughput by 8× if the inline-PTX path holds), and 5-4 real-data
  SIFT1M / GIST1M recall@k in `ferro-bench-runner`.
- **Phase C — `knn_search` end-to-end CUDA path**: the WGSL
  `top_k_select.wgsl` bitonic-merge top-K shader has been ported to
  CUDA (NVRTC, same algorithm, same tie-break), and
  `BinaryDistanceKernel::knn_search` now short-circuits through CUDA
  when the feature is enabled. The distance matrix stays on the device
  between GEMM and top-K, so only `Q × K × 8 B` crosses PCIe back.
  Measured on RTX 4070 Ti SUPER (CUDA path vs WGSL knn_search):
  | shape (Q × N × K × dim) | WGSL knn | CUDA knn |
  |--------------------------|----------|----------|
  | 64 × 10 000 × 100 × 768  | 4.92 ms  | 0.59 ms (cold) |
  | 64 × 100 000 × 100 × 768 | 13.27 ms | 5.97 ms (cold) |
  Two new public APIs support the cached calling pattern:
  - `CudaTensorCoreKernel::knn_search(...)` — one-shot, builds a
    cached corpus internally and runs end-to-end.
  - `CudaBinaryCorpus::knn_search(...)` — runs against a corpus
    handle that was already prepared via `prepare_corpus`. This is the
    production pattern for repeated query batches against the same
    BBQ segment.
  Bit-equivalence with the WGSL `top_k_select.wgsl` is asserted in the
  parity test `parity_knn_search_matches_cpu` (5 representative
  shapes) — distances *and* tie-broken ids match the CPU oracle
  exactly.
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

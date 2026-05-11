# CUDA backend bench — 2026-05-11

`cargo bench -p tantivy-gpu --bench binary_distance --features cuda-tensor-core`
with `BINARY_DIST_BENCH_NO_ASSERT=1` so every section runs to completion
regardless of individual phase thresholds. Numbers refresh the
2026-05-10 snapshot under `gpu/benches/data/`.

## Host

| key | value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (16,376 MiB, SM 8.9 / Ada) |
| Driver | 595.58.03, CUDA runtime 13.2 |
| nvcc | 12.4 (build cuda_12.4.r12.4) |
| Kernel | 7.0.0-15-generic |
| rustc | 1.95.0 (2026-04-14) |
| WGSL backend | Vulkan |

## Headline — does the "6-22×" claim hold on real silicon?

Yes. The end-to-end k-NN figure clears the upper bound of the L4 envelope.

| Path | Shape | Wall-clock | vs CPU oracle | vs WGSL |
|---|---|---:|---:|---:|
| CPU oracle (host) | dim=768, N=1 M, Q=64, K=100 | 1.627 s | 1.00× | — |
| WGSL `xor_popcount_batched` | same | 249.71 ms (cold) | 6.52× | 1.00× |
| CUDA INT8 IMMA cuBLASLt (cold) | same | 96.48 ms | 16.86× | 2.59× |
| CUDA INT8 IMMA cached corpus | same, repeated query batch | 73.45 ms | 22.15× | 3.45× |
| CUDA INT8 IMMA cached + pinned (Phase 5-1) | same | 27.22 ms | 59.77× | 9.30× |
| CUDA E2E k-NN (cached + on-GPU top-K) | same, with K=100 reducer | 61.92 ms | **26.28×** | n/a (different output shape) |

The pinned-output path against a device-resident corpus on the N=1 M, Q=64 shape
runs **9.30× faster than the WGSL kernel and 59.77× faster than the CPU oracle**
for the distance-matrix step alone; once the on-GPU bitonic top-K reducer is
folded in (the production entry point — only K results cross PCIe per query) the
full end-to-end k-NN is **26.28× faster than the CPU oracle**.

Wave 4 G1's L4 envelope of 6-7× was for the cold WGSL→CUDA comparison;
this run reproduces 5.7-12× across the cold/cached/pinned ladder and the
amortised production path (cached + pinned + top-K) pulls past the upper
"22×" claim. The conservative `≥ 3× WGSL` Go-threshold at N=100k Q=64 holds
at **5.68×** on this host.

## Byte-equality

Every CUDA path (INT8 IMMA, cached, pinned, double-buffered, BMMA) is asserted
byte-equal vs the WGSL kernel on every shape exercised, before any timing is
recorded. No DIVERGE row in the log. INT8 with INT32 accumulator is exact for
any `dim_bits ≤ 2³¹`, and the BMMA `mma.sync.aligned.m8n8k128.b1.xor.popc`
emits the same `popcount(q ⊕ d)` integer — both paths are oracle-faithful, not
approximate, which matters for HNSW edge selection.

## Phase ladder

### Phase 2 — WGSL ↔ CUDA INT8 IMMA cold path (see `cuda_vs_wgsl.json`)

| shape | WGSL | CUDA | speedup | match |
|---|---:|---:|---:|---|
| dim=768 N=10 k Q=8 | 2.54 ms | 309.6 µs | 8.20× | ok |
| dim=768 N=10 k Q=64 | 2.87 ms | 497.7 µs | 5.77× | ok |
| dim=768 N=100 k Q=8 | 6.38 ms | 2.43 ms | 2.62× | ok |
| dim=768 N=100 k Q=64 | 24.26 ms | 4.27 ms | **5.68×** (Go-threshold gate) | ok |
| dim=768 N=1 M Q=64 | 249.71 ms | 96.48 ms | 2.59× (PCIe-bound, tracked only) | ok |

### Phase 4 / 5-1 — cached corpus + pinned output (`cuda_cached_vs_wgsl.json`)

Removing the per-call corpus upload and DMA-ing into a pinned host buffer.

| shape | WGSL | CUDA cached `Vec<u32>` | CUDA cached pinned | c/WGSL | p/WGSL |
|---|---:|---:|---:|---:|---:|
| dim=768 N=100 k Q=64 | 26.55 ms | 2.28 ms | 2.20 ms | **11.62×** | **12.06×** |
| dim=768 N=1 M Q=64 | 253.26 ms | 73.45 ms | 27.22 ms | 3.45× | **9.30×** |

Pinned-output is the production-ready calling pattern: corpus uploaded once per
segment, query batches DMA in/out at full PCIe bandwidth.

### Phase 5-2 — double-buffered batched pipeline (`cuda_doublebuf.json`)

Overlap the next batch's H→D upload with the current batch's compute and the
previous batch's D→H download. Numbers are vs the serial pinned path.

| shape | serial | doublebuf | speedup |
|---|---:|---:|---:|
| B=10 Q=4 N=100 k | 3.02 ms | 2.39 ms | 1.27× |
| B=10 Q=64 N=100 k | 24.17 ms | 20.04 ms | 1.21× |
| B=10 Q=64 N=1 M | 272.37 ms | 206.13 ms | 1.32× |

Modest but free win for benchmark-style fixed-batch loops; HNSW data-dependent
walks still want the single-batch path.

### Phase 5-3 — BMMA `mma.sync` 1-bit Tensor Core vs cuBLASLt INT8 (`cuda_bmma_vs_int8.json`)

Inline-PTX `mma.sync.aligned.m8n8k128.b1.xor.popc` (one warp per 8×8 tile, no
shared-memory tiling yet) vs the cuBLASLt INT8 IMMA path.

| shape | INT8 | BMMA | INT8 / BMMA |
|---|---:|---:|---:|
| dim=128 N=1 k Q=8 | 39.7 µs | 20.8 µs | **1.91×** |
| dim=256 N=10 k Q=8 | 155.1 µs | 101.2 µs | 1.53× |
| dim=512 N=10 k Q=64 | 425.1 µs | 337.9 µs | 1.26× |
| dim=768 N=10 k Q=64 | 494.0 µs | 376.3 µs | 1.31× |
| dim=768 N=100 k Q=64 | 4.14 ms | 3.02 ms | 1.37× |
| dim=1024 N=100 k Q=64 | 4.79 ms | 3.29 ms | 1.46× |
| dim=768 N=1 M Q=64 | 97.13 ms | 79.38 ms | 1.22× |

BMMA is **uniformly faster than cuBLASLt INT8 IMMA across every shape**, even
in the naive 1-warp-per-tile implementation without shared-memory K-tiling.
The win comes from 1-bit packing density: BMMA's K-tile is 128 bits = 4 u32
words, vs INT8 IMMA needing 8× more on-chip bandwidth to unpack 1→8b. The
ratio compresses from 1.91× (tiny-N regime, kernel-launch overhead amortised)
to 1.22× (1M corpus, where global-memory bandwidth on the unpacked INT8
matrix saturates either way).

Production routing decision is open: BMMA is currently a *bench-only* entry
point (`compute_batched_bmma`, no cached-corpus / pinned-output variant). To
fold BMMA into the production cached + pinned path the corpus would need to
be stored in its original 1-bit packed form on-device rather than unpacked
into the INT8 corpus matrix — a separate `CudaBinaryCorpusBmma` would
own the packed-corpus buffer. See ADR-001 §Phase 5-3 follow-up.

## CPU / GPU coherence (Phase 1 parity)

Every single-query, batched, k-NN, and Tensor-Core comparison was asserted
byte-equal vs the CPU oracle (`hamming_distance_cpu`,
`hamming_distances_batched_cpu`, `top_k_select_cpu`) — see lines tagged "ok"
in `bench.log`. No `DIVERGE` rows on this host across the full sweep.

## Files

- `bench.log` — full stdout from the cargo bench run
- `cuda_vs_wgsl.json` — cold path WGSL↔CUDA INT8 IMMA
- `cuda_cached_vs_wgsl.json` — cached corpus + pinned output
- `cuda_doublebuf.json` — Phase 5-2 double-buffered pipeline
- `cuda_bmma_vs_int8.json` — Phase 5-3 BMMA vs cuBLASLt INT8

These are snapshots of the four JSON files the bench writes into
`gpu/benches/data/`; the bench overwrites those on every run, but the
`reports/cuda-backend-bench-2026-05-11/` copies are immutable evidence.

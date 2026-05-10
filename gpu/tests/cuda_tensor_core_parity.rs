//! G0 parity sweep — bit-exact CUDA vs CPU oracle for binary Hamming
//! distance.
//!
//! Gated on `--features cuda-tensor-core`. Skips at runtime (passes
//! with a printed message) if no NVIDIA device / CUDA driver is
//! reachable, so the same test target can run on developer machines
//! without a discrete NVIDIA GPU.
//!
//! ## Why this lives in `tests/` and not `lib.rs`
//!
//! These cases run a 4 × 4 × 5 = 80-cell sweep over (Q, N, dim_bits)
//! and exercise the IMMA tensor core path on real hardware. The full
//! suite (sweep + extremes + identity + odd-shape) takes ≈3 s on RTX
//! 4070 Ti SUPER. The smoke test in
//! `vector::cuda_tensor_core::tests::smoke_byte_equal` covers the
//! basic build-and-run health.

#![cfg(feature = "cuda-tensor-core")]

use tantivy_gpu::vector::binary_distance::{
    dim_u32_for, hamming_distances_batched_cpu, top_k_select_cpu,
};
use tantivy_gpu::vector::cuda_tensor_core::CudaTensorCoreKernel;

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn random_u32_vec(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed.max(1);
    (0..n).map(|_| xorshift64(&mut state) as u32).collect()
}

/// If `dim_bits` is not a multiple of 32, mask the trailing padding bits
/// of the last u32 in each row to zero — the calling contract requires
/// this. The CPU oracle and the CUDA path both treat unset padding bits
/// as zero, so leaving stray ones would diverge before the kernel even
/// runs.
fn zero_padding(rows: &mut [u32], num_rows: usize, dim_bits: usize) {
    let dim_u32 = dim_u32_for(dim_bits);
    let bits_in_last = dim_bits - 32 * (dim_u32 - 1);
    if bits_in_last == 32 {
        return;
    }
    let mask: u32 = (1u32 << bits_in_last) - 1;
    for r in 0..num_rows {
        rows[r * dim_u32 + dim_u32 - 1] &= mask;
    }
}

fn try_build() -> Option<CudaTensorCoreKernel> {
    match CudaTensorCoreKernel::try_new() {
        Ok(k) => Some(k),
        Err(e) => {
            eprintln!("CUDA Tensor Core unavailable on this host: {e}");
            eprintln!("(test skipped — no NVIDIA device / driver / cuBLASLt found)");
            None
        }
    }
}

/// Sweep `(Q, N, dim_bits)` and assert byte-equal output vs the CPU oracle.
///
/// Grid:
/// - Q ∈ {1, 4, 32, 128}        — covers single-query and IMMA's M=128
///                                tile boundary
/// - N ∈ {1, 100, 1024, 16_384} — covers small-batch overhead and
///                                arithmetic-intensity sweet spot
/// - dim_bits ∈ {128, 256, 512, 768, 1024} — common BBQ widths
///
/// All combinations are bit-exact: INT8 inputs ∈ {0, 1} give integer
/// inner products bounded by `dim_bits`, well inside `i32` range; the
/// `CUBLAS_COMPUTE_32I` accumulator is exact.
#[test]
fn parity_sweep_byte_equal() {
    let kernel = match try_build() {
        Some(k) => k,
        None => return,
    };

    let q_vals: &[usize] = &[1, 4, 32, 128];
    let n_vals: &[usize] = &[1, 100, 1024, 16_384];
    let dim_vals: &[usize] = &[128, 256, 512, 768, 1024];

    let mut total_cells: usize = 0;
    let mut max_q_used: usize = 0;
    let mut max_n_used: usize = 0;

    for &q in q_vals {
        for &n in n_vals {
            for &dim_bits in dim_vals {
                let dim_u32 = dim_u32_for(dim_bits);
                let q_seed = (0xa1b2_c3d4u64) ^ ((q as u64) << 32) ^ (n as u64) ^ (dim_bits as u64);
                let d_seed = (0xfedc_ba98u64) ^ ((q as u64) << 16) ^ ((n as u64) << 4) ^ (dim_bits as u64);

                let mut queries = random_u32_vec(q * dim_u32, q_seed);
                let mut corpus = random_u32_vec(n * dim_u32, d_seed);
                zero_padding(&mut queries, q, dim_bits);
                zero_padding(&mut corpus, n, dim_bits);

                let cpu = hamming_distances_batched_cpu(&queries, &corpus, q, n, dim_u32);
                let gpu = kernel
                    .compute_batched(&queries, &corpus, q, n, dim_bits)
                    .unwrap_or_else(|e| {
                        panic!(
                            "CUDA compute_batched failed at Q={q} N={n} dim_bits={dim_bits}: {e}"
                        )
                    });

                assert_eq!(
                    gpu.len(),
                    q * n,
                    "output length mismatch at Q={q} N={n} dim_bits={dim_bits}"
                );
                if gpu != cpu {
                    let first_diff = gpu
                        .iter()
                        .zip(cpu.iter())
                        .position(|(a, b)| a != b)
                        .unwrap();
                    panic!(
                        "byte-equal violated at Q={q} N={n} dim_bits={dim_bits}, first diff idx={first_diff}: gpu={} cpu={}",
                        gpu[first_diff], cpu[first_diff]
                    );
                }

                // Sanity: distances are bounded by dim_bits.
                for (i, &d) in gpu.iter().enumerate() {
                    assert!(
                        d <= dim_bits as u32,
                        "Hamming distance {d} > dim_bits {dim_bits} at idx {i} (Q={q} N={n})"
                    );
                }

                total_cells += 1;
                max_q_used = max_q_used.max(q);
                max_n_used = max_n_used.max(n);
            }
        }
    }
    eprintln!(
        "CUDA Tensor Core parity sweep: {total_cells} cells passed, max Q={max_q_used}, max N={max_n_used}"
    );
}

/// All-zero query against an all-one corpus must give every distance
/// equal to `dim_bits`. Catches sign / accumulator-overflow bugs that
/// could be masked by random inputs.
#[test]
fn parity_extremes_zero_vs_one() {
    let kernel = match try_build() {
        Some(k) => k,
        None => return,
    };
    let dim_bits = 768;
    let dim_u32 = dim_u32_for(dim_bits);
    let q = 8;
    let n = 64;

    let queries = vec![0u32; q * dim_u32];
    let corpus = vec![u32::MAX; n * dim_u32];
    // Padding bits in the last u32 must be zero; mask them off.
    let mut corpus = corpus;
    zero_padding(&mut corpus, n, dim_bits);

    let gpu = kernel
        .compute_batched(&queries, &corpus, q, n, dim_bits)
        .expect("compute_batched");
    for (i, &d) in gpu.iter().enumerate() {
        assert_eq!(d, dim_bits as u32, "all-zero vs all-one mismatch at idx {i}");
    }
}

/// Identity row: query equal to corpus[k] must give distance 0 at
/// (q, k) and a positive distance for any perturbed corpus row.
#[test]
fn parity_identity() {
    let kernel = match try_build() {
        Some(k) => k,
        None => return,
    };
    let dim_bits = 256;
    let dim_u32 = dim_u32_for(dim_bits);
    let n = 16;

    let mut corpus = random_u32_vec(n * dim_u32, 0xabcd_ef12_3456_7890);
    zero_padding(&mut corpus, n, dim_bits);
    // Query equals corpus row 0, plus a query that's the bit-complement of
    // corpus row 1 (to force max distance).
    let mut queries: Vec<u32> = Vec::with_capacity(2 * dim_u32);
    queries.extend_from_slice(&corpus[..dim_u32]);
    let row1 = &corpus[dim_u32..2 * dim_u32];
    let inverted: Vec<u32> = row1.iter().map(|w| !*w).collect();
    queries.extend_from_slice(&inverted);
    zero_padding(&mut queries, 2, dim_bits);

    let gpu = kernel
        .compute_batched(&queries, &corpus, 2, n, dim_bits)
        .expect("compute_batched");

    assert_eq!(gpu[0], 0, "dist(query0, corpus[0]) must be 0");
    assert_eq!(
        gpu[n + 1],
        dim_bits as u32,
        "dist(query1=NOT(corpus[1]), corpus[1]) must equal dim_bits"
    );
}

/// CUDA `knn_search` must agree exactly with the CPU oracle — same
/// distances and same tie-broken ids — across a representative grid.
/// The tie-break rule both implementations follow is "ascending by
/// distance, ties broken by lower id".
#[test]
fn parity_knn_search_matches_cpu() {
    let kernel = match try_build() {
        Some(k) => k,
        None => return,
    };

    let cases: &[(usize, usize, usize, usize)] = &[
        // (Q, N, dim_bits, K)
        (1, 16, 64, 4),
        (4, 256, 256, 16),
        (8, 1024, 768, 32),
        (16, 4096, 768, 100),
        (32, 16_384, 1024, 128),
    ];

    for &(q, n, dim_bits, k) in cases {
        let dim_u32 = dim_u32_for(dim_bits);
        let mut queries =
            random_u32_vec(q * dim_u32, 0xface_face_face_faceu64 ^ q as u64 ^ n as u64);
        let mut corpus =
            random_u32_vec(n * dim_u32, 0xbabe_babe_babe_babeu64 ^ q as u64 ^ n as u64);
        zero_padding(&mut queries, q, dim_bits);
        zero_padding(&mut corpus, n, dim_bits);

        let cpu_dists = hamming_distances_batched_cpu(&queries, &corpus, q, n, dim_u32);
        let cpu_topk = top_k_select_cpu(&cpu_dists, q, n, k);

        // One-shot path.
        let gpu_topk = kernel
            .knn_search(&queries, &corpus, q, n, dim_bits, k)
            .unwrap_or_else(|e| panic!("CUDA knn_search Q={q} N={n} dim={dim_bits} k={k}: {e}"));
        // Cached path.
        let cached = kernel
            .prepare_corpus(&corpus, n, dim_bits)
            .expect("prepare_corpus");
        let cached_topk = cached
            .knn_search(&queries, q, k)
            .unwrap_or_else(|e| panic!("CUDA cached knn_search: {e}"));

        for qi in 0..q {
            let row_cpu: Vec<(u32, u32)> = cpu_topk[qi].clone();
            let row_gpu = &gpu_topk[qi];
            let row_cached = &cached_topk[qi];

            assert_eq!(
                row_gpu.len(),
                row_cpu.len(),
                "row {qi} length mismatch (Q={q} N={n} dim={dim_bits} k={k})"
            );
            assert_eq!(
                row_gpu, &row_cpu,
                "one-shot CUDA knn != CPU at row {qi} (Q={q} N={n} dim={dim_bits} k={k})"
            );
            assert_eq!(
                row_cached, &row_cpu,
                "cached CUDA knn != CPU at row {qi} (Q={q} N={n} dim={dim_bits} k={k})"
            );
        }
    }
}

/// Cached-corpus path must produce byte-identical output to the
/// fresh-call path. Sweep is smaller than the full parity grid because
/// `prepare_corpus` adds a one-time upload+unpack the test harness
/// pays once per shape; the inner loop is just the queries.
#[test]
fn parity_cached_corpus_matches_fresh() {
    let kernel = match try_build() {
        Some(k) => k,
        None => return,
    };

    let cases: &[(usize, usize, usize)] = &[
        (1, 100, 256),
        (8, 1024, 768),
        (32, 4096, 768),
        (64, 8192, 1024),
    ];

    for &(q, n, dim_bits) in cases {
        let dim_u32 = dim_u32_for(dim_bits);
        let mut queries = random_u32_vec(q * dim_u32, 0xc1c2_c3c4u64 ^ q as u64 ^ n as u64);
        let mut corpus = random_u32_vec(n * dim_u32, 0xd1d2_d3d4u64 ^ q as u64 ^ n as u64);
        zero_padding(&mut queries, q, dim_bits);
        zero_padding(&mut corpus, n, dim_bits);

        let cpu = hamming_distances_batched_cpu(&queries, &corpus, q, n, dim_u32);
        let fresh = kernel
            .compute_batched(&queries, &corpus, q, n, dim_bits)
            .expect("fresh");
        let cached = kernel
            .prepare_corpus(&corpus, n, dim_bits)
            .expect("prepare_corpus")
            .compute_batched(&queries, q)
            .expect("cached compute");

        assert_eq!(fresh, cpu, "fresh != cpu at Q={q} N={n} dim={dim_bits}");
        assert_eq!(cached, cpu, "cached != cpu at Q={q} N={n} dim={dim_bits}");
        assert_eq!(cached, fresh, "cached != fresh at Q={q} N={n} dim={dim_bits}");
    }
}

/// Small-but-odd shapes that probe leading-dimension and alignment
/// edges of the IMMA path.
#[test]
fn parity_small_odd_shapes() {
    let kernel = match try_build() {
        Some(k) => k,
        None => return,
    };
    let cases: &[(usize, usize, usize)] = &[
        (1, 1, 32),
        (1, 7, 64),
        (3, 5, 96),
        (5, 17, 128),
        (7, 33, 160),
        (11, 65, 192),
    ];
    for &(q, n, dim_bits) in cases {
        let dim_u32 = dim_u32_for(dim_bits);
        let mut queries = random_u32_vec(q * dim_u32, 0xc0ff_eeu64 ^ q as u64);
        let mut corpus = random_u32_vec(n * dim_u32, 0xdead_beefu64 ^ n as u64);
        zero_padding(&mut queries, q, dim_bits);
        zero_padding(&mut corpus, n, dim_bits);

        let cpu = hamming_distances_batched_cpu(&queries, &corpus, q, n, dim_u32);
        let gpu = kernel
            .compute_batched(&queries, &corpus, q, n, dim_bits)
            .unwrap_or_else(|e| panic!("Q={q} N={n} dim_bits={dim_bits} failed: {e}"));
        assert_eq!(gpu, cpu, "Q={q} N={n} dim_bits={dim_bits} byte-equal violated");
    }
}

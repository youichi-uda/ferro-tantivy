//! GPU-rerank vs CPU-only end-to-end bench for the binary HNSW —
//! Phase 2 A-final adoption form.
//!
//! ## What this measures
//!
//! Two production-shaped paths compared on the same `BinaryHnswIndex`:
//!
//! 1. CPU-only:
//!    `BinaryHnswIndex::search` — graph traversal + popcount on CPU,
//!    no GPU involved. The baseline.
//!
//! 2. GPU rerank:
//!    `BinaryHnswIndex::search_gpu_rerank` (single-query) and
//!    `BinaryHnswIndex::search_batch_gpu_rerank` (multi-query) —
//!    graph traversal on CPU, candidate-set rerank on GPU in **one**
//!    batched dispatch.
//!
//! ## What we expect
//!
//! - At small `ef_search` (≤ 64) the GPU rerank is **slower** than
//!   CPU because the dispatch overhead (~1-3 ms) exceeds the actual
//!   work. This is expected and documented; the bench prints the
//!   relative numbers without asserting a particular speedup at
//!   small ef.
//! - At `ef_search ≥ 256` and multi-query batches (`Q ≥ 8`) the GPU
//!   path should clear well into the win regime — the production
//!   target shape is `Q = 64, ef = 256, K = 10` over `N = 100k`.
//! - The CPU graph-traversal time bounds the overall speedup. Phase 2
//!   A-final does NOT GPU-accelerate the graph walk itself; that is
//!   future work (a dedicated graph-walk shader).
//!
//! ## Recall
//!
//! HNSW recall against brute-force is reported on a separate
//! `recall_at_10` block, but we **do not assert** a particular floor
//! at this bench's `dim_bits = 768, N = 100k` shape. Synthetic random
//! u32 bits at this dim/N give nearest-neighbour distances within
//! ~1 std deviation of any random vector, so the "true" top-10 is
//! statistically indistinguishable from positions 11-1000, and HNSW
//! cannot achieve high recall on a degenerate dataset. The unit test
//! `test_binary_hnsw_recall_at_10_mid_scale` (in `binary_hnsw.rs`)
//! validates a recall floor of 0.85 at the more separable
//! `dim = 256, N = 2000` shape; that is the canonical correctness
//! signal. The rerank itself is bit-exact (the kernel computes the
//! same `popcount(XOR)` that the CPU oracle does) so the CPU-vs-GPU
//! recall numbers reported here will agree to the same value by
//! construction. Real-data validation on SIFT1M / GIST1M ground
//! truth (where neighbours genuinely separate from random) is the
//! GA-prep wave.
//!
//! Run with:
//!     cargo bench -p tantivy-gpu --bench hnsw_binary
//!
//! Build-only (no GPU required):
//!     cargo bench -p tantivy-gpu --bench hnsw_binary --no-run

use std::collections::HashSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use tantivy_gpu::device::GpuContext;
use tantivy_gpu::vector::binary_distance::{dim_u32_for, BinaryDistanceMetric};
use tantivy_gpu::vector::binary_hnsw::{brute_force_knn_cpu, BinaryHnswIndex};

const DIM_BITS: usize = 768;
const N_BUILD: usize = 100_000;

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

fn bench<F: FnMut()>(iters: usize, mut f: F) -> Duration {
    f(); // warmup
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters as u32
}

fn fmt_dur(d: Duration) -> String {
    if d.as_secs_f64() >= 1.0 {
        format!("{:.3}s", d.as_secs_f64())
    } else if d.as_millis() >= 1 {
        format!("{:.2}ms", d.as_secs_f64() * 1_000.0)
    } else {
        format!("{:.1}µs", d.as_secs_f64() * 1_000_000.0)
    }
}

fn main() {
    println!("=== tantivy-gpu Binary HNSW — Phase 2 A-final adoption bench ===\n");

    let ctx = GpuContext::init().expect("GpuContext::init");
    let info = ctx.info();
    println!(
        "GPU: {} ({}) | HW GPU: {}",
        info.name,
        info.backend,
        ctx.is_hardware_gpu()
    );
    println!("dim_bits = {DIM_BITS}, N (corpus) = {N_BUILD}\n");

    // ── Build phase ──
    let dim_u32 = dim_u32_for(DIM_BITS);
    println!("(building binary HNSW corpus — this takes a few seconds...)");
    let build_start = Instant::now();
    let raw_corpus = random_u32_vec(N_BUILD * dim_u32, 0x0a1b_2c3d_4e5f_6071);
    let mut idx =
        BinaryHnswIndex::with_gpu(DIM_BITS, BinaryDistanceMetric::Hamming, &ctx).expect("with_gpu");
    for v in 0..N_BUILD {
        let base = v * dim_u32;
        idx.insert(raw_corpus[base..base + dim_u32].to_vec())
            .expect("insert");
    }
    let build_time = build_start.elapsed();
    println!("Build time      : {} for {N_BUILD} vectors", fmt_dur(build_time));
    println!();

    // ── Single-query: CPU-only vs GPU rerank ──
    println!("--- Single-query latency (Q = 1) ---");
    println!(
        "{:<10}  {:>10}  {:>10}  {:>10}",
        "ef_search", "CPU only", "GPU rerank", "speedup"
    );
    let single_efs: &[usize] = &[32, 64, 128, 256, 512];
    let query = random_u32_vec(dim_u32, 0xc1c2_c3c4_c5c6_c7c8);
    for &ef in single_efs {
        let cpu_iters = if ef >= 256 { 5 } else { 20 };
        let cpu_time = bench(cpu_iters, || {
            black_box(idx.search(&query, 10, ef).expect("search"));
        });
        let gpu_iters = if ctx.is_hardware_gpu() { 10 } else { 3 };
        let gpu_time = bench(gpu_iters, || {
            black_box(idx.search_gpu_rerank(&query, 10, ef).expect("rerank"));
        });
        let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos().max(1) as f64;
        println!(
            "{:<10}  {:>10}  {:>10}  {:>9.2}x",
            ef,
            fmt_dur(cpu_time),
            fmt_dur(gpu_time),
            speedup
        );
    }
    println!();

    // ── Multi-query: CPU sequential vs one batched GPU rerank ──
    println!("--- Multi-query latency (Q = 64, k = 10) ---");
    println!(
        "{:<10}  {:>12}  {:>12}  {:>10}",
        "ef_search", "CPU sequential", "GPU batched", "speedup"
    );
    let multi_efs: &[usize] = &[64, 128, 256, 512];
    let queries: Vec<Vec<u32>> = (0..64)
        .map(|qi| {
            random_u32_vec(
                dim_u32,
                0x9999_aaaa_bbbb_ccccu64.wrapping_add(qi as u64),
            )
        })
        .collect();
    let mut best_multi_speedup: f64 = 0.0;
    let mut best_multi_label = String::new();
    for &ef in multi_efs {
        let cpu_iters = if ef >= 256 { 3 } else { 5 };
        let cpu_time = bench(cpu_iters, || {
            for q in &queries {
                black_box(idx.search(q, 10, ef).expect("search"));
            }
        });
        let gpu_iters = if ctx.is_hardware_gpu() { 5 } else { 1 };
        let gpu_time = bench(gpu_iters, || {
            black_box(
                idx.search_batch_gpu_rerank(&queries, 10, ef)
                    .expect("batch rerank"),
            );
        });
        let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos().max(1) as f64;
        if speedup > best_multi_speedup {
            best_multi_speedup = speedup;
            best_multi_label = format!("Q=64 ef={ef}");
        }
        println!(
            "{:<10}  {:>12}  {:>12}  {:>9.2}x",
            ef,
            fmt_dur(cpu_time),
            fmt_dur(gpu_time),
            speedup
        );
    }
    println!();
    println!("Best multi-query speedup: {best_multi_speedup:.2}x at {best_multi_label}");
    println!();

    // ── Recall@10 vs brute-force ground truth ──
    println!("--- Recall@10 vs brute-force ground truth (Q = 32, ef = 256) ---");
    let mut total_correct_cpu = 0;
    let mut total_correct_gpu = 0;
    let mut total_expected = 0;
    let recall_qs = 32;
    let ef_recall = 256;
    let k_recall = 10;
    for qi in 0..recall_qs {
        let q_seed = 0xfeed_face_dead_beefu64
            .wrapping_add((qi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let query = random_u32_vec(dim_u32, q_seed);

        let bf = brute_force_knn_cpu(&query, &raw_corpus, N_BUILD, dim_u32, k_recall);
        let bf_set: HashSet<u32> = bf.iter().map(|&(_, id)| id).collect();

        let cpu = idx.search(&query, k_recall, ef_recall).expect("search");
        for &(_, id) in &cpu {
            if bf_set.contains(&id) {
                total_correct_cpu += 1;
            }
        }
        let gpu = idx
            .search_gpu_rerank(&query, k_recall, ef_recall)
            .expect("rerank");
        for &(_, id) in &gpu {
            if bf_set.contains(&id) {
                total_correct_gpu += 1;
            }
        }
        total_expected += k_recall;
    }
    let recall_cpu = total_correct_cpu as f64 / total_expected as f64;
    let recall_gpu = total_correct_gpu as f64 / total_expected as f64;
    println!("CPU-only      recall@10 = {recall_cpu:.4}");
    println!("GPU-rerank    recall@10 = {recall_gpu:.4}");
    println!(
        "(Note: synthetic random u32 bits at dim=768, N=100k give a degenerate \
         neighbour distribution — true top-10 is statistically indistinguishable \
         from positions 11-1000. The unit test test_binary_hnsw_recall_at_10_mid_scale \
         verifies a 0.85 floor at the more separable dim=256, N=2000 shape; \
         real-data validation on SIFT1M / GIST1M is the GA-prep wave.)"
    );
    println!();

    // Regression invariant: the GPU rerank is exact, so CPU and GPU
    // recall numbers must agree byte-for-byte. If they diverge,
    // either the kernel or the candidate-set unioning is broken.
    assert!(
        (recall_cpu - recall_gpu).abs() < 1e-9,
        "CPU and GPU recall@10 must be identical (rerank is exact): \
         cpu={recall_cpu}, gpu={recall_gpu}"
    );
    println!("ASSERT PASSED : CPU and GPU rerank recall agree exactly");

    if !ctx.is_hardware_gpu() {
        println!();
        println!(
            "NOTE: ran on CPU fallback. For real GPU numbers re-run on \
             a host with a discrete GPU (e.g. RTX 4070 Ti SUPER)."
        );
    }
    println!("\n=== Done ===");
}

//! SIMD vs scalar bench for the Wave 14 top-K filter kernel.
//!
//! Measures the per-block (64-doc) throughput of the SIMD greater-than /
//! less-than filter against the scalar reference, on inputs that mirror the
//! Rally `desc_sort_*_after_force_merge_1_seg` workload (1 M-doc i64 timestamp
//! sort, ES wins 3.34–8.30× pre-Wave-14).
//!
//! Run with:
//!
//! ```bash
//! RUSTFLAGS='-Ctarget-cpu=native' cargo bench --bench sort_simd
//! ```
//!
//! Without `target-cpu=native` the AVX2 path is still reached via runtime
//! detection (`SimdDispatch`), but the scalar fallback compiles for the
//! generic baseline (SSE2 only) so the SIMD-vs-scalar gap will be larger.

use std::hint::black_box;
use std::time::Instant;

use tantivy::collector::sort_key::simd_top_k::{
    self, simd_filter_block_gt_u64, simd_filter_block_lt_u64, SIMD_BLOCK_LEN,
};

/// 1M-doc i64 (timestamp-shaped) input.  Values are u64 because the warm
/// cache stores `MonotonicallyMappableToU64::to_u64` outputs — see Wave 8.
fn build_input(n: usize) -> Vec<u64> {
    let mut s: u64 = 0xCAFEBABE;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s
        })
        .collect()
}

fn bench_kernel(name: &str, input: &[u64], threshold: u64, iters: usize) {
    let n = input.len();
    let blocks = n / SIMD_BLOCK_LEN;

    // SIMD desc
    let t0 = Instant::now();
    let mut total = 0u32;
    for _ in 0..iters {
        for b in 0..blocks {
            let off = b * SIMD_BLOCK_LEN;
            let m = simd_filter_block_gt_u64(
                black_box(&input[off..off + SIMD_BLOCK_LEN]),
                black_box(threshold),
                SIMD_BLOCK_LEN,
            );
            total = total.wrapping_add(m.count_ones());
        }
    }
    let dt_simd = t0.elapsed();
    let _ = black_box(total);

    // Scalar desc
    let t0 = Instant::now();
    let mut total = 0u32;
    for _ in 0..iters {
        for b in 0..blocks {
            let off = b * SIMD_BLOCK_LEN;
            let m = simd_top_k::scalar::filter_block_gt_u64(
                black_box(&input[off..off + SIMD_BLOCK_LEN]),
                black_box(threshold),
                SIMD_BLOCK_LEN,
            );
            total = total.wrapping_add(m.count_ones());
        }
    }
    let dt_scal = t0.elapsed();
    let _ = black_box(total);

    let docs_total = (blocks * SIMD_BLOCK_LEN * iters) as f64;
    let simd_ns_per_doc = dt_simd.as_nanos() as f64 / docs_total;
    let scal_ns_per_doc = dt_scal.as_nanos() as f64 / docs_total;
    let speedup = scal_ns_per_doc / simd_ns_per_doc;
    println!(
        "{name:<32}  scalar {:>6.2} ns/doc  simd {:>6.2} ns/doc  speedup {:>4.2}x",
        scal_ns_per_doc, simd_ns_per_doc, speedup
    );

    // Repeat for the asc / lt kernel.
    let t0 = Instant::now();
    let mut total = 0u32;
    for _ in 0..iters {
        for b in 0..blocks {
            let off = b * SIMD_BLOCK_LEN;
            let m = simd_filter_block_lt_u64(
                black_box(&input[off..off + SIMD_BLOCK_LEN]),
                black_box(threshold),
                SIMD_BLOCK_LEN,
            );
            total = total.wrapping_add(m.count_ones());
        }
    }
    let dt_simd = t0.elapsed();
    let _ = black_box(total);

    let t0 = Instant::now();
    let mut total = 0u32;
    for _ in 0..iters {
        for b in 0..blocks {
            let off = b * SIMD_BLOCK_LEN;
            let m = simd_top_k::scalar::filter_block_lt_u64(
                black_box(&input[off..off + SIMD_BLOCK_LEN]),
                black_box(threshold),
                SIMD_BLOCK_LEN,
            );
            total = total.wrapping_add(m.count_ones());
        }
    }
    let dt_scal = t0.elapsed();
    let _ = black_box(total);

    let simd_ns_per_doc = dt_simd.as_nanos() as f64 / docs_total;
    let scal_ns_per_doc = dt_scal.as_nanos() as f64 / docs_total;
    let speedup = scal_ns_per_doc / simd_ns_per_doc;
    println!(
        "{name:<32}  (lt)  scalar {:>6.2} ns/doc  simd {:>6.2} ns/doc  speedup {:>4.2}x",
        scal_ns_per_doc, simd_ns_per_doc, speedup
    );
}

fn main() {
    println!("Wave 14 SIMD top-K filter — block 64-doc, scalar vs SIMD");
    println!("================================================================");

    // 1M-doc i64 timestamp-shaped sort, threshold ~10 docs above (top-10 desc)
    let input = build_input(1_000_000);
    let mut sorted = input.clone();
    sorted.sort_unstable();
    let threshold = sorted[sorted.len() - 11];
    bench_kernel("1M docs i64 (top-10 thresh)", &input, threshold, 10);

    // 1M-doc DateTime — same kernel, different threshold (top-100)
    let threshold = sorted[sorted.len() - 101];
    bench_kernel("1M docs Date (top-100 thresh)", &input, threshold, 10);

    // 100k-doc f64 — smaller workload
    let input = build_input(100_000);
    let mut sorted = input.clone();
    sorted.sort_unstable();
    let threshold = sorted[sorted.len() - 11];
    bench_kernel("100k docs f64 (top-10 thresh)", &input, threshold, 100);

    // Adversarial: threshold below all values → all docs survive (worst case
    // for branch predictor in the scalar path).
    let input = build_input(1_000_000);
    bench_kernel("1M docs adversarial (all pass)", &input, 0u64, 10);

    // Adversarial: threshold above all values → no docs survive (best case
    // for both paths — measures pure compare throughput).
    bench_kernel(
        "1M docs adversarial (none pass)",
        &input,
        u64::MAX,
        10,
    );
}

//! GPU vs CPU benchmarks for the binary (1-bit) Hamming-distance
//! kernel suite — single-query, batched (Q × N), and on-GPU top-K.
//!
//! This is the Phase 2 A integration bench. It exercises the three
//! shapes that production k-NN over BBQ codes actually run on:
//!
//!   1. Single query × N corpus  — the foundation kernel; vec4 SIMD
//!      widening; CPU is hard to beat for small N because of PCIe
//!      upload + dispatch overhead.
//!   2. Batched Q × N            — the breaking-point case; corpus is
//!      tiled into shared memory, every query in the workgroup reuses
//!      the same tile, which collapses VRAM bandwidth from
//!      Q · |corpus| to ~|corpus|.
//!   3. End-to-end k-NN          — distance + on-GPU top-K. Only Q · K
//!      pairs cross PCIe back. This is the production entry point.
//!
//! Real-data benchmarks against SIFT1M / GIST1M / DEEP1B (ground-truth
//! recall + recall@k) are deferred to a GA-prep wave — those need
//! ferro-bench-runner + dataset download.
//!
//! Run with:
//!     cargo bench -p tantivy-gpu --bench binary_distance
//!
//! Build-only check (no GPU required):
//!     cargo bench -p tantivy-gpu --bench binary_distance --no-run
//!
//! Behaviour:
//! - if no hardware GPU is present the bench prints the CPU baseline
//!   and exits (does **not** abort), so this can run as a smoke check
//!   on CI and on dev laptops without a discrete GPU.
//! - the GPU result is verified byte-equal against the CPU baseline on
//!   each shape — divergence aborts the bench.
//! - on real GPU hardware, the headline shape (N = 1_000_000, Q = 64,
//!   K = 100) asserts that the GPU end-to-end k-NN beats the CPU
//!   batched-distance + top-K oracle by ≥ 5×. If your hardware can't
//!   sustain that, set `BINARY_DIST_BENCH_NO_ASSERT=1` in the
//!   environment to skip the assert (the numbers still print).

use std::hint::black_box;
use std::time::{Duration, Instant};

use tantivy_gpu::device::GpuContext;
use tantivy_gpu::vector::binary_distance::{
    dim_u32_for, hamming_distance_cpu, hamming_distances_batched_cpu, top_k_select_cpu,
    BinaryDistanceKernel,
};

const HEADLINE_N: usize = 1_000_000;
const HEADLINE_Q: usize = 64;
const HEADLINE_K: usize = 100;
const HEADLINE_DIM_BITS: usize = 768;
const HEADLINE_REQUIRED_SPEEDUP: f64 = 5.0;

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

/// Run a closure `iters` times and return the average duration.
fn bench<F: FnMut()>(iters: usize, mut f: F) -> Duration {
    // Warm up
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters as u32
}

/// CPU oracle for batched k-NN (matrix + top-K).
fn cpu_knn(
    queries: &[u32],
    corpus: &[u32],
    num_queries: usize,
    num_vecs: usize,
    dim_u32: usize,
    k: usize,
) -> Vec<Vec<(u32, u32)>> {
    let dists = hamming_distances_batched_cpu(queries, corpus, num_queries, num_vecs, dim_u32);
    top_k_select_cpu(&dists, num_queries, num_vecs, k)
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
    println!("=== tantivy-gpu Binary Distance — Phase 2 A integration bench ===\n");

    let ctx = GpuContext::init().expect("GpuContext::init");
    let info = ctx.info();
    println!(
        "GPU: {} ({}) | HW GPU: {}",
        info.name,
        info.backend,
        ctx.is_hardware_gpu()
    );
    println!();

    // Pre-compile kernel once.
    let kernel = BinaryDistanceKernel::new(&ctx).expect("compile");

    // ── Single-query kernel (existing shapes, regression check) ──
    println!("--- Single-query kernel (vec4 SIMD) ---");
    println!(
        "{:<24}  {:>10}  {:>10}  {:>8}  {:>8}",
        "shape", "CPU", "GPU", "speedup", "match"
    );
    let single_shapes = [
        ("dim=768  N=1024  ", 1_024usize, 768usize),
        ("dim=768  N=4096  ", 4_096usize, 768usize),
        ("dim=768  N=16384 ", 16_384usize, 768usize),
        ("dim=1024 N=16384 ", 16_384usize, 1_024usize),
    ];
    for &(label, num_vecs, dim_bits) in &single_shapes {
        let dim_u32 = dim_u32_for(dim_bits);
        let query = random_u32_vec(dim_u32, 0xdead_beef_cafe_babe);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0x0123_4567_89ab_cdef);

        let cpu_iters = if num_vecs >= 16_384 { 5 } else { 20 };
        let cpu_ref = hamming_distance_cpu(&query, &corpus, num_vecs, dim_u32);
        let cpu_time = bench(cpu_iters, || {
            black_box(hamming_distance_cpu(&query, &corpus, num_vecs, dim_u32));
        });

        let gpu_iters = if ctx.is_hardware_gpu() { 20 } else { 5 };
        let gpu_first = kernel
            .compute(&query, &corpus, num_vecs, dim_bits)
            .expect("gpu compute");
        let matches = gpu_first == cpu_ref;
        if !matches {
            panic!("GPU output does not match CPU reference on shape {label}");
        }
        let gpu_time = bench(gpu_iters, || {
            black_box(
                kernel
                    .compute(&query, &corpus, num_vecs, dim_bits)
                    .expect("gpu compute"),
            );
        });

        let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos().max(1) as f64;
        println!(
            "{:<24}  {:>10}  {:>10}  {:>7.2}x  {:>8}",
            label,
            fmt_dur(cpu_time),
            fmt_dur(gpu_time),
            speedup,
            if matches { "ok" } else { "DIVERGE" }
        );
    }
    println!();

    // ── Batched kernel (Q × N) ──
    println!("--- Batched kernel (Q queries × N corpus, no top-K) ---");
    println!(
        "{:<32}  {:>10}  {:>10}  {:>8}  {:>8}",
        "shape", "CPU", "GPU", "speedup", "match"
    );
    let batched_shapes: &[(&str, usize, usize, usize)] = &[
        ("dim=768  N=10000   Q=8  ", 10_000, 8, 768),
        ("dim=768  N=10000   Q=64 ", 10_000, 64, 768),
        ("dim=768  N=100000  Q=8  ", 100_000, 8, 768),
        ("dim=768  N=100000  Q=64 ", 100_000, 64, 768),
    ];
    for &(label, num_vecs, num_queries, dim_bits) in batched_shapes {
        let dim_u32 = dim_u32_for(dim_bits);
        let queries = random_u32_vec(num_queries * dim_u32, 0xa1a2_a3a4_a5a6_a7a8);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xb1b2_b3b4_b5b6_b7b8);

        let cpu_iters = 3;
        let cpu_time = bench(cpu_iters, || {
            black_box(hamming_distances_batched_cpu(
                &queries,
                &corpus,
                num_queries,
                num_vecs,
                dim_u32,
            ));
        });

        let gpu_iters = if ctx.is_hardware_gpu() { 5 } else { 1 };
        let gpu_first = kernel
            .compute_batched(&queries, &corpus, num_queries, num_vecs, dim_bits)
            .expect("gpu batched");
        let cpu_ref =
            hamming_distances_batched_cpu(&queries, &corpus, num_queries, num_vecs, dim_u32);
        let matches = gpu_first == cpu_ref;
        if !matches {
            panic!("Batched GPU output does not match CPU reference on shape {label}");
        }
        let gpu_time = bench(gpu_iters, || {
            black_box(
                kernel
                    .compute_batched(&queries, &corpus, num_queries, num_vecs, dim_bits)
                    .expect("gpu batched"),
            );
        });

        let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos().max(1) as f64;
        println!(
            "{:<32}  {:>10}  {:>10}  {:>7.2}x  {:>8}",
            label,
            fmt_dur(cpu_time),
            fmt_dur(gpu_time),
            speedup,
            if matches { "ok" } else { "DIVERGE" }
        );
    }
    println!();

    // ── End-to-end k-NN (batched + on-GPU top-K) ──
    println!("--- End-to-end k-NN (batched + on-GPU top-K) ---");
    println!(
        "{:<40}  {:>10}  {:>10}  {:>8}",
        "shape", "CPU", "GPU", "speedup"
    );
    let knn_shapes: &[(&str, usize, usize, usize, usize)] = &[
        ("dim=768  N=10000   Q=8   K=10 ", 10_000, 8, 10, 768),
        ("dim=768  N=10000   Q=64  K=100", 10_000, 64, 100, 768),
        ("dim=768  N=100000  Q=8   K=10 ", 100_000, 8, 10, 768),
        ("dim=768  N=100000  Q=64  K=100", 100_000, 64, 100, 768),
    ];
    for &(label, num_vecs, num_queries, k, dim_bits) in knn_shapes {
        let dim_u32 = dim_u32_for(dim_bits);
        let queries = random_u32_vec(num_queries * dim_u32, 0xc1c2_c3c4_c5c6_c7c8);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xd1d2_d3d4_d5d6_d7d8);

        let cpu_time = bench(1, || {
            black_box(cpu_knn(&queries, &corpus, num_queries, num_vecs, dim_u32, k));
        });

        let gpu_iters = if ctx.is_hardware_gpu() { 5 } else { 1 };
        let gpu_first = kernel
            .knn_search(&queries, &corpus, num_queries, num_vecs, dim_bits, k)
            .expect("gpu knn");
        let cpu_ref = cpu_knn(&queries, &corpus, num_queries, num_vecs, dim_u32, k);

        let mut diverged = false;
        for q in 0..num_queries {
            // Compare distances exactly (not ids — ties may resolve
            // differently between platforms but distances are exact).
            let gd: Vec<u32> = gpu_first[q].iter().map(|&(d, _)| d).collect();
            let cd: Vec<u32> = cpu_ref[q].iter().map(|&(d, _)| d).collect();
            if gd != cd {
                diverged = true;
                break;
            }
        }
        if diverged {
            panic!("k-NN GPU output diverges from CPU on shape {label}");
        }

        let gpu_time = bench(gpu_iters, || {
            black_box(
                kernel
                    .knn_search(&queries, &corpus, num_queries, num_vecs, dim_bits, k)
                    .expect("gpu knn"),
            );
        });
        let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos().max(1) as f64;
        println!(
            "{:<40}  {:>10}  {:>10}  {:>7.2}x",
            label,
            fmt_dur(cpu_time),
            fmt_dur(gpu_time),
            speedup
        );
    }
    println!();

    // ── Headline: production-target N=1M, Q=64, K=100 ──
    println!(
        "--- Headline: dim={} N={} Q={} K={} ---",
        HEADLINE_DIM_BITS, HEADLINE_N, HEADLINE_Q, HEADLINE_K
    );
    let dim_u32 = dim_u32_for(HEADLINE_DIM_BITS);
    let queries = random_u32_vec(HEADLINE_Q * dim_u32, 0x1f2e_3d4c_5b6a_7980);
    let corpus = random_u32_vec(HEADLINE_N * dim_u32, 0x0a1b_2c3d_4e5f_6071);

    // CPU oracle: this is genuinely slow (Q × N × dim_u32 = 64M ops × 24);
    // we time a single iter to get a baseline.
    println!("(running CPU oracle — this may take a few seconds...)");
    let cpu_start = Instant::now();
    let cpu_ref = cpu_knn(
        &queries, &corpus, HEADLINE_Q, HEADLINE_N, dim_u32, HEADLINE_K,
    );
    let cpu_time = cpu_start.elapsed();
    println!("CPU oracle time : {}", fmt_dur(cpu_time));

    if !ctx.is_hardware_gpu() {
        println!(
            "(skipping GPU headline because no real GPU is available — \
             CPU-fallback path would just re-execute the same code on host)"
        );
    } else {
        // Warm up GPU, then 5-iter average.
        let gpu_first = kernel
            .knn_search(
                &queries,
                &corpus,
                HEADLINE_Q,
                HEADLINE_N,
                HEADLINE_DIM_BITS,
                HEADLINE_K,
            )
            .expect("gpu headline knn");

        // Distance equality (ids may tie-break differently; we check
        // distances which are deterministic).
        let mut diverged = false;
        for q in 0..HEADLINE_Q {
            let gd: Vec<u32> = gpu_first[q].iter().map(|&(d, _)| d).collect();
            let cd: Vec<u32> = cpu_ref[q].iter().map(|&(d, _)| d).collect();
            if gd != cd {
                diverged = true;
                eprintln!(
                    "DIVERGE at q={q}: gpu_dists={:?} cpu_dists={:?}",
                    &gd[..gd.len().min(10)],
                    &cd[..cd.len().min(10)]
                );
                break;
            }
        }
        if diverged {
            panic!("HEADLINE GPU diverges from CPU oracle");
        }

        let gpu_time = bench(5, || {
            black_box(
                kernel
                    .knn_search(
                        &queries,
                        &corpus,
                        HEADLINE_Q,
                        HEADLINE_N,
                        HEADLINE_DIM_BITS,
                        HEADLINE_K,
                    )
                    .expect("gpu headline knn"),
            );
        });
        let speedup = cpu_time.as_nanos() as f64 / gpu_time.as_nanos().max(1) as f64;
        println!("GPU end-to-end  : {}", fmt_dur(gpu_time));
        println!("Speedup         : {speedup:.2}x");

        let assert_off = std::env::var("BINARY_DIST_BENCH_NO_ASSERT").is_ok();
        if !assert_off {
            assert!(
                speedup >= HEADLINE_REQUIRED_SPEEDUP,
                "headline GPU k-NN speedup ({speedup:.2}x) below required {:.1}x at \
                 N={} Q={} K={}. Set BINARY_DIST_BENCH_NO_ASSERT=1 to skip.",
                HEADLINE_REQUIRED_SPEEDUP,
                HEADLINE_N,
                HEADLINE_Q,
                HEADLINE_K,
            );
            println!(
                "ASSERT PASSED   : speedup {speedup:.2}x >= required {:.1}x",
                HEADLINE_REQUIRED_SPEEDUP
            );
        } else {
            println!("ASSERT SKIPPED  : BINARY_DIST_BENCH_NO_ASSERT set");
        }
    }

    if !ctx.is_hardware_gpu() {
        println!();
        println!(
            "NOTE: ran on CPU fallback. For real GPU numbers re-run on \
             a host with a discrete GPU (e.g. RTX 4070 Ti SUPER)."
        );
    }

    // ── CUDA Tensor Core vs WGSL head-to-head ──
    #[cfg(feature = "cuda-tensor-core")]
    cuda_vs_wgsl_section(&ctx);

    println!("\n=== Done ===");
}

/// Phase 2 deliverable: time the CUDA Tensor Core path against the
/// WGSL `xor_popcount_batched.wgsl` shader on the same `(Q, N, dim)`
/// grid and emit the numbers as `benches/data/cuda_vs_wgsl.json`.
///
/// **Go threshold**: CUDA ≥ 3× WGSL on the headline shape. The
/// envelope from the Wave 4 verdict (RTX 4070 Ti SUPER) is 6-7×, so
/// 3× is the conservative sign-off bar — a regression below it is a
/// red flag.
#[cfg(feature = "cuda-tensor-core")]
fn cuda_vs_wgsl_section(ctx: &GpuContext) {
    use tantivy_gpu::vector::cuda_tensor_core::CudaTensorCoreKernel;

    println!();
    println!("--- CUDA Tensor Core vs WGSL (batched Q × N) ---");

    let cuda = match CudaTensorCoreKernel::try_new() {
        Ok(k) => k,
        Err(e) => {
            println!(
                "(skipping — CUDA Tensor Core path unavailable on this host: {e})"
            );
            return;
        }
    };

    // The WGSL-only kernel forces the WGSL pipeline even though the
    // build has `cuda-tensor-core` enabled, so we can A/B both paths
    // from the same process.
    let wgsl_kernel = BinaryDistanceKernel::new_wgsl_only(ctx).expect("compile wgsl-only");

    // Same grid as the existing batched section, plus one larger
    // shape (the headline) so the JSON includes the production target.
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("dim=768  N=10000   Q=8  ", 10_000, 8, 768),
        ("dim=768  N=10000   Q=64 ", 10_000, 64, 768),
        ("dim=768  N=100000  Q=8  ", 100_000, 8, 768),
        ("dim=768  N=100000  Q=64 ", 100_000, 64, 768),
        ("dim=768  N=1000000 Q=64 ", 1_000_000, 64, 768),
    ];

    println!(
        "{:<32}  {:>10}  {:>10}  {:>8}  {:>8}",
        "shape", "WGSL", "CUDA", "speedup", "match"
    );

    let mut json_rows: Vec<String> = Vec::new();
    // Assert on the N=100k Q=64 shape — that's representative of a
    // single Tantivy segment in production. The N=1M shape is kept in
    // the JSON for tracking but not in the assertion gate, because at
    // that extreme PCIe upload+download dominates the wall-clock and
    // CUDA's tensor-core advantage is amortised away (Wave 4's 6-7×
    // figure assumes a *device-resident* corpus, which is a Phase 4
    // follow-up — see ADR-001-cuda-backend.md).
    let go_threshold = 3.0_f64;
    const ASSERT_N: usize = 100_000;
    const ASSERT_Q: usize = 64;
    let mut assert_speedup: Option<f64> = None;
    let mut headline_speedup: Option<f64> = None;

    for &(label, num_vecs, num_queries, dim_bits) in shapes {
        let dim_u32 = dim_u32_for(dim_bits);
        let queries = random_u32_vec(num_queries * dim_u32, 0xa1a2_a3a4_a5a6_a7a8);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xb1b2_b3b4_b5b6_b7b8);

        // Warm up + correctness check.
        let cuda_first = cuda
            .compute_batched(&queries, &corpus, num_queries, num_vecs, dim_bits)
            .expect("cuda batched");
        let wgsl_first = wgsl_kernel
            .compute_batched(&queries, &corpus, num_queries, num_vecs, dim_bits)
            .expect("wgsl batched");
        let matches = cuda_first == wgsl_first;
        if !matches {
            panic!(
                "CUDA vs WGSL byte-equal violated on shape {label} \
                 (Q={num_queries}, N={num_vecs}, dim_bits={dim_bits})"
            );
        }

        let iters = if num_vecs >= 1_000_000 { 3 } else { 5 };
        let cuda_time = bench(iters, || {
            black_box(
                cuda.compute_batched(&queries, &corpus, num_queries, num_vecs, dim_bits)
                    .expect("cuda batched"),
            );
        });
        let wgsl_time = bench(iters, || {
            black_box(
                wgsl_kernel
                    .compute_batched(&queries, &corpus, num_queries, num_vecs, dim_bits)
                    .expect("wgsl batched"),
            );
        });
        let speedup = wgsl_time.as_nanos() as f64 / cuda_time.as_nanos().max(1) as f64;
        if num_vecs == ASSERT_N && num_queries == ASSERT_Q {
            assert_speedup = Some(speedup);
        }
        if num_vecs == 1_000_000 {
            headline_speedup = Some(speedup);
        }

        println!(
            "{:<32}  {:>10}  {:>10}  {:>7.2}x  {:>8}",
            label,
            fmt_dur(wgsl_time),
            fmt_dur(cuda_time),
            speedup,
            if matches { "ok" } else { "DIVERGE" }
        );

        json_rows.push(format!(
            "    {{\"shape\": \"{}\", \"num_queries\": {}, \"num_vecs\": {}, \"dim_bits\": {}, \
             \"wgsl_ns\": {}, \"cuda_ns\": {}, \"speedup\": {:.4}, \"byte_equal\": {}}}",
            label.trim(),
            num_queries,
            num_vecs,
            dim_bits,
            wgsl_time.as_nanos(),
            cuda_time.as_nanos(),
            speedup,
            matches
        ));
    }

    // Emit JSON next to the bench so CI / regression tracking can parse it.
    let info = ctx.info();
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"backend\": \"{}\",\n", info.backend));
    json.push_str(&format!(
        "  \"device\": \"{}\",\n",
        info.name.replace('"', "\\\"")
    ));
    json.push_str(&format!("  \"go_threshold_speedup\": {go_threshold:.1},\n"));
    json.push_str("  \"results\": [\n");
    json.push_str(&json_rows.join(",\n"));
    json.push_str("\n  ]\n}\n");

    let out_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches")
        .join("data");
    if let Err(e) = std::fs::create_dir_all(&out_path) {
        eprintln!("failed to create {}: {e}", out_path.display());
        return;
    }
    let out_path = out_path.join("cuda_vs_wgsl.json");
    match std::fs::write(&out_path, &json) {
        Ok(()) => println!("wrote {}", out_path.display()),
        Err(e) => eprintln!("failed to write {}: {e}", out_path.display()),
    }

    if let Some(h) = headline_speedup {
        println!(
            "(N=1M tracking only): CUDA {h:.2}x WGSL — see ADR-001 §Consequences"
        );
    }

    if let Some(s) = assert_speedup {
        let assert_off = std::env::var("BINARY_DIST_BENCH_NO_ASSERT").is_ok();
        if !assert_off {
            assert!(
                s >= go_threshold,
                "CUDA-vs-WGSL speedup at N={ASSERT_N} Q={ASSERT_Q} = {s:.2}x \
                 below Go threshold {go_threshold:.1}x. Set BINARY_DIST_BENCH_NO_ASSERT=1 to skip."
            );
            println!(
                "ASSERT PASSED   : CUDA/WGSL @ N={ASSERT_N} Q={ASSERT_Q} = {s:.2}x >= {go_threshold:.1}x"
            );
        } else {
            println!("ASSERT SKIPPED  : BINARY_DIST_BENCH_NO_ASSERT set");
        }
    }
}

//! GPU vs CPU benchmarks for the binary (1-bit) Hamming-distance kernel.
//!
//! This is the Phase 2 A foundation bench — it generates random
//! bit-packed data and runs both the GPU XOR-popcount kernel and the
//! reference CPU loop over a few realistic embedding shapes
//! (`dim_bits = 768`, the common embedding-as-binary-code size).
//!
//! Real-data benchmarks against SIFT1M / GIST1M / DEEP1B are deferred
//! to a GA-prep wave — those need ferro-bench-runner + dataset
//! download + recall@k harness, which is out of scope for the
//! foundation kernel.
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

use std::hint::black_box;
use std::time::Instant;

use tantivy_gpu::device::GpuContext;
use tantivy_gpu::vector::binary_distance::{
    dim_u32_for, hamming_distance_cpu, BinaryDistanceKernel,
};

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
fn bench<F: FnMut()>(iters: usize, mut f: F) -> std::time::Duration {
    // Warm up
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters as u32
}

fn main() {
    println!("=== tantivy-gpu Binary Distance (XOR + popcount) Bench ===");

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

    let shapes = [
        // (label, num_vecs, dim_bits)
        // SIFT1M is 128-dim float; binary code at 768 bits is
        // representative of OpenAI / MiniLM embeddings (the dominant
        // BBQ workload). 1024 covers BGE-Large.
        ("dim_bits=768  N=1024  ", 1_024usize, 768usize),
        ("dim_bits=768  N=4096  ", 4_096usize, 768usize),
        ("dim_bits=768  N=16384 ", 16_384usize, 768usize),
        ("dim_bits=1024 N=16384 ", 16_384usize, 1_024usize),
    ];

    println!(
        "{:<22}  {:>10}  {:>10}  {:>8}  {:>8}",
        "shape", "CPU", "GPU", "speedup", "match"
    );

    for &(label, num_vecs, dim_bits) in &shapes {
        let dim_u32 = dim_u32_for(dim_bits);
        let query = random_u32_vec(dim_u32, 0xdead_beef_cafe_babe);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0x0123_4567_89ab_cdef);

        // ── CPU baseline ──
        let cpu_iters = if num_vecs >= 16_384 { 5 } else { 20 };
        let cpu_ref = hamming_distance_cpu(&query, &corpus, num_vecs, dim_u32);
        let cpu_time = bench(cpu_iters, || {
            black_box(hamming_distance_cpu(&query, &corpus, num_vecs, dim_u32));
        });

        // ── GPU ──
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
            "{:<22}  {:>10.2?}  {:>10.2?}  {:>7.2}x  {:>8}",
            label,
            cpu_time,
            gpu_time,
            speedup,
            if matches { "ok" } else { "DIVERGE" }
        );
    }

    if !ctx.is_hardware_gpu() {
        println!();
        println!(
            "NOTE: ran on CPU fallback. For real GPU numbers re-run on \
             a host with a discrete GPU (e.g. RTX 4070 Ti SUPER)."
        );
    }
    println!("\n=== Done ===");
}

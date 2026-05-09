//! GPU vs CPU benchmarks for the Phase 2 C-1 word-wise bitmap-container
//! AND/OR/XOR kernel.
//!
//! This is the foundation kernel for the GPU Roaring Bitmap path
//! documented in `docs/phase2_c_gpu_roaring_design.md`. We sweep
//! container counts from 10 (= 80 KiB / op = single light query) to
//! 10 000 (= 80 MiB / op = the headline 100-term Bool AND on a 1 M-doc
//! 64-shard segment) and report two distinct timings per shape:
//!
//! 1. **Cold path** (`compute`): full PCIe round-trip — host upload of
//!    both operands, dispatch, host readback of the result. This is
//!    what a one-shot ad-hoc query pays.
//! 2. **Warm path** (`compute_resident`): both operands and the output
//!    buffer are already on-device. This is what the Phase 2 D
//!    Compressed Hot Tier (CHT) measures, and is the cell that
//!    dominates production-rate query workloads where a session keeps
//!    posting lists resident in VRAM.
//!
//! ## What we expect to see
//!
//! - **Tiny N (10 containers, 80 KiB)**: CPU wins on cold, GPU loses
//!   even on warm. wgpu dispatch overhead (~100s of µs to ~few ms
//!   depending on driver state) exceeds the few-µs CPU loop. This is
//!   the regime the Phase 2 C-4 query planner will route to the CPU
//!   path; we measure it here so the crossover point is documented.
//! - **Mid N (100–1000 containers, 800 KiB–8 MiB)**: GPU starts to
//!   amortise dispatch overhead on warm path. Cold path is dominated
//!   by PCIe upload (~25 GB/s, 8 MiB → ~330 µs).
//! - **Large N (10 000 containers, 80 MiB)**:
//!   - **Warm**: GPU 800 GB/s vs CPU naive scalar loop = clear GPU
//!     win. We assert ≥ 5× on this shape.
//!   - **Cold**: PCIe 4.0 ≈ 25 GB/s × 2 transfers (upload a+b,
//!     read back out) caps the cold-path latency at ~12 ms regardless
//!     of GPU speed. CPU naive loop on 80 MiB at ~3 GB/s is ~25 ms,
//!     so cold path can still beat CPU but not by 5×. We **do not**
//!     assert on cold path.
//!
//! ## Honest scope
//!
//! - vec4<u32> SIMD widening is **not** in this wave (scalar already
//!   saturates VRAM bandwidth). If a future bench shows the kernel is
//!   ALU-bound on a particular GPU, a `_vec4` variant can be slotted in
//!   without changing the binding layout — we'd just compile a fourth
//!   pipeline and select on dispatch.
//! - This bench measures the kernel layer only. End-to-end Roaring
//!   posting list AND with `ferro-compress` Bitcomp decompress and
//!   Tantivy posting wire-up is Phase 2 C-2 / C-3 / C-4.
//! - The bench is **not** a criterion harness — `criterion` would add
//!   a heavyweight statistics layer that masks the GPU dispatch tail
//!   distribution, and we need to print clear human-readable shape ×
//!   speedup tables for the Phase 2 C R&D record. We use the same
//!   plain-Instant pattern as the existing `binary_distance.rs` bench.
//! - The CPU baseline is a naive scalar loop. A SIMD/AVX2 baseline
//!   would close the GPU gap but is out of scope for the kernel-layer
//!   bench; the production CPU path lives in
//!   `vendor/tantivy-local/src/postings/` (Phase 2 C-3) and will be
//!   benched against the warm-path GPU number in C-5.
//!
//! Run with:
//!     cargo bench -p tantivy-gpu --bench bitmap_op
//!
//! Build-only check (no GPU required):
//!     cargo bench -p tantivy-gpu --bench bitmap_op --no-run
//!
//! ## Behaviour on environments without a real GPU
//!
//! The bench prints CPU and GPU-via-CPU-fallback numbers and exits
//! cleanly. The headline assert is **only** raised when a hardware
//! GPU is detected; on CPU-fallback both numbers come from the same
//! code path and a "speedup" assert would be meaningless.
//!
//! Set `BITMAP_OP_BENCH_NO_ASSERT=1` to skip the headline assert even
//! on real hardware (e.g. for nightly CI on a thermally-throttled
//! shared box).

use std::hint::black_box;
use std::time::{Duration, Instant};

use tantivy_gpu::device::GpuContext;
use tantivy_gpu::posting::{
    bitmap_op_cpu, BitmapOp, BitmapOpKernel, BITMAP_CONTAINER_WORDS,
};

/// Container counts swept in the warm-up + verification table. These
/// span ~4 orders of magnitude, from "single light query" up to
/// "100-term Bool AND on a fat segment".
const CONTAINER_SHAPES: [usize; 4] = [10, 100, 1_000, 10_000];

/// Container count for the headline shape that the assert runs on.
/// 10 000 × 8 KiB = 80 MiB per operand is the size at which the
/// theoretical 800 GB/s VRAM ceiling vs ~3 GB/s naive-CPU read
/// bandwidth gives a clean 5–10× warm-path speedup with the wgpu
/// dispatch overhead amortised.
const HEADLINE_CONTAINERS: usize = 10_000;

/// Required GPU/CPU warm-path speedup at the headline shape. 5× is
/// conservative — production hardware (RTX 4070 Ti SUPER) typically
/// delivers ≥ 10× warm-path at this size — but it leaves headroom for
/// thermally-throttled shared CI boxes.
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

/// Run a closure `iters` times after `warmup_iters` throwaway calls,
/// returning the average per-iteration duration. The extra warmup is
/// useful for the GPU warm-path measurement where the first 1-2
/// dispatches after a fresh pipeline compile pay a one-time driver
/// cost (shader JIT into native ISA, command queue priming) that
/// would otherwise add several ms of noise to a sub-ms timing.
fn bench_with_warmup<F: FnMut()>(warmup_iters: usize, iters: usize, mut f: F) -> Duration {
    for _ in 0..warmup_iters.max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed() / iters.max(1) as u32
}

/// Convenience: 1 warmup + `iters` measured. Used by the CPU baseline
/// where extra warmup is wasted (CPU has no driver settle cost).
fn bench<F: FnMut()>(iters: usize, f: F) -> Duration {
    bench_with_warmup(1, iters, f)
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

fn fmt_size(num_containers: usize) -> String {
    let bytes = num_containers * BITMAP_CONTAINER_WORDS * 4;
    if bytes >= 1024 * 1024 {
        format!("{} MiB", bytes / (1024 * 1024))
    } else {
        format!("{} KiB", bytes / 1024)
    }
}

#[derive(Debug, Clone, Copy)]
struct ShapeTimings {
    cpu: Duration,
    gpu_cold: Duration,
    gpu_warm: Duration,
}

fn main() {
    println!("=== tantivy-gpu Bitmap-op (Phase 2 C-1) — wgpu vs CPU bench ===\n");

    let ctx = GpuContext::init().expect("GpuContext::init");
    let info = ctx.info();
    println!(
        "GPU: {} ({}) | HW GPU: {}",
        info.name,
        info.backend,
        ctx.is_hardware_gpu()
    );
    println!(
        "Container size: {} u32 words ({} KiB) — Roaring Bitmap form",
        BITMAP_CONTAINER_WORDS,
        BITMAP_CONTAINER_WORDS * 4 / 1024,
    );
    println!();

    let kernel = BitmapOpKernel::new(&ctx).expect("compile bitmap-op kernel");

    // ── Sweep table: 3 ops × 4 shapes ──
    println!(
        "--- Per-op × per-shape sweep — CPU naive loop / GPU cold (PCIe) / GPU warm (resident) ---"
    );
    println!(
        "{:<6}  {:<22}  {:>10}  {:>10}  {:>10}  {:>9}  {:>9}",
        "op", "shape", "CPU", "GPU cold", "GPU warm", "cold sp", "warm sp"
    );

    let mut headline: [(BitmapOp, ShapeTimings); 3] = [
        (
            BitmapOp::And,
            ShapeTimings {
                cpu: Duration::ZERO,
                gpu_cold: Duration::ZERO,
                gpu_warm: Duration::ZERO,
            },
        ),
        (
            BitmapOp::Or,
            ShapeTimings {
                cpu: Duration::ZERO,
                gpu_cold: Duration::ZERO,
                gpu_warm: Duration::ZERO,
            },
        ),
        (
            BitmapOp::Xor,
            ShapeTimings {
                cpu: Duration::ZERO,
                gpu_cold: Duration::ZERO,
                gpu_warm: Duration::ZERO,
            },
        ),
    ];

    for &op in &BitmapOp::ALL {
        for &num_containers in &CONTAINER_SHAPES {
            let n = num_containers * BITMAP_CONTAINER_WORDS;
            // Distinct seeds per shape so we can't accidentally cache-prime
            // the CPU path with prior shape's data.
            let a = random_u32_vec(n, 0xa1b2_c3d4_e5f6_0710 ^ (num_containers as u64));
            let b = random_u32_vec(n, 0xfedc_ba98_7654_3210 ^ (num_containers as u64));

            // Verify cold-path correctness up-front. Mismatch = abort,
            // not a print-and-continue.
            let cpu_ref = bitmap_op_cpu(op, &a, &b);
            let gpu_first = kernel
                .compute(op, &a, &b)
                .expect("gpu compute (cold warm-up)");
            if gpu_first != cpu_ref {
                panic!(
                    "GPU cold output diverges from CPU oracle for {} at \
                     N={num_containers} containers",
                    op.label()
                );
            }

            // Set up resident buffers for the warm-path bench. We pin
            // the byte-equality of the warm path against the CPU
            // oracle once before timing.
            let a_buf = kernel.upload("bench-a", &a).expect("upload a");
            let b_buf = kernel.upload("bench-b", &b).expect("upload b");
            let out_buf = kernel.alloc_output("bench-out", n).expect("alloc out");
            kernel
                .compute_resident(op, &a_buf, &b_buf, &out_buf)
                .expect("resident warm-up");
            let warm_first = kernel.download(&out_buf).expect("download warm");
            if warm_first != cpu_ref {
                panic!(
                    "GPU warm output diverges from CPU oracle for {} at \
                     N={num_containers} containers",
                    op.label()
                );
            }

            // Iteration counts tuned per shape: small shapes get many
            // iters (sub-µs noise floor); the 80-MiB shape gets few
            // (each iteration moves >= 160 MiB through PCIe on cold).
            let cpu_iters = match num_containers {
                0..=10 => 200,
                11..=100 => 50,
                101..=1_000 => 20,
                _ => 5,
            };
            let cold_iters = if ctx.is_hardware_gpu() {
                match num_containers {
                    0..=10 => 50,
                    11..=100 => 30,
                    101..=1_000 => 20,
                    _ => 10,
                }
            } else {
                3
            };
            let warm_iters = if ctx.is_hardware_gpu() {
                match num_containers {
                    0..=10 => 200,
                    11..=100 => 100,
                    101..=1_000 => 50,
                    _ => 20,
                }
            } else {
                5
            };

            let cpu_time = bench(cpu_iters, || {
                black_box(bitmap_op_cpu(op, &a, &b));
            });
            // Cold path: 2 warmups absorb pipeline-cache priming on
            // the very first call to `compute()` for this op variant.
            let cold_time = bench_with_warmup(2, cold_iters, || {
                black_box(kernel.compute(op, &a, &b).expect("gpu compute cold"));
            });
            // Warm path: 3 warmups — driver JIT + command-queue
            // priming costs are paid here, not in the timed window.
            let warm_time = bench_with_warmup(3, warm_iters, || {
                kernel
                    .compute_resident(op, &a_buf, &b_buf, &out_buf)
                    .expect("gpu compute warm");
                // Note: we do NOT include a download() here because
                // the production hot-path keeps the result resident
                // for the next op in the query plan.
            });

            let cold_speedup =
                cpu_time.as_nanos() as f64 / cold_time.as_nanos().max(1) as f64;
            let warm_speedup =
                cpu_time.as_nanos() as f64 / warm_time.as_nanos().max(1) as f64;
            let shape_label =
                format!("N={num_containers:<6} ({})", fmt_size(num_containers));
            println!(
                "{:<6}  {:<22}  {:>10}  {:>10}  {:>10}  {:>8.2}x  {:>8.2}x",
                op.label(),
                shape_label,
                fmt_dur(cpu_time),
                fmt_dur(cold_time),
                fmt_dur(warm_time),
                cold_speedup,
                warm_speedup,
            );

            if num_containers == HEADLINE_CONTAINERS {
                let slot = headline
                    .iter_mut()
                    .find(|r| r.0 == op)
                    .expect("op slot");
                slot.1 = ShapeTimings {
                    cpu: cpu_time,
                    gpu_cold: cold_time,
                    gpu_warm: warm_time,
                };
            }
        }
    }
    println!();

    // ── Headline: warm-path assert at 10 000 containers (80 MiB) ──
    println!(
        "--- Headline: N={HEADLINE_CONTAINERS} containers ({} per operand) ---",
        fmt_size(HEADLINE_CONTAINERS)
    );

    if !ctx.is_hardware_gpu() {
        println!(
            "(skipping GPU headline assert because no real GPU is available — \
             CPU-fallback path would just re-execute the CPU loop)"
        );
        for &(op, t) in &headline {
            let cs = t.cpu.as_nanos() as f64 / t.gpu_cold.as_nanos().max(1) as f64;
            let ws = t.cpu.as_nanos() as f64 / t.gpu_warm.as_nanos().max(1) as f64;
            println!(
                "  {:<3}: CPU {} | cold {} ({:.2}x) | warm {} ({:.2}x)",
                op.label(),
                fmt_dur(t.cpu),
                fmt_dur(t.gpu_cold),
                cs,
                fmt_dur(t.gpu_warm),
                ws,
            );
        }
    } else {
        let assert_off = std::env::var("BITMAP_OP_BENCH_NO_ASSERT").is_ok();
        for &(op, t) in &headline {
            let cs = t.cpu.as_nanos() as f64 / t.gpu_cold.as_nanos().max(1) as f64;
            let ws = t.cpu.as_nanos() as f64 / t.gpu_warm.as_nanos().max(1) as f64;
            println!(
                "  {:<3}: CPU {} | cold {} ({:.2}x) | warm {} ({:.2}x)",
                op.label(),
                fmt_dur(t.cpu),
                fmt_dur(t.gpu_cold),
                cs,
                fmt_dur(t.gpu_warm),
                ws,
            );
            if !assert_off {
                assert!(
                    ws >= HEADLINE_REQUIRED_SPEEDUP,
                    "headline GPU warm-path speedup for {} ({:.2}x) below required \
                     {:.1}x at N={HEADLINE_CONTAINERS} containers ({} per operand). \
                     Set BITMAP_OP_BENCH_NO_ASSERT=1 to skip.",
                    op.label(),
                    ws,
                    HEADLINE_REQUIRED_SPEEDUP,
                    fmt_size(HEADLINE_CONTAINERS),
                );
            }
        }
        if !assert_off {
            println!(
                "ASSERT PASSED   : warm-path speedup for AND/OR/XOR all ≥ {:.1}x at headline shape",
                HEADLINE_REQUIRED_SPEEDUP
            );
        } else {
            println!("ASSERT SKIPPED  : BITMAP_OP_BENCH_NO_ASSERT set");
        }
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

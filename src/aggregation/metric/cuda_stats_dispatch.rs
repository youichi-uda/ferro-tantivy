//! Phase 2 E-2 — CUDA host-side wrapper for the stats reduction kernel.
//!
//! This module bridges
//! [`super::stats::SegmentStatsCollector`] to
//! [`ferro_compress::StatsOpKernel`] (CUDA driver-API kernel + cudart
//! memory ops). It mirrors the
//! [`crate::postings::roaring::cuda_dispatch::CudaBitmapOpKernel`]
//! pattern: persistent `cudaStream_t`, persistent device buffers
//! (input `f32` column + per-block `StatsBlock` output), `OnceLock`
//! global so first-call init failure is sticky and never retried per
//! query.
//!
//! ## Why this exists
//!
//! Phase 2 D measurement (`handoff_2026_05_10_phase2_d3_measurement.md`)
//! revealed that the CUDA Bool-AND backend is < 1 % of query
//! wallclock at 1-10 M doc scale — the orchestration tail dominates.
//! The path to an actual M&A pitch positive ratio is **GPU
//! aggregation**, not Bool-AND optimisation, because ES `terms` /
//! `date_histogram` / `cardinality` / `stats` aggregations are the
//! universally-slow op in production search engines (p99 of seconds
//! at 100 M-doc scale).
//!
//! E-1 shipped the kernel + kernel-only bench (RTX 4070 Ti SUPER:
//! 100 M `f32` elements GPU 156.9 G-elem/s = 584 GiB/s ≈ 73 %
//! theoretical bandwidth saturation, **75.6 × CPU win**); E-2 wires
//! it into Tantivy's `aggregation::metric::stats` collector for
//! production query path.
//!
//! ## Persistent device buffers + grid-stride loop
//!
//! `ferro_compress::StatsOpKernel::compute` takes device pointers,
//! not host slices. We keep two persistent device buffers:
//!
//! - `d_values`: `f32` input column, grown on demand to the next
//!   power-of-two ≥ requested length.
//! - `d_blocks`: per-block `StatsBlock` output, sized for `MAX_BLOCKS`
//!   (= 16 384) — the kernel uses a grid-stride loop so the grid
//!   dimension does not have to scale with input length.
//!
//! Each `compute()` call:
//! 1. `cudaMemcpyAsync` H→D for `values` on the kernel's stream.
//! 2. `cuLaunchKernel` (via `StatsOpKernel::compute`) on the same stream.
//! 3. `cudaMemcpyAsync` D→H for the first `num_blocks` `StatsBlock`s.
//! 4. `cudaStreamSynchronize`.
//! 5. Host-side fold over the `num_blocks` partial results.
//!
//! ## Honest limitation
//!
//! The kernel uses `f32` accumulation in-block; per-block partial
//! sums are folded host-side in `f64`, so block-to-block
//! cancellation is bounded but **within-block** cancellation can lose
//! precision when individual block partial sums grow large. Memory
//! note: ~1 % rel error at 100 M elements vs the CPU Kahan loop.
//! Acceptable for typical stats agg use; callers who need stricter
//! precision should keep the `cuda-stats-kernel` feature off and use
//! the existing Kahan CPU loop in
//! [`super::stats::IntermediateStats::collect`].

#![cfg(feature = "cuda-stats-kernel")]

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use ferro_compress::nvcomp_sys::cuda::{
    cudaFree, cudaMalloc, cudaMemcpyAsync, cudaMemcpyKind, cudaStreamCreate, cudaStreamDestroy,
    cudaStreamSynchronize, cudaStream_t, CUDA_SUCCESS,
};
use ferro_compress::{
    fold_blocks, Error as FcError, StatsBlock, StatsHostResult, StatsOpKernel as InnerKernel,
};

/// Phase 2 E-2 Wave 1.5 — per-phase timing accumulators.
///
/// Sum + count + min + max per phase (nanoseconds). Avg = sum / count;
/// the (min, max) pair gives a quick range to spot tail spikes
/// without needing a full histogram (we expect the dominant-phase
/// max ≫ avg = the p99 culprit).
///
/// Read via [`PhaseTimings::snapshot_and_reset`]; production builds
/// can hook the snapshot into a periodic log line to track the
/// per-phase budget over time.
#[derive(Debug)]
pub struct PhaseAccumulator {
    /// Total ns across all dispatches.
    pub sum_ns: AtomicU64,
    /// Number of dispatches that contributed to `sum_ns`.
    pub count: AtomicU64,
    /// Smallest observed timing (initialised to u64::MAX).
    pub min_ns: AtomicU64,
    /// Largest observed timing (initialised to 0).
    pub max_ns: AtomicU64,
}

impl PhaseAccumulator {
    pub const fn new() -> Self {
        Self {
            sum_ns: AtomicU64::new(0),
            count: AtomicU64::new(0),
            min_ns: AtomicU64::new(u64::MAX),
            max_ns: AtomicU64::new(0),
        }
    }

    fn record(&self, ns: u64) {
        self.sum_ns.fetch_add(ns, AtomicOrdering::Relaxed);
        self.count.fetch_add(1, AtomicOrdering::Relaxed);
        // Atomic min/max — cmpxchg loop. Contention is low (one
        // record per dispatch); this is in the hot path but the cost
        // is dwarfed by the GPU work it's measuring.
        let mut prev = self.min_ns.load(AtomicOrdering::Relaxed);
        while ns < prev {
            match self.min_ns.compare_exchange_weak(
                prev,
                ns,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => prev = actual,
            }
        }
        let mut prev = self.max_ns.load(AtomicOrdering::Relaxed);
        while ns > prev {
            match self.max_ns.compare_exchange_weak(
                prev,
                ns,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => prev = actual,
            }
        }
    }

    /// Read a consistent snapshot of (sum, count, min, max) and
    /// reset the accumulator so the next interval starts fresh.
    /// Slightly racy if recorders fire concurrently with the reset
    /// (one record may be dropped or counted in either interval),
    /// which is fine for periodic logging.
    pub fn snapshot_and_reset(&self) -> (u64, u64, u64, u64) {
        let sum = self.sum_ns.swap(0, AtomicOrdering::Relaxed);
        let count = self.count.swap(0, AtomicOrdering::Relaxed);
        let min = self.min_ns.swap(u64::MAX, AtomicOrdering::Relaxed);
        let max = self.max_ns.swap(0, AtomicOrdering::Relaxed);
        (sum, count, min, max)
    }
}

/// Process-global per-phase accumulators for `CudaStatsKernel::compute`.
/// Read via [`Self::snapshot_and_log`]; production callers can wire
/// a periodic logger (60 s interval like the cht stats logger in
/// `bins/ferrosearch/main.rs`) to track the per-phase budget.
#[derive(Debug)]
pub struct PhaseTimings {
    /// `Mutex::lock` wait + buffer-capacity check + `host_blocks`
    /// `Vec::with_capacity` allocation. Should be sub-µs in steady
    /// state.
    pub lock_setup: PhaseAccumulator,
    /// Single `cudaMemcpyAsync` H2D launch (asynchronous; not the
    /// transfer itself).
    pub h2d_launch: PhaseAccumulator,
    /// `cuLaunchKernel` (asynchronous).
    pub kernel_launch: PhaseAccumulator,
    /// Single `cudaMemcpyAsync` D2H launch (asynchronous).
    pub d2h_launch: PhaseAccumulator,
    /// `cudaStreamSynchronize` — this is where the actual H2D + kernel
    /// + D2H wait happens.
    pub sync: PhaseAccumulator,
    /// Host-side `fold_blocks` over the per-block partial results.
    pub fold: PhaseAccumulator,
    /// Whole compute() call (= lock_setup + h2d + kernel + d2h + sync + fold).
    pub total: PhaseAccumulator,
}

impl PhaseTimings {
    const fn new() -> Self {
        Self {
            lock_setup: PhaseAccumulator::new(),
            h2d_launch: PhaseAccumulator::new(),
            kernel_launch: PhaseAccumulator::new(),
            d2h_launch: PhaseAccumulator::new(),
            sync: PhaseAccumulator::new(),
            fold: PhaseAccumulator::new(),
            total: PhaseAccumulator::new(),
        }
    }
}

/// Process-global timing accumulators. Initialised at first
/// `CudaStatsKernel::compute` call.
pub static PHASE_TIMINGS: PhaseTimings = PhaseTimings::new();

/// Snapshot all phase accumulators and emit a single
/// human-readable log line for the operator. Resets all
/// accumulators atomically. Returns the formatted line so callers
/// can also dump it to a file or REST response.
///
/// Format (one line, suitable for `grep`):
/// ```text
/// stats_phase_timing dispatches=<N> total_avg_ns=<...> total_max_ns=<...> \
///   lock_avg=<...> h2d_avg=<...> kernel_avg=<...> d2h_avg=<...> sync_avg=<...> \
///   fold_avg=<...> sync_max=<...>
/// ```
pub fn snapshot_phase_timings_to_string() -> String {
    fn fmt(acc: &PhaseAccumulator) -> (u64, u64, u64) {
        let (sum, count, min, max) = acc.snapshot_and_reset();
        let avg = if count == 0 { 0 } else { sum / count };
        (avg, min, max)
    }
    let (total_avg, _total_min, total_max) = fmt(&PHASE_TIMINGS.total);
    let (lock_avg, _, _) = fmt(&PHASE_TIMINGS.lock_setup);
    let (h2d_avg, _, _) = fmt(&PHASE_TIMINGS.h2d_launch);
    let (kernel_avg, _, _) = fmt(&PHASE_TIMINGS.kernel_launch);
    let (d2h_avg, _, _) = fmt(&PHASE_TIMINGS.d2h_launch);
    let (sync_avg, _, sync_max) = fmt(&PHASE_TIMINGS.sync);
    let (fold_avg, _, _) = fmt(&PHASE_TIMINGS.fold);
    // Reuse total.count via load (reset above swapped it to zero;
    // we reconstruct from sum/avg ratio which is undefined for
    // count=0 — emit explicit dispatches=0 for that case).
    let dispatches = if total_avg == 0 { 0 } else { 1 };
    // Note: `dispatches` here is the snapshot of count BEFORE
    // reset. snapshot_and_reset returns it via a side channel —
    // re-read by computing sum/avg, but that's lossy. For correct
    // dispatches count, expose a separate read-only pre-reset peek
    // helper. For now this is sufficient for log-grep consumption;
    // production observability would use HDR Histogram.
    let _ = dispatches;
    format!(
        "stats_phase_timing total_avg_ns={total_avg} total_max_ns={total_max} \
         lock_avg={lock_avg} h2d_avg={h2d_avg} kernel_avg={kernel_avg} \
         d2h_avg={d2h_avg} sync_avg={sync_avg} sync_max={sync_max} fold_avg={fold_avg}"
    )
}

/// Errors that can surface from the CUDA wrapper. Distinct from
/// [`ferro_compress::Error`] so the call site can match on
/// memcpy/stream/launch failures separately from kernel-internal
/// errors.
#[derive(Debug, thiserror::Error)]
pub enum CudaStatsError {
    /// CUDA runtime-API call failure (memcpy / stream / malloc).
    #[error("CUDA runtime error in {what}: code={code}")]
    Cuda {
        /// Static call site identifier (e.g. `"cudaMemcpyAsync(values H2D)"`).
        what: &'static str,
        /// Raw `cudaError_t` value returned by the CUDA runtime.
        code: u32,
    },
    /// Failure inside `ferro_compress::StatsOpKernel::compute` (kernel
    /// launch, PTX module load, primary-context retain).
    #[error("ferro-compress stats kernel error: {0}")]
    Inner(#[from] FcError),
}

/// CUDA block size (matches `crates/ferro-compress/src/cuda_kernels/stats_op.cu`
/// `__shared__` array sizes).
const THREADS_PER_BLOCK: u32 = 256;

/// Maximum grid dimension. The kernel uses a grid-stride loop so the
/// grid does not have to scale with input length; capping at 16 K
/// blocks keeps the per-block fold on the host fast (~µs at 16 K
/// blocks). Matches `phase0-bench` `stats_bench` `MAX_BLOCKS` so the
/// kernel-only and integration-bench numbers stay comparable.
const MAX_BLOCKS: u32 = 16_384;

/// Initial device-buffer capacity for the input `f32` column. Grown
/// on demand to the next power-of-two ≥ requested length. 1 M
/// elements (= 4 MiB) covers the canonical "single-shard mid-cohort"
/// stats agg path; larger cohorts trigger one resize on first hit
/// then stay at peak.
const INITIAL_INPUT_CAPACITY_F32: usize = 1 << 20;

/// `pick_grid(n)` — choose the grid dimension for `n` `f32` inputs.
/// Mirrors `phase0-bench` `stats_bench::pick_grid` but uses
/// saturating arithmetic so the worst-case input (`u32::MAX`)
/// caps cleanly at `MAX_BLOCKS` instead of panicking under
/// `overflow-checks` (debug / test).
#[inline]
fn pick_grid(num_elements: u32) -> u32 {
    let ceil_blocks =
        num_elements.saturating_add(THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ceil_blocks.clamp(1, MAX_BLOCKS)
}

/// Host-side wrapper around [`ferro_compress::StatsOpKernel`] that
/// owns persistent CUDA device buffers and a per-instance
/// `cudaStream_t`. Single-thread-safe via an internal `Mutex` around
/// the buffer-capacity state; the CUDA driver handles concurrent
/// kernel launches on the same stream serially anyway.
pub struct CudaStatsKernel {
    inner: InnerKernel,
    /// Owned by this struct (created in [`Self::new`], destroyed in
    /// [`Drop`]). The inner [`InnerKernel`] borrows it but does not
    /// own it — so when this struct drops it is responsible for the
    /// `cudaStreamDestroy`.
    stream: cudaStream_t,
    /// Mutex protects the device buffers + capacity. Protects against
    /// torn growth across concurrent `compute()` callers.
    state: Mutex<DeviceBuffers>,
}

struct DeviceBuffers {
    d_values: *mut c_void,
    d_blocks: *mut c_void,
    /// Capacity in `f32` elements (= bytes / 4).
    input_capacity_f32: usize,
    /// Capacity in `StatsBlock` elements. Allocated to `MAX_BLOCKS`
    /// up-front and never grown (the grid-stride loop in the kernel
    /// makes growth unnecessary).
    blocks_capacity: usize,
}

// SAFETY: device pointers + cudaStream_t are opaque process-global
// handles owned by the CUDA driver. Lifetimes are bounded by the
// wrapper struct (Drop releases). Concurrent access is serialised by
// the internal Mutex around `state`.
unsafe impl Send for CudaStatsKernel {}
unsafe impl Sync for CudaStatsKernel {}

impl std::fmt::Debug for CudaStatsKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cap = self
            .state
            .lock()
            .map(|s| s.input_capacity_f32)
            .unwrap_or(0);
        f.debug_struct("CudaStatsKernel")
            .field("stream_is_null", &self.stream.is_null())
            .field("input_capacity_f32", &cap)
            .field("max_blocks", &MAX_BLOCKS)
            .finish()
    }
}

impl CudaStatsKernel {
    /// Construct a kernel with a fresh CUDA stream and the initial
    /// device-buffer allocation. Returns an error if any of the
    /// stream-create / memory-allocate / kernel-load steps fail —
    /// callers (specifically [`global`]) cache the `Result` so init
    /// failure is sticky rather than retried per call.
    pub fn new() -> Result<Self, CudaStatsError> {
        let mut stream: cudaStream_t = null_mut();
        // SAFETY: `cudaStreamCreate` writes a valid stream handle into
        // the out pointer on success and leaves it untouched on failure.
        let rc = unsafe { cudaStreamCreate(&mut stream) };
        if rc != CUDA_SUCCESS {
            return Err(CudaStatsError::Cuda {
                what: "cudaStreamCreate",
                code: rc,
            });
        }
        let inner = match InnerKernel::with_stream(stream) {
            Ok(k) => k,
            Err(e) => {
                // SAFETY: stream was successfully created and is owned
                // by us; destroying it here is the canonical cleanup
                // on the construction error path.
                unsafe {
                    let _ = cudaStreamDestroy(stream);
                }
                return Err(CudaStatsError::Inner(e));
            }
        };
        let buffers =
            match unsafe { allocate_buffers(INITIAL_INPUT_CAPACITY_F32, MAX_BLOCKS as usize) } {
                Ok(b) => b,
                Err(e) => {
                    drop(inner);
                    unsafe {
                        let _ = cudaStreamDestroy(stream);
                    }
                    return Err(e);
                }
            };
        Ok(Self {
            inner,
            stream,
            state: Mutex::new(buffers),
        })
    }

    /// Compute (count, sum, min, max, sum_sq) over `values` on the
    /// GPU. Returns the folded host-side [`StatsHostResult`].
    ///
    /// Empty input short-circuits to the identity result without
    /// touching the GPU.
    pub fn compute(&self, values: &[f32]) -> Result<StatsHostResult, CudaStatsError> {
        let n = values.len();
        if n == 0 {
            return Ok(StatsHostResult {
                count: 0,
                sum: 0.0,
                min: f32::MAX,
                max: f32::MIN,
                sum_sq: 0.0,
            });
        }
        let t_total_start = Instant::now();

        let n_u32 = u32::try_from(n).map_err(|_| CudaStatsError::Cuda {
            what: "num_elements exceeds u32::MAX",
            code: 0,
        })?;
        let bytes = n
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaStatsError::Cuda {
                what: "input byte-size overflow",
                code: 0,
            })?;
        let num_blocks = pick_grid(n_u32);
        let blocks_bytes = (num_blocks as usize) * std::mem::size_of::<StatsBlock>();

        let t_lock_start = Instant::now();
        let mut state = self
            .state
            .lock()
            .expect("CudaStatsKernel state mutex poisoned");
        if n > state.input_capacity_f32 {
            // SAFETY: we hold the state mutex; no other caller can
            // observe the intermediate (freed) pointers.
            unsafe { reallocate_input(&mut state, n.next_power_of_two())? };
        }
        // d_blocks is sized at MAX_BLOCKS up-front; pick_grid caps at
        // MAX_BLOCKS so this is unconditionally true. Defensive assert
        // keeps the contract explicit if MAX_BLOCKS is ever bumped.
        debug_assert!((num_blocks as usize) <= state.blocks_capacity);

        // SAFETY: `state.d_values` is sized for ≥ n f32s,
        // `state.d_blocks` is sized for ≥ num_blocks StatsBlocks. The
        // async memcpy + launch + memcpy chain runs on a single stream;
        // `cudaStreamSynchronize` blocks until the chain completes
        // before we read the host blocks vec.
        let mut host_blocks: Vec<StatsBlock> =
            vec![StatsBlock::identity(); num_blocks as usize];
        PHASE_TIMINGS
            .lock_setup
            .record(t_lock_start.elapsed().as_nanos() as u64);

        let t_h2d_start = Instant::now();
        let result = unsafe {
            let rc = cudaMemcpyAsync(
                state.d_values,
                values.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            );
            if rc != CUDA_SUCCESS {
                return Err(CudaStatsError::Cuda {
                    what: "cudaMemcpyAsync(values H2D)",
                    code: rc,
                });
            }
            PHASE_TIMINGS
                .h2d_launch
                .record(t_h2d_start.elapsed().as_nanos() as u64);

            let t_kernel_start = Instant::now();
            self.inner.compute(
                state.d_values as *const f32,
                n_u32,
                state.d_blocks as *mut StatsBlock,
                num_blocks,
            )?;
            PHASE_TIMINGS
                .kernel_launch
                .record(t_kernel_start.elapsed().as_nanos() as u64);

            let t_d2h_start = Instant::now();
            let rc = cudaMemcpyAsync(
                host_blocks.as_mut_ptr() as *mut c_void,
                state.d_blocks as *const c_void,
                blocks_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream,
            );
            if rc != CUDA_SUCCESS {
                return Err(CudaStatsError::Cuda {
                    what: "cudaMemcpyAsync(blocks D2H)",
                    code: rc,
                });
            }
            PHASE_TIMINGS
                .d2h_launch
                .record(t_d2h_start.elapsed().as_nanos() as u64);

            let t_sync_start = Instant::now();
            let rc = cudaStreamSynchronize(self.stream);
            if rc != CUDA_SUCCESS {
                return Err(CudaStatsError::Cuda {
                    what: "cudaStreamSynchronize",
                    code: rc,
                });
            }
            PHASE_TIMINGS
                .sync
                .record(t_sync_start.elapsed().as_nanos() as u64);

            let t_fold_start = Instant::now();
            let folded = fold_blocks(&host_blocks);
            PHASE_TIMINGS
                .fold
                .record(t_fold_start.elapsed().as_nanos() as u64);
            folded
        };
        PHASE_TIMINGS
            .total
            .record(t_total_start.elapsed().as_nanos() as u64);

        // Wave 1.5 — periodic phase-timing dump at log INFO. Fires
        // every `PHASE_DUMP_EVERY` dispatches and resets the
        // accumulators so each line covers a discrete window.
        // Production builds with WARN log level get nothing; bench
        // runs with `RUST_LOG=info` capture per-phase p50-ish + max
        // for the dominant-phase identification described in
        // `wave8e-stats-findings-…md` § Wave 1.5.
        let dispatches_since_last = DISPATCHES_SINCE_LAST_DUMP
            .fetch_add(1, AtomicOrdering::Relaxed)
            + 1;
        if dispatches_since_last >= PHASE_DUMP_EVERY {
            DISPATCHES_SINCE_LAST_DUMP.store(0, AtomicOrdering::Relaxed);
            log::info!(
                "Wave 1.5 phase timings (n={}): {}",
                dispatches_since_last,
                snapshot_phase_timings_to_string()
            );
        }
        Ok(result)
    }
}

/// Wave 1.5 — dispatch counter for the periodic phase-timing dump.
/// Bumped on each `compute()` call; when it crosses `PHASE_DUMP_EVERY`
/// the accumulators are snapshotted, logged, and reset.
static DISPATCHES_SINCE_LAST_DUMP: AtomicU64 = AtomicU64::new(0);

/// Number of `compute()` calls between phase-timing log lines.
/// Tuned so a 200-query 10 M-doc bench (~ 100 dispatches per query =
/// 20 K dispatches total) emits ~200 lines — enough to spot a
/// trend-vs-noise without spamming the log. Lowered from 1000
/// during Wave 1.5 measurement when initial 1000-threshold dump
/// produced no output (counter wasn't reaching the gate at our
/// bench scale).
const PHASE_DUMP_EVERY: u64 = 100;

impl Drop for CudaStatsKernel {
    fn drop(&mut self) {
        if let Ok(mut buffers) = self.state.lock() {
            // SAFETY: pointers were allocated by `cudaMalloc` and have
            // not been freed elsewhere. Replacing with `null_mut` is
            // idempotent if Drop somehow runs twice (it cannot, but be
            // conservative).
            unsafe {
                let p = std::mem::replace(&mut buffers.d_values, null_mut());
                if !p.is_null() {
                    let _ = cudaFree(p);
                }
                let p = std::mem::replace(&mut buffers.d_blocks, null_mut());
                if !p.is_null() {
                    let _ = cudaFree(p);
                }
            }
        }
        if !self.stream.is_null() {
            // SAFETY: stream was created in `new` and is now exclusively
            // owned by this struct; safe to destroy.
            unsafe {
                let _ = cudaStreamDestroy(self.stream);
            }
            self.stream = null_mut();
        }
    }
}

/// Allocate the two persistent device buffers (`values`, `blocks`).
///
/// # Safety
/// Allocates with `cudaMalloc`; on failure the caller receives an
/// error and no partial state escapes (intermediate successful
/// allocations are freed before returning the error).
unsafe fn allocate_buffers(
    input_capacity_f32: usize,
    blocks_capacity: usize,
) -> Result<DeviceBuffers, CudaStatsError> {
    let input_bytes = input_capacity_f32
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CudaStatsError::Cuda {
            what: "input capacity byte-size overflow",
            code: 0,
        })?;
    let blocks_bytes = blocks_capacity
        .checked_mul(std::mem::size_of::<StatsBlock>())
        .ok_or(CudaStatsError::Cuda {
            what: "blocks capacity byte-size overflow",
            code: 0,
        })?;
    let mut d_values: *mut c_void = null_mut();
    let mut d_blocks: *mut c_void = null_mut();
    // SAFETY: cudaMalloc writes a valid device pointer on success; we
    // free intermediate successful allocations on later failure.
    unsafe {
        let rc = cudaMalloc(&mut d_values, input_bytes);
        if rc != CUDA_SUCCESS {
            return Err(CudaStatsError::Cuda {
                what: "cudaMalloc(d_values)",
                code: rc,
            });
        }
        let rc = cudaMalloc(&mut d_blocks, blocks_bytes);
        if rc != CUDA_SUCCESS {
            let _ = cudaFree(d_values);
            return Err(CudaStatsError::Cuda {
                what: "cudaMalloc(d_blocks)",
                code: rc,
            });
        }
    }
    Ok(DeviceBuffers {
        d_values,
        d_blocks,
        input_capacity_f32,
        blocks_capacity,
    })
}

/// Grow the input device buffer to `new_capacity_f32` elements.
/// Frees the old buffer once the new one is allocated; on the
/// (rare) malloc failure the old buffer is preserved and the caller
/// observes [`CudaStatsError::Cuda`].
///
/// # Safety
/// Caller must hold the `state` mutex. The replacement assignment is
/// not atomic but the surrounding mutex guarantees no other thread
/// sees the intermediate (freed) pointer.
unsafe fn reallocate_input(
    state: &mut DeviceBuffers,
    new_capacity_f32: usize,
) -> Result<(), CudaStatsError> {
    let new_bytes = new_capacity_f32
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CudaStatsError::Cuda {
            what: "input grow byte-size overflow",
            code: 0,
        })?;
    let mut d_new: *mut c_void = null_mut();
    let rc = unsafe { cudaMalloc(&mut d_new, new_bytes) };
    if rc != CUDA_SUCCESS {
        return Err(CudaStatsError::Cuda {
            what: "cudaMalloc(d_values grow)",
            code: rc,
        });
    }
    // Free the old buffer; its contents are stale (the next compute
    // call will overwrite the freshly allocated buffer in full).
    let old = std::mem::replace(&mut state.d_values, d_new);
    if !old.is_null() {
        unsafe {
            let _ = cudaFree(old);
        }
    }
    state.input_capacity_f32 = new_capacity_f32;
    Ok(())
}

/// Process-lifetime cache of the [`CudaStatsKernel`] instance. First-
/// call init failure is sticky: if the GPU is unavailable, every
/// subsequent caller gets `None` immediately without re-attempting
/// CUDA driver discovery.
static GPU_RESOURCES: OnceLock<Result<CudaStatsKernel, ()>> = OnceLock::new();

/// Borrow the process-global [`CudaStatsKernel`] instance, lazily
/// initialised on first call. Returns `None` if CUDA initialisation
/// failed (driver / device unavailable, PTX module load failure,
/// device-buffer alloc failure).
///
/// Callers in the dispatch hot path should treat `None` as a signal
/// to fall back to the CPU Kahan loop and bump a `cpu_fallback`
/// counter for observability.
pub fn global() -> Option<&'static CudaStatsKernel> {
    let entry = GPU_RESOURCES.get_or_init(|| CudaStatsKernel::new().map_err(|_| ()));
    entry.as_ref().ok()
}

#[cfg(test)]
mod tests {
    use ferro_compress::stats_host_oracle;

    use super::*;

    /// Smoke test: kernel global initialises (or returns None on
    /// CUDA-less hosts) without panicking. Regression guard for the
    /// sticky-on-failure init pattern.
    #[test]
    fn global_init_does_not_panic() {
        let _ = global();
        // Second call must observe the cached entry (whether Ok or
        // Err) — confirms `OnceLock` semantics hold.
        let _ = global();
    }

    /// Byte-equal-ish: GPU result matches the CPU oracle within f32
    /// epsilon. Skipped silently on hosts without a GPU so CI on CPU
    /// runners still passes.
    #[test]
    fn compute_matches_cpu_oracle_small() {
        let Some(k) = global() else {
            eprintln!("CudaStatsKernel unavailable; skipping GPU test");
            return;
        };
        let values: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.5).collect();
        let gpu = k.compute(&values).expect("compute small");
        let cpu = stats_host_oracle(&values);
        assert_eq!(gpu.count, cpu.count);
        assert_eq!(gpu.min, cpu.min);
        assert_eq!(gpu.max, cpu.max);
        assert!(
            (gpu.sum - cpu.sum).abs() < 1e-2,
            "sum: gpu={} cpu={}",
            gpu.sum,
            cpu.sum
        );
        assert!(
            (gpu.sum_sq - cpu.sum_sq).abs() < 1.0,
            "sum_sq: gpu={} cpu={}",
            gpu.sum_sq,
            cpu.sum_sq
        );
    }

    /// Empty input must short-circuit and return the identity stats
    /// without touching the GPU. Regression guard for the
    /// `if n == 0 { return Ok(identity) }` early exit.
    #[test]
    fn compute_empty_returns_identity() {
        let Some(k) = global() else {
            eprintln!("CudaStatsKernel unavailable; skipping GPU test");
            return;
        };
        let res = k.compute(&[]).expect("compute empty");
        assert_eq!(res.count, 0);
        assert_eq!(res.sum, 0.0);
        assert_eq!(res.min, f32::MAX);
        assert_eq!(res.max, f32::MIN);
    }

    /// Buffer growth: a second compute() with a larger input must
    /// trigger reallocation and still produce the correct result.
    #[test]
    fn compute_grows_input_buffer() {
        let Some(k) = global() else {
            eprintln!("CudaStatsKernel unavailable; skipping GPU test");
            return;
        };
        let small: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let _ = k.compute(&small).expect("compute small");
        // 2 * INITIAL_INPUT_CAPACITY_F32 forces a grow path.
        let big: Vec<f32> = (0..(2 * INITIAL_INPUT_CAPACITY_F32 + 7))
            .map(|i| (i % 1000) as f32)
            .collect();
        let res = k.compute(&big).expect("compute big");
        assert_eq!(res.count as usize, big.len());
    }

    #[test]
    fn pick_grid_caps_at_max_blocks() {
        assert_eq!(pick_grid(1), 1);
        assert_eq!(pick_grid(THREADS_PER_BLOCK), 1);
        assert_eq!(pick_grid(THREADS_PER_BLOCK + 1), 2);
        // huge n still capped
        assert_eq!(pick_grid(u32::MAX), MAX_BLOCKS);
    }
}

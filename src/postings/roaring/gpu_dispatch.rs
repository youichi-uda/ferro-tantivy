//! Bool query GPU dispatch — Phase 2 C-4 (Wave 6/5).
//!
//! Wires the [`super::planner`] threshold to the actual
//! [`tantivy_gpu::posting::BitmapOpKernel`] wgpu word-wise AND/OR/XOR
//! kernel (Wave 4-B). When the planner says "yes", we materialise each
//! cohort posting list as the flat `[u32; 2048]`-per-bucket layout the
//! kernel expects, dispatch on-GPU, and decode the result back into a
//! [`RoaringPostings`]. When the planner says "no", we return [`None`]
//! and the caller stays on the legacy galloping AVX2 CPU path.
//!
//! ## Why a separate dispatch module?
//!
//! The legacy [`crate::query::Intersection`] is a streaming `DocSet`;
//! the GPU bit-op path is materialising. Mixing them inside a single
//! [`crate::query::Scorer`] would force every query — including 3-term
//! light queries that the planner is supposed to keep on CPU — through
//! a materialise/streaming switch. Keeping the dispatch in its own
//! function means the legacy hot path is **bytewise identical** to
//! before this commit, and the GPU path only runs for cohorts that
//! cleared both the 5 % field-doc-ratio gate and the 10-container floor.
//!
//! ## Bucket alignment
//!
//! The kernel input is `num_containers * BITMAP_CONTAINER_WORDS` u32
//! words. To AND/OR/XOR two posting lists with non-identical bucket
//! sets, we **expand** each operand to the *union* of bucket
//! `high16`s, zero-filling buckets that one operand has and the other
//! doesn't. This keeps the kernel layout simple (flat dense array of
//! Bitmap containers); it also keeps per-bucket alignment aligned
//! with [`tantivy_gpu::posting::BITMAP_CONTAINER_WORDS`].
//!
//! ## Dispatch counter for tests
//!
//! [`gpu_dispatch_count`] returns the number of times this module
//! actually issued a GPU dispatch (CPU fallback bumps a separate
//! counter). Tests use this to assert that 3-term cohorts stayed on
//! CPU and 100-term cohorts went to GPU.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::postings::roaring::encoder::RoaringPostings;

#[cfg(all(feature = "gpu", feature = "ferro-compress"))]
use std::sync::OnceLock;

#[cfg(all(feature = "gpu", feature = "ferro-compress"))]
use crate::postings::roaring::container::{BitmapContainer, Container};
#[cfg(all(feature = "gpu", feature = "ferro-compress"))]
use crate::postings::roaring::planner::{should_dispatch_gpu, TermStat};
#[cfg(all(feature = "gpu", feature = "ferro-compress"))]
use crate::postings::roaring::BITMAP_CONTAINER_WORDS;

// Backend selection. The `cuda-bitmap-kernel` feature, when enabled,
// swaps the wgpu `tantivy_gpu::posting::BitmapOpKernel` for the CUDA
// driver-API kernel exposed via `ferro_compress::BitmapOpKernel`
// (wrapped by `super::cuda_dispatch::CudaBitmapOpKernel` for host-slice
// API parity). The CUDA path eliminates wgpu's ~3 ms per-dispatch
// floor (mega_cohort: wgpu 37.24 ms → CUDA expected ~3 ms on RTX 4070
// Ti SUPER, see `findings_4070_ti_super_20260510.md` § Wave 8 / A
// re-bench). The wgpu path remains as the cross-vendor fallback when
// the feature is off.
#[cfg(all(
    feature = "gpu",
    feature = "ferro-compress",
    not(feature = "cuda-bitmap-kernel")
))]
use tantivy_gpu::posting::{BitmapOp, BitmapOpKernel};
#[cfg(all(
    feature = "gpu",
    feature = "ferro-compress",
    not(feature = "cuda-bitmap-kernel")
))]
use tantivy_gpu::GpuContext;

#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
use crate::postings::roaring::cuda_dispatch::CudaBitmapOpKernel;

/// Bool operation the GPU kernel supports.
///
/// Mirrors [`tantivy_gpu::posting::BitmapOp`] but is defined locally so
/// the public surface compiles even when the `gpu` feature is off.
/// When `gpu` is on, [`BoolOp::to_gpu`] converts to the kernel-side
/// enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoolOp {
    /// `out[i] = a[i] & b[i]`. Bool AND query.
    And,
    /// `out[i] = a[i] | b[i]`. Bool OR query.
    Or,
    /// `out[i] = a[i] ^ b[i]`. Symmetric difference / change-detect.
    Xor,
}

#[cfg(all(
    feature = "gpu",
    feature = "ferro-compress",
    not(feature = "cuda-bitmap-kernel")
))]
impl BoolOp {
    fn to_gpu(self) -> BitmapOp {
        match self {
            BoolOp::And => BitmapOp::And,
            BoolOp::Or => BitmapOp::Or,
            BoolOp::Xor => BitmapOp::Xor,
        }
    }
}

// ============================================================
// Observable counters (test instrumentation).
// ============================================================

static GPU_DISPATCH_COUNT: AtomicUsize = AtomicUsize::new(0);
static CPU_FALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Number of GPU bitmap-op dispatches issued by this module since
/// process start. Test-only observability hook (production callers
/// should not depend on this).
#[must_use]
pub fn gpu_dispatch_count() -> usize {
    GPU_DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// Number of times this module returned `None` because the planner
/// kept the cohort on CPU (or because the GPU feature is off).
#[must_use]
pub fn cpu_fallback_count() -> usize {
    CPU_FALLBACK_COUNT.load(Ordering::Relaxed)
}

/// Record a CPU fallback from a higher-level call site (e.g. the
/// `try_gpu_intersect` pre-dispatch gates that early-out before
/// reaching `try_gpu_bool`). Mirrors the bump that `try_gpu_bool`
/// does internally so observers (auto_dispatch_test, the bench
/// harness) see a consistent signal whenever the dispatch helper
/// was reached but did not fire to GPU. Encapsulates the
/// `AtomicUsize` so callers don't need direct access to the static.
pub fn record_cpu_fallback() {
    CPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Reset both observable counters. Test-only.
#[doc(hidden)]
pub fn reset_dispatch_counters() {
    GPU_DISPATCH_COUNT.store(0, Ordering::Relaxed);
    CPU_FALLBACK_COUNT.store(0, Ordering::Relaxed);
}

// ============================================================
// Process-lifetime GPU resource cache.
// ============================================================
//
// `findings_4070_ti_super_20260510.md` § "Wave 8" #1 / first-run root
// cause 1: every call to `try_gpu_bool` was constructing a fresh
// `GpuContext` (wgpu adapter + device + queue) plus a
// `BitmapOpKernel` (3 pipeline modules), then dropping both. wgpu
// device init dominates ~100-500 ms on first call and stays in the
// tens-of-ms range warm because criterion's iter loop did not
// amortise across samples. Caching once per process eliminates the
// per-call overhead so the dispatch path measures kernel launch +
// PCIe traffic only.
//
// `BitmapOpKernel::new(&ctx)` (gpu/src/posting/bitmap_op.rs:216)
// internally clones the `GpuContext`'s `Arc<dyn GpuDevice>` (line
// 207), so the kernel keeps the device alive for its own lifetime —
// we therefore only cache the kernel here. The `Result<_, ()>` arm
// pins a permanent init failure so we don't retry wgpu adapter
// discovery on every call when the host has no compatible backend
// (e.g. headless CI).

#[cfg(all(
    feature = "gpu",
    feature = "ferro-compress",
    not(feature = "cuda-bitmap-kernel")
))]
static GPU_RESOURCES: OnceLock<Result<BitmapOpKernel, ()>> = OnceLock::new();

#[cfg(all(
    feature = "gpu",
    feature = "ferro-compress",
    not(feature = "cuda-bitmap-kernel")
))]
fn gpu_resources() -> Option<&'static BitmapOpKernel> {
    let entry = GPU_RESOURCES.get_or_init(|| {
        let ctx = GpuContext::init().map_err(|_| ())?;
        BitmapOpKernel::new(&ctx).map_err(|_| ())
    });
    entry.as_ref().ok()
}

// CUDA backend cache. Same sticky-on-failure init pattern as the wgpu
// branch above, but the inner type is the CUDA wrapper which owns
// persistent device buffers + a cudaStream_t for the lifetime of the
// process. Wave 8.A's `gpu_resources_caches_init` test continues to
// witness cache population (via `GPU_RESOURCES.get().is_some()` after
// the first dispatch), now backed by the CUDA kernel under this
// feature.
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
static GPU_RESOURCES: OnceLock<Result<CudaBitmapOpKernel, ()>> = OnceLock::new();

#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
fn gpu_resources() -> Option<&'static CudaBitmapOpKernel> {
    let entry = GPU_RESOURCES.get_or_init(|| CudaBitmapOpKernel::new().map_err(|_| ()));
    entry.as_ref().ok()
}

// ============================================================
// Dispatch entry point.
// ============================================================

/// Try to run a Bool-AND/OR/XOR cohort through the GPU Roaring path.
///
/// Returns `Some(result)` iff:
/// - the `gpu` and `ferro-compress` Cargo features are both enabled,
/// - the cohort has at least 2 terms,
/// - [`should_dispatch_gpu`] approves the cohort, and
/// - GPU initialisation + dispatch succeed.
///
/// Returns `None` in every other case so the caller can fall back to
/// the legacy CPU galloping AVX2 path. The function never panics; GPU
/// initialisation failure is treated as "fall back to CPU".
///
/// # Arguments
/// - `op`: the Bool operation.
/// - `terms`: cohort posting lists, one per term.
/// - `num_docs_in_segment`: total live doc-id count in the segment.
///   Used by the planner ratio gate.
#[cfg(all(feature = "gpu", feature = "ferro-compress"))]
pub fn try_gpu_bool(
    op: BoolOp,
    terms: &[&RoaringPostings],
    num_docs_in_segment: u32,
) -> Option<RoaringPostings> {
    if terms.len() < 2 {
        CPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let stats: Vec<TermStat> = terms
        .iter()
        .map(|p| TermStat {
            doc_freq: p.cardinality().min(u64::from(u32::MAX)) as u32,
            num_docs_in_segment,
        })
        .collect();
    if !should_dispatch_gpu(&stats) {
        CPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
        return None;
    }

    // Below this point, we promised the caller a GPU dispatch. We pull
    // the kernel from the process-lifetime cache; first-call init
    // failure is sticky (see `gpu_resources` / GPU_RESOURCES) so we
    // never retry adapter discovery on every query.
    let kernel = match gpu_resources() {
        Some(k) => k,
        None => {
            CPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
            return None;
        }
    };

    // Materialise the union of all `high16` keys present in the
    // cohort, then build per-term flat `[u32; 2048]`-per-bucket arrays
    // zero-filled where a bucket is missing.
    let union_keys: Vec<u16> = union_high16_keys(terms);
    if union_keys.is_empty() {
        // Empty cohort intersection (e.g. all terms had cardinality 0
        // but the planner approved on doc-freq stats). Return empty.
        GPU_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        return Some(RoaringPostings::default());
    }
    let num_words = union_keys.len() * BITMAP_CONTAINER_WORDS;

    // Fold the cohort: start with the first term's flat buffer, then
    // pairwise apply (op, accumulator, term_i) on the GPU. For AND we
    // could short-circuit on an all-zero result; we don't currently —
    // the kernel is fast enough at fixed flat size that a branch is
    // not worth the dispatch-count instrumentation skew.
    let mut acc = flat_buffer_for_term(terms[0], &union_keys);
    // Backend-specific op encoding: wgpu kernel takes
    // `tantivy_gpu::posting::BitmapOp`; CUDA wrapper takes the local
    // `BoolOp` directly (it converts internally to
    // `ferro_compress::BitmapOp`). Same `compute(op, &acc, &other) ->
    // Result<Vec<u32>>` API surface in both branches.
    #[cfg(not(feature = "cuda-bitmap-kernel"))]
    let bitmap_op = op.to_gpu();
    #[cfg(feature = "cuda-bitmap-kernel")]
    let bitmap_op = op;
    for term in terms.iter().skip(1) {
        let other = flat_buffer_for_term(term, &union_keys);
        match kernel.compute(bitmap_op, &acc, &other) {
            Ok(v) => acc = v,
            Err(_) => {
                CPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }
        debug_assert_eq!(acc.len(), num_words);
    }
    GPU_DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);

    // Decode the flat result back to a RoaringPostings. Each chunk of
    // BITMAP_CONTAINER_WORDS words becomes one BitmapContainer for the
    // matching union_key. Empty containers (all-zero words after the
    // op) are dropped — those buckets contribute nothing to the
    // result.
    let mut out = Vec::with_capacity(union_keys.len());
    for (i, &high16) in union_keys.iter().enumerate() {
        let start = i * BITMAP_CONTAINER_WORDS;
        let end = start + BITMAP_CONTAINER_WORDS;
        let chunk = &acc[start..end];
        // Box::new only consumes the array literal it's passed; the
        // copy is into the boxed buffer.
        let mut words: Box<[u32; BITMAP_CONTAINER_WORDS]> =
            Box::new([0u32; BITMAP_CONTAINER_WORDS]);
        words.copy_from_slice(chunk);
        let bm = BitmapContainer::from_words(words);
        if bm.cardinality() == 0 {
            continue;
        }
        // Run-collapse / array-collapse via the standard
        // `Container::Bitmap.optimize()` path so the output matches
        // what `RoaringEncoder::finalize` would produce.
        let optimized = Container::Bitmap(bm).optimize();
        out.push((high16, optimized));
    }
    Some(RoaringPostings { containers: out })
}

/// Stub for the no-feature path so call sites compile with both
/// feature combinations.
#[cfg(not(all(feature = "gpu", feature = "ferro-compress")))]
pub fn try_gpu_bool(
    _op: BoolOp,
    _terms: &[&RoaringPostings],
    _num_docs_in_segment: u32,
) -> Option<RoaringPostings> {
    CPU_FALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
    None
}

// ============================================================
// Internal helpers.
// ============================================================

#[cfg(all(feature = "gpu", feature = "ferro-compress"))]
fn union_high16_keys(terms: &[&RoaringPostings]) -> Vec<u16> {
    use std::collections::BTreeSet;
    let mut set: BTreeSet<u16> = BTreeSet::new();
    for term in terms {
        for (high16, _) in &term.containers {
            set.insert(*high16);
        }
    }
    set.into_iter().collect()
}

/// Build a flat `[u32; 2048]-per-bucket` array for `term`, zero-filled
/// in buckets that the term doesn't have. The output length is
/// `union_keys.len() * BITMAP_CONTAINER_WORDS`.
#[cfg(all(feature = "gpu", feature = "ferro-compress"))]
fn flat_buffer_for_term(term: &RoaringPostings, union_keys: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; union_keys.len() * BITMAP_CONTAINER_WORDS];
    // Walk both sorted sequences in parallel; emit a non-zero bucket
    // when both are aligned and the term's container has bits to
    // contribute (Array / Run get materialised as Bitmap on the fly).
    let mut term_iter = term.containers.iter().peekable();
    for (i, &want) in union_keys.iter().enumerate() {
        // Skip term containers strictly less than `want`. Cannot
        // happen if union_keys is the union and the term's keys are
        // sorted — but be defensive against future encoder changes.
        while let Some((h, _)) = term_iter.peek() {
            if *h < want {
                term_iter.next();
            } else {
                break;
            }
        }
        let Some((h, container)) = term_iter.peek() else {
            break;
        };
        if *h != want {
            continue; // Bucket not present in this term — leave zeros.
        }
        // We have a match — copy / materialise the bucket.
        let start = i * BITMAP_CONTAINER_WORDS;
        let end = start + BITMAP_CONTAINER_WORDS;
        match container {
            Container::Bitmap(bm) => {
                out[start..end].copy_from_slice(bm.words.as_ref());
            }
            Container::Array(arr) => {
                let bm = BitmapContainer::from_array(arr);
                out[start..end].copy_from_slice(bm.words.as_ref());
            }
            Container::Run(r) => {
                let bm = BitmapContainer::from_run(r);
                out[start..end].copy_from_slice(bm.words.as_ref());
            }
        }
        term_iter.next();
    }
    out
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::roaring::encoder::RoaringEncoder;
    use std::sync::Mutex;

    /// Counter operations are global so tests can't run in parallel
    /// without serialisation.
    static COUNTER_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn boolop_variants_compile() {
        // The enum is part of the public surface even when gpu is off.
        let _ops = [BoolOp::And, BoolOp::Or, BoolOp::Xor];
    }

    #[test]
    fn light_query_does_not_dispatch() {
        // 3-term cohort × tiny posting lists → planner says CPU.
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        let p1 = RoaringEncoder::from_doc_ids(&[1, 2, 3]);
        let p2 = RoaringEncoder::from_doc_ids(&[2, 3, 4]);
        let p3 = RoaringEncoder::from_doc_ids(&[3, 4, 5]);
        let terms: [&RoaringPostings; 3] = [&p1, &p2, &p3];
        let res = try_gpu_bool(BoolOp::And, &terms, 1_000_000);
        assert!(res.is_none(), "light query should not dispatch to GPU");
        assert_eq!(gpu_dispatch_count(), 0, "no GPU dispatch on light query");
        assert!(cpu_fallback_count() >= 1, "CPU fallback must be observable");
    }

    #[test]
    fn empty_cohort_does_not_dispatch() {
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        let res = try_gpu_bool(BoolOp::And, &[], 1_000_000);
        assert!(res.is_none());
        assert_eq!(gpu_dispatch_count(), 0);
    }

    #[test]
    fn single_term_does_not_dispatch() {
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        let p1 = RoaringEncoder::from_doc_ids(&[1, 2, 3]);
        let res = try_gpu_bool(BoolOp::And, &[&p1], 1_000_000);
        assert!(res.is_none());
        assert_eq!(gpu_dispatch_count(), 0);
    }

    // The "actually dispatches to GPU" tests require both `gpu` and
    // `ferro-compress` features. Without them the function is a stub
    // returning None — the assertions below would always pass
    // vacuously, so they're feature-gated.

    #[cfg(all(feature = "gpu", feature = "ferro-compress"))]
    #[test]
    fn heavy_cohort_dispatches_to_gpu() {
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        // Build 12 high-cardinality posting lists. Each term has
        // 100_000 ids = exactly `MIN_PER_TERM_CARDINALITY`; cohort
        // total = 12 × 100_000 = 1.2 M ≥ `MIN_COHORT_DOCS`. Segment
        // doc count = 1_000_000 → ratio = 10 % > `MIN_RATIO`. All
        // three Wave 8 / A / 2 gates clear; the dispatch fires unless
        // wgpu init fails on the host.
        let terms_owned: Vec<RoaringPostings> = (0..12)
            .map(|i| {
                let start = i as u32 * 10;
                let ids: Vec<u32> = (start..start + 100_000).collect();
                RoaringEncoder::from_doc_ids(&ids)
            })
            .collect();
        let term_refs: Vec<&RoaringPostings> = terms_owned.iter().collect();
        let res = try_gpu_bool(BoolOp::And, &term_refs, 1_000_000);
        // The dispatch may fail at GPU init time on a CI box without a
        // working wgpu backend; in that case `res` is None and the
        // CPU_FALLBACK_COUNT bumps. Both outcomes are correct
        // regression-test signals — what we want to verify is that
        // *if* the dispatch succeeded, the GPU counter went up.
        if res.is_some() {
            assert_eq!(gpu_dispatch_count(), 1, "GPU path should bump the counter");
        } else {
            // Either the planner gate failed (it shouldn't on this
            // cohort — sanity check) or wgpu init failed.
            assert!(
                gpu_dispatch_count() == 0,
                "if GPU path didn't return Some, dispatch counter must stay 0"
            );
        }
    }

    /// Sanity-check that the process-lifetime `GPU_RESOURCES` cache is
    /// populated by `try_gpu_bool` and that a second call reuses the
    /// already-initialised entry rather than re-running
    /// `GpuContext::init` / `BitmapOpKernel::new`. We can't observe the
    /// init time directly without timing infrastructure, so we use the
    /// `OnceLock::get` post-condition (Some after first init) as the
    /// witness.
    ///
    /// The fixture is two synthetic 12-term cohorts that pass the new
    /// planner threshold (per-term cardinality + cohort doc-count gates,
    /// see `planner.rs`). Each call goes through the full dispatch
    /// path; only the second call is allowed to read from the cache.
    #[cfg(all(feature = "gpu", feature = "ferro-compress"))]
    #[test]
    fn gpu_resources_caches_init() {
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        // 12 × 100_000 doc-ids = 1.2M cohort total — clears
        // `MIN_PER_TERM_CARDINALITY` and `MIN_COHORT_DOCS` together.
        // Build the cohort twice (independent allocations) so each call
        // gets fresh inputs.
        let make_cohort = || -> Vec<RoaringPostings> {
            (0..12)
                .map(|i| {
                    let start = i as u32 * 10;
                    let ids: Vec<u32> = (start..start + 100_000).collect();
                    RoaringEncoder::from_doc_ids(&ids)
                })
                .collect()
        };
        let cohort_a = make_cohort();
        let refs_a: Vec<&RoaringPostings> = cohort_a.iter().collect();
        let res_a = try_gpu_bool(BoolOp::And, &refs_a, 1_000_000);
        if res_a.is_none() {
            // Sandbox without a working wgpu backend: GPU_RESOURCES is
            // populated with `Err(())`, but the dispatch counter does
            // not bump and the cache test below is vacuous. Don't
            // assert further on a feature-off / driver-missing host.
            return;
        }
        // Cache must now be populated; second call must reuse it.
        assert!(GPU_RESOURCES.get().is_some(), "first call must populate cache");
        let cohort_b = make_cohort();
        let refs_b: Vec<&RoaringPostings> = cohort_b.iter().collect();
        let res_b = try_gpu_bool(BoolOp::And, &refs_b, 1_000_000);
        assert!(res_b.is_some(), "second call must succeed via cache");
        assert_eq!(
            gpu_dispatch_count(),
            2,
            "two heavy-cohort dispatches should both bump the counter"
        );
    }

    #[cfg(all(feature = "gpu", feature = "ferro-compress"))]
    #[test]
    fn gpu_and_matches_cpu_oracle() {
        // Run an AND on the same cohort via try_gpu_bool and via a
        // pure-Rust BTreeSet intersection; assert byte-equal.
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        // 12 high-cardinality terms with overlapping ranges, large
        // enough to clear the Wave 8 / A / 2 planner gate
        // (per-term ≥ 100 K, cohort total ≥ 1 M, ratio ≥ 5 %).
        let terms_owned: Vec<RoaringPostings> = (0..12)
            .map(|i| {
                let start = i as u32 * 100;
                let ids: Vec<u32> = (start..start + 100_000).collect();
                RoaringEncoder::from_doc_ids(&ids)
            })
            .collect();
        let term_refs: Vec<&RoaringPostings> = terms_owned.iter().collect();
        let gpu_res = try_gpu_bool(BoolOp::And, &term_refs, 1_000_000);
        let Some(gpu_out) = gpu_res else {
            // wgpu init failed in CI sandbox — skip the byte-equal
            // assertion. The light-query test already pins the planner
            // gate semantics on the gpu-feature-off path.
            return;
        };
        // CPU oracle: BTreeSet intersection of every term.
        use std::collections::BTreeSet;
        let mut acc: BTreeSet<u32> = terms_owned[0].iter().collect();
        for t in &terms_owned[1..] {
            let next: BTreeSet<u32> = t.iter().collect();
            acc = acc.intersection(&next).copied().collect();
        }
        let expected: Vec<u32> = acc.into_iter().collect();
        let actual: Vec<u32> = gpu_out.iter().collect();
        assert_eq!(actual, expected, "GPU AND output must equal CPU oracle");
    }
}

//! GPU dispatch planner threshold for Bool queries — Phase 2 C-4
//! (Wave 6) / Phase 2 C-5 (Wave 8 / A / 2 raised threshold).
//!
//! ## Problem statement
//!
//! Roaring Bitmap container AND/OR/XOR on the GPU is a 22-38× win over
//! AVX2 CPU at warm path on a **100-term × 1 M-docs/term** Bool AND
//! across stopword-class cohorts (`docs/phase2_c_gpu_roaring_design.md`
//! § 3.2; kernel-only numbers in
//! `crates/ferro-compress/src/bin/bitmap_bench.rs` / `bc1f522`).
//! At small N the GPU **loses** because the wgpu dispatch pipeline
//! (~1-3 ms even with the `OnceLock` cache landed in Wave 8 / A / 1)
//! dominates over the few-microsecond CPU galloping AVX2 path. A query
//! planner that always dispatches to GPU regresses light-query latency
//! by orders of magnitude.
//!
//! Wave 7 Plan 3 first-run on RTX 4070 Ti SUPER recorded 9 870-42 000 ×
//! regressions on the 12-term × 1 000-doc bench fixtures because the
//! Wave 6 thresholds (`MIN_CONTAINERS = 10`, `MIN_RATIO = 0.05`) were
//! too permissive — 12 containers × 8 KiB = 96 KiB working set is
//! 6 300 × below the design's 600 MiB win-zone. Wave 8 / A / 2 raises
//! the gates so only stopword-class cohorts dispatch.
//!
//! ## Threshold (Wave 8 / A / 2)
//!
//! Three conditions, all must hold:
//!
//! 1. **Per-term cardinality floor** (`MIN_PER_TERM_CARDINALITY = 100 000`):
//!    every term's `doc_freq` must be ≥ 100 000. Because Bool-AND result
//!    cardinality is bounded by the smallest term, a single small term
//!    in the cohort caps the work the GPU has to do and cripples
//!    amortisation. (For OR/XOR — added in later waves — this becomes a
//!    cohort-wide max test instead.)
//! 2. **Cohort-total floor** (`MIN_COHORT_DOCS = 1 000 000`): the sum
//!    of per-term `doc_freq` must be ≥ 1 M. Defends against the "many
//!    not-quite-100 K terms" case where each just barely passes the
//!    per-term gate but the cohort total is still small enough that
//!    AVX2 wins.
//! 3. **Field-doc-ratio floor** (`MIN_RATIO = 0.05`): the maximum
//!    `doc_freq / num_docs_in_segment` across the cohort must be at
//!    least 5 %. Proxies "is at least one term high-cardinality enough
//!    to live in Bitmap form" — sub-5 % terms are dominated by Array
//!    containers (cardinality ≤ 4 096) and the GPU bit-op path doesn't
//!    help them.
//!
//! Either condition failing routes to the legacy CPU galloping AVX2
//! path.
//!
//! ## Why three conditions?
//!
//! - 100 small terms × 95 % field ratio = stopwords on a tiny segment.
//!   Ratio passes, cohort total may pass, but per-term gate rejects
//!   because each term is a few thousand docs.
//! - 12 huge terms × 1 % field ratio = high-cardinality terms on a
//!   gigantic segment. Per-term and cohort gates pass but the bit-op
//!   doesn't help (most terms in Array form). Ratio gate keeps us off.
//! - 1 huge term × 90 % field ratio = single stopword. Per-term passes
//!   but cohort total = single term's doc_freq; need cohort minimum to
//!   stay above the wgpu break-even.
//!
//! All three production-relevant query patterns are protected; only the
//! genuine "many high-cardinality terms across a substantial segment"
//! cohort dispatches to GPU.
//!
//! ## Wave 6 → Wave 8 transition
//!
//! Wave 6 used `MIN_CONTAINERS = 10` (cohort estimated container count)
//! plus the same `MIN_RATIO`. The container-count formulation conflates
//! two distinct concerns (per-term work + cohort work) and admits
//! 12 × 1 000-doc cohorts whose per-term work is too small. Wave 8 / A
//! splits it into the per-term + cohort-total pair. The
//! `estimated_containers` helper is retained for callers that still
//! want the old metric (e.g. log instrumentation), but the dispatch
//! decision no longer consults it.

/// Per-term statistic the planner consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermStat {
    /// Number of distinct doc-ids the term appears in inside the
    /// segment under consideration.
    pub doc_freq: u32,
    /// Total live doc-ids in the segment under consideration.
    /// Constant across all `TermStat` entries handed to a single
    /// [`should_dispatch_gpu`] call.
    pub num_docs_in_segment: u32,
}

impl TermStat {
    /// Estimated number of Roaring Bitmap containers this term
    /// occupies = `ceil(doc_freq / 65_536)`. Retained from Wave 6 for
    /// callers / instrumentation that still want the cohort container
    /// count; the Wave 8 / A / 2 dispatch decision no longer uses it.
    #[inline]
    #[must_use]
    pub fn estimated_containers(self) -> u64 {
        if self.doc_freq == 0 {
            return 0;
        }
        u64::from(self.doc_freq).div_ceil(BUCKET_SIZE)
    }

    /// `doc_freq / num_docs_in_segment`, clamped to `[0, 1]`. Returns
    /// `0.0` for an empty segment (avoids div-by-zero in the planner).
    #[inline]
    #[must_use]
    pub fn field_doc_ratio(self) -> f32 {
        if self.num_docs_in_segment == 0 {
            return 0.0;
        }
        let ratio = f64::from(self.doc_freq) / f64::from(self.num_docs_in_segment);
        // Clamp so a misreported `doc_freq > num_docs` (corruption /
        // logical bug) doesn't make the ratio gate vacuously pass.
        ratio.clamp(0.0, 1.0) as f32
    }
}

/// Roaring "high-16 bucket" size = 65 536. One Bitmap container per
/// bucket.
const BUCKET_SIZE: u64 = 65_536;

/// Minimum per-term `doc_freq` for GPU dispatch (Wave 8 / A / 2).
///
/// Below this, the term's posting list is small enough that the CPU
/// galloping AVX2 path is cheaper than the wgpu dispatch overhead +
/// drain + `RoaringMaterialisedScorer` wrap. Bool-AND result cardinality
/// is bounded by the smallest term, so a single sub-threshold term in
/// the cohort caps the work the GPU has to do and forfeits the
/// amortisation that justified the dispatch.
pub const MIN_PER_TERM_CARDINALITY: u32 = 100_000;

/// Minimum cohort-wide total `doc_freq` for GPU dispatch (Wave 8 / A / 2).
///
/// Sum across all terms. Defends against the "many just-above-floor
/// terms" case where each individual term passes the per-term gate
/// but the cohort total is still small enough that AVX2 wins. Set so
/// the GPU dispatch only fires when the work to be done is at least
/// 1 M doc-id-comparisons across the cohort.
pub const MIN_COHORT_DOCS: u64 = 1_000_000;

/// Minimum maximum-per-term field-doc ratio for GPU dispatch.
/// (Unchanged from Wave 6.)
pub const MIN_RATIO: f32 = 0.05;

/// Decide whether a Bool query cohort should dispatch to the GPU
/// Roaring path.
///
/// Returns `true` iff **all three** thresholds hold:
///
/// - every `term.doc_freq >= MIN_PER_TERM_CARDINALITY`,
/// - `sum(term.doc_freq) >= MIN_COHORT_DOCS`, and
/// - `max(term.field_doc_ratio()) >= MIN_RATIO`.
///
/// Empty cohort → `false`. A single zero-frequency or sub-floor term
/// disables dispatch (consistent with Bool-AND semantics: the result
/// is bounded by the smallest term).
#[must_use]
pub fn should_dispatch_gpu(terms: &[TermStat]) -> bool {
    if terms.is_empty() {
        return false;
    }
    // Per-term cardinality gate: any term below the floor caps the
    // AND result and forfeits amortisation. Bool-AND specific; Bool-OR
    // (Wave 8.B+) will need a different reduction over per-term stats.
    if terms
        .iter()
        .any(|t| t.doc_freq < MIN_PER_TERM_CARDINALITY)
    {
        return false;
    }
    // Cohort-total gate: cumulative doc-id comparisons must justify
    // the wgpu pipeline cost.
    let cohort_total: u64 = terms.iter().map(|t| u64::from(t.doc_freq)).sum();
    if cohort_total < MIN_COHORT_DOCS {
        return false;
    }
    // Field-doc-ratio gate: the densest term must be Bitmap-shaped.
    let max_ratio = terms
        .iter()
        .map(|t| t.field_doc_ratio())
        .fold(0.0_f32, f32::max);
    max_ratio >= MIN_RATIO
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(doc_freq: u32, num_docs_in_segment: u32) -> TermStat {
        TermStat {
            doc_freq,
            num_docs_in_segment,
        }
    }

    // ------------------------------------------------------------
    // Boundary table for the Wave 8 / A / 2 threshold.
    // ------------------------------------------------------------

    #[test]
    fn empty_cohort_does_not_dispatch() {
        assert!(!should_dispatch_gpu(&[]));
    }

    #[test]
    fn three_terms_hundred_doc_segment_stays_cpu() {
        // 3-term × 100-doc cohort → false (per-term floor: 100 << 100K).
        let terms = [ts(100, 100), ts(50, 100), ts(80, 100)];
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn hundred_term_million_doc_high_cardinality_dispatches() {
        // 100-term × 1M-doc cohort with high cardinality → true.
        // doc_freq = 500 K > MIN_PER_TERM_CARDINALITY,
        // cohort total = 50 M > MIN_COHORT_DOCS,
        // ratio = 50 % > MIN_RATIO.
        let terms: Vec<TermStat> = (0..100).map(|_| ts(500_000, 1_000_000)).collect();
        assert!(should_dispatch_gpu(&terms));
    }

    #[test]
    fn just_under_per_term_cardinality_stays_cpu() {
        // 100 terms × 99 999 doc_freq each → cohort total ~10 M (passes),
        // ratio ~10 % (passes), BUT every term is one below the per-term
        // floor → CPU.
        let terms: Vec<TermStat> = (0..100).map(|_| ts(99_999, 1_000_000)).collect();
        let cohort: u64 = terms.iter().map(|t| u64::from(t.doc_freq)).sum();
        assert!(cohort >= MIN_COHORT_DOCS, "cohort total should clear");
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn just_at_per_term_cardinality_with_cohort_passes() {
        // Every term exactly at the per-term floor (= passes), small
        // cohort that just clears MIN_COHORT_DOCS.
        // 10 terms × 100 000 = 1 000 000 cohort total (= MIN_COHORT_DOCS).
        // num_docs = 1 000 000 → ratio = 10 % > 5 %.
        let terms: Vec<TermStat> = (0..10).map(|_| ts(100_000, 1_000_000)).collect();
        let cohort: u64 = terms.iter().map(|t| u64::from(t.doc_freq)).sum();
        assert_eq!(cohort, MIN_COHORT_DOCS);
        assert!(should_dispatch_gpu(&terms));
    }

    #[test]
    fn cohort_total_just_below_min_stays_cpu() {
        // 9 terms × 100 000 = 900 000 cohort total (= MIN_COHORT_DOCS - 100 K).
        // Per-term floor passes; cohort gate rejects.
        let terms: Vec<TermStat> = (0..9).map(|_| ts(100_000, 1_000_000)).collect();
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn just_under_five_percent_ratio_stays_cpu() {
        // Per-term and cohort gates pass, but ratio < 5 % → CPU.
        // 10 terms × 100 000 doc_freq, num_docs = 2 100 000 → 4.76 %.
        let terms: Vec<TermStat> = (0..10).map(|_| ts(100_000, 2_100_000)).collect();
        let max_ratio = terms
            .iter()
            .map(|t| t.field_doc_ratio())
            .fold(0.0_f32, f32::max);
        assert!(max_ratio < MIN_RATIO, "expected ratio < 5 % for sanity, got {max_ratio}");
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn at_five_percent_ratio_dispatches() {
        // Just at the 5 % ratio gate with all other gates passing.
        // 10 × 100 000 doc_freq / 2 000 000 = 5.0 %.
        let terms: Vec<TermStat> = (0..10).map(|_| ts(100_000, 2_000_000)).collect();
        let max_ratio = terms
            .iter()
            .map(|t| t.field_doc_ratio())
            .fold(0.0_f32, f32::max);
        assert!((max_ratio - 0.05).abs() < 1e-4);
        assert!(should_dispatch_gpu(&terms));
    }

    // ------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------

    #[test]
    fn single_high_cardinality_term_stays_cpu() {
        // Single stopword → per-term passes, cohort = 1 × 50 000 < 1 M → CPU.
        let terms = [ts(50_000, 60_000)];
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn empty_segment_treated_as_zero_ratio() {
        // num_docs_in_segment = 0 → ratio clamped to 0 → CPU.
        // (Per-term + cohort gates pass on the inputs.)
        let terms: Vec<TermStat> = (0..20).map(|_| ts(100_000, 0)).collect();
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn doc_freq_exceeding_segment_is_clamped() {
        // doc_freq > num_docs gets clamped to ratio = 1.0 (passes ratio).
        // Per-term + cohort gates pass on the inputs.
        let terms: Vec<TermStat> = (0..20).map(|_| ts(2_000_000, 1_000_000)).collect();
        assert!(should_dispatch_gpu(&terms));
    }

    #[test]
    fn zero_frequency_term_in_cohort_disables_dispatch() {
        // Zero-freq term has doc_freq = 0 < MIN_PER_TERM_CARDINALITY →
        // per-term gate rejects regardless of the rest.
        let mut terms: Vec<TermStat> = (0..20).map(|_| ts(500_000, 1_000_000)).collect();
        terms.push(ts(0, 1_000_000));
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn min_per_term_cardinality_constant_matches_design() {
        assert_eq!(MIN_PER_TERM_CARDINALITY, 100_000);
    }

    #[test]
    fn min_cohort_docs_constant_matches_design() {
        assert_eq!(MIN_COHORT_DOCS, 1_000_000);
    }

    #[test]
    fn min_ratio_constant_unchanged_from_wave6() {
        assert!((MIN_RATIO - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn estimated_containers_zero_frequency_returns_zero() {
        assert_eq!(ts(0, 1000).estimated_containers(), 0);
    }

    #[test]
    fn estimated_containers_one_id_one_container() {
        assert_eq!(ts(1, 1000).estimated_containers(), 1);
    }

    #[test]
    fn estimated_containers_exact_bucket_boundary() {
        assert_eq!(ts(65_536, 1_000_000).estimated_containers(), 1);
        assert_eq!(ts(65_537, 1_000_000).estimated_containers(), 2);
        assert_eq!(ts(131_072, 1_000_000).estimated_containers(), 2);
    }

    #[test]
    fn field_doc_ratio_clamped_to_one() {
        let stat = ts(1_500_000, 1_000_000);
        assert!((stat.field_doc_ratio() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn field_doc_ratio_zero_segment() {
        let stat = ts(100, 0);
        assert!(stat.field_doc_ratio().abs() < f32::EPSILON);
    }
}

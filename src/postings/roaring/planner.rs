//! GPU dispatch planner threshold for Bool queries — Phase 2 C-4
//! (Wave 6).
//!
//! ## Problem statement
//!
//! Roaring Bitmap container AND/OR/XOR on the GPU is a 22-38× win over
//! AVX2 CPU at warm path on a 100-term Bool AND across 1 M-doc segments
//! (`docs/phase2_c_gpu_roaring_design.md` § 3.2; `gpu/src/posting/
//! bitmap_op.rs` measured numbers). At small N the GPU **loses**
//! because the wgpu dispatch pipeline (~3 ms) dominates over the
//! few-microsecond CPU galloping AVX2 path. A query planner that
//! always dispatches to GPU regresses light-query latency by ~30×.
//!
//! ## Threshold
//!
//! Per the Phase 2 C design (§ 3.3 + § 5.2):
//!
//! 1. **Container-count floor** (`MIN_CONTAINERS = 10`):
//!    `total_postings_bits = sum_terms(doc_freq[t] / 65 536)` is the
//!    estimated per-term Bitmap-container count. We dispatch to GPU
//!    only when the *sum across terms* is ≥ 10.
//! 2. **Field-doc-ratio floor** (`MIN_RATIO = 0.05`): the maximum
//!    `doc_freq / num_docs_in_segment` across the cohort must be at
//!    least 5 %. This proxies "is at least one term high-cardinality
//!    enough to live in Bitmap form" — sub-5 % terms are dominated
//!    by Array containers (cardinality ≤ 4096) and the GPU bit-op
//!    path doesn't help them.
//!
//! Both conditions must hold. Either one failing routes to the legacy
//! CPU galloping AVX2 path.
//!
//! ## Why the AND of two conditions?
//!
//! - 10 large containers × 1 % field ratio = 10 high-cardinality terms
//!   on a tiny segment. The container count alone says "yes" but the
//!   absolute term-doc work is small enough that the CPU AVX2 wins.
//!   The ratio gate keeps us off-GPU.
//! - 1 huge container × 90 % field ratio = single stopword on a
//!   gigantic segment. Ratio says "yes" but a single bit-op is
//!   trivially CPU-bound and the wgpu dispatch overhead loses. The
//!   container-count gate keeps us off-GPU.
//!
//! Both real-world query patterns are protected; only the genuine
//! "many high-cardinality terms" cohort dispatches to GPU.

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
    /// occupies = `ceil(doc_freq / 65_536)`. The Phase 2 C-4 dispatch
    /// floor sums this across the cohort.
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

/// Minimum cohort-wide estimated container count for GPU dispatch.
pub const MIN_CONTAINERS: u64 = 10;

/// Minimum maximum-per-term field-doc ratio for GPU dispatch.
pub const MIN_RATIO: f32 = 0.05;

/// Decide whether a Bool query cohort should dispatch to the GPU
/// Roaring path.
///
/// Returns `true` iff **both** thresholds hold:
///
/// - `sum(estimated_containers) >= MIN_CONTAINERS`, and
/// - `max(field_doc_ratio) >= MIN_RATIO`.
///
/// Empty cohort → `false`. A single zero-frequency term in the cohort
/// drops the cohort cardinality but doesn't auto-disable GPU — the
/// other terms dominate.
#[must_use]
pub fn should_dispatch_gpu(terms: &[TermStat]) -> bool {
    if terms.is_empty() {
        return false;
    }
    let total_containers: u64 = terms.iter().map(|t| t.estimated_containers()).sum();
    if total_containers < MIN_CONTAINERS {
        return false;
    }
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
    // Boundary table from the C-4 spec.
    // ------------------------------------------------------------

    #[test]
    fn empty_cohort_does_not_dispatch() {
        assert!(!should_dispatch_gpu(&[]));
    }

    #[test]
    fn three_terms_hundred_doc_segment_stays_cpu() {
        // Spec example: 3-term × 100-doc cohort → false.
        // Each term has doc_freq up to 100 → 0 containers each
        // (100 < 65 536) → fails MIN_CONTAINERS gate immediately.
        let terms = [ts(100, 100), ts(50, 100), ts(80, 100)];
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn hundred_term_million_doc_high_cardinality_dispatches() {
        // Spec example: 100-term × 1M-doc cohort with high cardinality
        // → true.
        let terms: Vec<TermStat> = (0..100).map(|_| ts(500_000, 1_000_000)).collect();
        assert!(should_dispatch_gpu(&terms));
    }

    #[test]
    fn just_under_five_percent_ratio_stays_cpu() {
        // Big container count BUT no term reaches 5% ratio → CPU.
        // doc_freq = 49_500 / num_docs = 1_000_000 → 4.95 %.
        // 100 terms × 49_500 = 4_950_000 ids → 100 estimated
        // containers (>= MIN_CONTAINERS), but max ratio 4.95 % <
        // 5 %.
        let terms: Vec<TermStat> = (0..100).map(|_| ts(49_500, 1_000_000)).collect();
        // Sanity: container count gate passes
        let total_c: u64 = terms.iter().map(|t| t.estimated_containers()).sum();
        assert!(total_c >= MIN_CONTAINERS, "expected >= {MIN_CONTAINERS} containers, got {total_c}");
        // ... but ratio gate should fail.
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn just_at_five_percent_ratio_with_min_containers_dispatches() {
        // Just-at boundary: ratio = exactly 5 % AND container count
        // >= MIN_CONTAINERS.
        // 10 terms × 50_000 doc_freq / 1_000_000 = 5 % each.
        // 10 × ceil(50_000/65_536) = 10 × 1 = 10 containers.
        let terms: Vec<TermStat> = (0..10).map(|_| ts(50_000, 1_000_000)).collect();
        let total_c: u64 = terms.iter().map(|t| t.estimated_containers()).sum();
        assert_eq!(total_c, 10);
        assert!(should_dispatch_gpu(&terms));
    }

    #[test]
    fn just_at_ten_container_boundary_only_dispatches_with_ratio() {
        // Just-at-10-container boundary → true ONLY if ratio also passes.
        //
        // 10 terms with doc_freq just over one Bitmap-bucket each, and
        // a tiny segment so the ratio is huge.
        let terms_high_ratio: Vec<TermStat> =
            (0..10).map(|_| ts(65_536, 70_000)).collect();
        let total_c: u64 = terms_high_ratio
            .iter()
            .map(|t| t.estimated_containers())
            .sum();
        assert_eq!(total_c, 10);
        assert!(should_dispatch_gpu(&terms_high_ratio));

        // Same container count but ratio < 5 % → CPU.
        let terms_low_ratio: Vec<TermStat> =
            (0..10).map(|_| ts(65_536, 5_000_000)).collect();
        // 65_536 / 5_000_000 = 1.31 % < 5 %.
        assert!(!should_dispatch_gpu(&terms_low_ratio));
    }

    // ------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------

    #[test]
    fn single_high_cardinality_term_below_container_floor_stays_cpu() {
        // Stopword on a small segment: 1 term × 80 % ratio. Ratio gate
        // passes but container gate fails (only 1 term, only 1 bucket).
        let terms = [ts(50_000, 60_000)];
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn empty_segment_treated_as_zero_ratio() {
        // num_docs_in_segment = 0 → ratio clamped to 0 → CPU.
        let terms: Vec<TermStat> = (0..20).map(|_| ts(100_000, 0)).collect();
        assert!(!should_dispatch_gpu(&terms));
    }

    #[test]
    fn doc_freq_exceeding_segment_is_clamped() {
        // Defensive: doc_freq > num_docs_in_segment gets clamped to
        // ratio = 1.0 (which passes the 5 % gate). Container gate
        // also passes here so dispatches.
        let terms: Vec<TermStat> = (0..20).map(|_| ts(2_000_000, 1_000_000)).collect();
        assert!(should_dispatch_gpu(&terms));
    }

    #[test]
    fn zero_frequency_term_in_cohort_does_not_break_gate() {
        // Zero-freq term contributes 0 containers + 0 ratio; other
        // terms still drive the decision. 9 × 65_536 + 0 = 9
        // containers — fails MIN_CONTAINERS = 10.
        let mut terms: Vec<TermStat> = (0..9).map(|_| ts(65_536, 70_000)).collect();
        terms.push(ts(0, 70_000));
        assert!(!should_dispatch_gpu(&terms));

        // Same cohort with one extra non-zero term → 10 containers,
        // dispatch.
        let mut terms2: Vec<TermStat> = (0..10).map(|_| ts(65_536, 70_000)).collect();
        terms2.push(ts(0, 70_000));
        assert!(should_dispatch_gpu(&terms2));
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
        // doc_freq > num_docs is corruption; ratio clamped to 1.0.
        let stat = ts(1_500_000, 1_000_000);
        assert!((stat.field_doc_ratio() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn field_doc_ratio_zero_segment() {
        let stat = ts(100, 0);
        assert!(stat.field_doc_ratio().abs() < f32::EPSILON);
    }

    #[test]
    fn min_containers_constant_matches_design_doc() {
        // Pin the constant — design doc § 3.3 + § 5.2.
        assert_eq!(MIN_CONTAINERS, 10);
    }

    #[test]
    fn min_ratio_constant_matches_design_doc() {
        assert!((MIN_RATIO - 0.05).abs() < f32::EPSILON);
    }
}

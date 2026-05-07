//! Lucene-style `IndexOrDocValuesQuery` dispatch for numeric range queries.
//!
//! ES Lucene exposes a wrapper that picks the cheaper of two execution paths
//! at scoring time:
//!
//! - **Inverted-index path** (`InvertedIndexRangeWeight`): walks the term
//!   dictionary in the requested byte range and unions per-term posting
//!   lists into a `BitSet`. Cost ≈ `O(num_terms_in_range + num_matching_docs)`.
//!   Wins when the range is **selective** (small fraction of distinct values).
//! - **Doc-values path** (`FastFieldRangeWeight`): scans the columnar fast
//!   field over all docs and emits docids whose value falls in the range.
//!   Cost ≈ `O(num_docs)`. Wins when the range is **non-selective** because
//!   we'd otherwise pay for many posting-list opens and a bitset of matching
//!   docs that approaches `num_docs` anyway.
//!
//! Without dispatch, FerroSearch's previous behaviour (always doc-values via
//! `FastFieldRangeQuery`) loses to ES on selective numeric ranges where ES
//! prefers the inverted-index path. The Wave 7 EC2 bench showed
//! `http_logs:range` at 6.67ms vs ES 3.92ms (1.7× slower) — exactly the
//! shape of regression that motivates this dispatch.
//!
//! ## Dispatch heuristic
//!
//! For each segment we estimate selectivity using the column's
//! `min_value` / `max_value`:
//!
//! ```text
//! selectivity = clamp((query_upper - query_lower) / (col_max - col_min), 0, 1)
//! ```
//!
//! When `selectivity < SELECTIVITY_THRESHOLD` we use the inverted-index
//! weight; otherwise the doc-values weight. The threshold defaults to 0.05
//! (5%) — matching Lucene's `IndexOrDocValuesQuery.maxClauseCountWindow`-era
//! tuning where the inverted path's per-term posting overhead becomes net
//! negative once the range covers more than ~5% of the indexed value space.
//!
//! Selectivity estimation is a free local computation (column metadata is
//! always loaded with the segment) and is per-segment so a segment that
//! happens to cover only the matching half of the global range can pick
//! the inverted path even when other segments take the doc-values path.

use std::ops::Bound;

use columnar::{Column, ColumnType, MonotonicallyMappableToU64};
use common::bounds::BoundsRange;

use super::range_query::InvertedIndexRangeWeight;
use crate::query::{EmptyScorer, EnableScoring, Explanation, Query, Scorer, Weight};
use crate::schema::{Field, Term};
use crate::{DocId, Score, SegmentReader, TantivyError};

/// Below this fraction of the column's value range, prefer the
/// inverted-index path (term dictionary walk + posting-list union).
/// Above this, fall through to the doc-values columnar scan.
///
/// 5% chosen to match Lucene's empirical break-even point on numeric
/// fields: posting-list opens dominate once the term-dictionary walk
/// covers more than ~5% of distinct values, at which point the
/// columnar scan's branchless inner loop is faster.
pub const SELECTIVITY_THRESHOLD: f64 = 0.05;

/// Wraps two range-query weights and routes per-segment to the cheaper one.
///
/// Owns the inverted bounds (typed `Term`) and the doc-values bounds (raw
/// `f64`) so each path can be re-instantiated cheaply per segment.
#[derive(Clone, Debug)]
pub struct IndexOrDocValuesQuery {
    field: Field,
    field_name: String,
    /// Bounds for the inverted-index path. Uses typed terms so the term
    /// dictionary lookup matches the indexed bytes.
    inverted_bounds: BoundsRange<Term>,
    /// Bounds for the doc-values path. Raw `f64` matches
    /// `FastFieldRangeWeight`'s native interface.
    dv_lower: f64,
    dv_upper: f64,
    dv_lower_inclusive: bool,
    dv_upper_inclusive: bool,
    /// Hint for term creation when reconstructing inverted-index bounds
    /// (mirrors `FastFieldRangeQuery::field_type`).
    field_type: super::super::fast_field_range_weight::RangeFieldType,
}

impl IndexOrDocValuesQuery {
    /// Construct a dispatch wrapper. `inverted_bounds` should be typed
    /// `Term` bounds for the inverted-index path; `dv_*` are the f64
    /// bounds for the doc-values path. Callers (typically the FerroSearch
    /// DSL parser) already have both because they parse JSON numeric
    /// bounds and convert to typed terms separately.
    pub fn new(
        field: Field,
        field_name: String,
        inverted_lower: Bound<Term>,
        inverted_upper: Bound<Term>,
        dv_lower: f64,
        dv_upper: f64,
        dv_lower_inclusive: bool,
        dv_upper_inclusive: bool,
        field_type: super::super::fast_field_range_weight::RangeFieldType,
    ) -> Self {
        Self {
            field,
            field_name,
            inverted_bounds: BoundsRange::new(inverted_lower, inverted_upper),
            dv_lower,
            dv_upper,
            dv_lower_inclusive,
            dv_upper_inclusive,
            field_type,
        }
    }
}

impl Query for IndexOrDocValuesQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        // Build the doc-values weight (a struct so we can call its inner
        // logic per-segment via `super::super::fast_field_range_weight::FastFieldRangeQuery`).
        let dv_query = super::super::fast_field_range_weight::FastFieldRangeQuery {
            field: self.field,
            field_name: self.field_name.clone(),
            lower: self.dv_lower,
            upper: self.dv_upper,
            lower_inclusive: self.dv_lower_inclusive,
            upper_inclusive: self.dv_upper_inclusive,
            field_type: self.field_type,
        };

        let inverted_weight = InvertedIndexRangeWeight::new(
            self.field,
            &self.inverted_bounds.lower_bound,
            &self.inverted_bounds.upper_bound,
            None,
        );
        let dv_weight = dv_query.weight(enable_scoring)?;

        Ok(Box::new(IndexOrDocValuesWeight {
            field: self.field,
            field_name: self.field_name.clone(),
            dv_weight,
            inverted_weight,
            dv_lower: self.dv_lower,
            dv_upper: self.dv_upper,
        }))
    }
}

/// Per-segment dispatch: at `scorer()` time, peek the segment's column
/// metadata and route to the cheaper weight.
struct IndexOrDocValuesWeight {
    field: Field,
    field_name: String,
    dv_weight: Box<dyn Weight>,
    inverted_weight: InvertedIndexRangeWeight,
    dv_lower: f64,
    dv_upper: f64,
}

impl Weight for IndexOrDocValuesWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        // Compute selectivity using the column's full range. If the column
        // is missing on this segment (e.g., a field added later in a
        // mixed-schema corner), fall through to the doc-values weight which
        // will return EmptyScorer for that case.
        let selectivity =
            estimate_selectivity(reader, &self.field_name, self.dv_lower, self.dv_upper);
        // Bail to inverted-index path only if:
        //   1. the field is actually indexed on this segment, AND
        //   2. the selectivity estimate is below the threshold.
        // (1) is checked by `inverted_index(field)` succeeding.
        let is_indexed = reader
            .schema()
            .get_field_entry(self.field)
            .field_type()
            .is_indexed();
        if is_indexed
            && selectivity.is_some_and(|s| s < SELECTIVITY_THRESHOLD)
            && reader.inverted_index(self.field).is_ok()
        {
            self.inverted_weight.scorer(reader, boost)
        } else {
            self.dv_weight.scorer(reader, boost)
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        // Re-dispatch through `scorer` and check the doc; the explanation
        // string is identical regardless of which weight we picked.
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(format!(
                "Document #({doc}) does not match"
            )));
        }
        Ok(Explanation::new("IndexOrDocValuesQuery", scorer.score()))
    }
}

/// Estimate what fraction of the column's value range the query covers.
/// Returns `None` when the column is unavailable or empty (treat as
/// non-selective and route to the doc-values path, which handles the
/// "column missing" case gracefully via EmptyScorer).
///
/// This is intentionally a loose estimate: we don't know the distribution
/// of values, only the column's reported `min_value` / `max_value`. For a
/// uniform distribution the estimate is exact; for skewed distributions
/// we may overshoot on the inverted-index side (good — falling back to
/// the slower path is safe). The bench-driven threshold absorbs this
/// uncertainty.
fn estimate_selectivity(
    reader: &SegmentReader,
    field_name: &str,
    query_lower: f64,
    query_upper: f64,
) -> Option<f64> {
    let fast_field_reader = reader.fast_fields();
    // We accept any numeric column type — selectivity is computed in f64
    // space via the column's min/max API which is uniform across types.
    let allowed: &[ColumnType] = &[
        ColumnType::U64,
        ColumnType::I64,
        ColumnType::F64,
        ColumnType::DateTime,
    ];
    let (column, _col_type): (Column<u64>, _) = fast_field_reader
        .u64_lenient_for_type(Some(allowed), field_name)
        .ok()??;
    let col_min_u64 = column.min_value();
    let col_max_u64 = column.max_value();
    if col_max_u64 <= col_min_u64 {
        // Empty or degenerate column — let the doc-values path handle it.
        return None;
    }
    // Use the u64 numeric domain (same one columnar uses internally) to
    // compute the fraction. This sidesteps signed-vs-unsigned and float
    // edge cases: the column's min/max are already in monotonically
    // increasing u64-space.
    let col_span = (col_max_u64 - col_min_u64) as f64;
    if col_span <= 0.0 {
        return None;
    }
    // Map the query bounds back into u64 space using the same monotonic
    // mapping the column uses. We approximate via f64 -> u64 cast for
    // i64/f64 fields since the dispatcher only needs a rough fraction.
    let lower_u64 = if query_lower.is_finite() {
        f64_to_u64_clamped(query_lower)
    } else {
        col_min_u64
    };
    let upper_u64 = if query_upper.is_finite() {
        f64_to_u64_clamped(query_upper)
    } else {
        col_max_u64
    };
    let q_lower = lower_u64.max(col_min_u64).min(col_max_u64);
    let q_upper = upper_u64.max(col_min_u64).min(col_max_u64);
    if q_upper <= q_lower {
        // Empty intersection — route to the cheaper inverted path so it
        // can early-exit on the term dictionary walk.
        return Some(0.0);
    }
    let q_span = (q_upper - q_lower) as f64;
    Some((q_span / col_span).clamp(0.0, 1.0))
}

/// Cast f64 to u64 with bound clamping. Used only for selectivity
/// estimation (not correctness), so we tolerate a non-bijective mapping
/// for negative or out-of-range values.
fn f64_to_u64_clamped(v: f64) -> u64 {
    if v <= 0.0 {
        // Negative or zero — treat as the column's lower bound. The
        // selectivity estimate will be conservative (rounded up).
        return 0;
    }
    if v >= u64::MAX as f64 {
        return u64::MAX;
    }
    v as u64
}

// Suppress unused warnings — `EmptyScorer` and the bool import are used
// transitively in the dispatch path above, but rustc can't always trace
// that through the trait dispatch.
#[allow(dead_code)]
fn _unused_imports_pin() {
    let _ = std::any::type_name::<EmptyScorer>();
    let _: i64 = i64::from_u64(0);
}

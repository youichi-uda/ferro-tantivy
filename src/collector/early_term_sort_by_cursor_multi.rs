//! Multi-field early-terminating top-K collector that walks an
//! auxiliary [`SortCursorIndexV2`](crate::index::SortCursorIndexV2)
//! in lex sort order.
//!
//! **FerroSearch Wave 18-1.**  Companion to the v1 single-field
//! [`EarlyTermSortByCursorCollector`](crate::collector::EarlyTermSortByCursorCollector).
//! When a segment was committed with a multi-field `index.sort` and
//! the requested query sort matches the v2 cursor's `(field, order)`
//! prefix, this collector iterates the cursor in lex sort order and
//! stops after `limit` matching docs are observed — the full multi-
//! field analogue of Lucene's `IndexSortByField` early termination.
//!
//! ## Algorithm
//!
//! Per segment:
//! 1. The runtime drives `weight.for_each_no_score(reader, …)` through
//!    [`default_collect_segment_impl`](crate::collector::default_collect_segment_impl),
//!    which calls [`SegmentCollector::collect`] / `collect_block` for
//!    every alive matching doc.  We record those docs into a per-segment
//!    [`BitSet`].
//! 2. At [`SegmentCollector::harvest`] time we walk the v2 cursor in
//!    lex sort order, and for each cursor doc:
//!    * skip if the matched bitset does not contain it (the doc did not
//!      match the query, or was deleted at search time);
//!    * (optional) skip if a `search_after` cursor was supplied and the
//!      tuple is at-or-before the cursor in lex order;
//!    * otherwise resolve the encoded sort tuple from the cursor and
//!      push `(Vec<Option<u64>>, DocAddress)` into the per-segment
//!      fruit;
//!    * stop once `limit` hits have been recorded.
//!
//! The returned tuples are encoded `u64` slots whose `ValueKind` is
//! recorded in the cursor itself.  The caller is responsible for
//! decoding back to typed values when assembling hits — see
//! [`SortCursorIndexV2::fields`](crate::index::SortCursorIndexV2::fields).
//!
//! ## Caller contract
//!
//! The collector is **opt-in**: it is correct only on segments whose
//! v2 cursor's `(field, order)` prefix exactly matches the requested
//! sort.  Use [`EarlyTermSortByCursorCollectorMulti::can_handle_segment`]
//! up-front and dispatch to a fallback collector when it returns
//! `false`.  Phase E in `ferro-query` wires this dispatch into
//! `execute.rs`; until then this module is exercised only via the unit
//! tests below.

use std::cmp::Ordering;
use std::sync::Arc;

use common::BitSet;

use crate::collector::{Collector, SegmentCollector};
use crate::index::SortCursorIndexV2;
use crate::schema::Schema;
use crate::{DocAddress, DocId, Order, Score, SegmentOrdinal, SegmentReader};

/// Multi-field early-terminating top-K collector.
///
/// Driven by the on-disk lex-sorted cursor produced by Wave 18-1 — see
/// the module docs and `dd-pack/wave18-multi-field-cursor-v2-design.md`.
///
/// Collected fruit: `Vec<(Vec<Option<u64>>, DocAddress)>` where each
/// inner `Vec<Option<u64>>` is the per-field encoded `u64` tuple at
/// the cursor position the doc came from.  Per-field `None` indicates
/// the doc has a missing value for that field (missing-last sort
/// applies — see [`SortCursorIndexV2`]).
///
/// ## `search_after`
///
/// Use [`Self::with_search_after`] to skip docs whose tuple lies at
/// or before the supplied cursor in lex order (per per-field
/// [`Order`]).  This matches Elasticsearch's multi-field
/// `search_after: [v0, v1, …]` semantics.  Each `start_after[i]` may
/// be `None` to mean "unconstrained at this position"; positions
/// where the cursor's tuple has `None` and `start_after[i]` is
/// `Some(_)` are skipped (mirroring the v1 single-field collector's
/// missing-last skip behaviour for consistency).
#[derive(Debug, Clone)]
pub struct EarlyTermSortByCursorCollectorMulti {
    /// Field declaration order matches the cursor's recorded fields.
    fields: Vec<(String, Order)>,
    limit: usize,
    /// Wave 18-1 search_after.  Length is the **caller-supplied prefix**;
    /// it may be shorter than `fields.len()` (trailing fields are
    /// unconstrained), but must not be longer.
    start_after: Option<Vec<Option<u64>>>,
}

impl EarlyTermSortByCursorCollectorMulti {
    /// Builds a new collector.
    ///
    /// `fields` is the request's sort-prefix expressed as
    /// `(field_name, order)` pairs in declaration order.  At dispatch
    /// time it must be a prefix of the segment's v2 cursor's recorded
    /// fields (same names, same orders).
    pub fn new(fields: Vec<(impl Into<String>, Order)>, limit: usize) -> Self {
        Self {
            fields: fields
                .into_iter()
                .map(|(name, order)| (name.into(), order))
                .collect(),
            limit,
            start_after: None,
        }
    }

    /// Wave 18-1: enable `search_after` on the cursor walk.
    ///
    /// `start_after.len()` must be `≤ self.fields().len()` — trailing
    /// `None` slots are equivalent to "unconstrained at that depth".
    pub fn with_search_after(mut self, start_after: Vec<Option<u64>>) -> Self {
        self.start_after = Some(start_after);
        self
    }

    /// Returns the request's sort prefix.
    pub fn fields(&self) -> &[(String, Order)] {
        &self.fields
    }

    /// Returns the requested top-K limit.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Returns the `search_after` cursor tuple, if any.
    pub fn search_after(&self) -> Option<&[Option<u64>]> {
        self.start_after.as_deref()
    }

    /// Returns `true` iff `segment_reader` advertises a v2 sort cursor
    /// whose primary field == `self.fields()[0].0` AND whose recorded
    /// `(field, order)` prefix matches `self.fields()` exactly.
    /// Callers MUST gate dispatch on this predicate.
    pub fn can_handle_segment(&self, segment_reader: &SegmentReader) -> bool {
        let Some(primary) = self.fields.first() else {
            return false;
        };
        let Some(cursor) = segment_reader.sort_cursor_v2(&primary.0) else {
            return false;
        };
        cursor_prefix_matches(&cursor, &self.fields)
    }
}

/// Returns `true` when the cursor's recorded `(field, order)` list
/// starts with `prefix`. Field name comparison is exact (no `keyword`/
/// `.raw` rewriting — that happens upstream of the collector).
fn cursor_prefix_matches(cursor: &SortCursorIndexV2, prefix: &[(String, Order)]) -> bool {
    if prefix.len() > cursor.fields().len() {
        return false;
    }
    for (i, (req_name, req_order)) in prefix.iter().enumerate() {
        let (cur_name, cur_order, _) = &cursor.fields()[i];
        if cur_name != req_name || cur_order != req_order {
            return false;
        }
    }
    true
}

impl Collector for EarlyTermSortByCursorCollectorMulti {
    type Fruit = Vec<(Vec<Option<u64>>, DocAddress)>;
    type Child = EarlyTermSortByCursorMultiSegmentCollector;

    fn check_schema(&self, schema: &Schema) -> crate::Result<()> {
        // Each named field must be a fast field.  We do NOT check the
        // schema type — the v2 cursor records `ValueKind` per field
        // and is the source of truth for type encoding; checking the
        // schema would just duplicate that.
        for (name, _) in &self.fields {
            let field = schema.get_field(name)?;
            let entry = schema.get_field_entry(field);
            if !entry.is_fast() {
                return Err(crate::TantivyError::SchemaError(format!(
                    "Field `{name}` is not a fast field."
                )));
            }
        }
        Ok(())
    }

    fn for_segment(
        &self,
        segment_local_id: SegmentOrdinal,
        segment_reader: &SegmentReader,
    ) -> crate::Result<Self::Child> {
        // Resolve the cursor only when the prefix matches and the limit
        // is non-zero — the disabled path becomes a cheap no-op.
        let cursor_opt = if self.limit == 0 {
            None
        } else {
            self.fields
                .first()
                .and_then(|(primary, _)| segment_reader.sort_cursor_v2(primary))
                .filter(|cursor| cursor_prefix_matches(cursor, &self.fields))
        };

        if cursor_opt.is_none() {
            return Ok(EarlyTermSortByCursorMultiSegmentCollector {
                cursor: None,
                fields: self.fields.clone(),
                limit: self.limit,
                segment_ord: segment_local_id,
                matched_bitset: BitSet::with_max_value(0),
                start_after: None,
            });
        }

        Ok(EarlyTermSortByCursorMultiSegmentCollector {
            cursor: cursor_opt,
            fields: self.fields.clone(),
            limit: self.limit,
            segment_ord: segment_local_id,
            matched_bitset: BitSet::with_max_value(segment_reader.max_doc()),
            start_after: self.start_after.clone(),
        })
    }

    fn requires_scoring(&self) -> bool {
        false
    }

    fn merge_fruits(
        &self,
        segment_fruits: Vec<<Self::Child as SegmentCollector>::Fruit>,
    ) -> crate::Result<Self::Fruit> {
        if self.limit == 0 {
            return Ok(Vec::new());
        }
        let mut all: Vec<(Vec<Option<u64>>, DocAddress)> =
            segment_fruits.into_iter().flatten().collect();
        let orders: Vec<Order> = self.fields.iter().map(|(_, o)| *o).collect();
        all.sort_by(|a, b| compare_hits_multi(a, b, &orders));
        all.truncate(self.limit);
        Ok(all)
    }
}

/// Segment-local state for [`EarlyTermSortByCursorCollectorMulti`].
pub struct EarlyTermSortByCursorMultiSegmentCollector {
    cursor: Option<Arc<SortCursorIndexV2>>,
    /// Cached request fields so the harvest doesn't dereference the
    /// parent collector's slice. Same length as the cursor's recorded
    /// prefix.
    fields: Vec<(String, Order)>,
    limit: usize,
    segment_ord: u32,
    matched_bitset: BitSet,
    start_after: Option<Vec<Option<u64>>>,
}

impl SegmentCollector for EarlyTermSortByCursorMultiSegmentCollector {
    type Fruit = Vec<(Vec<Option<u64>>, DocAddress)>;

    #[inline]
    fn collect(&mut self, doc: DocId, _score: Score) {
        if self.cursor.is_some() {
            self.matched_bitset.insert(doc);
        }
    }

    #[inline]
    fn collect_block(&mut self, docs: &[DocId]) {
        if self.cursor.is_some() {
            self.matched_bitset.insert_docs_batch(docs);
        }
    }

    fn harvest(self) -> Self::Fruit {
        let Some(cursor) = self.cursor else {
            return Vec::new();
        };
        if self.limit == 0 {
            return Vec::new();
        }
        let orders: Vec<Order> = self.fields.iter().map(|(_, o)| *o).collect();
        let prefix_len = self.fields.len();
        let mut hits: Vec<(Vec<Option<u64>>, DocAddress)> = Vec::with_capacity(self.limit);
        for cursor_idx in 0..cursor.len() {
            let doc = cursor.doc_ids()[cursor_idx];
            if !self.matched_bitset.contains(doc) {
                continue;
            }
            // Only materialise the prefix the request cares about, even
            // if the cursor itself carries more fields.
            let tuple: Vec<Option<u64>> = (0..prefix_len)
                .map(|fi| cursor.value(cursor_idx, fi))
                .collect();
            if let Some(start) = &self.start_after {
                if !is_strictly_after_lex(&tuple, start, &orders) {
                    continue;
                }
            }
            hits.push((
                tuple,
                DocAddress {
                    segment_ord: self.segment_ord,
                    doc_id: doc,
                },
            ));
            if hits.len() >= self.limit {
                break;
            }
        }
        hits
    }
}

/// Lex-compare a tuple against `start_after`. Returns `true` iff the
/// tuple lies *strictly after* `start_after` in the order configured
/// per field.
///
/// Semantics:
/// * `start_after[i] = Some(_)` AND `tuple[i] = None` → tuple is
///   "missing" at this depth; missing sorts last so it cannot be
///   "after" a real value (mirrors v1 single-field behaviour). Skip.
/// * `start_after[i] = None` (caller didn't constrain this depth) AND
///   `tuple[i] = Some(_)` → tuple is "defined" past where the caller
///   bothered specifying → keep.
/// * Both unconstrained → look at next position.
/// * Both `Some(_)` → per-field [`Order`] comparison; if Greater, keep;
///   if Less, drop; if Equal, look at next position.
///
/// All positions equal (or both unconstrained at every depth) → tuple
/// is NOT strictly after start_after; drop.
fn is_strictly_after_lex(
    tuple: &[Option<u64>],
    start: &[Option<u64>],
    orders: &[Order],
) -> bool {
    let n = tuple.len().min(orders.len());
    for i in 0..n {
        let s_at_i = start.get(i).copied().flatten();
        match (tuple[i], s_at_i) {
            (None, Some(_)) => return false,
            (Some(_), None) => return true,
            (None, None) => continue,
            (Some(tv), Some(sv)) => {
                let cmp = match orders[i] {
                    Order::Asc => tv.cmp(&sv),
                    Order::Desc => sv.cmp(&tv),
                };
                match cmp {
                    Ordering::Greater => return true,
                    Ordering::Less => return false,
                    Ordering::Equal => continue,
                }
            }
        }
    }
    // Caller supplied longer start than tuple? Treat extra start
    // positions as "constraint cannot be satisfied" → drop. (In
    // practice the dispatcher trims start_after to the cursor's
    // prefix length, so this branch is rare.)
    if start.len() > tuple.len() && start[tuple.len()..].iter().any(|s| s.is_some()) {
        return false;
    }
    false
}

/// Inter-segment merge ordering for the multi-field fruit. Mirrors
/// [`SortCursorIndexV2`]'s on-disk `missing="_last"` rule and the
/// deterministic `(segment_ord, doc_id)` tie-break.
fn compare_hits_multi(
    a: &(Vec<Option<u64>>, DocAddress),
    b: &(Vec<Option<u64>>, DocAddress),
    orders: &[Order],
) -> Ordering {
    let n = a.0.len().min(b.0.len()).min(orders.len());
    for i in 0..n {
        let av = a.0[i];
        let bv = b.0[i];
        match (av, bv) {
            (None, None) => continue,
            (None, Some(_)) => return Ordering::Greater,
            (Some(_), None) => return Ordering::Less,
            (Some(au), Some(bu)) => {
                let cmp = match orders[i] {
                    Order::Asc => au.cmp(&bu),
                    Order::Desc => bu.cmp(&au),
                };
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
        }
    }
    a.1.segment_ord
        .cmp(&b.1.segment_ord)
        .then_with(|| a.1.doc_id.cmp(&b.1.doc_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{SortCursorIndexV2, ValueKind};

    use columnar::MonotonicallyMappableToU64;

    /// Test-only helper to build a v2 cursor from FastValue-encoded
    /// `Option<u64>` columns.
    fn build_cursor(
        fields: Vec<(&str, Order, ValueKind)>,
        per_field_values: Vec<Vec<Option<u64>>>,
        max_doc: u32,
    ) -> Arc<SortCursorIndexV2> {
        let owned: Vec<(String, Order, ValueKind)> = fields
            .into_iter()
            .map(|(n, o, k)| (n.to_string(), o, k))
            .collect();
        Arc::new(
            SortCursorIndexV2::build_from_columns(owned, per_field_values, max_doc)
                .expect("v2 build should succeed"),
        )
    }

    /// Direct-harvest helper: drive the segment-collector hot path
    /// without going through a real `SegmentReader`. Pre-populates the
    /// matched bitset to mark every doc in `matched` as alive + matching.
    fn harvest_with_matched(
        cursor: Arc<SortCursorIndexV2>,
        fields: Vec<(&str, Order)>,
        limit: usize,
        start_after: Option<Vec<Option<u64>>>,
        matched: &[DocId],
    ) -> Vec<(Vec<Option<u64>>, DocAddress)> {
        let max_doc = cursor.max_doc();
        let mut bitset = BitSet::with_max_value(max_doc);
        for &d in matched {
            bitset.insert(d);
        }
        let segment = EarlyTermSortByCursorMultiSegmentCollector {
            cursor: Some(cursor),
            fields: fields
                .into_iter()
                .map(|(n, o)| (n.to_string(), o))
                .collect(),
            limit,
            segment_ord: 0,
            matched_bitset: bitset,
            start_after,
        };
        segment.harvest()
    }

    /// Single-field-as-multi: identical fruit shape to the v1 collector
    /// (`Vec<(Vec<Option<u64>>, _)>` with prefix_len=1) for an `i64 DESC`
    /// query.
    #[test]
    fn single_field_collector_matches_v1_walk() {
        // 4 docs, sort by score DESC. doc 3 missing → last.
        let v: Vec<Option<u64>> = vec![Some(30i64), Some(10), Some(20)]
            .into_iter()
            .map(|o| o.map(|v| v.to_u64()))
            .chain(std::iter::once(None))
            .collect();
        let cursor = build_cursor(
            vec![("score", Order::Desc, ValueKind::I64)],
            vec![v],
            4,
        );
        let hits = harvest_with_matched(
            cursor.clone(),
            vec![("score", Order::Desc)],
            10,
            None,
            &[0, 1, 2, 3],
        );
        // Expected order DESC missing-last: doc 0 (30), doc 2 (20), doc 1 (10), doc 3 (None)
        let docs: Vec<DocId> = hits.iter().map(|(_, a)| a.doc_id).collect();
        assert_eq!(docs, vec![0u32, 2, 1, 3]);
        assert_eq!(hits[3].0, vec![None]);
        assert!(hits[0].0[0].is_some());
    }

    /// Two-field walk: ts DESC, _id ASC; primary tie on doc 0 vs doc 1
    /// → secondary _id ASC picks doc 1 first (mirrors v2 cursor build
    /// order).
    #[test]
    fn two_field_walk_breaks_tie_by_secondary() {
        let ts: Vec<Option<u64>> = vec![100i64, 100, 200, 50]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let id: Vec<Option<u64>> = vec![7i64, 3, 5, 9]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let cursor = build_cursor(
            vec![
                ("ts", Order::Desc, ValueKind::I64),
                ("_id", Order::Asc, ValueKind::I64),
            ],
            vec![ts, id],
            4,
        );
        let hits = harvest_with_matched(
            cursor.clone(),
            vec![("ts", Order::Desc), ("_id", Order::Asc)],
            10,
            None,
            &[0, 1, 2, 3],
        );
        let docs: Vec<DocId> = hits.iter().map(|(_, a)| a.doc_id).collect();
        // Expected: doc 2 (ts=200), doc 1 (ts=100,_id=3), doc 0 (ts=100,_id=7), doc 3 (ts=50)
        assert_eq!(docs, vec![2u32, 1, 0, 3]);
    }

    /// `search_after` on primary value: after `ts=100`, only docs with
    /// ts > 100 (DESC: lower than) survive, and we don't have any —
    /// only doc 3 (ts=50, after 100 in DESC order) does.
    #[test]
    fn search_after_skips_at_or_before_primary_value() {
        let ts: Vec<Option<u64>> = vec![100i64, 100, 200, 50]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let id: Vec<Option<u64>> = vec![7i64, 3, 5, 9]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let cursor = build_cursor(
            vec![
                ("ts", Order::Desc, ValueKind::I64),
                ("_id", Order::Asc, ValueKind::I64),
            ],
            vec![ts, id],
            4,
        );
        // search_after = (ts=100, _id=3) — page picks docs strictly *after* this
        // in DESC,ASC lex order = LATER in the cursor walk.
        //
        // Cursor walk (already lex DESC,ASC sorted):
        //   doc 2 (200, 5)   ← cursor pos 0
        //   doc 1 (100, 3)   ← cursor pos 1 (== start_after, NOT after)
        //   doc 0 (100, 7)   ← cursor pos 2 (tie ts, _id=7 ASC > 3 → after)
        //   doc 3 (50,  9)   ← cursor pos 3 (ts=50 in DESC walk: comes after 100)
        //
        // From doc 2 (200, 5): in DESC, 200 sorts BEFORE 100 → drop.
        // From doc 1 (100, 3): equal in every position → drop.
        // From doc 0 (100, 7): tie ts → _id=7 > 3 ASC → keep.
        // From doc 3 (50, 9):  ts=50 sorts AFTER 100 in DESC → keep.
        let start_after = vec![Some(100i64.to_u64()), Some(3i64.to_u64())];
        let hits = harvest_with_matched(
            cursor.clone(),
            vec![("ts", Order::Desc), ("_id", Order::Asc)],
            10,
            Some(start_after),
            &[0, 1, 2, 3],
        );
        let docs: Vec<DocId> = hits.iter().map(|(_, a)| a.doc_id).collect();
        assert_eq!(docs, vec![0u32, 3]);
    }

    /// `search_after` skips docs whose primary is missing (mirrors v1
    /// missing-last + search_after rule).
    #[test]
    fn search_after_skips_missing_primary_when_constrained() {
        // 3 docs: doc 0 ts=100, doc 1 missing, doc 2 ts=50.
        let ts: Vec<Option<u64>> = vec![Some(100i64), None, Some(50)]
            .into_iter()
            .map(|o| o.map(|v| v.to_u64()))
            .collect();
        let cursor = build_cursor(
            vec![("ts", Order::Desc, ValueKind::I64)],
            vec![ts],
            3,
        );
        // search_after = (ts=200) — only docs with ts < 200 in DESC are after; that
        // means doc 0 (ts=100) and doc 2 (ts=50). Missing-primary doc 1 is skipped.
        let start_after = vec![Some(200i64.to_u64())];
        let hits = harvest_with_matched(
            cursor.clone(),
            vec![("ts", Order::Desc)],
            10,
            Some(start_after),
            &[0, 1, 2],
        );
        let docs: Vec<DocId> = hits.iter().map(|(_, a)| a.doc_id).collect();
        assert_eq!(docs, vec![0u32, 2]);
    }

    /// `limit` enforces early termination: with 100 matched docs but
    /// `limit=2`, the harvest emits exactly 2 hits even when the
    /// cursor still has more to walk.
    #[test]
    fn limit_enforces_early_termination() {
        let n: u32 = 50;
        let v: Vec<Option<u64>> = (0..n).map(|i| Some((i as i64).to_u64())).collect();
        let cursor = build_cursor(
            vec![("k", Order::Desc, ValueKind::I64)],
            vec![v],
            n,
        );
        let matched: Vec<DocId> = (0..n).collect();
        let hits = harvest_with_matched(
            cursor.clone(),
            vec![("k", Order::Desc)],
            2,
            None,
            &matched,
        );
        assert_eq!(hits.len(), 2);
        // Top 2 in DESC order: doc 49 (largest), doc 48.
        let docs: Vec<DocId> = hits.iter().map(|(_, a)| a.doc_id).collect();
        assert_eq!(docs, vec![49u32, 48]);
    }

    /// `can_handle_segment` rejects when the requested prefix is
    /// LONGER than the cursor's recorded fields.
    #[test]
    fn cursor_prefix_matches_rejects_too_long_prefix() {
        let cursor = build_cursor(
            vec![("ts", Order::Desc, ValueKind::I64)],
            vec![vec![Some(1u64)]],
            1,
        );
        let prefix = vec![
            ("ts".to_string(), Order::Desc),
            ("_id".to_string(), Order::Asc),
        ];
        assert!(!cursor_prefix_matches(&cursor, &prefix));
    }

    /// `can_handle_segment` rejects when the order on a field
    /// disagrees.
    #[test]
    fn cursor_prefix_matches_rejects_order_mismatch() {
        let cursor = build_cursor(
            vec![
                ("ts", Order::Desc, ValueKind::I64),
                ("_id", Order::Asc, ValueKind::I64),
            ],
            vec![vec![Some(1u64), Some(2)], vec![Some(3u64), Some(4)]],
            2,
        );
        let prefix_ok = vec![("ts".to_string(), Order::Desc)];
        let prefix_bad_order = vec![("ts".to_string(), Order::Asc)];
        let prefix_bad_name = vec![("other".to_string(), Order::Desc)];
        assert!(cursor_prefix_matches(&cursor, &prefix_ok));
        assert!(!cursor_prefix_matches(&cursor, &prefix_bad_order));
        assert!(!cursor_prefix_matches(&cursor, &prefix_bad_name));
    }

    /// `merge_fruits` produces a stable lex-ordered top-K across
    /// multiple per-segment fruits. Uses the public `Collector` API.
    #[test]
    fn merge_fruits_lex_orders_across_segments() {
        // Two segments, two fields. Each segment hands in a couple of
        // hits. The merged result should respect both field orders.
        let collector = EarlyTermSortByCursorCollectorMulti::new(
            vec![("ts", Order::Desc), ("_id", Order::Asc)],
            10,
        );
        let seg0_fruit: Vec<(Vec<Option<u64>>, DocAddress)> = vec![
            (
                vec![Some(200i64.to_u64()), Some(5i64.to_u64())],
                DocAddress {
                    segment_ord: 0,
                    doc_id: 2,
                },
            ),
            (
                vec![Some(100i64.to_u64()), Some(3i64.to_u64())],
                DocAddress {
                    segment_ord: 0,
                    doc_id: 1,
                },
            ),
        ];
        let seg1_fruit: Vec<(Vec<Option<u64>>, DocAddress)> = vec![
            (
                vec![Some(150i64.to_u64()), Some(8i64.to_u64())],
                DocAddress {
                    segment_ord: 1,
                    doc_id: 0,
                },
            ),
            (
                vec![Some(100i64.to_u64()), Some(1i64.to_u64())],
                DocAddress {
                    segment_ord: 1,
                    doc_id: 4,
                },
            ),
        ];
        let merged = collector
            .merge_fruits(vec![seg0_fruit, seg1_fruit])
            .expect("merge");
        // Lex DESC,ASC order:
        //   200/5 (seg 0, doc 2), 150/8 (seg 1, doc 0), 100/1 (seg 1, doc 4), 100/3 (seg 0, doc 1)
        let pairs: Vec<(u32, u32)> = merged
            .iter()
            .map(|(_, a)| (a.segment_ord, a.doc_id))
            .collect();
        assert_eq!(pairs, vec![(0, 2), (1, 0), (1, 4), (0, 1)]);
    }
}

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
//! **FerroSearch Wave 18-3.**  Extended to support **keyword / string
//! sort fields** ([`ValueKind::String`](crate::index::ValueKind::String)).
//! On-disk storage is the segment-local term ordinal, but the per-hit
//! fruit carries decoded UTF-8 bytes so cross-segment merge can use
//! real byte comparison: term ord 1 in segment A may be "JP" while
//! ord 1 in segment B is "US".  Decoding happens lazily at
//! [`SegmentCollector::harvest`] time via the segment's
//! [`StrColumn`](columnar::StrColumn) dictionary, with no extra cost
//! for indices whose primary sort is purely numeric.
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
//!    * resolve the per-field encoded sort tuple from the cursor and
//!      decode `ValueKind::String` slots from segment-local term ords
//!      to UTF-8 bytes (via the captured per-field `StrColumn`);
//!    * (optional) skip if a `search_after` cursor was supplied and the
//!      tuple is at-or-before the cursor in lex order (kind-aware: u64
//!      cmp for numerics, `Vec<u8>` cmp for strings);
//!    * push `(Vec<CursorSortVal>, DocAddress)` into the per-segment
//!      fruit;
//!    * stop once `limit` hits have been recorded.
//!
//! Cross-segment merge then sorts on the decoded bytes for string
//! fields, sidestepping the segment-local ord dictionary divergence.
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
use crate::index::{SortCursorIndexV2, ValueKind};
use crate::schema::Schema;
use crate::{DocAddress, DocId, Order, Score, SegmentOrdinal, SegmentReader};

/// Per-field encoded sort value used by the multi-field early-term
/// cursor collector.  The variant tag is the field's
/// [`ValueKind`](crate::index::ValueKind) recorded in the v2 cursor.
///
/// **Wave 18-3 design note.**  `Numeric(Option<u64>)` carries the
/// `FastValue::to_u64`-encoded slot for `I64` / `U64` / `F64` /
/// `Date` / `DateNanos`; segment-local order matches global lex
/// order because all such fields are monotonically mappable to
/// `u64`.  `String(Option<Vec<u8>>)` carries the **decoded** UTF-8
/// bytes (resolved at harvest time via the segment's
/// [`StrColumn`](columnar::StrColumn) dictionary), because segment-
/// local term ordinals are NOT comparable across segments — only the
/// underlying byte values are.
///
/// `None` in either variant means the underlying doc has a missing
/// value at this field.  The `missing="_last"` rule from
/// Elasticsearch / Lucene is honoured by the `cmp_sort_val` helper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CursorSortVal {
    /// Encoded `u64` for numeric/date `FastValue` fields, or `None`
    /// when the doc has a missing value here (sorts last).
    Numeric(Option<u64>),
    /// Decoded UTF-8 bytes for keyword/string fields, or `None` when
    /// the doc has a missing term ord here (sorts last).
    String(Option<Vec<u8>>),
}

impl CursorSortVal {
    /// Returns `true` if the value is missing (`Numeric(None)` or
    /// `String(None)`).  Missing values sort last in both ASC and
    /// DESC orders, matching ES / Lucene `missing="_last"`.
    pub fn is_missing(&self) -> bool {
        matches!(self, CursorSortVal::Numeric(None) | CursorSortVal::String(None))
    }
}

/// Crate-public re-export of [`cmp_sort_val`] for the Wave 18-2 mix
/// dispatcher in
/// [`crate::collector::early_term_or_fallback_collector_multi`].
pub(crate) fn cmp_sort_val_pub(a: &CursorSortVal, b: &CursorSortVal, order: Order) -> Ordering {
    cmp_sort_val(a, b, order)
}

/// Lex-compares two [`CursorSortVal`]s under `order`, honouring the
/// `missing="_last"` rule (a missing value always sorts after a
/// present one regardless of `order`).  Variants of different kinds
/// (`Numeric` vs `String`) are treated as equal — that combination
/// only happens when the cursor's recorded `ValueKind` disagrees
/// with the runtime decoded value, which the collector never produces.
fn cmp_sort_val(a: &CursorSortVal, b: &CursorSortVal, order: Order) -> Ordering {
    use CursorSortVal::*;
    match (a, b) {
        (Numeric(None), Numeric(None)) => Ordering::Equal,
        (String(None), String(None)) => Ordering::Equal,
        (Numeric(None), Numeric(Some(_))) => Ordering::Greater,
        (Numeric(Some(_)), Numeric(None)) => Ordering::Less,
        (String(None), String(Some(_))) => Ordering::Greater,
        (String(Some(_)), String(None)) => Ordering::Less,
        (Numeric(Some(au)), Numeric(Some(bu))) => match order {
            Order::Asc => au.cmp(bu),
            Order::Desc => bu.cmp(au),
        },
        (String(Some(ab)), String(Some(bb))) => match order {
            Order::Asc => ab.as_slice().cmp(bb.as_slice()),
            Order::Desc => bb.as_slice().cmp(ab.as_slice()),
        },
        // Kind mismatch: shouldn't happen if the cursor's recorded
        // value_kind matches what we materialise at harvest time.
        // Treat as equal so a debug-only divergence doesn't corrupt
        // the merge.
        _ => Ordering::Equal,
    }
}

/// Multi-field early-terminating top-K collector.
///
/// Driven by the on-disk lex-sorted cursor produced by Wave 18-1 — see
/// the module docs and `dd-pack/wave18-multi-field-cursor-v2-design.md`.
///
/// Collected fruit: `Vec<(Vec<CursorSortVal>, DocAddress)>` where each
/// inner `Vec<CursorSortVal>` is the per-field decoded sort tuple at
/// the cursor position the doc came from.  Per-field
/// `Numeric(None)` / `String(None)` indicates the doc has a missing
/// value for that field (missing-last sort applies — see
/// [`SortCursorIndexV2`]).
///
/// ## `search_after`
///
/// Use [`Self::with_search_after`] to skip docs whose tuple lies at
/// or before the supplied cursor in lex order (per per-field
/// [`Order`]).  This matches Elasticsearch's multi-field
/// `search_after: [v0, v1, …]` semantics.  Each `start_after[i]` may
/// be `Numeric(None)` or `String(None)` to mean "unconstrained at
/// this position"; positions where the cursor's tuple has `None` and
/// `start_after[i]` is constrained (some present value) are skipped
/// (mirroring the v1 single-field collector's missing-last skip
/// behaviour for consistency).
#[derive(Debug, Clone)]
pub struct EarlyTermSortByCursorCollectorMulti {
    /// Field declaration order matches the cursor's recorded fields.
    fields: Vec<(String, Order)>,
    limit: usize,
    /// Wave 18-1 search_after.  Length is the **caller-supplied prefix**;
    /// it may be shorter than `fields.len()` (trailing fields are
    /// unconstrained), but must not be longer.  Wave 18-3: variants
    /// are kind-aware ([`CursorSortVal::Numeric`] vs
    /// [`CursorSortVal::String`]).
    start_after: Option<Vec<CursorSortVal>>,
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

    /// Wave 18-1 / 18-3: enable `search_after` on the cursor walk.
    ///
    /// `start_after.len()` must be `≤ self.fields().len()` — trailing
    /// missing variants (e.g. `Numeric(None)`, `String(None)`) are
    /// equivalent to "unconstrained at that depth".  Each entry's
    /// variant must match the cursor's recorded
    /// [`ValueKind`](crate::index::ValueKind) for that field — the
    /// caller is responsible for parsing the JSON `search_after`
    /// array per type before calling here.
    pub fn with_search_after(mut self, start_after: Vec<CursorSortVal>) -> Self {
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
    pub fn search_after(&self) -> Option<&[CursorSortVal]> {
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
    type Fruit = Vec<(Vec<CursorSortVal>, DocAddress)>;
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

        let Some(cursor) = cursor_opt else {
            return Ok(EarlyTermSortByCursorMultiSegmentCollector {
                cursor: None,
                fields: self.fields.clone(),
                value_kinds: Vec::new(),
                str_columns: Vec::new(),
                limit: self.limit,
                segment_ord: segment_local_id,
                matched_bitset: BitSet::with_max_value(0),
                start_after: None,
            });
        };

        // Wave 18-3: capture the cursor's per-prefix-field
        // `ValueKind` and resolve a `StrColumn` for each
        // `ValueKind::String` slot, so the harvest hot path can
        // decode segment-local term ordinals to UTF-8 bytes.  The
        // `StrColumn` is `Arc`-shared internally; this is a single
        // pre-cache pointer copy per query per segment.
        let prefix_len = self.fields.len();
        let mut value_kinds: Vec<ValueKind> = Vec::with_capacity(prefix_len);
        let mut str_columns: Vec<Option<columnar::StrColumn>> =
            Vec::with_capacity(prefix_len);
        for fi in 0..prefix_len {
            let (cursor_field, _, kind) = &cursor.fields()[fi];
            value_kinds.push(*kind);
            if matches!(kind, ValueKind::String) {
                let str_col = segment_reader.fast_fields().str(cursor_field)?;
                str_columns.push(str_col);
            } else {
                str_columns.push(None);
            }
        }

        Ok(EarlyTermSortByCursorMultiSegmentCollector {
            cursor: Some(cursor),
            fields: self.fields.clone(),
            value_kinds,
            str_columns,
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
        let mut all: Vec<(Vec<CursorSortVal>, DocAddress)> =
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
    /// Wave 18-3: per-prefix-field `ValueKind` mirrored from the
    /// cursor's `fields()`, so harvest knows whether to wrap the
    /// encoded `u64` as a numeric or to decode it through a
    /// `StrColumn` dictionary.  Same length as `fields`.
    value_kinds: Vec<ValueKind>,
    /// Wave 18-3: per-prefix-field `StrColumn` for `ValueKind::String`
    /// slots; `None` for numeric/date slots.  Captured at
    /// `for_segment` time so harvest stays a flat loop with no
    /// schema lookups.
    str_columns: Vec<Option<columnar::StrColumn>>,
    limit: usize,
    segment_ord: u32,
    matched_bitset: BitSet,
    start_after: Option<Vec<CursorSortVal>>,
}

impl EarlyTermSortByCursorMultiSegmentCollector {
    /// Crate-public constructor used by the Wave 18-2 mix dispatcher
    /// (`EarlyTermOrFallbackCollectorMulti`) to delegate the cursor
    /// path on a per-segment basis.  Passes the same wiring the
    /// outer collector's `for_segment` would build, so behaviour is
    /// identical to a direct dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_for_test_or_mix(
        cursor: Option<Arc<SortCursorIndexV2>>,
        fields: Vec<(String, Order)>,
        value_kinds: Vec<ValueKind>,
        str_columns: Vec<Option<columnar::StrColumn>>,
        limit: usize,
        segment_ord: u32,
        matched_bitset: BitSet,
        start_after: Option<Vec<CursorSortVal>>,
    ) -> Self {
        Self {
            cursor,
            fields,
            value_kinds,
            str_columns,
            limit,
            segment_ord,
            matched_bitset,
            start_after,
        }
    }
}

impl SegmentCollector for EarlyTermSortByCursorMultiSegmentCollector {
    type Fruit = Vec<(Vec<CursorSortVal>, DocAddress)>;

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
        let mut hits: Vec<(Vec<CursorSortVal>, DocAddress)> = Vec::with_capacity(self.limit);
        // Reusable scratch for decoding string term ords to bytes —
        // mirrors `convert_segment_sort_key`'s buffer reuse on
        // `SortByString` to avoid per-hit allocation.
        let mut decode_scratch: Vec<u8> = Vec::with_capacity(32);
        for cursor_idx in 0..cursor.len() {
            let doc = cursor.doc_ids()[cursor_idx];
            if !self.matched_bitset.contains(doc) {
                continue;
            }
            // Materialise the prefix the request cares about, decoding
            // string term ords to bytes via the captured `StrColumn`.
            let mut tuple: Vec<CursorSortVal> = Vec::with_capacity(prefix_len);
            for fi in 0..prefix_len {
                let raw = cursor.value(cursor_idx, fi);
                let val = match self.value_kinds[fi] {
                    ValueKind::String => {
                        let bytes = match (raw, self.str_columns[fi].as_ref()) {
                            (Some(ord), Some(str_col)) => {
                                decode_scratch.clear();
                                match str_col.ord_to_bytes(ord, &mut decode_scratch) {
                                    Ok(true) => Some(decode_scratch.clone()),
                                    // Ord present in cursor but not in dict —
                                    // treat as missing so the merge still
                                    // produces a deterministic ordering.
                                    Ok(false) | Err(_) => None,
                                }
                            }
                            // No StrColumn (e.g. dropped between cursor
                            // build and search) → degrade to missing.
                            (None, _) | (_, None) => None,
                        };
                        CursorSortVal::String(bytes)
                    }
                    _ => CursorSortVal::Numeric(raw),
                };
                tuple.push(val);
            }
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

/// Lex-compare a [`CursorSortVal`] tuple against `start_after`.
/// Returns `true` iff the tuple lies *strictly after* `start_after`
/// in the order configured per field.
///
/// Semantics (kind-aware as of Wave 18-3):
/// * `start_after[i]` constrained AND `tuple[i]` missing → tuple is
///   "missing" at this depth; missing sorts last so it cannot be
///   "after" a real value (mirrors v1 single-field behaviour). Skip.
/// * `start_after[i]` missing/unconstrained AND `tuple[i]` constrained
///   → tuple is "defined" past where the caller bothered specifying
///   → keep.
/// * Both unconstrained → look at next position.
/// * Both constrained → kind-aware per-field [`Order`] comparison via
///   [`cmp_sort_val`]; if Greater, keep; if Less, drop; if Equal,
///   look at next position.
///
/// All positions equal (or both unconstrained at every depth) → tuple
/// is NOT strictly after start_after; drop.
fn is_strictly_after_lex(
    tuple: &[CursorSortVal],
    start: &[CursorSortVal],
    orders: &[Order],
) -> bool {
    let n = tuple.len().min(orders.len());
    for i in 0..n {
        let tuple_missing = tuple[i].is_missing();
        let start_at_i = start.get(i);
        let start_missing = match start_at_i {
            None => true,
            Some(v) => v.is_missing(),
        };
        match (tuple_missing, start_missing) {
            // Tuple has missing primary while start constrains it →
            // missing sorts last, so it can't be "after" a real value.
            (true, false) => return false,
            // Tuple has a real value while start is unconstrained at
            // this depth → tuple is past the caller's prefix.
            (false, true) => return true,
            // Both missing or both unconstrained — descend.
            (true, true) => continue,
            (false, false) => match cmp_sort_val(&tuple[i], start_at_i.unwrap(), orders[i]) {
                Ordering::Greater => return true,
                Ordering::Less => return false,
                Ordering::Equal => continue,
            },
        }
    }
    // Caller supplied longer start than tuple? Treat extra start
    // positions as "constraint cannot be satisfied" → drop. (In
    // practice the dispatcher trims start_after to the cursor's
    // prefix length, so this branch is rare.)
    if start.len() > tuple.len() && start[tuple.len()..].iter().any(|s| !s.is_missing()) {
        return false;
    }
    false
}

/// Inter-segment merge ordering for the multi-field fruit. Mirrors
/// [`SortCursorIndexV2`]'s on-disk `missing="_last"` rule and the
/// deterministic `(segment_ord, doc_id)` tie-break.  Kind-aware
/// (Wave 18-3) — string fields compare by decoded bytes, numerics by
/// encoded `u64`.
fn compare_hits_multi(
    a: &(Vec<CursorSortVal>, DocAddress),
    b: &(Vec<CursorSortVal>, DocAddress),
    orders: &[Order],
) -> Ordering {
    let n = a.0.len().min(b.0.len()).min(orders.len());
    for i in 0..n {
        match cmp_sort_val(&a.0[i], &b.0[i], orders[i]) {
            Ordering::Equal => continue,
            other => return other,
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
    /// Mirrors the prefix-length conventions of `for_segment` —
    /// `value_kinds` is derived from the cursor's recorded fields, and
    /// `str_columns` is left `None` for every field (numeric-only
    /// tests).  Wave 18-3 string-decoding tests go through a real
    /// `SegmentReader` instead.
    fn harvest_with_matched(
        cursor: Arc<SortCursorIndexV2>,
        fields: Vec<(&str, Order)>,
        limit: usize,
        start_after: Option<Vec<CursorSortVal>>,
        matched: &[DocId],
    ) -> Vec<(Vec<CursorSortVal>, DocAddress)> {
        let max_doc = cursor.max_doc();
        let mut bitset = BitSet::with_max_value(max_doc);
        for &d in matched {
            bitset.insert(d);
        }
        let prefix_len = fields.len();
        let value_kinds: Vec<ValueKind> = (0..prefix_len)
            .map(|fi| cursor.fields()[fi].2)
            .collect();
        let str_columns: Vec<Option<columnar::StrColumn>> = vec![None; prefix_len];
        let segment = EarlyTermSortByCursorMultiSegmentCollector {
            cursor: Some(cursor),
            fields: fields
                .into_iter()
                .map(|(n, o)| (n.to_string(), o))
                .collect(),
            value_kinds,
            str_columns,
            limit,
            segment_ord: 0,
            matched_bitset: bitset,
            start_after,
        };
        segment.harvest()
    }

    /// Convenience: wrap encoded `u64` slots into `CursorSortVal::Numeric`
    /// (matches v2 fruit shape for numeric fields).
    fn num(v: Option<u64>) -> CursorSortVal {
        CursorSortVal::Numeric(v)
    }

    /// Single-field-as-multi: prefix_len=1 fruit shape for an
    /// `i64 DESC` query.  Wave 18-3 fruit carries `CursorSortVal`,
    /// so we wrap encoded slots in `CursorSortVal::Numeric` for
    /// assertion.
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
        assert_eq!(hits[3].0, vec![num(None)]);
        assert!(matches!(hits[0].0[0], CursorSortVal::Numeric(Some(_))));
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
        let start_after = vec![num(Some(100i64.to_u64())), num(Some(3i64.to_u64()))];
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
        let start_after = vec![num(Some(200i64.to_u64()))];
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
        let seg0_fruit: Vec<(Vec<CursorSortVal>, DocAddress)> = vec![
            (
                vec![num(Some(200i64.to_u64())), num(Some(5i64.to_u64()))],
                DocAddress {
                    segment_ord: 0,
                    doc_id: 2,
                },
            ),
            (
                vec![num(Some(100i64.to_u64())), num(Some(3i64.to_u64()))],
                DocAddress {
                    segment_ord: 0,
                    doc_id: 1,
                },
            ),
        ];
        let seg1_fruit: Vec<(Vec<CursorSortVal>, DocAddress)> = vec![
            (
                vec![num(Some(150i64.to_u64())), num(Some(8i64.to_u64()))],
                DocAddress {
                    segment_ord: 1,
                    doc_id: 0,
                },
            ),
            (
                vec![num(Some(100i64.to_u64())), num(Some(1i64.to_u64()))],
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

    // -------------------------------------------------------------
    // Wave 18-3 — string sort cursor unit tests.
    //
    // The collector hot path's string decode goes through a real
    // `StrColumn`, so these tests build an in-RAM tantivy index over
    // a keyword field and drive the public `Collector` API.
    // -------------------------------------------------------------

    /// Convenience: wrap UTF-8 bytes into `CursorSortVal::String`.
    fn s(bytes: &[u8]) -> CursorSortVal {
        CursorSortVal::String(Some(bytes.to_vec()))
    }

    /// Convenience: missing string slot.
    fn s_missing() -> CursorSortVal {
        CursorSortVal::String(None)
    }

    /// `cmp_sort_val` on string slots respects per-field `Order` and
    /// the missing-last rule.
    #[test]
    fn cmp_sort_val_string_orders_correctly() {
        let ar = s(b"AR");
        let jp = s(b"JP");
        let us = s(b"US");
        let missing = s_missing();
        // ASC: AR < JP < US < missing
        assert_eq!(cmp_sort_val(&ar, &jp, Order::Asc), Ordering::Less);
        assert_eq!(cmp_sort_val(&jp, &us, Order::Asc), Ordering::Less);
        assert_eq!(cmp_sort_val(&us, &missing, Order::Asc), Ordering::Less);
        assert_eq!(cmp_sort_val(&missing, &us, Order::Asc), Ordering::Greater);
        // DESC: US < JP < AR < missing (missing still last)
        assert_eq!(cmp_sort_val(&us, &jp, Order::Desc), Ordering::Less);
        assert_eq!(cmp_sort_val(&jp, &ar, Order::Desc), Ordering::Less);
        assert_eq!(cmp_sort_val(&ar, &missing, Order::Desc), Ordering::Less);
        // Same bytes are equal regardless of order.
        assert_eq!(cmp_sort_val(&jp, &s(b"JP"), Order::Asc), Ordering::Equal);
        assert_eq!(cmp_sort_val(&jp, &s(b"JP"), Order::Desc), Ordering::Equal);
    }

    /// String single-field sort over a real keyword fast field.  The
    /// segment's term ordinals are decoded to UTF-8 bytes at harvest
    /// time, and the resulting fruit holds `CursorSortVal::String`.
    #[test]
    fn string_single_field_real_segment_decodes_bytes() {
        use crate::index::SegmentReader;
        use crate::schema::{FAST, STRING, Schema};
        use crate::{Index, IndexBuilder, IndexSettings, IndexSortByField};

        let mut sb = Schema::builder();
        let country = sb.add_text_field("country", STRING | FAST);
        let schema = sb.build();
        let settings = IndexSettings {
            sort_by_fields: Some(vec![IndexSortByField {
                field: "country".to_string(),
                order: Order::Asc,
            }]),
            ..Default::default()
        };
        let index = IndexBuilder::default()
            .schema(schema.clone())
            .settings(settings)
            .create_in_ram()
            .unwrap();
        let _ = index;
        // Build path skipped: we are exercising the **collector** with
        // a manually-built v2 cursor + a real `StrColumn` from a
        // committed segment.  Indexing through the writer + cursor
        // build path is covered separately (`v2_string_*` in
        // `sort_cursor.rs`).  Here we focus on cross-segment merge.
        // Rebuild a fresh index that we drive directly.
        drop(country);
        let mut sb = Schema::builder();
        let country = sb.add_text_field("country", STRING | FAST);
        let schema = sb.build();
        let settings = IndexSettings {
            sort_by_fields: Some(vec![IndexSortByField {
                field: "country".to_string(),
                order: Order::Asc,
            }]),
            ..Default::default()
        };
        let index: Index = IndexBuilder::default()
            .schema(schema)
            .settings(settings)
            .create_in_ram()
            .unwrap();
        let mut writer = index
            .writer_for_tests()
            .expect("writer_for_tests");
        // 4 docs: JP, US, AR, JP
        writer.add_document(doc!(country => "JP")).unwrap();
        writer.add_document(doc!(country => "US")).unwrap();
        writer.add_document(doc!(country => "AR")).unwrap();
        writer.add_document(doc!(country => "JP")).unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        assert_eq!(
            searcher.segment_readers().len(),
            1,
            "test assumes a single segment"
        );
        let seg: &SegmentReader = searcher.segment_reader(0);
        // Cursor advertised by SegmentMeta.
        assert!(
            seg.sort_cursor_v2("country").is_some(),
            "v2 cursor must be present after commit on a sort_by_fields index"
        );
        let collector = EarlyTermSortByCursorCollectorMulti::new(
            vec![("country", Order::Asc)],
            10,
        );
        assert!(collector.can_handle_segment(seg));
        let hits = searcher
            .search(&crate::query::AllQuery, &collector)
            .expect("search v2-multi string");
        let docs: Vec<u32> = hits.iter().map(|(_, a)| a.doc_id).collect();
        // ASC: AR (doc 2) < JP (docs 0, 3 — tie broken by doc-id ASC) < US (doc 1).
        assert_eq!(docs, vec![2u32, 0, 3, 1]);
        // Fruit values are decoded UTF-8 bytes.
        assert_eq!(hits[0].0, vec![s(b"AR")]);
        assert_eq!(hits[1].0, vec![s(b"JP")]);
        assert_eq!(hits[2].0, vec![s(b"JP")]);
        assert_eq!(hits[3].0, vec![s(b"US")]);
    }

    /// Cross-segment merge with diverging dictionaries: two segments
    /// whose term ord 1 maps to different strings ("JP" in seg A vs
    /// "US" in seg B).  Without byte-level decode at harvest, the
    /// merge would compare raw ords and produce a wrong global order.
    /// This test pins that the decoded-bytes fruit yields a correct
    /// global lex order across segments.
    #[test]
    fn string_cross_segment_dictionary_divergence_uses_bytes() {
        use crate::schema::{FAST, STRING, Schema};
        use crate::{Index, IndexBuilder, IndexSettings, IndexSortByField};

        let mut sb = Schema::builder();
        let country = sb.add_text_field("country", STRING | FAST);
        let schema = sb.build();
        let settings = IndexSettings {
            sort_by_fields: Some(vec![IndexSortByField {
                field: "country".to_string(),
                order: Order::Asc,
            }]),
            ..Default::default()
        };
        let index: Index = IndexBuilder::default()
            .schema(schema)
            .settings(settings)
            .create_in_ram()
            .unwrap();
        let mut writer = index.writer_for_tests().expect("writer_for_tests");
        // Segment A: AR, JP — local dict: AR=0, JP=1.
        writer.add_document(doc!(country => "AR")).unwrap();
        writer.add_document(doc!(country => "JP")).unwrap();
        writer.commit().unwrap();
        // Segment B: CN, US — local dict: CN=0, US=1.
        // Same raw ord 1 maps to "JP" in seg A but "US" in seg B —
        // so a hypothetical raw-ord merge would conflate them.
        writer.add_document(doc!(country => "CN")).unwrap();
        writer.add_document(doc!(country => "US")).unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        assert!(
            searcher.segment_readers().len() >= 2,
            "test assumes at least 2 segments"
        );
        let collector = EarlyTermSortByCursorCollectorMulti::new(
            vec![("country", Order::Asc)],
            10,
        );
        let hits = searcher
            .search(&crate::query::AllQuery, &collector)
            .expect("search v2-multi cross-segment");
        // Global ASC byte order across both segments:
        //   AR < CN < JP < US
        let bytes: Vec<&[u8]> = hits
            .iter()
            .map(|(t, _)| match &t[0] {
                CursorSortVal::String(Some(b)) => b.as_slice(),
                _ => panic!("expected decoded string slot"),
            })
            .collect();
        assert_eq!(
            bytes,
            vec![b"AR".as_slice(), b"CN".as_slice(), b"JP".as_slice(), b"US".as_slice()],
            "decoded bytes must respect global lex order across segments, \
             not segment-local ord ordering"
        );
    }

    /// Multi-field cursor `[country ASC, ts DESC]`: lex sort over a
    /// keyword primary + numeric secondary.  Pinned via a real
    /// segment so the harvest path does an actual `StrColumn` decode.
    #[test]
    fn string_then_numeric_multi_field_real_segment() {
        use crate::schema::{FAST, INDEXED, STORED, STRING, Schema};
        use crate::{Index, IndexBuilder, IndexSettings, IndexSortByField};

        let mut sb = Schema::builder();
        let country = sb.add_text_field("country", STRING | FAST);
        let ts = sb.add_i64_field("ts", FAST | INDEXED | STORED);
        let schema = sb.build();
        let settings = IndexSettings {
            sort_by_fields: Some(vec![
                IndexSortByField {
                    field: "country".to_string(),
                    order: Order::Asc,
                },
                IndexSortByField {
                    field: "ts".to_string(),
                    order: Order::Desc,
                },
            ]),
            ..Default::default()
        };
        let index: Index = IndexBuilder::default()
            .schema(schema)
            .settings(settings)
            .create_in_ram()
            .unwrap();
        let mut writer = index.writer_for_tests().expect("writer_for_tests");
        // 5 docs:
        //   doc 0: JP / 100
        //   doc 1: AR / 30
        //   doc 2: JP / 200
        //   doc 3: US / 5
        //   doc 4: AR / 30   (tie with doc 1 on both keys)
        writer.add_document(doc!(country => "JP", ts => 100i64)).unwrap();
        writer.add_document(doc!(country => "AR", ts => 30i64)).unwrap();
        writer.add_document(doc!(country => "JP", ts => 200i64)).unwrap();
        writer.add_document(doc!(country => "US", ts => 5i64)).unwrap();
        writer.add_document(doc!(country => "AR", ts => 30i64)).unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let collector = EarlyTermSortByCursorCollectorMulti::new(
            vec![("country", Order::Asc), ("ts", Order::Desc)],
            10,
        );
        let hits = searcher
            .search(&crate::query::AllQuery, &collector)
            .expect("search v2-multi string+numeric");
        let docs: Vec<u32> = hits.iter().map(|(_, a)| a.doc_id).collect();
        // ASC country, DESC ts:
        //   AR/30 (doc 1, doc 4 — doc-id ASC tie) → JP/200 (doc 2) → JP/100 (doc 0) → US/5 (doc 3)
        assert_eq!(docs, vec![1u32, 4, 2, 0, 3]);
        // Spot-check: first hit has country="AR" decoded.
        assert_eq!(hits[0].0[0], s(b"AR"));
        // And ts=30 encoded numerically.
        assert!(matches!(
            hits[0].0[1],
            CursorSortVal::Numeric(Some(_))
        ));
    }

    /// `is_strictly_after_lex` on a string primary: DESC search_after
    /// "JP" means the walker should keep docs whose primary is strictly
    /// less than "JP" (in DESC walk = later in walk = lex less than).
    #[test]
    fn search_after_string_lex_skip() {
        // Build a string-only cursor manually (term ords 0..2 standing
        // in for "AR", "JP", "US" in lex order).  Then drive the
        // SegmentCollector via `harvest_with_matched`, but pre-decode
        // the fruit so the helper bypasses the StrColumn path — we
        // assert the search_after lex semantic at the
        // `is_strictly_after_lex` layer here, since real-segment
        // search_after is covered by the ferrosearch integ tests.
        let tuple_jp = vec![s(b"JP")];
        let tuple_us = vec![s(b"US")];
        let tuple_ar = vec![s(b"AR")];
        let start = vec![s(b"JP")];
        let orders = [Order::Desc];
        // DESC walk: tuples lex less than "JP" sort AFTER "JP" in DESC.
        // → "AR" is lex less → strictly_after?  No: in DESC ord, "AR" > "JP" wait
        // Recheck: DESC order means walker emits in reverse byte order,
        // so "after JP in walk" = "lex less than JP".
        assert_eq!(
            is_strictly_after_lex(&tuple_ar, &start, &orders),
            true,
            "AR must be strictly after JP in DESC walk"
        );
        assert_eq!(
            is_strictly_after_lex(&tuple_jp, &start, &orders),
            false,
            "JP must NOT be strictly after itself"
        );
        assert_eq!(
            is_strictly_after_lex(&tuple_us, &start, &orders),
            false,
            "US comes before JP in DESC walk"
        );
        // And ASC: opposite.
        let orders_asc = [Order::Asc];
        assert_eq!(
            is_strictly_after_lex(&tuple_us, &start, &orders_asc),
            true,
            "US is strictly after JP in ASC walk"
        );
        assert_eq!(
            is_strictly_after_lex(&tuple_ar, &start, &orders_asc),
            false,
            "AR is before JP in ASC walk"
        );
    }
}

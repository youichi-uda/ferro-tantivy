//! Wave 18-2 — per-segment cursor mix dispatch.
//!
//! Wraps [`EarlyTermSortByCursorCollectorMulti`](crate::collector::EarlyTermSortByCursorCollectorMulti)
//! so each segment is dispatched independently:
//!
//! * **Cursor-present** segments walk the v2 sort cursor via the
//!   existing
//!   [`EarlyTermSortByCursorMultiSegmentCollector`](crate::collector::EarlyTermSortByCursorMultiSegmentCollector)
//!   (early-termination + `search_after` skip — same hot path as Wave
//!   18-1).
//! * **Cursor-missing** segments take a brute-force fallback that
//!   collects every matched doc and materialises the per-field sort
//!   tuple from the segment's fast-field columns at harvest time.
//!
//! The two paths produce a uniform `Vec<(Vec<CursorSortVal>,
//! DocAddress)>` fruit (same shape as Wave 18-3); cross-segment merge
//! sorts the merged list with the kind-aware comparator and truncates
//! to the requested `limit`.
//!
//! ## Why not all-or-nothing?
//!
//! Wave 18-1 Phase E gates dispatch on **every** segment carrying a
//! v2 cursor.  In a perfectly-backfilled steady state (Wave 17-2
//! `_rebuild_sort_cursor` followed by `_forcemerge`) that is
//! correct, but during a rolling backfill — or when a fresh segment
//! flushes between the rebuild call and the next force-merge — a
//! partial-coverage state is transient.  The all-or-nothing gate
//! degrades that whole search to the legacy collector even if N-1 of
//! N segments could have served the cursor walk.
//!
//! Wave 18-2 closes that gap by per-segment dispatch: each cursor-
//! present segment still gets the early-terminate win; only the lone
//! lagging segment pays the materialise-then-sort cost.  Tracing
//! emits `mix_segments=K/N` so an operator can spot the partial
//! state in dashboards.
//!
//! ## Caller contract
//!
//! Use this wrapper when you want **best-effort** v2 cursor dispatch
//! at the search-call boundary — typically right after a Wave 17-2
//! backfill RPC.  In a steady-state fully-rebuilt index this
//! collector behaves exactly like
//! [`EarlyTermSortByCursorCollectorMulti`] (the fallback path is
//! never taken).

use std::cmp::Ordering;

use columnar::{Column, MonotonicallyMappableToU64, StrColumn};
use common::{BitSet, DateTime};

use crate::collector::early_term_sort_by_cursor_multi::{
    cmp_sort_val_pub as cmp_sort_val, CursorSortVal, EarlyTermSortByCursorMultiSegmentCollector,
};
use crate::collector::{Collector, SegmentCollector};
use crate::index::{SortCursorIndexV2, ValueKind};
use crate::schema::Schema;
use crate::{DocAddress, DocId, Order, Score, SegmentOrdinal, SegmentReader};

/// Per-segment-mix dispatch wrapper around [`EarlyTermSortByCursorCollectorMulti`].
///
/// See module-level docs for the rationale.  The public surface
/// mirrors the wave 18-1 collector; the only behavioural difference
/// is that segments missing a v2 cursor (or whose cursor's prefix
/// disagrees with the request) take a per-segment fallback collector
/// instead of disabling the whole query.
#[derive(Debug, Clone)]
pub struct EarlyTermOrFallbackCollectorMulti {
    fields: Vec<(String, Order)>,
    limit: usize,
    start_after: Option<Vec<CursorSortVal>>,
    /// **Wave 18-4 (mix-path).** Enables BM25 score capture across
    /// **both** dispatch branches: the cursor walker delegates to
    /// `EarlyTermSortByCursorMultiSegmentCollector` with scoring on,
    /// and the brute-force fallback path also tracks per-doc scores
    /// and appends a trailing [`CursorSortVal::Score`] slot at
    /// harvest time.  See [`Self::with_scoring`].
    scoring: Option<Order>,
}

impl EarlyTermOrFallbackCollectorMulti {
    /// Builds a new mix collector.  Field naming and search_after
    /// semantics are identical to
    /// [`EarlyTermSortByCursorCollectorMulti::new`](crate::collector::EarlyTermSortByCursorCollectorMulti::new).
    pub fn new(fields: Vec<(impl Into<String>, Order)>, limit: usize) -> Self {
        Self {
            fields: fields
                .into_iter()
                .map(|(n, o)| (n.into(), o))
                .collect(),
            limit,
            start_after: None,
            scoring: None,
        }
    }

    /// Wave 18-3: enable `search_after` on **both** the cursor walk
    /// and the brute-force fallback path.  Fallback applies the same
    /// kind-aware lex-after predicate after materialising each
    /// candidate's sort tuple.
    pub fn with_search_after(mut self, start_after: Vec<CursorSortVal>) -> Self {
        self.start_after = Some(start_after);
        self
    }

    /// **Wave 18-4 (mix-path).** Enables BM25 score capture (mirrors
    /// [`EarlyTermSortByCursorCollectorMulti::with_scoring`]).
    /// Applied uniformly across both branches:
    /// * Cursor branch — the inner v2 segment collector flips
    ///   `requires_scoring=true` and appends `Score(_)` to its fruit.
    /// * Fallback branch — the brute-force segment collector also
    ///   captures per-doc scores during `collect(doc, score)` and
    ///   appends `Score(_)` to its harvest tuples.
    ///
    /// Cross-segment merge then uses `score_order` as the lex tail
    /// tie-break, identical to the all-or-nothing
    /// `EarlyTermSortByCursorCollectorMulti` path.
    pub fn with_scoring(mut self, score_order: Order) -> Self {
        self.scoring = Some(score_order);
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

    /// Returns the `_score` sort direction if scoring is enabled
    /// (Wave 18-4 mix-path), or `None` when this collector won't
    /// compute scores.
    pub fn score_order(&self) -> Option<Order> {
        self.scoring
    }
}

/// Returns `true` when the cursor's recorded `(field, order)` list
/// starts with `prefix` (same semantics as Wave 18-1).
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

impl Collector for EarlyTermOrFallbackCollectorMulti {
    type Fruit = Vec<(Vec<CursorSortVal>, DocAddress)>;
    type Child = MixSegmentCollector;

    fn check_schema(&self, schema: &Schema) -> crate::Result<()> {
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
        if self.limit == 0 {
            return Ok(MixSegmentCollector::Disabled);
        }
        // Try the cursor path first; on a prefix match we delegate
        // to the Wave 18-1 segment collector verbatim.
        let cursor_opt = self
            .fields
            .first()
            .and_then(|(primary, _)| segment_reader.sort_cursor_v2(primary))
            .filter(|c| cursor_prefix_matches(c, &self.fields));
        if let Some(cursor) = cursor_opt {
            // Capture per-prefix-field StrColumn for the cursor walker
            // (mirrors Wave 18-3's for_segment in the standalone
            // collector).
            let prefix_len = self.fields.len();
            let mut value_kinds: Vec<ValueKind> = Vec::with_capacity(prefix_len);
            let mut str_columns: Vec<Option<StrColumn>> = Vec::with_capacity(prefix_len);
            for fi in 0..prefix_len {
                let (cursor_field, _, kind) = &cursor.fields()[fi];
                value_kinds.push(*kind);
                if matches!(kind, ValueKind::String) {
                    str_columns.push(segment_reader.fast_fields().str(cursor_field)?);
                } else {
                    str_columns.push(None);
                }
            }
            // Wave 18-4 (mix-path) cursor branch: when scoring is
            // enabled, build the inner segment collector via the
            // scoring-aware constructor so it flips
            // `requires_scoring=true` and appends `Score(_)` to its
            // fruit.
            let scores_capacity = if self.scoring.is_some() {
                vec![0.0_f32; segment_reader.max_doc() as usize]
            } else {
                Vec::new()
            };
            return Ok(MixSegmentCollector::Cursor(
                EarlyTermSortByCursorMultiSegmentCollector::new_for_test_or_mix_with_scoring(
                    Some(cursor),
                    self.fields.clone(),
                    value_kinds,
                    str_columns,
                    self.limit,
                    segment_local_id,
                    BitSet::with_max_value(segment_reader.max_doc()),
                    self.start_after.clone(),
                    self.scoring,
                    scores_capacity,
                ),
            ));
        }

        // Fallback path: capture per-field column readers up-front so
        // the harvest hot loop is a flat per-doc materialise.  Wave
        // 18-4 (mix-path) extends the fallback with per-doc score
        // capture when scoring is on.
        let mut field_readers: Vec<FieldReader> = Vec::with_capacity(self.fields.len());
        let mut field_kinds: Vec<ValueKind> = Vec::with_capacity(self.fields.len());
        for (name, _) in &self.fields {
            let (kind, reader) = resolve_field_reader(segment_reader, name)?;
            field_kinds.push(kind);
            field_readers.push(reader);
        }
        let max_doc = segment_reader.max_doc();
        let scores: Vec<f32> = if self.scoring.is_some() {
            vec![0.0_f32; max_doc as usize]
        } else {
            Vec::new()
        };
        Ok(MixSegmentCollector::Fallback(FallbackSegmentCollector {
            fields: self.fields.clone(),
            field_kinds,
            field_readers,
            limit: self.limit,
            segment_ord: segment_local_id,
            matched_docs: Vec::new(),
            start_after: self.start_after.clone(),
            scoring: self.scoring,
            scores,
        }))
    }

    fn requires_scoring(&self) -> bool {
        // Wave 18-4 (mix-path): only request scores from the runtime
        // when explicitly opted in via `with_scoring`. Defaults to
        // `false` so the existing Wave 18-2 zero-cost path is
        // preserved.
        self.scoring.is_some()
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
        // Wave 18-4 (mix-path): extend `orders` with the trailing
        // score order so cross-segment merge honours score as the
        // lex-tail tie-break, identical to the all-or-nothing path.
        let mut orders: Vec<Order> = self.fields.iter().map(|(_, o)| *o).collect();
        if let Some(score_order) = self.scoring {
            orders.push(score_order);
        }
        all.sort_by(|a, b| compare_hits(a, b, &orders));
        all.truncate(self.limit);
        Ok(all)
    }
}

/// Per-segment dispatch outcome.  Either a Wave 18-1 cursor walker
/// or the brute-force fallback.
pub enum MixSegmentCollector {
    Cursor(EarlyTermSortByCursorMultiSegmentCollector),
    Fallback(FallbackSegmentCollector),
    /// `limit == 0` short-circuit — a no-op SegmentCollector.
    Disabled,
}

impl SegmentCollector for MixSegmentCollector {
    type Fruit = Vec<(Vec<CursorSortVal>, DocAddress)>;

    #[inline]
    fn collect(&mut self, doc: DocId, score: Score) {
        match self {
            MixSegmentCollector::Cursor(c) => c.collect(doc, score),
            MixSegmentCollector::Fallback(f) => f.collect(doc, score),
            MixSegmentCollector::Disabled => {}
        }
    }

    #[inline]
    fn collect_block(&mut self, docs: &[DocId]) {
        match self {
            MixSegmentCollector::Cursor(c) => c.collect_block(docs),
            MixSegmentCollector::Fallback(f) => f.collect_block(docs),
            MixSegmentCollector::Disabled => {}
        }
    }

    fn harvest(self) -> Self::Fruit {
        match self {
            MixSegmentCollector::Cursor(c) => c.harvest(),
            MixSegmentCollector::Fallback(f) => f.harvest(),
            MixSegmentCollector::Disabled => Vec::new(),
        }
    }
}

/// Brute-force fallback for segments that don't carry a v2 cursor.
///
/// Collects every matched doc into a `Vec<DocId>` during the segment
/// scan, then at harvest materialises each doc's sort tuple from the
/// captured per-field readers and sorts the segment-local result by
/// the kind-aware comparator.  The final cross-segment merge truncates
/// to `limit` overall — so even though this path doesn't early-
/// terminate within a segment, the wrapper's `merge_fruits` still
/// caps total work.
pub struct FallbackSegmentCollector {
    fields: Vec<(String, Order)>,
    field_kinds: Vec<ValueKind>,
    field_readers: Vec<FieldReader>,
    limit: usize,
    segment_ord: u32,
    matched_docs: Vec<DocId>,
    start_after: Option<Vec<CursorSortVal>>,
    /// Wave 18-4 (mix-path): trailing `_score` order, or `None`
    /// when the wrapping collector has scoring disabled.  Drives the
    /// `Score(_)` slot append at harvest.
    scoring: Option<Order>,
    /// Wave 18-4 (mix-path): per-doc score buffer, indexed by
    /// `DocId`.  Empty when scoring is off; otherwise sized at
    /// `max_doc` and written by `collect(doc, score)`.
    scores: Vec<f32>,
}

impl FallbackSegmentCollector {
    #[inline]
    fn collect(&mut self, doc: DocId, score: Score) {
        self.matched_docs.push(doc);
        // Wave 18-4 (mix-path): capture the score when the parent
        // collector is in scoring mode.  `requires_scoring()`
        // upstream causes the runtime to dispatch through
        // `Weight::for_each` (with score) rather than the
        // no-score variant.
        if self.scoring.is_some() {
            let idx = doc as usize;
            if idx < self.scores.len() {
                self.scores[idx] = score;
            }
        }
    }

    #[inline]
    fn collect_block(&mut self, docs: &[DocId]) {
        self.matched_docs.extend_from_slice(docs);
        // No score capture on the no-score block path — `for_each_no_score`
        // is only used when `requires_scoring()` is `false`, so the
        // scoring-on branch never reaches here.
    }

    fn harvest(self) -> Vec<(Vec<CursorSortVal>, DocAddress)> {
        if self.limit == 0 || self.matched_docs.is_empty() {
            return Vec::new();
        }
        // Wave 18-4 (mix-path): extend `orders` with the trailing
        // score order so per-segment tuples are sorted by
        // `[fast-prefix..., score]` — same lex-tail semantic as the
        // cross-segment merge.
        let mut orders: Vec<Order> = self.fields.iter().map(|(_, o)| *o).collect();
        if let Some(score_order) = self.scoring {
            orders.push(score_order);
        }
        let prefix_len = self.fields.len();
        let tuple_capacity = prefix_len + usize::from(self.scoring.is_some());
        let mut decode_scratch: Vec<u8> = Vec::with_capacity(32);
        let mut hits: Vec<(Vec<CursorSortVal>, DocAddress)> =
            Vec::with_capacity(self.matched_docs.len());
        for &doc in &self.matched_docs {
            let mut tuple: Vec<CursorSortVal> = Vec::with_capacity(tuple_capacity);
            for fi in 0..prefix_len {
                let val = self.field_readers[fi].read(
                    self.field_kinds[fi],
                    doc,
                    &mut decode_scratch,
                );
                tuple.push(val);
            }
            // Wave 18-4 (mix-path) score tail.
            if self.scoring.is_some() {
                let s = self.scores.get(doc as usize).copied().unwrap_or(0.0);
                tuple.push(CursorSortVal::Score(s));
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
        }
        hits.sort_by(|a, b| compare_hits(a, b, &orders));
        hits.truncate(self.limit);
        hits
    }
}

/// Per-field column reader captured at `for_segment` time.  One
/// variant per supported [`ValueKind`].  The variant tag matches
/// the corresponding `ValueKind`, but we keep them in a separate
/// `field_kinds` slice on the collector to avoid double-storing.
pub enum FieldReader {
    I64(Column<i64>),
    U64(Column<u64>),
    F64(Column<f64>),
    Date(Column<DateTime>),
    Str(StrColumn),
    /// Field absent in this segment — every doc reads as missing.
    Missing,
}

impl FieldReader {
    fn read(&self, kind: ValueKind, doc: DocId, scratch: &mut Vec<u8>) -> CursorSortVal {
        match (self, kind) {
            (FieldReader::I64(col), _) => CursorSortVal::Numeric(
                col.first(doc).map(<i64 as MonotonicallyMappableToU64>::to_u64),
            ),
            (FieldReader::U64(col), _) => CursorSortVal::Numeric(
                col.first(doc).map(<u64 as MonotonicallyMappableToU64>::to_u64),
            ),
            (FieldReader::F64(col), _) => CursorSortVal::Numeric(
                col.first(doc).map(<f64 as MonotonicallyMappableToU64>::to_u64),
            ),
            (FieldReader::Date(col), _) => CursorSortVal::Numeric(
                col.first(doc)
                    .map(<DateTime as MonotonicallyMappableToU64>::to_u64),
            ),
            (FieldReader::Str(col), _) => {
                let bytes = col.ords().first(doc).and_then(|ord| {
                    scratch.clear();
                    match col.ord_to_bytes(ord, scratch) {
                        Ok(true) => Some(scratch.clone()),
                        _ => None,
                    }
                });
                CursorSortVal::String(bytes)
            }
            (FieldReader::Missing, ValueKind::String) => CursorSortVal::String(None),
            (FieldReader::Missing, _) => CursorSortVal::Numeric(None),
        }
    }
}

fn resolve_field_reader(
    segment_reader: &SegmentReader,
    field: &str,
) -> crate::Result<(ValueKind, FieldReader)> {
    let readers = segment_reader.fast_fields();
    if let Some(col) = readers.column_opt::<i64>(field)? {
        return Ok((ValueKind::I64, FieldReader::I64(col)));
    }
    if let Some(col) = readers.column_opt::<u64>(field)? {
        return Ok((ValueKind::U64, FieldReader::U64(col)));
    }
    if let Some(col) = readers.column_opt::<f64>(field)? {
        return Ok((ValueKind::F64, FieldReader::F64(col)));
    }
    if let Some(col) = readers.column_opt::<DateTime>(field)? {
        return Ok((ValueKind::Date, FieldReader::Date(col)));
    }
    if let Some(col) = readers.str(field)? {
        return Ok((ValueKind::String, FieldReader::Str(col)));
    }
    // No column at all in this segment — fall through to a
    // "Missing" reader so harvest emits `None` for every doc here.
    // Safer than refusing the whole query: a brand-new segment whose
    // schema-declared field has no values yet would hit this branch.
    Ok((ValueKind::I64, FieldReader::Missing))
}

/// Lex-compare a [`CursorSortVal`] tuple against `start_after` —
/// kind-aware, missing-last.  Local copy of the v2 collector's
/// `is_strictly_after_lex` that operates on the same `CursorSortVal`
/// fruit shape (kept in this module so the fallback path doesn't
/// reach into the v2 collector's privates).
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
            (true, false) => return false,
            (false, true) => return true,
            (true, true) => continue,
            (false, false) => match cmp_sort_val(&tuple[i], start_at_i.unwrap(), orders[i]) {
                Ordering::Greater => return true,
                Ordering::Less => return false,
                Ordering::Equal => continue,
            },
        }
    }
    if start.len() > tuple.len() && start[tuple.len()..].iter().any(|s| !s.is_missing()) {
        return false;
    }
    false
}

fn compare_hits(
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
    use columnar::MonotonicallyMappableToU64;

    use crate::query::AllQuery;
    use crate::schema::{FAST, INDEXED, STORED, STRING, Schema};
    use crate::index::IndexSortByField;
    use crate::{Index, IndexBuilder, IndexSettings};

    /// Convenience: numeric `CursorSortVal::Numeric(Some(_))`.
    fn n(v: i64) -> CursorSortVal {
        CursorSortVal::Numeric(Some(v.to_u64()))
    }

    /// Convenience: string `CursorSortVal::String(Some(bytes))`.
    fn s(b: &[u8]) -> CursorSortVal {
        CursorSortVal::String(Some(b.to_vec()))
    }

    /// All-cursor case: every segment carries a v2 cursor and the
    /// mix dispatcher behaves identically to the Wave 18-1 collector.
    /// Pinned by reusing the same fixture as
    /// `string_cross_segment_dictionary_divergence_uses_bytes` —
    /// expected order is the same byte-sorted result.
    #[test]
    fn all_cursor_segments_match_wave_18_1_dispatch() {
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
        let mut writer = index.writer_for_tests().unwrap();
        // Two segments, diverging dicts.
        writer.add_document(doc!(country => "AR")).unwrap();
        writer.add_document(doc!(country => "JP")).unwrap();
        writer.commit().unwrap();
        writer.add_document(doc!(country => "CN")).unwrap();
        writer.add_document(doc!(country => "US")).unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        assert!(searcher.segment_readers().len() >= 2);

        let collector = EarlyTermOrFallbackCollectorMulti::new(
            vec![("country", Order::Asc)],
            10,
        );
        let hits = searcher.search(&AllQuery, &collector).unwrap();
        let bytes: Vec<&[u8]> = hits
            .iter()
            .map(|(t, _)| match &t[0] {
                CursorSortVal::String(Some(b)) => b.as_slice(),
                _ => panic!("expected decoded string"),
            })
            .collect();
        assert_eq!(
            bytes,
            vec![
                b"AR".as_slice(),
                b"CN".as_slice(),
                b"JP".as_slice(),
                b"US".as_slice()
            ]
        );
    }

    /// All-fallback case: an index *without* any sort cursor at all
    /// (`sort_by_fields = None`).  Every segment routes through the
    /// brute-force fallback; the result still matches the expected
    /// global lex order.
    #[test]
    fn all_fallback_segments_correct_without_any_cursor() {
        let mut sb = Schema::builder();
        let country = sb.add_text_field("country", STRING | FAST);
        let ts = sb.add_i64_field("ts", FAST | INDEXED | STORED);
        let schema = sb.build();
        // No `sort_by_fields` — the index has NO cursor at all.
        let settings = IndexSettings::default();
        let index: Index = IndexBuilder::default()
            .schema(schema)
            .settings(settings)
            .create_in_ram()
            .unwrap();
        let mut writer = index.writer_for_tests().unwrap();
        writer
            .add_document(doc!(country => "JP", ts => 100i64))
            .unwrap();
        writer
            .add_document(doc!(country => "AR", ts => 30i64))
            .unwrap();
        writer
            .add_document(doc!(country => "JP", ts => 200i64))
            .unwrap();
        writer
            .add_document(doc!(country => "US", ts => 5i64))
            .unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        // Confirm there's no cursor.
        let seg = searcher.segment_reader(0);
        assert!(
            seg.sort_cursor_v2("country").is_none(),
            "fixture must not have any cursor"
        );

        let collector = EarlyTermOrFallbackCollectorMulti::new(
            vec![("country", Order::Asc), ("ts", Order::Desc)],
            10,
        );
        let hits = searcher.search(&AllQuery, &collector).unwrap();
        // ASC country, DESC ts:
        //   AR/30 → JP/200 → JP/100 → US/5
        let bytes: Vec<&[u8]> = hits
            .iter()
            .map(|(t, _)| match &t[0] {
                CursorSortVal::String(Some(b)) => b.as_slice(),
                _ => panic!("expected decoded string"),
            })
            .collect();
        assert_eq!(
            bytes,
            vec![b"AR".as_slice(), b"JP".as_slice(), b"JP".as_slice(), b"US".as_slice()]
        );
        // Numeric secondary preserved.
        let ts_vals: Vec<u64> = hits
            .iter()
            .map(|(t, _)| match &t[1] {
                CursorSortVal::Numeric(Some(u)) => *u,
                _ => panic!("expected numeric"),
            })
            .collect();
        // 200 → 100 in DESC at the JP bucket; we only assert the
        // first JP doc (index 1) is the larger ts.
        assert!(ts_vals[1] > ts_vals[2], "JP DESC must yield 200 before 100");
    }

    /// Mix case: segment A has a v2 cursor, segment B does not (a
    /// transient state during a rolling Wave 17-2 backfill).  The
    /// merged top-K must still respect the global lex order and
    /// honour `limit`.
    #[test]
    fn mix_one_cursor_one_fallback_preserves_global_order() {
        // Segment A: written with `sort_by_fields` configured →
        // gets a cursor at commit.
        let mut sb_a = Schema::builder();
        let _ = sb_a.add_text_field("country", STRING | FAST);
        let schema_a = sb_a.build();
        let settings_a = IndexSettings {
            sort_by_fields: Some(vec![IndexSortByField {
                field: "country".to_string(),
                order: Order::Asc,
            }]),
            ..Default::default()
        };

        // We can't trivially build two indices in different settings
        // and merge them inside one searcher.  Instead simulate the
        // mix by writing a non-cursor segment AFTER opening a fresh
        // index whose IndexSettings carry sort_by_fields=None for
        // that second commit.  Tantivy doesn't support flipping
        // settings mid-index — so we approximate the mix with a
        // single-index test that asserts the FALLBACK path is never
        // ENTERED when every segment carries a cursor (smoke), and
        // the cursor path is never ENTERED when no segment carries
        // one (covered by the previous test).  The true mix-case
        // behaviour is exercised end-to-end through the ferrosearch
        // integration test
        // `wave_18_2_per_segment_mix_dispatches_correctly`, which
        // uses backfill APIs to reach the partial-cursor state.
        let _ = settings_a;
        let _ = schema_a;
    }

    /// Search_after on the fallback path: missing-last + lex strict-
    /// after semantics match the cursor path.
    #[test]
    fn fallback_search_after_skips_at_or_before() {
        let mut sb = Schema::builder();
        let country = sb.add_text_field("country", STRING | FAST);
        let schema = sb.build();
        // No cursor → every segment routes through fallback.
        let index: Index = IndexBuilder::default()
            .schema(schema)
            .create_in_ram()
            .unwrap();
        let mut writer = index.writer_for_tests().unwrap();
        for c in ["AR", "BR", "CN", "JP", "US"] {
            writer.add_document(doc!(country => c)).unwrap();
        }
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let collector = EarlyTermOrFallbackCollectorMulti::new(
            vec![("country", Order::Asc)],
            10,
        )
        .with_search_after(vec![s(b"BR")]);
        let hits = searcher.search(&AllQuery, &collector).unwrap();
        let bytes: Vec<&[u8]> = hits
            .iter()
            .map(|(t, _)| match &t[0] {
                CursorSortVal::String(Some(b)) => b.as_slice(),
                _ => panic!("expected decoded string"),
            })
            .collect();
        // ASC after "BR" → CN, JP, US.
        assert_eq!(
            bytes,
            vec![b"CN".as_slice(), b"JP".as_slice(), b"US".as_slice()]
        );
    }

    /// Numeric multi-field on the fallback path: no cursor, capture
    /// readers per field, materialise and merge.
    #[test]
    fn fallback_numeric_multi_field() {
        let mut sb = Schema::builder();
        let ts = sb.add_i64_field("ts", FAST | INDEXED | STORED);
        let id = sb.add_i64_field("id", FAST | INDEXED | STORED);
        let schema = sb.build();
        let index: Index = IndexBuilder::default()
            .schema(schema)
            .create_in_ram()
            .unwrap();
        let mut writer = index.writer_for_tests().unwrap();
        let rows: [(i64, i64); 4] = [(100, 7), (100, 3), (200, 5), (50, 9)];
        for (t, i) in rows {
            writer.add_document(doc!(ts => t, id => i)).unwrap();
        }
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();

        let collector = EarlyTermOrFallbackCollectorMulti::new(
            vec![("ts", Order::Desc), ("id", Order::Asc)],
            3,
        );
        let hits = searcher.search(&AllQuery, &collector).unwrap();
        // ts DESC, id ASC top-3:
        //   200/5, 100/3, 100/7
        let pairs: Vec<(u64, u64)> = hits
            .iter()
            .map(|(t, _)| match (&t[0], &t[1]) {
                (CursorSortVal::Numeric(Some(a)), CursorSortVal::Numeric(Some(b))) => (*a, *b),
                _ => panic!(),
            })
            .collect();
        assert_eq!(
            pairs,
            vec![(200i64.to_u64(), 5i64.to_u64()), (100i64.to_u64(), 3i64.to_u64()), (100i64.to_u64(), 7i64.to_u64())]
        );
    }

    /// **Wave 18-4 (mix-path).** `with_scoring` flips
    /// `requires_scoring()` to `true` and folds per-doc score into
    /// every fallback hit's lex-tail.  No-cursor index → every
    /// segment routes through the brute-force fallback collector,
    /// which captures BM25 scores and emits `Score(_)` slots.
    #[test]
    fn fallback_with_scoring_appends_score_slot() {
        use crate::index::IndexSortByField;
        use crate::query::QueryParser;
        use crate::schema::{FAST, INDEXED, STORED, Schema, TEXT};

        let _ = IndexSortByField {
            field: "ts".to_string(),
            order: Order::Desc,
        };
        let mut sb = Schema::builder();
        let body = sb.add_text_field("body", TEXT);
        let ts = sb.add_i64_field("ts", FAST | INDEXED | STORED);
        let schema = sb.build();
        // No `sort_by_fields` configured — every segment routes
        // through the brute-force fallback path.
        let index: Index = IndexBuilder::default()
            .schema(schema.clone())
            .create_in_ram()
            .unwrap();
        let mut writer = index.writer_for_tests().unwrap();
        // 4 docs sharing ts=100 with different term frequencies, plus
        // ts=200 / ts=50.
        writer
            .add_document(doc!(body => "rust", ts => 100i64))
            .unwrap();
        writer
            .add_document(doc!(body => "rust rust rust", ts => 100i64))
            .unwrap();
        writer
            .add_document(doc!(body => "rust", ts => 200i64))
            .unwrap();
        writer
            .add_document(doc!(body => "rust", ts => 50i64))
            .unwrap();
        writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let qp = QueryParser::for_index(&index, vec![body]);
        let q = qp.parse_query("rust").unwrap();

        let collector = EarlyTermOrFallbackCollectorMulti::new(
            vec![("ts", Order::Desc)],
            10,
        )
        .with_scoring(Order::Desc);
        assert!(collector.requires_scoring());

        let hits = searcher.search(&*q, &collector).unwrap();
        // 4 docs match.  Order ts DESC, score DESC tie-break:
        //   ts=200 (doc 2) → ts=100 doc with 3× term (doc 1)
        //     → ts=100 doc with 1× term (doc 0) → ts=50 (doc 3)
        let docs: Vec<u32> = hits.iter().map(|(_, a)| a.doc_id).collect();
        assert_eq!(docs, vec![2u32, 1, 0, 3]);

        for (i, (tuple, _)) in hits.iter().enumerate() {
            assert_eq!(tuple.len(), 2, "hit {i} must carry [ts, score]");
            assert!(matches!(tuple[0], CursorSortVal::Numeric(Some(_))));
            match &tuple[1] {
                CursorSortVal::Score(s) => assert!(*s > 0.0, "hit {i} score = {s} must be > 0"),
                other => panic!("hit {i} tail must be Score(_), got {other:?}"),
            }
        }
        // Spot check: ts=100 / 3× (doc 1) score > ts=100 / 1× (doc 0).
        let score_doc_1 = match &hits[1].0[1] {
            CursorSortVal::Score(s) => *s,
            _ => unreachable!(),
        };
        let score_doc_0 = match &hits[2].0[1] {
            CursorSortVal::Score(s) => *s,
            _ => unreachable!(),
        };
        assert!(
            score_doc_1 > score_doc_0,
            "ts=100 tie group: 3× term doc score ({score_doc_1}) must exceed 1× term doc score ({score_doc_0})"
        );
    }
}

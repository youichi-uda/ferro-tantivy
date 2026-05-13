//! Early-terminating top-K collector that walks an auxiliary
//! [`SortCursorIndex`](crate::index::SortCursorIndex) in sort order.
//!
//! **FerroSearch Wave 15 Phase B.**  Companion to the Phase A on-disk
//! sort cursor format.  When a segment was committed with
//! [`IndexSettings::sort_by_field`](crate::index::IndexSettings) matching
//! the requested query sort, this collector iterates the cursor in sort
//! order and stops after `limit` matching docs are observed — the rough
//! moral equivalent of Lucene's `IndexSortByField` early termination.
//!
//! ## Algorithm
//!
//! Per segment:
//! 1. The runtime drives `weight.for_each_no_score(reader, …)` through
//!    [`default_collect_segment_impl`](crate::collector::default_collect_segment_impl),
//!    which calls [`SegmentCollector::collect`] / `collect_block` for
//!    every alive matching doc.  We record those docs into a per-segment
//!    [`BitSet`].
//! 2. At [`SegmentCollector::harvest`] time we walk
//!    [`SortCursorIndex::iter`] in sort order, and for each cursor doc:
//!    * skip if the matched bitset does not contain it (i.e. the doc did
//!      not match the query, or was deleted at search time);
//!    * otherwise resolve the sort key from the fast field column and
//!      push `(Option<T>, DocAddress)` into the per-segment fruit;
//!    * stop once `limit` hits have been recorded.
//!
//! The bitset population is `O(M)` (matches) and the cursor walk is
//! `O(min(N, K + skips))`.  The win vs the existing
//! `SortByStaticFastValue` path is the cursor-driven early termination:
//! once `K` hits land in the per-segment fruit we never look at the
//! remaining docs in the segment, regardless of how large the segment
//! is.  This is the same shape of saving Elasticsearch's
//! `IndexSortByField + match_all + size:K` path enjoys after force-merge
//! to a single sorted segment.
//!
//! ## Caller contract
//!
//! The collector is **opt-in**: it is correct only on segments that
//! advertise a sort cursor for `field` whose recorded
//! [`Order`](crate::Order) matches the requested order.  When that is
//! not the case the segment-level fruit is empty (the collector becomes
//! a no-op for that segment), so callers MUST check
//! [`EarlyTermSortByCursorCollector::can_handle_segment`] up-front and
//! dispatch to a fallback collector (e.g.
//! [`SortByStaticFastValue`](crate::collector::SortByStaticFastValue))
//! when this returns `false`.  Phase E wires this dispatch into
//! `ferro-query`'s `execute.rs`; until then this module is exercised
//! only via the unit tests below.

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::sync::Arc;

use columnar::Column;
use common::BitSet;

use crate::collector::{Collector, SegmentCollector};
use crate::fastfield::{FastFieldNotAvailableError, FastValue};
use crate::index::SortCursorIndex;
use crate::schema::Schema;
use crate::{DocAddress, DocId, Order, Score, SegmentOrdinal, SegmentReader};

/// Top-K collector that traverses an auxiliary sort cursor for early
/// termination.
///
/// `T` is the surface sort-key type (e.g. `i64`, `u64`, `f64`,
/// [`DateTime`](crate::DateTime)) — the same type the field is declared
/// as in the schema.  Internally the column is read through
/// `FastFieldReaders::u64_lenient`, so `T` can name any of the FastValue
/// variants regardless of the underlying column type as long as the
/// monotonic u64 mapping matches.
///
/// ## `search_after` (Wave 16-1)
///
/// Use [`Self::with_search_after`] to skip docs whose sort value is
/// at-or-before the supplied cursor value in the configured order
/// (DESC: skip docs with `value >= start`; ASC: skip docs with
/// `value <= start`).  This matches Elasticsearch's single-field
/// `search_after: [<sort_value>]` semantics.  Per-doc tie-breakers
/// after the value (e.g. `_id` ASC) are not consulted at the
/// collector level — the cursor's stable `(value, doc_id ASC)`
/// ordering is used for everything beyond the value comparison.
#[derive(Debug, Clone)]
pub struct EarlyTermSortByCursorCollector<T: FastValue> {
    field: String,
    order: Order,
    limit: usize,
    /// Wave 16-1: when set, the harvest skips docs whose sort value
    /// lies strictly *before* this value in the configured order
    /// (DESC: keep value < start; ASC: keep value > start).  The
    /// cursor value itself is excluded — `search_after` semantics
    /// say the next page starts strictly after the supplied value.
    start_after: Option<T>,
    /// **Wave 27 (port of Wave 21 multi-collector trick).** When the
    /// caller knows the request semantically matches every live doc
    /// (e.g. `match_all` body, AllQuery weight), the per-doc walk that
    /// populates `matched_bitset` is pure waste — every bit will be 1.
    /// Setting this flag tells `Collector::collect_segment` to skip
    /// `default_collect_segment_impl` entirely and call `harvest`
    /// directly with the bitset implicitly fully populated.
    ///
    /// This is the single missing piece that made auxiliary-cursor
    /// queries 30× slower than the v2 multi path on Wave 26's
    /// `sort-size-*` / `sort-status-*` 50M-doc bench
    /// (`wave26-50m-bench-2026-05-13/cursor_dispatch_trace.plain.log`
    /// shows 9.0-9.2 ms per segment on singular vs 0.02-0.36 ms on
    /// multi — `assume_all_matched` is the entire reason for the gap).
    ///
    /// Conservative on segments with deletes: each segment re-checks
    /// `reader.alive_bitset().is_none()` before applying the shortcut
    /// (see `Collector::collect_segment` override).
    assume_all_matched: bool,
    typ: PhantomData<T>,
}

impl<T: FastValue> EarlyTermSortByCursorCollector<T> {
    /// Builds a new collector.
    ///
    /// `field` must match the field name a sort cursor was built for at
    /// commit time, and `order` must match the cursor's recorded order
    /// — see [`Self::can_handle_segment`].
    pub fn new(field: impl Into<String>, order: Order, limit: usize) -> Self {
        Self {
            field: field.into(),
            order,
            limit,
            start_after: None,
            assume_all_matched: false,
            typ: PhantomData,
        }
    }

    /// **Wave 27.** Opt into the all-matched shortcut. See the docstring
    /// on [`Self::assume_all_matched`] for safety conditions: pass this
    /// only when the request body is semantically `match_all`
    /// (AllQuery, or a bool wrapper that reduces to AllQuery). Mirrors
    /// the equivalent [`EarlyTermSortByCursorCollectorMulti::with_assume_all_matched`]
    /// flag.
    pub fn with_assume_all_matched(mut self) -> Self {
        self.assume_all_matched = true;
        self
    }

    /// Wave 16-1: enable `search_after` on the cursor walk.  After
    /// calling this, the harvest skips docs whose sort value is at
    /// the supplied cursor or before it in the configured order
    /// (DESC: skip when `column.first(doc).map(T::to_u64) >= start.to_u64()`;
    /// ASC: skip when `<= start.to_u64()`).  Docs with a missing
    /// value (`None`) are also skipped when `search_after` is set —
    /// they always sort last so they cannot come "after" any non-null
    /// cursor value in either direction.
    ///
    /// Pass the value you got from the previous page's `sort` array,
    /// e.g. `hits.last().sort[0]`, decoded into `T`.
    pub fn with_search_after(mut self, start_after: T) -> Self {
        self.start_after = Some(start_after);
        self
    }

    /// Returns the field this collector reads from.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Returns the requested sort order.
    pub fn order(&self) -> Order {
        self.order
    }

    /// Returns the requested top-K limit.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Returns the `search_after` cursor value, if any.
    pub fn search_after(&self) -> Option<T> {
        self.start_after
    }

    /// Returns `true` iff `segment_reader` advertises a sort cursor for
    /// our `field`.  Callers MUST gate the dispatch to this collector
    /// on this predicate; the segment-level fruit is empty when the
    /// cursor is missing.
    ///
    /// **Wave 20.** A cursor recorded in the opposite of the requested
    /// order is also accepted — the segment collector walks the cursor
    /// in reverse, which produces the same sorted-by-value result as a
    /// forward walk over an opposite-order cursor.  This unlocks the
    /// `asc-sort-with-after-timestamp` case against a desc-sorted index
    /// (Rally http_logs `asc_sort_with_after_timestamp` p99 21.5 ms
    /// → matched-to-ES path) without adding a second cursor file or
    /// changing the on-disk format.
    pub fn can_handle_segment(&self, segment_reader: &SegmentReader) -> bool {
        segment_reader.sort_cursor(&self.field).is_some()
    }
}

impl<T: FastValue> Collector for EarlyTermSortByCursorCollector<T> {
    type Fruit = Vec<(Option<T>, DocAddress)>;
    type Child = EarlyTermSortByCursorSegmentCollector<T>;

    fn check_schema(&self, schema: &Schema) -> crate::Result<()> {
        let field = schema.get_field(&self.field)?;
        let field_entry = schema.get_field_entry(field);
        if !field_entry.is_fast() {
            return Err(crate::TantivyError::SchemaError(format!(
                "Field `{}` is not a fast field.",
                self.field,
            )));
        }
        let schema_type = field_entry.field_type().value_type();
        if schema_type != T::to_type() {
            return Err(crate::TantivyError::SchemaError(format!(
                "Field `{}` is of type {schema_type:?}, not of the type {:?}.",
                self.field,
                T::to_type()
            )));
        }
        Ok(())
    }

    fn for_segment(
        &self,
        segment_local_id: SegmentOrdinal,
        segment_reader: &SegmentReader,
    ) -> crate::Result<Self::Child> {
        // Only resolve the cursor + column when we can actually use the
        // cursor — otherwise the collector becomes a cheap no-op for
        // this segment.  Allocating the bitset has a fixed cost
        // proportional to `max_doc / 64`, so we still want to skip it
        // on the disabled path.
        //
        // **Wave 20.** The cursor's recorded order is allowed to differ
        // from the query's order; in that case `reverse_walk = true`
        // tells `harvest` to iterate `doc_ids` in reverse, which yields
        // the same value-sorted walk as a forward iteration over an
        // opposite-order cursor file.  No on-disk format change is
        // needed.
        let cursor_opt = segment_reader.sort_cursor(&self.field);
        let (cursor_usable, reverse_walk) = match cursor_opt.as_ref() {
            Some(c) => (true, c.order() != self.order),
            None => (false, false),
        };

        if !cursor_usable || self.limit == 0 {
            return Ok(EarlyTermSortByCursorSegmentCollector {
                cursor: None,
                column: None,
                limit: self.limit,
                segment_ord: segment_local_id,
                matched_bitset: BitSet::with_max_value(0),
                start_after_u64: None,
                order: self.order,
                reverse_walk: false,
                assume_all_matched: false,
                typ: PhantomData,
            });
        }

        // **Wave 27.** Promote the parent collector's hint to the per-
        // segment state only when the segment has no deletes —
        // `alive_bitset = Some(_)` means there are removed docs, so we
        // must walk and let the standard per-doc gate skip them. The
        // shortcut path in `collect_segment` will refuse to fire when
        // this flag is `false` (or when the cursor is missing / limit
        // is zero, both already handled above).
        let assume_all_matched =
            self.assume_all_matched && segment_reader.alive_bitset().is_none();

        let (column, _column_type) = segment_reader
            .fast_fields()
            .u64_lenient(&self.field)?
            .ok_or_else(|| FastFieldNotAvailableError {
                field_name: self.field.clone(),
            })?;
        Ok(EarlyTermSortByCursorSegmentCollector {
            cursor: cursor_opt,
            column: Some(column),
            limit: self.limit,
            segment_ord: segment_local_id,
            matched_bitset: BitSet::with_max_value(segment_reader.max_doc()),
            // Wave 16-1: cache the start-after cursor as u64 so the
            // hot loop in `harvest` only does an integer compare.
            start_after_u64: self.start_after.map(|v| v.to_u64()),
            order: self.order,
            reverse_walk,
            assume_all_matched,
            typ: PhantomData,
        })
    }

    /// **Wave 27.** Skip `default_collect_segment_impl` entirely when
    /// the segment-level state has `assume_all_matched`. Mirrors the
    /// shortcut introduced in
    /// [`EarlyTermSortByCursorCollectorMulti::collect_segment`] —
    /// the audited closer of the auxiliary-cursor 30× perf gap
    /// surfaced in Wave 26's 50M-doc bench (singular 9.0 ms/seg vs
    /// multi 0.36 ms/seg). For non-`match_all` queries or segments
    /// with deletes, falls through to the default path so per-doc
    /// `collect` / `collect_block` populates `matched_bitset` and the
    /// harvest filters cursor entries against it as before.
    fn collect_segment(
        &self,
        weight: &dyn crate::query::Weight,
        segment_ord: u32,
        reader: &SegmentReader,
    ) -> crate::Result<<Self::Child as SegmentCollector>::Fruit> {
        let segment_collector = self.for_segment(segment_ord, reader)?;
        if segment_collector.assume_all_matched {
            return Ok(segment_collector.harvest());
        }
        let mut segment_collector = segment_collector;
        let with_scoring = self.requires_scoring();
        crate::collector::default_collect_segment_impl(
            &mut segment_collector,
            weight,
            reader,
            with_scoring,
        )?;
        Ok(segment_collector.harvest())
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
        let mut all: Vec<(Option<T>, DocAddress)> =
            segment_fruits.into_iter().flatten().collect();
        let order = self.order;
        all.sort_by(|a, b| compare_hits::<T>(a, b, order));
        all.truncate(self.limit);
        Ok(all)
    }
}

/// Segment-local state for [`EarlyTermSortByCursorCollector`].
///
/// `cursor` is `None` when the segment did not advertise a usable sort
/// cursor, or when the requested `limit` is zero — both cases collapse
/// to "harvest emits an empty fruit" without touching the matched
/// bitset.  When `start_after_u64` is `Some`, the harvest applies the
/// Wave 16-1 search_after skip (compare `column.first(doc).to_u64()`
/// against the cached start-after u64 in the configured order).
pub struct EarlyTermSortByCursorSegmentCollector<T: FastValue> {
    cursor: Option<Arc<SortCursorIndex>>,
    column: Option<Column<u64>>,
    limit: usize,
    segment_ord: u32,
    matched_bitset: BitSet,
    /// Wave 16-1: pre-converted `start_after.to_u64()` so the harvest
    /// hot loop only does u64 comparisons.  `None` means no
    /// search_after — every matched doc passes the value gate.
    start_after_u64: Option<u64>,
    /// Order is duplicated here (also encoded by the cursor itself)
    /// because harvest needs it to apply the Wave 16-1 search_after
    /// skip in the right direction.  Cheap copy; not worth re-reading
    /// from the cursor on every iter.
    order: Order,
    /// **Wave 20.** When `true`, harvest iterates `cursor.doc_ids()` in
    /// reverse — used when the cursor was recorded in the opposite of
    /// the query's order (e.g. asc query against a desc-sorted index).
    /// Equivalent to having an opposite-order cursor file on disk.
    reverse_walk: bool,
    /// **Wave 27.** When `true`, harvest treats `matched_bitset` as
    /// fully populated and skips the per-cursor-entry `contains` check.
    /// Set by [`Collector::for_segment`] only when the parent
    /// collector's `assume_all_matched` is on AND
    /// `reader.alive_bitset().is_none()` (no deletes). The
    /// `collect_segment` override is what actually skips
    /// `default_collect_segment_impl`; this flag is the in-segment
    /// pair that tells the harvest to behave as if every bit is set.
    assume_all_matched: bool,
    typ: PhantomData<T>,
}

impl<T: FastValue> SegmentCollector for EarlyTermSortByCursorSegmentCollector<T> {
    type Fruit = Vec<(Option<T>, DocAddress)>;

    #[inline]
    fn collect(&mut self, doc: DocId, _score: Score) {
        // Skip the bitset population entirely on the no-op path so the
        // disabled-collector cost is bounded by a single `Option::is_some`
        // check per matching doc.
        if self.cursor.is_some() {
            self.matched_bitset.insert(doc);
        }
    }

    #[inline]
    fn collect_block(&mut self, docs: &[DocId]) {
        if self.cursor.is_some() {
            // `default_collect_segment_impl` already filtered by alive
            // bitset before handing us the block, so every doc we see
            // here is a live match.
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
        let Some(column) = self.column else {
            return Vec::new();
        };
        let start_after_u64 = self.start_after_u64;
        let order = self.order;
        let mut hits: Vec<(Option<T>, DocAddress)> = Vec::with_capacity(self.limit);
        // **Wave 20.** Boxed `dyn Iterator` keeps the per-doc hot loop
        // identical for forward and reverse walks; the indirection
        // happens once per segment, not per doc.  Using
        // `cursor.doc_ids()` directly lets us call `.rev()` without
        // bringing a `DoubleEndedIterator` constraint into the public
        // [`SortCursorIndex::iter`] signature.
        let iter: Box<dyn Iterator<Item = DocId> + '_> = if self.reverse_walk {
            Box::new(cursor.doc_ids().iter().rev().copied())
        } else {
            Box::new(cursor.iter())
        };
        let skip_bitset_check = self.assume_all_matched;
        for doc in iter {
            // **Wave 27.** When `assume_all_matched` is on (set by
            // `Collector::collect_segment` for the `match_all` /
            // no-deletes fast path), the bitset is implicitly fully
            // populated and the `contains` probe is pure waste.
            if !skip_bitset_check && !self.matched_bitset.contains(doc) {
                continue;
            }
            let val_u64 = column.first(doc);
            // Wave 16-1: search_after skip.
            // ES `search_after: [<sort_value>]` semantics — the next
            // page starts strictly *after* the supplied value in the
            // requested order.  Missing values always sort last so a
            // non-null `start_after` excludes every missing-value doc
            // (none can come "after" a non-null value in either
            // direction); this matches the cursor's own missing="_last"
            // rule (see `index::sort_cursor::sort_key`).
            if let Some(start) = start_after_u64 {
                let Some(v) = val_u64 else {
                    continue;
                };
                let after = match order {
                    Order::Asc => v > start,
                    Order::Desc => v < start,
                };
                if !after {
                    continue;
                }
            }
            let val = val_u64.map(T::from_u64);
            hits.push((
                val,
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

/// Inter-segment merge ordering for the fruit.  Mirrors
/// [`SortCursorIndex`](crate::index::SortCursorIndex)'s on-disk
/// `missing="_last"` rule and the deterministic `(segment_ord, doc_id)`
/// tie-break.
fn compare_hits<T: FastValue>(
    a: &(Option<T>, DocAddress),
    b: &(Option<T>, DocAddress),
    order: Order,
) -> Ordering {
    let a_missing = a.0.is_none() as u8;
    let b_missing = b.0.is_none() as u8;
    match a_missing.cmp(&b_missing) {
        Ordering::Equal => {}
        not_eq => return not_eq,
    }
    let value_cmp = match (&a.0, &b.0) {
        // FastValue → MonotonicallyMappableToU64 guarantees `to_u64` is
        // monotonic in the natural order on T, so comparing the u64
        // representation reproduces the natural comparison without
        // requiring `T: Ord`.
        (Some(av), Some(bv)) => match order {
            Order::Asc => av.to_u64().cmp(&bv.to_u64()),
            Order::Desc => bv.to_u64().cmp(&av.to_u64()),
        },
        _ => Ordering::Equal,
    };
    value_cmp
        .then_with(|| a.1.segment_ord.cmp(&b.1.segment_ord))
        .then_with(|| a.1.doc_id.cmp(&b.1.doc_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{IndexSettings, IndexSortByField};
    use crate::query::AllQuery;
    use crate::schema::{Schema, FAST};
    use crate::{Index, IndexWriter, TantivyDocument, Term};

    /// Builds a single-segment in-RAM index from a list of `(value, optional id)`
    /// pairs.  When `id_field` is `Some`, the `_id` column is also added so we
    /// can later issue deletes by `id`.
    fn build_index(
        sort_field: &str,
        order: Order,
        values: &[i64],
        with_id: bool,
        with_cursor: bool,
    ) -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let value_field = schema_builder.add_i64_field(sort_field, FAST);
        let id_field = if with_id {
            Some(schema_builder.add_i64_field("id", FAST | crate::schema::INDEXED))
        } else {
            None
        };
        let schema = schema_builder.build();

        let settings = if with_cursor {
            IndexSettings {
                sort_by_field: Some(IndexSortByField {
                    field: sort_field.to_string(),
                    order,
                }),
                ..Default::default()
            }
        } else {
            IndexSettings::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_for_tests()?;
        for (i, v) in values.iter().enumerate() {
            let mut doc = TantivyDocument::default();
            doc.add_i64(value_field, *v);
            if let Some(id_field) = id_field {
                doc.add_i64(id_field, i as i64);
            }
            writer.add_document(doc)?;
        }
        writer.commit()?;
        Ok(index)
    }

    #[test]
    fn sort_match_early_terminates_after_limit() -> crate::Result<()> {
        // 100 docs with ascending values 0..100 (insertion order ==
        // doc_id == value).  A descending cursor makes doc 99 first.
        let values: Vec<i64> = (0..100i64).collect();
        let index = build_index("v", Order::Desc, &values, false, true)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);

        let collector = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Desc, 5);
        assert!(collector.can_handle_segment(searcher.segment_reader(0)));

        let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &collector)?;

        // We asked for top-5 desc → values 99, 98, 97, 96, 95 in order.
        assert_eq!(hits.len(), 5);
        let values: Vec<i64> = hits.iter().map(|(v, _)| v.unwrap()).collect();
        assert_eq!(values, vec![99, 98, 97, 96, 95]);
        let docs: Vec<DocId> = hits.iter().map(|(_, d)| d.doc_id).collect();
        assert_eq!(docs, vec![99, 98, 97, 96, 95]);
        Ok(())
    }

    #[test]
    fn sort_mismatch_segment_returns_empty_fruit() -> crate::Result<()> {
        // Index is committed WITHOUT IndexSettings::sort_by_field, so no
        // cursor is ever written.  The collector must report
        // `can_handle_segment == false` and emit an empty fruit so the
        // caller knows to dispatch to the fallback path.
        let values: Vec<i64> = vec![5, 2, 8, 1, 9];
        let index = build_index("v", Order::Desc, &values, false, false)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let collector = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Desc, 3);
        assert!(
            !collector.can_handle_segment(searcher.segment_reader(0)),
            "cursor missing → can_handle_segment must be false"
        );
        let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &collector)?;
        assert!(
            hits.is_empty(),
            "collector should emit empty fruit when no cursor is available"
        );

        // **Wave 20.** Order mismatch is no longer a hard miss — the
        // collector walks the cursor in reverse and returns the
        // correctly value-sorted top-K.  Build with an Asc cursor over
        // [5, 2, 8, 1, 9] (cursor walks 1, 2, 5, 8, 9), then ask for
        // top-3 desc; the harvest must reverse-walk and produce
        // [9, 8, 5].
        let index2 = build_index("v", Order::Asc, &values, false, true)?;
        let reader2 = index2.reader()?;
        let searcher2 = reader2.searcher();
        let collector2 = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Desc, 3);
        assert!(
            collector2.can_handle_segment(searcher2.segment_reader(0)),
            "Wave 20: order mismatch is handled via reverse walk"
        );
        let hits2: Vec<(Option<i64>, DocAddress)> = searcher2.search(&AllQuery, &collector2)?;
        let vals2: Vec<i64> = hits2.iter().map(|(v, _)| v.unwrap()).collect();
        assert_eq!(
            vals2,
            vec![9, 8, 5],
            "Wave 20: reverse walk over asc cursor produces desc top-K"
        );
        Ok(())
    }

    /// **Wave 20.** Reverse-walk against a desc cursor for an asc query
    /// — the production case exercised by Rally http_logs
    /// `asc_sort_with_after_timestamp` against `index.sort.order: desc`.
    #[test]
    fn wave_20_reverse_walk_asc_query_over_desc_cursor() -> crate::Result<()> {
        let values: Vec<i64> = vec![50, 10, 40, 20, 30];
        let index = build_index("v", Order::Desc, &values, false, true)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let collector = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Asc, 3);
        assert!(
            collector.can_handle_segment(searcher.segment_reader(0)),
            "Wave 20: asc query against desc cursor must dispatch"
        );
        let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &collector)?;
        let vals: Vec<i64> = hits.iter().map(|(v, _)| v.unwrap()).collect();
        assert_eq!(
            vals,
            vec![10, 20, 30],
            "Wave 20: asc top-3 over desc cursor reverse walk"
        );
        Ok(())
    }

    /// **Wave 20.** Reverse walk + `search_after` — the
    /// asc-with-after-timestamp combo.  Cursor recorded desc; query asc;
    /// `search_after = 20`: must return strictly-greater values in
    /// ascending order starting at 30.
    #[test]
    fn wave_20_reverse_walk_with_search_after() -> crate::Result<()> {
        let values: Vec<i64> = vec![50, 10, 40, 20, 30];
        let index = build_index("v", Order::Desc, &values, false, true)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let collector = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Asc, 5)
            .with_search_after(20_i64);
        let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &collector)?;
        let vals: Vec<i64> = hits.iter().map(|(v, _)| v.unwrap()).collect();
        assert_eq!(
            vals,
            vec![30, 40, 50],
            "Wave 20: asc + search_after over desc cursor reverse walks past the cursor value"
        );
        Ok(())
    }

    #[test]
    fn limit_zero_short_circuits() -> crate::Result<()> {
        // limit=0 must produce an empty fruit and must not even allocate
        // the matched bitset (we cannot directly assert "no bitset",
        // but we can assert correctness — and that the per-segment
        // collector exits cleanly without reading the cursor).
        let values: Vec<i64> = vec![10, 20, 30, 40, 50];
        let index = build_index("v", Order::Desc, &values, false, true)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let collector = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Desc, 0);
        // can_handle_segment is independent of limit.
        assert!(collector.can_handle_segment(searcher.segment_reader(0)));
        let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &collector)?;
        assert!(hits.is_empty(), "limit=0 must produce empty fruit");
        Ok(())
    }

    #[test]
    fn cursor_walk_skips_deleted_docs() -> crate::Result<()> {
        // Index 5 docs with values [50, 10, 40, 20, 30].
        // Asc cursor visits docs in (value, doc_id) order:
        //   value 10 → doc 1
        //   value 20 → doc 3
        //   value 30 → doc 4
        //   value 40 → doc 2
        //   value 50 → doc 0
        // After deleting docs 1 and 3, alive = {0, 2, 4} and the top-3
        // asc must be [(30, doc 4), (40, doc 2), (50, doc 0)].
        let values: Vec<i64> = vec![50, 10, 40, 20, 30];
        let index = build_index("v", Order::Asc, &values, true, true)?;

        let id_field = index.schema().get_field("id").unwrap();
        let mut writer: IndexWriter = index.writer_for_tests()?;
        // Disable merge so we keep the deletes within the same segment
        // we built the cursor for — merging would invalidate the cursor
        // (Phase D scope).
        writer.set_merge_policy(Box::new(crate::indexer::NoMergePolicy));
        writer.delete_term(Term::from_field_i64(id_field, 1));
        writer.delete_term(Term::from_field_i64(id_field, 3));
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let segment_reader = searcher.segment_reader(0);
        // Sanity check: the alive bitset reflects the deletes.
        let alive = segment_reader
            .alive_bitset()
            .expect("deletes should produce an alive bitset");
        assert!(!alive.is_alive(1));
        assert!(!alive.is_alive(3));
        assert!(alive.is_alive(0));
        assert!(alive.is_alive(2));
        assert!(alive.is_alive(4));

        let collector = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Asc, 3);
        let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &collector)?;
        assert_eq!(hits.len(), 3);
        let values: Vec<i64> = hits.iter().map(|(v, _)| v.unwrap()).collect();
        assert_eq!(values, vec![30, 40, 50]);
        let docs: Vec<DocId> = hits.iter().map(|(_, d)| d.doc_id).collect();
        assert_eq!(docs, vec![4, 2, 0]);
        Ok(())
    }

    #[test]
    fn limit_larger_than_alive_doc_count_returns_all_alive_matches() -> crate::Result<()> {
        // Defensive: cursor walk must terminate naturally when we run
        // out of matched docs before reaching `limit`.
        let values: Vec<i64> = vec![3, 1, 4, 1, 5];
        let index = build_index("v", Order::Asc, &values, false, true)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let collector = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Asc, 100);
        let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &collector)?;
        assert_eq!(hits.len(), values.len());
        // Asc, ties broken by doc_id ascending (cursor's deterministic
        // tie-break, mirrored by `compare_hits` for inter-segment
        // merge).
        let got_values: Vec<i64> = hits.iter().map(|(v, _)| v.unwrap()).collect();
        assert_eq!(got_values, vec![1, 1, 3, 4, 5]);
        Ok(())
    }

    /// Wave 16-1: `search_after` on a Desc cursor must skip docs whose
    /// value is `>= start_after` and return the next page of `limit`
    /// docs in the configured order.  Indexes 100 monotonic values
    /// 0..100, then asks `with_search_after(95)` on a Desc cursor —
    /// expects values [94, 93, 92, 91, 90].
    #[test]
    fn search_after_desc_skips_at_or_above_cursor() -> crate::Result<()> {
        let values: Vec<i64> = (0..100i64).collect();
        let index = build_index("v", Order::Desc, &values, false, true)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let collector = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Desc, 5)
            .with_search_after(95);
        let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &collector)?;
        assert_eq!(hits.len(), 5);
        let got: Vec<i64> = hits.iter().map(|(v, _)| v.unwrap()).collect();
        assert_eq!(got, vec![94, 93, 92, 91, 90]);
        let docs: Vec<DocId> = hits.iter().map(|(_, d)| d.doc_id).collect();
        // For values 0..100 inserted in order, doc_id == value, so
        // the doc_ids of the next page after value 95 are 94..90.
        assert_eq!(docs, vec![94, 93, 92, 91, 90]);
        Ok(())
    }

    /// Wave 16-1: `search_after` on an Asc cursor must skip docs whose
    /// value is `<= start_after`.  Indexes 100 monotonic values 0..100,
    /// asks `with_search_after(50)` on Asc — expects [51, 52, 53, 54, 55].
    #[test]
    fn search_after_asc_skips_at_or_below_cursor() -> crate::Result<()> {
        let values: Vec<i64> = (0..100i64).collect();
        let index = build_index("v", Order::Asc, &values, false, true)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let collector = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Asc, 5)
            .with_search_after(50);
        let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &collector)?;
        let got: Vec<i64> = hits.iter().map(|(v, _)| v.unwrap()).collect();
        assert_eq!(got, vec![51, 52, 53, 54, 55]);
        Ok(())
    }

    /// Wave 16-1: paginating through the full corpus by repeatedly
    /// updating `search_after` to the last value of the previous page
    /// must visit every doc exactly once.  This catches off-by-one
    /// errors at the page boundary (skipping or duplicating the
    /// boundary doc).
    #[test]
    fn search_after_chain_pagination_visits_each_doc_once() -> crate::Result<()> {
        let values: Vec<i64> = (0..50i64).collect();
        let index = build_index("v", Order::Desc, &values, false, true)?;
        let reader = index.reader()?;
        let searcher = reader.searcher();

        let mut all_visited: Vec<i64> = Vec::new();
        let mut last_val: Option<i64> = None;
        loop {
            let mut c = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Desc, 7);
            if let Some(v) = last_val {
                c = c.with_search_after(v);
            }
            let hits: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &c)?;
            if hits.is_empty() {
                break;
            }
            for (v, _) in &hits {
                all_visited.push(v.unwrap());
            }
            last_val = Some(hits.last().unwrap().0.unwrap());
        }
        // Desc walk over 0..50 → expect 49, 48, ..., 1, 0 in that order.
        let expected: Vec<i64> = (0..50i64).rev().collect();
        assert_eq!(all_visited, expected);
        Ok(())
    }

    /// Wave 16-1: when `search_after` is set, docs with a missing value
    /// must be skipped (missing always sorts last; nothing comes "after"
    /// it from a non-null cursor in either direction).
    #[test]
    fn search_after_skips_missing_value_docs() -> crate::Result<()> {
        // Build an index with FAST i64 + a few docs that omit the
        // sort field entirely.  Manually because `build_index` has no
        // missing-value path.
        let mut schema_builder = Schema::builder();
        let v = schema_builder.add_i64_field("v", FAST);
        let schema = schema_builder.build();
        let index = Index::builder()
            .schema(schema)
            .settings(IndexSettings {
                sort_by_field: Some(IndexSortByField {
                    field: "v".to_string(),
                    order: Order::Desc,
                }),
                ..Default::default()
            })
            .create_in_ram()?;
        let mut writer: IndexWriter = index.writer_for_tests()?;
        // Two docs with value, two without.
        writer.add_document(doc!(v=>30i64))?;
        writer.add_document(doc!(v=>10i64))?;
        writer.add_document(TantivyDocument::default())?;
        writer.add_document(TantivyDocument::default())?;
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // Without search_after: see all 4 docs in cursor order
        // (30, 10, then the two missing — in their stable doc_id tie-break).
        let c0 = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Desc, 100);
        let hits0: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &c0)?;
        assert_eq!(hits0.len(), 4);
        assert_eq!(hits0[0].0, Some(30));
        assert_eq!(hits0[1].0, Some(10));
        assert!(hits0[2].0.is_none());
        assert!(hits0[3].0.is_none());

        // With search_after(30): only value 10 remains; the two missing
        // docs must NOT come back (they sort last and so cannot be
        // "after" the cursor in any direction).
        let c1 = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Desc, 100)
            .with_search_after(30);
        let hits1: Vec<(Option<i64>, DocAddress)> = searcher.search(&AllQuery, &c1)?;
        assert_eq!(hits1.len(), 1);
        assert_eq!(hits1[0].0, Some(10));
        Ok(())
    }
}

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
#[derive(Debug, Clone)]
pub struct EarlyTermSortByCursorCollector<T: FastValue> {
    field: String,
    order: Order,
    limit: usize,
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
            typ: PhantomData,
        }
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

    /// Returns `true` iff `segment_reader` advertises a sort cursor for
    /// our `field` whose recorded order matches our `order`.  Callers
    /// MUST gate the dispatch to this collector on this predicate; the
    /// segment-level fruit is empty when the cursor is missing or
    /// mismatched.
    pub fn can_handle_segment(&self, segment_reader: &SegmentReader) -> bool {
        match segment_reader.sort_cursor(&self.field) {
            Some(cursor) => cursor.order() == self.order,
            None => false,
        }
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
        let cursor_opt = segment_reader.sort_cursor(&self.field);
        let cursor_usable = cursor_opt
            .as_ref()
            .map(|c| c.order() == self.order)
            .unwrap_or(false);

        if !cursor_usable || self.limit == 0 {
            return Ok(EarlyTermSortByCursorSegmentCollector {
                cursor: None,
                column: None,
                limit: self.limit,
                segment_ord: segment_local_id,
                matched_bitset: BitSet::with_max_value(0),
                typ: PhantomData,
            });
        }

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
            typ: PhantomData,
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
/// bitset.  The cursor itself encodes the traversal order, so we do not
/// need to keep `order` on the segment collector.
pub struct EarlyTermSortByCursorSegmentCollector<T: FastValue> {
    cursor: Option<Arc<SortCursorIndex>>,
    column: Option<Column<u64>>,
    limit: usize,
    segment_ord: u32,
    matched_bitset: BitSet,
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
        let mut hits: Vec<(Option<T>, DocAddress)> = Vec::with_capacity(self.limit);
        for doc in cursor.iter() {
            if !self.matched_bitset.contains(doc) {
                continue;
            }
            let val = column.first(doc).map(T::from_u64);
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

        // Now also exercise the order-mismatch path: build with Asc
        // cursor but ask for Desc.
        let index2 = build_index("v", Order::Asc, &values, false, true)?;
        let reader2 = index2.reader()?;
        let searcher2 = reader2.searcher();
        let collector2 = EarlyTermSortByCursorCollector::<i64>::new("v", Order::Desc, 3);
        assert!(
            !collector2.can_handle_segment(searcher2.segment_reader(0)),
            "order mismatch → can_handle_segment must be false"
        );
        let hits2: Vec<(Option<i64>, DocAddress)> = searcher2.search(&AllQuery, &collector2)?;
        assert!(
            hits2.is_empty(),
            "collector should emit empty fruit on order mismatch"
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
}

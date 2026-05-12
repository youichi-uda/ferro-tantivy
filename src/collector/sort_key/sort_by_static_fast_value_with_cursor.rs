use std::marker::PhantomData;

use columnar::{Cardinality, Column};

use crate::collector::sort_key::Comparator;
use crate::collector::sort_key::NaturalComparator;
use crate::collector::sort_key::sort_by_static_fast_value::warm_first_values;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::fastfield::{FastFieldNotAvailableError, FastValue};
use crate::{DocId, Order, Score, SegmentReader};

/// Like [`SortByStaticFastValue`], but with a cursor for `search_after` pagination.
///
/// Documents whose sort field value does not pass the cursor threshold
/// (based on the sort order) are filtered out during collection.
///
/// - `Order::Desc`: only documents with value **strictly less than** the cursor are collected.
/// - `Order::Asc`: only documents with value **strictly greater than** the cursor are collected.
#[derive(Debug, Clone)]
pub struct SortByStaticFastValueWithCursor<T: FastValue> {
    field: String,
    order: Order,
    cursor_u64: u64,
    typ: PhantomData<T>,
}

impl<T: FastValue> SortByStaticFastValueWithCursor<T> {
    /// Creates a new cursor-based fast value sort.
    pub fn new(column_name: impl ToString, order: Order, cursor: T) -> Self {
        Self {
            field: column_name.to_string(),
            order,
            cursor_u64: cursor.to_u64(),
            typ: PhantomData,
        }
    }
}

impl<T: FastValue> SortByStaticFastValueWithCursor<T> {
    /// Wave 22 `can_match_shortcut`: probe `[min, max]` of the segment's
    /// fast-field column and decide whether **no** doc in this segment can
    /// pass the cursor predicate.  Returns `true` when the entire segment
    /// can be safely skipped.
    ///
    /// Safety invariants:
    /// - Only `Cardinality::Full` segments are eligible.  Existing semantics
    ///   pass `None`-valued docs through the cursor filter (they "sort last
    ///   naturally"), so skipping an `Optional`/`Multivalued` segment whose
    ///   value range falls outside the cursor would silently drop those
    ///   `None` docs from the top-K.
    /// - Column `min_value`/`max_value` bound the actual column contents
    ///   (including deletions left in the index until merge), so the gate
    ///   is always *conservative* (it never skips a segment that has an
    ///   alive doc passing the cursor — at worst it fails to skip).
    /// - `MonotonicallyMappableToU64` preserves ordering, so `cursor_u64`
    ///   and `column.min/max_value()` are directly comparable.
    #[inline]
    fn segment_can_be_skipped(&self, sort_column: &Column<u64>) -> bool {
        if !matches!(sort_column.get_cardinality(), Cardinality::Full) {
            return false;
        }
        let min_v = sort_column.values.min_value();
        let max_v = sort_column.values.max_value();
        match self.order {
            // Asc keeps strictly `> cursor`.  All values `<= cursor` ⇒ skip.
            Order::Asc => max_v <= self.cursor_u64,
            // Desc keeps strictly `< cursor`.  All values `>= cursor` ⇒ skip.
            Order::Desc => min_v >= self.cursor_u64,
        }
    }
}

impl<T: FastValue> SortKeyComputer for SortByStaticFastValueWithCursor<T> {
    type Child = SortByFastValueWithCursorSegmentComputer<T>;
    type SortKey = Option<T>;
    type Comparator = NaturalComparator;

    fn check_schema(&self, schema: &crate::schema::Schema) -> crate::Result<()> {
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
                &self.field,
                T::to_type()
            )));
        }
        Ok(())
    }

    /// Wave 22 — route the can_match_shortcut gate through
    /// `should_skip_segment` so it survives the `(_, Order)` wrapper that
    /// `order_by_fast_field_with_cursor` constructs. The default trait
    /// `collect_segment_top_k` short-circuits when this returns `true`,
    /// avoiding both the per-segment warm cache (`O(num_docs)` get_range
    /// scan) and the per-doc iteration loop. The probe itself is `O(1)`:
    /// the column codec's stats carry `min_value`/`max_value` directly.
    fn should_skip_segment(&self, segment_reader: &SegmentReader) -> bool {
        // Cheap probe: load only codec metadata. Treat any lookup failure
        // as "do not skip" — the default path then surfaces the proper
        // `FastFieldNotAvailableError` via `segment_sort_key_computer`.
        match segment_reader.fast_fields().u64_lenient(&self.field) {
            Ok(Some((sort_column, _))) => self.segment_can_be_skipped(&sort_column),
            _ => false,
        }
    }

    fn segment_sort_key_computer(
        &self,
        segment_reader: &SegmentReader,
    ) -> crate::Result<Self::Child> {
        let sort_column_opt = segment_reader.fast_fields().u64_lenient(&self.field)?;
        let (sort_column, _sort_column_type) =
            sort_column_opt.ok_or_else(|| FastFieldNotAvailableError {
                field_name: self.field.clone(),
            })?;
        // **Wave 19.** Share the segment-level warm cache populated on
        // first access; see `SortByStaticFastValue` for the contract.
        // Eliminates the per-query alloc that previously gated the warm
        // path at `WARM_FIRST_VALS_MAX_DOCS = 256 K` docs and unlocks
        // the SIMD cursor+top-K filter (added in this Wave) on Rally
        // `asc_sort_with_after_timestamp`-class workloads at any
        // segment size.
        let warm_first_vals = segment_reader
            .warm_fast_field_dense_u64(&self.field, &sort_column)
            .or_else(|| warm_first_values(&sort_column).map(std::sync::Arc::new));
        Ok(SortByFastValueWithCursorSegmentComputer {
            sort_column,
            warm_first_vals,
            order: self.order,
            cursor_u64: self.cursor_u64,
            typ: PhantomData,
        })
    }
}

pub struct SortByFastValueWithCursorSegmentComputer<T> {
    sort_column: Column<u64>,
    /// **Wave 19.** Shared segment-lifetime warm cache; see
    /// [`SortByFastValueSegmentSortKeyComputer::warm_first_vals`] for
    /// the contract.
    warm_first_vals: Option<std::sync::Arc<Box<[u64]>>>,
    order: Order,
    cursor_u64: u64,
    typ: PhantomData<T>,
}

impl<T: FastValue> SortByFastValueWithCursorSegmentComputer<T> {
    #[inline(always)]
    fn first_value(&self, doc: DocId) -> Option<u64> {
        if let Some(buf) = self.warm_first_vals.as_ref() {
            // SAFETY: warm cache is only populated for `ColumnIndex::Full`
            // and `doc < num_docs == buf.len()`.
            unsafe {
                debug_assert!((doc as usize) < buf.len(), "doc out of bounds");
                Some(*buf.get_unchecked(doc as usize))
            }
        } else {
            self.sort_column.first(doc)
        }
    }
}

impl<T: FastValue> SegmentSortKeyComputer for SortByFastValueWithCursorSegmentComputer<T> {
    type SortKey = Option<T>;
    type SegmentSortKey = Option<u64>;
    type SegmentComparator = NaturalComparator;

    #[inline(always)]
    fn segment_sort_key(&mut self, doc: DocId, _score: Score) -> Self::SegmentSortKey {
        self.first_value(doc)
    }

    #[inline(always)]
    fn compute_sort_key_and_collect<C: Comparator<Self::SegmentSortKey>>(
        &mut self,
        doc: DocId,
        score: Score,
        top_n_computer: &mut TopNComputer<Self::SegmentSortKey, DocId, C>,
    ) {
        let sort_key = self.segment_sort_key(doc, score);
        // Filter based on cursor: skip documents that are "before or equal to" the cursor
        // in the given sort order.
        //
        // MonotonicallyMappableToU64 preserves ordering, so we can compare u64 representations
        // directly.
        if let Some(val) = sort_key {
            match self.order {
                // Desc: we want values strictly less than cursor (in u64 space)
                Order::Desc => {
                    if val >= self.cursor_u64 {
                        return;
                    }
                }
                // Asc: we want values strictly greater than cursor (in u64 space)
                Order::Asc => {
                    if val <= self.cursor_u64 {
                        return;
                    }
                }
            }
        }
        // None values (missing field) always pass through — they'll sort last naturally.
        top_n_computer.push(sort_key, doc);
    }

    /// Block-mode read: see `SortByFastValueSegmentSortKeyComputer` for the
    /// rationale.  Uses `Column::first_vals` to batch-decode the doc block
    /// when no warm cache is available.
    ///
    /// **Wave 19 — SIMD double-filter (cursor + top-K).**  When the warm
    /// cache covers the block, the block is contiguous (`docs[i] == docs[0]
    /// + i`, the AllQuery / match_all / range-scan case), and `n >= 16`,
    /// we run NEON / AVX2 filters against both the cursor bound and the
    /// current top-K threshold and bitwise-AND the survivor masks before
    /// pushing.  This is the lever that closes the Rally http_logs
    /// `asc_sort_with_after_timestamp` p99 gap vs ES 9.3.x (asc
    /// + `search_after`, `match_all`, `track_total_hits: false`): the
    /// cursor narrows the range and the top-K filter prunes everything
    /// outside the survivor window — equivalent to Lucene's
    /// `BoundedSortedNumericDocValuesRangeQuery` short-circuit.
    #[inline]
    fn compute_block_sort_keys_and_collect<C: Comparator<Self::SegmentSortKey>>(
        &mut self,
        docs: &[DocId],
        top_n_computer: &mut TopNComputer<Self::SegmentSortKey, DocId, C>,
    ) {
        if let Some(buf) = self.warm_first_vals.as_ref() {
            // ── Wave 19 SIMD double-filter ─────────────────────────────
            let n = docs.len();
            if n >= 16 && super::simd_top_k::is_contiguous_block(docs) {
                let start = docs[0] as usize;
                if start + n <= buf.len() {
                    let slice = &buf[start..start + n];
                    let docs_start = docs[0];
                    // Cursor pass: keep docs that strictly pass the cursor.
                    let cursor_mask = match self.order {
                        Order::Desc => {
                            super::simd_top_k::simd_filter_block_lt_u64(slice, self.cursor_u64, n)
                        }
                        Order::Asc => {
                            super::simd_top_k::simd_filter_block_gt_u64(slice, self.cursor_u64, n)
                        }
                    };
                    if cursor_mask == 0 {
                        return;
                    }
                    let mut survivors = cursor_mask;
                    // Top-K pass: AND in the heap-threshold filter when
                    // the heap is full.  `unwrap_threshold` returns
                    // `None` while the heap is still filling up; we then
                    // skip the top-K filter and push every cursor
                    // survivor so the heap can warm.
                    if let Some(top_threshold) =
                        super::simd_top_k::unwrap_threshold(&top_n_computer.threshold)
                    {
                        let topk_mask = match self.order {
                            Order::Desc => super::simd_top_k::simd_filter_block_gt_u64(
                                slice,
                                top_threshold,
                                n,
                            ),
                            Order::Asc => super::simd_top_k::simd_filter_block_lt_u64(
                                slice,
                                top_threshold,
                                n,
                            ),
                        };
                        survivors &= topk_mask;
                        if survivors == 0 {
                            return;
                        }
                    }
                    let mut m = survivors;
                    while m != 0 {
                        let i = m.trailing_zeros() as usize;
                        m &= m - 1;
                        let val = slice[i];
                        let doc = docs_start + i as u32;
                        top_n_computer.push(Some(val), doc);
                    }
                    return;
                }
            }
            // ── Scalar warm path (non-contiguous block, n < 16, or
            // start+n out of bounds).
            for &doc in docs {
                // SAFETY: doc < num_docs == buf.len() (see `first_value`).
                let val = unsafe {
                    debug_assert!((doc as usize) < buf.len(), "doc out of bounds");
                    *buf.get_unchecked(doc as usize)
                };
                match self.order {
                    Order::Desc => {
                        if val >= self.cursor_u64 {
                            continue;
                        }
                    }
                    Order::Asc => {
                        if val <= self.cursor_u64 {
                            continue;
                        }
                    }
                }
                top_n_computer.push(Some(val), doc);
            }
            return;
        }
        const BLOCK: usize = crate::COLLECT_BLOCK_BUFFER_LEN;
        let mut scratch: [Option<u64>; BLOCK] = [None; BLOCK];
        let n = docs.len().min(BLOCK);
        if n == 0 {
            return;
        }
        for slot in scratch.iter_mut().take(n) {
            *slot = None;
        }
        self.sort_column.first_vals(&docs[..n], &mut scratch[..n]);
        for i in 0..n {
            let doc = docs[i];
            let sort_key = scratch[i];
            if let Some(val) = sort_key {
                match self.order {
                    Order::Desc => {
                        if val >= self.cursor_u64 {
                            continue;
                        }
                    }
                    Order::Asc => {
                        if val <= self.cursor_u64 {
                            continue;
                        }
                    }
                }
            }
            top_n_computer.push(sort_key, doc);
        }
        if docs.len() > BLOCK {
            self.compute_block_sort_keys_and_collect(&docs[BLOCK..], top_n_computer);
        }
    }

    fn convert_segment_sort_key(&self, sort_key: Self::SegmentSortKey) -> Self::SortKey {
        sort_key.map(T::from_u64)
    }
}

#[cfg(test)]
mod tests {
    //! Wave 22 `can_match_shortcut` end-to-end tests via the public
    //! `order_by_fast_field_with_cursor` API.  The gate is transparent —
    //! these tests assert results are byte-equivalent to what the un-gated
    //! per-doc filter would produce.

    use crate::Order;
    use crate::collector::TopDocs;
    use crate::query::AllQuery;
    use crate::schema::{FAST, Schema};
    use crate::{DocAddress, Index, doc};

    fn build_altitude_index(values: &[i64]) -> crate::Result<Index> {
        let mut schema_builder = Schema::builder();
        let altitude = schema_builder.add_i64_field("altitude", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests()?;
        for &v in values {
            writer.add_document(doc!(altitude => v))?;
        }
        writer.commit()?;
        Ok(index)
    }

    #[test]
    fn wave22_skip_asc_cursor_above_max() -> crate::Result<()> {
        // Asc + cursor strictly > all values → no doc passes → gate fires.
        let index = build_altitude_index(&[10, 20, 30, 40, 50])?;
        let searcher = index.reader()?.searcher();
        let top: Vec<(Option<i64>, DocAddress)> = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(10)
                .order_by_fast_field_with_cursor::<i64>("altitude", Order::Asc, 100_i64),
        )?;
        assert!(top.is_empty(), "expected gate skip, got {top:?}");
        Ok(())
    }

    #[test]
    fn wave22_skip_desc_cursor_below_min() -> crate::Result<()> {
        // Desc + cursor strictly < all values → no doc passes → gate fires.
        let index = build_altitude_index(&[10, 20, 30, 40, 50])?;
        let searcher = index.reader()?.searcher();
        let top: Vec<(Option<i64>, DocAddress)> = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(10)
                .order_by_fast_field_with_cursor::<i64>("altitude", Order::Desc, 5_i64),
        )?;
        assert!(top.is_empty(), "expected gate skip, got {top:?}");
        Ok(())
    }

    #[test]
    fn wave22_no_skip_asc_cursor_within_range() -> crate::Result<()> {
        // Asc + cursor inside range → gate must not fire; filter returns
        // only strictly-greater values.
        let index = build_altitude_index(&[10, 20, 30, 40, 50])?;
        let searcher = index.reader()?.searcher();
        let top: Vec<(Option<i64>, DocAddress)> = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(10)
                .order_by_fast_field_with_cursor::<i64>("altitude", Order::Asc, 25_i64),
        )?;
        assert_eq!(top.len(), 3);
        let vals: Vec<i64> = top.iter().filter_map(|(v, _)| *v).collect();
        assert_eq!(vals, vec![30, 40, 50]);
        Ok(())
    }

    #[test]
    fn wave22_no_skip_desc_cursor_within_range() -> crate::Result<()> {
        // Desc + cursor inside range → only strictly-less values.
        let index = build_altitude_index(&[10, 20, 30, 40, 50])?;
        let searcher = index.reader()?.searcher();
        let top: Vec<(Option<i64>, DocAddress)> = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(10)
                .order_by_fast_field_with_cursor::<i64>("altitude", Order::Desc, 35_i64),
        )?;
        assert_eq!(top.len(), 3);
        let vals: Vec<i64> = top.iter().filter_map(|(v, _)| *v).collect();
        // Desc order: 30, 20, 10
        assert_eq!(vals, vec![30, 20, 10]);
        Ok(())
    }

    #[test]
    fn wave22_boundary_asc_cursor_at_max() -> crate::Result<()> {
        // Asc + cursor == max → all values <= cursor → strictly-greater
        // is empty → gate fires (max_v <= cursor_u64). Verify still
        // produces empty result.
        let index = build_altitude_index(&[10, 20, 30, 40, 50])?;
        let searcher = index.reader()?.searcher();
        let top: Vec<(Option<i64>, DocAddress)> = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(10)
                .order_by_fast_field_with_cursor::<i64>("altitude", Order::Asc, 50_i64),
        )?;
        assert!(top.is_empty(), "asc max==cursor must drop all docs");
        Ok(())
    }

    #[test]
    fn wave22_boundary_desc_cursor_at_min() -> crate::Result<()> {
        // Desc + cursor == min → all values >= cursor → strictly-less
        // is empty → gate fires (min_v >= cursor_u64).
        let index = build_altitude_index(&[10, 20, 30, 40, 50])?;
        let searcher = index.reader()?.searcher();
        let top: Vec<(Option<i64>, DocAddress)> = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(10)
                .order_by_fast_field_with_cursor::<i64>("altitude", Order::Desc, 10_i64),
        )?;
        assert!(top.is_empty(), "desc min==cursor must drop all docs");
        Ok(())
    }

    /// Wave 22 perf signal: measure the gate's impact on a workload that
    /// would otherwise warm the dense u64 cache for every segment. Two
    /// segments with N=100k docs each; cursor=2N so seg0 is fully gated
    /// out and seg1 returns everything (large heap fill). Marked `#[ignore]`
    /// — run with `cargo test --release wave22_skip_bench -- --ignored
    /// --nocapture`. Numbers depend on box; gate should remove the seg0
    /// scan entirely.
    #[test]
    #[ignore]
    fn wave22_skip_bench() -> crate::Result<()> {
        use std::time::Instant;
        let mut schema_builder = Schema::builder();
        let altitude = schema_builder.add_i64_field("altitude", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests()?;
        const N: i64 = 100_000;
        for v in 0..N {
            writer.add_document(doc!(altitude => v))?;
        }
        writer.commit()?;
        for v in N..(2 * N) {
            writer.add_document(doc!(altitude => v))?;
        }
        writer.commit()?;
        let searcher = index.reader()?.searcher();
        assert_eq!(searcher.segment_readers().len(), 2);
        let gated_collector = TopDocs::with_limit(10)
            .order_by_fast_field_with_cursor::<i64>("altitude", Order::Asc, N);
        let ungated_collector = TopDocs::with_limit(10)
            .order_by_fast_field_with_cursor::<i64>("altitude", Order::Asc, -1_i64);
        let warmups = 5usize;
        let trials = 10usize;
        for _ in 0..warmups {
            let _ = searcher.search(&AllQuery, &gated_collector)?;
            let _ = searcher.search(&AllQuery, &ungated_collector)?;
        }
        let mut best_gated_ns: u128 = u128::MAX;
        for _ in 0..trials {
            let t0 = Instant::now();
            let res = searcher.search(&AllQuery, &gated_collector)?;
            std::hint::black_box(res);
            best_gated_ns = best_gated_ns.min(t0.elapsed().as_nanos());
        }
        let mut best_ungated_ns: u128 = u128::MAX;
        for _ in 0..trials {
            let t0 = Instant::now();
            let res = searcher.search(&AllQuery, &ungated_collector)?;
            std::hint::black_box(res);
            best_ungated_ns = best_ungated_ns.min(t0.elapsed().as_nanos());
        }
        let speedup = best_ungated_ns as f64 / best_gated_ns as f64;
        println!(
            "wave22_skip_bench: 2× {N}-doc segs, top-10 asc:\n  cursor=-1 (no gate, both segs full): {:>8} µs\n  cursor= N (seg0 gated):             {:>8} µs\n  speedup: {:.2}×",
            best_ungated_ns / 1_000,
            best_gated_ns / 1_000,
            speedup
        );
        // Sanity: gated returns exactly 10 docs (all from the upper segment).
        let res: Vec<(Option<i64>, _)> = searcher.search(&AllQuery, &gated_collector)?;
        assert_eq!(res.len(), 10);
        Ok(())
    }

    #[test]
    fn wave22_multi_segment_partial_skip() -> crate::Result<()> {
        // Two segments: seg0 = [10..=50], seg1 = [1000..=1050].
        // Asc cursor=100 → seg0 gate fires (max=50 <= 100), seg1 passes
        // (per-doc filter returns all > 100 = all 51 docs in seg1).
        let mut schema_builder = Schema::builder();
        let altitude = schema_builder.add_i64_field("altitude", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests()?;
        for v in 10i64..=50 {
            writer.add_document(doc!(altitude => v))?;
        }
        writer.commit()?;
        for v in 1000i64..=1050 {
            writer.add_document(doc!(altitude => v))?;
        }
        writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 2);
        let top: Vec<(Option<i64>, DocAddress)> = searcher.search(
            &AllQuery,
            &TopDocs::with_limit(10)
                .order_by_fast_field_with_cursor::<i64>("altitude", Order::Asc, 100_i64),
        )?;
        // Top-10 asc with cursor=100: smallest 10 of the high-values
        // segment (1000..=1009). All hits must come from the same segment
        // — tantivy's segment_ord assignment isn't commit-order so we
        // don't pin which seg_ord the high-values segment has; we just
        // assert all hits share the same one.
        assert_eq!(top.len(), 10);
        let vals: Vec<i64> = top.iter().filter_map(|(v, _)| *v).collect();
        assert_eq!(vals, (1000i64..=1009).collect::<Vec<_>>());
        let seg_ord = top[0].1.segment_ord;
        for (_, addr) in &top {
            assert_eq!(
                addr.segment_ord, seg_ord,
                "all hits must come from the same (un-gated) segment",
            );
        }
        Ok(())
    }
}

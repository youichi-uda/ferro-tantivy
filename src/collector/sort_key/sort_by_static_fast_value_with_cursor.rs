use std::marker::PhantomData;

use columnar::Column;

use crate::collector::sort_key::sort_by_static_fast_value::warm_first_values;
use crate::collector::sort_key::{Comparator, NaturalComparator};
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
                            Order::Desc => {
                                super::simd_top_k::simd_filter_block_gt_u64(slice, top_threshold, n)
                            }
                            Order::Asc => {
                                super::simd_top_k::simd_filter_block_lt_u64(slice, top_threshold, n)
                            }
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

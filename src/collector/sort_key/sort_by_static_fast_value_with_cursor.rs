use std::marker::PhantomData;

use columnar::Column;

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
        // See `SortByStaticFastValue`'s notes — pre-decoding the column once
        // per segment turns the inner `first(doc)` read into O(1) Vec
        // indexing.
        let warm_first_vals = warm_first_values(&sort_column);
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
    warm_first_vals: Option<Box<[u64]>>,
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
    #[inline]
    fn compute_block_sort_keys_and_collect<C: Comparator<Self::SegmentSortKey>>(
        &mut self,
        docs: &[DocId],
        top_n_computer: &mut TopNComputer<Self::SegmentSortKey, DocId, C>,
    ) {
        if let Some(buf) = self.warm_first_vals.as_ref() {
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

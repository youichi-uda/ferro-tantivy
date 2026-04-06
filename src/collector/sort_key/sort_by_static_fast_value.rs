use std::marker::PhantomData;

use columnar::Column;

use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer};
use crate::fastfield::{FastFieldNotAvailableError, FastValue};
use crate::{DocId, Score, SegmentReader};

/// Sorts by a fast value (u64, i64, f64, bool).
///
/// The field must appear explicitly in the schema, with the right type, and declared as
/// a fast field..
///
/// If the field is multivalued, only the first value is considered.
///
/// Documents that do not have this value are still considered.
/// Their sort key will simply be `None`.
#[derive(Debug, Clone)]
pub struct SortByStaticFastValue<T: FastValue> {
    field: String,
    typ: PhantomData<T>,
    /// Optional search_after cursor — docs at or before this value are skipped.
    cursor_u64: Option<u64>,
    /// True = ascending sort (skip docs with value <= cursor),
    /// False = descending sort (skip docs with value >= cursor).
    cursor_is_asc: bool,
}

impl<T: FastValue> SortByStaticFastValue<T> {
    /// Creates a new `SortByStaticFastValue` instance for the given field.
    pub fn for_field(column_name: impl ToString) -> SortByStaticFastValue<T> {
        Self {
            field: column_name.to_string(),
            typ: PhantomData,
            cursor_u64: None,
            cursor_is_asc: true,
        }
    }

    /// Sets a search_after cursor. Documents whose sort value is at or before
    /// the cursor are skipped during collection, eliminating the need for a
    /// BooleanQuery intersection with a RangeQuery.
    pub fn with_search_after(mut self, cursor_value: T, is_asc: bool) -> Self {
        self.cursor_u64 = Some(cursor_value.to_u64());
        self.cursor_is_asc = is_asc;
        self
    }
}

impl<T: FastValue> SortKeyComputer for SortByStaticFastValue<T> {
    type Child = SortByFastValueSegmentSortKeyComputer<T>;
    type SortKey = Option<T>;
    type Comparator = NaturalComparator;

    fn check_schema(&self, schema: &crate::schema::Schema) -> crate::Result<()> {
        // At the segment sort key computer level, we rely on the u64 representation.
        // The mapping is monotonic, so it is sufficient to compute our top-K docs.
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
        Ok(SortByFastValueSegmentSortKeyComputer {
            sort_column,
            typ: PhantomData,
            cursor_u64: self.cursor_u64,
            cursor_is_asc: self.cursor_is_asc,
        })
    }
}

pub struct SortByFastValueSegmentSortKeyComputer<T> {
    sort_column: Column<u64>,
    typ: PhantomData<T>,
    cursor_u64: Option<u64>,
    cursor_is_asc: bool,
}

impl<T: FastValue> SegmentSortKeyComputer for SortByFastValueSegmentSortKeyComputer<T> {
    type SortKey = Option<T>;
    type SegmentSortKey = Option<u64>;
    type SegmentComparator = NaturalComparator;

    #[inline(always)]
    fn segment_sort_key(&mut self, doc: DocId, _score: Score) -> Self::SegmentSortKey {
        self.sort_column.first(doc)
    }

    #[inline(always)]
    fn compute_sort_key_and_collect<C: Comparator<Self::SegmentSortKey>>(
        &mut self,
        doc: DocId,
        score: Score,
        top_n_computer: &mut crate::collector::TopNComputer<Self::SegmentSortKey, DocId, C>,
    ) {
        let sort_key = self.segment_sort_key(doc, score);
        // Skip docs at or before the search_after cursor — O(1) column check,
        // avoids the need for BooleanQuery(base, RangeQuery) intersection.
        if let Some(cursor) = self.cursor_u64 {
            if let Some(val) = sort_key {
                if self.cursor_is_asc {
                    if val <= cursor {
                        return;
                    }
                } else if val >= cursor {
                    return;
                }
            } else {
                // Null values: skip (null sorts last in both orders)
                return;
            }
        }
        top_n_computer.push(sort_key, doc);
    }

    fn convert_segment_sort_key(&self, sort_key: Self::SegmentSortKey) -> Self::SortKey {
        sort_key.map(T::from_u64)
    }
}

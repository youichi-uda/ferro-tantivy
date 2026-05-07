use columnar::StrColumn;

use crate::collector::sort_key::NaturalComparator;
use crate::collector::sort_key::sort_by_static_fast_value::warm_first_values;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer};
use crate::termdict::TermOrdinal;
use crate::{DocId, Score};

/// Sort by the first value of a string column.
///
/// The string can be dynamic (coming from a json field)
/// or static (being specificaly defined in the configuration).
///
/// If the field is multivalued, only the first value is considered.
///
/// Documents that do not have this value are still considered.
/// Their sort key will simply be `None`.
///
/// # Wave 8 perf notes
///
/// Sort by string compares term ordinals (the column-stored
/// `TermOrdinal`s, which are sorted in the same byte order as the dictionary)
/// in the inner loop, then resolves the actual UTF-8 term only for the final
/// top-K hits.  We pre-decode the term-ordinal column into a contiguous
/// `Vec<u64>` once per segment (same path as `SortByStaticFastValue`) when the
/// segment has at most `WARM_FIRST_VALS_MAX_DOCS` docs and the column has
/// `Full` cardinality, so the inner read becomes a Vec index instead of
/// bit-unpacking + dictionary rank lookup.  Above that segment size we keep
/// the streaming `column.first(doc)` path to avoid blowing per-query memory.
#[derive(Debug, Clone)]
pub struct SortByString {
    column_name: String,
}

impl SortByString {
    /// Creates a new sort by string sort key computer.
    pub fn for_field(column_name: impl ToString) -> Self {
        SortByString {
            column_name: column_name.to_string(),
        }
    }
}

impl SortKeyComputer for SortByString {
    type SortKey = Option<String>;
    type Child = ByStringColumnSegmentSortKeyComputer;
    type Comparator = NaturalComparator;

    fn segment_sort_key_computer(
        &self,
        segment_reader: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        let str_column_opt = segment_reader.fast_fields().str(&self.column_name)?;
        // When the string column exists, pre-decode its term-ordinal column.
        // The term-ordinal column is a `Column<u64>` indexed by doc, which is
        // exactly the shape `warm_first_values` is designed for.
        let warm_first_term_ords = str_column_opt.as_ref().and_then(|c| warm_first_values(c.ords()));
        Ok(ByStringColumnSegmentSortKeyComputer {
            str_column_opt,
            warm_first_term_ords,
        })
    }
}

pub struct ByStringColumnSegmentSortKeyComputer {
    str_column_opt: Option<StrColumn>,
    /// Pre-decoded `first(doc)` of the term-ordinal column.  `Full`
    /// cardinality guarantees every doc has a term ord, so we use a flat
    /// `Box<[u64]>` instead of `Box<[Option<u64>]>` (half the memory + no
    /// inner-loop discriminant check).
    warm_first_term_ords: Option<Box<[u64]>>,
}

impl SegmentSortKeyComputer for ByStringColumnSegmentSortKeyComputer {
    type SortKey = Option<String>;
    type SegmentSortKey = Option<TermOrdinal>;
    type SegmentComparator = NaturalComparator;

    #[inline(always)]
    fn segment_sort_key(&mut self, doc: DocId, _score: Score) -> Option<TermOrdinal> {
        if let Some(buf) = self.warm_first_term_ords.as_ref() {
            // SAFETY: warming uses the same num_docs as the str column, and
            // queries deliver only in-bounds doc ids.  Warm cache is only
            // populated for `ColumnIndex::Full`, so every doc has a term ord.
            unsafe {
                debug_assert!((doc as usize) < buf.len(), "doc out of bounds");
                Some(*buf.get_unchecked(doc as usize))
            }
        } else {
            let str_column = self.str_column_opt.as_ref()?;
            str_column.ords().first(doc)
        }
    }

    /// Block-mode override: when the warm cache is bypassed (segment too
    /// large or column not `Full`), batch-read the term-ordinal column with
    /// `Column::first_vals` to pay the bit-unpack cost once per block instead
    /// of once per doc.  Equivalent to Lucene's `SortedDocValues` block read.
    #[inline]
    fn compute_block_sort_keys_and_collect<C: crate::collector::sort_key::Comparator<Self::SegmentSortKey>>(
        &mut self,
        docs: &[DocId],
        top_n_computer: &mut crate::collector::TopNComputer<Self::SegmentSortKey, DocId, C>,
    ) {
        if let Some(buf) = self.warm_first_term_ords.as_ref() {
            for &doc in docs {
                // SAFETY: see `segment_sort_key` above.
                let val = unsafe {
                    debug_assert!((doc as usize) < buf.len(), "doc out of bounds");
                    *buf.get_unchecked(doc as usize)
                };
                top_n_computer.push(Some(val), doc);
            }
            return;
        }
        let Some(str_column) = self.str_column_opt.as_ref() else {
            // No string column at all — every doc has key None, but we still
            // need to push so the topN heap contains them (sort_by_string
            // semantics).  Push None for every doc.
            for &doc in docs {
                top_n_computer.push(None, doc);
            }
            return;
        };
        const BLOCK: usize = crate::COLLECT_BLOCK_BUFFER_LEN;
        let mut scratch: [Option<u64>; BLOCK] = [None; BLOCK];
        let n = docs.len().min(BLOCK);
        if n == 0 {
            return;
        }
        for slot in scratch.iter_mut().take(n) {
            *slot = None;
        }
        str_column.ords().first_vals(&docs[..n], &mut scratch[..n]);
        for i in 0..n {
            top_n_computer.push(scratch[i], docs[i]);
        }
        if docs.len() > BLOCK {
            self.compute_block_sort_keys_and_collect(&docs[BLOCK..], top_n_computer);
        }
    }

    fn convert_segment_sort_key(&self, term_ord_opt: Option<TermOrdinal>) -> Option<String> {
        // TODO: Individual lookups to the dictionary like this are very likely to repeatedly
        // decompress the same blocks. See https://github.com/quickwit-oss/tantivy/issues/2776
        let term_ord = term_ord_opt?;
        let str_column = self.str_column_opt.as_ref()?;
        let mut bytes = Vec::new();
        str_column
            .dictionary()
            .ord_to_term(term_ord, &mut bytes)
            .ok()?;
        String::try_from(bytes).ok()
    }
}

use std::cell::UnsafeCell;

use columnar::StrColumn;

use crate::collector::sort_key::simd_top_k::{
    detect_order, is_contiguous_block, simd_filter_block_gt_u64, simd_filter_block_lt_u64,
    unwrap_threshold, DetectedOrder,
};
use crate::collector::sort_key::sort_by_static_fast_value::warm_first_values;
use crate::collector::sort_key::NaturalComparator;
use crate::collector::sort_key::simd_top_k::{
    DetectedOrder, detect_order, is_contiguous_block, simd_filter_block_gt_u64,
    simd_filter_block_lt_u64, unwrap_threshold,
};
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
///
/// # Wave 14 perf notes — keyword sort ES gap closure
///
/// Wave 13 EC2 verify (`dd-pack/rally-bench-fullmode-2026-05-09-postwave13`)
/// flagged http_logs `sort_status_*` / `sort_keyword_*` running 3.34–8.30×
/// slower than ES 9.3.2 even after Wave 5 #D's term-ordinal warm cache.  ES
/// Lucene's `SortedDocValuesWriter`-fed comparator wins because:
///
/// 1. **Branchless threshold filter**: per-doc compare is just `ord > bottom_ord` (Desc) / `ord <
///    bottom_ord` (Asc), all in u64 space.  No `Option<u64>` discriminant, no generic Comparator
///    dispatch.
/// 2. **Block-level SIMD greater-than/less-than**: 64-doc blocks are filtered against the heap
///    threshold in 8–16 vector ops per block, so the 99%+ of docs that fail threshold never enter
///    the heap push function.
/// 3. **Lazy term materialization**: only the final top-K hits resolve their term ordinal back to a
///    UTF-8 string; the per-segment hot loop is pure u64.
///
/// All three lever points apply identically to keyword sort because the
/// term ordinal column **is** a `Column<u64>` (`StrColumn::ords()`).  The
/// dictionary stores terms in sorted byte order, so `ord_a > ord_b` is
/// equivalent to `term(ord_a) > term(ord_b)` lexicographically — exactly
/// what `SortedDocValues` gives Lucene.  See `super::simd_top_k` for the
/// SIMD kernel and the correctness oracle.
///
/// In addition, `convert_segment_sort_key` previously allocated a fresh
/// `Vec<u8>` on every top-K hit just to materialize the sort_value emitted
/// by `_search.hits[].sort`.  We now reuse a per-segment scratch buffer
/// so the allocator is hit at most twice per segment (initial alloc + at
/// most one growth), regardless of top-K size.
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
        let warm_first_term_ords = str_column_opt
            .as_ref()
            .and_then(|c| warm_first_values(c.ords()));
        Ok(ByStringColumnSegmentSortKeyComputer {
            str_column_opt,
            warm_first_term_ords,
            term_scratch: UnsafeCell::new(Vec::with_capacity(32)),
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
    /// Reusable scratch for `convert_segment_sort_key` so each top-K hit
    /// does not allocate a fresh `Vec<u8>` for the
    /// `dictionary().ord_to_term` decode.
    ///
    /// `UnsafeCell` (manual `Sync` impl below) because the trait signature
    /// takes `&self` and the wider trait infrastructure
    /// (`ErasedSegmentSortKeyComputer` for `SortByErasedType`) requires
    /// `Sync`.  Access discipline:
    ///   - `convert_segment_sort_key` is the only consumer.
    ///   - It is called from `harvest()` after the segment's collect-loop has finished, by the
    ///     single rayon worker that owned this segment.  No concurrent `&Self` exists during
    ///     conversion.
    ///   - The dictionary lookup is read-only and does not re-enter `convert_segment_sort_key`, so
    ///     reentrancy is impossible.
    /// These constraints make the access serialised in practice; the
    /// `&mut Vec<u8>` we materialise is unique for the duration of each
    /// call.
    term_scratch: UnsafeCell<Vec<u8>>,
}

// SAFETY: see `term_scratch` doc.  The `UnsafeCell<Vec<u8>>` is only ever
// accessed from `convert_segment_sort_key` on a single owner thread; we
// never share a reference to the inner Vec across threads, so the `Sync`
// impl is sound for the access pattern of the surrounding trait
// infrastructure.
unsafe impl Sync for ByStringColumnSegmentSortKeyComputer {}

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
    /// `Column::first_vals` to pay the bit-unpack cost once per block
    /// instead of once per doc.  Equivalent to Lucene's
    /// `SortedDocValues` block read.
    ///
    /// **Wave 14 — SIMD top-K threshold filter for keyword sort.**  When
    /// the warm cache is populated, the heap-full threshold is set, and
    /// the block delivered by `for_each_no_score` is contiguous
    /// (`docs[i] == docs[0] + i`, common case for AllQuery / match_all /
    /// range scans), we run a SIMD greater-than/less-than filter on the
    /// term-ordinal values against the threshold and **only push the
    /// survivors** into the heap.  Same trick as the numeric sort path;
    /// closes the 3.34–8.30× http_logs `sort_status_*` /
    /// `sort_keyword_*` gap to ES.
    #[inline]
    fn compute_block_sort_keys_and_collect<
        C: crate::collector::sort_key::Comparator<Self::SegmentSortKey>,
    >(
        &mut self,
        docs: &[DocId],
        top_n_computer: &mut crate::collector::TopNComputer<Self::SegmentSortKey, DocId, C>,
    ) {
        if let Some(buf) = self.warm_first_term_ords.as_ref() {
            // Wave 14 SIMD top-K filter — only viable when:
            //   1. block is contiguous (no gather needed),
            //   2. block is reasonably large (amortise SIMD setup),
            //   3. threshold is established (heap full).
            let n = docs.len();
            if n >= 16 && top_n_computer.threshold.is_some() && is_contiguous_block(docs) {
                let start = docs[0] as usize;
                if start + n <= buf.len() {
                    if let Some(threshold) = unwrap_threshold(&top_n_computer.threshold) {
                        let order = detect_order(top_n_computer.comparator_ref());
                        let slice = &buf[start..start + n];
                        let mask = match order {
                            DetectedOrder::Desc => simd_filter_block_gt_u64(slice, threshold, n),
                            DetectedOrder::Asc => simd_filter_block_lt_u64(slice, threshold, n),
                            DetectedOrder::Unknown => {
                                // Fall back to scalar — comparator we can't
                                // classify (e.g. user-supplied custom).
                                u64::MAX
                            }
                        };
                        if order != DetectedOrder::Unknown {
                            // Block-level early termination: when no doc in
                            // the block beats the threshold, the entire
                            // 64-doc block is skipped without entering the
                            // heap-push function once.  The dominant
                            // pattern for low-cardinality keyword sort
                            // (status: 5 distinct values) is that after the
                            // first 10 docs nothing else can ever displace
                            // the threshold — `mask == 0` shortcuts the
                            // rest.
                            if mask == 0 {
                                return;
                            }
                            // Iterate only set bits — survivors.
                            let docs_start = docs[0];
                            let mut m = mask;
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
                }
            }

            // Slow path: scalar per-doc loop (heap not yet full,
            // non-contiguous block, or unknown comparator).  This is also
            // the heap-fill phase, before the threshold is set.
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
            // No string column at all — every doc has key None, but we
            // still need to push so the topN heap contains them
            // (sort_by_string semantics).  Push None for every doc.
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
        // Wave 14 — reuse a per-segment scratch buffer instead of
        // allocating a fresh `Vec<u8>` per top-K hit.  Without this,
        // `_search` with size=10 paid 10 small allocations per shard
        // regardless of the dictionary block cache hit rate.  Top-K is
        // bounded so the buffer grows at most once.  Note: see
        // https://github.com/quickwit-oss/tantivy/issues/2776 for the
        // upstream "decompress same dictionary block 10 times" tracking
        // issue — that is orthogonal and applies to merged-segment
        // queries; the buffer reuse here helps regardless.
        let term_ord = term_ord_opt?;
        let str_column = self.str_column_opt.as_ref()?;
        // SAFETY: see `term_scratch` doc on
        // `ByStringColumnSegmentSortKeyComputer`.
        // `convert_segment_sort_key` is called sequentially from
        // `harvest()` for each top-K hit on a single thread; there is no
        // outstanding `&mut Vec<u8>` from anywhere else for the duration
        // of this borrow.  The dictionary lookup neither re-enters
        // `convert_segment_sort_key` nor stores the borrow.
        let scratch: &mut Vec<u8> = unsafe { &mut *self.term_scratch.get() };
        scratch.clear();
        if !str_column
            .dictionary()
            .ord_to_term(term_ord, scratch)
            .ok()?
        {
            return None;
        }
        // `str::from_utf8` does the same work as `String::try_from(Vec)`
        // but borrows; we allocate exactly once for the resulting
        // `String`.
        std::str::from_utf8(scratch).ok().map(str::to_owned)
    }
}

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use columnar::{Cardinality, Column};

use crate::collector::sort_key::{Comparator, NaturalComparator};
use crate::collector::sort_key_top_collector::TopBySortKeySegmentCollector;
use crate::collector::{
    default_collect_segment_impl, SegmentCollector as _, SegmentSortKeyComputer, SortKeyComputer,
    TopNComputer,
};
use crate::fastfield::{FastFieldNotAvailableError, FastValue};
use crate::DocAddress;
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
///
/// # Wave 8 perf notes
///
/// Two complementary throughput levers landed in Wave 8 to close the
/// `desc_sort_timestamp_*` / `sort_status_*` regressions vs ES 9.3.2:
///
/// 1. **Warm cache** for small-medium segments (≤[`WARM_FIRST_VALS_MAX_DOCS`]
///    docs, [`ColumnIndex::Full`](columnar::ColumnIndex::Full) cardinality).
///    The per-segment computer pre-decodes single-valued columns into a
///    contiguous `Box<[u64]>` (Lucene "doc-values warm"), so subsequent
///    `first(doc)` reads are O(1) Vec indexing instead of bit-unpacking.
///
/// 2. **Block-mode read** in [`Self::compute_block_sort_keys_and_collect`].
///    When the no-score iterator (`Weight::for_each_no_score`) hands us a
///    block of [`DocId`]s, we resolve all 64 sort keys in a single
///    `Column::first_vals` call.  This is the equivalent of Lucene's
///    `NumericComparator`'s block iteration over `LongValues` and pays the
///    bit-unpack overhead once per block instead of once per doc.  Critically
///    this also benefits **large** segments above the warm-cache cap (the
///    Rally http_logs / geonames tracks).
///
/// The `cursor_u64` filter (search_after) is applied inline in both the
/// per-doc and block paths; null values are dropped (null sorts last).
#[derive(Debug, Clone)]
pub struct SortByStaticFastValue<T: FastValue> {
    field: String,
    typ: PhantomData<T>,
    /// Optional search_after cursor — docs at or before this value are skipped.
    cursor_u64: Option<u64>,
    /// True = ascending sort (skip docs with value <= cursor),
    /// False = descending sort (skip docs with value >= cursor).
    cursor_is_asc: bool,
    /// **Wave 24.** Cross-segment running top-K threshold (best-effort skip
    /// gate). When set, [`SortKeyComputer::should_skip_segment`] compares
    /// the column's `[min, max]` to the most-recent K-th value submitted by
    /// any sibling segment and skips the entire segment when no doc can
    /// possibly enter the global top-K. Each call to
    /// [`Self::collect_segment_top_k`] tightens the threshold via an
    /// atomic `fetch_max` (desc) / `fetch_min` (asc) so later segments see
    /// the latest gate. Parallel segments race on initial load, so the
    /// gate is *best-effort*: it never produces a wrong result (the
    /// merge-fruits step always picks the global top-K), it only saves
    /// work.
    running_threshold: Option<Arc<AtomicU64>>,
    /// Threshold direction. `true` = descending top-K (keep largest values,
    /// threshold = current min of top-K, tightened via `fetch_max`).
    /// `false` = ascending top-K (keep smallest values, threshold = current
    /// max of top-K, tightened via `fetch_min`). Must match the [`Order`]
    /// the caller will wrap this computer in.
    threshold_is_desc: bool,
}

impl<T: FastValue> SortByStaticFastValue<T> {
    /// Creates a new `SortByStaticFastValue` instance for the given field.
    pub fn for_field(column_name: impl ToString) -> SortByStaticFastValue<T> {
        Self {
            field: column_name.to_string(),
            typ: PhantomData,
            cursor_u64: None,
            cursor_is_asc: true,
            running_threshold: None,
            threshold_is_desc: false,
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

    /// **Wave 24.** Enable cross-segment top-K skip via a shared atomic
    /// threshold. `is_desc` MUST match the [`Order`] this computer will be
    /// wrapped in by the collector (asc-order callers pass `false`,
    /// desc-order pass `true`). Initial threshold is `u64::MIN` for desc
    /// and `u64::MAX` for asc, so the first segment never skips itself.
    ///
    /// Returns `self` with `running_threshold = Some(Arc::new(...))`;
    /// callers that clone or share the computer across segments inherit
    /// the same atomic via `Arc::clone`.
    pub fn with_running_threshold(mut self, is_desc: bool) -> Self {
        let init = if is_desc { u64::MIN } else { u64::MAX };
        self.running_threshold = Some(Arc::new(AtomicU64::new(init)));
        self.threshold_is_desc = is_desc;
        self
    }

    /// **Wave 24.** `[min, max]` probe identical to Wave 22's cursor
    /// variant: returns `true` iff no doc in this segment can enter the
    /// global top-K given the most-recent K-th value seen by sibling
    /// segments. Conservative on cardinality (`Full` only — see
    /// `SortByStaticFastValueWithCursor::segment_can_be_skipped` for the
    /// invariants); falls through to the regular collection path
    /// otherwise.
    #[inline]
    fn segment_can_be_skipped_by_threshold(&self, sort_column: &Column<u64>) -> bool {
        let Some(thresh) = self.running_threshold.as_ref() else {
            return false;
        };
        if !matches!(sort_column.get_cardinality(), Cardinality::Full) {
            return false;
        }
        let min_v = sort_column.values.min_value();
        let max_v = sort_column.values.max_value();
        let threshold_value = thresh.load(Ordering::Relaxed);
        if self.threshold_is_desc {
            // Desc: top-K keeps largest. Threshold = current K-th (smallest
            // in top-K seen so far). Entry requires strict `> threshold`,
            // so a segment with `max <= threshold` cannot contribute.
            max_v <= threshold_value
        } else {
            // Asc: top-K keeps smallest. Threshold = current K-th (largest
            // in top-K seen so far). Entry requires strict `< threshold`,
            // so a segment with `min >= threshold` cannot contribute.
            min_v >= threshold_value
        }
    }

    /// **Wave 24.** After a segment's top-K is materialized, tighten the
    /// global threshold using this segment's K-th value (worst rank among
    /// the K kept). `fetch_max` for desc and `fetch_min` for asc are both
    /// monotone, so concurrent segments racing to publish never weaken the
    /// gate.
    ///
    /// Skips the update when:
    /// - no running threshold was configured;
    /// - the segment returned fewer than `k` entries (the gate would not
    ///   be a true K-th);
    /// - every kept entry has a `None` sort key (null-only segment carries
    ///   no useful threshold information).
    fn tighten_threshold_after_collect(&self, k: usize, segment_top_k: &[(Option<T>, DocAddress)]) {
        let Some(thresh) = self.running_threshold.as_ref() else {
            return;
        };
        if segment_top_k.len() < k {
            return;
        }
        // The K-th value (worst kept) is the segment's contribution to the
        // global threshold. For desc we want the *minimum* non-null kept
        // value; for asc, the *maximum*. Scanning is `O(k)` — k is the
        // top-K parameter, typically ≤ 10⁴ — and avoids a brittle
        // dependency on TopNComputer's harvest order.
        let kth_u64 = if self.threshold_is_desc {
            segment_top_k
                .iter()
                .filter_map(|(v, _)| v.map(T::to_u64))
                .min()
        } else {
            segment_top_k
                .iter()
                .filter_map(|(v, _)| v.map(T::to_u64))
                .max()
        };
        if let Some(kth) = kth_u64 {
            if self.threshold_is_desc {
                thresh.fetch_max(kth, Ordering::Relaxed);
            } else {
                thresh.fetch_min(kth, Ordering::Relaxed);
            }
        }
    }

    /// Test hook: read the current running threshold u64 (if configured).
    #[cfg(test)]
    pub(crate) fn current_threshold(&self) -> Option<u64> {
        self.running_threshold
            .as_ref()
            .map(|t| t.load(Ordering::Relaxed))
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

    /// **Wave 24.** Cross-segment top-K skip gate. Mirrors the Wave 22
    /// cursor probe in [`SortByStaticFastValueWithCursor`] but reads its
    /// threshold from a per-query [`AtomicU64`] tightened by sibling
    /// segments rather than from a constant cursor value. Cheap: only the
    /// column-codec stats are touched. Returns `false` (no skip) when no
    /// running threshold is configured, so the trait default path through
    /// the `(_, Order)` wrapper is preserved for non-Wave-24 callers.
    fn should_skip_segment(&self, segment_reader: &SegmentReader) -> bool {
        if self.running_threshold.is_none() {
            return false;
        }
        match segment_reader.fast_fields().u64_lenient(&self.field) {
            Ok(Some((sort_column, _))) => self.segment_can_be_skipped_by_threshold(&sort_column),
            _ => false,
        }
    }

    /// **Wave 24.** Override the default top-K collection just enough to
    /// tighten the running threshold after each segment finishes. The
    /// body mirrors the trait's default impl (Wave 8 block-mode and Wave
    /// 19 warm cache are inherited via `segment_sort_key_computer`); the
    /// only added work is one `O(k)` scan over the harvested top-K plus a
    /// relaxed atomic.
    fn collect_segment_top_k(
        &self,
        k: usize,
        weight: &dyn crate::query::Weight,
        reader: &crate::SegmentReader,
        segment_ord: u32,
    ) -> crate::Result<Vec<(Self::SortKey, DocAddress)>> {
        if self.should_skip_segment(reader) {
            return Ok(Vec::new());
        }
        let with_scoring = self.requires_scoring();
        let segment_sort_key_computer = self.segment_sort_key_computer(reader)?;
        let topn_computer = TopNComputer::new_with_comparator(k, self.comparator());
        let mut segment_top_key_collector = TopBySortKeySegmentCollector {
            topn_computer,
            segment_ord,
            segment_sort_key_computer,
        };
        default_collect_segment_impl(&mut segment_top_key_collector, weight, reader, with_scoring)?;
        let result = segment_top_key_collector.harvest();
        self.tighten_threshold_after_collect(k, &result);
        Ok(result)
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
        // first access by [`SegmentReader::warm_fast_field_dense_u64`].
        // The previous per-collector decode (the `warm_first_values`
        // helper below) capped at `WARM_FIRST_VALS_MAX_DOCS = 256 K`
        // because each Rally iteration paid the alloc cost; the
        // segment-level cache eliminates that and lets the SIMD top-K
        // filter (see `compute_block_sort_keys_and_collect`) fire at
        // any segment size — the lever that closes the Rally
        // http_logs `sort_size_*` / `sort_status_*` gap vs ES 9.3.x.
        //
        // Falls back to the deprecated per-collector decode for
        // segments below the legacy cap when the field is sparse
        // (cache hit returns `None` for `Optional` / `Multivalued`),
        // preserving the existing behaviour for non-dense workloads.
        let warm_first_vals = segment_reader
            .warm_fast_field_dense_u64(&self.field, &sort_column)
            .or_else(|| warm_first_values(&sort_column).map(|b| std::sync::Arc::new(b)));
        Ok(SortByFastValueSegmentSortKeyComputer {
            sort_column,
            warm_first_vals,
            typ: PhantomData,
            cursor_u64: self.cursor_u64,
            cursor_is_asc: self.cursor_is_asc,
        })
    }
}

/// Maximum segment size (in docs) for which we eagerly warm the column into a
/// dense `Vec<u64>`.  Above this we keep the streaming `Column::first` path
/// because the alloc + decode amortises to less than the per-doc bit-unpack
/// savings, and we don't want a 12M × 8B = 96 MB allocation per query.
///
/// This is only the eager-warm threshold.  For larger segments we use
/// block-mode `first_vals` reads in `collect_block`, which has the same
/// branchless tight-loop benefit without the per-segment alloc.
pub(crate) const WARM_FIRST_VALS_MAX_DOCS: u32 = 256 * 1024;

/// Pre-decode a single-valued, dense column into a contiguous `Vec<u64>`,
/// or return `None` when warming is not safe / profitable.
///
/// Warming is only enabled for `ColumnIndex::Full` (every doc has exactly one
/// value) and for segments with at most [`WARM_FIRST_VALS_MAX_DOCS`] docs.
///
/// We deliberately encode "no value" as `u64::MAX` instead of using
/// `Vec<Option<u64>>` so the warm buffer is half the size and avoids the extra
/// `Option` discriminant byte in the inner loop.  `ColumnIndex::Full` means
/// every doc has a value, so the sentinel is unreachable in practice; we only
/// pay the discriminant cost on the streaming fallback path.
pub(crate) fn warm_first_values(column: &Column<u64>) -> Option<Box<[u64]>> {
    use columnar::ColumnIndex;
    let num_docs = column.num_docs();
    // Skip warm if column is too small (overhead not worth it), empty, or
    // larger than the per-segment alloc cap.  The two boundary checks read
    // more naturally than a single `(min..=max).contains(...)` because
    // `MIN` and `MAX` are tuned for different reasons (alloc-amortise vs
    // per-query memory cap) and may diverge in future tuning.
    #[allow(clippy::manual_range_contains)]
    if num_docs < 1024 || num_docs > WARM_FIRST_VALS_MAX_DOCS {
        return None;
    }
    match &column.index {
        ColumnIndex::Full => {
            // Dense single-valued: pre-decode all values.  `Full` means
            // every doc has a value, so we can use a flat `Vec<u64>`.
            let mut buf = vec![0u64; num_docs as usize];
            column.values.get_range(0, &mut buf);
            Some(buf.into_boxed_slice())
        }
        // Optional / Multivalued: keep streaming path.  The doc→row rank
        // mapping cost outweighs the bit-unpack savings for sparse data,
        // and the rare-value None entries don't compress cleanly.
        _ => None,
    }
}

pub struct SortByFastValueSegmentSortKeyComputer<T> {
    sort_column: Column<u64>,
    /// **Wave 19.** Shared segment-lifetime warm cache populated lazily
    /// by [`SegmentReader::warm_fast_field_dense_u64`].  `Full`
    /// cardinality guarantees every doc has a value, so the cache is
    /// `Box<[u64]>` without an `Option` discriminant.  Wrapped in `Arc`
    /// so every concurrent collector against the same segment+field
    /// shares one decoded buffer (no per-Rally-iteration realloc).
    warm_first_vals: Option<std::sync::Arc<Box<[u64]>>>,
    typ: PhantomData<T>,
    cursor_u64: Option<u64>,
    cursor_is_asc: bool,
}

impl<T: FastValue> SortByFastValueSegmentSortKeyComputer<T> {
    /// Returns the precomputed first value for `doc`, falling back to the
    /// streaming column read when warming was skipped.
    #[inline(always)]
    fn first_value(&self, doc: DocId) -> Option<u64> {
        if let Some(buf) = self.warm_first_vals.as_ref() {
            // SAFETY: `doc < num_docs` is always true for docs delivered by
            // the query iterator (matches.alive_bitset() filtering), and
            // `buf.len() == num_docs as usize` by construction in
            // `warm_first_values`.  The warm cache is only populated for
            // `ColumnIndex::Full`, so every doc has a value and we return
            // `Some(_)` unconditionally.
            unsafe {
                debug_assert!((doc as usize) < buf.len(), "doc out of bounds");
                Some(*buf.get_unchecked(doc as usize))
            }
        } else {
            self.sort_column.first(doc)
        }
    }
}

impl<T: FastValue> SegmentSortKeyComputer for SortByFastValueSegmentSortKeyComputer<T> {
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

    /// Block-mode override: uses `Column::first_vals` to batch-decode the
    /// block of doc-values (Lucene NumericComparator-equivalent), then loops
    /// over the resolved keys with the cursor + threshold check.
    ///
    /// This avoids one bit-unpack call per document; for bitpacked codecs the
    /// inner loop becomes a tight strided read, which is the primary
    /// throughput lever for `sort_*_no_can_match_shortcut` Rally ops.
    ///
    /// We size the scratch buffer to match `tantivy::COLLECT_BLOCK_BUFFER_LEN`
    /// (the block size the query iterator hands us), but we read whatever the
    /// caller actually delivers — `for_each_no_score` may pass a final
    /// short block at the segment tail.
    ///
    /// **Wave 14 — SIMD top-K threshold filter.**  When the warm cache is
    /// available, the heap-full threshold is set, and the block delivered by
    /// `for_each_no_score` is contiguous (`docs[i] == docs[0] + i`, which is
    /// the common case for AllQuery / match_all / range scans), we run a
    /// SIMD greater-than/less-than filter against the threshold and **only
    /// push the survivors** into the heap.  This skips the generic
    /// `Comparator<Option<u64>>` dispatch on the 99%+ of docs that fail the
    /// threshold check after the heap fills up — exactly the pattern that
    /// gives ES Lucene's `NumericComparator` its 3–8× lead on
    /// http_logs `desc/asc-sort-*-after-force-merge-1-seg`.  See
    /// [`super::simd_top_k`] for the SIMD kernel and the correctness oracle.
    #[inline]
    fn compute_block_sort_keys_and_collect<C: Comparator<Self::SegmentSortKey>>(
        &mut self,
        docs: &[DocId],
        top_n_computer: &mut crate::collector::TopNComputer<Self::SegmentSortKey, DocId, C>,
    ) {
        // Fast path: warm cache covers everything, no first_vals needed.
        if let Some(buf) = self.warm_first_vals.as_ref() {
            // Wave 14 SIMD top-K filter — only viable when:
            //   1. block is contiguous (no gather needed),
            //   2. block is reasonably large (amortise SIMD setup),
            //   3. threshold is established (heap full),
            //   4. cursor is absent (cursor variant is in
            //      sort_by_static_fast_value_with_cursor.rs).
            //
            // We probe the comparator once with two known values to detect
            // asc vs desc semantics — the segment-level threshold/sort-key
            // is `Option<u64>` and the comparator's "Greater" semantics
            // determine which docs survive.
            let n = docs.len();
            if self.cursor_u64.is_none()
                && n >= 16
                && top_n_computer.threshold.is_some()
                && super::simd_top_k::is_contiguous_block(docs)
            {
                let start = docs[0] as usize;
                if start + n <= buf.len() {
                    if let Some(threshold) =
                        super::simd_top_k::unwrap_threshold(&top_n_computer.threshold)
                    {
                        let order =
                            super::simd_top_k::detect_order(top_n_computer.comparator_ref());
                        let slice = &buf[start..start + n];
                        let mask = match order {
                            super::simd_top_k::DetectedOrder::Desc => {
                                super::simd_top_k::simd_filter_block_gt_u64(slice, threshold, n)
                            }
                            super::simd_top_k::DetectedOrder::Asc => {
                                super::simd_top_k::simd_filter_block_lt_u64(slice, threshold, n)
                            }
                            super::simd_top_k::DetectedOrder::Unknown => {
                                // Fall back to scalar — comparator we can't classify
                                // (e.g. user-supplied custom).  Push everything.
                                u64::MAX
                            }
                        };
                        if order != super::simd_top_k::DetectedOrder::Unknown {
                            // Top-K block-level early termination: when the
                            // entire block fails the threshold, skip the
                            // per-doc loop entirely.  `count_ones() == 0` ⇒
                            // no survivors ⇒ skip block.
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

            // Slow path: scalar per-doc loop (cursor present, threshold not
            // set, non-contiguous block, or unknown comparator).
            for &doc in docs {
                // SAFETY: doc < num_docs == buf.len() (see `first_value`).
                let val = unsafe {
                    debug_assert!((doc as usize) < buf.len(), "doc out of bounds");
                    *buf.get_unchecked(doc as usize)
                };
                if let Some(cursor) = self.cursor_u64 {
                    if self.cursor_is_asc {
                        if val <= cursor {
                            continue;
                        }
                    } else if val >= cursor {
                        continue;
                    }
                }
                top_n_computer.push(Some(val), doc);
            }
            return;
        }
        // Streaming path: do a batch read into a stack-allocated scratch,
        // then push the resolved keys.  The scratch lifetime is the call,
        // so there is no per-segment alloc.
        const BLOCK: usize = crate::COLLECT_BLOCK_BUFFER_LEN;
        let mut scratch: [Option<u64>; BLOCK] = [None; BLOCK];
        let n = docs.len().min(BLOCK);
        if n == 0 {
            return;
        }
        // Reset only the slots we will write back into (`first_vals` only
        // overwrites entries that have a value for `Optional`/`Multivalued`;
        // for `Full` it overwrites all of them).
        for slot in scratch.iter_mut().take(n) {
            *slot = None;
        }
        self.sort_column.first_vals(&docs[..n], &mut scratch[..n]);
        for i in 0..n {
            let doc = docs[i];
            let sort_key = scratch[i];
            if let Some(cursor) = self.cursor_u64 {
                if let Some(val) = sort_key {
                    if self.cursor_is_asc {
                        if val <= cursor {
                            continue;
                        }
                    } else if val >= cursor {
                        continue;
                    }
                } else {
                    continue;
                }
            }
            top_n_computer.push(sort_key, doc);
        }
        // If `docs` is somehow larger than COLLECT_BLOCK_BUFFER_LEN (can
        // happen in custom callers — `for_each_no_score` itself respects
        // the constant), recurse for the tail.  This loop terminates because
        // `docs.len()` strictly decreases with each call.
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
    use super::{warm_first_values, SortByStaticFastValue, WARM_FIRST_VALS_MAX_DOCS};
    use crate::DocAddress;
    use columnar::column_values::VecColumn;
    use columnar::{Column, ColumnIndex};
    use std::sync::Arc;

    fn column_with_values(vs: Vec<u64>) -> Column<u64> {
        Column {
            index: ColumnIndex::Full,
            values: Arc::new(VecColumn::from(vs)),
        }
    }

    fn doc_addr(doc: u32) -> DocAddress {
        DocAddress {
            segment_ord: 0,
            doc_id: doc,
        }
    }

    #[test]
    fn wave24_no_threshold_does_not_skip() {
        // Default constructor — no running threshold. The probe must always
        // return false so callers that opt out of Wave 24 keep the old
        // behaviour (no skip).
        let sort = SortByStaticFastValue::<u64>::for_field("x");
        let col = column_with_values(vec![10, 20, 30, 40]);
        assert!(!sort.segment_can_be_skipped_by_threshold(&col));
        assert_eq!(sort.current_threshold(), None);
    }

    #[test]
    fn wave24_threshold_desc_skips_when_segment_max_at_or_below() {
        // Desc top-K wants the largest values. After earlier segments
        // tighten the threshold to 50, a segment whose max is 50 (tie with
        // the K-th — strict ">", so still skip) or below cannot contribute.
        let sort = SortByStaticFastValue::<u64>::for_field("x").with_running_threshold(true);
        sort.tighten_threshold_after_collect(
            2,
            &[(Some(100u64), doc_addr(0)), (Some(50u64), doc_addr(1))],
        );
        assert_eq!(sort.current_threshold(), Some(50));

        // Segment with max == 50 → skipped (strict-> entry).
        assert!(sort.segment_can_be_skipped_by_threshold(&column_with_values(vec![10, 30, 50])));
        // Segment with max == 51 → cannot skip.
        assert!(!sort.segment_can_be_skipped_by_threshold(&column_with_values(vec![10, 30, 51])));
    }

    #[test]
    fn wave24_threshold_asc_skips_when_segment_min_at_or_above() {
        // Asc top-K wants the smallest values. K-th is the max in top-K.
        let sort = SortByStaticFastValue::<u64>::for_field("x").with_running_threshold(false);
        sort.tighten_threshold_after_collect(
            2,
            &[(Some(10u64), doc_addr(0)), (Some(50u64), doc_addr(1))],
        );
        assert_eq!(sort.current_threshold(), Some(50));

        assert!(sort.segment_can_be_skipped_by_threshold(&column_with_values(vec![50, 80, 100])));
        assert!(!sort.segment_can_be_skipped_by_threshold(&column_with_values(vec![49, 80, 100])));
    }

    #[test]
    fn wave24_tighten_is_monotone_desc() {
        // Desc tightens via fetch_max — looser submissions never weaken the
        // gate even if a slower segment finishes after a tighter one.
        let sort = SortByStaticFastValue::<u64>::for_field("x").with_running_threshold(true);
        sort.tighten_threshold_after_collect(
            2,
            &[(Some(100u64), doc_addr(0)), (Some(80u64), doc_addr(1))],
        );
        assert_eq!(sort.current_threshold(), Some(80));

        // Subsequent segment with K-th = 30 — strictly worse than current 80.
        sort.tighten_threshold_after_collect(
            2,
            &[(Some(50u64), doc_addr(2)), (Some(30u64), doc_addr(3))],
        );
        assert_eq!(
            sort.current_threshold(),
            Some(80),
            "fetch_max keeps the tighter (larger) value"
        );

        // Subsequent segment with K-th = 90 — strictly better.
        sort.tighten_threshold_after_collect(
            2,
            &[(Some(110u64), doc_addr(4)), (Some(90u64), doc_addr(5))],
        );
        assert_eq!(sort.current_threshold(), Some(90));
    }

    #[test]
    fn wave24_tighten_skips_when_fewer_than_k() {
        // Partial top-K (segment didn't fill K slots) is not a valid global
        // threshold — would loosen the gate. Must be a no-op.
        let sort = SortByStaticFastValue::<u64>::for_field("x").with_running_threshold(true);
        // Seed a known-good threshold first.
        sort.tighten_threshold_after_collect(
            2,
            &[(Some(100u64), doc_addr(0)), (Some(80u64), doc_addr(1))],
        );
        assert_eq!(sort.current_threshold(), Some(80));

        // Now submit a partial top-K (1 entry, k=2): no update.
        sort.tighten_threshold_after_collect(2, &[(Some(50u64), doc_addr(2))]);
        assert_eq!(sort.current_threshold(), Some(80));
    }

    #[test]
    fn wave24_tighten_ignores_all_null_segments() {
        // A segment that only matched null-valued docs returns top-K with
        // `Option::None` sort keys. Nulls sort last by design (see Wave 8
        // doc-comment) so they should not contribute to the gate.
        let sort = SortByStaticFastValue::<u64>::for_field("x").with_running_threshold(true);
        sort.tighten_threshold_after_collect(
            2,
            &[(None::<u64>, doc_addr(0)), (None::<u64>, doc_addr(1))],
        );
        assert_eq!(
            sort.current_threshold(),
            Some(u64::MIN),
            "initial threshold preserved when no non-null values to publish"
        );
    }

    #[test]
    fn wave24_skip_requires_full_cardinality() {
        // Optional / Multivalued columns may emit `None` for some docs;
        // skipping such a segment would silently drop the null-sort-last
        // docs. Guard mirrors the Wave 22 cursor probe.
        let sort = SortByStaticFastValue::<u64>::for_field("x").with_running_threshold(true);
        sort.tighten_threshold_after_collect(
            2,
            &[(Some(100u64), doc_addr(0)), (Some(80u64), doc_addr(1))],
        );
        let optional_col = Column {
            index: ColumnIndex::Empty { num_docs: 4 },
            values: Arc::new(VecColumn::from(Vec::<u64>::new())),
        };
        assert!(
            !sort.segment_can_be_skipped_by_threshold(&optional_col),
            "non-Full cardinality must never skip"
        );
    }

    #[test]
    fn warm_first_values_full_column() {
        // num_docs >= 1024 to enter the warm path.
        let n = 1500u32;
        let raw: Vec<u64> = (0..n as u64).map(|i| i * 3).collect();
        let column = Column {
            index: ColumnIndex::Full,
            values: Arc::new(VecColumn::from(raw.clone())),
        };
        let warm = warm_first_values(&column).expect("full column with N>=1024 should warm");
        assert_eq!(warm.len(), n as usize);
        for (i, v) in warm.iter().enumerate() {
            assert_eq!(*v, raw[i]);
        }
    }

    #[test]
    fn warm_first_values_skipped_for_small_column() {
        let n = 100u32;
        let raw: Vec<u64> = (0..n as u64).collect();
        let column = Column {
            index: ColumnIndex::Full,
            values: Arc::new(VecColumn::from(raw)),
        };
        assert!(
            warm_first_values(&column).is_none(),
            "small columns should skip warming"
        );
    }

    #[test]
    fn warm_first_values_skipped_for_oversized_column() {
        // Above WARM_FIRST_VALS_MAX_DOCS we keep the streaming path to avoid
        // a per-query 96 MB allocation.  Build a column at the boundary.
        let n = WARM_FIRST_VALS_MAX_DOCS + 1;
        let raw: Vec<u64> = vec![42u64; n as usize];
        let column = Column {
            index: ColumnIndex::Full,
            values: Arc::new(VecColumn::from(raw)),
        };
        assert!(
            warm_first_values(&column).is_none(),
            "oversized columns should skip warming"
        );
    }
}

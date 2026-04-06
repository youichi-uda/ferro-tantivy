use std::collections::BTreeMap;
use std::sync::Arc;
use std::{fmt, io};

use crate::collector::Collector;
use crate::core::Executor;
use crate::index::{SegmentId, SegmentReader};
use crate::query::{Bm25StatisticsProvider, EnableScoring, Query};
use crate::schema::document::DocumentDeserialize;
use crate::schema::{Schema, Term};
use crate::space_usage::SearcherSpaceUsage;
use crate::store::{CacheStats, StoreReader};
use crate::{DocAddress, Index, Opstamp, TrackedObject};

/// Identifies the searcher generation accessed by a [`Searcher`].
///
/// While this might seem redundant, a [`SearcherGeneration`] contains
/// both a `generation_id` AND a list of `(SegmentId, DeleteOpstamp)`.
///
/// This is on purpose. This object is used by the [`Warmer`](crate::reader::Warmer) API.
/// Having both information makes it possible to identify which
/// artifact should be refreshed or garbage collected.
///
/// Depending on the use case, `Warmer`'s implementers can decide to
/// produce artifacts per:
/// - `generation_id` (e.g. some searcher level aggregates)
/// - `(segment_id, delete_opstamp)` (e.g. segment level aggregates)
/// - `segment_id` (e.g. for immutable document level information)
/// - `(generation_id, segment_id)` (e.g. for consistent dynamic column)
/// - ...
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SearcherGeneration {
    segments: BTreeMap<SegmentId, Option<Opstamp>>,
    generation_id: u64,
}

impl SearcherGeneration {
    pub(crate) fn from_segment_readers(
        segment_readers: &[SegmentReader],
        generation_id: u64,
    ) -> Self {
        let mut segment_id_to_del_opstamp = BTreeMap::new();
        for segment_reader in segment_readers {
            segment_id_to_del_opstamp
                .insert(segment_reader.segment_id(), segment_reader.delete_opstamp());
        }
        Self {
            segments: segment_id_to_del_opstamp,
            generation_id,
        }
    }

    /// Returns the searcher generation id.
    pub fn generation_id(&self) -> u64 {
        self.generation_id
    }

    /// Return a `(SegmentId -> DeleteOpstamp)` mapping.
    pub fn segments(&self) -> &BTreeMap<SegmentId, Option<Opstamp>> {
        &self.segments
    }
}

/// Holds a list of `SegmentReader`s ready for search.
///
/// It guarantees that the `Segment` will not be removed before
/// the destruction of the `Searcher`.
#[derive(Clone)]
pub struct Searcher {
    inner: Arc<SearcherInner>,
}

impl Searcher {
    /// Returns the `Index` associated with the `Searcher`
    pub fn index(&self) -> &Index {
        &self.inner.index
    }

    /// [`SearcherGeneration`] which identifies the version of the snapshot held by this `Searcher`.
    pub fn generation(&self) -> &SearcherGeneration {
        self.inner.generation.as_ref()
    }

    /// Fetches a document from tantivy's store given a [`DocAddress`].
    ///
    /// The searcher uses the segment ordinal to route the
    /// request to the right `Segment`.
    pub fn doc<D: DocumentDeserialize>(&self, doc_address: DocAddress) -> crate::Result<D> {
        let store_reader = &self.inner.store_readers[doc_address.segment_ord as usize];
        store_reader.get(doc_address.doc_id)
    }

    /// Fetches multiple documents in batch, optimizing block decompression.
    ///
    /// Documents are grouped by segment and sorted by doc_id within each segment
    /// to maximize DocStore block cache hits. The returned `Vec` preserves the
    /// same order as the input `doc_addresses` slice.
    ///
    /// Documents that fail to deserialize are returned as `None`.
    pub fn docs_batch<D: DocumentDeserialize>(
        &self,
        doc_addresses: &[DocAddress],
    ) -> Vec<Option<D>> {
        if doc_addresses.is_empty() {
            return Vec::new();
        }

        // Build (original_index, address) pairs and sort by (segment_ord, doc_id)
        // so that docs in the same compressed block are fetched consecutively.
        let mut indexed: Vec<(usize, DocAddress)> = doc_addresses
            .iter()
            .enumerate()
            .map(|(i, a)| (i, *a))
            .collect();
        indexed.sort_unstable_by_key(|(_, a)| (a.segment_ord, a.doc_id));

        // Fetch in sorted order, placing results back at original indices.
        let mut results: Vec<Option<D>> = (0..doc_addresses.len()).map(|_| None).collect();
        for (orig_idx, addr) in indexed {
            let store_reader = &self.inner.store_readers[addr.segment_ord as usize];
            results[orig_idx] = store_reader.get::<D>(addr.doc_id).ok();
        }
        results
    }

    /// Batch-extracts raw bytes of a single field from multiple stored documents.
    ///
    /// Documents are sorted by (segment_ord, doc_id) internally to maximize
    /// DocStore block cache hits. The returned `Vec` preserves the same order
    /// as the input `doc_addresses` slice.
    ///
    /// Much faster than `docs_batch()` when only one field is needed, because
    /// non-target fields are skipped without heap allocation during extraction.
    pub fn batch_get_field_bytes(
        &self,
        doc_addresses: &[DocAddress],
        target_field: crate::schema::Field,
    ) -> Vec<Option<Vec<u8>>> {
        if doc_addresses.is_empty() {
            return Vec::new();
        }

        // Sort by (segment_ord, doc_id) and group by segment
        let mut indexed: Vec<(usize, DocAddress)> = doc_addresses
            .iter()
            .enumerate()
            .map(|(i, a)| (i, *a))
            .collect();
        indexed.sort_unstable_by_key(|(_, a)| (a.segment_ord, a.doc_id));

        let mut results: Vec<Option<Vec<u8>>> = vec![None; doc_addresses.len()];

        // Process per-segment groups using batch_get_field_bytes_grouped
        // which decompresses each block only once for all docs in that block.
        let mut seg_start = 0;
        while seg_start < indexed.len() {
            let seg_ord = indexed[seg_start].1.segment_ord;
            let mut seg_end = seg_start + 1;
            while seg_end < indexed.len() && indexed[seg_end].1.segment_ord == seg_ord {
                seg_end += 1;
            }
            let group = &indexed[seg_start..seg_end];
            let doc_ids: Vec<crate::DocId> = group.iter().map(|(_, a)| a.doc_id).collect();
            let store_reader = &self.inner.store_readers[seg_ord as usize];
            let mut batch = store_reader.batch_get_field_bytes_grouped(&doc_ids, target_field);
            for (i, &(orig_idx, _)) in group.iter().enumerate() {
                results[orig_idx] = batch[i].take();
            }
            seg_start = seg_end;
        }
        results
    }

    /// Batch-reads raw bytes of a **bytes** fast field, falling back to DocStore
    /// for segments that lack the columnar column.
    ///
    /// Best for fields with repeated or small values where SSTable dictionary
    /// encoding is efficient. For large unique blobs (e.g. `_source`), prefer
    /// [`batch_get_field_bytes`] which uses DocStore directly — doc_id-ordered
    /// blocks with LRU cache are faster than content-sorted SSTable ordinals.
    ///
    /// The returned `Vec` preserves the input order.
    pub fn batch_fast_field_bytes(
        &self,
        doc_addresses: &[DocAddress],
        field_name: &str,
        stored_field: Option<crate::schema::Field>,
    ) -> Vec<Option<Vec<u8>>> {
        if doc_addresses.is_empty() {
            return Vec::new();
        }

        // Pre-cache BytesColumn per segment
        let mut bytes_cols: Vec<Option<columnar::BytesColumn>> =
            Vec::with_capacity(self.inner.segment_readers.len());
        for seg_reader in &self.inner.segment_readers {
            bytes_cols.push(seg_reader.fast_fields().bytes(field_name).ok().flatten());
        }

        let mut results: Vec<Option<Vec<u8>>> = vec![None; doc_addresses.len()];

        // Group by segment for batch SSTable streaming via sorted_ords_to_term_cb
        let mut seg_groups: Vec<Vec<(usize, u32)>> =
            vec![Vec::new(); self.inner.segment_readers.len()];
        let mut docstore_fallback: Vec<(usize, DocAddress)> = Vec::new();

        for (orig_idx, addr) in doc_addresses.iter().enumerate() {
            let seg_ord = addr.segment_ord as usize;
            if bytes_cols[seg_ord].is_some() {
                seg_groups[seg_ord].push((orig_idx, addr.doc_id));
            } else if stored_field.is_some() {
                docstore_fallback.push((orig_idx, *addr));
            }
        }

        // Fast path: batch read via sorted_ords_to_term_cb (streaming SSTable scan)
        for (seg_ord, group) in seg_groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let col = bytes_cols[seg_ord].as_ref().unwrap();

            // Collect (orig_idx, term_ord) pairs, sorted by term_ord for streaming
            let mut ord_pairs: Vec<(usize, u64)> = group
                .iter()
                .filter_map(|&(orig_idx, doc_id)| {
                    col.term_ords(doc_id).next().map(|ord| (orig_idx, ord))
                })
                .collect();
            ord_pairs.sort_unstable_by_key(|&(_, ord)| ord);

            // Stream-decode all terms in one pass (no per-doc block restart)
            let mut pair_idx = 0;
            let ords_iter = ord_pairs.iter().map(|&(_, ord)| ord);
            if let Err(e) = col.dictionary().sorted_ords_to_term_cb(ords_iter, |bytes| {
                if pair_idx < ord_pairs.len() {
                    let orig_idx = ord_pairs[pair_idx].0;
                    results[orig_idx] = Some(bytes.to_vec());
                    pair_idx += 1;
                }
                Ok(())
            }) {
                log::warn!("SSTable read error in batch_fast_field_bytes: {e}");
            }
        }

        // Fallback: DocStore single-field extraction for segments without BytesColumn
        // Sort by (segment_ord, doc_id) for block cache locality
        docstore_fallback.sort_unstable_by_key(|(_, a)| (a.segment_ord, a.doc_id));
        for (orig_idx, addr) in docstore_fallback {
            if let Some(field) = stored_field {
                let store_reader = &self.inner.store_readers[addr.segment_ord as usize];
                results[orig_idx] = store_reader
                    .get_field_bytes(addr.doc_id, field)
                    .ok()
                    .flatten();
            }
        }

        results
    }

    /// Batch-reads string values from a **str** fast field, falling back to
    /// DocStore for segments that lack the columnar column.
    ///
    /// Ideal for `_id` and other keyword fields marked as FAST.
    pub fn batch_fast_field_str(
        &self,
        doc_addresses: &[DocAddress],
        field_name: &str,
        stored_field: Option<crate::schema::Field>,
    ) -> Vec<Option<String>> {
        if doc_addresses.is_empty() {
            return Vec::new();
        }

        // Pre-cache StrColumn per segment
        let mut str_cols: Vec<Option<columnar::StrColumn>> =
            Vec::with_capacity(self.inner.segment_readers.len());
        for seg_reader in &self.inner.segment_readers {
            str_cols.push(seg_reader.fast_fields().str(field_name).ok().flatten());
        }

        let mut results: Vec<Option<String>> = vec![None; doc_addresses.len()];

        // Group by segment for batch SSTable streaming
        let mut seg_groups: Vec<Vec<(usize, u32)>> =
            vec![Vec::new(); self.inner.segment_readers.len()];
        let mut docstore_fallback: Vec<(usize, DocAddress)> = Vec::new();
        for (orig_idx, addr) in doc_addresses.iter().enumerate() {
            let seg_ord = addr.segment_ord as usize;
            if str_cols[seg_ord].is_some() {
                seg_groups[seg_ord].push((orig_idx, addr.doc_id));
            } else if stored_field.is_some() {
                docstore_fallback.push((orig_idx, *addr));
            }
        }

        for (seg_ord, group) in seg_groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let col = str_cols[seg_ord].as_ref().unwrap();

            let mut ord_pairs: Vec<(usize, u64)> = group
                .iter()
                .filter_map(|&(orig_idx, doc_id)| {
                    col.term_ords(doc_id).next().map(|ord| (orig_idx, ord))
                })
                .collect();
            ord_pairs.sort_unstable_by_key(|&(_, ord)| ord);

            let mut pair_idx = 0;
            let ords_iter = ord_pairs.iter().map(|&(_, ord)| ord);
            if let Err(e) = col.dictionary().sorted_ords_to_term_cb(ords_iter, |bytes| {
                if pair_idx < ord_pairs.len() {
                    let orig_idx = ord_pairs[pair_idx].0;
                    // SSTable data is valid UTF-8 for StrColumn; zero-copy via from_utf8
                    results[orig_idx] = Some(
                        String::from_utf8(bytes.to_vec())
                            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()),
                    );
                    pair_idx += 1;
                }
                Ok(())
            }) {
                log::warn!("SSTable read error in batch_fast_field_str: {e}");
            }
        }

        // Fallback: DocStore for segments without StrColumn (old indices)
        docstore_fallback.sort_unstable_by_key(|(_, a)| (a.segment_ord, a.doc_id));
        for (orig_idx, addr) in docstore_fallback {
            if let Some(field) = stored_field {
                let store_reader = &self.inner.store_readers[addr.segment_ord as usize];
                if let Ok(Some(bytes)) = store_reader.get_field_bytes(addr.doc_id, field) {
                    results[orig_idx] = String::from_utf8(bytes).ok();
                }
            }
        }

        results
    }

    /// The cache stats for the underlying store reader.
    ///
    /// Aggregates the sum for each segment store reader.
    pub fn doc_store_cache_stats(&self) -> CacheStats {
        let cache_stats: CacheStats = self
            .inner
            .store_readers
            .iter()
            .map(|reader| reader.cache_stats())
            .sum();
        cache_stats
    }

    /// Fetches a document in an asynchronous manner.
    #[cfg(feature = "quickwit")]
    pub async fn doc_async<D: DocumentDeserialize>(
        &self,
        doc_address: DocAddress,
    ) -> crate::Result<D> {
        let executor = self.inner.index.search_executor();
        let store_reader = &self.inner.store_readers[doc_address.segment_ord as usize];
        store_reader.get_async(doc_address.doc_id, executor).await
    }

    /// Access the schema associated with the index of this searcher.
    pub fn schema(&self) -> &Schema {
        &self.inner.schema
    }

    /// Returns the overall number of documents in the index.
    pub fn num_docs(&self) -> u64 {
        self.inner
            .segment_readers
            .iter()
            .map(|segment_reader| u64::from(segment_reader.num_docs()))
            .sum::<u64>()
    }

    /// Return the overall number of documents containing
    /// the given term.
    pub fn doc_freq(&self, term: &Term) -> crate::Result<u64> {
        let mut total_doc_freq = 0;
        for segment_reader in &self.inner.segment_readers {
            let inverted_index = segment_reader.inverted_index(term.field())?;
            let doc_freq = inverted_index.doc_freq(term)?;
            total_doc_freq += u64::from(doc_freq);
        }
        Ok(total_doc_freq)
    }

    /// Return the overall number of documents containing
    /// the given term in an asynchronous manner.
    #[cfg(feature = "quickwit")]
    pub async fn doc_freq_async(&self, term: &Term) -> crate::Result<u64> {
        let mut total_doc_freq = 0;
        for segment_reader in &self.inner.segment_readers {
            let inverted_index = segment_reader.inverted_index(term.field())?;
            let doc_freq = inverted_index.doc_freq_async(term).await?;
            total_doc_freq += u64::from(doc_freq);
        }
        Ok(total_doc_freq)
    }

    /// Return the list of segment readers
    pub fn segment_readers(&self) -> &[SegmentReader] {
        &self.inner.segment_readers
    }

    /// Returns the segment_reader associated with the given segment_ord
    pub fn segment_reader(&self, segment_ord: u32) -> &SegmentReader {
        &self.inner.segment_readers[segment_ord as usize]
    }

    /// Runs a query on the segment readers wrapped by the searcher.
    ///
    /// Search works as follows :
    ///
    ///  First the weight object associated with the query is created.
    ///
    ///  Then, the query loops over the segments and for each segment :
    ///  - setup the collector and informs it that the segment being processed has changed.
    ///  - creates a SegmentCollector for collecting documents associated with the segment
    ///  - creates a `Scorer` object associated for this segment
    ///  - iterate through the matched documents and push them to the segment collector.
    ///
    ///  Finally, the Collector merges each of the child collectors into itself for result usability
    ///  by the caller.
    pub fn search<C: Collector>(
        &self,
        query: &dyn Query,
        collector: &C,
    ) -> crate::Result<C::Fruit> {
        self.search_with_statistics_provider(query, collector, self)
    }

    /// Same as [`search(...)`](Searcher::search) but allows specifying
    /// a [Bm25StatisticsProvider].
    ///
    /// This can be used to adjust the statistics used in computing BM25
    /// scores.
    pub fn search_with_statistics_provider<C: Collector>(
        &self,
        query: &dyn Query,
        collector: &C,
        statistics_provider: &dyn Bm25StatisticsProvider,
    ) -> crate::Result<C::Fruit> {
        let enabled_scoring = if collector.requires_scoring() {
            EnableScoring::enabled_from_statistics_provider(statistics_provider, self)
        } else {
            EnableScoring::disabled_from_searcher(self)
        };
        let executor = self.inner.index.search_executor();
        self.search_with_executor(query, collector, executor, enabled_scoring)
    }

    /// Same as [`search(...)`](Searcher::search) but multithreaded.
    ///
    /// The current implementation is rather naive :
    /// multithreading is by splitting search into as many task
    /// as there are segments.
    ///
    /// It is powerless at making search faster if your index consists in
    /// one large segment.
    ///
    /// Also, keep in mind multithreading a single query on several
    /// threads will not improve your throughput. It can actually
    /// hurt it. It will however, decrease the average response time.
    pub fn search_with_executor<C: Collector>(
        &self,
        query: &dyn Query,
        collector: &C,
        executor: &Executor,
        enabled_scoring: EnableScoring,
    ) -> crate::Result<C::Fruit> {
        let weight = query.weight(enabled_scoring)?;
        collector.check_schema(self.schema())?;
        let segment_readers = self.segment_readers();
        let fruits = executor.map(
            |(segment_ord, segment_reader)| {
                collector.collect_segment(weight.as_ref(), segment_ord as u32, segment_reader)
            },
            segment_readers.iter().enumerate(),
        )?;
        collector.merge_fruits(fruits)
    }

    /// Summarize total space usage of this searcher.
    pub fn space_usage(&self) -> io::Result<SearcherSpaceUsage> {
        let mut space_usage = SearcherSpaceUsage::new();
        for segment_reader in self.segment_readers() {
            space_usage.add_segment(segment_reader.space_usage()?);
        }
        Ok(space_usage)
    }
}

impl From<Arc<SearcherInner>> for Searcher {
    fn from(inner: Arc<SearcherInner>) -> Self {
        Searcher { inner }
    }
}

/// Holds a list of `SegmentReader`s ready for search.
///
/// It guarantees that the `Segment` will not be removed before
/// the destruction of the `Searcher`.
pub(crate) struct SearcherInner {
    schema: Schema,
    index: Index,
    segment_readers: Vec<SegmentReader>,
    store_readers: Vec<StoreReader>,
    generation: TrackedObject<SearcherGeneration>,
}

impl SearcherInner {
    /// Creates a new `Searcher`
    pub(crate) fn new(
        schema: Schema,
        index: Index,
        segment_readers: Vec<SegmentReader>,
        generation: TrackedObject<SearcherGeneration>,
        doc_store_cache_num_blocks: usize,
    ) -> io::Result<SearcherInner> {
        assert_eq!(
            &segment_readers
                .iter()
                .map(|reader| (reader.segment_id(), reader.delete_opstamp()))
                .collect::<BTreeMap<_, _>>(),
            generation.segments(),
            "Set of segments referenced by this Searcher and its SearcherGeneration must match"
        );
        let store_readers: Vec<StoreReader> = segment_readers
            .iter()
            .map(|segment_reader| segment_reader.get_store_reader(doc_store_cache_num_blocks))
            .collect::<io::Result<Vec<_>>>()?;

        Ok(SearcherInner {
            schema,
            index,
            segment_readers,
            store_readers,
            generation,
        })
    }
}

impl fmt::Debug for Searcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let segment_ids = self
            .segment_readers()
            .iter()
            .map(SegmentReader::segment_id)
            .collect::<Vec<_>>();
        write!(f, "Searcher({segment_ids:?})")
    }
}

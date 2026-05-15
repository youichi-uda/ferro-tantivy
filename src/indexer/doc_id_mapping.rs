//! This module is used when sorting the index by a property, e.g.
//! to get mappings from old doc_id to new doc_id and vice versa, after sorting

use common::ReadOnlyBitSet;

use crate::DocAddress;

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum MappingType {
    /// Input segments concatenated in input order; merged segment doc_id
    /// equals the position in the concatenation.  No deletes, no reorder.
    /// The columnar merger uses the fast `StackMergeOrder` path here.
    Stacked,
    /// Input segments concatenated in input order, but with deleted docs
    /// skipped.  The columnar merger uses `ShuffleMergeOrder` to honour
    /// the alive bitsets per source segment.
    StackedWithDeletes,
    /// **FerroSearch Wave 15 Phase H-2.**  Documents are reordered so the
    /// merged segment's doc_id sequence matches the index-time sort
    /// (`IndexSettings::sort_by_field`).  After this merge, the existing
    /// `SortByStaticFastValue` path's WARM-cache + SIMD top-K threshold
    /// filter naturally early-terminates because per-doc-id iteration
    /// order coincides with the sort order, making the heap fill with the
    /// K extreme values immediately and the SIMD `mask == 0` block-skip
    /// kick in for every subsequent block.  The store path iterates in
    /// new-doc-id order (no `stack` optimisation, since the on-disk
    /// segment layout is no longer aligned to source segment boundaries).
    Sorted,
}

/// Struct to provide mapping from new doc_id to old doc_id and segment.
#[derive(Clone)]
pub(crate) struct SegmentDocIdMapping {
    pub(crate) new_doc_id_to_old_doc_addr: Vec<DocAddress>,
    pub(crate) alive_bitsets: Vec<Option<ReadOnlyBitSet>>,
    mapping_type: MappingType,
}

impl SegmentDocIdMapping {
    pub(crate) fn new(
        new_doc_id_to_old_doc_addr: Vec<DocAddress>,
        mapping_type: MappingType,
        alive_bitsets: Vec<Option<ReadOnlyBitSet>>,
    ) -> Self {
        Self {
            new_doc_id_to_old_doc_addr,
            mapping_type,
            alive_bitsets,
        }
    }

    pub fn mapping_type(&self) -> MappingType {
        self.mapping_type
    }

    /// Returns an iterator over the old document addresses, ordered by the new document ids.
    ///
    /// In the returned `DocAddress`, the `segment_ord` is the ordinal of targeted segment
    /// in the list of merged segments.
    pub(crate) fn iter_old_doc_addrs(&self) -> impl Iterator<Item = DocAddress> + '_ {
        self.new_doc_id_to_old_doc_addr.iter().copied()
    }
}

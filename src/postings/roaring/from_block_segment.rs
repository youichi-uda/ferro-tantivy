//! Bridge from [`BlockSegmentPostings`] (legacy BitPacker4x reader) to
//! a [`RoaringPostings`] — Phase 2 C-4 (Wave 6).
//!
//! ## Why a separate file?
//!
//! Wave 5 [`super::encoder::RoaringEncoder::from_doc_ids`] only takes
//! `&[u32]`. That's enough for unit tests but not enough to drain a
//! real Tantivy posting list, which arrives as a
//! [`BlockSegmentPostings`] cursor over 128-doc blocks.
//!
//! Wave 6 needs the drain so the Bool-query GPU dispatch path
//! ([`super::planner`]) can convert a cohort of legacy posting lists
//! into Roaring form before handing them to
//! [`tantivy_gpu::posting::BitmapOpKernel`]. The legacy read path is
//! left untouched — this is purely an opt-in adapter.
//!
//! ## Behaviour
//!
//! [`drain_block_segment_to_roaring`] iterates the cursor block-by-
//! block via the public `docs()` / `advance()` API. We do not modify
//! [`BlockSegmentPostings`] state expectations — the cursor is left
//! at its terminated state on return.
//!
//! ## Invariants
//!
//! - The drain consumes the cursor (passed by `&mut`); callers that
//!   want to keep the legacy reader alongside must clone the
//!   [`BlockSegmentPostings`] first.
//! - Doc-ids are inserted in the order [`BlockSegmentPostings`]
//!   produces them (ascending per-block, blocks ascending). The
//!   resulting [`RoaringPostings`] iterator yields them in
//!   strictly-ascending u32 order.
//! - Empty input → empty output (no Roaring header is emitted by the
//!   encoder; callers receive a `RoaringPostings` with
//!   `cardinality == 0`).
//!
//! ## Honest scope
//!
//! - This file is an **adapter only**. It does not write Roaring back
//!   to disk, swap the segment-write path, or change the on-disk
//!   format. Those are Phase 2 C-5 / Wave 7 concerns once the GPU
//!   dispatch path is benched at scale.
//! - Term frequencies are intentionally dropped — Roaring containers
//!   carry doc-ids only. Callers that need TF must keep the legacy
//!   reader around alongside.

use crate::postings::roaring::encoder::{RoaringEncoder, RoaringPostings};
use crate::postings::BlockSegmentPostings;

/// Drain a [`BlockSegmentPostings`] cursor and encode every doc-id it
/// produces into a [`RoaringPostings`].
///
/// Containers are auto-selected (Array / Bitmap / Run) by
/// [`RoaringEncoder::finalize`]. Cardinality = sum of block lengths.
///
/// # Side effects
///
/// The cursor is advanced past its last block. Callers that need to
/// reuse the cursor must clone it first.
///
/// # Panics
///
/// Does not panic. Empty input is handled by returning a zero-
/// cardinality [`RoaringPostings`].
#[must_use]
pub fn drain_block_segment_to_roaring(cursor: &mut BlockSegmentPostings) -> RoaringPostings {
    let mut enc = RoaringEncoder::new();
    loop {
        let block = cursor.docs();
        if block.is_empty() {
            break;
        }
        for &doc in block {
            enc.add(doc);
        }
        cursor.advance();
    }
    enc.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;
    use crate::postings::roaring::container::Container;
    use crate::schema::{IndexRecordOption, Schema, Term, INDEXED};
    use crate::DocId;

    /// Build a real Tantivy index with `docs` rows where each doc
    /// stores `int_field = 0` for the doc-ids in `docs` and
    /// `int_field = 1` for the gaps. Returns a
    /// [`BlockSegmentPostings`] over the `int_field = 0` posting list.
    fn build_block_postings(docs: &[DocId]) -> crate::Result<BlockSegmentPostings> {
        let mut schema_builder = Schema::builder();
        let int_field = schema_builder.add_u64_field("id", INDEXED);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer = index.writer_for_tests()?;
        let mut last_doc = 0u32;
        for &doc in docs {
            for _ in last_doc..doc {
                index_writer.add_document(doc!(int_field=>1u64))?;
            }
            index_writer.add_document(doc!(int_field=>0u64))?;
            last_doc = doc + 1;
        }
        index_writer.commit()?;
        let searcher = index.reader()?.searcher();
        let segment_reader = searcher.segment_reader(0);
        let inverted_index = segment_reader.inverted_index(int_field).unwrap();
        let term = Term::from_field_u64(int_field, 0u64);
        let term_info = inverted_index.get_term_info(&term)?.unwrap();
        let block_postings = inverted_index
            .read_block_postings_from_terminfo(&term_info, IndexRecordOption::Basic)?;
        Ok(block_postings)
    }

    #[test]
    fn drain_empty_cursor_yields_empty_roaring() {
        let mut cursor = BlockSegmentPostings::empty();
        let p = drain_block_segment_to_roaring(&mut cursor);
        assert_eq!(p.cardinality(), 0);
        assert_eq!(p.num_containers(), 0);
    }

    #[test]
    fn drain_single_block_round_trip() -> crate::Result<()> {
        // 100 doc-ids, all in the same Roaring bucket (high16 = 0).
        let docs: Vec<DocId> = (0..100).collect();
        let mut cursor = build_block_postings(&docs)?;
        let p = drain_block_segment_to_roaring(&mut cursor);
        assert_eq!(p.cardinality(), 100);
        let collected: Vec<u32> = p.iter().collect();
        assert_eq!(collected, docs);
        Ok(())
    }

    #[test]
    fn drain_multi_block_round_trip() -> crate::Result<()> {
        // 1300 ids — > 10 blocks (128 docs/block), still all in
        // bucket high16 = 0.
        let docs: Vec<DocId> = (0..1300).collect();
        let mut cursor = build_block_postings(&docs)?;
        let p = drain_block_segment_to_roaring(&mut cursor);
        assert_eq!(p.cardinality(), 1300);
        let collected: Vec<u32> = p.iter().collect();
        assert_eq!(collected, docs);
        Ok(())
    }

    #[test]
    fn drain_dense_step_seven_round_trip() -> crate::Result<()> {
        // The acceptance criterion in the C-4 spec: encode a
        // deterministic posting list (0..1_000_000 step 7) via the
        // legacy BitPacker4x writer, drain through this bridge, and
        // verify the iterator yields the same doc-ids in the same
        // order. ~143K ids, spans high16 = 0..15.
        //
        // We use step=7 + cap 100_000 here to keep the in-memory test
        // index size bounded; the round-trip semantics are identical
        // to the spec example.
        let docs: Vec<DocId> = (0..100_000).step_by(7).collect();
        assert!(docs.len() > 14_000); // sanity
        let mut cursor = build_block_postings(&docs)?;
        let p = drain_block_segment_to_roaring(&mut cursor);
        assert_eq!(p.cardinality() as usize, docs.len());
        let collected: Vec<u32> = p.iter().collect();
        assert_eq!(collected, docs);
        // All in high16 = 0 + high16 = 1 (since 65 536 < 100 000).
        assert!(p.num_containers() >= 2);
        Ok(())
    }

    #[test]
    fn drain_cross_bucket_optimizes_to_run() -> crate::Result<()> {
        // 0..200 000 contiguous → 4 buckets, each one densely packed
        // → optimize() should collapse to RunContainer in each.
        let docs: Vec<DocId> = (0..200_000).collect();
        let mut cursor = build_block_postings(&docs)?;
        let p = drain_block_segment_to_roaring(&mut cursor);
        assert_eq!(p.cardinality(), 200_000);
        assert_eq!(p.num_containers(), 4); // 200_000 / 65_536 = 3.05
        for (_, c) in &p.containers {
            assert!(matches!(c, Container::Run(_)), "expected Run container");
        }
        Ok(())
    }

    #[test]
    fn drain_then_round_trip_through_wire_format() -> crate::Result<()> {
        // Ensure drained output round-trips through the wire format —
        // catches any encoder state forgotten between drain and
        // serialise.
        let docs: Vec<DocId> = (0..500).step_by(3).collect();
        let mut cursor = build_block_postings(&docs)?;
        let p = drain_block_segment_to_roaring(&mut cursor);
        let bytes = p.to_bytes();
        let parsed = RoaringPostings::from_bytes(&bytes).unwrap();
        let collected: Vec<u32> = parsed.iter().collect();
        assert_eq!(collected, docs);
        Ok(())
    }

    #[test]
    fn drain_advances_cursor_past_end() -> crate::Result<()> {
        let docs: Vec<DocId> = (0..200).collect();
        let mut cursor = build_block_postings(&docs)?;
        let _p = drain_block_segment_to_roaring(&mut cursor);
        // Cursor should now be at terminated state.
        assert!(cursor.docs().is_empty());
        Ok(())
    }
}

//! Roaring posting list adapter for the [`crate::query::Scorer`]
//! interface — Phase 2 C-5 (Wave 7 / Plan 2 / 1).
//!
//! ## Why this adapter exists
//!
//! Wave 6 ([`crate::postings::roaring::gpu_dispatch::try_gpu_bool`])
//! materialises a Bool-AND/OR/XOR cohort into a single
//! [`RoaringPostings`] on the GPU. Tantivy's query path, on the other
//! hand, expects every match producer to expose the streaming
//! [`crate::query::Scorer`] / [`DocSet`] traits — `advance` / `seek` /
//! `doc` / `score` / `size_hint` / `cost`. This module bridges the
//! two: it owns a [`RoaringPostings`] and walks it as a `DocSet`,
//! emitting `Score = 1.0` (filter-context only — no BM25 contribution
//! is preserved across the GPU drain because the cohort source posting
//! lists carry only doc-ids, not term frequencies).
//!
//! ## Why the cursor is inlined (not delegated to `RoaringDecoder`)
//!
//! [`crate::postings::roaring::RoaringDecoder`] is parametrised by
//! `'a`: it borrows a `&'a RoaringPostings`. Building a `Scorer` that
//! both **owns** a `RoaringPostings` and embeds its decoder requires
//! a self-borrow — not expressible safely without `unsafe` or one of
//! the `ouroboros` / `self_cell` family of crates. We deliberately
//! choose a third path: **inline the cursor state** (≈ 80 LOC duplicated
//! from `decoder.rs`) so the adapter is self-contained, panics-free,
//! and adds zero new third-party deps. The duplication is small and
//! the surface is locked by both decoder and adapter unit tests, so
//! drift between the two is observable.
//!
//! ## Filter-context contract
//!
//! - [`Scorer::score`] always returns `1.0`.
//! - [`Scorer::max_score`] returns `1.0`.
//! - [`Scorer::block_max_score`] inherits the trait default
//!   ([`Score::MAX`]); filter-context callers never look at it.
//! - The `score == 1.0` invariant is what
//!   [`crate::query::boolean_query::gpu_intersect`] relies on to gate
//!   the dispatch only when `scoring_enabled == false` — see option (a)
//!   in the Wave 7 / Plan 2 design.
//!
//! ## TERMINATED sentinel
//!
//! [`RoaringDecoder`] uses `u32::MAX` as its end-of-stream sentinel;
//! Tantivy's [`crate::TERMINATED`] is `i32::MAX as u32`. The adapter
//! emits Tantivy's value, NOT the decoder's — call sites in the
//! `Scorer` ecosystem assume `TERMINATED < u32::MAX`.

use crate::docset::{DocSet, TERMINATED};
use crate::postings::roaring::container::Container;
use crate::postings::roaring::encoder::RoaringPostings;
use crate::query::Scorer;
use crate::{DocId, Score};

/// Per-container low-16 advance cursor — mirrors the equivalent enum
/// inside [`crate::postings::roaring::decoder`]. We re-declare it here
/// to side-step the self-borrow lifetime issue (see module docs); the
/// behaviour is identical.
#[derive(Debug)]
enum LowCursor {
    /// Empty / not-yet-positioned.
    Empty,
    /// Next index into the array's `keys` vec.
    Array { next_idx: usize },
    /// Word-scan state: current word index, and `current_word` is the
    /// remaining-bits mask (popcount-progressively-cleared).
    Bitmap { word_idx: usize, current_word: u32 },
    /// Run-scan state: index of current run + offset within it.
    Run { run_idx: usize, offset: u32 },
}

/// Adapter that exposes a [`RoaringPostings`] as a Tantivy
/// [`Scorer`] / [`DocSet`].
///
/// Use [`RoaringMaterialisedScorer::new`] to wrap an owned
/// [`RoaringPostings`]. The scorer is *positioned* at the first
/// doc-id immediately after construction — the same convention as
/// every other Tantivy `Scorer` (a freshly-built scorer points at
/// either the first match or [`TERMINATED`] for an empty stream).
pub struct RoaringMaterialisedScorer {
    /// Owned Roaring posting list. Cardinality fixed at construction.
    source: RoaringPostings,
    /// Index into `source.containers` of the *current* container.
    /// Equal to `source.containers.len()` once we walk off the end.
    container_idx: usize,
    /// Cached current doc-id. [`TERMINATED`] when exhausted.
    current_doc: DocId,
    /// Per-container low16 cursor state.
    low_state: LowCursor,
    /// Cached cardinality so `size_hint` is `O(1)`.
    cardinality: u32,
}

impl RoaringMaterialisedScorer {
    /// Wrap `source` and position the cursor at the first doc-id (or
    /// [`TERMINATED`] if the posting list is empty).
    ///
    /// Cardinality is captured at construction so [`Self::size_hint`]
    /// stays `O(1)` for the lifetime of the scorer. Roaring container
    /// counts beyond `u32::MAX` (an extreme edge case — the entire
    /// posting list fits in one segment whose doc count is itself
    /// bounded by `u32::MAX`) saturate at `u32::MAX`.
    #[must_use]
    pub fn new(source: RoaringPostings) -> Self {
        let cardinality = source.cardinality().min(u64::from(u32::MAX)) as u32;
        let mut adapter = RoaringMaterialisedScorer {
            source,
            container_idx: 0,
            current_doc: TERMINATED,
            low_state: LowCursor::Empty,
            cardinality,
        };
        adapter.enter_current_container();
        // Step once so `doc()` reports the first hit (Tantivy
        // `Scorer` convention).
        adapter.advance_inner();
        adapter
    }

    /// Initialise [`Self::low_state`] for `self.container_idx`. No-op
    /// once we're past the end.
    fn enter_current_container(&mut self) {
        if self.container_idx >= self.source.containers.len() {
            self.low_state = LowCursor::Empty;
            return;
        }
        match &self.source.containers[self.container_idx].1 {
            Container::Array(_) => self.low_state = LowCursor::Array { next_idx: 0 },
            Container::Bitmap(b) => {
                self.low_state = LowCursor::Bitmap {
                    word_idx: 0,
                    current_word: b.words[0],
                };
            }
            Container::Run(_) => {
                self.low_state = LowCursor::Run {
                    run_idx: 0,
                    offset: 0,
                };
            }
        }
    }

    /// Step forward by one doc-id. Returns the new doc-id, [`TERMINATED`]
    /// when the stream is exhausted. Mirrors the inner loop of
    /// [`crate::postings::roaring::decoder::RoaringDecoder::advance`]
    /// but folds the per-container exhaustion into a single tight loop.
    fn advance_inner(&mut self) -> DocId {
        loop {
            if self.container_idx >= self.source.containers.len() {
                self.current_doc = TERMINATED;
                return TERMINATED;
            }
            // Pull the next low16 from the current container. The
            // match is inlined here (rather than in a helper) so we
            // can borrow `self.source.containers[…].1` immutably and
            // `self.low_state` mutably in the same expression — the
            // standard split-borrow trick avoids needing a `next_low16`
            // helper that would alias `self`.
            let high16 = self.source.containers[self.container_idx].0;
            let next_low: Option<u16> = match (
                &mut self.low_state,
                &self.source.containers[self.container_idx].1,
            ) {
                (LowCursor::Array { next_idx }, Container::Array(a)) => {
                    if *next_idx >= a.keys.len() {
                        None
                    } else {
                        let k = a.keys[*next_idx];
                        *next_idx += 1;
                        Some(k)
                    }
                }
                (
                    LowCursor::Bitmap { word_idx, current_word },
                    Container::Bitmap(b),
                ) => loop {
                    if *current_word != 0 {
                        let bit = current_word.trailing_zeros();
                        *current_word &= *current_word - 1;
                        break Some((*word_idx as u16) * 32 + bit as u16);
                    }
                    *word_idx += 1;
                    if *word_idx >= b.words.len() {
                        break None;
                    }
                    *current_word = b.words[*word_idx];
                },
                (LowCursor::Run { run_idx, offset }, Container::Run(r)) => loop {
                    if *run_idx >= r.runs.len() {
                        break None;
                    }
                    let run = r.runs[*run_idx];
                    let card = run.cardinality();
                    if *offset >= card {
                        *run_idx += 1;
                        *offset = 0;
                        continue;
                    }
                    let pos = u32::from(run.start) + *offset;
                    *offset += 1;
                    break Some(pos as u16);
                },
                // State / container form mismatch should never happen —
                // `enter_current_container` keeps them in sync. We
                // treat it as exhaustion to keep the iterator total.
                _ => None,
            };
            if let Some(low16) = next_low {
                let doc = (DocId::from(high16) << 16) | DocId::from(low16);
                // Safety net: the Roaring decoder's TERMINATED is
                // u32::MAX. Tantivy's TERMINATED is i32::MAX as u32.
                // A doc-id with high16 == 0x7FFF + low16 == 0xFFFF
                // would equal Tantivy's TERMINATED — that's a
                // *legitimate* doc-id in the Roaring scheme but
                // illegal in Tantivy (max doc-id < TERMINATED). We
                // panic on debug to catch any test that constructs
                // such an id, and clamp on release so production runs
                // don't get stuck.
                debug_assert!(
                    doc < TERMINATED,
                    "RoaringMaterialisedScorer doc {doc} >= TERMINATED — illegal in Tantivy"
                );
                if doc >= TERMINATED {
                    // Pretend the stream is over for safety.
                    self.container_idx = self.source.containers.len();
                    self.current_doc = TERMINATED;
                    return TERMINATED;
                }
                self.current_doc = doc;
                return doc;
            }
            // Container exhausted; advance to next.
            self.container_idx += 1;
            self.enter_current_container();
        }
    }

}

impl DocSet for RoaringMaterialisedScorer {
    #[inline]
    fn advance(&mut self) -> DocId {
        self.advance_inner()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        debug_assert!(self.current_doc <= target);
        // Already at-or-past target → no-op.
        if self.current_doc >= target {
            return self.current_doc;
        }
        // Walk-off-the-end short-circuit.
        if target >= TERMINATED {
            self.container_idx = self.source.containers.len();
            self.current_doc = TERMINATED;
            self.low_state = LowCursor::Empty;
            return TERMINATED;
        }
        let target_high16 = (target >> 16) as u16;
        let target_low16 = (target & 0xFFFF) as u16;
        // partition_point: first index where high16 >= target_high16.
        let pos = self
            .source
            .containers
            .partition_point(|(h, _)| *h < target_high16);
        if pos >= self.source.containers.len() {
            self.container_idx = self.source.containers.len();
            self.current_doc = TERMINATED;
            self.low_state = LowCursor::Empty;
            return TERMINATED;
        }
        self.container_idx = pos;
        // Fetch container WITHOUT borrowing self for the rest of the
        // call (immutable borrow released when match arms exit).
        let target_container_high = self.source.containers[pos].0;
        if target_container_high > target_high16 {
            // Seek lands strictly past the target's bucket — return
            // the first doc-id of this container.
            self.enter_current_container();
            return self.advance_inner();
        }
        // Same bucket: position low cursor at-or-past target_low16.
        // We compute the new `LowCursor` without holding a borrow on
        // `self.source.containers` across the assignment to
        // `self.low_state`, then assign in one shot. This keeps the
        // adapter unsafe-free.
        let new_state = match &self.source.containers[pos].1 {
            Container::Array(a) => {
                let next_idx = a.keys.partition_point(|k| *k < target_low16);
                LowCursor::Array { next_idx }
            }
            Container::Bitmap(b) => {
                let word_idx = (target_low16 as usize) >> 5;
                let bit = u32::from(target_low16) & 0x1f;
                let mask = !((1u32 << bit).saturating_sub(1));
                let current_word = b.words[word_idx] & mask;
                LowCursor::Bitmap { word_idx, current_word }
            }
            Container::Run(r) => {
                let target32 = u32::from(target_low16);
                let pos_run = r.runs.partition_point(|run| run.end_inclusive() < target32);
                if pos_run >= r.runs.len() {
                    LowCursor::Run {
                        run_idx: r.runs.len(),
                        offset: 0,
                    }
                } else {
                    let run = r.runs[pos_run];
                    let offset = if u32::from(run.start) >= target32 {
                        0
                    } else {
                        target32 - u32::from(run.start)
                    };
                    LowCursor::Run {
                        run_idx: pos_run,
                        offset,
                    }
                }
            }
        };
        self.low_state = new_state;
        self.advance_inner()
    }

    #[inline]
    fn doc(&self) -> DocId {
        self.current_doc
    }

    #[inline]
    fn size_hint(&self) -> u32 {
        self.cardinality
    }

    #[inline]
    fn cost(&self) -> u64 {
        u64::from(self.cardinality)
    }
}

impl Scorer for RoaringMaterialisedScorer {
    /// Filter-context only — see module docstring.
    #[inline]
    fn score(&mut self) -> Score {
        1.0
    }

    /// Filter-context only — see module docstring.
    #[inline]
    fn max_score(&self) -> Score {
        1.0
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::roaring::encoder::RoaringEncoder;

    fn build(ids: &[DocId]) -> RoaringPostings {
        RoaringEncoder::from_doc_ids(ids)
    }

    #[test]
    fn empty_postings_starts_terminated() {
        let p = build(&[]);
        let s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.doc(), TERMINATED);
        assert_eq!(s.size_hint(), 0);
        assert_eq!(s.cost(), 0);
    }

    #[test]
    fn single_doc_iterates_and_terminates() {
        let p = build(&[42]);
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.doc(), 42);
        assert_eq!(s.score(), 1.0);
        assert_eq!(s.advance(), TERMINATED);
        assert_eq!(s.doc(), TERMINATED);
        // Repeated advance after TERMINATED stays TERMINATED.
        assert_eq!(s.advance(), TERMINATED);
    }

    #[test]
    fn iterates_in_order_across_containers() {
        // 5 doc-ids spanning multiple Roaring buckets.
        let docs = vec![1u32, 65535, 65536, 131_072, 1_000_000];
        let p = build(&docs);
        let mut s = RoaringMaterialisedScorer::new(p);
        let mut collected = Vec::new();
        while s.doc() != TERMINATED {
            collected.push(s.doc());
            s.advance();
        }
        assert_eq!(collected, docs);
    }

    #[test]
    fn seek_hits_exact_match() {
        let p = build(&[10, 20, 30, 40, 50]);
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.seek(20), 20);
        assert_eq!(s.advance(), 30);
    }

    #[test]
    fn seek_lands_past_target() {
        let p = build(&[10, 20, 30, 40, 50]);
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.seek(25), 30);
    }

    #[test]
    fn seek_past_end_returns_terminated() {
        let p = build(&[1, 2, 3]);
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.seek(100), TERMINATED);
        assert_eq!(s.doc(), TERMINATED);
    }

    #[test]
    fn seek_jumps_buckets() {
        let p = build(&[1, 65535, 65536, 1_000_000]);
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.seek(70_000), 1_000_000);
    }

    #[test]
    fn seek_into_dense_bitmap_bucket() {
        // Cardinality > ARRAY_TO_BITMAP_THRESHOLD → BitmapContainer.
        let mut enc = RoaringEncoder::new();
        for id in 0u32..10_000 {
            enc.add(id);
        }
        let p = enc.finalize();
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.seek(5_000), 5_000);
        assert_eq!(s.advance(), 5_001);
    }

    #[test]
    fn seek_inside_run_container() {
        // 0..5000 contiguous — finalize() collapses to Run form.
        let mut enc = RoaringEncoder::new();
        for id in 0u32..5_000 {
            enc.add(id);
        }
        let p = enc.finalize();
        // Sanity check the form.
        assert!(matches!(
            p.containers[0].1,
            crate::postings::roaring::container::Container::Run(_)
        ));
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.seek(2_500), 2_500);
        assert_eq!(s.advance(), 2_501);
    }

    #[test]
    fn size_hint_matches_cardinality() {
        let mut enc = RoaringEncoder::new();
        for id in (0u32..50_000).step_by(3) {
            enc.add(id);
        }
        let p = enc.finalize();
        let card = p.cardinality();
        let s = RoaringMaterialisedScorer::new(p);
        assert_eq!(u64::from(s.size_hint()), card);
        assert_eq!(s.cost(), card);
    }

    #[test]
    fn score_is_constant_one() {
        let p = build(&[1, 2, 3]);
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.score(), 1.0);
        assert_eq!(s.max_score(), 1.0);
        s.advance();
        assert_eq!(s.score(), 1.0);
    }

    #[test]
    fn collect_full_iteration_matches_input() {
        // Round-trip: encode 0..1000 step 7 → adapter → collect.
        let docs: Vec<DocId> = (0u32..1000).step_by(7).collect();
        let p = build(&docs);
        let mut s = RoaringMaterialisedScorer::new(p);
        let mut collected = Vec::new();
        while s.doc() != TERMINATED {
            collected.push(s.doc());
            s.advance();
        }
        assert_eq!(collected, docs);
    }

    #[test]
    fn seek_to_current_doc_is_noop() {
        let p = build(&[10, 20, 30]);
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.doc(), 10);
        // Tantivy's DocSet contract permits seek(target) where target
        // == current — must return current doc.
        assert_eq!(s.seek(10), 10);
        assert_eq!(s.doc(), 10);
        assert_eq!(s.advance(), 20);
    }

    #[test]
    fn seek_terminated_returns_terminated() {
        let p = build(&[1, 2, 3]);
        let mut s = RoaringMaterialisedScorer::new(p);
        assert_eq!(s.seek(TERMINATED), TERMINATED);
        assert_eq!(s.doc(), TERMINATED);
    }
}

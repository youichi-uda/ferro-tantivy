//! Roaring posting list decoder — Phase 2 C-3.
//!
//! Wraps a [`RoaringPostings`] in an iterator + galloping `seek` API
//! shaped after Tantivy's [`crate::docset::DocSet`] (without
//! implementing that trait directly — wiring into Tantivy's term-info
//! plumbing is a Phase 2 C-4 job).
//!
//! The decoder advances through containers in `high16` order. `seek`
//! short-circuits across whole containers via a `partition_point` on
//! the sorted `(high16, container)` table, then jumps into the
//! per-container low16 stream with a per-form fast path:
//!
//! - [`Container::Array`] → `binary_search` on the sorted `keys` vec
//! - [`Container::Bitmap`] → bit-scan from the target word
//! - [`Container::Run`] → `partition_point` on run starts + offset inside the matched run
//!
//! This keeps `seek(target)` at `O(log num_containers + log
//! container_size)` worst-case instead of the linear scan a naive
//! iterator would do.

use crate::postings::roaring::container::Container;
use crate::postings::roaring::encoder::RoaringPostings;

/// Sentinel returned by [`RoaringDecoder::doc`] before the first
/// `advance()` and after the last hit. Matches Tantivy's
/// [`crate::TERMINATED`] convention so call sites can be bridged
/// straightforwardly in C-4.
pub const TERMINATED: u32 = u32::MAX;

/// Iterator + seek decoder over a [`RoaringPostings`].
///
/// Build via [`RoaringDecoder::new`]. `advance()` returns the next
/// doc-id or [`None`] (= [`TERMINATED`]); `seek(target)` jumps to the
/// first doc-id ≥ `target`.
pub struct RoaringDecoder<'a> {
    /// Backing posting list. Borrowed for the decoder's lifetime so
    /// the per-container iterator state is cheap to spin up / tear
    /// down across `seek` calls.
    source: &'a RoaringPostings,
    /// Index into `source.containers` of the *current* container.
    /// Equal to `source.containers.len()` when we've walked off the
    /// end.
    container_idx: usize,
    /// Cached current doc-id. `TERMINATED` before the first
    /// `advance()` / after end-of-stream.
    current_doc: u32,
    /// Per-container low16 cursor state. Re-built on every container
    /// boundary so we never carry stale offsets across a `seek`.
    low_state: LowCursor,
}

/// Per-container low-16 advance state.
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

impl<'a> RoaringDecoder<'a> {
    /// Build a decoder positioned *before* the first doc-id. The
    /// caller must call [`Self::advance`] (or [`Self::seek`]) to
    /// consume the first hit.
    #[must_use]
    pub fn new(source: &'a RoaringPostings) -> Self {
        let mut dec = RoaringDecoder {
            source,
            container_idx: 0,
            current_doc: TERMINATED,
            low_state: LowCursor::Empty,
        };
        dec.enter_current_container();
        dec
    }

    /// Initialise [`Self::low_state`] for `self.container_idx`. No-op
    /// if we're past the end.
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
                }
            }
            Container::Run(_) => {
                self.low_state = LowCursor::Run {
                    run_idx: 0,
                    offset: 0,
                }
            }
        }
    }

    /// Current doc-id ([`TERMINATED`] if not started or exhausted).
    #[inline]
    #[must_use]
    pub fn doc(&self) -> u32 {
        self.current_doc
    }

    /// Total cardinality of the underlying [`RoaringPostings`].
    #[inline]
    #[must_use]
    pub fn cardinality(&self) -> u64 {
        self.source.cardinality()
    }

    /// Step forward by one doc-id. Returns the new doc-id, or [`None`]
    /// when the stream is exhausted.
    pub fn advance(&mut self) -> Option<u32> {
        loop {
            if self.container_idx >= self.source.containers.len() {
                self.current_doc = TERMINATED;
                return None;
            }
            let (high16, container) = &self.source.containers[self.container_idx];
            if let Some(low16) = self.next_low16(container) {
                let doc = (u32::from(*high16) << 16) | u32::from(low16);
                self.current_doc = doc;
                return Some(doc);
            }
            // Container exhausted; advance to next.
            self.container_idx += 1;
            self.enter_current_container();
        }
    }

    /// Pull the next low16 from the current container, given the
    /// active [`LowCursor`]. Returns [`None`] when the container is
    /// exhausted.
    fn next_low16(&mut self, container: &Container) -> Option<u16> {
        match (&mut self.low_state, container) {
            (LowCursor::Array { next_idx }, Container::Array(a)) => {
                if *next_idx >= a.keys.len() {
                    return None;
                }
                let k = a.keys[*next_idx];
                *next_idx += 1;
                Some(k)
            }
            (
                LowCursor::Bitmap {
                    word_idx,
                    current_word,
                },
                Container::Bitmap(b),
            ) => loop {
                if *current_word != 0 {
                    let bit = current_word.trailing_zeros();
                    *current_word &= *current_word - 1;
                    return Some((*word_idx as u16) * 32 + bit as u16);
                }
                *word_idx += 1;
                if *word_idx >= b.words.len() {
                    return None;
                }
                *current_word = b.words[*word_idx];
            },
            (LowCursor::Run { run_idx, offset }, Container::Run(r)) => loop {
                if *run_idx >= r.runs.len() {
                    return None;
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
                return Some(pos as u16);
            },
            // State / container form mismatch should never happen —
            // `enter_current_container` keeps them in sync.
            _ => None,
        }
    }

    /// Seek to the first doc-id ≥ `target`. Returns the new doc-id or
    /// [`None`] if `target` exceeds every doc-id in the stream.
    ///
    /// Two-stage galloping search:
    /// 1. Binary-search the `(high16, container)` table for the container that *might* contain
    ///    `target` (i.e. the first container whose `high16` ≥ `target_high16`).
    /// 2. If we land on the bucket that holds `target_high16`, binary-search within that
    ///    container's low16 form. Otherwise enter the bucket fresh (its first key is ≥ `target` by
    ///    construction).
    pub fn seek(&mut self, target: u32) -> Option<u32> {
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
            return None;
        }
        self.container_idx = pos;
        let (high16, container) = &self.source.containers[pos];
        if *high16 > target_high16 {
            // Seek lands strictly past the target's bucket — return
            // the first doc-id of this container.
            self.enter_current_container();
            return self.advance();
        }
        // *high16 == target_high16: position the low cursor at-or-past
        // target_low16 inside this container.
        self.position_within(container, target_low16)?;
        // Now `next_low16` will yield the first hit ≥ target_low16.
        self.advance()
    }

    /// Position the low cursor so the next [`Self::next_low16`] call
    /// returns the first key ≥ `target_low16`. Returns [`Some(())`]
    /// always; the [`None`] return type aligns with the call-site
    /// `?` flow but the only failure path (= "no key in this
    /// bucket ≥ target_low16") is signalled by the cursor landing
    /// past the end of the container, in which case the subsequent
    /// `advance()` walks into the next container.
    fn position_within(&mut self, container: &Container, target_low16: u16) -> Option<()> {
        match container {
            Container::Array(a) => {
                let next_idx = a.keys.partition_point(|k| *k < target_low16);
                self.low_state = LowCursor::Array { next_idx };
            }
            Container::Bitmap(b) => {
                let word_idx = (target_low16 as usize) >> 5;
                let bit = u32::from(target_low16) & 0x1f;
                // Mask off bits below `bit` in the current word.
                let mask = !((1u32 << bit).saturating_sub(1));
                let current_word = b.words[word_idx] & mask;
                self.low_state = LowCursor::Bitmap {
                    word_idx,
                    current_word,
                };
            }
            Container::Run(r) => {
                // Find the first run whose end ≥ target_low16.
                let target32 = u32::from(target_low16);
                let pos = r.runs.partition_point(|run| run.end_inclusive() < target32);
                if pos >= r.runs.len() {
                    self.low_state = LowCursor::Run {
                        run_idx: r.runs.len(),
                        offset: 0,
                    };
                } else {
                    let run = r.runs[pos];
                    let offset = if u32::from(run.start) >= target32 {
                        0
                    } else {
                        target32 - u32::from(run.start)
                    };
                    self.low_state = LowCursor::Run {
                        run_idx: pos,
                        offset,
                    };
                }
            }
        }
        Some(())
    }
}

impl Iterator for RoaringDecoder<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        self.advance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::roaring::encoder::RoaringEncoder;

    fn build(ids: &[u32]) -> RoaringPostings {
        RoaringEncoder::from_doc_ids(ids)
    }

    #[test]
    fn decoder_empty_advance_returns_none() {
        let p = build(&[]);
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.doc(), TERMINATED);
        assert_eq!(d.advance(), None);
    }

    #[test]
    fn decoder_single_doc() {
        let p = build(&[42]);
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.advance(), Some(42));
        assert_eq!(d.doc(), 42);
        assert_eq!(d.advance(), None);
    }

    #[test]
    fn decoder_iterates_in_order() {
        let p = build(&[100, 1, 1_000_000, 50, 100_000]);
        let d = RoaringDecoder::new(&p);
        let collected: Vec<u32> = d.collect();
        assert_eq!(collected, vec![1, 50, 100, 100_000, 1_000_000]);
    }

    #[test]
    fn decoder_iterates_dense() {
        let mut enc = RoaringEncoder::new();
        for id in 0u32..10000 {
            enc.add(id);
        }
        let p = enc.finalize();
        let d = RoaringDecoder::new(&p);
        let collected: Vec<u32> = d.collect();
        assert_eq!(collected.len(), 10000);
        assert_eq!(collected[0], 0);
        assert_eq!(collected[9999], 9999);
    }

    #[test]
    fn decoder_iterates_multi_bucket() {
        let p = build(&[0, 65535, 65536, 131072, 1_000_000]);
        let d = RoaringDecoder::new(&p);
        let collected: Vec<u32> = d.collect();
        assert_eq!(collected, vec![0, 65535, 65536, 131072, 1_000_000]);
    }

    #[test]
    fn decoder_seek_hit() {
        let p = build(&[10, 20, 30, 40, 50]);
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.seek(25), Some(30));
        assert_eq!(d.advance(), Some(40));
    }

    #[test]
    fn decoder_seek_exact_match() {
        let p = build(&[10, 20, 30]);
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.seek(20), Some(20));
    }

    #[test]
    fn decoder_seek_past_end() {
        let p = build(&[1, 2, 3]);
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.seek(100), None);
        assert_eq!(d.doc(), TERMINATED);
    }

    #[test]
    fn decoder_seek_jumps_buckets() {
        let p = build(&[1, 65535, 65536, 1_000_000]);
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.seek(70_000), Some(1_000_000));
    }

    #[test]
    fn decoder_seek_into_dense_bucket() {
        let mut enc = RoaringEncoder::new();
        for id in 0u32..10_000 {
            enc.add(id);
        }
        let p = enc.finalize();
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.seek(5_000), Some(5_000));
    }

    #[test]
    fn decoder_seek_zero() {
        let p = build(&[0, 1, 2]);
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.seek(0), Some(0));
    }

    #[test]
    fn decoder_cardinality_matches_iter_count() {
        let mut enc = RoaringEncoder::new();
        for id in (0u32..50_000).step_by(3) {
            enc.add(id);
        }
        let p = enc.finalize();
        let d = RoaringDecoder::new(&p);
        let collected: Vec<u32> = d.collect();
        assert_eq!(collected.len() as u64, p.cardinality());
    }

    #[test]
    fn decoder_seek_within_run_container() {
        // Build a container that optimises into Run form: contiguous
        // 0..5000 inside high16=0.
        let mut enc = RoaringEncoder::new();
        for id in 0u32..5000 {
            enc.add(id);
        }
        let p = enc.finalize();
        // Sanity: must be Run.
        assert!(matches!(p.containers[0].1, Container::Run(_)));
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.seek(2500), Some(2500));
        assert_eq!(d.advance(), Some(2501));
    }

    #[test]
    fn decoder_seek_target_below_first() {
        let p = build(&[100, 200, 300]);
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.seek(0), Some(100));
    }

    #[test]
    fn decoder_repeated_advance_terminates() {
        let p = build(&[1, 2]);
        let mut d = RoaringDecoder::new(&p);
        assert_eq!(d.advance(), Some(1));
        assert_eq!(d.advance(), Some(2));
        assert_eq!(d.advance(), None);
        assert_eq!(d.advance(), None);
        assert_eq!(d.doc(), TERMINATED);
    }

    #[test]
    fn decoder_seek_then_iterate() {
        let p = build(&[10, 20, 30, 40, 50]);
        let mut d = RoaringDecoder::new(&p);
        d.seek(25);
        let rest: Vec<u32> = std::iter::from_fn(|| d.advance()).collect();
        // After seek(25) → 30 emitted, advance gives 40, 50.
        assert_eq!(rest, vec![40, 50]);
    }
}

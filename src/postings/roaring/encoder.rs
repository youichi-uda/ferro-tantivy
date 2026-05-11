//! Roaring posting list encoder — Phase 2 C-3.
//!
//! Builds a [`RoaringPostings`] from a stream of `u32` doc-ids. The
//! flow is:
//!
//! 1. [`RoaringEncoder::add`] (or [`RoaringEncoder::add_sorted`]) splits each id into `(high16,
//!    low16)` and forwards `low16` into a per-bucket [`Container`] that grows in [`ArrayContainer`]
//!    form by default.
//! 2. [`ArrayContainer::insert`] auto-promotes to [`BitmapContainer`] once the per-bucket
//!    cardinality crosses [`crate::postings::roaring::container::ARRAY_TO_BITMAP_THRESHOLD`]. We
//!    promote *eagerly* on insert to keep the `Vec<u16>` insertion cost bounded.
//! 3. [`RoaringEncoder::finalize`] runs [`Container::optimize`] across every bucket — that's the
//!    moment a contiguous bucket gets rewritten as a [`RunContainer`].
//!
//! ## Wire format ([`RoaringPostings::to_bytes`])
//!
//! ```text
//! [magic 4][version u8][num_containers u32 LE]
//!   [high16 u16 LE][container_size u32 LE][container bytes]
//!   [high16 u16 LE][container_size u32 LE][container bytes]
//!   ...
//! ```
//!
//! `magic` = [`MAGIC`] = `b"ROAR"`. The current version is
//! [`VERSION_V1`]. Containers are written in strictly-increasing
//! `high16` order so [`RoaringDecoder::seek`] can binary-search the
//! per-bucket index.

use std::collections::BTreeMap;

use crate::postings::roaring::container::{
    ArrayContainer, BitmapContainer, Container, ContainerError, ARRAY_TO_BITMAP_THRESHOLD,
};

/// Wire format magic — 4 ASCII bytes, never overlap with the
/// BitPacker4x posting body.
pub const MAGIC: [u8; 4] = *b"ROAR";

/// Current wire version.
pub const VERSION_V1: u8 = 1;

/// Minimum bytes needed to peek the magic and version.
pub const HEADER_MIN: usize = MAGIC.len() + 1 + 4; // magic + version + num_containers

/// [`RoaringEncoder`] / [`RoaringPostings`] errors.
#[derive(Debug, thiserror::Error)]
pub enum RoaringFormatError {
    /// Input bytes did not start with [`MAGIC`].
    #[error("missing roaring magic; expected {MAGIC:?}, got {0:?}")]
    BadMagic([u8; 4]),
    /// Wire version is newer or unknown.
    #[error("unsupported roaring version: {0}")]
    UnsupportedVersion(u8),
    /// Generic truncation while parsing.
    #[error("roaring header truncated: needed {needed} bytes, got {got}")]
    Truncated {
        /// Bytes the parser expected.
        needed: usize,
        /// Bytes available.
        got: usize,
    },
    /// Per-container parse failure.
    #[error("container parse error at high16={high16}: {err}")]
    ContainerError {
        /// Bucket key whose container failed to parse.
        high16: u16,
        /// Underlying [`ContainerError`].
        err: ContainerError,
    },
    /// Containers were not in strictly-increasing `high16` order. The
    /// encoder always emits sorted output, so this is a corruption /
    /// foreign-encoder marker.
    #[error("roaring containers not in increasing high16 order")]
    UnsortedContainers,
}

/// Streaming encoder. Accepts doc-ids in any order; finalises into a
/// [`RoaringPostings`] with each per-bucket container individually
/// optimised.
#[derive(Debug, Default)]
pub struct RoaringEncoder {
    /// `high16 → Container` (Array form by default; promoted to
    /// Bitmap on threshold crossing).
    containers: BTreeMap<u16, Container>,
    /// Total cardinality (cheap, kept in sync with `containers`).
    cardinality: u64,
}

impl RoaringEncoder {
    /// Empty encoder.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        RoaringEncoder::default()
    }

    /// Insert a single doc-id.
    pub fn add(&mut self, doc_id: u32) {
        let high16 = (doc_id >> 16) as u16;
        let low16 = (doc_id & 0xFFFF) as u16;
        let entry = self
            .containers
            .entry(high16)
            .or_insert_with(|| Container::Array(ArrayContainer::new()));
        // Insert into the current container form.
        let already_present = entry.contains(low16);
        if !already_present {
            self.cardinality += 1;
            insert_low16(entry, low16);
        }
    }

    /// Bulk insert. Identical semantics to repeated [`Self::add`] —
    /// this is a placeholder for a future SIMD-batched fast path; for
    /// now it just forwards.
    pub fn add_sorted<I: IntoIterator<Item = u32>>(&mut self, ids: I) {
        for id in ids {
            self.add(id);
        }
    }

    /// Total cardinality (= count of distinct doc-ids inserted).
    #[inline]
    #[must_use]
    pub fn cardinality(&self) -> u64 {
        self.cardinality
    }

    /// Number of populated buckets.
    #[inline]
    #[must_use]
    pub fn num_containers(&self) -> usize {
        self.containers.len()
    }

    /// Finalise: run [`Container::optimize`] on each bucket, drop
    /// any bucket that ended up empty (defensive — the encoder never
    /// inserts empty containers, but `optimize()` could theoretically
    /// surface a zero-cardinality result).
    #[must_use]
    pub fn finalize(self) -> RoaringPostings {
        let mut out: Vec<(u16, Container)> = Vec::with_capacity(self.containers.len());
        for (high16, container) in self.containers {
            let optimized = container.optimize();
            if optimized.cardinality() > 0 {
                out.push((high16, optimized));
            }
        }
        RoaringPostings { containers: out }
    }

    /// Convenience: drain a [`crate::postings::BlockSegmentPostings`]
    /// into a [`RoaringPostings`]. Currently a `&[u32]` shim, since
    /// the actual `BlockSegmentPostings` -> Roaring wire is a
    /// Wave-6 (Phase 2 C-4) responsibility (we want to keep this
    /// crate's `pub` surface minimal until the segment-format
    /// dispatch lands).
    #[must_use]
    pub fn from_doc_ids(doc_ids: &[u32]) -> RoaringPostings {
        let mut enc = RoaringEncoder::new();
        for &id in doc_ids {
            enc.add(id);
        }
        enc.finalize()
    }
}

/// Insert a single low-16 key into the bucket container, eagerly
/// promoting an [`ArrayContainer`] that exceeds the threshold to a
/// [`BitmapContainer`].
fn insert_low16(container: &mut Container, low16: u16) {
    // Pre-emptive promotion: when we are about to push an Array into
    // crossing the threshold, swap to Bitmap *first* so the Vec
    // shift cost is bounded.
    let needs_promotion = matches!(container, Container::Array(a)
        if a.cardinality() >= ARRAY_TO_BITMAP_THRESHOLD);
    if needs_promotion {
        let promoted = match std::mem::replace(container, Container::Array(ArrayContainer::new())) {
            Container::Array(a) => Container::Bitmap(BitmapContainer::from_array(&a)),
            other => other,
        };
        *container = promoted;
    }
    match container {
        Container::Array(a) => a.insert(low16),
        Container::Bitmap(b) => b.insert(low16),
        Container::Run(r) => {
            // RunContainer doesn't have a single-key insert that
            // preserves the run invariant cheaply; rebuild via
            // bitmap. Encoder hot path never lands here (we never
            // construct a Run-form encoder bucket) but the dispatch
            // is here for completeness.
            let mut bm = BitmapContainer::from_run(r);
            bm.insert(low16);
            *container = Container::Bitmap(bm);
        }
    }
}

/// Serialised, sorted-by-high16 collection of containers.
///
/// Returned by [`RoaringEncoder::finalize`] and consumed by
/// [`crate::postings::roaring::RoaringDecoder`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RoaringPostings {
    /// `(high16, Container)` pairs in strictly-increasing `high16`
    /// order. Empty buckets are not present.
    pub containers: Vec<(u16, Container)>,
}

impl RoaringPostings {
    /// Empty.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        RoaringPostings::default()
    }

    /// Total cardinality.
    #[must_use]
    pub fn cardinality(&self) -> u64 {
        self.containers
            .iter()
            .map(|(_, c)| u64::from(c.cardinality()))
            .sum()
    }

    /// Number of buckets.
    #[inline]
    #[must_use]
    pub fn num_containers(&self) -> usize {
        self.containers.len()
    }

    /// Iterate every doc-id in strictly-increasing order.
    pub fn iter(&self) -> RoaringPostingsIter<'_> {
        RoaringPostingsIter {
            outer: self.containers.iter(),
            current: None,
        }
    }

    /// Encode to the wire format described in the module docs.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.push(VERSION_V1);
        let n = u32::try_from(self.containers.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&n.to_le_bytes());
        for (high16, container) in &self.containers {
            out.extend_from_slice(&high16.to_le_bytes());
            let body = container.to_bytes();
            let body_len = u32::try_from(body.len()).unwrap_or(u32::MAX);
            out.extend_from_slice(&body_len.to_le_bytes());
            out.extend(body);
        }
        out
    }

    /// Decode from the wire format. Returns
    /// [`RoaringFormatError::BadMagic`] for non-Roaring inputs (so a
    /// caller can dispatch BitPacker4x vs Roaring by attempting this
    /// parse) and [`RoaringFormatError::UnsupportedVersion`] for
    /// unknown versions.
    pub fn from_bytes(slice: &[u8]) -> Result<Self, RoaringFormatError> {
        if slice.len() < HEADER_MIN {
            return Err(RoaringFormatError::Truncated {
                needed: HEADER_MIN,
                got: slice.len(),
            });
        }
        let magic: [u8; 4] = [slice[0], slice[1], slice[2], slice[3]];
        if magic != MAGIC {
            return Err(RoaringFormatError::BadMagic(magic));
        }
        let version = slice[4];
        if version != VERSION_V1 {
            return Err(RoaringFormatError::UnsupportedVersion(version));
        }
        let num = u32::from_le_bytes([slice[5], slice[6], slice[7], slice[8]]) as usize;
        let mut cursor = HEADER_MIN;
        let mut containers: Vec<(u16, Container)> = Vec::with_capacity(num);
        let mut last_high16: Option<u16> = None;
        for _ in 0..num {
            if cursor + 6 > slice.len() {
                return Err(RoaringFormatError::Truncated {
                    needed: cursor + 6,
                    got: slice.len(),
                });
            }
            let high16 = u16::from_le_bytes([slice[cursor], slice[cursor + 1]]);
            if let Some(prev) = last_high16 {
                if high16 <= prev {
                    return Err(RoaringFormatError::UnsortedContainers);
                }
            }
            last_high16 = Some(high16);
            let body_len = u32::from_le_bytes([
                slice[cursor + 2],
                slice[cursor + 3],
                slice[cursor + 4],
                slice[cursor + 5],
            ]) as usize;
            cursor += 6;
            if cursor + body_len > slice.len() {
                return Err(RoaringFormatError::Truncated {
                    needed: cursor + body_len,
                    got: slice.len(),
                });
            }
            let body = &slice[cursor..cursor + body_len];
            let (container, consumed) = Container::from_bytes(body)
                .map_err(|err| RoaringFormatError::ContainerError { high16, err })?;
            if consumed != body_len {
                // Container body length mismatched the segment
                // declaration. Treat as truncation.
                return Err(RoaringFormatError::Truncated {
                    needed: cursor + body_len,
                    got: cursor + consumed,
                });
            }
            cursor += body_len;
            containers.push((high16, container));
        }
        Ok(RoaringPostings { containers })
    }
}

/// Iterator over [`RoaringPostings`] yielding 32-bit doc-ids in
/// strictly-increasing order. `O(cardinality)` total.
pub struct RoaringPostingsIter<'a> {
    outer: std::slice::Iter<'a, (u16, Container)>,
    current: Option<(u16, Box<dyn Iterator<Item = u16> + 'a>)>,
}

impl Iterator for RoaringPostingsIter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        loop {
            if let Some((high16, low_iter)) = self.current.as_mut() {
                if let Some(low16) = low_iter.next() {
                    return Some((u32::from(*high16) << 16) | u32::from(low16));
                }
                self.current = None;
            }
            match self.outer.next() {
                Some((h, c)) => {
                    self.current = Some((*h, c.iter()));
                }
                None => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::roaring::container::ArrayContainer;

    #[test]
    fn encoder_empty() {
        let enc = RoaringEncoder::new();
        assert_eq!(enc.cardinality(), 0);
        let p = enc.finalize();
        assert_eq!(p.cardinality(), 0);
        assert_eq!(p.num_containers(), 0);
    }

    #[test]
    fn encoder_single_doc() {
        let mut enc = RoaringEncoder::new();
        enc.add(42);
        assert_eq!(enc.cardinality(), 1);
        let p = enc.finalize();
        assert_eq!(p.cardinality(), 1);
    }

    #[test]
    fn encoder_full_bucket_then_next() {
        let mut enc = RoaringEncoder::new();
        for id in 0u32..100 {
            enc.add(id);
        }
        for id in 1_000_000u32..1_000_005 {
            enc.add(id);
        }
        let p = enc.finalize();
        assert_eq!(p.cardinality(), 105);
        // Bucket 0 plus high16 = (1_000_000 >> 16) = 15.
        assert_eq!(p.num_containers(), 2);
    }

    #[test]
    fn encoder_dedupes() {
        let mut enc = RoaringEncoder::new();
        for _ in 0..10 {
            enc.add(7);
        }
        assert_eq!(enc.cardinality(), 1);
    }

    #[test]
    fn encoder_promotes_array_to_bitmap() {
        let mut enc = RoaringEncoder::new();
        for id in 0u32..6000 {
            enc.add(id);
        }
        let p = enc.finalize();
        assert_eq!(p.cardinality(), 6000);
        // After optimize, contiguous 0..5999 should become Run.
        match &p.containers[0].1 {
            Container::Run(r) => assert_eq!(r.cardinality(), 6000),
            other => panic!("expected run, got {other:?}"),
        }
    }

    #[test]
    fn encoder_promotes_to_bitmap_then_runs_optimize() {
        let mut enc = RoaringEncoder::new();
        // Sparse pattern that will not run-collapse — every other
        // bit, giving cardinality > threshold but no contiguity.
        for id in (0u32..10000).step_by(2) {
            enc.add(id);
        }
        let p = enc.finalize();
        // Cardinality is 5000.
        assert_eq!(p.cardinality(), 5000);
        // 5000 keys, sparse alternating bits → bitmap should win
        // because run encoding would be 5000 runs * 4 + 2 = 20002
        // bytes vs 8 KiB.
        match &p.containers[0].1 {
            Container::Bitmap(_) => {}
            other => panic!("expected bitmap, got {other:?}"),
        }
    }

    #[test]
    fn encoder_add_sorted_alias() {
        let mut enc = RoaringEncoder::new();
        enc.add_sorted([1u32, 2, 3, 100, 1_000_000]);
        let p = enc.finalize();
        assert_eq!(p.cardinality(), 5);
    }

    #[test]
    fn encoder_from_doc_ids_helper() {
        let p = RoaringEncoder::from_doc_ids(&[1u32, 2, 1_000_000]);
        assert_eq!(p.cardinality(), 3);
    }

    #[test]
    fn postings_iter_order() {
        let p = RoaringEncoder::from_doc_ids(&[1_000_000, 5, 1, 100, 1_000_001]);
        let collected: Vec<u32> = p.iter().collect();
        assert_eq!(collected, vec![1, 5, 100, 1_000_000, 1_000_001]);
    }

    #[test]
    fn wire_round_trip_empty() {
        let p = RoaringPostings::new();
        let bytes = p.to_bytes();
        let parsed = RoaringPostings::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn wire_round_trip_typical() {
        let p = RoaringEncoder::from_doc_ids(&[1, 2, 3, 1_000_000, 1_000_001, 65_536, 65_537]);
        let bytes = p.to_bytes();
        let parsed = RoaringPostings::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.cardinality(), p.cardinality());
        let a: Vec<u32> = p.iter().collect();
        let b: Vec<u32> = parsed.iter().collect();
        assert_eq!(a, b);
    }

    #[test]
    fn wire_round_trip_dense() {
        let mut enc = RoaringEncoder::new();
        for id in 0u32..50_000 {
            enc.add(id);
        }
        let p = enc.finalize();
        let bytes = p.to_bytes();
        let parsed = RoaringPostings::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.cardinality(), 50_000);
    }

    #[test]
    fn wire_magic_validation() {
        let mut bytes = vec![b'X', b'Y', b'Z', b'W', VERSION_V1, 0, 0, 0, 0];
        bytes.extend(std::iter::repeat_n(0u8, 64));
        let res = RoaringPostings::from_bytes(&bytes);
        assert!(matches!(res, Err(RoaringFormatError::BadMagic(_))));
    }

    #[test]
    fn wire_version_validation() {
        let mut bytes = MAGIC.to_vec();
        bytes.push(99u8); // unknown version
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let res = RoaringPostings::from_bytes(&bytes);
        assert!(matches!(
            res,
            Err(RoaringFormatError::UnsupportedVersion(99))
        ));
    }

    #[test]
    fn wire_truncated_header() {
        let bytes = vec![b'R', b'O'];
        let res = RoaringPostings::from_bytes(&bytes);
        assert!(matches!(res, Err(RoaringFormatError::Truncated { .. })));
    }

    #[test]
    fn wire_truncated_container_table_entry() {
        // Magic + version + claim 1 container, but no entry follows.
        let mut bytes = MAGIC.to_vec();
        bytes.push(VERSION_V1);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        let res = RoaringPostings::from_bytes(&bytes);
        assert!(matches!(res, Err(RoaringFormatError::Truncated { .. })));
    }

    #[test]
    fn wire_truncated_container_body() {
        // Magic + version + 1 container + entry header that claims
        // 1000 body bytes, but only 5 are present.
        let mut bytes = MAGIC.to_vec();
        bytes.push(VERSION_V1);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes()); // high16 = 0
        bytes.extend_from_slice(&1000u32.to_le_bytes()); // body_len = 1000
        bytes.extend_from_slice(&[0, 0, 0, 0, 0]); // 5 bytes of body
        let res = RoaringPostings::from_bytes(&bytes);
        assert!(matches!(res, Err(RoaringFormatError::Truncated { .. })));
    }

    #[test]
    fn wire_unsorted_high16_rejected() {
        // Build an artificial 2-container payload where high16 goes
        // backwards.
        let arr_a = ArrayContainer::from_sorted(vec![1, 2]);
        let arr_b = ArrayContainer::from_sorted(vec![3, 4]);
        let body_a = Container::Array(arr_a).to_bytes();
        let body_b = Container::Array(arr_b).to_bytes();
        let mut bytes = MAGIC.to_vec();
        bytes.push(VERSION_V1);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&5u16.to_le_bytes()); // high16 = 5
        bytes.extend_from_slice(&u32::try_from(body_a.len()).unwrap().to_le_bytes());
        bytes.extend(body_a);
        bytes.extend_from_slice(&3u16.to_le_bytes()); // high16 = 3 < 5 → unsorted
        bytes.extend_from_slice(&u32::try_from(body_b.len()).unwrap().to_le_bytes());
        bytes.extend(body_b);
        let res = RoaringPostings::from_bytes(&bytes);
        assert!(matches!(res, Err(RoaringFormatError::UnsortedContainers)));
    }

    #[test]
    fn wire_corrupt_inner_container() {
        let mut bytes = MAGIC.to_vec();
        bytes.push(VERSION_V1);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes()); // high16 = 0
        bytes.extend_from_slice(&5u32.to_le_bytes()); // body_len = 5
        bytes.extend_from_slice(&[0xFFu8, 0, 0, 0, 0]); // unknown tag 0xFF
        let res = RoaringPostings::from_bytes(&bytes);
        assert!(matches!(
            res,
            Err(RoaringFormatError::ContainerError { .. })
        ));
    }

    #[test]
    fn encoder_one_million_ids_stress() {
        let mut enc = RoaringEncoder::new();
        for id in 0u32..1_000_000 {
            enc.add(id);
        }
        let p = enc.finalize();
        assert_eq!(p.cardinality(), 1_000_000);
        // 1M / 64K = 15.26 → 16 buckets.
        assert_eq!(p.num_containers(), 16);
        // Round-trip the full thing.
        let bytes = p.to_bytes();
        let parsed = RoaringPostings::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.cardinality(), 1_000_000);
    }

    #[test]
    fn encoder_high16_spread() {
        let mut enc = RoaringEncoder::new();
        for high in 0u32..10 {
            enc.add(high << 16);
        }
        let p = enc.finalize();
        assert_eq!(p.cardinality(), 10);
        assert_eq!(p.num_containers(), 10);
    }

    #[test]
    fn encoder_max_doc_id() {
        let mut enc = RoaringEncoder::new();
        enc.add(u32::MAX);
        let p = enc.finalize();
        assert_eq!(p.cardinality(), 1);
        let collected: Vec<u32> = p.iter().collect();
        assert_eq!(collected, vec![u32::MAX]);
    }

    #[test]
    fn encoder_num_containers_query() {
        let mut enc = RoaringEncoder::new();
        enc.add(1);
        enc.add(70_000);
        assert_eq!(enc.num_containers(), 2);
    }

    #[test]
    fn postings_from_bytes_then_iter() {
        let p = RoaringEncoder::from_doc_ids(&[1, 2, 100_000]);
        let bytes = p.to_bytes();
        let parsed = RoaringPostings::from_bytes(&bytes).unwrap();
        let collected: Vec<u32> = parsed.iter().collect();
        assert_eq!(collected, vec![1, 2, 100_000]);
    }
}

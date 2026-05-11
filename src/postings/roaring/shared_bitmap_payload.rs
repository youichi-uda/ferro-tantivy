//! Shared bitmap-expansion helper for the v2 / v3 VRAM CHT tiers.
//!
//! Wave Z-2 #1 extracts the container→bitmap expansion logic that
//! [`super::vram_cht::VramCht::build_entry`] and
//! [`super::vram_cht_v3::VramCompressedCht::build_entry`] each
//! reimplemented inline. Both tiers consume the **same**
//! `(bucket_index, total_words)` summary plus an uncompressed
//! `[u32; total_words]` host staging buffer; only the device-side
//! commit step differs (v2 = single H→D into the cache; v3 = H→D
//! into a temp + Bitcomp compress device→device).
//!
//! Extracting the expansion has two payoffs:
//!
//! 1. **Eliminates ~60 lines of duplicated container-walk** between
//!    `vram_cht.rs` and `vram_cht_v3.rs`. The two callers stay
//!    byte-identical (same staging layout, same `(high16,
//!    word_offset)` index, same `total_words` count) so existing
//!    persistence tests continue to round-trip the same wire bytes.
//! 2. **Unlocks cross-tier v2 → v3 promotion** (see
//!    [`super::vram_cht_v3::VramCompressedCht::promote_v2_to_v3`]).
//!    When v3 admits an entry that already lives in v2, the v2
//!    `Arc<VramTermEntry>` already carries the
//!    `(bucket_index, total_words)` pair — no host re-drain
//!    required. The helper here represents the *invariant* shape
//!    both tiers commit to.
//!
//! ## Scope
//!
//! - `pub(crate)` visibility — internal helper; no external API
//!   surface change. The v2/v3 caches keep their existing public
//!   `insert` / `promote` / `get` signatures.
//! - Pure CPU logic — no CUDA dependency. The struct is buildable
//!   on a CUDA-less host (e.g. CI box without nvcomp / NVIDIA
//!   driver) so test fixtures can verify the expansion in
//!   isolation.
//! - Feature-gated on the same `gpu + ferro-compress +
//!   cuda-bitmap-kernel` stack the v2/v3 modules require, because
//!   the helper's only callers are those modules. If a follow-up
//!   wave wants this helper outside the GPU stack, drop the cfg.

#![cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]

use crate::postings::roaring::container::{BitmapContainer, Container};
use crate::postings::roaring::encoder::RoaringPostings;
use crate::postings::roaring::BITMAP_CONTAINER_WORDS;

/// Shared bitmap representation produced by container→bitmap expansion.
///
/// Both v2 (uncompressed VRAM CHT) and v3 (Bitcomp-compressed VRAM
/// CHT) need the same `(high16, word_offset)` bucket index, the
/// total `u32` word count, and a host staging buffer to build their
/// respective device-side representations.
///
/// ## Layout invariants
///
/// - `bucket_index.len() == bucket_count`
/// - `total_words == bucket_count * BITMAP_CONTAINER_WORDS`
/// - `staging.len() == total_words`
/// - `bucket_index` is sorted in ascending `high16` order (the
///   source [`RoaringPostings::containers`] is itself sorted by
///   `high16`; the helper skips empty buckets without reordering).
/// - For bucket `i`, `staging[bucket_index[i].1 .. bucket_index[i].1 +
///   BITMAP_CONTAINER_WORDS]` holds the bitmap-form expansion of
///   the source container at `bucket_index[i].0`.
///
/// ## Lifetime
///
/// The helper owns its host buffers (`bucket_index` Vec + `staging`
/// Vec). Callers `cudaMemcpy` from `staging.as_ptr()` into the
/// device temp; once the H→D copy returns, the helper can be
/// dropped (the device buffer is the canonical copy from then on).
#[derive(Debug)]
pub(crate) struct SharedBitmapPayload {
    /// `(high16, word_offset)` for each non-empty bucket. Sorted by
    /// `high16` ascending; `word_offset` is in `u32` units relative
    /// to the start of `staging`.
    pub(crate) bucket_index: Vec<(u16, u32)>,
    /// Number of non-empty buckets = `bucket_index.len()`. Cached
    /// so callers don't repeat the `.len()` call across the build
    /// path.
    pub(crate) bucket_count: usize,
    /// Total `u32` words across all buckets = `bucket_count *
    /// BITMAP_CONTAINER_WORDS`. Matches v2's `total_words` and v3's
    /// `uncompressed_bytes / 4`.
    pub(crate) total_words: usize,
    /// Host-side flat `[u32; total_words]` staging buffer. Bucket
    /// `i` occupies `staging[bucket_index[i].1 .. bucket_index[i].1
    /// + BITMAP_CONTAINER_WORDS]`.
    pub(crate) staging: Vec<u32>,
}

impl SharedBitmapPayload {
    /// Walk a [`RoaringPostings`] and build the
    /// `(bucket_index, total_words, staging)` triple needed by both
    /// v2 and v3 build paths.
    ///
    /// Returns `None` when the source has zero non-empty buckets —
    /// caching is meaningless for an empty term (any cohort
    /// intersection involving it yields an empty result without
    /// touching the GPU). The caller's existing
    /// "empty roaring skipped" branch maps to this case.
    pub(crate) fn from_postings(postings: &RoaringPostings) -> Option<Self> {
        // Count non-empty buckets first so we can size both Vecs
        // exactly (no reallocs).
        let bucket_count = postings
            .containers
            .iter()
            .filter(|(_, c)| c.cardinality() > 0)
            .count();
        if bucket_count == 0 {
            return None;
        }
        let total_words = bucket_count * BITMAP_CONTAINER_WORDS;

        let mut staging: Vec<u32> = vec![0u32; total_words];
        let mut bucket_index: Vec<(u16, u32)> = Vec::with_capacity(bucket_count);
        let mut word_offset: u32 = 0;
        for (high16, container) in &postings.containers {
            if container.cardinality() == 0 {
                continue;
            }
            let start = word_offset as usize;
            let end = start + BITMAP_CONTAINER_WORDS;
            let chunk = &mut staging[start..end];
            match container {
                Container::Bitmap(bm) => {
                    chunk.copy_from_slice(bm.words.as_ref());
                }
                Container::Array(arr) => {
                    let bm = BitmapContainer::from_array(arr);
                    chunk.copy_from_slice(bm.words.as_ref());
                }
                Container::Run(rc) => {
                    let bm = BitmapContainer::from_run(rc);
                    chunk.copy_from_slice(bm.words.as_ref());
                }
            }
            bucket_index.push((*high16, word_offset));
            word_offset = word_offset
                .checked_add(BITMAP_CONTAINER_WORDS as u32)
                .expect("word_offset must not overflow u32 within total_words bound");
        }

        debug_assert_eq!(bucket_index.len(), bucket_count);
        debug_assert_eq!(staging.len(), total_words);
        Some(Self {
            bucket_index,
            bucket_count,
            total_words,
            staging,
        })
    }

    /// Total bytes of `staging` (= `total_words * 4`). Convenience
    /// helper because both tiers feed this into `cudaMalloc` /
    /// `cudaMemcpy` size arguments.
    #[inline]
    pub(crate) fn total_bytes(&self) -> usize {
        self.total_words * std::mem::size_of::<u32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::roaring::encoder::RoaringEncoder;

    fn small_roaring() -> RoaringPostings {
        RoaringEncoder::from_doc_ids(&[1, 2, 3, 100, 65540])
    }

    fn dense_roaring(seed: u32) -> RoaringPostings {
        let docs: Vec<u32> = (0..200_000).map(|i| seed.wrapping_add(i * 3)).collect();
        RoaringEncoder::from_doc_ids(&docs)
    }

    #[test]
    fn empty_postings_returns_none() {
        let empty = RoaringPostings::default();
        assert!(SharedBitmapPayload::from_postings(&empty).is_none());
    }

    #[test]
    fn small_postings_two_buckets() {
        // small_roaring spans bucket 0 (docs 1, 2, 3, 100) and
        // bucket 1 (doc 65540 = bucket 1, low16 = 4).
        let rp = small_roaring();
        let payload = SharedBitmapPayload::from_postings(&rp).unwrap();
        assert_eq!(payload.bucket_count, 2);
        assert_eq!(payload.bucket_index.len(), 2);
        assert_eq!(payload.total_words, 2 * BITMAP_CONTAINER_WORDS);
        assert_eq!(payload.staging.len(), 2 * BITMAP_CONTAINER_WORDS);
        assert_eq!(payload.total_bytes(), 2 * BITMAP_CONTAINER_WORDS * 4);

        let buckets: Vec<u16> = payload.bucket_index.iter().map(|(h, _)| *h).collect();
        assert_eq!(buckets, vec![0, 1]);
        for (i, (_, off)) in payload.bucket_index.iter().enumerate() {
            assert_eq!(*off as usize, i * BITMAP_CONTAINER_WORDS);
        }
    }

    #[test]
    fn dense_postings_layout_matches_legacy_v2() {
        // Cross-check against the documented v2 layout invariants:
        // - bucket_count = non-empty containers in the source
        // - bucket_index sorted ascending by high16
        // - word_offset[i] == i * BITMAP_CONTAINER_WORDS
        // - total_words = bucket_count * BITMAP_CONTAINER_WORDS
        let rp = dense_roaring(0);
        let expected_buckets = rp
            .containers
            .iter()
            .filter(|(_, c)| c.cardinality() > 0)
            .count();
        let payload = SharedBitmapPayload::from_postings(&rp).unwrap();
        assert_eq!(payload.bucket_count, expected_buckets);
        assert_eq!(
            payload.total_words,
            expected_buckets * BITMAP_CONTAINER_WORDS
        );
        // word_offset must be strictly monotonic in `i *
        // BITMAP_CONTAINER_WORDS` steps.
        let mut prev_off = -1i64;
        for (i, (_h, off)) in payload.bucket_index.iter().enumerate() {
            let off_i64 = *off as i64;
            assert!(off_i64 > prev_off);
            assert_eq!(off_i64, (i * BITMAP_CONTAINER_WORDS) as i64);
            prev_off = off_i64;
        }
    }

    #[test]
    fn bucket_index_high16_ascending() {
        // The helper relies on RoaringPostings' container vec being
        // sorted by high16; verify the output preserves that
        // invariant even after skipping empty containers.
        let rp = dense_roaring(0);
        let payload = SharedBitmapPayload::from_postings(&rp).unwrap();
        let highs: Vec<u16> = payload.bucket_index.iter().map(|(h, _)| *h).collect();
        let mut sorted = highs.clone();
        sorted.sort();
        assert_eq!(highs, sorted, "bucket_index must be sorted by high16");
    }

    #[test]
    fn staging_contains_bitmap_words_per_bucket() {
        // Each bucket's window in `staging` must equal the bitmap
        // expansion of that container — same invariant the legacy
        // v2/v3 build paths held inline.
        let rp = small_roaring();
        let payload = SharedBitmapPayload::from_postings(&rp).unwrap();
        for (h, off) in payload.bucket_index.iter() {
            let (_, container) = rp
                .containers
                .iter()
                .find(|(hh, _)| *hh == *h)
                .expect("bucket high16 must exist in source");
            let expected: Vec<u32> = match container {
                Container::Bitmap(bm) => bm.words.as_ref().to_vec(),
                Container::Array(arr) => BitmapContainer::from_array(arr).words.as_ref().to_vec(),
                Container::Run(rc) => BitmapContainer::from_run(rc).words.as_ref().to_vec(),
            };
            assert_eq!(expected.len(), BITMAP_CONTAINER_WORDS);
            let start = *off as usize;
            let end = start + BITMAP_CONTAINER_WORDS;
            assert_eq!(
                &payload.staging[start..end],
                expected.as_slice(),
                "staging window for bucket {h} must match container bitmap"
            );
        }
    }

    #[test]
    fn total_bytes_matches_total_words_times_four() {
        let rp = small_roaring();
        let payload = SharedBitmapPayload::from_postings(&rp).unwrap();
        assert_eq!(payload.total_bytes(), payload.total_words * 4);
    }
}

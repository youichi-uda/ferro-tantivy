//! Roaring Bitmap posting list — Phase 2 C-3 foundation.
//!
//! This module is the in-tree, **alternate** posting list format for
//! the Tantivy fork. It coexists with the existing block-wise
//! [`BitPacker4x`](crate::postings::compression) format, *not*
//! replacing it. Selection is done at segment-write time by a
//! [`PostingFormat`] tag (Phase 2 C-4 / Wave 6 will add the planner
//! threshold that decides which one to emit).
//!
//! # Module map
//!
//! - [`container`]: the three Roaring container forms
//!   ([`container::ArrayContainer`], [`container::BitmapContainer`],
//!   [`container::RunContainer`]) plus the [`container::Container`]
//!   dispatch enum and self-describing wire encoding.
//! - [`encoder`]: [`encoder::RoaringEncoder`] (`u32` doc-id stream →
//!   per-bucket containers) and [`encoder::RoaringPostings`]
//!   (finalised, sorted-by-`high16`, byte-serialisable form).
//! - [`decoder`]: [`decoder::RoaringDecoder`] — `advance` / `seek`
//!   over a [`encoder::RoaringPostings`] (`O(log n + log m)` seek).
//! - [`ferro_compress_bridge`]: the
//!   [`ferro_compress_bridge::BitcompCodec`] trait, the always-on
//!   [`ferro_compress_bridge::IdentityBitcompCodec`] passthrough, and
//!   the (currently short-circuiting)
//!   [`ferro_compress_bridge::FerroBitcompCodec`] wired in by the
//!   `ferro-compress` Cargo feature.
//!
//! # Wire format
//!
//! See [`encoder::RoaringPostings::to_bytes`]. The first 4 bytes are
//! the `b"ROAR"` magic ([`encoder::MAGIC`]) so a reader can dispatch
//! between Roaring and BitPacker4x bodies *purely by looking at the
//! head of the stream*.
//!
//! # GPU integration
//!
//! [`container::BitmapContainer`] keeps its raw layout aligned with
//! the [`crate::postings::roaring::BITMAP_CONTAINER_WORDS`] = 2048 u32
//! constant (re-exported from `tantivy-gpu`). That constant is the
//! exact word-count consumed by the wgpu / CUDA bitmap kernels (Wave
//! 4-B, `gpu/src/posting/bitmap_op.rs`). Wiring the GPU kernel to the
//! Tantivy query path is *not* this wave's job — it's Wave 6 (C-4),
//! gated on a planner threshold that keeps light queries on the CPU
//! galloping path.
//!
//! # What this wave (C-3) does NOT do
//!
//! - Make Roaring the default posting format. Existing segments and
//!   newly-written segments **continue to use BitPacker4x** until
//!   Wave 6 lands the dispatch threshold.
//! - Wire Roaring through `BlockSegmentPostings` / `SegmentPostings`
//!   read paths. Those continue to read BitPacker4x exclusively.
//! - Pull `ferro-compress` in as a hard dep. The trait abstraction
//!   keeps the Tantivy fork buildable standalone.
//! - Add bench harnesses. End-to-end Roaring vs BitPacker4x bench is
//!   Phase 2 C-5.

pub mod container;
pub mod decoder;
pub mod encoder;
pub mod ferro_compress_bridge;
pub mod from_block_segment;
pub mod gpu_dispatch;
pub mod planner;

pub use container::{
    ArrayContainer, BitmapContainer, Container, ContainerError, Run, RunContainer,
};
pub use decoder::{RoaringDecoder, TERMINATED};
pub use encoder::{RoaringEncoder, RoaringFormatError, RoaringPostings, MAGIC, VERSION_V1};
pub use ferro_compress_bridge::{
    compress_container, decompress_container, BitcompCodec, BitcompError, DefaultBitcompCodec,
    IdentityBitcompCodec,
};
pub use from_block_segment::drain_block_segment_to_roaring;
pub use gpu_dispatch::{
    cpu_fallback_count, gpu_dispatch_count, record_cpu_fallback, reset_dispatch_counters,
    try_gpu_bool, BoolOp,
};
pub use planner::{
    should_dispatch_gpu, TermStat, MIN_COHORT_DOCS, MIN_PER_TERM_CARDINALITY, MIN_RATIO,
};

#[cfg(feature = "ferro-compress")]
pub use ferro_compress_bridge::FerroBitcompCodec;

/// Number of `u32` words in a Roaring [`container::BitmapContainer`]
/// (= 2048 = 8 KiB = 65 536 bits).
///
/// Mirrors `tantivy_gpu::posting::BITMAP_CONTAINER_WORDS` so callers
/// in this crate don't need to depend on the optional `tantivy-gpu`
/// crate just to size a buffer; the equivalence is enforced by a
/// compile-time check in
/// [`tests::bitmap_container_words_matches_gpu`] when the `gpu`
/// feature is on.
pub const BITMAP_CONTAINER_WORDS: usize = 2048;

/// Tag for the on-disk posting list format.
///
/// Stored as a single byte (the variant discriminant) at the top of
/// each per-field segment metadata block. Unknown tags must be
/// treated as a corruption / forward-incompatible-segment marker by
/// readers — never silently coerce.
///
/// ## Wire encoding
///
/// - `0x00` → [`PostingFormat::BitPacker4x`] (legacy, default)
/// - `0x01` → [`PostingFormat::Roaring`]
///
/// Future formats (e.g. `Bitcomp` only, `RaBitQ`-residual) get
/// successive tags. Reserve `0xFF` for "extended descriptor follows"
/// if we ever need >256 forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PostingFormat {
    /// Block-wise SIMD-PFD (`BitPacker4x` + VInt tail). Legacy
    /// Tantivy default — what every existing segment uses.
    BitPacker4x = 0x00,
    /// Roaring Bitmap container format (this module).
    Roaring = 0x01,
}

impl PostingFormat {
    /// Wire byte for this format. Identical to `self as u8`.
    #[inline]
    #[must_use]
    pub fn tag(self) -> u8 {
        self as u8
    }

    /// Decode a wire byte into a [`PostingFormat`]. Returns [`None`]
    /// for unknown tags so callers can decide whether to treat the
    /// input as corrupt vs forward-incompatible.
    #[inline]
    #[must_use]
    pub fn from_tag(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(PostingFormat::BitPacker4x),
            0x01 => Some(PostingFormat::Roaring),
            _ => None,
        }
    }

    /// True iff this format is the legacy default (= what untagged
    /// segments are read as).
    #[inline]
    #[must_use]
    pub fn is_default(self) -> bool {
        matches!(self, PostingFormat::BitPacker4x)
    }
}

impl Default for PostingFormat {
    /// [`PostingFormat::BitPacker4x`] — backward-compatible with every
    /// pre-Phase-2 segment.
    fn default() -> Self {
        PostingFormat::BitPacker4x
    }
}

/// Lightweight format-dispatch handle.
///
/// Phase 2 C-3 ships this as a *scaffold*: callers can construct a
/// [`PostingFormat`] tag, ask which format will be used for a given
/// segment, and (in C-4) dispatch the read path. The actual read
/// path swap is Wave 6 — see the module docs.
///
/// # Backward-compat contract
///
/// - Segments without a format tag (= every existing index on disk)
///   read as [`PostingFormat::BitPacker4x`] — never as Roaring.
/// - Segments tagged `0x01` are Roaring and *must* fail-fast in
///   readers that don't understand them, never silently fall back.
///   The default reader path will keep working on `0x00` segments
///   identically to today; a Roaring-aware reader will check the
///   tag *before* attempting to parse the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostingFormatDispatch {
    /// Format selected at segment-write time.
    pub format: PostingFormat,
}

impl PostingFormatDispatch {
    /// Build a dispatch handle for [`PostingFormat::BitPacker4x`].
    #[inline]
    #[must_use]
    pub const fn legacy() -> Self {
        PostingFormatDispatch {
            format: PostingFormat::BitPacker4x,
        }
    }

    /// Build a dispatch handle for [`PostingFormat::Roaring`].
    #[inline]
    #[must_use]
    pub const fn roaring() -> Self {
        PostingFormatDispatch {
            format: PostingFormat::Roaring,
        }
    }

    /// Peek at a posting body and *guess* its format without
    /// committing to a parse.
    ///
    /// Roaring bodies start with the 4-byte
    /// [`encoder::MAGIC`] = `b"ROAR"`. Anything else is treated as
    /// BitPacker4x — keeping the legacy default safe.
    #[must_use]
    pub fn detect(body: &[u8]) -> PostingFormat {
        if body.len() >= MAGIC.len() && body[..MAGIC.len()] == MAGIC {
            PostingFormat::Roaring
        } else {
            PostingFormat::BitPacker4x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posting_format_default_is_bitpacker() {
        assert_eq!(PostingFormat::default(), PostingFormat::BitPacker4x);
    }

    #[test]
    fn posting_format_tag_round_trip() {
        for fmt in [PostingFormat::BitPacker4x, PostingFormat::Roaring] {
            let parsed = PostingFormat::from_tag(fmt.tag()).unwrap();
            assert_eq!(parsed, fmt);
        }
    }

    #[test]
    fn posting_format_unknown_tag_is_none() {
        assert!(PostingFormat::from_tag(0x42).is_none());
        assert!(PostingFormat::from_tag(0xFF).is_none());
    }

    #[test]
    fn posting_format_is_default() {
        assert!(PostingFormat::BitPacker4x.is_default());
        assert!(!PostingFormat::Roaring.is_default());
    }

    #[test]
    fn dispatch_legacy_is_bitpacker() {
        let d = PostingFormatDispatch::legacy();
        assert_eq!(d.format, PostingFormat::BitPacker4x);
    }

    #[test]
    fn dispatch_roaring() {
        let d = PostingFormatDispatch::roaring();
        assert_eq!(d.format, PostingFormat::Roaring);
    }

    #[test]
    fn dispatch_detect_roaring_magic() {
        let p = RoaringEncoder::from_doc_ids(&[1, 2, 3]);
        let bytes = p.to_bytes();
        assert_eq!(
            PostingFormatDispatch::detect(&bytes),
            PostingFormat::Roaring
        );
    }

    #[test]
    fn dispatch_detect_bitpacker_default() {
        let bytes = [0xAB, 0xCD, 0xEF, 0x12, 0x34];
        assert_eq!(
            PostingFormatDispatch::detect(&bytes),
            PostingFormat::BitPacker4x
        );
    }

    #[test]
    fn dispatch_detect_short_input_is_bitpacker() {
        // Anything smaller than the magic must default to BitPacker4x
        // so empty / partial bodies don't accidentally promote.
        assert_eq!(
            PostingFormatDispatch::detect(&[]),
            PostingFormat::BitPacker4x
        );
        assert_eq!(
            PostingFormatDispatch::detect(b"R"),
            PostingFormat::BitPacker4x
        );
    }

    #[test]
    fn bitmap_container_words_constant() {
        assert_eq!(BITMAP_CONTAINER_WORDS, 2048);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn bitmap_container_words_matches_gpu() {
        assert_eq!(
            BITMAP_CONTAINER_WORDS,
            tantivy_gpu::posting::BITMAP_CONTAINER_WORDS
        );
    }
}

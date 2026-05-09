//! Bridge between [`crate::postings::roaring::container::BitmapContainer`]
//! and the FerroSearch GPU compression stack (`ferro-compress` crate).
//!
//! ## Why a trait abstraction?
//!
//! The Tantivy fork (this crate) is OSS-licensed and must build
//! standalone — pulling `ferro-compress` in as a hard dependency
//! would couple every downstream user to FerroSearch's CUDA / nvCOMP
//! toolchain. Instead, we expose a single [`BitcompCodec`] trait and
//! ship two implementations behind a feature flag:
//!
//! - **Default** ([`feature = "ferro-compress"`] off): the
//!   [`IdentityBitcompCodec`] passes raw word bytes through,
//!   round-tripping cleanly. Useful for tests, the OSS fork, and any
//!   non-FerroSearch downstream.
//! - **Enabled** ([`feature = "ferro-compress"`] on): the
//!   [`FerroBitcompCodec`] *will* delegate to
//!   `ferro_compress::nvcomp::Algo::Bitcomp` once Cargo wiring lands
//!   in Wave 6 (Phase 2 C-4). Until then it returns
//!   [`BitcompError::WaitingOnWave6`] from every call — by design,
//!   so a misconfigured build fails fast rather than silently
//!   skipping compression.
//!
//! ## Wire considerations
//!
//! [`BitmapContainer`](crate::postings::roaring::container::BitmapContainer)
//! is a fixed `[u32; 2048]` (8 KiB). Passing it through this trait
//! lets us:
//!
//! 1. swap codecs without touching the encoder/decoder; and
//! 2. keep the GPU-native form (raw bitmap) and the on-disk form
//!    (Bitcomp-compressed bitmap) decoupled — the encoder writes
//!    `compress(raw_words)` to disk and the decoder reads
//!    `decompress(compressed_bytes) -> [u32; 2048]` back into VRAM /
//!    host memory.
//!
//! ## What this wave (C-3) does *not* do
//!
//! - No `Cargo.toml` `path = "../ferro-compress"` dependency wiring
//!   (Wave 6 / C-4).
//! - No `Cargo.lock` mutation outside this crate.
//! - No `BitmapContainer::compressed_bytes` helpers on the container
//!   itself (kept here, in the bridge module, to avoid a circular
//!   dep direction between `container.rs` and the codec layer).

use crate::postings::roaring::container::BitmapContainer;
use crate::postings::roaring::BITMAP_CONTAINER_WORDS;

/// Errors surfaced by [`BitcompCodec`] implementations.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BitcompError {
    /// Compressed input claimed a different uncompressed length than
    /// expected by the caller.
    #[error("bitcomp: length mismatch (expected {expected} u32 words, codec produced {got})")]
    LengthMismatch {
        /// Words the caller expected.
        expected: usize,
        /// Words the codec actually wrote.
        got: usize,
    },
    /// Real `ferro-compress` codec is selected via the
    /// `ferro-compress` feature, but the actual Cargo path-dep wiring
    /// is deferred to Wave 6.
    #[error("bitcomp: ferro-compress codec selected but Cargo wiring is deferred to Wave 6 (C-4)")]
    WaitingOnWave6,
    /// Decompression produced fewer / more bytes than the bitmap
    /// container fixed size.
    #[error(
        "bitcomp: decompressed output not 8 KiB ({} bytes) — got {got} bytes",
        BITMAP_CONTAINER_WORDS * 4
    )]
    NotContainerSized {
        /// Bytes actually produced.
        got: usize,
    },
    /// Generic codec failure (corrupt input, etc.).
    #[error("bitcomp: codec failure: {0}")]
    CodecFailure(String),
}

/// Codec abstraction over [`BitmapContainer`] payloads.
///
/// Implementations promise round-trip safety: for any
/// `raw_words: [u32; 2048]`,
/// `decompress(&compress(raw_words)) == raw_words`.
pub trait BitcompCodec {
    /// Compress an 8 KiB bitmap container's raw word array. Returns
    /// the codec-specific compressed payload, opaque to the caller.
    ///
    /// `raw_bitmap.len() == BITMAP_CONTAINER_WORDS` is guaranteed by
    /// callers — implementations may panic on length mismatch in
    /// debug builds; release-build behaviour is implementation-defined
    /// (typically: best-effort with a [`BitcompError::LengthMismatch`]
    /// returned via the failure path of a wrapper, see
    /// [`compress_container`]).
    fn compress(&self, raw_bitmap: &[u32]) -> Result<Vec<u8>, BitcompError>;

    /// Decompress into a fixed-size word array. The output buffer is
    /// always exactly `BITMAP_CONTAINER_WORDS` words; if the codec
    /// produces a different size it must surface
    /// [`BitcompError::NotContainerSized`].
    fn decompress(
        &self,
        compressed: &[u8],
        out: &mut [u32; BITMAP_CONTAINER_WORDS],
    ) -> Result<(), BitcompError>;
}

/// Compress a [`BitmapContainer`] via the supplied codec.
///
/// Convenience adapter that handles the borrow into
/// [`BitcompCodec::compress`] and avoids passing `&[u32]` directly
/// from the call site (which would lose the
/// `BITMAP_CONTAINER_WORDS` invariant).
pub fn compress_container(
    container: &BitmapContainer,
    codec: &dyn BitcompCodec,
) -> Result<Vec<u8>, BitcompError> {
    let words: &[u32] = container.words.as_ref();
    codec.compress(words)
}

/// Decompress a payload back into a [`BitmapContainer`].
pub fn decompress_container(
    compressed: &[u8],
    codec: &dyn BitcompCodec,
) -> Result<BitmapContainer, BitcompError> {
    let mut out = Box::new([0u32; BITMAP_CONTAINER_WORDS]);
    codec.decompress(compressed, out.as_mut())?;
    Ok(BitmapContainer::from_words(out))
}

// ============================================================
// Identity (passthrough) codec — always available.
// ============================================================

/// Passthrough codec used by the OSS fork and by any downstream
/// without `ferro-compress` enabled.
///
/// Encodes the word array to a little-endian byte stream and decodes
/// it back. Round-trip safe.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityBitcompCodec;

impl BitcompCodec for IdentityBitcompCodec {
    fn compress(&self, raw_bitmap: &[u32]) -> Result<Vec<u8>, BitcompError> {
        if raw_bitmap.len() != BITMAP_CONTAINER_WORDS {
            return Err(BitcompError::LengthMismatch {
                expected: BITMAP_CONTAINER_WORDS,
                got: raw_bitmap.len(),
            });
        }
        let mut out = Vec::with_capacity(BITMAP_CONTAINER_WORDS * 4);
        for &w in raw_bitmap {
            out.extend_from_slice(&w.to_le_bytes());
        }
        Ok(out)
    }

    fn decompress(
        &self,
        compressed: &[u8],
        out: &mut [u32; BITMAP_CONTAINER_WORDS],
    ) -> Result<(), BitcompError> {
        if compressed.len() != BITMAP_CONTAINER_WORDS * 4 {
            return Err(BitcompError::NotContainerSized {
                got: compressed.len(),
            });
        }
        for i in 0..BITMAP_CONTAINER_WORDS {
            let off = i * 4;
            out[i] = u32::from_le_bytes([
                compressed[off],
                compressed[off + 1],
                compressed[off + 2],
                compressed[off + 3],
            ]);
        }
        Ok(())
    }
}

// ============================================================
// FerroBitcompCodec — Wave-6 placeholder.
// ============================================================

/// Codec that delegates to `ferro_compress::nvcomp::Algo::Bitcomp`.
///
/// **Wave 5 status**: structurally complete, but every method short-
/// circuits to [`BitcompError::WaitingOnWave6`]. Wave 6 (Phase 2 C-4)
/// adds the `path = "../../ferrosearch-gpu-compress/crates/ferro-compress"`
/// dep + builds the actual Bitcomp call. Tests in this module pin the
/// short-circuit so the contract is observable today.
#[cfg(feature = "ferro-compress")]
#[derive(Debug, Default, Clone, Copy)]
pub struct FerroBitcompCodec;

#[cfg(feature = "ferro-compress")]
impl BitcompCodec for FerroBitcompCodec {
    fn compress(&self, _raw_bitmap: &[u32]) -> Result<Vec<u8>, BitcompError> {
        // Wave 6 wiring target:
        //
        //   use ferro_compress::nvcomp::{Backend, Algo, BitcompDataType};
        //   let backend = Backend::auto();
        //   let bytes: Vec<u8> = bytemuck::cast_slice(_raw_bitmap).to_vec();
        //   backend.compress(&bytes, Algo::Bitcomp { data_type: BitcompDataType::Uint32 })
        //
        // Until the dep lands, returning the contract error makes
        // misconfigurations visible at runtime.
        Err(BitcompError::WaitingOnWave6)
    }

    fn decompress(
        &self,
        _compressed: &[u8],
        _out: &mut [u32; BITMAP_CONTAINER_WORDS],
    ) -> Result<(), BitcompError> {
        Err(BitcompError::WaitingOnWave6)
    }
}

// ============================================================
// Default selector — feature-gated.
// ============================================================

/// Project-default codec.
///
/// - With `ferro-compress` **off** (default): [`IdentityBitcompCodec`].
/// - With `ferro-compress` **on**: [`FerroBitcompCodec`] (which still
///   short-circuits to [`BitcompError::WaitingOnWave6`] until C-4).
#[cfg(not(feature = "ferro-compress"))]
pub type DefaultBitcompCodec = IdentityBitcompCodec;

/// See the variant above. Selected by the `ferro-compress` feature.
#[cfg(feature = "ferro-compress")]
pub type DefaultBitcompCodec = FerroBitcompCodec;

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_container() -> BitmapContainer {
        let mut bm = BitmapContainer::new();
        for k in [0u16, 1, 31, 32, 100, 4096, 65535] {
            bm.insert(k);
        }
        bm
    }

    #[test]
    fn identity_round_trip() {
        let bm = make_container();
        let codec = IdentityBitcompCodec;
        let bytes = compress_container(&bm, &codec).unwrap();
        assert_eq!(bytes.len(), BITMAP_CONTAINER_WORDS * 4);
        let parsed = decompress_container(&bytes, &codec).unwrap();
        assert_eq!(parsed, bm);
    }

    #[test]
    fn identity_rejects_short_input() {
        let codec = IdentityBitcompCodec;
        let res = codec.compress(&[0u32; 10]);
        assert!(matches!(res, Err(BitcompError::LengthMismatch { .. })));
    }

    #[test]
    fn identity_rejects_bad_decompress_size() {
        let codec = IdentityBitcompCodec;
        let mut out = [0u32; BITMAP_CONTAINER_WORDS];
        let res = codec.decompress(&[0u8; 100], &mut out);
        assert!(matches!(res, Err(BitcompError::NotContainerSized { .. })));
    }

    #[test]
    fn identity_round_trip_random_words() {
        // Deterministic pseudo-random fill (no rand dep here): xorshift.
        let mut bm = BitmapContainer::new();
        let mut state: u64 = 0xdead_beef_cafe_babe;
        for _ in 0..2000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let key = (state as u32) % 65536;
            bm.insert(key as u16);
        }
        let codec = IdentityBitcompCodec;
        let bytes = compress_container(&bm, &codec).unwrap();
        let parsed = decompress_container(&bytes, &codec).unwrap();
        assert_eq!(parsed, bm);
    }

    #[test]
    fn identity_zero_container_round_trip() {
        let bm = BitmapContainer::new();
        let codec = IdentityBitcompCodec;
        let bytes = compress_container(&bm, &codec).unwrap();
        // Pure zero bytes — no padding tricks.
        assert!(bytes.iter().all(|&b| b == 0));
        let parsed = decompress_container(&bytes, &codec).unwrap();
        assert_eq!(parsed, bm);
    }

    #[test]
    fn identity_full_container_round_trip() {
        let mut bm = BitmapContainer::new();
        for k in 0u32..65_536 {
            bm.insert(k as u16);
        }
        assert_eq!(bm.cardinality(), 65_536);
        let codec = IdentityBitcompCodec;
        let bytes = compress_container(&bm, &codec).unwrap();
        let parsed = decompress_container(&bytes, &codec).unwrap();
        assert_eq!(parsed, bm);
    }

    #[cfg(not(feature = "ferro-compress"))]
    #[test]
    fn default_codec_is_identity_without_feature() {
        // Sanity: with the feature off, DefaultBitcompCodec is the
        // passthrough codec and round-trips. Under the feature flag
        // the default switches to FerroBitcompCodec, which short-
        // circuits to BitcompError::WaitingOnWave6 — covered by a
        // separate test below.
        let bm = make_container();
        let codec = DefaultBitcompCodec::default();
        let bytes = compress_container(&bm, &codec).unwrap();
        let parsed = decompress_container(&bytes, &codec).unwrap();
        assert_eq!(parsed, bm);
    }

    #[cfg(feature = "ferro-compress")]
    #[test]
    fn default_codec_is_ferro_bitcomp_with_feature() {
        // With the feature on, DefaultBitcompCodec aliases to
        // FerroBitcompCodec, which currently short-circuits to
        // BitcompError::WaitingOnWave6.
        let bm = make_container();
        let codec = DefaultBitcompCodec::default();
        let res = compress_container(&bm, &codec);
        assert_eq!(res, Err(BitcompError::WaitingOnWave6));
    }

    #[cfg(feature = "ferro-compress")]
    #[test]
    fn ferro_codec_short_circuits_until_wave6() {
        let codec = FerroBitcompCodec;
        let bm = make_container();
        let res = compress_container(&bm, &codec);
        assert_eq!(res, Err(BitcompError::WaitingOnWave6));
    }

    #[cfg(feature = "ferro-compress")]
    #[test]
    fn ferro_codec_decompress_short_circuits() {
        let codec = FerroBitcompCodec;
        let mut out = [0u32; BITMAP_CONTAINER_WORDS];
        let res = codec.decompress(&[0u8; 4], &mut out);
        assert_eq!(res, Err(BitcompError::WaitingOnWave6));
    }
}

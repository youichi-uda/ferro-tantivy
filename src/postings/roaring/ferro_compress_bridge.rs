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
//!   [`FerroBitcompCodec`] delegates to a real
//!   [`ferro_compress::Backend`]. The default backend resolution is
//!   [`ferro_compress::Backend::cpu`] with [`ferro_compress::Algo::Snappy`]
//!   — the Phase 0 fastest-decompress codec on json_logs / postings
//!   workloads (146 GB/s GPU decompress, 6.29 GB/s CPU). Callers that
//!   want Bitcomp / GPU dispatch construct a custom
//!   [`FerroBitcompCodec::with_backend`] with the appropriate algo.
//!
//! ## Wire considerations
//!
//! [`BitmapContainer`](crate::postings::roaring::container::BitmapContainer)
//! is a fixed `[u32; 2048]` (8 KiB). Passing it through this trait
//! lets us:
//!
//! 1. swap codecs without touching the encoder/decoder; and
//! 2. keep the GPU-native form (raw bitmap) and the on-disk form
//!    (compressed bitmap) decoupled — the encoder writes
//!    `compress(raw_words)` to disk and the decoder reads
//!    `decompress(compressed_bytes) -> [u32; 2048]` back into VRAM /
//!    host memory.

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
    /// Reserved variant: until Wave 6 the [`FerroBitcompCodec`] short-
    /// circuited to this error so misconfigured builds failed fast. Wave
    /// 6 swapped the short-circuit for a real
    /// [`ferro_compress::Backend`] round-trip; the variant is retained
    /// for back-compat with downstream `match` arms / tests pinned to
    /// the previous behaviour.
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
// FerroBitcompCodec — Wave 6 (C-4) real wiring.
// ============================================================

/// Codec that delegates to a [`ferro_compress::Backend`].
///
/// Wave 6 (Phase 2 C-4) lands the real round-trip. The default
/// constructor picks [`ferro_compress::Algo::Snappy`] on the CPU
/// backend — this is the Phase 0 fastest-decompress codec on
/// json_logs / postings workloads and avoids per-call CUDA overhead at
/// the 8 KiB Roaring container size (well under the
/// [`ferro_compress::AUTO_GPU_THRESHOLD_BYTES`] = 1 MiB GPU floor, so
/// `Backend::Auto` would dispatch to CPU anyway). Callers that want
/// Bitcomp / GPU paths construct via [`FerroBitcompCodec::with_backend`]
/// and pass an explicit [`ferro_compress::Algo`] / [`ferro_compress::Backend`]
/// pair — typically [`ferro_compress::Algo::for_postings_uint32`] for
/// the Bitcomp uint32-typed-numeric column win measured at Phase 0
/// (3.59× ratio / 366 GB/s decomp).
///
/// Round-trip safe by construction: every algorithm in
/// [`ferro_compress::Algo`] is lossless.
#[cfg(feature = "ferro-compress")]
pub struct FerroBitcompCodec {
    backend: ferro_compress::Backend,
    algo: ferro_compress::Algo,
}

#[cfg(feature = "ferro-compress")]
impl std::fmt::Debug for FerroBitcompCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FerroBitcompCodec")
            .field("backend_kind", &self.backend.kind())
            .field("algo", &self.algo)
            .finish()
    }
}

#[cfg(feature = "ferro-compress")]
impl Default for FerroBitcompCodec {
    fn default() -> Self {
        // Snappy on CPU: best CPU-decompress per Phase 0 (6.29 GB/s),
        // and 8 KiB containers fall well below the GPU dispatch
        // threshold so even `Backend::Auto` would route here. Picking
        // it explicitly keeps the default a fast, predictable path.
        FerroBitcompCodec {
            backend: ferro_compress::Backend::cpu(),
            algo: ferro_compress::Algo::Snappy,
        }
    }
}

#[cfg(feature = "ferro-compress")]
impl FerroBitcompCodec {
    /// Construct with an explicit [`ferro_compress::Backend`] and
    /// [`ferro_compress::Algo`]. Use this when the call site knows the
    /// column shape (e.g. typed-numeric postings → `Algo::Bitcomp`)
    /// and/or wants GPU dispatch via `Backend::Auto`.
    #[must_use]
    pub fn with_backend(backend: ferro_compress::Backend, algo: ferro_compress::Algo) -> Self {
        Self { backend, algo }
    }

    /// Convenience: Snappy on CPU. Same as [`Default::default`], named
    /// for discoverability.
    #[must_use]
    pub fn snappy_cpu() -> Self {
        Self::default()
    }

    /// Convenience: Bitcomp uint32 on whichever backend the supplied
    /// [`ferro_compress::Backend`] resolves to. Phase 0 measured 3.59×
    /// ratio / 366 GB/s decomp on `postings.bin` at this setting; the
    /// CPU backend rejects Bitcomp explicitly (no CPU implementation),
    /// so this constructor only succeeds at compress/decompress time
    /// when called against a backend that can service the algorithm.
    #[must_use]
    pub fn bitcomp_uint32(backend: ferro_compress::Backend) -> Self {
        Self {
            backend,
            algo: ferro_compress::Algo::for_postings_uint32(),
        }
    }
}

#[cfg(feature = "ferro-compress")]
impl BitcompCodec for FerroBitcompCodec {
    fn compress(&self, raw_bitmap: &[u32]) -> Result<Vec<u8>, BitcompError> {
        if raw_bitmap.len() != BITMAP_CONTAINER_WORDS {
            return Err(BitcompError::LengthMismatch {
                expected: BITMAP_CONTAINER_WORDS,
                got: raw_bitmap.len(),
            });
        }
        // Pack u32 words to LE bytes (codec sees a flat byte stream so
        // round-trip is endian-stable across architectures). Snappy /
        // LZ4 / zstd are byte-oriented and don't care; Bitcomp uses the
        // typed-numeric hint passed to the codec for layout — for
        // current 8 KiB Roaring containers we let `Backend::codec(algo)`
        // surface that decision rather than hard-coding `Uint32` here.
        let mut input_bytes = Vec::with_capacity(BITMAP_CONTAINER_WORDS * 4);
        for &w in raw_bitmap {
            input_bytes.extend_from_slice(&w.to_le_bytes());
        }
        let codec = self
            .backend
            .codec(self.algo)
            .map_err(|e| BitcompError::CodecFailure(format!("backend.codec failed: {e}")))?;
        let mut out = Vec::with_capacity(codec.max_compressed_len(input_bytes.len()));
        codec
            .compress(&input_bytes, &mut out)
            .map_err(|e| BitcompError::CodecFailure(format!("compress failed: {e}")))?;
        Ok(out)
    }

    fn decompress(
        &self,
        compressed: &[u8],
        out: &mut [u32; BITMAP_CONTAINER_WORDS],
    ) -> Result<(), BitcompError> {
        let codec = self
            .backend
            .codec(self.algo)
            .map_err(|e| BitcompError::CodecFailure(format!("backend.codec failed: {e}")))?;
        let mut decoded = Vec::with_capacity(BITMAP_CONTAINER_WORDS * 4);
        codec
            .decompress(compressed, &mut decoded)
            .map_err(|e| BitcompError::CodecFailure(format!("decompress failed: {e}")))?;
        if decoded.len() != BITMAP_CONTAINER_WORDS * 4 {
            return Err(BitcompError::NotContainerSized { got: decoded.len() });
        }
        for i in 0..BITMAP_CONTAINER_WORDS {
            let off = i * 4;
            out[i] = u32::from_le_bytes([
                decoded[off],
                decoded[off + 1],
                decoded[off + 2],
                decoded[off + 3],
            ]);
        }
        Ok(())
    }
}

// ============================================================
// Default selector — feature-gated.
// ============================================================

/// Project-default codec.
///
/// - With `ferro-compress` **off** (default): [`IdentityBitcompCodec`].
/// - With `ferro-compress` **on**: [`FerroBitcompCodec`] — Wave 6
///   wired this to a real [`ferro_compress::Backend::cpu`] +
///   [`ferro_compress::Algo::Snappy`] round-trip.
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
        // Wave 6: with the feature on, DefaultBitcompCodec aliases to
        // FerroBitcompCodec which now does a real ferro_compress
        // round-trip (Snappy on CPU by default). Round-trip cleanly.
        let bm = make_container();
        let codec = DefaultBitcompCodec::default();
        let bytes = compress_container(&bm, &codec).expect("compress should succeed");
        let parsed = decompress_container(&bytes, &codec).expect("decompress should succeed");
        assert_eq!(parsed, bm);
    }

    #[cfg(feature = "ferro-compress")]
    #[test]
    fn ferro_codec_round_trip_snappy_cpu() {
        // Default constructor — Snappy on CPU. Round-trip clean.
        let codec = FerroBitcompCodec::snappy_cpu();
        let bm = make_container();
        let bytes = compress_container(&bm, &codec).expect("compress should succeed");
        // Snappy compresses redundancy; for a sparsely-populated container
        // we expect output strictly smaller than the raw 8 KiB.
        assert!(
            bytes.len() < BITMAP_CONTAINER_WORDS * 4,
            "expected snappy to compress 8 KiB raw bitmap to < 8 KiB; got {} bytes",
            bytes.len()
        );
        let parsed = decompress_container(&bytes, &codec).expect("decompress should succeed");
        assert_eq!(parsed, bm);
    }

    #[cfg(feature = "ferro-compress")]
    #[test]
    fn ferro_codec_round_trip_zero_container() {
        // All-zero bitmap is the worst-case for Snappy frame metadata
        // overhead; round-trip must still be identity-correct.
        let codec = FerroBitcompCodec::snappy_cpu();
        let bm = BitmapContainer::new();
        let bytes = compress_container(&bm, &codec).expect("compress should succeed");
        let parsed = decompress_container(&bytes, &codec).expect("decompress should succeed");
        assert_eq!(parsed, bm);
    }

    #[cfg(feature = "ferro-compress")]
    #[test]
    fn ferro_codec_round_trip_full_container() {
        // Fully populated 65 536-bit container.
        let mut bm = BitmapContainer::new();
        for k in 0u32..65_536 {
            bm.insert(k as u16);
        }
        let codec = FerroBitcompCodec::snappy_cpu();
        let bytes = compress_container(&bm, &codec).expect("compress should succeed");
        let parsed = decompress_container(&bytes, &codec).expect("decompress should succeed");
        assert_eq!(parsed, bm);
    }

    #[cfg(feature = "ferro-compress")]
    #[test]
    fn ferro_codec_rejects_wrong_input_length() {
        let codec = FerroBitcompCodec::snappy_cpu();
        let res = codec.compress(&[0u32; 10]);
        assert!(matches!(res, Err(BitcompError::LengthMismatch { .. })));
    }

    #[cfg(feature = "ferro-compress")]
    #[test]
    fn ferro_codec_decompress_rejects_corrupt_input() {
        let codec = FerroBitcompCodec::snappy_cpu();
        let mut out = [0u32; BITMAP_CONTAINER_WORDS];
        // Garbage Snappy frame.
        let res = codec.decompress(&[0xFFu8; 16], &mut out);
        assert!(matches!(res, Err(BitcompError::CodecFailure(_))));
    }

    #[cfg(feature = "ferro-compress")]
    #[test]
    fn ferro_codec_round_trip_random_words() {
        // Deterministic xorshift fill — exercises codec on chaotic
        // input where Snappy ratio is poor (literal-heavy frame).
        let mut bm = BitmapContainer::new();
        let mut state: u64 = 0xdead_beef_cafe_babe;
        for _ in 0..3000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let key = (state as u32) % 65536;
            bm.insert(key as u16);
        }
        let codec = FerroBitcompCodec::snappy_cpu();
        let bytes = compress_container(&bm, &codec).expect("compress should succeed");
        let parsed = decompress_container(&bytes, &codec).expect("decompress should succeed");
        assert_eq!(parsed, bm);
    }
}

//! Phase 2 D-5 — warm-restart persistence common foundation.
//!
//! Shared on-disk wire-format primitives for the three CHT tiers
//! ([`super::cht`] v1 host, [`super::vram_cht`] v2 uncompressed VRAM,
//! [`super::vram_cht_v3`] v3 Bitcomp-compressed VRAM). Each tier's
//! `dump_to_path` / `load_from_path` impl threads its tier-specific
//! magic + body encoding through these helpers; cross-tier invariants
//! (version field, hash-function gate, end-of-stream sentinel, atomic
//! write protocol) live here so the three tiers stay in lockstep.
//!
//! ## Wire format common shape
//!
//! Every dump file is **little-endian** and follows:
//!
//! ```text
//! File header (24 bytes):
//!   magic            u32   tier-specific (FCV1 / FCV2 / FCV3)
//!   version          u32   = 1
//!   hash_function    u32   = HASH_FN_FXHASHER_V1 (= 0)
//!   entry_count      u64
//!   reserved         u32   = 0  (alignment + future flags)
//!
//! Per-entry ChtKey header (44 bytes):
//!   segment_id_hex   [u8; 32]   UUID as 32 ASCII lowercase hex chars
//!   field_id         u32
//!   term_hash        u64
//!
//! Tier-specific body — see [`super::cht`], [`super::vram_cht`],
//! [`super::vram_cht_v3`] for the per-tier encoding.
//!
//! Trailer (4 bytes):
//!   end_magic        u32   = MAGIC_END
//! ```
//!
//! ### Why ASCII hex for the segment id
//!
//! [`crate::index::SegmentId`] wraps `uuid::Uuid` but exposes only
//! `uuid_string()` / `from_uuid_string()` publicly — the inner 16-byte
//! representation is sealed. Storing the 32-char hex form costs 16
//! extra bytes per entry vs the raw UUID bytes; for the entry counts
//! the CHT admits (≤ 32 cohort terms × a few hundred segments) this
//! is negligible (KB-class overhead on a multi-MB dump) and keeps
//! D-5 from forking the segment_id surface.
//!
//! ## Hash-function gate
//!
//! `hash_function` in the file header pins the [`super::cht::ChtKey`]
//! `term_hash` provenance. Today that's [`HASH_FN_FXHASHER_V1`] (=
//! `rustc-hash::FxHasher` with the default seed, stable across calls
//! within a `rustc-hash` major version). A future hash-function swap
//! bumps this constant; loaders observing a mismatch return
//! [`LoadError::HashFunctionMismatch`] so the operator restarts cold
//! (no corruption, just an empty cache) rather than walk keys that no
//! longer line up with the current `hash_term_bytes` output.
//!
//! ## Atomic write protocol
//!
//! [`atomic_write`] writes `<path>.tmp`, calls `fsync` on the temp
//! file, then `rename`s it onto `<path>`. POSIX guarantees rename is
//! atomic on the same filesystem, so a crashed `dump` leaves either
//! the old file untouched or the fully-written new file — never a
//! half-written `cht_v{1,2,3}.bin`. The end-of-stream
//! [`MAGIC_END`] sentinel inside the body lets loaders distinguish a
//! truncated `.tmp` (caught by rename atomicity) from a corruption
//! that survived rename (caught by missing trailer).
//!
//! ## What this module does NOT do
//!
//! - **Per-tier byte encoding.** Body framing lives next to each
//!   tier's struct so the in-tree code paths stay co-located with
//!   their `Drop` / `cudaMalloc` semantics.
//! - **Per-entry CRC.** End-of-stream sentinel + filesystem rename
//!   atomicity are sufficient for the D-5 acceptance gate (operator
//!   shutdown is a controlled state). Per-entry CRC would catch silent
//!   disk-corruption scenarios; deferred as a future polish if dump
//!   sizes grow past a few GiB.
//! - **Encryption / KMS wrapping.** Posting bitmaps don't carry
//!   secret values but the operator may want at-rest encryption on
//!   shared hosts; that's a Phase 2 G follow-up (mirrors the
//!   per-request sampling persistence path).

#![cfg(all(feature = "gpu", feature = "ferro-compress"))]

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::index::SegmentId;
use crate::postings::roaring::cht::ChtKey;

/// Tier-specific file magic for the v1 host-memory CHT dump.
/// On disk the four bytes spell `"FCV1"` (operator-greppable in
/// `xxd` / `hexdump`).
pub const MAGIC_V1: u32 = u32::from_le_bytes(*b"FCV1");

/// Tier-specific file magic for the v2 uncompressed VRAM CHT dump.
/// On disk: `"FCV2"`.
pub const MAGIC_V2: u32 = u32::from_le_bytes(*b"FCV2");

/// Tier-specific file magic for the v3 Bitcomp-compressed VRAM CHT
/// dump (single-chunk per entry — pre-Wave-Z-6-#4 wire format). On
/// disk: `"FCV3"`. Loaded by [`super::vram_cht_v3::VramCompressedCht::load_from_path`]
/// as a legacy 1-chunk Multi entry for backward compatibility with
/// dumps produced by pre-Z-6-#4 binaries.
pub const MAGIC_V3: u32 = u32::from_le_bytes(*b"FCV3");

/// Tier-specific file magic for the v3 Bitcomp-compressed VRAM CHT
/// dump with multi-chunk records (Wave Z-6 #4 wire format). On disk:
/// `"FCV4"`. Per-term body emits `chunk_count: u32` followed by `N`
/// `(compressed_bytes: u32, body: comp_size bytes)` records; per-chunk
/// `uncompressed_bytes` is derived at load time from `bucket_count`
/// and `chunk_count` (bucket-boundary chunking invariant). Replaces
/// [`MAGIC_V3`] as the default dump magic; the V3 load path stays for
/// back-compat with operator-snapshotted dumps in the wild.
pub const MAGIC_V4: u32 = u32::from_le_bytes(*b"FCV4");

/// End-of-stream sentinel. On disk: `"ENDV"`. Located 4 bytes
/// before EOF; loaders observing a missing / mismatched trailer
/// reject the dump as truncated.
pub const MAGIC_END: u32 = u32::from_le_bytes(*b"ENDV");

/// Current wire-format version. Bumps in lockstep across the three
/// tiers — loaders observing a mismatch return
/// [`LoadError::VersionMismatch`].
pub const WIRE_VERSION: u32 = 1;

/// Hash-function identifier for `rustc-hash::FxHasher` with the
/// default seed (= the function [`super::cht::hash_term_bytes`]
/// implements as of Wave 11). Pinning the identifier in the file
/// header lets a future swap detect mismatched dumps cleanly.
pub const HASH_FN_FXHASHER_V1: u32 = 0;

/// File-header size in bytes (magic + version + hash_function +
/// entry_count + reserved).
pub const FILE_HEADER_BYTES: usize = 4 + 4 + 4 + 8 + 4;

/// Per-entry [`ChtKey`] header size in bytes (32 hex + field_id +
/// term_hash).
pub const CHTKEY_HEADER_BYTES: usize = 32 + 4 + 8;

/// File-trailer size in bytes (just [`MAGIC_END`]).
pub const FILE_TRAILER_BYTES: usize = 4;

/// Errors a `load_from_path` impl can return.
///
/// Variants are deliberately granular so an operator log line points
/// at the exact reason a dump was rejected: a `WrongMagic` /
/// `VersionMismatch` / `HashFunctionMismatch` is an expected
/// migration scenario (cold-start fallback); `TruncatedDump` /
/// `BadEndMagic` / `Io` signal something worse (disk corruption,
/// partial write that survived rename — should not happen with
/// [`atomic_write`]).
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// The file's magic header didn't match the expected tier
    /// constant. Either the file is for a different tier (someone
    /// pointed `cht_v2.bin` at the v3 loader, etc.) or it's not a
    /// CHT dump at all.
    #[error("dump file magic mismatch: expected 0x{expected:08x}, got 0x{got:08x}")]
    WrongMagic {
        /// Tier-specific expected magic.
        expected: u32,
        /// Magic bytes actually read.
        got: u32,
    },
    /// The file's `version` field didn't match [`WIRE_VERSION`].
    /// Future loaders may handle older versions explicitly; for now
    /// the loader returns cold-start.
    #[error("dump file version mismatch: expected {expected}, got {got}")]
    VersionMismatch {
        /// Loader's expected wire version.
        expected: u32,
        /// Version bytes actually read.
        got: u32,
    },
    /// The file's `hash_function` field didn't match the expected
    /// constant. The pre-restart process used a different
    /// `term_hash` source, so cached keys won't line up with keys
    /// the current process will build — operator restarts cold.
    #[error("dump file hash function mismatch: expected 0x{expected:08x}, got 0x{got:08x}")]
    HashFunctionMismatch {
        /// Loader's expected hash-function identifier.
        expected: u32,
        /// Hash-function id actually read.
        got: u32,
    },
    /// File ended before the expected number of entries / trailer
    /// were read. Likely a corrupted `.tmp` that survived rename
    /// (atomic_write's rename should prevent this; if seen, the
    /// host filesystem is the suspect).
    #[error("dump file truncated: needed {needed} more bytes")]
    TruncatedDump {
        /// Bytes the loader expected past the truncation point.
        needed: usize,
    },
    /// Trailer magic didn't match [`MAGIC_END`]. Same diagnostic as
    /// `TruncatedDump`: operator should investigate disk state.
    #[error("dump file end-magic mismatch: expected 0x{expected:08x}, got 0x{got:08x}")]
    BadEndMagic {
        /// Loader's expected end-magic.
        expected: u32,
        /// End-magic bytes actually read.
        got: u32,
    },
    /// The 32-byte segment-id hex slice wasn't valid UTF-8 or wasn't
    /// a parseable UUID. Indicates intra-entry corruption (more
    /// localised than `TruncatedDump`).
    #[error("dump file invalid segment_id hex at entry {entry_index}: {detail}")]
    InvalidSegmentId {
        /// Zero-based entry index where parsing failed.
        entry_index: u64,
        /// Human-readable parse failure.
        detail: String,
    },
    /// I/O failure during read (disk error, file disappeared, etc.).
    #[error("dump file I/O error: {0}")]
    Io(#[from] io::Error),
    /// A tier-specific body decode failed. The tier's body parser
    /// surfaces its own error type via this wrapper so the persist
    /// module stays oblivious to per-tier wire details.
    #[error("dump body decode failed at entry {entry_index}: {detail}")]
    BodyDecode {
        /// Zero-based entry index where parsing failed.
        entry_index: u64,
        /// Human-readable per-tier error.
        detail: String,
    },
    /// CUDA call failed during load (e.g. `cudaMalloc` for the
    /// per-entry device buffer). The loader treats this as
    /// best-effort: the failed entry is skipped, the rest continue.
    /// This variant is reported back as the *terminal* error only
    /// when **all** entries fail.
    #[error("CUDA error during load at entry {entry_index} (bytes={bytes}, code={code})")]
    Cuda {
        /// Zero-based entry index where the CUDA call failed.
        entry_index: u64,
        /// Bytes the failing call requested.
        bytes: usize,
        /// Raw `cudaError_t` value.
        code: u32,
    },
}

/// Errors a `dump_to_path` impl can return.
///
/// Most failures are I/O (disk full, permission denied, etc.); a
/// `Cuda` variant covers `cudaMemcpyDeviceToHost` failures on the
/// v2 / v3 tiers when staging device buffers back to host for write.
#[derive(Debug, thiserror::Error)]
pub enum DumpError {
    /// I/O failure during write.
    #[error("dump I/O error: {0}")]
    Io(#[from] io::Error),
    /// `cudaMemcpyDeviceToHost` failed when staging a device buffer
    /// back to host. Per-entry; the dumper aborts (the partial
    /// `.tmp` file is rolled back by [`atomic_write`]'s caller via
    /// not running the final rename).
    #[error("CUDA error during dump at entry {entry_index} (bytes={bytes}, code={code})")]
    Cuda {
        /// Zero-based entry index where the CUDA call failed.
        entry_index: u64,
        /// Bytes the failing call requested.
        bytes: usize,
        /// Raw `cudaError_t` value.
        code: u32,
    },
}

/// Convert a [`PathBuf`] argument into the `<path>.tmp` staging
/// path. Centralised so the dumpers can't drift on the tmp
/// convention.
#[must_use]
pub fn tmp_path_for(p: &Path) -> PathBuf {
    let mut tmp = p.as_os_str().to_owned();
    tmp.push(".tmp");
    PathBuf::from(tmp)
}

/// Open `<path>.tmp` for writing — creating / truncating as needed.
/// Wraps it in a `BufWriter` so per-entry writes coalesce into
/// page-sized chunks.
pub fn open_tmp_writer(path: &Path) -> io::Result<BufWriter<File>> {
    let tmp = tmp_path_for(path);
    // Best-effort: ensure parent dir exists. POSIX `rename` requires
    // the source and dest to be on the same filesystem; we don't
    // validate that (operator owns the path), but creating the dir
    // here is cheap and matches the lifecycle hook in
    // `bins/ferrosearch/src/main.rs` which calls `create_dir_all`
    // before invoking dumpers.
    if let Some(parent) = tmp.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)?;
    Ok(BufWriter::new(f))
}

/// Finalise the atomic-write protocol: flush + fsync the temp file,
/// then rename it over the final path. Caller passes the original
/// final path (not the `.tmp`); `<path>.tmp` is derived via
/// [`tmp_path_for`].
///
/// Failure modes:
/// - `flush` failure: partial write — `.tmp` is left behind for
///   operator inspection; final path is untouched.
/// - `sync_all` failure: same as above.
/// - `rename` failure: rare (same-fs constraint or permission); the
///   `.tmp` is left behind, final path is untouched.
///
/// Caller is expected to use [`open_tmp_writer`] earlier in the same
/// flow so the file actually exists at `<path>.tmp`.
pub fn finalise_atomic_write(mut writer: BufWriter<File>, path: &Path) -> io::Result<()> {
    writer.flush()?;
    let file = writer.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()?;
    let tmp = tmp_path_for(path);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Open `<path>` for read — wraps in `BufReader` so per-entry reads
/// coalesce.
pub fn open_reader(path: &Path) -> io::Result<BufReader<File>> {
    let f = File::open(path)?;
    Ok(BufReader::new(f))
}

/// Read exactly `n` bytes into a freshly-allocated `Vec<u8>`,
/// returning [`LoadError::TruncatedDump`] on premature EOF rather
/// than the generic `io::ErrorKind::UnexpectedEof`. Keeps the
/// per-tier loaders from having to translate every `read_exact`
/// failure.
pub fn read_exact_or_truncated<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>, LoadError> {
    let mut buf = vec![0u8; n];
    match r.read_exact(&mut buf) {
        Ok(()) => Ok(buf),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
            Err(LoadError::TruncatedDump { needed: n })
        }
        Err(e) => Err(LoadError::Io(e)),
    }
}

/// Read a little-endian `u32`. Surfaces [`LoadError::TruncatedDump`]
/// on EOF.
pub fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, LoadError> {
    let bytes = read_exact_or_truncated(r, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Read a little-endian `u64`.
pub fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, LoadError> {
    let bytes = read_exact_or_truncated(r, 8)?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

/// Read a little-endian `u16`.
pub fn read_u16_le<R: Read>(r: &mut R) -> Result<u16, LoadError> {
    let bytes = read_exact_or_truncated(r, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Write the file header in canonical little-endian order. Used by
/// every per-tier dumper so the 16-byte prefix stays identical.
pub fn write_file_header<W: Write>(
    w: &mut W,
    magic: u32,
    entry_count: u64,
) -> io::Result<()> {
    w.write_all(&magic.to_le_bytes())?;
    w.write_all(&WIRE_VERSION.to_le_bytes())?;
    w.write_all(&HASH_FN_FXHASHER_V1.to_le_bytes())?;
    w.write_all(&entry_count.to_le_bytes())?;
    w.write_all(&0u32.to_le_bytes())?; // reserved
    Ok(())
}

/// Read + validate the file header. Returns `entry_count`. The
/// `expected_magic` argument is tier-specific (caller passes
/// [`MAGIC_V1`] / [`MAGIC_V2`] / [`MAGIC_V3`]).
pub fn read_and_validate_file_header<R: Read>(
    r: &mut R,
    expected_magic: u32,
) -> Result<u64, LoadError> {
    let magic = read_u32_le(r)?;
    if magic != expected_magic {
        return Err(LoadError::WrongMagic {
            expected: expected_magic,
            got: magic,
        });
    }
    let version = read_u32_le(r)?;
    if version != WIRE_VERSION {
        return Err(LoadError::VersionMismatch {
            expected: WIRE_VERSION,
            got: version,
        });
    }
    let hash_fn = read_u32_le(r)?;
    if hash_fn != HASH_FN_FXHASHER_V1 {
        return Err(LoadError::HashFunctionMismatch {
            expected: HASH_FN_FXHASHER_V1,
            got: hash_fn,
        });
    }
    let entry_count = read_u64_le(r)?;
    let _reserved = read_u32_le(r)?;
    Ok(entry_count)
}

/// Same as [`read_and_validate_file_header`] but accepts a slice of
/// allowed magics and returns the matched magic alongside the entry
/// count. Used by tier loaders that support multiple wire-format
/// versions in parallel (e.g. v3 cache accepting both [`MAGIC_V3`]
/// single-chunk and [`MAGIC_V4`] multi-chunk dumps for back-compat).
///
/// `WrongMagic` reports the first (= "primary" / preferred) element
/// of `expected_magics` as the expected value when no magic matches —
/// the variant is informational, not exhaustive.
pub fn read_and_validate_file_header_multi<R: Read>(
    r: &mut R,
    expected_magics: &[u32],
) -> Result<(u32, u64), LoadError> {
    let magic = read_u32_le(r)?;
    if !expected_magics.iter().any(|&m| m == magic) {
        return Err(LoadError::WrongMagic {
            expected: *expected_magics.first().unwrap_or(&0),
            got: magic,
        });
    }
    let version = read_u32_le(r)?;
    if version != WIRE_VERSION {
        return Err(LoadError::VersionMismatch {
            expected: WIRE_VERSION,
            got: version,
        });
    }
    let hash_fn = read_u32_le(r)?;
    if hash_fn != HASH_FN_FXHASHER_V1 {
        return Err(LoadError::HashFunctionMismatch {
            expected: HASH_FN_FXHASHER_V1,
            got: hash_fn,
        });
    }
    let entry_count = read_u64_le(r)?;
    let _reserved = read_u32_le(r)?;
    Ok((magic, entry_count))
}

/// Write the trailing [`MAGIC_END`] sentinel.
pub fn write_file_trailer<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_all(&MAGIC_END.to_le_bytes())?;
    Ok(())
}

/// Read + validate the trailing [`MAGIC_END`] sentinel.
pub fn read_and_validate_file_trailer<R: Read>(r: &mut R) -> Result<(), LoadError> {
    let m = read_u32_le(r)?;
    if m != MAGIC_END {
        return Err(LoadError::BadEndMagic {
            expected: MAGIC_END,
            got: m,
        });
    }
    Ok(())
}

/// Serialise a [`ChtKey`] in the 44-byte common per-entry header
/// form. Uses [`SegmentId::uuid_string`] (32-char lowercase hex)
/// rather than the sealed inner UUID bytes.
pub fn write_chtkey<W: Write>(w: &mut W, k: &ChtKey) -> io::Result<()> {
    let hex = k.segment_id.uuid_string();
    debug_assert_eq!(hex.len(), 32, "uuid_string must return 32 hex chars");
    w.write_all(hex.as_bytes())?;
    w.write_all(&k.field.to_le_bytes())?;
    w.write_all(&k.term_hash.to_le_bytes())?;
    Ok(())
}

/// Parse a [`ChtKey`] from the 44-byte common per-entry header form.
/// `entry_index` is threaded through for error context only.
pub fn read_chtkey<R: Read>(r: &mut R, entry_index: u64) -> Result<ChtKey, LoadError> {
    let hex_bytes = read_exact_or_truncated(r, 32)?;
    let hex = std::str::from_utf8(&hex_bytes).map_err(|e| LoadError::InvalidSegmentId {
        entry_index,
        detail: format!("non-utf8 segment_id hex: {e}"),
    })?;
    let segment_id =
        SegmentId::from_uuid_string(hex).map_err(|e| LoadError::InvalidSegmentId {
            entry_index,
            detail: format!("malformed uuid hex: {e}"),
        })?;
    let field = read_u32_le(r)?;
    let term_hash = read_u64_le(r)?;
    Ok(ChtKey {
        segment_id,
        field,
        term_hash,
    })
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn magic_constants_match_ascii_text() {
        // The constants are documented to be the LE-bytes of the
        // ASCII strings — this test pins that contract because
        // operators may want to grep dump files in hex viewers.
        assert_eq!(MAGIC_V1.to_le_bytes(), *b"FCV1");
        assert_eq!(MAGIC_V2.to_le_bytes(), *b"FCV2");
        assert_eq!(MAGIC_V3.to_le_bytes(), *b"FCV3");
        assert_eq!(MAGIC_END.to_le_bytes(), *b"ENDV");
    }

    #[test]
    fn file_header_roundtrip() {
        let mut buf: Vec<u8> = Vec::new();
        write_file_header(&mut buf, MAGIC_V3, 42).unwrap();
        assert_eq!(buf.len(), FILE_HEADER_BYTES);
        let mut cur = Cursor::new(&buf);
        let n = read_and_validate_file_header(&mut cur, MAGIC_V3).unwrap();
        assert_eq!(n, 42);
    }

    #[test]
    fn file_header_wrong_magic_rejected() {
        let mut buf: Vec<u8> = Vec::new();
        write_file_header(&mut buf, MAGIC_V2, 0).unwrap();
        let mut cur = Cursor::new(&buf);
        let err = read_and_validate_file_header(&mut cur, MAGIC_V3).unwrap_err();
        match err {
            LoadError::WrongMagic { expected, got } => {
                assert_eq!(expected, MAGIC_V3);
                assert_eq!(got, MAGIC_V2);
            }
            other => panic!("expected WrongMagic, got {other:?}"),
        }
    }

    #[test]
    fn file_header_version_mismatch_rejected() {
        // Manually write a header with version != WIRE_VERSION.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&MAGIC_V3.to_le_bytes());
        buf.extend_from_slice(&999u32.to_le_bytes()); // version
        buf.extend_from_slice(&HASH_FN_FXHASHER_V1.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        let mut cur = Cursor::new(&buf);
        let err = read_and_validate_file_header(&mut cur, MAGIC_V3).unwrap_err();
        match err {
            LoadError::VersionMismatch { expected, got } => {
                assert_eq!(expected, WIRE_VERSION);
                assert_eq!(got, 999);
            }
            other => panic!("expected VersionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn file_header_hash_function_mismatch_rejected() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&MAGIC_V3.to_le_bytes());
        buf.extend_from_slice(&WIRE_VERSION.to_le_bytes());
        buf.extend_from_slice(&999u32.to_le_bytes()); // hash_function
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        let mut cur = Cursor::new(&buf);
        let err = read_and_validate_file_header(&mut cur, MAGIC_V3).unwrap_err();
        match err {
            LoadError::HashFunctionMismatch { expected, got } => {
                assert_eq!(expected, HASH_FN_FXHASHER_V1);
                assert_eq!(got, 999);
            }
            other => panic!("expected HashFunctionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn file_header_truncated_rejected() {
        // Only 4 bytes (= magic only).
        let buf = MAGIC_V3.to_le_bytes().to_vec();
        let mut cur = Cursor::new(&buf);
        let err = read_and_validate_file_header(&mut cur, MAGIC_V3).unwrap_err();
        assert!(matches!(err, LoadError::TruncatedDump { .. }));
    }

    #[test]
    fn file_trailer_roundtrip() {
        let mut buf: Vec<u8> = Vec::new();
        write_file_trailer(&mut buf).unwrap();
        assert_eq!(buf.len(), FILE_TRAILER_BYTES);
        let mut cur = Cursor::new(&buf);
        read_and_validate_file_trailer(&mut cur).unwrap();
    }

    #[test]
    fn file_trailer_bad_magic_rejected() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let mut cur = Cursor::new(&buf);
        let err = read_and_validate_file_trailer(&mut cur).unwrap_err();
        match err {
            LoadError::BadEndMagic { expected, got } => {
                assert_eq!(expected, MAGIC_END);
                assert_eq!(got, 0xDEAD_BEEF);
            }
            other => panic!("expected BadEndMagic, got {other:?}"),
        }
    }

    #[test]
    fn chtkey_roundtrip() {
        let original = ChtKey {
            segment_id: SegmentId::generate_random(),
            field: 0x1234_5678,
            term_hash: 0xCAFE_BABE_DEAD_BEEFu64,
        };
        let mut buf: Vec<u8> = Vec::new();
        write_chtkey(&mut buf, &original).unwrap();
        assert_eq!(buf.len(), CHTKEY_HEADER_BYTES);
        let mut cur = Cursor::new(&buf);
        let parsed = read_chtkey(&mut cur, 0).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn chtkey_invalid_hex_rejected() {
        let mut buf: Vec<u8> = vec![b'Z'; 32]; // non-hex chars
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let mut cur = Cursor::new(&buf);
        let err = read_chtkey(&mut cur, 7).unwrap_err();
        match err {
            LoadError::InvalidSegmentId { entry_index, .. } => {
                assert_eq!(entry_index, 7);
            }
            other => panic!("expected InvalidSegmentId, got {other:?}"),
        }
    }

    #[test]
    fn tmp_path_for_appends_tmp_suffix() {
        let p = Path::new("/data/cht/cht_v3.bin");
        let tmp = tmp_path_for(p);
        assert_eq!(tmp.to_string_lossy(), "/data/cht/cht_v3.bin.tmp");
    }

    #[test]
    fn atomic_write_lifecycle() {
        // Full atomic write: open tmp, write header + trailer,
        // finalise. Resulting file must exist at the final path
        // (not at .tmp) and contain exactly the bytes we wrote.
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("cht_test.bin");
        let mut writer = open_tmp_writer(&final_path).unwrap();
        write_file_header(&mut writer, MAGIC_V1, 0).unwrap();
        write_file_trailer(&mut writer).unwrap();
        finalise_atomic_write(writer, &final_path).unwrap();

        assert!(final_path.exists(), "final file must exist after rename");
        assert!(
            !tmp_path_for(&final_path).exists(),
            ".tmp must be gone after rename"
        );
        // Roundtrip read.
        let mut reader = open_reader(&final_path).unwrap();
        let n = read_and_validate_file_header(&mut reader, MAGIC_V1).unwrap();
        assert_eq!(n, 0);
        read_and_validate_file_trailer(&mut reader).unwrap();
    }
}

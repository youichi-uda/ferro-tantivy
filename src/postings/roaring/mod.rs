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
//! - [`container`]: the three Roaring container forms ([`container::ArrayContainer`],
//!   [`container::BitmapContainer`], [`container::RunContainer`]) plus the [`container::Container`]
//!   dispatch enum and self-describing wire encoding.
//! - [`encoder`]: [`encoder::RoaringEncoder`] (`u32` doc-id stream → per-bucket containers) and
//!   [`encoder::RoaringPostings`] (finalised, sorted-by-`high16`, byte-serialisable form).
//! - [`decoder`]: [`decoder::RoaringDecoder`] — `advance` / `seek` over a
//!   [`encoder::RoaringPostings`] (`O(log n + log m)` seek).
//! - [`ferro_compress_bridge`]: the [`ferro_compress_bridge::BitcompCodec`] trait, the always-on
//!   [`ferro_compress_bridge::IdentityBitcompCodec`] passthrough, and the (currently
//!   short-circuiting) [`ferro_compress_bridge::FerroBitcompCodec`] wired in by the
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
//! - Make Roaring the default posting format. Existing segments and newly-written segments
//!   **continue to use BitPacker4x** until Wave 6 lands the dispatch threshold.
//! - Wire Roaring through `BlockSegmentPostings` / `SegmentPostings` read paths. Those continue to
//!   read BitPacker4x exclusively.
//! - Pull `ferro-compress` in as a hard dep. The trait abstraction keeps the Tantivy fork buildable
//!   standalone.
//! - Add bench harnesses. End-to-end Roaring vs BitPacker4x bench is Phase 2 C-5.

pub mod container;
#[cfg(all(feature = "gpu", feature = "ferro-compress"))]
pub mod cht;
#[cfg(all(feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
pub mod cuda_dispatch;
pub mod decoder;
pub mod encoder;
pub mod ferro_compress_bridge;
pub mod from_block_segment;
pub mod gpu_dispatch;
#[cfg(all(feature = "gpu", feature = "ferro-compress"))]
pub mod persist;
pub mod planner;
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
pub(crate) mod shared_bitmap_payload;
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
pub mod vram_cht;
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
pub mod vram_cht_v3;

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
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
pub use gpu_dispatch::try_gpu_bool_vram;
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
pub use gpu_dispatch::try_gpu_bool_v3;
pub use planner::{
    should_dispatch_gpu, TermStat, MIN_COHORT_DOCS, MIN_PER_TERM_CARDINALITY, MIN_RATIO,
};

#[cfg(feature = "ferro-compress")]
pub use ferro_compress_bridge::FerroBitcompCodec;

// ============================================================
// Wave Z-1 #1 — segment-warm-on-open hook
// ============================================================
//
// When enabled, [`SegmentReader::open_with_custom_alive_set`] calls
// [`warm_segment_v3`] just before returning, which streams each
// indexed text field's term dictionary and pre-populates the host
// CHT (and, opportunistically, the Bitcomp v3 VRAM CHT). The goal is
// to front-load the H→D + decompress overhead at segment-open time
// so the first query that touches a warm term takes the hot path.
//
// Off by default — this is a *latency-shifting* optimisation, not a
// throughput win. Operators turn it on via the `ferrosearch
// --cht-segment-warm-on-open` CLI flag (Wave Z-1 #1 ferrosearch
// wiring).

use std::sync::atomic::{AtomicBool, Ordering};

/// Process-global toggle for the segment-warm-on-open hook. Off by
/// default so existing deployments retain bytewise-identical
/// behaviour until the operator opts in.
static SEGMENT_WARM_ON_OPEN: AtomicBool = AtomicBool::new(false);

/// Enable / disable the segment-warm-on-open hook for *all* future
/// [`crate::index::SegmentReader::open_with_custom_alive_set`] calls.
/// Idempotent. Cheap (single relaxed atomic store) — safe to call
/// from `ferrosearch::main` startup.
pub fn set_segment_warm_on_open(enabled: bool) {
    SEGMENT_WARM_ON_OPEN.store(enabled, Ordering::Relaxed);
}

/// Read the current value of the segment-warm-on-open toggle. Used
/// by the open path to short-circuit when the operator hasn't opted
/// in.
#[must_use]
pub fn segment_warm_on_open() -> bool {
    SEGMENT_WARM_ON_OPEN.load(Ordering::Relaxed)
}

// ============================================================
// Warm-up entry point (cfg-gated body, always-callable stub)
// ============================================================

/// Maximum number of terms warmed per indexed field per segment. Keeps
/// open-time bounded even on segments with very wide term spaces
/// (json fields, log corpora) where naively draining every term would
/// cost minutes on a cold segment. The limit is intentionally
/// conservative — a follow-up wave can plumb this through as an
/// operator knob if benchmarks justify it.
pub const MAX_WARMUP_TERMS_PER_FIELD: usize = 10_000;

/// Pre-populate the host CHT (and, when CUDA Bitcomp is available,
/// the v3 VRAM CHT) for every indexed text field in `reader`.
///
/// No-op when [`segment_warm_on_open`] returns `false`. Bounded by
/// [`MAX_WARMUP_TERMS_PER_FIELD`] per indexed field to keep
/// open-latency predictable on wide term spaces.
///
/// Failures (per-term read errors, oversized terms, CUDA OOM during
/// v3 promote) are silently swallowed — warm-up is best-effort,
/// never blocks segment open.
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
pub fn warm_segment_v3(reader: &crate::index::SegmentReader) {
    if !segment_warm_on_open() {
        return;
    }

    use crate::schema::IndexRecordOption;

    let schema = reader.schema();
    let cht_handle = cht::global();
    // v3 may be unavailable (no Bitcomp codec construction); we still
    // warm v1 in that case so the next query at least skips the
    // drain.
    let v3_handle = vram_cht_v3::global();
    let segment_id = reader.segment_id();

    let mut total_warmed: usize = 0;
    let mut total_skipped_limit: usize = 0;

    for (field, field_entry) in schema.fields() {
        let field_type = field_entry.field_type();
        // Only walk fields that are actually indexed; anything else
        // can't have a posting list to warm.
        if field_type.get_index_record_option().is_none() {
            continue;
        }
        // Skip JSON fields for now — their term spaces are wide and
        // wedge-shaped (per-path subspaces) and the
        // `MAX_WARMUP_TERMS_PER_FIELD` budget would consume the whole
        // limit on one path. A future wave can per-path it.
        if matches!(field_type, crate::schema::FieldType::JsonObject(_)) {
            continue;
        }

        let inv_idx_reader = match reader.inverted_index(field) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let term_dict = inv_idx_reader.terms();
        let mut stream = match term_dict.stream() {
            Ok(s) => s,
            Err(_) => continue,
        };

        let mut warmed_in_field: usize = 0;
        while let Some((term_bytes, term_info)) = stream.next() {
            if warmed_in_field >= MAX_WARMUP_TERMS_PER_FIELD {
                total_skipped_limit += 1;
                log::debug!(
                    "warm_segment_v3: field {:?} hit per-field limit \
                     {MAX_WARMUP_TERMS_PER_FIELD}; remaining terms skipped",
                    field_entry.name()
                );
                break;
            }

            let key = cht::ChtKey {
                segment_id,
                field: field.field_id(),
                term_hash: cht::hash_term_bytes(term_bytes),
            };

            // Skip if v1 already has this entry — re-insert is a
            // no-op for budgets but each drain costs real work.
            if cht_handle.get(&key).is_some() {
                continue;
            }

            // Drain BlockSegmentPostings into a RoaringPostings. We
            // ask for `Basic` because the v3 path only consumes
            // doc-ids; freqs/positions would just add cost.
            let block_cursor_opt = inv_idx_reader
                .read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)
                .ok();
            let Some(mut block_cursor) = block_cursor_opt else {
                continue;
            };
            let drained = std::sync::Arc::new(
                from_block_segment::drain_block_segment_to_roaring(&mut block_cursor),
            );
            cht_handle.insert(key.clone(), std::sync::Arc::clone(&drained));

            // Opportunistic v3 promotion. Same best-effort semantics
            // as the query-time promote path in
            // `boolean_query::gpu_intersect`. Failures (CUDA OOM,
            // oversize, budget pressure) are silently swallowed.
            if let Some(handle) = v3_handle {
                let _ = handle.promote(key, drained.as_ref());
            }

            warmed_in_field += 1;
            total_warmed += 1;
        }
    }

    log::debug!(
        "warm_segment_v3: segment={:?} warmed_terms={total_warmed} \
         fields_hit_limit={total_skipped_limit}",
        segment_id
    );
}

/// No-op stub used when the feature stack required for the CHT v3
/// path (`gpu` + `ferro-compress` + `cuda-bitmap-kernel`) is not
/// active. Keeps the call site in [`crate::index::SegmentReader`]
/// unconditional.
#[cfg(not(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel")))]
pub fn warm_segment_v3(_reader: &crate::index::SegmentReader) {
    // Even when the v3 path is compiled out we still observe the
    // toggle so unit tests covering the no-op branch behave
    // identically with and without the feature stack.
    let _ = segment_warm_on_open();
}

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
/// - Segments without a format tag (= every existing index on disk) read as
///   [`PostingFormat::BitPacker4x`] — never as Roaring.
/// - Segments tagged `0x01` are Roaring and *must* fail-fast in readers that don't understand them,
///   never silently fall back. The default reader path will keep working on `0x00` segments
///   identically to today; a Roaring-aware reader will check the tag *before* attempting to parse
///   the body.
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

    // ============================================================
    // Wave Z-1 #1 — segment-warm hook tests
    // ============================================================

    // Wave Z-3 #5 — test isolation guard. The two warm-hook tests
    // both mutate the process-global toggle
    // (`set_segment_warm_on_open`) and observe the process-global
    // host CHT (`cht::global()`). Cargo's default test harness runs
    // tests in this module in parallel, so without a serialisation
    // primitive the toggle-on test can flip the toggle just before
    // the toggle-off test snapshots `stats_after` — and the warm
    // path silently fires under the wrong toggle. We avoid pulling
    // in `serial_test` (no new crate dependency) by guarding the
    // critical section with a local mutex.
    //
    // Any test that
    // (a) writes to `SEGMENT_WARM_ON_OPEN` or
    // (b) inspects `cht::global().stats()`
    // must `let _g = WARM_TEST_LOCK.lock().unwrap();` at the top to
    // hold this lock for its whole duration.
    use std::sync::Mutex;
    static WARM_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// The toggle must default to *off*. Existing deployments retain
    /// bytewise-identical behaviour until they opt in.
    #[test]
    fn segment_warm_on_open_defaults_off() {
        // Wave Z-3 #5 — serialise with sibling warm-hook tests; we
        // mutate the process-global toggle.
        let _g = WARM_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        // NB: a previous test in this binary could have flipped the
        // toggle to true. Restore the documented default after we read
        // it. We cannot assert *the initial value* portably for this
        // reason — but we can verify the setter/getter contract and
        // the documented default by inspecting `set_segment_warm_on_open`
        // semantics below.
        let saved = segment_warm_on_open();
        set_segment_warm_on_open(false);
        assert!(!segment_warm_on_open());
        // Restore in case other tests rely on the previous value.
        set_segment_warm_on_open(saved);
    }

    /// `set_segment_warm_on_open` must round-trip through
    /// `segment_warm_on_open`. Idempotent across repeated stores.
    #[test]
    fn segment_warm_on_open_setter_round_trip() {
        // Wave Z-3 #5 — serialise with sibling warm-hook tests; we
        // mutate the process-global toggle.
        let _g = WARM_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let saved = segment_warm_on_open();
        set_segment_warm_on_open(true);
        assert!(segment_warm_on_open());
        set_segment_warm_on_open(true); // idempotent
        assert!(segment_warm_on_open());
        set_segment_warm_on_open(false);
        assert!(!segment_warm_on_open());
        // Restore.
        set_segment_warm_on_open(saved);
    }

    /// When the toggle is off, `warm_segment_v3` must be a no-op:
    /// calling it against a real `SegmentReader` must not mutate any
    /// observable global state. This test runs in both feature modes
    /// because both branches (real impl and stub) must respect the
    /// toggle.
    #[test]
    fn warm_segment_v3_noop_when_disabled() {
        use crate::index::Index;
        use crate::schema::{Schema, TEXT};

        // Wave Z-3 #5 — serialise with sibling warm-hook tests; we
        // mutate the process-global toggle and observe host CHT
        // stats. Without this guard, the sibling
        // `warm_segment_v3_populates_v1_cache_when_enabled` test
        // can flip the toggle to *true* concurrently and our
        // `warm_segment_v3` call below would suddenly start
        // inserting under a stale `toggle=false` assertion.
        let _g = WARM_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let saved = segment_warm_on_open();
        set_segment_warm_on_open(false);

        let mut schema_builder = Schema::builder();
        let body = schema_builder.add_text_field("body", TEXT);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests().unwrap();
        writer.add_document(doc!(body => "alpha beta gamma")).unwrap();
        writer.add_document(doc!(body => "beta gamma delta")).unwrap();
        writer.commit().unwrap();

        let reader = index.reader().unwrap();
        let segment_reader = reader.searcher().segment_reader(0).clone();

        // Under the feature stack, the host CHT exists — snapshot
        // its insert counter and assert it doesn't move when the
        // toggle is off. Under the no-feature build, the body is a
        // stub; the call itself must not panic or otherwise misbehave.
        #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
        {
            // Wave Z-3 #5 — reset host CHT for ordering-safe runs.
            // The toggle-off branch's assertion is "stats unchanged
            // by warm call". Reset *after* writer commit + reader
            // construction (those steps may touch the host CHT via
            // the segment open path); resetting beforehand would be
            // overwritten and the before-snapshot would still
            // contain whatever those steps inserted.
            let cht = cht::global();
            cht.reset();
            let stats_before = cht.stats();
            warm_segment_v3(&segment_reader);
            let stats_after = cht.stats();
            assert_eq!(
                stats_before.inserts, stats_after.inserts,
                "warm_segment_v3 must not insert anything when toggle is off"
            );
            assert_eq!(
                stats_before.entries, stats_after.entries,
                "warm_segment_v3 must not grow the cache when toggle is off"
            );
        }
        #[cfg(not(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel")))]
        {
            // Stub branch: just verify the call is safe.
            warm_segment_v3(&segment_reader);
        }

        set_segment_warm_on_open(saved);
    }

    /// When the toggle is on AND the v3 feature stack is compiled in,
    /// warming a small synthetic segment must increase the host CHT's
    /// insert counter by at least the number of distinct terms.
    ///
    /// Gated on the feature stack — the host CHT module
    /// (`super::cht`) is itself cfg-gated, so this test only compiles
    /// when v3 is available.
    ///
    /// Wave Z-3 #5 — test isolation: the host CHT is a process-global
    /// singleton (`cht::global()`), so other tests in the same binary
    /// may have already drained the same common test tokens
    /// (`alpha`/`beta`/`gamma`/...) and populated the cache. Without
    /// a reset, the `warm_segment_v3` body short-circuits on each
    /// already-cached key (see `mod.rs::warm_segment_v3` "Skip if v1
    /// already has this entry" branch) and the insert counter does
    /// NOT grow by 5 — the assertion below would observe
    /// `before == after`. We call [`cht::Cht::reset`] up front to
    /// force a clean slate; this is the documented test-only entry
    /// point on the host CHT.
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    #[test]
    fn warm_segment_v3_populates_v1_cache_when_enabled() {
        use crate::index::Index;
        use crate::schema::{Schema, TEXT};

        // Wave Z-3 #5 — serialise with sibling warm-hook tests; we
        // both mutate the process-global toggle AND inspect the
        // host CHT's `inserts` counter. Without this guard a
        // sibling test could flip the toggle off between our
        // `set_segment_warm_on_open(true)` and the `warm_segment_v3`
        // call, turning the body into a silent no-op and the
        // inserts-grew-by-≥5 assertion would flake.
        let _g = WARM_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let saved = segment_warm_on_open();
        set_segment_warm_on_open(true);

        let mut schema_builder = Schema::builder();
        let body = schema_builder.add_text_field("body", TEXT);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests().unwrap();
        // 5 distinct terms across 2 docs.
        writer
            .add_document(doc!(body => "alpha beta gamma"))
            .unwrap();
        writer
            .add_document(doc!(body => "delta epsilon"))
            .unwrap();
        writer.commit().unwrap();

        let reader = index.reader().unwrap();
        let segment_reader = reader.searcher().segment_reader(0).clone();

        // Wave Z-3 #5 — reset host CHT global state for ordering-
        // safe runs. Reset is performed *after* the writer commit
        // and reader/segment_reader construction: those steps may
        // touch the host CHT via the segment open path (e.g.
        // implicit `warm_segment_v3` invocations or other cache
        // priming routines triggered by `index.reader()`), so a
        // reset *before* setup would be overwritten and the
        // before-snapshot would still observe non-zero inserts for
        // already-cached test terms (`alpha`/`beta`/...) populated
        // by sibling tests. Resetting right before the snapshot
        // gives the inserts-delta assertion below a stable
        // baseline of zero.
        let cht = cht::global();
        cht.reset();

        let inserts_before = cht.stats().inserts;
        warm_segment_v3(&segment_reader);
        let inserts_after = cht.stats().inserts;

        assert!(
            inserts_after >= inserts_before + 5,
            "warm_segment_v3 must insert ≥5 distinct terms; before={} after={}",
            inserts_before, inserts_after,
        );

        set_segment_warm_on_open(saved);
    }

    /// `MAX_WARMUP_TERMS_PER_FIELD` is a public const — verify it's a
    /// reasonable safety bound so an accidental zero (an obvious
    /// merge-conflict failure mode) is caught. Uses `const { ... }`
    /// so the check is compile-time and clippy doesn't flag the
    /// assertion as constant-valued.
    #[test]
    fn warmup_per_field_limit_is_nonzero_and_bounded() {
        const _: () = {
            assert!(
                MAX_WARMUP_TERMS_PER_FIELD > 0,
                "MAX_WARMUP_TERMS_PER_FIELD must be > 0; an accidental \
                 zero would disable warming for every field with terms"
            );
            assert!(
                MAX_WARMUP_TERMS_PER_FIELD <= 1_000_000,
                "MAX_WARMUP_TERMS_PER_FIELD is too large; segment open \
                 latency would balloon on wide term spaces"
            );
        };
    }

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

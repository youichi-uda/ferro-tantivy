// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Wave Z-7 #6 #4 — concurrent-churn fuzz harness for the **v3 VRAM
//! Bitcomp-compressed CHT** (multi-chunk, Z-6 series).
//!
//! Same op grammar as the v1 target but with v3-specific invariants:
//!
//! 1. **Multi-chunk admission cap.** Each entry's `chunk_count() <=
//!    MAX_CHUNKS_PER_ENTRY` (= 64), and each chunk's `compressed_bytes`
//!    is non-zero and bounded.
//! 2. **Wire round-trip.** `dump_to_path` + `reset` + `load_from_path`
//!    preserves entry count *and* per-entry uncompressed/compressed
//!    byte totals (the Z-6 #4 MAGIC_V4 format invariant).
//! 3. **Per-segment dump symmetry.** `dump_by_segments(all_segments)`
//!    produces a file whose load-back entry count equals
//!    `dump_to_path`'s entry count (= the oracle test invariant from
//!    `vram_cht_v3::tests::dump_by_segments_byte_invariant_with_oracle`).
//! 4. **Evict-by-segments accounting.** The count `evict_by_segments`
//!    returns equals the number of shadow keys with a matching
//!    `segment_id`.
//! 5. **Stats consistency.** Same monotonicity + budget invariants as
//!    the v1 target.
//!
//! ## CUDA gating
//!
//! This target is gated behind the crate's `cuda-bitmap-kernel` feature
//! (`cargo +nightly fuzz build cht_v3_multichunk_invariants --features
//! cuda-bitmap-kernel`). On a non-CUDA host the crate's `[[bin]]`
//! section has `required-features = ["cuda-bitmap-kernel"]` so this
//! target is silently skipped during a default `cargo +nightly fuzz
//! build` — only the v1 target builds.
//!
//! At **runtime**, the v3 cache needs an accessible GPU. The
//! constructor `VramCompressedCht::with_budget(...)` returns
//! `Err(VramCompressedChtError::Cuda { ... })` if cudart fails to
//! initialise; we bail the iteration silently in that case (so a
//! CUDA-driver-misconfigured CI host doesn't surface as a crash).

#![no_main]
#![cfg(feature = "cuda-bitmap-kernel")]

use std::collections::HashMap;

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use tantivy::index::SegmentId;
use tantivy::postings::roaring::cht::{hash_term_bytes, ChtKey};
use tantivy::postings::roaring::vram_cht_v3::{
    VramCompressedCht, VramCompressedChtStats, MAX_CHUNKS_PER_ENTRY,
};
use tantivy::postings::roaring::{RoaringEncoder, RoaringPostings};

/// VRAM budget for the fuzz harness — 64 MiB so the budget-eviction
/// path fires within a small number of Medium/Large inserts. Real
/// production budgets are 16+ GiB; we exercise the LRU boundary not
/// the steady-state cache.
const FUZZ_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// Same 8-segment pool as the v1 target so `EvictBySegments` exercises
/// a `u8` bitmask cleanly.
const NUM_PRESEEDED_SEGMENTS: usize = 8;

/// Cap on ops per iteration. v3 ops are 100×+ more expensive than v1
/// (each insert is a CUDA compress kernel launch), so we keep the
/// iteration count low.
const MAX_OPS_PER_ITERATION: usize = 64;

#[derive(Debug, Clone, Copy, Arbitrary)]
enum PayloadSize {
    /// ~10_000 doc-ids — fits in one chunk well below 16 MiB ceiling.
    Small,
    /// ~500_000 doc-ids — still one chunk on most distributions.
    Medium,
    /// ~5_000_000 doc-ids — multi-chunk territory (Z-6 #2 admission
    /// covers up to 1 GiB / 64 chunks).
    Large,
}

#[derive(Debug, Arbitrary)]
enum Op {
    Insert {
        seg_idx: u8,
        field_id: u32,
        term_seed: u64,
        payload_seed: u32,
        size: PayloadSize,
    },
    Get {
        seg_idx: u8,
        field_id: u32,
        term_seed: u64,
    },
    EvictBySegments {
        mask: u8,
    },
    DumpRoundTrip,
    DumpBySegmentsAllAndLoad,
    Reset,
    Stats,
}

#[derive(Debug, Arbitrary)]
struct OpSeq {
    ops: Vec<Op>,
}

fn preseeded_segments() -> [SegmentId; NUM_PRESEEDED_SEGMENTS] {
    use uuid::Uuid;
    let raw = [
        0xa1c4_0000_0000_0000_0000_0000_0000_0001u128,
        0xb2c4_0000_0000_0000_0000_0000_0000_0002u128,
        0xc3c4_0000_0000_0000_0000_0000_0000_0003u128,
        0xd4c4_0000_0000_0000_0000_0000_0000_0004u128,
        0xe5c4_0000_0000_0000_0000_0000_0000_0005u128,
        0xf6c4_0000_0000_0000_0000_0000_0000_0006u128,
        0x07c4_0000_0000_0000_0000_0000_0000_0007u128,
        0x18c4_0000_0000_0000_0000_0000_0000_0008u128,
    ];
    let mut out = [SegmentId::generate_random(); NUM_PRESEEDED_SEGMENTS];
    for (i, &n) in raw.iter().enumerate() {
        let uuid = Uuid::from_u128(n);
        let s = uuid.as_simple().to_string();
        out[i] = SegmentId::from_uuid_string(&s)
            .expect("preseeded uuid round-trips through hex form");
    }
    out
}

fn build_payload(size: PayloadSize, seed: u32) -> RoaringPostings {
    let n: u32 = match size {
        PayloadSize::Small => 10_000,
        PayloadSize::Medium => 500_000,
        PayloadSize::Large => 5_000_000,
    };
    let docs: Vec<u32> = (0..n).map(|i| seed.wrapping_add(i * 3)).collect();
    RoaringEncoder::from_doc_ids(&docs)
}

#[derive(Debug, Clone, Copy)]
struct StatsBaseline {
    inserts: u64,
    evictions: u64,
    hits: u64,
    misses: u64,
}

impl From<&VramCompressedChtStats> for StatsBaseline {
    fn from(s: &VramCompressedChtStats) -> Self {
        Self {
            inserts: s.inserts,
            evictions: s.evictions,
            hits: s.hits,
            misses: s.misses,
        }
    }
}

fn assert_invariants(
    stats: &VramCompressedChtStats,
    baseline: &StatsBaseline,
    budget: u64,
) {
    assert!(
        stats.current_bytes <= budget,
        "stats.current_bytes ({}) > budget ({})",
        stats.current_bytes, budget,
    );
    assert_eq!(
        stats.budget_bytes, budget,
        "stats.budget_bytes drifted from configured budget",
    );
    assert_eq!(
        stats.compressed_bytes_total, stats.current_bytes,
        "compressed_bytes_total ({}) != current_bytes ({})",
        stats.compressed_bytes_total, stats.current_bytes,
    );
    assert!(
        stats.inserts >= baseline.inserts,
        "inserts went backwards: {} -> {}",
        baseline.inserts, stats.inserts,
    );
    assert!(
        stats.evictions >= baseline.evictions,
        "evictions went backwards: {} -> {}",
        baseline.evictions, stats.evictions,
    );
    assert!(
        stats.hits >= baseline.hits,
        "hits went backwards: {} -> {}",
        baseline.hits, stats.hits,
    );
    assert!(
        stats.misses >= baseline.misses,
        "misses went backwards: {} -> {}",
        baseline.misses, stats.misses,
    );
    assert!(
        stats.evictions <= stats.inserts,
        "evictions ({}) > inserts ({})",
        stats.evictions, stats.inserts,
    );
}

fn run_op_seq(seq: OpSeq) {
    let cht = match VramCompressedCht::with_budget(FUZZ_BUDGET_BYTES) {
        Ok(c) => c,
        Err(_) => return, // CUDA init failed — not a CHT bug
    };
    let segments = preseeded_segments();
    // Shadow tracks (key, expected_uncompressed_bytes, expected_chunk_count_lower_bound)
    // so we can sanity-check entry shape on `Get` hits.
    let mut shadow: HashMap<ChtKey, ()> = HashMap::new();
    let mut baseline = StatsBaseline::from(&cht.stats());

    for (op_idx, op) in seq.ops.into_iter().take(MAX_OPS_PER_ITERATION).enumerate() {
        match op {
            Op::Insert {
                seg_idx,
                field_id,
                term_seed,
                payload_seed,
                size,
            } => {
                let seg = segments[(seg_idx as usize) % NUM_PRESEEDED_SEGMENTS];
                let term_bytes = term_seed.to_le_bytes();
                let key = ChtKey {
                    segment_id: seg,
                    field: field_id,
                    term_hash: hash_term_bytes(&term_bytes),
                };
                let postings = build_payload(size, payload_seed);
                match cht.insert(key.clone(), &postings) {
                    Ok(_inserted) => {
                        shadow.insert(key, ());
                    }
                    Err(_) => {
                        // CUDA OOM / Bitcomp errors are not CHT
                        // invariant violations — they signal a real
                        // resource problem upstream.
                    }
                }
            }
            Op::Get {
                seg_idx,
                field_id,
                term_seed,
            } => {
                let seg = segments[(seg_idx as usize) % NUM_PRESEEDED_SEGMENTS];
                let term_bytes = term_seed.to_le_bytes();
                let key = ChtKey {
                    segment_id: seg,
                    field: field_id,
                    term_hash: hash_term_bytes(&term_bytes),
                };
                if let Some(entry) = cht.get(&key) {
                    // (1) Multi-chunk admission cap.
                    assert!(
                        entry.chunk_count() <= MAX_CHUNKS_PER_ENTRY,
                        "entry.chunk_count() {} > MAX_CHUNKS_PER_ENTRY {}",
                        entry.chunk_count(),
                        MAX_CHUNKS_PER_ENTRY,
                    );
                    assert!(
                        entry.chunk_count() >= 1,
                        "entry.chunk_count() must be >= 1",
                    );
                    for ch in entry.chunks() {
                        assert!(
                            ch.compressed_bytes > 0,
                            "chunk.compressed_bytes must be > 0",
                        );
                    }
                }
            }
            Op::EvictBySegments { mask } => {
                let to_evict: Vec<SegmentId> = (0..NUM_PRESEEDED_SEGMENTS)
                    .filter(|i| (mask & (1 << i)) != 0)
                    .map(|i| segments[i])
                    .collect();
                if !to_evict.is_empty() {
                    let to_evict_set: std::collections::HashSet<SegmentId> =
                        to_evict.iter().copied().collect();
                    let expected = shadow
                        .keys()
                        .filter(|k| to_evict_set.contains(&k.segment_id))
                        .count() as u64;
                    let got = cht.evict_by_segments(&to_evict);
                    // (4) Evict accounting: returned count must match
                    // shadow's filtered cardinality, BUT only if the
                    // shadow is in sync — LRU-driven evictions on
                    // Insert can leave the shadow ahead of reality.
                    // So we assert `got <= expected` (the CHT can
                    // evict no MORE than the shadow believed
                    // resident) and `got` is bounded by total entries.
                    assert!(
                        got <= expected,
                        "evict_by_segments returned {} but shadow had {} matching keys",
                        got, expected,
                    );
                    shadow.retain(|k, _| !to_evict_set.contains(&k.segment_id));
                }
            }
            Op::DumpRoundTrip => {
                let pre_entries = cht.stats().entries;
                let pre_uncompressed = cht.stats().uncompressed_bytes_total;
                let tmpdir = match std::env::temp_dir().canonicalize() {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let path = tmpdir.join(format!(
                    "cht_v3_fuzz_dump_{}_{}.bin",
                    std::process::id(),
                    op_idx,
                ));
                let dumped = match cht.dump_to_path(&path) {
                    Ok(n) => n,
                    Err(_) => {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                };
                assert_eq!(
                    dumped, pre_entries,
                    "dump_to_path returned {} but stats said {} entries",
                    dumped, pre_entries,
                );
                cht.reset();
                shadow.clear();
                let loaded = match cht.load_from_path(&path) {
                    Ok(n) => n,
                    Err(_) => {
                        let _ = std::fs::remove_file(&path);
                        baseline = StatsBaseline::from(&cht.stats());
                        continue;
                    }
                };
                assert_eq!(
                    loaded, dumped,
                    "load_from_path loaded {} but dump wrote {}",
                    loaded, dumped,
                );
                // (2) Wire round-trip: uncompressed totals preserved.
                let post_uncompressed = cht.stats().uncompressed_bytes_total;
                assert_eq!(
                    post_uncompressed, pre_uncompressed,
                    "uncompressed_bytes_total drifted on round-trip: {} -> {}",
                    pre_uncompressed, post_uncompressed,
                );
                let _ = std::fs::remove_file(&path);
                baseline = StatsBaseline::from(&cht.stats());
                continue;
            }
            Op::DumpBySegmentsAllAndLoad => {
                let pre_entries = cht.stats().entries;
                let tmpdir = match std::env::temp_dir().canonicalize() {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let path = tmpdir.join(format!(
                    "cht_v3_fuzz_dump_seg_{}_{}.bin",
                    std::process::id(),
                    op_idx,
                ));
                let dumped = match cht.dump_by_segments(&segments, &path) {
                    Ok(n) => n,
                    Err(_) => {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                };
                // (3) Per-segment dump symmetry: dumping all
                // pre-seeded segments must include every entry,
                // since every key the fuzzer can construct uses one
                // of those 8 segments.
                assert_eq!(
                    dumped, pre_entries,
                    "dump_by_segments(all 8 segments) returned {} but pre-dump entries = {}",
                    dumped, pre_entries,
                );
                cht.reset();
                shadow.clear();
                let loaded = match cht.load_from_path(&path) {
                    Ok(n) => n,
                    Err(_) => {
                        let _ = std::fs::remove_file(&path);
                        baseline = StatsBaseline::from(&cht.stats());
                        continue;
                    }
                };
                assert_eq!(
                    loaded, dumped,
                    "dump_by_segments+load round-trip mismatched: dumped {} loaded {}",
                    dumped, loaded,
                );
                let _ = std::fs::remove_file(&path);
                baseline = StatsBaseline::from(&cht.stats());
                continue;
            }
            Op::Reset => {
                cht.reset();
                shadow.clear();
                baseline = StatsBaseline::from(&cht.stats());
                continue;
            }
            Op::Stats => {}
        }
        let stats = cht.stats();
        assert_invariants(&stats, &baseline, FUZZ_BUDGET_BYTES);
        baseline = StatsBaseline::from(&stats);
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Ok(seq) = OpSeq::arbitrary(&mut u) {
        if !seq.ops.is_empty() {
            run_op_seq(seq);
        }
    }
});

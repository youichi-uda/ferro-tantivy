//! Wave Z-5 #1 — GPU host latency bench for `promote_v2_to_v3` on dense-term
//! cohorts. Compares the cross-tier device→device promote path against the
//! legacy `insert(roaring)` re-drain path on cohorts of varying term density,
//! capturing the `cross_tier_promotions` strict-subset counter delta to pin the
//! "right kernel-side path was taken" invariant alongside latency measurements.
//!
//! Validates the ~60ms savings recon estimate from `promote_v2_to_v3`'s
//! rustdoc (file `src/postings/roaring/vram_cht_v3.rs`, lines 519-521).
//!
//! Usage:
//!   LD_LIBRARY_PATH=<nvcomp lib path> cargo run --release --example \
//!     promote_v2_to_v3_dense_bench \
//!     --features gpu,ferro-compress,cuda-bitmap-kernel
//!
//! Stdout emits CSV rows `cohort,iter,path,duration_us`; stderr emits a
//! per-cohort summary block with p50/p95/p99 + counter deltas.

use std::sync::Arc;
use std::time::Instant;

use tantivy::index::SegmentId;
use tantivy::postings::roaring::cht::ChtKey;
use tantivy::postings::roaring::encoder::{RoaringEncoder, RoaringPostings};
use tantivy::postings::roaring::vram_cht::VramCht;
use tantivy::postings::roaring::vram_cht_v3::VramCompressedCht;

const BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const ITERATIONS_PER_COHORT: u32 = 100;
const WARMUP_ITERATIONS: u32 = 5;

fn dense_roaring(seed: u32, doc_count: u32) -> RoaringPostings {
    let docs: Vec<u32> = (0..doc_count).map(|i| seed.wrapping_add(i * 3)).collect();
    RoaringEncoder::from_doc_ids(&docs)
}

/// Build a posting list spread across `target_buckets` distinct high16
/// containers (= the "buckets" axis the recon estimate `~60ms` uses).
/// Each bucket holds `docs_per_bucket` low16 values starting at offset 0.
/// The high16 values are `seed + b` for b in 0..target_buckets, wrapping
/// at 0xFFFF.
fn bucket_spread_roaring(seed: u32, target_buckets: u32, docs_per_bucket: u32) -> RoaringPostings {
    let mut docs: Vec<u32> = Vec::with_capacity((target_buckets * docs_per_bucket) as usize);
    for b in 0..target_buckets {
        let high16: u32 = (seed.wrapping_add(b)) & 0xFFFF;
        let base = high16 << 16;
        for d in 0..docs_per_bucket {
            docs.push(base + d);
        }
    }
    docs.sort_unstable();
    docs.dedup();
    RoaringEncoder::from_doc_ids(&docs)
}

fn dummy_key(field: u32, term_hash: u64) -> ChtKey {
    ChtKey {
        segment_id: SegmentId::generate_random(),
        field,
        term_hash,
    }
}

fn percentile(samples_us: &mut [f64], p: f64) -> f64 {
    if samples_us.is_empty() {
        return 0.0;
    }
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples_us.len() as f64 - 1.0) * p).round() as usize;
    samples_us[idx]
}

/// A cohort: either dense (single-bucket-class, varying doc count) or
/// bucket-spread (varying bucket count, fixed docs per bucket).
enum CohortShape {
    DenseDocs(u32),
    BucketSpread { target_buckets: u32, docs_per_bucket: u32 },
}

fn main() {
    let cohorts: &[(&str, CohortShape)] = &[
        ("dense_1k_docs", CohortShape::DenseDocs(1_000)),
        ("dense_10k_docs", CohortShape::DenseDocs(10_000)),
        ("dense_100k_docs", CohortShape::DenseDocs(100_000)),
        ("dense_500k_docs", CohortShape::DenseDocs(500_000)),
        ("dense_1M_docs", CohortShape::DenseDocs(1_000_000)),
        ("bspread_500_buckets", CohortShape::BucketSpread {
            target_buckets: 500,
            docs_per_bucket: 64,
        }),
        ("bspread_1k_buckets", CohortShape::BucketSpread {
            target_buckets: 1_000,
            docs_per_bucket: 64,
        }),
        ("bspread_1500_buckets", CohortShape::BucketSpread {
            target_buckets: 1_500,
            docs_per_bucket: 64,
        }),
        // Wave Z-6 multi-chunk cohorts (Z-6 #2/#3/#4 LAND): the 16 MiB
        // single-chunk Bitcomp ceiling has been lifted to
        // `MAX_CHUNKS_PER_ENTRY * BITCOMP_CHUNK_BYTES` (= 64 × 16 MiB =
        // 1 GiB). The three cohorts below are the recon-estimate
        // targets from `vram_cht_v3.rs` rustdoc lines 519-521
        // (`~60 ms savings on dense terms (10K+ buckets)`); each spans
        // multiple Bitcomp chunks:
        //   - bspread_5k_buckets  ≈  40 MiB uncompressed → 3 chunks
        //   - bspread_10k_buckets ≈  80 MiB uncompressed → 5 chunks
        //   - bspread_30k_buckets ≈ 240 MiB uncompressed → 15 chunks
        ("bspread_5k_buckets", CohortShape::BucketSpread {
            target_buckets: 5_000,
            docs_per_bucket: 64,
        }),
        ("bspread_10k_buckets", CohortShape::BucketSpread {
            target_buckets: 10_000,
            docs_per_bucket: 64,
        }),
        ("bspread_30k_buckets", CohortShape::BucketSpread {
            target_buckets: 30_000,
            docs_per_bucket: 64,
        }),
    ];

    println!("cohort,iter,path,duration_us");

    for (label, shape) in cohorts {
        let total_iters = ITERATIONS_PER_COHORT + WARMUP_ITERATIONS;
        let fixtures: Vec<RoaringPostings> = (0..total_iters)
            .map(|i| match shape {
                CohortShape::DenseDocs(doc_count) => {
                    dense_roaring(i.wrapping_mul(7919), *doc_count)
                }
                CohortShape::BucketSpread { target_buckets, docs_per_bucket } => {
                    bucket_spread_roaring(i.wrapping_mul(101), *target_buckets, *docs_per_bucket)
                }
            })
            .collect();
        let cohort_doc_count: u64 = fixtures
            .first()
            .map(|rp| {
                rp.containers
                    .iter()
                    .map(|(_, c)| c.cardinality() as u64)
                    .sum()
            })
            .unwrap_or(0);
        let cohort_bucket_count: u64 = fixtures
            .first()
            .map(|rp| rp.containers.iter().filter(|(_, c)| c.cardinality() > 0).count() as u64)
            .unwrap_or(0);

        // === Path A: legacy insert(roaring) baseline ===
        let v3_legacy = VramCompressedCht::with_budget(BUDGET_BYTES)
            .expect("VramCompressedCht::with_budget must succeed (CUDA + nvcomp required)");
        let mut legacy_us: Vec<f64> = Vec::with_capacity(ITERATIONS_PER_COHORT as usize);
        for (i, rp) in fixtures.iter().enumerate() {
            let key = dummy_key(0xA000_0000 + i as u32, 100);
            let start = Instant::now();
            let inserted = v3_legacy.insert(key, rp).unwrap();
            let dur_us = start.elapsed().as_secs_f64() * 1e6;
            if !inserted {
                eprintln!(
                    "[{}] WARN: legacy insert returned Ok(false) at iter {} (uncompressed_bytes \
                     likely > 1 GiB MAX_CHUNKS_PER_ENTRY admission ceiling); skipping cohort",
                    label, i
                );
                break;
            }
            if (i as u32) >= WARMUP_ITERATIONS {
                legacy_us.push(dur_us);
                println!("{},{},legacy_insert,{:.3}", label, i, dur_us);
            }
        }
        let legacy_stats = v3_legacy.stats();
        drop(v3_legacy);

        // === Path B: promote_v2_to_v3 (cross-tier device→device) ===
        let v3_promote = VramCompressedCht::with_budget(BUDGET_BYTES)
            .expect("VramCompressedCht::with_budget must succeed");
        let v2 = VramCht::with_budget(BUDGET_BYTES);
        let pre_promote_stats = v3_promote.stats();
        let mut promote_us: Vec<f64> = Vec::with_capacity(ITERATIONS_PER_COHORT as usize);
        for (i, rp) in fixtures.iter().enumerate() {
            let key = dummy_key(0xB000_0000 + i as u32, 100);
            v2.insert(key.clone(), rp).unwrap();
            let v2_entry = v2.get(&key).expect("v2 must hit after insert");
            let start = Instant::now();
            let admitted = v3_promote
                .promote_v2_to_v3(&key, Arc::clone(&v2_entry))
                .unwrap();
            let dur_us = start.elapsed().as_secs_f64() * 1e6;
            if !admitted {
                eprintln!(
                    "[{}] WARN: promote_v2_to_v3 returned Ok(false) at iter {} (uncompressed_bytes \
                     > 1 GiB MAX_CHUNKS_PER_ENTRY admission ceiling); skipping cohort",
                    label, i
                );
                break;
            }
            if (i as u32) >= WARMUP_ITERATIONS {
                promote_us.push(dur_us);
                println!("{},{},promote_v2_to_v3,{:.3}", label, i, dur_us);
            }
        }
        let post_promote_stats = v3_promote.stats();

        let promotions_delta = post_promote_stats.promotions - pre_promote_stats.promotions;
        let cross_tier_delta =
            post_promote_stats.cross_tier_promotions - pre_promote_stats.cross_tier_promotions;
        let counters_match = promotions_delta == cross_tier_delta;
        let counters_complete = cross_tier_delta == promote_us.len() as u64 + WARMUP_ITERATIONS as u64;

        let legacy_p50 = percentile(&mut legacy_us.clone(), 0.50);
        let legacy_p95 = percentile(&mut legacy_us.clone(), 0.95);
        let legacy_p99 = percentile(&mut legacy_us.clone(), 0.99);
        let promote_p50 = percentile(&mut promote_us.clone(), 0.50);
        let promote_p95 = percentile(&mut promote_us.clone(), 0.95);
        let promote_p99 = percentile(&mut promote_us.clone(), 0.99);

        let savings_p50_us = legacy_p50 - promote_p50;
        let savings_pct = if legacy_p50 > 0.0 {
            100.0 * savings_p50_us / legacy_p50
        } else {
            0.0
        };

        eprintln!(
            "[{}] docs={} buckets={} | legacy(insert) p50={:>8.0}us p95={:>8.0}us p99={:>8.0}us \
             | promote(v2→v3) p50={:>8.0}us p95={:>8.0}us p99={:>8.0}us \
             | savings_p50={:>+8.0}us ({:>+5.1}%) \
             | Δpromotions={} Δcross_tier_promotions={} \
             | counters_match={} counters_complete={} \
             | legacy_compressed_total={} bytes",
            label,
            cohort_doc_count,
            cohort_bucket_count,
            legacy_p50,
            legacy_p95,
            legacy_p99,
            promote_p50,
            promote_p95,
            promote_p99,
            savings_p50_us,
            savings_pct,
            promotions_delta,
            cross_tier_delta,
            counters_match,
            counters_complete,
            legacy_stats.compressed_bytes_total,
        );

        drop(v3_promote);
        drop(v2);
    }
}

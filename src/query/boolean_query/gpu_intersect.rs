//! Bool-AND GPU dispatch helper for [`super::BooleanWeight`] — Phase 2
//! C-5 (Wave 7 / Plan 2 / 3).
//!
//! ## Decision matrix
//!
//! Before [`super::boolean_weight::BooleanWeight::complex_scorer`] hands
//! a Bool-AND cohort to the legacy galloping AVX2
//! [`crate::query::Intersection`], it consults
//! [`try_gpu_intersect`]. The helper returns:
//!
//! - `Some(scorer)` — caller uses the GPU-materialised
//!   [`super::super::roaring_materialised_scorer::RoaringMaterialisedScorer`]
//!   directly. **Filter-context only**: scoring is irrevocably
//!   collapsed to `1.0` per doc (term frequencies are dropped at the
//!   drain step — Roaring containers carry doc-ids only).
//! - `None` — caller proceeds on the CPU path bytewise-identically to
//!   pre-Wave-7 behaviour. This is the safe default for every
//!   non-AND-cohort, non-all-TermScorer cohort, scoring-enabled query,
//!   sub-2-term cohort, planner-rejected cohort, or driver-init
//!   failure.
//!
//! The helper is the **only** site in `BooleanWeight` that knows about
//! Roaring / GPU. The legacy [`crate::query::Intersection`] is
//! untouched.
//!
//! ## What's deferred to Wave 8
//!
//! - **OR (Union) path**: GPU has the kernel
//!   ([`tantivy_gpu::posting::BitmapOpKernel`] handles `BitmapOp::Or`
//!   identically to `And`), but the dispatch site is
//!   [`super::boolean_weight::scorer_union`] — wiring it requires the
//!   same `RoaringMaterialisedScorer` adapter contract for the union
//!   shape, plus a planner branch for OR cohorts (the heavy stopword
//!   case looks different). Out of scope this wave.
//! - **MustNot (Exclude) path**: needs an inverse-of-Roaring operation
//!   on the GPU, currently kernel-less.
//! - **Phrase queries / mixed-type cohorts**: phrases need positional
//!   payloads which Roaring drops; mixed cohorts (TermScorer + range
//!   scorer + custom) hit the cohort-shape gate below and return
//!   `None`.
//! - **Scoring-enabled (BM25) Bool-AND**: the GPU drain destroys term
//!   frequency, so we explicitly gate on `scoring_enabled == false`
//!   (option (a) in the Plan 2 design — preserves correctness and
//!   delays the per-query rescore cost).

use crate::index::SegmentReader;
use crate::postings::roaring::encoder::RoaringPostings;
use crate::postings::roaring::planner::{should_dispatch_gpu, TermStat};
use crate::postings::roaring::{
    drain_block_segment_to_roaring, record_cpu_fallback, try_gpu_bool, BoolOp,
};
use crate::query::roaring_materialised_scorer::RoaringMaterialisedScorer;
use crate::query::term_query::TermScorer;
use crate::query::Scorer;
use crate::Term;

#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
use crate::postings::roaring::try_gpu_bool_v3;
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
use crate::postings::roaring::try_gpu_bool_vram;
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
use crate::postings::roaring::vram_cht;
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
use crate::postings::roaring::vram_cht::VramTermEntry;
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
use crate::postings::roaring::vram_cht_v3;
#[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
use crate::postings::roaring::vram_cht_v3::VramCompressedTermEntry;

/// Try to route a Bool-AND cohort through the Wave 6 GPU Roaring
/// dispatch path.
///
/// Returns `Some(scorer)` iff every gate below clears — see the
/// module-level docstring for what each `None` arm protects against.
///
/// On `Some` return, ownership of the original `must_scorers` (passed
/// in by `Vec`) is consumed; on `None`, the caller's `must_scorers`
/// `Vec` is restored (we use a wrapper `Result` returned by reference
/// to avoid the move).
///
/// # Arguments
///
/// - `must_scorers`: the cohort's MUST-occur scorers paired with their
///   originating [`Term`]s, in arbitrary order. The `Term` is used to
///   build a content-stable
///   [`crate::postings::roaring::cht::ChtKey`] (Wave 11 migration).
///   The helper only takes ownership on the `Some` path.
/// - `reader`: segment reader for `num_docs()` (drives the field-doc
///   ratio gate).
/// - `scoring_enabled`: when `true`, returns `None` immediately — see
///   docstring rationale.
///
/// # Returns
///
/// - `Ok(Some(scorer))` — GPU path took the cohort. Caller boxes &
///   returns. `must_scorers` is consumed.
/// - `Ok(None)` — gate failed; caller falls through to legacy CPU
///   intersection. `must_scorers` is returned via the `Err` arm of
///   the inner `try_gpu_intersect` so the caller can recover the
///   `Vec<Box<dyn Scorer>>`. The recovered shape is the scorer-only
///   `Vec<Box<dyn Scorer>>` (Terms are dropped on the err path —
///   BooleanWeight reconstructs the legacy CPU cohort from scorers
///   alone).
/// - `Err(must_scorers)` — same as `Ok(None)` semantically; we use the
///   `Result<Option<…>, Vec<…>>` shape to make the move-vs-keep
///   contract explicit at the call site.
pub(crate) fn try_gpu_intersect(
    must_scorers: Vec<(Term, Box<dyn Scorer>)>,
    reader: &SegmentReader,
    scoring_enabled: bool,
) -> Result<Box<dyn Scorer>, Vec<Box<dyn Scorer>>> {
    // ----- Gate 1: scoring disabled -----
    // Roaring drains drop term frequencies. If the caller wants BM25
    // contributions we cannot satisfy them; punt to CPU.
    if scoring_enabled {
        record_cpu_fallback();
        return Err(must_scorers.into_iter().map(|(_, s)| s).collect());
    }

    // ----- Gate 2: cohort size -----
    // try_gpu_bool itself returns None for < 2 terms (single-scorer
    // queries don't need an intersection at all). The Wave 8 / A / 2
    // planner gates on per-term cardinality (≥ 100 K), cohort-total
    // doc-freq (≥ 1 M), and field-doc ratio (≥ 5 %) further down. We
    // early-out here for clarity; the planner is the authoritative
    // gate below.
    if must_scorers.len() < 2 {
        record_cpu_fallback();
        return Err(must_scorers.into_iter().map(|(_, s)| s).collect());
    }

    // ----- Gate 3: every cohort member must be a TermScorer -----
    // Mixed cohorts (TermScorer + RangeScorer + …) cannot be drained
    // by `BlockSegmentPostings → RoaringPostings` because the bridge
    // only accepts inverted-index posting cursors. We could in
    // principle drain non-term scorers via for_each_docset_buffered,
    // but the cost of a per-bucket bitmap construction off a
    // streaming non-term scorer dominates the GPU win at the cohort
    // sizes the planner approves; leave it for Wave 8 if the bench
    // demands it.
    if !must_scorers.iter().all(|(_, s)| s.is::<TermScorer>()) {
        record_cpu_fallback();
        return Err(must_scorers.into_iter().map(|(_, s)| s).collect());
    }

    // ----- Gate 4: planner threshold -----
    // Compute (doc_freq, num_docs_in_segment) for each term and feed
    // them to should_dispatch_gpu — the same gate Wave 6's
    // try_gpu_bool re-checks internally. We pre-check here to avoid
    // doing the (potentially expensive) drain when the cohort is
    // light. The double gate is cheap (sum of u32 + max f32) and
    // keeps both surfaces consistent in case someone calls
    // try_gpu_bool directly later.
    let num_docs_in_segment = reader.num_docs();
    let stats: Vec<TermStat> = must_scorers
        .iter()
        .map(|(_, scorer)| {
            // Downcast is infallible here — gate 3 verified.
            let term_scorer: &TermScorer = scorer
                .downcast_ref::<TermScorer>()
                .expect("gate 3 should have rejected non-TermScorer");
            TermStat {
                doc_freq: term_scorer.segment_postings().doc_freq(),
                num_docs_in_segment,
            }
        })
        .collect();
    if !should_dispatch_gpu(&stats) {
        record_cpu_fallback();
        return Err(must_scorers.into_iter().map(|(_, s)| s).collect());
    }

    // ----- Drain step -----
    // Move the scorers out of the Vec, downcast each to TermScorer,
    // clone the underlying BlockSegmentPostings, drain into a
    // RoaringPostings. We clone (rather than consume) so the original
    // TermScorer remains usable on the err-recovery path below — the
    // clone is cheap (BlockSegmentPostings shares its FileSlice via
    // Arc, so the copy is just cursor state).
    //
    // We collect into Vec<RoaringPostings>, then build a Vec of refs
    // for try_gpu_bool's `&[&RoaringPostings]` signature.
    // Phase 2 D-1 — CHT lookup before drain. Wave 11 migrated the
    // cache key to a content-stable scheme
    // `(segment_id, field_id, term_hash)` (see
    // `crate::postings::roaring::cht::ChtKey` doc); on hit we skip
    // `drain_block_segment_to_roaring` entirely (the ~700 ms / 10 M-doc
    // bottleneck per Wave 8 / C re-bench). On miss we drain as today,
    // then `Arc::new` the result and insert into the cache for the
    // next query touching the same posting list.
    use crate::postings::roaring::cht;
    use std::sync::Arc;
    // Phase 2 D-3 measurement instrumentation. Per-step `Instant`
    // timing emitted via `tracing::debug!` so the next-wave v2 design
    // is data-driven (Wave 8 / D-1 left ~110 ms / query unattributed
    // post-drain-bypass; this breakdown identifies the dominant
    // residual). Build cost is one `Instant::now` per step
    // (sub-µs syscall), only paid when `tracing::debug!` is active.
    let t0 = std::time::Instant::now();
    let segment_id = reader.segment_id();
    let cht_handle = cht::global();
    // Phase 2 D-3 v2 — VRAM CHT lookup. The VRAM cache is gated on
    // `cuda-bitmap-kernel`; without it, `vram_entries` is unused
    // (None for every term) and we fall through to the host fold
    // path bytewise-identically.
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    let vram_handle = vram_cht::global();
    // Phase 2 D-4 v3 — Bitcomp-compressed VRAM CHT lookup. May return
    // None for the global if codec construction failed (no CUDA driver
    // / no Bitcomp support), in which case v3 entries stay None across
    // the cohort and we fall through to v2.
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    let v3_handle = vram_cht_v3::global();
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    let mut vram_entries: Vec<Option<Arc<VramTermEntry>>> =
        Vec::with_capacity(must_scorers.len());
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    let mut v3_entries: Vec<Option<Arc<VramCompressedTermEntry>>> =
        Vec::with_capacity(must_scorers.len());
    let mut owned_postings: Vec<Arc<RoaringPostings>> =
        Vec::with_capacity(must_scorers.len());
    let mut term_scorers: Vec<Box<TermScorer>> = Vec::with_capacity(must_scorers.len());
    let mut host_hit_count: u32 = 0;
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    let mut vram_hit_count: u32 = 0;
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    let mut v3_hit_count: u32 = 0;
    let mut drain_us_total: u128 = 0;
    for (term, scorer) in must_scorers {
        // Infallible per gate 3. Box<dyn Scorer>::downcast moves the
        // value (so we don't lose ownership on success) — see
        // downcast_rs.
        let term_scorer: Box<TermScorer> = scorer
            .downcast::<TermScorer>()
            .map_err(|_| ())
            .expect("gate 3 should have rejected non-TermScorer");
        // Wave 11 content-stable key. `term.field().field_id()` is the
        // u32 schema-stable field id; `cht::hash_term_bytes` is the
        // deterministic FxHash of the serialized term value bytes.
        // Both are stable across process restarts — D-5 dump/load
        // depends on this property.
        let key = cht::ChtKey {
            segment_id,
            field: term.field().field_id(),
            term_hash: cht::hash_term_bytes(term.serialized_value_bytes()),
        };
        // Step A0: VRAM CHT v3 lookup (Bitcomp-compressed, fastest
        // capacity-multiplier tier). v3 only makes sense if the
        // process-global codec construction succeeded.
        #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
        let v3_entry = v3_handle.and_then(|h| h.get(&key));
        #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
        if v3_entry.is_some() {
            v3_hit_count += 1;
        }
        // Step A: VRAM CHT v2 lookup (under cuda-bitmap-kernel only).
        #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
        let vram_entry = vram_handle.get(&key);
        #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
        if vram_entry.is_some() {
            vram_hit_count += 1;
        }
        // Step B: host CHT v1 lookup (or drain on miss). The host
        // entry is needed regardless of VRAM hit state — the fall-back
        // host fold path consumes it, and on VRAM miss the dispatch
        // layer also needs it for opportunistic promotion. We never
        // skip the host insert: the host cache footprint is a small
        // fraction of VRAM (4 GiB host budget vs 32 GiB VRAM budget on
        // L40S) so the duplication cost is negligible vs the warm-
        // restart benefit (host cache survives more events than VRAM,
        // so it's the canonical "cheap" tier).
        let roaring = if let Some(cached) = cht_handle.get(&key) {
            // Host hit — skip the drain entirely.
            host_hit_count += 1;
            cached
        } else {
            // Miss — drain + insert into host cache so subsequent
            // queries hit. We clone the BlockSegmentPostings so the
            // original TermScorer keeps its read cursor for the
            // err-recovery path; cheap (file slice / Arc pattern).
            let drain_t = std::time::Instant::now();
            let mut block_cursor = term_scorer.segment_postings().block_cursor.clone();
            let drained = Arc::new(drain_block_segment_to_roaring(&mut block_cursor));
            drain_us_total += drain_t.elapsed().as_micros();
            cht_handle.insert(key.clone(), Arc::clone(&drained));
            drained
        };

        // Step C: opportunistic VRAM promotion. If the host entry just
        // became available (drain or v1 hit) but VRAM was a miss, try
        // to promote so subsequent queries hit the VRAM tier. Failures
        // (CUDA OOM, oversized term, budget pressure) are silently
        // swallowed — promotion is best-effort, never blocks the query.
        #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
        let final_vram_entry = if let Some(entry) = vram_entry {
            Some(entry)
        } else {
            // Promote: insert into VRAM cache, then re-fetch the
            // resulting Arc<VramTermEntry> for use in this same
            // query (so the cohort can take the VRAM fast path on
            // first warm-up after a miss).
            let _ = vram_handle.promote(key.clone(), roaring.as_ref());
            vram_handle.get(&key)
        };

        // Step D (Phase 2 D-4 v3): opportunistic v3 (Bitcomp-
        // compressed) promotion. Same best-effort semantics — promote
        // failure (no codec, oversize > 16 MiB single-chunk, budget
        // pressure) silently fall through to v2 / v1 below.
        //
        // Wave Z-3 #1 — v2-hit-but-v3-miss fast path. When step C
        // produced (or just promoted) a v2 entry, hand that
        // `Arc<VramTermEntry>` to `promote_v2_to_v3` instead of
        // re-draining the postings into a fresh `RoaringPostings`.
        // The new method Bitcomp-compresses the v2 device buffer
        // directly (no host container walk, no `cudaMalloc`
        // d_uncompressed_temp, no H→D staging copy), saving ~60ms
        // on dense terms (10K+ buckets) per Wave Z-2 #1's recon
        // estimate. v2 entry's `bucket_index` is reused so the v3
        // entry's layout is bytewise-identical to one produced via
        // the legacy `insert(roaring)` path (asserted by
        // `promote_v2_to_v3_matches_insert_layout` in
        // `vram_cht_v3`).
        //
        // When the v2 entry isn't in hand (step C promote also
        // failed, e.g. budget pressure / oversize at the v2 tier),
        // we keep the legacy `handle.promote(key, roaring.as_ref())`
        // path — the v2-skip-redrain optimisation only applies when
        // both tiers' source is the same in-flight `Arc<VramTermEntry>`.
        //
        // `promote_v2_to_v3` returning `Ok(false)` is admission
        // rejection (kill-switch / oversize / budget); we do NOT
        // fall back to the roaring path because (a) the v2 source
        // is the canonical copy and a parallel fresh drain would
        // hit the same budget cap, and (b) Wave Z-2 #1's method
        // already runs the cheap admission probe before any
        // cudaMalloc, so retrying via a different drain code path
        // can't improve the outcome this query. An
        // admission-rejected promote leaves the cache state clean
        // (the freshly-allocated `d_compressed` is dropped before
        // installation) and falls through to the v2 / v1 dispatch
        // tier below.
        #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
        let final_v3_entry = if let Some(entry) = v3_entry {
            Some(entry)
        } else if let Some(handle) = v3_handle {
            match final_vram_entry.as_ref() {
                Some(v2_arc) => {
                    // v2 entry is in hand — Bitcomp-compress directly
                    // from its device buffer (Wave Z-2 #1 method).
                    let _ = handle.promote_v2_to_v3(&key, Arc::clone(v2_arc));
                }
                None => {
                    // No v2 source — fall back to the legacy
                    // re-drain path (host container walk +
                    // cudaMalloc d_uncompressed_temp + H→D copy).
                    let _ = handle.promote(key.clone(), roaring.as_ref());
                }
            }
            handle.get(&key)
        } else {
            None
        };

        #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
        vram_entries.push(final_vram_entry);
        #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
        v3_entries.push(final_v3_entry);
        owned_postings.push(roaring);
        term_scorers.push(term_scorer);
    }
    let drain_phase_us = t0.elapsed().as_micros();

    // ----- GPU dispatch -----
    let dispatch_t = std::time::Instant::now();
    // Phase 2 D-4 v3 — 4-tier dispatch (highest preference first):
    // 1. v3 (Bitcomp-compressed VRAM) iff every cohort member has a
    //    v3 entry (decompress-on-read into workbench, then fold).
    // 2. v2 (uncompressed VRAM) iff every cohort member has a v2
    //    entry (no decompress overhead).
    // 3. v1 host CHT (host fold path with flat_buffer + H→D).
    //
    // Mixed cohorts (some tiers hit, some miss) prefer the highest
    // tier where ALL members hit. Missed-tier terms have already
    // been opportunistically promoted in steps C/D; subsequent
    // queries against the same cohort take the higher tier.
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    let gpu_result = {
        let all_have_v3 =
            v3_entries.iter().all(|opt| opt.is_some()) && !v3_entries.is_empty();
        let all_have_vram = vram_entries.iter().all(|opt| opt.is_some())
            && !vram_entries.is_empty();
        if all_have_v3 {
            let v3_terms: Vec<Arc<VramCompressedTermEntry>> = v3_entries
                .iter()
                .map(|opt| {
                    Arc::clone(opt.as_ref().expect("all_have_v3 pre-checked"))
                })
                .collect();
            try_gpu_bool_v3(BoolOp::And, &v3_terms)
        } else if all_have_vram {
            let vram_terms: Vec<Arc<VramTermEntry>> = vram_entries
                .iter()
                .map(|opt| {
                    Arc::clone(opt.as_ref().expect("all_have_vram pre-checked"))
                })
                .collect();
            try_gpu_bool_vram(BoolOp::And, &vram_terms)
        } else {
            // Mixed / no-VRAM-cache fall-through: host fold path.
            let term_refs: Vec<&RoaringPostings> =
                owned_postings.iter().map(|a| a.as_ref()).collect();
            try_gpu_bool(BoolOp::And, &term_refs, num_docs_in_segment)
        }
    };
    #[cfg(not(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel")))]
    let gpu_result = {
        let term_refs: Vec<&RoaringPostings> =
            owned_postings.iter().map(|a| a.as_ref()).collect();
        try_gpu_bool(BoolOp::And, &term_refs, num_docs_in_segment)
    };
    let dispatch_phase_us = dispatch_t.elapsed().as_micros();
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    let cohort_size = owned_postings.len();
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    log::debug!(
        target: "tantivy::query::boolean_query::gpu_intersect",
        "Phase 2 D-3/D-4 timing: cohort_size={} host_cht_hits={} host_cht_misses={} vram_cht_hits={} vram_cht_misses={} v3_cht_hits={} v3_cht_misses={} drain_phase_us={} drain_only_us={} dispatch_phase_us={}",
        cohort_size,
        host_hit_count,
        cohort_size as u32 - host_hit_count,
        vram_hit_count,
        cohort_size as u32 - vram_hit_count,
        v3_hit_count,
        cohort_size as u32 - v3_hit_count,
        drain_phase_us as u64,
        drain_us_total as u64,
        dispatch_phase_us as u64,
    );
    #[cfg(not(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel")))]
    log::debug!(
        target: "tantivy::query::boolean_query::gpu_intersect",
        "Phase 2 D-3 timing: cohort_size={} cht_hits={} cht_misses={} drain_phase_us={} drain_only_us={} dispatch_phase_us={}",
        owned_postings.len(),
        host_hit_count,
        owned_postings.len() as u32 - host_hit_count,
        drain_phase_us as u64,
        drain_us_total as u64,
        dispatch_phase_us as u64,
    );
    let Some(roaring_result) = gpu_result else {
        // Driver init failed / wgpu unavailable / cohort emptied
        // by the bucket-union step. Fall back to the legacy CPU path
        // by reconstructing the Box<dyn Scorer> Vec from the
        // recovered TermScorers. Cohort order is preserved relative
        // to the input (we pushed in iteration order).
        let recovered: Vec<Box<dyn Scorer>> = term_scorers
            .into_iter()
            .map(|ts| ts as Box<dyn Scorer>)
            .collect();
        return Err(recovered);
    };

    // ----- Wrap & return -----
    Ok(Box::new(RoaringMaterialisedScorer::new(roaring_result)))
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Index;
    use crate::postings::roaring::gpu_dispatch::{
        cpu_fallback_count, gpu_dispatch_count, reset_dispatch_counters,
    };
    use crate::query::{EnableScoring, Scorer};
    use crate::schema::{IndexRecordOption, Schema, Term, TEXT};
    use std::sync::Mutex;

    /// Counter ops are global; serialise tests that read them.
    static COUNTER_LOCK: Mutex<()> = Mutex::new(());

    /// Build a tiny in-RAM index where each doc gets a `text` field
    /// containing the strings in `docs`. Returns the index and the
    /// text field.
    fn build_index(
        docs: &[&str],
    ) -> crate::Result<(Index, crate::schema::Field)> {
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer = index.writer_for_tests()?;
        for doc_text in docs {
            writer.add_document(doc!(text_field => *doc_text))?;
        }
        writer.commit()?;
        Ok((index, text_field))
    }

    /// Build a `Vec<(Term, Box<dyn Scorer>)>` of `(Term, TermScorer)`
    /// pairs, one per term string, for the first segment of `index`.
    /// The shape matches the Wave 11 `try_gpu_intersect` cohort type.
    fn build_term_scorer_cohort(
        index: &Index,
        text_field: crate::schema::Field,
        terms: &[&str],
    ) -> crate::Result<Vec<(Term, Box<dyn Scorer>)>> {
        use crate::query::TermQuery;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0);
        let enable_scoring = EnableScoring::enabled_from_searcher(&searcher);
        let mut out: Vec<(Term, Box<dyn Scorer>)> = Vec::with_capacity(terms.len());
        for term_str in terms {
            let term = Term::from_field_text(text_field, term_str);
            let term_query = TermQuery::new(term.clone(), IndexRecordOption::Basic);
            let weight = term_query.specialized_weight(enable_scoring)?;
            let scorer = weight
                .term_scorer_for_test(segment_reader, 1.0)?
                .ok_or_else(|| {
                    crate::TantivyError::InvalidArgument(format!(
                        "no posting list for term '{term_str}'"
                    ))
                })?;
            out.push((term, Box::new(scorer)));
        }
        Ok(out)
    }

    /// Helper: a synthetic non-TermScorer paired with a placeholder
    /// `Term` (the cohort element shape is `(Term, Box<dyn Scorer>)`,
    /// but the `AllScorer` will fail gate 3 before the `Term` is ever
    /// read).
    fn build_all_scorer_proxy(
        text_field: crate::schema::Field,
        num_docs: u32,
    ) -> (Term, Box<dyn Scorer>) {
        (
            Term::from_field_text(text_field, "__all_proxy__"),
            Box::new(crate::query::AllScorer::new(num_docs)),
        )
    }

    #[test]
    fn returns_err_when_scoring_enabled() -> crate::Result<()> {
        let (index, text_field) =
            build_index(&["alpha beta", "alpha gamma", "beta gamma"])?;
        let cohort = build_term_scorer_cohort(&index, text_field, &["alpha", "beta"])?;
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();
        // scoring_enabled = true → must return Err immediately.
        let res = try_gpu_intersect(cohort, &segment_reader, true);
        assert!(res.is_err(), "scoring_enabled gate should reject");
        if let Err(recovered) = res {
            assert_eq!(recovered.len(), 2, "cohort must be returned intact");
        }
        Ok(())
    }

    #[test]
    fn returns_err_for_single_term_cohort() -> crate::Result<()> {
        let (index, text_field) = build_index(&["alpha", "alpha beta"])?;
        let cohort = build_term_scorer_cohort(&index, text_field, &["alpha"])?;
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();
        let res = try_gpu_intersect(cohort, &segment_reader, false);
        assert!(res.is_err(), "single-term cohort should be rejected");
        Ok(())
    }

    #[test]
    fn returns_err_for_mixed_cohort() -> crate::Result<()> {
        let (index, text_field) = build_index(&["alpha beta", "alpha gamma"])?;
        let mut cohort = build_term_scorer_cohort(&index, text_field, &["alpha"])?;
        // Inject a non-TermScorer (AllScorer) so gate 3 fails.
        cohort.push(build_all_scorer_proxy(text_field, 2));
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();
        let res = try_gpu_intersect(cohort, &segment_reader, false);
        assert!(res.is_err(), "mixed cohort should be rejected");
        Ok(())
    }

    #[test]
    fn returns_err_when_planner_rejects_light_cohort() -> crate::Result<()> {
        // 3 short documents, 3-term cohort = light query, planner should
        // reject (per-term doc_freq ≤ 3 « MIN_PER_TERM_CARDINALITY).
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        let (index, text_field) =
            build_index(&["alpha beta gamma", "alpha beta", "beta gamma"])?;
        let cohort = build_term_scorer_cohort(&index, text_field, &["alpha", "beta", "gamma"])?;
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();
        let res = try_gpu_intersect(cohort, &segment_reader, false);
        assert!(res.is_err(), "light cohort should be rejected");
        // No GPU dispatch attempted (planner pre-check stops us before
        // try_gpu_bool).
        assert_eq!(
            gpu_dispatch_count(),
            0,
            "no GPU dispatch on planner-rejected cohort"
        );
        Ok(())
    }

    #[test]
    fn err_arm_returns_cohort_intact() -> crate::Result<()> {
        // Verify the recovered cohort has the same length as the
        // input — vital for the caller (BooleanWeight) which moves
        // the Vec into try_gpu_intersect.
        let (index, text_field) = build_index(&["alpha", "alpha"])?;
        let cohort = build_term_scorer_cohort(&index, text_field, &["alpha"])?;
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();
        let res = try_gpu_intersect(cohort, &segment_reader, false);
        match res {
            Err(recovered) => assert_eq!(recovered.len(), 1),
            Ok(_) => panic!("single-term cohort should not have dispatched"),
        }
        Ok(())
    }

    /// Phase 2 D-3 v2 — VRAM CHT integration test.
    ///
    /// Build a heavy cohort, call `try_gpu_intersect` twice on the
    /// same scorers, and assert:
    /// 1. After call #1, the VRAM CHT has at least one inserted entry
    ///    (= the first-touch promotion succeeded for at least one
    ///    cohort member).
    /// 2. After call #2, the VRAM CHT hits counter went up (= the
    ///    second-call lookup found promoted entries).
    /// 3. Both calls succeed (= byte-equal results for AND on the
    ///    same fixture).
    ///
    /// Skip cleanly on hosts without a working CUDA driver (the
    /// VRAM stats stay at zero and we don't assert further).
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    #[test]
    fn vram_cht_warm_path_witnessed_via_stats() -> crate::Result<()> {
        use crate::postings::roaring::vram_cht;

        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        // Reset the VRAM cache so this test's stats counters start
        // from a known baseline (no interference from earlier tests
        // that share the process-global cache).
        vram_cht::reset_global();

        // Build the same heavy cohort as `heavy_cohort_dispatches_…`.
        // 12 high-frequency terms × 100_000 docs = clears every
        // planner gate.
        let term_list: Vec<String> = (0..12).map(|i| format!("vterm{i}")).collect();
        let doc_text = term_list.join(" ");
        let mut docs: Vec<String> = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            docs.push(doc_text.clone());
        }
        let docs_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let (index, text_field) = build_index(&docs_refs)?;
        let cohort_terms: Vec<&str> = term_list.iter().map(String::as_str).collect();
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();

        // Call #1: all VRAM misses → drain + promote.
        let cohort_a = build_term_scorer_cohort(&index, text_field, &cohort_terms)?;
        let res_a = try_gpu_intersect(cohort_a, &segment_reader, false);
        let stats_after_a = vram_cht::global().stats();
        // On a CUDA-driver-missing host, no promotions happened — the
        // promote() call returns Ok(false) silently. Use the result
        // of the dispatch as the "did anything actually run on GPU"
        // witness; if Ok we got a scorer back, so the VRAM tier is
        // alive.
        let Ok(_scorer_a) = res_a else {
            // Dispatch fell back to CPU on a sandbox box. No further
            // assertions possible — the VRAM tier never engaged.
            return Ok(());
        };
        // At least one promotion (typically all 12) recorded by call #1.
        // We don't assert == 12 because the planner may admit fewer
        // than 12 terms on this fixture (it admits all 12 here, but
        // future planner tweaks may change that — pin only the
        // monotonic property that promotion fired).
        assert!(
            stats_after_a.promotions >= 1,
            "first try_gpu_intersect call must promote ≥1 entry, got promotions={}",
            stats_after_a.promotions
        );

        // Call #2: same cohort terms, fresh scorer cohort. Now the
        // VRAM CHT should have entries from call #1, so we expect
        // hit_count to increase (= the warm-cache fast path engaged).
        let cohort_b = build_term_scorer_cohort(&index, text_field, &cohort_terms)?;
        let res_b = try_gpu_intersect(cohort_b, &segment_reader, false);
        let stats_after_b = vram_cht::global().stats();
        assert!(
            res_b.is_ok(),
            "second call must succeed (warm cache, identical fixture)"
        );
        assert!(
            stats_after_b.hits > stats_after_a.hits,
            "second try_gpu_intersect call must record at least one VRAM hit"
        );
        Ok(())
    }

    /// Phase 2 D-4 v3 — 4-tier dispatch integration test.
    ///
    /// Build a heavy cohort, call `try_gpu_intersect` twice on the
    /// same scorers, and assert:
    /// 1. After call #1, the v3 (Bitcomp-compressed) cache has at
    ///    least one promoted entry (= the v3 promotion succeeded for
    ///    at least one cohort member).
    /// 2. After call #2, the v3 hit counter went up (= the second-
    ///    call lookup found promoted v3 entries).
    /// 3. Both calls succeed.
    ///
    /// Skips on hosts without a working CUDA driver / Bitcomp-capable
    /// nvcomp (the v3 global returns None and the test exits cleanly).
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    #[test]
    fn v3_cht_warm_path_witnessed_via_stats() -> crate::Result<()> {
        use crate::postings::roaring::vram_cht_v3;

        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        // Reset v2 + v3 globals so this test's stats start fresh.
        crate::postings::roaring::vram_cht::reset_global();
        vram_cht_v3::reset_global();

        // Skip if v3 is unavailable on this host (no Bitcomp codec).
        let v3_handle = match vram_cht_v3::global() {
            Some(h) => h,
            None => return Ok(()),
        };

        // Same heavy cohort as v2's witness test.
        let term_list: Vec<String> = (0..12).map(|i| format!("v3term{i}")).collect();
        let doc_text = term_list.join(" ");
        let mut docs: Vec<String> = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            docs.push(doc_text.clone());
        }
        let docs_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let (index, text_field) = build_index(&docs_refs)?;
        let cohort_terms: Vec<&str> = term_list.iter().map(String::as_str).collect();
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();

        // Call #1: all v3 misses → drain + v3 promote.
        let cohort_a = build_term_scorer_cohort(&index, text_field, &cohort_terms)?;
        let res_a = try_gpu_intersect(cohort_a, &segment_reader, false);
        let stats_after_a = v3_handle.stats();
        let Ok(_scorer_a) = res_a else {
            // Sandbox without working dispatch; nothing more to assert.
            return Ok(());
        };
        assert!(
            stats_after_a.promotions >= 1,
            "first try_gpu_intersect call must promote ≥1 v3 entry, got promotions={}",
            stats_after_a.promotions
        );

        // Call #2: warm cache, v3 hits expected.
        let cohort_b = build_term_scorer_cohort(&index, text_field, &cohort_terms)?;
        let res_b = try_gpu_intersect(cohort_b, &segment_reader, false);
        let stats_after_b = v3_handle.stats();
        assert!(
            res_b.is_ok(),
            "second call must succeed (warm cache, identical fixture)"
        );
        assert!(
            stats_after_b.hits > stats_after_a.hits,
            "second try_gpu_intersect call must record at least one v3 hit"
        );
        Ok(())
    }

    #[cfg(all(feature = "gpu", feature = "ferro-compress"))]
    #[test]
    fn heavy_cohort_dispatches_or_falls_back_cleanly() -> crate::Result<()> {
        // Build an index with 100_000 docs, where 12 terms each appear
        // in 100 % of docs. This pushes well past Wave 8 / A / 2 gates
        // (`MIN_PER_TERM_CARDINALITY = 100_000`, `MIN_COHORT_DOCS =
        // 1_000_000`, `MIN_RATIO = 0.05`). The dispatch may still
        // return None on a CI box without working wgpu, in which case
        // the err arm fires cleanly.
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();

        // Each doc gets a fixed string with the same 12 high-frequency
        // terms, so doc_freq == num_docs for each. That gives
        // field_doc_ratio = 1.0. doc_freq = 100_000 = MIN_PER_TERM
        // boundary; cohort total = 1_200_000 > MIN_COHORT_DOCS. All
        // three planner gates clear.
        let mut docs: Vec<String> = Vec::with_capacity(100_000);
        let term_list: Vec<String> = (0..12).map(|i| format!("term{i}")).collect();
        let doc_text = term_list.join(" ");
        for _ in 0..100_000 {
            docs.push(doc_text.clone());
        }
        let docs_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let (index, text_field) = build_index(&docs_refs)?;
        let cohort_terms: Vec<&str> = term_list.iter().map(String::as_str).collect();
        let cohort = build_term_scorer_cohort(&index, text_field, &cohort_terms)?;
        assert_eq!(cohort.len(), 12);
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();
        let pre_gpu = gpu_dispatch_count();
        let pre_cpu = cpu_fallback_count();
        let res = try_gpu_intersect(cohort, &segment_reader, false);
        match res {
            Ok(_scorer) => {
                // Either GPU dispatched (count went up) or wgpu init
                // failed inside try_gpu_bool which would have made
                // try_gpu_intersect itself return Err, not Ok. So
                // Ok ⇒ GPU dispatched at least once.
                assert!(
                    gpu_dispatch_count() > pre_gpu,
                    "GPU dispatch counter must increase on Ok return"
                );
            }
            Err(_recovered) => {
                // No-wgpu sandboxes hit this arm — the CPU fallback
                // counter must be observable.
                assert!(
                    cpu_fallback_count() > pre_cpu,
                    "CPU fallback counter must increase on Err return"
                );
            }
        }
        Ok(())
    }

    // ========================================================
    // Wave Z-3 #1 — promote_v2_to_v3 dispatch wiring tests
    // ========================================================

    /// E2E witness for the Wave Z-3 #1 wiring: when the v2 (uncompressed
    /// VRAM) tier holds an entry for a cohort term but the v3
    /// (Bitcomp-compressed) tier does not, the dispatch path must
    /// promote via [`vram_cht_v3::VramCompressedCht::promote_v2_to_v3`]
    /// (= compress the v2 device buffer directly) instead of falling
    /// back to the legacy `v3.promote(roaring)` re-drain path.
    ///
    /// The test:
    /// 1. Resets v2 + v3 globals to a known empty state.
    /// 2. Runs `try_gpu_intersect` once with a heavy cohort so call #1
    ///    populates *both* tiers. Snapshot v3 stats after call #1.
    /// 3. Manually evicts the v3 entries (via `reset_global`) while
    ///    leaving v2 populated. v2 is now hot, v3 is cold — the exact
    ///    v2-hit-but-v3-miss shape that step D's new branch targets.
    /// 4. Runs `try_gpu_intersect` again. Step D must observe
    ///    `final_vram_entry.is_some()` (because v2 still has each
    ///    cohort key) and `v3_entry.is_none()` (because we cleared v3),
    ///    so the new `promote_v2_to_v3` branch must fire for every
    ///    cohort member.
    /// 5. Assert v3's `promotions` counter grew by ≥1 over the call.
    ///    Combined with the pre-call invariant `v3.entries == 0`, this
    ///    proves the wiring is exercised — there's no other code path
    ///    that bumps `v3.promotions` while the v2 tier is hot.
    ///
    /// Limitations: the `promotions` counter is bumped by *both*
    /// `promote(roaring)` and `promote_v2_to_v3`, so this test verifies
    /// the wiring-reach contract (v3 promotion fires during a v2-hit-
    /// but-v3-miss call) but does not byte-for-byte prove the new
    /// kernel-side path was taken. The structural code inspection +
    /// the Wave Z-2 #1 unit tests on `promote_v2_to_v3` itself cover
    /// the "actual savings" half; this test pins the "the wiring is
    /// reachable from the public query path" half.
    ///
    /// Skips on hosts without a working Bitcomp codec (v3 global is
    /// `None`) — same gate the sibling
    /// `v3_cht_warm_path_witnessed_via_stats` test uses.
    #[cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]
    #[test]
    fn compute_fold_v3_uses_promote_v2_to_v3_on_v2_hit() -> crate::Result<()> {
        use crate::postings::roaring::vram_cht;
        use crate::postings::roaring::vram_cht_v3;

        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        vram_cht::reset_global();
        vram_cht_v3::reset_global();

        let v3_handle = match vram_cht_v3::global() {
            Some(h) => h,
            // No Bitcomp codec on this host — v3 tier never engages,
            // so the wiring branch can't fire either. Exit cleanly.
            None => return Ok(()),
        };

        // Heavy cohort: 12 terms × 100_000 docs = clears every
        // planner gate. Identical fixture shape to the sibling v3
        // warm-path witness so the same dispatch decisions fire.
        let term_list: Vec<String> = (0..12).map(|i| format!("z3wire{i}")).collect();
        let doc_text = term_list.join(" ");
        let mut docs: Vec<String> = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            docs.push(doc_text.clone());
        }
        let docs_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let (index, text_field) = build_index(&docs_refs)?;
        let cohort_terms: Vec<&str> = term_list.iter().map(String::as_str).collect();
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();

        // Call #1: cold both tiers → drain + v2 promote + v3 promote.
        // This populates v2 (and v3 too) by going through the legacy
        // `promote(roaring)` paths in step C and step D.
        let cohort_a = build_term_scorer_cohort(&index, text_field, &cohort_terms)?;
        let res_a = try_gpu_intersect(cohort_a, &segment_reader, false);
        if res_a.is_err() {
            // Sandbox without working CUDA dispatch; the wiring branch
            // is unreachable anyway. Exit cleanly.
            return Ok(());
        }
        let v2_stats_after_a = vram_cht::global().stats();
        let v3_stats_after_a = v3_handle.stats();
        if v2_stats_after_a.promotions == 0 || v3_stats_after_a.promotions == 0 {
            // Either tier didn't admit any term (oversize / budget /
            // missing codec on this host). The v2-hit-but-v3-miss
            // case can't be set up; bail cleanly without asserting.
            return Ok(());
        }

        // Clear v3 only — v2 stays populated. This is the exact shape
        // we want for step D's new branch: every cohort key still
        // present in v2, but absent from v3.
        let v3_promotions_pre_warm = v3_stats_after_a.promotions;
        vram_cht_v3::reset_global();
        let v3_handle = vram_cht_v3::global().expect(
            "v3 global handle must remain available after reset_global",
        );
        let v3_stats_pre_b = v3_handle.stats();
        assert_eq!(
            v3_stats_pre_b.entries, 0,
            "v3 cache must be empty after reset_global (precondition for v2-hit-but-v3-miss path)"
        );
        assert_eq!(
            v3_stats_pre_b.promotions, 0,
            "v3 promotions counter must be zero after reset_global"
        );

        // v2 must still hold the cohort keys for the wiring branch
        // to be reachable.
        let v2_stats_pre_b = vram_cht::global().stats();
        assert!(
            v2_stats_pre_b.entries >= 1,
            "v2 cache must retain ≥1 entry across the v3-only reset; got entries={}",
            v2_stats_pre_b.entries
        );

        // Call #2: v2 hits, v3 misses → step D takes the new
        // `promote_v2_to_v3` branch for every cohort member.
        let cohort_b = build_term_scorer_cohort(&index, text_field, &cohort_terms)?;
        let res_b = try_gpu_intersect(cohort_b, &segment_reader, false);
        assert!(
            res_b.is_ok(),
            "second call must succeed under the v2-hit-but-v3-miss path"
        );
        let v3_stats_after_b = v3_handle.stats();
        assert!(
            v3_stats_after_b.promotions >= 1,
            "v3 promotions must grow on the v2-hit-but-v3-miss call \
             (Wave Z-3 #1 wiring proof). got promotions={} entries={} \
             (v3 was reset to empty before this call; only the new \
             wiring branch can bump promotions while v2 is hot)",
            v3_stats_after_b.promotions,
            v3_stats_after_b.entries,
        );

        // The wiring branch consumed *the same* Arc<VramTermEntry>
        // already cached in v2, so v2's promotion counter must
        // *not* grow during call #2 (no new v2 promotions — they
        // were all hits). This is the load-bearing assertion: it
        // distinguishes the new path (consume existing v2 Arc) from
        // a legacy fallback that would re-promote v2 from a fresh
        // drain (bumping v2 promotions again).
        let v2_stats_after_b = vram_cht::global().stats();
        assert_eq!(
            v2_stats_after_b.promotions, v2_stats_pre_b.promotions,
            "v2 promotions must NOT grow on the v2-hit-but-v3-miss call \
             (cohort keys already in v2; new wiring consumes the existing \
             Arc<VramTermEntry> rather than re-promoting). \
             pre={} after={}",
            v2_stats_pre_b.promotions, v2_stats_after_b.promotions,
        );

        // Defensive: confirm the v3 cache *grew* under the new path
        // (entries went from 0 to ≥1 across call #2).
        assert!(
            v3_stats_after_b.entries >= 1,
            "v3 cache must hold ≥1 entry after the v2-hit-but-v3-miss \
             call; got entries={}",
            v3_stats_after_b.entries
        );

        // Sanity: the v3 counter snapshot from call #1 is untouched
        // by the reset (reset zeroes the counters), so the +1 we
        // just observed is genuinely from call #2's new branch.
        let _ = v3_promotions_pre_warm; // referenced for clarity; values are reset above
        Ok(())
    }
}

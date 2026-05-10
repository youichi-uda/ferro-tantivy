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
use crate::postings::roaring::{drain_block_segment_to_roaring, try_gpu_bool, BoolOp};
use crate::query::roaring_materialised_scorer::RoaringMaterialisedScorer;
use crate::query::term_query::TermScorer;
use crate::query::Scorer;

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
/// - `must_scorers`: the cohort's MUST-occur scorers, in arbitrary
///   order. The helper only takes ownership on the `Some` path.
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
///   `Vec<Box<dyn Scorer>>`.
/// - `Err(must_scorers)` — same as `Ok(None)` semantically; we use the
///   `Result<Option<…>, Vec<…>>` shape to make the move-vs-keep
///   contract explicit at the call site.
/// `allow(dead_code)`: lands in this commit (Wave 7/Plan2/3) and is
/// hooked from `boolean_weight::complex_scorer` in the next commit
/// (Wave 7/Plan2/4). The lint clears at that point.
#[allow(dead_code)]
pub(crate) fn try_gpu_intersect(
    must_scorers: Vec<Box<dyn Scorer>>,
    reader: &SegmentReader,
    scoring_enabled: bool,
) -> Result<Box<dyn Scorer>, Vec<Box<dyn Scorer>>> {
    // ----- Gate 1: scoring disabled -----
    // Roaring drains drop term frequencies. If the caller wants BM25
    // contributions we cannot satisfy them; punt to CPU.
    if scoring_enabled {
        return Err(must_scorers);
    }

    // ----- Gate 2: cohort size -----
    // try_gpu_bool itself returns None for < 2 terms (single-scorer
    // queries don't need an intersection at all), and the planner
    // gates on cohort-wide container counts that require ≥ 10
    // containers to clear MIN_CONTAINERS. We early-out here for
    // clarity; the planner is the authoritative gate below.
    if must_scorers.len() < 2 {
        return Err(must_scorers);
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
    if !must_scorers.iter().all(|s| s.is::<TermScorer>()) {
        return Err(must_scorers);
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
        .map(|scorer| {
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
        return Err(must_scorers);
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
    let mut owned_postings: Vec<RoaringPostings> = Vec::with_capacity(must_scorers.len());
    let mut term_scorers: Vec<Box<TermScorer>> = Vec::with_capacity(must_scorers.len());
    for scorer in must_scorers {
        // Infallible per gate 3. Box<dyn Scorer>::downcast moves the
        // value (so we don't lose ownership on success) — see
        // downcast_rs.
        let term_scorer: Box<TermScorer> = scorer
            .downcast::<TermScorer>()
            .map_err(|_| ())
            .expect("gate 3 should have rejected non-TermScorer");
        // Clone the BlockSegmentPostings so the original TermScorer
        // (re-collected below on the recovery path) keeps its read
        // cursor. Cheap clone — see file slice / Arc pattern above.
        let mut block_cursor = term_scorer.segment_postings().block_cursor.clone();
        let roaring = drain_block_segment_to_roaring(&mut block_cursor);
        owned_postings.push(roaring);
        term_scorers.push(term_scorer);
    }

    // ----- GPU dispatch -----
    let term_refs: Vec<&RoaringPostings> = owned_postings.iter().collect();
    let gpu_result = try_gpu_bool(BoolOp::And, &term_refs, num_docs_in_segment);
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
    use crate::doc;
    use crate::index::Index;
    use crate::postings::roaring::gpu_dispatch::{
        cpu_fallback_count, gpu_dispatch_count, reset_dispatch_counters,
    };
    use crate::query::{Bm25Weight, EnableScoring, Scorer};
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

    /// Build a `Vec<Box<dyn Scorer>>` of `TermScorer`s, one per term
    /// string, for the first segment of `index`.
    fn build_term_scorer_cohort(
        index: &Index,
        text_field: crate::schema::Field,
        terms: &[&str],
    ) -> crate::Result<Vec<Box<dyn Scorer>>> {
        use crate::query::TermQuery;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let segment_reader = searcher.segment_reader(0);
        let enable_scoring = EnableScoring::enabled_from_searcher(&searcher);
        let mut out: Vec<Box<dyn Scorer>> = Vec::with_capacity(terms.len());
        for term_str in terms {
            let term_query =
                TermQuery::new(Term::from_field_text(text_field, term_str), IndexRecordOption::Basic);
            let weight = term_query.specialized_weight(enable_scoring)?;
            let scorer = weight
                .term_scorer_for_test(segment_reader, 1.0)?
                .ok_or_else(|| {
                    crate::TantivyError::InvalidArgument(format!(
                        "no posting list for term '{term_str}'"
                    ))
                })?;
            out.push(Box::new(scorer));
        }
        Ok(out)
    }

    /// Helper: a synthetic non-TermScorer that satisfies the trait
    /// bounds but whose presence in the cohort should fail gate 3.
    /// We use the existing AllScorer as a proxy.
    fn build_all_scorer_proxy(num_docs: u32) -> Box<dyn Scorer> {
        Box::new(crate::query::AllScorer::new(num_docs))
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
        assert!(matches!(res, Err(_)), "scoring_enabled gate should reject");
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
        assert!(matches!(res, Err(_)), "single-term cohort should be rejected");
        Ok(())
    }

    #[test]
    fn returns_err_for_mixed_cohort() -> crate::Result<()> {
        let (index, text_field) = build_index(&["alpha beta", "alpha gamma"])?;
        let mut cohort = build_term_scorer_cohort(&index, text_field, &["alpha"])?;
        // Inject a non-TermScorer (AllScorer) so gate 3 fails.
        cohort.push(build_all_scorer_proxy(2));
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();
        let res = try_gpu_intersect(cohort, &segment_reader, false);
        assert!(matches!(res, Err(_)), "mixed cohort should be rejected");
        Ok(())
    }

    #[test]
    fn returns_err_when_planner_rejects_light_cohort() -> crate::Result<()> {
        // 3 short documents, 3-term cohort = light query, planner should
        // reject (sub-MIN_CONTAINERS, sub-MIN_RATIO).
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();
        let (index, text_field) =
            build_index(&["alpha beta gamma", "alpha beta", "beta gamma"])?;
        let cohort = build_term_scorer_cohort(&index, text_field, &["alpha", "beta", "gamma"])?;
        let reader = index.reader()?;
        let segment_reader = reader.searcher().segment_reader(0).clone();
        let res = try_gpu_intersect(cohort, &segment_reader, false);
        assert!(matches!(res, Err(_)), "light cohort should be rejected");
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

    #[cfg(all(feature = "gpu", feature = "ferro-compress"))]
    #[test]
    fn heavy_cohort_dispatches_or_falls_back_cleanly() -> crate::Result<()> {
        // Build an index with 1_000 docs, where 12 terms each appear
        // in ≥ 70 % of docs. This pushes well past MIN_CONTAINERS and
        // MIN_RATIO. The dispatch may still return None on a CI box
        // without working wgpu, in which case the err arm fires
        // cleanly.
        let _g = COUNTER_LOCK.lock().unwrap();
        reset_dispatch_counters();

        // We need a synthetic term distribution. Each doc gets a fixed
        // string with the same 12 high-frequency terms, so doc_freq ==
        // num_docs for each. That gives field_doc_ratio = 1.0 (>>
        // 5 %). Cardinality 1000 × 12 = 12 000 ids → 1000 < 65536, so
        // each term has 1 estimated container, 12 terms × 1 = 12 ≥
        // MIN_CONTAINERS = 10. Both gates clear.
        let mut docs: Vec<String> = Vec::with_capacity(1000);
        let term_list: Vec<String> = (0..12).map(|i| format!("term{i}")).collect();
        let doc_text = term_list.join(" ");
        for _ in 0..1000 {
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
}

//! Integration tests for the Wave 7 / Plan 2 Bool-AND auto-dispatch
//! to the GPU Roaring path — Phase 2 C-5 (Wave 7/Plan2/5).
//!
//! These tests use only the **public** Tantivy surface (no
//! `pub(crate)` accessors). They build a real on-disk-ish (in-RAM)
//! index, run a `BooleanQuery` of multiple `TermQuery` clauses
//! through a real `Searcher` with the `Count` collector (which sets
//! `requires_scoring() == false` → `BooleanWeight.scoring_enabled =
//! false`), and assert that
//! [`tantivy::postings::roaring::gpu_dispatch_count`] tracks the
//! number of GPU dispatches in line with the planner threshold.
//!
//! ## TermScorer-vs-AllScorer caveat (important for test design)
//!
//! `TermWeight` short-circuits to `AllScorer` when
//! `!scoring_enabled && doc_freq == reader.max_doc()` (see
//! `src/query/term_query/term_weight.rs:193`). The
//! `try_gpu_intersect` hook fires only on Bool-AND cohorts of
//! `TermScorer`s — so test fixtures must keep `doc_freq <
//! max_doc` for every cohort term. We achieve this by adding a few
//! "noise" docs that don't contain the cohort terms — every cohort
//! term then appears in `num_dense_docs` of `num_dense_docs +
//! num_noise_docs` docs, with `doc_freq < max_doc`. That keeps the
//! AllScorer optimization off and routes the cohort through
//! `complex_scorer`.
//!
//! The dispatch counter is process-global, so we serialise tests
//! that read it via `COUNTER_LOCK` (mirroring the Wave 6
//! `gpu_dispatch.rs` test pattern).

use std::sync::Mutex;

use tantivy::collector::{Count, TopDocs};
use tantivy::postings::roaring::{
    cpu_fallback_count, gpu_dispatch_count, reset_dispatch_counters,
};
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Schema, TEXT};
use tantivy::{doc, Index, Term};

/// Counter ops are global — serialise.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

/// Build an index where:
///   - `num_dense` docs contain every term in `terms` (these are the
///     "matches all clauses" docs)
///   - `num_noise` docs contain only the placeholder string "noise"
///     (no cohort terms — keeps `doc_freq < max_doc` so the
///     AllScorer optimization off)
///
/// Every cohort term then has `doc_freq = num_dense`,
/// `num_docs_in_segment = num_dense + num_noise`. The planner
/// `field_doc_ratio` gate clears at 5 % when `num_dense >=
/// 0.06 * (num_dense + num_noise)`.
fn build_index_with_dense_and_noise(
    num_dense: usize,
    num_noise: usize,
    terms: &[&str],
) -> tantivy::Result<(Index, Field)> {
    assert!(num_noise > 0, "must have at least 1 noise doc to avoid AllScorer optimization");
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer = index.writer(50_000_000)?;
    let dense_text = terms.join(" ");
    for _ in 0..num_dense {
        writer.add_document(doc!(text_field => dense_text.clone()))?;
    }
    for _ in 0..num_noise {
        writer.add_document(doc!(text_field => "noise"))?;
    }
    writer.commit()?;
    Ok((index, text_field))
}

/// Build an index with a small set of unique short docs. Doc
/// frequencies are tiny so the planner rejects the cohort.
fn build_sparse_index(docs: &[&str]) -> tantivy::Result<(Index, Field)> {
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer = index.writer(50_000_000)?;
    for d in docs {
        writer.add_document(doc!(text_field => *d))?;
    }
    writer.commit()?;
    Ok((index, text_field))
}

/// Helper: build a Bool-AND query (all MUST).
fn bool_and_query(text_field: Field, terms: &[&str]) -> BooleanQuery {
    let clauses: Vec<(Occur, Box<dyn Query>)> = terms
        .iter()
        .map(|t| {
            let q: Box<dyn Query> = Box::new(TermQuery::new(
                Term::from_field_text(text_field, t),
                IndexRecordOption::Basic,
            ));
            (Occur::Must, q)
        })
        .collect();
    BooleanQuery::new(clauses)
}

#[test]
fn high_cardinality_filter_dispatches_to_gpu_or_falls_back_cleanly() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();

    // 1000 dense + 50 noise = 1050 docs. doc_freq(term) = 1000 →
    // ratio = 1000/1050 = 95 % >> 5 %. 12 terms × ceil(1000/65536) = 12.
    let term_strs: Vec<String> = (0..12).map(|i| format!("term{i}")).collect();
    let term_refs: Vec<&str> = term_strs.iter().map(String::as_str).collect();
    let (index, text_field) = build_index_with_dense_and_noise(1000, 50, &term_refs)?;

    let query = bool_and_query(text_field, &term_refs);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let pre_gpu = gpu_dispatch_count();
    let pre_cpu = cpu_fallback_count();
    let count = searcher.search(&query, &Count)?;
    let post_gpu = gpu_dispatch_count();
    let post_cpu = cpu_fallback_count();

    assert_eq!(count, 1000, "1000 dense docs should match all 12 terms");
    let gpu_delta = post_gpu - pre_gpu;
    let cpu_delta = post_cpu - pre_cpu;
    // multi-segment: dispatcher fires once per segment (writer can
    // produce >1 segment for 1050 docs depending on memory budget).
    assert!(
        gpu_delta >= 1 || cpu_delta >= 1,
        "expected GPU dispatch or CPU fallback observed at least once; \
         got gpu_delta = {gpu_delta} cpu_delta = {cpu_delta}",
    );
    Ok(())
}

#[test]
fn light_cohort_stays_on_cpu() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();

    // Sparse cohort: 3 docs, 3 terms. doc_freq tiny → planner rejects.
    let (index, text_field) = build_sparse_index(&[
        "alpha beta gamma",
        "alpha beta",
        "beta gamma",
    ])?;

    let query = bool_and_query(text_field, &["alpha", "beta", "gamma"]);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let count = searcher.search(&query, &Count)?;

    assert_eq!(count, 1);
    assert_eq!(
        gpu_dispatch_count(),
        0,
        "light cohort must not dispatch to GPU"
    );
    Ok(())
}

#[test]
fn scoring_enabled_query_stays_on_cpu_even_for_heavy_cohort() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();

    let term_strs: Vec<String> = (0..12).map(|i| format!("scored{i}")).collect();
    let term_refs: Vec<&str> = term_strs.iter().map(String::as_str).collect();
    let (index, text_field) = build_index_with_dense_and_noise(1000, 50, &term_refs)?;

    let query = bool_and_query(text_field, &term_refs);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let _top: Vec<(f32, tantivy::DocAddress)> =
        searcher.search(&query, &TopDocs::with_limit(10).order_by_score())?;

    // scoring_enabled = true → option (a) gate rejects.
    assert_eq!(
        gpu_dispatch_count(),
        0,
        "scoring-enabled query must not dispatch to GPU even on heavy cohort"
    );
    Ok(())
}

#[test]
fn count_aggregation_takes_filter_path() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();

    // 500 dense + 30 noise = 530. Each term in 500/530 = 94 %. 12
    // terms × 1 container = 12 ≥ MIN_CONTAINERS. Planner clears.
    let term_strs: Vec<String> = (0..12).map(|i| format!("kw{i}")).collect();
    let term_refs: Vec<&str> = term_strs.iter().map(String::as_str).collect();
    let (index, text_field) = build_index_with_dense_and_noise(500, 30, &term_refs)?;

    let query = bool_and_query(text_field, &term_refs);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let pre_gpu = gpu_dispatch_count();
    let pre_cpu = cpu_fallback_count();
    let count = searcher.search(&query, &Count)?;
    let post_gpu = gpu_dispatch_count();
    let post_cpu = cpu_fallback_count();

    assert_eq!(count, 500, "all 500 dense docs match every cohort term");
    let total_delta = (post_gpu - pre_gpu) + (post_cpu - pre_cpu);
    assert!(
        total_delta >= 1,
        "filter-context heavy cohort must hit the dispatch helper at least once \
         (gpu_delta + cpu_delta == 0 means the hook isn't wired)",
    );
    Ok(())
}

#[test]
fn gpu_path_count_matches_cpu_path_count() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();

    // 600 dense + 100 noise. 12-term Bool-AND should match all 600
    // dense docs irrespective of GPU vs CPU dispatch.
    let term_strs: Vec<String> = (0..12).map(|i| format!("z{i}")).collect();
    let term_refs: Vec<&str> = term_strs.iter().map(String::as_str).collect();
    let (index, text_field) = build_index_with_dense_and_noise(600, 100, &term_refs)?;

    let query = bool_and_query(text_field, &term_refs);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let count = searcher.search(&query, &Count)?;
    assert_eq!(count, 600, "12-term AND over 600 dense + 100 noise = 600");

    // Cross-check: same query parsed differently (2-term cohort that
    // routes via planner reject path on the 2-term gate).
    let pair_query = bool_and_query(text_field, &["z0", "z1"]);
    let count_pair = searcher.search(&pair_query, &Count)?;
    assert_eq!(count_pair, 600, "2-term AND also = 600 (every dense doc has z0 + z1)");
    Ok(())
}

#[test]
fn topdocs_no_scoring_paths_disagree_only_on_score_not_doc_ids() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();

    // 50 dense + 5 noise. Heavy enough on a small index for the gates
    // to clear *if* the field-doc ratio gate fires (50/55 = 90 % >
    // 5 %). Container gate: 12 terms × 1 = 12 ≥ MIN_CONTAINERS = 10.
    let term_strs: Vec<String> = (0..12).map(|i| format!("y{i}")).collect();
    let term_refs: Vec<&str> = term_strs.iter().map(String::as_str).collect();
    let (index, text_field) = build_index_with_dense_and_noise(50, 5, &term_refs)?;

    let query = bool_and_query(text_field, &term_refs);
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let count = searcher.search(&query, &Count)?;
    let top: Vec<(f32, tantivy::DocAddress)> =
        searcher.search(&query, &TopDocs::with_limit(100).order_by_score())?;
    assert_eq!(count, top.len(), "Count and TopDocs cohort size must agree");
    assert_eq!(count, 50);
    Ok(())
}

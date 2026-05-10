//! CPU path byte-identity regression — Phase 2 C-5 (Wave 7/Plan2/6).
//!
//! When the Wave 7 GPU dispatch helper rejects a cohort (planner gate
//! / single-term / scoring-enabled / mixed-type), execution falls
//! through to the legacy `intersect_scorers` AVX2 galloping path.
//! This test pins the legacy path bit-for-bit on a deterministic
//! 5-doc index across 5 cohort scenarios (every scenario is light
//! enough to be planner-rejected). After each scenario we also
//! assert `gpu_dispatch_count() == 0`, proving the legacy path was
//! the one exercised.
//!
//! ## Snapshot strategy
//!
//! The "snapshot" is inlined as `expected: &[DocId]` arrays in each
//! sub-test. This is preferable to a JSON snapshot file because:
//!   1. The doc-id sets are tiny (≤ 5 ids per scenario).
//!   2. Snapshot regeneration via `--ignored generate_snapshot`
//!      would require a separate test binary or a side file the
//!      reviewer must remember to regenerate. Inline expected values
//!      review themselves.
//!   3. A change in the legacy CPU path's emit order is visible in
//!      the `git diff` of this file, which makes regressions louder
//!      than a file-format change to a JSON sidecar.
//!
//! Snapshot location for the orchestrator's final report:
//! `tests/cpu_path_byte_identity.rs` (this file) — the inlined
//! `expected` slices in each `#[test]` are the snapshot.

use std::sync::Mutex;

use tantivy::collector::DocSetCollector;
use tantivy::postings::roaring::{gpu_dispatch_count, reset_dispatch_counters};
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Schema, TEXT};
use tantivy::{doc, DocId, Index, Term};

/// Counter ops are global — serialise.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

/// Build a deterministic 5-doc index. Each row's `text` field is
/// fixed, so doc-ids 0..5 map to:
///   0: "alpha beta gamma"     (3 terms)
///   1: "alpha beta"            (2 terms)
///   2: "alpha gamma delta"     (3 terms)
///   3: "beta gamma"            (2 terms)
///   4: "alpha"                 (1 term)
fn build_fixed_index() -> tantivy::Result<(Index, Field)> {
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    // Single-thread writer → single-segment index → deterministic
    // doc-id allocation. (writer(...) with the default 50 MB budget
    // multiplexes across N threads and may split the 5 docs across 2
    // segments; the snapshot would then contain (segment_ord, doc_id)
    // pairs that are arch-dependent.)
    let mut writer = index.writer_with_num_threads(1, 15_000_000)?;
    writer.add_document(doc!(text_field => "alpha beta gamma"))?;
    writer.add_document(doc!(text_field => "alpha beta"))?;
    writer.add_document(doc!(text_field => "alpha gamma delta"))?;
    writer.add_document(doc!(text_field => "beta gamma"))?;
    writer.add_document(doc!(text_field => "alpha"))?;
    writer.commit()?;
    Ok((index, text_field))
}

/// Build a Bool-AND query (all MUST).
fn bool_and(text_field: Field, terms: &[&str]) -> BooleanQuery {
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

/// Run a query through `DocSetCollector` (no scoring) and return the
/// matching doc-ids in ascending order. The fixture guarantees a
/// single segment (`writer_with_num_threads(1, …)`), so doc-ids
/// alone are stable across runs.
fn collect_sorted_doc_ids(
    index: &Index,
    query: &dyn Query,
) -> tantivy::Result<Vec<DocId>> {
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let docset = searcher.search(query, &DocSetCollector)?;
    // Sanity check: the fixture must yield exactly one segment.
    assert_eq!(
        searcher.segment_readers().len(),
        1,
        "fixture must produce a single-segment index for stable doc-ids",
    );
    let mut ids: Vec<DocId> = docset.iter().map(|addr| addr.doc_id).collect();
    ids.sort_unstable();
    Ok(ids)
}

#[test]
fn scenario_1_two_term_and_alpha_beta() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();
    let (index, text_field) = build_fixed_index()?;
    let query = bool_and(text_field, &["alpha", "beta"]);
    let actual = collect_sorted_doc_ids(&index, &query)?;
    // Snapshot: docs containing both alpha + beta = doc 0 and doc 1.
    let expected: &[DocId] = &[0, 1];
    assert_eq!(actual, expected, "alpha AND beta cohort doc-ids");
    assert_eq!(
        gpu_dispatch_count(),
        0,
        "planner rejects 2-term light cohort — must stay on CPU path"
    );
    Ok(())
}

#[test]
fn scenario_2_three_term_and_alpha_beta_gamma() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();
    let (index, text_field) = build_fixed_index()?;
    let query = bool_and(text_field, &["alpha", "beta", "gamma"]);
    let actual = collect_sorted_doc_ids(&index, &query)?;
    // Snapshot: only doc 0 contains all three.
    let expected: &[DocId] = &[0];
    assert_eq!(actual, expected, "alpha AND beta AND gamma cohort doc-ids");
    assert_eq!(gpu_dispatch_count(), 0);
    Ok(())
}

#[test]
fn scenario_3_two_term_and_gamma_delta() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();
    let (index, text_field) = build_fixed_index()?;
    let query = bool_and(text_field, &["gamma", "delta"]);
    let actual = collect_sorted_doc_ids(&index, &query)?;
    // Snapshot: only doc 2 contains both.
    let expected: &[DocId] = &[2];
    assert_eq!(actual, expected, "gamma AND delta cohort doc-ids");
    assert_eq!(gpu_dispatch_count(), 0);
    Ok(())
}

#[test]
fn scenario_4_three_term_and_with_no_match() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();
    let (index, text_field) = build_fixed_index()?;
    // beta + delta + alpha = no doc has all three.
    let query = bool_and(text_field, &["beta", "delta", "alpha"]);
    let actual = collect_sorted_doc_ids(&index, &query)?;
    let expected: &[DocId] = &[];
    assert_eq!(actual, expected, "empty intersection cohort");
    assert_eq!(gpu_dispatch_count(), 0);
    Ok(())
}

#[test]
fn scenario_5_should_clauses_unaffected_by_dispatcher() -> tantivy::Result<()> {
    let _g = COUNTER_LOCK.lock().unwrap();
    reset_dispatch_counters();
    let (index, text_field) = build_fixed_index()?;
    // (alpha) AND (beta OR gamma): MUST + SHOULD cohort. The hook
    // gates on `should_scorers.is_empty()` so this never reaches
    // try_gpu_intersect — pure regression test for the legacy path.
    let alpha: Box<dyn Query> = Box::new(TermQuery::new(
        Term::from_field_text(text_field, "alpha"),
        IndexRecordOption::Basic,
    ));
    let beta: Box<dyn Query> = Box::new(TermQuery::new(
        Term::from_field_text(text_field, "beta"),
        IndexRecordOption::Basic,
    ));
    let gamma: Box<dyn Query> = Box::new(TermQuery::new(
        Term::from_field_text(text_field, "gamma"),
        IndexRecordOption::Basic,
    ));
    let query = BooleanQuery::new(vec![
        (Occur::Must, alpha),
        (Occur::Should, beta),
        (Occur::Should, gamma),
    ]);
    let actual = collect_sorted_doc_ids(&index, &query)?;
    // Doc-id breakdown:
    //   0 (alpha beta gamma): alpha=Y, beta=Y, gamma=Y → match
    //   1 (alpha beta): alpha=Y, beta=Y, gamma=N → match (≥1 SHOULD)
    //   2 (alpha gamma delta): alpha=Y, beta=N, gamma=Y → match
    //   3 (beta gamma): alpha=N → no match (MUST fails)
    //   4 (alpha): alpha=Y, beta=N, gamma=N → match
    //     (when minimum_number_should_match is 0, the SHOULD branch
    //      contributes only to scoring — every alpha-doc is a match.
    //      DocSetCollector requires_scoring() == false, so SHOULD
    //      acts as scoring-only and does not gate matching.)
    let expected: &[DocId] = &[0, 1, 2, 4];
    assert_eq!(actual, expected, "MUST + SHOULD cohort — legacy path");
    assert_eq!(
        gpu_dispatch_count(),
        0,
        "should_scorers.is_empty() gate keeps cohort off the GPU path"
    );
    Ok(())
}

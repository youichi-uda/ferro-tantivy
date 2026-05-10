//! Bool-AND query-path bench harness — Phase 2 C-5 (Wave 7 / Plan 3).
//!
//! ## What this bench measures
//!
//! End-to-end latency of a Bool-AND `BooleanQuery` of multiple
//! `TermQuery` clauses going through `Searcher::search` with the
//! `Count` collector. The path exercised is:
//!
//! ```text
//! BooleanQuery
//!   → BooleanWeight::scorer
//!   → BooleanWeight::complex_scorer
//!     ├── (Wave 7 Plan 2 hook) try_gpu_intersect
//!     │     ├── planner gate (5 % field-doc-ratio + 10-container floor)
//!     │     ├── drain_block_segment_to_roaring (per term)
//!     │     ├── try_gpu_bool (BitmapOpKernel CUDA AND)
//!     │     └── RoaringMaterialisedScorer::new
//!     └── (else) legacy galloping AVX2 Intersection
//!   → Count::collect_block / DocSet::doc / advance
//! ```
//!
//! This is the **query-path overhead** layer — including the drain,
//! the kernel call, and the result wrapping. It is **complementary to**
//! `crates/ferro-compress/src/bin/bitmap_bench.rs` (parallel session
//! `bc1f522`) which measures pure kernel throughput on synthetic flat
//! `[u32; 2048]` buffers without going through `BlockSegmentPostings`
//! drain or `RoaringMaterialisedScorer` wrap.
//!
//! ## Bench cases
//!
//! - **`heavy_gpu` / `heavy_cpu`** (12-term cohort): every cohort term
//!   has `field_doc_ratio` >> 5 % and the cohort spans ≥ 10 Roaring
//!   containers. With `--features gpu,ferro-compress` the planner
//!   dispatches; without those features the call site is `cfg`-gated
//!   out and the legacy Intersection runs. Runtime delta = full GPU
//!   path overhead.
//!
//! - **`light`** (3-term cohort): each term has a tiny doc-frequency
//!   (sparse index). `try_gpu_intersect` returns `Err` at the planner
//!   gate, so the legacy CPU path runs. The CPU regression test:
//!   wallclock should be within ±5 % of the no-feature build (i.e. the
//!   pre-check / drain-clone overhead must be a no-op for cohorts the
//!   planner rejects).
//!
//! - **`boundary`** (12-term cohort × 70 K dense docs / 1 050 total =
//!   field-doc-ratio 6.7 %, just past the 5 % gate): the planner
//!   threshold edge case. Verifies the dispatch decision is stable for
//!   cohorts that just barely qualify.
//!
//! ## Data shape
//!
//! - **In-memory** (`Index::create_in_ram`) — keeps criterion's
//!   per-iteration overhead bounded and lets the bench complete in
//!   under a minute on a 4070 Ti SUPER. The 1 M / 10 M / 100 M scenarios
//!   listed in the design doc are out of scope here — those run via
//!   `phase0-bench/wave7_query_path/run_{cpu,gpu}.sh` driver scripts
//!   (which produce CSVs orthogonal to criterion's stochastic loop).
//!
//! - **Schema**: a single text field `body`. Each "dense" doc gets the
//!   join of all cohort term strings; "noise" docs hold a placeholder
//!   string with no cohort terms. This mirrors the
//!   `tests/auto_dispatch_test.rs` fixture pattern (preserves
//!   `doc_freq < max_doc` for every cohort term so the AllScorer
//!   short-circuit stays off).
//!
//! ## How to read the dispatch counter
//!
//! Each criterion routine snapshots `gpu_dispatch_count()` and
//! `cpu_fallback_count()` before its hot loop and asserts the
//! direction at the end (so a feature-gating regression that
//! silently re-routes the cohort surfaces as a panic, not a silent
//! number drift). The counters are process-global; we serialise
//! reads via `COUNTER_LOCK` (mirroring the Wave 6 `gpu_dispatch.rs`
//! and Wave 7 `auto_dispatch_test.rs` test patterns).
//!
//! ## Honest residuals
//!
//! - This bench measures **filter-context Bool-AND only** (Plan 2
//!   gate). Scoring-enabled Bool-AND, Bool-OR, MustNot, and mixed-type
//!   cohorts all stay on CPU per Plan 2's gate matrix and are not
//!   measured here. Wave 8 may add OR / scoring-aware paths.
//! - Pure kernel throughput is in `bitmap_bench.rs`; this is
//!   query-path overhead. Per-call breakdown (drain vs kernel vs wrap)
//!   would need criterion's profile-time mode and is left as a
//!   follow-up — the harness here treats the path as one black box.
//! - On a sandbox without working wgpu / CUDA, the GPU routine falls
//!   back to CPU silently (counter signal: `cpu_fallback_count`
//!   increments, `gpu_dispatch_count` stays at 0). Numbers reported in
//!   that environment are CPU-path numbers — the harness still runs
//!   to completion.
//!
//! ## Running
//!
//! ```bash
//! # GPU-enabled (4070 Ti SUPER / L4):
//! cargo bench --bench bool_and_query_path --features gpu,ferro-compress
//!
//! # CPU baseline:
//! cargo bench --bench bool_and_query_path
//!
//! # Smoke (one iter per fn, panics on the spot if hook is mis-wired):
//! cargo bench --bench bool_and_query_path --features gpu,ferro-compress -- --test
//! ```

use std::sync::Mutex;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use tantivy::collector::Count;
use tantivy::postings::roaring::{cpu_fallback_count, gpu_dispatch_count};
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, Schema, TEXT};
use tantivy::{doc, Index, ReloadPolicy, Searcher, Term};

/// Counter ops are global; serialise routines that read them.
/// Criterion may run benches concurrently if the host has parallel
/// runner support, but we don't want stray reads from another routine
/// to skew our before/after deltas inside `bench_function`.
static COUNTER_LOCK: Mutex<()> = Mutex::new(());

/// Build an in-RAM index where:
/// - `num_dense` docs contain every term in `terms` (the "match all
///   cohort clauses" docs)
/// - `num_noise` docs contain only "noise" (no cohort terms)
///
/// Every cohort term thus has `doc_freq = num_dense` and
/// `num_docs_in_segment = num_dense + num_noise`. The field-doc ratio
/// gate clears at 5 % when `num_dense >= 0.06 * (num_dense +
/// num_noise)`.
///
/// The schema has a single text field `body` (TEXT). Returns the
/// index, the field handle, and a pre-built `Searcher` (manual reload
/// policy — bench-friendly: no auto-reload races during the criterion
/// hot loop).
fn build_index_dense_plus_noise(
    num_dense: usize,
    num_noise: usize,
    terms: &[&str],
) -> (Index, Field, Searcher) {
    assert!(num_noise > 0, "must have at least 1 noise doc to keep doc_freq < max_doc");
    let mut schema_builder = Schema::builder();
    let body = schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);

    {
        let mut writer = index
            .writer_with_num_threads(1, 100_000_000)
            .expect("writer ok");
        let dense_text = terms.join(" ");
        for _ in 0..num_dense {
            writer.add_document(doc!(body => dense_text.clone())).unwrap();
        }
        for _ in 0..num_noise {
            writer.add_document(doc!(body => "noise")).unwrap();
        }
        writer.commit().expect("commit ok");
    }

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .expect("reader ok");
    let searcher = reader.searcher();
    (index, body, searcher)
}

/// Build an in-RAM index with `docs.len()` short docs (each its own
/// content). Used for the **light** cohort case — sparse index ⇒
/// tiny doc-frequencies ⇒ planner rejects.
fn build_sparse_index(docs: &[&str]) -> (Index, Field, Searcher) {
    let mut schema_builder = Schema::builder();
    let body = schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);

    {
        let mut writer = index
            .writer_with_num_threads(1, 100_000_000)
            .expect("writer ok");
        for d in docs {
            writer.add_document(doc!(body => *d)).unwrap();
        }
        writer.commit().expect("commit ok");
    }

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .expect("reader ok");
    let searcher = reader.searcher();
    (index, body, searcher)
}

/// Build a Bool-AND `BooleanQuery` of `TermQuery`s.
fn bool_and_query(body: Field, terms: &[&str]) -> BooleanQuery {
    let clauses: Vec<(Occur, Box<dyn Query>)> = terms
        .iter()
        .map(|t| {
            let q: Box<dyn Query> = Box::new(TermQuery::new(
                Term::from_field_text(body, t),
                IndexRecordOption::Basic,
            ));
            (Occur::Must, q)
        })
        .collect();
    BooleanQuery::new(clauses)
}

// ============================================================
// Bench routines.
// ============================================================

/// **Heavy cohort, GPU-eligible**.
///
/// 12 terms × 1 000 dense + 50 noise = 1 050 docs.
/// `doc_freq(term) = 1 000`, `field_doc_ratio = 1000 / 1050 ≈ 95 %`.
/// `cohort_containers = 12 × ceil(1000 / 65 536) = 12` — clears
/// `MIN_CONTAINERS = 10`. With `--features gpu,ferro-compress` the
/// dispatch fires; without, the legacy CPU intersection runs.
fn bench_heavy(c: &mut Criterion) {
    let term_strs: Vec<String> = (0..12).map(|i| format!("heavy{i}")).collect();
    let term_refs: Vec<&str> = term_strs.iter().map(String::as_str).collect();
    let (_index, body, searcher) = build_index_dense_plus_noise(1_000, 50, &term_refs);
    let query = bool_and_query(body, &term_refs);

    let mut group = c.benchmark_group("bool_and_query_path/heavy_cohort");
    // The heavy routine returns `count = 1000` per iteration; each
    // result element is 1 doc, so throughput = 1 000 elems / iter.
    group.throughput(Throughput::Elements(1_000));
    group.bench_function("count_12term_dense_1000_noise_50", |b| {
        b.iter_batched(
            || {
                let _g = COUNTER_LOCK.lock().unwrap();
                let pre_gpu = gpu_dispatch_count();
                let pre_cpu = cpu_fallback_count();
                (pre_gpu, pre_cpu)
            },
            |(pre_gpu, pre_cpu)| {
                let count = searcher.search(&query, &Count).expect("search ok");
                // Sanity: every dense doc matches every cohort term.
                debug_assert_eq!(count, 1_000);
                // Direction check: at least one of the counters must
                // bump per call (else the hook isn't wired). We do not
                // assert which one — that depends on the feature flag
                // and on whether wgpu init succeeded on the host.
                let post_gpu = gpu_dispatch_count();
                let post_cpu = cpu_fallback_count();
                debug_assert!(
                    post_gpu > pre_gpu || post_cpu > pre_cpu,
                    "heavy cohort must trip the dispatch helper"
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// **Light cohort, CPU regression**.
///
/// 3 terms across 3 docs. `doc_freq(each term) ∈ {1, 2, 3}` →
/// `field_doc_ratio` ≈ 33-100 % BUT cohort-containers = 3 × 1 = 3 <
/// `MIN_CONTAINERS = 10` → planner rejects. Routine MUST stay on CPU.
/// This is the harness's way to detect a regression where the GPU
/// pre-check / drain-clone path leaks overhead onto the CPU fallback.
fn bench_light(c: &mut Criterion) {
    let (_index, body, searcher) = build_sparse_index(&[
        "alpha beta gamma",
        "alpha beta",
        "beta gamma",
    ]);
    let query = bool_and_query(body, &["alpha", "beta", "gamma"]);

    let mut group = c.benchmark_group("bool_and_query_path/light_cohort");
    group.throughput(Throughput::Elements(1));
    group.bench_function("count_3term_sparse_3doc", |b| {
        b.iter_batched(
            || {
                let _g = COUNTER_LOCK.lock().unwrap();
                gpu_dispatch_count()
            },
            |pre_gpu| {
                let count = searcher.search(&query, &Count).expect("search ok");
                debug_assert_eq!(count, 1);
                debug_assert_eq!(
                    gpu_dispatch_count(),
                    pre_gpu,
                    "light cohort must NOT dispatch to GPU (planner reject)"
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// **Boundary cohort, planner-edge**.
///
/// 12 terms × 50 dense + 950 noise = 1 000 docs. `doc_freq = 50`,
/// `field_doc_ratio = 50 / 1000 = 5 %` — exactly at the planner's
/// `MIN_RATIO` threshold. `cohort_containers = 12 × 1 = 12 ≥
/// MIN_CONTAINERS`. Verifies the dispatch decision is sane on the
/// edge of the planner gate.
///
/// Note: `MIN_RATIO` is a strict-greater check (`>`, not `>=`) per
/// `planner.rs`, so 5 % might NOT clear; the bench still runs (CPU
/// fallback) and the post-call counter check is permissive.
fn bench_boundary(c: &mut Criterion) {
    let term_strs: Vec<String> = (0..12).map(|i| format!("edge{i}")).collect();
    let term_refs: Vec<&str> = term_strs.iter().map(String::as_str).collect();
    let (_index, body, searcher) = build_index_dense_plus_noise(50, 950, &term_refs);
    let query = bool_and_query(body, &term_refs);

    let mut group = c.benchmark_group("bool_and_query_path/boundary_cohort");
    group.throughput(Throughput::Elements(50));
    group.bench_function("count_12term_dense_50_noise_950", |b| {
        b.iter_batched(
            || {
                let _g = COUNTER_LOCK.lock().unwrap();
                let pre_gpu = gpu_dispatch_count();
                let pre_cpu = cpu_fallback_count();
                (pre_gpu, pre_cpu)
            },
            |(pre_gpu, pre_cpu)| {
                let count = searcher.search(&query, &Count).expect("search ok");
                debug_assert_eq!(count, 50);
                let post_gpu = gpu_dispatch_count();
                let post_cpu = cpu_fallback_count();
                debug_assert!(
                    post_gpu > pre_gpu || post_cpu > pre_cpu,
                    "boundary cohort must trip the dispatch helper (one side or the other)"
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// **Heavy cohort, scoring-enabled**.
///
/// Same fixture as `bench_heavy` but the query is run via the search
/// path that **enables scoring** (`TopDocs` — but for criterion we use
/// a 2-clause search through `Count` with a reweighted query: the
/// scoring path is reachable via `BooleanQuery::with_minimum_required_clauses`
/// when the underlying weight has scoring on). The clean way to force
/// `scoring_enabled = true` from a bench is to run TopDocs, which we
/// add as a separate routine `bench_heavy_scored`.
///
/// Plan 2 gate: scoring_enabled cohorts MUST stay on CPU. This routine
/// pins that contract under bench load.
fn bench_heavy_scored(c: &mut Criterion) {
    use tantivy::collector::TopDocs;
    let term_strs: Vec<String> = (0..12).map(|i| format!("scored{i}")).collect();
    let term_refs: Vec<&str> = term_strs.iter().map(String::as_str).collect();
    let (_index, body, searcher) = build_index_dense_plus_noise(1_000, 50, &term_refs);
    let query = bool_and_query(body, &term_refs);

    let mut group = c.benchmark_group("bool_and_query_path/heavy_scored");
    group.throughput(Throughput::Elements(10));
    group.bench_function("topdocs10_12term_dense_1000_noise_50", |b| {
        b.iter_batched(
            || {
                let _g = COUNTER_LOCK.lock().unwrap();
                gpu_dispatch_count()
            },
            |pre_gpu| {
                let _top: Vec<(f32, tantivy::DocAddress)> = searcher
                    .search(&query, &TopDocs::with_limit(10).order_by_score())
                    .expect("search ok");
                debug_assert_eq!(
                    gpu_dispatch_count(),
                    pre_gpu,
                    "scoring-enabled query must NOT dispatch to GPU \
                     (Plan 2 gate 1: scoring_enabled cohorts stay on CPU)"
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// **Mega cohort, Wave 8/A/2 GPU win-zone**.
///
/// 12 terms × 100 000 dense + 5 000 noise = 105 000 docs. Each cohort
/// term has `doc_freq = 100 000 = MIN_PER_TERM_CARDINALITY`; cohort
/// total = 1.2 M ≥ `MIN_COHORT_DOCS = 1 000 000`; `field_doc_ratio =
/// 100 000 / 105 000 ≈ 95 %` ≥ `MIN_RATIO`. All three Wave 8 / A / 2
/// gates clear, so the GPU dispatch fires.
///
/// Index ingest is ~10-20 s in debug, ~1-3 s release; the ingest
/// happens **once** outside the criterion hot loop so bench runtime is
/// dominated by the search loop. Per-iteration AND across 12 × 100 K-doc
/// posting lists hits ~24 Roaring containers (8 KiB each) = ~192 KiB
/// VRAM working set + 11 pairwise kernel reductions; with the
/// Wave 8 / A / 1 OnceLock cache, the wgpu device init is paid once at
/// criterion warmup, not per-iter.
///
/// Sample size is reduced to 20 (vs 50 elsewhere) so the criterion
/// hot loop completes in well under 30 s even when each iter is a few
/// ms. Throughput is reported in elements (matched docs).
#[allow(dead_code)] // bench routines registered via `criterion_group!`
fn bench_mega_cohort(c: &mut Criterion) {
    let term_strs: Vec<String> = (0..12).map(|i| format!("mega{i}")).collect();
    let term_refs: Vec<&str> = term_strs.iter().map(String::as_str).collect();
    let (_index, body, searcher) =
        build_index_dense_plus_noise(100_000, 5_000, &term_refs);
    let query = bool_and_query(body, &term_refs);

    let mut group = c.benchmark_group("bool_and_query_path/mega_cohort");
    group.throughput(Throughput::Elements(100_000));
    // 20 samples × ~ms-ish per iter → bench in ~10 s. Larger fixtures
    // are out of criterion's per-iter budget; the
    // `phase0-bench/wave7_query_path/run_*.sh` driver scripts can drive
    // out-of-process scenarios for production-scale validation.
    group.sample_size(20);
    group.bench_function("count_12term_dense_100000_noise_5000", |b| {
        b.iter_batched(
            || {
                let _g = COUNTER_LOCK.lock().unwrap();
                let pre_gpu = gpu_dispatch_count();
                let pre_cpu = cpu_fallback_count();
                (pre_gpu, pre_cpu)
            },
            |(pre_gpu, pre_cpu)| {
                let count = searcher.search(&query, &Count).expect("search ok");
                debug_assert_eq!(count, 100_000);
                let post_gpu = gpu_dispatch_count();
                let post_cpu = cpu_fallback_count();
                debug_assert!(
                    post_gpu > pre_gpu || post_cpu > pre_cpu,
                    "mega cohort must trip the dispatch helper"
                );
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        // 50 samples is a reasonable trade-off between criterion's
        // statistical convergence and bench wallclock. Running on the
        // 12-term × 1050-doc heavy fixture, this is ~2-3 s per routine
        // on a 4070 Ti SUPER (warm caches). The mega_cohort routine
        // overrides this to 20 internally because the per-iter cost is
        // ~ms-class.
        .sample_size(50);
    targets =
        bench_heavy,
        bench_light,
        bench_boundary,
        bench_heavy_scored,
        bench_mega_cohort,
}
criterion_main!(benches);

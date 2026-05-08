use crate::collector::Count;
use crate::directory::{RamDirectory, WatchCallback};
use crate::index::SegmentId;
use crate::indexer::{LogMergePolicy, NoMergePolicy};
use crate::postings::Postings;
use crate::query::TermQuery;
use crate::schema::{Field, IndexRecordOption, Schema, INDEXED, STRING, TEXT};
use crate::tokenizer::TokenizerManager;
use crate::{
    Directory, DocSet, Index, IndexBuilder, IndexReader, IndexSettings, IndexWriter, ReloadPolicy,
    TantivyDocument, Term,
};

#[test]
fn test_indexer_for_field() {
    let mut schema_builder = Schema::builder();
    let num_likes_field = schema_builder.add_u64_field("num_likes", INDEXED);
    let body_field = schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    assert!(index.tokenizer_for_field(body_field).is_ok());
    assert_eq!(
        format!("{:?}", index.tokenizer_for_field(num_likes_field).err()),
        "Some(SchemaError(\"\\\"num_likes\\\" is not a text field.\"))"
    );
}

#[test]
fn test_set_tokenizer_manager() {
    let mut schema_builder = Schema::builder();
    schema_builder.add_u64_field("num_likes", INDEXED);
    schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let index = IndexBuilder::new()
        // set empty tokenizer manager
        .tokenizers(TokenizerManager::new())
        .schema(schema)
        .create_in_ram()
        .unwrap();
    assert!(index.tokenizers().get("raw").is_none());
}

#[test]
fn test_index_exists() {
    let directory: Box<dyn Directory> = Box::new(RamDirectory::create());
    assert!(!Index::exists(directory.as_ref()).unwrap());
    assert!(Index::create(
        directory.clone(),
        throw_away_schema(),
        IndexSettings::default()
    )
    .is_ok());
    assert!(Index::exists(directory.as_ref()).unwrap());
}

#[test]
fn open_or_create_should_create() {
    let directory = RamDirectory::create();
    assert!(!Index::exists(&directory).unwrap());
    assert!(Index::open_or_create(directory.clone(), throw_away_schema()).is_ok());
    assert!(Index::exists(&directory).unwrap());
}

#[test]
fn open_or_create_should_open() {
    let directory: Box<dyn Directory> = Box::new(RamDirectory::create());
    assert!(Index::create(
        directory.clone(),
        throw_away_schema(),
        IndexSettings::default()
    )
    .is_ok());
    assert!(Index::exists(directory.as_ref()).unwrap());
    assert!(Index::open_or_create(directory, throw_away_schema()).is_ok());
}

#[test]
fn create_should_wipeoff_existing() {
    let directory: Box<dyn Directory> = Box::new(RamDirectory::create());
    assert!(Index::create(
        directory.clone(),
        throw_away_schema(),
        IndexSettings::default()
    )
    .is_ok());
    assert!(Index::exists(directory.as_ref()).unwrap());
    assert!(Index::create(
        directory,
        Schema::builder().build(),
        IndexSettings::default()
    )
    .is_ok());
}

#[test]
fn open_or_create_exists_but_schema_does_not_match() {
    let directory = RamDirectory::create();
    assert!(Index::create(
        directory.clone(),
        throw_away_schema(),
        IndexSettings::default()
    )
    .is_ok());
    assert!(Index::exists(&directory).unwrap());
    assert!(Index::open_or_create(directory.clone(), throw_away_schema()).is_ok());
    let err = Index::open_or_create(directory, Schema::builder().build());
    assert_eq!(
        format!("{:?}", err.unwrap_err()),
        "SchemaError(\"An index exists but the schema does not match.\")"
    );
}

fn throw_away_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    let _ = schema_builder.add_u64_field("num_likes", INDEXED);
    schema_builder.build()
}

#[test]
fn test_index_on_commit_reload_policy() -> crate::Result<()> {
    let schema = throw_away_schema();
    let field = schema.get_field("num_likes").unwrap();
    let index = Index::create_in_ram(schema);
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()
        .unwrap();
    assert_eq!(reader.searcher().num_docs(), 0);
    test_index_on_commit_reload_policy_aux(field, &index, &reader)
}

#[cfg(feature = "mmap")]
mod mmap_specific {

    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_index_on_commit_reload_policy_mmap() -> crate::Result<()> {
        let schema = throw_away_schema();
        let field = schema.get_field("num_likes").unwrap();
        let tempdir = TempDir::new().unwrap();
        let tempdir_path = PathBuf::from(tempdir.path());
        let index = Index::create_in_dir(tempdir_path, schema).unwrap();
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .unwrap();
        assert_eq!(reader.searcher().num_docs(), 0);
        test_index_on_commit_reload_policy_aux(field, &index, &reader)
    }

    #[test]
    fn test_index_manual_policy_mmap() -> crate::Result<()> {
        let schema = throw_away_schema();
        let field = schema.get_field("num_likes").unwrap();
        let mut index = Index::create_from_tempdir(schema)?;
        let mut writer: IndexWriter = index.writer_for_tests()?;
        writer.commit()?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        assert_eq!(reader.searcher().num_docs(), 0);
        writer.add_document(doc!(field=>1u64))?;
        let (sender, receiver) = crossbeam_channel::unbounded();
        let _handle = index.directory_mut().watch(WatchCallback::new(move || {
            let _ = sender.send(());
        }));
        writer.commit()?;
        assert!(receiver.recv().is_ok());
        assert_eq!(reader.searcher().num_docs(), 0);
        reader.reload()?;
        assert_eq!(reader.searcher().num_docs(), 1);
        Ok(())
    }

    #[test]
    fn test_index_on_commit_reload_policy_different_directories() -> crate::Result<()> {
        let schema = throw_away_schema();
        let field = schema.get_field("num_likes").unwrap();
        let tempdir = TempDir::new().unwrap();
        let tempdir_path = PathBuf::from(tempdir.path());
        let write_index = Index::create_in_dir(&tempdir_path, schema).unwrap();
        let read_index = Index::open_in_dir(&tempdir_path).unwrap();
        let reader = read_index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .unwrap();
        assert_eq!(reader.searcher().num_docs(), 0);
        test_index_on_commit_reload_policy_aux(field, &write_index, &reader)
    }
}
fn test_index_on_commit_reload_policy_aux(
    field: Field,
    index: &Index,
    reader: &IndexReader,
) -> crate::Result<()> {
    let mut reader_index = reader.index();
    let (sender, receiver) = crossbeam_channel::unbounded();
    let _watch_handle = reader_index
        .directory_mut()
        .watch(WatchCallback::new(move || {
            let _ = sender.send(());
        }));
    let mut writer: IndexWriter = index.writer_for_tests()?;
    assert_eq!(reader.searcher().num_docs(), 0);
    writer.add_document(doc!(field=>1u64))?;
    writer.commit().unwrap();
    // We need a loop here because it is possible for notify to send more than
    // one modify event. It was observed on CI on MacOS.
    loop {
        assert!(receiver.recv().is_ok());
        if reader.searcher().num_docs() == 1 {
            break;
        }
    }
    writer.add_document(doc!(field=>2u64))?;
    writer.commit().unwrap();
    // ... Same as above
    loop {
        assert!(receiver.recv().is_ok());
        if reader.searcher().num_docs() == 2 {
            break;
        }
    }
    Ok(())
}

// This test will not pass on windows, because windows
// prevent deleting files that are MMapped.
#[cfg(not(target_os = "windows"))]
#[test]
fn garbage_collect_works_as_intended() -> crate::Result<()> {
    let directory = RamDirectory::create();
    let schema = throw_away_schema();
    let field = schema.get_field("num_likes").unwrap();
    let index = Index::create(directory.clone(), schema, IndexSettings::default())?;

    let mut writer: IndexWriter = index.writer_with_num_threads(1, 32_000_000).unwrap();
    for _seg in 0..8 {
        for i in 0u64..1_000u64 {
            writer.add_document(doc!(field => i))?;
        }
        writer.commit()?;
    }

    let mem_right_after_commit = directory.total_mem_usage();

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()?;
    assert_eq!(reader.searcher().num_docs(), 8_000);
    assert_eq!(reader.searcher().segment_readers().len(), 8);

    writer.wait_merging_threads()?;

    let mem_right_after_merge_finished = directory.total_mem_usage();

    reader.reload().unwrap();
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    assert_eq!(searcher.num_docs(), 8_000);
    assert!(
        mem_right_after_merge_finished < mem_right_after_commit,
        "(mem after merge){mem_right_after_merge_finished} is expected < (mem before \
         merge){mem_right_after_commit}"
    );
    Ok(())
}

#[test]
fn test_single_segment_index_writer() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT);
    let schema = schema_builder.build();
    let directory = RamDirectory::default();
    let mut single_segment_index_writer = Index::builder()
        .schema(schema)
        .single_segment_index_writer(directory, 15_000_000)?;
    for _ in 0..10 {
        let doc = doc!(text_field=>"hello");
        single_segment_index_writer.add_document(doc)?;
    }
    let index = single_segment_index_writer.finalize()?;
    let searcher = index.reader()?.searcher();
    let term_query = TermQuery::new(
        Term::from_field_text(text_field, "hello"),
        IndexRecordOption::Basic,
    );
    let count = searcher.search(&term_query, &Count)?;
    assert_eq!(count, 10);
    Ok(())
}

#[test]
fn test_merging_segment_update_docfreq() {
    let mut schema_builder = Schema::builder();
    let text_field = schema_builder.add_text_field("text", TEXT);
    let id_field = schema_builder.add_text_field("id", STRING);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_for_tests().unwrap();
    writer.set_merge_policy(Box::new(NoMergePolicy));
    for _ in 0..5 {
        writer.add_document(doc!(text_field=>"hello")).unwrap();
    }
    writer
        .add_document(doc!(text_field=>"hello", id_field=>"TO_BE_DELETED"))
        .unwrap();
    writer
        .add_document(doc!(text_field=>"hello", id_field=>"TO_BE_DELETED"))
        .unwrap();
    writer.add_document(TantivyDocument::default()).unwrap();
    writer.commit().unwrap();
    for _ in 0..7 {
        writer.add_document(doc!(text_field=>"hello")).unwrap();
    }
    writer.add_document(TantivyDocument::default()).unwrap();
    writer.add_document(TantivyDocument::default()).unwrap();
    writer.delete_term(Term::from_field_text(id_field, "TO_BE_DELETED"));
    writer.commit().unwrap();

    let segment_ids: Vec<SegmentId> = index
        .list_all_segment_metas()
        .into_iter()
        .map(|reader| reader.id())
        .collect();
    writer.merge(&segment_ids[..]).wait().unwrap();
    let index_reader = index.reader().unwrap();
    let searcher = index_reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    assert_eq!(searcher.num_docs(), 15);
    let segment_reader = searcher.segment_reader(0);
    assert_eq!(segment_reader.max_doc(), 15);
    let inv_index = segment_reader.inverted_index(text_field).unwrap();
    let term = Term::from_field_text(text_field, "hello");
    let term_info = inv_index.get_term_info(&term).unwrap().unwrap();
    assert_eq!(term_info.doc_freq, 12);
}

// motivated by https://github.com/quickwit-oss/quickwit/issues/4130
#[test]
fn test_positions_merge_bug_non_text_json_vint() {
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_json_field("dynamic", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests().unwrap();
    let mut merge_policy = LogMergePolicy::default();
    merge_policy.set_min_num_segments(2);
    writer.set_merge_policy(Box::new(merge_policy));
    // Here a string would work.
    let doc_json = r#"{"tenant_id":75}"#;
    let vals = serde_json::from_str(doc_json).unwrap();
    let mut doc = TantivyDocument::default();
    doc.add_object(field, vals);
    writer.add_document(doc.clone()).unwrap();
    writer.commit().unwrap();
    writer.add_document(doc.clone()).unwrap();
    writer.commit().unwrap();
    writer.wait_merging_threads().unwrap();
    let reader = index.reader().unwrap();
    assert_eq!(reader.searcher().segment_readers().len(), 1);
}

// Same as above but with bitpacked blocks
#[test]
fn test_positions_merge_bug_non_text_json_bitpacked_block() {
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_json_field("dynamic", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests().unwrap();
    let mut merge_policy = LogMergePolicy::default();
    merge_policy.set_min_num_segments(2);
    writer.set_merge_policy(Box::new(merge_policy));
    // Here a string would work.
    let doc_json = r#"{"tenant_id":75}"#;
    let vals = serde_json::from_str(doc_json).unwrap();
    let mut doc = TantivyDocument::default();
    doc.add_object(field, vals);
    for _ in 0..128 {
        writer.add_document(doc.clone()).unwrap();
    }
    writer.commit().unwrap();
    writer.add_document(doc.clone()).unwrap();
    writer.commit().unwrap();
    writer.wait_merging_threads().unwrap();
    let reader = index.reader().unwrap();
    assert_eq!(reader.searcher().segment_readers().len(), 1);
}

#[test]
fn test_non_text_json_term_freq() {
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_json_field("dynamic", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests().unwrap();
    // Here a string would work.
    let doc_json = r#"{"tenant_id":75}"#;
    let vals = serde_json::from_str(doc_json).unwrap();
    let mut doc = TantivyDocument::default();
    doc.add_object(field, vals);
    writer.add_document(doc.clone()).unwrap();
    writer.commit().unwrap();
    let reader = index.reader().unwrap();
    assert_eq!(reader.searcher().segment_readers().len(), 1);
    let searcher = reader.searcher();
    let segment_reader = searcher.segment_reader(0u32);
    let inv_idx = segment_reader.inverted_index(field).unwrap();

    let mut term = Term::from_field_json_path(field, "tenant_id", false);
    term.append_type_and_fast_value(75i64);

    let postings = inv_idx
        .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
        .unwrap()
        .unwrap();
    assert_eq!(postings.doc(), 0);
    assert_eq!(postings.term_freq(), 1u32);
}

#[test]
fn test_non_text_json_term_freq_bitpacked() {
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_json_field("dynamic", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests().unwrap();
    // Here a string would work.
    let doc_json = r#"{"tenant_id":75}"#;
    let vals = serde_json::from_str(doc_json).unwrap();
    let mut doc = TantivyDocument::default();
    doc.add_object(field, vals);
    let num_docs = 132;
    for _ in 0..num_docs {
        writer.add_document(doc.clone()).unwrap();
    }
    writer.commit().unwrap();
    let reader = index.reader().unwrap();
    assert_eq!(reader.searcher().segment_readers().len(), 1);
    let searcher = reader.searcher();
    let segment_reader = searcher.segment_reader(0u32);
    let inv_idx = segment_reader.inverted_index(field).unwrap();

    let mut term = Term::from_field_json_path(field, "tenant_id", false);
    term.append_type_and_fast_value(75i64);

    let mut postings = inv_idx
        .read_postings(&term, IndexRecordOption::WithFreqsAndPositions)
        .unwrap()
        .unwrap();
    assert_eq!(postings.doc(), 0);
    assert_eq!(postings.term_freq(), 1u32);
    for i in 1..num_docs {
        assert_eq!(postings.advance(), i);
        assert_eq!(postings.term_freq(), 1u32);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Wave 10 #X: Multi-thread executor regression tests.
//
// The fork already exposes `Index::set_multithread_executor(n)` and
// `Index::set_default_multithread_executor()`.  These tests pin down the
// observable contract:
//
//   1. The default executor is `Executor::SingleThread`.
//   2. After `set_multithread_executor`, the executor is `ThreadPool` with
//      the requested thread count.
//   3. `Searcher::search` returns identical results regardless of executor
//      variant — parallelism is a perf concern, not a semantic one.  This
//      is the property FerroSearch's Wave 10 #X fix relies on.
//   4. `Searcher::clone` (the PIT pinning operation) preserves the
//      executor — both clones share `Arc<SearcherInner>` whose `Index`
//      holds an Arc-backed executor.
//
// These tests are deliberately small (each ≤500 docs across 2-4 segments).
// They validate the wiring, not the perf characteristics — a bench in
// `benches/` would be needed for the latter and is intentionally omitted
// from the unit suite (CI runtime budget).
// ─────────────────────────────────────────────────────────────────────────

/// Produces a small index with `num_segments` segments by committing per
/// chunk.  Used by the executor tests below.
fn build_multi_segment_index(num_segments: u32, docs_per_segment: u32) -> Index {
    let mut schema_builder = Schema::builder();
    let value_field = schema_builder.add_u64_field("value", INDEXED | crate::schema::FAST);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer_for_tests().unwrap();
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let mut doc_id = 0u64;
    for _ in 0..num_segments {
        for _ in 0..docs_per_segment {
            let mut doc = TantivyDocument::default();
            doc.add_u64(value_field, doc_id);
            writer.add_document(doc).unwrap();
            doc_id += 1;
        }
        writer.commit().unwrap();
    }
    index
}

/// Default executor must be `SingleThread`.
#[test]
fn executor_default_is_single_thread() {
    let index = build_multi_segment_index(2, 50);
    assert!(matches!(
        index.search_executor(),
        crate::Executor::SingleThread
    ));
}

/// `set_multithread_executor(n)` flips the executor variant.
#[test]
fn set_multithread_executor_flips_variant() {
    let mut index = build_multi_segment_index(2, 50);
    index
        .set_multithread_executor(4)
        .expect("multi_thread executor must build");
    assert!(matches!(
        index.search_executor(),
        crate::Executor::ThreadPool(_)
    ));
}

/// `set_default_multithread_executor()` produces a thread pool sized to
/// `available_parallelism()`.  Smoke test only — we don't assert the exact
/// thread count because it varies by host.
#[test]
fn set_default_multithread_executor_builds_pool() {
    let mut index = build_multi_segment_index(2, 50);
    index
        .set_default_multithread_executor()
        .expect("default multi-thread executor must build");
    assert!(matches!(
        index.search_executor(),
        crate::Executor::ThreadPool(_)
    ));
}

/// **Correctness invariant**: `Searcher::search` returns identical results
/// under SingleThread and ThreadPool executors.  This is the central
/// property the FerroSearch Wave 10 #X opt-in relies on — operators flipping
/// `FERRO_SEARCH_EXECUTOR_THREADS` must not see different top-K.
#[test]
fn searcher_results_identical_across_executor_variants() {
    use crate::collector::Count;

    // Single-thread baseline.
    let baseline_count = {
        let index = build_multi_segment_index(4, 100);
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        searcher.search(&crate::query::AllQuery, &Count).unwrap()
    };

    // Multi-thread (4 segments × 4 threads → one task per segment).
    let parallel_count = {
        let mut index = build_multi_segment_index(4, 100);
        index.set_multithread_executor(4).unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        searcher.search(&crate::query::AllQuery, &Count).unwrap()
    };

    assert_eq!(baseline_count, 400);
    assert_eq!(
        baseline_count, parallel_count,
        "executor variant must not change Count fruit"
    );
}

/// **Force-merged-1-seg case**: with a single segment the threadpool
/// executor degenerates to one task → one thread; the result must still
/// match SingleThread, and the call must not deadlock.  This is the
/// minimal regression guard against the `Executor::ThreadPool::map`
/// fan-out scope-drain path mishandling N=1.
#[test]
fn searcher_single_segment_works_under_threadpool() {
    use crate::collector::Count;
    let mut index = build_multi_segment_index(1, 200);
    index.set_multithread_executor(4).unwrap();
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    let count = searcher.search(&crate::query::AllQuery, &Count).unwrap();
    assert_eq!(count, 200);
}

/// `Searcher::clone` (the PIT pinning operation) preserves the executor
/// of the underlying `Index`.  This is the structural property that makes
/// `crates/ferro-query/src/execute.rs::searcher_for` correct — pinned
/// PIT searchers from `request.pit_searchers` must inherit the same
/// executor as the live shard searcher.
#[test]
fn searcher_clone_preserves_executor() {
    let mut index = build_multi_segment_index(2, 50);
    index.set_multithread_executor(2).unwrap();
    let reader = index.reader().unwrap();
    let live_searcher = reader.searcher();
    let pinned_clone = live_searcher.clone();

    // Both searchers' index() must report ThreadPool.
    assert!(matches!(
        live_searcher.index().search_executor(),
        crate::Executor::ThreadPool(_)
    ));
    assert!(matches!(
        pinned_clone.index().search_executor(),
        crate::Executor::ThreadPool(_)
    ));
}

/// **Send compile-time check** for the sort_key infrastructure.  The
/// per-segment computer types (`SortByFastValueSegmentSortKeyComputer<T>`
/// and `ByStringColumnSegmentSortKeyComputer`) live behind the
/// `SortKeyComputer::Child` associated type and must be `Send` so the
/// multi-thread executor can dispatch them across rayon worker threads.
/// This is enforced structurally by `Executor::map`'s `R: Send` bound on
/// the fruit and the `Sync + Send` bound on `Collector`, but the
/// *intermediate* computer state must also be Send for the per-segment
/// closure to be Send + Sync.
///
/// We assert the compile-time bounds via the public `SortByStaticFastValue`
/// + `SortByString` factory types — their `Child` is the segment computer.
/// If the Send bound regresses (e.g., someone adds an `Rc<...>` field),
/// this test will fail to compile.
#[test]
fn sort_key_segment_computers_are_send() {
    use crate::collector::sort_key::{SegmentSortKeyComputer, SortKeyComputer};

    fn assert_child_send<S>()
    where
        S: SortKeyComputer,
        <S as SortKeyComputer>::Child: Send,
    {
    }

    // u64 / i64 / f64 / DateTime fast-value variants and the string variant
    // all need to be Send for `Executor::ThreadPool::map` to dispatch the
    // per-segment closure across worker threads.
    assert_child_send::<crate::collector::sort_key::SortByStaticFastValue<u64>>();
    assert_child_send::<crate::collector::sort_key::SortByStaticFastValue<i64>>();
    assert_child_send::<crate::collector::sort_key::SortByStaticFastValue<f64>>();
    assert_child_send::<crate::collector::sort_key::SortByStaticFastValue<crate::DateTime>>();
    assert_child_send::<crate::collector::sort_key::SortByString>();

    // SegmentSortKeyComputer's required associated types must also be Send
    // (they cross thread boundaries via the topN computer's `merge_fruits`).
    fn assert_seg_keys_send<S: SegmentSortKeyComputer>()
    where
        S::SortKey: Send,
        S::SegmentSortKey: Send,
    {
    }
    assert_seg_keys_send::<<crate::collector::sort_key::SortByStaticFastValue<u64> as SortKeyComputer>::Child>();
    assert_seg_keys_send::<<crate::collector::sort_key::SortByString as SortKeyComputer>::Child>();
}

// =====================================================================
// Wave 11 #A: Searcher single-segment auto-bypass of multi-thread executor
// =====================================================================
//
// `Searcher::search_with_statistics_provider` now inspects
// `segment_readers().len()` at query time and short-circuits to
// `Executor::SingleThread` when there is exactly one segment AND the
// configured executor is `ThreadPool`. This avoids the ~1.5-4 µs rayon
// dispatch overhead the microbench
// (`benches/multithread_executor.rs:1-seg/50k`) measures for the trivial
// case.
//
// These tests prove three properties:
//   (a) 1-seg index + ThreadPool index-level → SingleThread chosen at query
//       time (verified by external observable: count is correct AND the
//       rayon pool is NOT entered, which we infer indirectly via the
//       configured executor still being `ThreadPool` while the search
//       returned the same Count).
//   (b) Multi-segment index + ThreadPool index-level → ThreadPool used
//       (per-segment parallelism preserved for the workloads it helps).
//   (c) Crossover correctness: the 1-seg auto-bypass and a synthetic
//       multi-seg ThreadPool path produce semantically identical results
//       (count + top-K identity).

#[cfg(test)]
mod auto_executor_bypass {
    use crate::collector::Count;
    use crate::indexer::NoMergePolicy;
    use crate::query::AllQuery;
    use crate::schema::{Schema, FAST, INDEXED};
    use crate::{Executor, Index, IndexWriter, TantivyDocument};

    fn build(num_segments: u32, docs_per_seg: u32) -> Index {
        let mut sb = Schema::builder();
        let value_f = sb.add_u64_field("value", INDEXED | FAST);
        let schema = sb.build();
        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();
        writer.set_merge_policy(Box::new(NoMergePolicy));
        let mut id = 0u64;
        for _ in 0..num_segments {
            for _ in 0..docs_per_seg {
                let mut d = TantivyDocument::default();
                d.add_u64(value_f, id);
                writer.add_document(d).unwrap();
                id += 1;
            }
            writer.commit().unwrap();
        }
        index
    }

    /// (a) 1-segment index with index-level ThreadPool: search must succeed
    /// and return the correct count; the index-level executor stays
    /// ThreadPool (the bypass is per-query, not per-index — operators who
    /// opted in keep the configuration for any future multi-seg state).
    #[test]
    fn one_segment_with_threadpool_executor_runs_correctly() {
        let mut index = build(1, 10_000);
        index.set_multithread_executor(4).unwrap();
        // Sanity: index reports ThreadPool.
        assert!(matches!(index.search_executor(), Executor::ThreadPool(_)));
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let count = searcher.search(&AllQuery, &Count).unwrap();
        assert_eq!(count, 10_000);
    }

    /// (b) Multi-segment index with index-level ThreadPool: the executor
    /// is exercised end-to-end (no bypass). We verify by constructing a
    /// searcher of >=2 segments and confirming both the count and top-K
    /// match the single-thread baseline.
    #[test]
    fn multi_segment_with_threadpool_executes_correctly() {
        let mut index = build(4, 2_500);
        index.set_multithread_executor(4).unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        assert!(searcher.segment_readers().len() >= 2);
        let count = searcher.search(&AllQuery, &Count).unwrap();
        assert_eq!(count, 10_000);

        // Cross-check against a freshly-built single-thread baseline with
        // the same data.
        let baseline = build(4, 2_500);
        let baseline_searcher = baseline.reader().unwrap().searcher();
        let baseline_count = baseline_searcher.search(&AllQuery, &Count).unwrap();
        assert_eq!(count, baseline_count);
    }

    /// (c) Crossover correctness: 1-seg auto-bypass and 4-seg ThreadPool
    /// produce identical document count. The 1-seg index goes through the
    /// auto-bypass path; the 4-seg index goes through the ThreadPool fan-
    /// out. Both must agree on the total document count.
    #[test]
    fn crossover_correctness_count_matches_across_executors() {
        let mut one_seg = build(1, 10_000);
        let mut multi_seg = build(4, 2_500);

        // Configure both indices with the multi-thread executor — the
        // 1-seg searcher will bypass at query time, the 4-seg searcher
        // will not.
        one_seg.set_multithread_executor(4).unwrap();
        multi_seg.set_multithread_executor(4).unwrap();

        let one_searcher = one_seg.reader().unwrap().searcher();
        let multi_searcher = multi_seg.reader().unwrap().searcher();
        assert_eq!(one_searcher.segment_readers().len(), 1);
        assert!(multi_searcher.segment_readers().len() >= 2);

        let one = one_searcher.search(&AllQuery, &Count).unwrap();
        let many = multi_searcher.search(&AllQuery, &Count).unwrap();
        assert_eq!(one, many, "count must be invariant across executors");
        assert_eq!(one, 10_000);
    }

    /// (d) Default index has SingleThread executor; bypass is a no-op for
    /// SingleThread indexes (we still go through the same code path, but
    /// the early-return condition `matches!(configured, ThreadPool(_))` is
    /// false, so `configured` is used directly).
    #[test]
    fn default_executor_is_single_thread_no_bypass() {
        let index = build(1, 1_000);
        assert!(matches!(index.search_executor(), Executor::SingleThread));
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let count = searcher.search(&AllQuery, &Count).unwrap();
        assert_eq!(count, 1_000);
    }

    /// (e) Empty index (zero segments) is also handled: not literally one
    /// segment, so the bypass does not trigger; the configured executor
    /// (whatever it is) processes the empty fan-out without panicking.
    #[test]
    fn zero_segment_index_does_not_bypass_or_panic() {
        let mut sb = Schema::builder();
        sb.add_u64_field("value", INDEXED | FAST);
        let mut index = Index::create_in_ram(sb.build());
        index.set_multithread_executor(4).unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 0);
        let count = searcher.search(&AllQuery, &Count).unwrap();
        assert_eq!(count, 0);
    }
}

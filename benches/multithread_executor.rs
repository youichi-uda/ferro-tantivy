// SPDX-License-Identifier: MIT
//
// Wave 10 #X: tantivy multi-thread executor micro-bench.
// Wave 11 #A: 1-segment auto-bypass coverage added — the
// `threadpool_4_1seg_autobypass` line proves the per-query bypass closes
// the regression the original `threadpool_4_1seg` line documents.
//
// This bench measures the throughput of `Searcher::search` on a multi-segment
// index under SingleThread vs ThreadPool executors.  It is the
// "before/after" reference for the FerroSearch
// `FERRO_SEARCH_EXECUTOR_THREADS` opt-in: with N segments each carrying
// significant per-segment work (here: u64 range scan + Count), the
// threadpool executor parallelises segment collection.
//
// Run with:
//   cargo bench --bench multithread_executor
//
// The bench is intentionally synthetic — real perf wins depend heavily on
// per-segment work density (BM25 vs match_all vs sort).  The signal we
// look for is "ThreadPool is at most 2× slower than SingleThread for the
// 1-segment case (dispatch noise dominates) and is meaningfully faster on
// the 8-segment case". With Wave 11 #A's auto-bypass, the 1-seg
// `threadpool_4_1seg_autobypass` line should match `single_thread_1seg`
// within bench noise (the bypass collapses the dispatch overhead).
// Absolute numbers are reported but should not be extrapolated to
// production.

use binggan::plugins::PeakMemAllocPlugin;
use binggan::{black_box, BenchRunner, PeakMemAlloc, INSTRUMENTED_SYSTEM};
use tantivy::collector::Count;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::AllQuery;
use tantivy::schema::{Schema, FAST, INDEXED};
use tantivy::{Index, IndexWriter, TantivyDocument};

#[global_allocator]
pub static GLOBAL: &PeakMemAlloc<std::alloc::System> = &INSTRUMENTED_SYSTEM;

fn build_index(num_segments: u32, docs_per_seg: u32) -> Index {
    let mut schema_builder = Schema::builder();
    let value_field = schema_builder.add_u64_field("value", INDEXED | FAST);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    // Single-thread writer so segment count exactly equals commit count
    // (the bench's `num_segments` invariant).  `15_000_000` matches
    // `MEMORY_BUDGET_NUM_BYTES_MIN`.
    let mut writer: IndexWriter = index.writer_with_num_threads(1, 15_000_000).unwrap();
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let mut id = 0u64;
    for _ in 0..num_segments {
        for _ in 0..docs_per_seg {
            let mut doc = TantivyDocument::default();
            doc.add_u64(value_field, id);
            writer.add_document(doc).unwrap();
            id += 1;
        }
        writer.commit().unwrap();
    }
    index
}

fn bench_executor(runner: &mut BenchRunner, index: &Index, label: &'static str) {
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let n_segs = searcher.segment_readers().len();
    runner.bench_function(label, move |_| {
        let count = searcher.search(&AllQuery, &Count).unwrap();
        black_box(count);
        Some(n_segs as u64)
    });
}

fn main() {
    let mut runner = BenchRunner::new();
    runner.add_plugin(PeakMemAllocPlugin::new(GLOBAL));

    // ----- 1 segment, 50_000 docs -----
    runner.set_name("AllQuery+Count on 1 segment / 50k docs");
    let mut single_seg = build_index(1, 50_000);
    bench_executor(&mut runner, &single_seg, "single_thread_1seg");

    single_seg.set_multithread_executor(4).unwrap();
    // Wave 11 #A: this row used to be ~1.8× slower than single_thread_1seg
    // due to rayon dispatch overhead. With the auto-bypass landed in
    // `Searcher::search_with_statistics_provider`, the configured
    // ThreadPool executor is short-circuited at query time when the
    // searcher holds exactly one segment, so this row should match
    // `single_thread_1seg` within bench noise. Keeping the original label
    // for back-compat with historical runs; the corrected label is the
    // `_autobypass` row below for clarity.
    bench_executor(&mut runner, &single_seg, "threadpool_4_1seg");
    bench_executor(&mut runner, &single_seg, "threadpool_4_1seg_autobypass");

    // ----- 8 segments, 6_250 docs each (50k total) -----
    runner.set_name("AllQuery+Count on 8 segments / 6.25k each / 50k total");
    let mut multi_seg = build_index(8, 6_250);
    bench_executor(&mut runner, &multi_seg, "single_thread_8seg");

    multi_seg.set_multithread_executor(4).unwrap();
    bench_executor(&mut runner, &multi_seg, "threadpool_4_8seg");

    multi_seg.set_multithread_executor(8).unwrap();
    bench_executor(&mut runner, &multi_seg, "threadpool_8_8seg");

    // ----- 4 segments, 100_000 docs each (400k total): per-segment work
    //       is large enough that thread dispatch cost is amortised.  This
    //       is the "force-merge isn't an option / sort is expensive"
    //       regime that motivated FerroSearch Wave 10 #X.
    runner.set_name("AllQuery+Count on 4 segments / 100k each / 400k total");
    let mut large_multi = build_index(4, 100_000);
    bench_executor(&mut runner, &large_multi, "single_thread_4x100k");

    large_multi.set_multithread_executor(4).unwrap();
    bench_executor(&mut runner, &large_multi, "threadpool_4_4x100k");
}

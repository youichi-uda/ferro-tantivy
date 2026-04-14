# Ferro-Tantivy Upstream Sync Audit

**Date:** 2026-04-14
**Fork:** `youichi-uda/ferro-tantivy`
**Upstream:** `quickwit-oss/tantivy` (`main`)
**Audit scope:** Read-only. No source edits, no rebase, no push.

## 1. Remote state

| remote | url | added |
|---|---|---|
| origin | github.com/youichi-uda/ferro-tantivy | pre-existing |
| upstream | github.com/quickwit-oss/tantivy | added 2026-04-14 (fetch-only use) |

`git fetch upstream` executed. No write operations performed on upstream refs.

## 2. Fork divergence

- Merge-base: `a65107135` (upstream commit predating the fork split)
- **Ahead:** 26 commits (fork-only)
- **Behind:** 4 commits (upstream-only, all on `upstream/main`)
- Fork tip: `6d8b567e7` (`gpu: serialize wgpu dispatch and readback per device`)
- Upstream tip: `58aa4b707` (`Fix cardinality aggregation using invalid coupons (#2893)`)

Cumulative fork delta vs merge-base: **+9,620 / -539** across 124 files (includes full `gpu/` crate: +36 files).

NOTE: Working tree currently has uncommitted edits (~20 files, +89/-46) from a concurrent agent session; this audit did not touch or commit them.

## 3. Custom patches (fork-only commits)

Listed newest → oldest. Subsystem tags map to the memory-recorded "8 patches + GPU + SQCache".

| # | SHA | LOC | Subsystem | Purpose |
|---|---|---|---|---|
| 1 | 6d8b567e7 | +21 | **GPU dispatch serialization** | Serialize wgpu dispatch+readback per device (concurrency safety) |
| 2 | 77a1b36d3 | +59/-5 | **Store reader poison recovery** | Recover from poisoned block-cache mutex (resilience) |
| 3 | 6bf1ee8e7 | +35 | **rewrite_schema_on_disk** | `Index::rewrite_schema_on_disk` for additive schema evolution |
| 4 | c1bc96c09 | +23/-2 | Fast field batch | Single-segment fast path in `batch_get_field_owned_bytes` |
| 5 | 2e20c4d68 | +27/-5 | FastFieldRangeQuery | I64/U64 field type support |
| 6 | 76198083c | +3/-3 | Range query fix | EmptyScorer fallback (no tracing dep) |
| 7 | b86f4912a | +6/-7 | Range query fix | Return EmptyScorer instead of panic |
| 8 | 5f0532037 | +4/-33 | Merge cleanup | Remove duplicate Searcher methods, suppress dead_code |
| 9 | ca0abd0dc | +1241/-49 | GPU merge | Merge `feat/gpu-acceleration` into main |
| 10 | 4ec934916 | +3 | Docs | VecColumn field docs |
| 11 | 9f4163c22 | +46/-33 | **MaxScoreBulkScorer** | Multi-essential union + index_writer batch improvements |
| 12 | bb8fe9dfd | +88/-24 | **Phrase query opt** | Batch BitSet insertion + match_phrase_prefix direct terminfo |
| 13 | 15c2ecbd0 | +1151/-12 | **BlockWAND + ferrosearch API** | Block-max WAND scoring + ferrosearch compatibility APIs |
| 14 | 0d0324da5 | +406/-15 | DocStore zero-copy | Zero-copy DocStore field extraction |
| 15 | 9a723c2eb | +24/-9 | SSTable CompositeKey | Cache order + reuse SmallVec |
| 16 | c678741fa | +12/-3 | DocStore robustness | Block-skip on decompression failure |
| 17 | 6ba9edfc2 | +107/-8 | DocStore batch | Batch block processing + combined (Count, TopDocs) |
| 18 | cfc5b308a | +4/-4 | Docs | batch_fast_field_bytes tradeoff clarification |
| 19 | e79c90c8b | +34/-25 | Review fixes | SSTable error handling, UTF-8, `_id` DocStore fallback |
| 20 | 75cc93a87 | +81/-30 | **SQCache / sorted_ords_to_term** | Batch fast-field reads via `sorted_ords_to_term_cb` |
| 21 | a690a0bee | +2/-2 | Sort fix | Derive is_asc from order |
| 22 | 032ded21f | +206 | Fast-field batch API | `order_by_fast_field_with_cursor` + batch APIs |
| 23 | 6206586b9 | +62/-2 | search_after cursor | Fast-field TopDocs cursor support |
| 24 | 813530629 | +943/-477 | **Performance bundle** | Read/write/query perf patches (broad churn) |
| 25 | 03fbeb8b3 | +3/-2 | Indexer tuning | PIPELINE_MAX_SIZE_IN_DOCS 10K→100K |
| 26 | f28e92c07 | +6459 | **tantivy-gpu crate** | New GPU acceleration layer (wgpu/WGSL) |

Total unique LOC across fork patches: ~**11,050 added** (dominated by `gpu/` crate at 6.4K + perf bundle 943 + BlockWAND 1151 + GPU merge 1241).

### Patch → memory-label mapping

- `rewrite_schema_on_disk` → #3
- Store reader poison recovery → #2
- GPU dispatch serialization → #1
- SQCache (sorted_ords_to_term path) → #20 (+ batch fast-field APIs #22, #4)
- BlockWAND exclude / block-max scoring → #13, #11
- Phrase query optimization → #12 (+ broader in #24)
- GPU crate → #26, #9 (merge commit)
- FastFieldRangeQuery extensions → #5, #6, #7
- Performance bundle → #24

## 4. Upstream hot items (4 commits since fork-base)

| SHA | Subject | Files | Impact on fork |
|---|---|---|---|
| 58aa4b707 | Fix cardinality aggregation invalid coupons (#2893) | cardinality.rs | Bug fix — worth pulling |
| 04beab3b2 | Performance: nested cardinality aggregation | 15 files incl. `cached_sub_aggs.rs → buffered_sub_aggs.rs` **rename** | Touches agg/composite/term_agg/sstable — overlaps fork patches |
| 3cd9011f8 | Make BucketEntries::iter / PercentileValuesVecEntry / TopNComputer::threshold public (#2890) | 3 files | API surface widening — trivial forward-merge |
| d2c1b8bc2 | Optimized intersection count (bitset when first leg dense) | bitset.rs, count_collector.rs, docset.rs, intersection.rs, term_scorer.rs | Touches `intersection.rs` / `term_scorer.rs` which fork also modified |

Upstream total: **+833 / -236 across 23 files.** Since there are only 4 commits, the sync window is small — a rebase now is much cheaper than letting it drift.

## 5. Overlapping files (fork × upstream)

12 files were modified by BOTH sides since merge-base:

```
Cargo.toml
common/src/bitset.rs
src/aggregation/bucket/composite/collector.rs
src/aggregation/bucket/composite/mod.rs
src/aggregation/bucket/filter.rs
src/aggregation/bucket/histogram/histogram.rs
src/aggregation/bucket/term_agg.rs
src/aggregation/mod.rs
src/collector/top_score_collector.rs
src/query/intersection.rs
src/query/term_query/term_scorer.rs
sstable/src/dictionary.rs
```

Additionally, upstream renamed `src/aggregation/cached_sub_aggs.rs → buffered_sub_aggs.rs`. Fork does not touch this file directly, but `src/aggregation/mod.rs` references it — rename conflict is resolvable but needs attention.

## 6. Rebase difficulty

| Patch subsystem | Difficulty | Reason |
|---|---|---|
| GPU crate (#26, #9, #1) | **easy** | New `gpu/` tree; no upstream overlap |
| rewrite_schema_on_disk (#3) | **easy** | `src/index/index.rs` untouched by upstream |
| Store poison recovery (#2) | **easy** | `src/store/reader.rs` untouched by upstream |
| DocStore zero-copy / batch (#14, #17, #16, #4) | **easy** | DocStore area untouched by upstream |
| Fast-field batch + cursor (#22, #20, #23, #21) | **easy** | Fast-field collector area untouched |
| FastFieldRangeQuery extensions (#5–#7) | **easy** | Isolated to `fast_field_range_weight.rs` |
| Phrase opt (#12) | **easy** | Phrase scorer/weight untouched upstream |
| MaxScoreBulkScorer (#11) | **medium** | Touches `term_scorer.rs` — upstream `d2c1b8bc2` adds a doc-gen method; mechanical merge |
| BlockWAND + ferrosearch API (#13) | **medium** | Largest functional patch; touches `boolean_query/*` + `top_score_collector.rs` (public API widened by upstream `3cd9011f8` — likely additive, not conflicting) |
| Perf bundle (#24, 58 files) | **medium→hard** | Broad churn; overlaps `bitset.rs`, `intersection.rs`, `term_scorer.rs`, `sstable/dictionary.rs`, composite collector, term_agg, filter, histogram, top_score_collector |
| Cargo.toml (every patch) | **easy** | Version/feature bumps; manual reconcile |
| Aggregation overlap (composite, term_agg, filter, histogram) | **hard** | Upstream `04beab3b2` is a heavy refactor (152 LoC in term_agg alone) + sub_aggs file rename; fork's aggregation touch-ups likely conflict line-by-line. Largest risk area. |
| sstable/dictionary.rs | **medium** | Upstream rewrote 108 LoC; fork touches 5 LoC — probably easy but verify semantics |

Overall: **MEDIUM**, dominated by the aggregation cluster. With only 4 upstream commits, a per-file 3-way merge is tractable in a single sitting.

## 7. Recommended strategy

**Preferred: incremental rebase onto `upstream/main`, preserving patch granularity.**

Rationale: 26 fork commits are already well-categorised (GPU / perf / API). Squashing would lose the attribution that makes future audits cheap. Only 4 upstream commits to absorb — the cost is low now and compounds if deferred.

### Proposed sequence

1. **Pre-flight** (separate working-tree clean required — resolve the 20 uncommitted files first; they are outside this audit's scope).
2. Create branch `sync/upstream-2026-04-14` from current `main`.
3. `git rebase upstream/main sync/upstream-2026-04-14` with `--rebase-merges` to preserve `ca0abd0dc` (GPU merge).
4. Expected conflict hotspots (resolve in this order — easiest first):
   - `sstable/src/dictionary.rs` (small)
   - `common/src/bitset.rs` (small)
   - `src/query/intersection.rs` + `term_scorer.rs` (BlockWAND interaction — cross-check with patch #13)
   - `src/collector/top_score_collector.rs` (public API expansion — accept upstream, keep fork additions)
   - `src/aggregation/bucket/{composite,term_agg,filter,histogram}` (hard; heaviest upstream refactor)
   - `src/aggregation/mod.rs` + `cached_sub_aggs → buffered_sub_aggs` rename
   - `Cargo.toml` (trivial)
5. After rebase: `cargo check --workspace --all-features`, `cargo test -p tantivy --lib` (aggregation unit tests are the risk gate).
6. Run FerroSearch-side smoke against the rebased fork before pushing.

### Alternative strategies (rejected)

- **Squash merge fork → upstream**: loses patch attribution; rejected.
- **Patch replay via `git format-patch`**: same result as rebase but with weaker tooling for conflict resolution.
- **Feature flag isolation**: already implicit via `gpu/` crate being separate; no further split needed.
- **Skip sync / pin upstream**: `04beab3b2` is a real perf win for nested cardinality — worth pulling. `58aa4b707` is a correctness fix. Skipping is not recommended.

### Watch items for next sync (post-rebase)

- Upstream has many active feature branches (wasm, typed-column, u128-columnar, warming). None are merged to `main` yet, but u128-columnar and warming would be disruptive if merged.
- FerroSearch-side API contract (`rewrite_schema_on_disk`, `order_by_fast_field_with_cursor`, `batch_get_field_owned_bytes`, BlockWAND exclusion) should be codified in an internal compat test before rebasing, so breakage is caught.
- Once rebased, tag `ferro-v34.1` or similar so FerroSearch can pin deterministically.

## 8. Build status

`cargo check` was NOT run in this audit — the working tree has ~20 files with uncommitted edits from a concurrent agent session. Running `cargo check` would reflect that state, not the committed HEAD. Recommend running post-handoff once the working tree is clean.

---

**Audit commits (not pushed):** 1 — this file only.

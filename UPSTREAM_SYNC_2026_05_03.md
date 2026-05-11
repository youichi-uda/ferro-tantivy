---
name: Ferro-Tantivy Upstream Sync Audit (refresh)
date: 2026-05-03
prev: UPSTREAM_SYNC.md (2026-04-14)
fork: youichi-uda/ferro-tantivy
upstream: quickwit-oss/tantivy main
scope: read-only audit + Tier-1 PR plan (no PR submission yet)
---

# Ferro-Tantivy Upstream Sync Audit — Refresh

19 days after the 2026-04-14 audit. Goal: send clean PRs upstream so the fork eventually converges to upstream code.

## 1. Divergence (current)

| metric | 2026-04-14 | **2026-05-03** | delta |
|---|---|---|---|
| Merge-base | a65107135 | a65107135 | unchanged |
| Fork ahead | 26 | **32** | +6 |
| Upstream ahead | 4 | **31** | +27 |
| Fork tip | 6d8b567e7 | 785884fd5 | |
| Upstream tip | 58aa4b707 | (current) | |
| Fork delta vs MB | +9,620 / -539 | **+10,569 / -664** | +949 / -125 |
| Upstream delta vs MB | +833 / -236 | **+2,635 / -414** | +1,802 / -178 |

The sync window has tripled in size. Merging upstream is now meaningfully more expensive than 19 days ago, but **none of the 5 Tier-1 PR candidates are affected** (per-file overlap = 0, see §3).

## 2. New upstream activity (27 commits since prev audit)

**Subsystems heavily touched:**

- **Block-Max WAND rework** — `block_wand.rs` renamed to `block_wand_union.rs` + new `block_wand_intersection.rs` (+464 LOC). Single-scorer BMW path, tightened thresholds.
  - 4480cf0a9 enable BMW for single-scorer
  - d27ca164a single-scorer path when only one scorer
  - 322286ee1 (#2897) tighten BMW in single-scorer
- **Top-K + intersection optimization** (Adrien Grand's ideas, #2865) — touches `term_scorer.rs`, intersection iteration
- **Aggregation perf** — composite `cached_sub_aggs.rs` → `buffered_sub_aggs.rs` rename + active-bucket-only memory measurement + nested cardinality fix
- **term_agg.rs** — heaviest churn (+446 LOC) — nested cardinality + early cutoff for order_by sub-agg
- **Query parser** — O(2^n) regression fix for deeply-nested queries (#2905)
- **Histogram memory fix** + facet sentinel ord skip (#2867)
- **CI hardening** — pinned action SHAs, OpenSSF Scorecard, restricted token permissions

**Implication for fork's BlockWAND patch (#13, +1151 LOC):** likely now MOSTLY OBSOLETE. Upstream re-implemented from scratch. Fork should evaluate whether to drop #13 entirely and adopt upstream's intersection/union split.

## 3. Tier-1 PR candidates — per-file overlap with upstream

Verified: none of these touched files have any upstream commits since merge-base (full history checked, not just since prev audit).

| # | SHA | Files | Upstream overlap | LOC | Status |
|---|---|---|---|---|---|
| #3 | 6bf1ee8e7 | `src/index/index.rs` | **0** | +35 | ✅ Ready |
| #2 | 77a1b36d3 | `src/store/reader.rs` | **0** | +59/-5 | ✅ Ready |
| #5–#7 | 2e20c4d68, 76198083c, b86f4912a | `src/query/fast_field_range_weight.rs`, `src/query/range_query/range_query_fastfield.rs` | **0** | +33/-10 | ✅ Ready (bundle as 1 PR) |
| #23 | 6206586b9 | `src/collector/mod.rs`, `sort_key/order.rs`, `sort_key/sort_by_static_fast_value.rs` | **0** | +62/-2 | ✅ Ready |
| (new) | 30c4be76c | `src/query/phrase_query/{phrase_query,phrase_scorer,phrase_weight}.rs` | **0** | +? | ✅ Ready |

**5 PRs total, all expected to apply cleanly.** Each will go on its own branch off `upstream/main`.

## 4. Re-classification of fork patches

Compared to the 2026-04-14 audit, the following patches changed tier:

| patch | prev tier | **new tier** | reason |
|---|---|---|---|
| #1 GPU dispatch serialization | T1 (easy) | **T3 (proposal)** | depends on #26 GPU crate (+6459 LOC) which doesn't exist upstream — single PR is impossible |
| #11 MaxScoreBulkScorer multi-essential | T2 (medium) | **T3 (likely obsolete)** | upstream rewrote BMW (#2897 + 4480cf0a9) — fork's design probably no longer applies |
| #13 BlockWAND + ferrosearch API (+1151) | T2 (medium) | **T3 (likely obsolete)** | upstream now has block_wand_intersection.rs + block_wand_union.rs split — fork should evaluate dropping |
| 785884fd5 warning silence | (new) | **already-applied** | upstream `searcher.rs` already drops `Field` import — no-op PR |
| f57fd1a98 DateTime ns→us | (new) | **T3 (breaking)** | precision change is breaking; needs upstream design buy-in before PR |
| 7d26e2cd0 composite date_histogram after_key + 1d collapse | (new) | **T2** | composite collector area — upstream has heavy churn (rename + active-bucket memory fix), needs rebase + design alignment |
| 30c4be76c phrase_query ordered flag | (new) | **T1** | clean isolated additive feature |

## 5. Tier-1 PR submission strategy

**Order (lowest risk first):**

1. **#3 rewrite_schema_on_disk** — pure additive API, smallest blast radius
2. **#2 store reader poison recovery** — bug fix, well-contained
3. **#5–#7 FastFieldRangeQuery i64/u64** — additive type support + 2 bug fixes (EmptyScorer fallback)
4. **30c4be76c phrase_query ordered flag** — additive feature, semantics matches existing interval-query design
5. **#23 search_after cursor** — additive collector feature (largest scope of the five, last)

**For each PR, the workflow is:**

```
1. git worktree add ../tantivy-pr-N upstream/main
2. cd ../tantivy-pr-N && git switch -c pr/<topic>
3. git cherry-pick <sha>  # or rebase if multi-commit
4. cargo check --workspace
5. cargo test -p tantivy --lib <relevant_module>
6. write PR body in PRs/<topic>.md (this repo)
7. STOP — wait for user approval before push to fork & PR creation upstream
```

**No `git push` and no `gh pr create` will be executed without explicit user approval.** All five branches will be staged locally first; the user reviews PR bodies + diffs, then approves submission in batch or one-by-one.

## 6. Tier-2 / Tier-3 (deferred)

These need upstream design discussion (proposal issue) before code PR is reasonable:

- **GPU crate (#26 + #9 + #1)** — propose as optional workspace member with feature gate; quickwit may prefer to keep out of main repo
- **BlockWAND + ferrosearch API (#13)** + **MaxScoreBulkScorer (#11)** — investigate whether fork can drop these and adopt upstream's new BMW path; if not, propose what's missing
- **Batch fast-field API (#20, #22, #4)** + **DocStore zero-copy (#14, #17)** — additive but wide API surface, propose first
- **DateTime ns→us (f57fd1a98)** — breaking; upstream may reject; propose as opt-in precision type or major-version change
- **Performance bundle (#24, +943)** — must be split per-subsystem before any PR

Phase B will draft proposal-issue text for each, separate from this audit.

## 7. Working-tree state

`src/aggregation/bucket/term_agg.rs` has 1 uncommitted file (+48 lines): a regression test `terms_aggregation_no_missing_param_repro` for a FerroSearch-side multi-field bucket-count bug. Preserved (not stashed, not committed by this audit). This file is in upstream's heavy-churn area (+446 LOC since prev audit) — the test should be re-run after rebase to confirm it still reproduces or has been fixed by 73c711ec7 / d47abdf10.

## 8. Watch items for the next sync

- Upstream BMW rework is still landing (4 commits in 19 days). Fork's BlockWAND patch becomes harder to defend each week. Decide soon: drop or rewrite.
- Upstream agg refactor (rename `cached_sub_aggs → buffered_sub_aggs`) means fork's `aggregation/mod.rs` will conflict on next rebase.
- OpenSSF Scorecard workflow (a5d297c75) touches CI — fork's ci will need to either adopt or explicitly opt out.

---

**Audit owner:** orchestrator (2026-05-03 session). No source files modified. UPSTREAM_SYNC.md (2026-04-14) preserved as historical record.

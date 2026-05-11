---
branch: pr/search-after-cursor (worktree: ../tantivy-pr-search-after)
base: upstream/main
commits: 1 (e2c2837cb)
files: src/collector/mod.rs (+1/-1), src/collector/sort_key/order.rs (+13), src/collector/sort_key/sort_by_static_fast_value.rs (+48/-1), src/collector/sort_key/mod.rs (+135 tests)
status: built + tested locally; awaiting user approval to push to fork & open PR
---

# PR title

`feat(collector): search_after cursor support for fast-field TopDocs`

# PR body

## Summary

Adds `with_search_after(cursor, is_asc)` to `SortByStaticFastValue`. When
set, the segment collector skips documents whose fast-field value is at or
before the cursor during collection, using an O(1) column lookup per
document.

Also fixes `SegmentSortKeyComputerWithComparator` to delegate
`compute_sort_key_and_collect` to the inner computer, so custom overrides
(such as the cursor skip) are preserved through the `Order` wrapper.

## Motivation

`search_after` is the cursor-based pagination model used by Elasticsearch
and similar search engines. Instead of `(offset, limit)` — which still
costs O(offset) because the engine ranks and discards the skipped docs —
the caller passes the sort value of the last result on the previous page,
and the next page begins strictly after it. This is the canonical way to
deep-paginate large result sets without re-reading earlier pages.

Today, a tantivy caller wanting `search_after` semantics on a fast field
must intersect their base query with a `RangeQuery(field, cursor.., ..)`
(or its descending counterpart) wrapped in a `BooleanQuery`. That works
but pays posting-list intersection cost on every page. Doing the skip
inside the segment collector is a single column read per candidate doc and
adds nothing to the query plan or posting-list traversal.

## Semantics

- For ascending order, docs whose value is `<= cursor` are skipped.
- For descending order, docs whose value is `>= cursor` are skipped.
- Null values (docs that lack the fast field) are skipped in both
  directions, consistent with null-sorts-last behavior in the existing
  collectors.
- The cursor is supplied in the typed `T: FastValue` form
  (`u64` / `i64` / `f64` / `bool` / `DateTime`) and converted via
  `T::to_u64()` so the segment-level comparison stays in u64 space, the
  same monotonic representation `SortByStaticFastValue` already uses.

## Order wrapper fix

`SegmentSortKeyComputerWithComparator` (in `sort_key/order.rs`) now
delegates `compute_sort_key_and_collect` to its inner computer rather
than relying on the trait's default body. Without this, the cursor-skip
override is bypassed whenever the computer is wrapped in an `Order`
(which is the normal call path through `TopDocs::order_by`). The fix is
a three-line forward and applies to any future override of
`compute_sort_key_and_collect` as well.

## Tests

Three new tests in `src/collector/sort_key/mod.rs`:

1. `test_search_after_cursor_f64` — covers ascending and descending
   cursor skips on an `f64` fast field, including the boundary case
   where the cursor matches the lowest/highest existing value.
2. `test_search_after_cursor_u64` — covers the `u64` typed path on the
   `id` field.
3. `test_search_after_cursor_pagination` — walks the index page-by-page
   with `limit=1`, feeding each page's last value back as the next
   cursor, and asserts the full sorted traversal matches the
   non-paginated baseline. This is the end-to-end Elasticsearch-style
   `search_after` flow.

All pass on `cargo test --lib collector::sort_key::`.

## Caller contract

- The cursor's `is_asc` argument must match the `Order` passed to
  `order_by`. Mismatching them will produce wrong-direction skips. Could
  be made type-safer in a follow-up by deriving `is_asc` from the
  `Order` at `order_by` time, but that requires plumbing through the
  `SortKeyComputer` trait — left as a separate question for the
  maintainers.
- Tied sort values: the current cursor uses strict inequality
  (`<= cursor` / `>= cursor`), so a doc whose value exactly equals the
  cursor will be skipped. For pagination this is the desired behavior
  (the caller already returned that doc on the previous page); for
  filter-style use it may not be. Open to changing to strict `<`/`>` if
  the maintainers prefer.

## Open questions for the maintainers

- Naming: `with_search_after` mirrors Elasticsearch terminology. Happy
  to rename to `with_skip_past`, `after`, or whatever fits the codebase
  conventions better.
- Should the cursor live on `SortByStaticFastValue` (current placement)
  or on `TopDocs` itself, parallel to `and_offset`? The latter would
  generalize across all sort-key types but requires defining the cursor
  shape generically. Current placement is the minimal change.
- For multi-key sorts, only the primary key is currently cursor-aware.
  The full Elasticsearch semantics (lexicographic compare across all
  sort keys) would need plumbing through the composite `SortKeyComputer`
  tuple impls — happy to follow up in a separate PR if there's interest.

## Diff stat

```
src/collector/mod.rs                              |   2 +-
src/collector/sort_key/mod.rs                     | 135 +++++++++++++++++++
src/collector/sort_key/order.rs                   |  13 ++
src/collector/sort_key/sort_by_static_fast_value.rs |  49 +++++++-
4 files changed, 197 insertions(+), 2 deletions(-)
```

## Local verification

- `cargo check --lib`: ok
- `cargo test --lib collector::sort_key::test_search_after`: 3/3 pass
- `cargo test --lib collector::`: 89/89 pass, 0 regressions
- `cargo clippy --lib`: no new warnings (pre-existing
  `cardinality.rs` warning unrelated to this change)

# Submission checklist (before push)

- [ ] User reviews PR body + diff
- [ ] Decide commit author/email (currently inherited from cherry-pick
      origin — should be set to public email before push)
- [ ] `git push origin pr/search-after-cursor` to fork
- [ ] `gh pr create --repo quickwit-oss/tantivy --base main --head youichi-uda:pr/search-after-cursor --title "..." --body "..."`

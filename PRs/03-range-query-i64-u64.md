---
branch: pr/range-query-i64-u64 (worktree: ../tantivy-pr-frangequery)
base: upstream/main
commits: 1 (a0695708b)
files: src/query/range_query/range_query_fastfield.rs (+33 -7)
status: built + tested locally; awaiting user approval to push to fork & open PR

scope-note: The original assignment bundled three commits intended to add
i64/u64 support to a fork-only `FastFieldRangeQuery` API plus an
EmptyScorer fallback. The i64/u64 portion (commit `2e20c4d68`) modifies
`src/query/fast_field_range_weight.rs`, a file that does not exist on
upstream — it was introduced by `15c2ecbd0` ("block-max WAND scoring +
ferrosearch compatibility APIs"), a large fork-only feature. Trying to
cherry-pick the i64/u64 commit fails with a `modify/delete` conflict,
and there is no clean way to land it without first upstreaming the
~1,151-line block-max WAND prerequisite as its own PR.

The EmptyScorer fallback (commits `b86f4912a` + `76198083c`), however,
applies cleanly to upstream's existing `range_query_fastfield.rs`: the
`assert_eq!` panic in `FastFieldRangeWeight::scorer` is still present
on `upstream/main` (lines 67-73). This PR keeps that valuable, narrow
fix and explicitly defers the i64/u64 work to a future PR that bundles
the prerequisite struct.
---

# PR title

`fix(query): return EmptyScorer instead of panic on range/field type mismatch`

# PR body

## Summary

`FastFieldRangeWeight::scorer` (in `src/query/range_query/range_query_fastfield.rs`)
currently asserts that the query term's type matches the field's value
type and panics via `assert_eq!` when they differ. This PR turns that
panic into an early `return Ok(Box::new(EmptyScorer))`, matching the
behavior of the surrounding branches (JSON / IP / missing column) that
already return an empty scorer when no matching column is present.

## Motivation

Type mismatches between a range query term and the underlying fast
field are reachable from query layers that construct `Term`s before the
field's exact value type is known — for example, a query DSL parser
that defaults numeric bounds to `f64` and then issues the resulting
`RangeQuery` against a schema where the field happens to be `i64` or
`u64`. The current behavior is `unwind`-on-mismatch, which is sharp:
any external server hosting a tantivy index becomes susceptible to a
crash from a malformed but otherwise well-typed range request.

Returning an empty scorer is the semantically correct answer: a column
whose values are of one numeric type can never contain a document whose
value is of a different numeric type, so the result set is necessarily
empty. The same module already follows this pattern in several other
places (see `EmptyScorer` returns at lines 98, 110, 122, 149 etc.) when
the requested column is missing or of an incompatible kind.

## Change

Replace:

```rust
assert_eq!(
    term.typ(),
    field_type.value_type(),
    "Field is of type {:?}, but got term of type {:?}",
    field_type,
    term.typ()
);
```

with:

```rust
if term.typ() != field_type.value_type() {
    // Type mismatch: the caller built a range term with the wrong type
    // (e.g., F64 term on an I64 field). Return empty results — no
    // documents can match when the types are incompatible.
    return Ok(Box::new(crate::query::EmptyScorer));
}
```

No public API changes; no behavioral change for any well-typed query.

## Tests

One new regression test added in the existing
`src/query/range_query/range_query_fastfield.rs` `mod tests`:

`test_range_query_term_type_mismatch_returns_empty_scorer` — builds an
index with a single `i64` fast field, then constructs a `RangeQuery`
whose lower and upper bounds are `Term::from_field_f64(...)` (i.e.,
deliberately wrong type). The test asserts that
`FastFieldRangeWeight::scorer(...)` returns `Ok(_)` and the resulting
scorer is positioned at `TERMINATED` (i.e., empty), rather than
panicking.

The test fails on `upstream/main` with the original `assert_eq!` and
passes with this change.

## Open questions for the maintainers

- Should the type-mismatch path log a debug-level warning, or stay
  silent? I left it silent to match the surrounding `EmptyScorer`
  branches and to avoid introducing a `tracing` dependency at this
  call site.
- Would maintainers prefer to keep the `assert_eq!` behind a
  `#[cfg(debug_assertions)]` block (panic in dev, empty in release)?
  Happy to change either way.

## Diff stat

```
src/query/range_query/range_query_fastfield.rs | 40 +++++++++++++++++++++-----
1 file changed, 33 insertions(+), 7 deletions(-)
```

## Local verification

- `cargo check --lib`: ok
- `cargo test --lib query::range_query::`: 30/30 pass, 0 regressions
- `cargo test --lib query::range_query::range_query_fastfield::tests::test_range_query_term_type_mismatch_returns_empty_scorer`: 1/1 pass
- `cargo clippy --lib`: no new warnings (pre-existing
  `cardinality.rs` warning unrelated to this change)

# Submission checklist (before push)

- [ ] User reviews PR body + diff
- [ ] Confirm commit author/email is acceptable
  (`youichi-uda <1589222+youichi-uda@users.noreply.github.com>`)
- [ ] `git push origin pr/range-query-i64-u64` to fork
- [ ] `gh pr create --repo quickwit-oss/tantivy --base main --head youichi-uda:pr/range-query-i64-u64 --title "fix(query): return EmptyScorer instead of panic on range/field type mismatch" --body-file PRs/03-range-query-i64-u64.md`

# Note on the original i64/u64 scope

The assignment bundled in commit `2e20c4d68` ("FastFieldRangeQuery —
I64/U64 field type support"), which adds a `RangeFieldType` enum and a
type-aware `make_term` helper to `src/query/fast_field_range_weight.rs`.
That file is **fork-only**; it does not exist on `upstream/main`. It was
introduced by commit `15c2ecbd0` ("feat: block-max WAND scoring +
ferrosearch compatibility APIs"), which adds ~1,151 LOC across 22 files
including new BMW scorers, MaxScoreBulkScorer, top-k cache, and the
parallel `FastFieldRangeQuery` struct.

The i64/u64 patch can only land upstream after the
`FastFieldRangeQuery` struct itself is upstreamed. That should be its
own PR (or part of a block-max WAND PR) and is out of scope for this
narrow fix.

The EmptyScorer fallback is independently valuable on upstream's
existing `RangeQuery` path and is the entire content of this PR.

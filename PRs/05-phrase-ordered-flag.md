---
branch: pr/phrase-ordered-flag (worktree: ../tantivy-pr-phrase-ordered)
base: upstream/main
commits: 1 (210127381)
files: src/query/phrase_query/{mod.rs,phrase_query.rs,phrase_scorer.rs,phrase_weight.rs} (+483 / -15)
status: built + tested locally; awaiting user approval to push to fork & open PR
---

# PR title

`feat(query): PhraseQuery::set_ordered for in-order phrase matching`

# PR body

## Summary

Adds `PhraseQuery::set_ordered(bool)` and `is_ordered()`, and threads the
flag through `PhraseWeight` and `PhraseScorer`. When set, slop-based
phrase matching additionally requires the phrase terms to appear in the
order given to `PhraseQuery::new`, rejecting reversed occurrences even
if their distance is within `slop`.

## Motivation

Today, `PhraseQuery` has two modes:

- `slop = 0`: terms must be exactly adjacent and in document order.
- `slop > 0`: terms may appear in any order within the slop budget; a
  transposition of two adjacent terms costs 2. This is the documented
  symmetric-slop behaviour.

Some callers want a third mode: "terms must appear in the given order,
within `k` positions of each other". Today they have to compose
`Intersection` / multiple sub-queries or implement a custom scorer.
Tantivy already has all the bookkeeping for ordered intersection in
`compute_phrase_match` — the bounded position arithmetic just needs an
in-order branch.

This PR adds an explicit opt-in flag so the existing scorer can serve
that case without breaking the long-standing default.

## Behaviour

| `slop` | `ordered` | semantics |
|--------|-----------|-----------|
| 0      | false     | exact phrase, in document order (unchanged) |
| 0      | true      | exact phrase, in document order (no-op flag) |
| > 0    | false     | terms within slop budget, any order (unchanged) |
| > 0    | true      | terms within slop budget, in given order (new) |

`set_ordered(true)` only affects the slop path; the no-slop path
already enforces order, so the flag is documented as a no-op there.

## Implementation

- `intersection_count_with_slop_ordered` and
  `intersection_exists_with_slop_ordered` for the 2-term phrase paths.
  These mirror the existing slop variants but require
  `right_pos >= left_pos` in the shifted-position coordinate space.
- `intersection_count_with_carrying_slop_ordered` for the 3+ term path,
  mirroring `intersection_count_with_carrying_slop` with the same
  ordered constraint and saturating arithmetic on the slop accumulator
  to avoid overflow when the gap is undefined.
- `PostingsWithOffset::position_shift()` accessor: returns the offset
  used to normalise this term's positions into the common phrase
  coordinate space. The lowest-offset (earliest) phrase term has the
  highest shift; the highest-offset (latest) term has shift 0. This
  lets `compute_phrase_match` recover term order after
  `Intersection::new` reorders docsets by cost.
- Public `PhraseScorer::new_with_ordered` and crate-internal
  `new_with_offset_and_ordered` constructors so the flag can be
  threaded through without breaking the existing `PhraseScorer::new`
  signature. Existing constructors delegate with `ordered = false`.

## Tests

Six new tests in `src/query/phrase_query/mod.rs`:

1. `test_phrase_query_ordered_default_is_false` — `set_ordered` defaults
   to `false`; `"blue ocean"` with `slop = 2` matches both `"blue ocean"`
   and `"ocean blue"`.
2. `test_phrase_query_ordered_two_terms_rejects_reverse` — with
   `set_ordered(true)`, `"ocean blue"` no longer matches the
   `["blue", "ocean"]` query.
3. `test_phrase_query_ordered_three_terms_rejects_reverse` — exercises
   the carrying-slop path; `["a", "b", "c"]` with `slop = 2`,
   `ordered = true` matches `"a b c"` and `"a x b x c"` but rejects
   `"c b a"`, `"a c b"`, `"b a c"`.
4. `test_phrase_query_ordered_unordered_includes_reverse` — same corpus
   as (3) with `ordered = false` matches more docs (regression guard
   for default behaviour).
5. `test_phrase_query_ordered_respects_slop` — ordered queries still
   reject hits that exceed the slop budget.
6. `test_phrase_query_ordered_no_slop_is_noop` — with `slop = 0`,
   `ordered = true` and `ordered = false` produce identical results.

`cargo test --lib query::phrase_query::` — 31/31 pass (1 pre-existing
`#[ignore]`d test untouched), 0 regressions.

## Open questions for the maintainers

- Naming: `set_ordered` follows the existing `set_slop` pattern. Happy
  to rename to `set_in_order`, `set_strict_order`, etc. if the
  maintainers prefer.
- The `position_shift()` accessor on `PostingsWithOffset` is `pub`
  rather than `pub(crate)` because `PostingsWithOffset` itself is
  module-private; the visibility doesn't actually leak outside the
  crate. If preferred I can lower it to `pub(super)` or add an
  `#[allow(dead_code)]` for the unordered-slop path.
- `PhraseScorer::new_with_ordered` is added alongside the existing
  `PhraseScorer::new` rather than changing the latter's signature, to
  preserve the public API. If a single constructor is preferred I'm
  happy to consolidate.

## Diff stat

```
src/query/phrase_query/mod.rs           |  95 +++++++++
src/query/phrase_query/phrase_query.rs  |  28 +++
src/query/phrase_query/phrase_scorer.rs | 355 ++++++++++++++++++++++++++++++--
src/query/phrase_query/phrase_weight.rs |  10 +-
4 files changed, 483 insertions(+), 15 deletions(-)
```

## Local verification

- `cargo check --lib`: ok
- `cargo test --lib query::phrase_query::`: 31/31 pass, 0 regressions
- `cargo clippy --lib`: no new warnings (pre-existing
  `cardinality.rs` `into_iter` warning unrelated to this change)

# Submission checklist (before push)

- [ ] User reviews PR body + diff
- [ ] Decide commit author/email (currently
  `Youichi Uda <youichi.uda@gmail.com>`, set during cherry-pick)
- [ ] `git push origin pr/phrase-ordered-flag` to fork
- [ ] `gh pr create --repo quickwit-oss/tantivy --base main --head youichi-uda:pr/phrase-ordered-flag --title "..." --body "..."`

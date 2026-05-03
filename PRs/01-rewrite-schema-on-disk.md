---
branch: pr/rewrite-schema-on-disk (worktree: ../tantivy-pr-rewrite-schema)
base: upstream/main
commits: 1 (275c38de4)
files: src/index/index.rs (+146)
status: built + tested locally; awaiting user approval to push to fork & open PR
---

# PR title

`feat(index): Index::rewrite_schema_on_disk for additive schema evolution`

# PR body

## Summary

Adds `Index::rewrite_schema_on_disk(new_schema)`, which rewrites `meta.json`
in place with a new schema while preserving the segment list, opstamp,
settings, and payload.

This enables additive schema evolution: new fields can be appended to the
schema without rebuilding existing segments. Reads of the new field on
documents written before the rewrite return missing.

## Motivation

Search systems built on tantivy that need to support dynamic mapping (new
fields appearing as documents arrive) currently have no way to evolve the
schema without dropping and re-indexing. This patch provides a minimal
on-disk meta rewrite that handles the additive case — the most common one
in log/observability and ES-compatible workloads.

The method is small (~10 lines + docs) and reuses existing
`load_metas` / `save_metas` plumbing. No invariants of the segment files
themselves are touched, so existing readers continue to work.

## Caller contract

Documented on the method itself:

- The new schema **must be a superset** of the existing one (field names
  and types of pre-existing fields must be unchanged). Removing or
  re-typing fields will corrupt reads against existing segments.
- Any in-flight writes must be committed and the existing `IndexWriter`
  dropped before calling. The method does not acquire the writer lock.
- Re-open the directory with `Index::open` after a successful call so
  subsequent writers use the new schema.

## Tests

Two new tests in `src/index/index.rs`:

1. `test_rewrite_schema_on_disk_appends_new_field` — creates an index with
   a 2-field schema, writes a doc, drops the writer, calls
   `rewrite_schema_on_disk` with a 3-field schema, re-opens and asserts
   the schema has 3 fields and the existing fast field is still readable.
2. `test_rewrite_schema_on_disk_preserves_opstamp_and_segments` — asserts
   that opstamp and segment IDs are unchanged after the rewrite.

Both pass on `cargo test --lib index::index::tests::`.

## Open questions for the maintainers

- Naming: `rewrite_schema_on_disk` is verbose but unambiguous. Happy to
  rename to `evolve_schema`, `set_schema`, or whatever fits the codebase
  better.
- Should the method validate the superset invariant itself rather than
  documenting it as a caller contract? Doing so would require comparing
  schemas field by field; for now I followed the pattern of other
  `Index` methods that defer to caller correctness.
- Any preference on placing the test in `src/core/tests.rs` instead of an
  inline module in `index.rs`?

## Diff stat

```
src/index/index.rs | 146 +++++++++++++++++++++++++++++++++++++++++++++++++++++
1 file changed, 146 insertions(+)
```

## Local verification

- `cargo check --lib`: ok
- `cargo test --lib index::index::tests::`: 2/2 pass, 0 regressions
- `cargo clippy --lib`: no new warnings (pre-existing `cardinality.rs`
  warning unrelated to this change)

# Submission checklist (before push)

- [ ] User reviews PR body + diff
- [x] Author normalized to `Youichi Uda <youichi.uda@gmail.com>` (matches fork tip identity)
- [ ] `git push origin pr/rewrite-schema-on-disk` to fork
- [ ] `gh pr create --repo quickwit-oss/tantivy --base main --head youichi-uda:pr/rewrite-schema-on-disk --title "..." --body "..."`

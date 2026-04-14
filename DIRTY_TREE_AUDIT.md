# Dirty Tree Audit — 2026-04-14

Audit of 20 modified files found in the ferro-tantivy working tree at the
time of the upstream-sync audit commit (`8ebecbd93`). Performed read-only;
no source files were touched, no resets, no stashes.

## TL;DR

**All 20 files are pure `rustfmt` whitespace/import-order noise** produced by
running `cargo fmt` on the **stable** toolchain against a tree whose
committed formatting was produced under **nightly** `rustfmt`.

- Semantic changes: **0**
- API surface changes: **0**
- Logic changes: **0**
- Net diff: `+89 / −46` lines, entirely whitespace + import reorder
- `cargo fmt --check` on the dirty tree: **exit 0** (stable idempotent)
- `cargo check`: **exit 0** (only pre-existing `dead_code` warnings on
  `BlockWandScorer`, unrelated to this tree state)

**Recommendation: discard all 20 files** (`git checkout -- .`). They add
nothing but drift from the nightly-formatted canonical source.

## Root cause

`rustfmt.toml` at repo root requests options that are **nightly-only**:

```toml
comment_width = 120
format_strings = true
group_imports = "StdExternalCrate"
imports_granularity = "Module"
normalize_comments = true
where_single_line = true
wrap_comments = true
```

Current toolchain is stable `rustfmt 1.8.0-stable`, which silently warns
`unstable features are only available in nightly channel` and ignores
these options. An earlier agent/session ran `cargo fmt` under stable,
producing "legal stable formatting" that diverges from the nightly
output that is actually committed. Most visible side-effects:

| nightly option (committed) | stable output (dirty tree) |
|---|---|
| `where_single_line = true` → `where T: Bar {` on one line | expanded to 3 lines `where\n    T: Bar,\n{` |
| `group_imports = StdExternalCrate` → std / external / crate blocks | reordered: `use common::` moved below `use crate::` |
| `imports_granularity = Module` → one `use` per module, items alphabetized | `{BinaryDocumentDeserializer, extract_field_bytes_from_doc, ...}` reordered |

## File-by-file classification

All 20 files fall in category **(c) — existing-commit cosmetic duplicate /
nightly-vs-stable fmt drift**. Discard-safe.

| File | +/− | Nature |
|---|---|---|
| `columnar/src/block_accessor.rs` | +6 −2 | `where` clause expansion |
| `columnar/src/column/mod.rs` | +1 −1 | import reorder (`use common::` vs `use crate::`) |
| `columnar/src/column_index/optional_index/mod.rs` | +2 −1 | `where Self: 'b` expansion |
| `columnar/src/column_index/optional_index/set.rs` | +2 −1 | same |
| `columnar/src/column_index/optional_index/set_block/dense.rs` | +2 −1 | same |
| `columnar/src/column_index/optional_index/set_block/sparse.rs` | +2 −1 | same |
| `columnar/src/column_values/monotonic_mapping.rs` | +6 −3 | `where` expansion (3 impls) |
| `columnar/src/column_values/u64_based/bitpacked.rs` | +1 −4 | expression line-wrap collapse (`indexes.windows(2)…`) |
| `columnar/src/columnar/reader/mod.rs` | +3 −1 | `where FileSlice: From<F>` expansion |
| `columnar/src/columnar/writer/mod.rs` | +3 −1 | `where` expansion |
| `columnar/src/iterable.rs` | +2 −1 | `where` expansion |
| `common/src/writer.rs` | +3 −1 | `where Self: Sized` expansion |
| `src/collector/sort_key/sort_by_static_fast_value_with_cursor.rs` | +1 −1 | import alphabetical reorder |
| `src/core/searcher.rs` | +9 −7 | `use common::OwnedBytes` reorder + 3 chained-call re-wraps (semantically identical) |
| `src/query/boolean_query/block_wand_scorer.rs` | +17 −3 | 3 iter-chain re-wraps (`.iter().map().min().unwrap_or()`) — pure whitespace |
| `src/query/top_k_cache.rs` | +1 −1 | import reorder |
| `src/query/weight.rs` | +1 −5 | `fn collect_term_scorers` signature collapsed to 1 line |
| `src/schema/document/de.rs` | +20 −6 | `reader.read_exact(...).map_err(...)?` re-wrap + type_codes `|` match arm expansion |
| `src/schema/document/mod.rs` | +3 −1 | `pub(crate) use self::de::{…}` reorder (`extract_field_bytes_*` alphabetized) |
| `src/store/reader.rs` | +4 −4 | method chain rewrap |

Recent commits touching these hot files (`c1bc96c09`, `5f0532037`,
`15c2ecbd0`) are the canonical nightly-formatted versions; the working
tree is a **downgrade** of that formatting, not a precursor.

## FerroSearch API impact

Zero. None of the changes touch any public item signature. Specifically
the ferrosearch-consumed API surface is **untouched**:

- `Index::rewrite_schema_on_disk` — not in this diff
- `StoreReader::batch_get_field_owned_bytes_grouped` — only the
  **call-site** indentation inside `Searcher` changes, not the signature
- `BlockWandScorer::{new, advance, seek}` — whitespace only
- `schema::document::de::*` re-exports — same identifiers, reordered
- `Weight::collect_term_scorers` — same signature, collapsed to one line

`vendor/tantivy-local` symlink consumers (ferro-index, ferro-query) will
see identical compiled output.

## Build status (dirty tree)

```
cargo check              → exit 0  (pre-existing dead_code warnings only)
cargo fmt --check        → exit 0  (stable rustfmt idempotent on dirty tree)
```

## Recommendation

1. **Discard all 20 files.** `git checkout -- .` in tantivy root.
   Rationale: the committed nightly-formatted version is canonical and
   required by `rustfmt.toml`; the stable reformat loses intent.
2. **Do not commit** any subset — there is no subset worth keeping.
3. **Investigate: none.**
4. **Process fix (follow-up, not in this audit):** either
   (a) pin `rust-toolchain.toml` to `nightly` for `cargo fmt` so stable
   users are blocked from reformatting, or
   (b) move the nightly-only options behind a comment block and adopt
   stable-only `rustfmt.toml` settings so both toolchains converge.
   This prevents a future agent from re-creating the same dirty tree.

## Provenance hypothesis

Unknown which agent produced this. Plausible candidates:
- A recent session that ran `cargo fmt` without noticing the
  nightly-only warnings (easy to miss — they are buffered at the top of
  stderr).
- The mac-migration session (see memory `project_session_handoff_mac.md`)
  where toolchain defaults may have changed from nightly to stable.

No evidence of in-progress SQCache / BlockWAND patch work in these
specific hunks — all of the perf-hot-path files (`block_wand_scorer.rs`,
`bitpacked.rs`, `block_accessor.rs`, `monotonic_mapping.rs`) show only
whitespace churn.

---
branch: pr/store-poison-recovery (worktree: ../tantivy-pr-store-poison)
base: upstream/main
commits: 1 (9fbac7cb2)
files: src/store/reader.rs (+55 / -4)
status: built + tested locally; awaiting user approval to push to fork & open PR
---

# PR title

`fix(store): recover from poisoned block cache mutex`

# PR body

## Summary

Replace `cache.lock().unwrap()` on `BlockCache`'s LRU mutex with a small
`Self::lock_cache(...)` helper that does
`lock().unwrap_or_else(|poisoned| poisoned.into_inner())`, so a panic in
one thread does not turn into a permanent panic for every subsequent
reader sharing the `StoreReader`.

## Motivation

`BlockCache` wraps an `LruCache<usize, Block>` in a `Mutex`. Today every
access — `get_from_cache`, `put_into_cache`, `len`, `peek_lru` — locks
that mutex with `.unwrap()`. If any thread panics while holding the lock
(e.g. an allocator failure under memory pressure, a panic in user code
hidden behind a callback, or any future change that introduces a panic
inside the critical section), the mutex becomes poisoned and **every
subsequent reader on the shared `StoreReader` panics on `PoisonError`**.

Because `StoreReader` is typically wrapped in an `Arc` and shared across
worker threads, this turns a single transient failure into a permanent
loss of the doc-store cache (and of every reader holding the reference).
The cache itself only stores decompressed block bytes — the panicking
thread cannot have left it in a logically inconsistent state — so
recovering the guard via `PoisonError::into_inner()` is safe and
strictly better than the current behaviour.

## Fix

A 5-line helper centralises the recovery:

```rust
fn lock_cache<'a>(
    cache: &'a Mutex<LruCache<usize, Block>>,
) -> std::sync::MutexGuard<'a, LruCache<usize, Block>> {
    cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
```

All four call sites (`get_from_cache`, `put_into_cache`, `len`,
`peek_lru`) are routed through it.

## Test

`test_block_cache_recovers_from_poisoned_mutex` (in
`src/store/reader.rs`):

1. Builds a `BlockCache` with a small `LruCache` backed by a `Mutex`.
2. Seeds one entry.
3. Spawns a worker thread that locks the mutex and `panic!`s while
   holding the guard, deliberately poisoning the mutex.
4. Joins the worker (asserts `is_err()`) and asserts
   `mutex.is_poisoned()` is `true`.
5. Calls `len()`, `get_from_cache(42)`, `put_into_cache(7, ...)` and
   `stats()` and asserts they all return the expected results — i.e. no
   panic, and the previously-cached entry is still accessible.

The test passes with `cargo test --lib store::reader::tests::test_block_cache_recovers_from_poisoned_mutex`.

## Diff stat

```
src/store/reader.rs | 59 +++++++++++++++++++++++++++++++++++++++++++++++++----
1 file changed, 55 insertions(+), 4 deletions(-)
```

## Local verification

- `cargo check --lib`: ok
- `cargo test --lib store::reader::tests::`: 3/3 pass, 0 regressions
  (the new test plus the two pre-existing `test_block_cache_*` tests)
- `cargo clippy --lib`: no new warnings (the one pre-existing warning
  in `cardinality.rs` is unrelated to this change)

## Open questions for the maintainers

- The helper is on `BlockCache` rather than free-standing because all
  callers are inside `impl BlockCache`. Happy to inline it back at each
  site if you prefer not to add a method.
- The test triggers a real panic across a thread boundary, so it prints
  a panic trace in the test output by default. Let me know if you'd
  rather it use `std::panic::set_hook` to silence the trace during the
  test.

# Submission checklist (before push)

- [ ] User reviews PR body + diff
- [x] Author normalized to `Youichi Uda <youichi.uda@gmail.com>` (matches fork tip identity)
- [ ] `git push origin pr/store-poison-recovery` to fork
- [ ] `gh pr create --repo quickwit-oss/tantivy --base main --head youichi-uda:pr/store-poison-recovery --title "..." --body "..."`

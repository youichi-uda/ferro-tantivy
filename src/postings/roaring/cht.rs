//! Phase 2 D-1 — Compressed Hot Tier (CHT) cache, host-memory v1.
//!
//! See `docs/phase2_d_cht_design.md` for the full architecture.
//! Short version: cache `(segment_id, field, term) →
//! Arc<RoaringPostings>` so steady-state warm queries skip
//! `drain_block_segment_to_roaring` (the ~700 ms / 10 M-doc
//! bottleneck identified by Wave 8 / C re-bench).
//!
//! ## v1 scope
//!
//! - **Host memory** only — the cached value is `Arc<RoaringPostings>`,
//!   cloned by `Arc::clone` on hit. No VRAM. v2/v3 (VRAM-resident
//!   uncompressed / Bitcomp-compressed) are deferred.
//! - **LRU eviction** under a configurable byte budget. Default
//!   2 GiB; production operators can raise via the
//!   `ferrosearch --cht-budget-bytes` flag (Wave 8 / D / 1 / 3 wiring).
//! - **Process-global** `OnceLock<Cht>` — same lifetime as
//!   `GPU_RESOURCES` in `gpu_dispatch.rs`. Multi-tenant per-index
//!   caches are Wave 12 scope.
//! - **No invalidation hook** — stale entries (segments deleted /
//!   merged away) just LRU-evict naturally. Different `segment_id`
//!   = different key, so stale entries never produce wrong results
//!   (only waste budget).
//!
//! ## Wave 11 — content-stable keys
//!
//! As of Wave 11 the key is `(segment_id, field_id, term_hash)` where
//! `term_hash` is a deterministic 64-bit `FxHash` of the term's serialized
//! value bytes. This replaces the v1 pragmatic scheme that keyed on the
//! `BlockSegmentPostings` data slice address (which was ASLR-randomised
//! across process restarts and so unusable as a persistence key). Keys
//! are now stable across process restarts, unlocking D-5 warm-restart
//! persistence.
//!
//! ## Lookup pattern
//!
//! ```ignore
//! let key = ChtKey {
//!     segment_id,
//!     field: term.field().field_id(),
//!     term_hash: hash_term_bytes(term.serialized_value_bytes()),
//! };
//! let roaring = match cht().get(&key) {
//!     Some(cached) => cached,                              // hit — skip drain
//!     None => {
//!         let drained = Arc::new(drain_block_segment_to_roaring(&mut cursor));
//!         cht().insert(key, drained.clone());              // populate cache
//!         drained
//!     }
//! };
//! ```
//!
//! Insertion is fire-and-forget: `insert` may evict the entry it
//! just inserted if the new entry already exceeds the budget alone
//! (cohort too large to ever cache); the helper handles this
//! gracefully so callers don't need to special-case.
//!
//! ## Observability
//!
//! `Cht::stats()` returns `ChtStats { hits, misses, inserts,
//! evictions, current_bytes, budget_bytes }`. Counters are
//! `AtomicU64` so the stats endpoint can read without locking the
//! LRU. The `try_gpu_intersect` test harness uses these as the
//! warm-cache-hit witness analogous to `gpu_dispatch_count`.

#![cfg(all(feature = "gpu", feature = "ferro-compress"))]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::index::SegmentId;
use crate::postings::roaring::container::Container;
use crate::postings::roaring::encoder::RoaringPostings;

/// Estimated host-memory footprint of a [`RoaringPostings`] (bytes).
///
/// Per-container estimates:
/// - `Bitmap`: 8192 bytes (2048 u32 words, fixed).
/// - `Array`: `cardinality × 2` + 16 bytes Vec overhead.
/// - `Run`: `num_runs × 4` (u16 start + u16 length pair) + 16 bytes
///   Vec overhead. We approximate `num_runs` as `cardinality / 4`
///   for sizing purposes; this is a heuristic, not a tight bound.
///
/// The whole `RoaringPostings` adds 24 bytes for the outer `Vec`
/// header plus 8 bytes per `(high16, Container)` pair for the
/// enum tag + alignment. Used as the LRU budget accounting unit.
#[must_use]
pub fn estimated_bytes(rp: &RoaringPostings) -> usize {
    let mut total: usize = 24; // outer Vec header
    for (_, container) in &rp.containers {
        total += 8; // (u16, enum tag) per pair
        total += match container {
            Container::Bitmap(_) => 8192,
            Container::Array(a) => (a.cardinality() as usize) * 2 + 16,
            Container::Run(r) => (r.cardinality() as usize / 4) * 4 + 16,
        };
    }
    total
}

/// Cache key for a single posting list in a segment.
///
/// **Wave 11 — content-stable design**: keyed on
/// `(segment_id, field_id, term_hash)`, where `term_hash` is the
/// deterministic 64-bit `FxHash` of `Term::serialized_value_bytes()`.
/// This replaces the v1 pragmatic scheme that keyed on the
/// `BlockSegmentPostings` data slice's (address, length) pair (an
/// ASLR-randomised pointer that could not be persisted).
///
/// Properties:
/// - **Unique** within a segment: two different `(field, term)` tuples
///   in the same segment hash to different `(field_id, term_hash)`
///   pairs (collision probability is `2⁻⁶⁴` per field, negligible at
///   any practical cardinality).
/// - **Stable** across queries and **across process restarts**:
///   `FxHasher` is deterministic (no random seed), `field_id` is a
///   schema-stable `u32`, and `segment_id` is the on-disk UUID. A
///   re-opened index produces byte-equal keys for the same term, so
///   D-5 dump/load preserves cache hits across restart.
/// - **Survives merges that produce a new segment**: new `segment_id`
///   = new key, old entries evict naturally. Same segment re-opened =
///   same key, entry survives the re-open.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ChtKey {
    /// Tantivy segment UUID (stable per segment file set, persisted
    /// in `meta.json`).
    pub segment_id: SegmentId,
    /// Field id (`Field::field_id()`) — stable per schema.
    pub field: u32,
    /// 64-bit `FxHash` of `Term::serialized_value_bytes()`. Stable
    /// across process restarts (no random seed). See
    /// [`hash_term_bytes`].
    pub term_hash: u64,
}

/// Deterministic 64-bit hash of a term value's serialized bytes. Used
/// to build the [`ChtKey::term_hash`] component. `FxHasher` is the
/// content-stable choice: no random seed, no version drift inside a
/// `rustc-hash` crate version. The hash is **not** cryptographic; we
/// rely on the 64-bit space to keep collision probability negligible
/// for the cohort sizes the planner admits (≤ 32 terms per AND).
///
/// The hash function is part of the persistence contract for D-5 — if
/// it changes, the on-disk wire format version must bump in lockstep
/// (`FCV1`/`FCV2`/`FCV3` magics in the dump headers).
#[inline]
#[must_use]
pub fn hash_term_bytes(term_value_bytes: &[u8]) -> u64 {
    use std::hash::Hasher;
    let mut hasher = rustc_hash::FxHasher::default();
    hasher.write(term_value_bytes);
    hasher.finish()
}

/// Observable counters for the CHT. All `Relaxed` — cheap to read
/// without coordinating with the LRU mutex.
#[derive(Debug, Clone, Copy)]
pub struct ChtStats {
    /// Cumulative cache hits since process start (or last [`Cht::reset`]).
    pub hits: u64,
    /// Cumulative cache misses.
    pub misses: u64,
    /// Cumulative successful insertions (includes those that evicted
    /// older entries; excludes silent drops of oversized values and
    /// LRU touches on duplicate inserts).
    pub inserts: u64,
    /// Cumulative LRU evictions caused by budget pressure.
    pub evictions: u64,
    /// Sum of [`estimated_bytes`] for all currently-cached entries.
    pub current_bytes: u64,
    /// Configured budget ceiling (insertions are skipped if a single
    /// value exceeds this; eviction kicks in to keep total ≤ budget).
    pub budget_bytes: u64,
    /// Number of cached entries (= cardinality of the LRU map).
    pub entries: u64,
}

/// Single CHT instance. Construct via [`Cht::with_budget`] for
/// custom budgets; access the process-global instance via
/// [`global`] / [`global_with_budget`].
#[allow(missing_docs)]
pub struct Cht {
    inner: Mutex<ChtInner>,
    budget_bytes: u64,
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    evictions: AtomicU64,
}

/// Internal LRU state. Held under a single `Mutex` because hit-rate
/// reordering and budget-aware insertion must be atomic relative to
/// each other. The lock is held for the duration of get/insert; in
/// practice these are O(log n) so the critical section is brief.
struct ChtInner {
    /// `(key → (entry, lru_token))`. The LRU token is a monotonic
    /// counter incremented on every access; entries are evicted in
    /// ascending token order. Simpler than a doubly-linked-list LRU
    /// at the cost of one `HashMap` scan on eviction.
    map: HashMap<ChtKey, (Arc<RoaringPostings>, u64)>,
    next_token: u64,
    current_bytes: u64,
}

impl Cht {
    /// Construct a fresh cache with the given byte budget.
    #[must_use]
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(ChtInner {
                map: HashMap::new(),
                next_token: 0,
                current_bytes: 0,
            }),
            budget_bytes,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Look up a cached posting list. On hit, bumps the LRU token
    /// so this entry is the freshest. Increments `hits` or `misses`
    /// counter accordingly.
    pub fn get(&self, key: &ChtKey) -> Option<Arc<RoaringPostings>> {
        let mut inner = self.inner.lock().expect("CHT mutex poisoned");
        // Compute the next token before re-borrowing `inner.map` so
        // the borrow checker can split the disjoint-field write.
        let next_token = inner.next_token + 1;
        let cloned = if let Some((entry, token)) = inner.map.get_mut(key) {
            *token = next_token;
            Some(Arc::clone(entry))
        } else {
            None
        };
        if cloned.is_some() {
            inner.next_token = next_token;
        }
        drop(inner);
        if cloned.is_some() {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        cloned
    }

    /// Insert a posting list under `key`. May evict older entries
    /// to fit the budget. If `value` alone exceeds the budget, the
    /// insertion is skipped (the cache stays consistent; this query
    /// just doesn't benefit from caching). No-op if `key` already
    /// has a cached entry — the existing entry's LRU token is
    /// bumped instead, so re-inserts of the same posting list don't
    /// thrash the LRU.
    pub fn insert(&self, key: ChtKey, value: Arc<RoaringPostings>) {
        let entry_bytes = estimated_bytes(&value) as u64;
        if entry_bytes > self.budget_bytes {
            // Cohort larger than the entire budget — never cacheable.
            // No state change.
            return;
        }
        let mut inner = self.inner.lock().expect("CHT mutex poisoned");
        // Already cached? Bump LRU token, return early. Pre-compute
        // the next token before re-borrowing `inner.map` so the
        // borrow checker can split the disjoint-field write.
        let next_token = inner.next_token + 1;
        let already_cached = inner.map.get_mut(&key).map(|(_, token)| {
            *token = next_token;
        });
        if already_cached.is_some() {
            inner.next_token = next_token;
            return;
        }
        // Evict to fit.
        while inner.current_bytes + entry_bytes > self.budget_bytes
            && !inner.map.is_empty()
        {
            let oldest_key = inner
                .map
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                if let Some((evicted_value, _)) = inner.map.remove(&k) {
                    let evicted_bytes = estimated_bytes(&evicted_value) as u64;
                    inner.current_bytes = inner.current_bytes.saturating_sub(evicted_bytes);
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }
        // Insert.
        inner.next_token += 1;
        let token = inner.next_token;
        inner.current_bytes += entry_bytes;
        inner.map.insert(key, (value, token));
        self.inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Snapshot of current cache stats. Cheap (atomic loads + one
    /// mutex acquire for the entry count).
    #[must_use]
    pub fn stats(&self) -> ChtStats {
        let (current_bytes, entries) = {
            let inner = self.inner.lock().expect("CHT mutex poisoned");
            (inner.current_bytes, inner.map.len() as u64)
        };
        ChtStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            current_bytes,
            budget_bytes: self.budget_bytes,
            entries,
        }
    }

    /// Reset all state — counters AND entries. Test-only.
    #[doc(hidden)]
    pub fn reset(&self) {
        let mut inner = self.inner.lock().expect("CHT mutex poisoned");
        inner.map.clear();
        inner.next_token = 0;
        inner.current_bytes = 0;
        drop(inner);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.inserts.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }
}

// ============================================================
// Process-global cache
// ============================================================

/// Default budget when no explicit configuration is provided.
/// 2 GiB host RAM — large enough to cover a small-to-mid index's
/// hot terms on commodity hardware. Operators tune via
/// `ferrosearch --cht-budget-bytes` (D-1 / step 3 wiring).
pub const DEFAULT_CHT_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

static GLOBAL_CHT: OnceLock<Cht> = OnceLock::new();

/// Get the process-global CHT, initialising it with the default
/// budget on first access. Subsequent calls return the same handle.
pub fn global() -> &'static Cht {
    GLOBAL_CHT.get_or_init(|| Cht::with_budget(DEFAULT_CHT_BUDGET_BYTES))
}

/// Get the process-global CHT, initialising it with a custom budget
/// **iff this is the first access**. Returns `None` if the global
/// is already initialised (the budget is fixed at first init; the
/// caller should arrange for this to run before any query path
/// touches the cache, e.g. from `ferrosearch::main` startup).
///
/// Returns `Some(&Cht)` on successful first-init, `None` if
/// already initialised.
pub fn global_with_budget(budget_bytes: u64) -> Option<&'static Cht> {
    if GLOBAL_CHT.get().is_some() {
        return None;
    }
    let _ = GLOBAL_CHT.set(Cht::with_budget(budget_bytes));
    GLOBAL_CHT.get()
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::roaring::encoder::RoaringEncoder;

    fn small_postings() -> RoaringPostings {
        RoaringEncoder::from_doc_ids(&[1, 2, 3, 100, 65540])
    }

    fn big_postings(seed: u32) -> RoaringPostings {
        // Large enough to exercise multiple Bitmap containers.
        let docs: Vec<u32> = (0..200_000).map(|i| seed.wrapping_add(i * 3)).collect();
        RoaringEncoder::from_doc_ids(&docs)
    }

    fn dummy_key(field: u32, term_hash: u64) -> ChtKey {
        ChtKey {
            // SegmentId::generate_random() exists and is the
            // intended way to construct a fresh ID for tests.
            segment_id: SegmentId::generate_random(),
            field,
            term_hash,
        }
    }

    #[test]
    fn miss_then_hit() {
        let cht = Cht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xdead, 100);
        assert!(cht.get(&key).is_none());
        let stats = cht.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        cht.insert(key.clone(), Arc::new(small_postings()));
        let got = cht.get(&key);
        assert!(got.is_some());
        let stats = cht.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.inserts, 1);
    }

    #[test]
    fn duplicate_insert_does_not_double_count() {
        let cht = Cht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xdead, 100);
        let value = Arc::new(small_postings());
        cht.insert(key.clone(), Arc::clone(&value));
        let bytes_after_first = cht.stats().current_bytes;
        cht.insert(key.clone(), Arc::clone(&value));
        let bytes_after_second = cht.stats().current_bytes;
        assert_eq!(
            bytes_after_first, bytes_after_second,
            "re-insert of same key should not double-count bytes"
        );
        // Re-insert bumps token but does NOT re-bump the inserts
        // counter (it's an LRU touch, not a real insert).
        assert_eq!(cht.stats().inserts, 1);
    }

    #[test]
    fn lru_evicts_oldest_first() {
        // Budget tight enough to fit only one big posting list.
        let one_big_bytes = estimated_bytes(&big_postings(0)) as u64;
        let cht = Cht::with_budget(one_big_bytes + 512);

        let k1 = dummy_key(0xa1f7, 100);
        let k2 = dummy_key(0xb273, 100);
        cht.insert(k1.clone(), Arc::new(big_postings(0)));
        cht.insert(k2.clone(), Arc::new(big_postings(1)));

        // k1 should have been evicted (k2 inserted after).
        assert!(cht.get(&k1).is_none(), "oldest key must have been evicted");
        assert!(cht.get(&k2).is_some(), "newest key must still be cached");
        let stats = cht.stats();
        assert!(stats.evictions >= 1);
    }

    #[test]
    fn entry_larger_than_budget_is_dropped() {
        let cht = Cht::with_budget(1024); // 1 KiB — much smaller than big_postings
        let key = dummy_key(0xc0ff, 100);
        cht.insert(key.clone(), Arc::new(big_postings(7)));
        assert!(
            cht.get(&key).is_none(),
            "value larger than budget must not be cached"
        );
        // No eviction should have happened — the insert short-circuited.
        let stats = cht.stats();
        assert_eq!(stats.inserts, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.current_bytes, 0);
    }

    #[test]
    fn lru_touch_protects_entry_from_eviction() {
        // Two entries fit, third evicts the OLDEST. We touch the
        // oldest so the newer one is evicted instead.
        let one_big_bytes = estimated_bytes(&big_postings(0)) as u64;
        let cht = Cht::with_budget(one_big_bytes * 2 + 512);

        let k_old = dummy_key(0x111, 100);
        let k_mid = dummy_key(0x222, 100);
        let k_new = dummy_key(0x333, 100);

        cht.insert(k_old.clone(), Arc::new(big_postings(0)));
        cht.insert(k_mid.clone(), Arc::new(big_postings(1)));
        // Touch `k_old` so it's now the freshest, NOT the oldest.
        let _ = cht.get(&k_old);
        // Insert `k_new` — should evict `k_mid` (oldest after touch).
        cht.insert(k_new.clone(), Arc::new(big_postings(2)));

        assert!(cht.get(&k_old).is_some(), "touched entry survives");
        assert!(cht.get(&k_mid).is_none(), "untouched middle entry evicted");
        assert!(cht.get(&k_new).is_some(), "newest entry survives");
    }

    #[test]
    fn estimated_bytes_bitmap_is_8192() {
        // Single Bitmap container = 8 KiB content + small overhead.
        // Use a high-cardinality run to force Bitmap form.
        let docs: Vec<u32> = (0..10_000).collect();
        let rp = RoaringEncoder::from_doc_ids(&docs);
        let est = estimated_bytes(&rp);
        // At least one Bitmap container worth of bytes.
        assert!(est >= 8192, "estimated bytes for Bitmap-class should be ≥ 8192, got {est}");
    }

    #[test]
    fn chtkey_hash_stable_across_construction() {
        // Wave 11 acceptance: same term bytes must hash to the same
        // ChtKey across independent construction sites. This is the
        // load-bearing invariant for D-5 warm-restart persistence —
        // an on-disk dump key must equal the key constructed at first
        // query post-restart.
        let bytes_a: &[u8] = &[0x10, 0x00, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
        let bytes_b: &[u8] = &[0x10, 0x00, 0x00, 0x00, b'h', b'e', b'l', b'l', b'o'];
        let bytes_c: &[u8] = &[0x10, 0x00, 0x00, 0x00, b'w', b'o', b'r', b'l', b'd'];
        let h_a = hash_term_bytes(bytes_a);
        let h_b = hash_term_bytes(bytes_b);
        let h_c = hash_term_bytes(bytes_c);
        assert_eq!(
            h_a, h_b,
            "identical term bytes must hash to the same term_hash \
             (FxHasher determinism — load-bearing for D-5 persistence)"
        );
        assert_ne!(
            h_a, h_c,
            "distinct term bytes must hash to distinct term_hash \
             (probabilistic, but pinned for these fixtures)"
        );
        // Empty value bytes must produce a well-defined, finite hash
        // (not a panic / UB), since some term kinds have zero-length
        // value bytes (e.g. a sentinel empty-string token). We assert
        // determinism — every call returns the same value — but do
        // not pin the literal value (FxHasher's empty-input state is
        // an implementation detail; D-5 wire format versions the hash
        // function explicitly).
        let h_empty_1 = hash_term_bytes(b"");
        let h_empty_2 = hash_term_bytes(b"");
        assert_eq!(
            h_empty_1, h_empty_2,
            "empty-input hash must be deterministic across calls"
        );
    }

    #[test]
    fn global_with_budget_respects_first_init_only() {
        // Note: process-global state — these assertions only hold if
        // the global hasn't been initialised by an earlier test in
        // the same process. To keep test independence we don't
        // assert the budget value, only the first-init semantics.
        let _first_attempt = global_with_budget(8 * 1024 * 1024);
        let second_attempt = global_with_budget(16 * 1024 * 1024);
        assert!(
            second_attempt.is_none(),
            "second global_with_budget call must return None — budget is fixed at first init"
        );
    }
}

//! Phase 2 D-3 — VRAM-resident Compressed Hot Tier (CHT v2).
//!
//! See `docs/phase2_d_cht_design.md` § 3 for the architectural
//! background. Short version: D-1 (host-memory CHT v1) eliminated the
//! per-query drain step (~700 ms / 10M-doc cohort); D-3 v2 takes the
//! next step by caching each term's posting list **already
//! pre-expanded to BitmapContainer form on the GPU device**, so warm
//! queries skip both `flat_buffer_for_term` (CPU expansion) and the
//! H→D `cudaMemcpyAsync` of that flat layout.
//!
//! ## v2 scope
//!
//! - **Device-memory cache** keyed on the same `ChtKey` as host CHT v1
//!   (`(segment_id, field_id, term_hash)` as of Wave 11 — content-stable
//!   across process restarts; see `cht::ChtKey` for the migration
//!   rationale). Stable per segment file set; survives merges that
//!   produce a new segment ID; survives process restart at the key
//!   level — restart-roundtrip persistence ships in D-5.
//! - **Per-term VRAM layout**: each cached entry stores all
//!   non-empty buckets of THIS term, expanded to the kernel's expected
//!   `[u32; BITMAP_CONTAINER_WORDS]`-per-bucket form, stacked
//!   contiguously in a single `cudaMalloc` allocation. A
//!   `Vec<(u16, u32)>` bucket_index records each bucket's `high16`
//!   and word-offset within the device buffer.
//! - **LRU eviction** under a configurable VRAM byte budget. Default
//!   is 32 GiB on L40S (= 48 GiB device - 16 GiB compute + scratch
//!   reserve). Operators tune via `ferrosearch --cht-vram-budget-bytes`.
//! - **Process-global** `OnceLock<VramCht>`, same lifetime model as
//!   the v1 host `Cht`.
//! - **Coexists with v1 host CHT**: a term may live in either, both,
//!   or neither cache. The dispatch layer
//!   ([`super::gpu_intersect::try_gpu_intersect`]) prefers v2 when all
//!   cohort members have a v2 hit; falls back to v1 host hit + flat
//!   buffer + H→D when any cohort member is v2-miss.
//!
//! ## What v2 does NOT do (deferred to D-4 / D-5)
//!
//! - **Bitcomp compression** (D-4): cached entries are uncompressed
//!   bitmap form, so each non-empty bucket consumes 8 KiB device
//!   memory. D-4 wraps the flat buffer in `ferro_compress` Bitcomp
//!   for ~4× capacity multiplier at the cost of a per-query decomp
//!   step.
//! - **Warm-restart** (D-5): the cache is process-local. Restart =
//!   cold cache.
//! - **Per-tenant isolation** (Wave 12): the cache is process-global,
//!   so multiple ferrosearch indexes share the budget.
//!
//! ## Empirical context
//!
//! Wave 8.D-1 measurement (commit `05504ad`) showed that with v1
//! host-memory cache eliminating the drain step, the residual
//! `try_gpu_intersect` walltime is dominated by upstream HTTP /
//! Searcher / BooleanWeight / Count orchestration (~99 % of 110 ms
//! at 10M doc scale). The H→D + flat_buffer step is ~1 % residual at
//! that scale. v2 therefore delivers diminishing absolute returns at
//! 10M scale BUT amplifies as cohort scale grows: at 50-term cohorts
//! × 100-bucket terms × 8 KiB per bucket = ~40 MB H→D per term × 50
//! terms = ~2 GB H→D / query, vs ~50 MB DtD / query for v2 (~24×
//! lower transfer volume). Combined with L40S 48 GiB VRAM (vs L4
//! 24 GiB), v2 enables a working-set scale that wasn't reachable
//! before.
//!
//! ## Lookup pattern
//!
//! ```ignore
//! // Inside try_gpu_intersect, per cohort (Term, scorer) pair:
//! let key = ChtKey {
//!     segment_id,
//!     field: term.field().field_id(),
//!     term_hash: cht::hash_term_bytes(term.serialized_value_bytes()),
//! };
//! let vram_entry = vram_cht().get(&key);
//! match vram_entry {
//!     Some(entry) => vram_entries.push(entry),  // v2 fast path
//!     None => {
//!         // v1 path: get-or-drain Arc<RoaringPostings>, then
//!         // opportunistically promote to v2 for next query.
//!         let roaring = host_cht_get_or_drain(&key);
//!         let _ = vram_cht().try_promote(&key, &roaring);
//!         host_entries.push(roaring);
//!     }
//! }
//! ```

#![cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ferro_compress::nvcomp_sys::cuda::{
    cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind, CUDA_SUCCESS,
};

use crate::index::SegmentId;
use crate::postings::roaring::cht::ChtKey;
use crate::postings::roaring::container::{BitmapContainer, Container};
use crate::postings::roaring::encoder::RoaringPostings;
use crate::postings::roaring::BITMAP_CONTAINER_WORDS;

/// Errors specific to the VRAM cache layer. Insert / promote failures
/// at the CUDA level are surfaced here so callers can distinguish them
/// from in-memory bookkeeping issues. Construction is infallible
/// (it requires no device call); `get` cannot fail. The error variants
/// cover the device-memory paths.
#[derive(Debug, thiserror::Error)]
pub enum VramChtError {
    /// `cudaMalloc` returned non-success.
    #[error("cudaMalloc failed (bytes={bytes}, code={code})")]
    Malloc {
        /// Bytes the allocation requested.
        bytes: usize,
        /// Raw `cudaError_t` value.
        code: u32,
    },
    /// `cudaMemcpy` host-to-device returned non-success.
    #[error("cudaMemcpy(H2D) failed (bytes={bytes}, code={code})")]
    Memcpy {
        /// Bytes the copy attempted.
        bytes: usize,
        /// Raw `cudaError_t` value.
        code: u32,
    },
}

/// One cached posting list, stored on the CUDA device.
///
/// Layout: `d_buckets` points to a contiguous `[u32; total_words]` device
/// allocation, where `total_words = bucket_count * BITMAP_CONTAINER_WORDS`.
/// Each non-empty bucket of the source [`RoaringPostings`] is expanded
/// to its 8 KiB Bitmap form (regardless of whether the source container
/// was Array, Run, or Bitmap) and stacked in ascending `high16` order.
///
/// `bucket_index[i] = (high16, word_offset)` records the i-th bucket's
/// 16-bit doc-id prefix and its starting word offset within `d_buckets`.
/// `word_offset` is in u32 units; bucket `i` occupies
/// `d_buckets[bucket_index[i].1 .. bucket_index[i].1 + BITMAP_CONTAINER_WORDS]`.
///
/// ## Drop semantics
///
/// Drop calls `cudaFree(d_buckets)` exactly once. The Drop is run when
/// the last `Arc<VramTermEntry>` holding this entry goes out of scope —
/// either via LRU eviction (the cache drops its `Arc`) or via query-side
/// release (the dispatch path drops its borrowed clone after the kernel
/// completes). Concurrent in-flight queries holding `Arc` clones keep
/// the buffer alive until the kernel finishes; eviction-during-query is
/// safe.
pub struct VramTermEntry {
    /// Device pointer to a flat `[u32; total_words]` array. Never null
    /// for a successfully-constructed entry; cleared to null on drop
    /// to make double-drop idempotent.
    d_buckets: *mut u32,
    /// `(high16, word_offset)` for each non-empty bucket, sorted by
    /// `high16` ascending. `word_offset` is in u32 units relative to
    /// `d_buckets`.
    bucket_index: Vec<(u16, u32)>,
    /// Total u32 words allocated = `bucket_index.len() *
    /// BITMAP_CONTAINER_WORDS`.
    total_words: usize,
}

// SAFETY: device pointer is opaque, owned by `VramTermEntry` for its
// whole lifetime. No `&self` method dereferences `d_buckets` from Rust;
// only the CUDA driver reads it via the kernel launch. `Send` + `Sync`
// allow the entry to be wrapped in `Arc` and passed across query
// threads.
unsafe impl Send for VramTermEntry {}
unsafe impl Sync for VramTermEntry {}

impl std::fmt::Debug for VramTermEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramTermEntry")
            .field("d_buckets_is_null", &self.d_buckets.is_null())
            .field("bucket_count", &self.bucket_index.len())
            .field("total_words", &self.total_words)
            .finish()
    }
}

impl VramTermEntry {
    /// Number of buckets (non-empty containers) cached.
    #[must_use]
    pub fn bucket_count(&self) -> usize {
        self.bucket_index.len()
    }

    /// Total u32 words in the device buffer.
    #[must_use]
    pub fn total_words(&self) -> usize {
        self.total_words
    }

    /// Total bytes in the device buffer = `total_words * 4`.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_words * 4
    }

    /// Slice of `(high16, word_offset)` pairs. Caller-visible to drive
    /// cohort scatter at the dispatch layer.
    #[must_use]
    pub fn bucket_index(&self) -> &[(u16, u32)] {
        &self.bucket_index
    }

    /// Raw device pointer for the kernel layer's
    /// `cudaMemcpyAsync(DeviceToDevice)` scatter. The pointer is
    /// stable for the lifetime of the `Arc<VramTermEntry>` clone.
    ///
    /// # Safety
    /// The returned pointer is a CUDA device pointer; dereferencing
    /// from Rust is undefined behaviour. Only pass it to CUDA runtime
    /// or driver API calls.
    #[must_use]
    pub fn device_ptr(&self) -> *const u32 {
        self.d_buckets as *const u32
    }
}

impl Drop for VramTermEntry {
    fn drop(&mut self) {
        if !self.d_buckets.is_null() {
            // SAFETY: pointer was allocated by `cudaMalloc` in
            // [`VramCht::insert_internal`] and has not been freed
            // elsewhere (the `Arc` in the cache map ensures unique
            // ownership of the `d_buckets` pointer). Replacing with
            // null makes the Drop idempotent in the unlikely event it
            // runs twice.
            unsafe {
                let p = std::mem::replace(&mut self.d_buckets, null_mut());
                let _ = cudaFree(p as *mut c_void);
            }
        }
    }
}

/// Observable counters for the VRAM cache. Same shape as
/// [`super::cht::ChtStats`], with an additional `promotions` counter
/// recording host→VRAM transitions (when a term that lived in the v1
/// host cache gets promoted to v2 by the dispatch layer).
#[derive(Debug, Clone, Copy)]
pub struct VramChtStats {
    /// Cumulative cache hits.
    pub hits: u64,
    /// Cumulative cache misses.
    pub misses: u64,
    /// Cumulative successful insertions (excludes silent drops).
    pub inserts: u64,
    /// Cumulative LRU evictions caused by budget pressure.
    pub evictions: u64,
    /// Cumulative host→VRAM promotions (subset of `inserts` driven by
    /// dispatch-layer promotion, rather than direct insert).
    pub promotions: u64,
    /// Sum of `total_bytes()` for all currently-cached entries.
    pub current_bytes: u64,
    /// Configured budget ceiling.
    pub budget_bytes: u64,
    /// Number of cached entries.
    pub entries: u64,
}

/// VRAM-resident CHT instance. Construct via [`VramCht::with_budget`]
/// for custom budgets; access the process-global instance via
/// [`global`] / [`global_with_budget`].
#[allow(missing_docs)]
pub struct VramCht {
    inner: Mutex<VramChtInner>,
    budget_bytes: u64,
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    evictions: AtomicU64,
    promotions: AtomicU64,
}

struct VramChtInner {
    map: HashMap<ChtKey, (Arc<VramTermEntry>, u64)>,
    next_token: u64,
    current_bytes: u64,
}

impl VramCht {
    /// Construct a fresh VRAM cache with the given byte budget.
    /// Budget is checked against [`VramTermEntry::total_bytes`] at insert
    /// time; entries individually larger than the budget are silently
    /// dropped.
    #[must_use]
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(VramChtInner {
                map: HashMap::new(),
                next_token: 0,
                current_bytes: 0,
            }),
            budget_bytes,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
        }
    }

    /// Look up a cached posting list. On hit, bumps the LRU token so
    /// this entry is now the freshest. The returned `Arc` keeps the
    /// device buffer alive across concurrent eviction; the pointer is
    /// stable for the `Arc`'s lifetime.
    ///
    /// When the `FERRO_DISABLE_VRAM_CHT` env var is set, returns `None`
    /// immediately and bumps the misses counter so observers see the
    /// kill-switch effect (the dispatch path then falls through to
    /// the host fold path bytewise-identically to D-1 alone).
    pub fn get(&self, key: &ChtKey) -> Option<Arc<VramTermEntry>> {
        if is_disabled() {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut inner = self.inner.lock().expect("VramCht mutex poisoned");
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

    /// Build a [`VramTermEntry`] from a host [`RoaringPostings`] and
    /// insert it under `key`. Triggers LRU eviction if the new entry
    /// would push the budget over.
    ///
    /// Returns `Ok(true)` on successful insert, `Ok(false)` if the
    /// entry alone exceeds the budget (skipped), or `Err(VramChtError)`
    /// on CUDA failure (in which case no state change occurs).
    ///
    /// Idempotent: re-inserting the same key bumps the LRU token but
    /// keeps the existing device entry; the new `roaring` is dropped
    /// without device allocation, saving an unnecessary `cudaMalloc`
    /// and `cudaMemcpy` round-trip.
    pub fn insert(
        &self,
        key: ChtKey,
        roaring: &RoaringPostings,
    ) -> Result<bool, VramChtError> {
        if is_disabled() {
            // Kill-switch active — pretend we accepted the entry but
            // don't allocate device memory. Subsequent `get` calls
            // will miss as expected.
            return Ok(false);
        }
        // Compute footprint up front before any device calls.
        let bucket_count = roaring
            .containers
            .iter()
            .filter(|(_, c)| c.cardinality() > 0)
            .count();
        if bucket_count == 0 {
            // Empty term — caching is meaningless (any cohort
            // intersection involving it produces empty results without
            // touching the GPU). Skip cleanly.
            return Ok(false);
        }
        let total_words = bucket_count * BITMAP_CONTAINER_WORDS;
        let entry_bytes = (total_words as u64).saturating_mul(4);
        if entry_bytes > self.budget_bytes {
            // Term larger than the entire budget — never cacheable.
            return Ok(false);
        }

        // Idempotency check before allocation: if this key is already
        // cached, just bump the LRU token. Avoids a wasted cudaMalloc
        // for hot terms re-presented after eviction-window stabilises.
        {
            let mut inner = self.inner.lock().expect("VramCht mutex poisoned");
            let next_token = inner.next_token + 1;
            let already_cached = inner
                .map
                .get_mut(&key)
                .map(|(_, token)| {
                    *token = next_token;
                });
            if already_cached.is_some() {
                inner.next_token = next_token;
                return Ok(false);
            }
        }

        // Device-side construction. Allocates a single buffer of
        // `total_words * 4` bytes and stages the buckets via a host
        // staging Vec. We use the synchronous `cudaMemcpy` (not
        // `cudaMemcpyAsync` + sync) because there's no kernel chained
        // off this transfer; sync is the cheapest path to "data is
        // visible to the kernel".
        let entry = self.build_entry(roaring, total_words)?;

        // LRU eviction + insert under the cache mutex.
        let mut inner = self.inner.lock().expect("VramCht mutex poisoned");
        // Re-check idempotency: a concurrent `insert` may have raced
        // us between the pre-alloc check and now. If so, the new entry
        // is dropped (its `cudaFree` runs in `Drop`).
        let next_token = inner.next_token + 1;
        if let Some((_, token)) = inner.map.get_mut(&key) {
            *token = next_token;
            inner.next_token = next_token;
            drop(inner);
            // `entry` drops here, freeing its newly-alloced buffer.
            return Ok(false);
        }
        // Evict to fit. Pre-compute eviction targets to avoid
        // borrow checker issues with iterating + removing.
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
                    let evicted_bytes = evicted_value.total_bytes() as u64;
                    inner.current_bytes = inner.current_bytes.saturating_sub(evicted_bytes);
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                    // `evicted_value` drop releases the device buffer.
                }
            } else {
                break;
            }
        }
        inner.next_token = next_token;
        inner.current_bytes += entry_bytes;
        inner.map.insert(key, (Arc::new(entry), next_token));
        self.inserts.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    /// Insert and additionally bump the `promotions` counter — used
    /// when the dispatch layer promotes a host-CHT-resident term to
    /// VRAM (vs a direct first-touch insert from the planner). Counter
    /// distinction lets observability distinguish "cache warmed by
    /// query path" from "cache directly populated".
    pub fn promote(
        &self,
        key: ChtKey,
        roaring: &RoaringPostings,
    ) -> Result<bool, VramChtError> {
        let inserted = self.insert(key, roaring)?;
        if inserted {
            self.promotions.fetch_add(1, Ordering::Relaxed);
        }
        Ok(inserted)
    }

    /// Snapshot of current stats. Cheap (atomic loads + one mutex
    /// acquire for the entry / bytes counts).
    #[must_use]
    pub fn stats(&self) -> VramChtStats {
        let (current_bytes, entries) = {
            let inner = self.inner.lock().expect("VramCht mutex poisoned");
            (inner.current_bytes, inner.map.len() as u64)
        };
        VramChtStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            promotions: self.promotions.load(Ordering::Relaxed),
            current_bytes,
            budget_bytes: self.budget_bytes,
            entries,
        }
    }

    /// Reset all state — counters AND entries. Test-only. The Drop on
    /// each evicted `Arc<VramTermEntry>` releases its device buffer.
    #[doc(hidden)]
    pub fn reset(&self) {
        let mut inner = self.inner.lock().expect("VramCht mutex poisoned");
        inner.map.clear();
        inner.next_token = 0;
        inner.current_bytes = 0;
        drop(inner);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.inserts.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.promotions.store(0, Ordering::Relaxed);
    }

    /// Budget configured at construction.
    #[must_use]
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Internal: allocate a device buffer, build the host staging
    /// layout, copy H→D synchronously. Failures `cudaFree` the
    /// allocation before returning the error so the caller's state
    /// remains consistent.
    fn build_entry(
        &self,
        roaring: &RoaringPostings,
        total_words: usize,
    ) -> Result<VramTermEntry, VramChtError> {
        let bytes = total_words * std::mem::size_of::<u32>();

        // Host staging buffer: build the full flat layout, then a single
        // cudaMemcpy. Smaller cudaMemcpys per bucket would multiply
        // launch overhead — at typical bucket counts (~10-100 per term)
        // a single H→D wins handily.
        let mut staging: Vec<u32> = vec![0u32; total_words];
        let mut bucket_index: Vec<(u16, u32)> = Vec::with_capacity(
            roaring.containers.len(),
        );
        let mut word_offset: u32 = 0;
        for (high16, container) in &roaring.containers {
            if container.cardinality() == 0 {
                continue;
            }
            let start = word_offset as usize;
            let end = start + BITMAP_CONTAINER_WORDS;
            let chunk = &mut staging[start..end];
            match container {
                Container::Bitmap(bm) => {
                    chunk.copy_from_slice(bm.words.as_ref());
                }
                Container::Array(arr) => {
                    let bm = BitmapContainer::from_array(arr);
                    chunk.copy_from_slice(bm.words.as_ref());
                }
                Container::Run(rc) => {
                    let bm = BitmapContainer::from_run(rc);
                    chunk.copy_from_slice(bm.words.as_ref());
                }
            }
            bucket_index.push((*high16, word_offset));
            word_offset = word_offset
                .checked_add(BITMAP_CONTAINER_WORDS as u32)
                .expect("word_offset must not overflow u32 within total_words bound");
        }

        // Allocate device buffer.
        let mut d_buckets: *mut c_void = null_mut();
        // SAFETY: cudaMalloc writes a valid device pointer on success
        // and returns a non-zero error code on failure (in which case
        // d_buckets is left untouched / null).
        let rc = unsafe { cudaMalloc(&mut d_buckets, bytes) };
        if rc != CUDA_SUCCESS {
            return Err(VramChtError::Malloc { bytes, code: rc });
        }

        // Copy H→D synchronously. The default stream sync at end of
        // cudaMemcpy guarantees the data is visible to subsequent
        // kernel launches on any stream.
        // SAFETY: d_buckets is a valid device pointer of `bytes` size;
        // staging.as_ptr() points to `bytes` of host memory we
        // populated above.
        let rc = unsafe {
            cudaMemcpy(
                d_buckets,
                staging.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };
        if rc != CUDA_SUCCESS {
            // Defensive cleanup: we allocated, the copy failed, free
            // before returning the error so the caller's state is
            // unaffected.
            // SAFETY: d_buckets was successfully allocated above; no
            // one else holds it.
            unsafe {
                let _ = cudaFree(d_buckets);
            }
            return Err(VramChtError::Memcpy { bytes, code: rc });
        }

        Ok(VramTermEntry {
            d_buckets: d_buckets as *mut u32,
            bucket_index,
            total_words,
        })
    }

    // ========================================================
    // Phase 2 D-5 — warm-restart persistence (uncompressed VRAM tier)
    // ========================================================

    /// Dump every cached entry to `<path>` in the v2 (FCV2) wire
    /// format. Each entry stages its device-resident
    /// `[u32; total_words]` flat bitmap back to host via
    /// `cudaMemcpyDeviceToHost` and writes the uncompressed bytes
    /// next to the bucket-index header.
    ///
    /// The atomic-write protocol from
    /// [`super::persist::finalise_atomic_write`] guarantees that a
    /// crash mid-dump leaves the destination path untouched; the
    /// staging `<path>.tmp` is left behind for operator inspection.
    ///
    /// Returns the number of entries written. The default-stream
    /// `cudaMemcpy` is synchronous; it implicitly serialises
    /// against in-flight kernels on the same stream so a query in
    /// progress on the same device-resident `Arc` is safe.
    pub fn dump_to_path(
        &self,
        path: &std::path::Path,
    ) -> Result<u64, super::persist::DumpError> {
        use super::persist::{
            finalise_atomic_write, open_tmp_writer, write_chtkey, write_file_header,
            write_file_trailer, DumpError, MAGIC_V2,
        };
        use std::io::Write;
        // Snapshot Arcs under the lock, release before staging
        // device → host copies (which can be slow on multi-MiB
        // entries). Arc keeps each entry's device buffer alive
        // until the snapshot is fully drained.
        let snapshot: Vec<(ChtKey, Arc<VramTermEntry>)> = {
            let inner = self.inner.lock().expect("VramCht mutex poisoned");
            inner
                .map
                .iter()
                .map(|(k, (v, _))| (k.clone(), Arc::clone(v)))
                .collect()
        };
        let entry_count = snapshot.len() as u64;
        let mut writer = open_tmp_writer(path)?;
        write_file_header(&mut writer, MAGIC_V2, entry_count)?;
        for (i, (key, entry)) in snapshot.iter().enumerate() {
            write_chtkey(&mut writer, key)?;
            // bucket_count (u32) + bucket_index pairs (u16 + u32 each).
            let bucket_count = u32::try_from(entry.bucket_index.len()).unwrap_or(u32::MAX);
            writer.write_all(&bucket_count.to_le_bytes())?;
            for (high16, word_offset) in &entry.bucket_index {
                writer.write_all(&high16.to_le_bytes())?;
                writer.write_all(&word_offset.to_le_bytes())?;
            }
            // uncompressed_bytes (= total_words * 4).
            let uncompressed_bytes = (entry.total_words as u64).saturating_mul(4);
            writer.write_all(&uncompressed_bytes.to_le_bytes())?;
            // Stage the device buffer back to host then write.
            let mut host_staging: Vec<u32> = vec![0u32; entry.total_words];
            let bytes = entry.total_words * 4;
            // SAFETY: entry.d_buckets points to `bytes` of device
            // memory owned by this Arc; host_staging holds `bytes`
            // of valid host memory.
            let rc = unsafe {
                cudaMemcpy(
                    host_staging.as_mut_ptr() as *mut c_void,
                    entry.d_buckets as *const c_void,
                    bytes,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            if rc != CUDA_SUCCESS {
                return Err(DumpError::Cuda {
                    entry_index: i as u64,
                    bytes,
                    code: rc,
                });
            }
            // bytemuck-free byte view: u32 staging vec → byte slice.
            let byte_view = unsafe {
                std::slice::from_raw_parts(
                    host_staging.as_ptr() as *const u8,
                    bytes,
                )
            };
            writer.write_all(byte_view)?;
        }
        write_file_trailer(&mut writer)?;
        finalise_atomic_write(writer, path)?;
        Ok(entry_count)
    }

    /// Load entries from `<path>` into this cache. For each entry
    /// read from disk, allocates a fresh device buffer via
    /// `cudaMalloc`, stages the uncompressed bytes back to device
    /// via `cudaMemcpyHostToDevice`, and inserts under the parsed
    /// [`ChtKey`].
    ///
    /// Per-entry CUDA failures are skipped (the dump's other
    /// entries continue loading); structural failures
    /// (magic / version / trailer) fail loud via
    /// [`super::persist::LoadError`]. Returns the number of
    /// entries successfully installed.
    pub fn load_from_path(
        &self,
        path: &std::path::Path,
    ) -> Result<u64, super::persist::LoadError> {
        use super::persist::{
            open_reader, read_and_validate_file_header,
            read_and_validate_file_trailer, read_chtkey, read_exact_or_truncated,
            read_u16_le, read_u32_le, read_u64_le, MAGIC_V2,
        };
        let mut reader = open_reader(path)?;
        let entry_count = read_and_validate_file_header(&mut reader, MAGIC_V2)?;
        let mut loaded: u64 = 0;
        for i in 0..entry_count {
            let key = read_chtkey(&mut reader, i)?;
            let bucket_count = read_u32_le(&mut reader)? as usize;
            let mut bucket_index: Vec<(u16, u32)> = Vec::with_capacity(bucket_count);
            for _ in 0..bucket_count {
                let high16 = read_u16_le(&mut reader)?;
                let word_offset = read_u32_le(&mut reader)?;
                bucket_index.push((high16, word_offset));
            }
            let uncompressed_bytes = read_u64_le(&mut reader)? as usize;
            let body = read_exact_or_truncated(&mut reader, uncompressed_bytes)?;
            // Cross-check: total_words derived from bucket_count
            // must match the uncompressed_bytes; mismatched dumps
            // are skipped (structural inconsistency without being
            // truncated).
            let expected_bytes = bucket_count * BITMAP_CONTAINER_WORDS * 4;
            if expected_bytes != uncompressed_bytes {
                continue;
            }
            // Allocate device buffer + copy H2D.
            let mut d_buckets: *mut c_void = null_mut();
            // SAFETY: cudaMalloc writes the device pointer on success.
            let rc = unsafe { cudaMalloc(&mut d_buckets, uncompressed_bytes) };
            if rc != CUDA_SUCCESS {
                continue; // Skip; budget pressure or device full.
            }
            // SAFETY: d_buckets is a valid device buffer of
            // uncompressed_bytes; body.as_ptr() holds the right
            // bytes.
            let rc = unsafe {
                cudaMemcpy(
                    d_buckets,
                    body.as_ptr() as *const c_void,
                    uncompressed_bytes,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            if rc != CUDA_SUCCESS {
                // Defensive cleanup.
                // SAFETY: just allocated above.
                unsafe {
                    let _ = cudaFree(d_buckets);
                }
                continue;
            }
            let entry = VramTermEntry {
                d_buckets: d_buckets as *mut u32,
                bucket_index,
                total_words: bucket_count * BITMAP_CONTAINER_WORDS,
            };
            // Install via the standard LRU eviction path so budget
            // accounting stays consistent.
            let entry_bytes = (entry.total_words as u64).saturating_mul(4);
            if entry_bytes > self.budget_bytes {
                // The dump was written under a larger budget than
                // this process; drop the entry cleanly (Drop runs
                // cudaFree).
                drop(entry);
                continue;
            }
            let mut inner = self.inner.lock().expect("VramCht mutex poisoned");
            let next_token = inner.next_token + 1;
            if inner.map.contains_key(&key) {
                // Already cached — race against another loader, or
                // duplicate dump entry; drop ours.
                drop(inner);
                drop(entry);
                continue;
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
                        let evicted_bytes = evicted_value.total_bytes() as u64;
                        inner.current_bytes =
                            inner.current_bytes.saturating_sub(evicted_bytes);
                        self.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    break;
                }
            }
            inner.next_token = next_token;
            inner.current_bytes += entry_bytes;
            inner.map.insert(key, (Arc::new(entry), next_token));
            self.inserts.fetch_add(1, Ordering::Relaxed);
            loaded += 1;
        }
        read_and_validate_file_trailer(&mut reader)?;
        Ok(loaded)
    }
}

// ============================================================
// Process-global cache
// ============================================================

/// Default VRAM budget when no explicit configuration is provided.
/// 4 GiB — conservative for 16 GiB-class GPUs (e.g. RTX 4070 Ti SUPER
/// dev box, L4) where compute scratch + nvCOMP + Tantivy overhead
/// reserve must still fit. Operators on L40S 48 GiB-class GPUs should
/// raise this via `ferrosearch --cht-vram-budget-bytes 32G`.
pub const DEFAULT_VRAM_CHT_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

static GLOBAL_VRAM_CHT: OnceLock<VramCht> = OnceLock::new();

/// Get the process-global VRAM cache, initialising with the default
/// budget on first access.
pub fn global() -> &'static VramCht {
    GLOBAL_VRAM_CHT.get_or_init(|| VramCht::with_budget(DEFAULT_VRAM_CHT_BUDGET_BYTES))
}

/// Phase 2 D-3 v2 kill-switch — when the `FERRO_DISABLE_VRAM_CHT`
/// env var is set to `1` / `true`, all `get` / `insert` / `promote`
/// calls short-circuit (return as if the cache were always empty
/// and full-budget rejected). The dispatch site
/// (`try_gpu_intersect`) sees `None` from `get` and falls through
/// to the host fold path, giving operators a clean A/B comparison
/// against the D-1 host CHT alone without recompiling.
///
/// Checked once on first call (process-lifetime cache); the env
/// var must be set before any query path touches the cache. Stats
/// counters still record the misses so observers can confirm the
/// kill-switch is active.
fn is_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("FERRO_DISABLE_VRAM_CHT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Whether the VRAM CHT kill-switch is currently active. Public so
/// the periodic stats logger and bench harnesses can report the
/// effective state alongside the hit/miss counters.
#[must_use]
pub fn kill_switch_active() -> bool {
    is_disabled()
}

/// Get the process-global VRAM cache, initialising with a custom
/// budget **iff this is the first access**. Returns `None` if the
/// global is already initialised.
pub fn global_with_budget(budget_bytes: u64) -> Option<&'static VramCht> {
    if GLOBAL_VRAM_CHT.get().is_some() {
        return None;
    }
    let _ = GLOBAL_VRAM_CHT.set(VramCht::with_budget(budget_bytes));
    GLOBAL_VRAM_CHT.get()
}

/// Reset the process-global VRAM cache. Test-only.
#[doc(hidden)]
pub fn reset_global() {
    if let Some(cht) = GLOBAL_VRAM_CHT.get() {
        cht.reset();
    }
}

// ============================================================
// Tests
// ============================================================
//
// These tests require a working CUDA device. On a driver-missing CI
// box the very first `cudaMalloc` returns a non-success code; the
// tests early-out gracefully via the `Err(Malloc)` arm so the suite
// stays green on machines without an NVIDIA GPU. Heavy CUDA correctness
// coverage already lives in `crates/ferro-compress/tests/bitmap_op_gpu.rs`
// and (for the v1 host cache) in `super::cht::tests`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postings::roaring::encoder::RoaringEncoder;

    fn small_roaring() -> RoaringPostings {
        RoaringEncoder::from_doc_ids(&[1, 2, 3, 100, 65540])
    }

    fn dense_roaring(seed: u32) -> RoaringPostings {
        let docs: Vec<u32> = (0..200_000).map(|i| seed.wrapping_add(i * 3)).collect();
        RoaringEncoder::from_doc_ids(&docs)
    }

    fn dummy_key(field: u32, term_hash: u64) -> ChtKey {
        ChtKey {
            segment_id: SegmentId::generate_random(),
            field,
            term_hash,
        }
    }

    /// Skip predicate: returns `true` if the host machine has no
    /// usable CUDA device, in which case any `insert` returns
    /// `Err(Malloc)` and the test body short-circuits cleanly.
    fn cuda_available() -> bool {
        let cht = VramCht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xfeed, 100);
        match cht.insert(key, &small_roaring()) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    #[test]
    fn miss_then_hit() {
        if !cuda_available() {
            return;
        }
        let cht = VramCht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xdead, 100);
        assert!(cht.get(&key).is_none());
        let stats = cht.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        let inserted = cht.insert(key.clone(), &small_roaring()).unwrap();
        assert!(inserted, "non-empty term must insert");
        let got = cht.get(&key);
        assert!(got.is_some());
        let stats = cht.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.inserts, 1);
    }

    #[test]
    fn duplicate_insert_reuses_existing_entry() {
        if !cuda_available() {
            return;
        }
        let cht = VramCht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xdead, 100);
        let inserted_1 = cht.insert(key.clone(), &small_roaring()).unwrap();
        assert!(inserted_1);
        let bytes_after_first = cht.stats().current_bytes;

        let inserted_2 = cht.insert(key.clone(), &small_roaring()).unwrap();
        assert!(!inserted_2, "second insert of same key must short-circuit");
        let bytes_after_second = cht.stats().current_bytes;
        assert_eq!(
            bytes_after_first, bytes_after_second,
            "re-insert of same key must not double-count bytes"
        );
        assert_eq!(cht.stats().inserts, 1);
    }

    #[test]
    fn lru_evicts_oldest_first() {
        if !cuda_available() {
            return;
        }
        // Compute one dense entry's bytes for budget sizing.
        let probe = VramCht::with_budget(1 << 30);
        let probe_key = dummy_key(0xff00, 100);
        probe.insert(probe_key.clone(), &dense_roaring(0)).unwrap();
        let one_dense = probe.stats().current_bytes;
        drop(probe);

        let cht = VramCht::with_budget(one_dense + 1024);
        let k1 = dummy_key(0xa1f7, 100);
        let k2 = dummy_key(0xb273, 100);
        cht.insert(k1.clone(), &dense_roaring(0)).unwrap();
        cht.insert(k2.clone(), &dense_roaring(1)).unwrap();
        assert!(cht.get(&k1).is_none(), "oldest key must have been evicted");
        assert!(cht.get(&k2).is_some(), "newest key must still be cached");
        assert!(cht.stats().evictions >= 1);
    }

    #[test]
    fn entry_larger_than_budget_skipped() {
        if !cuda_available() {
            return;
        }
        // 1 KiB budget << 200_000-doc bitmap class.
        let cht = VramCht::with_budget(1024);
        let key = dummy_key(0xc0ff, 100);
        let inserted = cht.insert(key.clone(), &dense_roaring(7)).unwrap();
        assert!(!inserted);
        let stats = cht.stats();
        assert_eq!(stats.inserts, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.current_bytes, 0);
        assert!(cht.get(&key).is_none());
    }

    #[test]
    fn empty_roaring_skipped_cleanly() {
        if !cuda_available() {
            return;
        }
        let cht = VramCht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xe1ee, 100);
        let empty = RoaringPostings::default();
        let inserted = cht.insert(key.clone(), &empty).unwrap();
        assert!(!inserted);
        assert_eq!(cht.stats().inserts, 0);
        assert!(cht.get(&key).is_none());
    }

    #[test]
    fn promote_bumps_promotions_counter() {
        if !cuda_available() {
            return;
        }
        let cht = VramCht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xdead, 100);
        let inserted = cht.promote(key.clone(), &small_roaring()).unwrap();
        assert!(inserted);
        let stats = cht.stats();
        assert_eq!(stats.promotions, 1);
        assert_eq!(stats.inserts, 1);
    }

    #[test]
    fn entry_bucket_index_matches_source() {
        if !cuda_available() {
            return;
        }
        let cht = VramCht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xface, 100);
        let rp = small_roaring();
        cht.insert(key.clone(), &rp).unwrap();
        let entry = cht.get(&key).unwrap();
        // small_roaring has docs [1,2,3,100] in bucket 0 and [65540 (=
        // bucket 1, low16=4)] in bucket 1.
        let buckets: Vec<u16> = entry.bucket_index().iter().map(|(h, _)| *h).collect();
        assert_eq!(buckets, vec![0, 1]);
        // Each bucket's word_offset must be a multiple of
        // BITMAP_CONTAINER_WORDS, ascending.
        for (i, (_, off)) in entry.bucket_index().iter().enumerate() {
            assert_eq!(*off as usize, i * BITMAP_CONTAINER_WORDS);
        }
        assert_eq!(entry.total_words(), 2 * BITMAP_CONTAINER_WORDS);
        assert_eq!(entry.bucket_count(), 2);
    }

    #[test]
    fn arc_clone_survives_eviction_then_reset() {
        // Phase 2 D-3 design doc § "Honest limitations" #8: eviction
        // during in-flight queries must not free a device buffer
        // that another thread still references. The `Arc<VramTermEntry>`
        // refcount is the safety mechanism — `cudaFree` only runs
        // when the LAST Arc drops. This test pins the property: take
        // a clone, evict (or reset) the cache entry, the clone is
        // still usable (we read its bucket_index, which is host-side,
        // and the device pointer is still valid because Drop hasn't
        // run yet).
        if !cuda_available() {
            return;
        }
        let cht = VramCht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0x5afe, 100);
        cht.insert(key.clone(), &small_roaring()).unwrap();
        let live_clone = cht.get(&key).expect("post-insert get must hit");
        // Reset the cache — this drops the Arc held by the cache map.
        // `live_clone` retains its own Arc, so the device buffer is
        // not freed yet.
        cht.reset();
        // The clone's metadata is still consistent (host-side fields
        // are immutable post-construction).
        assert_eq!(live_clone.bucket_count(), 2);
        assert!(!live_clone.device_ptr().is_null());
        // Drop the clone — this is when `cudaFree` actually runs.
        // We can't directly observe the cudaFree from Rust without
        // re-querying the device, but the test passing without segfault
        // (and the cache being re-usable below) is sufficient evidence.
        drop(live_clone);
        // Cache is empty + reusable post-reset.
        assert_eq!(cht.stats().entries, 0);
        let key2 = dummy_key(0x5aff, 100);
        let inserted = cht.insert(key2.clone(), &small_roaring()).unwrap();
        assert!(inserted);
        assert!(cht.get(&key2).is_some());
    }

    #[test]
    fn kill_switch_via_env_var() {
        // Phase 2 D-3 v2 follow-up: FERRO_DISABLE_VRAM_CHT=1 forces
        // get/insert/promote to short-circuit so operators can A/B
        // bench v2 vs D-1 alone without recompiling.
        //
        // This test cannot run in parallel with other tests that read
        // the kill-switch state because `is_disabled()` caches the
        // env-var lookup in a process-global `OnceLock`. We test the
        // API surface (the underlying static caching is process-local
        // to the test binary; in production the env var is set once
        // at startup before any cache touch).
        if !cuda_available() {
            return;
        }
        // We can't easily flip the env var mid-test (OnceLock caches
        // on first call). Instead, document that:
        //   - With FERRO_DISABLE_VRAM_CHT unset (default):
        //     get-then-insert-then-get sees: miss → insert OK → hit.
        //   - With FERRO_DISABLE_VRAM_CHT=1 (set before process start):
        //     get-then-insert-then-get sees: miss → insert OK(false) → miss.
        //
        // The following assertions cover the default path; the
        // kill-switch path is exercised end-to-end by the bench
        // harness's `FERRO_DISABLE_VRAM_CHT=1 cargo bench` invocation
        // (Phase 2 D-3 v2 follow-up A/B).
        let cht = VramCht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xb007, 100);
        if kill_switch_active() {
            // Env var was set before this test process started.
            // Verify get/insert short-circuit cleanly.
            assert!(cht.get(&key).is_none());
            let inserted = cht.insert(key.clone(), &small_roaring()).unwrap();
            assert!(!inserted, "kill-switch active: insert must short-circuit");
            assert!(cht.get(&key).is_none(), "kill-switch active: get always misses");
            // Stats: misses bumped (twice — once per get), inserts not bumped.
            let stats = cht.stats();
            assert_eq!(stats.inserts, 0);
            assert_eq!(stats.entries, 0);
            assert!(stats.misses >= 2);
        } else {
            // Default path: cache works as designed.
            assert!(cht.get(&key).is_none());
            let inserted = cht.insert(key.clone(), &small_roaring()).unwrap();
            assert!(inserted);
            assert!(cht.get(&key).is_some());
        }
    }

    // ========================================================
    // Phase 2 D-5 — v2 dump/load roundtrip tests
    // ========================================================

    #[test]
    fn vram_cht_v2_dump_then_load_roundtrip_byte_equal() {
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v2.bin");

        // Populate source cache with 3 entries.
        let src = VramCht::with_budget(256 * 1024 * 1024);
        let entries = vec![
            (
                dummy_key(0xAAAA, 0x1111_2222_3333_4444),
                small_roaring(),
            ),
            (
                dummy_key(0xBBBB, 0x5555_6666_7777_8888),
                dense_roaring(0),
            ),
            (
                dummy_key(0xCCCC, 0x9999_AAAA_BBBB_CCCC),
                dense_roaring(1),
            ),
        ];
        for (k, rp) in &entries {
            src.insert(k.clone(), rp).unwrap();
        }
        let n_dumped = src.dump_to_path(&path).unwrap();
        assert_eq!(n_dumped, entries.len() as u64);
        assert!(path.exists());
        // Capture per-entry (bucket_index, total_words, host bytes)
        // for byte-equal comparison post-load. We round-trip the
        // device buffer back to host once per source entry now so
        // we can verify the loader produced an identical buffer.
        let mut expected_bytes: Vec<(usize, Vec<u32>)> = Vec::with_capacity(entries.len());
        for (k, _) in &entries {
            let e = src.get(k).expect("src entry must hit");
            let mut staging = vec![0u32; e.total_words];
            let bytes = e.total_words * 4;
            let rc = unsafe {
                cudaMemcpy(
                    staging.as_mut_ptr() as *mut c_void,
                    e.device_ptr() as *const c_void,
                    bytes,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            assert_eq!(rc, CUDA_SUCCESS, "src D→H must succeed");
            expected_bytes.push((e.total_words, staging));
        }

        // Fresh cache; load; per-key byte-equal.
        let dst = VramCht::with_budget(256 * 1024 * 1024);
        let n_loaded = dst.load_from_path(&path).unwrap();
        assert_eq!(n_loaded, entries.len() as u64);
        for ((k, _), (expected_words, expected_buf)) in
            entries.iter().zip(expected_bytes.iter())
        {
            let got = dst.get(k).expect("loaded entry must hit");
            assert_eq!(got.total_words(), *expected_words);
            let mut staging = vec![0u32; got.total_words()];
            let bytes = got.total_words() * 4;
            let rc = unsafe {
                cudaMemcpy(
                    staging.as_mut_ptr() as *mut c_void,
                    got.device_ptr() as *const c_void,
                    bytes,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            assert_eq!(rc, CUDA_SUCCESS, "dst D→H must succeed");
            assert_eq!(
                &staging, expected_buf,
                "loaded device buffer must be byte-equal to dumped"
            );
        }
    }

    #[test]
    fn vram_cht_v2_restart_simulation_first_query_hits() {
        // Phase 2 D-5 acceptance gate: after dump + restart
        // simulation (= fresh VramCht) + load, the FIRST query for
        // a previously-cached key must observe a hit. v2 specific:
        // the loaded entry must be queryable identically to a
        // freshly-inserted entry (kernel scatter consumes the
        // device buffer via Arc, not the Cht itself).
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v2_restart.bin");

        let original = VramCht::with_budget(64 * 1024 * 1024);
        let k1 = dummy_key(0xCAFE, 0xBABE_0001);
        let k2 = dummy_key(0xCAFE, 0xBABE_0002);
        original.insert(k1.clone(), &small_roaring()).unwrap();
        original.insert(k2.clone(), &dense_roaring(0)).unwrap();
        original.dump_to_path(&path).unwrap();

        let restarted = VramCht::with_budget(64 * 1024 * 1024);
        let pre = restarted.stats();
        assert_eq!(pre.hits, 0);
        assert_eq!(pre.entries, 0);

        let loaded = restarted.load_from_path(&path).unwrap();
        assert_eq!(loaded, 2, "load installs both entries");
        let post_load = restarted.stats();
        assert_eq!(post_load.entries, 2);

        // Acceptance gate.
        let got = restarted.get(&k1);
        assert!(got.is_some(), "first-query-post-restart hits");
        let post_query = restarted.stats();
        assert_eq!(post_query.hits, 1);
        assert_eq!(post_query.misses, 0);

        // The device pointer is non-null (the Arc owns a freshly-
        // allocated buffer); bucket_index matches the source layout.
        let entry = got.unwrap();
        assert!(!entry.device_ptr().is_null());
        assert!(entry.bucket_count() >= 1);
    }

    #[test]
    fn vram_cht_v2_load_wrong_magic_rejected() {
        if !cuda_available() {
            return;
        }
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v2.bin");
        // Write v3-magic file.
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&super::super::persist::MAGIC_V3.to_le_bytes())
            .unwrap();
        f.write_all(&super::super::persist::WIRE_VERSION.to_le_bytes())
            .unwrap();
        f.write_all(&super::super::persist::HASH_FN_FXHASHER_V1.to_le_bytes())
            .unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&super::super::persist::MAGIC_END.to_le_bytes())
            .unwrap();
        drop(f);

        let dst = VramCht::with_budget(64 * 1024 * 1024);
        let err = dst.load_from_path(&path).unwrap_err();
        assert!(
            matches!(
                err,
                super::super::persist::LoadError::WrongMagic { .. }
            ),
            "v3 file at v2 path must reject as WrongMagic, got {err:?}"
        );
    }

    #[test]
    fn vram_cht_v2_dump_empty_cache_roundtrips() {
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v2_empty.bin");
        let src = VramCht::with_budget(64 * 1024 * 1024);
        let n = src.dump_to_path(&path).unwrap();
        assert_eq!(n, 0);
        let dst = VramCht::with_budget(64 * 1024 * 1024);
        let n_loaded = dst.load_from_path(&path).unwrap();
        assert_eq!(n_loaded, 0);
        assert_eq!(dst.stats().entries, 0);
    }

    #[test]
    fn concurrent_get_inserts_under_budget_pressure() {
        // Stress: spawn N reader threads, K writer threads, all hitting
        // the same cache under a tight budget that forces frequent
        // evictions. The test passes iff:
        //   1. No panics (mutex-poisoning, double-free, etc).
        //   2. The cache state is internally consistent after the
        //      threads join (entries == map.len() invariant).
        //   3. current_bytes ≤ budget_bytes at all times (tested via
        //      the stats snapshot post-join).
        if !cuda_available() {
            return;
        }
        // Probe one dense entry's footprint, then size the budget to
        // hold ~3 of them (force LRU pressure).
        let probe = VramCht::with_budget(1 << 30);
        let probe_key = dummy_key(0xff00, 100);
        probe.insert(probe_key, &dense_roaring(0)).unwrap();
        let one_dense = probe.stats().current_bytes;
        drop(probe);

        let cht = std::sync::Arc::new(VramCht::with_budget(one_dense * 3 + 4096));
        let mut handles = Vec::new();
        // Writers: insert dense roarings under unique keys.
        for w in 0..4 {
            let cht = std::sync::Arc::clone(&cht);
            handles.push(std::thread::spawn(move || {
                for i in 0..16u32 {
                    let key = dummy_key(0x10000 + (w * 100 + i), 100);
                    let rp = dense_roaring(w * 100 + i);
                    let _ = cht.insert(key, &rp);
                }
            }));
        }
        // Readers: poll arbitrary keys.
        for r in 0..4 {
            let cht = std::sync::Arc::clone(&cht);
            handles.push(std::thread::spawn(move || {
                for i in 0..32u32 {
                    let key = dummy_key(0x10000 + (r * 100 + i), 100);
                    let _ = cht.get(&key);
                }
            }));
        }
        for h in handles {
            h.join().expect("worker thread must not panic");
        }
        let stats = cht.stats();
        assert!(
            stats.current_bytes <= cht.budget_bytes(),
            "current_bytes ({}) must stay within budget ({})",
            stats.current_bytes,
            cht.budget_bytes()
        );
    }
}

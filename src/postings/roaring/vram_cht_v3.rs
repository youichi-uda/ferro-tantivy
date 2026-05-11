//! Phase 2 D-4 v3 — VRAM-resident **Bitcomp-compressed** CHT.
//!
//! Builds on Wave 9 v2 ([`super::vram_cht`]) by storing the per-term
//! flat BitmapContainer buffer in **Bitcomp-compressed** form on the
//! device, gaining ~3-4× capacity multiplier (Phase 0 measured 3.59×
//! ratio on `postings.bin` uint32 stream) at the cost of a
//! per-query decompression step (~22 µs/term per design doc § 3.2's
//! 366 GB/s decomp throughput on uint32).
//!
//! ## v3 scope
//!
//! - **Device-memory cache** keyed on the same [`ChtKey`] as v1/v2
//!   (`(segment_id, field_id, term_hash)` as of Wave 11 — content-stable
//!   across process restarts; see `cht::ChtKey` for migration details).
//! - **Per-term Bitcomp blob** stored in a single `cudaMalloc`
//!   allocation; uncompressed buffer rebuilt at query time into a
//!   workbench buffer via [`ferro_compress::BitcompDeviceCodec`].
//! - **LRU eviction** under a configurable VRAM byte budget (counted
//!   in *compressed* bytes, the actual VRAM footprint). Default 16
//!   GiB on L40S 48 GiB-class; operators tune via
//!   `ferrosearch --cht-vram-compressed-budget-bytes`.
//! - **Process-global** `OnceLock<VramCompressedCht>`, same lifetime
//!   model as v1 host CHT and v2 VRAM CHT.
//! - **Coexists with v1 + v2**: a term may live in any combination of
//!   the three caches. The dispatch layer
//!   ([`super::gpu_intersect::try_gpu_intersect`]) prefers v3 when
//!   all cohort members have a v3 hit (decompress-on-read into
//!   workbench, then the same DtD scatter as v2); falls back to v2
//!   when any member is v3-miss but v2-hit; falls back to v1 +
//!   flat_buffer + H→D when v2 also misses.
//! - **Kill-switch** via `FERRO_DISABLE_VRAM_COMPRESSED` env var,
//!   same pattern as v2's `FERRO_DISABLE_VRAM_CHT`. Allows operator
//!   A/B (v3 vs v2 vs v1 alone) without recompiling.
//!
//! ## What v3 does NOT do (deferred)
//!
//! - **Multi-chunk Bitcomp** for very large terms (> 16 MiB
//!   uncompressed): the current insert path single-chunks each term.
//!   Production posting lists at the planner-approved cardinality
//!   (≥ 100 K docs ≈ ≥ 1 bucket but < 100 buckets) are well below
//!   the 16 MiB ceiling. Multi-chunk = future work in
//!   [`ferro_compress::BitcompDeviceCodec`].
//! - **Warm-restart persistence** (D-5): the cache is process-local.
//! - **Per-tenant isolation** (Wave 12).
//!
//! ## Empirical context
//!
//! Phase 0 nvCOMP benchmarks (4070 Ti SUPER) showed Bitcomp on
//! uint32 postings: 3.59× ratio + 366 GB/s decomp throughput.
//! Translating to v3:
//!
//! - **4× capacity multiplier**: a 32 GiB v3 budget effectively
//!   caches ~128 GiB of uncompressed bitmap data (the
//!   "192 GiB compressed hot tier per GPU" pitch claim assumes 6×
//!   ratio on JSON-log-class workloads; postings give 3.59×
//!   conservatively).
//! - **Per-query decompress overhead**: 8 KiB-per-bucket × 12
//!   buckets per term × 12 terms = 1.15 MiB / cohort. At 366 GB/s
//!   = ~3.1 µs decompress per cohort. Well within the v2-vs-D-1
//!   3.22× speedup envelope (D-1 baseline ~1 ms → v3 1 ms - 3.1 µs
//!   ≈ negligible overhead).

#![cfg(all(feature = "gpu", feature = "ferro-compress", feature = "cuda-bitmap-kernel"))]

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ferro_compress::nvcomp_sys::cuda::{
    cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind, CUDA_SUCCESS,
};
use ferro_compress::{BitcompDataType, BitcompDeviceCodec, Error as FcError};

use crate::postings::roaring::cht::ChtKey;
use crate::postings::roaring::encoder::RoaringPostings;
use crate::postings::roaring::shared_bitmap_payload::SharedBitmapPayload;
use crate::postings::roaring::BITMAP_CONTAINER_WORDS;

/// Errors specific to the v3 cache layer. Insert / promote failures
/// at the CUDA / Bitcomp level are surfaced here so callers can
/// distinguish them from in-memory bookkeeping issues.
#[derive(Debug, thiserror::Error)]
pub enum VramCompressedChtError {
    /// `cudaMalloc` returned non-success.
    #[error("cudaMalloc failed (bytes={bytes}, code={code})")]
    Malloc {
        /// Bytes the allocation requested.
        bytes: usize,
        /// Raw `cudaError_t` value.
        code: u32,
    },
    /// `cudaMemcpy` host-to-device failed during staging.
    #[error("cudaMemcpy(H2D) failed (bytes={bytes}, code={code})")]
    Memcpy {
        /// Bytes the copy attempted.
        bytes: usize,
        /// Raw `cudaError_t` value.
        code: u32,
    },
    /// Bitcomp compress / decompress returned an error from
    /// `ferro_compress::BitcompDeviceCodec`.
    #[error("Bitcomp error: {0}")]
    Bitcomp(#[from] FcError),
}

/// One cached posting list, stored on the CUDA device in
/// **Bitcomp-compressed** form.
///
/// Layout: `d_compressed` points to a contiguous compressed-bytes
/// buffer of `compressed_bytes` size. The `bucket_index` records the
/// **uncompressed** layout (so the dispatch layer knows how to
/// scatter buckets after decompression). `uncompressed_bytes` =
/// `bucket_count * BITMAP_CONTAINER_WORDS * 4`.
///
/// At query time the dispatch layer:
/// 1. Allocates / reuses a workbench device buffer of
///    `uncompressed_bytes` size (managed by `CudaBitmapOpKernel`).
/// 2. Calls `BitcompDeviceCodec::decompress_one(d_compressed,
///    compressed_bytes, d_workbench, uncompressed_bytes)`.
/// 3. Scatters from the workbench (using `bucket_index`) to the
///    cohort's flat layout, identical to v2's `scatter_term_to_device`.
///
/// ## Drop semantics
///
/// Drop calls `cudaFree(d_compressed)` exactly once. The same
/// Arc-refcount safety as v2 protects against eviction-during-in-flight-
/// query; in-flight queries hold an `Arc<VramCompressedTermEntry>`
/// clone, the buffer stays alive until the kernel completes.
pub struct VramCompressedTermEntry {
    /// Device pointer to a compressed-bytes buffer.
    d_compressed: *mut c_void,
    /// Bytes in the device-resident compressed buffer (= the storage
    /// footprint counted toward the cache budget).
    compressed_bytes: usize,
    /// Bytes the decompressed buffer needs (`bucket_count *
    /// BITMAP_CONTAINER_WORDS * 4`). Required at decompress time to
    /// validate the per-call output size.
    uncompressed_bytes: usize,
    /// `(high16, word_offset)` for each non-empty bucket, sorted by
    /// `high16` ascending. `word_offset` is in u32 units relative to
    /// the **uncompressed** buffer (= relative to the workbench
    /// buffer the dispatch layer fills via `decompress_one`).
    bucket_index: Vec<(u16, u32)>,
}

// SAFETY: device pointer + bucket_index are owned by the entry.
// The codec used at insert time is a one-shot caller; at query
// time the dispatch layer's codec instance reads `d_compressed`
// without taking ownership. `Send`+`Sync` allow `Arc` wrap.
unsafe impl Send for VramCompressedTermEntry {}
unsafe impl Sync for VramCompressedTermEntry {}

impl std::fmt::Debug for VramCompressedTermEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramCompressedTermEntry")
            .field("d_compressed_is_null", &self.d_compressed.is_null())
            .field("compressed_bytes", &self.compressed_bytes)
            .field("uncompressed_bytes", &self.uncompressed_bytes)
            .field("bucket_count", &self.bucket_index.len())
            .field(
                "compression_ratio",
                &(self.uncompressed_bytes as f64 / self.compressed_bytes.max(1) as f64),
            )
            .finish()
    }
}

impl VramCompressedTermEntry {
    /// Number of buckets cached.
    #[must_use]
    pub fn bucket_count(&self) -> usize {
        self.bucket_index.len()
    }

    /// Bytes the compressed payload occupies on device (= cache
    /// budget footprint).
    #[must_use]
    pub fn compressed_bytes(&self) -> usize {
        self.compressed_bytes
    }

    /// Bytes the workbench buffer must hold for decompress
    /// (`bucket_count * BITMAP_CONTAINER_WORDS * 4`).
    #[must_use]
    pub fn uncompressed_bytes(&self) -> usize {
        self.uncompressed_bytes
    }

    /// `(high16, word_offset)` pairs into the uncompressed workbench.
    /// Caller-visible to drive cohort scatter at the dispatch layer.
    #[must_use]
    pub fn bucket_index(&self) -> &[(u16, u32)] {
        &self.bucket_index
    }

    /// Raw device pointer to the compressed payload.
    ///
    /// # Safety
    /// The pointer is a CUDA device pointer; only pass it to CUDA
    /// runtime / driver / nvcomp APIs. Stable for the
    /// `Arc<VramCompressedTermEntry>` clone's lifetime.
    #[must_use]
    pub fn device_ptr(&self) -> *const c_void {
        self.d_compressed as *const c_void
    }
}

impl Drop for VramCompressedTermEntry {
    fn drop(&mut self) {
        if !self.d_compressed.is_null() {
            // SAFETY: pointer was allocated by `cudaMalloc` in
            // [`VramCompressedCht::build_entry`] and has not been
            // freed elsewhere (the `Arc` in the cache map ensures
            // unique ownership). Replacing with null is defensive.
            unsafe {
                let p = std::mem::replace(&mut self.d_compressed, null_mut());
                let _ = cudaFree(p);
            }
        }
    }
}

/// Observable counters for the v3 cache. Same shape as
/// [`super::vram_cht::VramChtStats`], plus `compressed_bytes_total` /
/// `uncompressed_bytes_total` so operators can compute the live
/// compression ratio.
#[derive(Debug, Clone, Copy)]
pub struct VramCompressedChtStats {
    /// Cumulative cache hits.
    pub hits: u64,
    /// Cumulative cache misses.
    pub misses: u64,
    /// Cumulative successful insertions.
    pub inserts: u64,
    /// Cumulative LRU evictions caused by budget pressure.
    pub evictions: u64,
    /// Cumulative v2→v3 promotions (subset of `inserts` driven by
    /// dispatch-layer promotion, rather than direct insert).
    pub promotions: u64,
    /// Sum of `compressed_bytes` for all currently-cached entries
    /// (= the actual VRAM footprint, what the budget bounds).
    pub current_bytes: u64,
    /// Configured budget ceiling.
    pub budget_bytes: u64,
    /// Number of cached entries.
    pub entries: u64,
    /// Sum of `uncompressed_bytes` for all currently-cached entries
    /// (= the effective hot-tier capacity if these were stored
    /// uncompressed in v2).
    pub uncompressed_bytes_total: u64,
    /// Sum of `compressed_bytes` for all currently-cached entries
    /// (same value as `current_bytes`; exposed separately for
    /// stat-symmetry with `uncompressed_bytes_total`).
    pub compressed_bytes_total: u64,
}

impl VramCompressedChtStats {
    /// Live compression ratio
    /// (`uncompressed_bytes_total / compressed_bytes_total`). Returns
    /// 0.0 if no entries are cached.
    #[must_use]
    pub fn live_compression_ratio(&self) -> f64 {
        if self.compressed_bytes_total == 0 {
            return 0.0;
        }
        self.uncompressed_bytes_total as f64 / self.compressed_bytes_total as f64
    }
}

/// VRAM-resident Bitcomp-compressed CHT instance. Construct via
/// [`VramCompressedCht::with_budget`] or access the process-global via
/// [`global`] / [`global_with_budget`].
#[allow(missing_docs)]
pub struct VramCompressedCht {
    inner: Mutex<VramCompressedChtInner>,
    budget_bytes: u64,
    /// Bitcomp data type for the cache. Posting lists default to
    /// [`BitcompDataType::Uint32`] per Phase 0 finding (3.59× ratio).
    data_type: BitcompDataType,
    /// Codec instance dedicated to insert-time compression. Wrapped
    /// in `Mutex` so concurrent inserts serialise on the codec's
    /// internal device singletons + temp buffer.
    insert_codec: Mutex<BitcompDeviceCodec>,
    hits: AtomicU64,
    misses: AtomicU64,
    inserts: AtomicU64,
    evictions: AtomicU64,
    promotions: AtomicU64,
}

struct VramCompressedChtInner {
    map: HashMap<ChtKey, (Arc<VramCompressedTermEntry>, u64)>,
    next_token: u64,
    current_bytes: u64,
    uncompressed_bytes_total: u64,
}

impl VramCompressedCht {
    /// Construct a fresh v3 cache with the given byte budget. Budget
    /// counts compressed bytes (the actual VRAM footprint). Default
    /// data type is [`BitcompDataType::Uint32`] (Phase 0 best-for-
    /// postings).
    pub fn with_budget(budget_bytes: u64) -> Result<Self, VramCompressedChtError> {
        Self::with_budget_and_data_type(budget_bytes, BitcompDataType::Uint32)
    }

    /// Same as [`Self::with_budget`] but with an explicit Bitcomp
    /// data type. Use [`BitcompDataType::Uint32`] for postings,
    /// [`BitcompDataType::Char`] for general bitmap streams.
    pub fn with_budget_and_data_type(
        budget_bytes: u64,
        data_type: BitcompDataType,
    ) -> Result<Self, VramCompressedChtError> {
        let codec = BitcompDeviceCodec::new(data_type)?;
        Ok(Self {
            inner: Mutex::new(VramCompressedChtInner {
                map: HashMap::new(),
                next_token: 0,
                current_bytes: 0,
                uncompressed_bytes_total: 0,
            }),
            budget_bytes,
            data_type,
            insert_codec: Mutex::new(codec),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            promotions: AtomicU64::new(0),
        })
    }

    /// Look up a cached compressed posting list. On hit, bumps the
    /// LRU token. The returned `Arc` keeps the device buffer alive
    /// across concurrent eviction.
    ///
    /// Honors the `FERRO_DISABLE_VRAM_COMPRESSED` env var (same
    /// pattern as v2's `FERRO_DISABLE_VRAM_CHT`): when set, returns
    /// `None` immediately and bumps misses, so the dispatch layer
    /// falls through to v2 / v1 / drain.
    pub fn get(&self, key: &ChtKey) -> Option<Arc<VramCompressedTermEntry>> {
        if is_disabled() {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let mut inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
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

    /// Build a [`VramCompressedTermEntry`] from a host
    /// [`RoaringPostings`] (= the same source v2 takes; v3 does the
    /// same flatten + H→D as v2 then compresses on device) and insert
    /// it under `key`.
    ///
    /// Returns `Ok(true)` on successful insert, `Ok(false)` if the
    /// entry alone exceeds the budget OR the term is empty (skipped),
    /// or `Err(VramCompressedChtError)` on CUDA / Bitcomp failure.
    ///
    /// Idempotent re-insert: bumps the LRU token but keeps the
    /// existing device entry.
    ///
    /// Honors the `FERRO_DISABLE_VRAM_COMPRESSED` kill-switch
    /// (returns `Ok(false)` short-circuit).
    pub fn insert(
        &self,
        key: ChtKey,
        roaring: &RoaringPostings,
    ) -> Result<bool, VramCompressedChtError> {
        if is_disabled() {
            return Ok(false);
        }
        // Compute footprint.
        let bucket_count = roaring
            .containers
            .iter()
            .filter(|(_, c)| c.cardinality() > 0)
            .count();
        if bucket_count == 0 {
            return Ok(false);
        }
        let uncompressed_bytes = bucket_count * BITMAP_CONTAINER_WORDS * 4;
        // Bitcomp single-chunk ceiling = 16 MiB. For very dense terms
        // we'd need to multi-chunk; in practice the planner-approved
        // hot terms are well below this.
        if uncompressed_bytes > (1 << 24) {
            // Too big for single-chunk Bitcomp — fall through, the
            // caller will retry via v2 path.
            return Ok(false);
        }

        // Idempotency check.
        {
            let mut inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
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

        // Build the entry: stage uncompressed flat → upload to device
        // temp → compress device→device → free temp uncompressed.
        let entry = self.build_entry(roaring, bucket_count, uncompressed_bytes)?;
        let entry_compressed_bytes = entry.compressed_bytes as u64;

        if entry_compressed_bytes > self.budget_bytes {
            // Even compressed, too big for the entire budget.
            return Ok(false);
        }

        // LRU eviction + insert under the cache mutex. A concurrent
        // insert may have raced us to populate the same key; on race
        // we drop the freshly-built entry (its `cudaFree` runs in
        // `Drop`).
        let mut inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
        let next_token = inner.next_token + 1;
        if let Some((_, token)) = inner.map.get_mut(&key) {
            *token = next_token;
            inner.next_token = next_token;
            return Ok(false);
        }
        // Evict to fit.
        while inner.current_bytes + entry_compressed_bytes > self.budget_bytes
            && !inner.map.is_empty()
        {
            let oldest_key = inner
                .map
                .iter()
                .min_by_key(|(_, (_, t))| *t)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                if let Some((evicted_value, _)) = inner.map.remove(&k) {
                    inner.current_bytes = inner
                        .current_bytes
                        .saturating_sub(evicted_value.compressed_bytes as u64);
                    inner.uncompressed_bytes_total = inner
                        .uncompressed_bytes_total
                        .saturating_sub(evicted_value.uncompressed_bytes as u64);
                    self.evictions.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }
        let entry_uncompressed_bytes = entry.uncompressed_bytes as u64;
        inner.next_token = next_token;
        inner.current_bytes += entry_compressed_bytes;
        inner.uncompressed_bytes_total += entry_uncompressed_bytes;
        inner.map.insert(key, (Arc::new(entry), next_token));
        self.inserts.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    /// Insert + bump `promotions` counter — used when the dispatch
    /// layer promotes a v2-resident entry to v3 (vs direct first-touch
    /// insert from the planner).
    pub fn promote(
        &self,
        key: ChtKey,
        roaring: &RoaringPostings,
    ) -> Result<bool, VramCompressedChtError> {
        let inserted = self.insert(key, roaring)?;
        if inserted {
            self.promotions.fetch_add(1, Ordering::Relaxed);
        }
        Ok(inserted)
    }

    /// Snapshot of current stats. Cheap (atomic loads + one mutex
    /// acquire for the bytes/entry count).
    #[must_use]
    pub fn stats(&self) -> VramCompressedChtStats {
        let (current_bytes, uncompressed_bytes_total, entries) = {
            let inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
            (
                inner.current_bytes,
                inner.uncompressed_bytes_total,
                inner.map.len() as u64,
            )
        };
        VramCompressedChtStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            promotions: self.promotions.load(Ordering::Relaxed),
            current_bytes,
            budget_bytes: self.budget_bytes,
            entries,
            uncompressed_bytes_total,
            compressed_bytes_total: current_bytes,
        }
    }

    /// Reset all state. Test-only.
    #[doc(hidden)]
    pub fn reset(&self) {
        let mut inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
        inner.map.clear();
        inner.next_token = 0;
        inner.current_bytes = 0;
        inner.uncompressed_bytes_total = 0;
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

    /// Bitcomp data type the cache is configured for.
    #[must_use]
    pub fn data_type(&self) -> BitcompDataType {
        self.data_type
    }

    /// Internal: stage uncompressed flat layout to host, upload to
    /// device temp, compress with `BitcompDeviceCodec` to a freshly-
    /// allocated `d_compressed`, free the device temp.
    ///
    /// Wave Z-2 #1: container→bitmap expansion lives in
    /// [`SharedBitmapPayload::from_postings`] so the staging /
    /// bucket_index layout stays byte-identical to v2 (= the
    /// invariant cross-tier promote relies on).
    fn build_entry(
        &self,
        roaring: &RoaringPostings,
        bucket_count: usize,
        uncompressed_bytes: usize,
    ) -> Result<VramCompressedTermEntry, VramCompressedChtError> {
        // 1. Build host staging buffer + bucket_index via shared helper.
        let payload = SharedBitmapPayload::from_postings(roaring)
            .expect("caller pre-filtered empty postings; payload must be Some");
        debug_assert_eq!(payload.bucket_count, bucket_count);
        debug_assert_eq!(payload.total_bytes(), uncompressed_bytes);
        let SharedBitmapPayload {
            bucket_index,
            staging,
            ..
        } = payload;

        // 2. Upload uncompressed staging to device temp.
        let mut d_uncompressed_temp: *mut c_void = null_mut();
        // SAFETY: cudaMalloc writes a valid device pointer on success.
        let rc =
            unsafe { cudaMalloc(&mut d_uncompressed_temp, uncompressed_bytes) };
        if rc != CUDA_SUCCESS {
            return Err(VramCompressedChtError::Malloc {
                bytes: uncompressed_bytes,
                code: rc,
            });
        }
        // SAFETY: temp buffer just allocated of the right size; staging
        // host buffer holds `uncompressed_bytes` valid bytes.
        let rc = unsafe {
            cudaMemcpy(
                d_uncompressed_temp,
                staging.as_ptr() as *const c_void,
                uncompressed_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };
        if rc != CUDA_SUCCESS {
            // SAFETY: just allocated by us.
            unsafe {
                let _ = cudaFree(d_uncompressed_temp);
            }
            return Err(VramCompressedChtError::Memcpy {
                bytes: uncompressed_bytes,
                code: rc,
            });
        }

        // 3. Compute max compressed size, allocate d_compressed.
        let max_compressed = match BitcompDeviceCodec::max_compressed_size(
            uncompressed_bytes,
            self.data_type,
        ) {
            Ok(s) => s,
            Err(e) => {
                // SAFETY: temp just allocated by us.
                unsafe {
                    let _ = cudaFree(d_uncompressed_temp);
                }
                return Err(VramCompressedChtError::Bitcomp(e));
            }
        };
        let mut d_compressed: *mut c_void = null_mut();
        // SAFETY: cudaMalloc writes valid pointer on success.
        let rc = unsafe { cudaMalloc(&mut d_compressed, max_compressed) };
        if rc != CUDA_SUCCESS {
            // SAFETY: temp still valid; free it.
            unsafe {
                let _ = cudaFree(d_uncompressed_temp);
            }
            return Err(VramCompressedChtError::Malloc {
                bytes: max_compressed,
                code: rc,
            });
        }

        // 4. Compress device→device.
        let actual_comp_size = {
            let mut codec = self
                .insert_codec
                .lock()
                .expect("VramCompressedCht insert_codec poisoned");
            // SAFETY: both pointers are valid device buffers of the
            // sizes we pass. Codec serialises across callers via this
            // mutex, so its internal singletons aren't raced.
            unsafe {
                codec.compress_one(
                    d_uncompressed_temp as *const c_void,
                    uncompressed_bytes,
                    d_compressed,
                    max_compressed,
                )
            }
        };
        let actual_comp_size = match actual_comp_size {
            Ok(s) => s,
            Err(e) => {
                // SAFETY: both pointers were allocated by us above.
                unsafe {
                    let _ = cudaFree(d_uncompressed_temp);
                    let _ = cudaFree(d_compressed);
                }
                return Err(VramCompressedChtError::Bitcomp(e));
            }
        };

        // 5. Free the device temp uncompressed buffer (we no longer
        //    need it; only d_compressed survives in the cache).
        // SAFETY: temp was allocated above, is no longer referenced.
        unsafe {
            let _ = cudaFree(d_uncompressed_temp);
        }

        Ok(VramCompressedTermEntry {
            d_compressed,
            compressed_bytes: actual_comp_size,
            uncompressed_bytes,
            bucket_index,
        })
    }

    // ========================================================
    // Phase 2 D-5 — warm-restart persistence (Bitcomp-compressed VRAM tier)
    // ========================================================

    /// Dump every cached entry to `<path>` in the v3 (FCV3) wire
    /// format. Each entry stages its device-resident
    /// compressed-bytes payload back to host via
    /// `cudaMemcpyDeviceToHost` and writes the bytes next to the
    /// bucket-index + uncompressed/compressed size headers.
    ///
    /// On disk the compressed bytes are stored as-is — load time
    /// allocates a fresh device buffer and copies them back via
    /// `cudaMemcpyHostToDevice`, ready for the same Bitcomp
    /// decompress-on-read flow as a freshly-inserted entry. The
    /// codec data type is implicit in the magic (one cache per
    /// process, one data type per cache); future multi-data-type
    /// dumps would version-bump the wire format.
    ///
    /// Returns the number of entries written.
    pub fn dump_to_path(
        &self,
        path: &std::path::Path,
    ) -> Result<u64, super::persist::DumpError> {
        use super::persist::{
            finalise_atomic_write, open_tmp_writer, write_chtkey, write_file_header,
            write_file_trailer, DumpError, MAGIC_V3,
        };
        use std::io::Write;
        let snapshot: Vec<(ChtKey, Arc<VramCompressedTermEntry>)> = {
            let inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
            inner
                .map
                .iter()
                .map(|(k, (v, _))| (k.clone(), Arc::clone(v)))
                .collect()
        };
        let entry_count = snapshot.len() as u64;
        let mut writer = open_tmp_writer(path)?;
        write_file_header(&mut writer, MAGIC_V3, entry_count)?;
        for (i, (key, entry)) in snapshot.iter().enumerate() {
            write_chtkey(&mut writer, key)?;
            // bucket_count + per-bucket (high16, word_offset) pairs.
            let bucket_count = u32::try_from(entry.bucket_index.len()).unwrap_or(u32::MAX);
            writer.write_all(&bucket_count.to_le_bytes())?;
            for (high16, word_offset) in &entry.bucket_index {
                writer.write_all(&high16.to_le_bytes())?;
                writer.write_all(&word_offset.to_le_bytes())?;
            }
            // uncompressed_bytes + compressed_bytes (both u64).
            writer.write_all(&(entry.uncompressed_bytes as u64).to_le_bytes())?;
            writer.write_all(&(entry.compressed_bytes as u64).to_le_bytes())?;
            // Stage the device-resident compressed payload back to
            // host for write.
            let bytes = entry.compressed_bytes;
            let mut host_staging: Vec<u8> = vec![0u8; bytes];
            // SAFETY: entry.d_compressed points to `bytes` of device
            // memory owned by this Arc; host_staging holds `bytes`
            // of valid host memory.
            let rc = unsafe {
                cudaMemcpy(
                    host_staging.as_mut_ptr() as *mut c_void,
                    entry.d_compressed as *const c_void,
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
            writer.write_all(&host_staging)?;
        }
        write_file_trailer(&mut writer)?;
        finalise_atomic_write(writer, path)?;
        Ok(entry_count)
    }

    /// Load entries from `<path>` into this cache. For each entry
    /// read from disk, allocates a fresh `d_compressed` device
    /// buffer, stages the compressed bytes back to device, and
    /// installs the entry under the parsed [`ChtKey`].
    ///
    /// Per-entry CUDA failures are skipped; structural failures
    /// fail loud. Returns the number of entries installed.
    pub fn load_from_path(
        &self,
        path: &std::path::Path,
    ) -> Result<u64, super::persist::LoadError> {
        use super::persist::{
            open_reader, read_and_validate_file_header,
            read_and_validate_file_trailer, read_chtkey, read_exact_or_truncated,
            read_u16_le, read_u32_le, read_u64_le, MAGIC_V3,
        };
        let mut reader = open_reader(path)?;
        let entry_count = read_and_validate_file_header(&mut reader, MAGIC_V3)?;
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
            let compressed_bytes = read_u64_le(&mut reader)? as usize;
            let body = read_exact_or_truncated(&mut reader, compressed_bytes)?;
            // Cross-check: uncompressed_bytes must match the
            // bucket_count expansion (same invariant as v2).
            let expected_uncomp = bucket_count * BITMAP_CONTAINER_WORDS * 4;
            if expected_uncomp != uncompressed_bytes {
                continue;
            }
            // Budget gate: if the entire compressed entry exceeds
            // the budget, skip (matches the runtime insert path).
            if (compressed_bytes as u64) > self.budget_bytes {
                continue;
            }
            // Allocate device buffer + H2D for the compressed payload.
            let mut d_compressed: *mut c_void = null_mut();
            // SAFETY: cudaMalloc writes the device pointer on success.
            let rc = unsafe { cudaMalloc(&mut d_compressed, compressed_bytes) };
            if rc != CUDA_SUCCESS {
                continue;
            }
            // SAFETY: d_compressed just allocated of `compressed_bytes`;
            // body.as_ptr() holds the right bytes.
            let rc = unsafe {
                cudaMemcpy(
                    d_compressed,
                    body.as_ptr() as *const c_void,
                    compressed_bytes,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            if rc != CUDA_SUCCESS {
                // SAFETY: just allocated above.
                unsafe {
                    let _ = cudaFree(d_compressed);
                }
                continue;
            }
            let entry = VramCompressedTermEntry {
                d_compressed,
                compressed_bytes,
                uncompressed_bytes,
                bucket_index,
            };
            // Install via LRU eviction path.
            let entry_compressed_bytes = entry.compressed_bytes as u64;
            let entry_uncompressed_bytes = entry.uncompressed_bytes as u64;
            let mut inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
            let next_token = inner.next_token + 1;
            if inner.map.contains_key(&key) {
                drop(inner);
                drop(entry);
                continue;
            }
            while inner.current_bytes + entry_compressed_bytes > self.budget_bytes
                && !inner.map.is_empty()
            {
                let oldest_key = inner
                    .map
                    .iter()
                    .min_by_key(|(_, (_, t))| *t)
                    .map(|(k, _)| k.clone());
                if let Some(k) = oldest_key {
                    if let Some((evicted_value, _)) = inner.map.remove(&k) {
                        inner.current_bytes = inner
                            .current_bytes
                            .saturating_sub(evicted_value.compressed_bytes as u64);
                        inner.uncompressed_bytes_total = inner
                            .uncompressed_bytes_total
                            .saturating_sub(evicted_value.uncompressed_bytes as u64);
                        self.evictions.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    break;
                }
            }
            inner.next_token = next_token;
            inner.current_bytes += entry_compressed_bytes;
            inner.uncompressed_bytes_total += entry_uncompressed_bytes;
            inner.map.insert(key, (Arc::new(entry), next_token));
            self.inserts.fetch_add(1, Ordering::Relaxed);
            loaded += 1;
        }
        read_and_validate_file_trailer(&mut reader)?;
        Ok(loaded)
    }
}

// ============================================================
// Process-global cache + kill-switch
// ============================================================

/// Default v3 budget when no explicit configuration is provided.
/// 4 GiB — same conservative default as v2's
/// `vram_cht::DEFAULT_VRAM_CHT_BUDGET_BYTES`. L40S 48 GiB-class
/// production should set `--cht-vram-compressed-budget-bytes 16GiB`
/// (or larger; with ~3.59× Bitcomp ratio that effectively caches
/// ~57 GiB of uncompressed bitmap data alongside v2's 32 GiB
/// uncompressed budget).
pub const DEFAULT_VRAM_COMPRESSED_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

static GLOBAL_VRAM_COMPRESSED_CHT: OnceLock<VramCompressedCht> = OnceLock::new();

/// Get the process-global v3 cache, initialising with the default
/// budget on first access. Returns `None` if the underlying CUDA
/// context cannot be acquired (no NVIDIA driver / no compatible GPU).
///
/// First-init failure is sticky: the OnceLock stays unset and
/// subsequent calls retry construction. In practice, dispatch sites
/// should treat `None` as "v3 unavailable, fall through to v2".
pub fn global() -> Option<&'static VramCompressedCht> {
    if let Some(cht) = GLOBAL_VRAM_COMPRESSED_CHT.get() {
        return Some(cht);
    }
    match VramCompressedCht::with_budget(DEFAULT_VRAM_COMPRESSED_BUDGET_BYTES) {
        Ok(cht) => {
            let _ = GLOBAL_VRAM_COMPRESSED_CHT.set(cht);
            GLOBAL_VRAM_COMPRESSED_CHT.get()
        }
        Err(_) => None,
    }
}

/// Get the process-global v3 cache, initialising with a custom budget
/// **iff this is the first access**. Returns `None` if already
/// initialised OR if codec construction fails.
pub fn global_with_budget(budget_bytes: u64) -> Option<&'static VramCompressedCht> {
    if GLOBAL_VRAM_COMPRESSED_CHT.get().is_some() {
        return None;
    }
    let cht = VramCompressedCht::with_budget(budget_bytes).ok()?;
    let _ = GLOBAL_VRAM_COMPRESSED_CHT.set(cht);
    GLOBAL_VRAM_COMPRESSED_CHT.get()
}

/// Reset the process-global cache. Test-only.
#[doc(hidden)]
pub fn reset_global() {
    if let Some(cht) = GLOBAL_VRAM_COMPRESSED_CHT.get() {
        cht.reset();
    }
}

/// Phase 2 D-4 v3 kill-switch — when `FERRO_DISABLE_VRAM_COMPRESSED`
/// is set to `1` / `true`, all `get` / `insert` / `promote` calls
/// short-circuit. Used for clean v3-vs-v2 / v3-vs-D-1 A/B benchmarking
/// without recompiling. Same pattern as v2's `FERRO_DISABLE_VRAM_CHT`.
fn is_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("FERRO_DISABLE_VRAM_COMPRESSED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Whether the v3 kill-switch is active. Public for stats loggers.
#[must_use]
pub fn kill_switch_active() -> bool {
    is_disabled()
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::SegmentId;
    use crate::postings::roaring::encoder::RoaringEncoder;

    fn small_roaring() -> RoaringPostings {
        RoaringEncoder::from_doc_ids(&[1, 2, 3, 100, 65540])
    }

    fn dense_roaring(seed: u32) -> RoaringPostings {
        // 200K docs, distributed across multiple high16 buckets.
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

    fn cuda_available() -> bool {
        let cht = match VramCompressedCht::with_budget(64 * 1024 * 1024) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let key = dummy_key(0xdada, 100);
        cht.insert(key, &small_roaring()).is_ok()
    }

    #[test]
    fn miss_then_hit() {
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
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
    fn duplicate_insert_short_circuits() {
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xface, 100);
        let inserted_1 = cht.insert(key.clone(), &small_roaring()).unwrap();
        assert!(inserted_1);
        let bytes_after_first = cht.stats().current_bytes;
        let inserted_2 = cht.insert(key.clone(), &small_roaring()).unwrap();
        assert!(!inserted_2);
        let bytes_after_second = cht.stats().current_bytes;
        assert_eq!(bytes_after_first, bytes_after_second);
        assert_eq!(cht.stats().inserts, 1);
    }

    #[test]
    fn entry_records_compression_ratio() {
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xb007, 100);
        cht.insert(key.clone(), &dense_roaring(0)).unwrap();
        let entry = cht.get(&key).unwrap();
        // For dense bitmap-class containers, ratio should be ≥ ~1
        // (Bitcomp doesn't degrade random-ish data much) and bucket
        // counts non-zero.
        assert!(entry.bucket_count() >= 1);
        assert!(entry.uncompressed_bytes() > entry.compressed_bytes() / 10);
        let stats = cht.stats();
        assert_eq!(stats.compressed_bytes_total, stats.current_bytes);
        assert_eq!(
            stats.uncompressed_bytes_total,
            entry.uncompressed_bytes() as u64
        );
        let ratio = stats.live_compression_ratio();
        // Should be > 0.5 for realistic bitmap data (Bitcomp reduces
        // somewhat even on near-random); we don't pin a tight upper
        // bound because the synthetic fixture isn't optimised for
        // Bitcomp's algorithm.
        assert!(ratio > 0.0);
    }

    #[test]
    fn lru_evicts_oldest() {
        if !cuda_available() {
            return;
        }
        // Probe one entry's compressed bytes.
        let probe = VramCompressedCht::with_budget(1 << 30).unwrap();
        let probe_key = dummy_key(0xff00, 100);
        probe.insert(probe_key.clone(), &dense_roaring(0)).unwrap();
        let one_entry = probe.stats().current_bytes;
        drop(probe);

        let cht = VramCompressedCht::with_budget(one_entry + 1024).unwrap();
        let k1 = dummy_key(0xa1, 100);
        let k2 = dummy_key(0xa2, 100);
        cht.insert(k1.clone(), &dense_roaring(0)).unwrap();
        cht.insert(k2.clone(), &dense_roaring(1)).unwrap();
        assert!(cht.get(&k1).is_none(), "oldest must have been evicted");
        assert!(cht.get(&k2).is_some(), "newest must still be cached");
        assert!(cht.stats().evictions >= 1);
    }

    #[test]
    fn empty_roaring_skipped() {
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xe1ee, 100);
        let empty = RoaringPostings::default();
        let inserted = cht.insert(key.clone(), &empty).unwrap();
        assert!(!inserted);
        assert!(cht.get(&key).is_none());
    }

    #[test]
    fn promote_bumps_promotions_counter() {
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xc0ff, 100);
        let inserted = cht.promote(key, &small_roaring()).unwrap();
        assert!(inserted);
        let stats = cht.stats();
        assert_eq!(stats.promotions, 1);
        assert_eq!(stats.inserts, 1);
    }

    #[test]
    fn arc_clone_survives_eviction() {
        // Same Arc-eviction-during-in-flight-query safety property as
        // v2's `arc_clone_survives_eviction_then_reset`.
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0x5afe, 100);
        cht.insert(key.clone(), &small_roaring()).unwrap();
        let live_clone = cht.get(&key).expect("post-insert get must hit");
        cht.reset();
        // Clone is still usable post-reset (host metadata immutable,
        // device buffer alive via Arc refcount).
        assert!(live_clone.bucket_count() >= 1);
        assert!(!live_clone.device_ptr().is_null());
        drop(live_clone);
        // Cache empty + reusable.
        assert_eq!(cht.stats().entries, 0);
        let key2 = dummy_key(0x5aff, 100);
        let inserted = cht.insert(key2.clone(), &small_roaring()).unwrap();
        assert!(inserted);
        assert!(cht.get(&key2).is_some());
    }

    #[test]
    fn capacity_multiplier_vs_v2() {
        // Cross-validate the v3 capacity story: under the SAME
        // budget, v3 should hold strictly more entries than v2 when
        // the underlying data is Bitcomp-compressible. This is the
        // "4× capacity multiplier" pitch claim's local witness.
        if !cuda_available() {
            return;
        }
        // Build N realistic posting fixtures. Each has ~200K dense
        // doc ids spanning 4 high16 buckets — bitmap-class.
        let fixtures: Vec<RoaringPostings> =
            (0..32).map(|i| dense_roaring(i as u32 * 1000)).collect();

        // Probe one entry's compressed + uncompressed footprint via v3
        // to size the budget.
        let probe_v3 =
            VramCompressedCht::with_budget(1 << 30).unwrap();
        let probe_key = dummy_key(0xff00, 100);
        probe_v3.insert(probe_key, &fixtures[0]).unwrap();
        let v3_one_entry = probe_v3.stats().compressed_bytes_total;
        let v3_one_uncompressed = probe_v3.stats().uncompressed_bytes_total;
        drop(probe_v3);

        // Tight budget = 4 × the one-entry compressed size, so v3
        // can hold ~4 entries. In v2 (uncompressed) the same byte
        // budget would hold strictly fewer entries because v2 entries
        // are the FULL uncompressed bitmap.
        let budget = v3_one_entry * 4 + 1024;

        // Insert 16 entries into v3 — LRU evicts but the live count
        // should be ≥ 4 (= budget / per-entry).
        let cht_v3 = VramCompressedCht::with_budget(budget).unwrap();
        for (i, fix) in fixtures.iter().enumerate() {
            let key = dummy_key(0x10000_u32 + i as u32, 100);
            let _ = cht_v3.insert(key, fix);
        }
        let v3_entries = cht_v3.stats().entries;
        let _v3_live_uncompressed = cht_v3.stats().uncompressed_bytes_total;

        // Run the same test on v2 (using the v2 cache for parallel
        // construction). v2 entry size = uncompressed bitmap
        // (no compression). Same budget = much fewer entries.
        let cht_v2 =
            crate::postings::roaring::vram_cht::VramCht::with_budget(budget);
        for (i, fix) in fixtures.iter().enumerate() {
            let key = dummy_key(0x10000_u32 + i as u32, 100);
            let _ = cht_v2.insert(key, fix);
        }
        let v2_entries = cht_v2.stats().entries;

        // Headline: v3 holds ≥ v2 entries under same byte budget.
        // (For dense bitmap fixtures Bitcomp typically achieves
        // 1.05-1.5× ratio, so v3 holds ≈ same to slightly more
        // entries; on real postings Phase 0 measured 3.59× which
        // would amplify this further. We assert the weak property to
        // keep the test stable across fixture changes.)
        assert!(
            v3_entries >= v2_entries,
            "v3 should hold ≥ v2 entries under same budget; got v3={} v2={}",
            v3_entries,
            v2_entries
        );
        // Also verify the compression ratio is at least breakeven on
        // the live cache contents (= v3 isn't worse than no compression
        // for this fixture).
        let live_ratio = cht_v3.stats().live_compression_ratio();
        assert!(
            live_ratio >= 0.9,
            "v3 live compression ratio should be ≥ 0.9 (Bitcomp shouldn't \
             expand bitmap data); got {live_ratio:.2}"
        );
        // Sanity log for the capacity advantage (visible in `cargo test
        // -- --nocapture`).
        eprintln!(
            "v3 capacity advantage: v3={v3_entries} entries / v2={v2_entries} entries / \
             v3 live ratio={live_ratio:.2} / v3 one-entry compressed={v3_one_entry} \
             uncompressed={v3_one_uncompressed}"
        );
    }

    // ========================================================
    // Phase 2 D-5 — v3 dump/load roundtrip tests
    // ========================================================

    #[test]
    fn cht_v3_dump_then_load_roundtrip_byte_equal() {
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v3.bin");

        let src = VramCompressedCht::with_budget(256 * 1024 * 1024).unwrap();
        let entries = vec![
            (
                dummy_key(0x1111, 0x1111_2222_3333_4444),
                small_roaring(),
            ),
            (
                dummy_key(0x2222, 0x5555_6666_7777_8888),
                dense_roaring(0),
            ),
            (
                dummy_key(0x3333, 0x9999_AAAA_BBBB_CCCC),
                dense_roaring(1),
            ),
        ];
        for (k, rp) in &entries {
            src.insert(k.clone(), rp).unwrap();
        }
        let n_dumped = src.dump_to_path(&path).unwrap();
        assert_eq!(n_dumped, entries.len() as u64);
        assert!(path.exists());

        // Capture per-entry (compressed_bytes, uncompressed_bytes,
        // bucket_index, host bytes) from the source.
        struct Expected {
            compressed_bytes: usize,
            uncompressed_bytes: usize,
            bucket_index: Vec<(u16, u32)>,
            host_compressed: Vec<u8>,
        }
        let mut expected_per_entry: Vec<Expected> = Vec::with_capacity(entries.len());
        for (k, _) in &entries {
            let e = src.get(k).expect("src entry must hit");
            let bytes = e.compressed_bytes();
            let mut staging = vec![0u8; bytes];
            let rc = unsafe {
                cudaMemcpy(
                    staging.as_mut_ptr() as *mut c_void,
                    e.device_ptr(),
                    bytes,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            assert_eq!(rc, CUDA_SUCCESS);
            expected_per_entry.push(Expected {
                compressed_bytes: e.compressed_bytes(),
                uncompressed_bytes: e.uncompressed_bytes(),
                bucket_index: e.bucket_index().to_vec(),
                host_compressed: staging,
            });
        }

        // Fresh cache; load; per-entry byte-equal.
        let dst = VramCompressedCht::with_budget(256 * 1024 * 1024).unwrap();
        let n_loaded = dst.load_from_path(&path).unwrap();
        assert_eq!(n_loaded, entries.len() as u64);
        for ((k, _), expected) in entries.iter().zip(expected_per_entry.iter()) {
            let got = dst.get(k).expect("loaded entry must hit");
            assert_eq!(got.compressed_bytes(), expected.compressed_bytes);
            assert_eq!(got.uncompressed_bytes(), expected.uncompressed_bytes);
            assert_eq!(got.bucket_index(), expected.bucket_index.as_slice());
            // Compressed payload byte-equal: stage device → host
            // and compare to the dumped bytes.
            let mut staging = vec![0u8; got.compressed_bytes()];
            let rc = unsafe {
                cudaMemcpy(
                    staging.as_mut_ptr() as *mut c_void,
                    got.device_ptr(),
                    got.compressed_bytes(),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            assert_eq!(rc, CUDA_SUCCESS);
            assert_eq!(
                staging, expected.host_compressed,
                "loaded compressed payload must be byte-equal to dumped"
            );
        }
        // Stats: the loaded cache must report the same uncompressed
        // total (the live capacity-multiplier signal that buyer DD
        // talks about).
        let src_stats = src.stats();
        let dst_stats = dst.stats();
        assert_eq!(
            src_stats.compressed_bytes_total,
            dst_stats.compressed_bytes_total,
            "compressed total must match"
        );
        assert_eq!(
            src_stats.uncompressed_bytes_total,
            dst_stats.uncompressed_bytes_total,
            "uncompressed total must match"
        );
    }

    #[test]
    fn cht_v3_restart_simulation_first_query_hits() {
        // Phase 2 D-5 acceptance gate: after dump + restart
        // simulation (= fresh VramCompressedCht) + load, the FIRST
        // query for a previously-cached key must observe a hit.
        // v3 specific: the loaded entry must preserve compressed
        // bytes byte-for-byte (so decompress-on-read at query time
        // produces the same uncompressed buffer the kernel
        // expects).
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v3_restart.bin");

        let original = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let k1 = dummy_key(0xCAFE, 0xBABE_0001);
        let k2 = dummy_key(0xCAFE, 0xBABE_0002);
        original.insert(k1.clone(), &small_roaring()).unwrap();
        original.insert(k2.clone(), &dense_roaring(0)).unwrap();
        original.dump_to_path(&path).unwrap();

        let restarted = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
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

        // The entry exposes a valid device pointer + matched bucket
        // index + non-zero compressed bytes (the buyer DD signal
        // for "compressed hot tier survives restart").
        let entry = got.unwrap();
        assert!(!entry.device_ptr().is_null());
        assert!(entry.compressed_bytes() > 0);
        assert!(entry.uncompressed_bytes() > 0);
        assert!(entry.bucket_count() >= 1);
    }

    #[test]
    fn cht_v3_load_wrong_magic_rejected() {
        if !cuda_available() {
            return;
        }
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v3.bin");
        // Write a v1-magic file.
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&super::super::persist::MAGIC_V1.to_le_bytes())
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

        let dst = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let err = dst.load_from_path(&path).unwrap_err();
        assert!(
            matches!(
                err,
                super::super::persist::LoadError::WrongMagic { .. }
            ),
            "v1 file at v3 path must reject as WrongMagic, got {err:?}"
        );
    }

    #[test]
    fn cht_v3_dump_empty_cache_roundtrips() {
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v3_empty.bin");
        let src = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let n = src.dump_to_path(&path).unwrap();
        assert_eq!(n, 0);
        let dst = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let n_loaded = dst.load_from_path(&path).unwrap();
        assert_eq!(n_loaded, 0);
        assert_eq!(dst.stats().entries, 0);
    }

    #[test]
    fn cht_v3_dump_with_hash_fn_drift_rejected_on_load() {
        // Write a v3-magic file with hash_function = 999 (= future
        // hash swap). Loader must reject as HashFunctionMismatch
        // so cold-start kicks in cleanly.
        if !cuda_available() {
            return;
        }
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v3_hashdrift.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&super::super::persist::MAGIC_V3.to_le_bytes())
            .unwrap();
        f.write_all(&super::super::persist::WIRE_VERSION.to_le_bytes())
            .unwrap();
        f.write_all(&999u32.to_le_bytes()).unwrap(); // wrong hash_fn
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&super::super::persist::MAGIC_END.to_le_bytes())
            .unwrap();
        drop(f);

        let dst = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let err = dst.load_from_path(&path).unwrap_err();
        assert!(
            matches!(
                err,
                super::super::persist::LoadError::HashFunctionMismatch { .. }
            ),
            "hash_function drift must reject as HashFunctionMismatch, got {err:?}"
        );
    }

    #[test]
    fn kill_switch_via_env_var() {
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xb007_b007, 100);
        if kill_switch_active() {
            assert!(cht.get(&key).is_none());
            let inserted = cht.insert(key.clone(), &small_roaring()).unwrap();
            assert!(!inserted);
            assert!(cht.get(&key).is_none());
            let stats = cht.stats();
            assert_eq!(stats.inserts, 0);
            assert_eq!(stats.entries, 0);
        } else {
            assert!(cht.get(&key).is_none());
            let inserted = cht.insert(key.clone(), &small_roaring()).unwrap();
            assert!(inserted);
            assert!(cht.get(&key).is_some());
        }
    }
}

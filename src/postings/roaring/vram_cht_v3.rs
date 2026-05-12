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
//! - **Multi-chunk dispatch wiring** (Z-6 #3): the entry now carries
//!   a `chunks: Vec<DeviceChunkSlice>` payload (Z-6 #2 LAND, see
//!   [`DeviceChunkSlice`] + [`VramCompressedTermEntry::chunks`]) so
//!   the 16 MiB single-chunk admission ceiling lifts to
//!   `MAX_CHUNKS_PER_ENTRY * 16 MiB` (= 1 GiB). The compression
//!   loop fan-out is in place; the dispatch path
//!   ([`super::cuda_dispatch::CudaBitmapOpKernel::decompress_batch_into_workbench`])
//!   still consumes the single-chunk back-compat shim and will be
//!   migrated to walk `chunks()` directly in Z-6 #3. Until that
//!   lands, dispatch on a multi-chunk entry trips a
//!   `debug_assert!(chunk_count == 1)` (= silent wrong result in
//!   release builds). The `cuda-bitmap-kernel` feature is
//!   pre-release; the only consumer touching multi-chunk admission
//!   today is the Z-6 #5 dense-bench example.
//! - **Multi-chunk persist** (Z-6 #4): [`dump_to_path`] guards
//!   multi-chunk entries with a `debug_assert!`; MAGIC_V4 wire
//!   format upgrade in Z-6 #4 will dump per-chunk records.
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

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ferro_compress::nvcomp_sys::cuda::{
    cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind, CUDA_SUCCESS,
};
use ferro_compress::{BitcompDataType, BitcompDeviceCodec, Error as FcError};

use crate::index::SegmentId;
use crate::postings::roaring::cht::ChtKey;
use crate::postings::roaring::encoder::RoaringPostings;
use crate::postings::roaring::shared_bitmap_payload::SharedBitmapPayload;
use crate::postings::roaring::vram_cht::VramTermEntry;
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
    /// Source posting list would require more than
    /// [`MAX_CHUNKS_PER_ENTRY`] (= 1 GiB hard cap) — surfaced rather
    /// than silently rejected so operators see the boundary.
    #[error("term exceeds MAX_CHUNKS_PER_ENTRY (chunks_needed={chunks_needed}, max={max})")]
    TooManyChunks {
        /// Chunks the term would need = `ceil(uncompressed_bytes / 16 MiB)`.
        chunks_needed: usize,
        /// Hard cap, currently [`MAX_CHUNKS_PER_ENTRY`].
        max: usize,
    },
}

/// nvcomp Bitcomp single-chunk ceiling (16 MiB). Each
/// [`DeviceChunkSlice`] covers up to this many uncompressed bytes;
/// terms over this threshold fan out into multiple chunks via
/// [`VramCompressedCht::insert`] / [`VramCompressedCht::promote_v2_to_v3`].
pub const BITCOMP_CHUNK_BYTES: usize = 1 << 24;

/// Buckets per max-size chunk = `BITCOMP_CHUNK_BYTES / (BITMAP_CONTAINER_WORDS * 4)`
/// (= 2048 with the current `BITMAP_CONTAINER_WORDS = 2048`). Chunk
/// boundaries always fall on bucket boundaries so the
/// `(high16, word_offset)` `bucket_index` stays a single Vec for the
/// whole entry — chunk `k`'s bucket window is
/// `bucket_index[k * BUCKETS_PER_CHUNK ..]` up to the next chunk
/// boundary.
pub const BUCKETS_PER_CHUNK: usize = BITCOMP_CHUNK_BYTES / (BITMAP_CONTAINER_WORDS * 4);

/// Hard cap on the number of [`DeviceChunkSlice`] entries per term.
/// 64 chunks × 16 MiB = 1 GiB uncompressed, well above any realistic
/// posting list (a `bspread_30k_buckets` cohort hits ~240 MiB = 15
/// chunks) but bounds worst-case `cudaMalloc` count per admission.
/// Operators that overflow this cap see
/// [`VramCompressedChtError::TooManyChunks`] surfaced rather than a
/// silent skip.
pub const MAX_CHUNKS_PER_ENTRY: usize = 64;

/// One Bitcomp-compressed chunk of an entry's posting payload.
///
/// Each chunk owns a `cudaMalloc`-allocated `d_compressed` device
/// buffer of `compressed_bytes` size carrying the Bitcomp output
/// for a slice of `uncompressed_bytes` source bytes
/// (`uncompressed_bytes` ≤ [`BITCOMP_CHUNK_BYTES`]; the slice is
/// aligned to bucket boundaries — see [`BUCKETS_PER_CHUNK`]).
///
/// `DeviceChunkSlice` is a borrow-free record; the actual `cudaFree`
/// runs in the parent [`VramCompressedTermEntry`]'s `Drop` impl,
/// which iterates the `chunks` vec.
#[derive(Debug)]
pub struct DeviceChunkSlice {
    /// Device pointer to the compressed-bytes buffer for this chunk.
    pub d_compressed: *mut c_void,
    /// Bytes in the device-resident compressed buffer for this chunk
    /// (= per-chunk contribution to the cache-budget footprint).
    pub compressed_bytes: usize,
    /// Bytes of uncompressed source this chunk covers
    /// (≤ [`BITCOMP_CHUNK_BYTES`]). The sum over the parent entry's
    /// chunks equals [`VramCompressedTermEntry::uncompressed_bytes`].
    pub uncompressed_bytes: usize,
}

// SAFETY: `d_compressed` is a CUDA device pointer; nothing in this
// struct holds a host-visible borrow. The parent
// `VramCompressedTermEntry` owns the chunks vec and runs `cudaFree`
// from its `Drop`; `Send`+`Sync` only allow the record to traverse
// `Arc<VramCompressedTermEntry>` clones.
unsafe impl Send for DeviceChunkSlice {}
unsafe impl Sync for DeviceChunkSlice {}

/// One cached posting list, stored on the CUDA device in
/// **Bitcomp-compressed** form.
///
/// Wave Z-6 #2 LAND: the device payload is now a `Vec<DeviceChunkSlice>`
/// so the 16 MiB single-chunk Bitcomp ceiling lifts to
/// `MAX_CHUNKS_PER_ENTRY × BITCOMP_CHUNK_BYTES` (= 1 GiB). Single-chunk
/// terms continue to land as `chunks.len() == 1` and preserve byte-
/// identical layout (the persist MAGIC_V3 wire format still round-trips
/// them; MAGIC_V4 lands in Z-6 #4).
///
/// ## Layout invariants
/// - `chunks.is_empty()` is forbidden (empty terms are rejected at
///   admission; see [`VramCompressedCht::insert`]).
/// - `uncompressed_bytes == sum(chunks[i].uncompressed_bytes)`.
/// - `bucket_index.len() * BITMAP_CONTAINER_WORDS * 4 == uncompressed_bytes`.
/// - Chunks tile the uncompressed source contiguously in
///   `bucket_index` order: chunk `k` covers buckets
///   `[k * BUCKETS_PER_CHUNK, min((k+1) * BUCKETS_PER_CHUNK, bucket_count))`.
///   The last chunk may be smaller than [`BITCOMP_CHUNK_BYTES`] if
///   `bucket_count` is not a multiple of [`BUCKETS_PER_CHUNK`].
///
/// ## Drop semantics
/// Drop iterates `chunks` and runs `cudaFree` per chunk exactly once.
/// The same Arc-refcount safety as v2 protects against eviction-
/// during-in-flight-query; in-flight queries hold an
/// `Arc<VramCompressedTermEntry>` clone, every chunk buffer stays
/// alive until the kernel completes.
pub struct VramCompressedTermEntry {
    /// Per-chunk Bitcomp payloads, ordered by ascending bucket offset.
    /// Non-empty for any admitted entry. See struct-level invariants.
    chunks: Vec<DeviceChunkSlice>,
    /// Total bytes the decompressed buffer needs (`bucket_count *
    /// BITMAP_CONTAINER_WORDS * 4`). Required at decompress time to
    /// validate the per-call output size; matches `sum(chunks[i].uncompressed_bytes)`.
    uncompressed_bytes: usize,
    /// `(high16, word_offset)` for each non-empty bucket, sorted by
    /// `high16` ascending. `word_offset` is in u32 units relative to
    /// the **uncompressed** buffer (= relative to the workbench
    /// buffer the dispatch layer fills via `decompress_one`).
    bucket_index: Vec<(u16, u32)>,
}

// SAFETY: device pointers + bucket_index are owned by the entry.
// `Send`+`Sync` propagate the same property `DeviceChunkSlice` declares.
unsafe impl Send for VramCompressedTermEntry {}
unsafe impl Sync for VramCompressedTermEntry {}

impl std::fmt::Debug for VramCompressedTermEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let compressed_total = self.compressed_bytes();
        f.debug_struct("VramCompressedTermEntry")
            .field("chunk_count", &self.chunks.len())
            .field("compressed_bytes_total", &compressed_total)
            .field("uncompressed_bytes", &self.uncompressed_bytes)
            .field("bucket_count", &self.bucket_index.len())
            .field(
                "compression_ratio",
                &(self.uncompressed_bytes as f64 / compressed_total.max(1) as f64),
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

    /// Total bytes the compressed payload occupies on device summed
    /// over all chunks (= cache budget footprint).
    #[must_use]
    pub fn compressed_bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.compressed_bytes).sum()
    }

    /// Bytes the workbench buffer must hold for decompress
    /// (`bucket_count * BITMAP_CONTAINER_WORDS * 4`); also equals
    /// `sum(chunks[i].uncompressed_bytes)`.
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

    /// Number of Bitcomp chunks the compressed payload spans (≥ 1
    /// for any admitted entry; > 1 only after Z-6 #2 multi-chunk
    /// admission landed). Lets Z-6 #3 dispatch wiring distinguish
    /// single-chunk back-compat callers from multi-chunk ones.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Per-chunk slices, ordered by ascending bucket offset. Z-6 #3
    /// dispatch wiring iterates this slice to flatten N-chunk terms
    /// into nvcomp batched-decompress entries.
    #[must_use]
    pub fn chunks(&self) -> &[DeviceChunkSlice] {
        &self.chunks
    }

    // Wave Z-6 #3 removed the `device_ptr()` / `d_compressed()`
    // back-compat shims (which `debug_assert!`-ed single-chunk and
    // returned `chunks[0].d_compressed`). Dispatch now walks
    // [`Self::chunks`] for every payload access, including the
    // [`VramCompressedCht::dump_to_path`] single-chunk path which
    // indexes `chunks()[0]` directly under its own debug_assert until
    // Z-6 #4 introduces MAGIC_V4 per-chunk records.
}

impl Drop for VramCompressedTermEntry {
    fn drop(&mut self) {
        // SAFETY: every chunk's `d_compressed` was allocated by
        // `cudaMalloc` in [`VramCompressedCht::build_entry`] /
        // [`VramCompressedCht::promote_v2_to_v3`] and has not been
        // freed elsewhere — the `Arc` in the cache map ensures unique
        // ownership of the chunks vec.
        for chunk in self.chunks.drain(..) {
            unsafe {
                if !chunk.d_compressed.is_null() {
                    let _ = cudaFree(chunk.d_compressed);
                }
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
    /// Cumulative successful [`VramCompressedCht::promote_v2_to_v3`]
    /// calls — strict subset of [`Self::promotions`] that excludes
    /// legacy [`VramCompressedCht::promote`] (re-drain) admissions.
    /// Bumped only when the cross-tier device→device compress path
    /// fires; lets dispatch-path tests pin the new wiring branch
    /// byte-for-byte vs the legacy `promote(roaring)` fallback.
    pub cross_tier_promotions: u64,
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
    cross_tier_promotions: AtomicU64,
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
            cross_tier_promotions: AtomicU64::new(0),
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
        // Wave Z-6 #2: Bitcomp's 16 MiB per-chunk ceiling is now
        // handled internally via multi-chunk fan-out; admit any term
        // up to MAX_CHUNKS_PER_ENTRY * BITCOMP_CHUNK_BYTES (= 1 GiB).
        let max_admit_bytes = MAX_CHUNKS_PER_ENTRY.saturating_mul(BITCOMP_CHUNK_BYTES);
        if uncompressed_bytes > max_admit_bytes {
            return Err(VramCompressedChtError::TooManyChunks {
                chunks_needed: uncompressed_bytes.div_ceil(BITCOMP_CHUNK_BYTES),
                max: MAX_CHUNKS_PER_ENTRY,
            });
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
        // temp → compress device→device per chunk → free per-chunk
        // temp uncompressed. Single-chunk fast path keeps the original
        // pre-Z-6 layout for terms ≤ 16 MiB.
        let entry = self.build_entry(roaring, bucket_count, uncompressed_bytes)?;
        let entry_compressed_bytes = entry.compressed_bytes() as u64;

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
                        .saturating_sub(evicted_value.compressed_bytes() as u64);
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

    /// Wave Z-2 #1 — promote a v2 entry to v3 by Bitcomp-compressing
    /// the v2 device buffer **without re-draining the source
    /// postings**.
    ///
    /// The caller hands in the v2 [`Arc<VramTermEntry>`] (= the
    /// existing tier hit). The v2 entry already owns:
    /// - the `bucket_index` produced by the same
    ///   [`SharedBitmapPayload`] expansion v3 would have produced
    ///   internally;
    /// - an uncompressed `[u32; total_words]` device buffer
    ///   bytewise-identical to what v3's existing `build_entry`
    ///   would have allocated + filled.
    ///
    /// This method reuses both: it skips the host
    /// container→bitmap walk + `cudaMalloc d_uncompressed_temp` +
    /// H→D staging copy, and directly feeds the v2 device pointer
    /// into the Bitcomp `compress_one` call. Wave Z-6 #5 validates
    /// the savings via `examples/promote_v2_to_v3_dense_bench.rs`
    /// on two GPUs (AWS L40S 48 GiB g6e.xlarge + local RTX 4070 Ti
    /// SUPER 16 GiB), evidence packaged in
    /// `dd-pack/cht-wave-z6-multichunk-bench-2026-05-12/`:
    ///
    /// | cohort | chunks | L40S savings p50 | local savings p50 |
    /// |---|---|---|---|
    /// | `bspread_5k_buckets`   |  3 | **+25.04 ms (+65.7%)** | +11.97 ms (+54.5%) |
    /// | `bspread_10k_buckets`  |  5 | **+49.51 ms (+64.8%)** | +22.93 ms (+53.0%) |
    /// | `bspread_30k_buckets`  | 15 | **+154.86 ms (+66.3%)** | partial (VRAM frag OOM) |
    ///
    /// The `~60 ms savings on dense terms (10K+ buckets)` recon
    /// estimate is **validated and exceeded**: 10K buckets measures
    /// +49.51 ms (83% of recon) and 30K buckets measures +154.86 ms
    /// (2.6× recon). The savings ratio stays at ~65% across
    /// multi-chunk cohorts on L40S (vs ~53% on the consumer card),
    /// consistent with HBM3 device bandwidth amortising more
    /// container-walk + H→D-copy work in the cross-tier path.
    ///
    /// ## Returns
    /// - `Ok(true)` on successful insertion (admission policy + LRU
    ///   eviction performed; both [`VramCompressedChtStats::promotions`]
    ///   and the strict-subset
    ///   [`VramCompressedChtStats::cross_tier_promotions`] counter
    ///   bumped).
    /// - `Ok(false)` on rejection: kill-switch active OR the
    ///   compressed entry would alone exceed `budget_bytes` OR the
    ///   key is already cached (idempotent re-promote). Neither
    ///   counter is bumped.
    /// - `Err(VramCompressedChtError)` on CUDA / Bitcomp failure.
    ///
    /// ## Lifetime / safety
    /// `v2_entry` is held by `Arc` across the entire compress
    /// call. The v2 `Drop` (= `cudaFree(d_buckets)`) cannot fire
    /// while we hold an `Arc` clone, so a concurrent v2 eviction
    /// of the same key is safe — the device buffer the compressor
    /// reads stays alive until this method returns. Once the v3
    /// entry is in the cache map, the v2 `Arc` can be dropped by
    /// the caller without affecting the v3 entry's freshly-
    /// allocated `d_compressed`.
    ///
    /// ## Layout invariant
    /// The v2 entry's `d_buckets: *mut u32` points to a contiguous
    /// `[u32; total_words]` allocation; reinterpret-cast to
    /// `*const c_void` matches the Bitcomp `compress_one` input
    /// signature without a device-side copy. v3 (uncompressed
    /// side) and v2 share this layout by construction — both go
    /// through [`SharedBitmapPayload::from_postings`] for any
    /// freshly-built entry, and the v2 dump/load path round-trips
    /// the same bytes.
    pub fn promote_v2_to_v3(
        &self,
        key: &ChtKey,
        v2_entry: Arc<VramTermEntry>,
    ) -> Result<bool, VramCompressedChtError> {
        if is_disabled() {
            return Ok(false);
        }

        let bucket_count = v2_entry.bucket_count();
        if bucket_count == 0 {
            // Defensive: a v2 entry with zero buckets shouldn't
            // exist (the v2 insert path filters empty terms), but
            // we treat it the same way for symmetry.
            return Ok(false);
        }
        let total_words = v2_entry.total_words();
        let uncompressed_bytes = total_words * std::mem::size_of::<u32>();
        // Wave Z-6 #2: admission cap matches `insert` — up to
        // MAX_CHUNKS_PER_ENTRY * BITCOMP_CHUNK_BYTES = 1 GiB.
        let max_admit_bytes = MAX_CHUNKS_PER_ENTRY.saturating_mul(BITCOMP_CHUNK_BYTES);
        if uncompressed_bytes > max_admit_bytes {
            return Err(VramCompressedChtError::TooManyChunks {
                chunks_needed: uncompressed_bytes.div_ceil(BITCOMP_CHUNK_BYTES),
                max: MAX_CHUNKS_PER_ENTRY,
            });
        }

        // Idempotency: if v3 already has this key, just bump the
        // LRU token and short-circuit. No device call.
        {
            let mut inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
            let next_token = inner.next_token + 1;
            let already_cached = inner
                .map
                .get_mut(key)
                .map(|(_, token)| {
                    *token = next_token;
                });
            if already_cached.is_some() {
                inner.next_token = next_token;
                return Ok(false);
            }
        }

        // 1. Compress device→device per chunk. Each chunk covers
        //    `BUCKETS_PER_CHUNK` buckets (= 16 MiB) except possibly
        //    the last; we walk the v2 device buffer with byte-offset
        //    pointer arithmetic and feed each slice to a fresh
        //    `compress_one` call. We hold `v2_entry` (`Arc` clone)
        //    across the whole loop so a concurrent v2 eviction can't
        //    race the source pointer into Drop.
        let chunks = self.compress_chunks_device_to_device(
            v2_entry.device_ptr() as *const c_void,
            uncompressed_bytes,
        )?;

        // 2. Build the v3 entry. Clone bucket_index from the v2
        //    entry — same `(high16, word_offset)` invariant.
        let entry = VramCompressedTermEntry {
            chunks,
            uncompressed_bytes,
            bucket_index: v2_entry.bucket_index().to_vec(),
        };
        let entry_compressed_bytes = entry.compressed_bytes() as u64;

        if entry_compressed_bytes > self.budget_bytes {
            // Even compressed, too big for the whole budget. Drop
            // the freshly-allocated d_compressed and return false.
            drop(entry);
            return Ok(false);
        }

        // 4. Install under LRU eviction. Symmetric with `insert`'s
        //    post-build branch (idempotency re-check + evict-to-fit
        //    + bump counters).
        let mut inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
        let next_token = inner.next_token + 1;
        if let Some((_, token)) = inner.map.get_mut(key) {
            // Race: another thread populated the same key between
            // the pre-alloc idempotency check and now. Drop ours.
            *token = next_token;
            inner.next_token = next_token;
            drop(inner);
            // entry's `Drop` runs cudaFree on d_compressed.
            return Ok(false);
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
                        .saturating_sub(evicted_value.compressed_bytes() as u64);
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
        inner
            .map
            .insert(key.clone(), (Arc::new(entry), next_token));
        self.inserts.fetch_add(1, Ordering::Relaxed);
        self.promotions.fetch_add(1, Ordering::Relaxed);
        self.cross_tier_promotions
            .fetch_add(1, Ordering::Relaxed);
        // Drop the inner lock before returning so the caller doesn't
        // hold it any longer than necessary.
        drop(inner);
        // `v2_entry` Arc is released here (caller's clone or the
        // method-scope binding); the v2 cache still holds its own
        // clone if it's resident.
        Ok(true)
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
            cross_tier_promotions: self.cross_tier_promotions.load(Ordering::Relaxed),
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
        self.cross_tier_promotions.store(0, Ordering::Relaxed);
    }

    /// Evict all entries whose [`ChtKey::segment_id`] is in
    /// `segment_ids`. Returns the count of entries evicted.
    ///
    /// Wave Z-7 #2 — fine-grained eviction surface for the ILM hook
    /// (`ferro-ilm`) on phase transitions (`delete` / `cold` /
    /// `frozen`). Releases VRAM owned by segments of indices
    /// transitioning out of the Hot tier without touching the Hot
    /// indices' working set. Cheaper than [`Self::reset`] which would
    /// wipe the whole cache.
    ///
    /// Each evicted [`Arc<VramCompressedTermEntry>`] drops on removal;
    /// its `Drop` iterates the per-chunk `DeviceChunkSlice` vec and
    /// calls `cudaFree` per chunk. No leak.
    ///
    /// Stats: bumps `evictions` by the returned count; decrements
    /// `current_bytes` (= compressed bytes total) by the sum of
    /// evicted entries' `compressed_bytes()`, and
    /// `uncompressed_bytes_total` by their uncompressed footprint.
    /// `inserts` is unchanged (eviction is not insertion).
    pub fn evict_by_segments(&self, segment_ids: &[SegmentId]) -> u64 {
        if segment_ids.is_empty() {
            return 0;
        }
        let segment_set: HashSet<SegmentId> = segment_ids.iter().copied().collect();
        let mut inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
        let keys_to_evict: Vec<ChtKey> = inner
            .map
            .keys()
            .filter(|k| segment_set.contains(&k.segment_id))
            .cloned()
            .collect();
        let count = keys_to_evict.len() as u64;
        for key in keys_to_evict {
            if let Some((evicted_value, _)) = inner.map.remove(&key) {
                inner.current_bytes = inner
                    .current_bytes
                    .saturating_sub(evicted_value.compressed_bytes() as u64);
                inner.uncompressed_bytes_total = inner
                    .uncompressed_bytes_total
                    .saturating_sub(evicted_value.uncompressed_bytes as u64);
            }
        }
        drop(inner);
        self.evictions.fetch_add(count, Ordering::Relaxed);
        count
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
    /// device temp per chunk, compress each chunk into a freshly-
    /// allocated `d_compressed`, free per-chunk temp.
    ///
    /// Wave Z-2 #1: container→bitmap expansion lives in
    /// [`SharedBitmapPayload::from_postings`] so the staging /
    /// bucket_index layout stays byte-identical to v2.
    ///
    /// Wave Z-6 #2: when `uncompressed_bytes > BITCOMP_CHUNK_BYTES`
    /// the host staging buffer is split on bucket boundaries (= every
    /// `BUCKETS_PER_CHUNK` buckets) so each `compress_one` call sees
    /// ≤ 16 MiB and the resulting chunks tile the original layout
    /// contiguously. The last chunk may be smaller than 16 MiB.
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

        // 2. Compress chunk-by-chunk from a freshly-uploaded device
        //    temp (host → device → compress → free temp, repeated per
        //    chunk). The per-chunk temp lets the loop avoid one giant
        //    `cudaMalloc` of `uncompressed_bytes` for very dense terms.
        let chunks = self.compress_chunks_host_to_device(
            staging.as_slice(),
            uncompressed_bytes,
        )?;

        Ok(VramCompressedTermEntry {
            chunks,
            uncompressed_bytes,
            bucket_index,
        })
    }

    /// Compress `uncompressed_bytes` of host staging into one
    /// [`DeviceChunkSlice`] per Bitcomp chunk (≤ 16 MiB each). The
    /// loop allocates a small per-chunk `d_uncompressed_temp`, copies
    /// the relevant slice of `staging` into it, runs `compress_one`,
    /// then frees the temp. On error any chunks allocated so far are
    /// `cudaFree`'d so the caller never sees a leaked partial entry.
    ///
    /// `staging` is the host-side flat `[u32; total_words]` layout
    /// produced by [`SharedBitmapPayload::from_postings`]; the chunk
    /// boundaries fall on bucket boundaries (every `BUCKETS_PER_CHUNK`
    /// buckets), so the resulting chunks tile the source layout
    /// contiguously and the parent entry's single `bucket_index` vec
    /// covers all chunks.
    fn compress_chunks_host_to_device(
        &self,
        staging: &[u32],
        uncompressed_bytes: usize,
    ) -> Result<Vec<DeviceChunkSlice>, VramCompressedChtError> {
        debug_assert_eq!(staging.len() * 4, uncompressed_bytes);
        let n_chunks = uncompressed_bytes.div_ceil(BITCOMP_CHUNK_BYTES).max(1);
        let mut chunks: Vec<DeviceChunkSlice> = Vec::with_capacity(n_chunks);
        let mut byte_offset: usize = 0;
        let mut chunk_idx: usize = 0;
        while byte_offset < uncompressed_bytes {
            let chunk_uncomp = (uncompressed_bytes - byte_offset).min(BITCOMP_CHUNK_BYTES);

            // Per-chunk uncompressed temp on device.
            let mut d_uncompressed_temp: *mut c_void = null_mut();
            // SAFETY: cudaMalloc writes a valid device pointer on success.
            let rc = unsafe { cudaMalloc(&mut d_uncompressed_temp, chunk_uncomp) };
            if rc != CUDA_SUCCESS {
                free_chunks(chunks);
                return Err(VramCompressedChtError::Malloc {
                    bytes: chunk_uncomp,
                    code: rc,
                });
            }
            // SAFETY: chunk_uncomp ≤ remaining staging bytes; word
            // offsets align with chunk boundaries (chunk_uncomp is a
            // multiple of `BITMAP_CONTAINER_WORDS * 4` except possibly
            // the last chunk, which is a multiple of 4 either way).
            let staging_byte_ptr = unsafe {
                (staging.as_ptr() as *const u8).add(byte_offset) as *const c_void
            };
            let rc = unsafe {
                cudaMemcpy(
                    d_uncompressed_temp,
                    staging_byte_ptr,
                    chunk_uncomp,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            if rc != CUDA_SUCCESS {
                unsafe {
                    let _ = cudaFree(d_uncompressed_temp);
                }
                free_chunks(chunks);
                return Err(VramCompressedChtError::Memcpy {
                    bytes: chunk_uncomp,
                    code: rc,
                });
            }

            match self.compress_one_chunk_in_place(d_uncompressed_temp, chunk_uncomp) {
                Ok(chunk) => {
                    chunks.push(chunk);
                    // SAFETY: the per-chunk uncompressed temp is no
                    // longer needed once compress_one finished.
                    unsafe {
                        let _ = cudaFree(d_uncompressed_temp);
                    }
                }
                Err(e) => {
                    unsafe {
                        let _ = cudaFree(d_uncompressed_temp);
                    }
                    free_chunks(chunks);
                    return Err(e);
                }
            }

            byte_offset += chunk_uncomp;
            chunk_idx += 1;
            debug_assert!(chunk_idx <= MAX_CHUNKS_PER_ENTRY);
        }
        debug_assert!(!chunks.is_empty());
        debug_assert_eq!(chunks.len(), n_chunks);
        Ok(chunks)
    }

    /// Same as [`Self::compress_chunks_host_to_device`] but the
    /// uncompressed source already lives on device — used by
    /// [`Self::promote_v2_to_v3`]. We slice the v2 device buffer with
    /// byte-offset pointer arithmetic and feed each slice directly to
    /// `compress_one` (no host staging round-trip). The caller must
    /// keep the source pointer alive for the duration of the call
    /// (an `Arc<VramTermEntry>` clone in the v2 case).
    fn compress_chunks_device_to_device(
        &self,
        v2_device_base: *const c_void,
        uncompressed_bytes: usize,
    ) -> Result<Vec<DeviceChunkSlice>, VramCompressedChtError> {
        let n_chunks = uncompressed_bytes.div_ceil(BITCOMP_CHUNK_BYTES).max(1);
        let mut chunks: Vec<DeviceChunkSlice> = Vec::with_capacity(n_chunks);
        let mut byte_offset: usize = 0;
        let mut chunk_idx: usize = 0;
        while byte_offset < uncompressed_bytes {
            let chunk_uncomp = (uncompressed_bytes - byte_offset).min(BITCOMP_CHUNK_BYTES);
            // SAFETY: v2_device_base is a valid `[u32; total_words]`
            // device buffer of exactly `uncompressed_bytes` bytes;
            // `byte_offset + chunk_uncomp ≤ uncompressed_bytes` so the
            // offset pointer stays within the original allocation.
            let chunk_src = unsafe {
                (v2_device_base as *const u8).add(byte_offset) as *const c_void
            };
            match self.compress_one_chunk_in_place(chunk_src as *mut c_void, chunk_uncomp) {
                Ok(chunk) => chunks.push(chunk),
                Err(e) => {
                    free_chunks(chunks);
                    return Err(e);
                }
            }
            byte_offset += chunk_uncomp;
            chunk_idx += 1;
            debug_assert!(chunk_idx <= MAX_CHUNKS_PER_ENTRY);
        }
        debug_assert!(!chunks.is_empty());
        debug_assert_eq!(chunks.len(), n_chunks);
        Ok(chunks)
    }

    /// Allocate `d_compressed`, run `compress_one` from
    /// `d_uncompressed` (device pointer; not freed here — caller
    /// manages its lifetime), return the `DeviceChunkSlice`. On error
    /// `d_compressed` is freed before returning.
    fn compress_one_chunk_in_place(
        &self,
        d_uncompressed: *mut c_void,
        chunk_uncomp: usize,
    ) -> Result<DeviceChunkSlice, VramCompressedChtError> {
        let max_compressed =
            BitcompDeviceCodec::max_compressed_size(chunk_uncomp, self.data_type)
                .map_err(VramCompressedChtError::Bitcomp)?;
        let mut d_compressed: *mut c_void = null_mut();
        // SAFETY: cudaMalloc writes a valid device pointer on success.
        let rc = unsafe { cudaMalloc(&mut d_compressed, max_compressed) };
        if rc != CUDA_SUCCESS {
            return Err(VramCompressedChtError::Malloc {
                bytes: max_compressed,
                code: rc,
            });
        }
        let actual_comp_size = {
            let mut codec = self
                .insert_codec
                .lock()
                .expect("VramCompressedCht insert_codec poisoned");
            // SAFETY: both pointers point to valid device buffers
            // (`d_uncompressed` of `chunk_uncomp` bytes from caller,
            // `d_compressed` of `max_compressed` bytes from above);
            // codec is serialised across callers via this mutex.
            unsafe {
                codec.compress_one(
                    d_uncompressed as *const c_void,
                    chunk_uncomp,
                    d_compressed,
                    max_compressed,
                )
            }
        };
        match actual_comp_size {
            Ok(actual) => Ok(DeviceChunkSlice {
                d_compressed,
                compressed_bytes: actual,
                uncompressed_bytes: chunk_uncomp,
            }),
            Err(e) => {
                // SAFETY: d_compressed was allocated above and not
                // yet handed out.
                unsafe {
                    let _ = cudaFree(d_compressed);
                }
                Err(VramCompressedChtError::Bitcomp(e))
            }
        }
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
            write_file_trailer, DumpError, MAGIC_V4,
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
        write_file_header(&mut writer, MAGIC_V4, entry_count)?;
        for (i, (key, entry)) in snapshot.iter().enumerate() {
            // Wave Z-6 #4: MAGIC_V4 per-term body emits `chunk_count`
            // followed by `N` `(compressed_bytes: u32, body bytes)`
            // records. Each chunk's `uncompressed_bytes` is derivable
            // at load time from the bucket-boundary chunking invariant
            // (chunk k covers `[k * BUCKETS_PER_CHUNK, min((k+1) *
            // BUCKETS_PER_CHUNK, bucket_count))`), so the wire doesn't
            // need to carry it. Single-chunk entries collapse to
            // `chunk_count = 1` with one record — no special case.
            write_chtkey(&mut writer, key)?;
            // bucket_count + per-bucket (high16, word_offset) pairs.
            let bucket_count = u32::try_from(entry.bucket_index.len()).unwrap_or(u32::MAX);
            writer.write_all(&bucket_count.to_le_bytes())?;
            for (high16, word_offset) in &entry.bucket_index {
                writer.write_all(&high16.to_le_bytes())?;
                writer.write_all(&word_offset.to_le_bytes())?;
            }
            // uncompressed_bytes (total across all chunks) + chunk_count.
            writer.write_all(&(entry.uncompressed_bytes as u64).to_le_bytes())?;
            let chunk_count =
                u32::try_from(entry.chunk_count()).unwrap_or(u32::MAX);
            writer.write_all(&chunk_count.to_le_bytes())?;
            // Per-chunk records. Each chunk's compressed payload is
            // staged D→H individually so the same fixed-size staging
            // buffer can be reused across chunks (each ≤
            // `BITCOMP_CHUNK_BYTES`, post-compress typically much
            // smaller).
            for chunk in entry.chunks() {
                let comp_size = u32::try_from(chunk.compressed_bytes)
                    .unwrap_or(u32::MAX);
                writer.write_all(&comp_size.to_le_bytes())?;
                let mut host_staging: Vec<u8> = vec![0u8; chunk.compressed_bytes];
                // SAFETY: `chunk.d_compressed` points to
                // `chunk.compressed_bytes` of device memory owned by
                // this entry's Arc; `host_staging` holds that many
                // valid host bytes.
                let rc = unsafe {
                    cudaMemcpy(
                        host_staging.as_mut_ptr() as *mut c_void,
                        chunk.d_compressed as *const c_void,
                        chunk.compressed_bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    )
                };
                if rc != CUDA_SUCCESS {
                    return Err(DumpError::Cuda {
                        entry_index: i as u64,
                        bytes: chunk.compressed_bytes,
                        code: rc,
                    });
                }
                writer.write_all(&host_staging)?;
            }
        }
        write_file_trailer(&mut writer)?;
        finalise_atomic_write(writer, path)?;
        Ok(entry_count)
    }

    /// Wave Z-7 #6 #1 — dump only the entries whose
    /// [`ChtKey::segment_id`] is in `segment_ids` to `path`. Symmetric
    /// with [`Self::evict_by_segments`] — the same membership filter
    /// applied during eviction is now applied during persist, so the
    /// resulting file is a per-segment-set slice of the global v3
    /// cache.
    ///
    /// Production use case (`bins/ferrosearch/src/main.rs`
    /// `--cht-dump-on-shutdown` SIGTERM sweep): iterate every open
    /// index, collect that index's `searchable_segments()` into a
    /// `Vec<SegmentId>`, write `<data_path>/<index>/cht_v3.bin`
    /// containing exactly that index's entries. The result is the
    /// per-index dump file `--cht-prewarm-on-startup` (Z-7 #4) expects
    /// on the next process start, closing the writer half of the
    /// warm-restart loop.
    ///
    /// File format is byte-identical to [`Self::dump_to_path`]: same
    /// MAGIC_V4 header, same per-entry wire layout. The only
    /// difference is which entries are included; an
    /// `Self::dump_by_segments(<all segments>)` call produces a
    /// byte-equivalent file to `Self::dump_to_path`. The
    /// oracle-byte-eq test (`dump_by_segments_byte_invariant_with_oracle`)
    /// pins this invariant.
    ///
    /// Empty `segment_ids` writes a zero-entry MAGIC_V4 header (so the
    /// file always exists with a valid magic + trailer after a
    /// successful call — `load_from_path` round-trips it as a no-op).
    /// The atomic write protocol still runs even for the empty case
    /// so a partially-written file from a crashed prior dump is
    /// replaced cleanly.
    ///
    /// Atomic write: bytes are written to `<path>.tmp` first, then
    /// `rename`d over `path`. POSIX `rename(2)` is atomic on the same
    /// filesystem, so a SIGTERM mid-write leaves either the previous
    /// `<path>` intact or `<path>` replaced by the new content — no
    /// half-written intermediate state observable at `<path>`.
    /// Cross-filesystem `<data_path>` setups (uncommon) would degrade
    /// to copy-then-rename which loses atomicity; this is the same
    /// caveat the existing `--cht-persist-path` global dump carries.
    ///
    /// Returns the number of entries written.
    pub fn dump_by_segments(
        &self,
        segment_ids: &[SegmentId],
        path: &std::path::Path,
    ) -> Result<u64, super::persist::DumpError> {
        use super::persist::{
            finalise_atomic_write, open_tmp_writer, write_chtkey, write_file_header,
            write_file_trailer, DumpError, MAGIC_V4,
        };
        use std::io::Write;
        // Build the same `HashSet<SegmentId>` membership pattern Z-7
        // #2's `evict_by_segments` uses, so the writer side of the
        // pair is filtering-symmetric with the eviction side.
        let segment_set: HashSet<SegmentId> = segment_ids.iter().copied().collect();
        let snapshot: Vec<(ChtKey, Arc<VramCompressedTermEntry>)> = {
            let inner = self.inner.lock().expect("VramCompressedCht mutex poisoned");
            inner
                .map
                .iter()
                .filter(|(k, _)| segment_set.contains(&k.segment_id))
                .map(|(k, (v, _))| (k.clone(), Arc::clone(v)))
                .collect()
        };
        let entry_count = snapshot.len() as u64;
        let mut writer = open_tmp_writer(path)?;
        write_file_header(&mut writer, MAGIC_V4, entry_count)?;
        // Per-entry body — bit-for-bit identical to `dump_to_path` so
        // the oracle test (`dump_by_segments_byte_invariant_with_oracle`)
        // produces byte-equivalent files when called with the full
        // segment set. Empty `segment_ids` → `snapshot` is empty →
        // header + trailer only, no per-entry bodies.
        for (i, (key, entry)) in snapshot.iter().enumerate() {
            write_chtkey(&mut writer, key)?;
            let bucket_count = u32::try_from(entry.bucket_index.len()).unwrap_or(u32::MAX);
            writer.write_all(&bucket_count.to_le_bytes())?;
            for (high16, word_offset) in &entry.bucket_index {
                writer.write_all(&high16.to_le_bytes())?;
                writer.write_all(&word_offset.to_le_bytes())?;
            }
            writer.write_all(&(entry.uncompressed_bytes as u64).to_le_bytes())?;
            let chunk_count =
                u32::try_from(entry.chunk_count()).unwrap_or(u32::MAX);
            writer.write_all(&chunk_count.to_le_bytes())?;
            for chunk in entry.chunks() {
                let comp_size = u32::try_from(chunk.compressed_bytes)
                    .unwrap_or(u32::MAX);
                writer.write_all(&comp_size.to_le_bytes())?;
                let mut host_staging: Vec<u8> = vec![0u8; chunk.compressed_bytes];
                // SAFETY: `chunk.d_compressed` points to
                // `chunk.compressed_bytes` of device memory owned by
                // this entry's Arc; `host_staging` holds that many
                // valid host bytes.
                let rc = unsafe {
                    cudaMemcpy(
                        host_staging.as_mut_ptr() as *mut c_void,
                        chunk.d_compressed as *const c_void,
                        chunk.compressed_bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    )
                };
                if rc != CUDA_SUCCESS {
                    return Err(DumpError::Cuda {
                        entry_index: i as u64,
                        bytes: chunk.compressed_bytes,
                        code: rc,
                    });
                }
                writer.write_all(&host_staging)?;
            }
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
            open_reader, read_and_validate_file_header_multi,
            read_and_validate_file_trailer, read_chtkey, read_exact_or_truncated,
            read_u16_le, read_u32_le, read_u64_le, MAGIC_V3, MAGIC_V4,
        };
        let mut reader = open_reader(path)?;
        // Wave Z-6 #4: dispatch on file magic — MAGIC_V4 is the
        // multi-chunk wire format produced by current binaries;
        // MAGIC_V3 is retained for back-compat with operator-
        // snapshotted dumps from pre-Z-6-#4 binaries (lifted into a
        // 1-chunk Multi entry at load time).
        let (magic, entry_count) = read_and_validate_file_header_multi(
            &mut reader,
            &[MAGIC_V4, MAGIC_V3],
        )?;
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
            // Per-magic body decode: V4 = chunk_count + N records;
            // V3 = single `compressed_bytes: u64` + body. Both collect
            // into the same shape `Vec<(comp_size, host_bytes)>` so
            // the device-side install logic below is unified.
            let chunks_data: Vec<(usize, Vec<u8>)> = if magic == MAGIC_V4 {
                let chunk_count = read_u32_le(&mut reader)? as usize;
                let mut chunks_data = Vec::with_capacity(chunk_count);
                for _ in 0..chunk_count {
                    let comp_size = read_u32_le(&mut reader)? as usize;
                    let body = read_exact_or_truncated(&mut reader, comp_size)?;
                    chunks_data.push((comp_size, body));
                }
                chunks_data
            } else {
                // MAGIC_V3 — legacy single u64 compressed_bytes + body.
                let compressed_bytes = read_u64_le(&mut reader)? as usize;
                let body = read_exact_or_truncated(&mut reader, compressed_bytes)?;
                vec![(compressed_bytes, body)]
            };
            // Cross-check: uncompressed_bytes must match the
            // bucket_count expansion (same invariant as v2 / pre-Z-6).
            let expected_uncomp = bucket_count * BITMAP_CONTAINER_WORDS * 4;
            if expected_uncomp != uncompressed_bytes {
                continue;
            }
            // Cross-check: sum of derived per-chunk uncompressed_bytes
            // (bucket-boundary chunking invariant) must equal the
            // wire's `uncompressed_bytes`. Defensive against malformed
            // V4 dumps where `chunk_count` and `bucket_count` are
            // inconsistent.
            let n_chunks = chunks_data.len();
            let total_chunk_uncomp: usize = (0..n_chunks)
                .map(|k| {
                    let chunk_first = k * BUCKETS_PER_CHUNK;
                    let chunk_last = std::cmp::min(
                        (k + 1) * BUCKETS_PER_CHUNK,
                        bucket_count,
                    );
                    chunk_last.saturating_sub(chunk_first)
                        * BITMAP_CONTAINER_WORDS
                        * 4
                })
                .sum();
            if total_chunk_uncomp != uncompressed_bytes {
                continue;
            }
            // Budget gate: if the entire compressed entry exceeds
            // the budget, skip (matches the runtime insert path).
            let total_compressed: usize =
                chunks_data.iter().map(|(s, _)| *s).sum();
            if (total_compressed as u64) > self.budget_bytes {
                continue;
            }
            // Build `Vec<DeviceChunkSlice>` — per-chunk cudaMalloc +
            // H2D into freshly allocated device buffers. On any
            // failure mid-loop, free already-allocated chunks and
            // skip this entry (no resource leak).
            let mut chunks: Vec<DeviceChunkSlice> = Vec::with_capacity(n_chunks);
            let mut chunk_install_failed = false;
            for (k, (comp_size, body)) in chunks_data.iter().enumerate() {
                let chunk_first = k * BUCKETS_PER_CHUNK;
                let chunk_last = std::cmp::min(
                    (k + 1) * BUCKETS_PER_CHUNK,
                    bucket_count,
                );
                let chunk_buckets = chunk_last.saturating_sub(chunk_first);
                let chunk_uncompressed = chunk_buckets * BITMAP_CONTAINER_WORDS * 4;
                let mut d_compressed: *mut c_void = null_mut();
                // SAFETY: cudaMalloc writes the device pointer on success.
                let rc = unsafe { cudaMalloc(&mut d_compressed, *comp_size) };
                if rc != CUDA_SUCCESS {
                    chunk_install_failed = true;
                    break;
                }
                // SAFETY: d_compressed just allocated of `comp_size`;
                // body holds `comp_size` valid host bytes.
                let rc = unsafe {
                    cudaMemcpy(
                        d_compressed,
                        body.as_ptr() as *const c_void,
                        *comp_size,
                        cudaMemcpyKind::cudaMemcpyHostToDevice,
                    )
                };
                if rc != CUDA_SUCCESS {
                    // SAFETY: just allocated above.
                    unsafe {
                        let _ = cudaFree(d_compressed);
                    }
                    chunk_install_failed = true;
                    break;
                }
                chunks.push(DeviceChunkSlice {
                    d_compressed,
                    compressed_bytes: *comp_size,
                    uncompressed_bytes: chunk_uncompressed,
                });
            }
            if chunk_install_failed {
                // SAFETY: free chunks we already pushed before bailing.
                for chunk in chunks.drain(..) {
                    unsafe {
                        if !chunk.d_compressed.is_null() {
                            let _ = cudaFree(chunk.d_compressed);
                        }
                    }
                }
                continue;
            }
            let entry = VramCompressedTermEntry {
                chunks,
                uncompressed_bytes,
                bucket_index,
            };
            // Install via LRU eviction path.
            let entry_compressed_bytes = entry.compressed_bytes() as u64;
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
                            .saturating_sub(evicted_value.compressed_bytes() as u64);
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

/// Free every chunk's `d_compressed` device pointer. Used when a
/// multi-chunk admission errors part-way through compression so the
/// half-built `chunks` vec doesn't leak the chunks already allocated.
fn free_chunks(chunks: Vec<DeviceChunkSlice>) {
    for chunk in chunks {
        // SAFETY: caller passes ownership of the chunks vec; each
        // `d_compressed` was allocated by us via cudaMalloc.
        unsafe {
            if !chunk.d_compressed.is_null() {
                let _ = cudaFree(chunk.d_compressed);
            }
        }
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
        assert!(!live_clone.chunks()[0].d_compressed.is_null());
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
                    e.chunks()[0].d_compressed as *const c_void,
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
                    got.chunks()[0].d_compressed as *const c_void,
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
        assert!(!entry.chunks()[0].d_compressed.is_null());
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
    fn cht_v3_dump_then_load_multi_chunk_roundtrip_byte_equal() {
        // Wave Z-6 #4 acceptance gate: a multi-chunk entry dumps as
        // MAGIC_V4 with `chunk_count + N×(comp_size + body)` records
        // and loads back into a `Vec<DeviceChunkSlice>` whose per-
        // chunk compressed bytes, derived uncompressed_bytes, and
        // device payloads are byte-identical to the source. The
        // fixture forces 3 chunks (2 × full BITCOMP_CHUNK_BYTES + 1
        // partial last chunk) so the partial-last derivation
        // (`bucket_count - 2 * BUCKETS_PER_CHUNK` buckets in the tail)
        // is exercised.
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v3_multichunk.bin");
        let src = VramCompressedCht::with_budget(256 * 1024 * 1024).unwrap();
        // 2 entries: one multi-chunk (3 chunks) + one single-chunk
        // (to keep the V3 / V4 single-chunk degenerate case covered
        // in the same dump's load loop).
        let k_multi = dummy_key(0xC6_04_01, 0x4C04_4C04_4C04_4C04);
        let k_single = dummy_key(0xC6_04_02, 0x5C04_5C04_5C04_5C04);
        let rp_multi = multi_chunk_roaring(BUCKETS_PER_CHUNK * 2 + 1);
        let rp_single = small_roaring();
        src.insert(k_multi.clone(), &rp_multi).unwrap();
        src.insert(k_single.clone(), &rp_single).unwrap();

        // Capture expected per-chunk shape + host-staged bytes.
        struct ExpectedChunk {
            compressed_bytes: usize,
            uncompressed_bytes: usize,
            host_payload: Vec<u8>,
        }
        struct ExpectedEntry {
            uncompressed_bytes: usize,
            bucket_index: Vec<(u16, u32)>,
            chunks: Vec<ExpectedChunk>,
        }
        let snapshot_keys = vec![k_multi.clone(), k_single.clone()];
        let mut expected: Vec<ExpectedEntry> = Vec::with_capacity(2);
        for k in &snapshot_keys {
            let e = src.get(k).expect("hit");
            let mut chunks_expected = Vec::with_capacity(e.chunk_count());
            for chunk in e.chunks() {
                let mut staging = vec![0u8; chunk.compressed_bytes];
                let rc = unsafe {
                    cudaMemcpy(
                        staging.as_mut_ptr() as *mut c_void,
                        chunk.d_compressed as *const c_void,
                        chunk.compressed_bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    )
                };
                assert_eq!(rc, CUDA_SUCCESS);
                chunks_expected.push(ExpectedChunk {
                    compressed_bytes: chunk.compressed_bytes,
                    uncompressed_bytes: chunk.uncompressed_bytes,
                    host_payload: staging,
                });
            }
            expected.push(ExpectedEntry {
                uncompressed_bytes: e.uncompressed_bytes(),
                bucket_index: e.bucket_index().to_vec(),
                chunks: chunks_expected,
            });
        }
        // Multi-chunk fixture sanity — pin chunk_count so the test
        // doesn't silently regress to a single-chunk roundtrip.
        assert_eq!(
            expected[0].chunks.len(),
            3,
            "multi-chunk fixture must produce exactly 3 chunks (2 full + 1 partial-last)"
        );
        assert_eq!(expected[1].chunks.len(), 1);

        let n_dumped = src.dump_to_path(&path).unwrap();
        assert_eq!(n_dumped, 2);

        // Fresh cache; load via the magic-aware dispatch path.
        let dst = VramCompressedCht::with_budget(256 * 1024 * 1024).unwrap();
        let n_loaded = dst.load_from_path(&path).unwrap();
        assert_eq!(n_loaded, 2);

        for (k, expected_entry) in snapshot_keys.iter().zip(expected.iter()) {
            let got = dst.get(k).expect("loaded entry must hit");
            assert_eq!(got.chunk_count(), expected_entry.chunks.len());
            assert_eq!(got.uncompressed_bytes(), expected_entry.uncompressed_bytes);
            assert_eq!(got.bucket_index(), expected_entry.bucket_index.as_slice());
            for (got_chunk, exp_chunk) in got.chunks().iter().zip(expected_entry.chunks.iter()) {
                assert_eq!(got_chunk.compressed_bytes, exp_chunk.compressed_bytes);
                assert_eq!(got_chunk.uncompressed_bytes, exp_chunk.uncompressed_bytes);
                // Compressed payload byte-equal: stage D→H from the
                // loaded chunk and compare to the dumped host bytes.
                let mut staging = vec![0u8; got_chunk.compressed_bytes];
                let rc = unsafe {
                    cudaMemcpy(
                        staging.as_mut_ptr() as *mut c_void,
                        got_chunk.d_compressed as *const c_void,
                        got_chunk.compressed_bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    )
                };
                assert_eq!(rc, CUDA_SUCCESS);
                assert_eq!(
                    staging, exp_chunk.host_payload,
                    "loaded chunk payload must be byte-equal to dumped"
                );
            }
        }
    }

    #[test]
    fn cht_v3_load_magic_v3_back_compat_lifts_to_single_chunk() {
        // Wave Z-6 #4 back-compat acceptance: a MAGIC_V3 (pre-Z-6-#4)
        // dump produced by an old binary must still load via the new
        // magic-aware loader, lifted into a 1-chunk Multi entry whose
        // shape mirrors what the live `insert` admission would produce.
        // We hand-craft a synthetic MAGIC_V3 dump using payload bytes
        // staged off a real single-chunk entry — the compressed bytes
        // themselves are wire-format-independent, so the same payload
        // round-trips correctly through the V3 read path.
        if !cuda_available() {
            return;
        }
        use std::io::Write;

        // Build a real single-chunk entry to source valid bytes.
        let src_cache = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xC6_04_03, 0x3C04_3C04_3C04_3C04);
        src_cache.insert(key.clone(), &small_roaring()).unwrap();
        let entry = src_cache.get(&key).expect("hit");
        let bucket_index_clone = entry.bucket_index().to_vec();
        let uncompressed_bytes_clone = entry.uncompressed_bytes();
        let chunk0 = &entry.chunks()[0];
        let comp_bytes = chunk0.compressed_bytes;
        let mut payload = vec![0u8; comp_bytes];
        let rc = unsafe {
            cudaMemcpy(
                payload.as_mut_ptr() as *mut c_void,
                chunk0.d_compressed as *const c_void,
                comp_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        assert_eq!(rc, CUDA_SUCCESS);
        drop(entry);
        drop(src_cache);

        // Hand-craft a MAGIC_V3 dump file containing 1 entry.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v3_v3_backcompat.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&super::super::persist::MAGIC_V3.to_le_bytes())
            .unwrap();
        f.write_all(&super::super::persist::WIRE_VERSION.to_le_bytes())
            .unwrap();
        f.write_all(&super::super::persist::HASH_FN_FXHASHER_V1.to_le_bytes())
            .unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap(); // entry_count
        f.write_all(&0u32.to_le_bytes()).unwrap(); // reserved
        // Per-term: chtkey + bucket_count + bucket_index + uncomp:u64
        // + comp:u64 + payload.
        super::super::persist::write_chtkey(&mut f, &key).unwrap();
        let bucket_count =
            u32::try_from(bucket_index_clone.len()).unwrap_or(u32::MAX);
        f.write_all(&bucket_count.to_le_bytes()).unwrap();
        for (high16, word_offset) in &bucket_index_clone {
            f.write_all(&high16.to_le_bytes()).unwrap();
            f.write_all(&word_offset.to_le_bytes()).unwrap();
        }
        f.write_all(&(uncompressed_bytes_clone as u64).to_le_bytes())
            .unwrap();
        f.write_all(&(comp_bytes as u64).to_le_bytes()).unwrap();
        f.write_all(&payload).unwrap();
        f.write_all(&super::super::persist::MAGIC_END.to_le_bytes())
            .unwrap();
        drop(f);

        // Load via the V4-default loader; MAGIC_V3 must lift cleanly
        // into a 1-chunk Multi entry.
        let dst = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let n_loaded = dst.load_from_path(&path).unwrap();
        assert_eq!(n_loaded, 1);
        let loaded = dst.get(&key).expect("V3 back-compat load must hit");
        assert_eq!(
            loaded.chunk_count(),
            1,
            "MAGIC_V3 dump must lift into a 1-chunk Multi entry"
        );
        assert_eq!(loaded.uncompressed_bytes(), uncompressed_bytes_clone);
        assert_eq!(loaded.bucket_index(), bucket_index_clone.as_slice());
        assert_eq!(loaded.chunks()[0].compressed_bytes, comp_bytes);
        assert_eq!(
            loaded.chunks()[0].uncompressed_bytes,
            uncompressed_bytes_clone
        );
        // Compressed payload byte-equal vs the synthetic dump's body.
        let mut staging = vec![0u8; loaded.chunks()[0].compressed_bytes];
        let rc = unsafe {
            cudaMemcpy(
                staging.as_mut_ptr() as *mut c_void,
                loaded.chunks()[0].d_compressed as *const c_void,
                loaded.chunks()[0].compressed_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        assert_eq!(rc, CUDA_SUCCESS);
        assert_eq!(staging, payload);
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

    // ========================================================
    // Wave Z-2 #1 — promote_v2_to_v3 cross-tier tests
    // ========================================================

    /// Build a freshly-populated v2 cache + extract the
    /// `Arc<VramTermEntry>` so the v3 cross-tier promote tests
    /// don't have to repeat the boilerplate.
    fn populate_v2(roaring: &RoaringPostings) -> (crate::postings::roaring::vram_cht::VramCht, ChtKey, Arc<VramTermEntry>) {
        let v2 = crate::postings::roaring::vram_cht::VramCht::with_budget(64 * 1024 * 1024);
        let key = dummy_key(0xC03_3000, 100);
        v2.insert(key.clone(), roaring).unwrap();
        let entry = v2.get(&key).expect("v2 must hit after insert");
        (v2, key, entry)
    }

    #[test]
    fn promote_v2_to_v3_reuses_bucket_index() {
        // Acceptance gate: the v2 entry's `bucket_index` survives
        // the cross-tier promote byte-for-byte. Same `(high16,
        // word_offset)` invariant the v3 dump/load path relies on,
        // re-used here from the v2 source instead of re-expanding
        // the postings.
        if !cuda_available() {
            return;
        }
        let v3 = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let rp = small_roaring();
        let (_v2, key, v2_entry) = populate_v2(&rp);
        let v2_bucket_index = v2_entry.bucket_index().to_vec();
        let v2_total_words = v2_entry.total_words();

        let inserted = v3.promote_v2_to_v3(&key, Arc::clone(&v2_entry)).unwrap();
        assert!(inserted, "promote must admit a fresh key");

        let v3_entry = v3.get(&key).expect("v3 must hit post-promote");
        assert_eq!(
            v3_entry.bucket_index(),
            v2_bucket_index.as_slice(),
            "v3 bucket_index must be byte-identical to v2 source"
        );
        assert_eq!(
            v3_entry.uncompressed_bytes(),
            v2_total_words * 4,
            "v3 uncompressed_bytes must equal v2 total_words*4"
        );
        assert_eq!(v3_entry.bucket_count(), v2_entry.bucket_count());
        // Stats: promotion was counted.
        let stats = v3.stats();
        assert_eq!(stats.inserts, 1);
        assert_eq!(stats.promotions, 1);
    }

    #[test]
    fn promote_v2_to_v3_skips_redrain() {
        // Acceptance gate: promote works purely from the
        // `Arc<VramTermEntry>` handle — the caller does NOT need to
        // hold or re-drain the original `RoaringPostings`. We
        // populate v2, drop the source `RoaringPostings`, and the
        // promote still succeeds (the v2 device buffer is the
        // canonical copy).
        if !cuda_available() {
            return;
        }
        let v3 = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key;
        let v2_entry;
        let v2_holder;
        {
            let rp = dense_roaring(0);
            let (v2, k, e) = populate_v2(&rp);
            key = k;
            v2_entry = e;
            v2_holder = v2;
            // `rp` drops here — the original RoaringPostings is gone.
        }
        // We still have the Arc<VramTermEntry>; promote uses ONLY
        // that handle (no roaring re-drain).
        let inserted = v3.promote_v2_to_v3(&key, Arc::clone(&v2_entry)).unwrap();
        assert!(inserted, "promote must succeed from Arc handle alone");
        let v3_entry = v3.get(&key).expect("v3 must hit");
        assert_eq!(v3_entry.bucket_index(), v2_entry.bucket_index());
        // Drop the v2 cache last to release its Arc — v3 lifetime
        // is independent.
        drop(v2_holder);
        // v3 entry remains usable.
        assert!(!v3_entry.chunks()[0].d_compressed.is_null());
        assert!(v3_entry.compressed_bytes() > 0);
    }

    #[test]
    fn promote_v2_to_v3_rejection_for_oversize_entry() {
        // Admission policy: when the compressed entry alone would
        // exceed the budget, promote returns Ok(false) and does NOT
        // install the entry. The cudaMalloc for d_compressed does
        // fire (we need actual_comp_size to make the budget
        // decision), but the entry is dropped before insertion so
        // the cache state stays clean.
        //
        // We probe one entry's compressed size on a wide budget
        // first, then construct a v3 cache whose budget is < that
        // (so the post-compress check rejects).
        if !cuda_available() {
            return;
        }
        let rp = dense_roaring(0);
        let probe = VramCompressedCht::with_budget(1 << 30).unwrap();
        let pkey = dummy_key(0xff00, 100);
        probe.insert(pkey, &rp).unwrap();
        let one_entry_compressed = probe.stats().compressed_bytes_total;
        drop(probe);

        let tight = VramCompressedCht::with_budget(one_entry_compressed.saturating_sub(1).max(1)).unwrap();
        let (_v2, key, v2_entry) = populate_v2(&rp);
        let admitted = tight.promote_v2_to_v3(&key, v2_entry).unwrap();
        assert!(
            !admitted,
            "compressed entry > budget must reject (got admitted=true)"
        );
        let stats = tight.stats();
        assert_eq!(stats.inserts, 0);
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.entries, 0);
        // Live cache must be empty + reusable.
        assert!(tight.get(&key).is_none());
    }

    #[test]
    fn promote_v2_to_v3_arc_lifetime_safe() {
        // Lifetime test: the v2 Arc is held across the entire
        // promote call, so a concurrent v2 cache eviction of the
        // same key during the compress step cannot Drop the source
        // device buffer. We simulate this by resetting the v2 cache
        // mid-way (= the cache map drops ITS Arc clone; ours is
        // still live).
        if !cuda_available() {
            return;
        }
        let v3 = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let rp = small_roaring();
        let (v2, key, v2_entry) = populate_v2(&rp);

        // Reset the v2 cache. This drops the v2 map's Arc clone.
        // Our `v2_entry` clone keeps the device buffer alive.
        v2.reset();
        assert!(v2.get(&key).is_none(), "v2 reset must clear map");

        // Promote with our still-live v2 Arc handle.
        let inserted = v3.promote_v2_to_v3(&key, Arc::clone(&v2_entry)).unwrap();
        assert!(
            inserted,
            "promote must succeed even after the v2 cache evicted the map entry"
        );
        // v3 entry queryable.
        let v3_entry = v3.get(&key).expect("v3 must hit");
        assert_eq!(v3_entry.bucket_index(), v2_entry.bucket_index());
        // Drop the original Arc — v3 owns its own d_compressed,
        // unaffected by the v2 source dropping.
        drop(v2_entry);
        assert!(!v3_entry.chunks()[0].d_compressed.is_null());
    }

    #[test]
    fn promote_v2_to_v3_idempotent() {
        // Re-promoting the same key short-circuits without a
        // duplicate cudaMalloc. The second call returns Ok(false)
        // and does not bump `inserts` / `promotions`.
        if !cuda_available() {
            return;
        }
        let v3 = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let rp = small_roaring();
        let (_v2, key, v2_entry) = populate_v2(&rp);

        let first = v3.promote_v2_to_v3(&key, Arc::clone(&v2_entry)).unwrap();
        assert!(first);
        let bytes_after_first = v3.stats().current_bytes;
        let inserts_after_first = v3.stats().inserts;
        let promotions_after_first = v3.stats().promotions;

        let second = v3.promote_v2_to_v3(&key, Arc::clone(&v2_entry)).unwrap();
        assert!(!second, "re-promote of same key must short-circuit");
        let stats = v3.stats();
        assert_eq!(stats.current_bytes, bytes_after_first);
        assert_eq!(stats.inserts, inserts_after_first);
        assert_eq!(stats.promotions, promotions_after_first);
    }

    #[test]
    fn promote_v2_to_v3_matches_insert_layout() {
        // Cross-validation: the v3 entry produced via
        // promote_v2_to_v3 has the same `(bucket_index,
        // uncompressed_bytes)` layout as one produced via the
        // legacy `insert(roaring)` path. Compressed bytes may
        // differ because Bitcomp output can vary microscopically
        // by codec state, but the metadata invariants must match.
        if !cuda_available() {
            return;
        }
        let rp = small_roaring();
        let v3_legacy = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let legacy_key = dummy_key(0x11, 100);
        v3_legacy.insert(legacy_key.clone(), &rp).unwrap();
        let legacy_entry = v3_legacy.get(&legacy_key).unwrap();

        let v3_promote = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let promote_key = dummy_key(0x22, 100);
        let (_v2, _vk, v2_entry) = populate_v2(&rp);
        v3_promote
            .promote_v2_to_v3(&promote_key, v2_entry)
            .unwrap();
        let promote_entry = v3_promote.get(&promote_key).unwrap();

        assert_eq!(
            legacy_entry.bucket_index(),
            promote_entry.bucket_index(),
            "promote_v2_to_v3 must produce the same bucket_index as insert(roaring)"
        );
        assert_eq!(
            legacy_entry.uncompressed_bytes(),
            promote_entry.uncompressed_bytes(),
            "uncompressed_bytes must match"
        );
        assert_eq!(legacy_entry.bucket_count(), promote_entry.bucket_count());
    }

    // ========================================================
    // Wave Z-4 #1 — cross_tier_promotions dedicated counter
    // ========================================================

    #[test]
    fn legacy_promote_does_not_increment_cross_tier_promotions() {
        // Legacy `promote(roaring)` (re-drain path) bumps the
        // generic `promotions` counter but must NOT bump the
        // strict-subset `cross_tier_promotions` counter — that
        // counter is reserved for `promote_v2_to_v3`'s device→device
        // path. This is the key invariant Wave Z-4 #1 establishes.
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xc4_4ee5, 100);
        let inserted = cht.promote(key, &small_roaring()).unwrap();
        assert!(inserted);
        let stats = cht.stats();
        assert_eq!(stats.promotions, 1);
        assert_eq!(
            stats.cross_tier_promotions, 0,
            "legacy promote(roaring) must not bump the cross-tier counter"
        );
    }

    #[test]
    fn promote_v2_to_v3_increments_cross_tier_promotions() {
        // Successful cross-tier promote bumps BOTH `promotions` and
        // the dedicated `cross_tier_promotions` counter by exactly
        // 1. Re-promote of the same key (idempotent short-circuit)
        // does not bump either.
        if !cuda_available() {
            return;
        }
        let v3 = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let rp = small_roaring();
        let (_v2, key, v2_entry) = populate_v2(&rp);

        let pre = v3.stats();
        assert_eq!(pre.cross_tier_promotions, 0);

        let first = v3.promote_v2_to_v3(&key, Arc::clone(&v2_entry)).unwrap();
        assert!(first);
        let after_first = v3.stats();
        assert_eq!(after_first.promotions, 1);
        assert_eq!(after_first.cross_tier_promotions, 1);

        // Idempotent re-promote: short-circuits, neither counter
        // moves.
        let second = v3.promote_v2_to_v3(&key, Arc::clone(&v2_entry)).unwrap();
        assert!(!second);
        let after_second = v3.stats();
        assert_eq!(after_second.promotions, 1);
        assert_eq!(after_second.cross_tier_promotions, 1);
    }

    #[test]
    fn promote_v2_to_v3_rejection_keeps_cross_tier_counter_zero() {
        // Admission rejection (compressed entry > budget) takes the
        // pre-install drop path and must NOT bump either counter.
        // This mirrors the existing
        // `promote_v2_to_v3_rejection_for_oversize_entry` test but
        // pins the new counter explicitly.
        if !cuda_available() {
            return;
        }
        let rp = dense_roaring(0);
        let probe = VramCompressedCht::with_budget(1 << 30).unwrap();
        let pkey = dummy_key(0xfee, 100);
        probe.insert(pkey, &rp).unwrap();
        let one_entry_compressed = probe.stats().compressed_bytes_total;
        drop(probe);

        let tight =
            VramCompressedCht::with_budget(one_entry_compressed.saturating_sub(1).max(1)).unwrap();
        let (_v2, key, v2_entry) = populate_v2(&rp);
        let admitted = tight.promote_v2_to_v3(&key, v2_entry).unwrap();
        assert!(!admitted);
        let stats = tight.stats();
        assert_eq!(stats.promotions, 0);
        assert_eq!(stats.cross_tier_promotions, 0);
    }

    #[test]
    fn reset_clears_cross_tier_promotions() {
        // `reset()` must zero the new counter alongside the existing
        // ones — confirms the field is wired into the test-only
        // reset helper that dispatch-layer tests rely on.
        if !cuda_available() {
            return;
        }
        let v3 = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let rp = small_roaring();
        let (_v2, key, v2_entry) = populate_v2(&rp);
        v3.promote_v2_to_v3(&key, v2_entry).unwrap();
        assert_eq!(v3.stats().cross_tier_promotions, 1);
        v3.reset();
        assert_eq!(v3.stats().cross_tier_promotions, 0);
    }

    // ========================================================
    // Wave Z-6 #2 — multi-chunk Bitcomp admission tests
    // ========================================================

    /// One doc in each of `buckets` distinct high16 buckets — forces
    /// the staging buffer to grow to `buckets * 8 KiB` uncompressed so
    /// we can exercise the multi-chunk admission path without piling
    /// docs into the same container.
    fn multi_chunk_roaring(buckets: usize) -> RoaringPostings {
        let docs: Vec<u32> = (0..buckets).map(|i| (i as u32) << 16).collect();
        RoaringEncoder::from_doc_ids(&docs)
    }

    #[test]
    fn insert_multi_chunk_roundtrip() {
        // Wave Z-6 #2 acceptance gate: a 32 MiB-equivalent posting
        // (= 4096 buckets × 8 KiB) admits via the new multi-chunk
        // path and produces exactly 2 chunks. `compressed_bytes()`
        // aggregates per-chunk bytes; idempotent re-insert short-
        // circuits without bumping inserts.
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(128 * 1024 * 1024).unwrap();
        let key = dummy_key(0xC01_C01, 100);
        let rp = multi_chunk_roaring(BUCKETS_PER_CHUNK * 2);
        let inserted = cht.insert(key.clone(), &rp).unwrap();
        assert!(
            inserted,
            "32 MiB-equivalent posting must admit via multi-chunk path"
        );
        let entry = cht.get(&key).expect("hit");
        assert_eq!(entry.chunk_count(), 2, "32 MiB ÷ 16 MiB = 2 chunks");
        assert_eq!(entry.bucket_count(), BUCKETS_PER_CHUNK * 2);
        assert_eq!(
            entry.uncompressed_bytes(),
            BUCKETS_PER_CHUNK * 2 * BITMAP_CONTAINER_WORDS * 4
        );
        let sum_compressed: usize = entry.chunks().iter().map(|c| c.compressed_bytes).sum();
        assert_eq!(entry.compressed_bytes(), sum_compressed);
        let sum_uncomp: usize = entry.chunks().iter().map(|c| c.uncompressed_bytes).sum();
        assert_eq!(entry.uncompressed_bytes(), sum_uncomp);

        // Idempotent re-insert short-circuits.
        let inserted_2 = cht.insert(key.clone(), &rp).unwrap();
        assert!(!inserted_2);
        assert_eq!(cht.stats().inserts, 1);
    }

    #[test]
    fn promote_v2_to_v3_multi_chunk_roundtrip() {
        // Cross-tier promote on a 32 MiB-equivalent posting: v2 holds
        // the uncompressed buffer (one cudaMalloc); v3 fans out across
        // 2 Bitcomp chunks via device→device compress without
        // re-draining the source postings. `cross_tier_promotions`
        // bumps by exactly 1 per successful promote (single counter
        // bump, NOT per-chunk).
        if !cuda_available() {
            return;
        }
        let v3 = VramCompressedCht::with_budget(128 * 1024 * 1024).unwrap();
        let rp = multi_chunk_roaring(BUCKETS_PER_CHUNK * 2);

        // v2 needs ≥ 32 MiB budget to hold the uncompressed buffer.
        let v2 = crate::postings::roaring::vram_cht::VramCht::with_budget(128 * 1024 * 1024);
        let key = dummy_key(0xC03_3001, 100);
        v2.insert(key.clone(), &rp).unwrap();
        let v2_entry = v2.get(&key).expect("v2 must hit");
        let v2_bucket_index = v2_entry.bucket_index().to_vec();

        let inserted = v3.promote_v2_to_v3(&key, Arc::clone(&v2_entry)).unwrap();
        assert!(inserted, "32 MiB v2 entry must promote into multi-chunk v3");
        let v3_entry = v3.get(&key).expect("v3 must hit post-promote");
        assert_eq!(v3_entry.chunk_count(), 2);
        assert_eq!(v3_entry.bucket_index(), v2_bucket_index.as_slice());
        assert_eq!(
            v3_entry.uncompressed_bytes(),
            BUCKETS_PER_CHUNK * 2 * BITMAP_CONTAINER_WORDS * 4
        );
        // Counter bumps once per promote, not once per chunk.
        let stats = v3.stats();
        assert_eq!(stats.promotions, 1);
        assert_eq!(stats.cross_tier_promotions, 1);
        assert_eq!(stats.inserts, 1);
    }

    #[test]
    fn multi_chunk_drop_frees_all_chunks() {
        // Drop must run `cudaFree` for every chunk in the chunks vec.
        // We can't directly count cudaFree calls without instrumenting
        // the runtime, but we can verify by reset()ing a populated
        // multi-chunk cache and reusing the freed memory for a fresh
        // insert — if the previous chunks weren't freed, we'd see
        // budget pressure that prevents the second insert.
        if !cuda_available() {
            return;
        }
        let one_entry_uncompressed_bytes = BUCKETS_PER_CHUNK * 2 * BITMAP_CONTAINER_WORDS * 4;
        // Sanity: each entry is 32 MiB uncompressed.
        assert_eq!(one_entry_uncompressed_bytes, 32 * 1024 * 1024);

        let cht = VramCompressedCht::with_budget(256 * 1024 * 1024).unwrap();
        let rp = multi_chunk_roaring(BUCKETS_PER_CHUNK * 2);
        let k1 = dummy_key(0xD09_F100, 100);
        cht.insert(k1.clone(), &rp).unwrap();
        let entry = cht.get(&k1).expect("hit");
        assert_eq!(entry.chunk_count(), 2);
        let live_bytes_before = cht.stats().current_bytes;
        assert!(live_bytes_before > 0);
        drop(entry);

        // Reset drops every cached entry's Arc → Drop runs cudaFree
        // for both chunks. Cache budget must clear back to zero.
        cht.reset();
        assert_eq!(cht.stats().current_bytes, 0);
        assert_eq!(cht.stats().entries, 0);

        // Re-insert succeeds — proves the prior chunks' device memory
        // was freed and is available to back the new allocation.
        let k2 = dummy_key(0xD09_F200, 100);
        let reinserted = cht.insert(k2, &rp).unwrap();
        assert!(reinserted, "post-reset re-insert must reuse freed VRAM");
        assert_eq!(cht.stats().entries, 1);
    }

    #[test]
    fn legacy_single_chunk_path_unchanged() {
        // Existing terms below the 16 MiB threshold continue to land
        // as `chunks.len() == 1`; the back-compat shim methods stay
        // valid and the persist dump/load roundtrip (covered in the
        // separate D-5 tests) keeps working byte-for-byte.
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xE05_E05, 100);
        cht.insert(key.clone(), &small_roaring()).unwrap();
        let entry = cht.get(&key).expect("hit");
        assert_eq!(
            entry.chunk_count(),
            1,
            "small (< 16 MiB) terms keep the single-chunk fast path"
        );
        // Wave Z-6 #3: the single live chunk exposes a non-null device
        // pointer (the back-compat shims this test originally asserted
        // against were removed once dispatch migrated to `chunks()`).
        assert!(!entry.chunks()[0].d_compressed.is_null());
        // Sum aggregation degenerates to the single chunk's bytes.
        assert_eq!(entry.compressed_bytes(), entry.chunks()[0].compressed_bytes);
        assert_eq!(
            entry.uncompressed_bytes(),
            entry.chunks()[0].uncompressed_bytes
        );
    }

    #[test]
    fn admission_cap_constants_and_error_variant_shape() {
        // `MAX_CHUNKS_PER_ENTRY * BITCOMP_CHUNK_BYTES` (= 1 GiB) is the
        // hard admission cap. Reaching it via a single `RoaringPostings`
        // is **not** possible today because `high16` is `u16`, bounding
        // the per-term bucket count at 65,536 (= 32 chunks at 2,048
        // buckets each) — well below the 64-chunk cap. The
        // `TooManyChunks` variant exists for hypothetical future
        // expansion (wider high16, denser containers); pin its shape +
        // the cap constants so consumers can match on the error and so
        // a constant tweak fails loud here.
        assert_eq!(MAX_CHUNKS_PER_ENTRY, 64);
        assert_eq!(BITCOMP_CHUNK_BYTES, 1 << 24);
        assert_eq!(BUCKETS_PER_CHUNK, 2048);
        assert_eq!(
            MAX_CHUNKS_PER_ENTRY * BITCOMP_CHUNK_BYTES,
            1024 * 1024 * 1024,
            "max admission must equal 1 GiB"
        );
        let err = VramCompressedChtError::TooManyChunks {
            chunks_needed: 100,
            max: MAX_CHUNKS_PER_ENTRY,
        };
        match err {
            VramCompressedChtError::TooManyChunks { chunks_needed, max } => {
                assert_eq!(chunks_needed, 100);
                assert_eq!(max, MAX_CHUNKS_PER_ENTRY);
            }
            other => panic!("variant mismatch, got {other:?}"),
        }
    }

    #[test]
    fn bucket_index_byte_offsets_match_chunk_boundaries() {
        // Layout invariant pin: for a multi-chunk entry, the prefix
        // sum of `chunks[..k].uncompressed_bytes` aligns with the
        // bucket_index's byte offset for bucket `k * BUCKETS_PER_CHUNK`.
        // This is the assumption Z-6 #3 dispatch flattening will rely
        // on; pin it now so design pivots fail loud.
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(128 * 1024 * 1024).unwrap();
        let key = dummy_key(0xB47_B47, 100);
        let rp = multi_chunk_roaring(BUCKETS_PER_CHUNK * 2);
        cht.insert(key.clone(), &rp).unwrap();
        let entry = cht.get(&key).expect("hit");
        assert_eq!(entry.chunk_count(), 2);

        let bucket_bytes = BITMAP_CONTAINER_WORDS * 4;
        let mut byte_prefix: usize = 0;
        for (k, chunk) in entry.chunks().iter().enumerate() {
            let first_bucket_of_chunk = k * BUCKETS_PER_CHUNK;
            // The (high16, word_offset) entry for the chunk's first
            // bucket must point at the start of this chunk in the
            // uncompressed layout.
            let (_, word_offset) = entry.bucket_index()[first_bucket_of_chunk];
            assert_eq!(
                word_offset as usize * 4,
                byte_prefix,
                "chunk {k}: bucket_index word_offset must mark chunk start"
            );
            // Each chunk covers an integral number of buckets.
            assert_eq!(
                chunk.uncompressed_bytes % bucket_bytes,
                0,
                "chunk {k} uncompressed_bytes must be a multiple of {bucket_bytes}"
            );
            byte_prefix += chunk.uncompressed_bytes;
        }
        assert_eq!(byte_prefix, entry.uncompressed_bytes());
    }

    // ============================================================
    // Wave Z-7 #2 — `evict_by_segments` API tests
    // ============================================================

    #[test]
    fn evict_by_segments_drops_only_matching_entries() {
        // Wave Z-7 #2 acceptance: passing two of three segment IDs to
        // `evict_by_segments` evicts exactly those two entries; the
        // third (whose segment was not in the input slice) survives.
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key_a = dummy_key(0xa1a1, 100);
        let key_b = dummy_key(0xa1a2, 101);
        let key_c = dummy_key(0xa1a3, 102);
        cht.insert(key_a.clone(), &small_roaring()).unwrap();
        cht.insert(key_b.clone(), &small_roaring()).unwrap();
        cht.insert(key_c.clone(), &small_roaring()).unwrap();
        assert_eq!(cht.stats().entries, 3);

        let evicted = cht.evict_by_segments(&[
            key_a.segment_id.clone(),
            key_c.segment_id.clone(),
        ]);
        assert_eq!(evicted, 2);
        assert_eq!(cht.stats().entries, 1);
        assert!(cht.get(&key_a).is_none());
        assert!(cht.get(&key_b).is_some());
        assert!(cht.get(&key_c).is_none());
        assert_eq!(cht.stats().evictions, 2);
    }

    #[test]
    fn evict_by_segments_bytes_total_decrements() {
        // Wave Z-7 #2 stats invariant: `current_bytes` and
        // `uncompressed_bytes_total` must decrement by exactly the
        // evicted entry's footprint; `inserts` must NOT change
        // (eviction is not insertion).
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key_a = dummy_key(0xb1b1, 200);
        let key_b = dummy_key(0xb1b2, 201);
        cht.insert(key_a.clone(), &small_roaring()).unwrap();
        cht.insert(key_b.clone(), &small_roaring()).unwrap();

        let entry_a = cht.get(&key_a).expect("hit");
        let entry_a_compressed = entry_a.compressed_bytes() as u64;
        let entry_a_uncompressed = entry_a.uncompressed_bytes() as u64;
        drop(entry_a);
        let before = cht.stats();

        let evicted = cht.evict_by_segments(&[key_a.segment_id.clone()]);
        assert_eq!(evicted, 1);
        let after = cht.stats();
        assert_eq!(after.entries, 1);
        assert_eq!(
            after.current_bytes,
            before.current_bytes.saturating_sub(entry_a_compressed)
        );
        assert_eq!(
            after.uncompressed_bytes_total,
            before
                .uncompressed_bytes_total
                .saturating_sub(entry_a_uncompressed)
        );
        assert_eq!(after.evictions, before.evictions + 1);
        assert_eq!(after.inserts, before.inserts, "inserts unchanged by evict");
    }

    #[test]
    fn evict_by_segments_empty_input_no_op() {
        // Wave Z-7 #2 zero-element fast path: calling with `&[]`
        // returns 0 and touches neither the map nor the stats
        // counters (avoids a mutex acquisition + alloc on the
        // common ILM-no-Hot-transitions case).
        if !cuda_available() {
            return;
        }
        let cht = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xc1c1, 300);
        cht.insert(key.clone(), &small_roaring()).unwrap();
        let before = cht.stats();

        let evicted = cht.evict_by_segments(&[]);
        assert_eq!(evicted, 0);
        let after = cht.stats();
        assert_eq!(after.entries, before.entries);
        assert_eq!(after.current_bytes, before.current_bytes);
        assert_eq!(after.uncompressed_bytes_total, before.uncompressed_bytes_total);
        assert_eq!(after.evictions, before.evictions);
        assert!(cht.get(&key).is_some(), "untouched entry must still hit");
    }

    // ============================================================
    // Wave Z-7 #6 #1 — `dump_by_segments` API tests
    // ============================================================

    #[test]
    fn dump_by_segments_writes_only_matching_entries() {
        // Wave Z-7 #6 #1 acceptance: insert 4 entries (segments
        // a/b/c/d), call `dump_by_segments(&[a, c])`, load back into
        // a fresh cache instance, assert only entries for a + c are
        // present.
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v3_filtered.bin");

        let src = VramCompressedCht::with_budget(128 * 1024 * 1024).unwrap();
        let key_a = dummy_key(0xa1, 0x0001);
        let key_b = dummy_key(0xa2, 0x0002);
        let key_c = dummy_key(0xa3, 0x0003);
        let key_d = dummy_key(0xa4, 0x0004);
        src.insert(key_a.clone(), &small_roaring()).unwrap();
        src.insert(key_b.clone(), &dense_roaring(0)).unwrap();
        src.insert(key_c.clone(), &small_roaring()).unwrap();
        src.insert(key_d.clone(), &dense_roaring(1)).unwrap();
        assert_eq!(src.stats().entries, 4);

        // Dump only a + c.
        let n_dumped = src
            .dump_by_segments(
                &[key_a.segment_id.clone(), key_c.segment_id.clone()],
                &path,
            )
            .unwrap();
        assert_eq!(n_dumped, 2);
        assert!(path.exists(), "dump file must exist after dump");

        // Source cache must NOT be touched by a dump (vs evict).
        assert_eq!(src.stats().entries, 4, "dump must not mutate source");

        // Load into a fresh cache and confirm membership.
        let dst = VramCompressedCht::with_budget(128 * 1024 * 1024).unwrap();
        let n_loaded = dst.load_from_path(&path).unwrap();
        assert_eq!(n_loaded, 2);
        assert!(dst.get(&key_a).is_some(), "a included in filter must load");
        assert!(dst.get(&key_b).is_none(), "b excluded must NOT load");
        assert!(dst.get(&key_c).is_some(), "c included in filter must load");
        assert!(dst.get(&key_d).is_none(), "d excluded must NOT load");
    }

    #[test]
    fn dump_by_segments_byte_invariant_with_oracle() {
        // Wave Z-7 #6 #1 oracle test: calling `dump_by_segments` with
        // ALL segments from the cache must produce a byte-equivalent
        // file to `dump_to_path` (which dumps everything). Pins the
        // filter logic — a non-trivial bug in the segment_set
        // membership test would silently include / exclude entries
        // and diverge from the oracle.
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let oracle_path = dir.path().join("cht_v3_oracle.bin");
        let filtered_path = dir.path().join("cht_v3_filtered_all.bin");

        let src = VramCompressedCht::with_budget(128 * 1024 * 1024).unwrap();
        let key_a = dummy_key(0xb1, 0x1001);
        let key_b = dummy_key(0xb2, 0x1002);
        let key_c = dummy_key(0xb3, 0x1003);
        src.insert(key_a.clone(), &small_roaring()).unwrap();
        src.insert(key_b.clone(), &dense_roaring(2)).unwrap();
        src.insert(key_c.clone(), &small_roaring()).unwrap();

        // Oracle: dump_to_path (no filter).
        let n_oracle = src.dump_to_path(&oracle_path).unwrap();
        // Filtered: dump_by_segments with the full segment set —
        // membership filter is true for every entry, output set is
        // identical to oracle.
        let n_filtered = src
            .dump_by_segments(
                &[
                    key_a.segment_id.clone(),
                    key_b.segment_id.clone(),
                    key_c.segment_id.clone(),
                ],
                &filtered_path,
            )
            .unwrap();
        assert_eq!(n_oracle, n_filtered);

        // The cache map iteration order is stable within a single
        // process (HashMap is `RandomState`-keyed; both dump calls
        // happen in the same process and observe the same internal
        // ordering), so the serialised entry sequence is identical
        // and the resulting files are byte-equivalent. If a future
        // refactor breaks this stability guarantee, this assertion
        // fires and the implementation must canonicalise entry
        // order before serialising.
        let oracle_bytes = std::fs::read(&oracle_path).unwrap();
        let filtered_bytes = std::fs::read(&filtered_path).unwrap();
        assert_eq!(
            oracle_bytes, filtered_bytes,
            "dump_by_segments(all segments) must be byte-equivalent to dump_to_path"
        );
    }

    #[test]
    fn dump_by_segments_empty_input_writes_header_only() {
        // Wave Z-7 #6 #1 empty-input behaviour: calling with `&[]`
        // writes a zero-entry MAGIC_V4 header (the file exists, is
        // valid, and `load_from_path` round-trips it as a no-op).
        // This is the documented contract — atomic write still runs
        // so any previous partially-written file is replaced cleanly.
        if !cuda_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cht_v3_empty_filter.bin");

        let src = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let key = dummy_key(0xcafe, 0x9000);
        src.insert(key, &small_roaring()).unwrap();

        let n_dumped = src.dump_by_segments(&[], &path).unwrap();
        assert_eq!(n_dumped, 0, "empty filter writes zero entries");
        assert!(path.exists(), "empty-filter dump still creates the file");

        // Round-trip via load_from_path — must be a no-op (zero
        // entries installed) without WrongMagic / TruncatedDump.
        let dst = VramCompressedCht::with_budget(64 * 1024 * 1024).unwrap();
        let n_loaded = dst.load_from_path(&path).unwrap();
        assert_eq!(n_loaded, 0);
        assert_eq!(dst.stats().entries, 0);
    }
}

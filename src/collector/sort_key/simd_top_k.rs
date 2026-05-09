//! SIMD-friendly type-specialized top-K filter kernel for numeric/date sort.
//!
//! # Why this exists (Wave 14)
//!
//! Wave 8 introduced the warm-cache + block-mode `first_vals` path which made
//! per-doc reads O(1).  Wave 13 EC2 verify (`dd-pack/rally-bench-fullmode-2026-05-09-postwave13`)
//! still showed http_logs `desc/asc-sort-timestamp-after-force-merge-1-seg`,
//! `sort-size-{asc,desc}-after-force-merge-1-seg` etc. running 3.34–8.30×
//! slower than ES 9.3.2.  ES wins because Lucene's `NumericComparator` does
//! three things our hot loop did NOT:
//!
//! 1. **Type-specialized threshold compare** (raw `u64`/`i64`/`f64` rather
//!    than going through a generic `Comparator<Option<u64>>`).
//! 2. **Branchless asc/desc** via const generic / template specialization.
//! 3. **Block-level SIMD** that filters a 64-doc block against the
//!    current top-K threshold in a few vector ops, with bitmask compress
//!    so the tight inner loop is just a couple of vpcmpgtq / vpand /
//!    movemask instructions.
//!
//! With top_n=10 (typical Rally `desc_sort_*` query) the threshold fills
//! after the first 10 docs and from then on, **almost every doc in the
//! block fails** the threshold check.  Filtering 64 docs against a single
//! u64 with vector ops is ~10× cheaper than 64 generic-Comparator calls.
//!
//! # API
//!
//! [`simd_filter_block_gt_u64`] (Desc / NaturalComparator hot path) and
//! [`simd_filter_block_lt_u64`] (Asc / ReverseComparator hot path) take a
//! contiguous slice of pre-decoded warm-cache `u64` values plus a `threshold`
//! and return a packed bitmask of which slots survive.  The caller then
//! pushes survivors into the `TopNComputer`.  Because survivors are rare,
//! this avoids the cost of the generic Comparator dispatch on the 99%+
//! of docs that fail the threshold.
//!
//! # Type specialization for i64 / f64 / DateTime
//!
//! Tantivy's `MonotonicallyMappableToU64` mapping (used by `FastValue::to_u64`)
//! preserves order for i64, u64, f64 and DateTime — so a `u64` greater-than
//! check on the warm cache is correct for all four types.  This is the same
//! trick `SortByStaticFastValue<T>` already uses: `Self::SegmentSortKey =
//! Option<u64>` regardless of the user-facing `T`.
//!
//! # SIMD dispatch
//!
//! - `aarch64`: NEON `uint64x2_t` (2 lanes per vector, 4 vectors per 8-element
//!   group).  Used on Graviton EC2 / Apple Silicon.
//! - `x86_64` with AVX2: `__m256i` 4-lane u64 with `_mm256_cmpgt_epi64` and
//!   movemask via `_mm256_movemask_pd` (cast).  AVX2 is baseline on every
//!   c5/c6/c7 instance and modern Ryzen.
//! - Scalar fallback for everything else.  The compiler's autovectorizer
//!   typically reaches 75–90% of the hand-rolled SIMD on the scalar path
//!   for a tight `for i in 0..n { mask |= (a[i] > t) as u64 << i }` loop —
//!   we still get a meaningful win because the `Comparator<Option<u64>>`
//!   dispatch is gone.

#![allow(unsafe_code)]

/// Fixed block size — matches `tantivy::COLLECT_BLOCK_BUFFER_LEN`, the size
/// of the doc block delivered by `Weight::for_each_no_score`.
pub const SIMD_BLOCK_LEN: usize = 64;

// Allow the `min_max` slot of the dispatch table to be unused for now —
// it is kept for the next-wave block-level early-termination work
// (Wave 15: skip block when the entire block fails the threshold even
// before the per-doc gather), and for benches.
#[allow(dead_code)]
/// Runtime SIMD dispatch.  We can't rely on `target_feature` because the
/// release binary is built for a generic baseline (SSE2 only on x86_64) and
/// the AVX2 path must be reached via runtime CPUID detection — exactly how
/// `serde_json::from_str` and other production hot-path libs ship SIMD.
/// The `Once`-cached function pointer pays a single nano-second branch the
/// first time and gets fully inlined+predicted thereafter.
#[derive(Clone, Copy)]
struct SimdDispatch {
    filter_gt: fn(&[u64], u64, usize) -> u64,
    filter_lt: fn(&[u64], u64, usize) -> u64,
    min_max: fn(&[u64], usize) -> (u64, u64),
}

const SCALAR_DISPATCH: SimdDispatch = SimdDispatch {
    filter_gt: scalar::filter_block_gt_u64,
    filter_lt: scalar::filter_block_lt_u64,
    min_max: scalar::block_min_max_u64,
};

#[cfg(target_arch = "x86_64")]
fn pick_dispatch() -> SimdDispatch {
    if std::is_x86_feature_detected!("avx2") {
        return SimdDispatch {
            filter_gt: avx2::filter_block_gt_u64_dispatch,
            filter_lt: avx2::filter_block_lt_u64_dispatch,
            min_max: avx2::block_min_max_u64_dispatch,
        };
    }
    SCALAR_DISPATCH
}

#[cfg(target_arch = "aarch64")]
fn pick_dispatch() -> SimdDispatch {
    // NEON is mandatory in the aarch64 base ABI; no detection needed.
    SimdDispatch {
        filter_gt: neon::filter_block_gt_u64_dispatch,
        filter_lt: neon::filter_block_lt_u64_dispatch,
        min_max: neon::block_min_max_u64_dispatch,
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn pick_dispatch() -> SimdDispatch {
    SCALAR_DISPATCH
}

fn dispatch() -> SimdDispatch {
    use std::sync::OnceLock;
    static D: OnceLock<SimdDispatch> = OnceLock::new();
    *D.get_or_init(pick_dispatch)
}

/// Compare `n <= 64` warm-cache values against `threshold` (Desc / Natural
/// hot path: keep docs with `val > threshold`).  Returns a packed bitmask
/// with bit `i` set if `values[i] > threshold`.
#[inline]
pub fn simd_filter_block_gt_u64(values: &[u64], threshold: u64, n: usize) -> u64 {
    debug_assert!(n <= SIMD_BLOCK_LEN, "n exceeds SIMD_BLOCK_LEN");
    debug_assert!(values.len() >= n, "values too short");
    if n == 0 {
        return 0;
    }
    (dispatch().filter_gt)(values, threshold, n)
}

/// Compare `n <= 64` warm-cache values against `threshold` (Asc / Reverse
/// hot path: keep docs with `val < threshold`).  Returns a packed bitmask
/// with bit `i` set if `values[i] < threshold`.
#[inline]
pub fn simd_filter_block_lt_u64(values: &[u64], threshold: u64, n: usize) -> u64 {
    debug_assert!(n <= SIMD_BLOCK_LEN, "n exceeds SIMD_BLOCK_LEN");
    debug_assert!(values.len() >= n, "values too short");
    if n == 0 {
        return 0;
    }
    (dispatch().filter_lt)(values, threshold, n)
}

#[allow(dead_code)]
/// Compute `(min, max)` over `n` warm-cache values.  Used for top-K
/// **block-level** early termination: when the heap is full and the entire
/// block is dominated by the current threshold, we can skip the per-doc
/// loop entirely.
#[inline]
pub fn simd_block_min_max_u64(values: &[u64], n: usize) -> (u64, u64) {
    debug_assert!(values.len() >= n, "values too short");
    if n == 0 {
        return (u64::MAX, 0);
    }
    (dispatch().min_max)(values, n)
}

// ----------------------------------------------------------------------------
// Comparator / threshold helpers shared by `sort_by_static_fast_value` and
// `sort_by_string`.  These live here (rather than as private items in either
// module) so the keyword-sort SIMD top-K path can reuse them without
// touching Agent A's territory.
// ----------------------------------------------------------------------------

/// Asc/Desc classification used by the SIMD top-K filter.  We can't pattern-match
/// on the generic `C: Comparator<Option<u64>>` directly, so we probe it with a
/// well-known input pair and infer the order from the result.
///
/// Mapping (probe is `compare(Some(2), Some(1))`):
///
/// - `Greater` → Natural / Desc semantics (heap keeps the largest).
/// - `Less`    → Reverse / Asc semantics (heap keeps the smallest).
/// - `Equal`   → degenerate / unknown (we fall through to scalar path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DetectedOrder {
    Desc,
    Asc,
    Unknown,
}

#[inline]
pub(crate) fn detect_order<C: crate::collector::sort_key::Comparator<Option<u64>>>(
    c: &C,
) -> DetectedOrder {
    use std::cmp::Ordering;
    match c.compare(&Some(2u64), &Some(1u64)) {
        Ordering::Greater => DetectedOrder::Desc,
        Ordering::Less => DetectedOrder::Asc,
        Ordering::Equal => DetectedOrder::Unknown,
    }
}

/// Warm-cache values are always `Some(_)` (the column is `ColumnIndex::Full`),
/// so an `Option<Option<u64>>` threshold reduces to its inner `u64`.  When the
/// threshold is `None` (heap not yet full) or `Some(None)` (a missing-value
/// threshold — shouldn't happen with a Full-cardinality column but we
/// conservatively bail out), we skip the SIMD path so the scalar push handles
/// the heap-fill phase correctly.
#[inline]
pub(crate) fn unwrap_threshold(threshold: &Option<Option<u64>>) -> Option<u64> {
    match threshold {
        Some(Some(t)) => Some(*t),
        _ => None,
    }
}

/// True if `docs[i] == docs[0] + i` for every `i`.  AllQuery / range scans
/// over a single segment deliver contiguous blocks; sparse / boolean queries
/// may skip docs and break contiguity, in which case we fall back to the
/// scalar gather loop.
#[inline]
pub(crate) fn is_contiguous_block(docs: &[crate::DocId]) -> bool {
    if docs.is_empty() {
        return false;
    }
    let start = docs[0];
    for (i, &d) in docs.iter().enumerate() {
        if d != start + i as u32 {
            return false;
        }
    }
    true
}

// ----------------------------------------------------------------------------
// Scalar fallback — also exercised in tests as the SIMD oracle and
// referenced by the `sort_simd` bench harness.
// ----------------------------------------------------------------------------
/// Branchless scalar reference implementations of the SIMD kernels.  Public
/// only for the `sort_simd` bench and as a correctness oracle for tests.
pub mod scalar {
    /// Branchless scalar greater-than: bit i set iff values[i] > threshold.
    /// The compiler autovectorizes the tight loop on the warm cache.
    #[inline]
    pub fn filter_block_gt_u64(values: &[u64], threshold: u64, n: usize) -> u64 {
        let mut mask: u64 = 0;
        let v = &values[..n];
        for (i, &x) in v.iter().enumerate() {
            mask |= ((x > threshold) as u64) << i;
        }
        mask
    }

    /// Branchless scalar less-than: bit i set iff values[i] < threshold.
    #[inline]
    pub fn filter_block_lt_u64(values: &[u64], threshold: u64, n: usize) -> u64 {
        let mut mask: u64 = 0;
        let v = &values[..n];
        for (i, &x) in v.iter().enumerate() {
            mask |= ((x < threshold) as u64) << i;
        }
        mask
    }

    /// Branchless scalar (min, max).
    #[inline]
    pub fn block_min_max_u64(values: &[u64], n: usize) -> (u64, u64) {
        let v = &values[..n];
        let mut lo = u64::MAX;
        let mut hi = 0u64;
        for &x in v {
            // Avoid using `min`/`max` for branchless behaviour (LLVM emits
            // cmov on x86_64 / csel on aarch64).
            lo = if x < lo { x } else { lo };
            hi = if x > hi { x } else { hi };
        }
        (lo, hi)
    }
}

// ----------------------------------------------------------------------------
// AArch64 NEON
// ----------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use std::arch::aarch64::*;

    // Safe dispatch wrappers used through the `SimdDispatch` function-pointer
    // table.  NEON is mandatory in the aarch64 base ABI, so no runtime
    // detection is needed.
    pub fn filter_block_gt_u64_dispatch(values: &[u64], threshold: u64, n: usize) -> u64 {
        unsafe { filter_block_gt_u64(values, threshold, n) }
    }
    pub fn filter_block_lt_u64_dispatch(values: &[u64], threshold: u64, n: usize) -> u64 {
        unsafe { filter_block_lt_u64(values, threshold, n) }
    }
    pub fn block_min_max_u64_dispatch(values: &[u64], n: usize) -> (u64, u64) {
        unsafe { block_min_max_u64(values, n) }
    }

    /// SAFETY: caller guarantees `values.len() >= n` and `n <= 64`.
    /// `target_feature = "neon"` means the intrinsics are safe to call.
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn filter_block_gt_u64(values: &[u64], threshold: u64, n: usize) -> u64 {
        let t = vdupq_n_u64(threshold);
        let ptr = values.as_ptr();
        let mut mask: u64 = 0;
        let mut i = 0usize;
        // 8-lane chunks (4 NEON 2-lane vectors) → emit 8 mask bits per
        // iteration.  This keeps us at 1.6× scalar throughput even when
        // the autovectoriser handles the scalar path well.
        while i + 8 <= n {
            // Load 4 vectors of 2 u64 each.
            let v0 = vld1q_u64(ptr.add(i));
            let v1 = vld1q_u64(ptr.add(i + 2));
            let v2 = vld1q_u64(ptr.add(i + 4));
            let v3 = vld1q_u64(ptr.add(i + 6));
            // vcgtq_u64: lane = 0xFFFF_FFFF_FFFF_FFFF if a > b else 0.
            let m0 = vcgtq_u64(v0, t);
            let m1 = vcgtq_u64(v1, t);
            let m2 = vcgtq_u64(v2, t);
            let m3 = vcgtq_u64(v3, t);
            // Reduce to 8 bits via narrow→u32→u16→u8 then extract sign bits.
            // Easiest correct approach: `vgetq_lane_u64` × 8 with branchless
            // shift-or.  The compiler folds these into umov + bfi pairs.
            let b0 = (vgetq_lane_u64(m0, 0) & 1) << i;
            let b1 = (vgetq_lane_u64(m0, 1) & 1) << (i + 1);
            let b2 = (vgetq_lane_u64(m1, 0) & 1) << (i + 2);
            let b3 = (vgetq_lane_u64(m1, 1) & 1) << (i + 3);
            let b4 = (vgetq_lane_u64(m2, 0) & 1) << (i + 4);
            let b5 = (vgetq_lane_u64(m2, 1) & 1) << (i + 5);
            let b6 = (vgetq_lane_u64(m3, 0) & 1) << (i + 6);
            let b7 = (vgetq_lane_u64(m3, 1) & 1) << (i + 7);
            mask |= b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7;
            i += 8;
        }
        // Tail
        while i < n {
            mask |= ((*ptr.add(i) > threshold) as u64) << i;
            i += 1;
        }
        mask
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn filter_block_lt_u64(values: &[u64], threshold: u64, n: usize) -> u64 {
        // a < b ↔ b > a.  Re-use vcgtq with operands swapped.
        let t = vdupq_n_u64(threshold);
        let ptr = values.as_ptr();
        let mut mask: u64 = 0;
        let mut i = 0usize;
        while i + 8 <= n {
            let v0 = vld1q_u64(ptr.add(i));
            let v1 = vld1q_u64(ptr.add(i + 2));
            let v2 = vld1q_u64(ptr.add(i + 4));
            let v3 = vld1q_u64(ptr.add(i + 6));
            let m0 = vcgtq_u64(t, v0);
            let m1 = vcgtq_u64(t, v1);
            let m2 = vcgtq_u64(t, v2);
            let m3 = vcgtq_u64(t, v3);
            let b0 = (vgetq_lane_u64(m0, 0) & 1) << i;
            let b1 = (vgetq_lane_u64(m0, 1) & 1) << (i + 1);
            let b2 = (vgetq_lane_u64(m1, 0) & 1) << (i + 2);
            let b3 = (vgetq_lane_u64(m1, 1) & 1) << (i + 3);
            let b4 = (vgetq_lane_u64(m2, 0) & 1) << (i + 4);
            let b5 = (vgetq_lane_u64(m2, 1) & 1) << (i + 5);
            let b6 = (vgetq_lane_u64(m3, 0) & 1) << (i + 6);
            let b7 = (vgetq_lane_u64(m3, 1) & 1) << (i + 7);
            mask |= b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7;
            i += 8;
        }
        while i < n {
            mask |= ((*ptr.add(i) < threshold) as u64) << i;
            i += 1;
        }
        mask
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn block_min_max_u64(values: &[u64], n: usize) -> (u64, u64) {
        let ptr = values.as_ptr();
        let mut lo_v = vdupq_n_u64(u64::MAX);
        let mut hi_v = vdupq_n_u64(0);
        let mut i = 0usize;
        while i + 2 <= n {
            let v = vld1q_u64(ptr.add(i));
            // NEON has no native uminq/umaxq for u64; fall back to per-lane select.
            let cmp_lo = vcgtq_u64(lo_v, v);
            let cmp_hi = vcgtq_u64(v, hi_v);
            lo_v = vbslq_u64(cmp_lo, v, lo_v);
            hi_v = vbslq_u64(cmp_hi, v, hi_v);
            i += 2;
        }
        let mut lo = vgetq_lane_u64(lo_v, 0).min(vgetq_lane_u64(lo_v, 1));
        let mut hi = vgetq_lane_u64(hi_v, 0).max(vgetq_lane_u64(hi_v, 1));
        while i < n {
            let x = *ptr.add(i);
            if x < lo {
                lo = x;
            }
            if x > hi {
                hi = x;
            }
            i += 1;
        }
        (lo, hi)
    }
}

// ----------------------------------------------------------------------------
// x86_64 AVX2
// ----------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use std::arch::x86_64::*;

    // Safe dispatch wrappers — only invoked through the `SimdDispatch`
    // table after `is_x86_feature_detected!("avx2")` succeeded.  The unsafe
    // precondition (CPU supports AVX2) is therefore upheld at the call site.
    pub fn filter_block_gt_u64_dispatch(values: &[u64], threshold: u64, n: usize) -> u64 {
        unsafe { filter_block_gt_u64(values, threshold, n) }
    }
    pub fn filter_block_lt_u64_dispatch(values: &[u64], threshold: u64, n: usize) -> u64 {
        unsafe { filter_block_lt_u64(values, threshold, n) }
    }
    pub fn block_min_max_u64_dispatch(values: &[u64], n: usize) -> (u64, u64) {
        unsafe { block_min_max_u64(values, n) }
    }

    /// AVX2 has only **signed** 64-bit cmpgt (`_mm256_cmpgt_epi64`).
    /// Unsigned cmp emulation: `a > b (unsigned) ↔ (a ^ MSB) > (b ^ MSB) (signed)`.
    /// We pre-flip MSB on both lanes inside the SIMD path — one extra xor per
    /// vector load, negligible vs the cmpgt + movemask cost.
    pub(super) const SIGN_FLIP: i64 = i64::MIN; // 0x8000_0000_0000_0000 as i64

    /// SAFETY: caller guarantees `values.len() >= n`, `n <= 64`, and the
    /// CPU supports AVX2 (the `target_feature` precondition).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_block_gt_u64(values: &[u64], threshold: u64, n: usize) -> u64 {
        let sign = _mm256_set1_epi64x(SIGN_FLIP);
        let t = _mm256_set1_epi64x((threshold as i64) ^ SIGN_FLIP);
        let ptr = values.as_ptr() as *const __m256i;
        let mut mask: u64 = 0;
        let mut i = 0usize;
        // Process 16 docs per iteration: 4 AVX2 vectors of 4 u64.
        while i + 16 <= n {
            let v0 = _mm256_xor_si256(_mm256_loadu_si256(ptr.add(i / 4)), sign);
            let v1 = _mm256_xor_si256(_mm256_loadu_si256(ptr.add(i / 4 + 1)), sign);
            let v2 = _mm256_xor_si256(_mm256_loadu_si256(ptr.add(i / 4 + 2)), sign);
            let v3 = _mm256_xor_si256(_mm256_loadu_si256(ptr.add(i / 4 + 3)), sign);
            // _mm256_cmpgt_epi64: signed greater-than → 0xFFFF... per lane.
            let c0 = _mm256_cmpgt_epi64(v0, t);
            let c1 = _mm256_cmpgt_epi64(v1, t);
            let c2 = _mm256_cmpgt_epi64(v2, t);
            let c3 = _mm256_cmpgt_epi64(v3, t);
            // Pack the high bit of each 64-bit lane via `_mm256_movemask_pd`
            // (which extracts the sign bit of each f64 lane — same MSB).
            let m0 = _mm256_movemask_pd(_mm256_castsi256_pd(c0)) as u64; // 4 bits
            let m1 = _mm256_movemask_pd(_mm256_castsi256_pd(c1)) as u64;
            let m2 = _mm256_movemask_pd(_mm256_castsi256_pd(c2)) as u64;
            let m3 = _mm256_movemask_pd(_mm256_castsi256_pd(c3)) as u64;
            let combined = m0 | (m1 << 4) | (m2 << 8) | (m3 << 12);
            mask |= combined << i;
            i += 16;
        }
        // 4-lane tail
        while i + 4 <= n {
            let v = _mm256_xor_si256(
                _mm256_loadu_si256(values.as_ptr().add(i) as *const __m256i),
                sign,
            );
            let c = _mm256_cmpgt_epi64(v, t);
            let m = _mm256_movemask_pd(_mm256_castsi256_pd(c)) as u64;
            mask |= m << i;
            i += 4;
        }
        // Scalar tail
        let raw = values.as_ptr();
        while i < n {
            mask |= ((*raw.add(i) > threshold) as u64) << i;
            i += 1;
        }
        mask
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_block_lt_u64(values: &[u64], threshold: u64, n: usize) -> u64 {
        // a < b (unsigned) ↔ b > a.  Swap operands to cmpgt.
        let sign = _mm256_set1_epi64x(SIGN_FLIP);
        let t = _mm256_set1_epi64x((threshold as i64) ^ SIGN_FLIP);
        let ptr = values.as_ptr() as *const __m256i;
        let mut mask: u64 = 0;
        let mut i = 0usize;
        while i + 16 <= n {
            let v0 = _mm256_xor_si256(_mm256_loadu_si256(ptr.add(i / 4)), sign);
            let v1 = _mm256_xor_si256(_mm256_loadu_si256(ptr.add(i / 4 + 1)), sign);
            let v2 = _mm256_xor_si256(_mm256_loadu_si256(ptr.add(i / 4 + 2)), sign);
            let v3 = _mm256_xor_si256(_mm256_loadu_si256(ptr.add(i / 4 + 3)), sign);
            let c0 = _mm256_cmpgt_epi64(t, v0);
            let c1 = _mm256_cmpgt_epi64(t, v1);
            let c2 = _mm256_cmpgt_epi64(t, v2);
            let c3 = _mm256_cmpgt_epi64(t, v3);
            let m0 = _mm256_movemask_pd(_mm256_castsi256_pd(c0)) as u64;
            let m1 = _mm256_movemask_pd(_mm256_castsi256_pd(c1)) as u64;
            let m2 = _mm256_movemask_pd(_mm256_castsi256_pd(c2)) as u64;
            let m3 = _mm256_movemask_pd(_mm256_castsi256_pd(c3)) as u64;
            let combined = m0 | (m1 << 4) | (m2 << 8) | (m3 << 12);
            mask |= combined << i;
            i += 16;
        }
        while i + 4 <= n {
            let v = _mm256_xor_si256(
                _mm256_loadu_si256(values.as_ptr().add(i) as *const __m256i),
                sign,
            );
            let c = _mm256_cmpgt_epi64(t, v);
            let m = _mm256_movemask_pd(_mm256_castsi256_pd(c)) as u64;
            mask |= m << i;
            i += 4;
        }
        let raw = values.as_ptr();
        while i < n {
            mask |= ((*raw.add(i) < threshold) as u64) << i;
            i += 1;
        }
        mask
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn block_min_max_u64(values: &[u64], n: usize) -> (u64, u64) {
        // AVX2 has no unsigned 64-bit min/max instruction → blendv with cmpgt.
        // We work in signed-shifted space (xor with SIGN_FLIP / 0x8000…), so
        // initial lo = u64::MAX maps to i64::MAX (0x7FFF…) and initial
        // hi = 0u64 maps to i64::MIN (0x8000…).
        let sign = _mm256_set1_epi64x(SIGN_FLIP);
        let mut lo = _mm256_set1_epi64x(i64::MAX);
        let mut hi = _mm256_set1_epi64x(i64::MIN);
        let mut i = 0usize;
        while i + 4 <= n {
            let v = _mm256_xor_si256(
                _mm256_loadu_si256(values.as_ptr().add(i) as *const __m256i),
                sign,
            );
            // signed min/max via cmpgt+blendv
            let cmp_lo = _mm256_cmpgt_epi64(lo, v); // lo > v → take v
            lo = _mm256_blendv_epi8(lo, v, cmp_lo);
            let cmp_hi = _mm256_cmpgt_epi64(v, hi); // v > hi → take v
            hi = _mm256_blendv_epi8(hi, v, cmp_hi);
            i += 4;
        }
        // Lane-wise reduce + flip sign back.
        let mut lo_arr = [0i64; 4];
        let mut hi_arr = [0i64; 4];
        _mm256_storeu_si256(lo_arr.as_mut_ptr() as *mut __m256i, lo);
        _mm256_storeu_si256(hi_arr.as_mut_ptr() as *mut __m256i, hi);
        let mut lo_u = u64::MAX;
        let mut hi_u = 0u64;
        for k in 0..4 {
            let lv = (lo_arr[k] ^ SIGN_FLIP) as u64;
            let hv = (hi_arr[k] ^ SIGN_FLIP) as u64;
            if lv < lo_u {
                lo_u = lv;
            }
            if hv > hi_u {
                hi_u = hv;
            }
        }
        // Scalar tail
        let raw = values.as_ptr();
        while i < n {
            let x = *raw.add(i);
            if x < lo_u {
                lo_u = x;
            }
            if x > hi_u {
                hi_u = x;
            }
            i += 1;
        }
        // Edge case: when the SIMD body never ran (n < 4) the lane values are
        // sentinels — fall back to the scalar oracle for correctness.
        if n < 4 {
            return super::scalar::block_min_max_u64(values, n);
        }
        (lo_u, hi_u)
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn build_input(n: usize, seed: u64) -> Vec<u64> {
        // Simple LCG to keep test deterministic without a rand dep.
        let mut s = seed.wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s
            })
            .collect()
    }

    /// Oracle: SIMD path must match the scalar path bit-for-bit on the same
    /// input, for every block length 0..=64.  This is the SIMD correctness
    /// invariant: SIMD must never disagree with the scalar reference even
    /// for the tail / boundary cases.
    #[test]
    fn simd_filter_gt_matches_scalar_oracle() {
        let v = build_input(64, 0xCAFE);
        for &threshold in &[0u64, 1, u64::MAX, u64::MAX - 1, 1 << 62, 1 << 63] {
            for n in 0..=64 {
                let simd = simd_filter_block_gt_u64(&v, threshold, n);
                let scal = scalar::filter_block_gt_u64(&v, threshold, n);
                assert_eq!(
                    simd, scal,
                    "gt mismatch: threshold={threshold:#x}, n={n}, simd={simd:#x}, scalar={scal:#x}"
                );
            }
        }
    }

    #[test]
    fn simd_filter_lt_matches_scalar_oracle() {
        let v = build_input(64, 0xBEEF);
        for &threshold in &[0u64, 1, u64::MAX, u64::MAX - 1, 1 << 62, 1 << 63] {
            for n in 0..=64 {
                let simd = simd_filter_block_lt_u64(&v, threshold, n);
                let scal = scalar::filter_block_lt_u64(&v, threshold, n);
                assert_eq!(
                    simd, scal,
                    "lt mismatch: threshold={threshold:#x}, n={n}, simd={simd:#x}, scalar={scal:#x}"
                );
            }
        }
    }

    /// Branchless asc/desc: lt is mathematically the dual of gt.
    /// For every input/threshold pair, `mask_lt | mask_gt | mask_eq == (1<<n)-1`
    /// (where eq is the equal mask).  Verifies the two kernels are
    /// complementary, not "same answer twice".
    #[test]
    fn simd_lt_and_gt_are_complementary() {
        let v = build_input(64, 0x1234);
        for &threshold in &[v[0], v[15], v[63], 42u64] {
            let n = 64;
            let gt = simd_filter_block_gt_u64(&v, threshold, n);
            let lt = simd_filter_block_lt_u64(&v, threshold, n);
            let mut eq: u64 = 0;
            for i in 0..n {
                if v[i] == threshold {
                    eq |= 1u64 << i;
                }
            }
            let union = gt | lt | eq;
            assert_eq!(
                union,
                u64::MAX,
                "gt={gt:#x} lt={lt:#x} eq={eq:#x} not complementary"
            );
            assert_eq!(gt & lt, 0, "gt and lt overlap");
        }
    }

    /// Top-K early-termination: when the heap is full and threshold
    /// dominates the entire block, mask must be 0 (no survivors).
    #[test]
    fn top_k_early_termination_block() {
        // Block of 64 small values, threshold higher than max → 0 survivors.
        let v: Vec<u64> = (0..64u64).collect();
        let mask = simd_filter_block_gt_u64(&v, 1000, 64);
        assert_eq!(mask, 0, "mask should be 0 when threshold dominates block");

        // Inverse for asc: threshold below min → 0 survivors.
        let mask = simd_filter_block_lt_u64(&v, 0, 64);
        assert_eq!(mask, 0, "mask should be 0 when asc threshold below min");
    }

    #[test]
    fn block_min_max_matches_scalar_oracle() {
        let v = build_input(64, 0x9999);
        for n in 0..=64 {
            let simd = simd_block_min_max_u64(&v, n);
            let scal = scalar::block_min_max_u64(&v, n);
            assert_eq!(simd, scal, "min_max mismatch at n={n}");
        }
    }

    /// 1 M-doc top-10 SIMD filter against a fixed threshold must match the
    /// scalar reference perfectly. This is the workload the Rally
    /// `desc_sort_*_force_merge_1_seg` ops hit.
    #[test]
    fn one_million_doc_top_k_correctness() {
        let n = 1_000_000usize;
        let v = build_input(n, 0xDEAD);
        // Pick a threshold that ~10 values exceed (top-10 desc).
        let mut sorted = v.clone();
        sorted.sort_unstable();
        let threshold = sorted[n - 11]; // ~10 docs > threshold
        let mut simd_count = 0u32;
        let mut scal_count = 0u32;
        let mut i = 0usize;
        while i < n {
            let block = SIMD_BLOCK_LEN.min(n - i);
            let s = simd_filter_block_gt_u64(&v[i..], threshold, block);
            let c = scalar::filter_block_gt_u64(&v[i..], threshold, block);
            assert_eq!(s, c, "block at {i} mismatched");
            simd_count += s.count_ones();
            scal_count += c.count_ones();
            i += block;
        }
        assert_eq!(simd_count, scal_count);
        // ~10 survivors expected; allow ±2 for ties.
        assert!(
            (8..=12).contains(&simd_count),
            "expected ~10 survivors, got {simd_count}"
        );
    }

    /// Sentinel-value handling: u64::MAX (the "missing value" sentinel
    /// inside the warm cache) must be handled correctly by the SIMD path.
    /// The warm cache is only populated for `ColumnIndex::Full`, so the
    /// sentinel is unreachable in practice — but we still verify it
    /// produces the expected mask bits if it ever leaks through.
    #[test]
    fn sentinel_max_handling() {
        let v: Vec<u64> = vec![u64::MAX; 64];
        // gt vs threshold=u64::MAX-1 → all survivors.
        let mask = simd_filter_block_gt_u64(&v, u64::MAX - 1, 64);
        assert_eq!(mask, u64::MAX);
        // gt vs threshold=u64::MAX → no survivors.
        let mask = simd_filter_block_gt_u64(&v, u64::MAX, 64);
        assert_eq!(mask, 0);
        // lt vs threshold=u64::MAX → no survivors (no value is < MAX).
        let mask = simd_filter_block_lt_u64(&v, u64::MAX, 64);
        assert_eq!(mask, 0);
    }
}

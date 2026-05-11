//! CUDA Tensor Core fast path — failure-mode and contract regression tests.
//!
//! Pairs with `cuda_tensor_core_parity.rs` (which proves byte-equality on
//! the happy path) by pinning the **fallback contract**: the public API
//! must reject malformed inputs with the documented error variants
//! before any GPU call, and the BMMA dim-multiple-of-128 gate must
//! return `CpuFallback` rather than panicking on the inline-PTX kernel.
//!
//! Skips at runtime (passes with a printed message) on hosts without a
//! reachable NVIDIA device, so the same target runs unchanged on
//! AMD / Apple / Intel / CI runners without GPU passthrough — same
//! pattern as `cuda_tensor_core_parity.rs`.

#![cfg(feature = "cuda-tensor-core")]

use tantivy_gpu::error::GpuError;
use tantivy_gpu::vector::cuda_tensor_core::CudaTensorCoreKernel;

fn try_build() -> Option<CudaTensorCoreKernel> {
    match CudaTensorCoreKernel::try_new() {
        Ok(k) => Some(k),
        Err(e) => {
            eprintln!("CUDA Tensor Core unavailable on this host: {e}");
            None
        }
    }
}

// ── compute_batched ────────────────────────────────────────────────────

#[test]
fn compute_batched_empty_queries_returns_empty() {
    let Some(k) = try_build() else { return };
    let r = k.compute_batched(&[], &[0u32; 8], 0, 1, 256).expect("ok");
    assert!(r.is_empty(), "Q=0 must short-circuit to empty result");
}

#[test]
fn compute_batched_empty_corpus_returns_empty() {
    let Some(k) = try_build() else { return };
    let r = k.compute_batched(&[0u32; 8], &[], 1, 0, 256).expect("ok");
    assert!(r.is_empty(), "N=0 must short-circuit to empty result");
}

#[test]
fn compute_batched_zero_dim_returns_empty() {
    let Some(k) = try_build() else { return };
    // dim_bits = 0 short-circuits before any sizing check, so input
    // length checks aren't tripped — return value is just empty.
    let r = k.compute_batched(&[], &[], 1, 1, 0).expect("ok");
    assert!(
        r.is_empty(),
        "dim_bits=0 must short-circuit to empty result"
    );
}

#[test]
fn compute_batched_query_length_mismatch_errors() {
    let Some(k) = try_build() else { return };
    // dim_bits=256 → dim_u32=8. num_queries=2 expects 16 u32s; pass 7.
    let bad_queries = vec![0u32; 7];
    let corpus = vec![0u32; 8];
    let err = k
        .compute_batched(&bad_queries, &corpus, 2, 1, 256)
        .expect_err("must reject mismatched queries length");
    assert!(
        matches!(err, GpuError::ColumnTypeMismatch { .. }),
        "expected ColumnTypeMismatch, got {err:?}"
    );
}

#[test]
fn compute_batched_corpus_length_mismatch_errors() {
    let Some(k) = try_build() else { return };
    let queries = vec![0u32; 8];
    let bad_corpus = vec![0u32; 5];
    let err = k
        .compute_batched(&queries, &bad_corpus, 1, 1, 256)
        .expect_err("must reject mismatched corpus length");
    assert!(
        matches!(err, GpuError::ColumnTypeMismatch { .. }),
        "expected ColumnTypeMismatch, got {err:?}"
    );
}

// ── prepare_corpus ─────────────────────────────────────────────────────

#[test]
fn prepare_corpus_zero_vecs_errors() {
    let Some(k) = try_build() else { return };
    match k.prepare_corpus(&[], 0, 256) {
        Ok(_) => panic!("num_vecs=0 must reject, got Ok"),
        Err(GpuError::ColumnTypeMismatch { .. }) => {}
        Err(other) => panic!("expected ColumnTypeMismatch, got {other:?}"),
    }
}

#[test]
fn prepare_corpus_zero_dim_errors() {
    let Some(k) = try_build() else { return };
    match k.prepare_corpus(&[], 1, 0) {
        Ok(_) => panic!("dim_bits=0 must reject, got Ok"),
        Err(GpuError::ColumnTypeMismatch { .. }) => {}
        Err(other) => panic!("expected ColumnTypeMismatch, got {other:?}"),
    }
}

#[test]
fn prepare_corpus_length_mismatch_errors() {
    let Some(k) = try_build() else { return };
    let bad = vec![0u32; 3]; // num_vecs=2, dim_u32=8 → expects 16
    match k.prepare_corpus(&bad, 2, 256) {
        Ok(_) => panic!("must reject mismatched corpus length, got Ok"),
        Err(GpuError::ColumnTypeMismatch { .. }) => {}
        Err(other) => panic!("expected ColumnTypeMismatch, got {other:?}"),
    }
}

// ── CudaBinaryCorpus query-side contract ──────────────────────────────

#[test]
fn cached_compute_batched_query_length_mismatch_errors() {
    let Some(k) = try_build() else { return };
    let corpus = vec![0u32; 8];
    let cached = k.prepare_corpus(&corpus, 1, 256).expect("prepare");
    let bad_queries = vec![0u32; 5]; // expects num_queries=1 × dim_u32=8 = 8
    let err = cached
        .compute_batched(&bad_queries, 1)
        .expect_err("must reject mismatched query length");
    assert!(
        matches!(err, GpuError::ColumnTypeMismatch { .. }),
        "expected ColumnTypeMismatch, got {err:?}"
    );
}

#[test]
fn cached_compute_batched_into_pinned_too_small_buffer_errors() {
    let Some(k) = try_build() else { return };
    let corpus = vec![0u32; 8 * 4]; // num_vecs=4, dim_u32=8
    let cached = k.prepare_corpus(&corpus, 4, 256).expect("prepare");
    let queries = vec![0u32; 8 * 2]; // num_queries=2, dim_u32=8
    let mut pinned = k.alloc_pinned_u32(2 * 4 - 1).expect("alloc");
    let err = cached
        .compute_batched_into_pinned(&queries, 2, &mut pinned)
        .expect_err("must reject too-small pinned buffer");
    assert!(
        matches!(err, GpuError::ColumnTypeMismatch { .. }),
        "expected ColumnTypeMismatch on too-small pinned buffer, got {err:?}"
    );
}

#[test]
fn cached_compute_batched_into_pinned_zero_queries_returns_empty() {
    let Some(k) = try_build() else { return };
    let corpus = vec![0u32; 8 * 4];
    let cached = k.prepare_corpus(&corpus, 4, 256).expect("prepare");
    // Allocate a non-zero pinned buffer — Q=0 should short-circuit
    // before touching it.
    let mut pinned = k.alloc_pinned_u32(1).expect("alloc 1");
    cached
        .compute_batched_into_pinned(&[], 0, &mut pinned)
        .expect("Q=0 must short-circuit, not error");
    // Buffer is unchanged — we only assert the call returned cleanly.
}

// ── H-3 regression: OOM must propagate, not CpuFallback ────────────────
//
// `cuMemHostAlloc(absurd_size)` legitimately returns
// `CUDA_ERROR_OUT_OF_MEMORY`. The previous error mapping conflated this
// with init / config failures and produced `GpuError::CpuFallback`, so
// the caller had no way to distinguish "GPU unavailable on this host
// — retry on CPU" from "this allocation is too large — release working
// set or batch differently". OOM must surface as
// `GpuError::OutOfMemory` so callers can react appropriately.
#[test]
fn pinned_alloc_oom_propagates_as_out_of_memory() {
    let Some(k) = try_build() else { return };
    // 1 PiB request — guaranteed to exceed every realistic host's pinned
    // memory budget. We don't actually want this to succeed; we want the
    // failure to be classified.
    let absurd_len: usize = 1usize << 48;
    let err = match k.alloc_pinned_u32(absurd_len) {
        Ok(_) => panic!("1 PiB pinned alloc must fail, but succeeded"),
        Err(e) => e,
    };
    assert!(
        matches!(err, GpuError::OutOfMemory { .. }),
        "expected GpuError::OutOfMemory for cuMemHostAlloc OOM, got {err:?}"
    );
}

// ── C-1 regression: alloc_pinned_u32(0) must not UB ────────────────────
//
// `cuMemHostAlloc(0, ..)` is implementation-defined per the CUDA Driver
// API: some driver versions return NULL legitimately. The previous code
// guarded only with `debug_assert!(!raw.is_null())`, which is stripped in
// release builds — a release-mode caller could store NULL in `ptr` and
// trigger UB on the next `from_raw_parts`/`memcpy_dtoh_async`.
// `alloc_pinned_u32(0)` must therefore short-circuit before calling the
// driver and yield a safe (empty, never dereferenced) buffer.
#[test]
fn pinned_zero_length_alloc_does_not_underflow() {
    let Some(k) = try_build() else { return };
    let pinned = k.alloc_pinned_u32(0).expect(
        "alloc_pinned_u32(0) must succeed without invoking cuMemHostAlloc(0); release-mode null \
         deref is UB",
    );
    assert_eq!(pinned.len(), 0, "zero-length buffer must report len = 0");
    assert!(pinned.is_empty(), "zero-length buffer must be is_empty()");
    let slice = pinned.as_slice();
    assert_eq!(slice.len(), 0, "as_slice() must yield empty slice");
    // Dropping at scope end must not call cuMemFreeHost(NULL or dangling).
}

// ── knn_search ─────────────────────────────────────────────────────────

#[test]
fn knn_search_zero_k_returns_empty_rows() {
    let Some(k) = try_build() else { return };
    let queries = vec![0u32; 8 * 2];
    let corpus = vec![0u32; 8 * 4];
    let r = k
        .knn_search(&queries, &corpus, 2, 4, 256, 0)
        .expect("k=0 ok");
    assert_eq!(
        r.len(),
        2,
        "k=0 must still return one (empty) row per query"
    );
    assert!(r.iter().all(|row| row.is_empty()));
}

#[test]
fn knn_search_zero_queries_returns_empty() {
    let Some(k) = try_build() else { return };
    let corpus = vec![0u32; 8 * 4];
    let r = k.knn_search(&[], &corpus, 0, 4, 256, 10).expect("Q=0 ok");
    assert!(r.is_empty(), "Q=0 must short-circuit to empty Vec");
}

// ── compute_batched_bmma — dim-bits gate ───────────────────────────────

#[test]
fn bmma_non_multiple_of_128_falls_back() {
    let Some(k) = try_build() else { return };
    // dim_bits = 256 is OK, dim_bits = 100 is not (not a multiple of
    // 128). The PoC kernel's K-tile is fixed at 128 bits, so the
    // public API must reject this with CpuFallback so callers can
    // route to the INT8 IMMA path.
    let dim_bits = 100usize;
    let dim_u32 = dim_bits.div_ceil(32);
    let queries = vec![0u32; dim_u32];
    let corpus = vec![0u32; dim_u32];
    let err = k
        .compute_batched_bmma(&queries, &corpus, 1, 1, dim_bits)
        .expect_err("must reject non-128-multiple dim_bits");
    assert!(
        matches!(err, GpuError::CpuFallback { .. }),
        "expected CpuFallback for dim_bits=100, got {err:?}"
    );
}

#[test]
fn bmma_one_short_of_128_multiple_falls_back() {
    let Some(k) = try_build() else { return };
    // 384 = 3 × 128 (good); 383 = not a multiple → CpuFallback.
    let dim_bits = 383usize;
    let dim_u32 = dim_bits.div_ceil(32);
    let queries = vec![0u32; dim_u32];
    let corpus = vec![0u32; dim_u32];
    let err = k
        .compute_batched_bmma(&queries, &corpus, 1, 1, dim_bits)
        .expect_err("must reject 383");
    assert!(matches!(err, GpuError::CpuFallback { .. }));
}

#[test]
fn bmma_zero_queries_returns_empty() {
    let Some(k) = try_build() else { return };
    let r = k
        .compute_batched_bmma(&[], &[0u32; 4], 0, 1, 128)
        .expect("Q=0 ok");
    assert!(r.is_empty());
}

#[test]
fn bmma_zero_corpus_returns_empty() {
    let Some(k) = try_build() else { return };
    let r = k
        .compute_batched_bmma(&[0u32; 4], &[], 1, 0, 128)
        .expect("N=0 ok");
    assert!(r.is_empty());
}

#[test]
fn bmma_zero_dim_returns_empty() {
    let Some(k) = try_build() else { return };
    // The Q=0/N=0/dim_bits=0 short-circuit takes precedence over the
    // %128 gate: dim_bits=0 returns Ok(empty), not CpuFallback.
    let r = k
        .compute_batched_bmma(&[], &[], 1, 1, 0)
        .expect("dim_bits=0 ok");
    assert!(r.is_empty());
}

#[test]
fn bmma_query_length_mismatch_errors() {
    let Some(k) = try_build() else { return };
    // dim_bits=256 → dim_u32=8; pass 7 for num_queries=1.
    let bad = vec![0u32; 7];
    let corpus = vec![0u32; 8];
    let err = k
        .compute_batched_bmma(&bad, &corpus, 1, 1, 256)
        .expect_err("must reject mismatched queries length");
    assert!(
        matches!(err, GpuError::ColumnTypeMismatch { .. }),
        "expected ColumnTypeMismatch, got {err:?}"
    );
}

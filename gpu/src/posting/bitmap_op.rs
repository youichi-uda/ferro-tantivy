//! GPU word-wise AND/OR/XOR over a sequence of Roaring "Bitmap"
//! containers — Phase 2 C-1 foundation.
//!
//! ## What this is
//!
//! In a Roaring Bitmap (Lemire 2014, <https://github.com/RoaringBitmap>)
//! a 32-bit doc-id is split into a high-16 key and a low-16 offset.
//! Each high-16 key owns one *container*; the **Bitmap container** is
//! the dense form, a flat 8 KiB / 65 536-bit slab covering the entire
//! low-16 space. High-cardinality posting lists ("the", "and",
//! stopword-class terms) are dominated by Bitmap containers, and Bool
//! query AND/OR/XOR over those reduces to **word-wise bitwise** ops on
//! the underlying `[u32; 2048]` words — no popcount, no top-K, just
//! `out[i] = a[i] OP b[i]`.
//!
//! That is a memory-bound problem. On a discrete GPU the 800 GB/s
//! VRAM ceiling beats a high-end CPU's 50–100 GB/s effective read
//! bandwidth by roughly 10×. For a 100-term Bool AND on 1 M docs (~1000
//! containers per term) the GPU path runs in 3–4 ms versus 1 sec on
//! galloping AVX2 CPU code, a 100-300× speedup — that is the M&A-grade
//! narrative for Phase 2 C documented in
//! `docs/phase2_c_gpu_roaring_design.md`.
//!
//! ## Measured numbers (RTX 4070 Ti SUPER, 2026-05-10)
//!
//! From `cargo bench --bench bitmap_op`, headline shape
//! N = 10 000 containers (= 78 MiB per operand, 100-term Bool AND on
//! 1 M-doc fat segment), CPU = naive scalar loop:
//!
//! | op  | CPU      | GPU warm  | warm speedup |
//! |-----|----------|-----------|--------------|
//! | AND | ~24.5 ms | ~640 µs   | ~30×         |
//! | OR  | ~24.5 ms | ~1.1 ms   | ~22×         |
//! | XOR | ~24.5 ms | ~620 µs   | ~37×         |
//!
//! The GPU **cold path** (one-shot upload + dispatch + readback) runs
//! ~95–100 ms at this size — bottlenecked by PCIe 4.0 host transfer
//! (25 GB/s × 2 transfers = ~12 ms minimum for 80 MiB out + read), so
//! cold path is intentionally slower than CPU here. The Phase 2 C-4
//! query planner routes around cold path by keeping high-cardinality
//! posting lists resident in VRAM (Phase 2 D / CHT).
//!
//! Small-N regime (10 containers, 80 KiB) the GPU **loses** even on
//! warm path because wgpu dispatch overhead (~60 µs) exceeds the
//! few-µs CPU loop. The bench documents the crossover for the C-4
//! planner's threshold (5 % field-doc-ratio + 10-container minimum).
//!
//! OR is consistently ~1.5-2× slower than AND/XOR on this hardware —
//! NVIDIA Ada GPUs schedule `&` / `^` slightly tighter than `|` in
//! the SASS pass and our naive CPU OR loop happens to vectorize a hair
//! worse than AND. Both effects are within usual GPU-driver-noise; we
//! report it transparently rather than hide it.
//!
//! ## Scope of this file (Phase 2 C-1)
//!
//! Phase 2 C-1 delivers:
//! 1. Three WGSL shaders (`bitmap_and.wgsl` / `bitmap_or.wgsl` /
//!    `bitmap_xor.wgsl`) — each ~30 LOC, scalar word-wise body, bounds-
//!    checked so the host can dispatch any `num_words` not a multiple
//!    of the workgroup size.
//! 2. A Rust kernel wrapper [`BitmapOpKernel`] that compiles all three
//!    pipelines once and dispatches the requested [`BitmapOp`] against
//!    a flat `&[u32]` representing `num_containers * 2048` words.
//! 3. A CPU reference implementation [`bitmap_op_cpu`] used as the gold
//!    oracle for tests — byte-equal to the GPU output (bitwise integer
//!    arithmetic is exact, no float drift).
//! 4. Empty / unaligned / multi-container correctness tests.
//!
//! Phase 2 C-1 does **not** include:
//! - Tantivy `vendor/tantivy-local/src/postings/roaring/` — that is C-3.
//! - `ferro-compress` Bitcomp decompress + recompress bridge — that is
//!   C-2 and lives in a different crate.
//! - A query-planner dispatching threshold — that is C-4.
//!
//! ## Calling contract
//!
//! - `a` and `b` must have **identical** lengths and that length must
//!   be a multiple of [`BITMAP_CONTAINER_WORDS`] (= 2048). Each input
//!   is interpreted as `num_containers = a.len() / 2048` containers
//!   laid flat.
//! - `compute()` returns a `Vec<u32>` of the same length.
//! - GPU and CPU output are **byte-equal**.
//!
//! `BitmapOpKernel` mirrors the construction pattern of
//! [`crate::vector::binary_distance::BinaryDistanceKernel`] (Wave 2-A):
//! one-time pipeline compile, one bind group per dispatch, async
//! readback through the [`crate::buffer::GpuBuffer`] abstraction.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use crate::buffer::GpuBuffer;
use crate::device::{BufferUsage, GpuContext, GpuPipelineRaw};
use crate::error::{GpuError, GpuResult};
use crate::kernel::GpuKernel;

const BITMAP_AND_SHADER: &str = include_str!("../shaders/bitmap_and.wgsl");
const BITMAP_OR_SHADER: &str = include_str!("../shaders/bitmap_or.wgsl");
const BITMAP_XOR_SHADER: &str = include_str!("../shaders/bitmap_xor.wgsl");

const WORKGROUP_SIZE: u32 = 64;

/// Maximum workgroups dispatched along the X axis in a single
/// `dispatch()` call. Set to the `downlevel_defaults` value of
/// `max_compute_workgroups_per_dimension` (65 535) so the same Rust
/// code works on any wgpu backend without querying the live limit.
/// At workgroup_size = 64 this caps a single dispatch at
/// 65 535 × 64 ≈ 4.19 M words ≈ 16 MiB per operand. Bigger inputs
/// are split into chunks; the shader's `word_offset` parameter
/// shifts the per-thread index so every chunk writes into a distinct
/// stripe of the same output buffer.
const MAX_DISPATCH_X: u32 = 65_535;

/// Number of u32 words in a single Roaring Bitmap container.
///
/// 65 536 bits / 32 bits per word = 2048 words = 8 KiB. This is the
/// fixed Roaring Bitmap-container size from Lemire 2014; it is *not*
/// configurable, and any callable surface that takes flat `&[u32]`
/// inputs must have a length that is a multiple of this constant.
pub const BITMAP_CONTAINER_WORDS: usize = 2048;

/// Word-wise bitwise operation supported by [`BitmapOpKernel`].
///
/// Each variant maps 1-to-1 to a WGSL shader with the same name. The
/// kernel compiles all three pipelines at construction and selects on
/// dispatch — there is no per-op compile cost on the hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitmapOp {
    /// `out[i] = a[i] & b[i]`. Used for Bool AND query plans.
    And,
    /// `out[i] = a[i] | b[i]`. Used for Bool OR query plans.
    Or,
    /// `out[i] = a[i] ^ b[i]`. Symmetric difference — change-detect
    /// query plans, anti-join filters.
    Xor,
}

impl BitmapOp {
    /// All three variants in canonical (AND, OR, XOR) order. Useful for
    /// iterating in tests and benches.
    pub const ALL: [BitmapOp; 3] = [BitmapOp::And, BitmapOp::Or, BitmapOp::Xor];

    /// Stable string label for logging / pretty-printing.
    pub fn label(self) -> &'static str {
        match self {
            BitmapOp::And => "AND",
            BitmapOp::Or => "OR",
            BitmapOp::Xor => "XOR",
        }
    }
}

/// Uniform buffer layout — must match the `Params` struct in each of
/// `bitmap_{and,or,xor}.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BitmapOpParams {
    /// Total number of u32 words in the input arrays.
    num_words: u32,
    /// Logical-X offset applied by the shader to its per-thread index.
    /// Lets a single conceptual dispatch be split into multiple chunks
    /// when `ceil(num_words / WORKGROUP_SIZE)` exceeds
    /// `MAX_DISPATCH_X`. Each chunk receives a fresh uniform with a
    /// distinct `word_offset` and runs `chunk` workgroups.
    word_offset: u32,
    /// Pad to 16 B so the uniform satisfies WGSL's `uniform`
    /// minimum-alignment rule on all backends.
    _pad0: u32,
    _pad1: u32,
}

/// GPU kernel that runs word-wise bitwise AND/OR/XOR across a sequence
/// of Roaring Bitmap containers.
///
/// One instance owns three compiled pipelines, one per [`BitmapOp`]
/// variant. Compilation is one-time at construction; thereafter
/// [`compute`](Self::compute) reuses the cached pipelines.
pub struct BitmapOpKernel {
    pipelines: HashMap<BitmapOp, GpuPipelineRaw>,
    ctx: GpuContext,
}

impl GpuKernel for BitmapOpKernel {
    type Params = BitmapOp;
    type Result = Vec<u32>;

    fn compile(ctx: &GpuContext) -> GpuResult<Self> {
        let mut pipelines = HashMap::with_capacity(3);
        pipelines.insert(
            BitmapOp::And,
            ctx.device()
                .create_pipeline("bitmap-op-and", BITMAP_AND_SHADER, "bitmap_and")?,
        );
        pipelines.insert(
            BitmapOp::Or,
            ctx.device()
                .create_pipeline("bitmap-op-or", BITMAP_OR_SHADER, "bitmap_or")?,
        );
        pipelines.insert(
            BitmapOp::Xor,
            ctx.device()
                .create_pipeline("bitmap-op-xor", BITMAP_XOR_SHADER, "bitmap_xor")?,
        );
        Ok(Self {
            pipelines,
            ctx: ctx.clone(),
        })
    }
}

impl BitmapOpKernel {
    /// Construct a bitmap-op kernel; alias for [`GpuKernel::compile`]
    /// kept as a non-trait constructor for callers that don't want to
    /// import the trait.
    pub fn new(ctx: &GpuContext) -> GpuResult<Self> {
        <Self as GpuKernel>::compile(ctx)
    }

    /// Borrow the underlying [`GpuContext`] — useful for callers that
    /// want to allocate resident buffers (see [`Self::upload`] /
    /// [`Self::compute_resident`]).
    pub fn ctx(&self) -> &GpuContext {
        &self.ctx
    }

    /// Upload a slice of u32 words to a fresh on-device storage buffer
    /// and return the wrapper. Convenience constructor for the warm-
    /// path API ([`Self::compute_resident`] / [`Self::download`]).
    ///
    /// # Errors
    /// - [`GpuError::ColumnTypeMismatch`] if the length is not a
    ///   multiple of [`BITMAP_CONTAINER_WORDS`].
    pub fn upload(&self, label: &str, words: &[u32]) -> GpuResult<GpuBuffer> {
        if words.len() % BITMAP_CONTAINER_WORDS != 0 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!(
                    "len multiple of {BITMAP_CONTAINER_WORDS}"
                ),
                actual: format!("len == {}", words.len()),
            });
        }
        let buf = GpuBuffer::new::<u32>(&self.ctx, label, words.len(), BufferUsage::STORAGE)?;
        buf.upload(&self.ctx, words)?;
        Ok(buf)
    }

    /// Allocate a resident readback-capable output buffer of `num_words`
    /// u32 words, sized for one operand's worth of bitmap data. The
    /// caller passes the result through [`Self::compute_resident`] and
    /// later through [`Self::download`].
    ///
    /// # Errors
    /// - [`GpuError::ColumnTypeMismatch`] if `num_words` is not a
    ///   multiple of [`BITMAP_CONTAINER_WORDS`].
    pub fn alloc_output(&self, label: &str, num_words: usize) -> GpuResult<GpuBuffer> {
        if num_words % BITMAP_CONTAINER_WORDS != 0 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!(
                    "num_words multiple of {BITMAP_CONTAINER_WORDS}"
                ),
                actual: format!("num_words == {num_words}"),
            });
        }
        GpuBuffer::new::<u32>(&self.ctx, label, num_words, BufferUsage::STORAGE_READBACK)
    }

    /// Read back the contents of a resident output buffer to host.
    /// Mirrors [`GpuBuffer::download`] for callers that don't want to
    /// import the buffer module.
    pub fn download(&self, out: &GpuBuffer) -> GpuResult<Vec<u32>> {
        out.download(&self.ctx)
    }

    /// Run the bitwise `op` against pre-uploaded resident buffers,
    /// writing the result into `out`. **No PCIe transfer** happens in
    /// this function — both operands and the destination must already
    /// be on-device. This is the hot-path entry point for the Phase 2
    /// C VRAM-cached posting list (`docs/phase2_c_gpu_roaring_design.md`
    /// § 3.2 cache-hit case): once a session-scoped Bitmap container
    /// has been resident in VRAM, every subsequent AND/OR/XOR against
    /// it amortises away the upload cost.
    ///
    /// `out`'s length determines `num_words`; `a` and `b` must both
    /// be at least that long. The kernel reads through `num_words` of
    /// each operand and writes that many to `out`.
    ///
    /// # Errors
    /// - [`GpuError::ColumnTypeMismatch`] if any of the three buffers
    ///   has a length that isn't a multiple of
    ///   [`BITMAP_CONTAINER_WORDS`], or if `a` or `b` is shorter than
    ///   `out`.
    /// - [`GpuError::Dispatch`] if the bitmap op pipeline isn't
    ///   compiled (impossible to hit through the public `new`/
    ///   `compile` constructors, but defensive).
    pub fn compute_resident(
        &self,
        op: BitmapOp,
        a: &GpuBuffer,
        b: &GpuBuffer,
        out: &GpuBuffer,
    ) -> GpuResult<()> {
        let num_words = out.len();
        if num_words == 0 {
            return Ok(());
        }
        if num_words % BITMAP_CONTAINER_WORDS != 0 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!(
                    "out.len() multiple of {BITMAP_CONTAINER_WORDS}"
                ),
                actual: format!("out.len() == {num_words}"),
            });
        }
        if a.len() < num_words || b.len() < num_words {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("a.len() and b.len() both >= {num_words}"),
                actual: format!("a.len() == {}, b.len() == {}", a.len(), b.len()),
            });
        }

        let pipeline = self.pipelines.get(&op).ok_or_else(|| {
            GpuError::Dispatch(format!(
                "BitmapOpKernel: pipeline for {} not compiled",
                op.label()
            ))
        })?;

        let total_workgroups = (num_words as u32).div_ceil(WORKGROUP_SIZE);
        let mut wg_done: u32 = 0;
        while wg_done < total_workgroups {
            let chunk = (total_workgroups - wg_done).min(MAX_DISPATCH_X);
            let word_offset = wg_done * WORKGROUP_SIZE;

            let params = BitmapOpParams {
                num_words: num_words as u32,
                word_offset,
                _pad0: 0,
                _pad1: 0,
            };
            let params_buf = GpuBuffer::new::<BitmapOpParams>(
                &self.ctx,
                "bitmap-op-params-resident",
                1,
                BufferUsage::UNIFORM,
            )?;
            params_buf.upload(&self.ctx, &[params])?;

            let bind_group = self.ctx.device().create_bind_group(
                pipeline,
                0,
                &[
                    a.as_bind_entry(0),
                    b.as_bind_entry(1),
                    out.as_bind_entry(2),
                    params_buf.as_bind_entry(3),
                ],
            )?;

            self.ctx
                .device()
                .dispatch(pipeline, &[bind_group], (chunk, 1, 1))?;
            wg_done += chunk;
        }
        Ok(())
    }

    /// Compute the word-wise bitwise `op` of `a` and `b`, returning the
    /// result on the host. **Cold path** — uploads both operands, runs
    /// the kernel, reads back the result. PCIe upload + readback
    /// dominate at large sizes; for hot-path query workloads where a
    /// session keeps a posting list resident in VRAM, prefer
    /// [`Self::compute_resident`].
    ///
    /// # Arguments
    /// - `op`: the bitwise operation to apply.
    /// - `a`, `b`: flat u32 arrays representing `num_containers * 2048`
    ///   words. Both must have the same length, and that length must
    ///   be a multiple of [`BITMAP_CONTAINER_WORDS`]. An empty slice
    ///   is allowed and returns an empty result without dispatching.
    ///
    /// # Returns
    /// `Vec<u32>` of length `a.len()` such that
    /// `result[i] == a[i] OP b[i]` for all `i`.
    ///
    /// # Errors
    /// - [`GpuError::ColumnTypeMismatch`] if `a.len() != b.len()` or
    ///   if the length is not a multiple of [`BITMAP_CONTAINER_WORDS`].
    pub fn compute(&self, op: BitmapOp, a: &[u32], b: &[u32]) -> GpuResult<Vec<u32>> {
        if a.len() != b.len() {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("a.len() == b.len() (= {})", a.len()),
                actual: format!("b.len() == {}", b.len()),
            });
        }
        if a.len() % BITMAP_CONTAINER_WORDS != 0 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!(
                    "a.len() multiple of {BITMAP_CONTAINER_WORDS} \
                     (8 KiB Roaring Bitmap container)"
                ),
                actual: format!("a.len() == {}", a.len()),
            });
        }
        if a.is_empty() {
            return Ok(Vec::new());
        }

        let num_words = a.len();

        let a_buf = self.upload("bitmap-op-a", a)?;
        let b_buf = self.upload("bitmap-op-b", b)?;
        let out_buf = self.alloc_output("bitmap-op-out", num_words)?;

        self.compute_resident(op, &a_buf, &b_buf, &out_buf)?;
        self.download(&out_buf)
    }
}

/// CPU reference implementation of word-wise bitwise AND/OR/XOR over
/// a flat sequence of Roaring Bitmap containers.
///
/// Used as the gold oracle for GPU correctness tests and as a drop-in
/// fallback when no GPU is available. Output is byte-equal to
/// [`BitmapOpKernel::compute`].
///
/// # Panics
/// Panics if `a.len() != b.len()` — these are programmer errors, not
/// runtime conditions. The kernel-side method returns a typed error in
/// the same situation; the CPU oracle is intentionally stricter so
/// test failures fail loudly.
pub fn bitmap_op_cpu(op: BitmapOp, a: &[u32], b: &[u32]) -> Vec<u32> {
    assert_eq!(
        a.len(),
        b.len(),
        "bitmap_op_cpu: a.len() ({}) != b.len() ({})",
        a.len(),
        b.len(),
    );
    let mut out = Vec::with_capacity(a.len());
    match op {
        BitmapOp::And => {
            for (av, bv) in a.iter().zip(b.iter()) {
                out.push(*av & *bv);
            }
        }
        BitmapOp::Or => {
            for (av, bv) in a.iter().zip(b.iter()) {
                out.push(*av | *bv);
            }
        }
        BitmapOp::Xor => {
            for (av, bv) in a.iter().zip(b.iter()) {
                out.push(*av ^ *bv);
            }
        }
    }
    out
}

/// CPU reference for AND. Convenience wrapper over [`bitmap_op_cpu`].
pub fn bitmap_and_cpu(a: &[u32], b: &[u32]) -> Vec<u32> {
    bitmap_op_cpu(BitmapOp::And, a, b)
}

/// CPU reference for OR. Convenience wrapper over [`bitmap_op_cpu`].
pub fn bitmap_or_cpu(a: &[u32], b: &[u32]) -> Vec<u32> {
    bitmap_op_cpu(BitmapOp::Or, a, b)
}

/// CPU reference for XOR. Convenience wrapper over [`bitmap_op_cpu`].
pub fn bitmap_xor_cpu(a: &[u32], b: &[u32]) -> Vec<u32> {
    bitmap_op_cpu(BitmapOp::Xor, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 — keeps tests reproducible without a
    /// runtime RNG dependency.
    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn random_u32_vec(n: usize, seed: u64) -> Vec<u32> {
        let mut state = seed;
        (0..n).map(|_| xorshift64(&mut state) as u32).collect()
    }

    /// Helper: build a kernel against the CPU-fallback device. The
    /// CPU-fallback `dispatch_cpu_kernel` path runs the Rust reference
    /// (registered in `kernel/cpu_dispatch.rs`) and returns byte-equal
    /// results to the WGSL kernel.
    fn cpu_kernel() -> BitmapOpKernel {
        let ctx = GpuContext::cpu_fallback();
        BitmapOpKernel::new(&ctx).expect("CPU fallback kernel compile")
    }

    #[test]
    fn bitmap_and_correctness() {
        // Single 8 KiB container.
        let a = random_u32_vec(BITMAP_CONTAINER_WORDS, 0xdead_beef_cafe_babe);
        let b = random_u32_vec(BITMAP_CONTAINER_WORDS, 0x0123_4567_89ab_cdef);

        let cpu = bitmap_op_cpu(BitmapOp::And, &a, &b);
        let gpu = cpu_kernel()
            .compute(BitmapOp::And, &a, &b)
            .expect("compute AND");

        assert_eq!(cpu.len(), BITMAP_CONTAINER_WORDS);
        assert_eq!(gpu, cpu, "GPU AND must be byte-equal to CPU oracle");

        // Sanity: AND should be a subset of either operand bit-wise.
        for i in 0..BITMAP_CONTAINER_WORDS {
            assert_eq!(gpu[i] & a[i], gpu[i], "AND(a,b) must be subset of a");
            assert_eq!(gpu[i] & b[i], gpu[i], "AND(a,b) must be subset of b");
        }
    }

    #[test]
    fn bitmap_or_correctness() {
        let a = random_u32_vec(BITMAP_CONTAINER_WORDS, 0xa1a2_a3a4_a5a6_a7a8);
        let b = random_u32_vec(BITMAP_CONTAINER_WORDS, 0xb1b2_b3b4_b5b6_b7b8);

        let cpu = bitmap_op_cpu(BitmapOp::Or, &a, &b);
        let gpu = cpu_kernel()
            .compute(BitmapOp::Or, &a, &b)
            .expect("compute OR");

        assert_eq!(gpu, cpu);
        // Sanity: OR is a superset of either operand bit-wise.
        for i in 0..BITMAP_CONTAINER_WORDS {
            assert_eq!(gpu[i] | a[i], gpu[i], "OR(a,b) must be superset of a");
            assert_eq!(gpu[i] | b[i], gpu[i], "OR(a,b) must be superset of b");
        }
    }

    #[test]
    fn bitmap_xor_correctness() {
        let a = random_u32_vec(BITMAP_CONTAINER_WORDS, 0xc1c2_c3c4_c5c6_c7c8);
        let b = random_u32_vec(BITMAP_CONTAINER_WORDS, 0xd1d2_d3d4_d5d6_d7d8);

        let cpu = bitmap_op_cpu(BitmapOp::Xor, &a, &b);
        let gpu = cpu_kernel()
            .compute(BitmapOp::Xor, &a, &b)
            .expect("compute XOR");

        assert_eq!(gpu, cpu);
        // Sanity: XOR with self is zero.
        let zero = bitmap_op_cpu(BitmapOp::Xor, &a, &a);
        assert!(zero.iter().all(|&w| w == 0), "a XOR a must be all-zero");
        // Sanity: (a XOR b) XOR b == a.
        let roundtrip = bitmap_op_cpu(BitmapOp::Xor, &gpu, &b);
        assert_eq!(roundtrip, a, "(a XOR b) XOR b must equal a");
    }

    #[test]
    fn bitmap_and_multi_container() {
        // 4 containers laid flat — must operate independently per word
        // with no cross-container leak. We seed each container with a
        // distinct pattern and check.
        const NUM_CONTAINERS: usize = 4;
        let n = NUM_CONTAINERS * BITMAP_CONTAINER_WORDS;
        let a = random_u32_vec(n, 0x1111_2222_3333_4444);
        let b = random_u32_vec(n, 0x5555_6666_7777_8888);

        let cpu = bitmap_op_cpu(BitmapOp::And, &a, &b);
        let gpu = cpu_kernel()
            .compute(BitmapOp::And, &a, &b)
            .expect("compute multi-container AND");

        assert_eq!(gpu.len(), n);
        assert_eq!(gpu, cpu, "multi-container output must match CPU exactly");

        // Cross-container leak check: zero-out container #2 in `a`,
        // recompute, and verify only container #2's region went to 0.
        let mut a_zeroed = a.clone();
        let z_start = 2 * BITMAP_CONTAINER_WORDS;
        let z_end = 3 * BITMAP_CONTAINER_WORDS;
        for w in &mut a_zeroed[z_start..z_end] {
            *w = 0;
        }
        let gpu2 = cpu_kernel()
            .compute(BitmapOp::And, &a_zeroed, &b)
            .expect("compute zeroed multi-container AND");
        // Containers 0, 1, 3 must equal `gpu` (unchanged).
        for c in [0, 1, 3] {
            let s = c * BITMAP_CONTAINER_WORDS;
            let e = s + BITMAP_CONTAINER_WORDS;
            assert_eq!(
                &gpu2[s..e],
                &gpu[s..e],
                "container #{c} must be unaffected by zeroing container #2"
            );
        }
        // Container 2 must now be all-zero (AND with zero).
        for &w in &gpu2[z_start..z_end] {
            assert_eq!(w, 0, "zeroed container #2 must yield all-zero AND");
        }
    }

    #[test]
    fn bitmap_and_large_scale() {
        // 256 containers = 256 × 8 KiB = 2 MiB total per operand. Large
        // enough to exercise dispatch sizing (256 × 2048 = 524 288 words
        // → 8192 workgroups of 64 threads each) but small enough to run
        // on the CPU fallback in test mode without slowdown.
        const NUM_CONTAINERS: usize = 256;
        let n = NUM_CONTAINERS * BITMAP_CONTAINER_WORDS;
        let a = random_u32_vec(n, 0xfeed_face_dead_beef);
        let b = random_u32_vec(n, 0xcafe_babe_b16b_00b5);

        let cpu = bitmap_op_cpu(BitmapOp::And, &a, &b);
        let gpu = cpu_kernel()
            .compute(BitmapOp::And, &a, &b)
            .expect("compute large-scale AND");
        assert_eq!(gpu.len(), n);
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn edge_case_empty() {
        let kernel = cpu_kernel();
        let out = kernel
            .compute(BitmapOp::And, &[], &[])
            .expect("empty input must succeed");
        assert!(out.is_empty());

        let out = kernel
            .compute(BitmapOp::Or, &[], &[])
            .expect("empty input must succeed for OR");
        assert!(out.is_empty());

        let out = kernel
            .compute(BitmapOp::Xor, &[], &[])
            .expect("empty input must succeed for XOR");
        assert!(out.is_empty());
    }

    #[test]
    fn edge_case_unaligned_length_errors() {
        // Length not a multiple of BITMAP_CONTAINER_WORDS must return
        // a typed error rather than dispatching with a partial
        // container. We test 100 (a non-multiple of 64 *and* not a
        // multiple of 2048) and 4096 - 1 (one short of two
        // containers).
        let kernel = cpu_kernel();
        let a = vec![0u32; 100];
        let b = vec![0u32; 100];
        let err = kernel.compute(BitmapOp::And, &a, &b).unwrap_err();
        assert!(
            matches!(err, GpuError::ColumnTypeMismatch { .. }),
            "unaligned length must yield ColumnTypeMismatch, got {err:?}"
        );

        let a = vec![0u32; 2 * BITMAP_CONTAINER_WORDS - 1];
        let b = vec![0u32; 2 * BITMAP_CONTAINER_WORDS - 1];
        let err = kernel.compute(BitmapOp::Or, &a, &b).unwrap_err();
        assert!(matches!(err, GpuError::ColumnTypeMismatch { .. }));
    }

    #[test]
    fn mismatched_lengths_error() {
        let kernel = cpu_kernel();
        let a = vec![0u32; BITMAP_CONTAINER_WORDS];
        let b = vec![0u32; 2 * BITMAP_CONTAINER_WORDS];
        let err = kernel.compute(BitmapOp::And, &a, &b).unwrap_err();
        assert!(matches!(err, GpuError::ColumnTypeMismatch { .. }));
    }

    #[test]
    fn cpu_oracle_self_consistent() {
        // Direct CPU-to-CPU sanity (no GPU dependency at all).
        let a: Vec<u32> = vec![0xFFFF_FFFF, 0xAAAA_AAAA, 0x5555_5555, 0x0000_0000];
        let b: Vec<u32> = vec![0xFFFF_FFFF, 0x5555_5555, 0x5555_5555, 0xFFFF_FFFF];

        assert_eq!(
            bitmap_op_cpu(BitmapOp::And, &a, &b),
            vec![0xFFFF_FFFF, 0x0000_0000, 0x5555_5555, 0x0000_0000]
        );
        assert_eq!(
            bitmap_op_cpu(BitmapOp::Or, &a, &b),
            vec![0xFFFF_FFFF, 0xFFFF_FFFF, 0x5555_5555, 0xFFFF_FFFF]
        );
        assert_eq!(
            bitmap_op_cpu(BitmapOp::Xor, &a, &b),
            vec![0x0000_0000, 0xFFFF_FFFF, 0x0000_0000, 0xFFFF_FFFF]
        );
    }

    #[test]
    fn bitmap_op_label_is_stable() {
        assert_eq!(BitmapOp::And.label(), "AND");
        assert_eq!(BitmapOp::Or.label(), "OR");
        assert_eq!(BitmapOp::Xor.label(), "XOR");
    }

    #[test]
    fn bitmap_op_all_iterates_canonical_order() {
        let labels: Vec<&'static str> =
            BitmapOp::ALL.iter().map(|o| o.label()).collect();
        assert_eq!(labels, vec!["AND", "OR", "XOR"]);
    }

    #[test]
    fn resident_path_byte_equal_to_cold_path() {
        // Hot-path API (compute_resident with pre-uploaded buffers)
        // must produce byte-equal output to the cold-path `compute()`.
        let a = random_u32_vec(2 * BITMAP_CONTAINER_WORDS, 0xa1a2_a3a4_a5a6_a7a8);
        let b = random_u32_vec(2 * BITMAP_CONTAINER_WORDS, 0xb1b2_b3b4_b5b6_b7b8);

        let kernel = cpu_kernel();
        for &op in &BitmapOp::ALL {
            let cold = kernel.compute(op, &a, &b).expect("cold");

            let a_buf = kernel.upload("a", &a).expect("upload a");
            let b_buf = kernel.upload("b", &b).expect("upload b");
            let out_buf = kernel.alloc_output("out", a.len()).expect("alloc out");
            kernel
                .compute_resident(op, &a_buf, &b_buf, &out_buf)
                .expect("resident");
            let warm = kernel.download(&out_buf).expect("download");

            assert_eq!(
                warm, cold,
                "{:?}: resident path must be byte-equal to cold path",
                op
            );
        }
    }

    #[test]
    fn resident_rejects_unaligned_buffers() {
        let kernel = cpu_kernel();
        // out length not a multiple of BITMAP_CONTAINER_WORDS.
        let a_buf = GpuBuffer::new::<u32>(
            kernel.ctx(),
            "a",
            BITMAP_CONTAINER_WORDS,
            BufferUsage::STORAGE,
        )
        .expect("a");
        let b_buf = GpuBuffer::new::<u32>(
            kernel.ctx(),
            "b",
            BITMAP_CONTAINER_WORDS,
            BufferUsage::STORAGE,
        )
        .expect("b");
        let out_buf = GpuBuffer::new::<u32>(
            kernel.ctx(),
            "out",
            BITMAP_CONTAINER_WORDS - 1,
            BufferUsage::STORAGE_READBACK,
        )
        .expect("out (intentionally unaligned)");

        let err = kernel
            .compute_resident(BitmapOp::And, &a_buf, &b_buf, &out_buf)
            .unwrap_err();
        assert!(matches!(err, GpuError::ColumnTypeMismatch { .. }));
    }

    #[test]
    fn convenience_wrappers_match() {
        let a = random_u32_vec(BITMAP_CONTAINER_WORDS, 0x1234_5678_9abc_def0);
        let b = random_u32_vec(BITMAP_CONTAINER_WORDS, 0x0fed_cba9_8765_4321);
        assert_eq!(
            bitmap_and_cpu(&a, &b),
            bitmap_op_cpu(BitmapOp::And, &a, &b)
        );
        assert_eq!(bitmap_or_cpu(&a, &b), bitmap_op_cpu(BitmapOp::Or, &a, &b));
        assert_eq!(
            bitmap_xor_cpu(&a, &b),
            bitmap_op_cpu(BitmapOp::Xor, &a, &b)
        );
    }
}

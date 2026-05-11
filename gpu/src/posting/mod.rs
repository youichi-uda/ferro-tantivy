//! GPU-accelerated posting-list operations for FerroSearch.
//!
//! Phase 2 C foundation. The Roaring Bitmap container — 8 KiB (= 2048
//! u32 words = 65 536 bits) per 16-bit doc-id space — is the natural
//! GPU target inside a Roaring posting list: word-wise bitwise AND/OR/
//! XOR over a long sequence of containers maps cleanly to a memory-
//! bound compute shader and fully saturates VRAM bandwidth on a
//! discrete GPU.
//!
//! This module currently exports [`BitmapOpKernel`] and the
//! [`BitmapOp`] enum. Future Phase 2 C waves layer:
//! - **C-2**: `ferro-compress` Bitcomp decompress + bit-op pipeline (CUDA-only path) so containers
//!   can stay compressed in VRAM between dispatches.
//! - **C-3**: Tantivy fork — Roaring container 3-form encoder/decoder (Array / Bitmap / Run) inside
//!   `vendor/tantivy-local/src/postings/`.
//! - **C-4**: Bool query (AND/OR) GPU dispatch path with the planner threshold (5 % field-doc-ratio
//!   + 10-container minimum) so light queries avoid wgpu dispatch overhead and route through the
//!   CPU galloping AVX2 path.
//!
//! ## Honest scope
//!
//! Phase 2 C-1 is the *kernel layer foundation* only. Tantivy posting
//! list integration is Phase 2 C-3, query-planner dispatching is C-4,
//! and the `ferro-compress` Bitcomp bridge is C-2. None of those are
//! provided here; this module is the building block they will call
//! into.

pub mod bitmap_op;

pub use bitmap_op::{
    bitmap_and_cpu, bitmap_op_cpu, bitmap_or_cpu, bitmap_xor_cpu, BitmapOp, BitmapOpKernel,
    BITMAP_CONTAINER_WORDS,
};

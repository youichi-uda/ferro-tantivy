// Word-wise bitmap-container OR kernel for Tantivy GPU.
//
// Phase 2 C-1 (FerroSearch GPU posting list): computes the element-wise
// bitwise OR of two flat u32 arrays of Roaring "Bitmap" containers.
// See `bitmap_and.wgsl` for the layout rationale and bandwidth analysis
// — OR shares the same memory pattern (1 store, 2 loads per output
// word), and `|` is a single-cycle ALU op so the kernel is bandwidth-
// bound at exactly the same ceiling as AND. Empirically on consumer
// NVIDIA hardware OR runs identically to AND; if a regression appears
// on a different GPU the fault is on the bandwidth controller, not the
// shader.
//
// Layout:
//   @group(0) @binding(0) — a:    array<u32>  (length = num_words)
//   @group(0) @binding(1) — b:    array<u32>  (length = num_words)
//   @group(0) @binding(2) — out:  array<u32>  (length = num_words)
//   @group(0) @binding(3) — uniform: Params { num_words }

// See `bitmap_and.wgsl` for the `word_offset` chunking rationale —
// shared across all three Phase 2 C-1 kernels so a single host-side
// chunking helper can dispatch any of them.
struct Params {
    num_words: u32,
    word_offset: u32,
}

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn bitmap_or(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + params.word_offset;
    if i >= params.num_words {
        return;
    }
    out[i] = a[i] | b[i];
}

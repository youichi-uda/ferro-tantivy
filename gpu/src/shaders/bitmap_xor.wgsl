// Word-wise bitmap-container XOR kernel for Tantivy GPU.
//
// Phase 2 C-1 (FerroSearch GPU posting list): computes the element-wise
// bitwise XOR of two flat u32 arrays of Roaring "Bitmap" containers.
// XOR is the symmetric difference operation — useful for change-detect
// query plans ("docs that match exactly one of these two terms"). See
// `bitmap_and.wgsl` for the layout rationale and bandwidth analysis;
// `^` shares the 1-cycle ALU cost and the kernel is bandwidth-bound
// at the same ceiling as AND/OR.
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
fn bitmap_xor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + params.word_offset;
    if i >= params.num_words {
        return;
    }
    out[i] = a[i] ^ b[i];
}

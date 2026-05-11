// Word-wise bitmap-container AND kernel for Tantivy GPU.
//
// Phase 2 C-1 (FerroSearch GPU posting list): computes the element-wise
// bitwise AND of two flat u32 arrays representing a sequence of Roaring
// "Bitmap" containers. Each container is exactly 8 KiB (= 2048 u32 words
// = 65 536 bits) covering one 16-bit doc-id space. This kernel treats
// the two inputs as flat arrays of `num_words = num_containers * 2048`
// words and runs `out[i] = a[i] & b[i]` over the whole range — there is
// no cross-container coupling, so a single dispatch processes any number
// of containers in lock-step without needing a 2-D index space.
//
// The kernel is **memory-bound**: each thread does one load from `a`,
// one load from `b`, one AND, and one store. On RTX 4070 Ti SUPER (800
// GB/s peak) the steady-state cost is exactly the 16 B/word that has to
// move across VRAM (8 B read + 4 B written + 4 B read of the other
// operand → 12 B per output word, i.e. ~67 G output words per second at
// peak). vec4<u32> SIMD widening is **not** done here: scalar already
// fully saturates VRAM bandwidth and `&` is a 1-cycle ALU op, so vec4
// would only thicken the IR without lowering the bandwidth roof. If
// post-bench shows we are ALU-bound (e.g. on a future tail-fused
// pipeline), a `_vec4` variant can be added without changing the binding
// layout.
//
// Layout:
//   @group(0) @binding(0) — a:    array<u32>  (length = num_words)
//   @group(0) @binding(1) — b:    array<u32>  (length = num_words)
//   @group(0) @binding(2) — out:  array<u32>  (length = num_words)
//   @group(0) @binding(3) — uniform: Params { num_words }
//
// Workgroup size 64 matches NVIDIA warp×2 / AMD wave64. The bounds
// check `i >= num_words` lets the host dispatch `ceil(num_words / 64)`
// workgroups even when `num_words` is not a multiple of 64.

// `word_offset` shifts the per-thread index into the global flat
// arrays so that callers can split a single conceptual dispatch into
// multiple chunks when `ceil(num_words / 64)` exceeds the device's
// `max_compute_workgroups_per_dimension` (downlevel default 65 535).
// Each chunk runs `word_offset .. word_offset + chunk_words` of the
// logical range.
struct Params {
    num_words: u32,
    word_offset: u32,
}

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn bitmap_and(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + params.word_offset;
    if i >= params.num_words {
        return;
    }
    out[i] = a[i] & b[i];
}

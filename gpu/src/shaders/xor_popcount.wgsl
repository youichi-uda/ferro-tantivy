// Binary (1-bit) Hamming-distance kernel for Tantivy GPU.
//
// Computes Hamming distances between a bit-packed query vector and a
// batch of bit-packed corpus vectors using XOR + population count.
// This is the GPU offload path for BBQ (Binary Block Quantization)
// nearest-neighbour search: each f32 input vector is quantised to a
// `dim_bits`-bit code, packed into `dim_u32 = ceil(dim_bits / 32)`
// `u32` words, and Hamming distance approximates the angular distance.
//
// Layout:
//   @group(0) @binding(0) — query: array<u32>            (dim_u32 words)
//   @group(0) @binding(1) — corpus: array<u32>           (num_vecs * dim_u32 words)
//   @group(0) @binding(2) — dists: array<u32>            (num_vecs)
//   @group(0) @binding(3) — uniform: Params {num_vecs, dim_u32}
//
// Each thread (global_invocation_id.x) computes the Hamming distance
// for one corpus vector. Workgroup size 64 is chosen to match warp/wave
// granularity on common GPUs (NVIDIA warp=32, AMD wave=64) while keeping
// occupancy reasonable for short vectors (dim_u32 small).
//
// vec4<u32> SIMD widening (Phase 2 A integration): the inner loop is
// unrolled into 4-wide chunks via `vec4<u32>` ops on a vector of 4
// adjacent u32 words. `countOneBits()` is component-wise on vectors,
// and the resulting vec4 of popcounts is reduced by `dot()` against
// `vec4(1u, 1u, 1u, 1u)`. Modern shader compilers fuse this into a tight
// 4-popcount-per-step loop on hardware (PTX `popc.b32.b32` on NVIDIA,
// `S_BCNT1_*` on AMD). For typical embedding sizes (`dim_bits = 768` →
// `dim_u32 = 24`, divisible by 4) the entire inner loop runs in vec4
// mode; sub-multiple-of-4 tails (e.g. `dim_bits = 32` → `dim_u32 = 1`)
// fall back to a scalar pass over the leftover 1–3 words.

struct Params {
    num_vecs: u32,
    dim_u32: u32,
}

@group(0) @binding(0) var<storage, read> query: array<u32>;
@group(0) @binding(1) var<storage, read> corpus: array<u32>;
@group(0) @binding(2) var<storage, read_write> dists: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn xor_popcount(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let vec_id = global_id.x;
    if vec_id >= params.num_vecs {
        return;
    }

    let dim = params.dim_u32;
    let base = vec_id * dim;
    // Number of complete vec4<u32> chunks; tail = dim - chunks*4.
    let chunks = dim / 4u;
    let tail_start = chunks * 4u;

    var sum: u32 = 0u;

    // ── Vectorised body (4 words per iteration) ──
    for (var c: u32 = 0u; c < chunks; c = c + 1u) {
        let i = c * 4u;
        let qv = vec4<u32>(
            query[i],
            query[i + 1u],
            query[i + 2u],
            query[i + 3u],
        );
        let cv = vec4<u32>(
            corpus[base + i],
            corpus[base + i + 1u],
            corpus[base + i + 2u],
            corpus[base + i + 3u],
        );
        let pop = countOneBits(qv ^ cv);
        sum = sum + pop.x + pop.y + pop.z + pop.w;
    }

    // ── Scalar tail (0–3 leftover words) ──
    for (var i: u32 = tail_start; i < dim; i = i + 1u) {
        let q = query[i];
        let c = corpus[base + i];
        sum = sum + countOneBits(q ^ c);
    }

    dists[vec_id] = sum;
}

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
// WGSL 1.0 builtin `countOneBits()` is used — modern GPUs lower this to
// a hardware popcount instruction (PTX `popc`, SPIR-V `OpBitCount`).
// A vec4<u32> SIMD widening is possible but not required for correctness;
// it remains a future optimisation (Phase 2 B+).

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

    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let q = query[i];
        let c = corpus[base + i];
        sum = sum + countOneBits(q ^ c);
    }

    dists[vec_id] = sum;
}

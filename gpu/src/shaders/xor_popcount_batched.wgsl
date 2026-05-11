// Batched binary (1-bit) Hamming-distance kernel for Tantivy GPU.
//
// Computes a Q × N distance matrix in one dispatch: for each of `Q`
// query vectors and each of `N` corpus vectors, writes the Hamming
// distance to `dists[q * N + n]`.
//
// This is the high-throughput path for k-NN over BBQ-style binary
// codes. The single-query kernel (`xor_popcount.wgsl`) reads the entire
// corpus from VRAM once per query — for `Q` independent queries that's
// `Q × |corpus|` PCIe / VRAM bandwidth. Batching collapses the corpus
// read to once per workgroup tile by staging tiles of queries and
// corpus rows into workgroup-shared memory, then reusing each row for
// every query in the tile. For the typical workload of `dim_bits = 768`
// (`dim_u32 = 24`) and `Q = 64` this is a roughly Q-fold reduction in
// VRAM traffic and brings the kernel hard up against the 800 GB/s
// peak bandwidth of an RTX 4070 Ti SUPER.
//
// Tile dimensions (compile-time constants):
//   TILE_Q          = 8   queries per workgroup
//   TILE_N          = 8   corpus vectors per workgroup
//   MAX_DIM_U32     = 64  upper bound on `dim_u32` supported by this shader
//                          (covers `dim_bits` up to 2048 — wider than any
//                           production embedding model in 2025).
//
// The 2D dispatch is sized as
//     workgroups_x = ceil(N / TILE_N)
//     workgroups_y = ceil(Q / TILE_Q)
// with workgroup_size = (TILE_N, TILE_Q, 1) so each thread owns a single
// (query, corpus) cell.
//
// Layout:
//   @group(0) @binding(0) — queries: array<u32>          (Q * dim_u32 words)
//   @group(0) @binding(1) — corpus:  array<u32>          (N * dim_u32 words)
//   @group(0) @binding(2) — dists:   array<u32>          (Q * N)
//   @group(0) @binding(3) — uniform: Params {num_queries, num_vecs, dim_u32}

const TILE_Q: u32 = 8u;
const TILE_N: u32 = 8u;
const MAX_DIM_U32: u32 = 64u;

struct ParamsB {
    num_queries: u32,
    num_vecs: u32,
    dim_u32: u32,
    // Workgroup-x offset into the N dimension. Lets a single kernel
    // launch be split into multiple dispatches when
    // ceil(N / TILE_N) exceeds `max_compute_workgroups_per_dimension`
    // (downlevel-default 65535). Each chunk receives the same buffers
    // and runs `wg_offset_x..wg_offset_x + chunk_workgroups_x` of the
    // logical workgroup.x range.
    wg_offset_x: u32,
}

@group(0) @binding(0) var<storage, read> queries: array<u32>;
@group(0) @binding(1) var<storage, read> corpus: array<u32>;
@group(0) @binding(2) var<storage, read_write> dists: array<u32>;
@group(0) @binding(3) var<uniform> params: ParamsB;

// Workgroup-local query / corpus tiles. Indexed as
//     tile_q[q_local * MAX_DIM_U32 + word_idx]
//     tile_c[n_local * MAX_DIM_U32 + word_idx]
// where word_idx in [0, dim_u32). Slots beyond `dim_u32` are unused.
var<workgroup> tile_q: array<u32, 512>; // TILE_Q * MAX_DIM_U32 = 8 * 64
var<workgroup> tile_c: array<u32, 512>; // TILE_N * MAX_DIM_U32 = 8 * 64

@compute @workgroup_size(8, 8, 1)
fn xor_popcount_batched(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let dim = params.dim_u32;
    let n_local = local_id.x;     // 0..TILE_N
    let q_local = local_id.y;     // 0..TILE_Q
    // Logical workgroup index along N is (raw wg_id.x + wg_offset_x).
    let wg_x_logical = wg_id.x + params.wg_offset_x;
    let q_base_global = wg_id.y * TILE_Q;
    let n_base_global = wg_x_logical * TILE_N;
    let n = n_base_global + n_local; // logical corpus index
    let q = q_base_global + q_local; // logical query index

    // ── Cooperative tile load ──
    // Each of the 64 threads loads `dim/64 + 1` words. We sweep over
    // every word of every row (TILE_Q rows × dim words for the query
    // tile, TILE_N rows × dim words for the corpus tile). Threads that
    // would index past the end of either tile or off the end of a real
    // row simply skip.
    let thread_id_1d = q_local * TILE_N + n_local; // 0..63
    let total_threads = TILE_Q * TILE_N;            // 64

    // Load query tile.
    let q_tile_words = TILE_Q * dim;
    var w: u32 = thread_id_1d;
    loop {
        if w >= q_tile_words { break; }
        let row = w / dim;
        let col = w % dim;
        let q_idx = q_base_global + row;
        if q_idx < params.num_queries {
            tile_q[row * MAX_DIM_U32 + col] = queries[q_idx * dim + col];
        }
        w = w + total_threads;
    }

    // Load corpus tile.
    let c_tile_words = TILE_N * dim;
    var w2: u32 = thread_id_1d;
    loop {
        if w2 >= c_tile_words { break; }
        let row = w2 / dim;
        let col = w2 % dim;
        let n_idx = n_base_global + row;
        if n_idx < params.num_vecs {
            tile_c[row * MAX_DIM_U32 + col] = corpus[n_idx * dim + col];
        }
        w2 = w2 + total_threads;
    }

    workgroupBarrier();

    // Out-of-range threads: still synced, just don't write.
    if n >= params.num_vecs || q >= params.num_queries {
        return;
    }

    // ── Vectorised distance computation from shared memory ──
    let chunks = dim / 4u;
    let tail_start = chunks * 4u;
    let q_off = q_local * MAX_DIM_U32;
    let c_off = n_local * MAX_DIM_U32;

    var sum: u32 = 0u;
    for (var c: u32 = 0u; c < chunks; c = c + 1u) {
        let i = c * 4u;
        let qv = vec4<u32>(
            tile_q[q_off + i],
            tile_q[q_off + i + 1u],
            tile_q[q_off + i + 2u],
            tile_q[q_off + i + 3u],
        );
        let cv = vec4<u32>(
            tile_c[c_off + i],
            tile_c[c_off + i + 1u],
            tile_c[c_off + i + 2u],
            tile_c[c_off + i + 3u],
        );
        let pop = countOneBits(qv ^ cv);
        sum = sum + pop.x + pop.y + pop.z + pop.w;
    }
    for (var i: u32 = tail_start; i < dim; i = i + 1u) {
        let qw = tile_q[q_off + i];
        let cw = tile_c[c_off + i];
        sum = sum + countOneBits(qw ^ cw);
    }

    dists[q * params.num_vecs + n] = sum;
}

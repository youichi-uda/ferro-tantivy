// On-GPU top-K selection over a Q × N distance matrix.
//
// For each of `Q` queries, scans the row of `N` (distance, id) pairs and
// returns the K smallest distances together with their corpus ids,
// sorted ascending by distance. This is the second half of a batched
// k-NN pipeline — its main reason for existing is to avoid copying the
// full Q × N matrix back over PCIe when only Q × K is needed at the
// caller. For Q=64, N=1_000_000, K=100 this is a ~10000× reduction in
// readback bytes and is decisive for sub-millisecond k-NN over BBQ
// codes.
//
// ## Algorithm — batched bitonic merge top-K
//
// Of the two well-known on-GPU top-K options:
//
//   (a) Bitonic merge — maintain a sorted shared-memory buffer of
//       `K_padded` pairs (`UINT_MAX` sentinels initially), repeatedly
//       merge in a fresh batch of `K_padded` candidates via a 2K-wide
//       bitonic sort, then truncate back to `K_padded`. Cost
//       O(N/K · K log² K) per query, simple, in-shared-memory.
//
//   (b) Radix select / partial sort — partition by k-th element
//       pivots. Asymptotically faster (O(N + K log K)) but needs
//       multi-pass scans, atomic counters, careful tie handling.
//
// We pick (a). Reasoning: K ≤ 1024 is the production envelope for
// k-NN reranking workloads (recall ~95 % at K=100, full BBQ retrieval
// at K=1024). At those sizes, K log² K ≈ 1024 × 100 = 10⁵ cycles per
// merge step is dwarfed by the PCIe + scan cost; the simpler control
// flow has fewer correctness pitfalls; and there is no atomic
// contention to schedule. Radix select becomes attractive at K ≥ 4096
// or N ≥ 100 M, neither of which is the present design point.
//
// ## Layout
//
//   @group(0) @binding(0) — dists       : array<u32>   (Q * N input)
//   @group(0) @binding(1) — top_k_dists : array<u32>   (Q * K output, sorted asc)
//   @group(0) @binding(2) — top_k_ids   : array<u32>   (Q * K output, paired)
//   @group(0) @binding(3) — uniform Params { num_queries, num_vecs, k, k_padded }
//
// ## Dispatch
//
// 1 workgroup per query, workgroup_size = 256. Shared memory holds
// `2 * MAX_K_PADDED = 2048` (dist, id) pairs = 16 KB, well under the
// 32 KB shared-memory limit on every shader-model-6 GPU we target.

const MAX_K_PADDED: u32 = 1024u;
const WG_SIZE: u32 = 256u;
const SENTINEL: u32 = 0xFFFFFFFFu;

struct ParamsTopK {
    num_queries: u32,
    num_vecs: u32,
    k: u32,
    k_padded: u32,
}

@group(0) @binding(0) var<storage, read> dists: array<u32>;
@group(0) @binding(1) var<storage, read_write> top_k_dists: array<u32>;
@group(0) @binding(2) var<storage, read_write> top_k_ids: array<u32>;
@group(0) @binding(3) var<uniform> params: ParamsTopK;

// Two halves of the working buffer. The first MAX_K_PADDED slots hold
// the running top-K_padded; the second half is the merge staging area.
var<workgroup> buf_dist: array<u32, 2048>;  // 2 * MAX_K_PADDED
var<workgroup> buf_id:   array<u32, 2048>;

// Conditional swap: order pair (a, b) in buffer ascending by dist if
// `dir == 1u` (lo→hi), descending otherwise. Tie-break by id (lower id
// wins) keeps the algorithm deterministic when many corpus vectors
// share the same Hamming distance.
fn cmpswap(a: u32, b: u32, dir: u32) {
    let da = buf_dist[a];
    let db = buf_dist[b];
    let ia = buf_id[a];
    let ib = buf_id[b];
    // ascending iff (da > db) OR (da == db AND ia > ib) when dir == 1
    let asc = (da > db) || (da == db && ia > ib);
    let should_swap = (asc && dir == 1u) || (!asc && dir == 0u);
    if should_swap {
        buf_dist[a] = db;
        buf_dist[b] = da;
        buf_id[a] = ib;
        buf_id[b] = ia;
    }
}

// Sort `len` entries of buf_*[0..len) into ascending order via bitonic
// sort. `len` must be a power of two, ≤ 2 * MAX_K_PADDED.
fn bitonic_sort(len: u32, tid: u32) {
    var size: u32 = 2u;
    loop {
        if size > len { break; }

        var stride: u32 = size / 2u;
        loop {
            if stride == 0u { break; }

            // Each thread handles every (len / WG_SIZE) compare-swap.
            var idx: u32 = tid;
            loop {
                if idx >= len { break; }
                let pair = idx ^ stride;
                if pair > idx {
                    let dir_block = (idx & size) == 0u;
                    let dir: u32 = select(0u, 1u, dir_block);
                    cmpswap(idx, pair, dir);
                }
                idx = idx + WG_SIZE;
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        size = size * 2u;
    }
}

@compute @workgroup_size(256, 1, 1)
fn top_k_select(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let q = wg_id.x;
    let tid = local_id.x;
    if q >= params.num_queries {
        return;
    }

    let n = params.num_vecs;
    let k = params.k;
    let kp = params.k_padded;            // power of two, ≤ MAX_K_PADDED
    let total_len = kp * 2u;             // merge buffer length

    let row_base = q * n;

    // Initialize top half with sentinels; bottom half also for the
    // first batch. The bottom half is overwritten by every batch load.
    var i: u32 = tid;
    loop {
        if i >= total_len { break; }
        buf_dist[i] = SENTINEL;
        buf_id[i] = 0u;
        i = i + WG_SIZE;
    }
    workgroupBarrier();

    // ── Streaming merge: process N inputs in batches of `kp`. After
    // each batch, the lower half of `buf` holds the running top kp
    // candidates sorted ascending by dist. ──
    var processed: u32 = 0u;
    loop {
        if processed >= n { break; }

        // Load up to `kp` new candidates into the upper half of buf
        // (slots [kp .. 2*kp)). Padding slots beyond N keep sentinels.
        var j: u32 = tid;
        loop {
            if j >= kp { break; }
            let src = processed + j;
            if src < n {
                buf_dist[kp + j] = dists[row_base + src];
                buf_id[kp + j] = src;
            } else {
                buf_dist[kp + j] = SENTINEL;
                buf_id[kp + j] = 0u;
            }
            j = j + WG_SIZE;
        }
        workgroupBarrier();

        // Sort all 2*kp entries ascending. After sort, slots
        // [0..kp) contain the running top kp.
        bitonic_sort(total_len, tid);

        processed = processed + kp;
    }

    // ── Spill the first `k` entries to the output. ──
    var w: u32 = tid;
    loop {
        if w >= k { break; }
        top_k_dists[q * k + w] = buf_dist[w];
        top_k_ids[q * k + w] = buf_id[w];
        w = w + WG_SIZE;
    }
}

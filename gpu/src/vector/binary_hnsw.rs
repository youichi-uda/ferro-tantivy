//! HNSW (Hierarchical Navigable Small World) graph over **bit-packed
//! binary codes** with GPU-accelerated batched re-rank — the Phase 2
//! A-final adoption form for BBQ-style nearest-neighbour search.
//!
//! ## Why a separate index from [`HnswIndex`]
//!
//! [`HnswIndex`] (in `hnsw.rs`) stores `Vec<Vec<f32>>` and dispatches
//! float L2/cosine/dot kernels. Bit-packed codes have a fundamentally
//! different storage layout (`u32`-words, `dim_u32` per vector, flat
//! row-major) and a different distance shape (Hamming returns `u32`
//! not `f32`). Threading a `DistanceMetric` enum that mixes both
//! through `HnswIndex` would either:
//!  - waste memory storing float vectors alongside binary codes, or
//!  - require trait-object dispatch per edge (cache hostile, no
//!    win for the small per-edge work).
//!
//! Instead this module provides [`BinaryHnswIndex`], a **structural
//! mirror** of `HnswIndex` that swaps the storage and distance kernel
//! while reusing the exact same level/neighbour algorithm. The float
//! [`HnswIndex`] is therefore left **byte-identical** — no risk of
//! perturbing the float test suite, the persistence format, or any
//! downstream caller — while the binary path is built up cleanly with
//! its own kernel and its own test surface.
//!
//! ## Phase 2 A-final adoption form: graph traversal CPU + final rerank GPU batched
//!
//! Naïvely one might think *every* per-edge distance call along an
//! HNSW path should be dispatched to the GPU. That doesn't work: each
//! HNSW hop typically evaluates only `M = 16-32` neighbours, and a
//! single GPU dispatch costs ~1-3 ms of constant overhead (PCIe
//! upload, kernel launch, readback). The dispatch overhead **dwarfs**
//! the actual compute for a 16-neighbour batch — CPU `popcount` clears
//! the same work in single-digit microseconds.
//!
//! The **production win** for binary HNSW on GPU is therefore **not**
//! per-edge GPU dispatch. It is:
//!
//! ```text
//!   1. Run the HNSW graph traversal on CPU with bit-packed popcount
//!      (cheap, branchy, sequential — exactly what CPUs do well).
//!      This produces a candidate set of `ef_search` (typically
//!      256-1000) approximate nearest neighbours.
//!
//!   2. Re-rank that candidate set on the GPU using one batched
//!      dispatch (`BinaryDistanceKernel::knn_search`). At ef=256 the
//!      GPU clears the rerank in ~0.3-1.0 ms — well into the win
//!      regime — and the on-device top-K reduction collapses readback
//!      to `K` pairs.
//! ```
//!
//! For multi-query workloads (`Q > 1`, the realistic case for any
//! search-engine-grade system serving concurrent traffic), the
//! batched rerank is the same single dispatch with `Q × ef_search`
//! distance evaluations — the corpus tile reuse gives a Q-factor
//! shared-memory benefit on top of the per-query GPU win.
//!
//! ## Calling contract
//!
//! - `dim_bits` is fixed at construction; every inserted code must be
//!   exactly `dim_u32 = ceil(dim_bits / 32)` u32 words, with trailing
//!   padding bits zeroed by the caller.
//! - [`BinaryHnswIndex::insert`] takes a `Vec<u32>` of length `dim_u32`.
//! - [`BinaryHnswIndex::search`] performs CPU graph traversal with
//!   CPU popcount distance and returns top-K. **No GPU is involved.**
//! - [`BinaryHnswIndex::search_gpu_rerank`] does CPU graph traversal
//!   to produce an `ef_search`-sized candidate set, then dispatches
//!   one batched GPU rerank to produce the final top-K.
//! - [`BinaryHnswIndex::search_batch_gpu_rerank`] is the multi-query
//!   form: every query's candidate set is reranked in one fused GPU
//!   dispatch. This is the production entry point for serving
//!   workloads.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::device::GpuContext;
use crate::error::{GpuError, GpuResult};
use crate::kernel::GpuKernel;
use crate::vector::binary_distance::{
    dim_u32_for, hamming_distance_cpu, BinaryDistanceKernel, BinaryDistanceMetric,
};

/// Maximum number of connections per node at level 0.
const M: usize = 16;
/// Maximum number of connections per node at levels > 0.
const M_MAX0: usize = 32;
/// Size multiplier for construction candidate list.
const EF_CONSTRUCTION: usize = 200;

/// HNSW graph index over bit-packed binary codes.
///
/// Storage layout:
/// - `corpus_bits`: flat `num_vecs * dim_u32` u32 array (row-major).
/// - `graph[level][node_id] = Vec<neighbor_id>`.
/// - `levels[node_id] = level` (the topmost layer at which this node
///   participates).
///
/// The graph is byte-shape-identical to [`crate::vector::hnsw::HnswIndex`]
/// but with `vectors: Vec<Vec<f32>>` swapped for `corpus_bits: Vec<u32>`.
pub struct BinaryHnswIndex {
    /// Flat row-major bit-packed corpus: `num_vecs * dim_u32` u32 words.
    corpus_bits: Vec<u32>,
    /// Adjacency lists per level: `graph[level][node_id] = Vec<neighbor_id>`.
    graph: Vec<Vec<Vec<u32>>>,
    /// Topmost level at which each node participates.
    levels: Vec<usize>,
    /// Entry point (node with the highest level).
    entry_point: Option<u32>,
    /// Maximum level currently present in the graph.
    max_level: usize,
    /// Vector dimension in bits.
    dim_bits: usize,
    /// Words-per-vector — cached from `dim_bits`.
    dim_u32: usize,
    /// Distance metric (currently always Hamming; held for forward
    /// compatibility with future variants like Jaccard).
    metric: BinaryDistanceMetric,
    /// Cached GPU rerank kernel — `None` until [`BinaryHnswIndex::with_gpu`]
    /// or [`BinaryHnswIndex::attach_gpu`] is called.
    gpu_kernel: Option<BinaryDistanceKernel>,
}

/// A scored candidate during graph traversal: `(distance, node_id)`.
/// Distances are `u32` (Hamming) — exact, no float ties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    distance: u32,
    id: u32,
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: ascending distance. Tie-break: ascending id (stable,
        // matches the GPU top-K shader's deterministic tie-break).
        self.distance
            .cmp(&other.distance)
            .then(self.id.cmp(&other.id))
    }
}

impl BinaryHnswIndex {
    /// Create a new empty binary HNSW index. CPU-only; GPU rerank
    /// becomes available once [`Self::attach_gpu`] is called.
    pub fn new(dim_bits: usize, metric: BinaryDistanceMetric) -> Self {
        Self {
            corpus_bits: Vec::new(),
            graph: Vec::new(),
            levels: Vec::new(),
            entry_point: None,
            max_level: 0,
            dim_bits,
            dim_u32: dim_u32_for(dim_bits),
            metric,
            gpu_kernel: None,
        }
    }

    /// Create a new binary HNSW index with the GPU rerank kernel
    /// pre-compiled. If GPU compilation fails the error is bubbled up
    /// — callers that want graceful CPU-only fallback should call
    /// [`Self::new`] instead and (optionally) [`Self::attach_gpu`].
    pub fn with_gpu(
        dim_bits: usize,
        metric: BinaryDistanceMetric,
        ctx: &GpuContext,
    ) -> GpuResult<Self> {
        let kernel = BinaryDistanceKernel::compile(ctx)?;
        Ok(Self {
            corpus_bits: Vec::new(),
            graph: Vec::new(),
            levels: Vec::new(),
            entry_point: None,
            max_level: 0,
            dim_bits,
            dim_u32: dim_u32_for(dim_bits),
            metric,
            gpu_kernel: Some(kernel),
        })
    }

    /// Lazily attach a GPU kernel to an already-built index.
    pub fn attach_gpu(&mut self, ctx: &GpuContext) -> GpuResult<()> {
        let kernel = BinaryDistanceKernel::compile(ctx)?;
        self.gpu_kernel = Some(kernel);
        Ok(())
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        // `dim_u32 = 0` only if the user constructed the index with
        // `dim_bits = 0` — degenerate but legal; treat as empty.
        self.corpus_bits.len().checked_div(self.dim_u32).unwrap_or(0)
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.corpus_bits.is_empty()
    }

    /// Vector dimension in bits (e.g. 768 for BBQ-768).
    pub fn dim_bits(&self) -> usize {
        self.dim_bits
    }

    /// `dim_u32 = ceil(dim_bits / 32)` — the number of u32 words per
    /// stored vector.
    pub fn dim_u32(&self) -> usize {
        self.dim_u32
    }

    /// Distance metric (currently always Hamming).
    pub fn metric(&self) -> BinaryDistanceMetric {
        self.metric
    }

    /// Maximum level currently present in the graph.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Entry point node ID (the node with the highest level), or
    /// `None` if the index is empty.
    pub fn entry_point_id(&self) -> Option<u32> {
        self.entry_point
    }

    /// Whether the GPU rerank kernel is currently attached.
    pub fn has_gpu(&self) -> bool {
        self.gpu_kernel.is_some()
    }

    /// Borrow the flat corpus bit-buffer (row-major,
    /// `len() * dim_u32` u32 words).
    pub fn corpus_bits(&self) -> &[u32] {
        &self.corpus_bits
    }

    /// Insert a single bit-packed code vector.
    ///
    /// Returns the assigned node ID.
    ///
    /// # Errors
    /// Returns [`GpuError::ColumnTypeMismatch`] if `code.len() != dim_u32`.
    pub fn insert(&mut self, code: Vec<u32>) -> GpuResult<u32> {
        if code.len() != self.dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("dim_u32={}", self.dim_u32),
                actual: format!("code.len()={}", code.len()),
            });
        }

        let id = self.len() as u32;
        let level = self.random_level();

        // Ensure graph has enough levels.
        while self.graph.len() <= level {
            self.graph.push(Vec::new());
        }
        // Ensure each existing level has a slot for this node.
        for lvl in self.graph.iter_mut() {
            while lvl.len() <= id as usize {
                lvl.push(Vec::new());
            }
        }
        self.levels.push(level);
        self.corpus_bits.extend_from_slice(&code);

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_level = level;
            return Ok(id);
        }

        let entry = self.entry_point.expect("entry_point confirmed Some");

        // Phase 1: greedy descent from top down to level+1.
        let mut current = entry;
        for lev in (level + 1..=self.max_level).rev() {
            current = self.greedy_closest(lev, current, &code);
        }

        // Phase 2: insert at every level from `level` down to 0.
        let mut ep = vec![Candidate {
            distance: self.distance_to_node(current as usize, &code),
            id: current,
        }];

        for lev in (0..=level.min(self.max_level)).rev() {
            let max_conn = if lev == 0 { M_MAX0 } else { M };
            let candidates = self.search_layer(&code, &ep, EF_CONSTRUCTION, lev);
            let neighbors = self.select_neighbors(&candidates, max_conn);

            // Connect new node → neighbors.
            self.graph[lev][id as usize] = neighbors.iter().map(|c| c.id).collect();

            // Connect neighbors → new node, prune if over-connected.
            for &neighbor in &neighbors {
                let nid = neighbor.id as usize;
                self.graph[lev][nid].push(id);
                if self.graph[lev][nid].len() > max_conn {
                    let nv_base = nid * self.dim_u32;
                    let nv = self.corpus_bits[nv_base..nv_base + self.dim_u32].to_vec();
                    let mut scored: Vec<Candidate> = self.graph[lev][nid]
                        .iter()
                        .map(|&nbr| Candidate {
                            distance: self.distance_node_to_code(nbr as usize, &nv),
                            id: nbr,
                        })
                        .collect();
                    scored.sort();
                    self.graph[lev][nid] = scored[..max_conn].iter().map(|c| c.id).collect();
                }
            }

            ep = candidates;
        }

        if level > self.max_level {
            self.entry_point = Some(id);
            self.max_level = level;
        }

        Ok(id)
    }

    /// Pure-CPU graph search returning top-K `(distance, doc_id)`
    /// pairs sorted ascending by distance.
    pub fn search(&self, query: &[u32], k: usize, ef_search: usize) -> GpuResult<Vec<(u32, u32)>> {
        let candidates = self.collect_candidate_set(query, ef_search.max(k))?;
        Ok(candidates
            .into_iter()
            .take(k)
            .map(|c| (c.distance, c.id))
            .collect())
    }

    /// Phase 2 A-final production search path: CPU graph traversal
    /// produces an `ef_search`-sized candidate set, GPU batched rerank
    /// (one fused dispatch) computes the exact distances and the
    /// final top-K.
    ///
    /// For a single query the GPU dispatch overhead may dominate
    /// unless `ef_search ≥ ~256`; for multi-query workloads use
    /// [`Self::search_batch_gpu_rerank`].
    ///
    /// Returns top-K `(distance, doc_id)` pairs sorted ascending.
    /// Falls back to CPU `search` if no GPU kernel is attached.
    pub fn search_gpu_rerank(
        &self,
        query: &[u32],
        k: usize,
        ef_search: usize,
    ) -> GpuResult<Vec<(u32, u32)>> {
        if self.gpu_kernel.is_none() {
            return self.search(query, k, ef_search);
        }
        if query.len() != self.dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("query.len()={}", self.dim_u32),
                actual: format!("query.len()={}", query.len()),
            });
        }
        if self.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // Stage 1: CPU graph traversal → ef_search candidate IDs.
        let cand = self.collect_candidate_set(query, ef_search.max(k))?;
        if cand.is_empty() {
            return Ok(Vec::new());
        }

        // Stage 2: GPU batched rerank over the candidate set.
        // The candidate set is small (typically 256-1024 vectors),
        // so we materialise its sub-corpus into a contiguous buffer
        // and dispatch a single Q=1 × N=ef_search GPU kernel.
        let mut sub_corpus: Vec<u32> = Vec::with_capacity(cand.len() * self.dim_u32);
        for c in &cand {
            let base = c.id as usize * self.dim_u32;
            sub_corpus.extend_from_slice(&self.corpus_bits[base..base + self.dim_u32]);
        }

        let kernel = self.gpu_kernel.as_ref().expect("gpu_kernel confirmed Some");
        let k_eff = k.min(cand.len());
        let topk = kernel.knn_search(query, &sub_corpus, 1, cand.len(), self.dim_bits, k_eff)?;

        // Map sub-corpus indices back to original node IDs.
        let row = topk
            .into_iter()
            .next()
            .expect("knn_search yields one row per query");
        Ok(row
            .into_iter()
            .map(|(dist, sub_id)| (dist, cand[sub_id as usize].id))
            .collect())
    }

    /// Multi-query GPU rerank — the production multi-tenant entry point.
    ///
    /// For each query, performs CPU graph traversal to its `ef_search`
    /// candidate set, then **unions** every query's candidate set
    /// into a single shared sub-corpus and runs **one batched GPU
    /// dispatch** of shape `Q × |union|`. This produces a full
    /// `Q × |union|` distance matrix in one kernel; per-query top-K
    /// is then a CPU-side scan **filtered** to that query's own
    /// candidates (the union is a superset, so other queries'
    /// candidates contribute distances we simply ignore).
    ///
    /// Why union-and-filter rather than per-query GPU dispatch:
    /// - The batched GPU kernel requires a *single shared* corpus
    ///   across queries (it tiles corpus rows into shared memory
    ///   reused by every query in the workgroup). A per-query
    ///   dispatch would lose that shared-memory tile reuse.
    /// - The post-dispatch CPU filter is cheap: `Q × |union|` is
    ///   bounded by `Q × Q × ef_search`, typically a few thousand
    ///   `u32` lookups per query — single-digit microseconds.
    /// - Recall is unchanged: the filter projects each query's row
    ///   back onto its own candidate set, identical to what a
    ///   per-query rerank would have produced.
    ///
    /// Returns one `Vec<(distance, doc_id)>` per query, each sorted
    /// ascending.
    ///
    /// Falls back to per-query CPU search if no GPU kernel is
    /// attached.
    pub fn search_batch_gpu_rerank(
        &self,
        queries: &[Vec<u32>],
        k: usize,
        ef_search: usize,
    ) -> GpuResult<Vec<Vec<(u32, u32)>>> {
        if self.gpu_kernel.is_none() {
            return queries.iter().map(|q| self.search(q, k, ef_search)).collect();
        }
        if queries.is_empty() || k == 0 || self.is_empty() {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        // Validate all query lengths up front.
        for (qi, q) in queries.iter().enumerate() {
            if q.len() != self.dim_u32 {
                return Err(GpuError::ColumnTypeMismatch {
                    expected: format!("queries[{qi}].len()={}", self.dim_u32),
                    actual: format!("queries[{qi}].len()={}", q.len()),
                });
            }
        }

        // Build candidate sets, one per query, on CPU.
        let cand_sets: Vec<Vec<Candidate>> = queries
            .iter()
            .map(|q| self.collect_candidate_set(q, ef_search.max(k)))
            .collect::<GpuResult<Vec<_>>>()?;

        // Build the union of every query's candidate IDs. We keep
        // insertion order so that multiple paths (single-vs-batched)
        // see deterministic id orderings.
        let mut union_ids: Vec<u32> = Vec::new();
        let mut id_to_pos: std::collections::HashMap<u32, usize> =
            std::collections::HashMap::new();
        for cs in &cand_sets {
            for c in cs {
                if let std::collections::hash_map::Entry::Vacant(e) = id_to_pos.entry(c.id) {
                    e.insert(union_ids.len());
                    union_ids.push(c.id);
                }
            }
        }
        if union_ids.is_empty() {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        // Materialise the shared sub-corpus (union of candidates).
        let union_n = union_ids.len();
        let num_queries = queries.len();
        let mut sub_corpus: Vec<u32> = Vec::with_capacity(union_n * self.dim_u32);
        for &nid in &union_ids {
            let base = nid as usize * self.dim_u32;
            sub_corpus.extend_from_slice(&self.corpus_bits[base..base + self.dim_u32]);
        }

        // Flatten queries into a single row-major buffer for the kernel.
        let mut fused_queries: Vec<u32> = Vec::with_capacity(num_queries * self.dim_u32);
        for q in queries {
            fused_queries.extend_from_slice(q);
        }

        // One batched dispatch produces the full Q × union_n distance
        // matrix.
        let kernel = self.gpu_kernel.as_ref().expect("gpu_kernel confirmed Some");
        let dist_matrix = kernel.compute_batched(
            &fused_queries,
            &sub_corpus,
            num_queries,
            union_n,
            self.dim_bits,
        )?;

        // For each query, mask the row to only its own candidates and
        // take the top-K.
        let mut out = Vec::with_capacity(num_queries);
        for (qi, cs) in cand_sets.iter().enumerate() {
            if cs.is_empty() {
                out.push(Vec::new());
                continue;
            }
            let row_base = qi * union_n;
            let mut scored: Vec<(u32, u32)> = cs
                .iter()
                .map(|c| {
                    let pos = *id_to_pos
                        .get(&c.id)
                        .expect("every cand id must be in union");
                    let dist = dist_matrix[row_base + pos];
                    (dist, c.id)
                })
                .collect();
            scored.sort_by_key(|&(d, i)| (d, i));
            scored.truncate(k);
            out.push(scored);
        }
        Ok(out)
    }

    // ─── Internal helpers ───

    fn random_level(&self) -> usize {
        // ml = 1 / ln(M) — same heuristic as float `HnswIndex`.
        let ml = 1.0 / (M as f64).ln();
        let r: f64 = rand_f64();
        (-r.ln() * ml).floor() as usize
    }

    fn distance_to_node(&self, node_id: usize, query: &[u32]) -> u32 {
        let base = node_id * self.dim_u32;
        let node_slice = &self.corpus_bits[base..base + self.dim_u32];
        let mut sum: u32 = 0;
        for i in 0..self.dim_u32 {
            sum = sum.wrapping_add((query[i] ^ node_slice[i]).count_ones());
        }
        sum
    }

    fn distance_node_to_code(&self, node_id: usize, code: &[u32]) -> u32 {
        let base = node_id * self.dim_u32;
        let node_slice = &self.corpus_bits[base..base + self.dim_u32];
        let mut sum: u32 = 0;
        for i in 0..self.dim_u32 {
            sum = sum.wrapping_add((code[i] ^ node_slice[i]).count_ones());
        }
        sum
    }

    fn greedy_closest(&self, level: usize, start: u32, query: &[u32]) -> u32 {
        let mut current = start;
        let mut best_dist = self.distance_to_node(current as usize, query);

        loop {
            let mut changed = false;
            if level < self.graph.len() && (current as usize) < self.graph[level].len() {
                for &neighbor in &self.graph[level][current as usize] {
                    let dist = self.distance_to_node(neighbor as usize, query);
                    if dist < best_dist {
                        best_dist = dist;
                        current = neighbor;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    fn search_layer(
        &self,
        query: &[u32],
        entry_points: &[Candidate],
        ef: usize,
        level: usize,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        for ep in entry_points {
            visited.insert(ep.id);
            candidates.push(Reverse(*ep));
            results.push(*ep);
        }

        while let Some(Reverse(current)) = candidates.pop() {
            let worst_result = results.peek().map(|c| c.distance).unwrap_or(u32::MAX);
            if current.distance > worst_result && results.len() >= ef {
                break;
            }

            if level < self.graph.len() && (current.id as usize) < self.graph[level].len() {
                for &neighbor in &self.graph[level][current.id as usize] {
                    if visited.insert(neighbor) {
                        let dist = self.distance_to_node(neighbor as usize, query);
                        let worst = results.peek().map(|c| c.distance).unwrap_or(u32::MAX);
                        if results.len() < ef || dist < worst {
                            let c = Candidate {
                                distance: dist,
                                id: neighbor,
                            };
                            candidates.push(Reverse(c));
                            results.push(c);
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<Candidate> = results.into_vec();
        result_vec.sort();
        result_vec
    }

    /// Run the full HNSW descent and produce an `ef`-sized candidate
    /// set sorted ascending by distance. Used as the foundation of
    /// both [`Self::search`] and the GPU rerank entry points.
    fn collect_candidate_set(&self, query: &[u32], ef: usize) -> GpuResult<Vec<Candidate>> {
        if query.len() != self.dim_u32 {
            return Err(GpuError::ColumnTypeMismatch {
                expected: format!("query.len()={}", self.dim_u32),
                actual: format!("query.len()={}", query.len()),
            });
        }
        if self.entry_point.is_none() || ef == 0 {
            return Ok(Vec::new());
        }

        let entry = self.entry_point.expect("entry_point confirmed Some");
        let mut current = entry;

        // Greedy descent through upper levels.
        for lev in (1..=self.max_level).rev() {
            current = self.greedy_closest(lev, current, query);
        }

        // ef-search at level 0.
        let ep = vec![Candidate {
            distance: self.distance_to_node(current as usize, query),
            id: current,
        }];
        Ok(self.search_layer(query, &ep, ef, 0))
    }

    fn select_neighbors(&self, candidates: &[Candidate], max_conn: usize) -> Vec<Candidate> {
        // Same simple "closest M" heuristic used by the float
        // HnswIndex. A more advanced selector (the M-RNG-pruning of
        // hnswlib) is a future optimisation, not a Phase 2 A-final
        // requirement.
        candidates[..candidates.len().min(max_conn)].to_vec()
    }
}

/// CPU brute-force k-NN over a flat bit-packed corpus — used as the
/// gold oracle for binary HNSW recall measurements.
///
/// Returns top-K `(distance, doc_id)` pairs sorted ascending.
pub fn brute_force_knn_cpu(
    query: &[u32],
    corpus_bits: &[u32],
    num_vecs: usize,
    dim_u32: usize,
    k: usize,
) -> Vec<(u32, u32)> {
    let dists = hamming_distance_cpu(query, corpus_bits, num_vecs, dim_u32);
    let mut indexed: Vec<(u32, u32)> = dists
        .into_iter()
        .enumerate()
        .map(|(i, d)| (d, i as u32))
        .collect();
    indexed.sort_by_key(|&(d, i)| (d, i));
    indexed.truncate(k);
    indexed
}

/// Pseudo-random f64 in `[0, 1)` — same xorshift64 as
/// `hnsw::rand_f64`. Held local to this module to avoid coupling.
fn rand_f64() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        if x == 0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x12345678_9abcdef0);
            let tid = std::thread::current().id();
            let tid_hash = {
                let s = format!("{tid:?}");
                s.bytes()
                    .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
            };
            x = now ^ tid_hash;
            if x == 0 {
                x = 0x12345678_9abcdef0;
            }
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as f64) / (u64::MAX as f64)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 — keeps tests reproducible.
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

    /// Build a binary HNSW from a flat corpus + dim_bits, returning
    /// (index, corpus_bits) for downstream brute-force comparison.
    fn build_index(
        dim_bits: usize,
        num_vecs: usize,
        seed: u64,
    ) -> (BinaryHnswIndex, Vec<u32>) {
        let dim_u32 = dim_u32_for(dim_bits);
        let corpus = random_u32_vec(num_vecs * dim_u32, seed);
        let mut idx = BinaryHnswIndex::new(dim_bits, BinaryDistanceMetric::Hamming);
        for v in 0..num_vecs {
            let base = v * dim_u32;
            idx.insert(corpus[base..base + dim_u32].to_vec()).unwrap();
        }
        (idx, corpus)
    }

    #[test]
    fn test_binary_hnsw_empty() {
        let idx = BinaryHnswIndex::new(64, BinaryDistanceMetric::Hamming);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        let q = vec![0u32; 2];
        let out = idx.search(&q, 5, 50).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_binary_hnsw_dim_mismatch() {
        let mut idx = BinaryHnswIndex::new(64, BinaryDistanceMetric::Hamming);
        // dim_u32 = 2; passing length 3 must fail.
        let err = idx.insert(vec![0u32; 3]).unwrap_err();
        assert!(matches!(err, GpuError::ColumnTypeMismatch { .. }));

        idx.insert(vec![0u32; 2]).unwrap();
        let bad_q = vec![0u32; 3];
        let err = idx.search(&bad_q, 1, 10).unwrap_err();
        assert!(matches!(err, GpuError::ColumnTypeMismatch { .. }));
    }

    #[test]
    fn test_binary_hnsw_self_match() {
        // A query equal to a corpus vector must rank that vector
        // first with distance 0 — the most basic correctness signal.
        let dim_bits = 128;
        let (idx, corpus) = build_index(dim_bits, 100, 0xdead_beef_cafe_babe);
        let dim_u32 = dim_u32_for(dim_bits);
        let target_id = 42usize;
        let query = corpus[target_id * dim_u32..(target_id + 1) * dim_u32].to_vec();

        let out = idx.search(&query, 1, 50).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0, "self-match must have distance 0");
        assert_eq!(out[0].1, target_id as u32, "self-match must return self");
    }

    #[test]
    fn test_binary_hnsw_topk_sorted() {
        // top-K rows must come back ascending by distance.
        let dim_bits = 256;
        let (idx, _) = build_index(dim_bits, 500, 0x0123_4567_89ab_cdef);
        let dim_u32 = dim_u32_for(dim_bits);
        let query = random_u32_vec(dim_u32, 0xffff_eeee_dddd_cccc);
        let out = idx.search(&query, 10, 100).unwrap();
        assert_eq!(out.len(), 10);
        for w in 1..out.len() {
            assert!(
                out[w - 1].0 <= out[w].0,
                "results must be sorted ascending by distance: idx {w} broke order"
            );
        }
    }

    #[test]
    fn test_binary_hnsw_recall_small_scale() {
        // Small-scale: N=200, k=5, ef=100. The graph-walk top-K must
        // exactly equal the brute-force top-K (sorted by distance).
        // At this size HNSW should easily achieve recall 1.0.
        let dim_bits = 128;
        let num_vecs = 200;
        let (idx, corpus) = build_index(dim_bits, num_vecs, 0xa5a5_5a5a_a5a5_5a5a);
        let dim_u32 = dim_u32_for(dim_bits);
        let query = random_u32_vec(dim_u32, 0xbbbb_cccc_dddd_eeee);

        let hnsw_topk = idx.search(&query, 5, 100).unwrap();
        let bf_topk = brute_force_knn_cpu(&query, &corpus, num_vecs, dim_u32, 5);

        // Distances must match exactly (HNSW recall = 1.0 at this scale).
        let hnsw_dists: Vec<u32> = hnsw_topk.iter().map(|&(d, _)| d).collect();
        let bf_dists: Vec<u32> = bf_topk.iter().map(|&(d, _)| d).collect();
        assert_eq!(
            hnsw_dists, bf_dists,
            "small-scale HNSW must match brute force exactly: hnsw={hnsw_topk:?} bf={bf_topk:?}"
        );
    }

    #[test]
    fn test_binary_hnsw_recall_at_10_mid_scale() {
        // Mid-scale: N=2000, dim=256, k=10, ef=128. HNSW recall@10
        // against brute-force ground truth should hit ≥ 0.85 — the
        // expected floor for binary HNSW with default M/ef_construction.
        let dim_bits = 256;
        let num_vecs = 2_000;
        let k = 10;
        let ef_search = 128;
        let (idx, corpus) = build_index(dim_bits, num_vecs, 0x1234_5678_9abc_def0);
        let dim_u32 = dim_u32_for(dim_bits);

        let mut total_correct = 0;
        let mut total_expected = 0;
        let num_queries = 20;
        for qi in 0..num_queries {
            let q_seed = 0xf00d_face_dead_beefu64 ^ (qi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let query = random_u32_vec(dim_u32, q_seed);
            let hnsw_topk = idx.search(&query, k, ef_search).unwrap();
            let bf_topk = brute_force_knn_cpu(&query, &corpus, num_vecs, dim_u32, k);
            let bf_set: HashSet<u32> = bf_topk.iter().map(|&(_, id)| id).collect();
            for &(_, id) in &hnsw_topk {
                if bf_set.contains(&id) {
                    total_correct += 1;
                }
            }
            total_expected += k;
        }
        let recall = total_correct as f64 / total_expected as f64;
        assert!(
            recall >= 0.85,
            "binary HNSW recall@{k} = {recall:.3} below 0.85 floor (correct {total_correct} / {total_expected})"
        );
    }

    #[test]
    fn test_binary_hnsw_search_gpu_rerank_matches_cpu_search() {
        // GPU rerank with CPU-fallback context must return the same
        // top-K as the pure-CPU search (the rerank is exact).
        let ctx = GpuContext::cpu_fallback();
        let dim_bits = 128;
        let num_vecs = 300;
        let dim_u32 = dim_u32_for(dim_bits);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xa1b2_c3d4_e5f6_0708);

        let mut idx =
            BinaryHnswIndex::with_gpu(dim_bits, BinaryDistanceMetric::Hamming, &ctx).unwrap();
        for v in 0..num_vecs {
            let base = v * dim_u32;
            idx.insert(corpus[base..base + dim_u32].to_vec()).unwrap();
        }

        let query = random_u32_vec(dim_u32, 0x9999_8888_7777_6666);

        let cpu_only = idx.search(&query, 8, 64).unwrap();
        let gpu_rerank = idx.search_gpu_rerank(&query, 8, 64).unwrap();

        // Both walk the same graph, both rerank exactly. Distances
        // must agree exactly.
        let cpu_d: Vec<u32> = cpu_only.iter().map(|&(d, _)| d).collect();
        let gpu_d: Vec<u32> = gpu_rerank.iter().map(|&(d, _)| d).collect();
        assert_eq!(
            cpu_d, gpu_d,
            "GPU rerank distances must match CPU search exactly"
        );
    }

    #[test]
    fn test_binary_hnsw_batch_gpu_rerank_matches_per_query() {
        // The batched-rerank path must produce the same per-query
        // top-K as repeated single-query reranks.
        let ctx = GpuContext::cpu_fallback();
        let dim_bits = 128;
        let num_vecs = 256;
        let dim_u32 = dim_u32_for(dim_bits);
        let corpus = random_u32_vec(num_vecs * dim_u32, 0xc0ff_eec0_ffee_c0ff);

        let mut idx =
            BinaryHnswIndex::with_gpu(dim_bits, BinaryDistanceMetric::Hamming, &ctx).unwrap();
        for v in 0..num_vecs {
            let base = v * dim_u32;
            idx.insert(corpus[base..base + dim_u32].to_vec()).unwrap();
        }

        let queries: Vec<Vec<u32>> = (0..4)
            .map(|qi| random_u32_vec(dim_u32, 0xd00d_d00d_d00d_d00du64.wrapping_add(qi as u64)))
            .collect();

        let single_results: Vec<Vec<(u32, u32)>> = queries
            .iter()
            .map(|q| idx.search_gpu_rerank(q, 5, 64).unwrap())
            .collect();
        let batched = idx.search_batch_gpu_rerank(&queries, 5, 64).unwrap();

        assert_eq!(batched.len(), queries.len());
        for (q, (s, b)) in single_results.iter().zip(batched.iter()).enumerate() {
            // Distances at each rank must match — the only place the
            // two paths could legitimately differ is in tie-break id
            // ordering when two real candidates share a distance.
            // Distance-vector equality is the sound check.
            let sd: Vec<u32> = s.iter().map(|&(d, _)| d).collect();
            let bd: Vec<u32> = b.iter().map(|&(d, _)| d).collect();
            assert_eq!(sd, bd, "query {q}: single vs batched rerank distance mismatch");
        }
    }

    #[test]
    fn test_binary_hnsw_dispatch_with_gpu_no_kernel_falls_back() {
        // Without `with_gpu` / `attach_gpu`, search_gpu_rerank should
        // transparently delegate to CPU search.
        let dim_bits = 64;
        let (idx, _) = build_index(dim_bits, 50, 0x4242_4242_4242_4242);
        let dim_u32 = dim_u32_for(dim_bits);
        let query = random_u32_vec(dim_u32, 0xfacefaceu64);

        assert!(!idx.has_gpu());
        let cpu_path = idx.search(&query, 5, 50).unwrap();
        let rerank_path = idx.search_gpu_rerank(&query, 5, 50).unwrap();
        assert_eq!(cpu_path, rerank_path);
    }

    #[test]
    fn test_binary_hnsw_attach_gpu_lazy() {
        let dim_bits = 64;
        let (mut idx, _) = build_index(dim_bits, 30, 0x7777_8888_9999_aaaa);
        assert!(!idx.has_gpu());
        let ctx = GpuContext::cpu_fallback();
        idx.attach_gpu(&ctx).unwrap();
        assert!(idx.has_gpu());
    }

    #[test]
    fn test_brute_force_knn_cpu_oracle() {
        // Direct sanity check on the CPU oracle.
        let dim_bits = 64;
        let dim_u32 = 2;
        let corpus = vec![
            0xFFFF_FFFFu32, 0xFFFF_FFFF, // id 0: all 1s — dist 0 from query
            0u32, 0u32,                   // id 1: all 0s — dist 64
            0xFFFF_FFFFu32, 0u32,         // id 2: half — dist 32
            0u32, 0xFFFF_FFFFu32,         // id 3: half — dist 32
        ];
        let query = vec![0xFFFF_FFFFu32, 0xFFFF_FFFFu32];
        let out = brute_force_knn_cpu(&query, &corpus, 4, dim_u32, 3);
        assert_eq!(out.len(), 3);
        // First must be id 0 with dist 0; the next two are tied at
        // dist 32 with ids 2 and 3 (ascending tie-break).
        assert_eq!(out[0], (0, 0));
        assert_eq!(out[1].0, 32);
        assert_eq!(out[2].0, 32);
        let _ = dim_bits;
    }
}

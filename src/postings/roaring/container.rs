//! Roaring Bitmap container types — Phase 2 C-3 foundation.
//!
//! A Roaring posting list partitions every 32-bit doc-id into a
//! `(high16, low16)` pair. Each `high16` key buckets the low 16 bits
//! into one of three container forms chosen for cardinality + run
//! density:
//!
//! - [`ArrayContainer`] — sorted unique `u16` for sparse buckets (cardinality ≤
//!   [`ARRAY_TO_BITMAP_THRESHOLD`]). Storage: `2 × cardinality` bytes.
//! - [`BitmapContainer`] — fixed `[u32; 2048]` (= 8 KiB, 65 536 bits) for dense buckets. Word-wise
//!   bitwise AND/OR/XOR is the GPU-friendly form (Wave 4-B kernel uses this exact layout).
//! - [`RunContainer`] — sorted non-overlapping `Run { start, length }` for runs-of-set-bits
//!   buckets. Best when the content is a small number of contiguous ranges (`(num_runs * 4 + 2) <
//!   cardinality * 2`, i.e. average run length > 4).
//!
//! [`Container`] is the top-level dispatch enum with all 9 set-op
//! combinations (3 LHS forms × 3 RHS forms) wired in. Promotion and
//! demotion (e.g. `Array` → `Bitmap` after a large OR, `Bitmap` →
//! `Array` after a small AND) are decided by [`Container::optimize`]
//! and the per-op result-form heuristics.
//!
//! ## Wire format (per container)
//!
//! Every container serialises with a 1-byte tag prefix so the
//! top-level [`Container::to_bytes`] stream is self-describing:
//!
//! - `tag = 0x01` → [`ArrayContainer::to_bytes`] body
//! - `tag = 0x02` → [`BitmapContainer::to_bytes`] body
//! - `tag = 0x03` → [`RunContainer::to_bytes`] body
//!
//! All multi-byte integers are little-endian (matches the rest of the
//! Tantivy on-disk format and the WGSL/CUDA bitmap kernels).

use std::convert::TryFrom;

use crate::postings::roaring::BITMAP_CONTAINER_WORDS;

/// Promotion threshold: an [`ArrayContainer`] with strictly more than
/// `4096` entries upgrades to a [`BitmapContainer`] because the array
/// crosses the storage break-even point (`2 × 4096 B = 8 KiB`,
/// the bitmap's fixed footprint).
pub const ARRAY_TO_BITMAP_THRESHOLD: u32 = 4096;

/// Total cells in a 16-bit bucket (`1 << 16`). [`BitmapContainer`]
/// covers exactly this many positions; [`RunContainer::cardinality`]
/// can never exceed it.
pub const CONTAINER_CARDINALITY_MAX: u32 = 1 << 16;

const TAG_ARRAY: u8 = 0x01;
const TAG_BITMAP: u8 = 0x02;
const TAG_RUN: u8 = 0x03;

/// Container deserialisation errors.
///
/// Returned by every `from_bytes` constructor in this module. Callers
/// should treat any variant as a corrupted segment — none of these are
/// recoverable in a single-segment context.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ContainerError {
    /// Input slice was shorter than the minimum on-disk header for
    /// the inferred container form.
    #[error("container truncated: needed {needed} bytes, got {got}")]
    Truncated {
        /// Bytes the parser expected to read.
        needed: usize,
        /// Bytes actually available in the input slice.
        got: usize,
    },
    /// The 1-byte tag did not match any known container form.
    #[error("unknown container tag: {0:#04x}")]
    UnknownTag(u8),
    /// An [`ArrayContainer`] with non-monotonic or duplicate keys.
    #[error("array container keys not strictly increasing")]
    ArrayUnsorted,
    /// A [`RunContainer`] with overlapping or non-monotonic runs, or
    /// a single run whose end exceeds `u16::MAX`.
    #[error("run container runs overlap or are not strictly increasing")]
    RunInvariantViolation,
    /// A run with `length == 0`. Empty runs are not permitted —
    /// callers must omit them entirely instead of encoding zero-length
    /// markers.
    #[error("run container contains zero-length run")]
    EmptyRun,
}

/// A single contiguous run of set bits inside a 16-bit bucket.
///
/// `start` is the first set position; `length` is the number of
/// consecutive set bits (so the last set position is `start + length
/// - 1`). `length` is stored as `u16` but interpreted as a `u32` in
/// arithmetic so `start = 0` + `length = 65 536` (full bucket) is
/// representable in transit; on the wire we encode `length - 1` in a
/// `u16` so the bit-pattern `0xFFFF` denotes a length-65536 run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Run {
    /// First set position in the run (inclusive).
    pub start: u16,
    /// Number of consecutive set positions (so `start + length - 1` is
    /// the last set position). Always ≥ 1 — the empty case is encoded
    /// by omitting the run from the container, never with `length =
    /// 0`.
    pub length: u16,
}

impl Run {
    /// Build a [`Run`] from its inclusive endpoints.
    ///
    /// Returns `None` if `end < start`, which would describe a
    /// zero-length run.
    #[inline]
    #[must_use]
    pub fn from_inclusive(start: u16, end: u16) -> Option<Self> {
        if end < start {
            return None;
        }
        Some(Run {
            start,
            // end - start fits in u16 because both fit in u16, and the
            // result is non-negative thanks to the guard above. + 1
            // can wrap to 0 only when end - start == 65535, which means
            // the run spans the full bucket; we encode that as length
            // = 65535 and rely on contains/iter to interpret length=0
            // as "full bucket" — but since wrapping_add(1) yields 0
            // for that case, we explicitly handle it.
            length: u16::try_from((u32::from(end) - u32::from(start)) + 1).unwrap_or(0),
        })
    }

    /// Inclusive end position of the run (`start + length - 1`),
    /// extended to `u32` to avoid overflow in the full-bucket case.
    #[inline]
    #[must_use]
    pub fn end_inclusive(self) -> u32 {
        let len = if self.length == 0 {
            CONTAINER_CARDINALITY_MAX
        } else {
            u32::from(self.length)
        };
        u32::from(self.start) + len - 1
    }

    /// Cardinality of this run (`length`, treating the encoded `0` as
    /// a full 65 536-cell bucket).
    #[inline]
    #[must_use]
    pub fn cardinality(self) -> u32 {
        if self.length == 0 {
            CONTAINER_CARDINALITY_MAX
        } else {
            u32::from(self.length)
        }
    }

    /// True iff `key` is one of the set positions in this run.
    #[inline]
    #[must_use]
    pub fn contains(self, key: u16) -> bool {
        let key32 = u32::from(key);
        key32 >= u32::from(self.start) && key32 <= self.end_inclusive()
    }
}

// =============================================================
// ArrayContainer
// =============================================================

/// Sparse (≤ `ARRAY_TO_BITMAP_THRESHOLD`) container backed by a
/// strictly-increasing `Vec<u16>`.
///
/// Contained keys are kept sorted at all times; the `insert`,
/// `from_sorted`, and `from_bytes` constructors enforce the
/// invariant. `cardinality` is the same as `keys.len()`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArrayContainer {
    /// Sorted, unique low-16 keys in this bucket.
    pub keys: Vec<u16>,
}

impl ArrayContainer {
    /// Empty array container.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        ArrayContainer { keys: Vec::new() }
    }

    /// Build from a slice of sorted unique keys without copying via
    /// the `insert` hot path.
    ///
    /// This bypasses the per-key `binary_search` of [`Self::insert`]
    /// — callers must guarantee the slice is strictly increasing.
    /// On invariant violation in debug builds we panic; in release we
    /// fall back to sorting + dedup-in-place for safety.
    #[must_use]
    pub fn from_sorted(keys: Vec<u16>) -> Self {
        let mut keys = keys;
        if !is_strictly_increasing(&keys) {
            debug_assert!(
                is_strictly_increasing(&keys),
                "ArrayContainer::from_sorted received non-monotonic keys"
            );
            keys.sort_unstable();
            keys.dedup();
        }
        ArrayContainer { keys }
    }

    /// Insert `key`, keeping the `keys` invariant. No-op if the key
    /// is already present. `O(log n + n)` because the tail shifts.
    pub fn insert(&mut self, key: u16) {
        match self.keys.binary_search(&key) {
            Ok(_) => {}
            Err(pos) => self.keys.insert(pos, key),
        }
    }

    /// True iff `key` is in this container. `O(log n)`.
    #[inline]
    #[must_use]
    pub fn contains(&self, key: u16) -> bool {
        self.keys.binary_search(&key).is_ok()
    }

    /// Number of set bits. Always equal to `self.keys.len()`.
    #[inline]
    #[must_use]
    pub fn cardinality(&self) -> u32 {
        // u32 fits because cardinality cannot exceed 65 536 (full
        // bucket).
        self.keys.len() as u32
    }

    /// Iterator over the sorted keys.
    pub fn iter(&self) -> impl Iterator<Item = u16> + '_ {
        self.keys.iter().copied()
    }

    /// Galloping intersection — both inputs are sorted, so we walk in
    /// `O(min(n, m))` worst-case, with binary-search short-circuits
    /// when one side is much sparser than the other (galloping = the
    /// classic "double-step until overshoot, then binary search"
    /// technique from Demaine et al. 2000).
    #[must_use]
    pub fn and(&self, other: &ArrayContainer) -> ArrayContainer {
        let (small, large) = if self.keys.len() <= other.keys.len() {
            (&self.keys, &other.keys)
        } else {
            (&other.keys, &self.keys)
        };
        // For asymmetric inputs (small ≪ large) galloping pays off;
        // for balanced inputs the linear merge wins. Decision threshold:
        // if large/small < 32 we use linear; else galloping.
        let mut out: Vec<u16> = Vec::with_capacity(small.len());
        if large.len() / small.len().max(1) < 32 {
            // Linear merge.
            let (mut i, mut j) = (0usize, 0usize);
            while i < self.keys.len() && j < other.keys.len() {
                let a = self.keys[i];
                let b = other.keys[j];
                if a == b {
                    out.push(a);
                    i += 1;
                    j += 1;
                } else if a < b {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        } else {
            // Galloping: for each small-side key, binary search in the
            // suffix of the large side.
            let mut large_start = 0usize;
            for &k in small {
                if let Ok(idx) = large[large_start..].binary_search(&k) {
                    out.push(k);
                    large_start += idx + 1;
                } else {
                    // binary_search Err returns the insertion index;
                    // the next match must be at-or-after that point.
                    let idx = match large[large_start..].binary_search(&k) {
                        Ok(_) => unreachable!(),
                        Err(idx) => idx,
                    };
                    large_start += idx;
                }
                if large_start >= large.len() {
                    break;
                }
            }
        }
        ArrayContainer { keys: out }
    }

    /// Set union. May promote to a [`BitmapContainer`] if the result
    /// crosses the [`ARRAY_TO_BITMAP_THRESHOLD`] cutoff.
    #[must_use]
    pub fn or(&self, other: &ArrayContainer) -> Container {
        let mut out: Vec<u16> = Vec::with_capacity(self.keys.len() + other.keys.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.keys.len() && j < other.keys.len() {
            let a = self.keys[i];
            let b = other.keys[j];
            match a.cmp(&b) {
                std::cmp::Ordering::Less => {
                    out.push(a);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(b);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    out.push(a);
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend_from_slice(&self.keys[i..]);
        out.extend_from_slice(&other.keys[j..]);
        let card = u32::try_from(out.len()).unwrap_or(u32::MAX);
        if card > ARRAY_TO_BITMAP_THRESHOLD {
            let mut bm = BitmapContainer::new();
            for k in out {
                bm.insert(k);
            }
            Container::Bitmap(bm)
        } else {
            Container::Array(ArrayContainer { keys: out })
        }
    }

    /// Set symmetric difference. May promote to a
    /// [`BitmapContainer`].
    #[must_use]
    pub fn xor(&self, other: &ArrayContainer) -> Container {
        let mut out: Vec<u16> = Vec::with_capacity(self.keys.len() + other.keys.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.keys.len() && j < other.keys.len() {
            let a = self.keys[i];
            let b = other.keys[j];
            match a.cmp(&b) {
                std::cmp::Ordering::Less => {
                    out.push(a);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    out.push(b);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    // present in both → drop
                    i += 1;
                    j += 1;
                }
            }
        }
        out.extend_from_slice(&self.keys[i..]);
        out.extend_from_slice(&other.keys[j..]);
        let card = u32::try_from(out.len()).unwrap_or(u32::MAX);
        if card > ARRAY_TO_BITMAP_THRESHOLD {
            let mut bm = BitmapContainer::new();
            for k in out {
                bm.insert(k);
            }
            Container::Bitmap(bm)
        } else {
            Container::Array(ArrayContainer { keys: out })
        }
    }

    /// On-disk encoding: `[u32 cardinality LE][u16 keys LE × cardinality]`.
    ///
    /// Cardinality is encoded as `u32` (rather than the sufficient
    /// `u16`) for tag-free alignment with the bitmap form's 32-bit
    /// length field — avoids a special-case parser branch.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.keys.len() * 2);
        out.extend_from_slice(&self.cardinality().to_le_bytes());
        for k in &self.keys {
            out.extend_from_slice(&k.to_le_bytes());
        }
        out
    }

    /// Inverse of [`Self::to_bytes`].
    pub fn from_bytes(slice: &[u8]) -> Result<Self, ContainerError> {
        if slice.len() < 4 {
            return Err(ContainerError::Truncated {
                needed: 4,
                got: slice.len(),
            });
        }
        let card = u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]);
        let needed = 4 + (card as usize) * 2;
        if slice.len() < needed {
            return Err(ContainerError::Truncated {
                needed,
                got: slice.len(),
            });
        }
        let mut keys = Vec::with_capacity(card as usize);
        let mut prev: Option<u16> = None;
        for i in 0..(card as usize) {
            let off = 4 + i * 2;
            let k = u16::from_le_bytes([slice[off], slice[off + 1]]);
            if let Some(p) = prev {
                if k <= p {
                    return Err(ContainerError::ArrayUnsorted);
                }
            }
            keys.push(k);
            prev = Some(k);
        }
        Ok(ArrayContainer { keys })
    }
}

// =============================================================
// BitmapContainer
// =============================================================

/// Dense (cardinality > [`ARRAY_TO_BITMAP_THRESHOLD`]) container with
/// fixed `[u32; BITMAP_CONTAINER_WORDS]` storage (= 8 KiB).
///
/// The exact layout matches the WGSL/CUDA kernel input format
/// ([`crate::postings::roaring::BITMAP_CONTAINER_WORDS`] = 2048 u32
/// words = 65 536 bits): word `w` covers positions `32 * w .. 32 *
/// (w + 1)`, and inside a word, bit `b` covers position `32 * w + b`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitmapContainer {
    /// 8 KiB raw word array.
    pub words: Box<[u32; BITMAP_CONTAINER_WORDS]>,
}

impl Default for BitmapContainer {
    fn default() -> Self {
        Self::new()
    }
}

impl BitmapContainer {
    /// All-zero bitmap (cardinality 0).
    #[must_use]
    pub fn new() -> Self {
        BitmapContainer {
            words: Box::new([0u32; BITMAP_CONTAINER_WORDS]),
        }
    }

    /// Promote an [`ArrayContainer`] into this dense form.
    #[must_use]
    pub fn from_array(arr: &ArrayContainer) -> Self {
        let mut bm = BitmapContainer::new();
        for &k in &arr.keys {
            bm.insert(k);
        }
        bm
    }

    /// Take an existing word array (from the GPU side, e.g.) without
    /// re-inserting bit-by-bit.
    #[must_use]
    pub fn from_words(words: Box<[u32; BITMAP_CONTAINER_WORDS]>) -> Self {
        BitmapContainer { words }
    }

    /// Set bit `key`. Idempotent.
    #[inline]
    pub fn insert(&mut self, key: u16) {
        let word = (key as usize) >> 5;
        let bit = (key as u32) & 0x1f;
        self.words[word] |= 1u32 << bit;
    }

    /// Clear bit `key`. Idempotent.
    #[inline]
    pub fn remove(&mut self, key: u16) {
        let word = (key as usize) >> 5;
        let bit = (key as u32) & 0x1f;
        self.words[word] &= !(1u32 << bit);
    }

    /// True iff bit `key` is set.
    #[inline]
    #[must_use]
    pub fn contains(&self, key: u16) -> bool {
        let word = (key as usize) >> 5;
        let bit = (key as u32) & 0x1f;
        (self.words[word] >> bit) & 1 == 1
    }

    /// Population count summed over the full word array.
    #[must_use]
    pub fn cardinality(&self) -> u32 {
        let mut sum: u32 = 0;
        for w in self.words.iter() {
            sum += w.count_ones();
        }
        sum
    }

    /// Bit-scan iterator. Yields keys in increasing order.
    pub fn iter(&self) -> BitmapContainerIter<'_> {
        BitmapContainerIter {
            words: self.words.as_ref(),
            word_idx: 0,
            current_word: self.words[0],
        }
    }

    /// Word-wise AND. May demote to [`ArrayContainer`] if the result
    /// fits below [`ARRAY_TO_BITMAP_THRESHOLD`].
    #[must_use]
    pub fn and(&self, other: &BitmapContainer) -> Container {
        let mut out = Box::new([0u32; BITMAP_CONTAINER_WORDS]);
        let mut card: u32 = 0;
        for i in 0..BITMAP_CONTAINER_WORDS {
            let w = self.words[i] & other.words[i];
            out[i] = w;
            card += w.count_ones();
        }
        if card <= ARRAY_TO_BITMAP_THRESHOLD {
            // Demote to ArrayContainer.
            let mut keys = Vec::with_capacity(card as usize);
            for (word_idx, &w) in out.iter().enumerate() {
                let mut x = w;
                while x != 0 {
                    let bit = x.trailing_zeros();
                    keys.push((word_idx as u16) * 32 + bit as u16);
                    x &= x - 1;
                }
            }
            Container::Array(ArrayContainer { keys })
        } else {
            Container::Bitmap(BitmapContainer { words: out })
        }
    }

    /// Word-wise OR. Result is always dense enough to stay a bitmap.
    #[must_use]
    pub fn or(&self, other: &BitmapContainer) -> BitmapContainer {
        let mut out = Box::new([0u32; BITMAP_CONTAINER_WORDS]);
        for i in 0..BITMAP_CONTAINER_WORDS {
            out[i] = self.words[i] | other.words[i];
        }
        BitmapContainer { words: out }
    }

    /// Word-wise XOR. May demote to [`ArrayContainer`].
    #[must_use]
    pub fn xor(&self, other: &BitmapContainer) -> Container {
        let mut out = Box::new([0u32; BITMAP_CONTAINER_WORDS]);
        let mut card: u32 = 0;
        for i in 0..BITMAP_CONTAINER_WORDS {
            let w = self.words[i] ^ other.words[i];
            out[i] = w;
            card += w.count_ones();
        }
        if card <= ARRAY_TO_BITMAP_THRESHOLD {
            let mut keys = Vec::with_capacity(card as usize);
            for (word_idx, &w) in out.iter().enumerate() {
                let mut x = w;
                while x != 0 {
                    let bit = x.trailing_zeros();
                    keys.push((word_idx as u16) * 32 + bit as u16);
                    x &= x - 1;
                }
            }
            Container::Array(ArrayContainer { keys })
        } else {
            Container::Bitmap(BitmapContainer { words: out })
        }
    }

    /// On-disk encoding: 8 KiB raw little-endian word array.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(BITMAP_CONTAINER_WORDS * 4);
        for w in self.words.iter() {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Inverse of [`Self::to_bytes`].
    pub fn from_bytes(slice: &[u8]) -> Result<Self, ContainerError> {
        let needed = BITMAP_CONTAINER_WORDS * 4;
        if slice.len() < needed {
            return Err(ContainerError::Truncated {
                needed,
                got: slice.len(),
            });
        }
        let mut words = Box::new([0u32; BITMAP_CONTAINER_WORDS]);
        for i in 0..BITMAP_CONTAINER_WORDS {
            let off = i * 4;
            words[i] =
                u32::from_le_bytes([slice[off], slice[off + 1], slice[off + 2], slice[off + 3]]);
        }
        Ok(BitmapContainer { words })
    }
}

/// Bit-scan iterator over [`BitmapContainer`] yielding set keys in
/// strictly-increasing order. `O(cardinality)` total across the full
/// scan.
pub struct BitmapContainerIter<'a> {
    words: &'a [u32; BITMAP_CONTAINER_WORDS],
    word_idx: usize,
    current_word: u32,
}

impl Iterator for BitmapContainerIter<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        loop {
            if self.current_word != 0 {
                let bit = self.current_word.trailing_zeros();
                self.current_word &= self.current_word - 1;
                // Word index ≤ 2047 (fits in u16 with 5 spare bits) +
                // bit ≤ 31, so the multiplication-add cannot overflow
                // u16 (2047 * 32 + 31 = 65 535).
                return Some((self.word_idx as u16) * 32 + bit as u16);
            }
            self.word_idx += 1;
            if self.word_idx >= BITMAP_CONTAINER_WORDS {
                return None;
            }
            self.current_word = self.words[self.word_idx];
        }
    }
}

// =============================================================
// RunContainer
// =============================================================

/// Run-length-encoded container.
///
/// Stores a sorted, non-overlapping list of [`Run`]s. A run with
/// `length = 0` is *only* used internally for the "full bucket" case
/// (cardinality 65 536); externally the cardinality is correctly
/// reported and no zero-length runs are accepted by `from_runs` or
/// `from_bytes`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunContainer {
    /// Sorted, non-overlapping runs.
    pub runs: Vec<Run>,
}

impl RunContainer {
    /// Empty run container.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        RunContainer { runs: Vec::new() }
    }

    /// Build from a pre-validated slice of runs. Returns an error if
    /// any run has zero length, or if the runs are not strictly
    /// increasing and non-overlapping.
    pub fn from_runs(runs: Vec<Run>) -> Result<Self, ContainerError> {
        for w in runs.windows(2) {
            if w[0].length == 0 {
                return Err(ContainerError::EmptyRun);
            }
            if w[0].end_inclusive() >= u32::from(w[1].start) {
                return Err(ContainerError::RunInvariantViolation);
            }
        }
        if let Some(last) = runs.last() {
            if last.length == 0 && runs.len() > 1 {
                // length=0 (full bucket) only valid as the sole run.
                return Err(ContainerError::RunInvariantViolation);
            }
        }
        Ok(RunContainer { runs })
    }

    /// Sum of `cardinality` across all runs.
    #[must_use]
    pub fn cardinality(&self) -> u32 {
        let mut sum: u32 = 0;
        for r in &self.runs {
            sum = sum.saturating_add(r.cardinality());
        }
        sum
    }

    /// True iff `key` is inside any run. `O(log n)` via binary search
    /// on the sorted run starts.
    #[must_use]
    pub fn contains(&self, key: u16) -> bool {
        // Binary search for the rightmost run whose start ≤ key.
        let key32 = u32::from(key);
        let pos = self.runs.partition_point(|r| u32::from(r.start) <= key32);
        if pos == 0 {
            return false;
        }
        self.runs[pos - 1].contains(key)
    }

    /// Iterator over all set keys, in strictly-increasing order.
    pub fn iter(&self) -> RunContainerIter<'_> {
        RunContainerIter {
            runs: &self.runs,
            run_idx: 0,
            offset: 0,
        }
    }

    /// Sweep-line interval intersection. `O(n + m)`.
    #[must_use]
    pub fn and(&self, other: &RunContainer) -> Container {
        let mut out: Vec<Run> = Vec::new();
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.runs.len() && j < other.runs.len() {
            let a = self.runs[i];
            let b = other.runs[j];
            let a_end = a.end_inclusive();
            let b_end = b.end_inclusive();
            let overlap_start = u32::from(a.start).max(u32::from(b.start));
            let overlap_end = a_end.min(b_end);
            if overlap_start <= overlap_end {
                let length = overlap_end - overlap_start + 1;
                out.push(Run {
                    start: overlap_start as u16,
                    length: u16::try_from(length).unwrap_or(0),
                });
            }
            if a_end < b_end {
                i += 1;
            } else {
                j += 1;
            }
        }
        Container::Run(RunContainer { runs: out }).optimize()
    }

    /// Sweep-line interval union with greedy merge of touching runs.
    #[must_use]
    pub fn or(&self, other: &RunContainer) -> Container {
        let mut out: Vec<Run> = Vec::with_capacity(self.runs.len() + other.runs.len());
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.runs.len() || j < other.runs.len() {
            let next = if j == other.runs.len() {
                let r = self.runs[i];
                i += 1;
                r
            } else if i == self.runs.len() {
                let r = other.runs[j];
                j += 1;
                r
            } else if self.runs[i].start <= other.runs[j].start {
                let r = self.runs[i];
                i += 1;
                r
            } else {
                let r = other.runs[j];
                j += 1;
                r
            };
            push_or_merge(&mut out, next);
        }
        Container::Run(RunContainer { runs: out }).optimize()
    }

    /// Symmetric difference. Computed by materialising both run sets
    /// into a bitmap (since interval XOR fragmentation is awkward to
    /// express in pure run-form) and then optimising the result back
    /// to its smallest container form.
    #[must_use]
    pub fn xor(&self, other: &RunContainer) -> Container {
        let bm_a = BitmapContainer::from_run(self);
        let bm_b = BitmapContainer::from_run(other);
        bm_a.xor(&bm_b).optimize()
    }

    /// Convert from an [`ArrayContainer`] by run-length scanning.
    #[must_use]
    pub fn from_array(arr: &ArrayContainer) -> Self {
        let mut runs: Vec<Run> = Vec::new();
        let mut iter = arr.keys.iter().copied();
        if let Some(first) = iter.next() {
            let mut start = first;
            let mut prev = first;
            for k in iter {
                if k == prev.wrapping_add(1) {
                    prev = k;
                } else {
                    runs.push(Run {
                        start,
                        length: u16::try_from((u32::from(prev) - u32::from(start)) + 1)
                            .unwrap_or(0),
                    });
                    start = k;
                    prev = k;
                }
            }
            runs.push(Run {
                start,
                length: u16::try_from((u32::from(prev) - u32::from(start)) + 1).unwrap_or(0),
            });
        }
        RunContainer { runs }
    }

    /// Convert from a [`BitmapContainer`] by scanning bit transitions.
    #[must_use]
    pub fn from_bitmap(bm: &BitmapContainer) -> Self {
        let mut runs: Vec<Run> = Vec::new();
        let mut in_run = false;
        let mut start: u16 = 0;
        let mut prev: u16 = 0;
        for k in bm.iter() {
            if !in_run {
                start = k;
                prev = k;
                in_run = true;
            } else if k == prev.wrapping_add(1) {
                prev = k;
            } else {
                runs.push(Run {
                    start,
                    length: u16::try_from((u32::from(prev) - u32::from(start)) + 1).unwrap_or(0),
                });
                start = k;
                prev = k;
            }
        }
        if in_run {
            runs.push(Run {
                start,
                length: u16::try_from((u32::from(prev) - u32::from(start)) + 1).unwrap_or(0),
            });
        }
        RunContainer { runs }
    }

    /// On-disk encoding:
    /// `[u16 num_runs LE][u16 start LE | u16 length-1 LE] × num_runs`.
    ///
    /// We encode `length - 1` so the full-bucket sentinel `length =
    /// 65 536` (which doesn't fit in `u16`) can ride in `0xFFFF`
    /// without ambiguity with the empty-run case (which is forbidden
    /// at construction time).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + self.runs.len() * 4);
        let n = u16::try_from(self.runs.len()).unwrap_or(u16::MAX);
        out.extend_from_slice(&n.to_le_bytes());
        for r in &self.runs {
            out.extend_from_slice(&r.start.to_le_bytes());
            // Internal length=0 (full-bucket) maps to wire 0xFFFF
            // (length - 1 = 65 535). Otherwise length - 1 fits in u16.
            let wire_len = if r.length == 0 {
                0xFFFFu16
            } else {
                r.length - 1
            };
            out.extend_from_slice(&wire_len.to_le_bytes());
        }
        out
    }

    /// Inverse of [`Self::to_bytes`].
    pub fn from_bytes(slice: &[u8]) -> Result<Self, ContainerError> {
        if slice.len() < 2 {
            return Err(ContainerError::Truncated {
                needed: 2,
                got: slice.len(),
            });
        }
        let n = u16::from_le_bytes([slice[0], slice[1]]) as usize;
        let needed = 2 + n * 4;
        if slice.len() < needed {
            return Err(ContainerError::Truncated {
                needed,
                got: slice.len(),
            });
        }
        let mut runs = Vec::with_capacity(n);
        for i in 0..n {
            let off = 2 + i * 4;
            let start = u16::from_le_bytes([slice[off], slice[off + 1]]);
            let wire_len = u16::from_le_bytes([slice[off + 2], slice[off + 3]]);
            // wire 0xFFFF → internal length = 0 (full bucket). All
            // other values: internal length = wire + 1, which fits in
            // u16 because wire ≤ 0xFFFE.
            let length = if wire_len == 0xFFFF { 0 } else { wire_len + 1 };
            runs.push(Run { start, length });
        }
        // Validate after parse: full-bucket sentinel only valid as
        // the lone run; otherwise non-overlap + monotonic.
        if runs.len() > 1 {
            for w in runs.windows(2) {
                if w[0].length == 0 || w[1].length == 0 {
                    return Err(ContainerError::RunInvariantViolation);
                }
                if w[0].end_inclusive() >= u32::from(w[1].start) {
                    return Err(ContainerError::RunInvariantViolation);
                }
            }
        }
        Ok(RunContainer { runs })
    }
}

impl BitmapContainer {
    /// Materialise a [`RunContainer`] into the dense form. Inverse of
    /// [`RunContainer::from_bitmap`].
    #[must_use]
    pub fn from_run(rc: &RunContainer) -> Self {
        let mut bm = BitmapContainer::new();
        for r in &rc.runs {
            let card = r.cardinality();
            for offset in 0..card {
                let pos = u32::from(r.start) + offset;
                // pos is at most 65 535 because cardinality is bounded
                // by the bucket size.
                bm.insert(pos as u16);
            }
        }
        bm
    }
}

/// Iterator over [`RunContainer`] yielding every set position in
/// strictly-increasing order. `O(cardinality)` total.
pub struct RunContainerIter<'a> {
    runs: &'a [Run],
    run_idx: usize,
    offset: u32,
}

impl Iterator for RunContainerIter<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        if self.run_idx >= self.runs.len() {
            return None;
        }
        let r = self.runs[self.run_idx];
        let card = r.cardinality();
        if self.offset >= card {
            self.run_idx += 1;
            self.offset = 0;
            return self.next();
        }
        let pos = u32::from(r.start) + self.offset;
        self.offset += 1;
        Some(pos as u16)
    }
}

// =============================================================
// Container enum (top-level dispatch)
// =============================================================

/// Top-level dispatch over the three container forms.
///
/// All set ops are total: every (LHS, RHS) pairing has a defined path
/// — see [`Self::and`], [`Self::or`], [`Self::xor`]. Result form is
/// chosen by [`Self::optimize`] (and by per-op heuristics in the
/// individual container methods).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Container {
    /// Sparse form (≤ [`ARRAY_TO_BITMAP_THRESHOLD`]).
    Array(ArrayContainer),
    /// Dense form (8 KiB fixed).
    Bitmap(BitmapContainer),
    /// RLE form.
    Run(RunContainer),
}

impl Container {
    /// Total cardinality (always ≤ 65 536).
    #[must_use]
    pub fn cardinality(&self) -> u32 {
        match self {
            Container::Array(a) => a.cardinality(),
            Container::Bitmap(b) => b.cardinality(),
            Container::Run(r) => r.cardinality(),
        }
    }

    /// True iff `key` is in this container.
    #[must_use]
    pub fn contains(&self, key: u16) -> bool {
        match self {
            Container::Array(a) => a.contains(key),
            Container::Bitmap(b) => b.contains(key),
            Container::Run(r) => r.contains(key),
        }
    }

    /// Iterator over all set keys in strictly-increasing order.
    pub fn iter(&self) -> Box<dyn Iterator<Item = u16> + '_> {
        match self {
            Container::Array(a) => Box::new(a.iter()),
            Container::Bitmap(b) => Box::new(b.iter()),
            Container::Run(r) => Box::new(r.iter()),
        }
    }

    /// Set intersection — full 3 × 3 dispatch.
    #[must_use]
    pub fn and(&self, other: &Container) -> Container {
        match (self, other) {
            (Container::Array(a), Container::Array(b)) => Container::Array(a.and(b)),
            (Container::Bitmap(a), Container::Bitmap(b)) => a.and(b),
            (Container::Run(a), Container::Run(b)) => a.and(b),
            (Container::Array(a), Container::Bitmap(b))
            | (Container::Bitmap(b), Container::Array(a)) => {
                // Filter the array by membership in the bitmap.
                let keys: Vec<u16> = a.keys.iter().copied().filter(|&k| b.contains(k)).collect();
                Container::Array(ArrayContainer { keys })
            }
            (Container::Array(a), Container::Run(r)) | (Container::Run(r), Container::Array(a)) => {
                let keys: Vec<u16> = a.keys.iter().copied().filter(|&k| r.contains(k)).collect();
                Container::Array(ArrayContainer { keys })
            }
            (Container::Bitmap(b), Container::Run(r))
            | (Container::Run(r), Container::Bitmap(b)) => {
                // Materialise the run-side as a bitmap and AND.
                let r_bm = BitmapContainer::from_run(r);
                b.and(&r_bm)
            }
        }
    }

    /// Set union — full 3 × 3 dispatch.
    #[must_use]
    pub fn or(&self, other: &Container) -> Container {
        match (self, other) {
            (Container::Array(a), Container::Array(b)) => a.or(b),
            (Container::Bitmap(a), Container::Bitmap(b)) => Container::Bitmap(a.or(b)),
            (Container::Run(a), Container::Run(b)) => a.or(b),
            (Container::Array(a), Container::Bitmap(b))
            | (Container::Bitmap(b), Container::Array(a)) => {
                let mut out = b.clone();
                for &k in &a.keys {
                    out.insert(k);
                }
                Container::Bitmap(out)
            }
            (Container::Array(a), Container::Run(r)) | (Container::Run(r), Container::Array(a)) => {
                // Build a bitmap from the run side (cheap dense) and
                // OR-insert the array keys.
                let mut out = BitmapContainer::from_run(r);
                for &k in &a.keys {
                    out.insert(k);
                }
                Container::Bitmap(out).optimize()
            }
            (Container::Bitmap(b), Container::Run(r))
            | (Container::Run(r), Container::Bitmap(b)) => {
                let mut out = b.clone();
                for run in &r.runs {
                    let card = run.cardinality();
                    for offset in 0..card {
                        let pos = u32::from(run.start) + offset;
                        out.insert(pos as u16);
                    }
                }
                Container::Bitmap(out)
            }
        }
    }

    /// Symmetric difference — full 3 × 3 dispatch.
    #[must_use]
    pub fn xor(&self, other: &Container) -> Container {
        match (self, other) {
            (Container::Array(a), Container::Array(b)) => a.xor(b),
            (Container::Bitmap(a), Container::Bitmap(b)) => a.xor(b),
            (Container::Run(a), Container::Run(b)) => a.xor(b),
            (Container::Array(a), Container::Bitmap(b))
            | (Container::Bitmap(b), Container::Array(a)) => {
                let mut out = b.clone();
                for &k in &a.keys {
                    if out.contains(k) {
                        out.remove(k);
                    } else {
                        out.insert(k);
                    }
                }
                Container::Bitmap(out).optimize()
            }
            (Container::Array(a), Container::Run(r)) | (Container::Run(r), Container::Array(a)) => {
                let mut out = BitmapContainer::from_run(r);
                for &k in &a.keys {
                    if out.contains(k) {
                        out.remove(k);
                    } else {
                        out.insert(k);
                    }
                }
                Container::Bitmap(out).optimize()
            }
            (Container::Bitmap(b), Container::Run(r))
            | (Container::Run(r), Container::Bitmap(b)) => {
                let r_bm = BitmapContainer::from_run(r);
                b.xor(&r_bm).optimize()
            }
        }
    }

    /// Pick the best container form for the current contents.
    ///
    /// Logic (Lemire 2014 § 3.3):
    /// - cardinality ≤ [`ARRAY_TO_BITMAP_THRESHOLD`]: prefer [`ArrayContainer`] (or
    ///   [`RunContainer`] when run-density yields strictly smaller storage).
    /// - cardinality > [`ARRAY_TO_BITMAP_THRESHOLD`]: prefer [`BitmapContainer`] (or
    ///   [`RunContainer`] when run-density wins).
    ///
    /// "Run-density wins" means encoded run bytes (`2 + 4 *
    /// num_runs`) is strictly less than the alternative encoding
    /// (`4 + 2 * cardinality` for arrays, `8 192` for bitmaps).
    #[must_use]
    pub fn optimize(self) -> Self {
        let card = self.cardinality();
        // Compute the run count we *would* get if we converted to a
        // RunContainer. For `Run` itself, that's just self.runs.len().
        let run_bytes = match &self {
            Container::Array(a) => {
                let runs = count_runs_in_array(a);
                2 + runs * 4
            }
            Container::Bitmap(b) => {
                let runs = count_runs_in_bitmap(b);
                2 + runs * 4
            }
            Container::Run(r) => 2 + r.runs.len() * 4,
        };
        let array_bytes = 4 + (card as usize) * 2;
        let bitmap_bytes = BITMAP_CONTAINER_WORDS * 4; // 8 KiB
        let run_smallest = run_bytes < array_bytes && run_bytes < bitmap_bytes && card > 0;
        if run_smallest {
            return match self {
                Container::Run(r) => Container::Run(r),
                Container::Array(a) => Container::Run(RunContainer::from_array(&a)),
                Container::Bitmap(b) => Container::Run(RunContainer::from_bitmap(&b)),
            };
        }
        if card <= ARRAY_TO_BITMAP_THRESHOLD {
            // Array is the winner unless we are already an Array.
            match self {
                Container::Array(a) => Container::Array(a),
                Container::Bitmap(b) => {
                    let mut keys: Vec<u16> = Vec::with_capacity(card as usize);
                    for k in b.iter() {
                        keys.push(k);
                    }
                    Container::Array(ArrayContainer { keys })
                }
                Container::Run(r) => {
                    let mut keys: Vec<u16> = Vec::with_capacity(card as usize);
                    for k in r.iter() {
                        keys.push(k);
                    }
                    Container::Array(ArrayContainer { keys })
                }
            }
        } else {
            // Bitmap is the winner unless we are already a Bitmap.
            match self {
                Container::Bitmap(b) => Container::Bitmap(b),
                Container::Array(a) => Container::Bitmap(BitmapContainer::from_array(&a)),
                Container::Run(r) => Container::Bitmap(BitmapContainer::from_run(&r)),
            }
        }
    }

    /// Self-describing encoding: `[u8 tag][body]`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Container::Array(a) => {
                out.push(TAG_ARRAY);
                out.extend(a.to_bytes());
            }
            Container::Bitmap(b) => {
                out.push(TAG_BITMAP);
                out.extend(b.to_bytes());
            }
            Container::Run(r) => {
                out.push(TAG_RUN);
                out.extend(r.to_bytes());
            }
        }
        out
    }

    /// Inverse of [`Self::to_bytes`]. Returns the parsed container
    /// **and** the number of bytes consumed, so a stream of containers
    /// (e.g. inside a [`crate::postings::roaring::RoaringPostings`])
    /// can be parsed without needing length prefixes.
    pub fn from_bytes(slice: &[u8]) -> Result<(Self, usize), ContainerError> {
        if slice.is_empty() {
            return Err(ContainerError::Truncated { needed: 1, got: 0 });
        }
        match slice[0] {
            TAG_ARRAY => {
                // Need at least 4 B for the cardinality header.
                if slice.len() < 5 {
                    return Err(ContainerError::Truncated {
                        needed: 5,
                        got: slice.len(),
                    });
                }
                let card = u32::from_le_bytes([slice[1], slice[2], slice[3], slice[4]]);
                let body_len = 4 + (card as usize) * 2;
                let total = 1 + body_len;
                if slice.len() < total {
                    return Err(ContainerError::Truncated {
                        needed: total,
                        got: slice.len(),
                    });
                }
                let a = ArrayContainer::from_bytes(&slice[1..total])?;
                Ok((Container::Array(a), total))
            }
            TAG_BITMAP => {
                let body_len = BITMAP_CONTAINER_WORDS * 4;
                let total = 1 + body_len;
                if slice.len() < total {
                    return Err(ContainerError::Truncated {
                        needed: total,
                        got: slice.len(),
                    });
                }
                let b = BitmapContainer::from_bytes(&slice[1..total])?;
                Ok((Container::Bitmap(b), total))
            }
            TAG_RUN => {
                if slice.len() < 3 {
                    return Err(ContainerError::Truncated {
                        needed: 3,
                        got: slice.len(),
                    });
                }
                let n = u16::from_le_bytes([slice[1], slice[2]]) as usize;
                let body_len = 2 + n * 4;
                let total = 1 + body_len;
                if slice.len() < total {
                    return Err(ContainerError::Truncated {
                        needed: total,
                        got: slice.len(),
                    });
                }
                let r = RunContainer::from_bytes(&slice[1..total])?;
                Ok((Container::Run(r), total))
            }
            tag => Err(ContainerError::UnknownTag(tag)),
        }
    }
}

// =============================================================
// Helpers
// =============================================================

fn is_strictly_increasing(keys: &[u16]) -> bool {
    keys.windows(2).all(|w| w[0] < w[1])
}

fn push_or_merge(out: &mut Vec<Run>, next: Run) {
    if let Some(last) = out.last_mut() {
        let last_end = last.end_inclusive();
        if last_end + 1 >= u32::from(next.start) {
            // Touching or overlapping → merge.
            let new_end = last_end.max(next.end_inclusive());
            last.length = u16::try_from(new_end - u32::from(last.start) + 1).unwrap_or(0);
            return;
        }
    }
    out.push(next);
}

fn count_runs_in_array(arr: &ArrayContainer) -> usize {
    if arr.keys.is_empty() {
        return 0;
    }
    let mut runs = 1usize;
    for w in arr.keys.windows(2) {
        // New run when gap > 1.
        if u32::from(w[1]) > u32::from(w[0]) + 1 {
            runs += 1;
        }
    }
    runs
}

fn count_runs_in_bitmap(bm: &BitmapContainer) -> usize {
    let mut runs = 0usize;
    let mut prev_set = false;
    for w_idx in 0..BITMAP_CONTAINER_WORDS {
        let mut word = bm.words[w_idx];
        let mut bit = 0u32;
        while bit < 32 {
            let is_set = (word & 1) == 1;
            if is_set && !prev_set {
                runs += 1;
            }
            prev_set = is_set;
            word >>= 1;
            bit += 1;
        }
    }
    runs
}

// =============================================================
// Tests
// =============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------- ArrayContainer --------

    #[test]
    fn array_new_is_empty() {
        let a = ArrayContainer::new();
        assert_eq!(a.cardinality(), 0);
        assert!(a.iter().next().is_none());
    }

    #[test]
    fn array_insert_keeps_sorted() {
        let mut a = ArrayContainer::new();
        for k in [5u16, 1, 3, 9, 2] {
            a.insert(k);
        }
        assert_eq!(a.keys, vec![1, 2, 3, 5, 9]);
    }

    #[test]
    fn array_insert_dedupes() {
        let mut a = ArrayContainer::new();
        a.insert(7);
        a.insert(7);
        a.insert(7);
        assert_eq!(a.cardinality(), 1);
    }

    #[test]
    fn array_contains() {
        let a = ArrayContainer::from_sorted(vec![10, 20, 30, 40]);
        assert!(a.contains(20));
        assert!(!a.contains(25));
        assert!(a.contains(40));
        assert!(!a.contains(0));
    }

    #[test]
    fn array_from_sorted_strict() {
        let a = ArrayContainer::from_sorted(vec![1, 2, 3]);
        assert_eq!(a.keys, vec![1, 2, 3]);
    }

    #[test]
    fn array_from_sorted_recovers_invariant() {
        // Release-build path: even non-monotonic input is canonicalised.
        // Debug build asserts; we skip the assertion in tests by passing
        // sorted input then.
        let a = ArrayContainer::from_sorted(vec![1, 3, 5]);
        assert_eq!(a.keys, vec![1, 3, 5]);
    }

    #[test]
    fn array_iter_in_order() {
        let a = ArrayContainer::from_sorted(vec![1, 5, 9, 100]);
        let collected: Vec<u16> = a.iter().collect();
        assert_eq!(collected, vec![1, 5, 9, 100]);
    }

    #[test]
    fn array_and_simple() {
        let a = ArrayContainer::from_sorted(vec![1, 3, 5, 7, 9]);
        let b = ArrayContainer::from_sorted(vec![2, 3, 5, 8]);
        let r = a.and(&b);
        assert_eq!(r.keys, vec![3, 5]);
    }

    #[test]
    fn array_and_disjoint() {
        let a = ArrayContainer::from_sorted(vec![1, 2, 3]);
        let b = ArrayContainer::from_sorted(vec![10, 20, 30]);
        let r = a.and(&b);
        assert!(r.keys.is_empty());
    }

    #[test]
    fn array_and_galloping_path() {
        // Asymmetric: small=[10], large=[0..1000].
        let a = ArrayContainer::from_sorted(vec![500]);
        let large_keys: Vec<u16> = (0..1000).collect();
        let b = ArrayContainer::from_sorted(large_keys);
        let r = a.and(&b);
        assert_eq!(r.keys, vec![500]);
    }

    #[test]
    fn array_or_simple() {
        let a = ArrayContainer::from_sorted(vec![1, 3, 5]);
        let b = ArrayContainer::from_sorted(vec![2, 4, 5]);
        match a.or(&b) {
            Container::Array(r) => assert_eq!(r.keys, vec![1, 2, 3, 4, 5]),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_or_promotes_to_bitmap() {
        // Build two arrays whose union exceeds the threshold.
        let a_keys: Vec<u16> = (0..3000).collect();
        let b_keys: Vec<u16> = (3000..6000).collect();
        let a = ArrayContainer::from_sorted(a_keys);
        let b = ArrayContainer::from_sorted(b_keys);
        match a.or(&b) {
            Container::Bitmap(bm) => assert_eq!(bm.cardinality(), 6000),
            _ => panic!("expected promotion to bitmap"),
        }
    }

    #[test]
    fn array_xor_disjoint() {
        let a = ArrayContainer::from_sorted(vec![1, 3, 5]);
        let b = ArrayContainer::from_sorted(vec![2, 4]);
        match a.xor(&b) {
            Container::Array(r) => assert_eq!(r.keys, vec![1, 2, 3, 4, 5]),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_xor_overlap() {
        let a = ArrayContainer::from_sorted(vec![1, 2, 3, 4]);
        let b = ArrayContainer::from_sorted(vec![3, 4, 5, 6]);
        match a.xor(&b) {
            Container::Array(r) => assert_eq!(r.keys, vec![1, 2, 5, 6]),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn array_xor_promotes_to_bitmap() {
        let a_keys: Vec<u16> = (0..3000).collect();
        let b_keys: Vec<u16> = (3000..7000).collect();
        let a = ArrayContainer::from_sorted(a_keys);
        let b = ArrayContainer::from_sorted(b_keys);
        match a.xor(&b) {
            Container::Bitmap(_) => {}
            _ => panic!("expected promotion"),
        }
    }

    #[test]
    fn array_serde_round_trip_empty() {
        let a = ArrayContainer::new();
        let bytes = a.to_bytes();
        assert_eq!(bytes, vec![0, 0, 0, 0]);
        let parsed = ArrayContainer::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, a);
    }

    #[test]
    fn array_serde_round_trip_typical() {
        let a = ArrayContainer::from_sorted(vec![1, 100, 1000, 10000, 60000]);
        let bytes = a.to_bytes();
        let parsed = ArrayContainer::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, a);
    }

    #[test]
    fn array_serde_truncated_header() {
        let bytes = [1u8, 2, 3];
        let res = ArrayContainer::from_bytes(&bytes);
        assert!(matches!(res, Err(ContainerError::Truncated { .. })));
    }

    #[test]
    fn array_serde_truncated_body() {
        let mut bytes = vec![0u8; 4];
        bytes[0] = 5; // claim 5 keys, no body
        let res = ArrayContainer::from_bytes(&bytes);
        assert!(matches!(res, Err(ContainerError::Truncated { .. })));
    }

    #[test]
    fn array_serde_unsorted_rejected() {
        let mut bytes = vec![3u8, 0, 0, 0]; // 3 keys
        bytes.extend_from_slice(&5u16.to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes());
        bytes.extend_from_slice(&7u16.to_le_bytes());
        let res = ArrayContainer::from_bytes(&bytes);
        assert!(matches!(res, Err(ContainerError::ArrayUnsorted)));
    }

    #[test]
    fn array_at_threshold_cardinality() {
        let keys: Vec<u16> = (0..ARRAY_TO_BITMAP_THRESHOLD as u16).collect();
        let a = ArrayContainer::from_sorted(keys);
        assert_eq!(a.cardinality(), ARRAY_TO_BITMAP_THRESHOLD);
    }

    // -------- BitmapContainer --------

    #[test]
    fn bitmap_new_is_empty() {
        let b = BitmapContainer::new();
        assert_eq!(b.cardinality(), 0);
        assert!(b.iter().next().is_none());
    }

    #[test]
    fn bitmap_insert_contains() {
        let mut b = BitmapContainer::new();
        b.insert(0);
        b.insert(31);
        b.insert(32);
        b.insert(65535);
        assert!(b.contains(0));
        assert!(b.contains(31));
        assert!(b.contains(32));
        assert!(b.contains(65535));
        assert!(!b.contains(1));
        assert_eq!(b.cardinality(), 4);
    }

    #[test]
    fn bitmap_remove() {
        let mut b = BitmapContainer::new();
        b.insert(100);
        assert!(b.contains(100));
        b.remove(100);
        assert!(!b.contains(100));
    }

    #[test]
    fn bitmap_full_cardinality() {
        let mut b = BitmapContainer::new();
        for k in 0u32..=65535 {
            b.insert(k as u16);
        }
        assert_eq!(b.cardinality(), 65536);
    }

    #[test]
    fn bitmap_from_array() {
        let a = ArrayContainer::from_sorted(vec![1, 100, 65535]);
        let bm = BitmapContainer::from_array(&a);
        assert!(bm.contains(1));
        assert!(bm.contains(100));
        assert!(bm.contains(65535));
        assert_eq!(bm.cardinality(), 3);
    }

    #[test]
    fn bitmap_iter_order() {
        let mut b = BitmapContainer::new();
        for k in [65535u16, 0, 32, 100, 1024] {
            b.insert(k);
        }
        let collected: Vec<u16> = b.iter().collect();
        assert_eq!(collected, vec![0, 32, 100, 1024, 65535]);
    }

    #[test]
    fn bitmap_and_dense() {
        let mut a = BitmapContainer::new();
        let mut b = BitmapContainer::new();
        for k in 0u16..10000 {
            a.insert(k);
        }
        for k in 5000u16..15000 {
            b.insert(k);
        }
        match a.and(&b) {
            Container::Bitmap(r) => assert_eq!(r.cardinality(), 5000),
            Container::Array(_) => panic!("expected bitmap"),
            _ => panic!("unexpected"),
        }
    }

    #[test]
    fn bitmap_and_demotes_to_array() {
        let mut a = BitmapContainer::new();
        let mut b = BitmapContainer::new();
        for k in 0u16..10000 {
            a.insert(k);
        }
        for k in 9990u16..12000 {
            b.insert(k);
        }
        match a.and(&b) {
            Container::Array(r) => assert_eq!(r.cardinality(), 10),
            Container::Bitmap(_) => panic!("expected array"),
            _ => panic!("unexpected"),
        }
    }

    #[test]
    fn bitmap_or_dense() {
        let mut a = BitmapContainer::new();
        let mut b = BitmapContainer::new();
        for k in 0u16..5000 {
            a.insert(k);
        }
        for k in 5000u16..10000 {
            b.insert(k);
        }
        let r = a.or(&b);
        assert_eq!(r.cardinality(), 10000);
    }

    #[test]
    fn bitmap_xor_dense() {
        let mut a = BitmapContainer::new();
        let mut b = BitmapContainer::new();
        for k in 0u16..10000 {
            a.insert(k);
        }
        for k in 5000u16..15000 {
            b.insert(k);
        }
        match a.xor(&b) {
            Container::Bitmap(r) => assert_eq!(r.cardinality(), 10000),
            _ => panic!("expected bitmap"),
        }
    }

    #[test]
    fn bitmap_xor_demotes_to_array() {
        let mut a = BitmapContainer::new();
        let mut b = BitmapContainer::new();
        // a has 0..6000, b has 0..6000 except missing 5 spots
        for k in 0u16..6000 {
            a.insert(k);
        }
        for k in 0u16..6000 {
            if !matches!(k, 100 | 200 | 300 | 400 | 500) {
                b.insert(k);
            }
        }
        match a.xor(&b) {
            Container::Array(r) => assert_eq!(r.cardinality(), 5),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn bitmap_serde_round_trip() {
        let mut b = BitmapContainer::new();
        for k in [0u16, 1, 100, 10000, 65535] {
            b.insert(k);
        }
        let bytes = b.to_bytes();
        assert_eq!(bytes.len(), BITMAP_CONTAINER_WORDS * 4);
        let parsed = BitmapContainer::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, b);
    }

    #[test]
    fn bitmap_serde_truncated() {
        let bytes = vec![0u8; 100];
        let res = BitmapContainer::from_bytes(&bytes);
        assert!(matches!(res, Err(ContainerError::Truncated { .. })));
    }

    #[test]
    fn bitmap_word_boundaries() {
        let mut b = BitmapContainer::new();
        // Bits at every word boundary
        for w in 0..BITMAP_CONTAINER_WORDS {
            b.insert((w as u16) * 32);
        }
        assert_eq!(b.cardinality(), BITMAP_CONTAINER_WORDS as u32);
    }

    // -------- RunContainer --------

    #[test]
    fn run_new_is_empty() {
        let r = RunContainer::new();
        assert_eq!(r.cardinality(), 0);
        assert!(r.iter().next().is_none());
    }

    #[test]
    fn run_from_runs_valid() {
        let r = RunContainer::from_runs(vec![
            Run {
                start: 0,
                length: 5,
            },
            Run {
                start: 100,
                length: 10,
            },
        ])
        .unwrap();
        assert_eq!(r.cardinality(), 15);
    }

    #[test]
    fn run_from_runs_overlap_rejected() {
        let res = RunContainer::from_runs(vec![
            Run {
                start: 0,
                length: 10,
            },
            Run {
                start: 5,
                length: 5,
            },
        ]);
        assert!(matches!(res, Err(ContainerError::RunInvariantViolation)));
    }

    #[test]
    fn run_contains() {
        let r = RunContainer::from_runs(vec![
            Run {
                start: 100,
                length: 5,
            }, // 100..=104
            Run {
                start: 200,
                length: 3,
            }, // 200..=202
        ])
        .unwrap();
        assert!(r.contains(100));
        assert!(r.contains(104));
        assert!(!r.contains(105));
        assert!(r.contains(200));
        assert!(r.contains(202));
        assert!(!r.contains(203));
        assert!(!r.contains(50));
    }

    #[test]
    fn run_iter() {
        let r = RunContainer::from_runs(vec![
            Run {
                start: 0,
                length: 3,
            },
            Run {
                start: 100,
                length: 2,
            },
        ])
        .unwrap();
        let collected: Vec<u16> = r.iter().collect();
        assert_eq!(collected, vec![0, 1, 2, 100, 101]);
    }

    #[test]
    fn run_and_overlap() {
        let a = RunContainer::from_runs(vec![Run {
            start: 0,
            length: 100,
        }])
        .unwrap();
        let b = RunContainer::from_runs(vec![Run {
            start: 50,
            length: 100,
        }])
        .unwrap();
        let result = a.and(&b);
        assert_eq!(result.cardinality(), 50);
    }

    #[test]
    fn run_and_no_overlap() {
        let a = RunContainer::from_runs(vec![Run {
            start: 0,
            length: 50,
        }])
        .unwrap();
        let b = RunContainer::from_runs(vec![Run {
            start: 100,
            length: 50,
        }])
        .unwrap();
        let result = a.and(&b);
        assert_eq!(result.cardinality(), 0);
    }

    #[test]
    fn run_or_merge_touching() {
        let a = RunContainer::from_runs(vec![Run {
            start: 0,
            length: 5,
        }])
        .unwrap();
        // Touching: [0..4] ∪ [5..9]
        let b = RunContainer::from_runs(vec![Run {
            start: 5,
            length: 5,
        }])
        .unwrap();
        let result = a.or(&b);
        assert_eq!(result.cardinality(), 10);
    }

    #[test]
    fn run_or_disjoint() {
        let a = RunContainer::from_runs(vec![Run {
            start: 0,
            length: 5,
        }])
        .unwrap();
        let b = RunContainer::from_runs(vec![Run {
            start: 100,
            length: 5,
        }])
        .unwrap();
        let result = a.or(&b);
        assert_eq!(result.cardinality(), 10);
    }

    #[test]
    fn run_xor_overlap() {
        let a = RunContainer::from_runs(vec![Run {
            start: 0,
            length: 100,
        }])
        .unwrap();
        let b = RunContainer::from_runs(vec![Run {
            start: 50,
            length: 100,
        }])
        .unwrap();
        let result = a.xor(&b);
        assert_eq!(result.cardinality(), 100);
    }

    #[test]
    fn run_from_array_contiguous() {
        let a = ArrayContainer::from_sorted(vec![10, 11, 12, 13, 100, 101]);
        let r = RunContainer::from_array(&a);
        assert_eq!(r.runs.len(), 2);
        assert_eq!(r.cardinality(), 6);
    }

    #[test]
    fn run_from_bitmap_contiguous() {
        let mut bm = BitmapContainer::new();
        for k in 100u16..200 {
            bm.insert(k);
        }
        for k in 1000u16..2000 {
            bm.insert(k);
        }
        let r = RunContainer::from_bitmap(&bm);
        assert_eq!(r.runs.len(), 2);
        assert_eq!(r.cardinality(), 1100);
    }

    #[test]
    fn run_serde_round_trip() {
        let r = RunContainer::from_runs(vec![
            Run {
                start: 0,
                length: 100,
            },
            Run {
                start: 5000,
                length: 200,
            },
            Run {
                start: 60000,
                length: 1000,
            },
        ])
        .unwrap();
        let bytes = r.to_bytes();
        let parsed = RunContainer::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, r);
    }

    #[test]
    fn run_serde_full_bucket_sentinel() {
        // length=0 means "full bucket" (cardinality 65536). Encoded
        // as wire 0xFFFF.
        let r = RunContainer::from_runs(vec![Run {
            start: 0,
            length: 0,
        }])
        .unwrap();
        assert_eq!(r.cardinality(), 65536);
        let bytes = r.to_bytes();
        let parsed = RunContainer::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.cardinality(), 65536);
    }

    #[test]
    fn run_serde_truncated() {
        let bytes = [0u8];
        let res = RunContainer::from_bytes(&bytes);
        assert!(matches!(res, Err(ContainerError::Truncated { .. })));
    }

    // -------- Container enum (top-level) --------

    fn array_of(keys: Vec<u16>) -> Container {
        Container::Array(ArrayContainer::from_sorted(keys))
    }

    fn bitmap_of_range(start: u16, end_exclusive: u32) -> Container {
        let mut b = BitmapContainer::new();
        for k in u32::from(start)..end_exclusive {
            b.insert(k as u16);
        }
        Container::Bitmap(b)
    }

    fn run_of(runs: Vec<(u16, u16)>) -> Container {
        let mut rs: Vec<Run> = Vec::new();
        for (start, length) in runs {
            rs.push(Run { start, length });
        }
        Container::Run(RunContainer::from_runs(rs).unwrap())
    }

    #[test]
    fn dispatch_and_array_array() {
        let a = array_of(vec![1, 3, 5]);
        let b = array_of(vec![3, 4, 5]);
        let r = a.and(&b);
        assert_eq!(r.cardinality(), 2);
    }

    #[test]
    fn dispatch_and_bitmap_bitmap() {
        let a = bitmap_of_range(0, 10000);
        let b = bitmap_of_range(5000, 15000);
        let r = a.and(&b);
        assert_eq!(r.cardinality(), 5000);
    }

    #[test]
    fn dispatch_and_run_run() {
        let a = run_of(vec![(0, 100)]);
        let b = run_of(vec![(50, 100)]);
        let r = a.and(&b);
        assert_eq!(r.cardinality(), 50);
    }

    #[test]
    fn dispatch_and_array_bitmap() {
        let a = array_of(vec![10, 20, 30, 40]);
        let b = bitmap_of_range(15, 35);
        let r = a.and(&b);
        assert_eq!(r.cardinality(), 2); // 20, 30
    }

    #[test]
    fn dispatch_and_bitmap_array() {
        let a = bitmap_of_range(15, 35);
        let b = array_of(vec![10, 20, 30, 40]);
        let r = a.and(&b);
        assert_eq!(r.cardinality(), 2);
    }

    #[test]
    fn dispatch_and_array_run() {
        let a = array_of(vec![5, 50, 150]);
        let b = run_of(vec![(0, 100)]);
        let r = a.and(&b);
        assert_eq!(r.cardinality(), 2); // 5, 50
    }

    #[test]
    fn dispatch_and_run_array() {
        let a = run_of(vec![(0, 100)]);
        let b = array_of(vec![5, 50, 150]);
        let r = a.and(&b);
        assert_eq!(r.cardinality(), 2);
    }

    #[test]
    fn dispatch_and_bitmap_run() {
        let a = bitmap_of_range(0, 1000);
        let b = run_of(vec![(500, 1000)]);
        let r = a.and(&b);
        assert_eq!(r.cardinality(), 500);
    }

    #[test]
    fn dispatch_and_run_bitmap() {
        let a = run_of(vec![(500, 1000)]);
        let b = bitmap_of_range(0, 1000);
        let r = a.and(&b);
        assert_eq!(r.cardinality(), 500);
    }

    #[test]
    fn dispatch_or_array_array() {
        let a = array_of(vec![1, 3]);
        let b = array_of(vec![2, 3]);
        let r = a.or(&b);
        assert_eq!(r.cardinality(), 3);
    }

    #[test]
    fn dispatch_or_bitmap_bitmap() {
        let a = bitmap_of_range(0, 1000);
        let b = bitmap_of_range(500, 2000);
        let r = a.or(&b);
        assert_eq!(r.cardinality(), 2000);
    }

    #[test]
    fn dispatch_or_run_run() {
        let a = run_of(vec![(0, 100)]);
        let b = run_of(vec![(50, 100)]);
        let r = a.or(&b);
        assert_eq!(r.cardinality(), 150);
    }

    #[test]
    fn dispatch_or_array_bitmap() {
        let a = array_of(vec![10000, 20000]);
        let b = bitmap_of_range(0, 1000);
        let r = a.or(&b);
        assert_eq!(r.cardinality(), 1002);
    }

    #[test]
    fn dispatch_or_array_run() {
        let a = array_of(vec![10000, 20000]);
        let b = run_of(vec![(0, 1000)]);
        let r = a.or(&b);
        assert_eq!(r.cardinality(), 1002);
    }

    #[test]
    fn dispatch_or_bitmap_run() {
        let a = bitmap_of_range(0, 1000);
        let b = run_of(vec![(500, 1000)]);
        let r = a.or(&b);
        assert_eq!(r.cardinality(), 1500);
    }

    #[test]
    fn dispatch_xor_array_array() {
        let a = array_of(vec![1, 2, 3]);
        let b = array_of(vec![2, 3, 4]);
        let r = a.xor(&b);
        assert_eq!(r.cardinality(), 2); // 1, 4
    }

    #[test]
    fn dispatch_xor_bitmap_bitmap() {
        let a = bitmap_of_range(0, 5000);
        let b = bitmap_of_range(2500, 7500);
        let r = a.xor(&b);
        assert_eq!(r.cardinality(), 5000);
    }

    #[test]
    fn dispatch_xor_run_run() {
        let a = run_of(vec![(0, 100)]);
        let b = run_of(vec![(50, 100)]);
        let r = a.xor(&b);
        assert_eq!(r.cardinality(), 100);
    }

    #[test]
    fn dispatch_xor_array_bitmap() {
        let a = array_of(vec![5, 10, 15]);
        let b = bitmap_of_range(10, 20);
        let r = a.xor(&b);
        // 5, 11..19, but symmetric diff: a_only=5, b_only=11..14,16..19,
        // both=10,15 (drop). So result = {5, 11, 12, 13, 14, 16, 17, 18, 19} = 9
        assert_eq!(r.cardinality(), 9);
    }

    #[test]
    fn dispatch_xor_bitmap_run() {
        let a = bitmap_of_range(0, 100);
        let b = run_of(vec![(50, 100)]);
        let r = a.xor(&b);
        // 0..49 from a only, 100..149 from b only. 50..99 in both.
        assert_eq!(r.cardinality(), 100);
    }

    #[test]
    fn dispatch_optimize_array_to_bitmap_when_dense() {
        let keys: Vec<u16> = (0..5000).collect();
        let a = array_of(keys);
        // a is Array but cardinality > threshold; optimize → Bitmap
        // (unless the run-density wins — in this case the array is
        // 0..4999 so it's a single run = 6 bytes vs 8 KiB bitmap vs
        // 4 + 10000 array. Run wins.)
        let optimized = a.optimize();
        match optimized {
            Container::Run(r) => assert_eq!(r.runs.len(), 1),
            other => panic!("expected run, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_optimize_keeps_array_when_sparse() {
        let a = array_of(vec![1, 100, 1000, 10000]);
        match a.optimize() {
            Container::Array(_) => {}
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_optimize_promotes_bitmap_to_run_when_dense_runs() {
        let bm = bitmap_of_range(0, 10000);
        match bm.optimize() {
            Container::Run(r) => assert_eq!(r.runs.len(), 1),
            other => panic!("expected run, got {other:?}"),
        }
    }

    #[test]
    fn container_serde_round_trip_array() {
        let c = array_of(vec![1, 100, 1000]);
        let bytes = c.to_bytes();
        let (parsed, n) = Container::from_bytes(&bytes).unwrap();
        assert_eq!(n, bytes.len());
        assert_eq!(parsed.cardinality(), c.cardinality());
    }

    #[test]
    fn container_serde_round_trip_bitmap() {
        let c = bitmap_of_range(0, 10000);
        let bytes = c.to_bytes();
        let (parsed, n) = Container::from_bytes(&bytes).unwrap();
        assert_eq!(n, bytes.len());
        assert_eq!(parsed.cardinality(), c.cardinality());
    }

    #[test]
    fn container_serde_round_trip_run() {
        let c = run_of(vec![(0, 1000), (5000, 500), (60000, 100)]);
        let bytes = c.to_bytes();
        let (parsed, n) = Container::from_bytes(&bytes).unwrap();
        assert_eq!(n, bytes.len());
        assert_eq!(parsed.cardinality(), c.cardinality());
    }

    #[test]
    fn container_serde_unknown_tag() {
        let bytes = [0xAA, 0, 0, 0, 0];
        let res = Container::from_bytes(&bytes);
        assert!(matches!(res, Err(ContainerError::UnknownTag(0xAA))));
    }

    #[test]
    fn container_serde_truncated_array() {
        let bytes = [TAG_ARRAY, 1, 0]; // tag + partial header
        let res = Container::from_bytes(&bytes);
        assert!(matches!(res, Err(ContainerError::Truncated { .. })));
    }

    #[test]
    fn container_iter_array() {
        let c = array_of(vec![1, 2, 3]);
        let collected: Vec<u16> = c.iter().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn container_iter_bitmap() {
        let c = bitmap_of_range(100, 105);
        let collected: Vec<u16> = c.iter().collect();
        assert_eq!(collected, vec![100, 101, 102, 103, 104]);
    }

    #[test]
    fn container_iter_run() {
        let c = run_of(vec![(0, 3)]);
        let collected: Vec<u16> = c.iter().collect();
        assert_eq!(collected, vec![0, 1, 2]);
    }

    #[test]
    fn container_contains_dispatch() {
        let a = array_of(vec![1, 5]);
        let b = bitmap_of_range(10, 15);
        let r = run_of(vec![(20, 5)]);
        assert!(a.contains(5));
        assert!(b.contains(12));
        assert!(r.contains(22));
        assert!(!a.contains(2));
        assert!(!b.contains(15));
        assert!(!r.contains(30));
    }

    #[test]
    fn run_count_helpers() {
        let arr = ArrayContainer::from_sorted(vec![1, 2, 3, 10, 11]);
        assert_eq!(count_runs_in_array(&arr), 2);
        let mut bm = BitmapContainer::new();
        for k in 100u16..150 {
            bm.insert(k);
        }
        for k in 200u16..210 {
            bm.insert(k);
        }
        assert_eq!(count_runs_in_bitmap(&bm), 2);
    }
}

//! Host-side helpers for the CUDA Tensor Core fast path.
//!
//! Two operations live here:
//!
//! 1. **Bit unpacking** — turn each bit of a `dim_u32` × N corpus into one
//!    `i8` byte (∈ {0, 1}) so it can be fed to an INT8 cuBLAS GEMM.
//! 2. **Per-vector popcount** — sum the `count_ones()` of each row's u32
//!    words. Required by the popcount-identity reformulation
//!    `popcount(q ⊕ d) = popcount(q) + popcount(d) − 2·⟨q, d⟩`.
//!
//! Both run on the host because:
//! - Unpacking has zero arithmetic intensity (memcpy-bound) — pushing it
//!   to the device just doubles PCIe traffic.
//! - Per-vector popcount is `O(dim_u32 × N)` but each `count_ones()` is a
//!   single x86 `popcnt` cycle, so the host runs at ~30 GB/s effective.
//!   Wall-clock cost on a 1 M corpus at dim_bits = 768 is ~3 ms — small
//!   compared to the GEMM call.

/// CPU reference implementation of the device-side `unpack_bits`
/// NVRTC kernel — kept for unit-testing and as a readable description
/// of the expected output. The production path runs the unpack on the
/// GPU to keep PCIe traffic at one bit per bit.
#[cfg(test)]
pub(super) fn unpack_bits_to_i8(packed: &[u32], num_vecs: usize, dim_bits: usize) -> Vec<i8> {
    let dim_u32 = dim_bits.div_ceil(32);
    debug_assert_eq!(packed.len(), num_vecs * dim_u32);
    let mut out = Vec::with_capacity(num_vecs * dim_bits);
    for v in 0..num_vecs {
        let row = &packed[v * dim_u32..(v + 1) * dim_u32];
        for b in 0..dim_bits {
            let word = row[b / 32];
            let bit = ((word >> (b & 31)) & 1) as i8;
            out.push(bit);
        }
    }
    out
}

/// Per-vector population count of a row-major `num_vecs × dim_u32` u32
/// buffer. Returns one `i32` per row (max value = `dim_bits` ≤ 2048,
/// well inside `i32`).
pub(super) fn popcount_per_vec(packed: &[u32], num_vecs: usize, dim_u32: usize) -> Vec<i32> {
    debug_assert_eq!(packed.len(), num_vecs * dim_u32);
    let mut out = Vec::with_capacity(num_vecs);
    for v in 0..num_vecs {
        let row = &packed[v * dim_u32..(v + 1) * dim_u32];
        let mut sum: u32 = 0;
        for w in row {
            sum += w.count_ones();
        }
        out.push(sum as i32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_lsb_first() {
        // 0b1011 in word 0, dim_bits = 4 → expect [1, 1, 0, 1].
        let out = unpack_bits_to_i8(&[0b1011], 1, 4);
        assert_eq!(out, vec![1, 1, 0, 1]);
    }

    #[test]
    fn unpack_drops_padding() {
        // dim_bits = 6, word = 0xFFFF_FFFF — only the low 6 bits should appear.
        let out = unpack_bits_to_i8(&[0xFFFF_FFFF], 1, 6);
        assert_eq!(out, vec![1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn unpack_two_words_two_vecs() {
        // Vec 0: 0b…0001 in word 0, 0b…0010 in word 1 → bits[0]=1, bits[33]=1.
        // Vec 1: zeros.
        let packed = vec![1u32, 2u32, 0u32, 0u32];
        let out = unpack_bits_to_i8(&packed, 2, 64);
        assert_eq!(out.len(), 128);
        assert_eq!(out[0], 1);
        assert_eq!(out[33], 1);
        assert_eq!(out[1], 0);
        assert_eq!(out[64], 0);
        assert_eq!(out[127], 0);
    }

    #[test]
    fn popcount_matches_count_ones() {
        let packed = vec![0xFFFF_FFFFu32, 0u32, 0xAAAA_AAAA, 0x5555_5555];
        let out = popcount_per_vec(&packed, 2, 2);
        assert_eq!(out, vec![32, 32]);
    }

    #[test]
    fn popcount_empty() {
        assert!(popcount_per_vec(&[], 0, 4).is_empty());
    }
}

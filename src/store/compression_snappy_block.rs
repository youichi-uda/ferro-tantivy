//! Snappy block-format docstore compression.
//!
//! Phase 1 of the ferro-compress integration introduced Snappy as the Hot
//! tier default — Phase 0 nvCOMP measurements settled it, with 105-191 GB/s
//! GPU decompress and 1.6-7.4 GB/s CPU decompress on real workloads. Adding
//! Snappy at the Tantivy level lets the docstore itself cooperate with
//! ferro-storage's `Tier::Hot` codec choice without going through a
//! second compression layer.
//!
//! The on-disk layout mirrors the LZ4 module: a 4-byte little-endian
//! `u32` of the uncompressed length, followed by a raw Snappy block. We
//! don't use Snappy's framing format (which adds checksums, stream
//! markers) because Tantivy's docstore already block-aligns the writes
//! and verifies block boundaries via the outer footer.

use std::{io, mem};

use snap::raw::{decompress_len, max_compress_len, Decoder, Encoder};

#[inline]
#[expect(clippy::uninit_vec)]
pub fn compress(uncompressed: &[u8], compressed: &mut Vec<u8>) -> io::Result<()> {
    compressed.clear();
    let maximum_output_size = mem::size_of::<u32>() + max_compress_len(uncompressed.len());
    compressed.reserve(maximum_output_size);
    unsafe {
        compressed.set_len(maximum_output_size);
    }
    let mut encoder = Encoder::new();
    let bytes_written = encoder
        .compress(uncompressed, &mut compressed[4..])
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
    let num_bytes = uncompressed.len() as u32;
    compressed[0..4].copy_from_slice(&num_bytes.to_le_bytes());
    unsafe {
        compressed.set_len(bytes_written + mem::size_of::<u32>());
    }
    Ok(())
}

#[inline]
#[expect(clippy::uninit_vec)]
pub fn decompress(compressed: &[u8], decompressed: &mut Vec<u8>) -> io::Result<()> {
    decompressed.clear();
    let uncompressed_size_bytes: &[u8; 4] = compressed
        .get(..4)
        .ok_or(io::ErrorKind::InvalidData)?
        .try_into()
        .unwrap();
    let uncompressed_size = u32::from_le_bytes(*uncompressed_size_bytes) as usize;

    // Snappy embeds the uncompressed length in its own header too. Cross-
    // check the outer u32 against the inner Snappy length so a torn block
    // is caught before it produces silent zero-padding.
    let snappy_inner_len = decompress_len(&compressed[4..])
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
    if snappy_inner_len != uncompressed_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "doc store snappy length mismatch: outer u32 says {uncompressed_size}, snappy \
                 inner header says {snappy_inner_len}"
            ),
        ));
    }
    decompressed.reserve(uncompressed_size);
    unsafe {
        decompressed.set_len(uncompressed_size);
    }
    let mut decoder = Decoder::new();
    let bytes_written = decoder
        .decompress(&compressed[4..], decompressed)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
    if bytes_written != uncompressed_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "doc store block not completely decompressed, data corruption".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        let mut compressed = Vec::new();
        let mut decompressed = Vec::new();
        compress(&[], &mut compressed).unwrap();
        decompress(&compressed, &mut decompressed).unwrap();
        assert_eq!(decompressed, b"");
    }

    #[test]
    fn roundtrip_ascii() {
        let payload = b"the quick brown fox jumps over the lazy dog\n".repeat(64);
        let mut compressed = Vec::new();
        let mut decompressed = Vec::new();
        compress(&payload, &mut compressed).unwrap();
        decompress(&compressed, &mut decompressed).unwrap();
        assert_eq!(decompressed, payload);
    }

    #[test]
    fn roundtrip_binary_4kib() {
        let payload: Vec<u8> = (0..4096u32).map(|i| (i * 31) as u8).collect();
        let mut compressed = Vec::new();
        let mut decompressed = Vec::new();
        compress(&payload, &mut compressed).unwrap();
        decompress(&compressed, &mut decompressed).unwrap();
        assert_eq!(decompressed, payload);
    }
}

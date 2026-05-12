use std::io::Write;
use std::sync::Arc;
use std::{io, iter};

use common::{BinarySerializable, CountingWriter, DeserializeFrom, OwnedBytes};
use fastdivide::DividerU64;
use tantivy_bitpacker::{BitPacker, BitUnpacker, compute_num_bits};

use crate::MonotonicallyMappableToU64;
use crate::column_values::u64_based::line::Line;
use crate::column_values::u64_based::{ColumnCodec, ColumnCodecEstimator, ColumnStats};
use crate::column_values::{ColumnValues, VecColumn};

const BLOCK_SIZE: u32 = 512u32;

#[derive(Debug, Default)]
struct Block {
    line: Line,
    bit_unpacker: BitUnpacker,
    data_start_offset: usize,
}

impl BinarySerializable for Block {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> io::Result<()> {
        self.line.serialize(writer)?;
        self.bit_unpacker.bit_width().serialize(writer)?;
        Ok(())
    }

    fn deserialize<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let line = Line::deserialize(reader)?;
        let bit_width = u8::deserialize(reader)?;
        Ok(Block {
            line,
            bit_unpacker: BitUnpacker::new(bit_width),
            data_start_offset: 0,
        })
    }
}

fn compute_num_blocks(num_vals: u32) -> u32 {
    num_vals.div_ceil(BLOCK_SIZE)
}

pub struct BlockwiseLinearEstimator {
    block: Vec<u64>,
    values_num_bytes: u64,
    meta_num_bytes: u64,
}

impl Default for BlockwiseLinearEstimator {
    fn default() -> Self {
        Self {
            block: Vec::with_capacity(BLOCK_SIZE as usize),
            values_num_bytes: 0u64,
            meta_num_bytes: 0u64,
        }
    }
}

impl BlockwiseLinearEstimator {
    fn flush_block_estimate(&mut self) {
        if self.block.is_empty() {
            return;
        }
        let column = VecColumn::from(std::mem::take(&mut self.block));
        let line = Line::train(&column);
        self.block = column.into();

        let mut max_value = 0u64;
        for (i, buffer_val) in self.block.iter().enumerate() {
            let interpolated_val = line.eval(i as u32);
            let val = buffer_val.wrapping_sub(interpolated_val);
            max_value = val.max(max_value);
        }
        let bit_width = compute_num_bits(max_value) as usize;
        self.values_num_bytes += (bit_width * self.block.len() + 7) as u64 / 8;
        self.meta_num_bytes += 1 + line.num_bytes();
    }
}

impl ColumnCodecEstimator for BlockwiseLinearEstimator {
    fn collect(&mut self, value: u64) {
        self.block.push(value);
        if self.block.len() == BLOCK_SIZE as usize {
            self.flush_block_estimate();
            self.block.clear();
        }
    }
    fn estimate(&self, stats: &ColumnStats) -> Option<u64> {
        let mut estimate = 4 + stats.num_bytes() + self.meta_num_bytes + self.values_num_bytes;
        if stats.gcd.get() > 1 {
            let estimate_gain_from_gcd =
                (stats.gcd.get() as f32).log2().floor() * stats.num_rows as f32 / 8.0f32;
            estimate = estimate.saturating_sub(estimate_gain_from_gcd as u64);
        }
        Some(estimate)
    }

    fn finalize(&mut self) {
        self.flush_block_estimate();
    }

    fn serialize(
        &self,
        stats: &ColumnStats,
        mut vals: &mut dyn Iterator<Item = u64>,
        wrt: &mut dyn Write,
    ) -> io::Result<()> {
        stats.serialize(wrt)?;
        let mut buffer = Vec::with_capacity(BLOCK_SIZE as usize);
        let num_blocks = compute_num_blocks(stats.num_rows) as usize;
        let mut blocks = Vec::with_capacity(num_blocks);

        let mut bit_packer = BitPacker::new();

        let gcd_divider = DividerU64::divide_by(stats.gcd.get());

        for _ in 0..num_blocks {
            buffer.clear();
            buffer.extend(
                (&mut vals)
                    .map(MonotonicallyMappableToU64::to_u64)
                    .take(BLOCK_SIZE as usize),
            );

            for buffer_val in buffer.iter_mut() {
                *buffer_val = gcd_divider.divide(*buffer_val - stats.min_value);
            }

            let line = Line::train(&VecColumn::from(buffer.to_vec()));

            assert!(!buffer.is_empty());

            for (i, buffer_val) in buffer.iter_mut().enumerate() {
                let interpolated_val = line.eval(i as u32);
                *buffer_val = buffer_val.wrapping_sub(interpolated_val);
            }

            let bit_width = buffer.iter().copied().map(compute_num_bits).max().unwrap();

            for &buffer_val in &buffer {
                bit_packer.write(buffer_val, bit_width, wrt)?;
            }

            blocks.push(Block {
                line,
                bit_unpacker: BitUnpacker::new(bit_width),
                data_start_offset: 0,
            });
        }

        bit_packer.close(wrt)?;

        assert_eq!(blocks.len(), num_blocks);

        let mut counting_wrt = CountingWriter::wrap(wrt);
        for block in &blocks {
            block.serialize(&mut counting_wrt)?;
        }
        let footer_len = counting_wrt.written_bytes();
        (footer_len as u32).serialize(&mut counting_wrt)?;

        Ok(())
    }
}

pub struct BlockwiseLinearCodec;

impl ColumnCodec<u64> for BlockwiseLinearCodec {
    type ColumnValues = BlockwiseLinearReader;

    type Estimator = BlockwiseLinearEstimator;

    fn load(mut bytes: OwnedBytes) -> io::Result<Self::ColumnValues> {
        let stats = ColumnStats::deserialize(&mut bytes)?;
        let footer_len: u32 = (&bytes[bytes.len() - 4..]).deserialize()?;
        let footer_offset = bytes.len() - 4 - footer_len as usize;
        let (data, mut footer) = bytes.split(footer_offset);
        let num_blocks = compute_num_blocks(stats.num_rows);
        let mut blocks: Vec<Block> = iter::repeat_with(|| Block::deserialize(&mut footer))
            .take(num_blocks as usize)
            .collect::<io::Result<_>>()?;
        let mut start_offset = 0;
        for block in &mut blocks {
            block.data_start_offset = start_offset;
            start_offset += (block.bit_unpacker.bit_width() as usize) * BLOCK_SIZE as usize / 8;
        }
        Ok(BlockwiseLinearReader {
            blocks: blocks.into_boxed_slice().into(),
            data,
            stats,
        })
    }
}

#[derive(Clone)]
pub struct BlockwiseLinearReader {
    blocks: Arc<[Block]>,
    data: OwnedBytes,
    stats: ColumnStats,
}

impl ColumnValues for BlockwiseLinearReader {
    #[inline(always)]
    fn get_val(&self, idx: u32) -> u64 {
        let block_id = (idx / BLOCK_SIZE) as usize;
        let idx_within_block = idx % BLOCK_SIZE;
        let block = &self.blocks[block_id];
        let interpoled_val: u64 = block.line.eval(idx_within_block);
        let block_bytes = &self.data[block.data_start_offset..];
        let bitpacked_diff = block.bit_unpacker.get(idx_within_block, block_bytes);
        // TODO optimize me! the line parameters could be tweaked to include the multiplication and
        // remove the dependency.
        self.stats.min_value
            + self
                .stats
                .gcd
                .get()
                .wrapping_mul(interpoled_val.wrapping_add(bitpacked_diff))
    }

    fn get_range(&self, start: u64, output: &mut [u64]) {
        if output.is_empty() {
            return;
        }
        let start_u32 = start as u32;
        let end_u32 = start_u32 + output.len() as u32;
        let min_value = self.stats.min_value;
        let gcd = self.stats.gcd.get();
        let mut idx = start_u32;
        let mut out_cursor: usize = 0;
        let mut residual_buf: Vec<u32> = Vec::new();
        while idx < end_u32 {
            let block_id = (idx / BLOCK_SIZE) as usize;
            let block = &self.blocks[block_id];
            let block_start = (block_id as u32) * BLOCK_SIZE;
            let segment_end = (block_start + BLOCK_SIZE).min(end_u32);
            let segment_len = (segment_end - idx) as usize;
            let block_bytes = &self.data[block.data_start_offset..];
            let line = block.line;
            let bit_width = block.bit_unpacker.bit_width();
            let idx_within_block = idx - block_start;
            let out_slice = &mut output[out_cursor..out_cursor + segment_len];
            if bit_width == 0 {
                for (i, out) in out_slice.iter_mut().enumerate() {
                    let pos = idx_within_block + i as u32;
                    let interp = line.eval(pos);
                    *out = min_value.wrapping_add(gcd.wrapping_mul(interp));
                }
            } else if bit_width <= 32 {
                residual_buf.clear();
                residual_buf.resize(segment_len, 0u32);
                block.bit_unpacker.get_batch_u32s(
                    idx_within_block,
                    block_bytes,
                    &mut residual_buf,
                );
                for (i, (out, &res)) in
                    out_slice.iter_mut().zip(residual_buf.iter()).enumerate()
                {
                    let pos = idx_within_block + i as u32;
                    let interp = line.eval(pos);
                    *out = min_value.wrapping_add(
                        gcd.wrapping_mul(interp.wrapping_add(res as u64)),
                    );
                }
            } else {
                for (i, out) in out_slice.iter_mut().enumerate() {
                    let pos = idx_within_block + i as u32;
                    let interp = line.eval(pos);
                    let diff = block.bit_unpacker.get(pos, block_bytes);
                    *out = min_value.wrapping_add(
                        gcd.wrapping_mul(interp.wrapping_add(diff)),
                    );
                }
            }
            out_cursor += segment_len;
            idx = segment_end;
        }
    }

    #[inline(always)]
    fn min_value(&self) -> u64 {
        self.stats.min_value
    }

    #[inline(always)]
    fn max_value(&self) -> u64 {
        self.stats.max_value
    }

    #[inline(always)]
    fn num_vals(&self) -> u32 {
        self.stats.num_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_values::u64_based::tests::create_and_validate;

    #[test]
    fn test_with_codec_data_sets_simple() {
        create_and_validate::<BlockwiseLinearCodec>(
            &[11, 20, 40, 20, 10, 10, 10, 10, 10, 10],
            "simple test",
        )
        .unwrap();
    }

    #[test]
    fn test_with_codec_data_sets_simple_gcd() {
        let (_, actual_compression_rate) = create_and_validate::<BlockwiseLinearCodec>(
            &[10, 20, 40, 20, 10, 10, 10, 10, 10, 10],
            "name",
        )
        .unwrap();
        assert_eq!(actual_compression_rate, 0.175);
    }

    #[test]
    fn test_with_codec_data_sets() {
        let data_sets = crate::column_values::u64_based::tests::get_codec_test_datasets();
        for (mut data, name) in data_sets {
            create_and_validate::<BlockwiseLinearCodec>(&data, name);
            data.reverse();
            create_and_validate::<BlockwiseLinearCodec>(&data, name);
        }
    }

    #[test]
    fn test_blockwise_linear_fast_field_rand() {
        for _ in 0..500 {
            let mut data = (0..1 + rand::random::<u8>() as usize)
                .map(|_| rand::random::<i64>() as u64 / 2)
                .collect::<Vec<_>>();
            create_and_validate::<BlockwiseLinearCodec>(&data, "rand");
            data.reverse();
            create_and_validate::<BlockwiseLinearCodec>(&data, "rand");
        }
    }

    fn load_blockwise_linear_reader(values: &[u64]) -> BlockwiseLinearReader {
        use crate::column_values::u64_based::stats_collector::StatsCollector;
        let mut stats_collector = StatsCollector::default();
        let mut codec_estimator = BlockwiseLinearEstimator::default();
        for &v in values {
            stats_collector.collect(v);
            codec_estimator.collect(v);
        }
        codec_estimator.finalize();
        let stats = stats_collector.stats();
        let mut buffer = Vec::new();
        codec_estimator
            .serialize(&stats, &mut values.iter().copied(), &mut buffer)
            .unwrap();
        BlockwiseLinearCodec::load(OwnedBytes::new(buffer)).unwrap()
    }

    #[test]
    fn test_blockwise_linear_get_range_matches_get_val_multi_block() {
        // Force multiple blocks: 1500 values spans 3 blocks (each BLOCK_SIZE=512).
        let mut rng = rand::random::<u64>();
        let values: Vec<u64> = (0..1500u64)
            .map(|i| {
                // mix of linear trend + small residual so bit_width stays small (≤32 path)
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let residual = (rng % 4096) as u64;
                1_000_000_000u64.wrapping_add(i.wrapping_mul(17)).wrapping_add(residual)
            })
            .collect();
        let reader = load_blockwise_linear_reader(&values);

        let scalar: Vec<u64> = (0..1500u32).map(|i| reader.get_val(i)).collect();

        // Full range
        let mut batch = vec![0u64; 1500];
        reader.get_range(0, &mut batch);
        assert_eq!(scalar, batch, "full-range mismatch");

        // Cross-block range
        let mut batch = vec![0u64; 600];
        reader.get_range(400, &mut batch); // 400..1000 spans blocks 0+1
        assert_eq!(&scalar[400..1000], &batch[..], "cross-block 0->1 mismatch");

        // Within last partial block
        let mut batch = vec![0u64; 100];
        reader.get_range(1024, &mut batch); // 1024..1124, fully in block 2
        assert_eq!(&scalar[1024..1124], &batch[..], "last-block mismatch");

        // Single value via get_range
        let mut batch = vec![0u64; 1];
        reader.get_range(777, &mut batch);
        assert_eq!(scalar[777], batch[0], "single-value get_range mismatch");

        // Empty
        let mut batch: Vec<u64> = Vec::new();
        reader.get_range(0, &mut batch);
    }

    #[test]
    fn test_blockwise_linear_get_range_bit_width_zero() {
        // All-identical block → bit_width 0 path.
        let values: Vec<u64> = vec![42u64; 1500];
        let reader = load_blockwise_linear_reader(&values);

        let mut batch = vec![0u64; 1500];
        reader.get_range(0, &mut batch);
        for &v in &batch {
            assert_eq!(v, 42, "constant-block path should yield input");
        }
    }

    /// Wave 21 Phase 1 micro-bench: compare scalar `get_val` loop vs block-aware
    /// `get_range` on a 1 M-doc BlockwiseLinear column. Marked `#[ignore]`
    /// because it is a perf signal, not a correctness assertion — run with
    /// `cargo test --release blockwise_linear_get_range_bench -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn blockwise_linear_get_range_bench() {
        use std::time::Instant;
        const N: u64 = 1_000_000;
        let mut rng = 0x9E3779B97F4A7C15u64;
        let values: Vec<u64> = (0..N)
            .map(|i| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let residual = rng % 4096;
                1_000_000_000u64.wrapping_add(i.wrapping_mul(17)).wrapping_add(residual)
            })
            .collect();
        let reader = load_blockwise_linear_reader(&values);
        let warmups = 3usize;
        let trials = 5usize;
        for _ in 0..warmups {
            let mut sum: u64 = 0;
            for i in 0..N as u32 {
                sum = sum.wrapping_add(reader.get_val(i));
            }
            std::hint::black_box(sum);
        }
        let mut scalar_ns: u128 = u128::MAX;
        for _ in 0..trials {
            let t0 = Instant::now();
            let mut sum: u64 = 0;
            for i in 0..N as u32 {
                sum = sum.wrapping_add(reader.get_val(i));
            }
            std::hint::black_box(sum);
            scalar_ns = scalar_ns.min(t0.elapsed().as_nanos());
        }
        let mut buf: Vec<u64> = vec![0u64; N as usize];
        for _ in 0..warmups {
            reader.get_range(0, &mut buf);
            std::hint::black_box(&buf);
        }
        let mut range_ns: u128 = u128::MAX;
        for _ in 0..trials {
            let t0 = Instant::now();
            reader.get_range(0, &mut buf);
            std::hint::black_box(&buf);
            range_ns = range_ns.min(t0.elapsed().as_nanos());
        }
        let speedup = scalar_ns as f64 / range_ns as f64;
        println!(
            "BlockwiseLinear get_val-loop = {} ns, get_range = {} ns, speedup = {:.2}x",
            scalar_ns, range_ns, speedup
        );
        // Sanity: get_range must produce same output as scalar loop.
        for i in 0..N as u32 {
            assert_eq!(
                buf[i as usize],
                reader.get_val(i),
                "get_range mismatch at idx {i}"
            );
        }
    }

    #[test]
    fn test_blockwise_linear_get_range_wide_bit_width() {
        // Force bit_width > 32 by using values with wide random residuals.
        let mut rng = rand::random::<u64>();
        let values: Vec<u64> = (0..1500u64)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                rng
            })
            .collect();
        let reader = load_blockwise_linear_reader(&values);

        let scalar: Vec<u64> = (0..1500u32).map(|i| reader.get_val(i)).collect();
        let mut batch = vec![0u64; 1500];
        reader.get_range(0, &mut batch);
        assert_eq!(scalar, batch, "wide-bit-width scalar fallback must match");
    }
}

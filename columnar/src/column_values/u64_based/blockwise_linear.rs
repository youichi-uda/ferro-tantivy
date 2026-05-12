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

/// True when `indexes` is strictly +1 monotonic. O(N), short-circuits on
/// first non-sequential pair. Cheap enough to gate the SIMD get_range path
/// in get_vals.
#[inline]
fn is_sequential(indexes: &[u32]) -> bool {
    let mut iter = indexes.iter().copied();
    let Some(mut prev) = iter.next() else {
        return true;
    };
    for cur in iter {
        if cur != prev + 1 {
            return false;
        }
        prev = cur;
    }
    true
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

    /// Wave 21 Phase 2: block-aware batch lookup for arbitrary indexes.
    ///
    /// - Sequential indexes → delegate to get_range (full SIMD path).
    /// - Few indexes (≤ `RUN_THRESHOLD`) → walk per-block runs so block
    ///   state (line, bit_unpacker, block bytes) is loaded once per run.
    ///   This is the harvest path: per segment we have size=10-ish docs
    ///   that often cluster in a handful of blocks.
    /// - Many random indexes spread across blocks → fall back to the
    ///   default scalar 4-wide unroll. Per-block-run overhead doesn't pay
    ///   when avg run length is ~1 (synthetic 10K random over 1M-row
    ///   column measured 0.72× regression before adding this guard).
    ///
    /// The default ColumnValues::get_vals impl is a scalar 4-wide unroll
    /// over get_val, which re-derefs Arc<[Block]> + self.data per call —
    /// measurably the dominant cost on full-corpus sort harvest
    /// (Wave 20.1 profile).
    fn get_vals(&self, indexes: &[u32], output: &mut [u64]) {
        assert_eq!(indexes.len(), output.len());
        if output.is_empty() {
            return;
        }
        if indexes.len() > 1 && is_sequential(indexes) {
            self.get_range(indexes[0] as u64, output);
            return;
        }
        // Heuristic: only run the per-block-run path when the input is
        // small enough that the typical harvest pattern (few near-neighbour
        // doc_ids per segment) dominates. Larger random workloads use the
        // scalar unroll which the compiler keeps very tight.
        const RUN_THRESHOLD: usize = 64;
        if indexes.len() > RUN_THRESHOLD {
            let chunks = output.chunks_exact_mut(4).zip(indexes.chunks_exact(4));
            for (out_x4, idx_x4) in chunks {
                out_x4[0] = self.get_val(idx_x4[0]);
                out_x4[1] = self.get_val(idx_x4[1]);
                out_x4[2] = self.get_val(idx_x4[2]);
                out_x4[3] = self.get_val(idx_x4[3]);
            }
            let remainder = output
                .chunks_exact_mut(4)
                .into_remainder()
                .iter_mut()
                .zip(indexes.chunks_exact(4).remainder());
            for (out, idx) in remainder {
                *out = self.get_val(*idx);
            }
            return;
        }
        let min_value = self.stats.min_value;
        let gcd = self.stats.gcd.get();
        let mut i = 0usize;
        while i < indexes.len() {
            let block_id = (indexes[i] / BLOCK_SIZE) as usize;
            let block = &self.blocks[block_id];
            let block_bytes = &self.data[block.data_start_offset..];
            let line = block.line;
            let block_min = (block_id as u32) * BLOCK_SIZE;
            let block_max_excl = block_min + BLOCK_SIZE;
            // Process a run of indexes that fall in this block.
            let run_start = i;
            while i < indexes.len()
                && indexes[i] >= block_min
                && indexes[i] < block_max_excl
            {
                i += 1;
            }
            // The run is indexes[run_start..i]. Always non-empty (at least
            // the first element by construction).
            for j in run_start..i {
                let pos = indexes[j] - block_min;
                let interp = line.eval(pos);
                let diff = block.bit_unpacker.get(pos, block_bytes);
                output[j] = min_value
                    .wrapping_add(gcd.wrapping_mul(interp.wrapping_add(diff)));
            }
        }
    }

    /// Default impl loops scalar get_val; route through get_vals so callers
    /// of Column::first_vals on full single-valued columns benefit from the
    /// block-aware path. The Some-wrap is unconditional for Full columns,
    /// matching the default impl's behaviour.
    fn get_vals_opt(&self, indexes: &[u32], output: &mut [Option<u64>]) {
        assert_eq!(indexes.len(), output.len());
        if output.is_empty() {
            return;
        }
        let mut tmp: Vec<u64> = vec![0u64; indexes.len()];
        self.get_vals(indexes, &mut tmp);
        for (out, v) in output.iter_mut().zip(tmp) {
            *out = Some(v);
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
    fn test_blockwise_linear_get_vals_sequential_matches_get_range() {
        let mut rng = 0xDEAD_BEEFu64;
        let values: Vec<u64> = (0..1500u64)
            .map(|i| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let residual = (rng % 4096) as u64;
                1_000u64.wrapping_add(i.wrapping_mul(7)).wrapping_add(residual)
            })
            .collect();
        let reader = load_blockwise_linear_reader(&values);

        // Sequential indexes that cross multiple blocks → should hit fast path.
        let indexes: Vec<u32> = (300u32..1400u32).collect();
        let mut batch = vec![0u64; indexes.len()];
        reader.get_vals(&indexes, &mut batch);

        let expected: Vec<u64> = indexes.iter().map(|&i| reader.get_val(i)).collect();
        assert_eq!(batch, expected, "sequential get_vals must match per-doc get_val");
    }

    #[test]
    fn test_blockwise_linear_get_vals_random_matches_get_val() {
        let mut rng = 0xCAFE_F00Du64;
        let values: Vec<u64> = (0..2000u64)
            .map(|i| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let residual = (rng % 256) as u64;
                500u64.wrapping_add(i.wrapping_mul(3)).wrapping_add(residual)
            })
            .collect();
        let reader = load_blockwise_linear_reader(&values);

        // Random indexes — mix in-block and cross-block, including duplicates.
        let mut indexes: Vec<u32> = Vec::with_capacity(200);
        let mut s = 0xC0DEu64;
        for _ in 0..200 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            indexes.push((s % 2000) as u32);
        }
        let mut batch = vec![0u64; indexes.len()];
        reader.get_vals(&indexes, &mut batch);

        let expected: Vec<u64> = indexes.iter().map(|&i| reader.get_val(i)).collect();
        assert_eq!(batch, expected, "random get_vals must match per-doc get_val");
    }

    #[test]
    fn test_blockwise_linear_get_vals_opt_full_column() {
        let values: Vec<u64> = (0..700u64).map(|i| 42u64 + i).collect();
        let reader = load_blockwise_linear_reader(&values);
        let indexes: Vec<u32> = vec![0, 100, 511, 512, 513, 699];
        let mut batch: Vec<Option<u64>> = vec![None; indexes.len()];
        reader.get_vals_opt(&indexes, &mut batch);
        for (i, &idx) in indexes.iter().enumerate() {
            assert_eq!(batch[i], Some(reader.get_val(idx)));
        }
    }

    #[test]
    fn test_blockwise_linear_get_vals_empty_and_single() {
        let values: Vec<u64> = (0..1000u64).collect();
        let reader = load_blockwise_linear_reader(&values);

        // Empty
        let indexes: Vec<u32> = Vec::new();
        let mut batch: Vec<u64> = Vec::new();
        reader.get_vals(&indexes, &mut batch);
        assert!(batch.is_empty());

        // Single
        let indexes = vec![777u32];
        let mut batch = vec![0u64; 1];
        reader.get_vals(&indexes, &mut batch);
        assert_eq!(batch[0], reader.get_val(777));
    }

    #[test]
    fn test_blockwise_linear_get_vals_wide_bit_width() {
        let mut rng = rand::random::<u64>();
        let values: Vec<u64> = (0..1500u64)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                rng
            })
            .collect();
        let reader = load_blockwise_linear_reader(&values);

        // Sequential — exercises scalar fallback under sequential path
        // (delegates to get_range which has its own wide-bit-width path).
        let indexes: Vec<u32> = (0u32..1500u32).collect();
        let mut batch = vec![0u64; 1500];
        reader.get_vals(&indexes, &mut batch);
        let expected: Vec<u64> = (0..1500u32).map(|i| reader.get_val(i)).collect();
        assert_eq!(batch, expected, "wide-bit-width sequential get_vals must match");

        // Random — exercises the per-block-run path directly.
        let mut idx_rng = 0xF00Du64;
        let mut indexes: Vec<u32> = Vec::with_capacity(150);
        for _ in 0..150 {
            idx_rng = idx_rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            indexes.push((idx_rng % 1500) as u32);
        }
        let mut batch = vec![0u64; 150];
        reader.get_vals(&indexes, &mut batch);
        let expected: Vec<u64> = indexes.iter().map(|&i| reader.get_val(i)).collect();
        assert_eq!(batch, expected, "wide-bit-width random get_vals must match");
    }

    /// Wave 21 Phase 2 micro-bench: compare default scalar 4-wide unroll
    /// (via ColumnValues::get_vals default) vs our block-aware override
    /// on both sequential and random index sets. Marked `#[ignore]` —
    /// run with `cargo test --release blockwise_linear_get_vals_bench
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn blockwise_linear_get_vals_bench() {
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
        const QUERY_N: usize = 10_000;
        let mut idx_rng = 0xDEADBEEFu64;
        let random_idx: Vec<u32> = (0..QUERY_N)
            .map(|_| {
                idx_rng = idx_rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (idx_rng % N) as u32
            })
            .collect();
        let sequential_idx: Vec<u32> = (0u32..QUERY_N as u32).collect();
        let warmups = 3usize;
        let trials = 5usize;

        // Baseline: scalar 4-wide unroll via per-doc get_val loop.
        for _ in 0..warmups {
            let mut sum: u64 = 0;
            for &i in &random_idx {
                sum = sum.wrapping_add(reader.get_val(i));
            }
            std::hint::black_box(sum);
        }
        let mut scalar_random_ns: u128 = u128::MAX;
        for _ in 0..trials {
            let t0 = Instant::now();
            let mut sum: u64 = 0;
            for &i in &random_idx {
                sum = sum.wrapping_add(reader.get_val(i));
            }
            std::hint::black_box(sum);
            scalar_random_ns = scalar_random_ns.min(t0.elapsed().as_nanos());
        }

        // Block-aware: random get_vals.
        let mut buf = vec![0u64; QUERY_N];
        for _ in 0..warmups {
            reader.get_vals(&random_idx, &mut buf);
            std::hint::black_box(&buf);
        }
        let mut vals_random_ns: u128 = u128::MAX;
        for _ in 0..trials {
            let t0 = Instant::now();
            reader.get_vals(&random_idx, &mut buf);
            std::hint::black_box(&buf);
            vals_random_ns = vals_random_ns.min(t0.elapsed().as_nanos());
        }

        // Block-aware: sequential get_vals (should hit get_range fast path).
        for _ in 0..warmups {
            reader.get_vals(&sequential_idx, &mut buf);
            std::hint::black_box(&buf);
        }
        let mut vals_seq_ns: u128 = u128::MAX;
        for _ in 0..trials {
            let t0 = Instant::now();
            reader.get_vals(&sequential_idx, &mut buf);
            std::hint::black_box(&buf);
            vals_seq_ns = vals_seq_ns.min(t0.elapsed().as_nanos());
        }

        // Harvest-scale: 10 doc_ids clustered in the same block (mimics
        // size=10 sort scan where heap candidates share a segment range).
        const HARVEST_N: usize = 10;
        let cluster_base: u32 = 123_456;
        let mut harvest_rng = 0xBADC0FFEu64;
        let harvest_idx: Vec<u32> = (0..HARVEST_N)
            .map(|_| {
                harvest_rng = harvest_rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                cluster_base + (harvest_rng % 256) as u32
            })
            .collect();
        let mut harvest_buf = vec![0u64; HARVEST_N];
        for _ in 0..warmups {
            for (out, &i) in harvest_buf.iter_mut().zip(harvest_idx.iter()) {
                *out = reader.get_val(i);
            }
            std::hint::black_box(&harvest_buf);
        }
        let mut scalar_harvest_ns: u128 = u128::MAX;
        for _ in 0..trials * 1000 {
            let t0 = Instant::now();
            for (out, &i) in harvest_buf.iter_mut().zip(harvest_idx.iter()) {
                *out = reader.get_val(i);
            }
            std::hint::black_box(&harvest_buf);
            scalar_harvest_ns = scalar_harvest_ns.min(t0.elapsed().as_nanos());
        }
        for _ in 0..warmups {
            reader.get_vals(&harvest_idx, &mut harvest_buf);
            std::hint::black_box(&harvest_buf);
        }
        let mut block_harvest_ns: u128 = u128::MAX;
        for _ in 0..trials * 1000 {
            let t0 = Instant::now();
            reader.get_vals(&harvest_idx, &mut harvest_buf);
            std::hint::black_box(&harvest_buf);
            block_harvest_ns = block_harvest_ns.min(t0.elapsed().as_nanos());
        }
        let harvest_speedup = scalar_harvest_ns as f64 / block_harvest_ns as f64;

        let random_speedup = scalar_random_ns as f64 / vals_random_ns as f64;
        let seq_speedup = scalar_random_ns as f64 / vals_seq_ns as f64;
        println!(
            "BlockwiseLinear get_vals bench ({QUERY_N} lookups):\n  scalar  random:   {scalar_random_ns:>10} ns\n  block   random:   {vals_random_ns:>10} ns  ({random_speedup:.2}x vs scalar)\n  block   sequential:{vals_seq_ns:>10} ns  ({seq_speedup:.2}x vs scalar)\n\nHarvest-scale ({HARVEST_N} clustered lookups):\n  scalar:           {scalar_harvest_ns:>10} ns\n  block:            {block_harvest_ns:>10} ns  ({harvest_speedup:.2}x vs scalar)"
        );

        // Sanity: outputs must match scalar loop.
        let mut scalar_buf = vec![0u64; QUERY_N];
        for (out, &i) in scalar_buf.iter_mut().zip(random_idx.iter()) {
            *out = reader.get_val(i);
        }
        let mut block_buf = vec![0u64; QUERY_N];
        reader.get_vals(&random_idx, &mut block_buf);
        assert_eq!(scalar_buf, block_buf, "block-random get_vals must match scalar");
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

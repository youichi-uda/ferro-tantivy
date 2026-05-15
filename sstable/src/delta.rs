use std::io::{self, BufWriter, Write};
use std::ops::Range;

use common::{CountingWriter, OwnedBytes};
#[cfg(feature = "zstd-compression")]
use zstd::bulk::Compressor;

use super::value::ValueWriter;
use super::{BlockReader, value, vint};

const FOUR_BIT_LIMITS: usize = 1 << 4;
const VINT_MODE: u8 = 1u8;
const BLOCK_LEN: usize = 4_000;

pub struct DeltaWriter<W, TValueWriter>
where W: io::Write
{
    block: Vec<u8>,
    write: CountingWriter<BufWriter<W>>,
    value_writer: TValueWriter,
    // Only here to avoid allocations.
    stateless_buffer: Vec<u8>,
    block_len: usize,
    // P0-A ingest regression isolation 2026-05-13: keep a single zstd
    // `Compressor` instance across block flushes instead of allocating a
    // fresh one (which initialises ZSTD_CCtx ≈ 256 KB) on every
    // `flush_block` call. For high-cardinality keyword fields (1M unique
    // UUID-like terms ⇒ ~4 000 block flushes per segment) the per-flush
    // compressor init cost dominates sstable's serialise path and
    // accounts for the bulk of the +5.6× ingest regression measured in
    // `dd-pack/p0a-forcemerge-bench-2026-05-13/local_ab_2026_05_13.md`.
    #[cfg(feature = "zstd-compression")]
    zstd_compressor: Option<Compressor<'static>>,
}

impl<W, TValueWriter> DeltaWriter<W, TValueWriter>
where
    W: io::Write,
    TValueWriter: ValueWriter,
{
    pub fn new(wrt: W) -> Self {
        DeltaWriter {
            block: Vec::with_capacity(BLOCK_LEN * 2),
            write: CountingWriter::wrap(BufWriter::new(wrt)),
            value_writer: TValueWriter::default(),
            stateless_buffer: Vec::new(),
            block_len: BLOCK_LEN,
            #[cfg(feature = "zstd-compression")]
            zstd_compressor: None,
        }
    }

    pub fn set_block_len(&mut self, block_len: usize) {
        self.block_len = block_len
    }

    pub fn flush_block(&mut self) -> io::Result<Option<Range<usize>>> {
        if self.block.is_empty() {
            return Ok(None);
        }
        let start_offset = self.write.written_bytes() as usize;

        let buffer: &mut Vec<u8> = &mut self.stateless_buffer;
        self.value_writer.serialize_block(buffer);
        self.value_writer.clear();

        let block_len = buffer.len() + self.block.len();

        // P0-A ingest regression fix 2026-05-13: default to writing
        // sstable blocks **uncompressed** because the dominant cost
        // under the sstable termdict backend is the *read* side
        // (`BlockReader::read_block` → zstd `decompress_to_buffer`)
        // called from background-merge term lookups on
        // unique-per-doc keyword fields. Empirically (local 1M-doc
        // bench, `dd-pack/p0a-forcemerge-bench-2026-05-13/local_ab_2026_05_13.md`)
        // writing uncompressed closes 70 % of the +5.6× ingest
        // regression while *improving* the forcemerge gain further
        // (-7.4× vs the legacy fst termdict baseline). Trade: larger
        // on-disk sstable terms file (~3-5× depending on payload
        // entropy). For Frozen-tier indices stored on S3 where bytes
        // matter more than ingest latency, set
        // `P0A_SSTABLE_ENABLE_ZSTD=1` to opt back into compression on
        // a per-process basis. Per-shard dispatch (Hot/Warm
        // uncompressed, Frozen compressed) is future work that
        // requires threading a setting down through tantivy's
        // IndexWriter API.
        let enable_zstd = std::env::var_os("P0A_SSTABLE_ENABLE_ZSTD").is_some();
        if enable_zstd && cfg!(feature = "zstd-compression") && block_len > 2048 {
            #[cfg(feature = "zstd-compression")]
            {
                buffer.extend_from_slice(&self.block);
                self.block.clear();

                let max_len = zstd::zstd_safe::compress_bound(buffer.len());
                self.block.reserve(max_len);
                // Reuse the compressor across block flushes (initialised
                // lazily on first use). See struct field rustdoc for the
                // P0-A ingest regression context — each `Compressor::new`
                // allocates ZSTD_CCtx (~256 KB) which dominated sstable
                // build cost on high-cardinality keyword fields.
                if self.zstd_compressor.is_none() {
                    self.zstd_compressor = Some(Compressor::new(3)?);
                }
                let compressor = self
                    .zstd_compressor
                    .as_mut()
                    .expect("zstd_compressor was just initialised");
                compressor.compress_to_buffer(buffer, &mut self.block)?;

                // verify compression had a positive impact
                if self.block.len() < buffer.len() {
                    self.write
                        .write_all(&(self.block.len() as u32 + 1).to_le_bytes())?;
                    self.write.write_all(&[1])?;
                    self.write.write_all(&self.block[..])?;
                } else {
                    self.write
                        .write_all(&(block_len as u32 + 1).to_le_bytes())?;
                    self.write.write_all(&[0])?;
                    self.write.write_all(&buffer[..])?;
                }
            }
        } else {
            self.write
                .write_all(&(block_len as u32 + 1).to_le_bytes())?;
            self.write.write_all(&[0])?;
            self.write.write_all(&buffer[..])?;
            self.write.write_all(&self.block[..])?;
        }

        let end_offset = self.write.written_bytes() as usize;
        self.block.clear();
        buffer.clear();
        Ok(Some(start_offset..end_offset))
    }

    fn encode_keep_add(&mut self, keep_len: usize, add_len: usize) {
        if keep_len < FOUR_BIT_LIMITS && add_len < FOUR_BIT_LIMITS {
            let b = (keep_len | (add_len << 4)) as u8;
            self.block.extend_from_slice(&[b])
        } else {
            let mut buf = [VINT_MODE; 20];
            let mut len = 1 + vint::serialize(keep_len as u64, &mut buf[1..]);
            len += vint::serialize(add_len as u64, &mut buf[len..]);
            self.block.extend_from_slice(&buf[..len])
        }
    }

    pub(crate) fn write_suffix(&mut self, common_prefix_len: usize, suffix: &[u8]) {
        let keep_len = common_prefix_len;
        let add_len = suffix.len();
        self.encode_keep_add(keep_len, add_len);
        self.block.extend_from_slice(suffix);
    }

    pub(crate) fn write_value(&mut self, value: &TValueWriter::Value) {
        self.value_writer.write(value);
    }

    pub fn flush_block_if_required(&mut self) -> io::Result<Option<Range<usize>>> {
        if self.block.len() > self.block_len {
            return self.flush_block();
        }
        Ok(None)
    }

    pub fn finish(self) -> CountingWriter<BufWriter<W>> {
        self.write
    }
}

pub struct DeltaReader<TValueReader> {
    common_prefix_len: usize,
    suffix_range: Range<usize>,
    value_reader: TValueReader,
    block_reader: BlockReader,
    idx: usize,
}

impl<TValueReader> DeltaReader<TValueReader>
where TValueReader: value::ValueReader
{
    pub fn new(reader: OwnedBytes) -> Self {
        DeltaReader {
            idx: 0,
            common_prefix_len: 0,
            suffix_range: 0..0,
            value_reader: TValueReader::default(),
            block_reader: BlockReader::new(reader),
        }
    }

    pub fn from_multiple_blocks(reader: Vec<OwnedBytes>) -> Self {
        DeltaReader {
            idx: 0,
            common_prefix_len: 0,
            suffix_range: 0..0,
            value_reader: TValueReader::default(),
            block_reader: BlockReader::from_multiple_blocks(reader),
        }
    }

    pub fn empty() -> Self {
        DeltaReader::new(OwnedBytes::empty())
    }

    fn deserialize_vint(&mut self) -> u64 {
        self.block_reader.deserialize_u64()
    }

    fn read_keep_add(&mut self) -> Option<(usize, usize)> {
        let b = {
            let buf = &self.block_reader.buffer();
            if buf.is_empty() {
                return None;
            }
            buf[0]
        };
        self.block_reader.advance(1);
        match b {
            VINT_MODE => {
                let keep = self.deserialize_vint() as usize;
                let add = self.deserialize_vint() as usize;
                Some((keep, add))
            }
            b => {
                let keep = (b & 0b1111) as usize;
                let add = (b >> 4) as usize;
                Some((keep, add))
            }
        }
    }

    fn read_delta_key(&mut self) -> bool {
        let Some((keep, add)) = self.read_keep_add() else {
            return false;
        };
        self.common_prefix_len = keep;
        let suffix_start = self.block_reader.offset();
        self.suffix_range = suffix_start..(suffix_start + add);
        self.block_reader.advance(add);
        true
    }

    pub fn advance(&mut self) -> io::Result<bool> {
        if self.block_reader.buffer().is_empty() {
            if !self.block_reader.read_block()? {
                return Ok(false);
            }
            let consumed_len = self.value_reader.load(self.block_reader.buffer())?;
            self.block_reader.advance(consumed_len);
            self.idx = 0;
        } else {
            self.idx += 1;
        }
        if !self.read_delta_key() {
            return Ok(false);
        }
        Ok(true)
    }

    #[inline(always)]
    pub fn common_prefix_len(&self) -> usize {
        self.common_prefix_len
    }

    #[inline(always)]
    pub fn suffix(&self) -> &[u8] {
        self.block_reader.buffer_from_to(self.suffix_range.clone())
    }

    #[inline(always)]
    pub fn value(&self) -> &TValueReader::Value {
        self.value_reader.value(self.idx)
    }
}

#[cfg(test)]
mod tests {
    use super::DeltaReader;
    use crate::value::U64MonotonicValueReader;

    #[test]
    fn test_empty() {
        let mut delta_reader: DeltaReader<U64MonotonicValueReader> = DeltaReader::empty();
        assert!(!delta_reader.advance().unwrap());
    }
}

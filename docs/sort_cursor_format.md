# Sort Cursor file format (FerroSearch Wave 15)

`SortCursorIndex` persists a permutation of a segment's `DocId`s ordered
by a configured fast field, written to an auxiliary file alongside the
segment. A subsequent top-K query whose sort matches the configured
`(field, order)` can iterate the cursor and stop after K hits without
scanning the whole segment — equivalent in spirit to Elasticsearch's
`index.sort.field`.

This file lives under `vendor/tantivy-local/src/index/sort_cursor.rs`.
The implementation is a FerroSearch fork extension; upstream tantivy
0.25 has no equivalent. It is opt-in per index via
`IndexSettings::sort_by_field`.

## File naming

```
<segment_uuid>.<field>.sortcursor
```

The path is produced by `SegmentMeta::sort_cursor_path(field)`. The
field appears verbatim in the path (no escaping is applied — index
sort fields are validated upstream and do not contain path separators
or shell metacharacters).

A segment's `meta.json` advertises the cursors it owns via
`InnerSegmentMeta.sort_cursor_fields: Vec<String>` (skipped when
empty). Directory GC uses this list to retain cursor files across
writer cycles; segments produced before the extension existed
deserialise with an empty `sort_cursor_fields`, matching their
on-disk state (no cursor files present).

## Wire format

All multi-byte integers are **little-endian**. The format is:

```
+-------------------------------------+----+--------------------------+
| field                               | sz | notes                    |
+-------------------------------------+----+--------------------------+
| magic                               |  4 | 0x4952_4353 (LE u32; bytes "SCRI" on disk) |
| version                             |  1 | 1 (current)              |
| order                               |  1 | 0 = Asc, 1 = Desc        |
| reserved                            |  2 | must be 0                |
| field_name_len                      |  2 | u16                      |
| field_name                          |  N | UTF-8, exact length above|
| num_doc_ids                         |  4 | u32                      |
| doc_ids                             | 4M | u32 each, M = num_doc_ids|
| trailing magic                      |  4 | 0x4952_4353 (same as leading)              |
+-------------------------------------+----+--------------------------+
```

Total size = `16 + N + 4 * M` bytes, where `N` is the field name byte
length and `M` is `num_doc_ids`.

For a 343,000-doc segment with a 10-byte field name, the file is
about **1.34 MiB**.

## Sort semantics

- The cursor enumerates `0 .. segment.max_doc()` (live and deleted docs
  alike). Consumers filter via the segment's alive bitset.
- When the configured `Order` is `Asc`, `column.first(doc_ids[i])` is
  non-decreasing along `doc_ids`. When `Desc`, it is non-increasing.
- Documents with no value (`column.first(doc) == None`) are placed at
  the end of the cursor regardless of order, matching Elasticsearch's
  default `missing="_last"` semantics.
- Ties are broken by ascending `DocId` for deterministic output.

## Validation

`SortCursorIndex::open(slice)` and `from_bytes(buf)` reject cursors
whose:

- leading magic ≠ `0x4952_4353`
- version ≠ `1`
- order byte ∉ `{0, 1}`
- field name is not valid UTF-8
- payload is shorter than `4 * num_doc_ids + 4` bytes after the field
  name (covers the doc_id array and the trailing magic)
- trailing magic ≠ `0x4952_4353`

Any of these surface as `TantivyError::DataCorruption` and abort
segment open. The trailing magic catches truncated files (e.g. a
crash mid-write); together with the leading magic it pins both ends
of the payload.

## Forward compatibility

- The `version` byte is the negotiation point for future layout
  changes. Readers reject unknown versions outright; mixing readers
  and writers across the boundary requires a re-index.
- The two `reserved` bytes are intended for a future
  `missing_position` flag (e.g. `_first` vs `_last` policy override
  per cursor) and a `key_type` tag (so a v2 reader can know whether
  the cursor was built from `i64` / `u64` / `f64` / `DateTime`
  without re-reading the fast field).
- Multi-field (lexicographic) sort cursors are out of scope for v1
  and would warrant either a `version=2` with composite-key encoding
  or a different file extension.

## What v1 deliberately omits

- **Per-cursor checksum**: the leading + trailing magic and length
  prefixes catch most truncation/corruption modes. A CRC over the
  whole file would catch silent bit-flips but is not justified for
  Phase A; if needed, the surrounding `Directory` already supports
  page-level integrity in mmap mode.
- **Compression**: the doc id array is small and read sequentially.
  Bitpacking would save ~0.5× at the cost of the iteration hot path.
- **Merge survival**: per the design note, force-merging a sorted
  index invalidates per-segment cursors. Phase A leaves the rebuild
  to Phase D; the merger does not touch sort cursor files.

## Phase A acceptance

- `SortCursorIndex::write` and `from_bytes` round-trip arbitrary
  byte buffers.
- `Index::commit()` with `IndexSettings::sort_by_field = Some(...)`
  produces a `<uuid>.<field>.sortcursor` file alongside the segment,
  and `SegmentMeta::list_files()` includes it.
- `SegmentReader::sort_cursor(field)` returns the corresponding
  `Arc<SortCursorIndex>` with `field` / `order` matching the index
  setting and `doc_ids` permuting `0..max_doc` per the rules above.
- 11 unit tests in `src/index/sort_cursor.rs#tests` cover build
  correctness (asc / desc / missing / ties), large-doc ordering,
  empty-cursor handling, end-to-end commit, and three corruption
  rejection cases.

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

## v2 wire format design (Wave 16, deferred)

The Wave 15 v1 format only encodes a single sort field.  TSDB indices
default to multi-field lex sort (e.g. `["_tsid", "@timestamp"]` with
`["asc", "desc"]`), and `IndexSortConfig` already carries the full
`Vec<String>` for both fields and orders.  v2 adds composite key
encoding so Phase E early-term dispatch and Phase H-2 merge reorder
both honour the lex tail.

### Layout

```
+-------------------------------------+----+--------------------------+
| field                               | sz | notes                    |
+-------------------------------------+----+--------------------------+
| magic                               |  4 | 0x4952_4353 (LE u32; "SCRI")            |
| version                             |  1 | 2                                       |
| flags                               |  1 | bit 0 = missing_first; bits 1-7 reserved|
| num_fields                          |  2 | u16, ≥ 1, ≤ 16 (cap to limit cursor size)|
| field_descriptors                   | N  | num_fields × FieldDescriptor (see below)|
| num_doc_ids                         |  4 | u32                                     |
| doc_ids                             | 4M | u32 each, M = num_doc_ids               |
| trailing magic                      |  4 | 0x4952_4353                             |
+-------------------------------------+----+--------------------------+

FieldDescriptor (variable length):
+--------------+----+-------------------------------------------+
| field        | sz | notes                                     |
+--------------+----+-------------------------------------------+
| order        |  1 | 0 = Asc, 1 = Desc                         |
| key_type     |  1 | 0 = i64 / 1 = u64 / 2 = f64 / 3 = DateTime|
| field_len    |  2 | u16                                       |
| field_name   |  K | UTF-8, exact length above                 |
+--------------+----+-------------------------------------------+
```

### Build semantics

`build_sort_cursor_from_fast_fields_multi(readers, &[(field, order)],
num_docs, missing_first)` reads each field's column and produces a
`Vec<(CompositeKey, DocId)>` where `CompositeKey` is a tuple of u64
values (one per field, monotonically mapped per `key_type`).  The
sort uses lex order on the tuple with per-field asc/desc applied
inline (`b.cmp(a)` for desc fields, `a.cmp(b)` for asc).  Missing
values within any field are encoded as `u64::MAX` and bumped to the
end via the same missing-flag mechanism v1 uses, with the
`missing_first` flag inverting the policy when set.

### Read semantics

`SortCursorIndex::open` dispatches on `version`:
- version 1 → existing single-field reader (this doc).
- version 2 → multi-field reader; exposes
  `field_descriptors() -> &[FieldDescriptor]` and the same
  `iter() -> impl Iterator<Item = DocId>`.  Consumers that only care
  about doc_id order (Phase H-2 merge reorder) work unchanged; Phase
  E callers gain a `prefix_match(query_sort) -> bool` that walks
  field_descriptors and checks each `(field, order)` pair against
  the query's sort.

### Caller-side dispatch (Phase E v2)

`EarlyTermSortByCursorCollector` already takes a single `T: FastValue`.
v2 needs a tuple variant `EarlyTermSortByCompositeCursorCollector<(T1,
T2, ...)>` with const-generic field count.  When the query's sort
matches the *prefix* (not full) of the cursor's field descriptors,
the collector still walks the cursor and returns docs in the
multi-field-sorted order — only the `Fruit` shape changes (sort_key
becomes a tuple).

### Migration

v1 cursors stay readable.  v2 writers default to v2 only when
`IndexSortConfig.is_multi_field() == true`; single-field configs
continue emitting v1 (smaller files, no break for existing readers).
A `--cursor-version=2` `IndexWriter` opt-in can force v2 even for
single-field indices for testing.

### Out of scope for v2

- String / keyword sort fields (would need term-ordinal-based
  composite encoding; v1 also doesn't support).
- Cursor file compression (per-field key is u64 fixed, doc_id is
  u32 fixed; bitpacking saves ~0.5× of the cursor file at iteration
  cost).
- Mixed-segment v1 + v2 within one index — readers must handle
  both simultaneously since merges produce v2 from v1 inputs when
  multi-field is enabled mid-life.

### Multi-field cursor file size

For `["_tsid", "@timestamp"]` on a 343,000-doc segment:
- field descriptors: ~30 bytes (2 fields × ~15 bytes each)
- doc_ids: 4 × 343,000 = 1.34 MiB
- header + trailing: ~32 bytes
- **Total: ~1.34 MiB** (same as v1 — the field descriptors are
  negligible vs the doc_id payload).

### Implementation effort estimate

- Wire format + build/read on tantivy fork: ~400 LOC + 8 tests
- Phase E composite collector dispatch: ~250 LOC + 4 integration tests
- Phase H-2 merger: works unchanged (uses doc_id_mapping which is the
  same for v1 / v2 since the cursor encodes the SAME `Vec<DocId>` —
  composite affects ORDERING, not the doc_id list itself)
- Wave 16 estimate: 2-3 days end-to-end, gated on TSDB demand
  signal (no blocker for non-TSDB Wave 15 production)

## Live-ingest monotonicity caveat

**Wave 15 Phase H-2** physically reorders alive docs at merge time so
the merged segment's doc-id sequence matches the index sort order.
This works at the **segment-creation boundary only**:

- Docs added between merges land in **flush order** (insertion order,
  not sort order) within their fresh per-flush segment.
- Each per-flush segment carries its own sort cursor (Phase A hook),
  so cursor-aware queries still find the right docs — just walked
  per-segment via the cursor instead of via doc-id-ascending iteration.
- Once a merge runs over those small fresh segments, the merged output
  **does** get the H-2 reorder and the existing
  `SortByStaticFastValue` SIMD top-K filter naturally early-terminates.

In other words: H-2's perf benefit is **post-merge only**.  Until a
merge consolidates fresh segments, sort-by-indexed-field queries pay
the per-segment cursor walk cost (slightly slower than the optimised
WARM-cache + SIMD baseline, per the Phase G empirical finding) on the
just-flushed shards.

### Operator practice for sort-cursor production indices

The recommended deployment pattern when you want maximum sort-query
throughput:

1. **Bulk-load + force-merge promotion.**  After a bulk write
   completes (e.g. nightly ingest, snapshot restore, a reindex), call
   `POST /<index>/_forcemerge?max_num_segments=1` once.  The forced
   merge runs through Phase H-1 (cursor rebuild) + Phase H-2 (physical
   reorder) and produces a single sorted segment from which subsequent
   sort queries get the SIMD early-term win.
2. **For continuous-write workloads** (logs, telemetry, append-only
   streams): rely on `LogMergePolicy`'s background merges to
   consolidate small segments.  Latest-arrival docs in fresh
   un-consolidated segments still answer queries correctly, just at
   the slightly slower per-segment cursor walk rate; the next
   background merge restores monotonicity.
3. **Mixed read-write workload tuning**: lower
   `index.merge.policy.max_merged_segment` to encourage smaller
   merged-segment caps, which pulls fresh docs into sorted layout
   sooner (at the cost of more overall merge throughput).

### Why H-2 doesn't sort at flush time

Phase H-2's reorder lives in `IndexMerger::write` rather than
`SegmentWriter::flush`:

- Flush-time reorder (the "Alternative A pure" variant) requires
  buffering ALL docs in RAM before writing — high memory pressure for
  large per-flush ranges, and changes the latency profile of
  `index_writer.commit()` non-trivially.
- Merge-time reorder reuses the input segments' on-disk fast field
  columns to build the sort permutation, with no extra memory ceiling
  beyond the existing merger's working set.
- ES IndexSortByField uses the flush-time pattern, which is why ES
  achieves consistent sorted layout from doc 1; Wave 15 trades that
  consistency for a smaller blast radius (no SegmentWriter rewrite,
  no per-flush memory cap risk) at the cost of the post-merge-only
  guarantee.

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

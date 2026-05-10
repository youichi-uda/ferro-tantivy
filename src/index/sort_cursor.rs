//! Auxiliary "sort cursor" index — FerroSearch Wave 15 extension.
//!
//! See `vendor/tantivy-local/docs/sort_cursor_format.md` for the on-disk
//! file layout. The high-level idea is to store a permutation of the
//! segment's `DocId`s ordered by a configured fast field's first value.
//! A subsequent top-K query that matches the configured sort can iterate
//! the cursor and stop after K hits without scanning the whole segment.
//!
//! This module is deliberately decoupled from the `Collector` trait — it
//! only owns the build/persist/iterate primitives. The Wave 15 Phase B
//! collector lives in `crate::collector::early_term_sort_by_cursor`.

use std::cmp::Ordering;
use std::fmt::Debug;
use std::io::{self, Write};

use columnar::Column;
use common::{BinarySerializable, DateTime};

use crate::directory::FileSlice;
use crate::error::DataCorruption;
use crate::index::Order;
use crate::DocId;

/// Magic prefix `"SCRI"` (Sort Cursor RaIl).
///
/// Written as little-endian `u32`. Distinct from any other tantivy file
/// magic to make corrupted/cross-mounted files fail loudly.
const SORT_CURSOR_MAGIC: u32 = 0x4952_4353; // 'S','C','R','I' little-endian
/// Format version. Bump on incompatible layout changes.
const SORT_CURSOR_VERSION: u8 = 1;

/// Auxiliary sort cursor for a single (segment, field) pair.
///
/// Owns a permutation `doc_ids` of the segment's live `DocId`s such that
/// `column.first(doc_ids[i])` is non-decreasing (asc) or non-increasing
/// (desc). Documents whose value is absent (`column.first(doc) == None`)
/// are placed at the end regardless of order, matching Elasticsearch's
/// default `missing="_last"` semantics.
///
/// The struct is cheap to clone (it owns a single `Vec<DocId>` and a
/// short header), but in production the recommended access path is to
/// open a `FileSlice` view via [`SortCursorIndex::open`] and iterate via
/// [`SortCursorIndex::iter`] without copying the bulk array.
#[derive(Clone, Debug)]
pub struct SortCursorIndex {
    field: String,
    order: Order,
    doc_ids: Vec<DocId>,
}

/// Internal sort key that mirrors Elasticsearch's `missing="_last"` rule.
/// `None` always sorts last regardless of the requested `Order`.
fn sort_key<T: PartialOrd>(
    a: (DocId, &Option<T>),
    b: (DocId, &Option<T>),
    order: Order,
) -> Ordering {
    let a_missing = a.1.is_none() as u8;
    let b_missing = b.1.is_none() as u8;
    match a_missing.cmp(&b_missing) {
        Ordering::Equal => {}
        not_eq => return not_eq,
    }
    let value_cmp = match (a.1, b.1) {
        (Some(av), Some(bv)) => match order {
            Order::Asc => av.partial_cmp(bv).unwrap_or(Ordering::Equal),
            Order::Desc => bv.partial_cmp(av).unwrap_or(Ordering::Equal),
        },
        _ => Ordering::Equal,
    };
    // Stable tie-breaker: doc_id ascending. This makes the on-disk
    // representation deterministic across builds with the same data.
    value_cmp.then_with(|| a.0.cmp(&b.0))
}

impl SortCursorIndex {
    /// Builds a sort cursor by reading every doc's first value from `column`.
    ///
    /// `num_docs` is the segment's `max_doc` (the cursor enumerates
    /// `0..num_docs`; deleted docs are filtered later by the consumer).
    /// `field` is recorded verbatim in the file header so a later
    /// `open` can verify the cursor matches the requested field.
    pub fn build_from_column<T>(
        field: impl Into<String>,
        order: Order,
        column: &Column<T>,
        num_docs: u32,
    ) -> Self
    where
        T: PartialOrd + Copy + Debug + Send + Sync + 'static,
    {
        let mut keyed: Vec<(DocId, Option<T>)> = (0..num_docs)
            .map(|doc_id| (doc_id, column.first(doc_id)))
            .collect();
        keyed.sort_by(|a, b| sort_key((a.0, &a.1), (b.0, &b.1), order));
        let doc_ids = keyed.into_iter().map(|(d, _)| d).collect();
        Self {
            field: field.into(),
            order,
            doc_ids,
        }
    }

    /// Specialization that builds from a raw `Vec<Option<T>>` keyed by
    /// `DocId`. Useful in tests where constructing a `Column<T>` is
    /// awkward.
    #[doc(hidden)]
    pub fn build_from_values<T>(
        field: impl Into<String>,
        order: Order,
        values: Vec<Option<T>>,
    ) -> Self
    where
        T: PartialOrd + Copy + Debug + Send + Sync + 'static,
    {
        let mut keyed: Vec<(DocId, Option<T>)> = values
            .into_iter()
            .enumerate()
            .map(|(d, v)| (d as DocId, v))
            .collect();
        keyed.sort_by(|a, b| sort_key((a.0, &a.1), (b.0, &b.1), order));
        let doc_ids = keyed.into_iter().map(|(d, _)| d).collect();
        Self {
            field: field.into(),
            order,
            doc_ids,
        }
    }

    /// Field name recorded in the cursor header.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Sort order recorded in the cursor header.
    pub fn order(&self) -> Order {
        self.order
    }

    /// Number of doc ids in the cursor (== `max_doc` at build time).
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Returns `true` if the cursor is empty (no docs at build time).
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// Iterates the doc ids in sort order.
    pub fn iter(&self) -> impl Iterator<Item = DocId> + '_ {
        self.doc_ids.iter().copied()
    }

    /// Returns the underlying `Vec<DocId>` (already sorted).
    pub fn doc_ids(&self) -> &[DocId] {
        &self.doc_ids
    }

    /// Serialises the cursor to `writer` using the layout described in
    /// `docs/sort_cursor_format.md`.
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        SORT_CURSOR_MAGIC.serialize(writer)?;
        SORT_CURSOR_VERSION.serialize(writer)?;
        let order_byte: u8 = match self.order {
            Order::Asc => 0,
            Order::Desc => 1,
        };
        order_byte.serialize(writer)?;
        // 2 reserved bytes for future flags (e.g. missing-position policy).
        0u16.serialize(writer)?;
        // Field name length (u16) + UTF-8 bytes.
        let field_bytes = self.field.as_bytes();
        let field_len = u16::try_from(field_bytes.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "sort cursor field name exceeds 65,535 bytes",
            )
        })?;
        field_len.serialize(writer)?;
        writer.write_all(field_bytes)?;
        // Number of doc ids, then the doc ids themselves (u32 LE).
        let num_docs = u32::try_from(self.doc_ids.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "sort cursor doc count exceeds u32::MAX",
            )
        })?;
        num_docs.serialize(writer)?;
        for &doc_id in &self.doc_ids {
            doc_id.serialize(writer)?;
        }
        // Trailing magic for sanity-checking truncation.
        SORT_CURSOR_MAGIC.serialize(writer)?;
        Ok(())
    }

    /// Deserialises a cursor from a `FileSlice`.
    ///
    /// Returns `DataCorruption` on magic / version / size mismatch.
    pub fn open(slice: FileSlice) -> crate::Result<Self> {
        let bytes = slice.read_bytes()?;
        Self::from_bytes(bytes.as_slice())
    }

    /// Parse a cursor from an in-memory byte buffer.
    pub fn from_bytes(buf: &[u8]) -> crate::Result<Self> {
        let mut reader = buf;
        let magic = u32::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor: failed to read magic: {e}"))
        })?;
        if magic != SORT_CURSOR_MAGIC {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor: bad magic 0x{magic:08x}, expected 0x{SORT_CURSOR_MAGIC:08x}"
            ))
            .into());
        }
        let version = u8::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor: failed to read version: {e}"))
        })?;
        if version != SORT_CURSOR_VERSION {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor: unsupported version {version}, expected {SORT_CURSOR_VERSION}"
            ))
            .into());
        }
        let order_byte = u8::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor: failed to read order: {e}"))
        })?;
        let order = match order_byte {
            0 => Order::Asc,
            1 => Order::Desc,
            other => {
                return Err(DataCorruption::comment_only(format!(
                    "sort cursor: invalid order byte {other}"
                ))
                .into());
            }
        };
        let _reserved = u16::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor: failed to read reserved: {e}"))
        })?;
        let field_len = u16::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor: failed to read field_len: {e}"))
        })? as usize;
        if reader.len() < field_len {
            return Err(DataCorruption::comment_only(
                "sort cursor: field name truncated".to_string(),
            )
            .into());
        }
        let (field_bytes, rest) = reader.split_at(field_len);
        let field = std::str::from_utf8(field_bytes)
            .map_err(|e| {
                DataCorruption::comment_only(format!("sort cursor: field name not utf8: {e}"))
            })?
            .to_string();
        reader = rest;
        let num_docs = u32::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor: failed to read num_docs: {e}"))
        })? as usize;
        let needed_bytes = num_docs.checked_mul(4).and_then(|n| n.checked_add(4));
        let Some(needed_bytes) = needed_bytes else {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor: num_docs ({num_docs}) overflow"
            ))
            .into());
        };
        if reader.len() < needed_bytes {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor: payload truncated (need {needed_bytes} bytes, have {})",
                reader.len()
            ))
            .into());
        }
        let mut doc_ids = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            doc_ids.push(u32::deserialize(&mut reader).map_err(|e| {
                DataCorruption::comment_only(format!("sort cursor: failed to read doc_id: {e}"))
            })?);
        }
        let trailing_magic = u32::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!(
                "sort cursor: failed to read trailing magic: {e}"
            ))
        })?;
        if trailing_magic != SORT_CURSOR_MAGIC {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor: bad trailing magic 0x{trailing_magic:08x}"
            ))
            .into());
        }
        Ok(Self {
            field,
            order,
            doc_ids,
        })
    }
}

/// Builds a sort cursor by trying common fast-field column types in order
/// (`i64`, `u64`, `f64`, `DateTime`).
///
/// Returns an error if no compatible column exists for `field`. This is
/// the function the `SegmentWriter` finalize hook calls when
/// `IndexSettings::sort_by_field` is set.
pub fn build_sort_cursor_from_fast_fields(
    readers: &crate::fastfield::FastFieldReaders,
    field: &str,
    order: Order,
    num_docs: u32,
) -> crate::Result<SortCursorIndex> {
    if let Some(col) = readers.column_opt::<i64>(field)? {
        return Ok(SortCursorIndex::build_from_column(field, order, &col, num_docs));
    }
    if let Some(col) = readers.column_opt::<u64>(field)? {
        return Ok(SortCursorIndex::build_from_column(field, order, &col, num_docs));
    }
    if let Some(col) = readers.column_opt::<f64>(field)? {
        return Ok(SortCursorIndex::build_from_column(field, order, &col, num_docs));
    }
    if let Some(col) = readers.column_opt::<DateTime>(field)? {
        return Ok(SortCursorIndex::build_from_column(field, order, &col, num_docs));
    }
    Err(crate::TantivyError::SchemaError(format!(
        "Field `{field}` is not a numeric or date fast field; cannot build sort cursor"
    )))
}

/// Builds and persists the auxiliary sort cursor file(s) for `segment`,
/// based on `segment.index().settings().sort_by_field`.
///
/// **FerroSearch extension (Wave 15).** Called by `index_documents` and
/// `SingleSegmentIndexWriter::finalize` after the main segment files have
/// been laid out on disk and `with_max_doc` has been applied. Returns
/// the field names whose cursor was successfully written, so the caller
/// can advertise them in `SegmentMeta::sort_cursor_fields` via
/// [`crate::index::Segment::with_sort_cursor_fields`].
///
/// A no-op (`Ok(Vec::new())`) when the index has no `sort_by_field`
/// configured — keeps the indexing path zero-cost for indices that do
/// not opt in.
pub fn build_and_write_sort_cursors(
    segment: &mut crate::index::Segment,
) -> crate::Result<Vec<String>> {
    use common::TerminatingWrite;

    let sort_by = match segment.index().settings().sort_by_field.clone() {
        Some(sb) => sb,
        None => return Ok(Vec::new()),
    };
    let max_doc = segment.meta().max_doc();
    if max_doc == 0 {
        // An empty segment carries no doc ids; skip writing a cursor.
        return Ok(Vec::new());
    }
    let reader = crate::index::SegmentReader::open(segment)?;
    let cursor = build_sort_cursor_from_fast_fields(
        reader.fast_fields(),
        &sort_by.field,
        sort_by.order,
        max_doc,
    )?;
    let mut writer = segment.open_sort_cursor_write(&sort_by.field)?;
    cursor.write(&mut writer)?;
    writer.terminate()?;
    Ok(vec![sort_by.field])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(cursor: &SortCursorIndex) -> SortCursorIndex {
        let mut buf = Vec::new();
        cursor.write(&mut buf).expect("write should succeed");
        SortCursorIndex::from_bytes(&buf).expect("parse should succeed")
    }

    #[test]
    fn build_correctness_asc() {
        // values per doc_id: doc 0 = 30, doc 1 = 10, doc 2 = 20
        let cursor = SortCursorIndex::build_from_values(
            "score",
            Order::Asc,
            vec![Some(30i64), Some(10), Some(20)],
        );
        assert_eq!(cursor.doc_ids(), &[1u32, 2, 0]);
        assert_eq!(cursor.field(), "score");
        assert_eq!(cursor.order(), Order::Asc);
    }

    #[test]
    fn build_correctness_desc() {
        let cursor = SortCursorIndex::build_from_values(
            "score",
            Order::Desc,
            vec![Some(30i64), Some(10), Some(20)],
        );
        assert_eq!(cursor.doc_ids(), &[0u32, 2, 1]);
        assert_eq!(cursor.order(), Order::Desc);
    }

    #[test]
    fn missing_values_sort_last_for_both_orders() {
        // doc 1 has no value; should always be at the tail.
        let cursor_asc = SortCursorIndex::build_from_values(
            "ts",
            Order::Asc,
            vec![Some(5i64), None, Some(1)],
        );
        assert_eq!(cursor_asc.doc_ids(), &[2u32, 0, 1]);

        let cursor_desc = SortCursorIndex::build_from_values(
            "ts",
            Order::Desc,
            vec![Some(5i64), None, Some(1)],
        );
        assert_eq!(cursor_desc.doc_ids(), &[0u32, 2, 1]);
    }

    #[test]
    fn ties_break_by_doc_id_for_determinism() {
        let cursor = SortCursorIndex::build_from_values(
            "k",
            Order::Asc,
            vec![Some(7i64), Some(7), Some(7), Some(1)],
        );
        // 1 first, then ties broken by ascending doc_id.
        assert_eq!(cursor.doc_ids(), &[3u32, 0, 1, 2]);
    }

    #[test]
    fn write_open_roundtrip_small() {
        let cursor = SortCursorIndex::build_from_values(
            "@timestamp",
            Order::Desc,
            vec![Some(100i64), Some(50), Some(75), None],
        );
        let restored = roundtrip(&cursor);
        assert_eq!(restored.field(), "@timestamp");
        assert_eq!(restored.order(), Order::Desc);
        assert_eq!(restored.doc_ids(), cursor.doc_ids());
    }

    #[test]
    fn write_open_roundtrip_large() {
        // 10K docs with semi-random keys.
        let n = 10_000u32;
        let values: Vec<Option<i64>> = (0..n)
            .map(|i| Some(((i as i64) * 2654435761) & 0xFFFF))
            .collect();
        let cursor = SortCursorIndex::build_from_values("k", Order::Asc, values);
        let restored = roundtrip(&cursor);
        assert_eq!(restored.len(), n as usize);
        assert_eq!(restored.doc_ids(), cursor.doc_ids());
        // Independently verify monotonicity of the restored ordering by
        // re-keying with the same sort.
        // (We rebuild the values map and walk the doc_ids to check
        // values are non-decreasing.)
        let values2: Vec<i64> = (0..n)
            .map(|i| ((i as i64) * 2654435761) & 0xFFFF)
            .collect();
        let mut prev: Option<i64> = None;
        for &doc in restored.doc_ids() {
            let v = values2[doc as usize];
            if let Some(p) = prev {
                assert!(p <= v, "non-monotonic at doc {doc}: {p} > {v}");
            }
            prev = Some(v);
        }
    }

    #[test]
    fn empty_cursor_roundtrips() {
        let cursor =
            SortCursorIndex::build_from_values::<i64>("empty", Order::Asc, Vec::new());
        assert!(cursor.is_empty());
        let restored = roundtrip(&cursor);
        assert!(restored.is_empty());
        assert_eq!(restored.field(), "empty");
    }

    #[test]
    fn corrupted_magic_is_rejected() {
        let cursor = SortCursorIndex::build_from_values(
            "f",
            Order::Asc,
            vec![Some(1i64), Some(2), Some(3)],
        );
        let mut buf = Vec::new();
        cursor.write(&mut buf).unwrap();
        buf[0] = 0; // corrupt leading magic
        let err = SortCursorIndex::from_bytes(&buf).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("magic"), "expected magic error, got: {msg}");
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let cursor = SortCursorIndex::build_from_values(
            "f",
            Order::Asc,
            vec![Some(1i64), Some(2)],
        );
        let mut buf = Vec::new();
        cursor.write(&mut buf).unwrap();
        // Version byte sits right after the 4-byte magic.
        buf[4] = 99;
        let err = SortCursorIndex::from_bytes(&buf).unwrap_err();
        assert!(err.to_string().contains("version"));
    }

    #[test]
    fn truncated_payload_is_rejected() {
        let cursor = SortCursorIndex::build_from_values(
            "f",
            Order::Asc,
            (0..256i64).map(Some).collect(),
        );
        let mut buf = Vec::new();
        cursor.write(&mut buf).unwrap();
        buf.truncate(buf.len() - 16);
        let err = SortCursorIndex::from_bytes(&buf).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("truncated") || msg.contains("trailing"),
            "expected truncation error, got: {msg}"
        );
    }

    #[test]
    fn end_to_end_index_commit_persists_cursor() -> crate::Result<()> {
        use crate::index::{IndexSettings, IndexSortByField};
        use crate::schema::{Schema, FAST};
        use crate::{Index, IndexWriter, TantivyDocument};

        let mut schema_builder = Schema::builder();
        let value_field = schema_builder.add_i64_field("value", FAST);
        let schema = schema_builder.build();

        let settings = IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "value".to_string(),
                order: Order::Desc,
            }),
            ..Default::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_for_tests()?;
        // Insertion order is intentionally scrambled so the cursor must
        // do non-trivial work to recover the descending sort.
        for v in [40i64, 10, 30, 20, 50] {
            let mut doc = TantivyDocument::default();
            doc.add_i64(value_field, v);
            writer.add_document(doc)?;
        }
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let segment_reader = searcher.segment_reader(0);
        // Meta should advertise the cursor.
        let cursor_fields: Vec<&str> = segment_reader.sort_cursor_fields().collect();
        assert_eq!(cursor_fields, vec!["value"]);

        let cursor = segment_reader
            .sort_cursor("value")
            .expect("sort cursor should be present");
        assert_eq!(cursor.field(), "value");
        assert_eq!(cursor.order(), Order::Desc);
        // Recovered doc order: descending by value → 50, 40, 30, 20, 10
        // Insertion DocIds: 0=40, 1=10, 2=30, 3=20, 4=50
        // → expected cursor.doc_ids() = [4, 0, 2, 3, 1].
        assert_eq!(cursor.doc_ids(), &[4u32, 0, 2, 3, 1]);
        Ok(())
    }

    /// **FerroSearch Wave 15 Phase H-1.** After force-merging multiple
    /// segments down to one, the merged segment must carry a freshly
    /// rebuilt sort cursor.  Without the Phase H-1 hook in
    /// `segment_updater::merge`, post-merge segments would have an
    /// empty `sort_cursor_fields` list (each input segment owned its
    /// own cursor; merging produces a fresh segment from scratch) and
    /// the Phase E dispatch gate would silently fall back to the
    /// legacy `SortByStaticFastValue` path.
    ///
    /// This is exactly the http_logs Rally
    /// `*-after-force-merge-1-seg` ES-wins case — without H-1, those
    /// queries cannot benefit from the early-term cursor.
    #[test]
    fn post_force_merge_segment_carries_rebuilt_cursor() -> crate::Result<()> {
        use crate::index::{IndexSettings, IndexSortByField};
        use crate::indexer::NoMergePolicy;
        use crate::schema::{Schema, FAST};
        use crate::{Index, IndexWriter, TantivyDocument};

        let mut schema_builder = Schema::builder();
        let value_field = schema_builder.add_i64_field("value", FAST);
        let schema = schema_builder.build();

        let settings = IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "value".to_string(),
                order: Order::Desc,
            }),
            ..Default::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()?;

        // Disable auto-merge so we can deterministically build several
        // small segments first, then trigger an explicit force-merge.
        let mut writer: IndexWriter = index.writer_for_tests()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Build 3 segments by committing in batches.  Insertion order is
        // scrambled inside each batch so the per-segment cursor must do
        // real work to recover the descending sort.
        for batch in [&[40i64, 10, 30][..], &[20, 50][..], &[60, 5][..]] {
            for v in batch {
                let mut doc = TantivyDocument::default();
                doc.add_i64(value_field, *v);
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }

        // Pre-merge sanity: the reader sees 3 segments, each with its
        // own cursor (Phase A invariant — every fresh segment that
        // commits with `sort_by_field` set carries a cursor).
        let pre_merge_searcher = index.reader()?.searcher();
        assert_eq!(
            pre_merge_searcher.segment_readers().len(),
            3,
            "expected 3 pre-merge segments"
        );
        for sr in pre_merge_searcher.segment_readers() {
            assert!(
                sr.sort_cursor("value").is_some(),
                "every pre-merge segment must carry the cursor (Phase A)"
            );
        }

        // Trigger an in-place force-merge via `IndexWriter::merge`.
        let segment_ids: Vec<crate::index::SegmentId> = pre_merge_searcher
            .segment_readers()
            .iter()
            .map(|sr| sr.segment_id())
            .collect();
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        // After merge: a single new segment must exist AND it must
        // carry the rebuilt sort cursor.  Without Phase H-1 this
        // assertion fails (`sort_cursor("value")` returns None).
        let post_merge_searcher = index.reader()?.searcher();
        assert_eq!(
            post_merge_searcher.segment_readers().len(),
            1,
            "force-merge should collapse to a single segment"
        );
        let merged_sr = post_merge_searcher.segment_reader(0);
        let cursor = merged_sr
            .sort_cursor("value")
            .expect("Phase H-1: merged segment must carry rebuilt cursor");
        assert_eq!(cursor.field(), "value");
        assert_eq!(cursor.order(), Order::Desc);
        // Cursor enumerates exactly `max_doc` of the merged segment.
        // We don't assert a literal count because the writer may elide
        // documents at commit boundaries in some configurations.
        assert_eq!(
            cursor.len() as u32,
            merged_sr.max_doc(),
            "cursor must enumerate every doc in the merged segment"
        );
        // Walk the cursor and verify monotonicity (desc) by re-reading
        // the value column for each emitted doc id — this validates
        // that the cursor's permutation actually corresponds to the
        // post-merge doc layout, not stale pre-merge doc ids.
        let column = merged_sr
            .fast_fields()
            .i64("value")
            .expect("value column must exist");
        let mut prev: Option<i64> = None;
        for doc_id in cursor.iter() {
            let v = column.first(doc_id).expect("dense column has all values");
            if let Some(p) = prev {
                assert!(
                    p >= v,
                    "cursor must be non-increasing for Desc: {p} then {v} at doc {doc_id}"
                );
            }
            prev = Some(v);
        }
        Ok(())
    }

    /// **FerroSearch Wave 15 Phase H-2 (Alternative-A-at-merge).** After
    /// force-merging, the merged segment's *physical doc-id sequence*
    /// must match the index sort order — not just the cursor's
    /// permutation.  When that is true, the cursor is a strict identity
    /// (`cursor.iter()` yields `0, 1, 2, ..., max_doc-1`) and the
    /// existing `SortByStaticFastValue` SIMD top-K threshold filter
    /// naturally early-terminates because doc-id-ascending iteration
    /// coincides with sort order — the heap fills with the K extreme
    /// values immediately and the SIMD `mask == 0` block-skip kicks in
    /// for every subsequent block.
    ///
    /// Without H-2, the cursor would be a non-identity permutation
    /// (e.g. `[2, 3, 4, 1, 0, 5]` for the H-1 test's input) — the
    /// per-segment cursor still serves Phase E's early-term collector
    /// correctly, but the WARM-cache + SIMD baseline would walk in
    /// doc-id order and pay the full O(N log K) heap cost with no
    /// SIMD block-skip win.
    #[test]
    fn post_force_merge_segment_doc_ids_match_sort_order() -> crate::Result<()> {
        use crate::index::{IndexSettings, IndexSortByField};
        use crate::indexer::NoMergePolicy;
        use crate::schema::{Schema, FAST};
        use crate::{Index, IndexWriter, TantivyDocument};

        let mut schema_builder = Schema::builder();
        let value_field = schema_builder.add_i64_field("value", FAST);
        let schema = schema_builder.build();
        let settings = IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "value".to_string(),
                order: Order::Desc,
            }),
            ..Default::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_for_tests()?;
        writer.set_merge_policy(Box::new(NoMergePolicy));

        // Build 3 segments with intentionally scrambled per-segment
        // value orderings so the H-2 reorder must do real work to
        // produce a strictly Desc post-merge sequence.
        for batch in [&[10i64, 15][..], &[40, 30][..], &[20, 5][..]] {
            for v in batch {
                let mut doc = TantivyDocument::default();
                doc.add_i64(value_field, *v);
                writer.add_document(doc)?;
            }
            writer.commit()?;
        }

        let segment_ids = index.searchable_segment_ids()?;
        assert_eq!(segment_ids.len(), 3);
        writer.merge(&segment_ids).wait()?;
        writer.wait_merging_threads()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let seg = searcher.segment_reader(0);
        let column = seg.fast_fields().i64("value")?;

        // The killer assertion: merged segment doc-id 0 has the LARGEST
        // value, doc-id 1 the next-largest, etc.  This is strict
        // monotonicity by doc-id — the property the existing
        // `SortByStaticFastValue` SIMD top-K filter needs to early-
        // terminate without a separate cursor walk.
        let max_doc = seg.max_doc();
        let mut values_by_doc_id: Vec<i64> = Vec::with_capacity(max_doc as usize);
        for doc in 0..max_doc {
            values_by_doc_id.push(column.first(doc).expect("value present"));
        }
        assert_eq!(
            values_by_doc_id,
            vec![40, 30, 20, 15, 10, 5],
            "Phase H-2: merged segment's doc-id sequence must match sort order \
             (identity cursor); without H-2 the order would still be the \
             input-segment concatenation"
        );

        // Cursor must STILL be present (Phase H-1 hook still runs after
        // H-2 reorder — the cursor file is now an identity permutation
        // and the dispatch gate keeps working).
        let cursor = seg
            .sort_cursor("value")
            .expect("cursor must still be written by H-1 hook");
        let cursor_doc_ids: Vec<u32> = cursor.iter().collect();
        assert_eq!(
            cursor_doc_ids,
            vec![0, 1, 2, 3, 4, 5],
            "Phase H-2: post-reorder cursor is the identity permutation"
        );

        Ok(())
    }
}

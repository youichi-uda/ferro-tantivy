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

use crate::directory::{Directory, FileSlice};
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

// ---------------------------------------------------------------------------
// Wave 18-1: multi-field sort cursor (v2 wire format).
//
// See `dd-pack/wave18-multi-field-cursor-v2-design.md` for the rationale.
// v1 (above) is preserved verbatim — single-field deployments keep using
// it. v2 is opt-in: callers that explicitly request multi-field
// `index.sort` get a v2 file written via `build_sort_cursor_v2_from_fast_fields`
// + `SortCursorIndexV2::write`.  Reader-side dispatch goes through
// [`SortCursorAny::open`] which peeks the version byte and returns
// either a v1 or v2 cursor.
// ---------------------------------------------------------------------------

/// v2 format version byte. Distinct from [`SORT_CURSOR_VERSION`] (=1)
/// so a v1 reader rejects v2 files cleanly with `unsupported version`.
const SORT_CURSOR_VERSION_V2: u8 = 2;
/// Header `flags` bit set in v2 to mark the file as multi-field.
/// Reader rejects a v2 file whose flag is unset as corrupt.
const SORT_CURSOR_FLAG_MULTI_FIELD: u8 = 0x01;
/// Maximum number of fields a v2 cursor may carry. Bounded so the
/// `num_fields: u8` field cannot overflow and the missing-bitmap memory
/// per segment stays small even for large indices.
pub const SORT_CURSOR_MAX_FIELDS: usize = 8;

/// Encoded value kind for a v2 field. Recorded in the per-field
/// descriptor so a reader can pair the on-disk encoded `u64` with the
/// right `FastValue` variant when re-resolving against the fast-field
/// column at search time.
///
/// Numeric values 5..255 are reserved for future variants (e.g. string
/// term ordinals — Wave 18-3). Bumping past 255 requires a v3 format.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ValueKind {
    I64 = 0,
    U64 = 1,
    F64 = 2,
    Date = 3,
    DateNanos = 4,
}

impl ValueKind {
    fn to_byte(self) -> u8 {
        self as u8
    }

    fn from_byte(b: u8) -> crate::Result<Self> {
        match b {
            0 => Ok(ValueKind::I64),
            1 => Ok(ValueKind::U64),
            2 => Ok(ValueKind::F64),
            3 => Ok(ValueKind::Date),
            4 => Ok(ValueKind::DateNanos),
            other => Err(DataCorruption::comment_only(format!(
                "sort cursor v2: invalid value_kind byte {other}"
            ))
            .into()),
        }
    }
}

#[inline]
fn missing_bitmap_byte_count(max_doc: u32) -> usize {
    ((max_doc as usize) + 7) / 8
}

#[inline]
fn missing_bitmap_get(bm: &[u8], doc_id: u32) -> bool {
    let byte_idx = (doc_id / 8) as usize;
    let bit_idx = (doc_id % 8) as u8;
    match bm.get(byte_idx) {
        Some(b) => (b >> bit_idx) & 1 == 1,
        None => false,
    }
}

#[inline]
fn missing_bitmap_set(bm: &mut [u8], doc_id: u32) {
    let byte_idx = (doc_id / 8) as usize;
    let bit_idx = (doc_id % 8) as u8;
    bm[byte_idx] |= 1 << bit_idx;
}

/// Multi-field analogue of the v1 `sort_key`. Lex-compares two doc-id
/// rows whose per-field encoded values are tagged with `Option<u64>`,
/// honouring the per-field `Order` and ES's `missing="_last"` rule
/// (a `None` always sorts after a `Some(_)` regardless of order).
fn sort_key_multi(
    a_doc: DocId,
    a_vals: &[Option<u64>],
    b_doc: DocId,
    b_vals: &[Option<u64>],
    orders: &[Order],
) -> Ordering {
    debug_assert_eq!(a_vals.len(), b_vals.len());
    debug_assert_eq!(a_vals.len(), orders.len());
    for i in 0..a_vals.len() {
        let av = a_vals[i];
        let bv = b_vals[i];
        match (av, bv) {
            (None, None) => continue,
            (None, Some(_)) => return Ordering::Greater,
            (Some(_), None) => return Ordering::Less,
            (Some(au), Some(bu)) => {
                let cmp = match orders[i] {
                    Order::Asc => au.cmp(&bu),
                    Order::Desc => bu.cmp(&au),
                };
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
        }
    }
    a_doc.cmp(&b_doc)
}

/// Multi-field auxiliary sort cursor (Wave 18-1, v2 wire format).
///
/// Owns a permutation of the segment's live `DocId`s and a row-major
/// table of encoded `u64` values such that lex comparison of consecutive
/// rows respects each field's [`Order`].  Missing values are tracked
/// out-of-band in `missing_bitmaps` (one bitmap per field, doc-id-
/// indexed) and follow ES's `missing="_last"` rule.
///
/// On-disk layout:
///
/// ```text
///   u32  SORT_CURSOR_MAGIC                         (LE)
///   u8   version (= SORT_CURSOR_VERSION_V2)
///   u8   flags    (SORT_CURSOR_FLAG_MULTI_FIELD set)
///   u16  reserved (= 0)
///   u8   num_fields ∈ 1..=SORT_CURSOR_MAX_FIELDS
///   for each field:
///     u16  field_name_len (UTF-8 byte length)
///     [u8] field_name
///     u8   order   (0=Asc, 1=Desc)
///     u8   value_kind
///   u32  num_docs
///   [u32; num_docs]                       doc_ids permutation
///   [u64; num_docs * num_fields]          values (row-major)
///   for each field:
///     [u8; ⌈num_docs / 8⌉]                missing_bitmap (doc-id ordered)
///   u32  SORT_CURSOR_MAGIC trailing magic (LE)
/// ```
#[derive(Clone, Debug)]
pub struct SortCursorIndexV2 {
    fields: Vec<(String, Order, ValueKind)>,
    doc_ids: Vec<DocId>,
    /// Row-major: `values[cursor_idx * num_fields + field_idx]`.
    /// When the underlying doc has a missing value, the slot is set to
    /// `0` and `missing_bitmaps[field_idx][doc_id]` is set; consult
    /// [`SortCursorIndexV2::value`] rather than reading `values` directly.
    values: Vec<u64>,
    /// One bitmap per field, indexed by **doc id** (not cursor position).
    /// Length: `missing_bitmap_byte_count(max_doc)`.
    missing_bitmaps: Vec<Vec<u8>>,
    /// `max_doc` at build time. Required because the missing bitmap is
    /// addressed by doc-id, not by cursor position.
    max_doc: u32,
}

impl SortCursorIndexV2 {
    /// Builds a v2 cursor from per-field encoded `Option<u64>` value
    /// arrays. Each value array has length `max_doc as usize` and is
    /// indexed by doc id.
    ///
    /// Returns an error when `fields.is_empty()` or
    /// `fields.len() > SORT_CURSOR_MAX_FIELDS`, or when any per-field
    /// value array length disagrees with `max_doc`.
    pub fn build_from_columns(
        fields: Vec<(String, Order, ValueKind)>,
        per_field_values: Vec<Vec<Option<u64>>>,
        max_doc: u32,
    ) -> crate::Result<Self> {
        if fields.is_empty() {
            return Err(crate::TantivyError::SchemaError(
                "sort cursor v2: at least one field is required".to_string(),
            ));
        }
        if fields.len() > SORT_CURSOR_MAX_FIELDS {
            return Err(crate::TantivyError::SchemaError(format!(
                "sort cursor v2: at most {SORT_CURSOR_MAX_FIELDS} fields supported, got {}",
                fields.len()
            )));
        }
        if per_field_values.len() != fields.len() {
            return Err(crate::TantivyError::SchemaError(format!(
                "sort cursor v2: per-field value array count {} != fields count {}",
                per_field_values.len(),
                fields.len()
            )));
        }
        for (idx, vals) in per_field_values.iter().enumerate() {
            if vals.len() != max_doc as usize {
                return Err(crate::TantivyError::SchemaError(format!(
                    "sort cursor v2: field #{idx} has {} values, expected max_doc={max_doc}",
                    vals.len()
                )));
            }
        }
        let num_fields = fields.len();
        let orders: Vec<Order> = fields.iter().map(|(_, o, _)| *o).collect();

        // Sort permutation by lex (with per-field Order + missing-last).
        let mut keyed: Vec<(DocId, Vec<Option<u64>>)> = (0..max_doc)
            .map(|doc_id| {
                let row: Vec<Option<u64>> = per_field_values
                    .iter()
                    .map(|col| col[doc_id as usize])
                    .collect();
                (doc_id, row)
            })
            .collect();
        keyed.sort_by(|a, b| sort_key_multi(a.0, &a.1, b.0, &b.1, &orders));

        // Row-major values + per-field missing bitmaps.
        let bm_len = missing_bitmap_byte_count(max_doc);
        let mut missing_bitmaps: Vec<Vec<u8>> = vec![vec![0u8; bm_len]; num_fields];
        for (field_idx, col) in per_field_values.iter().enumerate() {
            for (doc_id, slot) in col.iter().enumerate() {
                if slot.is_none() {
                    missing_bitmap_set(&mut missing_bitmaps[field_idx], doc_id as u32);
                }
            }
        }

        let mut doc_ids: Vec<DocId> = Vec::with_capacity(max_doc as usize);
        let mut values: Vec<u64> = Vec::with_capacity(max_doc as usize * num_fields);
        for (doc_id, row) in keyed {
            doc_ids.push(doc_id);
            for slot in row {
                values.push(slot.unwrap_or(0));
            }
        }

        Ok(Self {
            fields,
            doc_ids,
            values,
            missing_bitmaps,
            max_doc,
        })
    }

    /// Number of doc ids in the cursor (== `max_doc` at build time).
    pub fn len(&self) -> usize {
        self.doc_ids.len()
    }

    /// Returns `true` if the cursor is empty.
    pub fn is_empty(&self) -> bool {
        self.doc_ids.is_empty()
    }

    /// `(name, order, value_kind)` triples in declaration order.
    pub fn fields(&self) -> &[(String, Order, ValueKind)] {
        &self.fields
    }

    /// Primary field name (== `self.fields[0].0`).
    pub fn primary_field(&self) -> &str {
        &self.fields[0].0
    }

    /// Primary order (== `self.fields[0].1`).
    pub fn primary_order(&self) -> Order {
        self.fields[0].1
    }

    /// Iterates the doc ids in lex sort order.
    pub fn iter(&self) -> impl Iterator<Item = DocId> + '_ {
        self.doc_ids.iter().copied()
    }

    /// Returns the underlying `Vec<DocId>` (already lex-sorted).
    pub fn doc_ids(&self) -> &[DocId] {
        &self.doc_ids
    }

    /// Encoded `u64` value for the `cursor_idx`-th cursor position and
    /// the `field_idx`-th field. Returns `None` when the underlying doc
    /// has a missing value for that field (missing-last sort applies).
    ///
    /// # Panics
    ///
    /// Panics if `cursor_idx >= self.len()` or `field_idx >=
    /// self.fields().len()` — these are programmer errors at the
    /// collector layer.
    pub fn value(&self, cursor_idx: usize, field_idx: usize) -> Option<u64> {
        let num_fields = self.fields.len();
        assert!(cursor_idx < self.doc_ids.len(), "cursor_idx out of range");
        assert!(field_idx < num_fields, "field_idx out of range");
        let doc_id = self.doc_ids[cursor_idx];
        if missing_bitmap_get(&self.missing_bitmaps[field_idx], doc_id) {
            None
        } else {
            Some(self.values[cursor_idx * num_fields + field_idx])
        }
    }

    /// Convenience: returns the full encoded tuple at `cursor_idx`.
    /// Each `Option<u64>` is `None` when the corresponding field is
    /// missing for the doc.
    pub fn tuple(&self, cursor_idx: usize) -> Vec<Option<u64>> {
        (0..self.fields.len())
            .map(|f| self.value(cursor_idx, f))
            .collect()
    }

    /// `max_doc` recorded at build time.
    pub fn max_doc(&self) -> u32 {
        self.max_doc
    }

    /// Serialises the cursor to `writer` using the v2 layout described
    /// in the struct doc-comment.
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        SORT_CURSOR_MAGIC.serialize(writer)?;
        SORT_CURSOR_VERSION_V2.serialize(writer)?;
        SORT_CURSOR_FLAG_MULTI_FIELD.serialize(writer)?;
        // 2 reserved bytes, kept zero.
        0u16.serialize(writer)?;
        let num_fields = u8::try_from(self.fields.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "sort cursor v2: num_fields exceeds u8",
            )
        })?;
        num_fields.serialize(writer)?;
        for (name, order, kind) in &self.fields {
            let bytes = name.as_bytes();
            let name_len = u16::try_from(bytes.len()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "sort cursor v2: field name exceeds 65,535 bytes",
                )
            })?;
            name_len.serialize(writer)?;
            writer.write_all(bytes)?;
            let order_byte: u8 = match order {
                Order::Asc => 0,
                Order::Desc => 1,
            };
            order_byte.serialize(writer)?;
            kind.to_byte().serialize(writer)?;
        }
        let num_docs = u32::try_from(self.doc_ids.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "sort cursor v2: doc count exceeds u32::MAX",
            )
        })?;
        num_docs.serialize(writer)?;
        for &doc_id in &self.doc_ids {
            doc_id.serialize(writer)?;
        }
        for &v in &self.values {
            v.serialize(writer)?;
        }
        for bm in &self.missing_bitmaps {
            writer.write_all(bm)?;
        }
        SORT_CURSOR_MAGIC.serialize(writer)?;
        Ok(())
    }

    /// Deserialises a v2 cursor from a [`FileSlice`].
    pub fn open(slice: FileSlice) -> crate::Result<Self> {
        let bytes = slice.read_bytes()?;
        Self::from_bytes(bytes.as_slice())
    }

    /// Parse a v2 cursor from an in-memory byte buffer.
    pub fn from_bytes(buf: &[u8]) -> crate::Result<Self> {
        let mut reader = buf;
        let magic = u32::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor v2: failed to read magic: {e}"))
        })?;
        if magic != SORT_CURSOR_MAGIC {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor v2: bad magic 0x{magic:08x}, expected 0x{SORT_CURSOR_MAGIC:08x}"
            ))
            .into());
        }
        let version = u8::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor v2: failed to read version: {e}"))
        })?;
        if version != SORT_CURSOR_VERSION_V2 {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor v2: unsupported version {version}, expected {SORT_CURSOR_VERSION_V2}"
            ))
            .into());
        }
        let flags = u8::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor v2: failed to read flags: {e}"))
        })?;
        if flags & SORT_CURSOR_FLAG_MULTI_FIELD == 0 {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor v2: MULTI_FIELD flag bit 0 not set (flags=0x{flags:02x})"
            ))
            .into());
        }
        let _reserved = u16::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor v2: failed to read reserved: {e}"))
        })?;
        let num_fields = u8::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!(
                "sort cursor v2: failed to read num_fields: {e}"
            ))
        })? as usize;
        if num_fields == 0 || num_fields > SORT_CURSOR_MAX_FIELDS {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor v2: num_fields={num_fields} out of range 1..={SORT_CURSOR_MAX_FIELDS}"
            ))
            .into());
        }
        let mut fields: Vec<(String, Order, ValueKind)> = Vec::with_capacity(num_fields);
        for i in 0..num_fields {
            let name_len = u16::deserialize(&mut reader).map_err(|e| {
                DataCorruption::comment_only(format!(
                    "sort cursor v2: failed to read field #{i} name_len: {e}"
                ))
            })? as usize;
            if reader.len() < name_len {
                return Err(DataCorruption::comment_only(format!(
                    "sort cursor v2: field #{i} name truncated"
                ))
                .into());
            }
            let (name_bytes, rest) = reader.split_at(name_len);
            let name = std::str::from_utf8(name_bytes)
                .map_err(|e| {
                    DataCorruption::comment_only(format!(
                        "sort cursor v2: field #{i} name not utf8: {e}"
                    ))
                })?
                .to_string();
            reader = rest;
            let order_byte = u8::deserialize(&mut reader).map_err(|e| {
                DataCorruption::comment_only(format!(
                    "sort cursor v2: failed to read field #{i} order: {e}"
                ))
            })?;
            let order = match order_byte {
                0 => Order::Asc,
                1 => Order::Desc,
                other => {
                    return Err(DataCorruption::comment_only(format!(
                        "sort cursor v2: field #{i} invalid order byte {other}"
                    ))
                    .into());
                }
            };
            let kind_byte = u8::deserialize(&mut reader).map_err(|e| {
                DataCorruption::comment_only(format!(
                    "sort cursor v2: failed to read field #{i} value_kind: {e}"
                ))
            })?;
            let kind = ValueKind::from_byte(kind_byte)?;
            fields.push((name, order, kind));
        }
        let num_docs = u32::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!("sort cursor v2: failed to read num_docs: {e}"))
        })? as usize;
        let bm_byte_count = ((num_docs) + 7) / 8;
        let needed = (num_docs.checked_mul(4))
            .and_then(|x| x.checked_add(num_docs.checked_mul(num_fields)?.checked_mul(8)?))
            .and_then(|x| x.checked_add(bm_byte_count.checked_mul(num_fields)?))
            .and_then(|x| x.checked_add(4));
        let Some(needed) = needed else {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor v2: num_docs ({num_docs}) overflow"
            ))
            .into());
        };
        if reader.len() < needed {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor v2: payload truncated (need {needed} bytes, have {})",
                reader.len()
            ))
            .into());
        }
        let mut doc_ids: Vec<DocId> = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            doc_ids.push(u32::deserialize(&mut reader).map_err(|e| {
                DataCorruption::comment_only(format!("sort cursor v2: failed to read doc_id: {e}"))
            })?);
        }
        let mut values: Vec<u64> = Vec::with_capacity(num_docs * num_fields);
        for _ in 0..(num_docs * num_fields) {
            values.push(u64::deserialize(&mut reader).map_err(|e| {
                DataCorruption::comment_only(format!("sort cursor v2: failed to read value: {e}"))
            })?);
        }
        let mut missing_bitmaps: Vec<Vec<u8>> = Vec::with_capacity(num_fields);
        for _ in 0..num_fields {
            if reader.len() < bm_byte_count {
                return Err(DataCorruption::comment_only(
                    "sort cursor v2: missing bitmap truncated".to_string(),
                )
                .into());
            }
            let (bm_bytes, rest) = reader.split_at(bm_byte_count);
            missing_bitmaps.push(bm_bytes.to_vec());
            reader = rest;
        }
        let trailing_magic = u32::deserialize(&mut reader).map_err(|e| {
            DataCorruption::comment_only(format!(
                "sort cursor v2: failed to read trailing magic: {e}"
            ))
        })?;
        if trailing_magic != SORT_CURSOR_MAGIC {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor v2: bad trailing magic 0x{trailing_magic:08x}"
            ))
            .into());
        }
        let max_doc = u32::try_from(num_docs).map_err(|_| {
            DataCorruption::comment_only(format!(
                "sort cursor v2: num_docs ({num_docs}) exceeds u32::MAX"
            ))
        })?;
        Ok(Self {
            fields,
            doc_ids,
            values,
            missing_bitmaps,
            max_doc,
        })
    }
}

/// Reader-side dispatch type. Peeks the version byte after the magic
/// prefix and returns either a v1 [`SortCursorIndex`] or a v2
/// [`SortCursorIndexV2`]. Existing code that only deals with v1 keeps
/// calling [`SortCursorIndex::open`] directly; new readers that need to
/// transparently handle both formats use [`SortCursorAny::open`].
#[derive(Clone, Debug)]
pub enum SortCursorAny {
    V1(SortCursorIndex),
    V2(SortCursorIndexV2),
}

impl SortCursorAny {
    /// Opens a v1-or-v2 cursor by peeking the version byte.
    pub fn open(slice: FileSlice) -> crate::Result<Self> {
        let bytes = slice.read_bytes()?;
        Self::from_bytes(bytes.as_slice())
    }

    /// Parse from an in-memory byte buffer, dispatching by version.
    pub fn from_bytes(buf: &[u8]) -> crate::Result<Self> {
        if buf.len() < 5 {
            return Err(DataCorruption::comment_only(
                "sort cursor: file too short to determine version".to_string(),
            )
            .into());
        }
        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic != SORT_CURSOR_MAGIC {
            return Err(DataCorruption::comment_only(format!(
                "sort cursor: bad magic 0x{magic:08x}, expected 0x{SORT_CURSOR_MAGIC:08x}"
            ))
            .into());
        }
        match buf[4] {
            SORT_CURSOR_VERSION => Ok(SortCursorAny::V1(SortCursorIndex::from_bytes(buf)?)),
            SORT_CURSOR_VERSION_V2 => {
                Ok(SortCursorAny::V2(SortCursorIndexV2::from_bytes(buf)?))
            }
            other => Err(DataCorruption::comment_only(format!(
                "sort cursor: unsupported version {other}"
            ))
            .into()),
        }
    }

    /// Convenience — true when the underlying cursor is v1 (single field).
    pub fn is_v1(&self) -> bool {
        matches!(self, SortCursorAny::V1(_))
    }

    /// Convenience — true when the underlying cursor is v2 (multi field).
    pub fn is_v2(&self) -> bool {
        matches!(self, SortCursorAny::V2(_))
    }
}

/// Builds a v2 sort cursor by reading each field's first value through
/// the segment's `FastFieldReaders`. Returns an error if any field
/// cannot be resolved as a numeric / date fast field, or if the field
/// count is outside `1..=SORT_CURSOR_MAX_FIELDS`.
pub fn build_sort_cursor_v2_from_fast_fields(
    readers: &crate::fastfield::FastFieldReaders,
    fields: &[(String, Order)],
    num_docs: u32,
) -> crate::Result<SortCursorIndexV2> {
    if fields.is_empty() {
        return Err(crate::TantivyError::SchemaError(
            "sort cursor v2: at least one field is required".to_string(),
        ));
    }
    if fields.len() > SORT_CURSOR_MAX_FIELDS {
        return Err(crate::TantivyError::SchemaError(format!(
            "sort cursor v2: at most {SORT_CURSOR_MAX_FIELDS} fields supported, got {}",
            fields.len()
        )));
    }
    let mut field_specs: Vec<(String, Order, ValueKind)> = Vec::with_capacity(fields.len());
    let mut per_field_values: Vec<Vec<Option<u64>>> = Vec::with_capacity(fields.len());
    for (field, order) in fields {
        let (kind, values) = read_field_to_u64(readers, field, num_docs)?;
        field_specs.push((field.clone(), *order, kind));
        per_field_values.push(values);
    }
    SortCursorIndexV2::build_from_columns(field_specs, per_field_values, num_docs)
}

/// Reads a fast-field column for `field`, encoding each doc's first
/// value to a `FastValue::to_u64()` slot. Tries `i64`, `u64`, `f64`,
/// then `DateTime` in order — same shape as the v1 helper.
fn read_field_to_u64(
    readers: &crate::fastfield::FastFieldReaders,
    field: &str,
    num_docs: u32,
) -> crate::Result<(ValueKind, Vec<Option<u64>>)> {
    use columnar::MonotonicallyMappableToU64;
    if let Some(col) = readers.column_opt::<i64>(field)? {
        let values: Vec<Option<u64>> = (0..num_docs)
            .map(|d| col.first(d).map(<i64 as MonotonicallyMappableToU64>::to_u64))
            .collect();
        return Ok((ValueKind::I64, values));
    }
    if let Some(col) = readers.column_opt::<u64>(field)? {
        let values: Vec<Option<u64>> = (0..num_docs)
            .map(|d| col.first(d).map(<u64 as MonotonicallyMappableToU64>::to_u64))
            .collect();
        return Ok((ValueKind::U64, values));
    }
    if let Some(col) = readers.column_opt::<f64>(field)? {
        let values: Vec<Option<u64>> = (0..num_docs)
            .map(|d| col.first(d).map(<f64 as MonotonicallyMappableToU64>::to_u64))
            .collect();
        return Ok((ValueKind::F64, values));
    }
    if let Some(col) = readers.column_opt::<DateTime>(field)? {
        let values: Vec<Option<u64>> = (0..num_docs)
            .map(|d| col.first(d).map(<DateTime as MonotonicallyMappableToU64>::to_u64))
            .collect();
        return Ok((ValueKind::Date, values));
    }
    Err(crate::TantivyError::SchemaError(format!(
        "Field `{field}` is not a numeric or date fast field; cannot build v2 cursor"
    )))
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
/// based on `segment.index().settings().sort_by_field` (v1, single
/// field) or `sort_by_fields` (v2, multi field).
///
/// **FerroSearch extension (Wave 15 / Wave 18-1).** Called by
/// `index_documents` and `SingleSegmentIndexWriter::finalize` after
/// the main segment files have been laid out on disk and `with_max_doc`
/// has been applied. Returns the field names whose cursor was
/// successfully written (single-field name for v1; the **primary**
/// field name for v2), so the caller can advertise them in
/// `SegmentMeta::sort_cursor_fields` via
/// [`crate::index::Segment::with_sort_cursor_fields`].
///
/// A no-op (`Ok(Vec::new())`) when neither `sort_by_field` nor
/// `sort_by_fields` is configured — keeps the indexing path zero-cost
/// for indices that do not opt in.
///
/// `IndexSettings::validate_sort_settings()` (called from
/// `IndexBuilder::create`) guarantees `sort_by_field` and
/// `sort_by_fields` are mutually exclusive — this function trusts that
/// invariant and prefers `sort_by_fields` when both happen to be set
/// at runtime (via direct struct mutation).
pub fn build_and_write_sort_cursors(
    segment: &mut crate::index::Segment,
) -> crate::Result<Vec<String>> {
    use common::TerminatingWrite;

    let settings = segment.index().settings().clone();
    let max_doc = segment.meta().max_doc();
    if max_doc == 0 {
        // An empty segment carries no doc ids; skip writing a cursor.
        return Ok(Vec::new());
    }

    // Wave 18-1: prefer multi-field (v2) when configured.
    if let Some(fields) = settings.sort_by_fields.as_ref() {
        if fields.is_empty() {
            return Ok(Vec::new());
        }
        let reader = crate::index::SegmentReader::open(segment)?;
        let pairs: Vec<(String, Order)> =
            fields.iter().map(|f| (f.field.clone(), f.order)).collect();
        let cursor = build_sort_cursor_v2_from_fast_fields(reader.fast_fields(), &pairs, max_doc)?;
        // The on-disk file is keyed by the **primary** field name so a
        // later reader's `Segment::meta().sort_cursor_fields()` listing
        // (which carries field names verbatim) finds it.
        let primary_field = pairs[0].0.clone();
        let mut writer = segment.open_sort_cursor_write(&primary_field)?;
        cursor.write(&mut writer)?;
        writer.terminate()?;
        // Same Phase H-5 reasoning as the v1 path below.
        segment.index().directory().sync_directory()?;
        return Ok(vec![primary_field]);
    }

    let sort_by = match settings.sort_by_field {
        Some(sb) => sb,
        None => return Ok(Vec::new()),
    };
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
    // **FerroSearch Wave 15 Phase H-5.** Sync the directory so the cursor
    // file's directory entry is durably visible before the caller
    // publishes `SegmentMeta::sort_cursor_fields = [<field>]`.  Phase H-4
    // EC2 stress-bench (Rally http_logs `bulk_indexing_clients=8` over
    // 8 indices × 5 shards) hit 2,968 `FileDoesNotExist` errors in 5
    // minutes because the auto-refresh that follows each bulk commit
    // tried to load segments whose meta advertised the cursor before
    // the cursor file's `dirent` was visible to a fresh `MmapDirectory`
    // open.  Without the sync, `terminate()`'s buffered writes can sit
    // in the page cache while the SegmentMeta is published — readers
    // then fail with `OpenReadError(FileDoesNotExist)`.
    //
    // The sync cost is bounded (one fsync of the segment dir per
    // segment commit when an index sort is configured), and aligns
    // with how `save_metas` already syncs the dir before the atomic
    // meta.json swap.
    segment.index().directory().sync_directory()?;
    Ok(vec![sort_by.field])
}

/// Builds and persists the auxiliary sort cursor file for a single,
/// caller-supplied (field, order) pair, **bypassing**
/// `IndexSettings::sort_by_field`.
///
/// **FerroSearch extension (Wave 17-2 backfill).** Unlike
/// [`build_and_write_sort_cursors`], which reads the index-level
/// `sort_by_field` setting and is invoked at commit / merge time, this
/// helper takes the field + order from the caller and is the building
/// block for `IndexWriter::backfill_sort_cursor` — the post-creation
/// "operator just enabled `index.sort.field`, please backfill the
/// existing segments" admin path.
///
/// Returns `Ok(true)` when a cursor file was written, `Ok(false)` when
/// the segment had `max_doc == 0` (skipped), and an `Err` when the fast
/// field column for `field` could not be resolved (e.g. the field is
/// not declared FAST in the schema, or its type is not numeric/date —
/// see [`build_sort_cursor_from_fast_fields`]).
///
/// On success, syncs the directory so the cursor file's directory
/// entry is durably visible before the caller publishes the updated
/// `SegmentMeta::sort_cursor_fields`.  Mirrors the Phase H-5 sync rule
/// in [`build_and_write_sort_cursors`].
pub fn build_and_write_sort_cursor_for(
    segment: &mut crate::index::Segment,
    field: &str,
    order: crate::index::Order,
) -> crate::Result<bool> {
    use common::TerminatingWrite;

    let max_doc = segment.meta().max_doc();
    if max_doc == 0 {
        return Ok(false);
    }
    let reader = crate::index::SegmentReader::open(segment)?;
    let cursor = build_sort_cursor_from_fast_fields(reader.fast_fields(), field, order, max_doc)?;
    let mut writer = segment.open_sort_cursor_write(field)?;
    cursor.write(&mut writer)?;
    writer.terminate()?;
    // Same Phase H-5 reasoning as `build_and_write_sort_cursors`: sync
    // the directory entry so a later reader's `MmapDirectory` open
    // doesn't race the buffered cursor file write.
    segment.index().directory().sync_directory()?;
    Ok(true)
}

/// Builds and persists the auxiliary **multi-field** (v2) sort cursor
/// file for a single, caller-supplied list of `(field, order)` pairs,
/// **bypassing** `IndexSettings::sort_by_fields`.
///
/// **FerroSearch Wave 18-1 backfill primitive.** This is the
/// multi-field analogue of [`build_and_write_sort_cursor_for`] —
/// designed to be the building block for
/// `IndexWriter::backfill_sort_cursor_v2`, the post-creation
/// "operator just changed `index.sort.field` to a multi-field array,
/// please backfill the existing segments" admin path.
///
/// The on-disk file is keyed by the **primary** field name (the
/// first entry in `pairs`), matching the `SegmentMeta::sort_cursor_fields`
/// naming convention used by the v1 path.
///
/// Returns `Ok(true)` when a cursor file was written, `Ok(false)`
/// when `max_doc == 0` (skipped), and an `Err` when the field list
/// is empty / exceeds `SORT_CURSOR_MAX_FIELDS` / contains a non-fast-
/// or non-numeric field. Returns the primary field name via
/// [`SortCursorIndexV2::primary_field`] semantics — callers should
/// take the first entry of `pairs`.
pub fn build_and_write_sort_cursor_v2_for(
    segment: &mut crate::index::Segment,
    pairs: &[(String, crate::index::Order)],
) -> crate::Result<bool> {
    use common::TerminatingWrite;

    if pairs.is_empty() {
        return Err(crate::TantivyError::InvalidArgument(
            "sort cursor v2 backfill: at least one field is required".to_string(),
        ));
    }
    if pairs.len() > SORT_CURSOR_MAX_FIELDS {
        return Err(crate::TantivyError::InvalidArgument(format!(
            "sort cursor v2 backfill: at most {SORT_CURSOR_MAX_FIELDS} fields, got {}",
            pairs.len()
        )));
    }
    let max_doc = segment.meta().max_doc();
    if max_doc == 0 {
        return Ok(false);
    }
    let reader = crate::index::SegmentReader::open(segment)?;
    let cursor = build_sort_cursor_v2_from_fast_fields(reader.fast_fields(), pairs, max_doc)?;
    let primary_field = &pairs[0].0;
    let mut writer = segment.open_sort_cursor_write(primary_field)?;
    cursor.write(&mut writer)?;
    writer.terminate()?;
    // Same Phase H-5 reasoning as the v1 backfill: ensure the dirent
    // is durably visible before the caller publishes the updated
    // `SegmentMeta::sort_cursor_fields`.
    segment.index().directory().sync_directory()?;
    Ok(true)
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

    // -----------------------------------------------------------------------
    // Wave 18-1: v2 (multi-field) sort cursor tests.
    // -----------------------------------------------------------------------

    /// Test-only helper: build a v2 cursor from raw FastValue-encoded
    /// `Option<u64>` value columns, bypassing the FastFieldReaders path.
    fn build_v2_from_values(
        fields: Vec<(&str, Order, ValueKind)>,
        per_field_values: Vec<Vec<Option<u64>>>,
        max_doc: u32,
    ) -> SortCursorIndexV2 {
        let owned: Vec<(String, Order, ValueKind)> = fields
            .into_iter()
            .map(|(n, o, k)| (n.to_string(), o, k))
            .collect();
        SortCursorIndexV2::build_from_columns(owned, per_field_values, max_doc)
            .expect("v2 build should succeed")
    }

    fn roundtrip_v2(cursor: &SortCursorIndexV2) -> SortCursorIndexV2 {
        let mut buf = Vec::new();
        cursor.write(&mut buf).expect("v2 write should succeed");
        SortCursorIndexV2::from_bytes(&buf).expect("v2 parse should succeed")
    }

    /// Single-field-as-v2 sort matches the v1 sort permutation.
    /// Encodes `i64` via `FastValue::to_u64` (xor with 1<<63) so plain
    /// `u64::cmp` in the v2 sort_key recovers the natural i64 order.
    #[test]
    fn v2_single_field_matches_v1_for_i64_desc() {
        use columnar::MonotonicallyMappableToU64;
        // doc 0 = 30, doc 1 = 10, doc 2 = 20, doc 3 = missing
        // desc order = [0, 2, 1, 3] (missing-last)
        let v: Vec<Option<u64>> = vec![Some(30i64), Some(10), Some(20)]
            .into_iter()
            .map(|o| o.map(|v| v.to_u64()))
            .chain(std::iter::once(None))
            .collect();
        let cursor = build_v2_from_values(
            vec![("score", Order::Desc, ValueKind::I64)],
            vec![v],
            4,
        );
        assert_eq!(cursor.doc_ids(), &[0u32, 2, 1, 3]);
        assert_eq!(cursor.primary_field(), "score");
        assert_eq!(cursor.primary_order(), Order::Desc);
    }

    /// Two-field lex sort: primary `ts DESC`, secondary `_id ASC`. Two
    /// docs share the primary value — tie-break must follow secondary
    /// order (NOT raw doc-id order, the Wave 17-3 fallback).
    #[test]
    fn v2_two_field_lex_sort_breaks_tie_by_secondary() {
        use columnar::MonotonicallyMappableToU64;
        // ts:    [100, 100, 200, 50]
        // _id:   [ 7,   3,   5,   9]
        // Wanted desc-ts ASC-_id order:
        //   doc 2 (ts=200, _id=5)
        //   doc 1 (ts=100, _id=3)   ← tie on ts: secondary _id=3 < 7 → doc 1
        //   doc 0 (ts=100, _id=7)
        //   doc 3 (ts=50,  _id=9)
        let ts: Vec<Option<u64>> = vec![100i64, 100, 200, 50]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let id: Vec<Option<u64>> = vec![7i64, 3, 5, 9]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let cursor = build_v2_from_values(
            vec![
                ("ts", Order::Desc, ValueKind::I64),
                ("_id", Order::Asc, ValueKind::I64),
            ],
            vec![ts, id],
            4,
        );
        assert_eq!(cursor.doc_ids(), &[2u32, 1, 0, 3]);
    }

    /// Four-field cursor exercises the larger header roundtrip and a
    /// chain of tie-breakers. Each successive field decides for some
    /// pair of docs.
    #[test]
    fn v2_four_field_lex_sort_and_roundtrip() {
        use columnar::MonotonicallyMappableToU64;
        // 3 docs, 4 fields. All docs tie on f0 (=100) and f1 (=5).
        // Doc 0: (100, 5, 1.0, 10)
        // Doc 1: (100, 5, 1.0,  3)   ← decided by f3 ASC: 3 < 10
        // Doc 2: (100, 5, 2.0,  0)   ← decided by f2 DESC: 2.0 > 1.0
        // Order f0 ASC, f1 DESC, f2 DESC, f3 ASC:
        //   doc 2 (f2=2.0 wins over both)
        //   doc 1 (tie on f0/f1/f2 with doc 0 → f3=3 < 10)
        //   doc 0
        let f0: Vec<Option<u64>> = vec![100i64, 100, 100]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let f1: Vec<Option<u64>> = vec![5i64, 5, 5]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let f2: Vec<Option<u64>> = vec![1.0f64, 1.0, 2.0]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let f3: Vec<Option<u64>> = vec![10i64, 3, 0]
            .into_iter()
            .map(|v| Some(v.to_u64()))
            .collect();
        let cursor = build_v2_from_values(
            vec![
                ("f0", Order::Asc, ValueKind::I64),
                ("f1", Order::Desc, ValueKind::I64),
                ("f2", Order::Desc, ValueKind::F64),
                ("f3", Order::Asc, ValueKind::I64),
            ],
            vec![f0, f1, f2, f3],
            3,
        );
        assert_eq!(cursor.doc_ids(), &[2u32, 1, 0]);

        let restored = roundtrip_v2(&cursor);
        assert_eq!(restored.doc_ids(), cursor.doc_ids());
        assert_eq!(restored.fields().len(), 4);
        assert_eq!(restored.fields()[0].0, "f0");
        assert_eq!(restored.fields()[2].2, ValueKind::F64);
        assert_eq!(restored.fields()[3].1, Order::Asc);
    }

    /// Missing-last semantics in multi-field. A doc whose PRIMARY is
    /// `None` always sorts AFTER any doc with `Some(_)`, regardless of
    /// the order on the primary or any secondary value.
    #[test]
    fn v2_missing_value_sorts_last_at_any_field_position() {
        use columnar::MonotonicallyMappableToU64;
        // doc 0: (Some(100), Some(5))
        // doc 1: (None,      Some(99))   ← primary missing: last
        // doc 2: (Some(50),  None)        ← primary present: before doc 1
        // Expected primary DESC, secondary ASC:
        //   doc 0 (primary=100)
        //   doc 2 (primary=50, secondary missing — still before doc 1)
        //   doc 1 (primary missing — last)
        let f0: Vec<Option<u64>> = vec![Some(100i64), None, Some(50)]
            .into_iter()
            .map(|o| o.map(|v| v.to_u64()))
            .collect();
        let f1: Vec<Option<u64>> = vec![Some(5i64), Some(99), None]
            .into_iter()
            .map(|o| o.map(|v| v.to_u64()))
            .collect();
        let cursor = build_v2_from_values(
            vec![
                ("a", Order::Desc, ValueKind::I64),
                ("b", Order::Asc, ValueKind::I64),
            ],
            vec![f0, f1],
            3,
        );
        assert_eq!(cursor.doc_ids(), &[0u32, 2, 1]);
        // Per-position queries return None at the right positions.
        assert!(cursor.value(0, 0).is_some());
        assert!(cursor.value(1, 1).is_none()); // doc 2 secondary
        assert!(cursor.value(2, 0).is_none()); // doc 1 primary
    }

    /// All fields tie → final stable tie-breaker is doc-id ASC. Mirrors
    /// the v1 `ties_break_by_doc_id_for_determinism` invariant.
    #[test]
    fn v2_ties_break_by_doc_id_when_all_fields_equal() {
        use columnar::MonotonicallyMappableToU64;
        let f0: Vec<Option<u64>> = vec![Some(7i64), Some(7), Some(7), Some(1)]
            .into_iter()
            .map(|o| o.map(|v| v.to_u64()))
            .collect();
        let f1: Vec<Option<u64>> = vec![Some(0u64), Some(0), Some(0), Some(0)]
            .into_iter()
            .collect();
        let cursor = build_v2_from_values(
            vec![
                ("k", Order::Asc, ValueKind::I64),
                ("z", Order::Asc, ValueKind::U64),
            ],
            vec![f0, f1],
            4,
        );
        // 1 first (smallest f0), then ties on (7, 0) broken by doc_id ASC.
        assert_eq!(cursor.doc_ids(), &[3u32, 0, 1, 2]);
    }

    /// Roundtrip with a longer doc count and 3 fields exercises the
    /// row-major values vector and the per-field missing bitmaps.
    /// Walks the restored cursor and verifies lex monotonicity.
    #[test]
    fn v2_write_open_roundtrip_realistic() {
        use columnar::MonotonicallyMappableToU64;
        let n: u32 = 1_000;
        // f0: pseudo-random i64 with collisions every 17 docs.
        // f1: monotonically growing _id-like u64.
        // f2: occasional missing f64 (every 13th doc).
        let f0: Vec<Option<u64>> = (0..n)
            .map(|i| Some((((i as i64) % 17) * 1000).to_u64()))
            .collect();
        let f1: Vec<Option<u64>> = (0..n).map(|i| Some((i as u64).to_u64())).collect();
        let f2: Vec<Option<u64>> = (0..n)
            .map(|i| {
                if i % 13 == 0 {
                    None
                } else {
                    Some(((i as f64) / 7.5).to_u64())
                }
            })
            .collect();
        let cursor = build_v2_from_values(
            vec![
                ("ts", Order::Desc, ValueKind::Date),
                ("_id", Order::Asc, ValueKind::U64),
                ("score", Order::Desc, ValueKind::F64),
            ],
            vec![f0, f1, f2],
            n,
        );
        let restored = roundtrip_v2(&cursor);
        assert_eq!(restored.len(), n as usize);
        assert_eq!(restored.doc_ids(), cursor.doc_ids());
        assert_eq!(restored.fields(), cursor.fields());

        // Walk the restored cursor and check lex monotonicity.
        let orders = [Order::Desc, Order::Asc, Order::Desc];
        let mut prev: Option<Vec<Option<u64>>> = None;
        for cursor_idx in 0..restored.len() {
            let row: Vec<Option<u64>> = (0..restored.fields().len())
                .map(|fi| restored.value(cursor_idx, fi))
                .collect();
            if let Some(p) = &prev {
                let order_cmp = sort_key_multi(0, p, 1, &row, &orders);
                assert!(
                    order_cmp != Ordering::Greater,
                    "non-monotonic at cursor_idx {cursor_idx}: {p:?} > {row:?}"
                );
            }
            prev = Some(row);
        }
    }

    /// Reader rejects a v2 file whose header has the MULTI_FIELD flag
    /// bit unset — defends against accidental v1/v2 confusion.
    #[test]
    fn v2_corruption_rejects_zero_flags() {
        let cursor = build_v2_from_values(
            vec![("f", Order::Asc, ValueKind::I64)],
            vec![vec![Some(1u64), Some(2), Some(3)]],
            3,
        );
        let mut buf = Vec::new();
        cursor.write(&mut buf).unwrap();
        // Header layout: magic(4) version(1) flags(1) reserved(2) num_fields(1).
        // Flip flags → 0 so the MULTI_FIELD bit is unset.
        buf[5] = 0;
        let err = SortCursorIndexV2::from_bytes(&buf).unwrap_err();
        assert!(
            err.to_string().contains("MULTI_FIELD flag"),
            "expected MULTI_FIELD-flag error, got: {err}"
        );
    }

    /// Build path rejects > 8 fields. Reader rejects a hand-corrupted
    /// `num_fields = 9` byte too.
    #[test]
    fn v2_max_fields_cap_enforced_on_build_and_read() {
        // Build path: 9 fields → error.
        let too_many: Vec<(String, Order, ValueKind)> = (0..9)
            .map(|i| (format!("f{i}"), Order::Asc, ValueKind::I64))
            .collect();
        let too_many_vals: Vec<Vec<Option<u64>>> = (0..9)
            .map(|_| vec![Some(0u64), Some(1), Some(2)])
            .collect();
        let err = SortCursorIndexV2::build_from_columns(too_many, too_many_vals, 3).unwrap_err();
        assert!(
            err.to_string().contains("at most"),
            "expected cap error, got: {err}"
        );

        // Reader path: corrupt num_fields → 9 in a serialised buffer.
        let cursor = build_v2_from_values(
            vec![("f", Order::Asc, ValueKind::I64)],
            vec![vec![Some(1u64), Some(2)]],
            2,
        );
        let mut buf = Vec::new();
        cursor.write(&mut buf).unwrap();
        // num_fields sits at offset 8 (magic 4 + version 1 + flags 1 + reserved 2).
        buf[8] = 9;
        let err = SortCursorIndexV2::from_bytes(&buf).unwrap_err();
        let msg = err.to_string();
        // The reader trips on either the num_fields cap, downstream
        // header parse failures from the corrupted descriptor count, or
        // trailing-magic mismatch. Any of those is a valid rejection path.
        assert!(
            msg.contains("num_fields")
                || msg.contains("out of range")
                || msg.contains("v2")
                || msg.contains("truncated")
                || msg.contains("trailing"),
            "expected v2-corruption error, got: {msg}"
        );
    }

    /// `SortCursorAny::open` peeks the version byte and returns the
    /// correct variant for v1 and v2 buffers.
    #[test]
    fn sort_cursor_any_dispatches_by_version() {
        // v1 cursor → SortCursorAny::V1
        let v1 = SortCursorIndex::build_from_values(
            "x",
            Order::Asc,
            vec![Some(1i64), Some(2)],
        );
        let mut buf_v1 = Vec::new();
        v1.write(&mut buf_v1).unwrap();
        let any = SortCursorAny::from_bytes(&buf_v1).unwrap();
        assert!(any.is_v1());
        assert!(!any.is_v2());

        // v2 cursor → SortCursorAny::V2
        let v2 = build_v2_from_values(
            vec![
                ("ts", Order::Desc, ValueKind::I64),
                ("id", Order::Asc, ValueKind::U64),
            ],
            vec![vec![Some(10u64), Some(20)], vec![Some(0u64), Some(1)]],
            2,
        );
        let mut buf_v2 = Vec::new();
        v2.write(&mut buf_v2).unwrap();
        let any = SortCursorAny::from_bytes(&buf_v2).unwrap();
        assert!(any.is_v2());
        assert!(!any.is_v1());

        // Unsupported version → error.
        let mut buf_bad = buf_v2.clone();
        buf_bad[4] = 99;
        let err = SortCursorAny::from_bytes(&buf_bad).unwrap_err();
        assert!(
            err.to_string().contains("unsupported version"),
            "expected unsupported version error, got: {err}"
        );
    }

    /// **Wave 18-1 Phase C.** When `IndexSettings::sort_by_fields` is
    /// set, a commit writes a v2 cursor file to disk and the
    /// `SegmentReader::sort_cursor_v2()` accessor returns it. Mirrors
    /// `end_to_end_index_commit_persists_cursor` (v1) but for the
    /// multi-field path.
    #[test]
    fn end_to_end_v2_commit_persists_multi_field_cursor() -> crate::Result<()> {
        use crate::index::{IndexSettings, IndexSortByField};
        use crate::schema::{Schema, FAST};
        use crate::{Index, IndexWriter, TantivyDocument};

        let mut schema_builder = Schema::builder();
        let ts_field = schema_builder.add_i64_field("ts", FAST);
        let id_field = schema_builder.add_i64_field("id", FAST);
        let schema = schema_builder.build();

        let settings = IndexSettings {
            sort_by_fields: Some(vec![
                IndexSortByField {
                    field: "ts".to_string(),
                    order: Order::Desc,
                },
                IndexSortByField {
                    field: "id".to_string(),
                    order: Order::Asc,
                },
            ]),
            ..Default::default()
        };
        let index = Index::builder()
            .schema(schema)
            .settings(settings)
            .create_in_ram()?;

        let mut writer: IndexWriter = index.writer_for_tests()?;
        // (ts, id) docs:
        //   doc 0: (100, 7)
        //   doc 1: (100, 3)   ← ties with doc 0 on ts; id=3 < 7 → doc 1 first
        //   doc 2: (200, 5)
        //   doc 3: (50,  9)
        for (ts, id) in [(100i64, 7i64), (100, 3), (200, 5), (50, 9)] {
            let mut doc = TantivyDocument::default();
            doc.add_i64(ts_field, ts);
            doc.add_i64(id_field, id);
            writer.add_document(doc)?;
        }
        writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();
        assert_eq!(searcher.segment_readers().len(), 1);
        let sr = searcher.segment_reader(0);

        // Meta should advertise the cursor under the PRIMARY field name.
        let cursor_fields: Vec<&str> = sr.sort_cursor_fields().collect();
        assert_eq!(cursor_fields, vec!["ts"]);

        // v1 accessor returns None — this is a v2 cursor.
        assert!(sr.sort_cursor("ts").is_none());

        // v2 accessor returns the cursor.
        let cursor = sr
            .sort_cursor_v2("ts")
            .expect("v2 cursor should be present after commit");
        assert_eq!(cursor.fields().len(), 2);
        assert_eq!(cursor.fields()[0].0, "ts");
        assert_eq!(cursor.fields()[0].1, Order::Desc);
        assert_eq!(cursor.fields()[1].0, "id");
        assert_eq!(cursor.fields()[1].1, Order::Asc);
        // Lex sort order: doc 2 (200,5), doc 1 (100,3), doc 0 (100,7), doc 3 (50,9).
        assert_eq!(cursor.doc_ids(), &[2u32, 1, 0, 3]);
        Ok(())
    }

    /// **Wave 18-1 Phase C.** `IndexSettings::validate_sort_settings`
    /// rejects setting both `sort_by_field` and `sort_by_fields`.
    #[test]
    fn index_settings_rejects_both_sort_by_field_and_fields_set() {
        use crate::index::{IndexSettings, IndexSortByField};

        let settings = IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "a".to_string(),
                order: Order::Asc,
            }),
            sort_by_fields: Some(vec![IndexSortByField {
                field: "b".to_string(),
                order: Order::Desc,
            }]),
            ..Default::default()
        };
        let err = settings.validate_sort_settings().unwrap_err();
        assert!(
            err.to_string().contains("mutually exclusive"),
            "expected mutual-exclusion error, got: {err}"
        );
    }

    /// **Wave 18-1 Phase C.** Cap is enforced on multi-field sort: an
    /// empty `sort_by_fields` returns an error (use `None` instead),
    /// and 9 fields exceeds the `SORT_CURSOR_MAX_FIELDS=8` cap.
    #[test]
    fn index_settings_rejects_empty_or_too_many_fields() {
        use crate::index::{IndexSettings, IndexSortByField};

        let empty = IndexSettings {
            sort_by_fields: Some(vec![]),
            ..Default::default()
        };
        let err = empty.validate_sort_settings().unwrap_err();
        assert!(
            err.to_string().contains("cannot be empty"),
            "expected empty-fields error, got: {err}"
        );

        let too_many = IndexSettings {
            sort_by_fields: Some(
                (0..9)
                    .map(|i| IndexSortByField {
                        field: format!("f{i}"),
                        order: Order::Asc,
                    })
                    .collect(),
            ),
            ..Default::default()
        };
        let err = too_many.validate_sort_settings().unwrap_err();
        assert!(
            err.to_string().contains("at most"),
            "expected cap error, got: {err}"
        );
    }

    /// **Wave 18-1 Phase C.** `IndexBuilder::create` propagates the
    /// validation error so misuse is caught at index-create time
    /// rather than at commit time.
    #[test]
    fn index_builder_propagates_sort_settings_validation_error() {
        use crate::index::{IndexSettings, IndexSortByField};
        use crate::schema::Schema;
        use crate::Index;

        let schema = Schema::builder().build();
        let bad = IndexSettings {
            sort_by_field: Some(IndexSortByField {
                field: "a".to_string(),
                order: Order::Asc,
            }),
            sort_by_fields: Some(vec![IndexSortByField {
                field: "b".to_string(),
                order: Order::Desc,
            }]),
            ..Default::default()
        };
        let err = Index::builder()
            .schema(schema)
            .settings(bad)
            .create_in_ram()
            .unwrap_err();
        assert!(
            err.to_string().contains("mutually exclusive"),
            "expected mutual-exclusion error from IndexBuilder, got: {err}"
        );
    }
}

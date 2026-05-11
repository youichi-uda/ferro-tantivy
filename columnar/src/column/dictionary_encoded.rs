use std::ops::Deref;
use std::sync::Arc;
use std::{fmt, io};

use sstable::{Dictionary, VoidSSTable};

use crate::RowId;
use crate::column::Column;

/// Dictionary encoded column.
///
/// The column simply gives access to a regular u64-column that, in
/// which the values are term-ordinals.
///
/// These ordinals are ids uniquely identify the bytes that are stored in
/// the column. These ordinals are small, and sorted in the same order
/// as the term_ord_column.
#[derive(Clone)]
pub struct BytesColumn {
    pub(crate) dictionary: Arc<Dictionary<VoidSSTable>>,
    pub(crate) term_ord_column: Column<u64>,
}

impl fmt::Debug for BytesColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytesColumn")
            .field("term_ord_column", &self.term_ord_column)
            .finish()
    }
}

impl BytesColumn {
    pub fn empty(num_docs: u32) -> BytesColumn {
        BytesColumn {
            dictionary: Arc::new(Dictionary::empty()),
            term_ord_column: Column::build_empty_column(num_docs),
        }
    }

    /// Fills the given `output` buffer with the term associated to the ordinal `ord`.
    ///
    /// Returns `false` if the term does not exist (e.g. `term_ord` is greater or equal to the
    /// overll number of terms).
    pub fn ord_to_bytes(&self, ord: u64, output: &mut Vec<u8>) -> io::Result<bool> {
        self.dictionary.ord_to_term(ord, output)
    }

    /// Returns the number of rows in the column.
    pub fn num_rows(&self) -> RowId {
        self.term_ord_column.num_docs()
    }

    pub fn term_ords(&self, row_id: RowId) -> impl Iterator<Item = u64> + '_ {
        self.term_ord_column.values_for_doc(row_id)
    }

    /// Returns the column of ordinals
    pub fn ords(&self) -> &Column<u64> {
        &self.term_ord_column
    }

    pub fn num_terms(&self) -> usize {
        self.dictionary.num_terms()
    }

    pub fn dictionary(&self) -> &Dictionary<VoidSSTable> {
        self.dictionary.as_ref()
    }

    /// Returns the (min, max) raw byte terms recorded in this column's
    /// dictionary, or `None` if the column has no terms.
    ///
    /// The dictionary stores terms in sorted byte order, so min == term at
    /// ordinal `0` and max == term at ordinal `num_terms - 1`. Cost is two
    /// SSTable seeks (O(log n) on the block index, no full scan).
    ///
    /// Used by query-plan-time `can_match` shortcuts: when the query carries
    /// a range/term filter on this field and the requested range is disjoint
    /// from `[min, max]`, the segment cannot contribute and the search may
    /// skip it entirely.
    ///
    /// Note: returned bounds are an over-approximation of *surviving* docs —
    /// deleted docs' terms remain in the dictionary until merge. This is
    /// safe for shortcut decisions (false positives — running search on a
    /// segment that turns out to have nothing — are correct, just slower;
    /// false negatives — wrongly skipping — would corrupt results, but the
    /// dictionary upper-bounds the actual range so this never happens).
    pub fn min_max_bytes(&self) -> io::Result<Option<(Vec<u8>, Vec<u8>)>> {
        let n = self.dictionary.num_terms();
        if n == 0 {
            return Ok(None);
        }
        let mut min_buf = Vec::new();
        if !self.dictionary.ord_to_term(0, &mut min_buf)? {
            return Ok(None);
        }
        if n == 1 {
            return Ok(Some((min_buf.clone(), min_buf)));
        }
        let mut max_buf = Vec::new();
        if !self.dictionary.ord_to_term((n - 1) as u64, &mut max_buf)? {
            return Ok(None);
        }
        Ok(Some((min_buf, max_buf)))
    }
}

#[derive(Clone)]
pub struct StrColumn(BytesColumn);

impl fmt::Debug for StrColumn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.term_ord_column)
    }
}

impl From<StrColumn> for BytesColumn {
    fn from(str_column: StrColumn) -> BytesColumn {
        str_column.0
    }
}

impl StrColumn {
    pub fn wrap(bytes_column: BytesColumn) -> StrColumn {
        StrColumn(bytes_column)
    }

    pub fn dictionary(&self) -> &Dictionary<VoidSSTable> {
        self.0.dictionary.as_ref()
    }

    /// Returns the (min, max) UTF-8 string terms recorded in this column's
    /// dictionary, or `None` if the column has no terms or any term is not
    /// valid UTF-8 (callers should fall back to `BytesColumn::min_max_bytes`).
    ///
    /// See `BytesColumn::min_max_bytes` for the can_match shortcut semantics.
    pub fn min_max_str(&self) -> io::Result<Option<(String, String)>> {
        let Some((min_b, max_b)) = self.0.min_max_bytes()? else {
            return Ok(None);
        };
        let min_s = String::from_utf8(min_b).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("min term not utf-8: {e}"),
            )
        })?;
        let max_s = String::from_utf8(max_b).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("max term not utf-8: {e}"),
            )
        })?;
        Ok(Some((min_s, max_s)))
    }

    /// Fills the buffer
    pub fn ord_to_str(&self, term_ord: u64, output: &mut String) -> io::Result<bool> {
        unsafe {
            let buf = output.as_mut_vec();
            if !self.0.dictionary.ord_to_term(term_ord, buf)? {
                return Ok(false);
            }
            // TODO consider remove checks if it hurts performance.
            if std::str::from_utf8(buf.as_slice()).is_err() {
                buf.clear();
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Not valid utf-8",
                ));
            }
        }
        Ok(true)
    }
}

impl Deref for StrColumn {
    type Target = BytesColumn;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

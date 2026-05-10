use std::fmt;
use std::path::PathBuf;

use super::SegmentComponent;
use crate::directory::error::{OpenReadError, OpenWriteError};
use crate::directory::{Directory, FileSlice, WritePtr};
use crate::index::{Index, SegmentId, SegmentMeta};
use crate::schema::Schema;
use crate::Opstamp;

/// A segment is a piece of the index.
#[derive(Clone)]
pub struct Segment {
    index: Index,
    meta: SegmentMeta,
}

impl fmt::Debug for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Segment({:?})", self.id().uuid_string())
    }
}

impl Segment {
    /// Creates a new segment given an `Index` and a `SegmentId`
    pub(crate) fn for_index(index: Index, meta: SegmentMeta) -> Segment {
        Segment { index, meta }
    }

    /// Returns the index the segment belongs to.
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Returns our index's schema.
    pub fn schema(&self) -> Schema {
        self.index.schema()
    }

    /// Returns the segment meta-information
    pub fn meta(&self) -> &SegmentMeta {
        &self.meta
    }

    /// Updates the max_doc value from the `SegmentMeta`.
    ///
    /// This method is only used when updating `max_doc` from 0
    /// as we finalize a fresh new segment.
    pub fn with_max_doc(self, max_doc: u32) -> Segment {
        Segment {
            index: self.index,
            meta: self.meta.with_max_doc(max_doc),
        }
    }

    /// **FerroSearch extension (Wave 15).** Records that auxiliary sort
    /// cursor files were written for `fields` when this segment was
    /// finalised, so the cursors are advertised in `meta.json` and
    /// retained by the directory GC.
    #[must_use]
    pub fn with_sort_cursor_fields(self, fields: Vec<String>) -> Segment {
        Segment {
            index: self.index,
            meta: self.meta.with_sort_cursor_fields(fields),
        }
    }

    #[doc(hidden)]
    #[must_use]
    pub fn with_delete_meta(self, num_deleted_docs: u32, opstamp: Opstamp) -> Segment {
        Segment {
            index: self.index,
            meta: self.meta.with_delete_meta(num_deleted_docs, opstamp),
        }
    }

    /// Returns the segment's id.
    pub fn id(&self) -> SegmentId {
        self.meta.id()
    }

    /// Returns the relative path of a component of our segment.
    ///
    /// It just joins the segment id with the extension
    /// associated with a segment component.
    pub fn relative_path(&self, component: SegmentComponent) -> PathBuf {
        self.meta.relative_path(component)
    }

    /// Open one of the component file for a *regular* read.
    pub fn open_read(&self, component: SegmentComponent) -> Result<FileSlice, OpenReadError> {
        let path = self.relative_path(component);
        self.index.directory().open_read(&path)
    }

    /// Open one of the component file for *regular* write.
    pub fn open_write(&mut self, component: SegmentComponent) -> Result<WritePtr, OpenWriteError> {
        let path = self.relative_path(component);
        let write = self.index.directory_mut().open_write(&path)?;
        Ok(write)
    }

    /// Opens the auxiliary sort cursor file for `field` for reading.
    ///
    /// **FerroSearch extension (Wave 15).** The path is
    /// `{segment_uuid}.{field}.sortcursor`.
    pub fn open_sort_cursor_read(&self, field: &str) -> Result<FileSlice, OpenReadError> {
        let path = self.meta.sort_cursor_path(field);
        self.index.directory().open_read(&path)
    }

    /// Opens the auxiliary sort cursor file for `field` for writing.
    ///
    /// **FerroSearch extension (Wave 15).**
    pub fn open_sort_cursor_write(&mut self, field: &str) -> Result<WritePtr, OpenWriteError> {
        let path = self.meta.sort_cursor_path(field);
        self.index.directory_mut().open_write(&path)
    }
}

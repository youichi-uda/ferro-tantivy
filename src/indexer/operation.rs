use crate::query::Weight;
use crate::schema::document::Document;
use crate::schema::{TantivyDocument, Term};
use crate::Opstamp;

/// Target for a delete operation.
pub enum DeleteTarget {
    /// Delete by term and build the actual weight lazily when applying deletes.
    Term(Term),
    /// Delete by an arbitrary compiled query weight.
    Weight(Box<dyn Weight>),
}

/// Timestamped Delete operation.
pub struct DeleteOperation {
    /// Operation stamp.
    /// It is used to check whether the delete operation
    /// applies to an added document operation.
    pub opstamp: Opstamp,
    /// Target used to define the set of documents to be deleted.
    pub target: DeleteTarget,
}

/// Timestamped Add operation.
#[derive(Eq, PartialEq, Debug)]
pub struct AddOperation<D: Document = TantivyDocument> {
    /// Operation stamp.
    pub opstamp: Opstamp,
    /// Document to be added.
    pub document: D,
}

/// UserOperation is an enum type that encapsulates other operation types.
#[derive(Eq, PartialEq, Debug)]
pub enum UserOperation<D: Document = TantivyDocument> {
    /// Add operation
    Add(D),
    /// Delete operation
    Delete(Term),
}

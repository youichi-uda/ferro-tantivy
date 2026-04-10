use std::ops::DerefMut;

use downcast_rs::impl_downcast;

use crate::docset::DocSet;
use crate::{DocId, Score, TERMINATED};

/// Scored set of documents matching a query within a specific segment.
///
/// See [`Query`](crate::query::Query).
pub trait Scorer: downcast_rs::Downcast + DocSet + 'static {
    /// Returns the score.
    ///
    /// This method will perform a bit of computation and is not cached.
    fn score(&mut self) -> Score;

    /// Returns the maximum score for the current block.
    fn block_max_score(&mut self) -> Score {
        Score::MAX
    }

    /// Positions the block cursor on the block containing `target`
    /// without loading individual postings.
    fn shallow_seek(&mut self, _target: DocId) -> DocId {
        TERMINATED
    }

    /// Returns the last `DocId` in the current block.
    fn last_doc_in_block(&self) -> DocId {
        TERMINATED
    }

    /// Returns the global maximum score this scorer can produce.
    fn max_score(&self) -> Score {
        Score::MAX
    }

    /// Returns the maximum score for docs up to `up_to` (inclusive).
    fn get_max_score(&self, _up_to: DocId) -> Score {
        Score::MAX
    }
}

impl_downcast!(Scorer);

impl Scorer for Box<dyn Scorer> {
    #[inline]
    fn score(&mut self) -> Score {
        self.deref_mut().score()
    }

    #[inline]
    fn block_max_score(&mut self) -> Score {
        self.deref_mut().block_max_score()
    }

    #[inline]
    fn shallow_seek(&mut self, target: DocId) -> DocId {
        self.deref_mut().shallow_seek(target)
    }

    #[inline]
    fn last_doc_in_block(&self) -> DocId {
        self.as_ref().last_doc_in_block()
    }

    #[inline]
    fn max_score(&self) -> Score {
        self.as_ref().max_score()
    }

    #[inline]
    fn get_max_score(&self, up_to: DocId) -> Score {
        self.as_ref().get_max_score(up_to)
    }
}

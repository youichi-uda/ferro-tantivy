//! BlockWandScorer — a Scorer wrapper for block-max WAND unions.
//!
//! This replaces BufferedUnionScorer when block-max scoring is desired.
//! BufferedUnionScorer uses a sliding window that conflicts with the
//! block-max API's internal state, so this scorer is purpose-built
//! to work with shallow_seek/block_max_score without state collisions.

use crate::docset::DocSet;
use crate::query::Scorer;
use crate::{DocId, Score, TERMINATED};

/// A union scorer that uses block-max WAND to skip non-competitive blocks.
///
/// Unlike BufferedUnionScorer, this scorer does not buffer documents.
/// Instead it iterates via find_min_and_score which advances the minimum
/// scorer and computes the combined score in a single pass.
#[allow(dead_code)]
pub struct BlockWandScorer {
    scorers: Vec<Box<dyn Scorer>>,
    current_doc: DocId,
    current_score: Score,
}

#[allow(dead_code)]
impl BlockWandScorer {
    pub fn new(mut scorers: Vec<Box<dyn Scorer>>) -> BlockWandScorer {
        // Remove terminated scorers.
        scorers.retain(|s| s.doc() != TERMINATED);

        // Find the initial minimum doc.
        let current_doc = scorers.iter().map(|s| s.doc()).min().unwrap_or(TERMINATED);

        let mut bws = BlockWandScorer {
            scorers,
            current_doc,
            current_score: 0.0,
        };

        if bws.current_doc != TERMINATED {
            bws.compute_score();
        }

        bws
    }

    /// Compute score for current_doc by summing all scorers positioned on it.
    fn compute_score(&mut self) {
        self.current_score = 0.0;
        for scorer in &mut self.scorers {
            if scorer.doc() == self.current_doc {
                self.current_score += scorer.score();
            }
        }
    }

    /// Advance to the next doc and compute its score.
    fn find_next(&mut self) {
        // Advance all scorers that are on current_doc.
        for scorer in &mut self.scorers {
            if scorer.doc() == self.current_doc {
                scorer.advance();
            }
        }

        // Remove terminated.
        self.scorers.retain(|s| s.doc() != TERMINATED);

        if self.scorers.is_empty() {
            self.current_doc = TERMINATED;
            self.current_score = 0.0;
            return;
        }

        // Find new minimum.
        self.current_doc = self.scorers.iter().map(|s| s.doc()).min().unwrap_or(TERMINATED);

        if self.current_doc != TERMINATED {
            self.compute_score();
        } else {
            self.current_score = 0.0;
        }
    }
}

impl DocSet for BlockWandScorer {
    fn advance(&mut self) -> DocId {
        self.find_next();
        self.current_doc
    }

    fn seek(&mut self, target: DocId) -> DocId {
        if self.current_doc >= target {
            return self.current_doc;
        }

        // Seek all scorers to target.
        for scorer in &mut self.scorers {
            if scorer.doc() < target {
                scorer.seek(target);
            }
        }

        // Remove terminated.
        self.scorers.retain(|s| s.doc() != TERMINATED);

        if self.scorers.is_empty() {
            self.current_doc = TERMINATED;
            self.current_score = 0.0;
            return TERMINATED;
        }

        self.current_doc = self.scorers.iter().map(|s| s.doc()).min().unwrap_or(TERMINATED);

        if self.current_doc != TERMINATED {
            self.compute_score();
        } else {
            self.current_score = 0.0;
        }

        self.current_doc
    }

    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        self.scorers.iter().map(|s| s.size_hint()).max().unwrap_or(0)
    }
}

impl Scorer for BlockWandScorer {
    fn score(&mut self) -> Score {
        self.current_score
    }

    fn block_max_score(&mut self) -> Score {
        self.scorers.iter_mut().map(|s| s.block_max_score()).sum()
    }

    fn shallow_seek(&mut self, target: DocId) -> DocId {
        let mut min_last = TERMINATED;
        for scorer in &mut self.scorers {
            let last = scorer.shallow_seek(target);
            if last < min_last {
                min_last = last;
            }
        }
        min_last
    }

    fn last_doc_in_block(&self) -> DocId {
        self.scorers
            .iter()
            .map(|s| s.last_doc_in_block())
            .min()
            .unwrap_or(TERMINATED)
    }

    fn max_score(&self) -> Score {
        self.scorers.iter().map(|s| s.max_score()).sum()
    }

    fn get_max_score(&self, up_to: DocId) -> Score {
        self.scorers.iter().map(|s| s.get_max_score(up_to)).sum()
    }
}

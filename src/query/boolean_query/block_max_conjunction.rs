//! BlockMaxConjunctionBulkScorer — block-max pruning for Must (AND) queries.
//!
//! For conjunction queries, all scorers must match the same doc. This scorer
//! uses block-max scores to skip entire blocks when the sum of block-max scores
//! across all clauses cannot exceed the current threshold.
//!
//! Key insight: use window-local block_max_score (not global max_score).

use crate::query::Scorer;
use crate::{DocId, Score, TERMINATED};

/// Runs block-max conjunction scoring over a set of Must scorers.
///
/// All scorers must match the same doc for it to be a candidate.
/// Block-max scores are used to skip windows where the combined
/// maximum possible score cannot exceed the threshold.
pub fn block_max_conjunction(
    mut scorers: Vec<Box<dyn Scorer>>,
    mut threshold: Score,
    callback: &mut dyn FnMut(DocId, Score) -> Score,
) {
    if scorers.is_empty() {
        return;
    }

    // Sort by cost ascending — lead iterator is the rarest term.
    scorers.sort_by_key(|s| s.size_hint());

    // Remove any terminated scorers.
    scorers.retain(|s| s.doc() != TERMINATED);

    if scorers.is_empty() {
        return;
    }

    let num_scorers = scorers.len();

    loop {
        // The lead scorer (index 0) drives iteration.
        let mut candidate = scorers[0].doc();
        if candidate == TERMINATED {
            break;
        }

        // Compute the window: find the minimum last_doc_in_block across all scorers.
        let mut window_end = TERMINATED;
        for scorer in scorers.iter_mut() {
            scorer.shallow_seek(candidate);
            let last = scorer.last_doc_in_block();
            if last < window_end {
                window_end = last;
            }
        }

        // Compute max possible score in this window (sum of block_max_scores).
        let max_window_score: Score = scorers.iter_mut().map(|s| s.block_max_score()).sum();

        if max_window_score <= threshold {
            // This window can't beat threshold — skip ahead.
            let next = if window_end >= TERMINATED - 1 {
                TERMINATED
            } else {
                window_end + 1
            };
            for scorer in scorers.iter_mut() {
                scorer.seek(next);
            }
            if scorers[0].doc() == TERMINATED {
                break;
            }
            continue;
        }

        // Process docs in [candidate, window_end].
        while candidate <= window_end && candidate != TERMINATED {
            // Try to align all scorers on candidate.
            let mut all_match = true;
            for i in 1..num_scorers {
                let d = scorers[i].seek(candidate);
                if d != candidate {
                    if d == TERMINATED {
                        return;
                    }
                    // Scorer went past candidate — this doc isn't a match.
                    // Advance lead to the new position.
                    candidate = d;
                    all_match = false;
                    // Re-check from the lead.
                    let lead_doc = scorers[0].seek(candidate);
                    if lead_doc == TERMINATED {
                        return;
                    }
                    candidate = lead_doc;
                    break;
                }
            }

            if !all_match {
                // candidate was updated, recheck from window boundary.
                if candidate > window_end {
                    break;
                }
                continue;
            }

            // All scorers aligned on candidate — compute actual score.
            let mut score: Score = 0.0;
            let mut skip = false;

            // Compute sum_of_other_clauses for early termination.
            // Use window-local block_max_score, NOT global max_score.
            let mut remaining_max: Score = 0.0;
            for scorer in scorers.iter_mut() {
                remaining_max += scorer.block_max_score();
            }

            for scorer in scorers.iter_mut() {
                remaining_max -= scorer.block_max_score();
                if score + remaining_max + scorer.block_max_score() <= threshold {
                    skip = true;
                    break;
                }
                score += scorer.score();
            }

            if !skip && score > threshold {
                threshold = callback(candidate, score);
            }

            // Advance lead.
            candidate = scorers[0].advance();
        }

        // Move past window.
        if window_end >= TERMINATED - 1 {
            break;
        }
        let next = window_end + 1;
        for scorer in scorers.iter_mut() {
            if scorer.doc() <= window_end {
                scorer.seek(next);
            }
        }
        if scorers[0].doc() == TERMINATED {
            break;
        }
    }
}

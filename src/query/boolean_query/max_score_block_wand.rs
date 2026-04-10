//! MaxScoreBulkScorer — block-max WAND for dyn Scorer unions.
//!
//! Ported from Lucene 10.3.2's MaxScoreBulkScorer.
//! Uses the block-max API on the Scorer trait to partition scorers into
//! essential (must evaluate) and non-essential (can skip) sets, then
//! processes documents in windows defined by block boundaries.

use crate::query::Scorer;
use crate::{DocId, Score, TERMINATED};

/// Entry wrapping a scorer with its cached max-window score and cost.
struct ScorerEntry {
    scorer: Box<dyn Scorer>,
    /// Max score this scorer can contribute within the current window.
    max_window_score: Score,
    /// Estimated number of documents (for ordering heuristics).
    cost: u32,
}

/// Partitions scorers into essential / non-essential based on threshold.
///
/// `partition_order[..num_essential]` = essential scorer indices (sorted by cost asc).
/// `partition_order[num_essential..]` = non-essential scorer indices.
///
/// Key insight from the memory doc: we use indirect `partition_order` array
/// rather than physically reordering the `entries` Vec — physical reorder
/// causes 2-3x regression.
fn partition_scorers(
    entries: &[ScorerEntry],
    partition_order: &mut [usize],
    threshold: Score,
) -> usize {
    // Sum max_window_scores from highest to lowest to find the partition point.
    // Non-essential scorers are those whose cumulative max score (from the top)
    // does not exceed the threshold.

    // Sort partition_order by max_window_score descending for partitioning.
    partition_order.sort_by(|&a, &b| {
        entries[b]
            .max_window_score
            .partial_cmp(&entries[a].max_window_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumulative_score: Score = 0.0;
    let mut first_essential = partition_order.len();

    for (i, &idx) in partition_order.iter().enumerate() {
        cumulative_score += entries[idx].max_window_score;
        if cumulative_score > threshold {
            first_essential = i;
            break;
        }
    }

    // partition_order[first_essential..] are essential.
    // Reverse so essential are at the front.
    let num_essential = partition_order.len() - first_essential;

    // Rearrange: essential scorers to front, non-essential to back.
    // We do this by rotating the array.
    partition_order.rotate_left(first_essential);

    // Sort essential portion (front) by cost ascending for efficient iteration.
    partition_order[..num_essential].sort_by_key(|&idx| entries[idx].cost);

    num_essential
}

/// Runs the MaxScore block-WAND algorithm over a set of scorers.
///
/// This is the main entry point for disjunctive (OR) queries with pruning.
pub fn max_score_bulk_scorer(
    mut scorers: Vec<Box<dyn Scorer>>,
    mut threshold: Score,
    callback: &mut dyn FnMut(DocId, Score) -> Score,
) {
    if scorers.is_empty() {
        return;
    }

    if scorers.len() == 1 {
        // Single scorer fast path — use simple block-max iteration.
        let mut scorer = scorers.pop().unwrap();
        let mut doc = scorer.doc();
        loop {
            while scorer.block_max_score() < threshold {
                let last = scorer.last_doc_in_block();
                if last == TERMINATED {
                    return;
                }
                doc = last + 1;
                scorer.shallow_seek(doc);
            }
            doc = scorer.seek(doc);
            if doc == TERMINATED {
                break;
            }
            loop {
                let score = scorer.score();
                if score > threshold {
                    threshold = callback(doc, score);
                }
                if doc >= scorer.last_doc_in_block() {
                    break;
                }
                doc = scorer.advance();
                if doc == TERMINATED {
                    return;
                }
            }
            doc += 1;
            scorer.shallow_seek(doc);
        }
        return;
    }

    // Build entries.
    let mut entries: Vec<ScorerEntry> = scorers
        .into_iter()
        .filter(|s| s.doc() != TERMINATED)
        .map(|s| {
            let cost = s.size_hint();
            let max_window_score = s.max_score();
            ScorerEntry {
                scorer: s,
                max_window_score,
                cost,
            }
        })
        .collect();

    if entries.is_empty() {
        return;
    }

    let n = entries.len();
    let mut partition_order: Vec<usize> = (0..n).collect();

    // Main window loop.
    loop {
        // Find the minimum doc across all scorers and the window end.
        let mut window_min = TERMINATED;
        let mut window_end = TERMINATED;

        for entry in entries.iter() {
            let d = entry.scorer.doc();
            if d < window_min {
                window_min = d;
            }
        }

        if window_min == TERMINATED {
            break;
        }

        // Compute window end as min(last_doc_in_block) across all scorers
        // whose current doc is within reach.
        for entry in entries.iter_mut() {
            entry.scorer.shallow_seek(window_min);
            let last = entry.scorer.last_doc_in_block();
            if last < window_end {
                window_end = last;
            }
            entry.max_window_score = entry.scorer.block_max_score();
        }

        if window_end == TERMINATED {
            // Fallback: process remaining docs one by one.
            window_end = TERMINATED - 1;
        }

        // Partition into essential / non-essential.
        let num_essential = partition_scorers(&entries, &mut partition_order, threshold);

        if num_essential == 0 {
            // No scorer can exceed threshold in this window — skip ahead.
            // Advance all scorers past window_end.
            let next = if window_end == TERMINATED - 1 {
                TERMINATED
            } else {
                window_end + 1
            };
            for entry in entries.iter_mut() {
                if entry.scorer.doc() <= window_end {
                    entry.scorer.seek(next);
                }
            }
            // Remove terminated.
            entries.retain(|e| e.scorer.doc() != TERMINATED);
            if entries.is_empty() {
                break;
            }
            partition_order = (0..entries.len()).collect();
            continue;
        }

        // Process docs in the window [window_min, window_end].
        // Iterate essential scorers to find candidate docs.
        if num_essential == 1 {
            // Single essential scorer — iterate it directly.
            let eidx = partition_order[0];
            let mut doc = entries[eidx].scorer.doc();
            while doc != TERMINATED && doc <= window_end {
                let mut score = entries[eidx].scorer.score();

                // Add contributions from non-essential scorers.
                for &neidx in &partition_order[num_essential..] {
                    let ne = &mut entries[neidx];
                    if ne.scorer.doc() < doc {
                        ne.scorer.seek(doc);
                    }
                    if ne.scorer.doc() == doc {
                        score += ne.scorer.score();
                    }
                }

                if score > threshold {
                    threshold = callback(doc, score);
                }

                doc = entries[eidx].scorer.advance();
            }
        } else {
            // Multiple essential scorers — iterate the union of all essential
            // scorers (find min doc across all essentials at each step).
            loop {
                // Find the minimum doc across all essential scorers.
                let mut doc = TERMINATED;
                for i in 0..num_essential {
                    let eidx = partition_order[i];
                    let d = entries[eidx].scorer.doc();
                    if d < doc {
                        doc = d;
                    }
                }

                if doc == TERMINATED || doc > window_end {
                    break;
                }

                // Score: sum from all essential scorers that are on this doc.
                let mut score: Score = 0.0;
                let mut sum_of_other_max: Score = 0.0;

                // First compute total max_window_score for non-essential scorers.
                for &neidx in &partition_order[num_essential..] {
                    sum_of_other_max += entries[neidx].max_window_score;
                }

                // Score from essential scorers on this doc.
                for i in 0..num_essential {
                    let eidx = partition_order[i];
                    let e = &mut entries[eidx];
                    if e.scorer.doc() == doc {
                        score += e.scorer.score();
                    }
                }

                // Early termination check.
                if score + sum_of_other_max > threshold {
                    // Add non-essential contributions.
                    for &neidx in &partition_order[num_essential..] {
                        let ne = &mut entries[neidx];
                        if ne.scorer.doc() < doc {
                            ne.scorer.seek(doc);
                        }
                        if ne.scorer.doc() == doc {
                            score += ne.scorer.score();
                        }
                    }

                    if score > threshold {
                        threshold = callback(doc, score);
                    }
                }

                // Advance all essential scorers that are on the current doc.
                for i in 0..num_essential {
                    let eidx = partition_order[i];
                    if entries[eidx].scorer.doc() == doc {
                        entries[eidx].scorer.advance();
                    }
                }
            }
        }

        // Advance all scorers past window_end for next window.
        let next_window = if window_end >= TERMINATED - 1 {
            TERMINATED
        } else {
            window_end + 1
        };
        for entry in entries.iter_mut() {
            if entry.scorer.doc() != TERMINATED && entry.scorer.doc() <= window_end {
                entry.scorer.seek(next_window);
            }
        }

        // Remove terminated scorers.
        entries.retain(|e| e.scorer.doc() != TERMINATED);
        if entries.is_empty() {
            break;
        }
        partition_order = (0..entries.len()).collect();
    }
}

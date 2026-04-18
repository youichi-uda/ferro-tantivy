use super::PhraseScorer;
use crate::fieldnorm::FieldNormReader;
use crate::index::SegmentReader;
use crate::postings::SegmentPostings;
use crate::query::bm25::Bm25Weight;
use crate::query::explanation::does_not_match;
use crate::query::{EmptyScorer, Explanation, Scorer, Weight};
use crate::schema::{IndexRecordOption, Term};
use crate::{DocId, DocSet, Score, TERMINATED};

pub struct PhraseWeight {
    phrase_terms: Vec<(usize, Term)>,
    similarity_weight_opt: Option<Bm25Weight>,
    slop: u32,
    ordered: bool,
}

impl PhraseWeight {
    /// Creates a new phrase weight.
    /// If `similarity_weight_opt` is None, then scoring is disabled
    pub fn new(
        phrase_terms: Vec<(usize, Term)>,
        similarity_weight_opt: Option<Bm25Weight>,
    ) -> PhraseWeight {
        let slop = 0;
        PhraseWeight {
            phrase_terms,
            similarity_weight_opt,
            slop,
            ordered: false,
        }
    }

    /// Enable ordered matching.
    pub fn set_ordered(&mut self, ordered: bool) {
        self.ordered = ordered;
    }

    fn fieldnorm_reader(&self, reader: &SegmentReader) -> crate::Result<FieldNormReader> {
        let field = self.phrase_terms[0].1.field();
        if self.similarity_weight_opt.is_some() {
            if let Some(fieldnorm_reader) = reader.fieldnorms_readers().get_field(field)? {
                return Ok(fieldnorm_reader);
            }
        }
        Ok(FieldNormReader::constant(reader.max_doc(), 1))
    }

    pub(crate) fn phrase_scorer(
        &self,
        reader: &SegmentReader,
        boost: Score,
    ) -> crate::Result<Option<PhraseScorer<SegmentPostings>>> {
        let similarity_weight_opt = self
            .similarity_weight_opt
            .as_ref()
            .map(|similarity_weight| similarity_weight.boost_by(boost));
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let mut term_postings_list = Vec::new();
        for &(offset, ref term) in &self.phrase_terms {
            if let Some(postings) = reader
                .inverted_index(term.field())?
                .read_postings(term, IndexRecordOption::WithFreqsAndPositions)?
            {
                term_postings_list.push((offset, postings));
            } else {
                return Ok(None);
            }
        }
        Ok(Some(PhraseScorer::new_with_ordered(
            term_postings_list,
            similarity_weight_opt,
            fieldnorm_reader,
            self.slop,
            self.ordered,
        )))
    }

    pub fn slop(&mut self, slop: u32) {
        self.slop = slop;
    }
}

impl Weight for PhraseWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        if let Some(scorer) = self.phrase_scorer(reader, boost)? {
            Ok(Box::new(scorer))
        } else {
            Ok(Box::new(EmptyScorer))
        }
    }

    /// Two-phase pruning: skip position decoding for docs whose upper-bound
    /// score (based on term frequency alone) cannot beat the current threshold.
    fn for_each_pruning(
        &self,
        threshold: Score,
        reader: &SegmentReader,
        callback: &mut dyn FnMut(DocId, Score) -> Score,
    ) -> crate::Result<()> {
        const PRUNING_WARMUP_DOCS: usize = 32;

        let scorer_opt = self.phrase_scorer(reader, 1.0)?;
        let Some(mut scorer) = scorer_opt else {
            return Ok(());
        };
        let mut threshold = threshold;
        // The first doc is already phrase-verified in PhraseScorer::new().
        let mut doc = scorer.doc();
        let mut docs_seen = 0usize;
        let mut use_two_phase = threshold.is_finite() && threshold > Score::MIN;
        while doc != TERMINATED {
            let score = scorer.score();
            if score > threshold {
                threshold = callback(doc, score);
                if !use_two_phase {
                    use_two_phase = threshold.is_finite() && threshold > Score::MIN;
                }
            }
            docs_seen += 1;
            if !use_two_phase && docs_seen >= PRUNING_WARMUP_DOCS {
                use_two_phase = true;
            }
            doc = if use_two_phase {
                scorer.advance_two_phase(threshold)
            } else {
                scorer.advance()
            };
        }
        Ok(())
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let scorer_opt = self.phrase_scorer(reader, 1.0)?;
        if scorer_opt.is_none() {
            return Err(does_not_match(doc));
        }
        let mut scorer = scorer_opt.unwrap();
        if scorer.seek(doc) != doc {
            return Err(does_not_match(doc));
        }
        let fieldnorm_reader = self.fieldnorm_reader(reader)?;
        let fieldnorm_id = fieldnorm_reader.fieldnorm_id(doc);
        let phrase_count = scorer.phrase_count();
        let mut explanation = Explanation::new("Phrase Scorer", scorer.score());
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            explanation.add_detail(similarity_weight.explain(fieldnorm_id, phrase_count));
        }
        Ok(explanation)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::create_index;
    use crate::collector::TopDocs;
    use crate::docset::TERMINATED;
    use crate::query::{EnableScoring, PhraseQuery, Scorer};
    use crate::{DocAddress, DocSet, Score, Term};

    fn build_phrase_docs(
        num_docs: usize,
        filler_step: usize,
        mut make_doc: impl FnMut(usize, String) -> String,
    ) -> Vec<String> {
        (0..num_docs)
            .map(|i| make_doc(i, "z ".repeat(i * filler_step)))
            .collect()
    }

    #[test]
    pub fn test_phrase_count() -> crate::Result<()> {
        let index = create_index(&["a c", "a a b d a b c", " a b"])?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "b"),
        ]);
        let enable_scoring = EnableScoring::enabled_from_searcher(&searcher);
        let phrase_weight = phrase_query.phrase_weight(enable_scoring).unwrap();
        let mut phrase_scorer = phrase_weight
            .phrase_scorer(searcher.segment_reader(0u32), 1.0)?
            .unwrap();
        assert_eq!(phrase_scorer.doc(), 1);
        assert_eq!(phrase_scorer.phrase_count(), 2);
        assert_eq!(phrase_scorer.advance(), 2);
        assert_eq!(phrase_scorer.doc(), 2);
        assert_eq!(phrase_scorer.phrase_count(), 1);
        assert_eq!(phrase_scorer.advance(), TERMINATED);
        Ok(())
    }

    /// Verify that two-phase pruning (via `for_each_pruning` / TopDocs) returns
    /// the same results as the eager advance path (via `for_each` / full
    /// collection).
    #[test]
    pub fn test_two_phase_pruning_matches_eager() -> crate::Result<()> {
        let docs = build_phrase_docs(180, 8, |i, filler| {
            if i % 3 == 0 {
                format!("x a b y {filler}")
            } else if i % 3 == 1 {
                format!("a x b {filler}")
            } else {
                format!("a x x {filler}")
            }
        });
        let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let index = create_index(&doc_refs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "b"),
        ]);

        // Collect via TopDocs (exercises for_each_pruning with two-phase)
        let top_docs: Vec<(Score, DocAddress)> =
            searcher.search(&phrase_query, &TopDocs::with_limit(10).order_by_score())?;

        // Collect ALL results via the scorer directly (eager advance)
        let enable_scoring = EnableScoring::enabled_from_searcher(&searcher);
        let phrase_weight = phrase_query.phrase_weight(enable_scoring).unwrap();
        let mut phrase_scorer = phrase_weight
            .phrase_scorer(searcher.segment_reader(0u32), 1.0)?
            .unwrap();
        let mut all_docs_scored = Vec::new();
        while phrase_scorer.doc() != TERMINATED {
            all_docs_scored.push((phrase_scorer.score(), phrase_scorer.doc()));
            phrase_scorer.advance();
        }
        assert!(
            all_docs_scored.first().unwrap().0 > all_docs_scored.last().unwrap().0,
            "expected field-length variation to produce score variation"
        );
        // Sort by score descending, take top 10
        all_docs_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        let expected_top: Vec<u32> = all_docs_scored.iter().take(10).map(|&(_, d)| d).collect();
        let actual_top: Vec<u32> = top_docs.iter().map(|&(_, addr)| addr.doc_id).collect();

        assert_eq!(
            actual_top.len(),
            expected_top.len(),
            "TopDocs count mismatch"
        );
        // The same documents should appear (order may differ for equal scores)
        let mut expected_sorted = expected_top.clone();
        let mut actual_sorted = actual_top.clone();
        expected_sorted.sort();
        actual_sorted.sort();
        assert_eq!(
            actual_sorted, expected_sorted,
            "Two-phase pruning returned different docs than eager advance"
        );

        Ok(())
    }

    #[test]
    pub fn test_two_phase_prunes_lower_scoring_phrase_matches() -> crate::Result<()> {
        let docs = build_phrase_docs(256, 16, |_, filler| format!("a b {filler}"));
        let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let index = create_index(&doc_refs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "b"),
        ]);
        let enable_scoring = EnableScoring::enabled_from_searcher(&searcher);
        let phrase_weight = phrase_query.phrase_weight(enable_scoring).unwrap();
        let mut phrase_scorer = phrase_weight
            .phrase_scorer(searcher.segment_reader(0u32), 1.0)?
            .unwrap();

        let best_score = phrase_scorer.score();
        assert_eq!(
            phrase_scorer.advance_two_phase(best_score),
            TERMINATED,
            "all later phrase matches should be pruned once the best score is known"
        );
        Ok(())
    }

    fn collect_top_docs_eager(
        phrase_query: &PhraseQuery,
        searcher: &crate::Searcher,
        limit: usize,
    ) -> crate::Result<Vec<u32>> {
        let enable_scoring = EnableScoring::enabled_from_searcher(searcher);
        let phrase_weight = phrase_query.phrase_weight(enable_scoring).unwrap();
        let mut phrase_scorer = phrase_weight
            .phrase_scorer(searcher.segment_reader(0u32), 1.0)?
            .unwrap();
        let mut all_docs_scored = Vec::new();
        while phrase_scorer.doc() != TERMINATED {
            all_docs_scored.push((phrase_scorer.score(), phrase_scorer.doc()));
            phrase_scorer.advance();
        }
        all_docs_scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        Ok(all_docs_scored
            .into_iter()
            .take(limit)
            .map(|(_, doc)| doc)
            .collect())
    }

    #[test]
    pub fn test_two_phase_pruning_matches_eager_with_slop_and_three_terms() -> crate::Result<()> {
        let docs = build_phrase_docs(120, 8, |i, filler| {
            if i % 4 == 0 {
                format!("a x b c {filler}")
            } else if i % 4 == 1 {
                format!("a x b x c {filler}")
            } else if i % 4 == 2 {
                format!("a b c {filler}")
            } else {
                format!("a c b {filler}")
            }
        });
        let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let index = create_index(&doc_refs)?;
        let schema = index.schema();
        let text_field = schema.get_field("text").unwrap();
        let searcher = index.reader()?.searcher();
        let mut phrase_query = PhraseQuery::new(vec![
            Term::from_field_text(text_field, "a"),
            Term::from_field_text(text_field, "b"),
            Term::from_field_text(text_field, "c"),
        ]);
        phrase_query.set_slop(1);

        let top_docs: Vec<(Score, DocAddress)> =
            searcher.search(&phrase_query, &TopDocs::with_limit(10).order_by_score())?;
        let expected_top = collect_top_docs_eager(&phrase_query, &searcher, 10)?;
        let actual_top: Vec<u32> = top_docs.iter().map(|&(_, addr)| addr.doc_id).collect();

        let mut expected_sorted = expected_top;
        let mut actual_sorted = actual_top;
        expected_sorted.sort();
        actual_sorted.sort();
        assert_eq!(actual_sorted, expected_sorted);
        Ok(())
    }
}

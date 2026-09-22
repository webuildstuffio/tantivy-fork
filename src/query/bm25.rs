use std::cell::Cell;
use std::sync::Arc;

use crate::fieldnorm::FieldNormReader;
use crate::query::Explanation;
use crate::schema::Field;
use crate::{Score, Searcher, Term};

// INF-135: Thread-local BM25 params — set before search, reset after. Thread-safe because
// each search is bounded to a single thread (Tantivy searcher is not Send across threads).
thread_local! {
    static CURRENT_BM25_PARAMS: Cell<Bm25Params> = const { Cell::new(Bm25Params { k1: K1_DEFAULT, b: B_DEFAULT, delta: 0.0 }) };
}

/// INF-135: Set per-workspace BM25 parameters for the current thread before executing a search.
/// Must be reset to default after the search completes to avoid leaking into subsequent searches.
/// Usage: `set_thread_bm25_params(Bm25Params { k1: 1.2, b: 0.20 })`; run search; `reset_thread_bm25_params()`.
pub fn set_thread_bm25_params(params: Bm25Params) {
    CURRENT_BM25_PARAMS.with(|p| p.set(params));
}

/// INF-135: Reset per-workspace BM25 parameters to default for the current thread.
pub fn reset_thread_bm25_params() {
    CURRENT_BM25_PARAMS.with(|p| p.set(Bm25Params::default()));
}

/// INF-135: Read the current thread-local BM25 parameters.
#[inline]
pub(crate) fn thread_bm25_params() -> Bm25Params {
    CURRENT_BM25_PARAMS.with(|p| p.get())
}

// INF-135: Default BM25 parameters matching Lucene/Tantivy baseline.
// Configurable per-workspace via Bm25Params; defaults preserved for zero-diff baseline.
pub const K1_DEFAULT: Score = 1.2;
pub const B_DEFAULT: Score = 0.75;

// Internal aliases kept for backward-compat with existing uses (explain text, etc.).
const K1: Score = K1_DEFAULT;

/// An interface to compute the statistics needed in BM25 scoring.
///
/// The standard implementation is a [Searcher] but you can also
/// create your own to adjust the statistics.
pub trait Bm25StatisticsProvider {
    /// The total number of tokens in a given field across all documents in
    /// the index.
    fn total_num_tokens(&self, field: Field) -> crate::Result<u64>;

    /// The total number of documents in the index.
    fn total_num_docs(&self) -> crate::Result<u64>;

    /// The number of documents containing the given term.
    fn doc_freq(&self, term: &Term) -> crate::Result<u64>;
}

impl Bm25StatisticsProvider for Searcher {
    fn total_num_tokens(&self, field: Field) -> crate::Result<u64> {
        let mut total_num_tokens = 0u64;

        for segment_reader in self.segment_readers() {
            let inverted_index = segment_reader.inverted_index(field)?;
            total_num_tokens += inverted_index.total_num_tokens();
        }
        Ok(total_num_tokens)
    }

    fn total_num_docs(&self) -> crate::Result<u64> {
        let mut total_num_docs = 0u64;

        for segment_reader in self.segment_readers() {
            total_num_docs += u64::from(segment_reader.max_doc());
        }
        Ok(total_num_docs)
    }

    fn doc_freq(&self, term: &Term) -> crate::Result<u64> {
        self.doc_freq(term)
    }
}

pub(crate) fn idf(doc_freq: u64, doc_count: u64) -> Score {
    assert!(doc_count >= doc_freq, "{doc_count} >= {doc_freq}");
    let x = ((doc_count - doc_freq) as Score + 0.5) / (doc_freq as Score + 0.5);
    (1.0 + x).ln()
}

fn cached_tf_component(fieldnorm: u32, average_fieldnorm: Score, k1: Score, b: Score) -> Score {
    k1 * (1.0 - b + b * fieldnorm as Score / average_fieldnorm)
}

fn compute_tf_cache(average_fieldnorm: Score, k1: Score, b: Score) -> Arc<[Score; 256]> {
    let mut cache: [Score; 256] = [0.0; 256];
    for (fieldnorm_id, cache_mut) in cache.iter_mut().enumerate() {
        let fieldnorm = FieldNormReader::id_to_fieldnorm(fieldnorm_id as u8);
        *cache_mut = cached_tf_component(fieldnorm, average_fieldnorm, k1, b);
    }
    Arc::new(cache)
}

/// INF-135: Per-workspace BM25 k1/b parameters. Defaults reproduce Lucene/Tantivy baseline.
/// SCR-310: added `delta` for BM25+ formula (Lü & Callan 2011). delta=0.0 = standard BM25.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    /// BM25 term-frequency saturation parameter (default 1.2).
    pub k1: Score,
    /// BM25 document-length normalization parameter (default 0.75).
    pub b: Score,
    /// BM25+ lower-bound delta: added to TF factor to prevent zero-score starvation.
    /// 0.0 = standard BM25 (default). Typical values: 0.5–2.0.
    pub delta: Score,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: K1_DEFAULT, b: B_DEFAULT, delta: 0.0 }
    }
}

/// A struct used for computing BM25 scores.
#[derive(Clone)]
pub struct Bm25Weight {
    idf_explain: Option<Explanation>,
    weight: Score,
    cache: Arc<[Score; 256]>,
    average_fieldnorm: Score,
    // INF-135: k1/b stored for explain output; scoring uses cache/weight computed at construction.
    k1: Score,
    b: Score,
    // SCR-310: BM25+ delta lower bound. 0.0 = standard BM25.
    delta: Score,
}

impl Bm25Weight {
    /// Increase the weight by a multiplicative factor.
    pub fn boost_by(&self, boost: Score) -> Bm25Weight {
        if boost == 1.0f32 {
            return self.clone();
        }
        Bm25Weight {
            idf_explain: self.idf_explain.clone(),
            weight: self.weight * boost,
            cache: self.cache.clone(),
            average_fieldnorm: self.average_fieldnorm,
            k1: self.k1,
            b: self.b,
            delta: self.delta,
        }
    }

    /// Construct a [Bm25Weight] for a phrase of terms.
    pub fn for_terms(
        statistics: &dyn Bm25StatisticsProvider,
        terms: &[Term],
    ) -> crate::Result<Bm25Weight> {
        // INF-135: read per-workspace k1/b from thread-local (set by MegaMem before search).
        Self::for_terms_with_params(statistics, terms, thread_bm25_params())
    }

    /// Construct a [Bm25Weight] for a phrase of terms with custom BM25 parameters.
    /// INF-135: allows per-workspace k1/b tuning without forking callers.
    pub fn for_terms_with_params(
        statistics: &dyn Bm25StatisticsProvider,
        terms: &[Term],
        params: Bm25Params,
    ) -> crate::Result<Bm25Weight> {
        assert!(!terms.is_empty(), "Bm25 requires at least one term");
        let field = terms[0].field();
        for term in &terms[1..] {
            assert_eq!(
                term.field(),
                field,
                "All terms must belong to the same field."
            );
        }

        let total_num_tokens = statistics.total_num_tokens(field)?;
        let total_num_docs = statistics.total_num_docs()?;
        let average_fieldnorm = total_num_tokens as Score / total_num_docs as Score;

        if terms.len() == 1 {
            let term_doc_freq = statistics.doc_freq(&terms[0])?;
            Ok(Bm25Weight::for_one_term_with_params(
                term_doc_freq,
                total_num_docs,
                average_fieldnorm,
                params,
            ))
        } else {
            let mut idf_sum: Score = 0.0;
            for term in terms {
                let term_doc_freq = statistics.doc_freq(term)?;
                idf_sum += idf(term_doc_freq, total_num_docs);
            }
            let idf_explain = Explanation::new("idf", idf_sum);
            Ok(Bm25Weight::new_with_params(idf_explain, average_fieldnorm, params))
        }
    }

    /// Construct a [Bm25Weight] for a single term.
    pub fn for_one_term(
        term_doc_freq: u64,
        total_num_docs: u64,
        avg_fieldnorm: Score,
    ) -> Bm25Weight {
        // INF-135: read per-workspace k1/b from thread-local.
        Self::for_one_term_with_params(term_doc_freq, total_num_docs, avg_fieldnorm, thread_bm25_params())
    }

    /// Construct a [Bm25Weight] for a single term with custom BM25 parameters.
    pub fn for_one_term_with_params(
        term_doc_freq: u64,
        total_num_docs: u64,
        avg_fieldnorm: Score,
        params: Bm25Params,
    ) -> Bm25Weight {
        let idf = idf(term_doc_freq, total_num_docs);
        let mut idf_explain =
            Explanation::new("idf, computed as log(1 + (N - n + 0.5) / (n + 0.5))", idf);
        idf_explain.add_const(
            "n, number of docs containing this term",
            term_doc_freq as Score,
        );
        idf_explain.add_const("N, total number of docs", total_num_docs as Score);
        Bm25Weight::new_with_params(idf_explain, avg_fieldnorm, params)
    }

    /// Construct a [Bm25Weight] for a single term.
    /// This method does not carry the [Explanation] for the idf.
    /// NOTE: INF-135 — serializer uses fixed default params (index-time, not query-time).
    pub fn for_one_term_without_explain(
        term_doc_freq: u64,
        total_num_docs: u64,
        avg_fieldnorm: Score,
    ) -> Bm25Weight {
        let idf = idf(term_doc_freq, total_num_docs);
        // Serializer path uses default params — k1/b are query-time, not index-time.
        Bm25Weight::new_without_explain(idf, avg_fieldnorm)
    }

    pub(crate) fn new(idf_explain: Explanation, average_fieldnorm: Score) -> Bm25Weight {
        Self::new_with_params(idf_explain, average_fieldnorm, Bm25Params::default())
    }

    /// INF-135: internal constructor with explicit k1/b parameters.
    /// SCR-310: also stores delta for BM25+ scoring.
    pub(crate) fn new_with_params(idf_explain: Explanation, average_fieldnorm: Score, params: Bm25Params) -> Bm25Weight {
        let weight = idf_explain.value() * (1.0 + params.k1);
        Bm25Weight {
            idf_explain: Some(idf_explain),
            weight,
            cache: compute_tf_cache(average_fieldnorm, params.k1, params.b),
            average_fieldnorm,
            k1: params.k1,
            b: params.b,
            delta: params.delta,
        }
    }

    pub(crate) fn new_without_explain(idf: f32, average_fieldnorm: Score) -> Bm25Weight {
        let params = Bm25Params::default();
        let weight = idf * (1.0 + params.k1);
        Bm25Weight {
            idf_explain: None,
            weight,
            cache: compute_tf_cache(average_fieldnorm, params.k1, params.b),
            average_fieldnorm,
            k1: params.k1,
            b: params.b,
            delta: params.delta,
        }
    }

    /// Compute the BM25 score of a single document.
    #[inline]
    pub fn score(&self, fieldnorm_id: u8, term_freq: u32) -> Score {
        self.weight * self.tf_factor(fieldnorm_id, term_freq)
    }

    /// Compute the maximum possible BM25 score given this weight.
    pub fn max_score(&self) -> Score {
        self.score(255u8, 2_013_265_944)
    }

    #[inline]
    pub(crate) fn tf_factor(&self, fieldnorm_id: u8, term_freq: u32) -> Score {
        let term_freq = term_freq as Score;
        let norm = self.cache[fieldnorm_id as usize];
        // SCR-310: BM25+ formula — add delta lower bound so rare-term matches
        // in long documents get a minimum positive TF contribution (Lü & Callan 2011).
        // delta=0.0 (default) is identical to standard BM25.
        term_freq / (term_freq + norm) + self.delta
    }

    /// Produce an [Explanation] of a BM25 score.
    pub fn explain(&self, fieldnorm_id: u8, term_freq: u32) -> Explanation {
        // The explain format is directly copied from Lucene's.
        // (So, Kudos to Lucene)
        let score = self.score(fieldnorm_id, term_freq);

        let norm = self.cache[fieldnorm_id as usize];
        let term_freq = term_freq as Score;
        let right_factor = term_freq / (term_freq + norm);

        let mut tf_explanation = Explanation::new(
            "freq / (freq + k1 * (1 - b + b * dl / avgdl))",
            right_factor,
        );

        tf_explanation.add_const("freq, occurrences of term within document", term_freq);
        tf_explanation.add_const("k1, term saturation parameter", self.k1);
        tf_explanation.add_const("b, length normalization parameter", self.b);
        tf_explanation.add_const(
            "dl, length of field",
            FieldNormReader::id_to_fieldnorm(fieldnorm_id) as Score,
        );
        tf_explanation.add_const("avgdl, average length of field", self.average_fieldnorm);

        let mut explanation = Explanation::new("TermQuery, product of...", score);
        explanation.add_detail(Explanation::new("(K1+1)", K1 + 1.0));
        if let Some(idf_explain) = &self.idf_explain {
            explanation.add_detail(idf_explain.clone());
        }
        explanation.add_detail(tf_explanation);
        explanation
    }
}

#[cfg(test)]
mod tests {

    use super::idf;
    use crate::{assert_nearly_equals, Score};

    #[test]
    fn test_idf() {
        let score: Score = 2.0;
        assert_nearly_equals!(idf(1, 2), score.ln());
    }
}

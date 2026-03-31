use kodama::Method;
use std::collections::HashMap;
use std::convert::TryInto;
use std::str::FromStr;
use std::sync::Arc;

/// Candidate-pair generation strategy used by the string pipeline.
///
/// Controls whether [`crate::group_similar`] computes the full O(n²) distance
/// matrix or builds a sparse candidate graph first. See [`crate::qgram_candidates`]
/// for the q-gram strategy details.
#[derive(Debug, Clone)]
pub enum Blocking {
    /// Compute the full distance matrix. Ground-truth partition, O(n²) memory
    /// and work. Right for small inputs or when correctness matters more than
    /// throughput.
    Dense,
    /// Use a character trigram inverted index to emit candidate pairs, then
    /// cluster within connected components. Dramatically reduces work for
    /// sparse-similar data at the cost of possible cluster fragmentation
    /// when truly-similar records share few trigrams.
    QGram {
        /// Filter aggressiveness. Higher values reject more candidates;
        /// typical range: 0.1–0.4.
        tau: f64,
    },
}

impl Default for Blocking {
    fn default() -> Self {
        Self::Dense
    }
}

/// Pre-fit cosine model carried alongside the distance closure when the
/// metric is one of the TF-IDF cosine variants. Lets the string pipeline
/// vectorize each representative once instead of re-vectorizing on every
/// pairwise comparison.
pub(super) struct CosineData {
    pub(super) vocab: Arc<HashMap<String, u32>>,
    pub(super) idf: Arc<Vec<f32>>,
    pub(super) positional: bool,
}

/// Configuration for the string pipeline ([`crate::group_similar`]).
///
/// Bundles a similarity metric (`compare`), a clustering threshold and
/// linkage method, an optional pre-cluster string normalizer, and a
/// candidate-pair strategy ([`Blocking`]).
pub struct Config<V> {
    pub(super) threshold: Threshold,
    pub(super) method: Method,
    /// Distance closure: takes two values and returns a [`crate::Distance`]
    /// in `[0.0, 1.0]` where smaller = more similar. Must be symmetric and
    /// idempotent. Set via the metric constructors or
    /// [`Config::with_compare`] — the latter also clears any metric-specific
    /// precompute (e.g., the cached TF-IDF vocab from [`Config::token_cosine`])
    /// so the new closure is the sole source of truth.
    #[allow(clippy::type_complexity)]
    pub(super) compare: Box<dyn Fn(&V, &V) -> crate::Distance + Send + Sync>,
    pub(super) normalize: Box<dyn Fn(&str) -> String + Send + Sync>,
    pub(super) blocking: Blocking,
    pub(super) cosine: Option<CosineData>,
}

impl<V: AsRef<str>> Config<V> {
    /// Construct a configuration using Jaro-Winkler for structure comparison
    pub fn jaro_winkler(threshold: Threshold) -> Self {
        Config {
            threshold,
            method: Method::Complete,
            compare: Box::new(|a, b| {
                crate::Distance::clamped(1.0 - jaro_winkler::jaro_winkler(a.as_ref(), b.as_ref()))
            }),
            normalize: Box::new(crate::normalize::identity),
            blocking: Blocking::default(),
            cosine: None,
        }
    }

    /// Set a normalizer that transforms strings before deduplication.
    ///
    /// Records whose normalized forms are identical are treated as duplicates,
    /// collapsing them to a single representative before pairwise comparison.
    /// Original values are preserved in output.
    pub fn with_normalizer<F>(mut self, normalize: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.normalize = Box::new(normalize);
        self
    }

    /// Replace the distance closure.
    ///
    /// Also clears any metric-specific precompute on this config — e.g., a
    /// [`Config::token_cosine`] config's cached vocab/IDF — so the new
    /// closure is the only thing [`crate::group_similar`] consults.
    pub fn with_compare<F>(mut self, compare: F) -> Self
    where
        F: Fn(&V, &V) -> crate::Distance + Send + Sync + 'static,
    {
        self.compare = Box::new(compare);
        self.cosine = None;
        self
    }

    /// Enable q-gram blocking with the given filter aggressiveness.
    ///
    /// Trades dense O(n²) work for a candidate graph built from shared
    /// character trigrams. `tau` controls the filter: higher values reject
    /// more pairs (faster, but more risk of fragmenting clusters whose
    /// members share few trigrams). Typical range: 0.1–0.4.
    pub fn with_blocking(mut self, tau: f64) -> Self {
        self.blocking = Blocking::QGram { tau };
        self
    }

    /// Force the dense pipeline (full O(n²) distance matrix). This is the
    /// default — provided as an explicit setter so callers can override a
    /// previously-configured [`Blocking::QGram`].
    pub fn without_blocking(mut self) -> Self {
        self.blocking = Blocking::Dense;
        self
    }

    /// Construct a configuration that compares records using IDF-weighted
    /// token cosine similarity, fitted to the supplied corpus.
    ///
    /// Tokenizes each record on non-alphanumeric boundaries, computes inverse
    /// document frequency across the corpus, and represents each record as a
    /// sparse TF-IDF vector. The `compare` closure returns `1.0 - cosine`, so
    /// the same threshold semantics apply (threshold = max acceptable
    /// dissimilarity, in `[0, 1]`).
    ///
    /// Best for grouping records by entity identity when records contain heavy
    /// shared boilerplate — common tokens get downweighted automatically, so
    /// rare distinctive tokens (entity identifiers, codes) dominate the score.
    pub fn token_cosine(corpus: &[V], threshold: Threshold) -> Self {
        let (vocab, idf) = crate::tokens::build_idf(corpus);
        let vocab = Arc::new(vocab);
        let idf = Arc::new(idf);

        let compare = {
            let vocab = Arc::clone(&vocab);
            let idf = Arc::clone(&idf);
            move |a: &V, b: &V| -> crate::Distance {
                let va = crate::tokens::vectorize(a.as_ref(), &vocab, &idf);
                let vb = crate::tokens::vectorize(b.as_ref(), &vocab, &idf);
                crate::Distance::clamped(1.0 - crate::tokens::sparse_cosine(&va, &vb))
            }
        };

        Config {
            threshold,
            method: Method::Complete,
            compare: Box::new(compare),
            normalize: Box::new(crate::normalize::identity),
            blocking: Blocking::default(),
            cosine: Some(CosineData {
                vocab,
                idf,
                positional: false,
            }),
        }
    }

    /// Like [`token_cosine`], but each token's contribution is scaled by
    /// `1 / (1 + position)` — earlier tokens dominate the similarity score.
    ///
    /// Useful when distinctive identifiers typically lead the string and
    /// shared boilerplate or formatting tokens trail it. With the positional
    /// decay, two records sharing only trailing tokens score much lower than
    /// records sharing their leading tokens, even at the same token-set
    /// overlap.
    pub fn token_cosine_positional(corpus: &[V], threshold: Threshold) -> Self {
        let (vocab, idf) = crate::tokens::build_idf(corpus);
        let vocab = Arc::new(vocab);
        let idf = Arc::new(idf);

        let compare = {
            let vocab = Arc::clone(&vocab);
            let idf = Arc::clone(&idf);
            move |a: &V, b: &V| -> crate::Distance {
                let va = crate::tokens::vectorize_positional(a.as_ref(), &vocab, &idf);
                let vb = crate::tokens::vectorize_positional(b.as_ref(), &vocab, &idf);
                crate::Distance::clamped(1.0 - crate::tokens::sparse_cosine(&va, &vb))
            }
        };

        Config {
            threshold,
            method: Method::Complete,
            compare: Box::new(compare),
            normalize: Box::new(crate::normalize::identity),
            blocking: Blocking::default(),
            cosine: Some(CosineData {
                vocab,
                idf,
                positional: true,
            }),
        }
    }
}

/// `Threshold` is a newtype wrapper describing how permissive comparisons are for a given
/// comparison closure.
///
/// This value is configurable and is a float between 0 and 1; 0 represents a threshold of exact
/// matches, while 1 represents entirely permissive values.
#[derive(Debug, Clone)]
pub struct Threshold(f64);

impl Threshold {
    pub(super) fn within(&self, dissimilarity: f32) -> bool {
        let dissimilarity = dissimilarity as f64;
        dissimilarity <= self.0
    }

    pub(super) fn value(&self) -> f64 {
        self.0
    }
}

impl std::fmt::Display for Threshold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for Threshold {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        f64::from_str(s)
            .map_err(|e| format!("{}", e))
            .and_then(|v| v.try_into())
    }
}

impl Default for Threshold {
    fn default() -> Self {
        Threshold(0.25)
    }
}

impl std::convert::TryFrom<f64> for Threshold {
    type Error = String;

    fn try_from(input: f64) -> Result<Self, Self::Error> {
        if !(0.0..=1.0).contains(&input) {
            Err("Threshold must be between 0 and 1".to_string())
        } else {
            Ok(Threshold(input))
        }
    }
}

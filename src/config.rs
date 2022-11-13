use kodama::Method;
use std::convert::TryInto;
use std::str::FromStr;

/// `Config` manages grouping configuration based on three settings (managed internally);
/// `threshold`, `method`, and `compare`.
///
/// `threshold` captures how strict or permissive the comparison is.
///
/// `method` represents the dendrogram linkage method, and (at least for Jaro-Winkler) should
/// be set to `Method::Complete`.
///
/// Finally, `compare` is the closure used to determine how similar two values are to each other.
/// For string-based comparisons, we make available Jaro-Winkler. This closure must be idempotent.
pub struct Config<V> {
    pub(super) threshold: Threshold,
    pub(super) method: Method,
    /// A closure that takes two values and returns a float between 0.0 and 1.0
    pub compare: Box<dyn Fn(&V, &V) -> f64 + Send + Sync>,
}

impl<V: AsRef<str>> Config<V> {
    /// Construct a configuration using Jaro-Winkler for structure comparison
    pub fn jaro_winkler(threshold: Threshold) -> Self {
        Config {
            threshold,
            method: Method::Complete,
            compare: Box::new(|a, b| 1.0 - jaro_winkler::jaro_winkler(&a.as_ref(), &b.as_ref())),
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
    pub(super) fn within(&self, dissimilarity: f64) -> bool {
        dissimilarity <= self.0
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
        if input < 0.0 || input > 1.0 {
            Err("Threshold must be between 0 and 1".to_string())
        } else {
            Ok(Threshold(input))
        }
    }
}

#![deny(missing_docs)]

//! This crate enables grouping values based on string similarity via
//! [Jaro-Winkler distance](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) and
//! [complete-linkage clustering](https://en.wikipedia.org/wiki/Complete-linkage_clustering).
//!
//! # Example: Identify likely repeated merchants based on merchant name
//!
//! ```
//! use group_similar::{Config, Named, Threshold, group_similar};
//!
//! #[derive(Eq, PartialEq, std::hash::Hash, Debug)]
//! struct Merchant {
//!   id: usize,
//!   name: String
//! }
//!
//! impl Named for Merchant {
//!     fn name(&self) -> &str {
//!         &self.name
//!     }
//! }
//!
//! let merchants = vec![
//!     Merchant {
//!         id: 1,
//!         name: "McDonalds 105109".to_string()
//!     },
//!     Merchant {
//!         id: 2,
//!         name: "McDonalds 105110".to_string()
//!     },
//!     Merchant {
//!         id: 3,
//!         name: "Target ID1244".to_string()
//!     },
//!     Merchant {
//!         id: 4,
//!         name: "Target ID125".to_string()
//!     },
//!     Merchant {
//!         id: 5,
//!         name: "Amazon.com TID120159120".to_string()
//!     },
//!     Merchant {
//!         id: 6,
//!         name: "Target".to_string()
//!     },
//!     Merchant {
//!         id: 7,
//!         name: "Target.com".to_string()
//!     },
//! ];
//!
//! let config = Config::jaro_winkler(Threshold::default());
//! let results = group_similar(&merchants, &config);
//!
//! assert_eq!(results.get(&merchants[0]), Some(&vec![&merchants[1]]));
//! assert_eq!(results.get(&merchants[2]), Some(&vec![&merchants[3], &merchants[5], &merchants[6]]));
//! assert_eq!(results.get(&merchants[4]), Some(&vec![]));
//! ```
//!
mod config;

pub use config::{Config, Named, Threshold};
use indicatif::ParallelProgressIterator;
use kodama::linkage;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;
use std::convert::TryInto;

/// Group records based on a particular configuration
pub fn group_similar<'a, 'b, V>(
    records: &'a [V],
    config: &'b Config<V>,
) -> HashMap<&'a V, Vec<&'a V>>
where
    V: std::hash::Hash + Eq + Sync,
{
    let mut results = HashMap::new();

    let mut condensed = similarity_matrix(records, &config.compare);
    let dend = linkage(&mut condensed, records.len(), config.method);
    let mut dendro: Dendro<f64, &V> = HashMap::default();

    for (idx, record) in records.iter().enumerate() {
        dendro.insert(idx, Dendrogram::Node(record));
    }

    let base = records.len();
    for (idx, step) in dend.steps().iter().enumerate() {
        dendro.insert(base + idx, Dendrogram::Group(step));
    }

    // We reverse this to start with the most dissimilar first, ultimately operating within the
    // threshold. The result is that the largest groups bubble up to the top, ensuring that we're
    // casting the widest net from closer matches when building the result set.
    for (idx, step) in dend.steps().iter().enumerate().rev() {
        if config.threshold.within(step.dissimilarity) {
            match extract_values(&mut dendro, base + idx).as_slice() {
                [first, rest @ ..] => {
                    results.entry(*first).or_insert(vec![]).extend(rest);
                }
                [] => {}
            }
        }
    }

    for node in get_remaining_nodes(&dendro) {
        results.insert(node, vec![]);
    }

    results
}

type Dendro<'a, F, V> = HashMap<usize, Dendrogram<'a, F, V>>;

/// Recursively retrieve this and all child values to completion
///
/// If the value retrieved by index is present in the HashMap, we remove it. If it's an item
/// directly, we push that onto the list of results. If it's a `Step` (which contains pointers to
/// other steps or values), we recurse both sides (as a step can be linked to a value, or even
/// another step).
fn extract_values<'a, 'b, F, V>(dendro: &mut Dendro<'a, F, V>, at: usize) -> Vec<V> {
    let mut results = vec![];

    if let Some(result) = dendro.remove(&at) {
        match result {
            Dendrogram::Group(step) => {
                results.extend(extract_values(dendro, step.cluster1));
                results.extend(extract_values(dendro, step.cluster2));
            }
            Dendrogram::Node(v) => results.push(v),
        }
    }

    results
}

/// Extract all item values from the HashMap
///
/// Given `extract_values` mutates the HashMap, removing values (so as not to duplicate results across
/// different groups, the net result is a HashMap that may contain steps (e.g. that didn't meet the
/// threshold criterion as the steps are too dissimilar) or items (if an item itself is too
/// dissimilar from any other items or steps).
///
/// In this case, we still need to return those items, and have their associated matches be an
/// emtpy Vec, when we add them to the returned HashMap.
fn get_remaining_nodes<'a, 'b, F, V>(dendro: &'b Dendro<'a, F, V>) -> Vec<&'b V> {
    let mut results = vec![];

    for value in dendro.values() {
        if let Dendrogram::Node(v) = value {
            results.push(v);
        }
    }

    results
}

#[derive(Debug)]
enum Dendrogram<'a, F, V> {
    Group(&'a kodama::Step<F>),
    Node(V),
}

/// This function is very heavily inspired by the example provided within the kodama documentation
/// (at https://docs.rs/kodama)
///
/// The biggest differences? This allows for the comparison operation to be provided, and it
/// compares values leveraging Rayon to speed up the operations. Due to the use of Rayon, we
/// construct an intermediate vector to hold positions for calculations.
fn similarity_matrix<V, F>(inputs: &[V], compare: &F) -> Vec<f64>
where
    F: Fn(&V, &V) -> f64 + Send + Sync,
    V: Sync,
{
    let mut intermediate = vec![];
    for row in 0..inputs.len() - 1 {
        for col in row + 1..inputs.len() {
            intermediate.push((row, col))
        }
    }

    let pb = indicatif::ProgressBar::new(intermediate.len().try_into().unwrap());
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("[{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%, {eta_precise})")
            .progress_chars("#>-"),
    );

    intermediate
        .par_iter()
        .progress_with(pb)
        .map(|(row, col)| (compare)(&inputs[*row], &inputs[*col]))
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{group_similar, Config, Named};
    use std::collections::HashMap;
    use std::convert::TryInto;

    impl Named for &str {
        fn name(&self) -> &str {
            self
        }
    }

    #[test]
    fn groups_all_together_with_permissive_threshold() {
        let threshold = 1.0_f64.try_into().expect("permissive threshold");

        let config = Config::jaro_winkler(threshold);

        let values = vec!["hello", "world"];

        let mut outcome = HashMap::new();
        outcome.insert(&"hello", vec![&"world"]);
        assert_eq!(outcome, group_similar(&values, &config));
    }

    #[test]
    fn groups_nothing_together_with_strict_threshold() {
        let threshold = 0.0_f64.try_into().expect("strict threshold");

        let config = Config::jaro_winkler(threshold);

        let values = vec!["hello", "world"];

        let mut outcome = HashMap::new();
        outcome.insert(&"hello", vec![]);
        outcome.insert(&"world", vec![]);
        assert_eq!(outcome, group_similar(&values, &config));
    }

    #[test]
    fn groups_similar_values() {
        let threshold = 0.25_f64.try_into().expect("fairly strict threshold");

        let config = Config::jaro_winkler(threshold);

        let values = vec!["Jane", "June", "Joan", "Joseph"];

        let mut outcome = HashMap::new();
        outcome.insert(&"Jane", vec![&"June"]);
        outcome.insert(&"Joan", vec![]);
        outcome.insert(&"Joseph", vec![]);
        assert_eq!(outcome, group_similar(&values, &config));
    }

    #[test]
    fn groups_similar_values_loosely() {
        let threshold = 0.5_f64.try_into().expect("fairly permissive threshold");

        let config = Config::jaro_winkler(threshold);

        let values = vec![
            "Henry", "Jane", "June", "Joan", "José", "Barry", "Joseph", "Mary", "Henry", "Harry",
        ];

        let mut outcome = HashMap::new();
        outcome.insert(&"José", vec![&"Joseph", &"Joan", &"Jane", &"June"]);
        outcome.insert(&"Henry", vec![&"Henry", &"Mary", &"Barry", &"Harry"]);
        assert_eq!(outcome, group_similar(&values, &config));
    }
}

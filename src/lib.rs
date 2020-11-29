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
use kodama::linkage;
use rayon::prelude::*;
use std::collections::HashMap;

/// Group records based on a particular configuration
pub fn group_similar<'a, 'b, V>(
    records: &'a [V],
    config: &'b Config<V>,
) -> HashMap<&'a V, Vec<&'a V>>
where
    V: std::hash::Hash + Eq + Sync,
{
    let mut ret = HashMap::new();

    let mut condensed = similarity_matrix(records, &config.compare);
    let dend = linkage(&mut condensed, records.len(), config.method);
    let mut full: Full<f64, &V> = HashMap::default();

    for (idx, r) in records.iter().enumerate() {
        full.insert(idx, Item::ItemRef(r));
    }

    let base = records.len();
    for (idx, step) in dend.steps().iter().enumerate() {
        full.insert(base + idx, Item::StepRef(step));
    }

    for (idx, step) in dend.steps().iter().enumerate().rev() {
        if config.threshold.within(step.dissimilarity) {
            match get_values(&mut full, base + idx).as_slice() {
                [first, rest @ ..] => {
                    ret.entry(*first).or_insert(vec![]).extend(rest);
                }
                [] => {}
            }
        }
    }

    for item in get_items(&full) {
        ret.insert(item, vec![]);
    }

    ret
}

type Full<'a, F, V> = HashMap<usize, Item<'a, F, V>>;

fn get_values<'a, 'b, F, V>(full: &mut Full<'a, F, V>, at: usize) -> Vec<V> {
    let mut results = vec![];

    if let Some(result) = full.remove(&at) {
        match result {
            Item::StepRef(step) => {
                results.extend(get_values(full, step.cluster1));
                results.extend(get_values(full, step.cluster2));
            }
            Item::ItemRef(v) => results.push(v),
        }
    }

    results
}

fn get_items<'a, 'b, F, V>(full: &'b Full<'a, F, V>) -> Vec<&'b V> {
    let mut results = vec![];

    for value in full.values() {
        if let Item::ItemRef(v) = value {
            results.push(v);
        }
    }

    results
}

#[derive(Debug)]
enum Item<'a, F, V> {
    StepRef(&'a kodama::Step<F>),
    ItemRef(V),
}

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

    intermediate
        .par_iter()
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

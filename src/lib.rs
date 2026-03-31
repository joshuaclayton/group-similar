#![deny(missing_docs)]

//! Hierarchical clustering by similarity, with batteries-included string metrics.
//!
//! The crate exposes two layers:
//!
//! - **Layer 1 (metric-agnostic):** [`cluster`] and [`cluster_with_candidates`]
//!   run [complete-linkage hierarchical clustering](https://en.wikipedia.org/wiki/Complete-linkage_clustering)
//!   over any `&[T]` with a user-supplied distance function. Output is a
//!   [`Clusters`] of indices into the input slice.
//!
//! - **Layer 2 (string pipeline):** [`group_similar`] adds normalize → dedup
//!   → cluster → expand glue for string-bearing records, using a [`Config`]
//!   that bundles a metric (Jaro-Winkler, IDF-weighted token cosine, …) with
//!   a threshold, normalizer, and candidate-pair strategy ([`Blocking`]).
//!
//! # Example: string pipeline
//!
//! ```
//! use group_similar::{Config, Threshold, group_similar};
//! use std::convert::TryInto;
//!
//! #[derive(Eq, PartialEq, std::hash::Hash, Debug, Ord, PartialOrd)]
//! struct Merchant {
//!     id: usize,
//!     name: String,
//! }
//!
//! impl AsRef<str> for Merchant {
//!     fn as_ref(&self) -> &str { &self.name }
//! }
//!
//! let merchants = vec![
//!     Merchant { id: 1, name: "McDonalds 105109".to_string() },
//!     Merchant { id: 2, name: "McDonalds 105110".to_string() },
//!     Merchant { id: 3, name: "Target ID1244".to_string() },
//!     Merchant { id: 4, name: "Target ID125".to_string() },
//!     Merchant { id: 5, name: "Amazon.com TID120159120".to_string() },
//!     Merchant { id: 6, name: "Target".to_string() },
//!     Merchant { id: 7, name: "Target.com".to_string() },
//! ];
//!
//! // Fit IDF-weighted token cosine similarity to the corpus, with leading-
//! // token positional weighting (distinctive identifiers tend to lead the
//! // string; shared boilerplate trails).
//! let threshold: Threshold = 0.4_f64.try_into().unwrap();
//! let config = Config::token_cosine_positional(&merchants, threshold);
//! let results = group_similar(&merchants, &config);
//!
//! // Every input appears exactly once across keys + values.
//! let total: usize = results.iter().map(|(_, vs)| 1 + vs.len()).sum();
//! assert_eq!(total, merchants.len());
//! ```
//!
//! # Example: metric-agnostic clustering
//!
//! ```
//! use group_similar::{cluster, Clusters, Distance, Method, Threshold};
//! use std::convert::TryInto;
//!
//! // Cluster floats in [0, 1] by absolute difference.
//! let items = vec![0.0_f32, 0.05, 0.1, 0.9, 0.95];
//! let threshold: Threshold = 0.2_f64.try_into().unwrap();
//!
//! let clusters: Clusters = cluster(
//!     &items,
//!     |a: &f32, b: &f32| Distance::clamped((a - b).abs()),
//!     threshold,
//!     Method::Complete,
//! );
//!
//! // Two clusters: {0.0, 0.05, 0.1} and {0.9, 0.95}.
//! assert_eq!(clusters.matched.len(), 2);
//! assert_eq!(clusters.unmatched.len(), 0);
//! ```

mod config;
pub mod normalize;
mod tokens;

pub use config::{Blocking, Config, Threshold};
pub use kodama::Method;

use kodama::linkage;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Pairwise distance in `[0.0, 1.0]`. Smaller = more similar.
///
/// Layer 1 ([`cluster`], [`cluster_with_candidates`]) requires every distance
/// value to land in this range; the type system enforces it. Built-in metric
/// constructors on [`Config`] always return valid `Distance` values. For
/// user-supplied distance closures, construct values via [`Distance::new`]
/// (fail-fast on out-of-range) or [`Distance::clamped`] (saturate, `NaN` →
/// [`Distance::MAX`]).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Distance(f32);

impl Distance {
    /// Identity — two items are exactly the same.
    pub const MIN: Self = Self(0.0);
    /// Maximally dissimilar — "definitely not a match."
    pub const MAX: Self = Self(1.0);

    /// Construct from a raw `f32`. Returns `None` if the value is `NaN`,
    /// infinite, or outside `[0.0, 1.0]`.
    pub fn new(value: f32) -> Option<Self> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Construct, clamping out-of-range values to the nearest endpoint.
    /// `NaN` becomes [`Distance::MAX`] — treating an unknowable distance as
    /// "definitely not a match" so it never merges two clusters.
    pub fn clamped(value: f32) -> Self {
        if value.is_nan() {
            Self::MAX
        } else {
            Self(value.clamp(0.0, 1.0))
        }
    }

    /// Underlying `f32`, for interop with raw-numerical APIs (e.g. building
    /// a kodama condensed matrix).
    pub fn value(self) -> f32 {
        self.0
    }
}

/// Result of clustering: groups of items that merged into clusters, plus the
/// singletons that did not.
///
/// All members are referenced by index into the slice passed to [`cluster`] or
/// [`cluster_with_candidates`]. Use the indices to project back to whatever
/// payload type you care about.
#[derive(Debug, Clone, Default)]
pub struct Clusters {
    /// Groups of 2+ items whose pairwise distances were within threshold.
    pub matched: Vec<Vec<usize>>,
    /// Items that did not cluster with any other.
    pub unmatched: Vec<usize>,
}

/// Run hierarchical clustering over `items` using `distance`.
///
/// Builds the full O(n²) condensed distance matrix in parallel, runs kodama
/// linkage with the supplied `method`, then walks the dendrogram and splits
/// steps whose dissimilarity is within `threshold` into matched groups; the
/// remaining leaves are reported as unmatched.
///
/// `T` may be anything `Sync`; `distance` must be symmetric and idempotent.
/// The [`Distance`] return type enforces values in `[0, 1]` at the
/// construction site.
pub fn cluster<T, F>(items: &[T], distance: F, threshold: Threshold, method: Method) -> Clusters
where
    T: Sync,
    F: Fn(&T, &T) -> Distance + Send + Sync,
{
    let n = items.len();
    if n == 0 {
        return Clusters::default();
    }
    if n == 1 {
        return Clusters {
            matched: vec![],
            unmatched: vec![0],
        };
    }

    let mut condensed = similarity_matrix(items, &distance);
    let dend = linkage(&mut condensed, n, method);
    extract_index_clusters(&dend, n, &threshold)
}

/// Run hierarchical clustering using a caller-supplied set of candidate pairs.
///
/// Distance is computed only for `candidates`; pairs with distance within
/// `threshold` define an edge graph whose connected components run
/// complete-link clustering independently. When building each component's
/// distance matrix, any pair not present in `candidates` has its distance
/// computed at that moment — a single missing edge would otherwise be
/// treated as "above threshold" by complete-link and fragment a cluster
/// that should be intact.
///
/// Use this when the full O(n²) matrix is infeasible and you have a cheap
/// way to enumerate likely-similar pairs. For strings, [`qgram_candidates`]
/// is a built-in candidate generator.
pub fn cluster_with_candidates<T, F>(
    items: &[T],
    candidates: &[(usize, usize)],
    distance: F,
    threshold: Threshold,
    method: Method,
) -> Clusters
where
    T: Sync,
    F: Fn(&T, &T) -> Distance + Send + Sync,
{
    let n = items.len();
    if n == 0 {
        return Clusters::default();
    }
    if n == 1 {
        return Clusters {
            matched: vec![],
            unmatched: vec![0],
        };
    }

    let threshold_dis = threshold.value() as f32;

    let edges: Vec<(usize, usize, f32)> = candidates
        .par_iter()
        .filter_map(|&(i, j)| {
            let dist = distance(&items[i], &items[j]).value();
            if dist > threshold_dis {
                return None;
            }
            Some((i, j, dist))
        })
        .collect();

    let mut uf = UnionFind::new(n);
    for &(i, j, _) in &edges {
        uf.union(i, j);
    }

    let mut edge_map: HashMap<(usize, usize), f32> = HashMap::with_capacity(edges.len());
    for &(i, j, d) in &edges {
        edge_map.insert((i.min(j), i.max(j)), d);
    }

    let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        components.entry(uf.find(i)).or_default().push(i);
    }

    let mut matched: Vec<Vec<usize>> = Vec::new();
    let mut unmatched: Vec<usize> = Vec::new();

    for (_, members) in components {
        if members.len() == 1 {
            unmatched.push(members[0]);
            continue;
        }

        let k = members.len();
        let mut condensed = Vec::with_capacity(k * (k - 1) / 2);
        for ii in 0..k {
            for jj in (ii + 1)..k {
                let a = members[ii].min(members[jj]);
                let b = members[ii].max(members[jj]);
                let d = match edge_map.get(&(a, b)) {
                    Some(&d) => d,
                    None => distance(&items[a], &items[b]).value(),
                };
                condensed.push(d);
            }
        }

        let dend = linkage(&mut condensed, k, method);
        let local = extract_index_clusters(&dend, k, &threshold);

        for group in local.matched {
            matched.push(group.into_iter().map(|local_i| members[local_i]).collect());
        }
        for local_i in local.unmatched {
            unmatched.push(members[local_i]);
        }
    }

    Clusters { matched, unmatched }
}

/// Build a condensed pairwise distance matrix using `compare`, parallelized
/// with Rayon. Returns the upper triangle in row-major order (`n*(n-1)/2`
/// entries) as raw `f32`s — the layout expected by kodama. The closure's
/// [`Distance`] return is unwrapped at the matrix boundary.
pub fn similarity_matrix<T, F>(inputs: &[T], compare: &F) -> Vec<f32>
where
    F: Fn(&T, &T) -> Distance + Send + Sync,
    T: Sync,
{
    if inputs.is_empty() {
        return vec![];
    }

    let ilen = inputs.len();
    let intermediate = (0..ilen - 1)
        .flat_map(|row| (row + 1..ilen).map(move |col| (row, col)))
        .collect::<Vec<_>>();

    intermediate
        .par_iter()
        .map(|(row, col)| (compare)(&inputs[*row], &inputs[*col]).value())
        .collect::<Vec<_>>()
}

/// Generate candidate pairs for [`cluster_with_candidates`] using a character
/// trigram inverted index.
///
/// A pair `(i, j)` is emitted when records `i` and `j` share at least
/// `max(1, tau_coef × min(unique_trigrams_i, unique_trigrams_j))` trigrams.
/// Records with fewer than 15 unique trigrams bypass the filter (they don't
/// have enough signal for the heuristic to be reliable). Very common trigrams
/// (present in ≥80% of records) are skipped — they don't discriminate and
/// dominate scan cost.
///
/// `tau_coef` controls aggressiveness: higher values reject more pairs but
/// risk false negatives (within-threshold pairs whose trigram overlap was
/// below `tau`). Typical range: 0.1–0.4. `cluster_with_candidates` recovers
/// most false negatives within a component, but pairs that should bridge two
/// components and don't appear here will fragment the cluster.
pub fn qgram_candidates<S: AsRef<str>>(items: &[S], tau_coef: f64) -> Vec<(usize, usize)> {
    let strs: Vec<&str> = items.iter().map(|s| s.as_ref()).collect();
    let (index, qgram_counts) = build_qgram_index(&strs);
    generate_candidates(&strs, &index, &qgram_counts, tau_coef)
}

fn extract_index_clusters(
    dend: &kodama::Dendrogram<f32>,
    n: usize,
    threshold: &Threshold,
) -> Clusters {
    let mut dendro: BTreeMap<usize, Dendro<f32, usize>> = BTreeMap::default();
    for i in 0..n {
        dendro.insert(i, Dendro::Node(i));
    }
    let base = n;
    for (idx, step) in dend.steps().iter().enumerate() {
        dendro.insert(base + idx, Dendro::Group(step));
    }

    // Walk steps from the most-dissimilar end downward so the largest groups
    // within threshold are emitted first; smaller subclusters are then
    // absorbed by `extract_values` removing nodes from the dendro map.
    let mut matched = Vec::new();
    for (idx, step) in dend.steps().iter().enumerate().rev() {
        if threshold.within(step.dissimilarity) {
            let group = extract_values(&mut dendro, base + idx);
            if !group.is_empty() {
                matched.push(group);
            }
        }
    }

    let unmatched = dendro
        .into_values()
        .filter_map(|v| match v {
            Dendro::Node(i) => Some(i),
            _ => None,
        })
        .collect();

    Clusters { matched, unmatched }
}

#[derive(Debug)]
enum Dendro<'a, F, V> {
    Group(&'a kodama::Step<F>),
    Node(V),
}

fn extract_values<'a, F, V>(dendro: &mut BTreeMap<usize, Dendro<'a, F, V>>, at: usize) -> Vec<V> {
    let mut results = vec![];

    if let Some(result) = dendro.remove(&at) {
        match result {
            Dendro::Group(step) => {
                results.extend(extract_values(dendro, step.cluster1));
                results.extend(extract_values(dendro, step.cluster2));
            }
            Dendro::Node(v) => results.push(v),
        }
    }

    results
}

/// Disjoint-set / union-find with path compression and union-by-rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cur = x;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] = self.rank[rx].saturating_add(1);
            }
        }
    }
}

fn qgram3_to_u32(b: &[u8]) -> u32 {
    (b[0] as u32) | ((b[1] as u32) << 8) | ((b[2] as u32) << 16)
}

fn build_qgram_index(strs: &[&str]) -> (HashMap<u32, Vec<usize>>, Vec<u32>) {
    let n = strs.len();
    let mut index: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut qgram_counts = vec![0u32; n];

    for (i, s) in strs.iter().enumerate() {
        let bytes = s.as_bytes();
        if bytes.len() < 3 {
            continue;
        }
        let mut seen: HashSet<u32> = HashSet::new();
        for w in bytes.windows(3) {
            let g = qgram3_to_u32(w);
            if seen.insert(g) {
                index.entry(g).or_default().push(i);
            }
        }
        qgram_counts[i] = seen.len() as u32;
    }

    (index, qgram_counts)
}

fn generate_candidates(
    strs: &[&str],
    index: &HashMap<u32, Vec<usize>>,
    qgram_counts: &[u32],
    tau_coef: f64,
) -> Vec<(usize, usize)> {
    let n = strs.len();
    let common_threshold = (n * 4) / 5;

    (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let bytes = strs[i].as_bytes();
            if bytes.len() < 3 {
                return Vec::new();
            }

            let mut seen: HashSet<u32> = HashSet::new();
            let mut counts = vec![0u32; n];

            for w in bytes.windows(3) {
                let g = qgram3_to_u32(w);
                if !seen.insert(g) {
                    continue;
                }
                if let Some(bucket) = index.get(&g) {
                    if bucket.len() > common_threshold {
                        continue;
                    }
                    for &j in bucket {
                        if j > i {
                            counts[j] += 1;
                        }
                    }
                }
            }

            const MIN_QGRAMS_FOR_FILTER: u32 = 15;

            let qi_count = qgram_counts[i];
            ((i + 1)..n)
                .filter_map(|j| {
                    let qj_count = qgram_counts[j];
                    if qi_count < MIN_QGRAMS_FOR_FILTER || qj_count < MIN_QGRAMS_FOR_FILTER {
                        return Some((i, j));
                    }
                    let min_q = qi_count.min(qj_count) as f64;
                    let tau = (tau_coef * min_q).max(1.0) as u32;
                    if counts[j] >= tau {
                        Some((i, j))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Deduplicated records: unique representatives plus a map from normalized
/// string back to all original records sharing that normalized form.
struct Deduplicated<'a, V> {
    representatives: Vec<&'a V>,
    duplicates: BTreeMap<String, Vec<&'a V>>,
}

/// Collapse duplicate strings, keeping one representative per unique
/// normalized value. Preserves insertion order of first-seen normalized forms
/// so that the representative list is deterministic relative to the input.
fn deduplicate<'a, V>(records: &'a [V], normalize: &dyn Fn(&str) -> String) -> Deduplicated<'a, V>
where
    V: AsRef<str>,
{
    let mut order: Vec<String> = Vec::new();
    let mut duplicates: BTreeMap<String, Vec<&'a V>> = BTreeMap::new();

    for record in records {
        let key = normalize(record.as_ref());
        duplicates
            .entry(key)
            .or_insert_with_key(|k| {
                order.push(k.clone());
                Vec::new()
            })
            .push(record);
    }

    let representatives: Vec<&'a V> = order.iter().map(|k| duplicates[k][0]).collect();
    Deduplicated {
        representatives,
        duplicates,
    }
}

/// Cluster output projected back to representative references — the shape
/// `expand` consumes.
struct Clustered<'a, V> {
    matched: Vec<Vec<&'a V>>,
    unmatched: Vec<&'a V>,
}

fn resolve_string_clusters<'a, V>(
    clusters: Clusters,
    representatives: &[&'a V],
) -> Clustered<'a, V> {
    let matched: Vec<Vec<&'a V>> = clusters
        .matched
        .into_iter()
        .map(|group| group.into_iter().map(|i| representatives[i]).collect())
        .collect();
    let unmatched: Vec<&'a V> = clusters
        .unmatched
        .into_iter()
        .map(|i| representatives[i])
        .collect();
    Clustered { matched, unmatched }
}

/// Expand clustered representative groups back to include every original
/// duplicate, producing the final result map keyed by representative.
fn expand<'a, V>(
    clustered: Clustered<'a, V>,
    duplicates: &BTreeMap<String, Vec<&'a V>>,
    normalize: &dyn Fn(&str) -> String,
) -> BTreeMap<&'a V, Vec<&'a V>>
where
    V: AsRef<str> + Ord,
{
    let mut results = BTreeMap::new();

    for group in &clustered.matched {
        let mut expanded: Vec<&'a V> = Vec::new();
        for rep in group {
            let key = normalize(rep.as_ref());
            expanded.extend(duplicates[&key].iter().copied());
        }
        if let [first, rest @ ..] = expanded.as_slice() {
            results.entry(*first).or_insert(vec![]).extend(rest);
        }
    }

    for rep in &clustered.unmatched {
        let key = normalize(rep.as_ref());
        let all = &duplicates[&key];
        let (first, rest) = all.split_first().unwrap();
        results.entry(*first).or_insert_with(|| rest.to_vec());
    }

    results
}

/// Group string-bearing records via normalize → dedup → cluster → expand.
///
/// Candidate-pair generation is controlled by [`Config::with_blocking`] /
/// [`Config::without_blocking`]; the default is [`Blocking::Dense`] (full
/// O(n²) matrix, ground-truth partition). For larger inputs where the dense
/// matrix is expensive, switch to [`Blocking::QGram`] via
/// `config.with_blocking(tau)`.
pub fn group_similar<'a, V>(records: &'a [V], config: &Config<V>) -> BTreeMap<&'a V, Vec<&'a V>>
where
    V: std::hash::Hash + AsRef<str> + Eq + Sync + Ord,
{
    if records.is_empty() {
        return BTreeMap::new();
    }

    let normalize = &*config.normalize;
    let deduped = deduplicate(records, normalize);

    if deduped.representatives.len() <= 1 {
        return expand(
            Clustered {
                matched: vec![deduped.representatives],
                unmatched: vec![],
            },
            &deduped.duplicates,
            normalize,
        );
    }

    let candidates: Option<Vec<(usize, usize)>> = match &config.blocking {
        Blocking::Dense => None,
        Blocking::QGram { tau } => Some(qgram_candidates(&deduped.representatives, *tau)),
    };

    let clusters = if let Some(cd) = &config.cosine {
        // Fast path: vectorize each representative once, then cluster over the
        // cached sparse vectors. Avoids re-vectorizing both sides of every
        // pairwise comparison, which dominates the runtime for cosine metrics
        // with non-trivial corpora.
        let vocab = &*cd.vocab;
        let idf = &*cd.idf;
        let vectors: Vec<Vec<(u32, f32)>> = if cd.positional {
            deduped
                .representatives
                .par_iter()
                .map(|r| crate::tokens::vectorize_positional(r.as_ref(), vocab, idf))
                .collect()
        } else {
            deduped
                .representatives
                .par_iter()
                .map(|r| crate::tokens::vectorize(r.as_ref(), vocab, idf))
                .collect()
        };
        let distance = |a: &Vec<(u32, f32)>, b: &Vec<(u32, f32)>| {
            Distance::clamped(1.0 - crate::tokens::sparse_cosine(a, b))
        };
        match &candidates {
            None => cluster(&vectors, distance, config.threshold.clone(), config.method),
            Some(c) => cluster_with_candidates(
                &vectors,
                c,
                distance,
                config.threshold.clone(),
                config.method,
            ),
        }
    } else {
        let distance = |a: &&V, b: &&V| (config.compare)(*a, *b);
        match &candidates {
            None => cluster(
                &deduped.representatives,
                distance,
                config.threshold.clone(),
                config.method,
            ),
            Some(c) => cluster_with_candidates(
                &deduped.representatives,
                c,
                distance,
                config.threshold.clone(),
                config.method,
            ),
        }
    };
    let clustered = resolve_string_clusters(clusters, &deduped.representatives);

    expand(clustered, &deduped.duplicates, normalize)
}

#[cfg(test)]
mod tests {
    use crate::{group_similar, Config};
    use std::collections::BTreeMap;
    use std::convert::TryInto;

    #[test]
    fn allows_for_empty_list() {
        let threshold = 1.0_f64.try_into().expect("permissive threshold");

        let config: Config<&str> = Config::jaro_winkler(threshold);

        let values = vec![];

        let outcome = BTreeMap::new();
        assert_eq!(outcome, group_similar(&values, &config));
    }

    #[test]
    fn groups_all_together_with_permissive_threshold() {
        let threshold = 1.0_f64.try_into().expect("permissive threshold");

        let config = Config::jaro_winkler(threshold);

        let values = vec!["hello", "world"];

        let mut outcome = BTreeMap::new();
        outcome.insert(&"hello", vec![&"world"]);
        assert_eq!(outcome, group_similar(&values, &config));
    }

    #[test]
    fn groups_nothing_together_with_strict_threshold() {
        let threshold = 0.0_f64.try_into().expect("strict threshold");

        let config = Config::jaro_winkler(threshold);

        let values = vec!["hello", "world"];

        let mut outcome = BTreeMap::new();
        outcome.insert(&"hello", vec![]);
        outcome.insert(&"world", vec![]);
        assert_eq!(outcome, group_similar(&values, &config));
    }

    #[test]
    fn groups_similar_values() {
        let threshold = 0.25_f64.try_into().expect("fairly strict threshold");

        let config = Config::jaro_winkler(threshold);

        let values = vec!["Jane", "June", "Joan", "Joseph"];

        let mut outcome = BTreeMap::new();
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

        let mut outcome = BTreeMap::new();
        outcome.insert(&"José", vec![&"Joseph", &"Joan", &"Jane", &"June"]);
        outcome.insert(&"Henry", vec![&"Henry", &"Mary", &"Barry", &"Harry"]);
        assert_eq!(outcome, group_similar(&values, &config));
    }

    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn prop_every_input_appears_in_output(values: Vec<String>) -> bool {
        if values.is_empty() {
            return true;
        }
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let config: Config<&str> = Config::jaro_winkler(0.25_f64.try_into().unwrap());
        let result = group_similar(&refs, &config);

        let mut seen: Vec<&&str> = Vec::new();
        for (k, vs) in &result {
            seen.push(k);
            seen.extend(vs.iter());
        }
        seen.sort();

        let mut expected: Vec<&&str> = refs.iter().collect();
        expected.sort();

        seen == expected
    }

    #[quickcheck]
    fn prop_normalizer_does_not_lose_records(values: Vec<String>) -> bool {
        if values.is_empty() {
            return true;
        }
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let config: Config<&str> = Config::jaro_winkler(0.25_f64.try_into().unwrap())
            .with_normalizer(crate::normalize::default_normalizer());
        let result = group_similar(&refs, &config);

        let mut seen: Vec<&&str> = Vec::new();
        for (k, vs) in &result {
            seen.push(k);
            seen.extend(vs.iter());
        }
        seen.len() == refs.len()
    }

    #[quickcheck]
    fn prop_empty_threshold_produces_singletons(values: Vec<String>) -> bool {
        if values.is_empty() {
            return true;
        }
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let config: Config<&str> = Config::jaro_winkler(0.0_f64.try_into().unwrap());
        let result = group_similar(&refs, &config);

        result.iter().all(|(k, vs)| vs.iter().all(|v| k == v))
    }

    #[test]
    fn layer1_cluster_partitions_by_distance() {
        use crate::{cluster, Distance, Method};

        // Two well-separated clusters in [0, 1].
        let items = vec![0.0_f32, 0.05, 0.1, 0.9, 0.95];
        let threshold = 0.2_f64.try_into().unwrap();
        let result = cluster(
            &items,
            |a: &f32, b: &f32| Distance::clamped((a - b).abs()),
            threshold,
            Method::Complete,
        );

        assert_eq!(result.matched.len(), 2);
        assert_eq!(result.unmatched.len(), 0);

        let sizes: Vec<usize> = {
            let mut s: Vec<usize> = result.matched.iter().map(|g| g.len()).collect();
            s.sort();
            s
        };
        assert_eq!(sizes, vec![2, 3]);
    }

    #[test]
    fn layer1_cluster_with_candidates_matches_dense_on_separable_data() {
        use crate::{cluster, cluster_with_candidates, Distance, Method};

        let items = vec![0.0_f32, 0.05, 0.1, 0.9, 0.95];
        let threshold: crate::Threshold = 0.2_f64.try_into().unwrap();
        let distance = |a: &f32, b: &f32| Distance::clamped((a - b).abs());

        let dense = cluster(&items, distance, threshold.clone(), Method::Complete);

        // Hand-supplied candidates covering every within-cluster pair.
        let candidates = vec![(0, 1), (0, 2), (1, 2), (3, 4)];
        let blocked =
            cluster_with_candidates(&items, &candidates, distance, threshold, Method::Complete);

        let mut dense_sizes: Vec<usize> = dense.matched.iter().map(|g| g.len()).collect();
        let mut blocked_sizes: Vec<usize> = blocked.matched.iter().map(|g| g.len()).collect();
        dense_sizes.sort();
        blocked_sizes.sort();
        assert_eq!(dense_sizes, blocked_sizes);
    }

    #[test]
    fn layer1_empty_input_returns_empty_clusters() {
        use crate::{cluster, Distance, Method};

        let items: Vec<f32> = vec![];
        let result = cluster(
            &items,
            |a: &f32, b: &f32| Distance::clamped((a - b).abs()),
            0.5_f64.try_into().unwrap(),
            Method::Complete,
        );
        assert!(result.matched.is_empty());
        assert!(result.unmatched.is_empty());
    }

    #[test]
    fn layer1_single_input_is_unmatched() {
        use crate::{cluster, Distance, Method};

        let items = vec![1.0_f32];
        let result = cluster(
            &items,
            |a: &f32, b: &f32| Distance::clamped((a - b).abs()),
            0.5_f64.try_into().unwrap(),
            Method::Complete,
        );
        assert!(result.matched.is_empty());
        assert_eq!(result.unmatched, vec![0]);
    }

    #[test]
    fn config_with_blocking_dispatches_through_group_similar() {
        // Both paths should produce identical partitions on this clearly
        // separable input.
        let threshold = 0.5_f64.try_into().unwrap();
        let values = vec![
            "Henry", "Jane", "June", "Joan", "José", "Barry", "Joseph", "Mary", "Henry", "Harry",
        ];

        let dense_config = Config::jaro_winkler(0.5_f64.try_into().unwrap());
        let blocked_config = Config::jaro_winkler(threshold).with_blocking(0.0);

        let dense = group_similar(&values, &dense_config);
        let blocked = group_similar(&values, &blocked_config);
        assert_eq!(dense, blocked);
    }
}

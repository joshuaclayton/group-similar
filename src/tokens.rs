//! IDF-weighted token cosine similarity.
//!
//! Tokenize each record on non-alphanumeric boundaries, compute inverse
//! document frequency over the corpus, represent each record as a sparse
//! TF-IDF vector, and use cosine similarity between vectors as the metric.
//!
//! Best for "is this the same entity" problems where records share rare
//! distinctive tokens (entity identifiers, codes) but vary in common
//! boilerplate tokens that should be downweighted.

use std::collections::HashMap;

/// Split a string into alphanumeric tokens (whitespace and punctuation are
/// separators). Tokens are case-sensitive — callers can lowercase upstream
/// if case-insensitive matching is desired.
pub(crate) fn tokenize(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
}

/// Scan the corpus and build a vocabulary plus IDF vector.
///
/// Returns `(vocab, idf)` where `vocab: HashMap<String, u32>` maps each token
/// to a stable integer id, and `idf[id]` is its inverse document frequency
/// `ln(N / df(t))` (using `ln_1p` to keep the result strictly positive when
/// a token appears in every record).
pub(crate) fn build_idf<V: AsRef<str>>(corpus: &[V]) -> (HashMap<String, u32>, Vec<f32>) {
    let mut vocab: HashMap<String, u32> = HashMap::new();
    let mut df: Vec<u32> = Vec::new();
    let mut next_id: u32 = 0;

    for record in corpus {
        let mut seen_ids: Vec<u32> = Vec::new();
        for tok in tokenize(record.as_ref()) {
            let id = match vocab.get(tok) {
                Some(&id) => id,
                None => {
                    let id = next_id;
                    next_id += 1;
                    vocab.insert(tok.to_string(), id);
                    df.push(0);
                    id
                }
            };
            // Per-document dedup so df counts documents, not occurrences.
            if !seen_ids.contains(&id) {
                seen_ids.push(id);
                df[id as usize] += 1;
            }
        }
    }

    let n = corpus.len() as f32;
    // ln_1p(x) = ln(1 + x) — keeps result strictly positive when df = N.
    let idf: Vec<f32> = df.iter().map(|c| (n / (*c as f32)).ln_1p()).collect();

    (vocab, idf)
}

/// Vectorize a single string against a precomputed vocab and idf.
/// Tokens not in the vocab are skipped. Returns a sparse vector sorted by
/// token id, suitable for the merge-style sparse cosine below.
pub(crate) fn vectorize(s: &str, vocab: &HashMap<String, u32>, idf: &[f32]) -> Vec<(u32, f32)> {
    let mut tf: HashMap<u32, f32> = HashMap::new();
    for tok in tokenize(s) {
        if let Some(&id) = vocab.get(tok) {
            *tf.entry(id).or_insert(0.0) += 1.0;
        }
    }
    let mut vec: Vec<(u32, f32)> = tf
        .into_iter()
        .map(|(id, count)| (id, count * idf[id as usize]))
        .collect();
    vec.sort_by_key(|(id, _)| *id);
    vec
}

/// Vectorize with positional decay — each token's contribution is weighted
/// by `1 / (1 + position)`, so early tokens dominate. Combined with IDF this
/// emphasizes rare distinctive tokens that lead the string and downweights
/// trailing boilerplate or formatting tokens.
pub(crate) fn vectorize_positional(
    s: &str,
    vocab: &HashMap<String, u32>,
    idf: &[f32],
) -> Vec<(u32, f32)> {
    let mut weights: HashMap<u32, f32> = HashMap::new();
    for (pos, tok) in tokenize(s).enumerate() {
        if let Some(&id) = vocab.get(tok) {
            let pos_weight = 1.0 / (1.0 + pos as f32);
            *weights.entry(id).or_insert(0.0) += pos_weight;
        }
    }
    let mut vec: Vec<(u32, f32)> = weights
        .into_iter()
        .map(|(id, w)| (id, w * idf[id as usize]))
        .collect();
    vec.sort_by_key(|(id, _)| *id);
    vec
}

/// Cosine similarity between two sparse vectors sorted by id.
/// Returns 0.0 when either side has zero norm.
pub(crate) fn sparse_cosine(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let mut dot = 0.0f32;
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                dot += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
        }
    }
    let norm_a: f32 = a.iter().map(|(_, v)| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|(_, v)| v * v).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_strips_punctuation_and_whitespace() {
        let toks: Vec<&str> = tokenize("ALPHA BETA GAMMA DELTA.NET/PATH X123ABC").collect();
        assert_eq!(
            toks,
            vec!["ALPHA", "BETA", "GAMMA", "DELTA", "NET", "PATH", "X123ABC"]
        );
    }

    #[test]
    fn idf_downweights_common_tokens() {
        let corpus = vec![
            "alpha shared",
            "beta shared",
            "gamma shared",
            "alpha unrelated",
        ];
        let (vocab, idf) = build_idf(&corpus);
        // "shared" appears in 3 of 4 records → low IDF.
        // "gamma" appears in 1 → high IDF.
        let shared_idf = idf[*vocab.get("shared").unwrap() as usize];
        let gamma_idf = idf[*vocab.get("gamma").unwrap() as usize];
        assert!(gamma_idf > shared_idf);
    }

    #[test]
    fn cosine_same_entity_high_different_low() {
        // Three records of one entity sharing leading tokens, two records of
        // another entity sharing different leading tokens.
        let corpus = vec![
            "alpha bravo charlie delta echo abc111",
            "alpha bravo charlie delta echo abc222",
            "alpha bravo charlie delta echo abc333",
            "foxtrot golf hotel xyz111",
            "foxtrot golf hotel xyz222",
        ];
        let (vocab, idf) = build_idf(&corpus);
        let v0 = vectorize(corpus[0], &vocab, &idf);
        let v1 = vectorize(corpus[1], &vocab, &idf);
        let v3 = vectorize(corpus[3], &vocab, &idf);

        let same_entity = sparse_cosine(&v0, &v1);
        let cross_entity = sparse_cosine(&v0, &v3);
        assert!(
            same_entity > 0.5,
            "same-entity cosine should be high; got {}",
            same_entity
        );
        assert!(
            cross_entity < 0.1,
            "cross-entity cosine should be near 0; got {}",
            cross_entity
        );
    }
}

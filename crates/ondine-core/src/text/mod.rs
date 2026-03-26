use std::collections::HashMap;

pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty() && w.len() > 1)
        .collect()
}

pub fn term_frequency(tokens: &[String]) -> HashMap<&str, f64> {
    let mut tf: HashMap<&str, f64> = HashMap::new();
    let len = tokens.len() as f64;
    if len == 0.0 {
        return tf;
    }
    for token in tokens {
        *tf.entry(token.as_str()).or_default() += 1.0;
    }
    for count in tf.values_mut() {
        *count /= len;
    }
    tf
}

/// TF-IDF cosine similarity between two texts, treating them as a 2-document corpus.
/// Returns 0.0 for empty/disjoint texts, 1.0 for identical texts.
pub fn tfidf_cosine_similarity(a: &str, b: &str) -> f64 {
    let tokens_a = tokenize(a);
    let tokens_b = tokenize(b);

    if tokens_a.is_empty() && tokens_b.is_empty() {
        return 1.0;
    }
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }

    let tf_a = term_frequency(&tokens_a);
    let tf_b = term_frequency(&tokens_b);

    let num_docs = 2.0_f64;
    let mut vocabulary: HashMap<&str, f64> = HashMap::new();
    for term in tf_a.keys().chain(tf_b.keys()) {
        vocabulary.entry(term).or_insert_with(|| {
            let mut df = 0.0;
            if tf_a.contains_key(term) { df += 1.0; }
            if tf_b.contains_key(term) { df += 1.0; }
            (num_docs / df).ln() + 1.0
        });
    }

    let mut dot_product = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for (term, &idf) in &vocabulary {
        let tfidf_a = tf_a.get(term).unwrap_or(&0.0) * idf;
        let tfidf_b = tf_b.get(term).unwrap_or(&0.0) * idf;
        dot_product += tfidf_a * tfidf_b;
        norm_a += tfidf_a * tfidf_a;
        norm_b += tfidf_b * tfidf_b;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot_product / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_one() {
        let text = "The quick brown fox jumps over the lazy dog";
        let sim = tfidf_cosine_similarity(text, text);
        assert!((sim - 1.0).abs() < 1e-10, "expected 1.0, got {sim}");
    }

    #[test]
    fn disjoint_is_zero() {
        let sim = tfidf_cosine_similarity("alpha beta gamma delta", "epsilon zeta eta theta");
        assert!(sim.abs() < 1e-10, "expected 0.0, got {sim}");
    }

    #[test]
    fn empty_texts_is_one() {
        assert!((tfidf_cosine_similarity("", "") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn one_empty_is_zero() {
        assert!(tfidf_cosine_similarity("hello world", "").abs() < 1e-10);
        assert!(tfidf_cosine_similarity("", "hello world").abs() < 1e-10);
    }

    #[test]
    fn partial_overlap_is_moderate() {
        let a = "The analysis recommends option A due to strong revenue growth and market position";
        let b = "The analysis recommends option A because of revenue growth potential and competitive advantage";
        let sim = tfidf_cosine_similarity(a, b);
        assert!(sim > 0.3 && sim < 0.95, "expected moderate, got {sim}");
    }
}

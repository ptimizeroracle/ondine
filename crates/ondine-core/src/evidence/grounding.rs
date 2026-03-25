use crate::evidence::store::{EvidenceError, EvidenceGraph};
use crate::text::tfidf_cosine_similarity;
use crate::types::{AssertionContext, Claim, ClaimType, SourceType};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const DEFAULT_GROUNDING_THRESHOLD: f64 = 0.3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDocumentClaim {
    pub text: String,
    pub claim_type: String,
    pub source_location: String,
    pub doc_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocSentences {
    pub doc_id: String,
    pub sentences: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedClaim {
    pub claim_id: String,
    pub claim_text: String,
    pub source: String,
    pub confidence: f64,
}

fn parse_claim_type(s: &str) -> ClaimType {
    match s.to_lowercase().as_str() {
        "factual" => ClaimType::Factual,
        "analytical" => ClaimType::Analytical,
        "predictive" => ClaimType::Predictive,
        _ => ClaimType::Conjecture,
    }
}

/// Ground raw document claims against source text via TF-IDF cosine similarity,
/// then store grounded claims in the Evidence Graph.
///
/// Claims with similarity below `threshold` are silently discarded.
pub fn ground_and_store(
    graph: &EvidenceGraph,
    decision_id: Uuid,
    raw_claims: &[RawDocumentClaim],
    doc_sentences: &[DocSentences],
    threshold: f64,
) -> Result<Vec<GroundedClaim>, EvidenceError> {
    if raw_claims.is_empty() {
        return Ok(Vec::new());
    }

    let threshold = if threshold <= 0.0 { DEFAULT_GROUNDING_THRESHOLD } else { threshold };
    let mut grounded = Vec::new();

    for raw in raw_claims {
        let best_sim = find_best_sentence_match(&raw.text, &raw.doc_id, doc_sentences);

        if best_sim < threshold {
            continue;
        }

        let claim_id = Uuid::new_v4();
        let claim = Claim {
            claim_id,
            text: raw.text.clone(),
            claim_type: parse_claim_type(&raw.claim_type),
            source_type: SourceType::LlmResponse,
            source_ref: format!("{}:{}", raw.doc_id, raw.source_location),
            asserted_by: "grounding_engine".to_string(),
            asserted_during: AssertionContext {
                decision_id,
                phase: 0,
                round: None,
            },
            valid_from: None,
            superseded_by: None,
            contradiction_of: Vec::new(),
        };

        graph.assert_claim(&claim)?;

        grounded.push(GroundedClaim {
            claim_id: claim_id.to_string(),
            claim_text: raw.text.clone(),
            source: format!("{}:{}", raw.doc_id, raw.source_location),
            confidence: best_sim,
        });
    }

    Ok(grounded)
}

fn find_best_sentence_match(claim_text: &str, doc_id: &str, doc_sentences: &[DocSentences]) -> f64 {
    let mut best = 0.0_f64;

    for doc in doc_sentences {
        if doc.doc_id != doc_id {
            continue;
        }
        for sentence in &doc.sentences {
            let sim = tfidf_cosine_similarity(claim_text, sentence);
            if sim > best {
                best = sim;
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_raw_claim(text: &str, doc_id: &str, location: &str) -> RawDocumentClaim {
        RawDocumentClaim {
            text: text.to_string(),
            claim_type: "factual".to_string(),
            source_location: location.to_string(),
            doc_id: doc_id.to_string(),
        }
    }

    fn make_doc_sentences(doc_id: &str, sentences: Vec<&str>) -> DocSentences {
        DocSentences {
            doc_id: doc_id.to_string(),
            sentences: sentences.into_iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn grounded_claim_above_threshold_is_stored() {
        let graph = EvidenceGraph::open_in_memory().unwrap();
        let decision_id = Uuid::new_v4();

        let raw = vec![make_raw_claim(
            "Product X is categorized as Organic Cereals",
            "catalogue.pdf",
            "page 3",
        )];
        let docs = vec![make_doc_sentences(
            "catalogue.pdf",
            vec!["Product X is categorized as Organic Cereals according to the product database."],
        )];

        let result = ground_and_store(&graph, decision_id, &raw, &docs, 0.3).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result[0].confidence > 0.3);
        assert_eq!(result[0].source, "catalogue.pdf:page 3");

        let stored = graph.get_claim(result[0].claim_id.parse().unwrap()).unwrap();
        assert_eq!(stored.text, "Product X is categorized as Organic Cereals");
    }

    #[test]
    fn ungrounded_claim_below_threshold_is_discarded() {
        let graph = EvidenceGraph::open_in_memory().unwrap();
        let decision_id = Uuid::new_v4();

        let raw = vec![make_raw_claim(
            "This product will dominate the global market by 2030",
            "catalogue.pdf",
            "page 1",
        )];
        let docs = vec![make_doc_sentences(
            "catalogue.pdf",
            vec!["Weather patterns in the arctic region have been changing rapidly."],
        )];

        let result = ground_and_store(&graph, decision_id, &raw, &docs, 0.3).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn empty_claims_returns_empty() {
        let graph = EvidenceGraph::open_in_memory().unwrap();
        let decision_id = Uuid::new_v4();
        let docs = vec![make_doc_sentences("doc.pdf", vec!["Some sentence."])];

        let result = ground_and_store(&graph, decision_id, &[], &docs, 0.3).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn exact_quote_gets_high_confidence() {
        let graph = EvidenceGraph::open_in_memory().unwrap();
        let decision_id = Uuid::new_v4();
        let exact_text = "Product X belongs to category Organic Premium Cereals";

        let raw = vec![make_raw_claim(exact_text, "catalogue.pdf", "page 7")];
        let docs = vec![make_doc_sentences(
            "catalogue.pdf",
            vec![exact_text],
        )];

        let result = ground_and_store(&graph, decision_id, &raw, &docs, 0.3).unwrap();

        assert_eq!(result.len(), 1);
        assert!(
            result[0].confidence > 0.95,
            "expected near 1.0 for exact quote, got {}",
            result[0].confidence
        );
    }

    #[test]
    fn claim_type_parsing_covers_all_variants() {
        assert!(matches!(parse_claim_type("factual"), ClaimType::Factual));
        assert!(matches!(parse_claim_type("analytical"), ClaimType::Analytical));
        assert!(matches!(parse_claim_type("predictive"), ClaimType::Predictive));
        assert!(matches!(parse_claim_type("Factual"), ClaimType::Factual));
        assert!(matches!(parse_claim_type("unknown"), ClaimType::Conjecture));
    }
}

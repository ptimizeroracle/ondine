use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Claim {
    pub claim_id: Uuid,
    pub text: String,
    pub claim_type: ClaimType,
    pub source_type: SourceType,
    pub source_ref: String,
    pub asserted_by: String,
    pub asserted_during: AssertionContext,
    pub valid_from: Option<DateTime<Utc>>,
    pub superseded_by: Option<Uuid>,
    pub contradiction_of: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClaimType {
    Factual,
    Analytical,
    Predictive,
    Conjecture,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SourceType {
    Document,
    LlmResponse,
    UserCorrection,
    External,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AssertionContext {
    pub decision_id: Uuid,
    pub phase: u8,
    pub round: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Evidence {
    pub claim: Claim,
    pub relevance_score: f64,
    pub support_count: u32,
    pub challenge_count: u32,
    pub net_strength: f64,
}

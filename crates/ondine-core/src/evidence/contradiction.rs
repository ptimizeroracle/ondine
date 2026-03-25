use uuid::Uuid;
use super::store::{EvidenceGraph, EvidenceError};

impl EvidenceGraph {
    pub fn detect_and_store_contradiction(
        &self,
        new_claim_id: Uuid,
        existing_claim_id: Uuid,
    ) -> Result<(), EvidenceError> {
        self.add_contradiction(new_claim_id, existing_claim_id)
    }
}

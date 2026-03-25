use crate::types::{Claim, Evidence};
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum EvidenceError {
    #[error("claim not found: {0}")]
    ClaimNotFound(Uuid),
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("embedding error: {0}")]
    Embedding(String),
}

pub type EmbedCallback = Box<dyn Fn(&[String]) -> Result<Vec<Vec<f32>>, String> + Send>;

pub struct EvidenceGraph {
    conn: rusqlite::Connection,
    embed_callback: Option<EmbedCallback>,
}

const SCHEMA_VERSION: i32 = 1;

impl EvidenceGraph {
    pub fn open_in_memory() -> Result<Self, EvidenceError> {
        let conn = rusqlite::Connection::open_in_memory()?;
        let graph = Self { conn, embed_callback: None };
        graph.initialize_schema()?;
        Ok(graph)
    }

    pub fn open(path: &str) -> Result<Self, EvidenceError> {
        let conn = rusqlite::Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
        let graph = Self { conn, embed_callback: None };
        graph.initialize_schema()?;
        Ok(graph)
    }

    pub fn set_embed_callback(&mut self, callback: EmbedCallback) {
        self.embed_callback = Some(callback);
    }

    pub fn has_embeddings(&self) -> bool {
        self.embed_callback.is_some()
    }

    fn initialize_schema(&self) -> Result<(), EvidenceError> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);")?;
        let has_version: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM schema_version)", [], |row| row.get(0),
        )?;
        if !has_version {
            self.conn.execute("INSERT INTO schema_version (version) VALUES (?1)",
                rusqlite::params![SCHEMA_VERSION])?;
        }

        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                claim_type TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                asserted_by TEXT NOT NULL,
                decision_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                round INTEGER,
                valid_from TEXT,
                superseded_by TEXT,
                data TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS provenance_links (
                claim_id TEXT NOT NULL REFERENCES claims(claim_id),
                asserted_by TEXT NOT NULL,
                decision_id TEXT NOT NULL,
                phase INTEGER NOT NULL,
                PRIMARY KEY (claim_id, asserted_by, decision_id, phase)
            );

            CREATE TABLE IF NOT EXISTS contradictions (
                claim_a TEXT NOT NULL REFERENCES claims(claim_id),
                claim_b TEXT NOT NULL REFERENCES claims(claim_id),
                PRIMARY KEY (claim_a, claim_b)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
                claim_id,
                text,
                content=claims,
                content_rowid=rowid
            );

            CREATE TABLE IF NOT EXISTS claim_embeddings (
                claim_id TEXT PRIMARY KEY REFERENCES claims(claim_id),
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL
            );"
        )?;
        Ok(())
    }

    pub fn assert_claim(
        &self,
        claim: &Claim,
    ) -> Result<Uuid, EvidenceError> {
        let data = serde_json::to_string(claim).unwrap();
        let claim_id_str = claim.claim_id.to_string();

        self.conn.execute(
            "INSERT OR IGNORE INTO claims (claim_id, text, claim_type, source_type, source_ref, asserted_by, decision_id, phase, round, valid_from, superseded_by, data)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            rusqlite::params![
                claim_id_str,
                claim.text,
                serde_json::to_string(&claim.claim_type).unwrap(),
                serde_json::to_string(&claim.source_type).unwrap(),
                claim.source_ref,
                claim.asserted_by,
                claim.asserted_during.decision_id.to_string(),
                claim.asserted_during.phase,
                claim.asserted_during.round,
                claim.valid_from.map(|t| t.to_rfc3339()),
                claim.superseded_by.map(|u| u.to_string()),
                data,
            ],
        )?;

        self.conn.execute(
            "INSERT OR IGNORE INTO provenance_links (claim_id, asserted_by, decision_id, phase) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                claim_id_str,
                claim.asserted_by,
                claim.asserted_during.decision_id.to_string(),
                claim.asserted_during.phase,
            ],
        )?;

        self.rebuild_fts_for_claim(&claim_id_str)?;

        Ok(claim.claim_id)
    }

    fn rebuild_fts_for_claim(&self, claim_id: &str) -> Result<(), EvidenceError> {
        let rowid: i64 = self.conn.query_row(
            "SELECT rowid FROM claims WHERE claim_id = ?1",
            [claim_id],
            |row| row.get(0),
        )?;
        let text: String = self.conn.query_row(
            "SELECT text FROM claims WHERE claim_id = ?1",
            [claim_id],
            |row| row.get(0),
        )?;
        self.conn.execute(
            "INSERT OR REPLACE INTO claims_fts(rowid, claim_id, text) VALUES (?1, ?2, ?3)",
            rusqlite::params![rowid, claim_id, text],
        )?;
        Ok(())
    }

    /// Search for claims relevant to the question.
    /// Uses hybrid search (dense + FTS5 + RRF) when embeddings are available,
    /// falls back to FTS5-only otherwise.
    pub fn query(&self, question: &str, limit: usize) -> Result<Vec<Evidence>, EvidenceError> {
        self.query_hybrid(question, limit)
    }

    pub fn get_claim(&self, claim_id: Uuid) -> Result<Claim, EvidenceError> {
        let data: String = self.conn.query_row(
            "SELECT data FROM claims WHERE claim_id = ?1",
            [claim_id.to_string()],
            |row| row.get(0),
        ).map_err(|_| EvidenceError::ClaimNotFound(claim_id))?;

        Ok(serde_json::from_str(&data).unwrap())
    }

    pub fn add_contradiction(&self, claim_a: Uuid, claim_b: Uuid) -> Result<(), EvidenceError> {
        self.conn.execute(
            "INSERT OR IGNORE INTO contradictions (claim_a, claim_b) VALUES (?1, ?2)",
            rusqlite::params![claim_a.to_string(), claim_b.to_string()],
        )?;
        self.conn.execute(
            "INSERT OR IGNORE INTO contradictions (claim_a, claim_b) VALUES (?1, ?2)",
            rusqlite::params![claim_b.to_string(), claim_a.to_string()],
        )?;
        Ok(())
    }

    /// Embed and store vectors for claims that don't yet have embeddings.
    pub fn embed_pending_claims(&self, model_name: &str) -> Result<usize, EvidenceError> {
        let callback = match &self.embed_callback {
            Some(cb) => cb,
            None => return Ok(0),
        };

        let mut stmt = self.conn.prepare(
            "SELECT c.claim_id, c.text FROM claims c
             LEFT JOIN claim_embeddings e ON c.claim_id = e.claim_id
             WHERE e.claim_id IS NULL"
        )?;

        let pending: Vec<(String, String)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        if pending.is_empty() {
            return Ok(0);
        }

        let texts: Vec<String> = pending.iter().map(|(_, t)| t.clone()).collect();
        let embeddings = callback(&texts).map_err(EvidenceError::Embedding)?;

        if embeddings.len() != pending.len() {
            return Err(EvidenceError::Embedding(format!(
                "expected {} embeddings, got {}", pending.len(), embeddings.len()
            )));
        }

        let dims = embeddings.first().map_or(0, |e| e.len());

        for (i, (claim_id, _)) in pending.iter().enumerate() {
            let blob = f32_vec_to_bytes(&embeddings[i]);
            self.conn.execute(
                "INSERT OR REPLACE INTO claim_embeddings (claim_id, embedding, model, dimensions)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![claim_id, blob, model_name, dims as i32],
            )?;
        }

        Ok(pending.len())
    }

    /// Hybrid query: dense (cosine) + sparse (FTS5) + Reciprocal Rank Fusion.
    /// Falls back to FTS5-only if no embedding callback is configured.
    pub fn query_hybrid(&self, question: &str, limit: usize) -> Result<Vec<Evidence>, EvidenceError> {
        let dense_results = self.try_dense_search(question, limit * 2);

        let sparse_results = self.query_fts(question, limit * 2)?;

        match dense_results {
            Some(dense) if !dense.is_empty() => {
                let fused = rrf_fuse(&dense, &sparse_results, limit);
                self.hydrate_evidence(&fused)
            }
            _ => {
                let ids: Vec<(String, f64)> = sparse_results.into_iter().take(limit).collect();
                self.hydrate_evidence(&ids)
            }
        }
    }

    fn try_dense_search(&self, question: &str, limit: usize) -> Option<Vec<(String, f64)>> {
        let callback = self.embed_callback.as_ref()?;

        let embeddings = callback(&[question.to_string()]).ok()?;
        let query_vec = embeddings.into_iter().next()?;

        let mut stmt = self.conn.prepare(
            "SELECT claim_id, embedding FROM claim_embeddings"
        ).ok()?;

        let mut scored: Vec<(String, f64)> = stmt
            .query_map([], |row| {
                let claim_id: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((claim_id, blob))
            })
            .ok()?
            .filter_map(|r| r.ok())
            .map(|(id, blob)| {
                let stored = bytes_to_f32_vec(&blob);
                let sim = cosine_similarity(&query_vec, &stored);
                (id, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Some(scored)
    }

    fn query_fts(&self, question: &str, limit: usize) -> Result<Vec<(String, f64)>, EvidenceError> {
        let fts_query = question
            .split_whitespace()
            .map(|w| format!("\"{}\"", w.replace('"', "")))
            .collect::<Vec<_>>()
            .join(" OR ");

        if fts_query.is_empty() {
            return Ok(Vec::new());
        }

        let mut stmt = self.conn.prepare(
            "SELECT fts.claim_id, fts.rank
             FROM claims_fts fts
             WHERE claims_fts MATCH ?1
             ORDER BY fts.rank
             LIMIT ?2"
        )?;

        let results = stmt
            .query_map(rusqlite::params![fts_query, limit], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(results.into_iter().map(|(id, rank)| (id, -rank)).collect())
    }

    fn hydrate_evidence(&self, scored_ids: &[(String, f64)]) -> Result<Vec<Evidence>, EvidenceError> {
        let mut results = Vec::with_capacity(scored_ids.len());

        for (claim_id, score) in scored_ids {
            let data: String = match self.conn.query_row(
                "SELECT data FROM claims WHERE claim_id = ?1",
                [claim_id],
                |row| row.get(0),
            ) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let claim: Claim = serde_json::from_str(&data).unwrap();

            let support_count: u32 = self.conn.query_row(
                "SELECT COUNT(*) FROM provenance_links WHERE claim_id = ?1",
                [claim_id.as_str()],
                |row| row.get(0),
            )?;

            results.push(Evidence {
                claim,
                relevance_score: *score,
                support_count,
                challenge_count: 0,
                net_strength: support_count as f64,
            });
        }

        Ok(results)
    }

    pub fn get_contradictions(&self, claim_id: Uuid) -> Result<Vec<Uuid>, EvidenceError> {
        let mut stmt = self.conn.prepare(
            "SELECT claim_b FROM contradictions WHERE claim_a = ?1"
        )?;
        let ids = stmt.query_map([claim_id.to_string()], |row| {
            let s: String = row.get(0)?;
            Ok(s.parse::<Uuid>().unwrap())
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(ids)
    }
}

fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    bytes
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        let bi = b[i] as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Reciprocal Rank Fusion: merges two ranked lists.
/// k=60 is standard. Returns (claim_id, rrf_score) sorted descending.
fn rrf_fuse(
    dense: &[(String, f64)],
    sparse: &[(String, f64)],
    limit: usize,
) -> Vec<(String, f64)> {
    use std::collections::HashMap;
    const K: f64 = 60.0;

    let mut scores: HashMap<&str, f64> = HashMap::new();

    for (rank, (id, _)) in dense.iter().enumerate() {
        *scores.entry(id.as_str()).or_default() += 1.0 / (K + rank as f64 + 1.0);
    }
    for (rank, (id, _)) in sparse.iter().enumerate() {
        *scores.entry(id.as_str()).or_default() += 1.0 / (K + rank as f64 + 1.0);
    }

    let mut fused: Vec<(String, f64)> = scores
        .into_iter()
        .map(|(id, score)| (id.to_string(), score))
        .collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use uuid::Uuid;

    fn make_test_claim(text: &str) -> Claim {
        Claim {
            claim_id: Uuid::new_v4(),
            text: text.to_string(),
            claim_type: ClaimType::Factual,
            source_type: SourceType::Document,
            source_ref: "test_doc.pdf".to_string(),
            asserted_by: "pipeline_run_1".to_string(),
            asserted_during: AssertionContext {
                decision_id: Uuid::new_v4(),
                phase: 0,
                round: None,
            },
            valid_from: None,
            superseded_by: None,
            contradiction_of: Vec::new(),
        }
    }

    #[test]
    fn assert_and_retrieve_claim() {
        let graph = EvidenceGraph::open_in_memory().unwrap();
        let claim = make_test_claim("Product X is categorized as Organic Cereals");
        let id = graph.assert_claim(&claim).unwrap();
        let retrieved = graph.get_claim(id).unwrap();
        assert_eq!(retrieved.text, claim.text);
    }

    #[test]
    fn duplicate_assertion_adds_provenance() {
        let graph = EvidenceGraph::open_in_memory().unwrap();
        let mut claim = make_test_claim("Bio Valley Granola is Organic");
        graph.assert_claim(&claim).unwrap();

        claim.asserted_by = "pipeline_run_2".to_string();
        graph.assert_claim(&claim).unwrap();

        let support: u32 = graph.conn.query_row(
            "SELECT COUNT(*) FROM provenance_links WHERE claim_id = ?1",
            [claim.claim_id.to_string()],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(support, 2);
    }

    #[test]
    fn query_returns_relevant_claims() {
        let graph = EvidenceGraph::open_in_memory().unwrap();
        graph.assert_claim(&make_test_claim("Product X is Organic Cereals")).unwrap();
        graph.assert_claim(&make_test_claim("Product Y is Frozen Vegetables")).unwrap();
        graph.assert_claim(&make_test_claim("Product Z is Organic Snacks")).unwrap();

        let results = graph.query("Organic", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].claim.text.contains("Organic"));
    }

    #[test]
    fn contradiction_detection() {
        let graph = EvidenceGraph::open_in_memory().unwrap();
        let claim_a = make_test_claim("Product X is Cereals");
        let claim_b = make_test_claim("Product X is Snacks");
        graph.assert_claim(&claim_a).unwrap();
        graph.assert_claim(&claim_b).unwrap();

        graph.add_contradiction(claim_a.claim_id, claim_b.claim_id).unwrap();

        let contradictions = graph.get_contradictions(claim_a.claim_id).unwrap();
        assert_eq!(contradictions.len(), 1);
        assert_eq!(contradictions[0], claim_b.claim_id);

        let reverse = graph.get_contradictions(claim_b.claim_id).unwrap();
        assert_eq!(reverse.len(), 1);
        assert_eq!(reverse[0], claim_a.claim_id);
    }

    #[test]
    fn f32_roundtrip_through_bytes() {
        let original = vec![1.0_f32, -0.5, 0.0, 3.14159];
        let bytes = f32_vec_to_bytes(&original);
        let recovered = bytes_to_f32_vec(&bytes);
        assert_eq!(original.len(), recovered.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10, "expected 1.0, got {sim}");
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10, "expected 0.0, got {sim}");
    }

    #[test]
    fn rrf_merges_two_rankings() {
        let dense = vec![
            ("claim_a".to_string(), 0.9),
            ("claim_b".to_string(), 0.7),
            ("claim_c".to_string(), 0.5),
        ];
        let sparse = vec![
            ("claim_b".to_string(), 5.0),
            ("claim_d".to_string(), 3.0),
            ("claim_a".to_string(), 1.0),
        ];
        let fused = rrf_fuse(&dense, &sparse, 10);

        assert!(!fused.is_empty());
        let top_ids: Vec<&str> = fused.iter().take(2).map(|(id, _)| id.as_str()).collect();
        assert!(top_ids.contains(&"claim_a") || top_ids.contains(&"claim_b"));
        assert!(fused.iter().any(|(id, _)| id == "claim_d"));
    }

    #[test]
    fn embed_pending_claims_stores_embeddings() {
        let mut graph = EvidenceGraph::open_in_memory().unwrap();

        let claim = make_test_claim("Product has strong organic certification");
        graph.assert_claim(&claim).unwrap();

        graph.set_embed_callback(Box::new(|texts: &[String]| {
            Ok(texts.iter().map(|_| vec![0.1_f32, 0.2, 0.3, 0.4]).collect())
        }));

        let count = graph.embed_pending_claims("test-model").unwrap();
        assert_eq!(count, 1);

        let count2 = graph.embed_pending_claims("test-model").unwrap();
        assert_eq!(count2, 0);
    }

    #[test]
    fn query_hybrid_uses_dense_when_embeddings_available() {
        let mut graph = EvidenceGraph::open_in_memory().unwrap();

        let claim_organic = make_test_claim("Product X is certified organic and premium");
        let claim_frozen = make_test_claim("Product Y is frozen vegetables budget");
        let claim_bio = make_test_claim("Product Z is bio-certified organic granola");
        graph.assert_claim(&claim_organic).unwrap();
        graph.assert_claim(&claim_frozen).unwrap();
        graph.assert_claim(&claim_bio).unwrap();

        graph.set_embed_callback(Box::new(move |texts: &[String]| {
            Ok(texts.iter().map(|t| {
                if t.contains("organic") || t.contains("bio") {
                    vec![0.9_f32, 0.1, 0.0, 0.0]
                } else if t.contains("frozen") || t.contains("budget") {
                    vec![0.0_f32, 0.1, 0.9, 0.0]
                } else {
                    vec![0.5_f32, 0.5, 0.5, 0.5]
                }
            }).collect())
        }));

        graph.embed_pending_claims("test-model").unwrap();

        let results = graph.query("organic products", 10).unwrap();
        assert!(!results.is_empty());
        let top_text = &results[0].claim.text;
        assert!(
            top_text.contains("organic") || top_text.contains("bio"),
            "expected organic/bio at top, got: {top_text}"
        );
    }

    #[test]
    fn query_falls_back_to_fts_without_embeddings() {
        let graph = EvidenceGraph::open_in_memory().unwrap();
        graph.assert_claim(&make_test_claim("Product X is Organic Cereals")).unwrap();
        graph.assert_claim(&make_test_claim("Product Y is Frozen Vegetables")).unwrap();

        let results = graph.query("Organic", 10).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].claim.text.contains("Organic"));
    }
}

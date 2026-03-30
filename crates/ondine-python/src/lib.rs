use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Mutex;

fn open_evidence(path: &str) -> PyResult<ondine_core::evidence::EvidenceGraph> {
    if path == ":memory:" {
        ondine_core::evidence::EvidenceGraph::open_in_memory()
    } else {
        ondine_core::evidence::EvidenceGraph::open(path)
    }
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Evidence error: {e}")))
}

/// Return the Rust engine version.
#[pyfunction]
fn version() -> &'static str {
    ondine_core::VERSION
}

/// Compute TF-IDF cosine similarity between two texts.
/// Returns a float in [0.0, 1.0].
#[pyfunction]
fn tfidf_similarity(a: &str, b: &str) -> f64 {
    ondine_core::text::tfidf_cosine_similarity(a, b)
}

/// Open or create an evidence database. Returns an opaque handle.
/// Use ":memory:" for an in-memory database (testing).
#[pyclass]
struct EvidenceDB {
    inner: Mutex<ondine_core::evidence::EvidenceGraph>,
}

#[pymethods]
impl EvidenceDB {
    #[new]
    #[pyo3(signature = (path = ":memory:"))]
    fn new(path: &str) -> PyResult<Self> {
        let graph = open_evidence(path)?;
        Ok(Self { inner: Mutex::new(graph) })
    }

    /// Store a claim (JSON-serialized) in the evidence graph.
    /// Returns the claim_id as a string.
    fn store_claim(&self, claim_json: &str) -> PyResult<String> {
        let claim: ondine_core::types::Claim = serde_json::from_str(claim_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid claim JSON: {e}")))?;
        let graph = self.inner.lock().unwrap();
        let id = graph.assert_claim(&claim)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Store error: {e}")))?;
        Ok(id.to_string())
    }

    /// Query the evidence graph for claims relevant to a question.
    /// Returns a JSON array of Evidence objects.
    #[pyo3(signature = (question, limit = 5))]
    fn query(&self, question: &str, limit: usize) -> PyResult<String> {
        let graph = self.inner.lock().unwrap();
        let results = graph.query(question, limit)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Query error: {e}")))?;
        serde_json::to_string(&results)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    /// Get a single claim by ID (UUID string). Returns JSON.
    fn get_claim(&self, claim_id: &str) -> PyResult<String> {
        let uuid: uuid::Uuid = claim_id.parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid UUID: {e}")))?;
        let graph = self.inner.lock().unwrap();
        let claim = graph.get_claim(uuid)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Get error: {e}")))?;
        serde_json::to_string(&claim)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    /// Ground raw claims against source sentences via TF-IDF.
    /// Returns JSON array of GroundedClaim objects.
    #[pyo3(signature = (decision_id, raw_claims_json, doc_sentences_json, threshold = 0.3))]
    fn ground_and_store(
        &self,
        decision_id: &str,
        raw_claims_json: &str,
        doc_sentences_json: &str,
        threshold: f64,
    ) -> PyResult<String> {
        let did: uuid::Uuid = decision_id.parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid UUID: {e}")))?;

        let raw_claims: Vec<ondine_core::evidence::RawDocumentClaim> =
            serde_json::from_str(raw_claims_json)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                    format!("Invalid raw_claims JSON: {e}")))?;

        let doc_sentences: Vec<ondine_core::evidence::DocSentences> =
            serde_json::from_str(doc_sentences_json)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                    format!("Invalid doc_sentences JSON: {e}")))?;

        let graph = self.inner.lock().unwrap();
        let grounded = ondine_core::evidence::grounding::ground_and_store(
            &graph, did, &raw_claims, &doc_sentences, threshold,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Grounding error: {e}")))?;

        serde_json::to_string(&grounded)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    /// Register a contradiction between two claims.
    fn add_contradiction(&self, claim_a_id: &str, claim_b_id: &str) -> PyResult<()> {
        let a: uuid::Uuid = claim_a_id.parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid UUID a: {e}")))?;
        let b: uuid::Uuid = claim_b_id.parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid UUID b: {e}")))?;
        let graph = self.inner.lock().unwrap();
        graph.detect_and_store_contradiction(a, b)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Contradiction error: {e}")))?;
        Ok(())
    }

    /// Get all contradictions for a claim. Returns JSON array of UUID strings.
    fn get_contradictions(&self, claim_id: &str) -> PyResult<String> {
        let uuid: uuid::Uuid = claim_id.parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid UUID: {e}")))?;
        let graph = self.inner.lock().unwrap();
        let ids = graph.get_contradictions(uuid)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Error: {e}")))?;
        let strings: Vec<String> = ids.iter().map(|u| u.to_string()).collect();
        serde_json::to_string(&strings)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    // ── Knowledge Base chunk operations ────────────────────────────────

    /// Store a document chunk in the knowledge base.
    #[pyo3(signature = (chunk_id, text, source, metadata_json = "{}"))]
    fn store_chunk(
        &self,
        chunk_id: &str,
        text: &str,
        source: &str,
        metadata_json: &str,
    ) -> PyResult<()> {
        let graph = self.inner.lock().unwrap();
        graph.store_chunk(chunk_id, text, source, metadata_json)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("store_chunk error: {e}")))
    }

    /// Hybrid search over KB chunks. Returns JSON array of
    /// [chunk_id, text, source, metadata_json, score] tuples.
    #[pyo3(signature = (question, limit = 5))]
    fn query_chunks(&self, question: &str, limit: usize) -> PyResult<String> {
        let graph = self.inner.lock().unwrap();
        let results = graph.query_chunks(question, limit)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("query_chunks error: {e}")))?;
        serde_json::to_string(&results)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Serialization error: {e}")))
    }

    /// Embed pending KB chunks using the same callback mechanism as claims.
    #[pyo3(signature = (callback, model_name = "text-embedding-3-small"))]
    fn embed_pending_chunks(
        &self,
        py: Python<'_>,
        callback: PyObject,
        model_name: &str,
    ) -> PyResult<usize> {
        let mut graph = self.inner.lock().unwrap();

        let cb_ref = callback.clone_ref(py);
        let rust_embed: ondine_core::evidence::store::EmbedCallback =
            Box::new(move |texts: &[String]| {
                Python::with_gil(|py| {
                    let py_texts = PyList::new(
                        py,
                        texts.iter().map(|t| t.as_str()),
                    ).map_err(|e| format!("Failed to create Python list: {e}"))?;

                    let result = cb_ref
                        .call1(py, (py_texts,))
                        .map_err(|e| format!("Python embed callback error: {e}"))?;

                    let outer_list = result
                        .downcast_bound::<PyList>(py)
                        .map_err(|e| format!("Embed callback must return list of lists: {e}"))?;

                    let mut embeddings = Vec::with_capacity(outer_list.len());
                    for item in outer_list.iter() {
                        let inner = item
                            .downcast::<PyList>()
                            .map_err(|e| format!("Inner embedding must be list: {e}"))?;
                        let vec: Vec<f32> = inner
                            .iter()
                            .map(|v| v.extract::<f32>().unwrap_or(0.0))
                            .collect();
                        embeddings.push(vec);
                    }
                    Ok(embeddings)
                })
            });

        graph.set_embed_callback(rust_embed);
        graph.embed_pending_chunks(model_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Embedding error: {e}")))
    }

    /// Number of chunks in the knowledge base.
    fn chunk_count(&self) -> PyResult<usize> {
        let graph = self.inner.lock().unwrap();
        graph.chunk_count()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("chunk_count error: {e}")))
    }

    /// Embed pending claims using a Python callback.
    /// callback receives a list of strings and returns a list of list[float].
    #[pyo3(signature = (callback, model_name = "text-embedding-3-small"))]
    fn embed_pending(
        &self,
        py: Python<'_>,
        callback: PyObject,
        model_name: &str,
    ) -> PyResult<usize> {
        let mut graph = self.inner.lock().unwrap();

        let cb_ref = callback.clone_ref(py);
        let rust_embed: ondine_core::evidence::store::EmbedCallback =
            Box::new(move |texts: &[String]| {
                Python::with_gil(|py| {
                    let py_texts = PyList::new(
                        py,
                        texts.iter().map(|t| t.as_str()),
                    ).map_err(|e| format!("Failed to create Python list: {e}"))?;

                    let result = cb_ref
                        .call1(py, (py_texts,))
                        .map_err(|e| format!("Python embed callback error: {e}"))?;

                    let outer_list = result
                        .downcast_bound::<PyList>(py)
                        .map_err(|e| format!("Embed callback must return list of lists: {e}"))?;

                    let mut embeddings = Vec::with_capacity(outer_list.len());
                    for item in outer_list.iter() {
                        let inner = item
                            .downcast::<PyList>()
                            .map_err(|e| format!("Inner embedding must be list: {e}"))?;
                        let vec: Vec<f32> = inner
                            .iter()
                            .map(|v| v.extract::<f32>().unwrap_or(0.0))
                            .collect();
                        embeddings.push(vec);
                    }
                    Ok(embeddings)
                })
            });

        graph.set_embed_callback(rust_embed);
        graph.embed_pending_claims(model_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Embedding error: {e}")))
    }
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(tfidf_similarity, m)?)?;
    m.add_class::<EvidenceDB>()?;
    Ok(())
}

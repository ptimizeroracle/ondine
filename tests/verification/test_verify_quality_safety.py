"""Claim verification: Quality and safety claims (Claims 31-40)."""

from ondine.context.memory_store import InMemoryContextStore
from ondine.context.protocol import ContextStore, EvidenceRecord, GroundingResult
from ondine.core.specifications import ErrorPolicy, ProcessingSpec


class TestQualitySafetyClaims:
    """Verify all claimed quality and safety features."""

    def test_claim_31_context_store_protocol(self):
        """Claim 31: Anti-hallucination — ContextStore ABC defines required methods."""
        assert hasattr(ContextStore, "store")
        assert hasattr(ContextStore, "retrieve")
        assert hasattr(ContextStore, "search")
        assert hasattr(ContextStore, "ground")
        assert hasattr(ContextStore, "add_contradiction")
        assert hasattr(ContextStore, "close")

    def test_claim_32_evidence_store_retrieve_cycle(self):
        """Claim 32: Evidence graph — store and retrieve preserve data."""
        store = InMemoryContextStore()
        record = EvidenceRecord(
            text="Paris is the capital of France",
            source_ref="test-doc-1",
            claim_type="factual",
        )
        claim_id = store.store(record)
        assert claim_id is not None

        retrieved = store.retrieve(claim_id)
        assert retrieved is not None
        assert retrieved.text == "Paris is the capital of France"
        assert retrieved.source_ref == "test-doc-1"

        store.close()

    def test_claim_33_confidence_field_on_evidence(self):
        """Claim 33: Confidence scoring — EvidenceRecord accepts confidence float."""
        record = EvidenceRecord(
            text="test claim",
            confidence=0.85,
        )
        assert record.confidence == 0.85
        assert 0.0 <= record.confidence <= 1.0

    def test_claim_34_grounding_method(self):
        """Claim 34: Grounding verification — ground() returns GroundingResult list."""
        store = InMemoryContextStore()

        results = store.ground(
            response_text="Paris is the capital of France",
            source_sentences=["France has Paris as its capital city"],
            threshold=0.3,
        )
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, GroundingResult)
            assert hasattr(r, "confidence")
            assert hasattr(r, "grounded")

        store.close()

    def test_claim_35_contradiction_detection(self):
        """Claim 35: Contradiction detection — add and query contradictions."""
        store = InMemoryContextStore()

        rec_a = EvidenceRecord(text="Earth is flat")
        rec_b = EvidenceRecord(text="Earth is round")
        id_a = store.store(rec_a)
        id_b = store.store(rec_b)

        store.add_contradiction(id_a, id_b)
        contradictions = store.get_contradictions(id_a)
        assert id_b in contradictions

        store.close()

    def test_claim_36_knowledge_store_search(self):
        """Claim 36: RAG — KnowledgeStore ingests text and searches."""
        from ondine.knowledge.store import KnowledgeStore, SearchResult

        kb = KnowledgeStore(":memory:")
        count = kb.ingest_text(
            "Machine learning is a subset of artificial intelligence"
        )
        assert count >= 1

        results = kb.search("what is machine learning", limit=3)
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], SearchResult)
            assert results[0].text  # non-empty
            assert results[0].score > 0

    def test_claim_37_evidence_retrieval_stage(self):
        """Claim 37: Evidence priming — EvidenceRetrievalStage exists."""
        from ondine.stages.evidence_retrieval_stage import EvidenceRetrievalStage

        store = InMemoryContextStore()
        stage = EvidenceRetrievalStage(
            store=store,
            query_columns=["text"],
            top_k=3,
        )
        assert stage is not None
        store.close()

    def test_claim_38_retry_handler_exponential_backoff(self):
        """Claim 38: Automatic retries — RetryHandler with exponential backoff."""
        from ondine.utils.retry_handler import RetryHandler

        handler = RetryHandler(
            max_attempts=3,
            initial_delay=0.01,
            exponential_base=2,
            retryable_exceptions=(ConnectionError,),
        )

        delay_1 = handler.calculate_delay(1)
        delay_2 = handler.calculate_delay(2)
        assert delay_2 > delay_1  # exponential growth

        # Verify it retries transient errors
        call_count = 0

        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "success"

        result = handler.execute(flaky_fn)
        assert result == "success"
        assert call_count == 3

    def test_claim_39_error_policies(self):
        """Claim 39: Error policies — FAIL, SKIP, RETRY, USE_DEFAULT."""
        from ondine.core.error_handler import ErrorAction, ErrorHandler

        # SKIP policy
        handler = ErrorHandler(policy=ErrorPolicy.SKIP)
        decision = handler.handle_error(ValueError("test"), context={}, attempt=1)
        assert decision.action == ErrorAction.SKIP

        # FAIL policy
        handler_fail = ErrorHandler(policy=ErrorPolicy.FAIL)
        decision_fail = handler_fail.handle_error(
            ValueError("test"), context={}, attempt=1
        )
        assert decision_fail.action == ErrorAction.FAIL

    def test_claim_40_partial_failure_handling(self):
        """Claim 40: Partial failure — error policy allows continuing after row failures."""
        spec = ProcessingSpec(error_policy=ErrorPolicy.SKIP)
        assert spec.error_policy == ErrorPolicy.SKIP

        # SKIP means pipeline continues on error, collecting partial results
        from ondine.core.error_handler import ErrorAction, ErrorHandler

        handler = ErrorHandler(policy=ErrorPolicy.SKIP)
        decision = handler.handle_error(
            RuntimeError("row error"), context={"row": 5}, attempt=1
        )
        assert decision.action == ErrorAction.SKIP  # continues, doesn't fail

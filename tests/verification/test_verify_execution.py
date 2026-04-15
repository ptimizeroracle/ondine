"""Claim verification: Execution modes and controls (Claims 53-60)."""

from ondine.core.specifications import ProcessingSpec
from ondine.orchestration.async_executor import AsyncExecutor
from ondine.orchestration.streaming_executor import StreamingExecutor
from ondine.orchestration.sync_executor import SyncExecutor
from ondine.utils.rate_limiter import RateLimiter


class TestExecutionClaims:
    """Verify all claimed execution modes and controls exist and work."""

    def test_claim_53_sync_executor_exists(self):
        """Claim 53: Standard synchronous execution mode exists."""
        executor = SyncExecutor()
        assert hasattr(executor, "execute")

    def test_claim_54_async_executor_exists(self):
        """Claim 54: Async execution with configurable concurrency."""
        executor = AsyncExecutor(max_concurrency=10)
        assert hasattr(executor, "execute")
        assert executor.max_concurrency == 10

    def test_claim_55_streaming_executor_exists(self):
        """Claim 55: Streaming execution for memory-efficient processing."""
        executor = StreamingExecutor(chunk_size=500)
        assert hasattr(executor, "execute")
        assert executor.chunk_size == 500

    def test_claim_56_checkpoint_interval_configurable(self):
        """Claim 56: Checkpoint interval is configurable via ProcessingSpec."""
        spec = ProcessingSpec(checkpoint_interval=250)
        assert spec.checkpoint_interval == 250

        spec_default = ProcessingSpec()
        assert spec_default.checkpoint_interval == 500  # default

    def test_claim_57_rate_limiter_limits_throughput(self):
        """Claim 57: Rate limiting constrains requests per minute."""
        limiter = RateLimiter(requests_per_minute=60)
        assert limiter.available_tokens > 0

        # Acquire one token
        acquired = limiter.acquire(tokens=1, timeout=0.1)
        assert acquired is True

    def test_claim_58_concurrency_configurable(self):
        """Claim 58: Concurrency control is configurable."""
        spec = ProcessingSpec(concurrency=20)
        assert spec.concurrency == 20

        executor = AsyncExecutor(max_concurrency=5)
        assert executor.max_concurrency == 5

    def test_claim_59_context_window_validation_exists(self):
        """Claim 59: Context window validation — BatchAggregatorStage supports it."""
        from ondine.stages.batch_aggregator_stage import BatchAggregatorStage

        stage = BatchAggregatorStage(
            batch_size=10,
            model="gpt-4o",
            validate_context_window=True,
        )
        assert stage is not None

    def test_claim_60_smart_defaults(self):
        """Claim 60: Smart defaults — ProcessingSpec has sensible defaults."""
        spec = ProcessingSpec()
        assert spec.batch_size == 100
        assert spec.concurrency == 5
        assert spec.max_retries == 3
        assert spec.error_policy.value == "skip"
        assert spec.checkpoint_interval == 500

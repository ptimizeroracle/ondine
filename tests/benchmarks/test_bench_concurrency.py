"""Concurrency and rate limiter benchmarks for Ondine pipeline.

Measures the overhead of concurrency control mechanisms to establish
baselines before optimization. Key focus: blocking RateLimiter cost
in async context, and semaphore vs ConcurrencyController overhead.

Run with:
    pytest tests/benchmarks/test_bench_concurrency.py --benchmark-only
"""

import asyncio
import time

import pytest

from ondine.orchestration.concurrency_controller import ConcurrencyController
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.llm_invocation_stage import LLMInvocationStage
from ondine.utils.rate_limiter import RateLimiter

from .conftest import BenchmarkMockLLMClient, make_prompt_batches

# ---------------------------------------------------------------------------
# RateLimiter benchmarks
# ---------------------------------------------------------------------------


class TestBenchRateLimiter:
    """Benchmark RateLimiter.acquire() to measure blocking overhead.

    Regression this catches: rate limiter becoming slower or the
    async wrapper (run_in_executor) adding unexpected overhead.
    """

    def test_bench_rate_limiter_acquire_burst(self, benchmark, rate_limiter_600rpm):
        """Measure acquire throughput when tokens are available (no waiting).

        This is the fast path — tokens available, just decrement and return.
        Should be sub-microsecond per acquire.
        """
        # Pre-fill tokens
        rate_limiter_600rpm.reset()

        def acquire_burst():
            """Acquire 10 tokens in a burst (no waiting needed)."""
            limiter = RateLimiter(requests_per_minute=6000, burst_size=100)
            for _ in range(100):
                limiter.acquire()

        benchmark(acquire_burst)

    def test_bench_rate_limiter_blocking_cost_in_async(self):
        """Measure the real cost of running blocking acquire in async context.

        This quantifies the overhead of:
        1. run_in_executor thread spawn
        2. time.sleep(0.1) polling loop
        3. Thread pool contention under concurrent load

        Not a pytest-benchmark test — uses manual timing for async control.
        """
        limiter = RateLimiter(requests_per_minute=120, burst_size=10)
        n_acquires = 10

        async def _run_blocking():
            """Simulate LLMInvocationStage's current pattern."""
            loop = asyncio.get_running_loop()
            tasks = []
            for _ in range(n_acquires):
                tasks.append(loop.run_in_executor(None, limiter.acquire))
            await asyncio.gather(*tasks)

        t0 = time.perf_counter()
        asyncio.run(_run_blocking())
        elapsed = time.perf_counter() - t0

        # At 120 RPM = 2/sec, 10 acquires should take ~5s if fully rate-limited.
        # But burst_size=10 means first 10 go through immediately.
        # The key metric is overhead vs theoretical minimum.
        # With no overhead, 10 burst acquires should be <10ms.
        assert elapsed < 2.0, (
            f"Blocking rate limiter took {elapsed:.3f}s for {n_acquires} burst acquires. "
            f"Expected <2s for burst capacity={limiter.capacity}."
        )

    def test_bench_rate_limiter_contention(self):
        """Measure rate limiter under concurrent async contention.

        Simulates 20 coroutines competing for a 60 RPM limiter via
        run_in_executor — the exact pattern in LLMInvocationStage.
        """
        limiter = RateLimiter(requests_per_minute=600, burst_size=20)
        n_tasks = 20

        async def _run_contention():
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(None, limiter.acquire) for _ in range(n_tasks)
            ]
            await asyncio.gather(*tasks)

        t0 = time.perf_counter()
        asyncio.run(_run_contention())
        elapsed = time.perf_counter() - t0

        # 20 burst acquires with burst_size=20 should be near-instant
        # Thread pool overhead should be <500ms
        assert elapsed < 1.0, (
            f"Rate limiter contention: {elapsed:.3f}s for {n_tasks} concurrent acquires"
        )


# ---------------------------------------------------------------------------
# ConcurrencyController vs raw Semaphore benchmarks
# ---------------------------------------------------------------------------


class TestBenchConcurrencyController:
    """Benchmark ConcurrencyController vs raw asyncio.Semaphore.

    Regression this catches: ConcurrencyController adding measurable
    overhead vs the raw semaphore pattern currently used in LLMInvocationStage.
    """

    def test_bench_raw_semaphore_throughput(self, benchmark):
        """Baseline: raw asyncio.Semaphore acquire/release cycle."""

        def _run():
            async def _semaphore_cycle():
                sem = asyncio.Semaphore(10)
                for _ in range(1000):
                    await sem.acquire()
                    sem.release()

            asyncio.run(_semaphore_cycle())

        benchmark(_run)

    def test_bench_controller_throttle_throughput(self, benchmark):
        """ConcurrencyController.throttle() cycle — same work as semaphore."""

        def _run():
            async def _controller_cycle():
                ctrl = ConcurrencyController(max_concurrent=10)
                for _ in range(1000):
                    async with ctrl.throttle():
                        pass

            asyncio.run(_controller_cycle())

        benchmark(_run)


# ---------------------------------------------------------------------------
# LLMInvocationStage end-to-end concurrency benchmarks
# ---------------------------------------------------------------------------


class TestBenchLLMInvocationConcurrency:
    """Benchmark LLMInvocationStage at different concurrency levels.

    Regression this catches: concurrency scaling not improving throughput
    linearly (semaphore starvation, rate limiter blocking, task overhead).
    """

    @pytest.mark.parametrize("concurrency", [1, 5, 10])
    def test_bench_llm_stage_mock_latency(self, concurrency):
        """Measure wall-clock time for LLM stage with simulated API latency.

        With 50ms mock latency and N concurrent tasks on 100 items:
        - concurrency=1: ~5000ms (sequential)
        - concurrency=5: ~1000ms (5x speedup)
        - concurrency=10: ~500ms (10x speedup)

        Not a pytest-benchmark test — uses manual timing for async precision.
        """
        client = BenchmarkMockLLMClient(delay_ms=50.0)
        batches = make_prompt_batches(100, batch_size=100)
        context = ExecutionContext(total_rows=100)
        stage = LLMInvocationStage(
            llm_client=client,
            concurrency=concurrency,
        )

        t0 = time.perf_counter()
        result = stage.process(batches, context)
        elapsed = time.perf_counter() - t0

        # Verify correctness
        total_responses = sum(len(b.responses) for b in result)
        assert total_responses == 100

        # Verify concurrency actually helps
        # With 50ms delay and 100 items:
        # Theoretical minimum = (100 / concurrency) * 0.05
        theoretical_min = (100 / concurrency) * 0.05
        # Allow 3x overhead for task scheduling, event loop, etc.
        max_allowed = theoretical_min * 3 + 0.5  # +0.5s for startup overhead

        assert elapsed < max_allowed, (
            f"concurrency={concurrency}: {elapsed:.2f}s elapsed, "
            f"theoretical_min={theoretical_min:.2f}s, max_allowed={max_allowed:.2f}s"
        )

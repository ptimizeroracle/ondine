"""Unit tests for ResponseCache."""

import tempfile
from decimal import Decimal

from ondine.core.models import LLMResponse
from ondine.utils.response_cache import ResponseCache, get_memory_cache


class TestResponseCache:
    """Test suite for ResponseCache."""

    def test_in_memory_cache(self):
        """Test in-memory cache basic operations."""
        cache = ResponseCache(":memory:")

        # Create a test response
        response = LLMResponse(
            text="Test response",
            tokens_in=10,
            tokens_out=5,
            model="gpt-4o-mini",
            cost=Decimal("0.001"),
            latency_ms=100.0,
        )

        # Test cache miss
        cache_key = "test_key_123"
        assert cache.get(cache_key) is None

        # Store in cache
        cache.set(cache_key, response)

        # Test cache hit
        cached = cache.get(cache_key)
        assert cached is not None
        assert cached.text == response.text
        assert cached.tokens_in == response.tokens_in
        assert cached.tokens_out == response.tokens_out
        assert cached.model == response.model
        assert cached.cost == str(response.cost)  # JSON stores as string
        assert cached.latency_ms == response.latency_ms

        cache.close()

    def test_persistent_cache(self):
        """Test persistent cache survives restart."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Create cache and store response
        cache1 = ResponseCache(db_path)
        response = LLMResponse(
            text="Persistent response",
            tokens_in=20,
            tokens_out=10,
            model="claude-3-5-sonnet",
            cost=Decimal("0.002"),
            latency_ms=200.0,
        )
        cache_key = "persistent_key"
        cache1.set(cache_key, response)
        cache1.close()

        # Create new cache instance with same DB
        cache2 = ResponseCache(db_path)
        cached = cache2.get(cache_key)

        assert cached is not None
        assert cached.text == response.text
        assert cached.model == response.model

        cache2.close()

    def test_cache_key_generation(self):
        """Test deterministic cache key generation."""
        # Same inputs should produce same key
        key1 = ResponseCache.generate_cache_key(
            prompt="Test prompt",
            system_message="You are a helper",
            model="gpt-4o-mini",
            temperature=0.7,
        )
        key2 = ResponseCache.generate_cache_key(
            prompt="Test prompt",
            system_message="You are a helper",
            model="gpt-4o-mini",
            temperature=0.7,
        )
        assert key1 == key2

        # Different inputs should produce different keys
        key3 = ResponseCache.generate_cache_key(
            prompt="Test prompt",
            system_message="You are a helper",
            model="gpt-4o-mini",
            temperature=0.8,  # Different temperature
        )
        assert key1 != key3

        # Different prompt should produce different key
        key4 = ResponseCache.generate_cache_key(
            prompt="Different prompt",
            system_message="You are a helper",
            model="gpt-4o-mini",
            temperature=0.7,
        )
        assert key1 != key4

    def test_cache_key_includes_all_params(self):
        """Test cache key includes all relevant parameters."""
        base_key = ResponseCache.generate_cache_key(
            prompt="Test",
            system_message="System",
            model="gpt-4",
            temperature=0.7,
        )

        # Different max_tokens should produce different key
        key_with_max_tokens = ResponseCache.generate_cache_key(
            prompt="Test",
            system_message="System",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
        )
        assert base_key != key_with_max_tokens

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ResponseCache(":memory:")

        stats = cache.get_stats()
        assert stats["cache_size"] == 0
        assert stats["db_path"] == "in_memory"

        # Add some responses
        for i in range(5):
            response = LLMResponse(
                text=f"Response {i}",
                tokens_in=10,
                tokens_out=5,
                model="gpt-4o-mini",
                cost=Decimal("0.001"),
                latency_ms=100.0,
            )
            cache.set(f"key_{i}", response)

        stats = cache.get_stats()
        assert stats["cache_size"] == 5

        cache.close()

    def test_clear_cache(self):
        """Test clearing cache."""
        cache = ResponseCache(":memory:")

        # Add response
        response = LLMResponse(
            text="To be cleared",
            tokens_in=10,
            tokens_out=5,
            model="gpt-4o-mini",
            cost=Decimal("0.001"),
            latency_ms=100.0,
        )
        cache_key = "clear_test"
        cache.set(cache_key, response)

        # Verify it's there
        assert cache.get(cache_key) is not None

        # Clear cache
        cache.clear()

        # Verify it's gone
        assert cache.get(cache_key) is None
        assert cache.get_stats()["cache_size"] == 0

        cache.close()

    def test_get_memory_cache(self):
        """Test global memory cache instance."""
        cache1 = get_memory_cache()
        cache2 = get_memory_cache()

        # Should be same instance
        assert cache1 is cache2

        # Should work
        response = LLMResponse(
            text="Global cache test",
            tokens_in=10,
            tokens_out=5,
            model="gpt-4o-mini",
            cost=Decimal("0.001"),
            latency_ms=100.0,
        )
        cache1.set("global_key", response)

        cached = cache2.get("global_key")
        assert cached is not None
        assert cached.text == response.text

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        import threading

        cache = ResponseCache(":memory:")

        results = []

        def worker(worker_id):
            for i in range(10):
                cache_key = f"key_{worker_id}_{i}"
                response = LLMResponse(
                    text=f"Response {worker_id}_{i}",
                    tokens_in=10,
                    tokens_out=5,
                    model="gpt-4o-mini",
                    cost=Decimal("0.001"),
                    latency_ms=100.0,
                )
                cache.set(cache_key, response)
                cached = cache.get(cache_key)
                results.append(cached.text if cached else None)

        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All operations should succeed
        assert len(results) == 50
        assert all(r is not None for r in results)
        assert cache.get_stats()["cache_size"] == 50

        cache.close()

    def test_response_serialization(self):
        """Test that response serialization preserves all fields."""
        cache = ResponseCache(":memory:")

        # Create response with metadata
        original = LLMResponse(
            text="Test with metadata",
            tokens_in=15,
            tokens_out=8,
            model="gpt-4o-mini",
            cost=Decimal("0.0015"),
            latency_ms=150.0,
            metadata={"key": "value", "number": 42},
        )

        cache_key = "metadata_test"
        cache.set(cache_key, original)
        cached = cache.get(cache_key)

        assert cached.text == original.text
        assert cached.tokens_in == original.tokens_in
        assert cached.tokens_out == original.tokens_out
        assert cached.model == original.model
        assert cached.cost == str(original.cost)  # JSON stores as string
        assert cached.latency_ms == original.latency_ms
        assert cached.metadata == original.metadata

        cache.close()

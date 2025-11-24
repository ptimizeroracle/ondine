"""
Response caching for LLM calls using Redis.

Caches LLM responses to avoid duplicate API calls and reduce costs.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RedisResponseCache:
    """
    Redis-based LLM response cache.

    Caches responses based on:
    - Model identifier
    - Prompt content
    - Temperature
    - System message

    Usage:
        cache = RedisResponseCache("redis://localhost:6379", ttl=3600)

        # Check cache
        cached = cache.get(cache_key)
        if cached:
            return cached

        # Make API call
        response = await llm.invoke(prompt)

        # Cache response
        cache.set(cache_key, response.model_dump())
    """

    def __init__(self, redis_url: str, ttl: int = 3600):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379")
            ttl: Time-to-live for cached responses in seconds (default: 1 hour)
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self.redis_client = None

        try:
            import redis

            self.redis_client = redis.from_url(redis_url)
            logger.info(f"Connected to Redis cache at {redis_url}")
        except ImportError:
            logger.warning("redis package not installed, caching disabled")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}, caching disabled")

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """
        Get cached response.

        Args:
            cache_key: Cache key (MD5 hash of prompt+params)

        Returns:
            Cached response dict or None if not found
        """
        if not self.redis_client:
            return None

        try:
            import json

            cached_json = self.redis_client.get(cache_key)
            if cached_json:
                return json.loads(cached_json)
            return None
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def set(self, cache_key: str, response: dict[str, Any]) -> None:
        """
        Cache response with TTL.

        Args:
            cache_key: Cache key
            response: Response dict to cache
        """
        if not self.redis_client:
            return

        try:
            import json

            response_json = json.dumps(response)
            self.redis_client.setex(cache_key, self.ttl, response_json)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

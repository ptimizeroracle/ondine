"""
Response cache for deduplicating LLM API calls.

Provides persistent storage of LLM responses to avoid redundant API calls
for identical inputs. Thread-safe implementation using SQLite.
"""

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import asdict
from typing import Any

from ondine.core.models import LLMResponse


class ResponseCache:
    """
    Thread-safe cache for LLM responses with SQLite persistence.

    Uses SHA256 hashing of prompt + metadata to create deterministic cache keys.
    Stores responses as JSON for easy serialization/deserialization.
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize response cache.

        Args:
            db_path: Path to SQLite database. Use ":memory:" for in-memory cache
                    or a file path for persistent cache.
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_created_at ON responses(created_at)"
            )
            conn.commit()
            self._conn = conn

    def get(self, cache_key: str) -> LLMResponse | None:
        """
        Retrieve cached response if exists.

        Args:
            cache_key: SHA256 hash of the request content

        Returns:
            LLMResponse if found in cache, None otherwise
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT response FROM responses WHERE cache_key = ?", (cache_key,)
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return LLMResponse(**data)
        return None

    def set(self, cache_key: str, response: LLMResponse) -> None:
        """
        Store response in cache.

        Args:
            cache_key: SHA256 hash of the request content
            response: LLMResponse to cache
        """
        with self._lock:
            # Convert Decimal to string for JSON serialization
            response_dict = asdict(response)
            response_dict["cost"] = str(response_dict["cost"])
            response_json = json.dumps(response_dict)
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (cache_key, response, created_at) VALUES (?, ?, ?)",
                (cache_key, response_json, time.time()),
            )
            self._conn.commit()

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and hit rate (if tracked externally)
        """
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM responses")
            count = cursor.fetchone()[0]
        return {
            "cache_size": count,
            "db_path": self.db_path if self.db_path != ":memory:" else "in_memory",
        }

    def clear(self) -> None:
        """Clear all cached responses."""
        with self._lock:
            self._conn.execute("DELETE FROM responses")
            self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            self._conn.close()

    @staticmethod
    def generate_cache_key(
        prompt: str,
        system_message: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs,
    ) -> str:
        """
        Generate deterministic cache key from request components.

        Includes all components that affect LLM response:
        - prompt (required)
        - system_message (optional)
        - model (optional)
        - temperature (optional)
        - Any additional kwargs (max_tokens, etc.)

        Args:
            prompt: The formatted prompt string
            system_message: System prompt for caching
            model: Model identifier
            temperature: Generation temperature
            **kwargs: Additional generation parameters

        Returns:
            SHA256 hex digest of the canonical request representation
        """
        # Create canonical representation
        cache_content = {
            "prompt": prompt,
            "system_message": system_message or "",
            "model": model or "",
            "temperature": temperature or 0.0,
        }
        # Add any additional generation parameters
        cache_content.update(kwargs)

        # Create deterministic JSON representation
        canonical_json = json.dumps(
            cache_content, sort_keys=True, separators=(",", ":")
        )

        # Generate SHA256 hash
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# Global in-memory cache instance for convenience
_memory_cache: ResponseCache | None = None
_cache_lock = threading.Lock()


def get_memory_cache() -> ResponseCache:
    """
    Get global in-memory cache instance.

    Returns:
        Shared ResponseCache instance using in-memory SQLite
    """
    global _memory_cache
    with _cache_lock:
        if _memory_cache is None:
            _memory_cache = ResponseCache(":memory:")
    return _memory_cache

"""Parse rate-limit retry hints out of HTTP response headers.

Different LLM providers expose "when can I try again" in different
header shapes:

* ``retry-after`` — RFC 7231 integer delta-seconds or HTTP-date
* ``retry-after-ms`` — OpenAI/Azure millisecond-precision variant
* ``x-ratelimit-reset-requests`` / ``-tokens`` — Groq/OpenAI human
  durations like ``6m0s``, ``7.66s``
* ``anthropic-ratelimit-*-reset`` — ISO-8601 absolute timestamps

This module centralises the parsing so that callers (error mapping in
the LiteLLM client, retry handler) speak a single shape: an optional
float ``retry_after_seconds``. ``None`` means "no hint, fall back to
exponential backoff"; ``0.0`` means "the server said retry now".
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

# Header names the parser recognises, in descending order of precision.
# Each entry is (header_name_lowercase, kind).
_HEADERS: list[tuple[str, str]] = [
    ("retry-after-ms", "delta_ms"),
    ("retry-after", "delta_or_date"),
    ("anthropic-ratelimit-requests-reset", "iso_absolute"),
    ("anthropic-ratelimit-tokens-reset", "iso_absolute"),
    ("anthropic-ratelimit-input-tokens-reset", "iso_absolute"),
    ("anthropic-ratelimit-output-tokens-reset", "iso_absolute"),
    ("x-ratelimit-reset-requests", "human_duration"),
    ("x-ratelimit-reset-tokens", "human_duration"),
]


_DURATION_RE = re.compile(r"(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?$")


class RetryAfterParser:
    """Extract a retry-after delay (seconds) from a response-header map.

    The parser is stateless except for two injectable clocks — one
    monotonic (unused today but reserved for future caching) and one
    wall-clock (needed to turn absolute HTTP-dates/ISO timestamps into
    deltas). Dependency injection keeps tests deterministic.

    Args:
        monotonic: Callable returning a monotonic timestamp in seconds.
            Reserved; present for API stability.
        utcnow: Callable returning the current UTC ``datetime``. Used
            to compute deltas from absolute timestamps.
        max_delay_s: Hard cap. Providers have been observed sending
            ``retry-after: 3600`` during outages; without a cap a
            single 429 stalls the pipeline. Defaults to 300 seconds.
    """

    def __init__(
        self,
        monotonic: Callable[[], float] | None = None,
        utcnow: Callable[[], datetime] | None = None,
        max_delay_s: float = 300.0,
    ) -> None:
        import time as _time

        self._monotonic = monotonic or _time.monotonic
        self._utcnow = utcnow or (lambda: datetime.now(tz=timezone.utc))
        self._max_delay = max_delay_s

    def parse(self, headers: Mapping[str, str]) -> float | None:
        """Return a retry-after delay in seconds, or ``None`` if no
        recognised header was present (or all were unparseable).

        ``0.0`` is a distinct, meaningful answer: the server said
        "retry now" and the local token bucket should be updated
        accordingly — not silently dropped.
        """
        # Normalise to lowercase once so lookups are case-insensitive.
        lower = {k.lower(): v for k, v in headers.items()}
        for name, kind in _HEADERS:
            if name not in lower:
                continue
            value = lower[name].strip()
            parsed = self._interpret(value, kind)
            if parsed is None:
                continue
            return self._clamp(parsed)
        return None

    # ── internal interpretation per header shape ───────────────────

    def _interpret(self, value: str, kind: str) -> float | None:
        if kind == "delta_ms":
            ms = self._try_float(value)
            return None if ms is None else ms / 1000.0
        if kind == "delta_or_date":
            delta = self._try_float(value)
            if delta is not None:
                return delta
            return self._parse_http_date(value)
        if kind == "iso_absolute":
            return self._parse_iso_absolute(value)
        if kind == "human_duration":
            return self._parse_human_duration(value)
        return None

    def _parse_http_date(self, value: str) -> float | None:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        return self._delta_from_now(when)

    def _parse_iso_absolute(self, value: str) -> float | None:
        # Python 3.10's datetime.fromisoformat rejects the "Z" UTC
        # suffix used by Anthropic's anthropic-ratelimit-*-reset
        # headers; 3.11+ accepts it. Normalise explicitly so the
        # parser behaves uniformly across supported Pythons
        # (3.10-3.13).
        candidate = value
        if candidate.endswith(("Z", "z")):
            candidate = candidate[:-1] + "+00:00"
        try:
            when = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return self._delta_from_now(when)

    def _parse_human_duration(self, value: str) -> float | None:
        # Patterns like "6m0s", "2s", "7.66s", "1.5m"
        match = _DURATION_RE.fullmatch(value)
        if not match or match.group("m") is None and match.group("s") is None:
            return None
        minutes = float(match.group("m") or 0.0)
        seconds = float(match.group("s") or 0.0)
        return minutes * 60.0 + seconds

    def _delta_from_now(self, when: datetime) -> float:
        now = self._utcnow()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return max(0.0, (when - now).total_seconds())

    def _clamp(self, delay: float) -> float | None:
        if delay < 0:
            return None
        if delay > self._max_delay:
            return self._max_delay
        return delay

    @staticmethod
    def _try_float(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

"""Tests for RetryAfterParser.

Each test articulates a specific regression: when a provider starts
sending an unexpected header shape, or stops sending headers
altogether, the parser must degrade predictably to "no hint" rather
than raise, over-sleep, or under-sleep.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest

from ondine.utils.retry_after import RetryAfterParser


@pytest.fixture
def now_monotonic() -> float:
    """Stable clock — tests inject wallclock dates via now_wall()."""
    return 1_000_000.0


@pytest.fixture
def now_wall() -> datetime:
    return datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def parser(now_monotonic: float, now_wall: datetime) -> RetryAfterParser:
    return RetryAfterParser(
        monotonic=lambda: now_monotonic,
        utcnow=lambda: now_wall,
        max_delay_s=300.0,
    )


# ── delta-seconds integer form ─────────────────────────────────────────


def test_parses_integer_delta_seconds(parser: RetryAfterParser) -> None:
    """Regression: OpenAI/Groq/Azure send `retry-after: 30`."""
    assert parser.parse({"retry-after": "30"}) == pytest.approx(30.0)


def test_treats_zero_as_zero_not_none(parser: RetryAfterParser) -> None:
    """Regression: `retry-after: 0` must still update the bucket to 'now'
    so other callers see the signal; returning None would silently drop it."""
    assert parser.parse({"retry-after": "0"}) == 0.0


def test_clamps_absurd_values_to_max_delay(parser: RetryAfterParser) -> None:
    """Regression: during outages providers have sent `retry-after: 3600`.
    Unclamped, one 429 stalls the whole pipeline for an hour."""
    assert parser.parse({"retry-after": "3600"}) == 300.0


def test_rejects_negative_values(parser: RetryAfterParser) -> None:
    """Regression: malformed servers may return negatives — must not
    produce a negative delay that underflows downstream arithmetic."""
    assert parser.parse({"retry-after": "-5"}) is None


# ── millisecond precision (OpenAI/Azure) ───────────────────────────────


def test_prefers_retry_after_ms_over_retry_after(parser: RetryAfterParser) -> None:
    """Regression: OpenAI/Azure send both `retry-after: 2` and
    `retry-after-ms: 1250` — the latter is the precise value."""
    headers = {"retry-after": "2", "retry-after-ms": "1250"}
    assert parser.parse(headers) == pytest.approx(1.25)


def test_retry_after_ms_handles_fractional_fallback_to_seconds(
    parser: RetryAfterParser,
) -> None:
    headers = {"retry-after-ms": "not a number", "retry-after": "3"}
    assert parser.parse(headers) == pytest.approx(3.0)


# ── HTTP-date form (RFC 1123) ──────────────────────────────────────────


def test_parses_http_date_computes_delta_from_utcnow(
    parser: RetryAfterParser,
    now_wall: datetime,
) -> None:
    """Regression: some providers return an RFC 1123 date, not a
    delta."""
    future = now_wall + timedelta(seconds=45)
    header = format_datetime(future)
    assert parser.parse({"retry-after": header}) == pytest.approx(45.0, abs=1.0)


def test_http_date_in_the_past_returns_none_not_negative(
    parser: RetryAfterParser,
    now_wall: datetime,
) -> None:
    past = now_wall - timedelta(seconds=10)
    header = format_datetime(past)
    # Past date means 'retry now' — equivalent to 0, not None, so the
    # bucket is updated instead of silently falling through to local
    # backoff.
    assert parser.parse({"retry-after": header}) == 0.0


# ── anthropic absolute ISO-8601 timestamps ─────────────────────────────


def test_parses_anthropic_reset_iso8601_timestamp(
    parser: RetryAfterParser,
    now_wall: datetime,
) -> None:
    """Regression: Anthropic's `anthropic-ratelimit-*-reset` are
    ISO-8601 absolute, not deltas."""
    future = now_wall + timedelta(seconds=12)
    header = future.isoformat()
    headers = {"anthropic-ratelimit-requests-reset": header}
    assert parser.parse(headers) == pytest.approx(12.0, abs=1.0)


def test_parses_anthropic_reset_with_z_utc_suffix(
    parser: RetryAfterParser,
    now_wall: datetime,
) -> None:
    """Regression (CodeRabbit #141): real Anthropic responses send
    the UTC suffix as ``Z`` (e.g. ``2026-04-17T12:05:30Z``). Python
    3.10's ``datetime.fromisoformat`` rejects that literal; the
    parser must normalise it so behaviour is uniform across 3.10-3.13.
    """
    future = now_wall + timedelta(seconds=18)
    # Replace +00:00 with the Z form real servers emit.
    header = future.isoformat().replace("+00:00", "Z")
    headers = {"anthropic-ratelimit-tokens-reset": header}
    assert parser.parse(headers) == pytest.approx(18.0, abs=1.0)


# ── groq/openai human-duration form ────────────────────────────────────


def test_parses_groq_human_duration_like_6m0s(
    now_monotonic: float,
    now_wall: datetime,
) -> None:
    """Regression: Groq and OpenAI sometimes send durations in
    `x-ratelimit-reset-requests`/`-tokens` as strings like `6m0s`.

    Uses a parser with a generous clamp so this test isolates the
    parsing rule; clamping behaviour is covered separately.
    """
    wide_parser = RetryAfterParser(
        monotonic=lambda: now_monotonic,
        utcnow=lambda: now_wall,
        max_delay_s=600.0,
    )
    assert wide_parser.parse({"x-ratelimit-reset-requests": "6m0s"}) == pytest.approx(
        360.0
    )


def test_parses_groq_fractional_seconds(parser: RetryAfterParser) -> None:
    assert parser.parse({"x-ratelimit-reset-tokens": "7.66s"}) == pytest.approx(7.66)


# ── absence & malformed ────────────────────────────────────────────────


def test_no_relevant_headers_returns_none(parser: RetryAfterParser) -> None:
    """Regression: when a provider omits all rate-limit headers (e.g.,
    Together in some error paths), parser must return None so callers
    fall back to exponential backoff — not raise."""
    assert parser.parse({"content-type": "application/json"}) is None


def test_empty_headers_returns_none(parser: RetryAfterParser) -> None:
    assert parser.parse({}) is None


def test_malformed_header_value_returns_none(parser: RetryAfterParser) -> None:
    """Regression: unparseable values must not crash the retry loop."""
    assert parser.parse({"retry-after": "sometime next tuesday"}) is None


# ── case insensitivity ────────────────────────────────────────────────


def test_header_lookup_is_case_insensitive(parser: RetryAfterParser) -> None:
    """Regression: httpx/aiohttp normalise differently; must handle
    both casings."""
    assert parser.parse({"Retry-After": "7"}) == pytest.approx(7.0)


# ── priority ordering ──────────────────────────────────────────────────


def test_priority_ms_over_seconds_over_reset_headers(
    parser: RetryAfterParser,
) -> None:
    """Regression: when many headers present, precision order must be
    ms > seconds > ratelimit-reset. Scrambling this hides real
    retry-after intent behind coarser values."""
    headers = {
        "retry-after": "10",
        "retry-after-ms": "2500",
        "x-ratelimit-reset-requests": "60s",
    }
    assert parser.parse(headers) == pytest.approx(2.5)

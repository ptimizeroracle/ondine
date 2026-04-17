"""Tests: RateLimitError carries retry_after_s, and LiteLLM error
mapping extracts it from provider headers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ondine.adapters.unified_litellm_client import _extract_retry_after
from ondine.utils.retry_handler import RateLimitError


def test_rate_limit_error_defaults_retry_after_to_none() -> None:
    """Regression: constructing without the kwarg must not break any
    existing call site."""
    err = RateLimitError("boom")
    assert err.retry_after_s is None


def test_rate_limit_error_preserves_retry_after_s() -> None:
    err = RateLimitError("boom", retry_after_s=30.0)
    assert err.retry_after_s == 30.0


# ── header extraction ────────────────────────────────────────────────


def _mk_error(**attrs: Any) -> Exception:
    err = RuntimeError("upstream 429")
    for k, v in attrs.items():
        setattr(err, k, v)
    return err


def test_extract_reads_litellm_response_headers_attribute() -> None:
    """Regression: newer LiteLLM surfaces headers on the exception
    directly as ``litellm_response_headers``."""
    err = _mk_error(litellm_response_headers={"retry-after": "12"})
    assert _extract_retry_after(err) == pytest.approx(12.0)


def test_extract_reads_openai_response_headers() -> None:
    """Regression: OpenAI-shaped exceptions expose
    ``error.response.headers`` (httpx.Headers-like)."""
    response = SimpleNamespace(headers={"retry-after-ms": "1250"})
    err = _mk_error(response=response)
    assert _extract_retry_after(err) == pytest.approx(1.25)


def test_extract_returns_none_when_no_headers_present() -> None:
    """Regression: error mapping must never raise on missing data —
    an error mapper that raises is a footgun during a 429 storm."""
    err = _mk_error()
    assert _extract_retry_after(err) is None


def test_extract_never_raises_on_malformed_exception() -> None:
    """Regression: defensive — even if attribute access returns
    something bizarre (object that raises on dict coercion), we
    degrade to None, not blow up the retry loop."""

    class Nasty:
        def __iter__(self) -> Any:
            raise RuntimeError("nope")

    err = _mk_error(litellm_response_headers=Nasty())
    assert _extract_retry_after(err) is None

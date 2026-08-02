"""
Unit tests for the runtime Instructor mode-fallback chain (issue #187).

The Instructor client is built once with a single mode. If a provider
structurally rejects that mode ("tools not supported", "response_format is
unsupported", Instructor's parse_tools assertion), every row failed with no
attempt at a mode the provider does accept.

The client now walks a fallback chain, with three properties these tests pin:

* **Narrow.** Only structural rejections trigger a fallback. Rate limits, auth
  failures, timeouts and network errors must propagate untouched — masking
  those would re-hide the failures PR #206 just made loud.
* **Sticky.** Once a mode is found to work, it is reused. A 100k-row run must
  not re-discover the dead mode 100k times.
* **Bounded.** The chain is finite and never revisits a mode.
"""

import asyncio
from types import SimpleNamespace

import instructor
import pytest

from ondine.adapters.unified_litellm_client import (
    UnifiedLiteLLMClient,
    _is_mode_rejection_error,
    _mode_fallback_chain,
)


class _FakeBadRequestError(Exception):
    """Stands in for litellm.BadRequestError (message-shaped classification)."""


class TestErrorClassificationIsNarrow:
    """Only structural mode rejections may trigger a fallback."""

    @pytest.mark.parametrize(
        "message",
        [
            "litellm.BadRequestError: tools is not supported by this model",
            "response_format is not supported for this model",
            "json_schema is unsupported on this endpoint",
            "Invalid parameter: 'functions' is not supported",
            "This model does not support tool use",
        ],
    )
    def test_structural_rejections_are_recognised(self, message):
        assert _is_mode_rejection_error(_FakeBadRequestError(message)) is True

    @pytest.mark.parametrize(
        "message",
        [
            # The exact upstream 429 observed on OpenRouter during testing.
            "litellm.RateLimitError: RateLimitError: OpenrouterException - "
            '{"error":{"message":"Provider returned error","code":429}}',
            "AuthenticationError: Invalid API key provided",
            "401 Unauthorized",
            "Request timed out after 60s",
            "Connection error: failed to reach host",
            "Budget exceeded: run halted",
            "insufficient_quota: you exceeded your current quota",
        ],
    )
    def test_transient_and_fatal_errors_never_trigger_fallback(self, message):
        """These must stay loud — a fallback here would mask a real failure."""
        assert _is_mode_rejection_error(_FakeBadRequestError(message)) is False

    def test_parse_tools_assertion_is_recognised(self):
        """Instructor asserts exactly one tool call; a violation is structural."""
        assert (
            _is_mode_rejection_error(AssertionError("Expected exactly one tool call"))
            is True
        )

    def test_unrelated_assertion_is_not_a_mode_rejection(self):
        assert _is_mode_rejection_error(AssertionError("some other invariant")) is False


class TestFallbackChain:
    """The chain is ordered, finite, and never revisits the starting mode."""

    @pytest.mark.parametrize(
        "start",
        [
            instructor.Mode.TOOLS,
            instructor.Mode.JSON_SCHEMA,
            instructor.Mode.JSON,
        ],
    )
    def test_chain_excludes_the_starting_mode(self, start):
        chain = _mode_fallback_chain(start, provider=None)

        assert start not in chain

    @pytest.mark.parametrize(
        "start",
        [instructor.Mode.TOOLS, instructor.Mode.JSON_SCHEMA, instructor.Mode.JSON],
    )
    def test_chain_has_no_duplicates_and_is_finite(self, start):
        chain = _mode_fallback_chain(start, provider=None)

        assert len(chain) == len(set(chain))
        assert len(chain) <= 3

    def test_json_is_the_terminal_fallback(self):
        """JSON is the most widely supported mode, so it must come last."""
        chain = _mode_fallback_chain(instructor.Mode.TOOLS, provider=None)

        assert chain, "TOOLS must have at least one fallback"
        assert chain[-1] == instructor.Mode.JSON

    def test_anthropic_chain_stays_within_anthropic_modes(self):
        """A native-Anthropic client must never be handed a LiteLLM-only mode."""
        anthropic_json = getattr(instructor.Mode, "ANTHROPIC_JSON", None)
        if anthropic_json is None:  # pragma: no cover - depends on instructor version
            pytest.skip("instructor build has no ANTHROPIC_JSON")

        chain = _mode_fallback_chain(
            instructor.Mode.ANTHROPIC_TOOLS, provider="anthropic"
        )

        assert instructor.Mode.JSON_SCHEMA not in chain
        assert anthropic_json in chain


class _StubClient:
    """Minimal stand-in for UnifiedLiteLLMClient's fallback collaborators.

    Only the pieces the fallback loop touches are real; the transport is a
    counter. Using the genuine methods (bound off the real class) keeps the
    test honest — it exercises shipped logic, not a reimplementation.
    """

    def __init__(self, fail_modes, error, chain=None):
        self.model = "test/model"
        self._uses_direct_anthropic_instructor = False
        self._fail_modes = set(fail_modes)
        self._error = error
        self._mode_fallbacks = (
            list(chain)
            if chain is not None
            else [instructor.Mode.JSON_SCHEMA, instructor.Mode.JSON]
        )
        self._mode_lock = None
        self.attempts: list = []
        self.rebuilds: list = []
        self.instructor_client = SimpleNamespace(mode=instructor.Mode.TOOLS)

    # -- real logic under test, bound from the shipped class ---------------
    _invoke_structured_with_fallback = (
        UnifiedLiteLLMClient._invoke_structured_with_fallback
    )
    _advance_mode = UnifiedLiteLLMClient._advance_mode

    # -- collaborators --------------------------------------------------
    def _apply_mode_kwargs(self, call_kwargs):
        return dict(call_kwargs)

    def _build_instructor_client(self, mode):
        self.rebuilds.append(mode)
        self.instructor_client = SimpleNamespace(mode=mode)

    def _map_provider_error(self, error):
        return error

    async def _create_structured(self, call_kwargs):
        mode = self.instructor_client.mode
        self.attempts.append(mode)
        # Yield before failing so concurrent callers all reach the failure on
        # the *same* mode before any of them takes the switch lock. Without
        # this the coroutines run to completion one at a time and never
        # contend, which would make the concurrency test below vacuous.
        await asyncio.sleep(0)
        if mode in self._fail_modes:
            raise self._error
        return ("ok", None)


class TestFallbackLoopBehaviour:
    async def test_falls_back_until_a_mode_is_accepted(self):
        client = _StubClient(
            fail_modes={instructor.Mode.TOOLS, instructor.Mode.JSON_SCHEMA},
            error=_FakeBadRequestError("tools is not supported"),
        )

        result, _ = await client._invoke_structured_with_fallback({})

        assert result == "ok"
        assert client.instructor_client.mode == instructor.Mode.JSON
        assert client.attempts == [
            instructor.Mode.TOOLS,
            instructor.Mode.JSON_SCHEMA,
            instructor.Mode.JSON,
        ]

    async def test_rate_limit_propagates_without_switching_mode(self):
        """Regression guard: a 429 must never be mistaken for mode rejection."""
        err = _FakeBadRequestError(
            "litellm.RateLimitError: 429 Provider returned error"
        )
        client = _StubClient(fail_modes={instructor.Mode.TOOLS}, error=err)

        with pytest.raises(_FakeBadRequestError):
            await client._invoke_structured_with_fallback({})

        assert client.rebuilds == [], "must not switch mode on a rate limit"
        assert client.attempts == [instructor.Mode.TOOLS]

    async def test_exhausted_chain_raises_the_rejection(self):
        client = _StubClient(
            fail_modes={
                instructor.Mode.TOOLS,
                instructor.Mode.JSON_SCHEMA,
                instructor.Mode.JSON,
            },
            error=_FakeBadRequestError("tools is not supported"),
        )

        with pytest.raises(_FakeBadRequestError):
            await client._invoke_structured_with_fallback({})

        assert client._mode_fallbacks == []

    async def test_mode_switch_is_sticky_across_calls(self):
        """A 100k-row run must not re-discover the dead mode on every row."""
        client = _StubClient(
            fail_modes={instructor.Mode.TOOLS},
            error=_FakeBadRequestError("tools is not supported"),
        )

        for _ in range(5):
            await client._invoke_structured_with_fallback({})

        # TOOLS attempted once; every later call went straight to JSON_SCHEMA.
        assert client.attempts.count(instructor.Mode.TOOLS) == 1
        assert client.rebuilds == [instructor.Mode.JSON_SCHEMA]

    async def test_concurrent_failures_consume_only_one_fallback(self):
        """Ten rows failing at once must not burn ten candidates."""
        client = _StubClient(
            fail_modes={instructor.Mode.TOOLS},
            error=_FakeBadRequestError("tools is not supported"),
        )

        await asyncio.gather(
            *(client._invoke_structured_with_fallback({}) for _ in range(10))
        )

        assert client.rebuilds == [instructor.Mode.JSON_SCHEMA]
        assert client._mode_fallbacks == [instructor.Mode.JSON]

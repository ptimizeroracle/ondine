"""A provider that makes silent data loss impossible to miss.

Every bug this repository shipped in its first month had the same shape: the
pipeline finished, reported success, and quietly dropped work. A `[SKIPPED]`
frame counted as processed (#150). A response cache nothing could read (#234).
A changelog missing its largest feature (#191). A smoke matrix that verified
nothing (#189). In each case the *shape* of the output was right — correct row
count, correct columns, `success=True` — and only the *values* were wrong.

The test suite could not catch any of them, because the stub provider it used
answered every prompt with the same constant string. When the answer is the
same for every row, no assertion can tell whether row 7 got row 7's answer,
row 3's answer, or a placeholder for an API call that never happened.

`LedgerClient` fixes that at the root. Each row carries a unique token, every
answer is derived from the token that asked for it, and the client records
every call it received. That makes three otherwise-invisible classes of bug
loud:

- **misalignment** — row 7 holding row 3's answer is a value mismatch
- **omission** — a row never sent is a missing entry in the ledger
- **duplication** — a resumed run redoing finished work is a repeat entry

The client speaks both dialects the pipeline uses: one prompt per row, and the
JSON mega-batch that `with_batch_size()` builds. Which one is in play is
inferred from the prompt, so a single client covers both paths.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from decimal import Decimal
from typing import Any

import pandas as pd
from pydantic import BaseModel

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import LLMProvider, LLMSpec
from ondine.utils.retry_handler import NetworkError

#: Row markers. Fixed width so lexical and numeric order agree, which keeps
#: assertion failures readable when a whole run is printed.
TOKEN_PATTERN = re.compile(r"row-\d{4}")

#: Answers are a pure function of the asking token, so the expected output of
#: any run can be computed without running it.
ANSWER_PREFIX = "ANS-"

# Priced so the arithmetic in a cost assertion is exact rather than nearly
# right: Decimal, and a per-call cost that divides evenly.
COST_PER_CALL = Decimal("0.000100")
TOKENS_IN_PER_CALL = 8
TOKENS_OUT_PER_CALL = 4


def token_for(row: int) -> str:
    """The marker carried by row `row` of a conformance frame."""
    return f"row-{row:04d}"


def answer_for(token: str) -> str:
    """The one answer this provider will ever give for `token`."""
    return f"{ANSWER_PREFIX}{token}"


def conformance_frame(rows: int) -> pd.DataFrame:
    """Input where every row is distinguishable from every other row."""
    return pd.DataFrame({"marker": [token_for(index) for index in range(rows)]})


def expected_answers(rows: int) -> list[str]:
    """What `conformance_frame(rows)` must produce, in order."""
    return [answer_for(token_for(index)) for index in range(rows)]


def conformance_spec() -> LLMSpec:
    """A spec whose prices make cost assertions exact."""
    return LLMSpec(
        provider=LLMProvider.OPENAI,
        model="ledger-1",
        temperature=0.0,
        input_cost_per_1k_tokens=Decimal("0.001"),
        output_cost_per_1k_tokens=Decimal("0.001"),
    )


class ProviderError(RuntimeError):
    """Raised by the ledger when a token is scripted to fail."""


class LedgerClient(LLMClient):
    """Answers from the prompt's own content and records every call.

    Args:
        spec: Model spec; only pricing and the model name matter here.
        fail_tokens: Tokens whose calls raise instead of answering.
        fail_times: How many times each token in `fail_tokens` fails before
            succeeding. `None` means "always", which is how a permanently
            failing row is simulated.
        transient: Raise `NetworkError` rather than a plain error. Only
            `RetryableError` subclasses reach the retry handler — everything
            else is classified non-retryable and fails the run — so this is
            what separates "the API hiccuped" from "this request is wrong".

    Thread-safe: the async executor runs many `ainvoke()` calls at once, and a
    ledger with lost entries under concurrency would fake the very bug this
    class exists to detect.
    """

    def __init__(
        self,
        spec: LLMSpec | None = None,
        *,
        fail_tokens: set[str] | None = None,
        fail_times: int | None = None,
        transient: bool = False,
    ) -> None:
        super().__init__(spec or conformance_spec())
        self._fail_tokens = set(fail_tokens or ())
        self._fail_times = fail_times
        self._failure_type = NetworkError if transient else ProviderError
        self._lock = threading.Lock()
        self.calls: Counter[str] = Counter()
        self.prompts: list[str] = []
        self.start_count = 0
        self.stop_count = 0

    # ── ledger ────────────────────────────────────────────────────────────

    @property
    def tokens_seen(self) -> set[str]:
        """Every token this client was asked about, at least once."""
        return set(self.calls)

    @property
    def duplicated_tokens(self) -> set[str]:
        """Tokens asked about more than once — repeated work."""
        return {token for token, count in self.calls.items() if count > 1}

    def _record(self, token: str) -> int:
        with self._lock:
            self.calls[token] += 1
            return self.calls[token]

    # ── answering ─────────────────────────────────────────────────────────

    def _answer_token(self, token: str) -> str:
        attempt = self._record(token)
        if token in self._fail_tokens and (
            self._fail_times is None or attempt <= self._fail_times
        ):
            raise self._failure_type(
                f"scripted failure for {token} (attempt {attempt})"
            )
        return answer_for(token)

    def _respond(self, prompt: str) -> str:
        """Answer either dialect: a batched JSON array, or one plain prompt."""
        items = _batch_items(prompt)
        if items is not None:
            return json.dumps(
                [
                    {
                        "id": item["id"],
                        "result": self._answer_token(_token_in(str(item["input"]))),
                    }
                    for item in items
                ]
            )
        return self._answer_token(_token_in(prompt))

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        with self._lock:
            self.prompts.append(prompt)
        return LLMResponse(
            text=self._respond(prompt),
            tokens_in=TOKENS_IN_PER_CALL,
            tokens_out=TOKENS_OUT_PER_CALL,
            model=self.model,
            cost=COST_PER_CALL,
            latency_ms=1.0,
        )

    def structured_invoke(self, prompt: str, output_cls: Any, **kwargs: Any):
        return self.invoke(prompt, **kwargs)

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    # Lifecycle hooks are counted, not overridden in behaviour: a run that
    # never calls start()/stop() would leave a real client's session unopened.
    async def start(self) -> None:  # pragma: no cover - trivial counter
        self.start_count += 1

    async def stop(self) -> None:  # pragma: no cover - trivial counter
        self.stop_count += 1


def _batch_items(prompt: str) -> list[dict[str, Any]] | None:
    """The items of a JSON mega-batch prompt, or None if this is a plain one.

    `JsonBatchStrategy` writes the array after an `INPUT:` line. Matching on
    the array itself rather than the surrounding instructions keeps this from
    breaking every time that prompt's wording is tuned.
    """
    match = re.search(r"^\s*(\[.*?\])\s*$", prompt, re.MULTILINE | re.DOTALL)
    if match is None:
        return None
    try:
        items = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list) or not items:
        return None
    if not all(
        isinstance(item, dict) and "id" in item and "input" in item for item in items
    ):
        return None
    return items


def _token_in(text: str) -> str:
    """The row marker inside a prompt.

    Raises rather than returning a placeholder: a prompt with no marker means
    the pipeline sent something other than the row it claimed to send, and a
    fake provider must not paper over that.
    """
    found = TOKEN_PATTERN.findall(text)
    if not found:
        raise AssertionError(f"prompt carries no row marker: {text!r}")
    if len(set(found)) > 1:
        raise AssertionError(
            f"prompt mixes markers from several rows: {sorted(set(found))}"
        )
    return found[0]


# ── structured output ─────────────────────────────────────────────────────
#
# The structured path does not go through JsonBatchStrategy: the model returns
# a Pydantic object and the disaggregator matches its items to rows. So it
# needs its own fake, and its own answers-carry-their-own-identity trick — a
# result object that names the row it belongs to is the only way a test can
# tell a correctly-ordered batch from a reordered one.


class StructuredAnswer(BaseModel):
    """One row's answer, carrying the marker of the row that asked for it."""

    token: str
    category: str


class StructuredAnswerBatch(BaseModel):
    """What a batched structured call returns: one item per row, by position."""

    items: list[StructuredAnswer]


class StructuredLedgerClient(LLMClient):
    """Answers structurally from the prompt's own markers.

    Args:
        reverse: Return the items in reverse order. Models reorder, renumber
            and re-sort batch responses in practice; this makes that concrete
            rather than hypothetical.
    """

    def __init__(self, spec: LLMSpec | None = None, *, reverse: bool = False) -> None:
        super().__init__(spec or conformance_spec())
        self._reverse = reverse
        self.calls: Counter[str] = Counter()

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        raise AssertionError(
            "a structured pipeline must call structured_invoke, not invoke"
        )

    def structured_invoke(self, prompt: str, output_cls: Any, **kwargs: Any):
        tokens = TOKEN_PATTERN.findall(prompt)
        if not tokens:
            raise AssertionError(f"prompt carries no row marker: {prompt!r}")
        for token in tokens:
            self.calls[token] += 1

        answers = [
            StructuredAnswer(token=token, category=answer_for(token))
            for token in tokens
        ]
        if self._reverse:
            answers.reverse()

        result = (
            output_cls(items=answers)
            if "items" in output_cls.model_fields
            else output_cls(token=answers[0].token, category=answers[0].category)
        )
        response = LLMResponse(
            text=result.model_dump_json(),
            tokens_in=TOKENS_IN_PER_CALL,
            tokens_out=TOKENS_OUT_PER_CALL,
            model=self.model,
            cost=COST_PER_CALL,
            latency_ms=1.0,
        )
        response.structured_result = result
        return response

    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

"""Bare-install smoke test: exercise the public API the way a user gets it.

Every other CI job installs the dev environment (``uv sync --all-extras``),
which masks two whole classes of bug: a dependency that is only declared in
an extra, and a code path that the unit tests reach only through a mock. Both
shipped in 1.11.0. This script is what a user does — install the wheel, call
``enrich()``, look at the DataFrame — so those bugs fail here instead of in
someone's terminal.

It talks to a local stub of the OpenAI chat-completions API rather than a real
provider, so it needs no key, costs nothing, and cannot flake on the network.
The stub is reached through ``OPENAI_BASE_URL`` because ``enrich()`` has no
base_url parameter (issue #208); if that gap is ever closed, prefer passing it
explicitly here.

Run it against any interpreter that has ondine installed::

    python scripts/smoke_bare_install.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

# The answer the stub gives to every prompt. Asserting on this exact string is
# what makes the test meaningful: a run that silently fails every row comes
# back as [SKIPPED] markers, which no longer matches.
ANSWER = "positive"


class _ChatCompletionsStub(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible endpoint: one canned answer, any prompt.

    Answers whichever structured-output dialect the request asks for. Ondine
    currently sends ``response_format: json_schema`` for OpenAI models, but it
    falls back to tool calls for providers that reject that (issue #187), so
    both are handled here rather than pinning the stub to today's choice.
    """

    def do_POST(self) -> None:  # noqa: N802  (http.server's required spelling)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or "{}")
        response_format = body.get("response_format") or {}

        if body.get("tools"):
            function = body["tools"][0].get("function", {})
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_stub",
                        "type": "function",
                        "function": {
                            "name": function.get("name", "Response"),
                            "arguments": json.dumps(
                                self._fill(function.get("parameters", {}))
                            ),
                        },
                    }
                ],
            }
        elif response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema", {})
            message = {"role": "assistant", "content": json.dumps(self._fill(schema))}
        else:
            message = {"role": "assistant", "content": ANSWER}

        self._respond(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "gpt-4o-mini"),
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    @staticmethod
    def _fill(schema: dict) -> dict:
        """Build an object that satisfies the caller's own JSON schema.

        Every declared property gets ANSWER, or a plausible value for its
        declared type, so the stub validates against whatever model the test
        defines without this file having to know about it.
        """
        by_type = {"integer": 1, "number": 1.0, "boolean": True, "array": []}
        return {
            field: by_type.get(spec.get("type"), ANSWER)
            for field, spec in schema.get("properties", {}).items()
        }

    def _respond(self, payload: dict) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: object) -> None:
        """Silence per-request logging; the smoke output should be the checks."""


def _start_stub() -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), _ChatCompletionsStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    os.environ["OPENAI_API_KEY"] = "sk-stub-not-a-real-key"  # pragma: allowlist secret
    os.environ["OPENAI_BASE_URL"] = f"http://{host}:{port}/v1"
    os.environ["OPENAI_API_BASE"] = f"http://{host}:{port}/v1"
    return server


def _assert_answered(frame, column: str, rows: int, label: str) -> None:
    """Fail unless every row carries the stub's answer.

    Checking the values (not just the shape) is the point: a totally failed
    run still has the right row count and column names.
    """
    values = list(frame[column])
    if len(values) != rows:
        raise AssertionError(f"{label}: expected {rows} rows, got {len(values)}")
    if any(value != ANSWER for value in values):
        raise AssertionError(f"{label}: expected every row {ANSWER!r}, got {values}")
    print(f"  ok  {label}")


def main() -> int:
    _start_stub()

    import pandas as pd
    import polars as pl
    from pydantic import BaseModel

    import ondine

    print(f"ondine {ondine.__version__} @ {sys.executable}")

    rows = {"review": ["great product", "terrible service"]}
    prompt = "Classify the sentiment of: {review}"

    # pandas in, pandas out.
    _assert_answered(
        ondine.enrich(pd.DataFrame(rows), prompt, output_columns=["sentiment"]),
        "sentiment",
        2,
        "enrich(pandas)",
    )

    # Polars in, Polars out. This is the case that caught the undeclared
    # pyarrow dependency: the conversion to pandas needs it, and it lives in
    # an extra, so only a bare install can see it missing.
    polars_result = ondine.enrich(
        pl.DataFrame(rows), prompt, output_columns=["sentiment"]
    )
    if not isinstance(polars_result, pl.DataFrame):
        raise AssertionError(
            f"enrich(polars) must return Polars, got {type(polars_result).__name__}"
        )
    _assert_answered(polars_result, "sentiment", 2, "enrich(polars)")

    # A file path. Exercises the loader stage rather than an attached frame.
    with TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "reviews.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        _assert_answered(
            ondine.enrich(str(csv_path), prompt, output_columns=["sentiment"]),
            "sentiment",
            2,
            "enrich(csv path)",
        )

    # Structured output. This is the case that caught the builder rebuild bug:
    # passing schema= reconstructs the pipeline from its specifications, and
    # the frame has to survive that trip.
    class Sentiment(BaseModel):
        sentiment: str

    _assert_answered(
        ondine.enrich(
            pd.DataFrame(rows), prompt, output_columns=["sentiment"], schema=Sentiment
        ),
        "sentiment",
        2,
        "enrich(schema=)",
    )

    print("bare-install smoke passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

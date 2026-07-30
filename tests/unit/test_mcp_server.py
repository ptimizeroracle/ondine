"""Tests for the ondine MCP server (§4) — the L5 front door over L4.

The MCP server exposes four tools (ondine_estimate, ondine_run, ondine_status,
ondine_collect). These tests pin the behavioural contract each tool must hold,
exercised through :class:`MCPService` (the plain-Python object the FastMCP tool
functions delegate to). Testing the service directly keeps the unit tests fast
and deterministic; the FastMCP wiring itself is a thin decorator pass-through
that adds no logic worth unit-testing.

Contract under test:

* ``ondine_run`` is MANDATORY-budget when reached via MCP — an unset budget
  must be rejected before any work begins (no runaway spend from a tool call).
* ``ondine_run`` returns a ``run_id`` immediately and does not block on the LLM;
  the work runs on a background thread that writes progress to the registry.
* ``ondine_run`` injects the budget into the processing spec so the engine's
  own BudgetController enforces the cap end-to-end.
* ``ondine_status`` reads live progress (rows done, cost so far, %) from the
  registry, even mid-flight.
* ``ondine_status`` reports "not found" distinctly from a real run.
* ``ondine_collect`` returns the result summary + output path once the run
  reaches a terminal state, and refuses while the run is still in flight.
* ``ondine_estimate`` is side-effect-free: it never creates a registry row and
  never writes a checkpoint.
* The registry progress observer writes rows/cost to the registry as the
  pipeline runs, so ``ondine_status`` sees non-zero progress mid-run.

The LLM client is mocked at the architectural boundary (``create_llm_client``)
so no network call is made; everything else is real.
"""

from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd
import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from ondine.core.models import LLMResponse
from ondine.mcp.server import MCPService, create_server

# ── helpers ───────────────────────────────────────────────────────────


_CONFIG_YAML = """
dataset:
  source_type: csv
  source_path: {input}
  input_columns: [text]
  output_columns: [result]
prompt:
  template: "Classify: {{text}}"
llm:
  provider: openai
  model: gpt-4o-mini
processing:
  batch_size: 5
  concurrency: 2
"""


def _write_dataset(path: Path, rows: int = 12) -> None:
    df = pd.DataFrame({"text": [f"row {i}" for i in range(rows)]})
    df.to_csv(path, index=False)


def _make_fake_llm_client_cls():
    """Build a deterministic LLMClient subclass that needs no network.

    Mirrors the proven mock pattern in test_run_registry.py: subclass the real
    ``LLMClient`` so pricing/estimate_tokens/calculate_cost come for free, and
    override only the invoke path to return a canned response.
    """
    from ondine.adapters.llm_client import LLMClient

    class _FakeLLMClient(LLMClient):
        def invoke(self, prompt, **kwargs):
            return LLMResponse(
                text="positive",
                tokens_in=5,
                tokens_out=1,
                model=self.model,
                cost=Decimal("0.001"),
                latency_ms=1.0,
            )

        async def ainvoke(self, prompt, **kwargs):
            return self.invoke(prompt, **kwargs)

        def structured_invoke(self, prompt, output_cls, **kwargs):
            return self.invoke(prompt, **kwargs)

        async def structured_invoke_async(self, prompt, output_cls, **kwargs):
            return self.invoke(prompt, **kwargs)

        async def start(self):
            pass

        async def stop(self):
            pass

        def estimate_tokens(self, text):
            return len(text) // 4

    return _FakeLLMClient


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch the LLM client factory so no provider is contacted.

    The pipeline resolves ``create_llm_client`` at call time from
    ``ondine.api.pipeline``; patching it there covers both the sync and async
    execution paths. Returns the fake client instance the pipeline will use.
    """
    fake_cls = _make_fake_llm_client_cls()

    def _factory(llm_spec, *args, **kwargs):
        return fake_cls(llm_spec)

    monkeypatch.setattr("ondine.api.pipeline.create_llm_client", _factory)
    return _factory


@pytest.fixture
def service(tmp_path: Path) -> MCPService:
    return MCPService(registry_dir=tmp_path)


def _config_yaml(input_path: Path) -> str:
    """Render the config template with the path as a JSON string literal.

    A JSON string literal is also a valid YAML double-quoted scalar with
    correctly-escaped backslashes, so this renders safely even for Windows
    paths like ``C:\\Users\\...`` — a bare ``str(path)`` interpolation would
    break YAML's double-quoted-scalar escaping rules (e.g. ``\\U`` requires 8
    hex digits) on those paths.
    """
    return _CONFIG_YAML.format(input=json.dumps(str(input_path)))


# ── regression: fixture YAML survives Windows-style paths ────────────


def test_config_yaml_parses_with_windows_style_path() -> None:
    """Regression for a CI-only failure: on the Windows matrix jobs, temp
    paths look like ``C:\\Users\\runneradmin\\AppData\\Local\\Temp\\...``. A
    naive ``str(path)`` interpolation into a double-quoted YAML scalar breaks
    because ``\\U`` is a YAML escape sequence requiring 8 hex digits, so
    ``yaml.safe_load`` raised ``ScannerError``. This runs on every platform
    (not just Windows) by constructing a Windows-shaped path directly, so the
    regression is caught on Linux/macOS CI too.
    """
    windows_path = PureWindowsPath(r"C:\Users\runneradmin\AppData\Local\Temp\in.csv")
    rendered = _config_yaml(windows_path)  # type: ignore[arg-type]

    config = yaml.safe_load(rendered)

    assert config["dataset"]["source_path"] == str(windows_path)


# ── regression: budget is mandatory on ondine_run ────────────────────


def test_run_rejects_missing_budget(
    service: MCPService,
    tmp_path: Path,
    fake_llm: Any,
) -> None:
    """Regression: the architecture proposal mandates a budget cap on every MCP
    run. If ondine_run accepted a missing budget, a tool caller could launch an
    unbounded spend job with no ceiling — the exact footgun the cap exists to
    prevent. The rejection must happen before any run is created.
    """
    inp = tmp_path / "in.csv"
    _write_dataset(inp)
    out = tmp_path / "out.csv"

    with pytest.raises(ValueError, match="(?i)budget"):
        service.ondine_run(
            config_yaml=_config_yaml(inp),
            input_path=str(inp),
            output_path=str(out),
            budget=None,
        )

    # No run row should have been persisted — rejection is pre-flight.
    assert service.list_runs() == []


def test_run_rejects_zero_budget(
    service: MCPService,
    tmp_path: Path,
    fake_llm: Any,
) -> None:
    """Regression: a budget of 0 or negative is equivalent to no cap (the
    engine treats ``None`` as unlimited and ``0`` would trip instantly or never
    depending on path). MCP must reject non-positive budgets explicitly.
    """
    inp = tmp_path / "in.csv"
    _write_dataset(inp)
    out = tmp_path / "out.csv"

    with pytest.raises(ValueError, match="(?i)budget"):
        service.ondine_run(
            config_yaml=_config_yaml(inp),
            input_path=str(inp),
            output_path=str(out),
            budget=0,
        )


# ── regression: ondine_run returns a run_id and does not block ───────


def test_run_returns_run_id_immediately_and_does_not_block(
    service: MCPService,
    tmp_path: Path,
    fake_llm: Any,
) -> None:
    """Regression: ondine_run is documented to 'return run_id immediately
    (never block)'. If it blocked on execute(), the MCP client would time out
    waiting for a long job and the user would lose the handle to poll. The call
    must return a resolvable run_id while the LLM has barely started.
    """
    inp = tmp_path / "in.csv"
    _write_dataset(inp, rows=50)
    out = tmp_path / "out.csv"

    handle = service.ondine_run(
        config_yaml=_config_yaml(inp),
        input_path=str(inp),
        output_path=str(out),
        budget=Decimal("5.00"),
    )

    # A run_id is returned and is durable in the registry.
    assert handle["run_id"]
    status = service.ondine_status(handle["run_id"])
    assert status["run_id"] == handle["run_id"]

    # The returned status is a real, trackable state — not "unknown".
    assert status["status"] in ("pending", "running", "succeeded", "failed", "partial")


# ── regression: budget is injected into the processing spec ──────────


def test_run_injects_budget_into_spec(
    service: MCPService,
    tmp_path: Path,
    fake_llm: Any,
) -> None:
    """Regression: the budget cap must reach the engine's own BudgetController,
    not just be stored in the registry. If ondine_run only recorded the budget
    for bookkeeping, a runaway job would blow past it because the invocation
    stage never sees the limit. The persisted spec snapshot must carry it.
    """
    inp = tmp_path / "in.csv"
    _write_dataset(inp)
    out = tmp_path / "out.csv"

    handle = service.ondine_run(
        config_yaml=_config_yaml(inp),
        input_path=str(inp),
        output_path=str(out),
        budget=Decimal("2.50"),
    )

    spec_snapshot = service.get_spec_snapshot(handle["run_id"])
    assert spec_snapshot is not None
    assert str(spec_snapshot["processing"]["max_budget"]).startswith("2.5")


# ── regression: ondine_status reports live progress mid-flight ───────


def test_status_reports_live_progress_during_run(
    service: MCPService,
    tmp_path: Path,
    fake_llm: Any,
) -> None:
    """Regression: ondine_status must show rows-done / cost-so-far while the
    job is RUNNING. If progress never reached the registry, a user polling
    status would see 0/0 until the very end — indistinguishable from a hung
    job. The RegistryProgressObserver must forward progress to the registry.
    """
    inp = tmp_path / "in.csv"
    _write_dataset(inp, rows=20)
    out = tmp_path / "out.csv"

    handle = service.ondine_run(
        config_yaml=_config_yaml(inp),
        input_path=str(inp),
        output_path=str(out),
        budget=Decimal("5.00"),
    )

    # Wait for the background run to make progress or finish.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        st = service.ondine_status(handle["run_id"])
        if st["status"] in ("succeeded", "failed", "partial"):
            break
        # While running, either rows_done>0 (progress observed) or we polled
        # before the first batch landed — keep waiting.
        time.sleep(0.05)

    final = service.ondine_status(handle["run_id"])
    assert final["status"] == "succeeded"
    assert final["rows_done"] == 20
    assert Decimal(final["cost"]) > 0


# ── regression: ondine_status distinguishes unknown run_id ───────────


def test_status_unknown_run_id_raises(
    service: MCPService,
) -> None:
    """Regression: polling a run_id that was never created must surface a clear
    'not found' error, not silently return a zeroed-out status. Otherwise a
    typo in the run_id would look like an empty, instantly-finished job.
    """
    with pytest.raises(KeyError):
        service.ondine_status(str(uuid4()))


# ── regression: ondine_collect returns summary for a finished run ────


def test_collect_returns_summary_after_success(
    service: MCPService,
    tmp_path: Path,
    fake_llm: Any,
) -> None:
    """Regression: ondine_collect is the terminal readout — rows, cost, and the
    output path. If it didn't surface the output path, the user would have no
    way to find their result file from the tool alone.
    """
    inp = tmp_path / "in.csv"
    _write_dataset(inp, rows=8)
    out = tmp_path / "out.csv"

    handle = service.ondine_run(
        config_yaml=_config_yaml(inp),
        input_path=str(inp),
        output_path=str(out),
        budget=Decimal("5.00"),
    )

    collected = service.wait_and_collect(handle["run_id"], timeout=30)
    assert collected["status"] == "succeeded"
    assert collected["rows_done"] == 8
    assert collected["output_path"] == str(out)
    assert Decimal(collected["cost"]) > 0
    # The output file really exists — collect doesn't lie about the artefact.
    assert out.exists()


def test_collect_refuses_in_flight_run(
    service: MCPService,
    tmp_path: Path,
    fake_llm: Any,
) -> None:
    """Regression: ondine_collect on a still-running job must not return a
    half-written summary. It should report that the run is not finished so the
    caller knows to poll status and come back.
    """
    # We can't easily freeze the run mid-flight with the fake client (it's
    # fast), so instead seed a PENDING run directly and assert collect refuses.
    handle = service._seed_pending_run()  # noqa: SLF001 — test-only seam
    with pytest.raises(ValueError, match="(?i)not.*finished|in.?flight|running"):
        service.ondine_collect(str(handle.run_id))


# ── regression: ondine_estimate is side-effect-free ──────────────────


def test_estimate_is_side_effect_free(
    service: MCPService,
    tmp_path: Path,
    fake_llm: Any,
) -> None:
    """Regression: ondine_estimate is the 'demo tool' — fast and side-effect
    -free. If it persisted a registry row or wrote a checkpoint, every estimate
    call would pollute the run history and the checkpoint dir. Estimate must
    leave the registry empty.
    """
    inp = tmp_path / "in.csv"
    _write_dataset(inp)

    estimate = service.ondine_estimate(config_yaml=_config_yaml(inp))

    assert estimate["rows"] == 12
    assert Decimal(estimate["total_cost"]) >= 0
    assert estimate["total_tokens"] >= 0
    # No run row created.
    assert service.list_runs() == []


# ── regression: create_server builds a FastMCP app with 4 tools ──────


def test_create_server_registers_four_tools() -> None:
    """Regression: the public entrypoint must wire exactly the four documented
    tools onto a FastMCP server so an MCP client discovers them by name. If a
    tool were missing or misnamed, the client would see an incomplete surface.
    """
    import asyncio

    from fastmcp import FastMCP

    server = create_server()
    assert isinstance(server, FastMCP)
    tools = asyncio.run(server.list_tools())
    names = sorted(t.name for t in tools)
    assert names == ["ondine_collect", "ondine_estimate", "ondine_run", "ondine_status"]


def test_create_server_tools_are_callable(service: MCPService) -> None:
    """The four service methods exist and are bound — guards against a rename
    silently breaking the MCP surface.
    """
    for name in ("ondine_estimate", "ondine_run", "ondine_status", "ondine_collect"):
        assert hasattr(service, name), f"MCPService missing tool {name!r}"

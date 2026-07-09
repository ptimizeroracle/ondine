"""ondine MCP server — an L5 front door exposing pipeline ops as MCP tools.

Architecture (see /plans/ARCHITECTURE_PROPOSAL.md §4):

* This module sits at L5 (front door). It imports only L4 (Pipeline) and
  ondine.config — never stages or engines. The four tools are thin façades.
* ``ondine_run`` hands back a ``run_id`` immediately and runs the pipeline on a
  daemon thread, writing progress to the shared RunRegistry (§2). A second
  tool call (``ondine_status`` / ``ondine_collect``) — even from another
  process — reads the durable registry row to report live progress and final
  results.
* A budget cap is MANDATORY on every ``ondine_run``: the whole point of
  exposing dataset processing to an LLM tool client is that the caller may not
  understand the cost implications, so the server refuses to launch without an
  explicit ceiling and injects it into the processing spec so the engine's own
  BudgetController enforces it end-to-end.

The MCP wire layer (FastMCP 3.x) is a pass-through: ``create_server()`` decorates
the four ``MCPService`` methods and returns the app; all behaviour lives in
:class:`MCPService` so it is directly unit-testable without an MCP client.
"""

from __future__ import annotations

import threading
import uuid
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ondine.api.pipeline import Pipeline
from ondine.config.config_loader import ConfigLoader
from ondine.mcp.progress import RegistryProgressObserver
from ondine.utils import get_logger
from ondine.utils.optional_dependencies import _matches_missing_dependency

if TYPE_CHECKING:
    from ondine.orchestration.run_registry import (
        RunHandle,
        RunRegistry,
        RunStatus,
    )

logger = get_logger(__name__)


def _require_fastmcp() -> Any:
    """Import FastMCP lazily so importing this module never forces the extra.

    The ``ondine[mcp]`` extra installs ``fastmcp``; users who never run the
    server should not pay the import cost or see an import error. This helper
    raises a clear, actionable error pointing at the extra on missing dep.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        if _matches_missing_dependency(exc, ("fastmcp", "mcp")):
            raise ImportError(
                "The ondine MCP server requires the 'mcp' extra. "
                "Install with: pip install 'ondine[mcp]'"
            ) from exc
        raise
    return FastMCP


class MCPService:
    """Plain-Python implementation of the four MCP tools.

    Every method maps 1:1 to an MCP tool (``ondine_estimate`` →
    :meth:`ondine_estimate`, etc.). Keeping the logic in a transport-agnostic
    class means the behaviour is fully testable without spinning up an MCP
    client, and the FastMCP layer in :func:`create_server` is a trivial
    decorator pass-through.

    State held across calls:

    * ``_registry`` — the shared :class:`RunRegistry` (§2). This is the
      cross-call (and cross-process) channel for run identity, status, and
      live progress. It is the *only* durable state the service owns.
    * ``_threads`` — in-process background workers keyed by run_id. v1 runs
      jobs on a daemon thread of the server process; the registry's on-disk
      row means a crash or restart still leaves a resumable checkpoint and a
      discoverable run_id. True out-of-process workers are §3's scope.
    """

    def __init__(
        self,
        registry_dir: str | Path | None = None,
        registry: RunRegistry | None = None,
    ) -> None:
        if registry is not None:
            self._registry = registry
            self._owns_registry = False
        else:
            from ondine.orchestration.run_registry import RunRegistry

            reg_dir = (
                Path(registry_dir) if registry_dir is not None else Path(".checkpoints")
            )
            reg_dir.mkdir(parents=True, exist_ok=True)
            self._registry = RunRegistry(reg_dir)
            self._owns_registry = True
        self._threads: dict[str, threading.Thread] = {}

    # ── ondine_estimate ─────────────────────────────────────────────

    def ondine_estimate(
        self,
        config_yaml: str,
        sample_rows: int | None = None,
    ) -> dict[str, Any]:
        """Estimate cost/token/row counts for a config WITHOUT running.

        Side-effect-free: builds a Pipeline purely to call ``estimate_cost()``,
        never touches the registry and never writes a checkpoint.
        """
        pipeline = self._build_pipeline(config_yaml)
        estimate = pipeline.estimate_cost()
        return {
            "total_cost": str(estimate.total_cost),
            "total_tokens": estimate.total_tokens,
            "input_tokens": estimate.input_tokens,
            "output_tokens": estimate.output_tokens,
            "rows": estimate.rows,
            "confidence": estimate.confidence,
        }

    # ── ondine_run ──────────────────────────────────────────────────

    def ondine_run(
        self,
        config_yaml: str,
        input_path: str,
        output_path: str,
        budget: float | Decimal | str | None,
    ) -> dict[str, Any]:
        """Launch a pipeline run, returning ``run_id`` immediately.

        ``budget`` is MANDATORY and must be positive — an LLM tool client must
        not be able to launch an unbounded-spend job. The budget is injected
        into the processing spec so the engine's BudgetController enforces it,
        and persisted in the run's spec snapshot for audit.
        """
        budget_decimal = self._require_positive_budget(budget)

        pipeline = self._build_pipeline(config_yaml)
        # Inject the mandatory budget into the spec the engine will actually
        # run, so the BudgetController caps spend at the source.
        pipeline.specifications.processing.max_budget = budget_decimal

        # Override I/O paths from the tool arguments (they are the source of
        # truth for an MCP call, superseding whatever the YAML carried).
        pipeline.specifications.dataset.source_path = Path(input_path)
        self._apply_output(pipeline, Path(output_path))

        # Register the run BEFORE launching so the id is durable even if the
        # worker thread fails to start. The snapshot carries the budget for
        # audit and the input/output paths for collect().
        spec_snapshot = pipeline.specifications.model_dump(mode="json")
        spec_snapshot.setdefault("_mcp", {})
        spec_snapshot["_mcp"].update(
            {"input_path": str(input_path), "output_path": str(output_path)}
        )
        from ondine.orchestration.run_registry import RunSpec

        run_spec = RunSpec(
            pipeline_id=str(pipeline.id),
            dataset=str(input_path),
            spec_snapshot=spec_snapshot,
        )
        handle = self._registry.create(run_spec)
        run_id_str = str(handle.run_id)

        # Wire pipeline progress → registry so ondine_status sees live rows/cost.
        pipeline.add_observer(RegistryProgressObserver(self._registry, handle.run_id))

        # Run on a daemon thread: the call returns the run_id now; the thread
        # drives the registry through RUNNING → SUCCEEDED|FAILED.
        thread = threading.Thread(
            target=self._run_worker,
            name=f"ondine-mcp-{run_id_str[:8]}",
            args=(pipeline, handle.run_id),
            daemon=True,
        )
        self._threads[run_id_str] = thread
        thread.start()
        return {"run_id": run_id_str}

    # ── ondine_status ───────────────────────────────────────────────

    def ondine_status(self, run_id: str) -> dict[str, Any]:
        """Live status: state, progress %, rows done, cost so far."""
        handle = self._registry.get(_to_uuid(run_id))
        if handle is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        metrics = handle.metrics or {}
        total = metrics.get("total_rows", 0) or 0
        done = metrics.get("processed_rows", metrics.get("rows_done", 0)) or 0
        progress_pct = (done / total * 100) if total else 0.0
        return {
            "run_id": str(handle.run_id),
            "status": handle.status.value,
            "progress_pct": round(progress_pct, 2),
            "rows_done": done,
            "total_rows": total,
            "cost": str(metrics.get("cost", "0")),
            "updated_at": handle.updated_at,
        }

    # ── ondine_collect ──────────────────────────────────────────────

    def ondine_collect(self, run_id: str) -> dict[str, Any]:
        """Terminal readout: rows, cost, and the output path.

        Refuses a non-terminal run so a caller never gets a half-written
        summary. Use :meth:`ondine_status` to follow an in-flight run.
        """
        handle = self._registry.get(_to_uuid(run_id))
        if handle is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        if handle.status.value not in ("succeeded", "failed", "partial"):
            raise ValueError(
                f"Run {run_id} is not finished (status={handle.status.value}); "
                "poll ondine_status until terminal."
            )
        metrics = handle.metrics or {}
        mcp_meta = (handle.spec_snapshot or {}).get("_mcp", {})
        output_path = mcp_meta.get("output_path") or metrics.get("output_path", "")
        return {
            "run_id": str(handle.run_id),
            "status": handle.status.value,
            "rows_done": metrics.get("processed_rows", metrics.get("rows_done", 0)),
            "total_rows": metrics.get("total_rows", 0),
            "cost": str(metrics.get("cost", "0")),
            "output_path": str(output_path),
            "error": metrics.get("error"),
            "updated_at": handle.updated_at,
        }

    # ── convenience / test seams ────────────────────────────────────

    def list_runs(self) -> list[RunHandle]:
        """List all known runs (newest-first), for inspection/testing."""
        return self._registry.list()

    def get_spec_snapshot(self, run_id: str) -> dict[str, Any] | None:
        handle = self._registry.get(_to_uuid(run_id))
        return dict(handle.spec_snapshot) if handle is not None else None

    def wait_and_collect(self, run_id: str, timeout: float = 60.0) -> dict[str, Any]:
        """Block until the run is terminal, then return :meth:`ondine_collect`.

        Convenience for tests and synchronous callers; the MCP tool itself
        never blocks (ondine_run returns immediately).
        """
        import time

        uid = _to_uuid(run_id)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            handle = self._registry.get(uid)
            if handle is not None and handle.status.value in (
                "succeeded",
                "failed",
                "partial",
            ):
                return self.ondine_collect(run_id)
            time.sleep(0.05)
        raise TimeoutError(f"Run {run_id} did not finish within {timeout}s")

    def _seed_pending_run(self) -> RunHandle:
        """Create a PENDING run with no worker — a test seam for asserting
        that ondine_collect refuses an in-flight run."""
        from ondine.orchestration.run_registry import RunSpec

        return self._registry.create(RunSpec(pipeline_id="test-seed", dataset="seed"))

    def close(self) -> None:
        if self._owns_registry:
            self._registry.close()

    # ── internals ───────────────────────────────────────────────────

    @staticmethod
    def _require_positive_budget(
        budget: float | Decimal | str | None,
    ) -> Decimal:
        if budget is None:
            raise ValueError(
                "A budget cap is mandatory for ondine_run. Pass a positive "
                "budget (USD) to set the maximum spend for this run."
            )
        try:
            dec = Decimal(str(budget))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Budget must be a number, got {budget!r}") from exc
        if dec <= 0:
            raise ValueError(
                f"Budget must be positive (got {dec}); an MCP run cannot be "
                "launched without a spend ceiling."
            )
        return dec

    @staticmethod
    def _build_pipeline(config_yaml: str) -> Pipeline:
        """Parse a YAML/JSON config string into a Pipeline (no execution).

        Reuses ConfigLoader so MCP and CLI accept identical config shapes.
        """
        config_dict = yaml.safe_load(config_yaml)
        if not isinstance(config_dict, dict):
            raise ValueError(
                "config_yaml must parse to a mapping (dict); got "
                f"{type(config_dict).__name__}"
            )
        specs = ConfigLoader._dict_to_specifications(config_dict)  # noqa: SLF001
        return Pipeline(specs)

    @staticmethod
    def _apply_output(pipeline: Pipeline, output_path: Path) -> None:
        from ondine.core.specifications import (
            DataSourceType,
            MergeStrategy,
            OutputSpec,
        )

        suffix = output_path.suffix.lower()
        dst_type = {
            ".csv": DataSourceType.CSV,
            ".json": DataSourceType.CSV,
            ".parquet": DataSourceType.PARQUET,
            ".xlsx": DataSourceType.EXCEL,
            ".xls": DataSourceType.EXCEL,
        }.get(suffix, DataSourceType.CSV)
        pipeline.specifications.output = OutputSpec(
            destination_type=dst_type,
            destination_path=output_path,
            merge_strategy=MergeStrategy.REPLACE,
        )

    def _run_worker(self, pipeline: Pipeline, run_id: uuid.UUID) -> None:
        """Background worker: drive the registry through the run lifecycle.

        Errors are swallowed here (and recorded as FAILED in the registry by
        Pipeline.execute itself) so a failing worker never crashes the server
        process or leaves the thread pool in a bad state.
        """

        # Re-derive total rows for progress math before execute mutates state.
        try:
            loader_df = pipeline.dataframe
            if loader_df is None and pipeline.specifications.dataset.source_path:
                import pandas as pd

                loader_df = pd.read_csv(pipeline.specifications.dataset.source_path)
            total_rows = int(len(loader_df)) if loader_df is not None else 0
            self._registry.update_metrics(run_id, {"total_rows": total_rows})
        except Exception:  # noqa: BLE001
            # Non-fatal: progress % will just read 0 total until the observer
            # writes the real count. Don't block the run on row-count math.
            pass

        try:
            pipeline.execute(run_id=run_id, registry=self._registry)
        except Exception as exc:  # noqa: BLE001
            # Pipeline.execute already recorded FAILED in the registry; this
            # guard is for failures in the thread plumbing itself.
            logger.exception("ondine_run worker for %s crashed", run_id)
            try:
                self._registry.transition(
                    run_id,
                    self._terminal_for(run_id),
                    metrics={"error": f"{type(exc).__name__}: {exc}"},
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._threads.pop(str(run_id), None)

    def _terminal_for(self, run_id: uuid.UUID) -> RunStatus:
        """Pick a legal terminal status from the run's current state."""
        from ondine.orchestration.run_registry import RunStatus

        handle = self._registry.get(run_id)
        if handle is not None and handle.status is RunStatus.PENDING:
            return RunStatus.FAILED  # PENDING can only move to RUNNING or FAILED
        return RunStatus.FAILED


def _to_uuid(run_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(run_id))
    except (ValueError, TypeError) as exc:
        raise KeyError(f"Invalid run_id (not a UUID): {run_id!r}") from exc


def create_server(service: MCPService | None = None) -> Any:
    """Build the FastMCP app with the four ondine tools registered.

    Returns the FastMCP instance; call ``.run()`` (stdio) to serve. Importing
    this function does not import FastMCP — only calling it does — so users
    without the ``ondine[mcp]`` extra can still import the rest of ondine.
    """
    fastmcp_cls = _require_fastmcp()
    svc = service if service is not None else MCPService()
    mcp = fastmcp_cls("ondine")

    @mcp.tool(
        name="ondine_estimate",
        description=(
            "Estimate the cost, token usage, and row count for an ondine "
            "pipeline configuration WITHOUT running it. Fast and side-effect "
            "-free. Pass the full config as a YAML string."
        ),
    )
    def _estimate(
        config_yaml: str,
        sample_rows: int | None = None,
    ) -> dict[str, Any]:
        return svc.ondine_estimate(config_yaml, sample_rows)

    @mcp.tool(
        name="ondine_run",
        description=(
            "Launch an ondine pipeline run. Returns run_id immediately — the "
            "job runs in the background; poll ondine_status for progress and "
            "call ondine_collect once it finishes. A positive budget (USD) is "
            "MANDATORY and caps total spend."
        ),
    )
    def _run(
        config_yaml: str,
        input_path: str,
        output_path: str,
        budget: float,
    ) -> dict[str, Any]:
        return svc.ondine_run(config_yaml, input_path, output_path, budget)

    @mcp.tool(
        name="ondine_status",
        description=(
            "Poll the live status of an ondine run: state, progress %, rows "
            "done, and cost so far. Returns 'unknown run_id' if the id was "
            "never created."
        ),
    )
    def _status(run_id: str) -> dict[str, Any]:
        return svc.ondine_status(run_id)

    @mcp.tool(
        name="ondine_collect",
        description=(
            "Collect the final result of a finished ondine run: rows, cost, "
            "the output file path, and any error. Refuses a still-running run "
            "(poll ondine_status first)."
        ),
    )
    def _collect(run_id: str) -> dict[str, Any]:
        return svc.ondine_collect(run_id)

    return mcp


def main() -> None:
    """Console-script entrypoint: ``ondine-mcp`` serves over stdio."""
    server = create_server()
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()

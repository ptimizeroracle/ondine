"""
Langfuse observer for LLM-specific observability.

Uses Langfuse SDK directly. No dependency on LiteLLM's internal callback
mechanism.

Langfuse v3 replaced the v2 `client.trace(...)` / `trace.generation(...)` /
`trace.span(...)` API with an OpenTelemetry-based model: observations are
created via `client.start_observation(as_type=...)` (or the equivalent method
on a parent observation), updated with `.update(...)`, and closed with
`.end()`. A "trace" is no longer a distinct object — it is simply the root
observation, optionally pinned to a caller-chosen trace id via
`trace_context={"trace_id": ...}` (which must be a 32-char lowercase hex
string, hence the `uuid.UUID(...).hex` conversions below).
"""

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any, cast

from ondine.observability.base import PipelineObserver
from ondine.observability.events import (
    LLMCallEvent,
    PipelineEndEvent,
    PipelineStartEvent,
    ProviderCooldownEvent,
    ProviderRecoveredEvent,
)
from ondine.observability.registry import observer

if TYPE_CHECKING:
    from langfuse import Langfuse, LangfuseSpan
    from langfuse.types import TraceContext

logger = logging.getLogger(__name__)


@observer("langfuse")
class LangfuseObserver(PipelineObserver):
    """
    Observer that uses Langfuse SDK directly for LLM observability.

    This implementation:
    - Targets the Langfuse v3+ OTEL-based observation API
    - Does NOT depend on LiteLLM's internal callbacks
    - Receives events from Ondine's ObserverDispatcher

    Tracks:
    - Full prompts and completions
    - Token usage and costs
    - Latency metrics
    - Model information
    - Pipeline traces with nested generations

    Configuration:
    - Requires environment variables: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
    - Optional: LANGFUSE_HOST (defaults to cloud)
    - Or pass in 'config' dict with keys

    Example:
        observer = LangfuseObserver(config={
            "public_key": "pk-lf-xxx",  # pragma: allowlist secret
            "secret_key": "sk-lf-xxx",  # pragma: allowlist secret
            "host": "https://cloud.langfuse.com"
        })
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Langfuse observer with direct SDK client.
        """
        super().__init__(config)

        # Quoted (not unwrapped by ruff's UP037): `Langfuse` is only bound as a
        # local name a few lines below (inside the try block's import), so an
        # unquoted annotation here trips a "referenced before assignment"
        # false positive even though annotations aren't evaluated at runtime.
        self._client: "Langfuse | None" = None  # noqa: UP037
        self._current_trace: "LangfuseSpan | None" = None  # noqa: UP037

        # Initialize Langfuse client directly
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=self.config.get("public_key")
                or os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=self.config.get("secret_key")
                or os.getenv("LANGFUSE_SECRET_KEY"),
                host=self.config.get("host")
                or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )
            logger.info("Langfuse observer initialized (direct SDK)")

        except ImportError:
            logger.warning(
                "Langfuse SDK not installed. Install with: pip install langfuse"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")

    @staticmethod
    def _to_trace_id(value: str) -> str:
        """
        Normalize an Ondine trace/run id (a `str(uuid.uuid4())`, with dashes)
        into the 32-char lowercase hex string Langfuse requires for
        `trace_context={"trace_id": ...}`. Falls back to the raw value if it
        isn't a parseable UUID, letting Langfuse's own validation warn.
        """
        try:
            return uuid.UUID(value).hex
        except ValueError:
            return value

    def on_pipeline_start(self, event: PipelineStartEvent) -> None:
        """
        Create a new root span (the pipeline "trace") for the pipeline run.
        """
        if not self._client:
            return

        try:
            # mypy can't match an inline dict literal against the TraceContext
            # TypedDict through start_observation's @overload set (spurious
            # "no overload matches" / NotRequired-key false positive), so we
            # cast the literal explicitly rather than losing type-checking on
            # the rest of the call via `Any`.
            trace_context = cast("TraceContext", {"trace_id": event.run_id.hex})
            self._current_trace = self._client.start_observation(
                name="ondine-pipeline",
                as_type="span",
                trace_context=trace_context,
                metadata={
                    "pipeline_id": str(event.pipeline_id),
                    "total_rows": event.total_rows,
                    **event.metadata,
                },
            )
            logger.debug(f"Created Langfuse trace: {event.run_id}")
        except Exception as e:
            logger.warning(f"Failed to create Langfuse trace: {e}")

    def on_llm_call(self, event: LLMCallEvent) -> None:
        """
        Log LLM call as a generation observation in Langfuse.
        """
        if not self._client:
            return

        try:
            usage_details = {
                "input": event.input_tokens,
                "output": event.output_tokens,
                "total": event.total_tokens,
            }
            metadata = {
                "provider": event.provider,
                "temperature": event.temperature,
                "max_tokens": event.max_tokens,
                "latency_ms": event.latency_ms,
                "cost": float(event.cost),
                "row_index": event.row_index,
                "stage_name": event.stage_name,
                **event.metadata,
            }

            # Nest under the current pipeline trace if available, otherwise
            # start a standalone root generation.
            if self._current_trace:
                generation = self._current_trace.start_observation(
                    name=f"llm-{event.model}",
                    as_type="generation",
                    model=event.model,
                    input=event.prompt,
                    output=event.completion,
                    usage_details=usage_details,
                    metadata=metadata,
                )
            else:
                # See on_pipeline_start for why this needs an explicit cast.
                trace_context = cast(
                    "TraceContext", {"trace_id": self._to_trace_id(event.trace_id)}
                )
                generation = self._client.start_observation(
                    name=f"llm-{event.model}",
                    as_type="generation",
                    trace_context=trace_context,
                    model=event.model,
                    input=event.prompt,
                    output=event.completion,
                    usage_details=usage_details,
                    metadata=metadata,
                )
            generation.end()

        except Exception as e:
            logger.debug(f"Failed to log LLM call to Langfuse: {e}")

    def on_pipeline_end(self, event: PipelineEndEvent) -> None:
        """
        Update the root span with final pipeline metrics and close it.
        """
        if not self._client or not self._current_trace:
            return

        try:
            self._current_trace.update(
                output={
                    "success": event.success,
                    "rows_processed": event.rows_processed,
                    "rows_succeeded": event.rows_succeeded,
                    "rows_failed": event.rows_failed,
                    "total_cost": float(event.total_cost),
                    "total_tokens": event.total_tokens,
                    "duration_ms": event.total_duration_ms,
                },
            )
            self._current_trace.end()
            logger.debug("Updated Langfuse trace with final metrics")
        except Exception as e:
            logger.debug(f"Failed to update Langfuse trace: {e}")

    def on_provider_cooldown(self, event: ProviderCooldownEvent) -> None:
        """
        Log provider cooldown as a span in Langfuse.
        """
        if not self._client:
            return

        try:
            metadata = {
                "provider": event.provider,
                "deployment_id": event.deployment_id,
                "reason": event.reason,
                "cooldown_duration": event.cooldown_duration,
                "fail_count": event.fail_count,
                "event_type": "circuit_breaker_triggered",
                **event.metadata,
            }
            # Nest under the current pipeline trace if available, otherwise
            # create a standalone root span.
            if self._current_trace:
                span = self._current_trace.start_observation(
                    name="provider-cooldown",
                    as_type="span",
                    metadata=metadata,
                    level="WARNING",
                )
            else:
                span = self._client.start_observation(
                    name="provider-cooldown",
                    as_type="span",
                    metadata=metadata,
                    level="WARNING",
                )
            span.end()
        except Exception as e:
            logger.debug(f"Failed to log provider cooldown to Langfuse: {e}")

    def on_provider_recovered(self, event: ProviderRecoveredEvent) -> None:
        """
        Log provider recovery as a span in Langfuse.
        """
        if not self._client or not self._current_trace:
            return

        try:
            span = self._current_trace.start_observation(
                name="provider-recovered",
                as_type="span",
                metadata={
                    "provider": event.provider,
                    "deployment_id": event.deployment_id,
                    "cooldown_duration": event.cooldown_duration,
                    "event_type": "circuit_breaker_recovered",
                    **event.metadata,
                },
                level="DEFAULT",
            )
            span.end()
        except Exception as e:
            logger.debug(f"Failed to log provider recovery to Langfuse: {e}")

    def flush(self) -> None:
        """Flush buffered events to Langfuse."""
        if not self._client:
            return

        try:
            self._client.flush()
        except Exception as e:
            logger.debug(f"Failed to flush Langfuse: {e}")

    def close(self) -> None:
        """Cleanup Langfuse client."""
        self.flush()
        if self._client:
            try:
                # Langfuse v2 uses shutdown(), v3 may differ
                if hasattr(self._client, "shutdown"):
                    self._client.shutdown()
            except Exception:  # nosec B110
                # Cleanup errors are non-critical
                pass
        self._current_trace = None

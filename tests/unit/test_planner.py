"""Tests for ondine.plan() intent layer.

These tests exercise the planner as a pure transformation:
  goal + data sample + LLM client  ->  Plan(PipelineSpecifications)

The LLM client is the only non-deterministic / networked collaborator
and is the architectural boundary, so we inject a Fake here. The SUT
is never mocked. See the tdd-enterprise skill: mock the boundary, not
the SUT; verify observable outcomes through the public interface.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pydantic import BaseModel

from ondine.adapters.llm_client import LLMClient
from ondine.core.models import LLMResponse
from ondine.core.specifications import PipelineSpecifications
from ondine.orchestration.intent.planner import Plan, plan

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _ScriptedClient(LLMClient):
    """LLM client that returns a canned structured payload.

    A Fake: a working, in-memory implementation of the LLMClient port.
    We use it (not a mock) because the planner only needs the structured
    payload, and a Fake keeps the assertion on real production wiring
    instead of on call counts of a mock.
    """

    def __init__(self, spec: Any, payload: dict[str, Any]) -> None:
        # Intentionally bypass LLMClient.__init__ heavy work.
        self.spec = spec
        self._payload = payload
        self.captured_prompt: str | None = None

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        raise AssertionError("plan() must use structured_invoke, not invoke()")

    def structured_invoke(
        self,
        prompt: str,
        output_cls: type[BaseModel],
        **kwargs: Any,
    ) -> LLMResponse:
        self.captured_prompt = prompt
        # Build the real Pydantic object the planner expects.
        instance = output_cls.model_validate(self._payload)
        return LLMResponse(
            text="",
            tokens_in=0,
            tokens_out=0,
            model="fake",
            cost=Decimal("0"),
            latency_ms=0.0,
            structured_result=instance,
        )

    def estimate_tokens(self, text: str) -> int:  # pragma: no cover - unused
        return len(text.split())


def _categorize_payload() -> dict[str, Any]:
    """A realistic draft the LLM might return for a categorization goal."""
    return {
        "input_columns": ["description"],
        "output_columns": ["category", "confidence"],
        "system_message": "You are a product categorization expert.",
        "prompt_template": (
            "Categorize this product into the most specific category.\n\n"
            "Product: {description}\n\nCategory:"
        ),
        "response_format": "json",
        "temperature": 0.0,
        "rationale": "Two structured fields because the goal asked for "
        "category and how confident the model is.",
    }


def _single_column_payload() -> dict[str, Any]:
    return {
        "input_columns": ["text"],
        "output_columns": ["summary"],
        "system_message": None,
        "prompt_template": "Summarize in one sentence: {text}",
        "response_format": "raw",
        "temperature": 0.0,
        "rationale": "Single free-text field.",
    }


# ---------------------------------------------------------------------------
# Plan value object
# ---------------------------------------------------------------------------


class TestPlanValueObject:
    def test_plan_holds_specifications_and_original_goal(self):
        """Plan is a container for a drafted spec + provenance, nothing more."""
        specs = PipelineSpecifications.model_validate(
            {
                "dataset": {
                    "source_type": "dataframe",
                    "input_columns": ["text"],
                    "output_columns": ["summary"],
                },
                "prompt": {"template": "Summarize: {text}"},
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                },
            }
        )
        plan_obj = Plan(specifications=specs, goal="summarize", rationale="r")

        assert plan_obj.specifications is specs
        assert plan_obj.goal == "summarize"

    def test_preview_yaml_is_round_trippable(self):
        """preview_yaml() must produce a YAML the ConfigLoader can parse back.

        This is the approval-by-inspection contract: the user reads the YAML
        and either edits it or hands it to build(). If the YAML is not valid
        for the loader, the loop is broken.
        """
        specs = PipelineSpecifications.model_validate(
            {
                "dataset": {
                    "source_type": "dataframe",
                    "input_columns": ["text"],
                    "output_columns": ["summary"],
                },
                "prompt": {"template": "Summarize: {text}"},
                "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            }
        )
        plan_obj = Plan(specifications=specs, goal="summarize", rationale="r")
        yaml_text = plan_obj.preview_yaml()

        import tempfile
        from pathlib import Path

        from ondine.config.config_loader import ConfigLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = Path(f.name)

        loaded = ConfigLoader.from_yaml(path)
        assert loaded.dataset.input_columns == ["text"]
        assert loaded.prompt.template == "Summarize: {text}"


# ---------------------------------------------------------------------------
# plan() — the deep module
# ---------------------------------------------------------------------------


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "description": [
                "Wireless mouse, ergonomic",
                "USB-C charger 65W",
            ],
            "sku": ["M-100", "C-065"],
        }
    )


class TestPlanFunction:
    def test_plan_returns_plan_with_drafted_specs(self):
        """Regression: with a valid LLM draft, plan() builds a Plan whose
        specifications carry the LLM-chosen columns and template, not
        arbitrary defaults."""
        client = _ScriptedClient(spec=None, payload=_categorize_payload())

        plan_obj = plan(
            data=_sample_df(),
            goal="Categorize products and score confidence",
            budget=Decimal("5.0"),
            model="openai/gpt-4o-mini",
            llm_client=client,
        )

        assert isinstance(plan_obj, Plan)
        assert plan_obj.specifications.dataset.input_columns == ["description"]
        assert plan_obj.specifications.dataset.output_columns == [
            "category",
            "confidence",
        ]
        assert "Product: {description}" in plan_obj.specifications.prompt.template
        assert plan_obj.specifications.prompt.response_format == "json"

    def test_plan_attaches_model_and_budget_to_specs(self):
        """The model and budget passed to plan() must end up on the spec,
        not be silently dropped — those are the user's hard constraints."""
        client = _ScriptedClient(spec=None, payload=_categorize_payload())

        plan_obj = plan(
            data=_sample_df(),
            goal="Categorize products",
            budget=Decimal("2.5"),
            model="anthropic/claude-3-5-sonnet",
            llm_client=client,
        )

        assert "claude-3-5-sonnet" in plan_obj.specifications.llm.model
        assert plan_obj.specifications.processing.max_budget == Decimal("2.5")

    def test_plan_sends_sample_and_goal_to_llm(self):
        """The LLM must actually receive the user's goal and a sample of the
        data — otherwise the draft is disconnected from the request."""
        client = _ScriptedClient(spec=None, payload=_categorize_payload())

        plan(
            data=_sample_df(),
            goal="Categorize products and score confidence",
            budget=Decimal("1.0"),
            model="openai/gpt-4o-mini",
            llm_client=client,
        )

        assert client.captured_prompt is not None
        # Goal is communicated verbatim...
        assert "Categorize products and score confidence" in client.captured_prompt
        # ...and a sample of the real data columns/values is included so the
        # model can ground its choices.
        assert "description" in client.captured_prompt
        assert "Wireless mouse" in client.captured_prompt

    def test_plan_rejects_llm_input_columns_not_in_data(self):
        """Defensive contract: if the LLM hallucinates columns that don't
        exist in the data, plan() must fail loudly rather than ship a spec
        that will crash at execution time."""
        bad = _categorize_payload()
        bad["input_columns"] = ["nonexistent_column"]
        client = _ScriptedClient(spec=None, payload=bad)

        with pytest.raises(ValueError, match="input_columns.*data"):
            plan(
                data=_sample_df(),
                goal="Categorize",
                budget=Decimal("1.0"),
                model="openai/gpt-4o-mini",
                llm_client=client,
            )

    def test_plan_rejects_overlapping_input_and_output_columns(self):
        """If the LLM produces overlapping in/out columns, plan() must
        reject it — DatasetSpec forbids the same, and surfacing the error
        here beats a cryptic Pydantic failure later."""
        bad = _categorize_payload()
        bad["output_columns"] = ["description", "confidence"]
        client = _ScriptedClient(spec=None, payload=bad)

        with pytest.raises(ValueError, match="overlap|output_columns"):
            plan(
                data=_sample_df(),
                goal="Categorize",
                budget=Decimal("1.0"),
                model="openai/gpt-4o-mini",
                llm_client=client,
            )

    def test_plan_caps_sample_rows_for_llm_context(self):
        """The sample sent to the LLM must be bounded so a 1M-row DataFrame
        doesn't blow up the drafting prompt. A 10k-row frame should produce
        a prompt mentioning far fewer rows."""
        big = pd.DataFrame({"description": [f"item {i}" for i in range(10_000)]})
        # Payload must reference the real column so the draft validates.
        payload = _single_column_payload()
        payload["input_columns"] = ["description"]
        payload["prompt_template"] = "Summarize: {description}"
        client = _ScriptedClient(spec=None, payload=payload)

        plan(
            data=big,
            goal="summarize",
            budget=Decimal("1.0"),
            model="openai/gpt-4o-mini",
            llm_client=client,
        )

        # 10k rows would be huge; verify the prompt is small.
        assert client.captured_prompt is not None
        # 50 sample rows * ~10 chars + framing -> well under a few KB.
        assert len(client.captured_prompt) < 20_000

    def test_plan_requires_non_empty_goal(self):
        """An empty goal is a programming error, not an LLM task."""
        client = _ScriptedClient(spec=None, payload=_categorize_payload())
        with pytest.raises(ValueError, match="goal"):
            plan(
                data=_sample_df(),
                goal="   ",
                budget=Decimal("1.0"),
                model="openai/gpt-4o-mini",
                llm_client=client,
            )

    def test_plan_requires_positive_budget(self):
        """A non-positive budget cannot buy any LLM call; reject early."""
        client = _ScriptedClient(spec=None, payload=_categorize_payload())
        with pytest.raises(ValueError, match="budget"):
            plan(
                data=_sample_df(),
                goal="Categorize",
                budget=Decimal("0"),
                model="openai/gpt-4o-mini",
                llm_client=client,
            )

    def test_plan_supports_polars_dataframe(self):
        """Data is data: polars frames must work without conversion ceremony."""
        pl = pytest.importorskip("polars")
        client = _ScriptedClient(spec=None, payload=_categorize_payload())

        # Build the polars frame directly — converting a pandas frame with
        # object-dtype columns would require pyarrow, which we don't depend
        # on here.
        plan_obj = plan(
            data=pl.DataFrame(
                {
                    "description": ["Wireless mouse", "USB-C charger"],
                    "sku": ["M-100", "C-065"],
                }
            ),
            goal="Categorize",
            budget=Decimal("1.0"),
            model="openai/gpt-4o-mini",
            llm_client=client,
        )

        assert plan_obj.specifications.dataset.input_columns == ["description"]


# ---------------------------------------------------------------------------
# Plan.build()
# ---------------------------------------------------------------------------


class TestPlanBuild:
    def test_build_returns_real_pipeline(self):
        """build() must produce a real ondine.Pipeline wired from the drafted
        spec — this is the 'Generates PipelineSpecifications ONLY' contract:
        there is no separate execution path, the planner just feeds the
        existing one."""
        from ondine.api.pipeline import Pipeline

        client = _ScriptedClient(spec=None, payload=_categorize_payload())

        plan_obj = plan(
            data=_sample_df(),
            goal="Categorize",
            budget=Decimal("1.0"),
            model="openai/gpt-4o-mini",
            llm_client=client,
        )

        pipeline = plan_obj.build()
        assert isinstance(pipeline, Pipeline)
        # The drafted columns flow through to the built pipeline.
        assert pipeline.specifications.dataset.output_columns == [
            "category",
            "confidence",
        ]


# ---------------------------------------------------------------------------
# Plan.estimated_cost
# ---------------------------------------------------------------------------


class TestPlanEstimatedCost:
    def test_plan_surfaces_a_cost_estimate(self):
        """A Plan must expose a projected cost — this is the whole safety
        story of plan(): the user inspects projected spend before ever
        calling build(). No network call should occur; the estimate is
        derived from the drafted spec + a token estimate, reusing
        Pipeline.estimate_cost() (the same estimator the rest of Ondine
        already uses), not a second cost model or an extra LLM call."""
        from ondine.core.models import CostEstimate

        client = _ScriptedClient(spec=None, payload=_categorize_payload())

        plan_obj = plan(
            data=_sample_df(),
            goal="Categorize products and score confidence",
            budget=Decimal("5.0"),
            model="openai/gpt-4o-mini",
            llm_client=client,
        )

        estimate = plan_obj.estimated_cost

        assert isinstance(estimate, CostEstimate)
        assert estimate.total_cost >= 0
        assert estimate.rows == len(_sample_df())
        # The scripted drafting client only implements structured_invoke();
        # estimated_cost must not have called back into it for anything.
        assert client.captured_prompt is not None  # sanity: drafting did run

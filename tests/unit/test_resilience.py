"""
Unit tests for circuit breaker / resilience configuration.

Tests that resilience parameters flow correctly through the pipeline builder
and that cooldown events are properly emitted.
"""

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.observability.base import PipelineObserver
from ondine.observability.dispatcher import ObserverDispatcher
from ondine.observability.events import (
    ProviderCooldownEvent,
    ProviderRecoveredEvent,
)


class TestResilienceConfig:
    """Tests for resilience configuration in PipelineBuilder."""

    def test_default_resilience_params(self):
        """Test that default resilience params are set correctly."""
        df = pd.DataFrame({"text": ["test"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Test: {text}")
            .with_router(
                model_list=[
                    {
                        "model_name": "test",
                        "litellm_params": {"model": "openai/gpt-4o-mini"},
                    }
                ]
            )
            .build()
        )

        # Check that default resilience params are in router_config
        router_config = pipeline.specifications.llm.router_config
        assert router_config["allowed_fails"] == 3  # Default
        assert router_config["cooldown_time"] == 60  # Default

    def test_custom_resilience_params(self):
        """Test that custom resilience params override defaults."""
        df = pd.DataFrame({"text": ["test"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Test: {text}")
            .with_router(
                model_list=[
                    {
                        "model_name": "test",
                        "litellm_params": {"model": "openai/gpt-4o-mini"},
                    }
                ],
                allowed_fails=5,
                cooldown_time=120,
            )
            .build()
        )

        router_config = pipeline.specifications.llm.router_config
        assert router_config["allowed_fails"] == 5
        assert router_config["cooldown_time"] == 120

    def test_disable_circuit_breaker(self):
        """Test that circuit breaker can be disabled."""
        df = pd.DataFrame({"text": ["test"]})

        pipeline = (
            PipelineBuilder.create()
            .from_dataframe(df, input_columns=["text"], output_columns=["result"])
            .with_prompt("Test: {text}")
            .with_router(
                model_list=[
                    {
                        "model_name": "test",
                        "litellm_params": {"model": "openai/gpt-4o-mini"},
                    }
                ],
                allowed_fails=0,  # Disable cooldowns
                cooldown_time=0,
            )
            .build()
        )

        router_config = pipeline.specifications.llm.router_config
        assert router_config["allowed_fails"] == 0
        assert router_config["cooldown_time"] == 0


class TestCooldownEvents:
    """Tests for cooldown event emission."""

    def test_provider_cooldown_event_structure(self):
        """Test that ProviderCooldownEvent has correct structure."""
        event = ProviderCooldownEvent(
            pipeline_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            timestamp=datetime.now(),
            trace_id="trace-123",
            span_id="span-456",
            provider="groq/llama-3.3-70b",
            deployment_id="deployment-abc",
            reason="Rate limit exceeded",
            cooldown_duration=60,
            fail_count=3,
        )

        assert event.provider == "groq/llama-3.3-70b"
        assert event.deployment_id == "deployment-abc"
        assert event.reason == "Rate limit exceeded"
        assert event.cooldown_duration == 60
        assert event.fail_count == 3

    def test_provider_recovered_event_structure(self):
        """Test that ProviderRecoveredEvent has correct structure."""
        event = ProviderRecoveredEvent(
            pipeline_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            timestamp=datetime.now(),
            trace_id="trace-123",
            span_id="span-456",
            provider="groq/llama-3.3-70b",
            deployment_id="deployment-abc",
            cooldown_duration=60,
        )

        assert event.provider == "groq/llama-3.3-70b"
        assert event.deployment_id == "deployment-abc"
        assert event.cooldown_duration == 60


class TestCooldownObserverDispatch:
    """Tests for cooldown event dispatch to observers."""

    def test_dispatcher_calls_cooldown_handler(self):
        """Test that dispatcher correctly routes cooldown events."""
        # Create mock observer
        mock_observer = MagicMock(spec=PipelineObserver)
        mock_observer.on_provider_cooldown = MagicMock()

        dispatcher = ObserverDispatcher([mock_observer])

        event = ProviderCooldownEvent(
            pipeline_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            timestamp=datetime.now(),
            trace_id="trace-123",
            span_id="span-456",
            provider="groq/llama-3.3-70b",
            deployment_id="deployment-abc",
            reason="Rate limit exceeded",
            cooldown_duration=60,
            fail_count=3,
        )

        dispatcher.dispatch("provider_cooldown", event)

        mock_observer.on_provider_cooldown.assert_called_once_with(event)

    def test_dispatcher_calls_recovered_handler(self):
        """Test that dispatcher correctly routes recovered events."""
        mock_observer = MagicMock(spec=PipelineObserver)
        mock_observer.on_provider_recovered = MagicMock()

        dispatcher = ObserverDispatcher([mock_observer])

        event = ProviderRecoveredEvent(
            pipeline_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            timestamp=datetime.now(),
            trace_id="trace-123",
            span_id="span-456",
            provider="groq/llama-3.3-70b",
            deployment_id="deployment-abc",
            cooldown_duration=60,
        )

        dispatcher.dispatch("provider_recovered", event)

        mock_observer.on_provider_recovered.assert_called_once_with(event)

    def test_dispatcher_handles_missing_handler(self):
        """Test that dispatcher gracefully handles observers without cooldown handlers."""
        # Create observer without cooldown handler
        mock_observer = MagicMock(spec=PipelineObserver)
        # Remove the cooldown handler
        del mock_observer.on_provider_cooldown

        dispatcher = ObserverDispatcher([mock_observer])

        event = ProviderCooldownEvent(
            pipeline_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            timestamp=datetime.now(),
            trace_id="trace-123",
            span_id="span-456",
            provider="groq/llama-3.3-70b",
            deployment_id="deployment-abc",
            reason="Rate limit exceeded",
            cooldown_duration=60,
            fail_count=3,
        )

        # Should not raise
        dispatcher.dispatch("provider_cooldown", event)

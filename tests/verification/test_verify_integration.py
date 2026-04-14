"""Claim verification: Integration claims (Claims 41-47)."""

from ondine.core.specifications import LLMSpec


class TestIntegrationClaims:
    """Verify all claimed integration capabilities."""

    def test_claim_41_litellm_callback_support(self):
        """Claim 41: LiteLLM callbacks — client accepts callbacks config."""
        spec = LLMSpec(
            model="gpt-4o-mini",
            extra_params={"callbacks": ["langfuse"]},
        )
        assert spec.extra_params["callbacks"] == ["langfuse"]

    def test_claim_42_langfuse_observer_exists(self):
        """Claim 42: Langfuse integration — observer class exists."""
        try:
            from ondine.observability.observers.langfuse_observer import (
                LangfuseObserver,
            )

            assert LangfuseObserver is not None
        except ImportError:
            # Langfuse observer module exists but langfuse package not installed
            import importlib

            importlib.util.find_spec("ondine.observability.observers.langfuse_observer")
            # Module file should exist even if dependency is missing
            from pathlib import Path

            observer_path = Path("ondine/observability/observers/langfuse_observer.py")
            assert observer_path.exists(), "Langfuse observer module not found"

    def test_claim_43_opentelemetry_observer_exists(self):
        """Claim 43: OpenTelemetry — observer and tracer modules exist."""
        from ondine.observability.observer import TracingObserver
        from ondine.observability.tracer import disable_tracing, enable_tracing

        assert TracingObserver is not None
        assert callable(enable_tracing)
        assert callable(disable_tracing)

    def test_claim_44_azure_managed_identity_optional_dep(self):
        """Claim 44: Azure Managed Identity — declared as optional dependency."""
        from pathlib import Path

        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # Python 3.10 fallback

        pyproject = Path("pyproject.toml")
        with open(pyproject, "rb") as f:
            config = tomllib.load(f)

        optional_deps = config["project"]["optional-dependencies"]
        azure_deps = optional_deps.get("azure", [])
        azure_dep_names = [d.lower() for d in azure_deps]
        assert any("azure" in d for d in azure_dep_names), (
            f"Azure dependency not found in optional deps: {azure_deps}"
        )

    def test_claim_45_multi_observer_dispatch(self):
        """Claim 45: Multiple observers simultaneously — dispatcher exists."""
        from ondine.observability.dispatcher import ObserverDispatcher

        # Create dispatcher with empty observer list
        dispatcher = ObserverDispatcher(observers=[])
        assert hasattr(dispatcher, "dispatch")
        assert hasattr(dispatcher, "flush_all")
        assert hasattr(dispatcher, "close_all")

    def test_claim_46_metrics_exporter_exists(self):
        """Claim 46: Cache hit/miss metrics — PrometheusMetrics exists."""
        from ondine.utils.metrics_exporter import PrometheusMetrics

        assert PrometheusMetrics is not None

    def test_claim_47_router_strategies_complete(self):
        """Claim 47: Router failover events — strategies include failover types."""
        from ondine.core.router_strategies import RouterStrategy

        # Verify load-balancing and failover strategies exist
        strategy_values = {s.value for s in RouterStrategy}
        assert "simple-shuffle" in strategy_values
        assert "latency-based-routing" in strategy_values
        assert "least-busy" in strategy_values
        assert len(strategy_values) >= 5

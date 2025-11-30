"""Tests for DeploymentTracker."""

import pytest

from ondine.orchestration.deployment_tracker import DeploymentInfo, DeploymentTracker


class TestDeploymentTracker:
    """Test DeploymentTracker functionality."""

    def test_init_empty(self):
        """Test initialization without model list."""
        tracker = DeploymentTracker()
        assert tracker.total_requests == 0
        assert tracker.get_distribution() == {}

    def test_init_with_model_list(self):
        """Test initialization with model list."""
        model_list = [
            {
                "model_name": "fast-model",
                "model_id": "groq-1",
                "litellm_params": {"model": "groq/llama-3.3-70b"},
            },
            {
                "model_name": "fast-model",
                "model_id": "cerebras-1",
                "litellm_params": {"model": "cerebras/llama-3.3-70b"},
            },
        ]
        tracker = DeploymentTracker(model_list)

        # Should have prepared friendly names
        deployments = tracker.get_deployments_for_progress()
        assert len(deployments) == 2
        assert deployments[0]["model_id"] == "groq-1_0"
        assert deployments[1]["model_id"] == "cerebras-1_1"

    def test_register_deployment_first_seen(self):
        """Test First-Seen-First-Assigned mapping."""
        model_list = [
            {"model_id": "provider-a", "litellm_params": {"model": "groq/model-a"}},
            {"model_id": "provider-b", "litellm_params": {"model": "groq/model-b"}},
        ]
        tracker = DeploymentTracker(model_list)

        # First unknown hash gets first friendly name
        friendly_id = tracker.register_deployment("hash-xyz-123")
        assert friendly_id == "provider-a_0"

        # Same hash returns same friendly ID
        assert tracker.register_deployment("hash-xyz-123") == "provider-a_0"

        # Second unknown hash gets second friendly name
        friendly_id2 = tracker.register_deployment("hash-abc-456")
        assert friendly_id2 == "provider-b_1"

    def test_register_deployment_no_names_available(self):
        """Test registration when no friendly names available."""
        tracker = DeploymentTracker()  # No model list

        # Should return hash as-is
        result = tracker.register_deployment("some-hash")
        assert result == "some-hash"

    def test_get_friendly_id(self):
        """Test getting friendly ID without registering."""
        model_list = [
            {"model_id": "test-model", "litellm_params": {"model": "test/model"}},
        ]
        tracker = DeploymentTracker(model_list)

        # Not registered yet
        assert tracker.get_friendly_id("unknown-hash") is None

        # Register it
        tracker.register_deployment("unknown-hash")

        # Now should return friendly ID
        assert tracker.get_friendly_id("unknown-hash") == "test-model_0"

    def test_get_label(self):
        """Test getting display label."""
        model_list = [
            {"model_id": "my-model", "litellm_params": {"model": "groq/llama-3.3"}},
        ]
        tracker = DeploymentTracker(model_list)

        # Register
        tracker.register_deployment("hash-123")

        # Should return label
        label = tracker.get_label("hash-123")
        assert "groq" in label
        assert "llama-3.3" in label

    def test_record_request(self):
        """Test recording requests."""
        tracker = DeploymentTracker()

        tracker.record_request("deployment-a")
        tracker.record_request("deployment-a")
        tracker.record_request("deployment-b")

        distribution = tracker.get_distribution()
        assert distribution["deployment-a"] == 2
        assert distribution["deployment-b"] == 1
        assert tracker.total_requests == 3

    def test_get_distribution_summary(self):
        """Test distribution summary with percentages."""
        model_list = [
            {"model_id": "fast", "litellm_params": {"model": "groq/model"}},
            {"model_id": "slow", "litellm_params": {"model": "openai/model"}},
        ]
        tracker = DeploymentTracker(model_list)

        # Register and record
        tracker.register_deployment("hash-a")
        tracker.register_deployment("hash-b")

        for _ in range(3):
            tracker.record_request("hash-a")
        tracker.record_request("hash-b")

        summary = tracker.get_distribution_summary()

        assert summary["hash-a"]["count"] == 3
        assert summary["hash-a"]["percentage"] == 75.0
        assert summary["hash-b"]["count"] == 1
        assert summary["hash-b"]["percentage"] == 25.0

    def test_repr(self):
        """Test string representation."""
        model_list = [
            {"model_id": "test", "litellm_params": {"model": "test/model"}},
        ]
        tracker = DeploymentTracker(model_list)
        tracker.register_deployment("hash-123")
        tracker.record_request("hash-123")

        repr_str = repr(tracker)
        assert "DeploymentTracker" in repr_str
        assert "deployments=1" in repr_str
        assert "total_requests=1" in repr_str


class TestDeploymentInfo:
    """Test DeploymentInfo dataclass."""

    def test_create_deployment_info(self):
        """Test creating DeploymentInfo."""
        info = DeploymentInfo(
            id="test-id",
            label="Test Label",
            model="groq/llama",
            provider="groq",
        )
        assert info.id == "test-id"
        assert info.label == "Test Label"
        assert info.model == "groq/llama"
        assert info.provider == "groq"


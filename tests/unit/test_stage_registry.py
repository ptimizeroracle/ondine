"""
Unit tests for StageRegistry.

Tests the plugin system for custom pipeline stages.
"""

import pytest

from ondine.stages.pipeline_stage import PipelineStage
from ondine.stages.stage_registry import StageRegistry, stage


class TestStageRegistry:
    """Test suite for StageRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        StageRegistry._reset()

    def test_register_custom_stage(self):
        """Should register custom stage successfully."""

        class CustomStage(PipelineStage):
            def execute(self, context):
                return {"success": True}

        # Register
        StageRegistry.register("custom_stage", CustomStage)

        # Verify registered
        assert StageRegistry.is_registered("custom_stage")
        assert "custom_stage" in StageRegistry.list_stages()

    def test_register_duplicate_raises_error(self):
        """Should raise error when registering duplicate stage name."""

        class CustomStage(PipelineStage):
            pass

        StageRegistry.register("duplicate", CustomStage)

        with pytest.raises(ValueError, match="already registered"):
            StageRegistry.register("duplicate", CustomStage)

    def test_get_registered_stage(self):
        """Should retrieve registered stage class."""

        class CustomStage(PipelineStage):
            pass

        StageRegistry.register("test_stage", CustomStage)

        retrieved_class = StageRegistry.get("test_stage")
        assert retrieved_class is CustomStage

    def test_get_unregistered_stage_raises_error(self):
        """Should raise error for unregistered stage."""
        with pytest.raises(ValueError, match="Unknown stage.*nonexistent"):
            StageRegistry.get("nonexistent")

    def test_list_stages_returns_all_registered(self):
        """Should list all registered stages."""

        class Stage1(PipelineStage):
            pass

        class Stage2(PipelineStage):
            pass

        StageRegistry.register("stage1", Stage1)
        StageRegistry.register("stage2", Stage2)

        stages = StageRegistry.list_stages()

        assert "stage1" in stages
        assert "stage2" in stages
        assert stages["stage1"] is Stage1
        assert stages["stage2"] is Stage2

    def test_list_stages_empty_by_default(self):
        """Should return empty dict when no stages registered."""
        stages = StageRegistry.list_stages()
        assert stages == {}

    def test_unregister_stage(self):
        """Should unregister stage successfully."""

        class CustomStage(PipelineStage):
            pass

        StageRegistry.register("to_remove", CustomStage)
        assert StageRegistry.is_registered("to_remove")

        StageRegistry.unregister("to_remove")
        assert not StageRegistry.is_registered("to_remove")

    def test_unregister_nonexistent_raises_error(self):
        """Should raise error when unregistering non-existent stage."""
        with pytest.raises(ValueError, match="not registered"):
            StageRegistry.unregister("nonexistent")

    def test_stage_decorator(self):
        """Should register stage via decorator."""

        @stage("decorated_stage")
        class DecoratedStage(PipelineStage):
            def execute(self, context):
                return {"decorated": True}

        assert StageRegistry.is_registered("decorated_stage")
        assert StageRegistry.get("decorated_stage") is DecoratedStage

    def test_stage_decorator_returns_class(self):
        """Should return decorated class unchanged (enables inheritance)."""

        @stage("test")
        class TestStage(PipelineStage):
            custom_attr = "test"

        # Should have access to original class attributes
        assert TestStage.custom_attr == "test"

    def test_create_instance_from_registry(self):
        """Should create working instances from registered stages."""
        from ondine.stages.stage_registry import stage as stage_decorator

        @stage_decorator("test_instance")
        class TestStage(PipelineStage):
            def __init__(self, param1: str, param2: int = 10):
                super().__init__(name="test_instance")
                self.param1 = param1
                self.param2 = param2

            def execute(self, context):
                return {"param1": self.param1, "param2": self.param2}

            def process(self, data):
                return data

            def validate_input(self, data):
                return True

            def estimate_cost(self, data):
                return 0.0

        # Get class and instantiate
        stage_class = StageRegistry.get("test_instance")
        stage_instance = stage_class(param1="value", param2=20)

        # Verify it works
        assert stage_instance.param1 == "value"
        assert stage_instance.param2 == 20
        result = stage_instance.execute(None)
        assert result["param1"] == "value"
        assert result["param2"] == 20

    def test_registry_isolation_between_tests(self):
        """Should maintain registry state across operations."""

        @stage("test1")
        class Test1(PipelineStage):
            pass

        @stage("test2")
        class Test2(PipelineStage):
            pass

        stages = StageRegistry.list_stages()
        assert "test1" in stages
        assert "test2" in stages

    def test_register_with_same_class_different_names(self):
        """Should allow registering same class under different names."""

        class SharedStage(PipelineStage):
            pass

        StageRegistry.register("stage_a", SharedStage)
        StageRegistry.register("stage_b", SharedStage)

        assert StageRegistry.get("stage_a") is SharedStage
        assert StageRegistry.get("stage_b") is SharedStage

    def test_reset_clears_all_stages(self):
        """Should clear all stages on reset."""

        @stage("custom")
        class CustomStage(PipelineStage):
            pass

        assert StageRegistry.is_registered("custom")

        # Reset
        StageRegistry._reset()

        # Should be empty
        assert len(StageRegistry._stages) == 0
        assert not StageRegistry.is_registered("custom")

    def test_stage_with_complex_initialization(self):
        """Should support stages with complex initialization."""
        from ondine.stages.stage_registry import stage as stage_decorator

        @stage_decorator("complex_stage")
        class ComplexStage(PipelineStage):
            def __init__(
                self,
                vector_store: str,
                index_name: str,
                top_k: int = 3,
                filters: dict | None = None,
            ):
                super().__init__(name="complex_stage")
                self.vector_store = vector_store
                self.index_name = index_name
                self.top_k = top_k
                self.filters = filters or {}

            def execute(self, context):
                return {
                    "vector_store": self.vector_store,
                    "index_name": self.index_name,
                    "top_k": self.top_k,
                    "filters": self.filters,
                }

            def process(self, data):
                return data

            def validate_input(self, data):
                return True

            def estimate_cost(self, data):
                return 0.0

        # Instantiate with complex params
        stage_class = StageRegistry.get("complex_stage")
        stage_instance = stage_class(
            vector_store="pinecone",
            index_name="my-docs",
            top_k=5,
            filters={"category": "tech"},
        )

        result = stage_instance.execute(None)
        assert result["vector_store"] == "pinecone"
        assert result["index_name"] == "my-docs"
        assert result["top_k"] == 5
        assert result["filters"] == {"category": "tech"}

    def test_is_registered_false_for_unregistered(self):
        """Should return False for unregistered stages."""
        assert not StageRegistry.is_registered("nonexistent")

    def test_list_stages_returns_copy(self):
        """Should return copy of registry (mutations don't affect internal state)."""

        @stage("test")
        class TestStage(PipelineStage):
            pass

        stages = StageRegistry.list_stages()
        stages["fake"] = object()

        # Internal registry should be unchanged
        assert "fake" not in StageRegistry.list_stages()

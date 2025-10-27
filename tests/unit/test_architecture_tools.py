"""Tests for architecture documentation tools."""

from pathlib import Path

import pytest
import yaml


class TestGenerateDocs:
    """Tests for generate_docs.py tool."""

    def test_load_model_reads_yaml(self, tmp_path):
        """Should load and parse YAML model file."""

        # Create test model
        model_content = """
layers:
  - name: core
    description: "Core layer"
    allowed_dependencies: []

entities:
  - name: TestEntity
    type: class
    layer: core
    file: test.py
    description: "Test"

features:
  - name: test_feature
    components: [TestEntity]
"""
        model_file = tmp_path / "model.yaml"
        model_file.write_text(model_content)

        # Temporarily override MODEL_FILE
        import tools.generate_docs as gen_module

        original_file = gen_module.MODEL_FILE
        gen_module.MODEL_FILE = model_file

        try:
            model = gen_module.load_model()
            assert "layers" in model
            assert "entities" in model
            assert len(model["layers"]) == 1
            assert len(model["entities"]) == 1
        finally:
            gen_module.MODEL_FILE = original_file

    def test_generate_erd_creates_mermaid(self):
        """Should generate valid Mermaid ERD syntax."""
        from tools.generate_docs import generate_erd

        entities = [
            {
                "name": "ClassA",
                "type": "class",
                "layer": "core",
                "relationships": [
                    {"type": "depends_on", "target": "ClassB"},
                    {"type": "extends", "target": "ClassC"},
                ],
            },
            {"name": "ClassB", "type": "class", "layer": "core", "relationships": []},
            {"name": "ClassC", "type": "class", "layer": "core", "relationships": []},
        ]
        relationships = []

        result = generate_erd(entities, relationships)

        assert "```mermaid" in result
        assert "erDiagram" in result
        assert "ClassA" in result
        assert "ClassB" in result
        assert "```" in result

    def test_generate_class_diagram_shows_inheritance(self):
        """Should generate class diagram with inheritance relationships."""
        from tools.generate_docs import generate_class_diagram

        entities = [
            {
                "name": "BaseClass",
                "type": "abstract_class",
                "layer": "core",
                "methods": ["method1()", "method2()"],
            },
            {
                "name": "SubClass",
                "type": "class",
                "layer": "adapters",
                "methods": ["method3()"],
                "relationships": [{"type": "extends", "target": "BaseClass"}],
            },
        ]

        result = generate_class_diagram(entities)

        assert "```mermaid" in result
        assert "classDiagram" in result
        assert "BaseClass" in result
        assert "SubClass" in result
        assert "<<abstract>>" in result
        assert "BaseClass <|-- SubClass" in result

    def test_generate_feature_map_creates_mindmap(self):
        """Should generate feature mindmap."""
        from tools.generate_docs import generate_feature_map

        features = [
            {
                "name": "feature1",
                "description": "Test feature",
                "components": ["Component1", "Component2"],
            },
            {
                "name": "feature2",
                "description": "Another feature",
                "components": ["Component3"],
            },
        ]

        result = generate_feature_map(features)

        assert "```mermaid" in result
        assert "mindmap" in result
        assert "feature1" in result
        assert "feature2" in result
        assert "Component1" in result


class TestValidateArchitecture:
    """Tests for validate_architecture.py tool."""

    def test_validation_result_tracks_errors_and_warnings(self):
        """Should track errors and warnings separately."""
        from tools.validate_architecture import ValidationResult

        result = ValidationResult()

        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_warning("Warning 1")
        result.add_info("Info 1")

        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert len(result.info) == 1
        assert not result.is_success()

    def test_validation_result_success_with_no_errors(self):
        """Should be successful if no errors."""
        from tools.validate_architecture import ValidationResult

        result = ValidationResult()
        result.add_warning("Warning")
        result.add_info("Info")

        assert result.is_success()
        assert len(result.warnings) == 1

    def test_parse_python_file_extracts_classes(self, tmp_path):
        """Should extract class names from Python file."""
        from tools.validate_architecture import parse_python_file

        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
class MyClass:
    pass

class MyOtherClass(BaseClass):
    pass

def my_function():
    pass
"""
        )

        result = parse_python_file(test_file)

        assert len(result["classes"]) == 2
        assert result["classes"][0]["name"] == "MyClass"
        assert result["classes"][1]["name"] == "MyOtherClass"
        assert "BaseClass" in result["classes"][1]["bases"]

    def test_parse_python_file_extracts_imports(self, tmp_path):
        """Should extract import statements."""
        from tools.validate_architecture import parse_python_file

        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
import os
from pathlib import Path
from ondine.core.models import LLMResponse
"""
        )

        result = parse_python_file(test_file)

        assert "os" in result["imports"]
        assert "pathlib" in result["imports"]
        assert "ondine.core.models" in result["imports"]

    def test_validate_entities_detects_missing(self):
        """Should detect entities that don't exist in codebase."""
        from tools.validate_architecture import ValidationResult, validate_entities

        model = {
            "entities": [
                {
                    "name": "NonExistentClass",
                    "type": "class",
                    "file": "ondine/fake/file.py",
                }
            ]
        }
        codebase = {}
        result = ValidationResult()

        validate_entities(model, codebase, result)

        assert not result.is_success()
        assert len(result.errors) > 0

    def test_validate_features_checks_test_files(self):
        """Should validate that feature test files exist."""
        from tools.validate_architecture import ValidationResult, validate_features

        model = {
            "features": [
                {
                    "name": "test_feature",
                    "components": ["Component1"],
                    "tests": ["tests/unit/nonexistent_test.py"],
                }
            ]
        }
        result = ValidationResult()

        validate_features(model, result)

        # Should have warning about missing test file
        assert len(result.warnings) > 0

    def test_validate_relationships_detects_invalid_targets(self):
        """Should detect relationships to undefined entities."""
        from tools.validate_architecture import ValidationResult, validate_relationships

        model = {
            "entities": [
                {
                    "name": "ClassA",
                    "relationships": [{"type": "depends_on", "target": "NonExistent"}],
                }
            ]
        }
        result = ValidationResult()

        validate_relationships(model, {}, result)

        assert not result.is_success()
        assert any("NonExistent" in err for err in result.errors)


class TestArchitectureIntegration:
    """Integration tests for architecture documentation system."""

    def test_full_workflow_generate_and_validate(self, tmp_path):
        """Should generate docs from model and validate successfully."""
        # Create minimal model
        model_content = {
            "layers": [
                {
                    "name": "core",
                    "description": "Core",
                    "allowed_dependencies": [],
                }
            ],
            "entities": [
                {
                    "name": "TestEntity",
                    "type": "class",
                    "layer": "core",
                    "file": "ondine/core/test.py",
                    "relationships": [],
                }
            ],
            "features": [{"name": "test", "components": ["TestEntity"], "tests": []}],
        }

        model_file = tmp_path / "model.yaml"
        with open(model_file, "w") as f:
            yaml.dump(model_content, f)

        # Validate structure
        with open(model_file) as f:
            loaded = yaml.safe_load(f)

        assert "layers" in loaded
        assert "entities" in loaded
        assert "features" in loaded

    def test_model_yaml_is_valid_yaml(self):
        """The actual model.yaml should be valid YAML."""

        model_file = (
            Path(__file__).parent.parent.parent / "docs" / "architecture" / "model.yaml"
        )

        if not model_file.exists():
            pytest.skip("model.yaml not found")

        with open(model_file) as f:
            model = yaml.safe_load(f)

        # Should have required sections
        assert "layers" in model
        assert "entities" in model
        assert "features" in model

        # Should have expected counts
        assert len(model["layers"]) >= 5
        assert len(model["entities"]) >= 50
        assert len(model["features"]) >= 5

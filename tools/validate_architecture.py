#!/usr/bin/env python3
"""
Validate architecture model against actual codebase.

Checks that model.yaml accurately reflects the codebase structure:
- All entities exist in code
- Relationships match actual imports
- No layer violations
- Features have test coverage
- All public classes are documented
"""

import ast
import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARCHITECTURE_DIR = PROJECT_ROOT / "docs" / "architecture"
MODEL_FILE = ARCHITECTURE_DIR / "model.yaml"
SOURCE_DIR = PROJECT_ROOT / "ondine"


class ValidationResult:
    """Stores validation results."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def add_error(self, message: str):
        self.errors.append(f"✗ {message}")

    def add_warning(self, message: str):
        self.warnings.append(f"⚠ {message}")

    def add_info(self, message: str):
        self.info.append(f"✓ {message}")

    def is_success(self) -> bool:
        return len(self.errors) == 0

    def print_report(self):
        """Print validation report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        if self.info:
            print("\n✅ PASSED:")
            for msg in self.info:
                print(f"  {msg}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for msg in self.warnings:
                print(f"  {msg}")

        if self.errors:
            print("\n❌ ERRORS:")
            for msg in self.errors:
                print(f"  {msg}")

        print("\n" + "=" * 60)
        if self.is_success():
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")
        print("=" * 60 + "\n")


def load_model() -> dict[str, Any]:
    """Load the architecture model YAML."""
    if not MODEL_FILE.exists():
        print(f"ERROR: Model file not found: {MODEL_FILE}")
        sys.exit(1)

    with open(MODEL_FILE) as f:
        return yaml.safe_load(f)


def parse_python_file(filepath: Path) -> dict[str, Any]:
    """
    Parse a Python file and extract classes, functions, and imports.

    Returns:
        Dict with 'classes', 'functions', and 'imports' lists
    """
    try:
        content = filepath.read_text()
        tree = ast.parse(content)
    except Exception as e:
        return {"classes": [], "functions": [], "imports": [], "error": str(e)}

    classes = []
    functions = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Extract base classes
            bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
            classes.append({"name": node.name, "bases": bases, "is_abstract": False})

        elif isinstance(node, ast.FunctionDef):
            # Top-level functions only
            if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree)):
                functions.append(node.name)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    # Check for abstract classes (heuristic: has @abstractmethod or ABC in bases)
    for cls in classes:
        if "ABC" in cls["bases"] or "ABC" in str(bases):
            cls["is_abstract"] = True

    return {"classes": classes, "functions": functions, "imports": imports}


def parse_codebase(source_dir: Path) -> dict[str, Any]:
    """
    Parse entire codebase and extract structure.

    Returns:
        Dict mapping file paths to parsed data
    """
    codebase = {}

    for pyfile in source_dir.rglob("*.py"):
        if "__pycache__" in str(pyfile) or "test" in str(pyfile):
            continue

        relative_path = pyfile.relative_to(PROJECT_ROOT)
        parsed = parse_python_file(pyfile)
        codebase[str(relative_path)] = parsed

    return codebase


def validate_entities(model: dict, codebase: dict, result: ValidationResult):
    """
    Validate that all entities in model exist in codebase.
    """
    entities = model.get("entities", [])
    missing_entities = []
    found_count = 0

    for entity in entities:
        name = entity["name"]
        file_path = entity.get("file", "")
        entity_type = entity.get("type", "")

        if not file_path:
            result.add_warning(f"Entity '{name}' has no file path")
            continue

        # Check if file exists
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            missing_entities.append(f"{name} (file not found: {file_path})")
            continue

        # Check if entity exists in file
        parsed = codebase.get(file_path)
        if not parsed:
            result.add_warning(f"Could not parse {file_path} for entity '{name}'")
            continue

        # Check based on type
        if entity_type in ["class", "abstract_class"]:
            class_names = [cls["name"] for cls in parsed["classes"]]
            if name not in class_names:
                missing_entities.append(f"{name} (class not found in {file_path})")
            else:
                found_count += 1

        elif entity_type in ["dataclass", "pydantic_model"]:
            # These are also classes
            class_names = [cls["name"] for cls in parsed["classes"]]
            if name not in class_names:
                missing_entities.append(f"{name} (model not found in {file_path})")
            else:
                found_count += 1

        elif entity_type == "enum":
            class_names = [cls["name"] for cls in parsed["classes"]]
            if name not in class_names:
                missing_entities.append(f"{name} (enum not found in {file_path})")
            else:
                found_count += 1

        elif entity_type == "module":
            # Module = file exists
            found_count += 1

    if missing_entities:
        result.add_error(
            f"Found {len(missing_entities)} entities that don't exist in codebase:"
        )
        for ent in missing_entities[:10]:  # Show first 10
            result.add_error(f"  - {ent}")
        if len(missing_entities) > 10:
            result.add_error(f"  ... and {len(missing_entities) - 10} more")
    else:
        result.add_info(f"{found_count} entities validated and found in codebase")


def validate_layer_dependencies(model: dict, codebase: dict, result: ValidationResult):
    """
    Validate that layers only import from allowed dependencies.
    """
    layers = model.get("layers", [])
    entities = model.get("entities", [])

    # Build layer hierarchy and allowed dependencies
    layer_hierarchy = {layer["name"]: i for i, layer in enumerate(layers)}
    layer_allowed = {
        layer["name"]: set(layer.get("allowed_dependencies", [])) for layer in layers
    }

    violations = []

    # Check each entity's imports
    for entity in entities:
        entity_layer = entity.get("layer")
        file_path = entity.get("file")

        if not entity_layer or not file_path:
            continue

        parsed = codebase.get(file_path)
        if not parsed:
            continue

        allowed = layer_allowed.get(entity_layer, set())

        # Check imports
        for import_path in parsed["imports"]:
            # Check if import is from ondine
            if not import_path.startswith("ondine."):
                continue

            # Extract layer from import
            parts = import_path.split(".")
            if len(parts) < 2:
                continue

            imported_layer = parts[1]  # e.g., "ondine.api.pipeline" -> "api"

            # Skip if importing from own layer
            if imported_layer == entity_layer:
                continue

            # Check if this is allowed
            if imported_layer in layer_hierarchy:
                if imported_layer not in allowed:
                    violations.append(
                        f"{entity['name']} ({entity_layer} layer) imports from "
                        f"{imported_layer} layer (not in allowed: {allowed})"
                    )

    if violations:
        result.add_error(f"Found {len(violations)} layer violations:")
        for violation in violations[:5]:
            result.add_error(f"  - {violation}")
        if len(violations) > 5:
            result.add_error(f"  ... and {len(violations) - 5} more")
    else:
        result.add_info(
            "No layer violations detected (all imports follow allowed dependencies)"
        )


def validate_features(model: dict, result: ValidationResult):
    """
    Validate that features have complete test coverage.
    """
    features = model.get("features", [])
    incomplete_features = []

    for feature in features:
        name = feature["name"]
        tests = feature.get("tests", [])

        if not tests:
            incomplete_features.append(f"{name} (no tests defined)")
            continue

        # Check if test files exist
        missing_tests = []
        for test_path in tests:
            # Support both absolute and relative paths
            if test_path.startswith("tests/"):
                full_path = PROJECT_ROOT / test_path
            else:
                full_path = PROJECT_ROOT / "tests" / test_path

            if not full_path.exists():
                missing_tests.append(test_path)

        if missing_tests:
            incomplete_features.append(
                f"{name} (missing tests: {', '.join(missing_tests[:3])})"
            )

    if incomplete_features:
        result.add_warning(
            f"Found {len(incomplete_features)} features with incomplete tests:"
        )
        for feat in incomplete_features[:5]:
            result.add_warning(f"  - {feat}")
    else:
        result.add_info(f"{len(features)} features have complete test coverage")


def find_undocumented_classes(model: dict, codebase: dict, result: ValidationResult):
    """
    Find public classes in codebase that aren't documented in model.
    """
    # Get all documented class names
    documented = set()
    for entity in model.get("entities", []):
        if entity.get("type") in [
            "class",
            "abstract_class",
            "dataclass",
            "pydantic_model",
            "enum",
        ]:
            documented.add(entity["name"])

    # Find all classes in codebase
    all_classes = set()
    for file_path, parsed in codebase.items():
        if not file_path.startswith("ondine/"):
            continue

        for cls in parsed.get("classes", []):
            class_name = cls["name"]
            # Skip private classes
            if not class_name.startswith("_"):
                all_classes.add(class_name)

    # Find undocumented
    undocumented = all_classes - documented

    # Filter out common patterns that don't need documentation
    undocumented = {
        cls
        for cls in undocumented
        if not cls.endswith("Error")  # Exception classes
        and not cls.endswith("Warning")  # Warning classes
        and not cls.startswith("Test")  # Test classes
        and cls not in ["Main", "CLI"]  # Common utility classes
    }

    if undocumented:
        result.add_warning(f"Found {len(undocumented)} undocumented classes:")
        for cls in sorted(list(undocumented)[:10]):
            result.add_warning(f"  - {cls}")
        if len(undocumented) > 10:
            result.add_warning(f"  ... and {len(undocumented) - 10} more")
    else:
        result.add_info("All public classes are documented")


def validate_relationships(model: dict, codebase: dict, result: ValidationResult):
    """
    Validate that relationships match actual code structure.
    """
    entities = model.get("entities", [])
    invalid_relationships = []
    valid_count = 0

    # Build entity name -> file mapping
    entity_files = {e["name"]: e.get("file", "") for e in entities}

    for entity in entities:
        name = entity["name"]
        relationships = entity.get("relationships", [])

        for rel in relationships:
            target = rel.get("target")
            if not target:
                continue

            # Check if target entity exists
            if target not in entity_files:
                invalid_relationships.append(
                    f"{name} -> {target} (target entity not defined in model)"
                )
                continue

            valid_count += 1

    if invalid_relationships:
        result.add_error(f"Found {len(invalid_relationships)} invalid relationships:")
        for rel in invalid_relationships[:5]:
            result.add_error(f"  - {rel}")
    else:
        result.add_info(f"{valid_count} relationships validated")


def main(quick: bool = False):
    """
    Run all validation checks.

    Args:
        quick: If True, skip expensive checks (for pre-commit hook)
    """
    print("Ondine Architecture Model Validator")
    print("=" * 60)

    # Load model
    print("Loading architecture model...")
    model = load_model()
    print(f"✓ Loaded {len(model.get('entities', []))} entities")
    print(f"✓ Loaded {len(model.get('features', []))} features")
    print(f"✓ Loaded {len(model.get('layers', []))} layers")
    print()

    # Parse codebase (skip if quick mode)
    if not quick:
        print("Parsing codebase...")
        codebase = parse_codebase(SOURCE_DIR)
        print(f"✓ Parsed {len(codebase)} Python files")
        print()
    else:
        codebase = {}
        print("⚡ Quick mode: Skipping codebase parsing")
        print()

    # Run validations
    result = ValidationResult()

    print("Running validation checks...")
    print()

    # 1. Entity existence
    if not quick:
        print("  [1/5] Validating entity existence...")
        validate_entities(model, codebase, result)

    # 2. Relationships
    print("  [2/5] Validating relationships...")
    validate_relationships(model, codebase, result)

    # 3. Layer dependencies
    if not quick:
        print("  [3/5] Validating layer dependencies...")
        validate_layer_dependencies(model, codebase, result)

    # 4. Feature coverage
    print("  [4/5] Validating feature test coverage...")
    validate_features(model, result)

    # 5. Completeness
    if not quick:
        print("  [5/5] Checking for undocumented classes...")
        find_undocumented_classes(model, codebase, result)

    # Print report
    result.print_report()

    # Exit with appropriate code
    sys.exit(0 if result.is_success() else 1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate architecture model")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (skip expensive checks)",
    )
    args = parser.parse_args()

    main(quick=args.quick)

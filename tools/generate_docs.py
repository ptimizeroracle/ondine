#!/usr/bin/env python3
"""
Generate architecture documentation from model.yaml.

Creates Mermaid diagrams and consolidated ARCHITECTURE.md from the
architecture model defined in docs/architecture/model.yaml.
"""

import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARCHITECTURE_DIR = PROJECT_ROOT / "docs" / "architecture"
MODEL_FILE = ARCHITECTURE_DIR / "model.yaml"
DIAGRAMS_DIR = ARCHITECTURE_DIR / "diagrams"
OUTPUT_FILE = PROJECT_ROOT / "docs" / "ARCHITECTURE.md"


def load_model() -> dict[str, Any]:
    """Load and parse the architecture model YAML."""
    if not MODEL_FILE.exists():
        print(f"ERROR: Model file not found: {MODEL_FILE}")
        sys.exit(1)

    with open(MODEL_FILE) as f:
        model = yaml.safe_load(f)

    print(f"✓ Loaded model from {MODEL_FILE}")
    return model


def generate_erd(entities: list[dict], relationships: list[dict]) -> str:
    """
    Generate Mermaid Entity-Relationship Diagram.

    Shows all entities and their relationships.
    """
    lines = ["```mermaid", "erDiagram"]

    # Group entities by layer
    by_layer: dict[str, list[dict]] = {}
    for entity in entities:
        layer = entity.get("layer", "unknown")
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(entity)

    # Define entities with attributes (sample only, full attrs too verbose)
    for layer in ["core", "adapters", "stages", "orchestration", "api", "utils"]:
        if layer not in by_layer:
            continue

        for entity in by_layer[layer]:
            name = entity["name"]
            entity_type = entity.get("type", "class")

            # Add entity definition with key attributes
            attrs = entity.get("attributes", [])
            if attrs and isinstance(attrs, list) and len(attrs) > 0:
                # Show first 3 attributes as sample
                attr_sample = attrs[:3]
                attr_lines = []
                for attr in attr_sample:
                    if isinstance(attr, dict):
                        for key, value in attr.items():
                            attr_lines.append(f"        {value} {key}")
                    else:
                        attr_lines.append(f"        string {attr}")

                if len(attrs) > 3:
                    attr_lines.append("        string ...")

                lines.append(f"    {name} {{")
                lines.extend(attr_lines)
                lines.append("    }")
            else:
                lines.append(f"    {name} {{")
                lines.append(f"        string {entity_type}")
                lines.append("    }")

    # Add relationships
    for entity in entities:
        name = entity["name"]
        rels = entity.get("relationships", [])

        for rel in rels:
            rel_type = rel.get("type", "")
            target = rel.get("target", "")

            if not target:
                continue

            # Map relationship types to Mermaid syntax
            if rel_type == "extends":
                lines.append(f'    {name} ||--||{{ {target} : "extends"')
            elif rel_type == "implements":
                lines.append(f'    {name} ||..||{{ {target} : "implements"')
            elif rel_type == "depends_on":
                lines.append(f'    {name} }}}}o--|| {target} : "depends_on"')
            elif rel_type == "uses":
                lines.append(f'    {name} }}}}o..|| {target} : "uses"')
            elif rel_type == "creates":
                lines.append(f'    {name} ||--|| {target} : "creates"')
            elif rel_type == "aggregates":
                lines.append(f'    {name} ||--o{{{{ {target} : "aggregates"')
            else:
                lines.append(f'    {name} }}}}|..|| {target} : "{rel_type}"')

    lines.append("```")
    return "\n".join(lines)


def generate_class_diagram(entities: list[dict]) -> str:
    """
    Generate Mermaid Class Diagram showing inheritance hierarchies.
    """
    lines = ["```mermaid", "classDiagram"]

    # Group by layer for organization
    by_layer: dict[str, list[dict]] = {}
    for entity in entities:
        if entity.get("type") not in ["class", "abstract_class"]:
            continue  # Only classes
        layer = entity.get("layer", "unknown")
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(entity)

    # Process each layer
    for layer in ["core", "adapters", "stages", "orchestration", "api", "utils"]:
        if layer not in by_layer:
            continue

        # Add comment for layer
        lines.append(f"    %% {layer.upper()} LAYER")

        for entity in by_layer[layer]:
            name = entity["name"]
            is_abstract = entity.get("type") == "abstract_class"

            # Define class
            if is_abstract:
                lines.append(f"    class {name} {{")
                lines.append("        <<abstract>>")
            else:
                lines.append(f"    class {name} {{")

            # Add methods (sample only)
            methods = entity.get("methods", [])
            if methods:
                for method in methods[:3]:  # First 3 methods
                    if isinstance(method, str):
                        lines.append(f"        +{method}")
                if len(methods) > 3:
                    lines.append("        +...")

            lines.append("    }")

            # Add inheritance relationships
            rels = entity.get("relationships", [])
            for rel in rels:
                if rel.get("type") == "extends":
                    target = rel.get("target")
                    lines.append(f"    {target} <|-- {name}")
                elif rel.get("type") == "implements":
                    target = rel.get("target")
                    lines.append(f"    {target} <|.. {name}")

    lines.append("```")
    return "\n".join(lines)


def generate_package_dependencies(entities: list[dict], layers: list[dict]) -> str:
    """
    Generate Mermaid flowchart showing package/layer dependencies.
    """
    lines = ["```mermaid", "flowchart TD"]

    # Add layer nodes
    for layer in layers:
        name = layer["name"]
        desc = layer.get("description", "")
        lines.append(f'    {name}["{name.upper()}<br/>{desc}"]')

    # Add dependencies between layers
    for layer in layers:
        name = layer["name"]
        allowed = layer.get("allowed_dependencies", [])
        for dep in allowed:
            lines.append(f"    {name} --> {dep}")

    # Style layers
    lines.append("    classDef coreLayer fill:#e1f5ff,stroke:#01579b")
    lines.append("    classDef adapterLayer fill:#fff3e0,stroke:#e65100")
    lines.append("    classDef stageLayer fill:#f3e5f5,stroke:#4a148c")
    lines.append("    classDef orchestrationLayer fill:#e8f5e9,stroke:#1b5e20")
    lines.append("    classDef apiLayer fill:#fce4ec,stroke:#880e4f")
    lines.append("    classDef utilLayer fill:#f5f5f5,stroke:#424242")

    lines.append("    class core coreLayer")
    lines.append("    class adapters adapterLayer")
    lines.append("    class stages stageLayer")
    lines.append("    class orchestration orchestrationLayer")
    lines.append("    class api apiLayer")
    lines.append("    class utils utilLayer")

    lines.append("```")
    return "\n".join(lines)


def generate_feature_map(features: list[dict]) -> str:
    """
    Generate Mermaid mind map showing feature-to-component mappings.
    """
    lines = ["```mermaid", "mindmap"]
    lines.append("  root((Ondine<br/>Features))")

    for feature in features:
        name = feature["name"]
        desc = feature.get("description", "")
        components = feature.get("components", [])

        lines.append(f"    {name}")
        lines.append(f"      {desc}")

        # Add components
        for comp in components[:5]:  # First 5 to keep readable
            lines.append(f"      ({comp})")

        if len(components) > 5:
            lines.append(f"      (+{len(components) - 5} more)")

    lines.append("```")
    return "\n".join(lines)


def generate_layer_architecture(layers: list[dict]) -> str:
    """
    Generate detailed layer architecture diagram.
    """
    lines = ["```mermaid", "graph TB"]
    lines.append('    subgraph "Ondine Architecture"')

    # Reverse order (API at top, Utils at bottom)
    layer_order = ["api", "orchestration", "stages", "adapters", "core", "utils"]

    for i, layer_name in enumerate(layer_order):
        # Find layer definition
        layer = next((lyr for lyr in layers if lyr["name"] == layer_name), None)
        if not layer:
            continue

        desc = layer.get("description", "")
        lines.append(f'        subgraph layer{i}["{layer_name.upper()} - {desc}"]')
        lines.append(f"            {layer_name}_content[Components]")
        lines.append("        end")

    lines.append("    end")

    # Add dependencies
    for i, layer_name in enumerate(layer_order):
        layer = next((lyr for lyr in layers if lyr["name"] == layer_name), None)
        if not layer:
            continue

        allowed = layer.get("allowed_dependencies", [])
        for dep in allowed:
            # Find dependency index
            if dep in layer_order:
                lines.append(f"    layer{i} --> layer{layer_order.index(dep)}")

    lines.append("```")
    return "\n".join(lines)


def generate_consolidated_doc(model: dict[str, Any], diagrams: dict[str, str]) -> str:
    """
    Generate consolidated ARCHITECTURE.md with all diagrams embedded.
    """
    lines = []

    # Header
    lines.append("# Ondine Architecture")
    lines.append("")
    lines.append("**Auto-generated from `docs/architecture/model.yaml`**")
    lines.append("")
    lines.append("**DO NOT EDIT THIS FILE MANUALLY**")
    lines.append("")
    lines.append("To update this documentation:")
    lines.append("1. Edit `docs/architecture/model.yaml`")
    lines.append("2. Run `python tools/generate_docs.py`")
    lines.append("3. Commit both `model.yaml` and this generated file")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Table of Contents
    lines.append("## Table of Contents")
    lines.append("")
    lines.append("- [Overview](#overview)")
    lines.append("- [Layer Architecture](#layer-architecture)")
    lines.append("- [Package Dependencies](#package-dependencies)")
    lines.append("- [Entity Relationships](#entity-relationships)")
    lines.append("- [Class Hierarchies](#class-hierarchies)")
    lines.append("- [Features](#features)")
    lines.append("- [Feature Map](#feature-map)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "Ondine follows a **5-layer clean architecture** with strict dependency rules:"
    )
    lines.append("")
    lines.append("- **Utils**: Cross-cutting utilities (no dependencies)")
    lines.append("- **Core**: Domain models and specifications")
    lines.append("- **Adapters**: External system integrations")
    lines.append("- **Stages**: Data transformation pipeline stages")
    lines.append("- **Orchestration**: Execution control and state management")
    lines.append("- **API**: User-facing high-level interfaces")
    lines.append("")
    lines.append(
        "**Dependency Rule**: Dependencies only point inward (higher layers depend on lower layers, never the reverse)."
    )
    lines.append("")
    lines.append(
        "See `docs/architecture/decisions/ADR-001-layered-architecture.md` for detailed rationale."
    )
    lines.append("")

    # Layer Architecture
    lines.append("## Layer Architecture")
    lines.append("")
    lines.append(diagrams["layer_architecture"])
    lines.append("")

    # Layer Definitions
    lines.append("### Layer Definitions")
    lines.append("")
    layers = model.get("layers", [])
    for layer in layers:
        name = layer["name"]
        desc = layer.get("description", "")
        allowed = layer.get("allowed_dependencies", [])
        prohibited = layer.get("prohibited_imports", [])

        lines.append(f"#### {name.upper()}")
        lines.append(f"**Description**: {desc}")
        lines.append("")
        lines.append(
            f"**Allowed Dependencies**: {', '.join(allowed) if allowed else 'None (leaf layer)'}"
        )
        lines.append("")
        if prohibited:
            lines.append(f"**Prohibited Imports**: {', '.join(prohibited)}")
            lines.append("")

    # Package Dependencies
    lines.append("## Package Dependencies")
    lines.append("")
    lines.append(diagrams["package_dependencies"])
    lines.append("")

    # Entity Relationships
    lines.append("## Entity Relationships")
    lines.append("")
    lines.append("This diagram shows all entities and their relationships:")
    lines.append("")
    lines.append(diagrams["erd"])
    lines.append("")

    # Class Hierarchies
    lines.append("## Class Hierarchies")
    lines.append("")
    lines.append("This diagram shows inheritance relationships:")
    lines.append("")
    lines.append(diagrams["class_diagram"])
    lines.append("")

    # Features
    lines.append("## Features")
    lines.append("")
    features = model.get("features", [])
    for feature in features:
        name = feature["name"]
        desc = feature.get("description", "")
        components = feature.get("components", [])
        tests = feature.get("tests", [])
        examples = feature.get("examples", [])
        introduced = feature.get("introduced_in", "")
        deps = feature.get("dependencies", [])

        lines.append(f"### {name}")
        lines.append(f"**Description**: {desc}")
        lines.append("")
        if introduced:
            lines.append(f"**Introduced**: {introduced}")
            lines.append("")
        if deps:
            lines.append(f"**Dependencies**: {', '.join(deps)}")
            lines.append("")
        lines.append(f"**Components** ({len(components)}):")
        for comp in components:
            lines.append(f"- `{comp}`")
        lines.append("")
        if tests:
            lines.append(f"**Tests** ({len(tests)}):")
            for test in tests:
                lines.append(f"- `{test}`")
            lines.append("")
        if examples:
            lines.append(f"**Examples** ({len(examples)}):")
            for example in examples:
                lines.append(f"- `{example}`")
            lines.append("")

    # Feature Map
    lines.append("## Feature Map")
    lines.append("")
    lines.append(diagrams["feature_map"])
    lines.append("")

    # Statistics
    entities = model.get("entities", [])
    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- **Total Entities**: {len(entities)}")
    lines.append(f"- **Total Features**: {len(features)}")
    lines.append(f"- **Total Layers**: {len(layers)}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("**Generated by**: `tools/generate_docs.py`")
    lines.append("")
    lines.append("**Source**: `docs/architecture/model.yaml`")
    lines.append("")
    lines.append("**Last Generated**: Run `python tools/generate_docs.py` to update")
    lines.append("")

    return "\n".join(lines)


def main():
    """Generate all architecture documentation."""
    print("Ondine Architecture Documentation Generator")
    print("=" * 60)

    # Create diagrams directory if it doesn't exist
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model()

    entities = model.get("entities", [])
    layers = model.get("layers", [])
    features = model.get("features", [])

    print(f"✓ Found {len(entities)} entities")
    print(f"✓ Found {len(layers)} layers")
    print(f"✓ Found {len(features)} features")
    print()

    # Generate diagrams
    print("Generating diagrams...")

    diagrams = {}

    print("  - Entity-Relationship Diagram...")
    # Extract relationships from entities
    all_relationships = []
    for entity in entities:
        rels = entity.get("relationships", [])
        for rel in rels:
            all_relationships.append({"source": entity["name"], **rel})
    diagrams["erd"] = generate_erd(entities, all_relationships)
    (DIAGRAMS_DIR / "entity_relationships.mmd").write_text(diagrams["erd"])

    print("  - Class Hierarchies...")
    diagrams["class_diagram"] = generate_class_diagram(entities)
    (DIAGRAMS_DIR / "class_hierarchies.mmd").write_text(diagrams["class_diagram"])

    print("  - Package Dependencies...")
    diagrams["package_dependencies"] = generate_package_dependencies(entities, layers)
    (DIAGRAMS_DIR / "package_dependencies.mmd").write_text(
        diagrams["package_dependencies"]
    )

    print("  - Feature Map...")
    diagrams["feature_map"] = generate_feature_map(features)
    (DIAGRAMS_DIR / "feature_map.mmd").write_text(diagrams["feature_map"])

    print("  - Layer Architecture...")
    diagrams["layer_architecture"] = generate_layer_architecture(layers)
    (DIAGRAMS_DIR / "layer_architecture.mmd").write_text(diagrams["layer_architecture"])

    print(f"✓ Generated {len(diagrams)} diagrams in {DIAGRAMS_DIR}")
    print()

    # Generate consolidated documentation
    print("Generating consolidated documentation...")
    consolidated = generate_consolidated_doc(model, diagrams)
    OUTPUT_FILE.write_text(consolidated)
    print(f"✓ Generated {OUTPUT_FILE}")
    print()

    print("=" * 60)
    print("✅ Documentation generation complete!")
    print()
    print("Generated files:")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {DIAGRAMS_DIR / 'entity_relationships.mmd'}")
    print(f"  - {DIAGRAMS_DIR / 'class_hierarchies.mmd'}")
    print(f"  - {DIAGRAMS_DIR / 'package_dependencies.mmd'}")
    print(f"  - {DIAGRAMS_DIR / 'feature_map.mmd'}")
    print(f"  - {DIAGRAMS_DIR / 'layer_architecture.mmd'}")
    print()
    print("Next steps:")
    print("  1. Review generated ARCHITECTURE.md")
    print("  2. Commit both model.yaml and generated files")
    print("  3. Run validation: python tools/validate_architecture.py")


if __name__ == "__main__":
    main()

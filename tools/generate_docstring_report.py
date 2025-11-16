#!/usr/bin/env python3
"""
Generate Docstring Quality Report

Analyzes docstring quality beyond just coverage:
- Checks for examples in docstrings
- Validates Args/Returns/Raises sections
- Identifies opportunities for improvement

Usage:
    python tools/generate_docstring_report.py
    python tools/generate_docstring_report.py --output report.json
"""

# ruff: noqa: N802
import ast
import json
import sys
from pathlib import Path


class DocstringQualityAnalyzer(ast.NodeVisitor):
    """Analyze docstring quality metrics."""

    def __init__(self, filename: str):
        self.filename = filename
        self.results = []

    def analyze_docstring(self, node, node_type: str):
        """Analyze quality of a single docstring."""
        if node.name.startswith("_"):
            return  # Skip private

        docstring = ast.get_docstring(node)

        if not docstring:
            return

        # Check for key sections
        has_args = "Args:" in docstring or "Parameters:" in docstring
        has_returns = "Returns:" in docstring or "Return:" in docstring
        has_raises = "Raises:" in docstring
        has_example = (
            "Example:" in docstring or "```" in docstring or ">>>" in docstring
        )
        has_note = "Note:" in docstring or "Warning:" in docstring

        # Count lines
        lines = docstring.strip().split("\n")
        line_count = len(lines)

        # Check if it's just a one-liner
        is_one_liner = line_count == 1

        # Quality score (0-100)
        score = 0
        if line_count > 1:
            score += 20  # Multi-line
        if has_args:
            score += 20
        if has_returns:
            score += 20
        if has_example:
            score += 30  # Examples are most valuable
        if has_raises:
            score += 5
        if has_note:
            score += 5

        self.results.append(
            {
                "name": node.name,
                "type": node_type,
                "line": node.lineno,
                "score": score,
                "line_count": line_count,
                "is_one_liner": is_one_liner,
                "has_args": has_args,
                "has_returns": has_returns,
                "has_raises": has_raises,
                "has_example": has_example,
                "has_note": has_note,
            }
        )

    def visit_ClassDef(self, node: ast.ClassDef):
        """Analyze class docstrings."""
        self.analyze_docstring(node, "class")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze function docstrings."""
        self.analyze_docstring(node, "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Analyze async function docstrings."""
        self.analyze_docstring(node, "async_function")
        self.generic_visit(node)


def analyze_file(filepath: Path) -> dict:
    """Analyze docstring quality for a file."""
    try:
        content = filepath.read_text()
        tree = ast.parse(content)
        analyzer = DocstringQualityAnalyzer(str(filepath))
        analyzer.visit(tree)

        # Calculate average score
        scores = [r["score"] for r in analyzer.results]
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            "file": str(filepath),
            "items": analyzer.results,
            "avg_score": avg_score,
            "needs_examples": sum(1 for r in analyzer.results if not r["has_example"]),
            "one_liners": sum(1 for r in analyzer.results if r["is_one_liner"]),
        }
    except Exception as e:
        return {"file": str(filepath), "error": str(e)}


def scan_directory(directory: Path) -> list[dict]:
    """Scan all Python files."""
    results = []

    for py_file in sorted(directory.rglob("*.py")):
        if "test_" in py_file.name or "__pycache__" in str(py_file):
            continue

        result = analyze_file(py_file)
        if "error" not in result:
            results.append(result)

    return results


def print_quality_report(results: list[dict]):
    """Print quality report."""
    print("=" * 80)
    print("DOCSTRING QUALITY REPORT")
    print("=" * 80)
    print()

    # Overall metrics
    total_items = sum(len(r["items"]) for r in results)
    avg_score = (
        sum(r["avg_score"] * len(r["items"]) for r in results) / total_items
        if total_items > 0
        else 0
    )
    needs_examples = sum(r["needs_examples"] for r in results)
    one_liners = sum(r["one_liners"] for r in results)

    print(f"Overall Quality Score: {avg_score:.1f}/100")
    print(f"Total Items Analyzed: {total_items}")
    print(
        f"Items Without Examples: {needs_examples} ({needs_examples / total_items * 100:.1f}%)"
    )
    print(f"One-liner Docstrings: {one_liners} ({one_liners / total_items * 100:.1f}%)")
    print()

    # Top files needing improvement
    print("=" * 80)
    print("FILES NEEDING MOST IMPROVEMENT (by example count)")
    print("=" * 80)

    files_by_need = sorted(results, key=lambda x: x["needs_examples"], reverse=True)

    for r in files_by_need[:10]:
        if r["needs_examples"] == 0:
            continue

        rel_path = Path(r["file"]).relative_to(Path.cwd())
        print(f"  {rel_path}")
        print(
            f"    Score: {r['avg_score']:.1f}/100, Missing examples: {r['needs_examples']}"
        )

    print()
    print("=" * 80)
    print("RECOMMENDATION: Focus on adding examples to high-priority API files")
    print("=" * 80)


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    ondine_dir = project_root / "ondine"

    print("Analyzing docstring quality in ondine/ directory...")
    print()

    results = scan_directory(ondine_dir)

    # Print report
    print_quality_report(results)

    # Save JSON if requested
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        output_file = sys.argv[idx + 1]
        Path(output_file).write_text(json.dumps(results, indent=2))
        print(f"\n✅ Saved detailed report to: {output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check Docstring Coverage for Ondine

Scans Python files and reports docstring coverage metrics.
Identifies missing docstrings in public APIs.

Usage:
    python tools/check_docstring_coverage.py
    python tools/check_docstring_coverage.py --detailed
"""

# ruff: noqa: N802
import ast
import sys
from pathlib import Path


class DocstringChecker(ast.NodeVisitor):
    """AST visitor to check for missing docstrings."""

    def __init__(self, filename: str):
        self.filename = filename
        self.missing = []
        self.total = 0
        self.documented = 0

    def visit_ClassDef(self, node: ast.ClassDef):
        """Check class docstrings."""
        if not node.name.startswith("_"):  # Public class
            self.total += 1
            if ast.get_docstring(node):
                self.documented += 1
            else:
                self.missing.append(f"Class: {node.name} (line {node.lineno})")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check function/method docstrings."""
        if not node.name.startswith("_"):  # Public function
            self.total += 1
            if ast.get_docstring(node):
                self.documented += 1
            else:
                self.missing.append(f"Function: {node.name} (line {node.lineno})")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Check async function docstrings."""
        if not node.name.startswith("_"):
            self.total += 1
            if ast.get_docstring(node):
                self.documented += 1
            else:
                self.missing.append(f"Async Function: {node.name} (line {node.lineno})")
        self.generic_visit(node)


def check_file(filepath: Path) -> dict:
    """Check docstring coverage for a single file."""
    try:
        content = filepath.read_text()
        tree = ast.parse(content)
        checker = DocstringChecker(str(filepath))
        checker.visit(tree)

        return {
            "file": filepath,
            "total": checker.total,
            "documented": checker.documented,
            "missing": checker.missing,
            "coverage": (
                (checker.documented / checker.total * 100) if checker.total > 0 else 100
            ),
        }
    except Exception as e:
        return {
            "file": filepath,
            "total": 0,
            "documented": 0,
            "missing": [],
            "coverage": 0,
            "error": str(e),
        }


def scan_directory(directory: Path) -> list[dict]:
    """Scan all Python files in directory."""
    results = []

    for py_file in sorted(directory.rglob("*.py")):
        # Skip test files and __pycache__
        if "test_" in py_file.name or "__pycache__" in str(py_file):
            continue

        result = check_file(py_file)
        results.append(result)

    return results


def print_report(results: list[dict], detailed: bool = False):
    """Print coverage report."""
    print("=" * 80)
    print("DOCSTRING COVERAGE REPORT")
    print("=" * 80)
    print()

    # Calculate totals
    total_items = sum(r["total"] for r in results)
    total_documented = sum(r["documented"] for r in results)
    overall_coverage = (
        (total_documented / total_items * 100) if total_items > 0 else 100
    )

    # Print summary
    print(f"Overall Coverage: {overall_coverage:.1f}%")
    print(f"Total Items: {total_items}")
    print(f"Documented: {total_documented}")
    print(f"Missing: {total_items - total_documented}")
    print()

    # Print by module
    print("=" * 80)
    print("BY MODULE")
    print("=" * 80)

    # Group by module
    modules = {}
    for r in results:
        module = r["file"].parent.name
        if module not in modules:
            modules[module] = []
        modules[module].append(r)

    for module in sorted(modules.keys()):
        module_results = modules[module]
        module_total = sum(r["total"] for r in module_results)
        module_documented = sum(r["documented"] for r in module_results)
        module_coverage = (
            (module_documented / module_total * 100) if module_total > 0 else 100
        )

        status = (
            "✅" if module_coverage >= 80 else "⚠️" if module_coverage >= 60 else "❌"
        )
        print(
            f"{status} {module:20s} {module_coverage:5.1f}% ({module_documented}/{module_total})"
        )

    # Print detailed file-by-file report
    if detailed:
        print()
        print("=" * 80)
        print("DETAILED FILE REPORT")
        print("=" * 80)

        for r in results:
            if r["total"] == 0:
                continue

            status = (
                "✅" if r["coverage"] >= 80 else "⚠️" if r["coverage"] >= 60 else "❌"
            )
            rel_path = r["file"].relative_to(Path.cwd())
            print(
                f"\n{status} {rel_path} - {r['coverage']:.1f}% ({r['documented']}/{r['total']})"
            )

            if r["missing"] and r["coverage"] < 100:
                print("   Missing docstrings:")
                for item in r["missing"][:5]:  # Show first 5
                    print(f"     - {item}")
                if len(r["missing"]) > 5:
                    print(f"     ... and {len(r['missing']) - 5} more")

    print()
    print("=" * 80)

    # Return exit code based on coverage
    if overall_coverage < 80:
        print(f"❌ Coverage below 80% threshold: {overall_coverage:.1f}%")
        return 1
    print(f"✅ Coverage meets 80% threshold: {overall_coverage:.1f}%")
    return 0


def main():
    """Main entry point."""
    detailed = "--detailed" in sys.argv

    project_root = Path(__file__).parent.parent
    ondine_dir = project_root / "ondine"

    print("Scanning ondine/ directory for docstring coverage...")
    print()

    results = scan_directory(ondine_dir)
    exit_code = print_report(results, detailed=detailed)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

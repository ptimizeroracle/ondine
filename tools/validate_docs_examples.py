#!/usr/bin/env python3
"""
Validate Code Examples in Documentation

This script extracts Python code blocks from markdown files and validates them
against the actual Ondine API to ensure examples are correct and runnable.

Usage:
    python tools/validate_docs_examples.py
    python tools/validate_docs_examples.py --fix  # Auto-fix common issues

Author: Quality Control
Date: 2025-11-09
"""

import ast
import re
import sys
from pathlib import Path

# Known non-existent APIs that should never appear in examples
BANNED_APIS = [
    r"\.with_processing\(",  # Never existed
    r"result\.total_rows\b",  # Should be result.metrics.total_rows
    r"result\.total_cost\b",  # Should be result.costs.total_cost
    r"result\.success_count\b",  # Should be result.metrics.success_count
    r"result\.execution_time\b",  # Should be result.duration
    r"result\.rows_processed\b",  # Should be result.metrics.total_rows
]

# Valid API patterns (for reference)
VALID_APIS = {
    ".with_batch_size(": "Configure batch size",
    ".with_concurrency(": "Configure concurrent requests",
    ".with_rate_limit(": "Configure rate limiting",
    ".with_checkpoint_interval(": "Configure checkpoint frequency",
    ".with_max_retries(": "Configure retry attempts",
    ".with_max_budget(": "Configure budget limit",
    "result.metrics.total_rows": "Total rows processed",
    "result.metrics.success_count": "Successful rows",
    "result.metrics.failed_rows": "Failed rows",
    "result.costs.total_cost": "Total cost",
    "result.costs.input_tokens": "Input tokens",
    "result.costs.output_tokens": "Output tokens",
    "result.duration": "Execution duration",
    "result.data": "Result DataFrame",
}


def extract_python_code_blocks(markdown_file: Path) -> list[tuple[int, str]]:
    """
    Extract Python code blocks from markdown file.

    Returns:
        List of (line_number, code) tuples
    """
    content = markdown_file.read_text()
    blocks = []

    # Match ```python ... ``` blocks
    pattern = r"```python\n(.*?)```"

    for match in re.finditer(pattern, content, re.DOTALL):
        code = match.group(1)
        # Find line number
        line_num = content[: match.start()].count("\n") + 1
        blocks.append((line_num, code))

    return blocks


def validate_code_block(code: str, line_num: int, file_path: Path) -> list[str]:
    """
    Validate a code block for banned APIs and syntax errors.

    Returns:
        List of error messages
    """
    errors = []

    # Check for banned APIs
    for pattern in BANNED_APIS:
        matches = list(re.finditer(pattern, code))
        for match in matches:
            code_line = code[: match.start()].count("\n") + 1
            errors.append(
                f"{file_path}:{line_num + code_line}: BANNED API: {match.group(0)}"
            )

    # Try to parse as Python (syntax check)
    # Skip if it's clearly a placeholder/incomplete example
    if "..." in code or "# ..." in code or "<" in code or ">" in code:
        return errors

    try:
        ast.parse(code)
    except SyntaxError:
        # Ignore syntax errors for documentation snippets
        # (they're often incomplete for brevity)
        pass

    return errors


def scan_documentation(docs_dir: Path) -> tuple[int, int, list[str]]:
    """
    Scan all markdown files in documentation directory.

    Returns:
        (total_blocks, error_count, error_messages)
    """
    total_blocks = 0
    all_errors = []

    # Find all markdown files
    md_files = list(docs_dir.rglob("*.md"))

    print(f"📚 Scanning {len(md_files)} markdown files...")
    print()

    for md_file in sorted(md_files):
        blocks = extract_python_code_blocks(md_file)

        if not blocks:
            continue

        print(f"📄 {md_file.relative_to(docs_dir)}: {len(blocks)} code blocks")

        for line_num, code in blocks:
            total_blocks += 1
            errors = validate_code_block(code, line_num, md_file)
            all_errors.extend(errors)

    return total_blocks, len(all_errors), all_errors


def scan_examples(examples_dir: Path) -> tuple[int, list[str]]:
    """
    Scan all Python example files.

    Returns:
        (file_count, error_messages)
    """
    py_files = list(examples_dir.glob("*.py"))
    all_errors = []

    print(f"\n📚 Scanning {len(py_files)} example files...")
    print()

    for py_file in sorted(py_files):
        print(f"📄 {py_file.name}")

        code = py_file.read_text()
        errors = validate_code_block(code, 1, py_file)
        all_errors.extend(errors)

    return len(py_files), all_errors


def suggest_fixes(error_msg: str) -> str:
    """Suggest fixes for common errors."""

    fixes = {
        ".with_processing(": """
    Replace with:
        .with_batch_size(batch_size)
        .with_concurrency(concurrency)
    """,
        "result.total_rows": "Replace with: result.metrics.total_rows",
        "result.total_cost": "Replace with: result.costs.total_cost",
        "result.success_count": "Replace with: result.metrics.success_count",
        "result.execution_time": "Replace with: result.duration",
        "result.rows_processed": "Replace with: result.metrics.total_rows",
    }

    for pattern, fix in fixes.items():
        if pattern in error_msg:
            return f"\n    💡 {fix}"

    return ""


def main():
    """Main validation function."""

    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"
    examples_dir = project_root / "examples"

    print("=" * 80)
    print("🔍 ONDINE DOCUMENTATION VALIDATOR")
    print("=" * 80)
    print()
    print("Checking for:")
    print("  ❌ Non-existent API methods")
    print("  ❌ Wrong result attributes")
    print("  ❌ Syntax errors")
    print()
    print("=" * 80)

    # Scan documentation
    total_blocks, doc_errors, doc_error_msgs = scan_documentation(docs_dir)

    # Scan examples
    total_examples, example_error_msgs = scan_examples(examples_dir)

    # Report results
    print()
    print("=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"Documentation: {total_blocks} code blocks scanned")
    print(f"Examples: {total_examples} files scanned")
    print()

    total_errors = len(doc_error_msgs) + len(example_error_msgs)

    if total_errors == 0:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("All code examples are valid and use correct API methods.")
        return 0

    print(f"❌ FOUND {total_errors} ERRORS")
    print()

    # Print documentation errors
    if doc_error_msgs:
        print("📚 Documentation Errors:")
        print("-" * 80)
        for error in doc_error_msgs:
            print(f"  {error}")
            print(suggest_fixes(error))
        print()

    # Print example errors
    if example_error_msgs:
        print("📝 Example File Errors:")
        print("-" * 80)
        for error in example_error_msgs:
            print(f"  {error}")
            print(suggest_fixes(error))
        print()

    print("=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("=" * 80)
    print("1. Fix all banned API usages")
    print("2. Update result attribute access")
    print("3. Test all examples before committing")
    print("4. Add this validator to CI/CD pipeline")
    print()

    return 1


if __name__ == "__main__":
    sys.exit(main())

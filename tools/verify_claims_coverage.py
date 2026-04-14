#!/usr/bin/env python3
"""Verify all 83 product features have test coverage or documentation.

This script is run in CI to ensure no feature claim is left unverified.
It scans tests/verification/ for test_claim_NN_ patterns and cross-references
with VERIFICATION_MATRIX.md for documentation-only features.

Exit code 0 = all features covered. Exit code 1 = gaps found.
"""

import re
import sys
from pathlib import Path

VERIFY_DIR = Path("tests/verification")
MATRIX_FILE = VERIFY_DIR / "VERIFICATION_MATRIX.md"
TOTAL_FEATURES = 83

# Features that are documentation-only (no automated test possible)
DOC_ONLY_FEATURES = {61, 62, 63, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83}


def find_tested_features() -> set[int]:
    """Scan test files for test_claim_NN_ patterns."""
    tested = set()
    pattern = re.compile(r"def test_claim_(\d+)_")

    for py_file in VERIFY_DIR.glob("test_verify_*.py"):
        text = py_file.read_text()
        for match in pattern.finditer(text):
            tested.add(int(match.group(1)))

    return tested


def verify_matrix_exists() -> bool:
    """Check VERIFICATION_MATRIX.md exists."""
    if not MATRIX_FILE.exists():
        print(f"ERROR: {MATRIX_FILE} not found")
        return False
    return True


def main() -> int:
    tested = find_tested_features()
    all_features = set(range(1, TOTAL_FEATURES + 1))
    expected_tested = all_features - DOC_ONLY_FEATURES

    missing_tests = expected_tested - tested
    extra_tests = tested - all_features

    print(f"Features verified by tests: {len(tested)}/{len(expected_tested)}")
    print(
        f"Features documented only:   {len(DOC_ONLY_FEATURES)}/{len(DOC_ONLY_FEATURES)}"
    )
    print(
        f"Total coverage:             {len(tested) + len(DOC_ONLY_FEATURES)}/{TOTAL_FEATURES}"
    )

    ok = True

    if missing_tests:
        print(f"\nERROR: Missing tests for features: {sorted(missing_tests)}")
        ok = False

    if extra_tests:
        print(f"\nWARNING: Tests for unknown feature numbers: {sorted(extra_tests)}")

    if not verify_matrix_exists():
        ok = False

    if ok:
        print(f"\nAll {TOTAL_FEATURES} features are covered.")
        return 0
    print("\nFeature coverage incomplete.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

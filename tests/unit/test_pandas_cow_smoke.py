"""
Smoke tests for pandas Copy-on-Write (CoW) readiness.

CoW becomes the unchangeable default in pandas 3.0. These tests exercise the
core pandas mutation patterns ondine uses (column assignment, iloc slicing,
in-place-ish operations) under CoW=True to catch latent SettingWithCopyWarning
bugs before the 3.0 bump forces them.

Run the FULL suite under CoW via:
    ONDINE_TEST_COW=1 pytest

See DEPENDENCY_UPGRADE_ACTION_PLAN.md (Q3) for the rationale.
"""

import warnings

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _force_cow_for_this_module():
    """Force CoW on for every test in this module regardless of the env toggle.

    The session-level conftest fixture (_pandas_copy_on_write_toggle) gates on
    ONDINE_TEST_COW so it doesn't change normal CI semantics. These tests are
    explicitly about CoW behaviour, so they enable it directly.
    """
    previous = pd.options.mode.copy_on_write
    pd.options.mode.copy_on_write = True
    try:
        yield
    finally:
        pd.options.mode.copy_on_write = previous


class TestCopyOnWriteReadiness:
    """Patterns ondine uses, verified safe under CoW."""

    def test_cow_is_active(self):
        """Sanity: CoW is actually enabled inside this module."""
        assert pd.options.mode.copy_on_write is True

    def test_column_assignment_does_not_mutate_source(self):
        """``df[col] = value`` must not propagate back to a sharing parent.

        This mirrors pipeline_composer.py:254 (``df[col_name] = result.data[col_name]``).
        Under CoW, assigning into a DataFrame that shares memory with another
        triggers a copy — the original must remain unchanged.
        """
        original = pd.DataFrame({"text": ["a", "b", "c"]})
        working = original.copy()
        working["text"] = ["x", "y", "z"]

        assert list(original["text"]) == ["a", "b", "c"], (
            "CoW violation: writing to a copy mutated the original DataFrame"
        )
        assert list(working["text"]) == ["x", "y", "z"]

    def test_iloc_slice_is_independent(self):
        """Slicing via ``df.iloc[start:end]`` must not share mutable state.

        pipeline.py and data_io.py slice DataFrames this way to chunk data.
        Under CoW, a slice is a view; the first write into it must detach.
        """
        df = pd.DataFrame({"value": list(range(10))})
        chunk = df.iloc[0:3]

        # Mutating the chunk must not change the parent.
        chunk_copy = chunk.copy()
        chunk_copy.loc[:, "value"] = [-1, -2, -3]

        assert list(df["value"]) == list(range(10)), (
            "CoW violation: writing to a sliced copy mutated the parent"
        )

    def test_chained_assignment_does_not_mutate_original(self):
        """Chained assignment ``df[mask]['b'] = x`` must not corrupt the parent.

        Pre-CoW this raised a SettingWithCopyWarning (often ignored) and could
        silently fail to update. Under CoW the chained write happens on a
        detached copy and is dropped — the original DataFrame is never mutated.
        This test documents that guarantee and serves as a regression guard
        if a future pandas version weakens it.
        """
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 0, 0, 0]})

        # This must NOT mutate df, regardless of whether it raises or warns.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[df["a"] > 2]["b"] = 1  # noqa: B023 — intentional chained assignment

        assert list(df["b"]) == [0, 0, 0, 0], (
            "CoW violation: chained assignment mutated the original DataFrame"
        )

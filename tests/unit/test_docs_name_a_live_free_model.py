"""Docs and examples must name the free model the daily smoke test exercises.

Free hosted models are retired without notice. The live smoke workflow calls
the one named in ``tests/integration/live_models.py`` every day, so its
retirement is noticed within a day and fixed in that one file. This test ties
the docs to the same name: once ``live_models`` moves on, every page and
example still recommending the retired model fails here, instead of going
stale the way the Groq Llama references did.
"""

import os
import re
from pathlib import Path

import pytest

from tests.integration.live_models import FREE_MODEL

# An override points the live suite at a stand-in model for one run; the docs
# are not expected to follow it.
pytestmark = pytest.mark.skipif(
    "ONDINE_TEST_FREE_MODEL" in os.environ,
    reason="live model overridden for this run",
)

REPO = Path(__file__).resolve().parents[2]
FREE_MODEL_ID = re.compile(r"openrouter/[\w.\-]+/[\w.\-]+:free")
DOCUMENTED = [
    *REPO.glob("README.md"),
    *(REPO / "docs").rglob("*.md"),
    *(REPO / "examples").rglob("*.py"),
    *(REPO / "examples").rglob("*.yaml"),
    *(REPO / "examples").rglob("*.ipynb"),
    *(REPO / "examples").rglob("*.md"),
]


def test_every_documented_free_model_is_the_one_under_daily_test():
    stale = {
        f"{path.relative_to(REPO)}: {model}"
        for path in DOCUMENTED
        for model in FREE_MODEL_ID.findall(path.read_text(encoding="utf-8"))
        if model != FREE_MODEL
    }

    assert not stale, (
        f"These name a free model other than {FREE_MODEL!r}, the one the live "
        f"smoke test proves still exists: {sorted(stale)}"
    )


def test_the_docs_do_name_the_free_model():
    """Guards the guard: a pattern that matches nothing would always pass."""
    mentions = [
        path for path in DOCUMENTED if FREE_MODEL in path.read_text(encoding="utf-8")
    ]

    assert mentions, f"no doc or example mentions {FREE_MODEL!r}"

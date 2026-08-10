"""Every ondine name the docs and examples use must exist.

The docs are the first code anyone runs, and nothing executed them. What that
allowed is not subtle drift: `with_retry_policy()` appeared in five places and
has never existed in this library, so the first snippet a new user copied
raised AttributeError. `result.metrics.success_count` was read in the
error-handling guide; `ProcessingStats` has no such field (#242).

Executing every snippet is not realistic — they open `data.csv`, they call
paid providers, they run for an hour. But *none* of the bugs above were logic
errors. They were names that do not exist, and that is checkable without
running anything:

- the snippet parses at all
- every `.with_*()` it calls exists on `PipelineBuilder`
- every field it reads off `result.metrics` / `result.costs` exists
- every `from ondine... import X` resolves

A snippet doing something genuinely outside this can opt out with an
``<!-- docs-check: skip -->`` comment on the line before its fence, which is
visible in review rather than silent.
"""

from __future__ import annotations

import ast
import importlib
import re
import textwrap
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from ondine.api.pipeline_builder import PipelineBuilder
from ondine.core.models import CostEstimate, ProcessingStats

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Prose files whose python fences are checked.
DOC_GLOBS = ("README.md", "docs/**/*.md", "examples/README.md")

#: Runnable scripts, checked whole. They are real python, so they need no
#: fence extraction — but they drift exactly the same way, and running them
#: needs a provider, so the same name checks are what is available cheaply.
SCRIPT_GLOBS = ("examples/*.py",)

FENCE = re.compile(r"^```(?:python|py)\s*$", re.MULTILINE)
SKIP_MARKER = "<!-- docs-check: skip -->"

BUILDER_METHODS = {name for name in dir(PipelineBuilder) if not name.startswith("_")}
METRICS_FIELDS = {field.name for field in dataclass_fields(ProcessingStats)} | {
    name for name in dir(ProcessingStats) if not name.startswith("_")
}
COSTS_FIELDS = {field.name for field in dataclass_fields(CostEstimate)} | {
    name for name in dir(CostEstimate) if not name.startswith("_")
}


def _drop_elisions(text: str) -> str:
    """Remove lines that are a bare `...`.

    Guides elide the middle of a builder chain to keep a section about one
    method. The surrounding calls are still real and still worth checking.
    """
    return "\n".join(
        line for line in text.splitlines() if line.strip() not in {"...", "# ..."}
    )


def _ground_dangling_chains(text: str) -> str:
    """Give a receiver to method calls written with none.

    `.with_batch_size(100)` starting a line at statement level is a fragment;
    the same text one indent inside an open paren is a continuation and must
    be left alone. Paren depth is what tells them apart.
    """
    grounded: list[str] = []
    depth = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if depth == 0 and stripped.startswith("."):
            line = "_receiver" + stripped
        depth = max(0, depth + line.count("(") - line.count(")"))
        grounded.append(line)
    return "\n".join(grounded)


class Snippet:
    """One fenced python block, with enough context to report it."""

    def __init__(self, path: Path, line: int, source: str) -> None:
        self.path = path
        self.line = line
        self.source = source

    @property
    def where(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.line}"

    def parse(self) -> ast.AST | None:
        """The snippet as a tree, or None if it is not parseable python.

        Docs show fragments as often as whole programs: a method chain with
        its receiver left off, a signature with no body, the inside of a class.
        Each shape gets the smallest wrapper that makes it parse, rather than
        being skipped — a fragment is exactly where a wrong method name hides.
        """
        body = textwrap.dedent(self.source)
        grounded = _ground_dangling_chains(_drop_elisions(body))
        candidates = (
            self.source,
            body,
            # `.with_batch_size(100)` on its own, and `...` standing in for
            # the parts of a chain the section is not about
            grounded,
            # the inside of a class: `def ...` / `async def ...` at top level
            "class _Scaffold:\n" + textwrap.indent(body, "    "),
            # an API signature quoted without its body, with or without `def`
            f"{body.rstrip()}: ...",
            f"def {body.strip()}: ...",
        )
        for candidate in candidates:
            try:
                return ast.parse(candidate)
            except SyntaxError:
                continue
        return None


def _collect_snippets() -> list[Snippet]:
    snippets: list[Snippet] = []
    for glob in DOC_GLOBS:
        for path in sorted(REPO_ROOT.glob(glob)):
            lines = path.read_text(encoding="utf-8").splitlines()
            index = 0
            while index < len(lines):
                if not FENCE.match(lines[index] + "\n"):
                    index += 1
                    continue
                opened_at = index
                index += 1
                body: list[str] = []
                while index < len(lines) and lines[index].strip() != "```":
                    body.append(lines[index])
                    index += 1
                index += 1
                preceding = lines[opened_at - 1] if opened_at else ""
                if SKIP_MARKER in preceding:
                    continue
                snippets.append(Snippet(path, opened_at + 2, "\n".join(body)))
    for glob in SCRIPT_GLOBS:
        for path in sorted(REPO_ROOT.glob(glob)):
            snippets.append(Snippet(path, 1, path.read_text(encoding="utf-8")))
    return snippets


SNIPPETS = _collect_snippets()


def _ondine_import_targets(tree: ast.AST) -> list[tuple[str, str]]:
    """(module, name) pairs imported from ondine by this snippet."""
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "ondine"
        ):
            for alias in node.names:
                targets.append((node.module or "", alias.name))
    return targets


def _builder_calls(tree: ast.AST) -> list[str]:
    """Names of `.with_*()` methods this snippet calls."""
    return [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("with_")
    ]


def _attributes_read_after(tree: ast.AST, holder: str) -> list[str]:
    """Attribute names read off `<anything>.<holder>`, e.g. result.metrics.X."""
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == holder
        ):
            found.append(node.attr)
    return found


def test_the_documentation_contains_snippets_to_check():
    """A glob that silently matches nothing would make this file a no-op."""
    assert len(SNIPPETS) > 100


@pytest.mark.parametrize("snippet", SNIPPETS, ids=lambda s: s.where)
def test_snippet_uses_only_names_that_exist(snippet: Snippet):
    """Catches copy-pasteable AttributeErrors in the docs.

    Every failure here is a snippet that cannot run as written, no matter what
    the reader's data or credentials look like.
    """
    tree = snippet.parse()
    if tree is None:
        pytest.fail(
            f"{snippet.where}: python fence does not parse. Mark it "
            f"`{SKIP_MARKER}` if it is deliberately pseudo-code."
        )

    problems: list[str] = []

    for method in _builder_calls(tree):
        if method not in BUILDER_METHODS:
            problems.append(f"PipelineBuilder has no method {method}()")

    for attribute in _attributes_read_after(tree, "metrics"):
        if attribute not in METRICS_FIELDS:
            problems.append(f"ProcessingStats has no field {attribute!r}")

    for attribute in _attributes_read_after(tree, "costs"):
        if attribute not in COSTS_FIELDS:
            problems.append(f"CostEstimate has no field {attribute!r}")

    for module, name in _ondine_import_targets(tree):
        try:
            imported = importlib.import_module(module)
        except ImportError as error:
            problems.append(f"cannot import {module}: {error}")
            continue
        if not hasattr(imported, name):
            problems.append(f"{module} exports no {name!r}")

    assert not problems, f"{snippet.where}: " + "; ".join(problems)

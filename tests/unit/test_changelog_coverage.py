"""Tests for the changelog-coverage check (issue #191).

The check exists because release-please skips commits its conventional-commit
parser cannot read, writes the changelog without them, and still reports
success. These tests pin the two judgements that decide whether an omission is
reported: which commits are *supposed* to appear, and which part of the
changelog counts as evidence that they did.

Both are easy to get subtly wrong in a direction that silently disables the
check — treating a hidden type as documented produces noise, and searching the
whole changelog instead of the newest section lets an old entry mask a fresh
omission.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_changelog_coverage.py"
)
_spec = importlib.util.spec_from_file_location("check_changelog_coverage", _SCRIPT)
coverage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coverage)


class TestWhichCommitsMustBeDocumented:
    @pytest.mark.parametrize(
        "subject",
        [
            "feat(orchestration): unified protocol (#180)",
            "fix(config): expand ${ENV_VAR} (#167)",
            "perf(rust): faster batching (#12)",
            "docs: rewrite the readme (#175)",
            "deps: upgrade everything (#182)",
            "revert: back out the parser change (#99)",
        ],
    )
    def test_types_that_reach_the_changelog_are_required(self, subject):
        assert coverage.documented_prs([subject])

    @pytest.mark.parametrize(
        "subject",
        [
            "chore(main): release 1.11.0 (#161)",
            "chore(deps): add types-PyYAML (#184)",
            "ci: pin actions (#149)",
            "test(mcp): join background threads (#210)",
            "build(deps): bump codecov (#172)",
            "style: reformat (#7)",
            "refactor(core): extract a stage (#8)",
        ],
    )
    def test_hidden_types_are_not_required(self, subject):
        """These never appear in the changelog, so demanding them is noise."""
        assert coverage.documented_prs([subject]) == {}

    def test_breaking_change_marker_is_still_a_feat(self):
        assert coverage.documented_prs(["feat(api)!: drop Python 3.9 (#42)"]) == {
            42: "feat(api)!: drop Python 3.9 (#42)"
        }

    def test_last_reference_wins_when_a_subject_cites_several(self):
        """ "fix: x (#166) (#167)" cites the issue, then the PR that closed it.

        The PR number is what release-please renders, so matching on the issue
        would report a false omission on every such commit.
        """
        subject = "fix(config): expand ${ENV_VAR} in config strings (#166) (#167)"

        assert coverage.documented_prs([subject]) == {167: subject}

    def test_commit_without_a_pr_reference_is_skipped(self):
        """A direct push to main has no PR number to match against."""
        assert coverage.documented_prs(["feat: pushed straight to main"]) == {}

    def test_non_conventional_subject_is_skipped(self):
        assert coverage.documented_prs(["Merge branch 'main' (#5)"]) == {}


class TestWhichChangelogSectionCounts:
    CHANGELOG = """# Changelog

## [1.11.0](https://example.com/compare/v1.10.1...v1.11.0) (2026-07-30)

### Features

* **intent:** add plan() ([#181](https://example.com/issues/181))

## [1.10.1](https://example.com/compare/v1.10.0...v1.10.1) (2026-04-23)

### Bug Fixes

* derive __version__ from metadata ([#163](https://example.com/issues/163))
"""

    def test_only_the_newest_section_is_considered(self):
        """An older release citing a number must not vouch for a new one.

        Searching the whole file would let #163 — documented under 1.10.1 —
        silently satisfy a 1.11.0 release that had dropped it.
        """
        cited = coverage.cited_prs(coverage.newest_section(self.CHANGELOG))

        assert 181 in cited
        assert 163 not in cited

    def test_missing_release_is_reported(self):
        """The end-to-end judgement: shipped but not cited."""
        shipped = coverage.documented_prs(
            [
                "feat(intent): add plan() (#181)",
                "feat(orchestration): unified protocol (#180)",
            ]
        )
        cited = coverage.cited_prs(coverage.newest_section(self.CHANGELOG))

        assert [pr for pr in shipped if pr not in cited] == [180]

    def test_changelog_with_no_sections_yet(self):
        assert coverage.newest_section("# Changelog\n\nNothing released.\n") == ""

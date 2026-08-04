"""Fail when a release's changelog omits a PR that shipped in it.

release-please parses each commit as a conventional commit. When that parse
fails it skips the commit, writes the changelog without it, and still reports
success — so the only symptom is an entry that quietly is not there. In 1.11.0
this swallowed #180, the largest feature in the release, because a body line
contained nested parentheses (`json.dumps(str(path))`) that its grammar could
not read (#191).

Nothing downstream notices: the version bump is usually implied by some *other*
commit, so the release completes and looks right. This script turns that silent
omission into a failed check by comparing the PRs merged since the last release
against the PR numbers cited in the changelog's newest section.

Usage:
    python scripts/check_changelog_coverage.py [--since TAG] [--changelog PATH]

Exits 0 when every documented-type commit is cited, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import subprocess  # nosec B404 — fixed `git` argv below, never a shell
import sys
from pathlib import Path

# Commit types release-please renders into the changelog for this repo. Types
# outside this set (chore, ci, test, build, style, refactor) are deliberately
# hidden, so their absence is correct and must not be reported.
DOCUMENTED_TYPES = frozenset({"feat", "fix", "perf", "deps", "docs", "revert"})

# "feat(scope)!: subject" — scope and the breaking-change "!" are optional.
COMMIT_TYPE = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]*\))?!?:")

# Trailing "(#123)" references. A subject may carry several — "fix: x (#166)
# (#167)" cites the issue and then the PR that closed it — and the last one is
# the PR that merged the commit.
PR_REFERENCE = re.compile(r"\(#(\d+)\)")

# release-please starts every release section with "## [1.2.3](compare-url)".
SECTION_HEADING = re.compile(r"^## \[")


def documented_prs(subjects: list[str]) -> dict[int, str]:
    """Map PR number -> commit subject, for commits that belong in a changelog.

    Commits with no PR reference (a direct push to main) are skipped: there is
    nothing to match them against.
    """
    found: dict[int, str] = {}
    for subject in subjects:
        match = COMMIT_TYPE.match(subject)
        if not match or match.group("type") not in DOCUMENTED_TYPES:
            continue
        references = PR_REFERENCE.findall(subject)
        if references:
            found[int(references[-1])] = subject
    return found


def newest_section(changelog: str) -> str:
    """Return the topmost release section — the one this release PR adds.

    Scoped deliberately: searching the whole file would let a number cited by
    some earlier release mask an omission in this one.
    """
    lines = changelog.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if SECTION_HEADING.match(line)), None
    )
    if start is None:
        return ""
    end = next(
        (i for i in range(start + 1, len(lines)) if SECTION_HEADING.match(lines[i])),
        len(lines),
    )
    return "\n".join(lines[start:end])


def cited_prs(section: str) -> set[int]:
    """Every issue/PR number referenced in a changelog section."""
    return {int(n) for n in re.findall(r"#(\d+)", section)}


def git_subjects(since: str, until: str) -> list[str]:
    # List argv with shell=False, so a ref is an argument to git and can never
    # reach a shell; `git` resolves from PATH like every other tool in CI.
    out = subprocess.run(  # nosec B603 B607
        ["git", "log", "--format=%s", f"{since}..{until}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def latest_tag() -> str:
    out = subprocess.run(  # nosec B603 B607
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=None, help="tag to compare from")
    parser.add_argument("--until", default="HEAD")
    parser.add_argument("--changelog", default="CHANGELOG.md", type=Path)
    args = parser.parse_args()

    since = args.since or latest_tag()
    shipped = documented_prs(git_subjects(since, args.until))
    if not shipped:
        print(f"No documented-type commits since {since} — nothing to check.")
        return 0

    cited = cited_prs(newest_section(args.changelog.read_text()))
    missing = {pr: subject for pr, subject in shipped.items() if pr not in cited}

    print(f"Commits since {since} that belong in the changelog: {len(shipped)}")
    print(f"PR numbers cited in the newest changelog section: {len(cited)}")

    if not missing:
        print("✅ Every shipped PR is present in the changelog.")
        return 0

    print(f"::error::{len(missing)} PR(s) shipped but missing from the changelog:")
    for pr, subject in sorted(missing.items()):
        print(f"  #{pr}  {subject}")
    print(
        "release-please skips commits its conventional-commit parser cannot "
        "read and still reports success (#191). Check the Release Please run "
        "log for 'commit could not be parsed', then add the entries by hand."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

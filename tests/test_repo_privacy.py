"""Repo-privacy gate: no private working-notes path may ever be tracked.

torchlens is a PUBLIC repository. Internal planning notes live in gitignored
territory (``.research/``, ``.project-context/`` minus the two curated docs,
and repo-root sprint ledgers). The primary defense is the ``no-internal-notes``
pre-commit hook, but hooks are one ``--no-verify`` (or a global
``core.hooksPath`` override) away from silence and have zero CI layers.

This test is the hook-independent layer: it asserts against ``git ls-files``
directly, so a private path that has already been committed -- however it got
past the hook -- turns the smoke tier red on every machine and CI leg.

The path matcher below MUST stay in lockstep with the ``files:`` regex of the
``no-internal-notes`` hook in ``.pre-commit-config.yaml``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parent.parent

# Lockstep mirror of .pre-commit-config.yaml no-internal-notes `files:` regex.
PRIVATE_PATH_PATTERN = re.compile(
    r"^\.research/"
    r"|^\.project-context/(?!(architecture|state_of_torchlens)\.md$)"
    r"|^(FORKS|PROGRESS)\.md$"
    r"|^[^/]*_RESULTS\.md$"
    r"|^[^/]*_(REPORT|AUDIT|SUMMARY|LEDGER|BATON|FINDINGS|NOTES|STATE|PLAN)(_[^/]*)?\.md$"
    r"|^(HUNT|SPRINT|ROUND)_[^/]*\.md$"
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git checkout (sdist/wheel install); privacy gate not applicable")
    return [p for p in result.stdout.decode("utf-8", "surrogateescape").split("\0") if p]


def find_private_violations(paths: list[str]) -> list[str]:
    return [p for p in paths if PRIVATE_PATH_PATTERN.search(p)]


def test_no_private_paths_are_tracked() -> None:
    violations = find_private_violations(_tracked_files())
    assert violations == [], (
        "PRIVATE paths are tracked in this PUBLIC repo -- untrack with "
        f"`git rm --cached` and never `git add -f` them: {violations}"
    )


def test_matcher_is_red_capable() -> None:
    """Non-vacuity: the matcher fires on each private class and spares the whitelist."""

    hits = find_private_violations(
        [
            ".research/notes.md",
            ".project-context/todos.md",
            ".project-context/torchlens_glossary.md",
            "FORKS.md",
            "PROGRESS.md",
            "sprint_RESULTS.md",
        ]
    )
    assert len(hits) == 6
    assert (
        find_private_violations(
            [
                ".project-context/architecture.md",
                ".project-context/state_of_torchlens.md",
                "RESULTS.md",
                "benchmarks/intervention_overhead_results.md",
                "torchlens/__init__.py",
            ]
        )
        == []
    )


def _git_config_reference_violations(paths_and_texts: list[tuple[str, str]]) -> list[str]:
    """Private-path references inside git CONFIG files (rule lines name paths).

    r7 R82 (opus b10 MED): ``.gitattributes`` published five internal
    ``.research/docs-plan-project_PLAN*.md`` filenames in this PUBLIC repo
    -- dead LFS rules left behind when the files were untracked (8f74a25b).
    The path gate above cannot see them (the leak is file CONTENT), and the
    pre-commit hook matched staged PATHS only. A git config rule line whose
    subject path is private is a leak of the artifact's name and the
    sprint's existence even when the rule is inert.
    """

    violations: list[str] = []
    for path, text in paths_and_texts:
        for line_number, line in enumerate(text.splitlines(), start=1):
            subject = line.strip().split(" ", 1)[0].split("\t", 1)[0]
            if subject and PRIVATE_PATH_PATTERN.search(subject):
                violations.append(f"{path}:{line_number}: {line.strip()}")
    return violations


def test_git_config_files_reference_no_private_paths() -> None:
    """No tracked .gitattributes/.lfsconfig rule may name a private path."""

    config_paths = [
        p
        for p in _tracked_files()
        if p == ".lfsconfig" or p.endswith(".gitattributes") or p.endswith("/.lfsconfig")
    ]
    contents = [
        (p, (REPO_ROOT / p).read_text(encoding="utf-8", errors="replace")) for p in config_paths
    ]
    violations = _git_config_reference_violations(contents)
    assert violations == [], (
        "git config rule(s) in this PUBLIC repo reference private paths -- "
        f"delete the rule lines (they leak internal artifact names): {violations}"
    )


def test_git_config_reference_matcher_is_red_capable() -> None:
    """Non-vacuity: the content matcher fires on the leaked class and spares public rules."""

    hits = _git_config_reference_violations(
        [
            (
                ".gitattributes",
                ".research/docs-plan-project_PLAN.md filter=lfs diff=lfs merge=lfs -text\n"
                ".project-context/todos.md -text\n"
                "*.ipynb filter=nbstripout\n"
                ".project-context/architecture.md -text\n",
            )
        ]
    )
    assert len(hits) == 2
    assert all(".research/" in hit or "todos" in hit for hit in hits)


def test_matcher_mirrors_precommit_hook() -> None:
    """The hook config's regex and this test's matcher may never drift apart."""

    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    match = re.search(r"id: no-internal-notes.*?files: '([^']+)'", config, flags=re.DOTALL)
    assert match is not None, "no-internal-notes hook (or its files: regex) is gone"
    hook_regex = re.compile(match.group(1))
    probes = [
        ".research/x.md",
        ".project-context/todos.md",
        ".project-context/architecture.md",
        ".project-context/state_of_torchlens.md",
        "FORKS.md",
        "PROGRESS.md",
        "lane_RESULTS.md",
        "RESULTS.md",
        "docs/guide.md",
    ]
    for probe in probes:
        assert bool(hook_regex.search(probe)) == bool(PRIVATE_PATH_PATTERN.search(probe)), (
            f"hook regex and test matcher disagree on {probe!r}"
        )

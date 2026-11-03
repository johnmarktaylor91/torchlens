"""Grep audit for deferred intervention features."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _harness import REPO_ROOT, run_checks


def _repo_root() -> Path:
    """Return the repository root.

    Returns
    -------
    Path
        Absolute repository root path.
    """

    return REPO_ROOT


def _grep_for_pattern(pattern: str, paths: list[str]) -> list[str]:
    """Search repository paths for a forbidden pattern.

    Parameters
    ----------
    pattern:
        Regular expression passed to ``rg``.
    paths:
        Repository-relative paths to scan.

    Returns
    -------
    list[str]
        Matching ``path:line:text`` records.
    """

    result = subprocess.run(
        ["git", "grep", "-E", "--line-number", pattern, "--", *paths],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr)
    return [line for line in result.stdout.splitlines() if line]


def check_not_mvp_features_not_in_public_surface() -> None:
    """Deferred features should not appear in public code, docs, or tests."""

    forbidden = [
        r"tl\.skip_module",
        r"tl\.fork\(",
        r"tl\.replay\(",
        r"tl\.rerun\(",
        r"cone-aware capture",
        r"selector chains",
        r"bundle-of-bundles",
    ]
    paths = ["torchlens/", "README.md", "docs/", "tests/"]
    allowlisted_files = {"scripts/checks/not_mvp_audit.py"}
    failures: list[str] = []

    for pattern in forbidden:
        for match in _grep_for_pattern(pattern, paths):
            rel_path = match.split(":", 1)[0]
            if rel_path in allowlisted_files:
                continue
            if "not-mvp-allow" in match:
                continue
            failures.append(f"{pattern}: {match}")

    assert failures == []


CHECKS = (check_not_mvp_features_not_in_public_surface,)


def main() -> int:
    """Run every not_mvp_audit check and return the exit status."""

    return run_checks("not_mvp_audit", CHECKS)


if __name__ == "__main__":
    sys.exit(main())

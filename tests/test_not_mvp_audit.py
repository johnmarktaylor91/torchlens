"""Grep audit for deferred intervention features."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _repo_root() -> Path:
    """Return the repository root.

    Returns
    -------
    Path
        Absolute repository root path.
    """

    return Path(__file__).resolve().parents[1]


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
        ["rg", "--line-number", pattern, *paths],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr)
    return [line for line in result.stdout.splitlines() if line]


def test_not_mvp_features_not_in_public_surface() -> None:
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
    allowlisted_files = {"tests/test_not_mvp_audit.py"}
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

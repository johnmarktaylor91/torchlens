"""Small shared repository-inspection helpers for crawler acceptance tests."""

from __future__ import annotations

import subprocess
from pathlib import Path


def repository_root() -> Path:
    """Return the checked-out repository root containing this acceptance package."""

    return Path(__file__).resolve().parents[2]


def tracked_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return every Git-tracked path without consulting ignored runtime files."""

    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return tuple(Path(value.decode("utf-8")) for value in completed.stdout.split(b"\0") if value)


def fabricated_crawler_locks(repo_root: Path) -> tuple[Path, ...]:
    """Return tracked crawler solve outputs that must be generated only on target."""

    return tuple(
        path
        for path in tracked_paths(repo_root)
        if path.parts[:3] == ("menagerie", "crawler", "envs")
        and (
            path.suffix == ".lock"
            or path.name.endswith(".resolved.json")
            or path.name.endswith(".resolved.sha256")
        )
    )

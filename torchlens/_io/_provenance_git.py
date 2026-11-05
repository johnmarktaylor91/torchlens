"""Bounded Git commit probe that stamps save-time bundle provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git_commit_hash(cwd: Path) -> str | None:
    """Return the Git commit for ``cwd`` with a short best-effort timeout.

    the provenance git commit is resolved from the TORCHLENS package
    directory, not ``Path.cwd()``. Stamping the working directory's repository
    recorded the USER's unrelated repo commit (a privacy leak) and made the same
    capture saved from two directories differ (an ambient-environment dependence in a
    supposedly portable artifact). Resolving from the package dir yields torchlens's
    own commit for a dev/editable install and ``None`` for a released wheel (whose
    ``torchlens_version`` already identifies it) -- deterministic either way.

    Parameters
    ----------
    cwd:
        Directory whose repository HEAD is probed (the torchlens package dir).

    Returns
    -------
    str | None
        Full commit hash, or ``None`` on any failure or timeout.
    """

    try:
        # R40: routed through the ONE bounded spawn seam so the timeout path
        # tears down the whole process group instead of orphaning a wedged
        # git helper's grandchildren.
        from .._utils._subprocess import run_bounded_subprocess

        completed = run_bounded_subprocess(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=0.25,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not 7 <= len(commit) <= 64:
        return None
    if any(character not in "0123456789abcdef" for character in commit.lower()):
        return None
    return commit

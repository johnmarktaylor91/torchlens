#!/usr/bin/env python3
"""Release-time hard stop: refuse a MAJOR version bump without written approval.

Runs in ``.github/workflows/release.yml`` immediately before
``semantic-release version`` (and in the release dry run). It takes the next
version that semantic-release computed (``semantic-release --noop version
--print``, which has no side effects and needs no token) and exits nonzero
when that version's major is greater than the current major, unless the
tracked approval file ``.github/MAJOR_RELEASE_APPROVED`` exists and its content
is exactly the computed version (one trailing newline allowed).

The current major is the LOWER of the ``project.version`` major in
``pyproject.toml`` and the highest ``vX.Y.Z`` release tag's major, so a hand
edit of ``pyproject.toml`` alone cannot raise the baseline.

Only the maintainer's explicit command authorizes creating the approval file.

Usage::

    python scripts/check_release_major.py --next 2.37.0 [--root PATH]

Exit codes:
    0  -- not a major bump, or a major bump approved for exactly this version
    1  -- unapproved major bump, or unparsable computed version (fail closed)
    2  -- the current version could not be determined
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

APPROVAL_RELPATH = Path(".github") / "MAJOR_RELEASE_APPROVED"
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_PROJECT_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"')
_TAG_RE = re.compile(r"^v(\d+)\.\d+\.\d+$")


def _major(version: str) -> int | None:
    match = _VERSION_RE.match(version.strip())
    return int(match.group(1)) if match else None


def pyproject_major(root: Path) -> int | None:
    """Return the major of ``project.version`` in ``root/pyproject.toml``."""

    # A line scan, not tomllib: the test tier runs on Python 3.10.
    in_project = False
    for line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
        elif in_project and (match := _PROJECT_VERSION_RE.match(stripped)):
            return _major(match.group(1))
    return None


def highest_tag_major(root: Path) -> int | None:
    """Return the highest major among ``vX.Y.Z`` tags, or None without tags."""

    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "tag", "--list", "v*"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    majors = [int(m.group(1)) for line in out.split() if (m := _TAG_RE.match(line))]
    return max(majors) if majors else None


def evaluate(next_version: str, current_major: int, approval_text: str | None) -> tuple[int, str]:
    """Decide the guard outcome; return ``(exit_code, message)``."""

    target = next_version.strip()
    next_major = _major(target)
    if next_major is None:
        return 1, f"computed next version {target!r} is not X.Y.Z; refusing (fail closed)"
    if next_major <= current_major:
        return 0, f"next version {target} keeps major {current_major}; not a major bump"
    if approval_text is not None and approval_text.rstrip("\n") == target:
        return 0, f"major bump to {target} approved by {APPROVAL_RELPATH}"
    detail = (
        "no approval file"
        if approval_text is None
        else f"approval file names {approval_text.strip()!r}, not {target!r}"
    )
    return 1, (
        f"REFUSING major bump: next version {target} > current major {current_major} "
        f"({detail}). Only the maintainer's explicit command authorizes creating "
        f"{APPROVAL_RELPATH} containing exactly {target}."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next", required=True, dest="next_version")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    root: Path = args.root
    majors = [m for m in (pyproject_major(root), highest_tag_major(root)) if m is not None]
    if not majors:
        print("[check_release_major] cannot determine the current major", file=sys.stderr)
        return 2
    approval = root / APPROVAL_RELPATH
    approval_text = approval.read_text(encoding="utf-8") if approval.is_file() else None
    code, message = evaluate(args.next_version, min(majors), approval_text)
    print(f"[check_release_major] {message}", file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())

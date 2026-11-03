#!/usr/bin/env python3
"""Release-time hard stop: refuse a MAJOR version without written approval.

Runs in the ``build`` job of ``.github/workflows/release.yml`` on every pull
request to main and on every pushed ``vX.Y.Z`` tag, before anything is built;
``publish`` needs that job, so a refusal stops the upload. It checks every
candidate version: the ``project.version`` in ``pyproject.toml``, the
``__version__`` in ``torchlens/__init__.py`` and, with ``--next``, the tag being
released. A candidate whose major is greater than the current major is refused
unless the tracked approval file ``.github/MAJOR_RELEASE_APPROVED`` exists and
its content is exactly that candidate (one trailing newline allowed).

The current major is the LOWEST of: the highest major among ``vX.Y.Z`` tags
that do not contain ``HEAD`` (the tag being released and any later tag are not
part of the baseline), the ``pyproject.toml`` major, and, with ``--released`` or
``--released-from-pypi``, the highest major actually published. So neither a
new tag, a stray earlier tag whose upload was refused, nor a hand edit of the
version files can raise it. With ``--require-tags`` (the release workflow) a
checkout without such a tag fails closed instead of falling back to the version
files; a failed PyPI lookup also fails closed.

Only the maintainer's explicit command authorizes creating the approval file.

Usage::

    python scripts/check_release_major.py --require-tags --released-from-pypi \
        [--next 2.38.0] [--released 2.37.0] [--root PATH]

Exit codes:
    0  -- no candidate is a major bump, or each major bump is approved for exactly that version
    1  -- an unapproved major bump, or an unparsable candidate version (fail closed)
    2  -- the current major could not be determined
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

APPROVAL_RELPATH = Path(".github") / "MAJOR_RELEASE_APPROVED"
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)([A-Za-z0-9.+-]*)$")
_PROJECT_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"')
_INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)
_TAG_RE = re.compile(r"^v(\d+)\.\d+\.\d+$")
_PYPI_JSON_URL = "https://pypi.org/pypi/torchlens/json"


def _major(version: str) -> int | None:
    match = _VERSION_RE.match(version.strip())
    return int(match.group(1)) if match else None


def pyproject_version(root: Path) -> str | None:
    """Return ``project.version`` from ``root/pyproject.toml``."""

    # A line scan, not tomllib: the test tier runs on Python 3.10.
    in_project = False
    for line in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
        elif in_project and (match := _PROJECT_VERSION_RE.match(stripped)):
            return match.group(1)
    return None


def init_version(root: Path) -> str | None:
    """Return ``__version__`` from ``root/torchlens/__init__.py`` if the file exists."""

    init = root / "torchlens" / "__init__.py"
    if not init.is_file():
        return None
    match = _INIT_VERSION_RE.search(init.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def released_tag_major(root: Path) -> int | None:
    """Return the highest major among ``vX.Y.Z`` tags that do not contain HEAD."""

    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "tag", "--list", "v*", "--no-contains", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    majors = [int(m.group(1)) for line in out.split() if (m := _TAG_RE.match(line))]
    return max(majors) if majors else None


def pypi_released_major(url: str = _PYPI_JSON_URL) -> int | None:
    """Return the highest major among the releases PyPI lists, or None on any failure."""

    try:
        with urlopen(url, timeout=30) as response:  # noqa: S310 (fixed https URL)
            releases = json.load(response)["releases"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    majors = [m for m in (_major(str(version)) for version in releases) if m is not None]
    return max(majors) if majors else None


def evaluate(candidate: str, current_major: int, approval_text: str | None) -> tuple[int, str]:
    """Decide the guard outcome for one candidate; return ``(exit_code, message)``."""

    target = candidate.strip()
    next_major = _major(target)
    if next_major is None:
        return 1, f"version {target!r} is not X.Y.Z; refusing (fail closed)"
    if next_major <= current_major:
        return 0, f"version {target} keeps major {current_major}; not a major bump"
    if approval_text is not None and approval_text.rstrip("\n") == target:
        return 0, f"major bump to {target} approved by {APPROVAL_RELPATH}"
    detail = (
        "no approval file"
        if approval_text is None
        else f"approval file names {approval_text.strip()!r}, not {target!r}"
    )
    return 1, (
        f"REFUSING major bump: version {target} > current major {current_major} "
        f"({detail}). Only the maintainer's explicit command authorizes creating "
        f"{APPROVAL_RELPATH} containing exactly {target}."
    )


def current_major(root: Path, *, require_tags: bool, released: int | None = None) -> int | None:
    """Return the baseline major, or None when it cannot be determined."""

    tag_major = released_tag_major(root)
    if tag_major is None and require_tags:
        return None
    project = pyproject_version(root)
    project_major = _major(project) if project else None
    majors = [m for m in (tag_major, project_major, released) if m is not None]
    return min(majors) if majors else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next", action="append", default=[], dest="next_versions")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--require-tags", action="store_true")
    parser.add_argument("--released", default=None, help="highest published version")
    parser.add_argument("--released-from-pypi", action="store_true")
    args = parser.parse_args(argv)

    root: Path = args.root
    released: int | None = None
    if args.released is not None:
        released = _major(args.released)
        if released is None:
            print(
                f"[check_release_major] --released {args.released!r} is not X.Y.Z; refusing",
                file=sys.stderr,
            )
            return 2
    if args.released_from_pypi:
        pypi_major = pypi_released_major()
        if pypi_major is None:
            print("[check_release_major] PyPI release lookup failed; refusing", file=sys.stderr)
            return 2
        released = pypi_major if released is None else min(released, pypi_major)
    baseline = current_major(root, require_tags=args.require_tags, released=released)
    if baseline is None:
        print(
            "[check_release_major] cannot determine the current major "
            "(no vX.Y.Z tag below HEAD; fetch tags); refusing",
            file=sys.stderr,
        )
        return 2
    candidates = [v for v in (pyproject_version(root), init_version(root)) if v is not None]
    candidates += [v.removeprefix("v") for v in args.next_versions]
    if not candidates:
        print("[check_release_major] no version to check; refusing", file=sys.stderr)
        return 1
    approval = root / APPROVAL_RELPATH
    approval_text = approval.read_text(encoding="utf-8") if approval.is_file() else None
    code = 0
    for candidate in candidates:
        result, message = evaluate(candidate, baseline, approval_text)
        print(f"[check_release_major] {message}", file=sys.stderr if result else sys.stdout)
        code = max(code, result)
    return code


if __name__ == "__main__":
    sys.exit(main())

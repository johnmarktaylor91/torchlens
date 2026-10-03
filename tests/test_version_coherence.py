"""Version coherence: pyproject.toml and torchlens.__version__ must agree.

Releases are cut by python-semantic-release, which rewrites both declared
version strings together -- but a manual edit desyncs them and ships in every
source install until the next release. Nothing else pins them equal.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import torchlens

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_version_matches_package_version() -> None:
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml not present (installed-package run)")
    # Regex, not tomllib: the declared floor is python 3.10, where tomllib is absent.
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        pyproject.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert match is not None, 'pyproject.toml has no version = "..." line'
    declared = match.group(1)
    assert declared == torchlens.__version__, (
        f"pyproject.toml declares {declared} but torchlens.__version__ is "
        f"{torchlens.__version__}; python-semantic-release updates both -- a manual "
        "edit must keep them in lockstep"
    )

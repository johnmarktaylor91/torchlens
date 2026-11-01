"""The shipped backend scorecards are packaged, one pattern for every backend.

``tl.report.doctor()`` reads ``torchlens/backends/<name>/scorecard.json`` from
the installed package (``torchlens/backends/_scorecard_rows.py``). Without a
package-data pattern an installed wheel ships no record and every backend row
reads ``not_tested``; without the MANIFEST.in line the sdist drops them too.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from torchlens.backends._scorecard_rows import SCORECARD_FILE_NAME, shipped_scorecard_path
from torchlens.backends.registry import registered_backend_specs

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_DATA_PATTERN = "backends/*/scorecard.json"
_MANIFEST_LINE = "recursive-include torchlens/backends scorecard.json"

# What a wheel build needs from the tree; the copy keeps the build off the
# repository (no stray build/ or dist/ in the checkout, no dependence on which
# scorecards happen to be committed).
_BUILD_INPUTS = ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE", "NOTICE")


def _package_data_block() -> str:
    # Regex on purpose: tomllib is 3.11+ and the suite's floor row runs 3.10.
    text = (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"\[tool\.setuptools\.package-data\]\s*\n(?P<body>.*?)(?=\n\[)", text, re.S)
    assert match is not None, "pyproject.toml has no [tool.setuptools.package-data] table"
    return match.group("body")


@pytest.mark.smoke
def test_scorecard_package_data_pattern_covers_every_backend() -> None:
    """One wheel pattern and one sdist line name every backend's scorecard."""

    assert f'"{_PACKAGE_DATA_PATTERN}"' in _package_data_block(), (
        f"[tool.setuptools.package-data] lacks {_PACKAGE_DATA_PATTERN!r}: an installed wheel "
        "ships no backend scorecard and tl.report.doctor() reads every backend as not_tested"
    )
    manifest = (_PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    assert _MANIFEST_LINE in manifest, f"MANIFEST.in lacks {_MANIFEST_LINE!r}"

    # The reader and the pattern agree on where a record lives, for every
    # registered backend (a backend named later matches the same glob).
    package_root = Path(importlib.util.find_spec("torchlens").origin or "").resolve().parent
    for spec in registered_backend_specs():
        relative = shipped_scorecard_path(spec.name).relative_to(package_root).as_posix()
        assert relative == f"backends/{spec.name}/{SCORECARD_FILE_NAME}"
        assert Path(relative).match(_PACKAGE_DATA_PATTERN), relative


def _wheel_build_command(source: Path, wheel_dir: Path) -> list[str]:
    """``python -m build`` when importable, else ``pip wheel``; see test_packaging_diet."""

    try:
        has_build_main = importlib.util.find_spec("build.__main__") is not None
    except ModuleNotFoundError:
        has_build_main = False
    if has_build_main:
        return [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir), str(source)]
    if importlib.util.find_spec("pip") is not None:
        return [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(source),
            "--no-deps",
            "-w",
            str(wheel_dir),
        ]
    pytest.skip("neither build nor pip is importable for wheel construction")


def _copy_build_tree(destination: Path) -> None:
    """Copy the library and build inputs, never the tests or the git metadata."""

    shutil.copytree(
        _PROJECT_ROOT / "torchlens",
        destination / "torchlens",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in _BUILD_INPUTS:
        source = _PROJECT_ROOT / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _plant_missing_scorecards(tree: Path) -> dict[str, bool]:
    """Give every registered backend a record; return which were real (committed)."""

    real: dict[str, bool] = {}
    for spec in registered_backend_specs():
        target = tree / "torchlens" / "backends" / spec.name / SCORECARD_FILE_NAME
        real[spec.name] = target.is_file()
        if not real[spec.name]:
            target.parent.mkdir(parents=True, exist_ok=True)
            placeholder = {
                "scorecard_version": "packaging-test",
                "backend": spec.name,
                "cells": [],
                "counts": {},
                "highest_tier": "",
            }
            target.write_text(json.dumps(placeholder), encoding="utf-8")
    return real


@pytest.mark.slow
def test_built_wheel_ships_every_backend_scorecard(tmp_path: Path) -> None:
    """A wheel built from the tree carries torch's scorecard and every other backend's.

    The build runs on a copy so the result does not depend on which records are
    committed at the time: a backend without one gets a placeholder record first,
    and the assertion is about the package-data pattern, not the record's body.
    """

    source = tmp_path / "src"
    _copy_build_tree(source)
    real = _plant_missing_scorecards(source)
    assert "torch" in real, "the torch backend is not registered"

    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    # Neutral cwd with an explicit srcdir: `python -m build` puts the cwd on
    # sys.path[0], and a setuptools build/ directory there shadows the module.
    subprocess.run(_wheel_build_command(source, wheel_dir), cwd=tmp_path, check=True)
    wheels = sorted(wheel_dir.glob("torchlens-*.whl"))
    assert len(wheels) == 1, wheels

    with zipfile.ZipFile(wheels[0]) as wheel_zip:
        members = set(wheel_zip.namelist())

    missing = [
        name for name in real if f"torchlens/backends/{name}/{SCORECARD_FILE_NAME}" not in members
    ]
    assert not missing, (
        f"wheel lacks the scorecard of backend(s) {missing}; the package-data pattern "
        f"{_PACKAGE_DATA_PATTERN!r} must cover every torchlens/backends/<name>/ package"
    )
    assert f"torchlens/backends/torch/{SCORECARD_FILE_NAME}" in members

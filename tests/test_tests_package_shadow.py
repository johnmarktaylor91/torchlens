"""The name ``tests`` resolves to this directory even when site-packages ships one.

The repository's ``tests`` is a namespace package, and a regular ``tests`` package
anywhere on ``sys.path`` wins over a namespace package. Installed peers do ship one
(brainscore_vision 2.4.4 put a top-level ``tests`` package into site-packages), and
then every ``from tests.support ...`` import failed with ``ModuleNotFoundError``.
``tests/conftest.py`` binds ``sys.modules["tests"]`` to this directory first; these
checks plant a fake regular ``tests`` package ahead on ``sys.path`` in a child process
and run a real pytest session against it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import capability_routes

# Each child is a full pytest session (torch, torchlens and the root conftest).
pytestmark = pytest.mark.heavy

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
_THIS_FILE = Path(__file__).resolve().relative_to(_REPO_ROOT).as_posix()
_FAKE_MARKER = "planted-foreign-tests-package"


def _plant_foreign_tests_package(root: Path) -> Path:
    """Write a regular ``tests`` package, as a peer's wheel would, under ``root``."""

    package = root / "tests"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"MARKER = {_FAKE_MARKER!r}\n", encoding="utf-8")
    return root


def _child_env(planted: Path) -> dict[str, str]:
    """Return this process's environment with ``planted`` first on ``PYTHONPATH``."""

    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_XDIST")}
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(planted), env.get("PYTHONPATH", "")]))
    return env


def test_tests_package_is_this_directory() -> None:
    """``tests`` and ``tests.support`` are this repository's, not a planted package's."""

    package = sys.modules["tests"]
    assert getattr(package, "MARKER", None) is None
    assert _TESTS_DIR in [Path(entry).resolve() for entry in package.__path__]
    module_file = Path(capability_routes.__file__ or "").resolve()
    assert module_file == _TESTS_DIR / "support" / "capability_routes.py"


def test_the_planted_package_shadows_tests_without_the_conftest(tmp_path: Path) -> None:
    """Precondition: outside pytest, the planted package wins over the repository's."""

    planted = _plant_foreign_tests_package(tmp_path / "site")
    probe = "import sys, tests; print(getattr(tests, 'MARKER', None))"
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(_REPO_ROOT)!r}); {probe}"],
        cwd=tmp_path,
        env=_child_env(planted),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == _FAKE_MARKER


@pytest.mark.parametrize(
    "early_plugin",
    [[], ["-p", "tests"]],
    ids=["planted-on-path", "planted-already-imported"],
)
def test_pytest_resolves_tests_support_past_a_planted_package(
    tmp_path: Path, early_plugin: list[str]
) -> None:
    """A pytest session with a planted ``tests`` package still imports ours.

    The second case loads the planted package as an early plugin, so it is in
    ``sys.modules`` before the root conftest runs and must be replaced, not kept.
    """

    planted = _plant_foreign_tests_package(tmp_path / "site")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *early_plugin,
            f"{_THIS_FILE}::test_tests_package_is_this_directory",
            "-p",
            "no:xdist",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            f"--basetemp={tmp_path / 'basetemp'}",
            "-q",
        ],
        cwd=_REPO_ROOT,
        env=_child_env(planted),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    assert "1 passed" in result.stdout, result.stdout[-4000:]

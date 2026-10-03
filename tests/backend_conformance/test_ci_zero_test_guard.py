"""Zero-executed-tests CI guard conformance.

The preview backend CI legs guard optional-dependency suites: with the
framework missing, every test importorskips away and pytest exits 0, leaving
a leg green while covering nothing (sol review, 2026-08-11). The nightly
workflow closes this with ``scripts/check_ci_executed_tests.py``; these tests
simulate the failure mode end-to-end against that exact script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import torchlens as tl

pytestmark = pytest.mark.backend_parity

_REPO_ROOT = Path(tl.__file__).resolve().parent.parent
_GUARD = _REPO_ROOT / "scripts" / "check_ci_executed_tests.py"


def _run_guard(junit_path: Path, floor: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_GUARD), str(junit_path), str(floor)],
        capture_output=True,
        text=True,
    )


def _write_junit(path: Path, *, tests: int, skipped: int) -> None:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<testsuites>"
        f'<testsuite name="pytest" tests="{tests}" skipped="{skipped}" '
        'errors="0" failures="0" />'
        "</testsuites>",
        encoding="utf-8",
    )


def test_guard_fails_fully_skipped_suite(tmp_path: Path) -> None:
    """An all-skipped junit report (sol's missing-TF reproduction) must fail."""

    junit = tmp_path / "all-skipped.xml"
    _write_junit(junit, tests=6, skipped=6)
    result = _run_guard(junit, 1)
    assert result.returncode == 1
    assert "covering nothing" in result.stderr


def test_guard_fails_missing_report(tmp_path: Path) -> None:
    """A missing junit report (pytest never ran) must fail, not pass vacuously."""

    result = _run_guard(tmp_path / "never-written.xml", 1)
    assert result.returncode == 1


def test_guard_passes_executed_suite(tmp_path: Path) -> None:
    """A suite with executed tests above the floor passes."""

    junit = tmp_path / "executed.xml"
    _write_junit(junit, tests=6, skipped=2)
    result = _run_guard(junit, 1)
    assert result.returncode == 0


def test_guard_end_to_end_importorskip_simulation(tmp_path: Path) -> None:
    """Full CI simulation: a real pytest run whose only test importorskips a
    missing framework exits 0, and the guard is what catches it."""

    # importorskip at call time mirrors the real legs: the suite COLLECTS
    # (so pytest exits 0, not 5) and every test skips.
    (tmp_path / "test_preview_stub.py").write_text(
        "import pytest\n"
        "def test_never_runs() -> None:\n"
        '    pytest.importorskip("torchlens_nonexistent_framework_xyz")\n'
        "    assert True\n",
        encoding="utf-8",
    )
    junit = tmp_path / "sim.junit.xml"
    pytest_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test_preview_stub.py",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert pytest_run.returncode == 0, pytest_run.stdout + pytest_run.stderr
    result = _run_guard(junit, 1)
    assert result.returncode == 1


def test_nightly_legs_declare_meaningful_executed_floors() -> None:
    """Every preview matrix leg carries an executed-test floor well above 1.

    A floor of 1 lets a 49-of-50 skip stay green, adding nothing beyond the
    import sentinel; each leg must pin a floor sized to its real suite.
    """

    import re

    workflow = (_REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text(encoding="utf-8")
    backends = re.findall(r"- backend: (\w+)", workflow)
    floors = [int(value) for value in re.findall(r"executed_floor: (\d+)", workflow)]
    assert sorted(backends) == ["jax", "mlx", "paddle", "tf", "tinygrad"]
    assert len(floors) == len(backends), "every leg must declare executed_floor"
    assert all(floor >= 10 for floor in floors), floors
    assert '"${{ matrix.executed_floor }}"' in workflow


def test_passed_ids_floor_accepts_path_spelled_node_ids(tmp_path: Path) -> None:
    """The exact-passed-ID floor matches pytest path ids against junit classnames.

    The RG floor file spells nodes ``tests/pkg/test_mod.py::test_x`` (what
    ``--collect-only`` prints and the manifest lint checks); the junit report
    spells the same node ``classname="tests.pkg.test_mod" name="test_x"``. The
    check must join the two spellings, and a genuinely missing or failed node
    must still be named.
    """

    junit = tmp_path / "rg.junit.xml"
    junit.write_text(
        '<testsuites><testsuite name="pytest" tests="2" skipped="0" failures="1" errors="0">'
        '<testcase classname="tests.pkg.test_mod" name="test_ok" time="0.1"/>'
        '<testcase classname="tests.pkg.test_mod" name="test_red" time="0.1">'
        "<failure>boom</failure></testcase>"
        "</testsuite></testsuites>"
    )
    ids = tmp_path / "ids.txt"
    ids.write_text(
        "# floor\ntests/pkg/test_mod.py::test_ok\ntests/pkg/test_mod.py::test_red\n"
        "tests/pkg/test_mod.py::test_absent\n"
    )
    result = subprocess.run(
        [sys.executable, str(_GUARD), str(junit), "0", "--passed-ids", str(ids)],
        capture_output=True,
        text=True,
        check=False,
    )
    report = result.stdout + result.stderr
    assert result.returncode == 1, report
    assert "tests/pkg/test_mod.py::test_red" in report
    assert "tests/pkg/test_mod.py::test_absent" in report
    assert "test_ok" not in report.split("did not PASS", 1)[1]

    ids.write_text("tests/pkg/test_mod.py::test_ok\n")
    result = subprocess.run(
        [sys.executable, str(_GUARD), str(junit), "0", "--passed-ids", str(ids)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

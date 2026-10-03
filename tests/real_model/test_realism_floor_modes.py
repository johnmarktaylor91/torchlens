"""Red-capability plants for the floor script's PASSED modes (memo A6/D4).

``--min-passed`` and ``--passed-ids`` exist because ``executed = total -
skipped`` counts failures as executed -- unsound on any non-blocking leg.
Every capability here is proven in BOTH directions: green on an honest
junit, red on each planted degradation (failure counted, skipped required
node, renamed node, absent node).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.real_model

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_ci_executed_tests.py"

GREEN_JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="0" skipped="0" tests="3">
<testcase classname="tests.pkg.test_mod" name="test_a"/>
<testcase classname="tests.pkg.test_mod" name="test_b"/>
<testcase classname="tests.pkg.test_mod" name="test_c"/>
</testsuite></testsuites>
"""

# One failure, one skip: executed = 2 (failure counts!), passed = 1.
DEGRADED_JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="0" failures="1" skipped="1" tests="3">
<testcase classname="tests.pkg.test_mod" name="test_a"/>
<testcase classname="tests.pkg.test_mod" name="test_b"><failure message="boom"/></testcase>
<testcase classname="tests.pkg.test_mod" name="test_c"><skipped message="gone"/></testcase>
</testsuite></testsuites>
"""


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


@pytest.fixture()
def junits(tmp_path):
    green = tmp_path / "green.xml"
    green.write_text(GREEN_JUNIT)
    degraded = tmp_path / "degraded.xml"
    degraded.write_text(DEGRADED_JUNIT)
    return green, degraded


def test_min_passed_green_and_red(junits):
    green, degraded = junits
    assert _run(str(green), "0", "--min-passed", "3").returncode == 0
    # THE unsound case: the executed floor tolerates the failure...
    assert _run(str(degraded), "2").returncode == 0
    # ...and the passed floor does not.
    result = _run(str(degraded), "0", "--min-passed", "2")
    assert result.returncode == 1
    assert "passed-count check FAILED" in result.stderr


def test_passed_ids_exact_floor(junits, tmp_path):
    green, degraded = junits
    ids = tmp_path / "ids.txt"
    ids.write_text(
        "# comment\n"
        "tests.pkg.test_mod::test_a\n"
        "tests.pkg.test_mod::test_b\n"
        "tests.pkg.test_mod::test_c\n"
    )
    assert _run(str(green), "0", "--passed-ids", str(ids)).returncode == 0
    # Planted reds: a failed node and a skipped node both break the floor.
    result = _run(str(degraded), "0", "--passed-ids", str(ids))
    assert result.returncode == 1
    assert "test_b" in result.stderr and "test_c" in result.stderr
    # A renamed/absent node breaks it even on an all-green junit.
    ids.write_text("tests.pkg.test_mod::test_renamed\n")
    result = _run(str(green), "0", "--passed-ids", str(ids))
    assert result.returncode == 1
    assert "test_renamed" in result.stderr


def test_positional_interface_unchanged(junits):
    green, degraded = junits
    assert _run(str(green), "3").returncode == 0
    assert _run(str(green), "4").returncode == 1
    assert _run(str(degraded), "2", "0.5").returncode == 0
    assert _run(str(degraded), "2", "0.1").returncode == 1


@pytest.mark.parametrize(
    ("manifest_name", "expected_count"),
    [("canary_passed_ids.txt", 1), ("r1_core_passed_ids.txt", 3)],
)
def test_passed_id_manifests_name_real_collectible_nodes(manifest_name, expected_count):
    """The committed passed-ID manifests stay in lockstep with the suite."""

    manifest = REPO_ROOT / "tests" / "real_model" / "r1" / manifest_name
    required = [
        line.strip()
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(required) == expected_count
    source = (
        REPO_ROOT / "tests" / "real_model" / "r1" / "test_realism_pretrained_core.py"
    ).read_text()
    for node_id in required:
        classname, name = node_id.rsplit("::", 1)
        assert classname == "tests.real_model.r1.test_realism_pretrained_core"
        assert f"def {name}(" in source, f"manifest names a nonexistent test: {name}"

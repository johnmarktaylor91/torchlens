"""U-gate lockstep pins (AUD-CODE 0.3 companion).

``tests/test_gate_infra_floor_drift.py`` governs the tests.yml floor literal
against the live smoke selection, but the captain's ``sprint/tools/u_gate.sh``
carries its OWN copy of the literal, which drifted identically (2830 against
a 9218-test tier). The two literals stay equal here, and the opt-in backstop
stage stays documented and gated on ``U_GATE_BACKSTOP=1``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parent.parent
_U_GATE = _REPO_ROOT / "sprint" / "tools" / "u_gate.sh"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "tests.yml"
_WORKFLOW_FLOOR_RE = re.compile(
    r"check_ci_executed_tests\.py\s+\"\$\{\{ runner\.temp \}\}/smoke\.junit\.xml\"\s+(\d+)\s+0\.15"
)


def _u_gate_text() -> str:
    if not _U_GATE.exists():
        pytest.skip("sprint/tools/u_gate.sh is not shipped in this checkout")
    return _U_GATE.read_text()


def test_u_gate_floor_matches_the_workflow_floor() -> None:
    gate_match = re.search(r"^FLOOR=(\d+)$", _u_gate_text(), re.MULTILINE)
    assert gate_match, "u_gate.sh lost its FLOOR literal"
    workflow_matches = _WORKFLOW_FLOOR_RE.findall(_WORKFLOW.read_text())
    assert len(workflow_matches) == 1
    assert int(gate_match.group(1)) == int(workflow_matches[0]), (
        "u_gate.sh FLOOR and the tests.yml smoke floor must be re-trued together"
    )


def test_backstop_stage_is_opt_in_and_documented() -> None:
    text = _u_gate_text()
    assert "U_GATE_BACKSTOP=1" in text.split("set -u")[0], "the header must document the knob"
    stage = re.search(
        r'if \[ "\$\{U_GATE_BACKSTOP:-0\}" = "1" \]; then\s*\n\s*run_stage backstop .*?'
        r"-m 'not rare and not slow and not heavy'",
        text,
        re.DOTALL,
    )
    assert stage, "the backstop stage must run the mid selection only under U_GATE_BACKSTOP=1"
    assert "--ignore-glob='*menagerie*' --ignore=tests/crawler" in stage.group(0) or (
        "--ignore-glob='*menagerie*'" in text.split("run_stage backstop", 1)[1]
    )


def test_u_gate_script_parses() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash unavailable")
    _u_gate_text()
    completed = subprocess.run([bash, "-n", str(_U_GATE)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

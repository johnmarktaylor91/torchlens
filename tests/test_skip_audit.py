"""``python -O`` leg self-verification for the ``requires_assertions`` marker.

The ``requires_assertions`` marker exists so assert-based audits skip (not
silently pass) under ``-O``; no CI workflow runs that leg, so this file proves
the contract in-suite with one targeted subprocess: marked tests SKIP under
``-O`` while a plain sentinel test executes.

The skip-surface ledgers and scanners (importorskip, unconditional skips,
unsatisfiable and device-gated skipifs, tripwire classification) live in
``scripts/checks/skip_audit.py``; this probe drives pytest and the conftest
hook, so it stays a pytest test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# NOTE: no module-level smoke pytestmark -- the -O subprocess test below is
# `slow`, and the tier markers are additive/disjoint (tests/test_marker_lint.py).
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def test_o_leg_sentinel_executes() -> None:
    """Trivial sentinel proving the ``-O`` subprocess really executes tests.

    Deliberately assert-free (``-O`` strips assert statements): failure is an
    explicit ``pytest.fail``.
    """

    if (1, 2)[0] != 1:  # pragma: no cover - arithmetic sanity sentinel
        pytest.fail("sentinel arithmetic failed")


@pytest.mark.slow
def test_requires_assertions_marker_engages_under_python_O(tmp_path: Path) -> None:
    """``requires_assertions`` tests SKIP (never pass/fail) under ``python -O``.

    No CI workflow runs a ``-O`` leg, so without this subprocess probe the
    marker and its conftest hook would rot unverified. One targeted run under
    ``-O`` proves both directions: the marked tests are collected and SKIPPED
    with the documented reason, while an ordinary test executes and passes on
    the same interpreter. Red-capable end to end: removing the conftest hook
    makes the marked assert-based tests FAIL under ``-O`` (stripped asserts
    never raise), and removing the marker breaks the exact skip count.
    """

    marked_nodes = [
        "tests/test_postprocess_contract_arming.py::test_unknown_step_contract_is_rejected",
        "tests/test_postprocess_contract_arming.py::test_undeclared_write_is_rejected",
    ]
    plain_node = "tests/test_skip_audit.py::test_o_leg_sentinel_executes"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(tmp_path / "oleg-bt"),
            *marked_nodes,
            plain_node,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"-O leg subprocess failed:\n{output}"
    assert "1 passed" in output, f"plain sentinel did not execute under -O:\n{output}"
    assert "2 skipped" in output, (
        f"expected exactly the two requires_assertions tests to skip under -O:\n{output}"
    )
    assert "requires assertions" in output, (
        f"skip reason does not name the requires_assertions contract:\n{output}"
    )

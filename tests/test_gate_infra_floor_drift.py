"""Floor-drift tripwire for the smoke executed-test floor (megaplan D13).

The smoke leg of ``.github/workflows/tests.yml`` attests execution with
``check_ci_executed_tests.py <junit> <floor> 0.15`` where ``<floor>`` is a
hand-written literal intended to sit at roughly HALF the smoke selection. That
literal has drifted twice (r7 R41: 1,900 against ~3,900; 2026-08-26: 2,500
against 5,659 -- a tolerated 56% hollow-out), because nothing re-derived it
from the live tier. This test re-derives the floor from the actual selection
count so the drift class goes red instead of waiting for the next audit.

Contract: the workflow's literal floor must sit within [0.45, 0.55] of the
live ``-m smoke`` selection count (the same selection expression the workflow
runs). When this fails, re-true the literal in tests.yml to ~half the printed
selection count -- do NOT widen the band; the band IS the ~50% intent the
inline comment has always claimed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow]  # full-suite collection is ~1 min; never smoke-tier

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

# The one smoke-leg attestation line this test governs.
FLOOR_RE = re.compile(
    r"check_ci_executed_tests\.py\s+\"\$\{\{ runner\.temp \}\}/smoke\.junit\.xml\"\s+(\d+)\s+0\.15"
)


def _workflow_floor() -> int:
    text = WORKFLOW.read_text()
    matches = FLOOR_RE.findall(text)
    assert len(matches) == 1, (
        f"expected exactly one smoke executed-floor attestation in {WORKFLOW}, "
        f"found {len(matches)}; keep this regex and the workflow in lockstep"
    )
    return int(matches[0])


def _live_smoke_selection_count() -> int:
    """Count tests the smoke leg's own selection expression selects today."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-m",
        "smoke",
        "--ignore-glob=*menagerie*",
        "--ignore=tests/crawler",
        "--collect-only",
        "-q",
        "-p",
        "no:randomly",
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900, check=False
    )
    # Summary line: "5659/15307 tests collected (9648 deselected) in 51.60s"
    # or "5659 tests collected in 51.60s" when nothing is deselected.
    match = re.search(r"(\d+)(?:/\d+)? tests collected", proc.stdout)
    assert match, (
        "could not parse the smoke selection count from collect-only output; "
        f"exit={proc.returncode}\nstdout tail:\n{proc.stdout[-2000:]}"
        f"\nstderr tail:\n{proc.stderr[-2000:]}"
    )
    return int(match.group(1))


def test_smoke_executed_floor_tracks_selection() -> None:
    floor = _workflow_floor()
    selected = _live_smoke_selection_count()
    assert selected > 0
    ratio = floor / selected
    assert 0.45 <= ratio <= 0.55, (
        f"smoke executed-floor drift: tests.yml floor {floor} is {ratio:.1%} of the live "
        f"smoke selection ({selected} tests). The floor's contract is ~50% of the tier "
        f"(band 45-55%). Re-true the literal in .github/workflows/tests.yml to "
        f"~{selected // 2} (and update its inline comment); do NOT widen this band."
    )

"""Live drift tripwire for AGENTS.md's documented test-tier counts.

Split out of ``test_docs_lockstep_names.py`` (which is smoke-tier; this run
costs a full collect-only, so it is ``heavy``). The pre-r3 lockstep gate
hard-asserted the literal doc string, so once the counts drifted the gate
ENFORCED the stale claim (R41/R81/R88, 3-lab). This test parses the claim and
compares it against a live collection instead: beyond the tolerance the fix
is refreshing AGENTS.md's dated tier record — never widening the tolerance.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from test_docs_lockstep_names import TIER_CLAIM_RE

# slow, not heavy: the live collect-only subprocess pays the root conftest
# torch import plus a full 12k-test collection -- 46s charged in the merged
# full not-slow session and 77s isolated, an order past the 20s heavy ceiling.
pytestmark = pytest.mark.slow

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_documented_smoke_count_tracks_live_collection() -> None:
    """The documented collect-only counts must stay within 10% of reality."""

    guide = (_REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claim = TIER_CLAIM_RE.search(guide)
    assert claim is not None, "structured tier claim missing (see the lockstep test)"
    documented_smoke = int(claim.group("smoke").replace(",", ""))
    documented_total = int(claim.group("total").replace(",", ""))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--collect-only",
            "-q",
            "-m",
            "smoke",
            "-p",
            "no:randomly",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    match = re.search(r"(\d+)/(\d+) tests collected", proc.stdout)
    assert match is not None, f"could not parse collect-only output:\n{proc.stdout[-2000:]}"
    live_smoke, live_total = int(match.group(1)), int(match.group(2))

    drift_smoke = abs(live_smoke - documented_smoke) / documented_smoke
    drift_total = abs(live_total - documented_total) / documented_total
    assert drift_smoke <= 0.10 and drift_total <= 0.10, (
        f"AGENTS.md's tier record is stale: documents {documented_smoke}/"
        f"{documented_total} but live collection is {live_smoke}/{live_total} "
        f"(drift {drift_smoke:.1%}/{drift_total:.1%}). Update the dated claim "
        "in AGENTS.md 'Testing Tiers' with fresh measured numbers"
    )

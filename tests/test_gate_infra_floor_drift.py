"""Floor-drift tripwire for the smoke executed-test floor (megaplan D13).

The smoke legs of ``.github/workflows/pr.yml`` and ``nightly.yml`` declare
their executed-test floor as ``smoke.junit.xml tier:smoke 0.15``: the floor is
half the smoke count recorded in the test-tier collection artifact
(``tests/tiers/collection.json``), resolved by
``scripts/check_ci_executed_tests.py``. Hand-written literals drifted twice
(r7 R41: 1,900 against ~3,900; 2026-08-26: 2,500 against 5,659 -- a tolerated
56% hollow-out) because nothing re-derived them from the live tier; the
artifact can drift the same way, so this test re-derives the floor from the
live selection count.

Contract: the resolved floor must sit within [0.45, 0.55] of the live
``-m smoke`` selection count (the same selection expression the workflow
runs). When this fails, regenerate the artifact with
``tools/tiers/collection.py`` -- do NOT widen the band; the band IS the ~50%
intent the floor has always claimed.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.slow]  # full-suite collection is ~1 min; never smoke-tier

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    REPO_ROOT / ".github" / "workflows" / "pr.yml",
    REPO_ROOT / ".github" / "workflows" / "nightly.yml",
)

# The one smoke-leg floor declaration per workflow this test governs.
FLOOR_RE = re.compile(r'echo "smoke\.junit\.xml (\S+) 0\.15"')


def _load_floor_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_ci_executed_tests", REPO_ROOT / "scripts" / "check_ci_executed_tests.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workflow_floor() -> int:
    tokens = set()
    for workflow in WORKFLOWS:
        matches = FLOOR_RE.findall(workflow.read_text())
        assert len(matches) == 1, (
            f"expected exactly one smoke executed-floor declaration in {workflow}, "
            f"found {len(matches)}; keep this regex and the workflow in lockstep"
        )
        tokens.add(matches[0])
    assert tokens == {"tier:smoke"}, (
        f"the smoke legs must take their floor from the collection artifact: {tokens}"
    )
    return _load_floor_checker().resolve_floor("tier:smoke")


@functools.lru_cache(maxsize=1)
def _live_smoke_selection_count() -> int:
    """Count tests the smoke leg's own selection expression selects today."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-m",
        "smoke",
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
        f"smoke executed-floor drift: workflow floor {floor} is {ratio:.1%} of the live "
        f"smoke selection ({selected} tests). The floor's contract is ~50% of the tier "
        f"(band 45-55%). Regenerate tests/tiers/collection.json with "
        f"tools/tiers/collection.py; do NOT widen this band."
    )


def test_smoke_selection_stays_under_size_ceiling() -> None:
    """The live whole-tree smoke selection stays at or under the tier's ceiling."""
    from tests.conftest import SMOKE_TIER_SIZE_CEILING

    selected = _live_smoke_selection_count()
    assert selected <= SMOKE_TIER_SIZE_CEILING, (
        f"the live smoke selection is {selected} tests, above the ceiling of "
        f"{SMOKE_TIER_SIZE_CEILING} in tests/conftest.py; demote tests rather than "
        "raising the ceiling without a measured sub-5-minute smoke wall time."
    )

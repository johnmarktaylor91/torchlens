"""Tier-durations check: no bounded test file exceeds its tier budget.

``tests/tiers/durations.csv`` records the slowest measured test per file and
the loosest tier the file spells; ``scripts/tier_durations.py check`` judges
each bounded file (smoke and unmarked 5s, heavy 20s per test, scaled by the
measurement's provenance) and lists test files on disk that the measurement
never saw. This adapter runs that command over the checkout for ``run_all.py``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import FAIL, OK, SKIP, Outcome  # noqa: E402

DURATIONS = Path("tests/tiers/durations.csv")
SCRIPT = Path("scripts/tier_durations.py")


def _load_script(repo: Path):
    spec = importlib.util.spec_from_file_location("tier_durations", repo / SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(repo: Path) -> Outcome:
    """Run ``tier_durations.py check`` over ``repo``; skip a tree without the file."""

    if not (repo / DURATIONS).is_file() or not (repo / SCRIPT).is_file():
        return Outcome(SKIP, "tier-durations: no tests/tiers/durations.csv in this tree")
    module = _load_script(repo)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        status = module.main(
            [
                "check",
                "--durations",
                str(repo / DURATIONS),
                "--meta",
                str(repo / module.META_PATH.relative_to(module.REPO_ROOT)),
            ]
        )
    lines = [line for line in buffer.getvalue().splitlines() if line]
    summary = lines[-1] if lines else "tier-durations"
    details = tuple(line for line in lines if line.startswith("FAIL"))
    return Outcome(OK if status == 0 else FAIL, summary, details)

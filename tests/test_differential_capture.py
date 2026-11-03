"""CI gate for the stage-1 mode-vs-wrapper differential harness.

Runs ``tools/differential_capture.py`` cross-process (the mode side must
observe PRISTINE torch, which this pytest process — with torchlens wrapped —
cannot provide) and asserts an op-for-op aligned verdict for every corpus
model. Also unit-tests the aligner's named canonicalization rules directly:
every rule must be NARROW (per-event, per-name), and unexplained events must
fail loudly — the discipline is "named carve-outs, never whole-column masks".
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = REPO_ROOT / "tools" / "differential_capture.py"

sys.path.insert(0, str(REPO_ROOT / "tools"))
from differential_capture import MODEL_SPECS  # noqa: E402


def _mode_events(names: list[str | None]) -> dict[str, object]:
    return {
        "side": "mode",
        "model": "unit",
        "events": [{"name": n, "unmapped_repr": None if n else "<raw>"} for n in names],
    }


def _wrapper_stream(
    ops: list[str],
    expansions: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    return {
        "side": "wrapper",
        "model": "unit",
        "ops": ops,
        "funcs_not_to_log": ["numpy", "__array__", "size", "dim"],
        "composite_expansions": expansions or {},
    }


@pytest.mark.heavy
@pytest.mark.parametrize("model_name", sorted(MODEL_SPECS))
def test_differential_capture_corpus(model_name: str) -> None:
    """Every corpus model aligns op-for-op across the two capture mechanisms."""
    proc = subprocess.run(
        [sys.executable, str(HARNESS), "--compare", "--model", model_name],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"harness mismatch:\n{proc.stdout}\n{proc.stderr}"
    report = json.loads(proc.stdout)
    assert report["matched"], json.dumps(report["mismatches"], indent=2)
    # A vacuous alignment (nothing compared) must never count as a pass.
    compared = [
        r
        for r in report["ledger"]
        if r["rule"] in ("EXACT", "DUNDER_RESPELL", "COMPOSITE_EXPANSION")
    ]
    assert compared, "alignment ledger is empty; harness compared nothing"

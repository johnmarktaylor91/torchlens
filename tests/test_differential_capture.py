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
from differential_capture import MODEL_SPECS, align_streams  # noqa: E402


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


class TestAlignerRules:
    """The four named rules, each exercised in isolation."""

    def test_exact_match(self) -> None:
        report = align_streams(_mode_events(["relu"]), _wrapper_stream(["relu"]))
        assert report["matched"]
        assert [row["rule"] for row in report["ledger"]] == ["EXACT"]

    def test_structural_rows_dropped_and_ledgered(self) -> None:
        report = align_streams(
            _mode_events(["relu"]),
            _wrapper_stream(["none", "relu", "identity", "none"]),
        )
        assert report["matched"]
        structural = [r for r in report["ledger"] if r["rule"] == "STRUCTURAL_ROW"]
        assert len(structural) == 3

    def test_not_logged_drop_uses_torchlens_table(self) -> None:
        report = align_streams(_mode_events(["dim", "relu"]), _wrapper_stream(["relu"]))
        assert report["matched"]
        assert report["ledger"][0] == {"rule": "NOT_LOGGED", "mode_index": 0, "name": "dim"}

    def test_dunder_respell_is_pairwise(self) -> None:
        report = align_streams(
            _mode_events(["add", "mul"]), _wrapper_stream(["__add__", "__mul__"])
        )
        assert report["matched"]
        assert all(row["rule"] == "DUNDER_RESPELL" for row in report["ledger"])

    def test_composite_expansion_consumes_exact_span(self) -> None:
        report = align_streams(
            _mode_events(["softsign", "relu"]),
            _wrapper_stream(
                ["__abs__", "__add__", "__truediv__", "relu"],
                expansions={"softsign": ["__abs__", "__add__", "__truediv__"]},
            ),
        )
        assert report["matched"]
        assert report["ledger"][0]["rule"] == "COMPOSITE_EXPANSION"
        assert report["ledger"][0]["wrapper_span"] == ["__abs__", "__add__", "__truediv__"]

    def test_wrong_composite_span_fails_loudly(self) -> None:
        report = align_streams(
            _mode_events(["softsign"]),
            _wrapper_stream(
                ["__abs__", "__mul__"],
                expansions={"softsign": ["__abs__", "__add__", "__truediv__"]},
            ),
        )
        assert not report["matched"]
        assert report["mismatches"][0]["kind"] == "COMPOSITE_SPAN_MISMATCH"

    def test_name_mismatch_fails_loudly(self) -> None:
        report = align_streams(_mode_events(["relu"]), _wrapper_stream(["tanh"]))
        assert not report["matched"]
        assert report["mismatches"][0]["kind"] == "NAME_MISMATCH"

    def test_unconsumed_wrapper_ops_fail_loudly(self) -> None:
        report = align_streams(_mode_events(["relu"]), _wrapper_stream(["relu", "tanh"]))
        assert not report["matched"]
        assert report["mismatches"][0]["kind"] == "UNCONSUMED_WRAPPER_OP"

    def test_unmapped_mode_event_fails_loudly(self) -> None:
        report = align_streams(_mode_events([None]), _wrapper_stream([]))
        assert not report["matched"]
        assert report["mismatches"][0]["kind"] == "UNMAPPED_MODE_EVENT"

    def test_exhausted_wrapper_stream_fails_loudly(self) -> None:
        report = align_streams(_mode_events(["relu", "tanh"]), _wrapper_stream(["relu"]))
        assert not report["matched"]
        assert report["mismatches"][0]["kind"] == "WRAPPER_STREAM_EXHAUSTED"


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

"""Decision tests for the interleaved nightly perf gate (``benchmarks/perf_ab.py``).

A synthetic cell runner stands in for ``perf_runner`` subprocesses, so these
tests pin the statistics and the verdict logic without timing anything: an A/A
run passes, planted regressions still fail (a gross one even on a noisy
runner), a transient flag that the re-measure does not reproduce passes,
null-control drift is inconclusive, and the cumulative anchor check runs
alongside the per-commit one.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks import perf_ab

CELLS: list[perf_ab.Cell] = [
    ("raw_forward", "tinynet", "cpu"),
    ("raw_inference_mode", "tinynet", "cpu"),
    ("global_wrap_dummy", "tinynet", "cpu"),
    ("tl_trace", "tinynet", "cpu"),
    ("aux_save", "tinynet", "cpu"),
]
BASE_MS = {
    "raw_forward": 4.2,
    "raw_inference_mode": 4.2,
    "global_wrap_dummy": 1400.0,
    "tl_trace": 150.0,
    "aux_save": 100.0,
}
COLD = {"global_wrap_dummy"}
Slowdown = Callable[[str, str, int], float]


def _runner(slowdown: Slowdown, *, jitter: float = 0.03, seed: int = 0) -> perf_ab.CellRunner:
    """Return a fake cell runner; ``slowdown(arm, operation, call)`` scales a cell."""

    rng = random.Random(seed)
    calls: dict[tuple[str, str], int] = {}

    def run(tree: Path, cell: perf_ab.Cell, samples: int, out: Path) -> dict[str, Any]:
        del out
        arm, operation = tree.name, cell[0]
        call = calls.setdefault((arm, operation), 0)
        calls[(arm, operation)] = call + 1
        count = 1 if operation in COLD else samples
        scale = slowdown(arm, operation, call)
        values = [
            BASE_MS[operation] * scale * (1 + rng.uniform(-jitter, jitter)) for _ in range(count)
        ]
        return {
            "operation": operation,
            "model": cell[1],
            "device": cell[2],
            "status": "ok",
            "timing": {"samples_ms": values, "cpu_samples_ms": values},
        }

    return run


def _gate(tmp_path: Path, runner: perf_ab.CellRunner, *, anchor: bool = False) -> dict[str, Any]:
    arms = ["base", "current"] + (["anchor"] if anchor else [])
    trees = {arm: tmp_path / arm for arm in arms}
    return perf_ab.run_gate(trees, CELLS, runner, rounds=5, samples=4, workdir=tmp_path / "w")


def _none(arm: str, operation: str, call: int) -> float:
    del arm, operation, call
    return 1.0


def test_aa_run_passes(tmp_path: Path) -> None:
    result = _gate(tmp_path, _runner(_none), anchor=True)
    assert result["passed"] is True
    assert {ref: v["verdict"] for ref, v in result["verdicts"].items()} == {
        "base": "pass",
        "anchor": "pass",
    }
    assert result["confirmation_payloads"] == {}


def test_cold_rows_get_one_sample_per_round(tmp_path: Path) -> None:
    result = _gate(tmp_path, _runner(_none))
    rows = {row["operation"]: row for row in result["primary_payloads"]["current"]["rows"]}
    assert rows["global_wrap_dummy"]["passes"]["timing"]["timing"]["cpu_sample_count"] == 5
    assert rows["tl_trace"]["passes"]["timing"]["timing"]["cpu_sample_count"] == 20


def test_arm_order_rotates_per_round(tmp_path: Path) -> None:
    order: list[str] = []

    def record(tree: Path, cell: perf_ab.Cell, samples: int, out: Path) -> dict[str, Any]:
        if cell[0] == "tl_trace":
            order.append(tree.name)
        return _runner(_none)(tree, cell, samples, out)

    perf_ab.measure(
        {"base": tmp_path / "base", "current": tmp_path / "current"},
        CELLS,
        record,
        rounds=4,
        samples=2,
        workdir=tmp_path / "w",
    )
    assert order == ["base", "current", "current", "base"] * 2


def test_planted_regression_fails(tmp_path: Path) -> None:
    def planted(arm: str, operation: str, call: int) -> float:
        del call
        return 1.25 if arm == "current" and operation == "tl_trace" else 1.0

    result = _gate(tmp_path, _runner(planted))
    verdict = result["verdicts"]["base"]
    assert result["passed"] is False and verdict["verdict"] == "fail"
    assert verdict["confirmed_regressions"] == [["tl_trace", "tinynet", "cpu"]]


def test_gross_planted_regression_fails_even_when_controls_drift(tmp_path: Path) -> None:
    def gross_on_noisy_runner(arm: str, operation: str, call: int) -> float:
        del call
        if arm != "current":
            return 1.0
        return {"tl_trace": 2.0, "raw_forward": 1.4}.get(operation, 1.0)

    verdict = _gate(tmp_path, _runner(gross_on_noisy_runner))["verdicts"]["base"]
    assert verdict["verdict"] == "fail"
    assert verdict["gross_regressions"] == [["tl_trace", "tinynet", "cpu"]]
    assert ["raw_forward", "tinynet", "cpu"] in verdict["control_drift"]


def test_modest_flag_on_noisy_runner_is_inconclusive(tmp_path: Path) -> None:
    def noisy(arm: str, operation: str, call: int) -> float:
        del call
        if arm != "current":
            return 1.0
        return {"tl_trace": 1.2, "raw_forward": 1.4}.get(operation, 1.0)

    result = _gate(tmp_path, _runner(noisy))
    assert result["passed"] is True
    assert result["inconclusive"] == ["base"]
    assert result["verdicts"]["base"]["confirmed_regressions"] == [["tl_trace", "tinynet", "cpu"]]


def test_control_drift_alone_is_inconclusive(tmp_path: Path) -> None:
    def drift(arm: str, operation: str, call: int) -> float:
        del call
        return 1.4 if arm == "current" and operation == "raw_forward" else 1.0

    result = _gate(tmp_path, _runner(drift))
    assert result["passed"] is True
    assert result["verdicts"]["base"]["verdict"] == "inconclusive"


def test_unconfirmed_flag_passes_with_disclosure(tmp_path: Path) -> None:
    def transient(arm: str, operation: str, call: int) -> float:
        # Only the primary measurement's rounds (calls 0-4) are slow.
        return 1.3 if arm == "current" and operation == "tl_trace" and call < 5 else 1.0

    verdict = _gate(tmp_path, _runner(transient))["verdicts"]["base"]
    assert verdict["verdict"] == "pass"
    assert verdict["unconfirmed_flags"] == [["tl_trace", "tinynet", "cpu"]]


def test_failed_torchlens_cell_fails_structurally(tmp_path: Path) -> None:
    ok = _runner(_none)

    def crash(tree: Path, cell: perf_ab.Cell, samples: int, out: Path) -> dict[str, Any]:
        payload = ok(tree, cell, samples, out)
        if tree.name == "current" and cell[0] == "aux_save":
            return {**payload, "status": "error", "error": "boom"}
        return payload

    verdict = _gate(tmp_path, crash)["verdicts"]["base"]
    assert verdict["verdict"] == "fail"
    assert "status_failures" in verdict["structural_failures"]


def test_cumulative_anchor_check_runs_beside_per_commit(tmp_path: Path) -> None:
    def drifted_since_release(arm: str, operation: str, call: int) -> float:
        del call
        return 0.7 if arm == "anchor" and operation == "tl_trace" else 1.0

    result = _gate(tmp_path, _runner(drifted_since_release), anchor=True)
    assert result["verdicts"]["base"]["verdict"] == "pass"
    assert result["verdicts"]["anchor"]["verdict"] == "fail"
    assert result["passed"] is False

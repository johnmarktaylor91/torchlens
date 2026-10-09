"""Interleaved same-runner A/B performance gate for the nightly workflow.

The previous nightly gate ran three whole smoke suites back to back (parent,
release anchor, current) and compared single medians. Runner drift over those
five minutes read as a regression: a row with no TorchLens code failed the gate
at 1.40x, and the cold-start rows carried ONE sample each, so their IQR term
was zero. This orchestrator removes both failure modes without loosening the
per-row tolerance policy of :mod:`benchmarks.perf_gate`:

1. **Interleaving.** For every timing cell, each round runs one subprocess per
   arm (parent, current, and optionally the release anchor), rotating the arm
   order per round, so drift lands on every arm alike. Samples are pooled
   across rounds; every cold-start row gets one cold sample per round.
2. **Null controls.** Rows no TorchLens code runs in (``raw_forward``,
   ``raw_inference_mode``; :func:`benchmarks.op_ownership.is_torchlens_operation`
   is False) cannot regress from a TorchLens change. When one moves past
   tolerance the runner was noisy and the comparison is INCONCLUSIVE: reported
   with a warning, neither passed silently nor failed.
3. **Confirmation.** A flagged TorchLens row is re-measured in fresh
   interleaved rounds (with the controls) and fails only when the re-measure
   flags it again.
4. **Gross regressions always fail.** A row confirmed at or above
   ``gross_ratio`` in both measurements fails even on a noisy runner.

Structural failures (failed or vanished TorchLens rows, missing metrics) fail
at once: they are not timing noise. Both comparisons (current vs parent at the
per-commit tolerance, current vs release anchor at the cumulative tolerance)
are always computed and reported.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from benchmarks.op_ownership import is_torchlens_operation
from benchmarks.perf_gate import (
    DEFAULT_FLOOR_MS,
    DEFAULT_IQR_MULTIPLIER,
    DEFAULT_REL_TOLERANCE,
    SCHEMA,
    compare_gate_payloads,
)

DEFAULT_ROUNDS = 5
DEFAULT_SAMPLES = 4
DEFAULT_GROSS_RATIO = 1.5
DEFAULT_ANCHOR_TOLERANCE = 0.25
Cell = tuple[str, str, str]
#: ``runner(tree, cell, samples, out_path) -> cell payload`` (perf_runner JSON).
CellRunner = Callable[[Path, Cell, int, Path], dict[str, Any]]
STRUCTURAL_KEYS = (
    "status_failures",
    "missing_current_rows",
    "unmatched_current_rows",
    "uncomparable_rows",
    "wall_clock_fallback_blocking_rows",
)


def smoke_cells() -> list[Cell]:
    """Return the ``(operation, model, device)`` timing cells of the smoke suite.

    Returns
    -------
    list[Cell]
        Timing cells in suite order.
    """

    from benchmarks.perf_suite import _matrix

    return [(op, model, device) for op, model, device, kind in _matrix(True) if kind == "timing"]


def subprocess_runner(
    python: str, timeout: int, threads: int
) -> Callable[[Path, Cell, int, Path], dict[str, Any]]:
    """Build a runner that times one cell in a fresh ``perf_runner`` process.

    Parameters
    ----------
    python:
        Interpreter shared by every arm (one venv, as in the old gate).
    timeout:
        Per-subprocess timeout in seconds.
    threads:
        Torch intra-op thread pin forwarded to the runner.

    Returns
    -------
    Callable[[Path, Cell, int, Path], dict[str, Any]]
        The cell runner.
    """

    def run(tree: Path, cell: Cell, samples: int, out: Path) -> dict[str, Any]:
        operation, model, device = cell
        cmd = [python, "-m", "benchmarks.perf_runner", "--operation", operation]
        cmd += ["--model", model, "--device", device, "--pass-type", "timing"]
        cmd += ["--out", str(out), "--samples", str(samples), "--threads", str(threads)]
        env = {**os.environ, "PYTHONPATH": str(tree)}
        try:
            done = subprocess.run(
                cmd, cwd=tree, env=env, timeout=timeout, text=True, capture_output=True
            )
        except subprocess.TimeoutExpired:
            return _error_payload(cell, f"cell timed out after {timeout}s")
        if not out.exists():
            return _error_payload(cell, f"runner wrote no JSON: {done.stderr[-2000:]}")
        return dict(json.loads(out.read_text()))

    return run


def _error_payload(cell: Cell, error: str) -> dict[str, Any]:
    """Return a failed cell payload that the gate reports as a status failure."""

    operation, model, device = cell
    return {
        "operation": operation,
        "model": model,
        "device": device,
        "status": "error",
        "error": error,
    }


def measure(
    trees: dict[str, Path],
    cells: Sequence[Cell],
    runner: CellRunner,
    *,
    rounds: int,
    samples: int,
    workdir: Path,
) -> dict[str, dict[Cell, list[dict[str, Any]]]]:
    """Time every cell in every arm, interleaved, rotating arm order per round.

    Parameters
    ----------
    trees:
        Arm name to source tree (``{"base": ..., "current": ...}``).
    cells:
        Timing cells to run.
    runner:
        Cell runner.
    rounds:
        Interleaved rounds; each cold-start row gets one sample per round.
    samples:
        Timing samples per subprocess for the warm rows.
    workdir:
        Directory for the runners' per-cell JSON.

    Returns
    -------
    dict[str, dict[Cell, list[dict[str, Any]]]]
        Per arm and cell, the payload of every round.
    """

    arms = list(trees)
    results: dict[str, dict[Cell, list[dict[str, Any]]]] = {arm: {} for arm in arms}
    for round_index in range(rounds):
        order = arms[round_index % len(arms) :] + arms[: round_index % len(arms)]
        for cell in cells:
            for arm in order:
                out = workdir / f"r{round_index}__{arm}__{'__'.join(cell)}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                payload = runner(trees[arm], cell, samples, out)
                results[arm].setdefault(cell, []).append(payload)
    return results


def _pooled_stats(samples_ms: list[float], prefix: str) -> dict[str, Any]:
    """Summarize pooled millisecond samples with the runner's key names."""

    if not samples_ms:
        return {}
    if len(samples_ms) > 1:
        q1, _, q3 = statistics.quantiles(samples_ms, n=4, method="inclusive")
    else:
        q1 = q3 = samples_ms[0]
    return {
        f"{prefix}samples_ms": samples_ms,
        f"{prefix}sample_count": len(samples_ms),
        f"{prefix}median_ms": statistics.median(samples_ms),
        f"{prefix}iqr_ms": q3 - q1,
    }


def pool_row(cell: Cell, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge one cell's per-round payloads into a gate row.

    Parameters
    ----------
    cell:
        The cell.
    payloads:
        Its payload from every round.

    Returns
    -------
    dict[str, Any]
        Gate row whose timing statistics cover the pooled samples. Any failed
        round makes the row failed: a flaky crash is a finding, not noise.
    """

    operation, model, device = cell
    bad = [p for p in payloads if p.get("status", "ok") != "ok"]
    row: dict[str, Any] = {
        "model": model,
        "device": device,
        "operation": operation,
        "status": bad[0].get("status", "error") if bad else "ok",
    }
    if bad:
        row["error"] = bad[0].get("error") or bad[0].get("skip_reason")
        return row
    timing: dict[str, Any] = {"rounds": len(payloads)}
    for prefix in ("", "cpu_"):
        pooled = [s for p in payloads for s in p.get("timing", {}).get(f"{prefix}samples_ms", [])]
        timing.update(_pooled_stats([float(s) for s in pooled], prefix))
    row["passes"] = {"timing": {"timing": timing}}
    return row


def build_payload(
    arm_results: dict[Cell, list[dict[str, Any]]], environment: dict[str, Any], sha: str
) -> dict[str, Any]:
    """Return a :mod:`benchmarks.perf_gate` payload for one arm.

    Parameters
    ----------
    arm_results:
        Per-cell payloads of the arm.
    environment:
        Environment metadata recorded with the payload.
    sha:
        Source commit of the arm.

    Returns
    -------
    dict[str, Any]
        Gate payload.
    """

    rows = [pool_row(cell, payloads) for cell, payloads in arm_results.items()]
    return {"schema": SCHEMA, "rows": rows, "environment": environment, "source_sha": sha}


def _row_id(check: dict[str, Any]) -> Cell:
    """Return the cell of a comparison check."""

    return (str(check["operation"]), str(check["model"]), str(check["device"]))


def flagged_rows(comparison: dict[str, Any]) -> tuple[list[Cell], list[Cell]]:
    """Split a comparison's regressions into TorchLens rows and null controls.

    Parameters
    ----------
    comparison:
        :func:`benchmarks.perf_gate.compare_gate_payloads` output.

    Returns
    -------
    tuple[list[Cell], list[Cell]]
        ``(flagged TorchLens rows, drifted control rows)``.
    """

    cells = [_row_id(check) for check in comparison["regressions"]]
    owned = [cell for cell in cells if is_torchlens_operation(cell[0])]
    return owned, [cell for cell in cells if cell not in owned]


def decide(
    primary: dict[str, Any],
    confirmation: dict[str, Any] | None,
    *,
    gross_ratio: float = DEFAULT_GROSS_RATIO,
) -> dict[str, Any]:
    """Return the verdict for one comparison and its confirmation re-measure.

    Parameters
    ----------
    primary:
        Comparison over the full interleaved measurement.
    confirmation:
        Comparison over the re-measured flagged rows plus the controls, or
        ``None`` when nothing was flagged.
    gross_ratio:
        Current/baseline ratio at or above which a row confirmed in both
        measurements fails even on a noisy runner.

    Returns
    -------
    dict[str, Any]
        ``verdict`` (``"pass"``, ``"fail"`` or ``"inconclusive"``) with the
        evidence lists behind it.
    """

    structural = {key: primary[key] for key in STRUCTURAL_KEYS if primary.get(key)}
    if confirmation is not None:
        structural |= {
            f"confirmation_{key}": confirmation[key]
            for key in STRUCTURAL_KEYS
            if confirmation.get(key)
        }
    flagged, drift = flagged_rows(primary)
    confirm_flagged, confirm_drift = flagged_rows(confirmation) if confirmation else ([], [])
    confirmed = [cell for cell in flagged if cell in confirm_flagged]
    ratios = {
        (_row_id(c), name): c["ratio"]
        for name, comp in (("primary", primary), ("confirmation", confirmation or {}))
        for c in comp.get("regressions", [])
        if "ratio" in c
    }
    gross = [
        cell
        for cell in confirmed
        if min(ratios[(cell, "primary")], ratios[(cell, "confirmation")]) >= gross_ratio
    ]
    control_drift = sorted(set(drift) | set(confirm_drift))
    if structural or gross or (confirmed and not control_drift):
        verdict = "fail"
    elif control_drift:
        verdict = "inconclusive"
    else:
        verdict = "pass"
    return {
        "verdict": verdict,
        "structural_failures": structural,
        "confirmed_regressions": [list(cell) for cell in confirmed],
        "gross_regressions": [list(cell) for cell in gross],
        "unconfirmed_flags": [list(cell) for cell in flagged if cell not in confirmed],
        "control_drift": [list(cell) for cell in control_drift],
    }


def _compare(
    baseline: dict[str, Any], current: dict[str, Any], rel_tolerance: float
) -> dict[str, Any]:
    """Compare two arm payloads under the gate's tolerance policy."""

    return compare_gate_payloads(
        baseline,
        current,
        rel_tolerance=rel_tolerance,
        iqr_multiplier=DEFAULT_IQR_MULTIPLIER,
        floor_ms=DEFAULT_FLOOR_MS,
    )


def run_gate(
    trees: dict[str, Path],
    cells: Sequence[Cell],
    runner: CellRunner,
    *,
    rounds: int = DEFAULT_ROUNDS,
    samples: int = DEFAULT_SAMPLES,
    workdir: Path,
    environment: dict[str, Any] | None = None,
    shas: dict[str, str] | None = None,
    rel_tolerance: float = DEFAULT_REL_TOLERANCE,
    anchor_tolerance: float = DEFAULT_ANCHOR_TOLERANCE,
    gross_ratio: float = DEFAULT_GROSS_RATIO,
) -> dict[str, Any]:
    """Measure interleaved arms, compare, re-measure flagged rows and decide.

    Parameters
    ----------
    trees:
        ``"base"`` and ``"current"`` trees, plus an optional ``"anchor"``
        (the latest release) for the cumulative comparison.
    cells, runner, rounds, samples, workdir:
        See :func:`measure`.
    environment, shas:
        Metadata recorded in the arm payloads.
    rel_tolerance:
        Per-commit tolerance (current vs base).
    anchor_tolerance:
        Cumulative tolerance (current vs anchor).
    gross_ratio:
        See :func:`decide`.

    Returns
    -------
    dict[str, Any]
        ``passed``, per-comparison ``verdicts``, and the payloads and
        comparisons behind them.
    """

    env, shas = environment or {}, shas or {}
    tolerances = {"base": rel_tolerance, "anchor": anchor_tolerance}
    refs = [arm for arm in ("base", "anchor") if arm in trees]

    def payloads(results: dict[str, dict[Cell, list[dict[str, Any]]]]) -> dict[str, Any]:
        return {arm: build_payload(res, env, shas.get(arm, "")) for arm, res in results.items()}

    primary_payloads = payloads(
        measure(trees, cells, runner, rounds=rounds, samples=samples, workdir=workdir / "primary")
    )
    primary = {
        ref: _compare(primary_payloads[ref], primary_payloads["current"], tolerances[ref])
        for ref in refs
    }
    flagged = sorted({cell for ref in refs for cell in flagged_rows(primary[ref])[0]})
    confirmation: dict[str, dict[str, Any] | None] = dict.fromkeys(refs)
    confirm_payloads: dict[str, Any] = {}
    if flagged:
        controls = [cell for cell in cells if not is_torchlens_operation(cell[0])]
        confirm_payloads = payloads(
            measure(
                trees,
                flagged + controls,
                runner,
                rounds=rounds,
                samples=samples,
                workdir=workdir / "confirm",
            )
        )
        confirmation = {
            ref: _compare(confirm_payloads[ref], confirm_payloads["current"], tolerances[ref])
            for ref in refs
        }
    verdicts = {
        ref: decide(primary[ref], confirmation[ref], gross_ratio=gross_ratio) for ref in refs
    }
    return {
        "schema": "torchlens.perf_ab.v1",
        "passed": all(v["verdict"] != "fail" for v in verdicts.values()),
        "inconclusive": [ref for ref, v in verdicts.items() if v["verdict"] == "inconclusive"],
        "verdicts": verdicts,
        "policy": {
            "rounds": rounds,
            "samples_per_round": samples,
            "gross_ratio": gross_ratio,
            "tolerances": {ref: tolerances[ref] for ref in refs},
        },
        "primary_comparisons": primary,
        "confirmation_comparisons": confirmation,
        "primary_payloads": primary_payloads,
        "confirmation_payloads": confirm_payloads,
    }


def _git_sha(tree: Path) -> str:
    """Return the short commit of ``tree``, or an empty string."""

    done = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "--short", "HEAD"], text=True, capture_output=True
    )
    return done.stdout.strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv:
        Arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Parent-commit tree")
    parser.add_argument("--current", type=Path, required=True, help="Commit-under-test tree")
    parser.add_argument("--anchor", type=Path, help="Latest-release tree (cumulative check)")
    parser.add_argument("--out", type=Path, required=True, help="Result JSON path")
    parser.add_argument("--workdir", type=Path, required=True, help="Per-cell JSON directory")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rel-tolerance", type=float, default=DEFAULT_REL_TOLERANCE)
    parser.add_argument("--anchor-tolerance", type=float, default=DEFAULT_ANCHOR_TOLERANCE)
    parser.add_argument("--gross-ratio", type=float, default=DEFAULT_GROSS_RATIO)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interleaved gate; return the process exit code.

    Parameters
    ----------
    argv:
        Arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        1 when any comparison fails, else 0 (inconclusive is reported, not failed).
    """

    args = parse_args(argv)
    trees = {"base": args.base, "current": args.current}
    if args.anchor is not None:
        trees["anchor"] = args.anchor
    result = run_gate(
        {arm: tree.resolve() for arm, tree in trees.items()},
        smoke_cells(),
        subprocess_runner(sys.executable, args.timeout, args.threads),
        rounds=args.rounds,
        samples=args.samples,
        workdir=args.workdir,
        environment={"runner": "perf_ab", "threads": args.threads},
        shas={arm: _git_sha(tree) for arm, tree in trees.items()},
        rel_tolerance=args.rel_tolerance,
        anchor_tolerance=args.anchor_tolerance,
        gross_ratio=args.gross_ratio,
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    for ref, verdict in result["verdicts"].items():
        summary = {k: v for k, v in verdict.items() if v and k != "verdict"}
        print(f"current vs {ref}: {verdict['verdict'].upper()} {json.dumps(summary)}")
        if verdict["verdict"] == "inconclusive":
            print(f"::warning::perf gate vs {ref} INCONCLUSIVE: null-control rows drifted")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a capture benchmark matrix, one fresh subprocess per cell, into one result document.

The matrix covers Suite A (matched contracts across arms), Suite B (what each TorchLens
operation keeps, beside plain-forward and hook references) and the operation fixture
sweep. Cells an arm cannot serve are written as ``unsupported`` rows without starting a
process, so they stay visible. The process index loop is outermost, so slow drift on the
host spreads across arms instead of landing on one. An A/A pair (the same raw forward
measured in two separate processes, at the start and at the end) gives the noise floor;
a cell group whose spread across processes exceeds it is listed as noisy.

    python benchmarks/capture/driver.py --smoke --out outputs/capture-smoke.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.capture import schema  # noqa: E402
from benchmarks.capture.arms import CASES  # noqa: E402
from benchmarks.host_label import redact_local_paths  # noqa: E402

SUPPORT = {
    "raw": set(CASES),
    "hooks": set(CASES),
    "nnsight": set(CASES),
    "tl_trace_sparse": set(CASES),
    "tl_record": set(CASES) - {"same_site_gradient"},
    "tl_extract": {"single_module_output", "matched_retained", "ordered_dataset_features"},
    "tl_bind": {"replacement_edit", "repeated_steering"},
    "tl_extract_dataset": {"ordered_dataset_features"},
    "tv_fx": {"single_module_output", "matched_retained", "ordered_dataset_features"},
    "thingsvision": {"ordered_dataset_features"},
    "captum": {"same_site_gradient"},
    "transformer_lens": set(),
}
UNSUPPORTED_REASON = {
    "tl_record": "this comparison uses tl.record without backward capture",
    "tl_extract": "tl.extract reads outputs; edits and gradients use other entry points",
    "tl_bind": "a bound edit records nothing; it serves edit contracts only",
    "tl_extract_dataset": "dataset extraction serves the dataset contract only",
    "tv_fx": "a feature extractor reads outputs; it has no edit or gradient contract",
    "thingsvision": "thingsvision is measured only on its extraction task",
    "captum": "Captum is measured only on its matching attribution contract",
    "transformer_lens": (
        "TransformerLens reimplements its supported decoder families; generic subjects are not "
        "among them (decoders are measured in the mech-interp workloads)"
    ),
}
SMOKE_A_SUBJECTS = (
    "fx-pointwise-100-small",
    "fx-attention_decomposed-1000-small",
    "resnet18",
    "lstm-tagger",
    "encoder-small",
    "dynamic-router",
)
SWEEP_ROWS = ("raw_forward", "hooks_all_modules", "full_trace_default")
AA_SUBJECT = "fx-pointwise-1000-small"
CELL_TIMEOUT_S = 900


def plan(
    smoke: bool, subjects: list[str] | None = None, sweep: bool = True
) -> list[tuple[str, str, str, str]]:
    """Return the (suite, case, arm, subject) cells of a run.

    Parameters
    ----------
    smoke:
        The CPU smoke matrix (small subjects) instead of the full CPU matrix.
    subjects:
        Override the Suite A and B subjects.
    sweep:
        Include the fixture sweep.

    Returns
    -------
    list[tuple[str, str, str, str]]
        Cells in run order.
    """

    from benchmarks.capture.fixtures import fixture_matrix
    from benchmarks.capture.panel import smoke_panel
    from benchmarks.capture.suite_b import ROWS

    fixtures = fixture_matrix(smoke) if sweep else []
    main = list(
        subjects or (SMOKE_A_SUBJECTS if smoke else (*fixture_matrix(smoke), *smoke_panel()))
    )
    cells = [("A", case, arm, subj) for subj in main for case in CASES for arm in SUPPORT]
    for subj in main:
        for name, row in ROWS.items():
            cells.append(("B", name, "torchlens" if row.uses_torchlens else "reference", subj))
    for subj in fixtures:
        if subj in main:
            continue
        for name in SWEEP_ROWS:
            arm = "torchlens" if ROWS[name].uses_torchlens else "reference"
            cells.append(("fixtures", name, arm, subj))
    return cells


def _spawn(cell: tuple[str, str, str, str], index: int, args: argparse.Namespace) -> dict[str, Any]:
    suite, case, arm, subject = cell
    cli_suite = "A" if suite == "A" else "B"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "benchmarks" / "capture" / "cell.py"),
        *("--suite", cli_suite, "--case", case, "--arm", arm, "--subject", subject),
        *("--warmups", str(args.warmups), "--samples", str(args.samples)),
        *("--memory-runs", str(args.memory_runs), "--budget-s", str(args.budget_s)),
        *("--threads", str(args.threads), "--process-index", str(index)),
        *("--device", args.device),
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=CELL_TIMEOUT_S, cwd=REPO_ROOT
        )
    except subprocess.TimeoutExpired:
        row = schema.make_cell(
            suite, case, arm, subject, "timeout", reason=f"over {CELL_TIMEOUT_S} s"
        )
    else:
        row = _parse(proc, cell)
    row["suite"] = suite
    row["cell_id"] = f"{suite}/{case}/{arm}/{subject}"
    row["process_index"] = index
    row["wall_s"] = time.perf_counter() - start
    return row


def _parse(
    proc: subprocess.CompletedProcess[str], cell: tuple[str, str, str, str]
) -> dict[str, Any]:
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    tail = redact_local_paths((proc.stdout + proc.stderr)[-600:], REPO_ROOT)
    return schema.make_cell(*cell, "error", reason=f"rc={proc.returncode}: {tail}")


def _static_unsupported(cell: tuple[str, str, str, str]) -> dict[str, Any] | None:
    suite, case, arm, subject = cell
    if suite != "A" or case in SUPPORT[arm]:
        return None
    return schema.make_cell(
        suite, case, arm, subject, "unsupported", reason=UNSUPPORTED_REASON[arm]
    )


def aa_noise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the A/A pairs: relative difference between identical cells."""

    medians = [r["timing"]["median_ms"] for r in rows if r.get("status") == "ok"]
    if len(medians) < 2:
        return {"pairs": len(medians), "max_rel_diff": None, "note": "fewer than two A/A cells"}
    lo, hi = min(medians), max(medians)
    return {"pairs": len(medians), "medians_ms": medians, "max_rel_diff": (hi - lo) / lo}


def group_cells(
    cells: list[dict[str, Any]], aa: dict[str, Any], floor: float
) -> list[dict[str, Any]]:
    """Fold per-process rows into one row per cell id with spread and a noisy flag."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault(cell["cell_id"], []).append(cell)
    threshold = max(floor, 2 * (aa.get("max_rel_diff") or 0.0))
    out = []
    for cell_id, rows in groups.items():
        medians = [
            r["timing"]["median_ms"] for r in rows if r.get("status") == "ok" and "timing" in r
        ]
        entry: dict[str, Any] = {
            "cell_id": cell_id,
            "statuses": sorted({r["status"] for r in rows}),
        }
        if medians:
            centre = statistics.median(medians)
            spread = (max(medians) - min(medians)) / centre if centre else 0.0
            entry.update(
                {
                    "median_ms": centre,
                    "process_spread": spread,
                    "noisy": len(medians) > 1 and spread > threshold,
                }
            )
        out.append(entry)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the matrix and write the document."""

    cells = plan(args.smoke, args.subjects.split(",") if args.subjects else None, not args.no_sweep)
    doc = schema.new_document(
        "capture",
        {"smoke": args.smoke, "cells_planned": len(cells), "threads": args.threads},
        device=args.device,
        protocol={
            "warmups": args.warmups,
            "samples": args.samples,
            "processes": args.processes,
            "memory_runs": args.memory_runs,
            "budget_s_per_cell": args.budget_s,
            "min_samples": 3,
        },
    )
    doc["suites_included"] = ["A", "B", "fixtures"]
    aa_rows = [_spawn(("aa", "raw_forward", "reference", AA_SUBJECT), 0, args)]
    for index in range(args.processes):
        for cell in cells:
            static = _static_unsupported(cell)
            if static is not None:
                if index == 0:
                    doc["cells"].append(static)
                continue
            row = _spawn(cell, index, args)
            doc["cells"].append(row)
            print(
                f"{row['cell_id']}#{index}: {row['status']} {row.get('reason', '')[:120]}",
                flush=True,
            )
    aa_rows.append(_spawn(("aa", "raw_forward", "reference", AA_SUBJECT), 1, args))
    doc["aa_cells"] = aa_rows
    doc["aa_noise"] = aa_noise(aa_rows)
    doc["groups"] = group_cells(doc["cells"], doc["aa_noise"], args.noise_floor)
    doc["wall_s"] = sum(c.get("wall_s", 0.0) for c in doc["cells"])
    Path(args.out).write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    return doc


def parse_args() -> argparse.Namespace:
    """Parse the driver's command line."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--no-sweep", action="store_true", help="skip the fixture sweep")
    parser.add_argument("--out", default="capture-results.json")
    parser.add_argument("--processes", type=int, default=schema.PROTOCOL_DEFAULTS["processes"])
    parser.add_argument("--warmups", type=int, default=schema.PROTOCOL_DEFAULTS["warmups"])
    parser.add_argument("--samples", type=int, default=schema.PROTOCOL_DEFAULTS["samples"])
    parser.add_argument("--memory-runs", type=int, default=3)
    parser.add_argument("--budget-s", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--noise-floor", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    doc = run(parse_args())
    problems = schema.validate_document(doc)
    statuses: dict[str, int] = {}
    for cell in doc["cells"]:
        statuses[cell["status"]] = statuses.get(cell["status"], 0) + 1
    print(json.dumps({"statuses": statuses, "aa_noise": doc["aa_noise"], "wall_s": doc["wall_s"]}))
    if problems:
        raise SystemExit("result document violates the format: " + "; ".join(problems[:5]))


if __name__ == "__main__":
    main()

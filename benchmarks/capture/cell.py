"""Run one capture benchmark cell in this (fresh) process and print its JSON row.

The driver starts one interpreter per cell and per process index, so no arm inherits
another arm's warm state, and arms that do not use TorchLens never import it (its torch
wrappers would otherwise slow them). Timing reuses the existing protocol helpers in
``benchmarks/perf_runner.py``: untimed warmups, then timed samples with wall and
process time, then a separate memory pass with peak RSS and device memory.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.capture import arms, schema  # noqa: E402
from benchmarks.host_label import redact_local_paths  # noqa: E402
from benchmarks.perf_runner import (  # noqa: E402
    _pin_threads,
    _run_memory,
    _run_timing,
    _set_determinism,
)

MIN_SAMPLES = 3


def build_subject(name: str, device: str = "cpu") -> arms.Subject:
    """Build a fixture (``fx-...``) or a smoke panel model as a prepared subject."""

    if name.startswith("fx-"):
        from benchmarks.capture.fixtures import build_fixture

        model, x, meta = build_fixture(name)
    else:
        from benchmarks.capture.panel import build_panel_entry

        model, x, meta = build_panel_entry(name)
        if name == "lstm-tagger":
            meta["edit_site"] = "embed"
    return arms.prepare_subject(name, model, x, meta, device)


def _sample_budget(fn: Any, requested: int, budget_s: float) -> tuple[int, float]:
    """Time one call and cut the sample count so the timed loop fits the budget."""

    start = time.perf_counter()
    fn()
    first = time.perf_counter() - start
    if first <= 0:
        return requested, first
    return max(MIN_SAMPLES, min(requested, int(budget_s / first))), first


def measure(fn: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Time and memory-profile one callable under the protocol."""

    samples, first_s = _sample_budget(fn, args.samples, args.budget_s)
    warmups = args.warmups if samples == args.samples else min(args.warmups, 1)
    timing = _run_timing(fn, args.device, warmups, samples)
    timing.update(
        {"samples_requested": args.samples, "warmups_used": warmups, "first_call_ms": first_s * 1e3}
    )
    memory = _run_memory(fn, args.device, args.memory_runs)
    return {"timing": timing, "memory": memory}


def _suite_a(args: argparse.Namespace, subject: arms.Subject) -> dict[str, Any]:
    arm_cls = arms.ARMS[args.arm]
    try:
        arm = arm_cls(subject)
        result = arm.run(args.case)
    except arms.Unsupported as exc:
        return {"status": "unsupported", "reason": str(exc)}
    fields: dict[str, Any] = {"contract": args.case}
    if args.arm == "raw":
        fields["exact"] = {"matched": None, "note": "floor: no contract output"}
    elif args.arm != "hooks":
        reference = arms.HooksArm(subject).run(args.case)
        fields["exact"] = arms.compare(result, reference)
    fields.update(measure(lambda: arm.run(args.case), args))
    status = "mismatch" if fields.get("exact", {}).get("matched") is False else "ok"
    if status == "mismatch":
        fields["reason"] = "the arm's output differs from the hooks reference"
    return {"status": status, **fields}


def _suite_b(args: argparse.Namespace, subject: arms.Subject, workdir: Path) -> dict[str, Any]:
    from benchmarks.capture.suite_b import ROWS, build_row

    row = ROWS[args.case]
    if row.needs_graphviz:
        import shutil

        if shutil.which("dot") is None:
            return {
                "status": "unsupported",
                "reason": "Graphviz is not installed",
                "keeps": row.keeps,
            }
    fn, facts = build_row(args.case, subject, workdir)
    fields = {"keeps": row.keeps, **facts, **measure(fn, args)}
    num_ops = facts.get("num_ops")
    median = fields["timing"].get("median_ms")
    if num_ops and median is not None:
        fields["ms_per_op"] = median / num_ops
    return {"status": "ok", **fields}


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    """Run one cell and return its row (errors become ``error`` rows)."""

    _set_determinism()
    _pin_threads(args.threads)
    base = {"process_index": args.process_index, "device": args.device}
    try:
        subject = build_subject(args.subject, args.device)
        with tempfile.TemporaryDirectory(prefix="tl-bench-") as tmp:
            if args.suite == "A":
                fields = _suite_a(args, subject)
            else:
                fields = _suite_b(args, subject, Path(tmp))
    except Exception as exc:  # noqa: BLE001 - a crashing arm is a visible cell
        reason = redact_local_paths(f"{type(exc).__name__}: {exc}"[:600], REPO_ROOT)
        fields = {"status": "error", "reason": reason}
    fields["torchlens_imported"] = "torchlens" in sys.modules
    status = fields.pop("status")
    return schema.make_cell(args.suite, args.case, args.arm, args.subject, status, **base, **fields)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the cell's command line."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", choices=("A", "B"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmups", type=int, default=schema.PROTOCOL_DEFAULTS["warmups"])
    parser.add_argument("--samples", type=int, default=schema.PROTOCOL_DEFAULTS["samples"])
    parser.add_argument("--memory-runs", type=int, default=3)
    parser.add_argument("--budget-s", type=float, default=60.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--process-index", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    """Command-line entry point: print the cell row as one JSON line."""

    print(json.dumps(run_cell(parse_args()), default=str), flush=True)


if __name__ == "__main__":
    main()

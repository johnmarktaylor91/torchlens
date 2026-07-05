"""S2 capture-spine per-op overhead benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecordContext


class ThousandOpModel(nn.Module):
    """Tiny model that emits about one thousand eager torch operations."""

    def __init__(self, steps: int = 250) -> None:
        """Initialize the synthetic op-count model.

        Parameters
        ----------
        steps
            Number of repeated four-op blocks.
        """

        super().__init__()
        self.steps = steps
        self.weight = nn.Parameter(torch.randn(16, 16) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run repeated elementwise and matmul operations."""

        y = x
        for _ in range(self.steps):
            y = y + 0.01
            y = torch.relu(y)
            y = y @ self.weight
            y = torch.tanh(y)
        return y


def _save_no_ops(ctx: RecordContext) -> bool:
    """Reject every operation for predicate-False benchmark cases."""

    return False


def _plain_forward(model: ThousandOpModel, x: torch.Tensor) -> None:
    """Run the model without TorchLens capture."""

    with torch.no_grad():
        model(x)


def _trace_exhaustive(model: ThousandOpModel, x: torch.Tensor) -> None:
    """Run exhaustive trace capture."""

    tl.trace(model, x, layers_to_save="none", random_seed=123)


def _trace_predicate_false(model: ThousandOpModel, x: torch.Tensor) -> None:
    """Run trace predicate capture with a predicate that never saves."""

    tl.trace(model, x, save=_save_no_ops, random_seed=123)


def _record_fastlog(model: ThousandOpModel, x: torch.Tensor) -> None:
    """Run fastlog recording with a predicate that never saves."""

    tl.record(model, x, save=_save_no_ops, random_seed=123)


def _time_case(
    name: str,
    func: Any,
    *,
    repeats: int,
    steps: int,
) -> dict[str, Any]:
    """Measure one benchmark case.

    Parameters
    ----------
    name
        Benchmark case name.
    func
        Callable accepting ``(model, x)``.
    repeats
        Number of timed repetitions.
    steps
        Repeated block count for the model.

    Returns
    -------
    dict[str, Any]
        Timing row with median seconds and per-op microseconds.
    """

    durations: list[float] = []
    for index in range(repeats + 1):
        torch.manual_seed(123)
        model = ThousandOpModel(steps=steps).eval()
        x = torch.randn(8, 16)
        started = time.perf_counter()
        func(model, x)
        duration = time.perf_counter() - started
        if index > 0:
            durations.append(duration)
    median_seconds = statistics.median(durations)
    op_count = steps * 4
    return {
        "name": name,
        "median_seconds": median_seconds,
        "per_op_us": (median_seconds / op_count) * 1_000_000,
        "runs": durations,
        "op_count_estimate": op_count,
    }


def run_benchmark(*, repeats: int = 5, steps: int = 250) -> dict[str, Any]:
    """Run all S2 capture-spine overhead cases.

    Parameters
    ----------
    repeats
        Number of timed repetitions after one warm-up.
    steps
        Repeated block count for the synthetic model.

    Returns
    -------
    dict[str, Any]
        JSON-serializable benchmark payload.
    """

    cases = (
        ("plain_forward", _plain_forward),
        ("trace_exhaustive", _trace_exhaustive),
        ("trace_predicate_false", _trace_predicate_false),
        ("record_fastlog", _record_fastlog),
    )
    rows = [_time_case(name, func, repeats=repeats, steps=steps) for name, func in cases]
    baseline = next(row for row in rows if row["name"] == "plain_forward")
    baseline_us = float(baseline["per_op_us"])
    for row in rows:
        row["over_plain_per_op_us"] = float(row["per_op_us"]) - baseline_us
        row["ratio_to_plain"] = float(row["per_op_us"]) / baseline_us if baseline_us else None
    return {
        "schema": "torchlens.s2_capture_spine_overhead.v1",
        "repeats": repeats,
        "steps": steps,
        "rows": rows,
    }


def main() -> None:
    """Run the benchmark from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/perf/s2_capture_spine_overhead.json"),
    )
    args = parser.parse_args()

    payload = run_benchmark(repeats=args.repeats, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

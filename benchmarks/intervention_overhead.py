"""Benchmark TorchLens intervention API overhead on a fixed small MLP."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchlens as tl  # noqa: E402

RESULTS_PATH = REPO_ROOT / "benchmarks" / "intervention_overhead_results.md"


class TinyMLP(nn.Module):
    """Small fixed-size feedforward model used for intervention benchmarks."""

    def __init__(self) -> None:
        """Initialize deterministic linear layers."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, 8)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, 8)``.
        """

        return self.net(x)


def _zero_hook(activation: torch.Tensor, *, hook: object) -> torch.Tensor:
    """Return a zero-valued activation for hook benchmarks.

    Parameters
    ----------
    activation:
        Activation supplied by TorchLens.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Zero-valued activation.
    """

    del hook
    return activation * 0


def _time_repeated(label: str, fn: Callable[[], object], repeats: int) -> tuple[str, float]:
    """Measure mean wall time for a repeated callable.

    Parameters
    ----------
    label:
        Human-readable benchmark label.
    fn:
        Callable to execute.
    repeats:
        Number of measured repetitions.

    Returns
    -------
    tuple[str, float]
        Benchmark label and mean seconds per call.
    """

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return label, statistics.mean(samples)


def _capture(model: nn.Module, x: torch.Tensor, *, intervention_ready: bool) -> tl.ModelLog:
    """Capture a TorchLens log for the benchmark model.

    Parameters
    ----------
    model:
        Model to capture.
    x:
        Forward input.
    intervention_ready:
        Whether to capture replay metadata.

    Returns
    -------
    tl.ModelLog
        Captured model log.
    """

    return tl.log_forward_pass(
        model,
        x,
        vis_opt="none",
        intervention_ready=intervention_ready,
        detach_saved_tensors=True,
    )


def _format_ratio(value: float, baseline: float) -> str:
    """Format a ratio relative to a baseline.

    Parameters
    ----------
    value:
        Measured value.
    baseline:
        Baseline value.

    Returns
    -------
    str
        Formatted multiplier.
    """

    return "n/a" if baseline == 0 else f"{value / baseline:.2f}x"


def _markdown_table(rows: list[tuple[str, float, str, str]]) -> str:
    """Build the markdown benchmark report.

    Parameters
    ----------
    rows:
        Benchmark rows as ``(name, seconds, ratio, notes)``.

    Returns
    -------
    str
        Markdown report contents.
    """

    lines = [
        "# Intervention Overhead Results",
        "",
        "Model: `TinyMLP` with dimensions 8 -> 16 -> 16 -> 8, batch size 32.",
        "",
        "Budget reference: PLAN.md v5.2 section 13.12 is qualitative. It requires "
        "inactive overhead to stay behind existing cheap gates, replay to scale with "
        "cone size, rerun to use pre-normalized hook dispatch, fork to stay shallow, "
        "and Bundle supergraph construction to remain lazy.",
        "",
        "| Benchmark | Mean seconds | Ratio | Notes |",
        "|---|---:|---:|---|",
    ]
    for name, seconds, ratio, notes in rows:
        lines.append(f"| {name} | {seconds:.6f} | {ratio} | {notes} |")
    lines.extend(
        [
            "",
            "Budget misses: none automatically enforced; review ratios against the "
            "qualitative v5.2 budget.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmarks() -> str:
    """Run intervention overhead benchmarks and write markdown results.

    Returns
    -------
    str
        Markdown report contents.
    """

    torch.manual_seed(0)
    model = TinyMLP().eval()
    x = torch.randn(32, 8)

    with torch.no_grad():
        for _ in range(5):
            model(x)

    baseline_label, baseline_s = _time_repeated(
        "Baseline forward",
        lambda: model(x),
        repeats=100,
    )
    nonready_label, nonready_s = _time_repeated(
        "log_forward_pass(intervention_ready=False)",
        lambda: _capture(model, x, intervention_ready=False),
        repeats=15,
    )
    ready_label, ready_s = _time_repeated(
        "log_forward_pass(intervention_ready=True)",
        lambda: _capture(model, x, intervention_ready=True),
        repeats=15,
    )

    replay_log = _capture(model, x, intervention_ready=True)
    replay_label, replay_s = _time_repeated(
        "replay(hook=zero relu)",
        lambda: replay_log.replay(hooks={tl.func("relu"): _zero_hook}),
        repeats=25,
    )

    rerun_log = _capture(model, x, intervention_ready=True)
    rerun_log.attach_hooks(tl.func("relu"), _zero_hook, confirm_mutation=True)
    rerun_label, rerun_s = _time_repeated(
        "rerun(model, x)",
        lambda: rerun_log.rerun(model, x),
        repeats=15,
    )

    logs = [_capture(model, x + idx * 0.01, intervention_ready=True) for idx in range(4)]
    bundle = tl.bundle({f"run_{idx}": log for idx, log in enumerate(logs)})
    first_relu_label = next(
        layer.layer_label for layer in logs[0].layer_list if layer.func_name == "relu"
    )
    assert bundle._supergraph is None
    bundle_label, bundle_s = _time_repeated(
        "Bundle.node() lazy supergraph build",
        lambda: bundle.node(tl.label(first_relu_label)),
        repeats=1,
    )

    rows = [
        (baseline_label, baseline_s, "1.00x", "PyTorch only"),
        (nonready_label, nonready_s, _format_ratio(nonready_s, baseline_s), "TorchLens capture"),
        (
            ready_label,
            ready_s,
            _format_ratio(ready_s, nonready_s),
            "Incremental vs non-ready capture",
        ),
        (replay_label, replay_s, _format_ratio(replay_s, baseline_s), "Saved-DAG propagation"),
        (rerun_label, rerun_s, _format_ratio(rerun_s, baseline_s), "Full forward through hooks"),
        (bundle_label, bundle_s, "n/a", "First node access builds supergraph"),
    ]
    report = _markdown_table(rows)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    """Run benchmarks and print the markdown report."""

    print(run_benchmarks())


if __name__ == "__main__":
    main()

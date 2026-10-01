"""Echo narration benchmark rows (lane F28; snoop memo build row 9).

Measures, on real architectures (torchvision resnet18 with pretrained
weights; the real HF gpt2 checkpoint when cached, else the config-built
architecture), for SUCCESS and CRASH paths:

- baseline raw forward;
- the record substrate (no echo) and record + metadata echo into a
  line-flushed file sink, a buffered /dev/null-style sink, and a callable;
- the sampled stats rung;
- the trace substrate and trace + echo;
- the crash-flush row: wall time from forward start to the tail's bytes
  being durable on disk.

Honesty rules from the memo: no published "echo at Nx" without a measured
echo row (this file IS that row's generator); absolute microseconds are
box-local and must be re-measured on the canonical perf host before any
number lands in a docstring; the RATIOS are the design-deciding output.

Run: ``python benchmarks/snoop_echo_bench.py [--repeats 5]``
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from collections.abc import Callable
from typing import Any

import torch

import torchlens as tl
from torchlens.options import EchoOptions


def _time_best(fn: Callable[[], Any], repeats: int) -> float:
    """Return the best-of-N wall seconds for one callable (min-of-N)."""

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _build_resnet18() -> tuple[torch.nn.Module, torch.Tensor]:
    """Real torchvision resnet18 with pretrained weights."""

    import torchvision

    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    ).eval()
    torch.manual_seed(0)
    return model, torch.randn(1, 3, 224, 224)


def _build_gpt2() -> tuple[torch.nn.Module, torch.Tensor, str]:
    """Real gpt2 checkpoint when cached; config-built architecture otherwise."""

    import transformers

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "gpt2", local_files_only=True
        ).eval()
        provenance = "gpt2 (real checkpoint, local cache)"
    except OSError:  # local_files_only miss: fall back to the config-built architecture
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config()).eval()
        provenance = "gpt2 (real architecture, config-built weights)"
    torch.manual_seed(0)
    input_ids = torch.randint(0, 50257, (1, 32))
    return model, input_ids, provenance


def _bench_model(name: str, model: torch.nn.Module, x: torch.Tensor, repeats: int) -> dict:
    """Measure the success-path rows for one model."""

    rows: dict[str, float] = {}
    with torch.no_grad():
        rows["baseline_forward_s"] = _time_best(lambda: model(x), repeats)

    def record_plain() -> None:
        """Record substrate, no echo, keep nothing but metadata events."""

        tl.record(model, x, save=lambda ctx: False)

    rows["record_substrate_s"] = _time_best(record_plain, repeats)

    devnull = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - held for the run
    line_counter = {"n": 0}

    def count_sink(_line: str) -> None:
        """Callable sink counting delivered lines."""

        line_counter["n"] += 1

    scenarios: dict[str, EchoOptions] = {
        "record_echo_devnull_s": EchoOptions(select=True, sink=devnull),
        "record_echo_callable_s": EchoOptions(select=True, sink=count_sink),
        "record_echo_sampled_s": EchoOptions(select=True, sink=devnull, stats="sampled"),
    }
    for key, options in scenarios.items():
        rows[key] = _time_best(lambda opt=options: tl.record(model, x, echo=opt), repeats)
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as handle:
        flush_path = handle.name
    rows["record_echo_flushed_file_s"] = _time_best(
        lambda: tl.record(model, x, echo=EchoOptions(select=True, sink=flush_path)), repeats
    )
    os.unlink(flush_path)
    rows["trace_substrate_s"] = _time_best(lambda: tl.trace(model, x, save=None), repeats)
    rows["trace_echo_devnull_s"] = _time_best(
        lambda: tl.trace(model, x, save=None, echo=EchoOptions(select=True, sink=devnull)),
        repeats,
    )
    devnull.close()

    line_counter["n"] = 0
    tl.record(model, x, echo=EchoOptions(select=True, sink=count_sink))
    lines = line_counter["n"]
    base = rows["baseline_forward_s"]

    def per_line(delta_s: float) -> float | None:
        """Microseconds per narrated line; None when below run-to-run noise."""

        if delta_s <= 0:
            return None  # below the noise floor -- never report a fake zero
        return delta_s / max(lines, 1) * 1e6

    return {
        "model": name,
        "narrated_lines": lines,
        "record_substrate_x": rows["record_substrate_s"] / base,
        "record_echo_x": rows["record_echo_devnull_s"] / base,
        "trace_substrate_x": rows["trace_substrate_s"] / base,
        "trace_echo_x": rows["trace_echo_devnull_s"] / base,
        "echo_us_per_line_devnull": per_line(
            rows["record_echo_devnull_s"] - rows["record_substrate_s"]
        ),
        "sampled_us_per_line": per_line(rows["record_echo_sampled_s"] - rows["record_substrate_s"]),
        "flushed_file_us_per_line": per_line(
            rows["record_echo_flushed_file_s"] - rows["record_substrate_s"]
        ),
        **{key: round(value, 6) for key, value in rows.items()},
    }


def _bench_crash_flush(name: str, model: torch.nn.Module, x: torch.Tensor) -> dict:
    """Measure the crash row: forward start -> tail bytes durable on disk."""

    class CrashHead(torch.nn.Module):
        """Wraps the model and crashes on an incompatible add."""

        def __init__(self) -> None:
            """Hold the real model plus a mis-shaped bias."""

            super().__init__()
            self.body = model
            self.bias = torch.nn.Parameter(torch.zeros(3))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Crash after the real body completes."""

            out = self.body(inputs)
            logits = out.logits if hasattr(out, "logits") else out
            return logits + self.bias

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as handle:
        flush_path = handle.name
    crasher = CrashHead()
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.record(
                crasher,
                x,
                echo=EchoOptions(select=True, sink=flush_path, tail_on_error=20),
                on_forward_error="return_partial",
            )
    except Exception:  # noqa: BLE001, S110 - the crash row MEASURES a failing forward
        pass
    wall = time.perf_counter() - start
    with open(flush_path, encoding="utf-8") as tail_file:
        tail_text = tail_file.read()
    os.unlink(flush_path)
    return {
        "model": name,
        "crash_wall_s": round(wall, 4),
        "tail_on_disk": "!! forward failed" in tail_text,
        "attempted_marker": "attempted call=" in tail_text,
    }


def main() -> None:
    """Run every benchmark row and print one JSON document."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    results: dict[str, Any] = {
        "host_note": (
            "box-local numbers; RATIOS are the design output, absolute values must be "
            "re-measured on the canonical perf host before entering any docstring"
        ),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    resnet, image = _build_resnet18()
    results["resnet18_success"] = _bench_model("resnet18", resnet, image, args.repeats)
    results["resnet18_crash"] = _bench_crash_flush("resnet18", resnet, image)
    gpt2, ids, provenance = _build_gpt2()
    results["gpt2_success"] = _bench_model(provenance, gpt2, ids, args.repeats)
    results["gpt2_crash"] = _bench_crash_flush(provenance, gpt2, ids)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

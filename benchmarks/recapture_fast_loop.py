"""CPU process-time yardstick for guarded static feature-extraction loops."""

from __future__ import annotations

import argparse
import copy
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One model's settled per-forward CPU process times."""

    model: str
    raw_ms: float
    hook_ms: float
    live_fast_ms: float
    sparse_fast_ms: float

    @property
    def live_marginal_ms(self) -> float:
        """Return live TorchLens-specific marginal above native plus save."""

        return self.live_fast_ms - self.hook_ms

    @property
    def sparse_marginal_ms(self) -> float:
        """Return sparse TorchLens-specific marginal above native plus save."""

        return self.sparse_fast_ms - self.hook_ms


class HookHarness:
    """Hand-rolled persistent forward-hook activation-save baseline."""

    def __init__(self, model: nn.Module, addresses: Sequence[str]) -> None:
        """Install target hooks and initialize the activation dictionary."""

        self.activations: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        modules = dict(model.named_modules())
        for address in addresses:
            module = modules[address]

            def hook(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                *,
                label: str = address,
            ) -> None:
                """Clone one tensor module output into the baseline dictionary."""

                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"Baseline target {label!r} did not return one tensor.")
                self.activations[label] = output.detach().clone()

            self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        """Remove all baseline hooks."""

        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _measure_cpu_jobs(
    jobs: Mapping[str, Callable[[], Any]], *, warmup: int, repeats: int
) -> dict[str, float]:
    """Measure rotating, interleaved median CPU process time for equivalent jobs."""

    names = tuple(jobs)
    samples: dict[str, list[float]] = {name: [] for name in names}
    with torch.inference_mode():
        for _ in range(warmup):
            for job in jobs.values():
                job()
        for repeat in range(repeats):
            offset = repeat % len(names)
            for name in (*names[offset:], *names[:offset]):
                started = time.process_time()
                jobs[name]()
                samples[name].append((time.process_time() - started) * 1_000.0)
    medians: dict[str, float] = {}
    for name, values in samples.items():
        values.sort()
        medians[name] = values[len(values) // 2]
    return medians


def _selector(addresses: Sequence[str]) -> Any:
    """Build one module-boundary save selector for benchmark targets."""

    selected = tl.module(addresses[0])
    for address in addresses[1:]:
        selected = selected | tl.module(address)
    return selected


def _benchmark_model(
    name: str,
    model: nn.Module,
    inputs: torch.Tensor,
    addresses: Sequence[str],
    *,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    """Benchmark raw, hand-hook, live-fast, and loaded-sparse-fast execution."""

    raw_model = copy.deepcopy(model).eval()
    hook_model = copy.deepcopy(model).eval()
    live_model = copy.deepcopy(model).eval()
    harness = HookHarness(hook_model, addresses)

    selector = _selector(addresses)
    trace = tl.trace(
        live_model,
        inputs,
        save=selector,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.run(inputs=inputs, fast=True)

    with tempfile.TemporaryDirectory(prefix=f"torchlens-{name}-fast-") as directory:
        artifact = Path(directory) / "model.tlspec"
        trace.save(artifact, level="runnable")
        loaded = tl.load(artifact)
        loaded.load_state_dict(live_model.state_dict())
        verified = loaded.run(inputs=inputs, fast=True)
        if verified.report.path_faithfulness.value != "verified":
            raise RuntimeError(
                f"{name} verify-once run settled {verified.report.path_faithfulness.value!r}."
            )
        timings = _measure_cpu_jobs(
            {
                "raw": lambda: raw_model(inputs),
                "hooks": lambda: hook_model(inputs),
                "live": lambda: trace.run(inputs=inputs, fast=True),
                "sparse": lambda: loaded.run(inputs=inputs, fast=True),
            },
            warmup=warmup,
            repeats=repeats,
        )
        loaded.cleanup()
    trace.cleanup()
    harness.close()
    return BenchmarkResult(
        name,
        timings["raw"],
        timings["hooks"],
        timings["live"],
        timings["sparse"],
    )


def _model_cases() -> tuple[tuple[str, nn.Module, torch.Tensor, tuple[str, ...]], ...]:
    """Construct the two required torchvision CPU benchmark cases."""

    from torchvision import models

    torch.manual_seed(20260811)
    return (
        (
            "alexnet",
            models.alexnet(weights=None),
            torch.randn(1, 3, 112, 112),
            ("features.3", "features.10", "classifier.2"),
        ),
        (
            "resnet50",
            models.resnet50(weights=None),
            torch.randn(1, 3, 112, 112),
            ("layer1.2.conv3", "layer2.3.conv3", "layer3.5.conv3", "layer4.2.conv3"),
        ),
    )


def _print_results(results: Sequence[BenchmarkResult]) -> None:
    """Print a Markdown table in the required marginal-overhead framing."""

    print(f"load average at report: {os.getloadavg()}")
    print(
        "| model | raw ms | hand hooks ms | live fast ms | live TL marginal ms | "
        "sparse fast ms | sparse TL marginal ms |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result.model} | {result.raw_ms:.3f} | {result.hook_ms:.3f} | "
            f"{result.live_fast_ms:.3f} | {result.live_marginal_ms:+.3f} | "
            f"{result.sparse_fast_ms:.3f} | {result.sparse_marginal_ms:+.3f} |"
        )


def main() -> None:
    """Run the required locked CPU benchmark cases."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    results = tuple(
        _benchmark_model(
            name,
            model,
            inputs,
            addresses,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for name, model, inputs, addresses in _model_cases()
    )
    _print_results(results)


if __name__ == "__main__":
    main()

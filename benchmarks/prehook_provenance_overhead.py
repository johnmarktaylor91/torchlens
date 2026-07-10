"""Benchmark pre-hook provenance overhead on a large module chain."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.modules.module as torch_module
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchlens as tl  # noqa: E402
from torchlens.backends.torch import prehook_provenance as provenance  # noqa: E402


class DeepReLUChain(nn.Module):
    """A root module containing a configurable chain of leaf modules."""

    def __init__(self, depth: int) -> None:
        """Initialize the module chain.

        Parameters
        ----------
        depth:
            Number of leaf ReLU modules. Including the root, the benchmark has
            ``depth + 1`` module invocations.
        """

        super().__init__()
        self.layers = nn.Sequential(*(nn.ReLU() for _ in range(depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input through every leaf module.

        Parameters
        ----------
        x:
            Benchmark input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after the full module chain.
        """

        return self.layers(x)


def _noop_global_pre_hook(_module: nn.Module, _args: tuple[Any, ...]) -> None:
    """Execute a process-global observational pre-hook."""


def _capture(model: nn.Module, x: torch.Tensor) -> None:
    """Capture one exhaustive trace and discard it.

    Parameters
    ----------
    model:
        Model under test.
    x:
        Forward input.
    """

    tl.trace(model, x)


def _capture_without_provenance(model: nn.Module, x: torch.Tensor) -> None:
    """Capture once with only provenance installation disabled.

    Parameters
    ----------
    model:
        Model under test.
    x:
        Forward input.
    """

    original = provenance.install_prehook_provenance

    def disabled_install(
        _trace: tl.Trace,
        _model: nn.Module,
        *,
        forward_hook_wrapper_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Stand in for provenance installation in the paired control."""

        del forward_hook_wrapper_factory

    setattr(provenance, "install_prehook_provenance", disabled_install)
    try:
        tl.trace(model, x)
    finally:
        setattr(provenance, "install_prehook_provenance", original)


def _median_seconds(fn: Callable[[], None], *, warmups: int, repeats: int) -> float:
    """Return median wall time for a callable.

    Parameters
    ----------
    fn:
        Callable to time.
    warmups:
        Number of unmeasured calls.
    repeats:
        Number of measured calls.

    Returns
    -------
    float
        Median seconds per call.
    """

    for _ in range(warmups):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def run_benchmark(*, depth: int, warmups: int, repeats: int) -> dict[str, float | int | str]:
    """Measure hookless and process-global-hook capture times.

    Parameters
    ----------
    depth:
        Number of leaf modules in the chain.
    warmups:
        Number of warmup captures per scenario.
    repeats:
        Number of measured captures per scenario.

    Returns
    -------
    dict[str, float | int | str]
        Machine-readable benchmark results.
    """

    torch.manual_seed(0)
    torch.set_num_threads(1)
    model = DeepReLUChain(depth).eval()
    x = torch.randn(32, 64)
    hookless_control = _median_seconds(
        lambda: _capture_without_provenance(model, x),
        warmups=warmups,
        repeats=repeats,
    )
    hookless = _median_seconds(
        lambda: _capture(model, x),
        warmups=warmups,
        repeats=repeats,
    )
    handle = torch_module.register_module_forward_pre_hook(_noop_global_pre_hook)
    try:
        global_hook_control = _median_seconds(
            lambda: _capture_without_provenance(model, x),
            warmups=warmups,
            repeats=repeats,
        )
        global_hook = _median_seconds(
            lambda: _capture(model, x),
            warmups=warmups,
            repeats=repeats,
        )
    finally:
        handle.remove()
    return {
        "torchlens_file": str(Path(tl.__file__).resolve()),
        "module_invocations": depth + 2,
        "warmups": warmups,
        "repeats": repeats,
        "hookless_control_seconds": hookless_control,
        "hookless_seconds": hookless,
        "hookless_delta_percent": (hookless / hookless_control - 1.0) * 100.0,
        "global_hook_control_seconds": global_hook_control,
        "global_hook_seconds": global_hook,
        "global_hook_delta_percent": (global_hook / global_hook_control - 1.0) * 100.0,
    }


def main() -> None:
    """Parse command-line arguments and print benchmark JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=179)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(
        json.dumps(
            run_benchmark(depth=args.depth, warmups=args.warmups, repeats=args.repeats),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

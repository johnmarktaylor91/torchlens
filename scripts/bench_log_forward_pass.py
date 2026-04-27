"""Benchmark ``log_forward_pass`` on a 20-layer MLP."""

import argparse
import os
import sys
import statistics
import time
from typing import Sequence

import torch
from torch import nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torchlens import log_forward_pass  # noqa: E402


class TwentyLayerMLP(nn.Module):
    """Simple 20-layer MLP for local logging benchmarks."""

    def __init__(self, width: int = 128) -> None:
        """Initialize the benchmark model.

        Args:
            width: Hidden dimension for every linear layer.
        """
        super().__init__()
        layers = []
        for _ in range(20):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward pass.

        Args:
            x: Input tensor of shape ``(batch_size, width)``.

        Returns:
            Model output tensor.
        """
        return self.net(x)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional command-line argument sequence.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="Timed benchmark runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warmup runs.")
    parser.add_argument("--width", type=int, default=128, help="MLP hidden width.")
    parser.add_argument("--batch-size", type=int, default=32, help="Input batch size.")
    return parser.parse_args(argv)


def run_once(model: nn.Module, x: torch.Tensor) -> float:
    """Time one ``log_forward_pass`` call.

    Args:
        model: Model to log.
        x: Input tensor for the model.

    Returns:
        Elapsed wall-clock seconds.
    """
    start = time.perf_counter()
    log_forward_pass(model, x)
    return time.perf_counter() - start


def main(argv: Sequence[str] | None = None) -> None:
    """Run the benchmark and print timing statistics.

    Args:
        argv: Optional command-line argument sequence.
    """
    args = parse_args(argv)
    torch.manual_seed(0)
    model = TwentyLayerMLP(width=args.width).eval()
    x = torch.randn(args.batch_size, args.width)

    with torch.no_grad():
        for _ in range(args.warmup):
            run_once(model, x)
        times = [run_once(model, x) for _ in range(args.runs)]

    print(f"runs: {args.runs}")
    print(f"warmup: {args.warmup}")
    print(f"width: {args.width}")
    print(f"batch_size: {args.batch_size}")
    print(f"mean_seconds: {statistics.mean(times):.6f}")
    print(f"median_seconds: {statistics.median(times):.6f}")
    print(f"min_seconds: {min(times):.6f}")
    print(f"max_seconds: {max(times):.6f}")
    print("all_seconds: " + ", ".join(f"{elapsed:.6f}" for elapsed in times))


if __name__ == "__main__":
    main()

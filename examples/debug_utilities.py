"""Short example for ``tl.debug`` helpers."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class DebugModel(nn.Module):
    """Tiny model with one numerically unstable operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a non-finite value after a linear layer."""

        hidden = torch.relu(x + 1)
        return hidden / (hidden - hidden)


def main() -> None:
    """Run NaN bisection and hot-path profiling."""

    trace = tl.trace(DebugModel(), torch.randn(2, 4))

    nan_result = tl.debug.bisect_nan(trace)
    print(nan_result.message)

    hot = tl.debug.hot_path(trace, by="memory")
    print(hot.head())


if __name__ == "__main__":
    main()

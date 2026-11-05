"""Validation completeness for an unused output of a multi-output call with no parents.

``torch.slogdet(self.weight)[1]`` reads only ``logabsdet``. The call has no recorded
parents (the weight is a parameter), so the unused ``sign`` output is kept only
because its call group survives orphan pruning. It must be logged as an internal sink
like any other dead-end output, or the backward completeness walk can never reach it.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


def _orthogonal(n: int) -> torch.Tensor:
    return torch.linalg.qr(torch.randn(n, n))[0]


class _ParamLogDet(nn.Module):
    """Invertible-convolution style log-determinant of a parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(_orthogonal(4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x @ self.weight, torch.slogdet(self.weight)[1] * 3


class _AccumulatedLogDet(nn.Module):
    """Two flow steps accumulating the log-determinant onto an input-derived zero."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(_orthogonal(4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = x.sum(1) * 0
        for _ in range(2):
            logdet = logdet + torch.slogdet(self.weight).logabsdet * 5
            x = x @ self.weight
        return x, logdet


def test_unused_sign_of_parameter_slogdet_validates() -> None:
    """The unused ``sign`` output no longer leaves the completeness walk short."""

    torch.manual_seed(0)
    assert tl.validate(_ParamLogDet(), torch.randn(3, 4), scope="forward")


def test_accumulated_parameter_logdet_validates() -> None:
    """Every pass of a looped parameter slogdet is reached."""

    torch.manual_seed(1)
    assert tl.validate(_AccumulatedLogDet(), torch.randn(3, 4), scope="forward")


def test_unused_sign_is_an_internal_sink() -> None:
    """The kept-but-unused sibling is recorded as an internal sink, not left dangling."""

    torch.manual_seed(2)
    trace = tl.trace(_ParamLogDet(), torch.randn(3, 4))
    slogdet_ops = [op for op in trace.layers if op.func_name == "slogdet"]
    assert len(slogdet_ops) == 2
    unused = [op for op in slogdet_ops if not op.children]
    assert len(unused) == 1
    assert unused[0].graph.is_internal_sink

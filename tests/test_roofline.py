"""Tests for access-time theoretical roofline properties."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._quantities import Bytes


class _ViewModel(nn.Module):
    """Model composed of logical view operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape and transpose without logical tensor movement.

        Parameters
        ----------
        x:
            Contiguous input tensor.

        Returns
        -------
        torch.Tensor
            Transposed view.
        """

        return x.reshape(2, 6).transpose(0, 1)


def _op(trace: tl.Trace, func_name: str) -> Any:
    """Return the first operation with a requested function name.

    Parameters
    ----------
    trace:
        Captured trace.
    func_name:
        Operation function name.

    Returns
    -------
    Any
        Matching operation.
    """

    return next(op for op in trace._layer_list if op.func_name == func_name)


def _bundle_bytes(path: Path) -> dict[str, bytes]:
    """Return relative file bytes for one tlspec bundle.

    Parameters
    ----------
    path:
        Bundle directory.

    Returns
    -------
    dict[str, bytes]
        Relative-path to exact file contents.
    """

    return {
        str(item.relative_to(path)): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def test_linear_theoretical_bytes_and_intensity() -> None:
    """Linear traffic is input plus weights/bias read and output written."""

    trace = tl.trace(nn.Linear(4, 3), torch.ones(2, 4, dtype=torch.float32))
    op = _op(trace, "linear")
    # Read: input 2*4*4 = 32; params (3*4 + 3)*4 = 60; total 92 bytes.
    # Write: output 2*3*4 = 24 bytes.
    assert op.cost.bytes_read == Bytes(92)
    assert op.cost.bytes_written == Bytes(24)
    assert op.cost.arithmetic_intensity == pytest.approx(float(op.cost.flops_forward) / 116)


def test_view_ops_report_zero_logical_traffic() -> None:
    """Logical views do not overcount their tensor shape as moved bytes."""

    trace = tl.trace(_ViewModel(), torch.ones(3, 4))

    for func_name in ("reshape", "transpose"):
        op = _op(trace, func_name)
        assert op.cost.bytes_read == Bytes(0)
        assert op.cost.bytes_written == Bytes(0)
        assert op.cost.arithmetic_intensity is None


def test_conv2d_theoretical_bytes_and_intensity() -> None:
    """Conv2d traffic includes the input, kernel, bias, and output once."""

    trace = tl.trace(
        nn.Conv2d(2, 3, kernel_size=3, bias=True),
        torch.ones(1, 2, 5, 5, dtype=torch.float32),
    )
    op = _op(trace, "conv2d")
    # Read: input 1*2*5*5*4 = 200; params (3*2*3*3 + 3)*4 = 228; total 428.
    # Write: output 1*3*3*3*4 = 108 bytes; total traffic 536 bytes.
    assert op.cost.bytes_read == Bytes(428)
    assert op.cost.bytes_written == Bytes(108)
    assert op.cost.arithmetic_intensity == pytest.approx(float(op.cost.flops_forward) / 536)

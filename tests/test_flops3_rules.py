"""Forward and executed-backward FLOP rules added for the classics-corpus acceptance.

Each test pins one rule against a closed form, or one TorchLens count against
another run of the same math (fused attention against the MATH backend).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._capture.compute_record import derive_compute_record_for_op


def _ops(trace: Any, func_name: str) -> list[Any]:
    return [op for op in trace._layer_list if getattr(op, "func_name", None) == func_name]


def _macs(trace: Any, func_name: str) -> int:
    """True MACs summed over one function's ops (every split must be known)."""

    records = [derive_compute_record_for_op(op) for op in _ops(trace, func_name)]
    assert records and all(record is not None and record.fma_macs is not None for record in records)
    return sum(record.fma_macs for record in records)


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/model_prep.py.
def test_weight_normed_convolution_is_counted() -> None:
    """A weight computed in forward (weight norm) is read from the call's arguments."""

    conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(4, 6, 3))
    trace = tl.trace(conv, torch.randn(2, 4, 10))
    try:
        assert _macs(trace, "conv1d") == 2 * 8 * 6 * 4 * 3
    finally:
        trace.cleanup()

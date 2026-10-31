"""Forward and executed-backward FLOP rules added for the classics-corpus acceptance.

Each test pins one rule against a closed form, or one TorchLens count against
another run of the same math (fused attention against the MATH backend).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
from torchlens.capture.compute_record import derive_compute_record_for_op
from torchlens.capture.flops import COUNTED_BEYOND_FLOP_COUNTER_MODE
from torchlens.report import backward_status


def _ops(trace: Any, func_name: str) -> list[Any]:
    return [op for op in trace._layer_list if getattr(op, "func_name", None) == func_name]


def _macs(trace: Any, func_name: str) -> int:
    """True MACs summed over one function's ops (every split must be known)."""

    records = [derive_compute_record_for_op(op) for op in _ops(trace, func_name)]
    assert records and all(record is not None and record.fma_macs is not None for record in records)
    return sum(record.fma_macs for record in records)


class _Call(nn.Module):
    def __init__(self, fn: Any) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *xs: Any) -> Any:
        return self.fn(*xs)


def test_linear_with_an_input_weight_and_keyword_bias_is_counted() -> None:
    """``F.linear(x, w, bias=b)`` with traced ``w`` and ``b``: 2MKN plus the bias adds."""

    x, w, b = torch.randn(3, 5), torch.randn(7, 5), torch.randn(7)
    trace = tl.trace(_Call(lambda x, w, b: F.linear(x, w, bias=b)), [x, w, b])
    try:
        (op,) = _ops(trace, "linear")
        assert op.flops_forward == 2 * 3 * 5 * 7 + 3 * 7
        assert _macs(trace, "linear") == 3 * 5 * 7
    finally:
        trace.cleanup()


def test_weight_normed_convolution_is_counted() -> None:
    """A weight computed in forward (weight norm) is read from the call's arguments."""

    conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(4, 6, 3))
    trace = tl.trace(conv, torch.randn(2, 4, 10))
    try:
        assert _macs(trace, "conv1d") == 2 * 8 * 6 * 4 * 3
    finally:
        trace.cleanup()


@pytest.mark.parametrize(
    ("module", "gates"),
    [
        (nn.RNN(8, 16, num_layers=2, bidirectional=True), 1),
        (nn.GRU(8, 16, num_layers=2, bidirectional=True), 3),
        (nn.LSTM(8, 16, num_layers=2, bidirectional=True), 4),
    ],
    ids=["rnn_tanh", "gru", "lstm"],
)
def test_fused_recurrent_kernels_count_every_layer_and_direction(
    module: nn.Module, gates: int
) -> None:
    """MACs = rows x (layer 0: G*H*(I+H) + layer 1: G*H*(2H+H)) per direction; states carry 0."""

    t, b, i, h = 5, 3, 8, 16
    trace = tl.trace(module, torch.randn(t, b, i))
    try:
        name = {1: "rnn_tanh", 3: "gru", 4: "lstm"}[gates]
        expected = t * b * 2 * gates * h * ((i + h) + (2 * h + h))
        assert _macs(trace, name) == expected
        assert sorted(op.flops_forward for op in _ops(trace, name))[0] == 0  # a final state
    finally:
        trace.cleanup()


def test_lstm_cell_splits_its_call_over_its_two_outputs() -> None:
    """Both outputs share one shape: half the call on each, the total exact."""

    cell = nn.LSTMCell(8, 16)
    trace = tl.trace(cell, torch.randn(3, 8))
    try:
        ops = _ops(trace, "lstm_cell")
        assert len(ops) == 2 and ops[0].flops_forward == ops[1].flops_forward
        assert _macs(trace, "lstm_cell") == 3 * 4 * 16 * (8 + 16)
    finally:
        trace.cleanup()


def test_multi_dot_counts_its_optimal_chain() -> None:
    """(10x100)(100x5)(5x50): (AB)C costs 5000 + 2500 MACs, A(BC) 25000 + 50000."""

    a, b, c = torch.randn(10, 100), torch.randn(100, 5), torch.randn(5, 50)
    trace = tl.trace(_Call(lambda a, b, c: torch.linalg.multi_dot([a, b, c])), [a, b, c])
    try:
        assert _macs(trace, "multi_dot") == 10 * 100 * 5 + 10 * 5 * 50
    finally:
        trace.cleanup()


def test_einsum_with_an_ellipsis_is_counted() -> None:
    """``...ij,...j->...i`` over (2,3,4,5) and (2,3,5): a batched matrix-vector product."""

    a, v = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 5)
    trace = tl.trace(_Call(lambda a, v: torch.einsum("...ij,...j->...i", a, v)), [a, v])
    try:
        assert _macs(trace, "einsum") == 2 * 3 * 4 * 5
    finally:
        trace.cleanup()


class _Attend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.qkv = nn.Linear(8, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).view(2, 6, 3, 2, 4).permute(2, 0, 3, 1, 4)
        return F.scaled_dot_product_attention(q, k, v)


def _executed(model: nn.Module, x: torch.Tensor) -> Any:
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True))
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    status = backward_status(trace)
    trace.cleanup()
    return status


def test_fused_attention_backward_equals_the_math_backend() -> None:
    """The CPU fused attention backward counts dP, dV, dQ, dK: the MATH backend's matmuls."""

    from torch.nn.attention import SDPBackend, sdpa_kernel

    x = torch.randn(2, 6, 8)
    fused = _executed(_Attend(), x)
    with sdpa_kernel(SDPBackend.MATH):
        math_status = _executed(_Attend(), x)
    names = dict(fused.uncounted_op_counts)
    assert "_scaled_dot_product_flash_attention_for_cpu_backward" not in names
    assert fused.executed_flops is not None and fused.executed_flops == math_status.executed_flops


class _SequenceOutput(nn.Module):
    def __init__(self, recurrent: nn.Module) -> None:
        super().__init__()
        self.recurrent = recurrent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.recurrent(x)[0]


def test_onednn_lstm_backward_is_counted() -> None:
    """The oneDNN LSTM layer backward forms dX, dW_ih, dH and dW_hh: twice the forward MACs."""

    torch.manual_seed(0)
    status = _executed(_SequenceOutput(nn.LSTM(8, 16)), torch.randn(5, 3, 8))
    assert "mkldnn_rnn_layer_backward" in COUNTED_BEYOND_FLOP_COUNTER_MODE
    assert "mkldnn_rnn_layer_backward" not in dict(status.uncounted_op_counts)
    assert status.executed_flops == 2 * 2 * 5 * 3 * 4 * 16 * (8 + 16)

"""Focused coverage for glossary-conformance phase B2b derived properties."""

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class _B2bModel(nn.Module):
    """Small model with trainable, frozen, buffer, and uncalled records."""

    def __init__(self) -> None:
        """Initialize modules used by the B2b conformance tests."""

        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.trainable = nn.Linear(4, 2)
        self.norm = nn.BatchNorm1d(4)
        self.unused = nn.Linear(4, 4)
        for param in self.frozen.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the logged forward pass."""

        return self.trainable(torch.relu(self.norm(self.frozen(x))))


def _make_trace() -> Any:
    """Create a trace with saved activations and gradients enabled."""

    model = _B2bModel()
    x = torch.randn(3, 4, requires_grad=True)
    return tl.trace(
        model,
        x,
        capture=CaptureOptions(save_grads="all"),
        save="all",
    )


def test_record_trace_aliases_and_ordinal_indexes_round_trip() -> None:
    """Records expose the owning Trace through the universal trace alias."""

    trace = _make_trace()
    loss = trace[trace.output_layers[0]].out.sum()
    trace.backward(loss)

    op = trace.ops[0]
    layer = trace.layers[0]
    module = trace.modules[0]
    module_call = trace.module_calls[0]
    param = trace.params[0]
    buffer = trace.buffers[0]
    grad_fn = trace.autograd.grad_fns[0]
    grad_fn_call = trace.autograd.grad_fn_calls[0]

    assert op.trace is trace
    assert layer.trace is trace
    assert module.trace is trace
    assert module_call.trace is trace
    assert param.trace is trace
    assert buffer.trace is trace
    assert grad_fn.trace is trace
    assert grad_fn_call.trace is trace

    assert trace.params[param.index] is param
    assert trace.autograd.grad_fns[grad_fn.ordinal_index] is grad_fn
    assert trace.autograd.grad_fn_calls[grad_fn_call.ordinal_index] is grad_fn_call
    assert sum(1 for g in trace.autograd.grad_fns if g.has_op) == sum(
        1 for record in trace.autograd.grad_fns if record.has_op
    )

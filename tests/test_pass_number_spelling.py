"""One public word for the one-based pass counter: ``pass_number``.

Every public keyword and field that takes or reports a one-based pass counts it as
``pass_number``; the older ``pass_index`` spelling refuses as an unknown keyword.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.recording._generation import GenerationStep


class _TwoLayer(nn.Module):
    """Tiny model with two saved ops on the backward path."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x))).sum()


@pytest.fixture
def two_pass_trace() -> Iterator[tl.Trace]:
    """A trace with two retained backward passes and saved gradients."""

    torch.manual_seed(0)
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(
        _TwoLayer(),
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_grads=True),
    )
    loss = trace[trace.output_layers[0]].out
    trace.backward(loss, retain_graph=True)
    trace.backward(loss, retain_graph=True)
    try:
        yield trace
    finally:
        trace.cleanup()


def test_for_pass_takes_pass_number_keyword(two_pass_trace: tl.Trace) -> None:
    """``for_pass(pass_number=)`` works on backward passes, gradient records and grad-fn calls."""

    passes = two_pass_trace.autograd.backward_passes
    assert passes.for_pass(pass_number=2).pass_number == 2

    op = next(op for op in two_pass_trace.saved_grad_ops if len(op.grads) == 2)
    assert op.grads.for_pass(pass_number=1) is op.grads.for_pass(1)

    grad_fn = next(
        grad_fn for grad_fn in two_pass_trace.autograd.grad_fns if len(grad_fn.calls) == 2
    )
    assert grad_fn.calls.for_pass(pass_number=2) is grad_fn.calls.for_pass(2)


def test_for_pass_refuses_pass_index_keyword(two_pass_trace: tl.Trace) -> None:
    """The removed ``pass_index=`` keyword refuses on all three accessors."""

    op = next(op for op in two_pass_trace.saved_grad_ops if len(op.grads) == 2)
    grad_fn = next(
        grad_fn for grad_fn in two_pass_trace.autograd.grad_fns if len(grad_fn.calls) == 2
    )
    for accessor in (two_pass_trace.autograd.backward_passes, op.grads, grad_fn.calls):
        with pytest.raises(TypeError, match="pass_index"):
            accessor.for_pass(pass_index=1)


def test_generation_step_counts_its_pass_as_pass_number() -> None:
    """``GenerationStep`` names the recorder pass holding its records ``pass_number``."""

    names = {field.name for field in dataclasses.fields(GenerationStep)}
    assert "pass_number" in names
    assert "pass_index" not in names


def test_tracelike_lookups_take_pass_number(two_pass_trace: tl.Trace) -> None:
    """``TraceLike`` pass lookups take ``pass_number``; ``pass_index=`` refuses."""

    trace = two_pass_trace
    assert trace.activations_by_pass(pass_number=1) == trace.activations_by_pass(1)
    record = next(
        record
        for record in trace._iter_activation_records()
        if trace._activation_pass_index(record) is not None
    )
    raw_label = trace._activation_raw_label(record)
    pass_number = trace._activation_pass_index(record)
    assert trace.activation_by_raw_label(raw_label, pass_number=pass_number) is record
    with pytest.raises(TypeError, match="pass_index"):
        trace.activation_by_raw_label(raw_label, pass_index=pass_number)

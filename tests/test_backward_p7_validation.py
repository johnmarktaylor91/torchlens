"""P7 backward validation oracle and event-flow corruption tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.ir.events import BackwardPassEnd, GradFnFired, OpGradObserved
from torchlens.validation.invariants import MetadataInvariantError, check_metadata_invariants


class _ValidationModel(nn.Module):
    """Small model with module grads and a nontrivial backward graph."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.fc2(self.relu(self.fc1(x)))


def _captured_backward_trace() -> tl.Trace:
    """Return a trace with materialized backward events and projections.

    Returns
    -------
    tl.Trace
        Trace after one saved backward pass.
    """

    torch.manual_seed(0)
    model = _ValidationModel()
    x = torch.randn(3, 4, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save="all", save_grads="all")
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss)
    check_metadata_invariants(trace)
    return trace


def test_backward_event_flow_invariants_trip_on_missing_bracket_end() -> None:
    """A pass start without a matching end is rejected."""

    trace = _captured_backward_trace()
    try:
        events = trace._capture_events.backward_events
        end_index = next(
            index for index, event in enumerate(events) if isinstance(event, BackwardPassEnd)
        )
        events.pop(end_index)

        with pytest.raises(MetadataInvariantError, match="exactly one start and end"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_backward_event_flow_invariants_trip_on_missing_op_label() -> None:
    """An ``OpGradObserved`` event cannot point at an unknown op."""

    trace = _captured_backward_trace()
    try:
        events = trace._capture_events.backward_events
        op_index = next(
            index for index, event in enumerate(events) if isinstance(event, OpGradObserved)
        )
        events[op_index] = replace(events[op_index], op_label="missing_op_1")

        with pytest.raises(MetadataInvariantError, match="missing op label"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_backward_event_flow_invariants_trip_on_projection_mismatch() -> None:
    """Projected op gradient records must mirror the event stream."""

    trace = _captured_backward_trace()
    try:
        victim = next(layer for layer in trace.layer_list if getattr(layer, "_grad_records", ()))
        victim._grad_records.clear()

        with pytest.raises(MetadataInvariantError, match="projected op gradient records"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_backward_event_flow_invariants_trip_on_missing_grad_fn_fire_target() -> None:
    """A ``GradFnFired`` event cannot point at an unknown GradFn record."""

    trace = _captured_backward_trace()
    try:
        events = trace._capture_events.backward_events
        fired_index = next(
            index for index, event in enumerate(events) if isinstance(event, GradFnFired)
        )
        events[fired_index] = replace(events[fired_index], object_id=-1)

        with pytest.raises(MetadataInvariantError, match="missing grad_fn id"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()

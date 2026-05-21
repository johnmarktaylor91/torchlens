"""Backward observer tests for tap and record spans."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.observers import TapObserver


class _TinyRelu(nn.Module):
    """Tiny module with a paired ReLU grad_fn_handle."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer followed by ReLU.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.relu(self.linear(x))


def _trace_with_tap(tap: TapObserver) -> object:
    """Capture a trace with a tap and run backward.

    Parameters
    ----------
    tap:
        Tap observer to mount during capture.

    Returns
    -------
    object
        Trace after backward capture.
    """

    torch.manual_seed(12)
    model = _TinyRelu()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, intervention_ready=True, hooks=tap)
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss)
    return trace


def test_tap_backward_tensor_mount_forward_selector() -> None:
    """Backward taps mount through direction-agnostic forward metadata selectors."""

    tap = tl.tap(tl.label("output_1"), direction="backward")
    _trace_with_tap(tap)

    assert len(tap.records) == 1
    record = tap.records[0]
    assert record.direction == "backward"
    assert record.grad_kind == "grad_output"
    assert record.backward_call_index == 1
    assert torch.equal(record.value, torch.ones(2, 3))


def test_tap_backward_grad_fn_mount_backward_selector() -> None:
    """Backward taps mount directly on grad_fn_handle selectors."""

    tap = tl.tap(tl.grad_fn(type="relu"), direction="backward")
    _trace_with_tap(tap)

    assert len(tap.records) == 1
    assert tap.records[0].direction == "backward"
    assert tap.records[0].site_label is not None
    assert tap.records[0].site_label.startswith("relu_back")


def test_tap_backward_both_directions_fire() -> None:
    """A both-direction tap records forward and backward events."""

    tap = tl.tap(tl.contains("linear"), direction="both")
    _trace_with_tap(tap)

    assert [record.direction for record in tap.records] == ["forward", "backward"]
    assert tap.records[0].backward_call_index is None
    assert tap.records[1].backward_call_index == 1


def test_tap_backward_intervening_grad_fn_only_grad_fn_mount() -> None:
    """Intervening selectors fire on backward-only grad_fn_handle sites."""

    tap = tl.tap(tl.intervening(), direction="backward")
    _trace_with_tap(tap)

    assert tap.records
    assert all(record.direction == "backward" for record in tap.records)
    assert any(
        record.site_label is not None and "sum_back" in record.site_label for record in tap.records
    )


def test_record_span_backward_direction() -> None:
    """Backward records include active span names."""

    tap = tl.tap(tl.grad_fn(type="relu"), direction="backward")
    with tl.record_span("backward_phase", direction="backward"):
        _trace_with_tap(tap)

    assert tap.records[0].span_names == ("backward_phase",)


def test_record_span_both_direction_captures_both() -> None:
    """Both-direction spans annotate forward and backward tap records."""

    tap = tl.tap(tl.contains("linear"), direction="both")
    with tl.record_span("both_phase", direction="both"):
        _trace_with_tap(tap)

    assert [record.span_names for record in tap.records] == [
        ("both_phase",),
        ("both_phase",),
    ]


def test_record_span_backward_event_inside_active_span() -> None:
    """A span opened after forward capture still annotates backward records."""

    torch.manual_seed(12)
    tap = tl.tap(tl.grad_fn(type="relu"), direction="backward")
    model = _TinyRelu()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x, intervention_ready=True, hooks=tap)
    loss = trace[trace.output_layers[0]].out.sum()

    with tl.record_span("backward_only", direction="backward"):
        trace.log_backward(loss)

    assert tap.records[0].span_names == ("backward_only",)


def test_tap_record_direction_field() -> None:
    """TapRecord.direction distinguishes forward from backward events."""

    tap = tl.tap(tl.contains("linear"), direction="both")
    _trace_with_tap(tap)

    assert {record.direction for record in tap.records} == {"forward", "backward"}


def test_tap_record_grad_kind_field() -> None:
    """Backward TapRecord.grad_kind identifies the captured gradient payload."""

    tap = tl.tap(tl.grad_fn(type="relu"), direction="backward")
    _trace_with_tap(tap)

    assert tap.records[0].grad_kind in {"grad_input", "grad_output"}


def test_tap_record_backward_call_index() -> None:
    """Backward TapRecord.backward_call_index is one-based."""

    tap = tl.tap(tl.grad_fn(type="relu"), direction="backward")
    _trace_with_tap(tap)

    assert tap.records[0].backward_call_index == 1


def test_tap_per_call_vs_per_event_count() -> None:
    """Backward taps record one event for each retained backward sweep."""

    torch.manual_seed(12)
    tap = tl.tap(tl.grad_fn(type="relu"), direction="backward")
    trace = tl.trace(_TinyRelu(), torch.randn(2, 3), intervention_ready=True, hooks=tap)
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss, retain_graph=True)
    trace.log_backward(loss, retain_graph=True)

    assert [record.backward_call_index for record in tap.records] == [1, 2]

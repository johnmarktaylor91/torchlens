"""Backward observer tests for tap and record spans."""

from __future__ import annotations

import pytest
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


class _Identity(nn.Module):
    """Identity module for input/output alias coverage."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Unchanged input tensor.
        """

        return x


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
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(replayable=True, hooks=tap),
    )
    loss = trace[trace.output_layers[0]].out.sum()
    trace.backward(loss)
    return trace


@pytest.mark.parametrize(
    "label",
    ["output_1", "relu_1_2"],
    ids=[
        "tap_backward_tensor_mount_forward_selector",
        "tap_backward_tensor_mount_compute_op_still_matches",
    ],
)
def test_tap_backward_tensor_mount(label: str) -> None:
    """tl.grad(label) taps the output gradient on either mount point.

    ``output_1`` mounts through direction-agnostic forward metadata selectors;
    a tap mounted on the ReLU compute op still sees the gradient stream.

    Parameters
    ----------
    label:
        Final op label the backward tap is mounted on.
    """

    tap = tl.observers.tap(tl.grad(tl.label(label)))
    _trace_with_tap(tap)

    assert len(tap.records) == 1
    record = tap.records[0]
    assert record.signal == "backward"
    assert record.site_label == label
    assert record.grad_kind == "grad_output"
    assert record.backward_call_index == 1
    assert torch.equal(record.value, torch.ones(2, 3))


def test_tap_backward_tensor_mount_input_alias_matches() -> None:
    """Backward taps mounted on a non-leaf input alias see its gradient stream."""

    torch.manual_seed(12)
    base = torch.randn(2, 3, requires_grad=True)
    x = base * 2
    tap = tl.observers.tap(tl.grad(tl.label("input_1")))
    trace = tl.trace(
        _Identity(),
        x,
        capture=tl.options.CaptureOptions(replayable=True, hooks=tap),
    )
    loss = trace[trace.output_layers[0]].out.sum()
    trace.backward(loss)

    assert len(tap.records) == 1
    record = tap.records[0]
    assert record.signal == "backward"
    assert record.grad_kind == "grad_output"
    assert record.backward_call_index == 1
    assert torch.equal(record.value, torch.ones(2, 3))


def test_tap_backward_grad_fn_mount_backward_selector() -> None:
    """Backward taps mount directly on grad_fn_handle selectors."""

    tap = tl.observers.tap(tl.grad_fn(name="relu"))
    _trace_with_tap(tap)

    assert len(tap.records) == 1
    assert tap.records[0].signal == "backward"
    assert tap.records[0].site_label is not None
    assert tap.records[0].site_label.startswith("relu_back")


def test_tap_gradient_and_activation_union_fires_both() -> None:
    """tap(tl.grad(s) | s) records the activation and the gradient."""

    tap = tl.observers.tap(tl.grad(tl.label_contains("linear")) | tl.label_contains("linear"))
    _trace_with_tap(tap)

    assert [record.signal for record in tap.records] == ["forward", "backward"]
    assert tap.records[0].backward_call_index is None
    assert tap.records[1].backward_call_index == 1


def test_tap_backward_intervening_grad_fn_only_grad_fn_mount() -> None:
    """Intervening selectors fire on backward-only grad_fn_handle sites."""

    tap = tl.observers.tap(tl.without_op())
    _trace_with_tap(tap)

    assert tap.records
    assert all(record.signal == "backward" for record in tap.records)
    assert any(
        record.site_label is not None and "sum_back" in record.site_label for record in tap.records
    )


def test_record_span_backward_signal() -> None:
    """Backward records include active span names."""

    tap = tl.observers.tap(tl.grad_fn(name="relu"))
    with tl.observers.span("backward_phase", signal="backward"):
        _trace_with_tap(tap)

    assert tap.records[0].span_names == ("backward_phase",)


def test_record_span_both_signal_captures_both() -> None:
    """signal="both" spans annotate forward and backward tap records."""

    tap = tl.observers.tap(tl.grad(tl.label_contains("linear")) | tl.label_contains("linear"))
    with tl.observers.span("both_phase", signal="both"):
        _trace_with_tap(tap)

    assert [record.span_names for record in tap.records] == [
        ("both_phase",),
        ("both_phase",),
    ]


def test_record_span_backward_event_inside_active_span() -> None:
    """A span opened after forward capture still annotates backward records."""

    torch.manual_seed(12)
    tap = tl.observers.tap(tl.grad_fn(name="relu"))
    model = _TinyRelu()
    x = torch.randn(2, 3)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(replayable=True, hooks=tap),
    )
    loss = trace[trace.output_layers[0]].out.sum()

    with tl.observers.span("backward_only", signal="backward"):
        trace.backward(loss)

    assert tap.records[0].span_names == ("backward_only",)


def test_tap_record_signal_field() -> None:
    """TapRecord.signal distinguishes forward from backward events."""

    tap = tl.observers.tap(tl.grad(tl.label_contains("linear")) | tl.label_contains("linear"))
    _trace_with_tap(tap)

    assert {record.signal for record in tap.records} == {"forward", "backward"}


def test_tap_record_grad_kind_field() -> None:
    """Backward TapRecord.grad_kind identifies the captured gradient payload."""

    tap = tl.observers.tap(tl.grad_fn(name="relu"))
    _trace_with_tap(tap)

    assert tap.records[0].grad_kind in {"grad_input", "grad_output"}


def test_tap_record_backward_call_index() -> None:
    """Backward TapRecord.backward_call_index is one-based."""

    tap = tl.observers.tap(tl.grad_fn(name="relu"))
    _trace_with_tap(tap)

    assert tap.records[0].backward_call_index == 1


def test_tap_per_call_vs_per_event_count() -> None:
    """Backward taps record one event for each retained backward sweep."""

    torch.manual_seed(12)
    tap = tl.observers.tap(tl.grad_fn(name="relu"))
    trace = tl.trace(
        _TinyRelu(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(replayable=True, hooks=tap),
    )
    loss = trace[trace.output_layers[0]].out.sum()
    trace.backward(loss, retain_graph=True)
    trace.backward(loss, retain_graph=True)

    assert [record.backward_call_index for record in tap.records] == [1, 2]

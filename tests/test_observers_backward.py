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


def test_tap_backward_intervening_grad_fn_only_grad_fn_mount() -> None:
    """Intervening selectors fire on backward-only grad_fn_handle sites."""

    tap = tl.observers.tap(tl.without_op())
    _trace_with_tap(tap)

    assert tap.records
    assert all(record.signal == "backward" for record in tap.records)
    assert any(
        record.site_label is not None and "sum_back" in record.site_label for record in tap.records
    )


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


def test_tap_record_backward_call_index() -> None:
    """Backward TapRecord.backward_call_index is one-based."""

    tap = tl.observers.tap(tl.grad_fn(name="relu"))
    _trace_with_tap(tap)

    assert tap.records[0].backward_call_index == 1

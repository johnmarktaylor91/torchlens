"""Observers on gradient sites: tap, narrate and span(signal=).

The site decides which tensor an observer sees; ``tap`` and ``narrate`` take
no pass keyword, and ``span(name, signal=...)`` is the one door that keeps a
keyword because a span has no site. The first block drives the observers with
hook contexts directly; the second needs ``tl.grad`` and the output-gradient
mount and is skipped until both are present.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.types import TargetSpec


def _ctx(signal: str, label: str = "relu_1_2", **kwargs: Any) -> Any:
    return SimpleNamespace(signal=signal, op={"layer_label": label}, kwargs=kwargs)


def test_tap_has_no_pass_field_or_keyword() -> None:
    tap = tl.observers.tap(tl.func("relu"))
    assert not hasattr(tap, "direction")
    assert not hasattr(tap, "signal")
    assert list(inspect.signature(tl.observers.tap).parameters) == ["site"]


def test_tap_records_the_pass_its_hook_context_names() -> None:
    tap = tl.observers.tap(tl.func("relu"))
    value = torch.ones(2)
    assert tap(value, ctx=_ctx("forward")) is value
    assert tap(2 * value, ctx=_ctx("backward", grad_kind="grad_output", call_index=1)) is not None
    forward, backward = tap.records
    assert forward.signal == "forward"
    assert forward.grad_kind is None
    assert backward.signal == "backward"
    assert backward.grad_kind == "grad_output"
    assert backward.backward_call_index == 1
    assert backward.site_label == "relu_1_2"
    assert torch.equal(backward.value, 2 * value)
    assert not hasattr(forward, "direction")


def test_span_scopes_records_by_signal() -> None:
    tap = tl.observers.tap(tl.func("relu"))
    value = torch.ones(2)
    with tl.observers.span("bwd-only", signal="backward") as record:
        assert record["signal"] == "backward"
        assert "direction" not in record
        tap(value, ctx=_ctx("forward"))
        tap(value, ctx=_ctx("backward", grad_kind="grad_output", call_index=1))
    with tl.observers.span("everything"):
        tap(value, ctx=_ctx("forward"))
    assert [r.span_names for r in tap.records] == [(), ("bwd-only",), ("everything",)]


def test_span_refuses_an_unknown_signal() -> None:
    with (
        pytest.raises(ValueError, match="signal must be 'forward', 'backward', or 'both'"),
        tl.observers.span("bad", signal="sideways"),  # type: ignore[arg-type]
    ):
        pass
    assert "direction" not in inspect.signature(tl.observers.span).parameters


def test_narrate_signal_is_derived_from_the_site() -> None:
    grad_site = TargetSpec(selector_kind="grad", selector_value=TargetSpec("label", "relu_1_2"))
    assert tl.stats.narrate(tl.func("relu")).signal == "forward"
    assert tl.stats.narrate(grad_site).signal == "backward"
    assert "signal" not in inspect.signature(tl.stats.narrate).parameters
    assert "direction" not in inspect.signature(tl.stats.narrate).parameters


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


_X = torch.randn(3, 4, generator=torch.Generator().manual_seed(11))


def _relu_label(model: nn.Module) -> str:
    trace = tl.trace(model, _X)
    return next(str(op.layer_label) for op in trace.layer_list if op.func_name == "relu")


def test_tap_on_a_gradient_site_records_backward_fires() -> None:  # A29
    model = _Net()
    relu = _relu_label(model)
    tap = tl.observers.tap(tl.grad(tl.func("relu")))
    trace = tl.trace(
        model,
        _X,
        capture=tl.options.CaptureOptions(intervention_ready=True, backward_ready=True, hooks=tap),
    )
    trace.backward(trace[trace.output_layers[0]].out.sum())
    (record,) = tap.records
    assert record.signal == "backward"
    assert record.grad_kind == "grad_output"
    assert record.site_label == relu
    assert record.backward_call_index is not None


def test_tap_on_a_mixed_site_records_both_passes() -> None:  # A29
    model = _Net()
    relu = _relu_label(model)
    tap = tl.observers.tap(tl.grad(tl.func("relu")) | tl.func("relu"))
    trace = tl.trace(
        model,
        _X,
        capture=tl.options.CaptureOptions(intervention_ready=True, backward_ready=True, hooks=tap),
    )
    trace.backward(trace[trace.output_layers[0]].out.sum())
    assert {record.signal for record in tap.records} == {"forward", "backward"}
    assert tl.stats.narrate(tl.grad(relu)).signal == "backward"
    assert tl.stats.narrate(relu).signal == "forward"

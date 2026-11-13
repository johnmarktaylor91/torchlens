"""Trace.backward and Trace.capture_backward dispatch through the backend registry."""

from __future__ import annotations

import dataclasses
import types
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends import _backward_neutral as neutral, registry
from torchlens.backends._backward_native import NativeBackwardSurface
from torchlens.errors import RunCapabilityUnavailableError


class _Chain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _trace(seed: int = 0) -> tl.Trace:
    torch.manual_seed(seed)
    return tl.trace(
        _Chain(),
        torch.randn(3, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
        save="all",
    )


def _patched(monkeypatch: pytest.MonkeyPatch, name: str, surface: Any) -> None:
    spec = registry.get_backend_spec(name)
    replaced = dataclasses.replace(
        spec,
        capabilities=dataclasses.replace(spec.capabilities, backward_capture=True),
        capability_implementations={
            **(spec.capability_implementations or {}),
            "backward_capture": lambda: surface,
        },
    )
    monkeypatch.setitem(registry._REGISTRY, name, replaced)


def test_torch_backward_dispatches_through_the_registered_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _trace()
    seen: list[tuple[str, Any]] = []
    stub = types.SimpleNamespace(
        trace_backward=lambda trace, loss, **kw: seen.append(("backward", kw)) or trace,
        recording_backward=lambda trace: seen.append(("capture", None)) or "context",
    )
    _patched(monkeypatch, "torch", stub)
    loss = trace[trace.output_layers[0]].out.sum()
    assert trace.backward(loss, retain_graph=True) is trace
    assert trace.capture_backward() == "context"
    assert seen == [("backward", {"retain_graph": True}), ("capture", None)]


def test_torch_backward_still_records_through_the_default_registry() -> None:
    trace = _trace()
    trace.backward(trace[trace.output_layers[0]].out.sum())
    assert len(trace.autograd.backward_passes) == 1
    assert len(trace.autograd.grad_fns) > 0


def _donor_events() -> tuple[list[Any], dict[int, list[int]], tl.Trace]:
    """Backward events observed on an identical trace, to replay as a native lane."""

    donor = _trace()
    donor.backward(donor[donor.output_layers[0]].out.sum())
    assert len(donor.autograd.grad_fns) > 0
    events = [dataclasses.replace(event) for event in donor._capture_events.backward_events]
    return events, dict(donor.__dict__.get("_backward_roots_by_pass", {})), donor


def _native_module(events: list[Any], roots: dict[int, list[int]]) -> types.ModuleType:
    module = types.ModuleType("fake_native_backward")
    calls: list[Any] = []

    def record_backward(trace: Any, loss: Any, **kwargs: Any) -> None:
        calls.append((loss, kwargs))
        for event in events:
            neutral.journal_backward_event(trace, event)
        for pass_index, root_ids in roots.items():
            neutral.set_pass_roots(trace, pass_index, root_ids)

    module.record_backward = record_backward  # type: ignore[attr-defined]
    module.calls = calls  # type: ignore[attr-defined]
    return module


def test_native_surface_builds_records_from_journaled_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, roots, donor = _donor_events()
    module = _native_module(events, roots)
    _patched(monkeypatch, "jax", NativeBackwardSurface("jax", module))
    target = _trace()
    target._provenance_backend = "jax"

    assert target.backward("loss-callable", retain_graph=True) is target
    assert module.calls == [("loss-callable", {"retain_graph": True})]
    want = [record.class_name for record in donor._grad_fn_logs.values()]
    assert [record.class_name for record in target._grad_fn_logs.values()] == want
    assert sorted(target._backward_pass_logs) == [1]
    observed = [
        event
        for event in events
        if type(event).__name__ == "OpGradObserved" and isinstance(event.payload_ref, torch.Tensor)
    ]
    assert observed
    for event in observed:
        op = target._layer_dict_all_keys[event.op_label]
        grads = [record.grad for record in op._slot("_grad_records") or ()]
        assert any(grad is event.payload_ref for grad in grads), event.op_label


def test_native_sync_runs_on_read_through_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    events, roots, donor = _donor_events()
    _patched(monkeypatch, "jax", NativeBackwardSurface("jax", _native_module([], {})))
    target = _trace()
    target._provenance_backend = "jax"
    for event in events:
        neutral.journal_backward_event(target, event)
    for pass_index, root_ids in roots.items():
        neutral.set_pass_roots(target, pass_index, root_ids)

    assert len(target.autograd.grad_fns) == len(donor._grad_fn_logs)
    assert target._backward_projection_revision == target._capture_events.backward_revision


def test_native_surface_refuses_capture_backward_without_a_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patched(monkeypatch, "jax", NativeBackwardSurface("jax", _native_module([], {})))
    target = _trace()
    target._provenance_backend = "jax"
    with pytest.raises(RunCapabilityUnavailableError, match="capture_backward"):
        target.capture_backward()

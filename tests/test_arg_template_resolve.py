"""The argument template resolves from facts observed at the call, after the call is gone.

``_set_arg_template_fields`` observes every fact the replay template carries at the call
and hands them to ``resolve_arg_template_fields``. The resolver must be a pure function of
those facts and the layout plan: resolved again after the capture, once the call's tensors
were mutated in place and freed, it gives the values the capture stored at the call. The
recorded facts hold no argument tensor except inside a literal component (the call-time
copy of an unlabeled tensor, or a parameter the template holds by design).
"""

from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends.torch import (
    _arg_template_resolve,
    _args_template_cache,
    _ops_argument_fields,
    _ops_arguments,
)
from torchlens.intervention._types import (
    CapturedArgTemplate,
    LiteralTensor,
    LiteralValue,
    ParentRef,
)


class _MixedArgs(nn.Module):
    """Literal, keyword, list, parameter, non-contiguous and unlabeled tensor arguments."""

    def __init__(self) -> None:
        """Build a linear layer and an unregistered plain tensor attribute."""
        super().__init__()
        self.lin = nn.Linear(6, 6)
        # Neither a parameter, a buffer nor a closure constant: an unlabeled argument.
        self.offset = torch.linspace(-1.0, 1.0, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix every argument kind the template classifies."""
        h = self.lin(x)
        h = torch.clamp(h, min=-0.5, max=0.5)
        h = h * 2.0 + self.offset
        side = h.t()
        prod = torch.matmul(side, h)
        joined = torch.cat([h, prod[: h.shape[0]]], dim=1)
        return torch.nn.functional.linear(joined[:, :6], self.lin.weight, bias=None).sum(dim=-1)


def _norm(component: Any) -> Any:
    """Return a comparable form of one template component (tensors by value)."""

    if isinstance(component, ParentRef):
        layout = component._capture_layout
        return ("parent", component.parent_label, None if layout is None else repr(layout))
    if isinstance(component, LiteralTensor):
        value = component.value
        return (
            "tensor",
            id(value) if isinstance(value, nn.Parameter) else None,
            tuple(value.shape),
            str(value.dtype),
            value.detach().flatten().tolist(),
            component.param_barcode,
            repr(component._capture_param_fingerprint),
        )
    if isinstance(component, CapturedArgTemplate):
        return (
            "template",
            _norm(component.args),
            tuple((name, _norm(value)) for name, value in component.kwargs),
            component.func_id,
            component.notes,
        )
    if isinstance(component, tuple):
        return tuple(_norm(item) for item in component)
    return (type(component).__name__, repr(component))


def _norm_fields(fields: dict[str, Any]) -> tuple[Any, ...]:
    """Return a comparable form of the three fields the resolver writes."""

    return (
        _norm(fields["args_template"]),
        _norm(fields["kwargs_template"]),
        fields["func_id"],
        fields["kwargs_template"] is fields["args_template"],
    )


def _stray_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor in recorded facts that is not inside a literal component."""

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _stray_tensors(item)
    elif isinstance(value, ParentRef | LiteralValue):
        return


def _record(
    monkeypatch: pytest.MonkeyPatch, model: nn.Module, x: torch.Tensor
) -> list[tuple[Any, tuple[Any, ...], tuple[Any, ...]]]:
    """Trace ``model`` and return ``(observed facts, stored fields, first-resolve norm)`` per op."""

    captured: list[tuple[Any, dict[str, Any]]] = []
    real = _ops_argument_fields.resolve_arg_template_fields

    def recording(fields_dict: dict[str, Any], observed: Any, parts: Any) -> None:
        """Resolve as the capture does and keep the facts with the stored values."""
        real(fields_dict, observed, parts)
        stored = {
            name: fields_dict[name] for name in ("args_template", "kwargs_template", "func_id")
        }
        captured.append((observed, stored))

    with monkeypatch.context() as patch:
        patch.setattr(_ops_argument_fields, "resolve_arg_template_fields", recording)
        tl.trace(model, x)
    assert captured, "the capture never observed a template"
    # Normalised after the capture: ``_norm`` runs torch functions, which the capture
    # would log (and template again) if they ran inside it. Nothing has been mutated yet.
    return [(observed, stored, _norm_fields(stored)) for observed, stored in captured]


def _resolve(observed: Any) -> tuple[Any, ...]:
    """Resolve recorded facts into a fresh field dict and return its comparable form."""

    fields: dict[str, Any] = {}
    _arg_template_resolve.resolve_arg_template_fields(
        fields, observed, _ops_arguments._TEMPLATE_PARTS
    )
    return _norm_fields(fields)


def test_resolver_gives_call_time_values_after_mutation_and_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolving after the inputs were mutated and freed returns what the call stored."""

    torch.manual_seed(0)
    model = _MixedArgs()
    x = torch.randn(4, 6)
    records = _record(monkeypatch, model, x)
    kinds = {type(c).__name__ for _, stored, _ in records for c in _walk(stored["args_template"])}
    assert {"ParentRef", "LiteralTensor", "LiteralValue"} <= kinds, kinds
    assert any(
        isinstance(c, ParentRef) and c._capture_layout is not None
        for _, stored, _ in records
        for c in _walk(stored["args_template"])
    ), "no non-contiguous parent argument was templated"
    assert any(stored["kwargs_template"] is not None for _, stored, _ in records)

    # Mutate every tensor the call read, then drop the model and the input.
    with torch.no_grad():
        x.mul_(-3.0)
        model.offset.add_(10.0)
        model.lin.weight.mul_(0.25)
    del x, model
    gc.collect()

    for observed, stored, at_call in records:
        assert _strip_param_values(_norm_fields(stored)) == _strip_param_values(at_call)
        again = _resolve(observed)
        # Parameter literals hold the parameter itself (by design), so their values follow
        # the in-place change on both sides; every other component is the call-time value.
        assert _strip_param_values(again) == _strip_param_values(at_call)
        assert again == _resolve(observed), "the resolver is not deterministic"


def test_recorded_facts_hold_no_live_argument_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every tensor in the facts is a literal component's copy or a parameter."""

    torch.manual_seed(0)
    model = _MixedArgs()
    x = torch.randn(4, 6)
    records = _record(monkeypatch, model, x)
    for observed, _, _ in records:
        stray = [t for t in _stray_tensors(observed) if not isinstance(t, nn.Parameter)]
        assert stray == [], f"facts hold {len(stray)} live tensors"
        for component in _walk(observed):
            if isinstance(component, LiteralTensor) and not isinstance(
                component.value, nn.Parameter
            ):
                assert component.value is not x
                assert component.value is not model.offset


def test_general_observation_resolves_like_the_classifier() -> None:
    """With no plan, the observed components resolve to ``_build_args_template``'s value."""

    t = torch.arange(6.0).reshape(2, 3)
    call = (torch.add, "add", (t, [t, 2], 1.5), {"alpha": 2})
    expected = _norm(
        _ops_arguments._build_args_template(torch.add, call[2], call[3], None, func_name="add")
    )
    facts = _args_template_cache.observe_template(_ops_arguments._TEMPLATE_PARTS, call, None, None)
    assert facts[0] is None, "a call with no trace has no plan"
    t.mul_(7.0)
    del t
    gc.collect()
    resolved = _arg_template_resolve.resolve_arg_template(_ops_arguments._TEMPLATE_PARTS, facts)
    assert _norm(resolved) == expected


def test_no_template_capture_resolves_function_identity_only() -> None:
    """A capture that records no template resolves the observed function key alone."""

    key = _ops_arguments._function_registry_key(torch.relu, "relu", None)
    fields: dict[str, Any] = {}
    _arg_template_resolve.resolve_arg_template_fields(
        fields, (None, False, False, key), _ops_arguments._TEMPLATE_PARTS
    )
    assert fields == {"args_template": None, "kwargs_template": None, "func_id": key}
    assert list(fields) == ["args_template", "kwargs_template", "func_id"]


def test_general_seam_is_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forcing the plan off sends the capture through the general classifier."""

    calls = []
    real = _ops_arguments._TEMPLATE_PARTS.classify

    def counting(*args: Any, **kwargs: Any) -> Any:
        """Count general classifications."""
        calls.append(1)
        return real(*args, **kwargs)

    torch.manual_seed(0)
    model, x = _MixedArgs(), torch.randn(4, 6)
    parts = _ops_arguments._TEMPLATE_PARTS
    forced = type(parts)(
        **{**{f: getattr(parts, f) for f in parts.__slots__}, "classify": counting}
    )
    with monkeypatch.context() as patch:
        patch.setattr(_ops_arguments, "_TEMPLATE_PARTS", forced)
        tl.trace(model, x)
        planned_calls = len(calls)
        calls.clear()
        patch.setattr(_args_template_cache, "_plan_of", lambda *_args: None)
        tl.trace(model, x)
    assert len(calls) > planned_calls


def _walk(value: Any) -> Iterator[Any]:
    """Yield every component and fact nested in ``value``."""

    yield value
    if isinstance(value, CapturedArgTemplate):
        yield from _walk(value.args)
        yield from _walk(tuple(component for _, component in value.kwargs))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)


def _strip_param_values(normed: Any) -> Any:
    """Drop the value of parameter literals (a parameter is held, not copied)."""

    if isinstance(normed, tuple):
        if len(normed) == 7 and normed[0] == "tensor" and normed[1] is not None:
            return normed[:4] + normed[5:]
        return tuple(_strip_param_values(item) for item in normed)
    return normed

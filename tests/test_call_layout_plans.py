"""Per-call argument layout plans change no recorded field.

A flat call (every argument a tensor or a value no argument walk looks inside) is
split, snapshotted and witnessed through the plan of its layout, kept per capture
and guarded on every call by the callable and the exact argument and keyword types
(``backends.torch._call_layout``). Each subject is traced with plans on and with
every call sent down the general path; the parents, argument positions, witness
verdicts, non-tensor arguments, templates, equivalence classes and parameter edges
must be equal op for op. The subjects drive one call site through the layout
changes the guard must catch: a tensor where a non-tensor was (and the reverse), a
changed keyword set, a changed non-tensor value, a Parameter swapped for a plain
tensor, an in-place call after an out-of-place one, a flat call after a nested one,
an unattributed tensor argument, and recursion through one site.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens.backends.torch import _call_layout, _ops_emission, _ops_shared_fields
from torchlens.backends.torch._ops_arg_slots import _SlotCall
from torchlens.intervention._types import CapturedArgTemplate, LiteralTensor, ParentRef


def _add_at_one_site(y: torch.Tensor, other: Any, **kwargs: Any) -> torch.Tensor:
    """Call ``torch.add`` from one line, whatever the arguments."""
    return torch.add(y, other, **kwargs)


class _SlotKinds(nn.Module):
    """One site gets a tensor, then an int, then a tensor, then a float."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alternate tensor and non-tensor second operands."""
        y = x
        for other in (x, 2, x * 2, 1.5, x):
            y = _add_at_one_site(y, other)
        return y


class _KwargSets(nn.Module):
    """One site gets no keywords, then ``alpha``, then a different ``alpha``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Change the keyword set and value at one site."""
        y = x
        for kwargs in ({}, {"alpha": 2}, {"alpha": 3}, {}, {"alpha": 2}):
            y = _add_at_one_site(y, x, **kwargs)
        return y


class _HashValues(nn.Module):
    """One site gets different non-tensor values (they feed the hash and template)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add values that change, repeat and change type."""
        y = x
        for value in (1, 2, 1, 2.5, True, 1, 10**12, 10**12 + 1):
            y = _add_at_one_site(y, value)
        return y


class _ParamSwap(nn.Module):
    """One linear site gets the Parameter, then a plain tensor copy, then the Parameter."""

    def __init__(self) -> None:
        """Build one weight."""
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Swap the weight operand between a Parameter and a plain tensor."""
        plain = self.weight.detach().clone()
        y = x
        for weight in (self.weight, plain, self.weight, plain):
            y = F.linear(y, weight)
        return y


class _InplaceAfter(nn.Module):
    """One site runs an out-of-place add, then an in-place add, then out-of-place."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alternate ``torch.add`` and ``Tensor.add_`` at one line."""
        y = x * 1
        for op in (torch.add, torch.Tensor.add_, torch.add, torch.Tensor.add_):
            y = op(y, 1)
        return y


class _NestedThenFlat(nn.Module):
    """One ``view`` site gets a shape tuple (nested), then ints (flat), then a tuple."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape through both argument forms at one line."""
        y = x
        for shape in ((16,), (4, 4), (2, 8), (4, 4)):
            y = y.view(*shape) if len(shape) > 1 else y.view(shape)
            y = y + 1
        return y


class _Unattributed(nn.Module):
    """A plain attribute tensor (neither buffer nor Parameter) enters a flat call."""

    def __init__(self) -> None:
        """Keep a plain tensor attribute."""
        super().__init__()
        self.extra = torch.randn(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the attribute tensor at the same site as an input-derived one."""
        y = _add_at_one_site(x, self.extra)
        return _add_at_one_site(y, x)


class _Recursive(nn.Module):
    """A recursive helper reenters one site at several depths."""

    def step(self, t: torch.Tensor, depth: int) -> torch.Tensor:
        """Recurse, then add at one site on the way out."""
        if depth == 0:
            return t
        return _add_at_one_site(self.step(t * 2, depth - 1), t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recurse three levels deep, twice."""
        return self.step(self.step(x, 3), 2)


_SUBJECTS: tuple[tuple[str, type[nn.Module]], ...] = (
    ("slot_kinds", _SlotKinds),
    ("kwarg_sets", _KwargSets),
    ("hash_values", _HashValues),
    ("param_swap", _ParamSwap),
    ("inplace_after", _InplaceAfter),
    ("nested_then_flat", _NestedThenFlat),
    ("unattributed", _Unattributed),
    ("recursive", _Recursive),
)


def _norm(value: Any) -> Any:
    """Return a comparable form of a recorded value (tensors by value, others by repr)."""

    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), value.detach().flatten().tolist())
    if isinstance(value, dict):
        return {key: _norm(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_norm(item) for item in value)
    return repr(value)


def _norm_component(component: Any) -> Any:
    """Return a comparable form of one template component (tensors by value)."""

    if isinstance(component, ParentRef):
        layout = component._capture_layout
        return ("parent", component.parent_label, None if layout is None else repr(layout))
    if isinstance(component, LiteralTensor):
        value = component.value
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            value.detach().flatten().tolist(),
            component.param_barcode is None,
            component._capture_param_fingerprint is None,
        )
    if isinstance(component, tuple):
        return tuple(_norm_component(item) for item in component)
    return (type(component).__name__, repr(component))


def _norm_template(template: Any) -> Any:
    """Return a comparable form of a captured argument template (or ``None``)."""

    if template is None:
        return None
    assert isinstance(template, CapturedArgTemplate)
    return (
        _norm_component(template.args),
        tuple((key, _norm_component(value)) for key, value in template.kwargs),
        template.notes,
        repr(template.func_id),
    )


def _fields(model_cls: type[nn.Module]) -> list[tuple[Any, ...]]:
    """Trace ``model_cls`` (fixed seed) and return each op's fields."""

    torch.manual_seed(0)
    model = model_cls()
    x = torch.randn(4, 4)
    trace = tl.trace(model, x)
    # A parameterized op's class holds a barcode drawn per trace; compare the
    # pattern (which ops share a class, numbered by first appearance) instead.
    classes: dict[str, int] = {}
    for op in trace.ops:
        classes.setdefault(op.graph.equivalence_class, len(classes))
    return [
        (
            index,
            list(op.parents),
            {kind: dict(op.call.parent_arg_positions[kind]) for kind in ("args", "kwargs")},
            tuple(op.call.unattributed_tensor_args),
            tuple(op.call.dropped_edge_tensor_args),
            _norm(op.call.non_tensor_pos_args),
            _norm(op.call.non_tensor_kwargs),
            _norm_template(op.call.args_template),
            _norm_template(op.call.kwargs_template),
            classes[op.graph.equivalence_class],
            [tuple(param.shape) for param in op.graph.parent_params],
            repr(op.call.config),
        )
        for index, op in enumerate(trace.ops)
    ]


def _general_call(trace: Any, func: Any, func_name: str, args: Any, kwargs: Any) -> _SlotCall:
    """Return the call with no layout plan: every walk runs its general path."""

    return _SlotCall(func, func_name, args, kwargs)


def _assert_same(name: str, left: list[tuple[Any, ...]], right: list[tuple[Any, ...]]) -> None:
    """Assert two per-op field lists are equal, naming the first differing fields."""

    assert len(left) == len(right), name
    diffs = [
        (a[0], index)
        for a, b in zip(left, right, strict=True)
        for index in range(1, len(a))
        if a[index] != b[index]
    ]
    assert diffs == [], f"{name}: fields differ (op, field index): {diffs[:6]}"


@pytest.mark.parametrize(("name", "model_cls"), _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_layout_plans_equal_general_path(
    monkeypatch: pytest.MonkeyPatch, name: str, model_cls: type[nn.Module]
) -> None:
    """Every compared field is the general path's, op for op."""

    planned = _fields(model_cls)
    with monkeypatch.context() as patch:
        patch.setattr(_ops_shared_fields, "planned_slot_call", _general_call)
        general = _fields(model_cls)
    _assert_same(name, general, planned)


def test_flat_calls_take_the_planned_witness(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated flat call is witnessed from its plan; the general witness is not run."""

    calls: list[str] = []
    real = _ops_emission._unattributed_tensor_arg_positions

    def counting(*args: Any, **kwargs: Any) -> Any:
        """Count general witness runs."""
        calls.append(args[3])
        return real(*args, **kwargs)

    monkeypatch.setattr(_ops_emission, "_unattributed_tensor_arg_positions", counting)
    tl.trace(_SlotKinds(), torch.randn(4, 4))
    assert "add" not in calls
    with monkeypatch.context() as patch:
        patch.setattr(_ops_shared_fields, "planned_slot_call", _general_call)
        tl.trace(_SlotKinds(), torch.randn(4, 4))
    assert "add" in calls


class _ClassOverride:
    """An object that reports another class through ``__class__``."""

    @property
    def __class__(self) -> type:  # type: ignore[override]
        """Claim to be a list."""
        return list


@pytest.mark.parametrize(
    ("args", "kwargs", "flat"),
    [
        ((torch.ones(1), 1), {}, True),
        ((torch.ones(1), 1.5), {"alpha": 2}, True),
        ((torch.ones(1), [torch.ones(1)]), {}, False),
        ((torch.ones(1), (1, 2)), {}, False),
        ((torch.ones(1),), {"dims": {"a": 1}}, False),
        ((torch.ones(1), torch.Size([1])), {}, False),
        ((torch.ones(1), _ClassOverride()), {}, False),
    ],
)
def test_layout_flatness(args: tuple[Any, ...], kwargs: dict[str, Any], flat: bool) -> None:
    """Containers, mappings, sizes and ``__class__`` overrides are not flat."""

    layout = _call_layout._new_layout(torch.add, args, kwargs, ((), (), ()))
    assert layout.flat is flat
    if flat:
        assert layout.tensor_slots == (("args", 0),)
        assert layout.other_args == (1,)
        assert layout.other_kwargs == tuple(kwargs)

"""The argument scan's shortcuts record what its reference paths record.

The scan keeps a flat call's replay-template plan on the call's ``CallLayout``
instead of a per-capture table keyed on the layout, decides the exact ``Tensor`` and
``Parameter`` types without the Parameter metaclass check, skips the
``AsyncCollectiveTensor`` probe for those two exact types, hands the template the
snapshot's label dict instead of rebuilding it, and resolves a parameter the capture
already validated without repeating the checks. Each subject is traced twice: once as
shipped and once with every shortcut replaced by its reference (the keyed plan table,
``isinstance`` for every type test, the probe for every tensor, a label dict built from
the slots and the full parameter checks). Parents, argument positions, the witness
pair, the templates and the parameter edges must be equal op for op, live and after a
save and load.

The subjects cover a tensor mutated and resized after its op, aliasing views written
in place, arguments that are a buffer, a parameter, an input, a closure constant, an
unattributed tensor and a plain tensor flagged ``_is_param``, nested containers,
keyword arguments and ``out=``, one callable at changing argument layouts, and chains
on either side of the column path's op threshold.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens.backends.torch import (
    _args_template_cache,
    _ops_arg_slots,
    _ops_argument_fields,
    _ops_arguments,
    _ops_predicate_events,
)
from torchlens.intervention._types import CapturedArgTemplate, LiteralTensor, ParentRef

_CLOSURE_CONSTANT = torch.linspace(-1.0, 1.0, 4)


class _MutateAfterUse(nn.Module):
    """A tensor read by an op, then written in place and resized; a strided view."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use ``h``, mutate it, resize it, then read it again."""
        h = x + 1
        y = h * 2
        h.add_(3)
        z = h.t() * 0.5
        h.resize_(2, 2)
        return y.sum() + z.sum() + h.sum()


class _AliasInplace(nn.Module):
    """Views of one base written in place, then read alongside the base."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write through a flat view and a column view."""
        base = x * 1.0
        flat = base.view(-1)
        flat.mul_(2)
        column = base[:, 0]
        column.add_(1)
        return base + flat.sum() + column.unsqueeze(1)


class _Sources(nn.Module):
    """Arguments of every provenance the scan resolves."""

    def __init__(self) -> None:
        """Build a parameter, a buffer, a plain tensor attribute and a flagged one."""
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.register_buffer("shift", torch.full((4,), 0.25))
        self.raw = torch.ones(4)
        flagged = torch.full((4,), 2.0)
        flagged._is_param = True  # type: ignore[attr-defined]
        self.flagged = flagged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix the input with a parameter, buffer, constant and unattributed tensors."""
        h = F.linear(x, self.weight)
        h = h + self.shift
        h = h * _CLOSURE_CONSTANT
        h = torch.mul(h, self.raw)
        h = torch.add(h, self.flagged)
        return torch.addcmul(h, self.weight[0], self.flagged)


class _Nested(nn.Module):
    """Tensors inside lists and tuples, including a repeated tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate, stack and broadcast nested arguments."""
        y = x * 2
        joined = torch.cat([x, y, x], dim=0)
        stacked = torch.stack((y, x))
        pair = torch.broadcast_tensors(x, y)
        return joined.sum() + stacked.sum() + pair[0].sum()


class _Keywords(nn.Module):
    """Keyword tensors, keyword scalars and an ``out=`` destination."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call through keywords and write into a pre-allocated buffer."""
        y = torch.add(x, other=x * 3, alpha=2)
        y = F.softmax(y, dim=-1)
        y = torch.clamp(y, min=0.1, max=0.9)
        buffer = torch.empty_like(y)
        torch.mul(y, x, out=buffer)
        return torch.where(buffer > 0.2, input=buffer, other=y)


class _ChangingLayouts(nn.Module):
    """One callable at several argument layouts, revisited in a loop."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alternate the argument types and keywords of ``torch.add``."""
        y = x
        for step in range(3):
            y = torch.add(y, 1)
            y = torch.add(y, x)
            y = torch.add(y, 1.5 * step)
            y = torch.add(y, x, alpha=2)
            y = torch.add(y, [x, x][step % 2].t().t())
        return y


def _chain(length: int) -> type[nn.Module]:
    """Return a module class adding one ``length`` times."""

    class _Chain(nn.Module):
        """A tensor-plus-int chain."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add one repeatedly."""
            for _ in range(length):
                x = x + 1
            return x

    _Chain.__name__ = f"_Chain{length}"
    return _Chain


_SUBJECTS: list[tuple[str, type[nn.Module], tuple[int, ...]]] = [
    ("mutate_after_use", _MutateAfterUse, (2, 3)),
    ("alias_inplace", _AliasInplace, (3, 4)),
    ("sources", _Sources, (2, 4)),
    ("nested", _Nested, (2, 3)),
    ("keywords", _Keywords, (2, 5)),
    ("changing_layouts", _ChangingLayouts, (3, 3)),
    ("chain_8_facade", _chain(8), (2, 2)),
    ("chain_64_column", _chain(64), (2, 2)),
]


class _NeverExact:
    """A type no argument has: every exact-type test against it fails."""


def _reference_labels(trace: Any, args: Any, kwargs: Any) -> dict[int, str | None] | None:
    """Return the snapshot labels by tensor id, rebuilt from the slots."""

    slots = _ops_arg_slots.current_arg_slots(trace, args, kwargs)
    return None if slots is None else {slot.value_id: slot.label for slot in slots}


@contextmanager
def _reference_paths(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace every argument-scan shortcut with its reference path."""

    keyed_plan: Callable[..., Any] = _ops_arguments.planned_args_template

    def keyed_only(parts: Any, call: Any, trace: Any, known: Any, _holder: Any = None) -> Any:
        """Look the plan up in the capture's keyed table, never on the layout."""
        return keyed_plan(parts, call, trace, known)

    keyed_lookup: Callable[..., Any] = _args_template_cache._plan_of

    def keyed_plan_only(parts: Any, call: Any, trace: Any, _holder: Any) -> Any:
        """The capture's own plan lookup, through the keyed table only."""
        return keyed_lookup(parts, call, trace, None)

    with monkeypatch.context() as patch:
        patch.setattr(_ops_arguments, "planned_args_template", keyed_only)
        patch.setattr(_ops_arguments, "arg_slot_labels", _reference_labels)
        patch.setattr(_args_template_cache, "_plan_of", keyed_plan_only)
        patch.setattr(_ops_argument_fields, "arg_slot_labels", _reference_labels)
        patch.setattr(_ops_argument_fields, "_TENSOR", _NeverExact)
        patch.setattr(_ops_argument_fields, "_PLAIN_TENSOR_TYPES", frozenset())
        patch.setattr(_ops_predicate_events, "_validated_params", lambda _trace: None)
        yield


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


def _op_fields(trace: Any) -> list[tuple[Any, ...]]:
    """Return each op's scan-derived fields in a comparable form."""

    return [
        (
            index,
            list(op.parents),
            {kind: dict(op.call.parent_arg_positions[kind]) for kind in ("args", "kwargs")},
            tuple(op.call.unattributed_tensor_args),
            tuple(op.call.dropped_edge_tensor_args),
            _norm_template(op.call.args_template),
            _norm_template(op.call.kwargs_template),
            sorted(op.graph.parent_param_ops.values()),
            [tuple(param.shape) for param in op.graph.parent_params],
        )
        for index, op in enumerate(trace.ops)
    ]


def _trace(
    model_cls: type[nn.Module], shape: tuple[int, ...], replayable: bool = False
) -> tuple[Any, list[str]]:
    """Trace a fresh ``model_cls`` on a fixed-seed input (a replayable capture if asked).

    Returns the trace and the texts of the warnings the capture raised (the closure
    constant and the unattributed tensor raise the missing-provenance warning), so a
    caller can compare them across the two paths.
    """

    torch.manual_seed(0)
    model = model_cls()
    capture = tl.options.CaptureOptions(replayable=True) if replayable else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if capture is None:
            trace = tl.trace(model, torch.randn(*shape))
        else:
            trace = tl.trace(model, torch.randn(*shape), capture=capture)
    return trace, sorted(str(item.message) for item in caught)


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


@pytest.mark.parametrize("replayable", [False, True], ids=["default", "replayable"])
@pytest.mark.parametrize(("name", "model_cls", "shape"), _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_scan_shortcuts_equal_reference_paths(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
    replayable: bool,
) -> None:
    """Every op's scan-derived fields match the reference paths, live."""

    shipped, shipped_warnings = _trace(model_cls, shape, replayable)
    with _reference_paths(monkeypatch):
        reference, reference_warnings = _trace(model_cls, shape, replayable)
    _assert_same(name, _op_fields(reference), _op_fields(shipped))
    assert shipped_warnings == reference_warnings


@pytest.mark.parametrize(
    ("name", "model_cls", "shape"),
    [_SUBJECTS[0], _SUBJECTS[2], _SUBJECTS[5], _SUBJECTS[7]],
    ids=["mutate_after_use", "sources", "changing_layouts", "chain_64_column"],
)
def test_scan_shortcuts_equal_reference_paths_after_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
) -> None:
    """A saved and loaded trace carries the same fields under either path."""

    loaded = []
    for label, reference in (("shipped", False), ("reference", True)):
        path = tmp_path / f"{label}.tlspec"
        if reference:
            with _reference_paths(monkeypatch):
                _trace(model_cls, shape, replayable=True)[0].save(path)
        else:
            _trace(model_cls, shape, replayable=True)[0].save(path)
        loaded.append(_op_fields(tl.load(path)))
    _assert_same(name, loaded[1], loaded[0])


def test_exact_tensor_parameter_test_is_its_flag() -> None:
    """For an exact ``Tensor`` the Parameter check is its ``_is_param`` flag alone.

    The scan's exact-type shortcut relies on this: an exact ``Tensor`` subclasses
    nothing, so the Parameter metaclass answers it from the flag.
    """

    class _SubParameter(nn.Parameter):
        """A Parameter subclass, which keeps the ordinary type test."""

    plain = torch.ones(3)
    flagged = torch.ones(3)
    flagged._is_param = True  # type: ignore[attr-defined]
    cleared = torch.ones(3)
    cleared._is_param = False  # type: ignore[attr-defined]
    for value in (plain, flagged, cleared):
        assert type(value) is torch.Tensor
        assert isinstance(value, nn.Parameter) is bool(getattr(value, "_is_param", False))
    assert isinstance(nn.Parameter(torch.ones(2)), nn.Parameter)
    assert isinstance(_SubParameter(torch.ones(2)), nn.Parameter)
    assert not isinstance(3, nn.Parameter)

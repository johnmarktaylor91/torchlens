"""Planned argument observation records what the general observation records.

A flat call's slot snapshot reads the facts of plain stamped tensors and prepared
Parameters through ``_arg_observation.plain_facts`` (reusing the resolve loop's
label reads), a Parameter's first sight resolves its prep address with one dict read,
the parent labels come from the snapshot's label dict and the witness is the planned
one. Each subject is traced as shipped and with every one of those replaced by its
reference: the general snapshot walk, the full parameter check, the slot-built parent
labels and the general witness on every call. The snapshot slots, parents, argument
positions, witness pair, templates and parameter edges must be equal op for op.

The subjects cover layers whose parameters are each read once, a cell reused in a
loop, buffers and running statistics, a parameter mutated in place, tied and fresh
in-forward parameters, a tensor flagged ``_is_param``, a tensor rebound to new
storage, a tensor stamped by an earlier capture, keyword tensors and ``out=``, a
repeated tensor and one callable at changing argument layouts.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens.backends.torch import (
    _arg_observation,
    _ops_arg_slots,
    _ops_argument_fields,
    _ops_arguments,
    _ops_emission,
)
from torchlens.intervention._types import CapturedArgTemplate, LiteralTensor, ParentRef


class _Varied(nn.Module):
    """Distinct layers: every parameter is read once per capture."""

    def __init__(self) -> None:
        """Build six differently sized linear layers."""
        super().__init__()
        sizes = (6, 7, 5, 8, 4, 6, 3)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layers with a nonlinearity between them."""
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class _LoopCell(nn.Module):
    """One cell applied in a loop: its parameters are read again and again."""

    def __init__(self) -> None:
        """Build the cell's two linear maps."""
        super().__init__()
        self.inp = nn.Linear(5, 5)
        self.rec = nn.Linear(5, 5, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unroll the cell over four steps."""
        h = torch.zeros_like(x)
        for _ in range(4):
            h = torch.tanh(self.inp(x) + self.rec(h))
        return h


class _Buffers(nn.Module):
    """A registered buffer read by flat calls and a batch norm's running statistics."""

    def __init__(self) -> None:
        """Build a buffer and a batch norm."""
        super().__init__()
        self.register_buffer("shift", torch.full((4,), 0.5))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shift, normalize and scale by the buffer again."""
        h = x + self.shift
        h = self.norm(h)
        return h * self.shift


class _MutatedParam(nn.Module):
    """A parameter written in place mid-forward, then read."""

    def __init__(self) -> None:
        """Build the parameter."""
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read, mutate, read again."""
        y = x @ self.weight
        with torch.no_grad():
            self.weight.mul_(0.5)
        return y + x @ self.weight + self.weight.sum()


class _TiedAndFresh(nn.Module):
    """A weight tied across two layers and a Parameter made inside the forward."""

    def __init__(self) -> None:
        """Tie the second layer's weight to the first."""
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)
        self.second.weight = self.first.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the tied layers and a fresh Parameter."""
        fresh = nn.Parameter(torch.ones(4))
        return self.second(self.first(x)) * fresh


class _Unusual(nn.Module):
    """A flagged tensor, a rebound tensor and a closure constant."""

    def __init__(self) -> None:
        """Build the flagged tensor."""
        super().__init__()
        flagged = torch.full((4,), 2.0)
        flagged._is_param = True  # type: ignore[attr-defined]
        self.flagged = flagged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix in the flagged tensor, then rebind a labeled tensor's storage."""
        h = torch.add(x, self.flagged)
        g = h * 2
        g.data = torch.ones_like(g)
        return torch.mul(g, h) + torch.add(h, 1)


class _Remembers(nn.Module):
    """Keeps a tensor from one capture and reads it in the next."""

    def __init__(self) -> None:
        """Start with nothing remembered."""
        super().__init__()
        self.kept: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read the previous capture's tensor, then remember this one's."""
        h = x * 3
        out = h + (self.kept if self.kept is not None else 0)
        self.kept = h
        return out


class _Keywords(nn.Module):
    """Keyword tensors, an ``out=`` destination and a repeated tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call through keywords, write into a buffer, square a tensor."""
        y = torch.add(x, other=x * 3, alpha=2)
        y = F.softmax(y, dim=-1)
        buffer = torch.empty_like(y)
        torch.mul(y, x, out=buffer)
        return torch.add(buffer, buffer) + y * y


class _ChangingLayouts(nn.Module):
    """One callable at several argument layouts, revisited in a loop."""

    def __init__(self) -> None:
        """Build a parameter to swap in for a plain tensor."""
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alternate the argument types and keywords of ``torch.add``."""
        y = x
        for step in range(3):
            y = torch.add(y, 1)
            y = torch.add(y, x)
            y = torch.add(y, self.weight)
            y = torch.add(y, 1.5 * step, alpha=2)
            y = torch.add(self.weight, y, alpha=0.5)
        return y


_SUBJECTS: list[tuple[str, type[nn.Module], tuple[int, ...]]] = [
    ("varied", _Varied, (2, 6)),
    ("loop_cell", _LoopCell, (2, 5)),
    ("buffers", _Buffers, (3, 4)),
    ("mutated_param", _MutatedParam, (2, 4)),
    ("tied_and_fresh", _TiedAndFresh, (2, 4)),
    ("unusual", _Unusual, (2, 4)),
    ("keywords", _Keywords, (2, 5)),
    ("changing_layouts", _ChangingLayouts, (3, 3)),
]


@contextmanager
def _reference(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace every planned observation step with its reference."""

    with monkeypatch.context() as patch:
        patch.setattr(_ops_argument_fields, "snapshot_flat_slots", lambda *_args: None)
        patch.setattr(
            _arg_observation, "validated_parameter", _ops_arguments._session_validated_parameter
        )
        patch.setattr(_ops_argument_fields, "slot_parent_labels", _ops_arg_slots.parents_from_slots)
        patch.setattr(_ops_emission, "_planned_witness", lambda *_args: None)
        yield


@contextmanager
def _recorded_slots(monkeypatch: pytest.MonkeyPatch, sink: list[Any]) -> Iterator[None]:
    """Record every op's snapshot slots, tensors named by first appearance."""

    real = _ops_argument_fields.snapshot_arg_slots

    def recording(*args: Any) -> Any:
        """Take the snapshot and record a comparable copy of its slots."""
        slots = real(*args)
        order: dict[int, int] = {}
        sink.append(
            [
                (
                    slot.kind,
                    slot.key,
                    order.setdefault(slot.value_id, len(order)),
                    slot.label,
                    slot.known_provenance,
                    slot.session_parent_labels,
                    slot.data_alias,
                    slot.param_unmutated,
                    slot.param_validated,
                )
                for slot in slots
            ]
        )
        return slots

    with monkeypatch.context() as patch:
        patch.setattr(_ops_argument_fields, "snapshot_arg_slots", recording)
        yield


def _norm(component: Any) -> Any:
    """Return a comparable form of a template or one of its components (tensors by value)."""

    if component is None:
        return None
    if isinstance(component, CapturedArgTemplate):
        kwargs = tuple((key, _norm(value)) for key, value in component.kwargs)
        return (_norm(component.args), kwargs, component.notes, repr(component.func_id))
    if isinstance(component, ParentRef):
        layout = component._capture_layout
        return ("parent", component.parent_label, None if layout is None else repr(layout))
    if isinstance(component, LiteralTensor):
        value = component.value
        return ("tensor", tuple(value.shape), value.detach().flatten().tolist())
    if isinstance(component, tuple):
        return tuple(_norm(item) for item in component)
    return (type(component).__name__, repr(component))


def _op_fields(trace: Any) -> list[tuple[Any, ...]]:
    """Return each op's observation-derived fields in a comparable form."""

    return [
        (
            index,
            list(op.parents),
            {kind: dict(op.call.parent_arg_positions[kind]) for kind in ("args", "kwargs")},
            tuple(op.call.unattributed_tensor_args),
            tuple(op.call.dropped_edge_tensor_args),
            _norm(op.call.args_template),
            sorted(op.graph.parent_param_ops.values()),
            [tuple(param.shape) for param in op.graph.parent_params],
        )
        for index, op in enumerate(trace.ops)
    ]


def _observe(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
    replayable: bool,
) -> tuple[list[Any], list[Any], list[str]]:
    """Trace a fresh model twice (the second capture is compared) on fixed inputs."""

    torch.manual_seed(0)
    model = model_cls()
    capture = tl.options.CaptureOptions(replayable=True) if replayable else None
    slots: list[Any] = []
    options: dict[str, Any] = {} if capture is None else {"capture": capture}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, torch.randn(*shape), **options)
        with _recorded_slots(monkeypatch, slots):
            trace = tl.trace(model, torch.randn(*shape), **options)
    return slots, _op_fields(trace), sorted(str(item.message) for item in caught)


@pytest.mark.parametrize("replayable", [False, True], ids=["default", "replayable"])
@pytest.mark.parametrize(("name", "model_cls", "shape"), _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_planned_observation_equals_reference(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
    replayable: bool,
) -> None:
    """Every op's slots and observation-derived fields match the reference, live."""

    shipped = _observe(monkeypatch, model_cls, shape, replayable)
    with _reference(monkeypatch):
        reference = _observe(monkeypatch, model_cls, shape, replayable)
    assert len(shipped[0]) == len(reference[0]), name
    slot_diffs = [i for i, (a, b) in enumerate(zip(shipped[0], reference[0])) if a != b]
    assert slot_diffs == [], f"{name}: snapshot slots differ at ops {slot_diffs[:6]}"
    field_diffs = [a[0] for a, b in zip(shipped[1], reference[1], strict=True) if a != b]
    assert len(shipped[1]) == len(reference[1]) and field_diffs == [], f"{name}: {field_diffs[:6]}"
    assert shipped[2] == reference[2]


def test_planned_parameter_check_equals_full_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cheaper first-sight parameter check answers as the full check, every call."""

    real = _arg_observation.validated_parameter
    full = _ops_arguments._session_validated_parameter
    answers: list[tuple[bool, bool]] = []

    def both(trace: Any, value: torch.Tensor) -> bool:
        """Answer with the planned check, then the full one from a cold cache."""
        planned = real(trace, value)
        trace._raw_graph_ws._validated_params.pop(id(value), None)
        answers.append((planned, full(trace, value)))
        return planned

    monkeypatch.setattr(_arg_observation, "validated_parameter", both)
    for model_cls, shape in ((_Varied, (2, 6)), (_TiedAndFresh, (2, 4)), (_LoopCell, (2, 5))):
        torch.manual_seed(0)
        tl.trace(model_cls(), torch.randn(*shape))
    assert answers, "no parameter was checked"
    assert all(planned == reference for planned, reference in answers)
    assert any(planned for planned, _ in answers)


def test_flat_calls_take_the_planned_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flat call's plain tensors and Parameters get planned facts; others the general read."""

    left: list[bool] = []
    real = _arg_observation.plain_facts

    def counting(*args: Any) -> Any:
        """Record whether each tensor is left to the general read."""
        facts = real(*args)
        left.append(facts is None)
        return facts

    monkeypatch.setattr(_arg_observation, "plain_facts", counting)
    torch.manual_seed(0)
    tl.trace(_Varied(), torch.randn(2, 6))
    assert left and not any(left)
    left.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the flagged tensor has no provenance
        tl.trace(_Unusual(), torch.randn(2, 4))
    assert any(left)


def test_record_builders_equal_the_dataclass_constructors() -> None:
    """The one-step builders make the records the generated constructors make, frozen."""

    facts = ("op_1", True, ("op_1", "buffer_2"), False, False, False)
    slot = _ops_arg_slots.new_arg_slot("kwargs", "mask", 7, facts)
    assert slot == _ops_arg_slots._ArgSlot("kwargs", "mask", 7, *facts)
    assert dataclasses.replace(slot, label=None).label is None
    context = _ops_arg_slots.new_slot_context((1,), {}, (slot,), {("kwargs", "mask"): slot}, {})
    assert context == _ops_arg_slots._SlotContext((1,), {}, (slot,), {("kwargs", "mask"): slot}, {})
    for record, field in ((slot, "label"), (context, "labels")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, field, None)

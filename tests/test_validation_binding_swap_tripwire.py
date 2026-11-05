"""Validation fails a call whose arguments are bound to the wrong parents.

Capture stores a per-child version snapshot of a parent whenever the value in the slot
the parent is bound to differs from the parent's saved out, and replay feeds the child
from that snapshot. A wrong binding therefore records snapshots that carry the swapped
content, and argument logging and replay both agreed with it. Each test plants a wrong
binding after capture exactly as capture would record it (the positions move, and each
moved parent's snapshot is the content of its new slot when that differs from its saved
out), and validation must fail; the unplanted control of each model validates.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils.tensor_utils import tensor_nanequal
from torchlens.validation._core import _op_for_validation_label


class _Binary(nn.Module):
    """``a - b`` over two same-shape parents."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract two products of the input."""

        return torch.sub(x * 2.0, x + 1.0)


class _BaseAndView(nn.Module):
    """One call reading a base and a column view of it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a column view of ``y`` back onto ``y``."""

        y = x * 2.0
        return torch.add(y, y[:, 1:2])


class _WriteBack(nn.Module):
    """``y[:, 1:3] += 1``: the write-back call reads the base and the written view."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one into a slice of ``y`` through augmented assignment."""

        y = x * 2.0
        y[:, 1:3] += 1.0
        return y * 3.0


class _Cat(nn.Module):
    """``cat([a, b, c])`` of three same-shape parents."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate three functions of the input."""

        return torch.cat([x * 2.0, x + 1.0, x - 3.0])


class _AliasesAfterViewWrite(nn.Module):
    """A base and a full view of it, both read after a write through another view."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into ``y[:, 0:4]``, then add ``y`` and ``y.view(y.shape)``."""

        y = x * 2.0
        z = y.view(y.shape)
        a = y[:, 0:4]
        a.add_(1.0)
        return torch.add(y, z)


class _TwoInplaceDifferentShapes(nn.Module):
    """Two in-place writes of different shapes, read by one broadcasting call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write ``p`` (3 x 8) and ``q`` (8) in place, then subtract them."""

        p = x * 2.0
        q = x[0] * 3.0
        p.add_(1.0)
        q.mul_(2.0)
        return torch.sub(p, q)


class _SiblingViewAfterWrite(nn.Module):
    """A sibling view of another shape, taken before a write through a view, read after it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into ``y[:, 3:5]``, then subtract a fresh tensor from the sibling ``y[:, 2:6]``."""

        y = x * 2.0
        sibling = y[:, 2:6]
        other = x[:, 0:4] + 1.0
        y[:, 3:5].mul_(3.0)
        return torch.sub(sibling, other)


class _SiblingViewReturnedAfterWrite(nn.Module):
    """Two views of one tensor returned after a write through the first (a planner's head)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale ``y[:, :2]`` in place, return it, its sibling ``y[:, 2:3]`` and a fresh tensor."""

        y = x * 2.0
        first, sibling = y[:, :2], y[:, 2:3]
        first *= 3.0
        return first, sibling, torch.add(x[:, 3:4], 1.0)


def _trace(model: nn.Module, x: torch.Tensor) -> Any:
    """Capture ``model`` with saved arguments, as validation does."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=True, save_rng=True))


def _op(trace: Any, func_name: str) -> Any:
    """Return the one op with ``func_name``."""

    (op,) = [op for op in trace._layer_list if op.func_name == func_name]
    return op


def _slot_value(op: Any, key: Any) -> Any:
    """Return the saved positional argument at a parent-position key."""

    if isinstance(key, tuple):
        return op.call.saved_args[key[0]][key[1]]
    return op.call.saved_args[key]


def _bind(trace: Any, child: Any, bindings: dict[Any, str]) -> None:
    """Rebind ``child``'s positional slots to ``bindings`` the way capture records a binding."""

    positions = child.call.parent_arg_positions["args"]
    before = {_op_for_validation_label(trace, label) for label in positions.values()}
    positions.update(bindings)
    child.parents = type(child.parents)(dict.fromkeys(positions.values()))
    after = {_op_for_validation_label(trace, label) for label in positions.values()}
    for parent in before | after:
        versions = parent.out_versions_by_child
        versions.pop(child.label, None)
        versions.pop(child.layer_label, None)
        if parent not in after:
            parent.children = type(parent.children)(
                c for c in parent.children if c != child.layer_label
            )
            continue
        if child.layer_label not in parent.children:
            parent.children = type(parent.children)([*parent.children, child.layer_label])
        key = next(
            k for k, label in positions.items() if _op_for_validation_label(trace, label) is parent
        )
        value = _slot_value(child, key)
        if not (value.shape == parent.out.shape and tensor_nanequal(value, parent.out)):
            versions[child.layer_label] = value.detach().clone()


def _swap(func_name: str, first: Any, second: Any) -> Callable[[Any], None]:
    """Return a plant that swaps the parents bound at two slots of ``func_name``."""

    def plant(trace: Any) -> None:
        """Swap the two slots' parents."""

        child = _op(trace, func_name)
        positions = child.call.parent_arg_positions["args"]
        _bind(trace, child, {first: positions[second], second: positions[first]})

    return plant


def _bind_past_the_write(trace: Any) -> None:
    """Bind ``add(y, z)`` to the pre-write producers, dropping the in-place op's consumer edge."""

    child = _op(trace, "add")
    _bind(trace, child, {0: _op(trace, "__mul__").layer_label, 1: _op(trace, "view").layer_label})


def _output_node(trace: Any, parent_func_name: str) -> Any:
    """Return the one output node whose parent is the op named ``parent_func_name``."""

    (node,) = [
        op
        for op in trace._layer_list
        if op.is_output
        and _op_for_validation_label(trace, op.parents[0]).func_name == parent_func_name
    ]
    return node


def _bind_fresh_output_to_the_write(trace: Any) -> None:
    """Bind the fresh ``(3, 1)`` output to the in-place write, as the returned sibling is."""

    _bind(trace, _output_node(trace, "add"), {0: _op(trace, "__imul__").layer_label})


_PLANTS: dict[str, tuple[type[nn.Module], Callable[[Any], None]]] = {
    "binary_swapped": (_Binary, _swap("sub", 0, 1)),
    "base_and_view_swapped": (_BaseAndView, _swap("add", 0, 1)),
    "write_back_base_and_view_swapped": (_WriteBack, _swap("__setitem__", 0, 2)),
    "cat_reordered": (_Cat, _swap("cat", (0, 0), (0, 2))),
    "consumer_bound_past_the_write": (_AliasesAfterViewWrite, _bind_past_the_write),
    "inplace_parents_of_different_shapes_swapped": (
        _TwoInplaceDifferentShapes,
        _swap("sub", 0, 1),
    ),
    "sibling_alias_and_fresh_operand_swapped": (_SiblingViewAfterWrite, _swap("sub", 0, 1)),
    "fresh_output_bound_to_the_write": (
        _SiblingViewReturnedAfterWrite,
        _bind_fresh_output_to_the_write,
    ),
}


def _validate(model: nn.Module, plant: Callable[[Any], None] | None) -> bool:
    """Capture, optionally plant a wrong binding, and run forward replay validation."""

    torch.manual_seed(0)
    x = torch.randn(3, 8)
    trace = _trace(model, x)
    if plant is not None:
        plant(trace)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ground_truth = model(x.clone())
        return bool(trace.validate_forward_pass(ground_truth, validate_metadata=False))


@pytest.mark.parametrize("case", sorted(_PLANTS))
def test_unplanted_control_validates(case: str) -> None:
    """Each model validates as captured."""

    model_class, _ = _PLANTS[case]
    assert _validate(model_class(), None)


@pytest.mark.parametrize("case", sorted(_PLANTS))
def test_planted_wrong_binding_fails_validation(case: str) -> None:
    """A wrong binding recorded the way capture records it fails forward validation."""

    model_class, plant = _PLANTS[case]
    assert not _validate(model_class(), plant)


def test_snapshot_that_matches_its_parent_is_not_questioned() -> None:
    """A snapshot equal to its parent's saved out needs no explanation (contract snapshots)."""

    from torchlens.validation._core import _saved_out_payload
    from torchlens.validation._snapshot_binding import snapshot_binding_explained

    trace = _trace(_Binary(), torch.randn(3, 8))
    child = _op(trace, "sub")
    parent = _op_for_validation_label(trace, child.call.parent_arg_positions["args"][0])
    parent.out_versions_by_child[child.layer_label] = parent.out.detach().clone()
    assert snapshot_binding_explained(
        child, parent, _saved_out_payload, lambda label: _op_for_validation_label(trace, label)
    )


def test_known_blind_spot_two_same_shape_inplace_parents_swapped() -> None:
    """KNOWN BLIND SPOT: the snapshot check accepts two same-shape in-place parents swapped.

    A snapshot of an in-place parent may legitimately differ from its saved out (a reader
    of the base, or of another alias of the view it wrote through, is bound to the write),
    and the record keeps no storage geometry to tell that alias apart from a swapped
    operand of the same shape, so the check explains both swapped snapshots. This pins
    the gap so a stricter check shows up here.
    """

    from torchlens.validation._core import _saved_out_payload
    from torchlens.validation._snapshot_binding import snapshot_binding_explained

    class _TwoBaseWrites(nn.Module):
        """Two bases, each written in place, read by one call."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Write ``p`` and ``q`` in place, then subtract them."""

            p = x * 2.0
            q = x * 3.0
            p.add_(1.0)
            q.mul_(2.0)
            return torch.sub(p, q)

    trace = _trace(_TwoBaseWrites(), torch.randn(3, 8))
    _swap("sub", 0, 1)(trace)
    child = _op(trace, "sub")
    parents = [_op_for_validation_label(trace, label) for label in child.parents]
    assert all(parent.is_inplace for parent in parents)
    assert all(
        snapshot_binding_explained(
            child, parent, _saved_out_payload, lambda label: _op_for_validation_label(trace, label)
        )
        for parent in parents
    )


class _SquareAliasesAfterWrite(nn.Module):
    """A square base and its transpose, both read after a write through a row slice."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into ``y[0:2]``, then add ``y`` and ``y.t()``."""

        y = x * 2.0
        v = y.t()
        a = y[0:2]
        a.add_(1.0)
        return torch.add(y, v)


def _inplace_slot_parent(trace: Any) -> tuple[Any, Any]:
    """Return ``add`` and the in-place parent bound to its first slot."""

    child = _op(trace, "add")
    parent = _op_for_validation_label(trace, child.call.parent_arg_positions["args"][0])
    assert parent.is_inplace
    return child, parent


def test_inplace_parent_snapshot_off_the_write_chain_is_not_explained() -> None:
    """An in-place parent's snapshot must have the shape of an op on the write's receiver chain."""

    from torchlens.validation._core import _saved_out_payload
    from torchlens.validation._snapshot_binding import snapshot_binding_explained

    trace = _trace(_SquareAliasesAfterWrite(), torch.randn(4, 4))
    child, parent = _inplace_slot_parent(trace)
    parent.out_versions_by_child[child.layer_label] = torch.randn(2, 8)
    assert not snapshot_binding_explained(
        child, parent, _saved_out_payload, lambda label: _op_for_validation_label(trace, label)
    )
    parent.out_versions_by_child[child.layer_label] = torch.randn(4, 4).double()
    assert not snapshot_binding_explained(
        child, parent, _saved_out_payload, lambda label: _op_for_validation_label(trace, label)
    )


def test_known_blind_spot_base_and_same_shape_view_swapped() -> None:
    """KNOWN BLIND SPOT: an in-place parent's snapshot of a same-shape alias is not told apart.

    ``add(y, y.t())`` after a write through ``y[0:2]`` binds both slots to the write. A
    swapped binding records the transpose's content as the base slot's snapshot; it has
    the base's shape and dtype, the check compares no content, and it is explained. This
    pins the gap (the base/view same-shape swap) so a stricter check shows up here.
    """

    from torchlens.validation._core import _saved_out_payload
    from torchlens.validation._snapshot_binding import snapshot_binding_explained

    trace = _trace(_SquareAliasesAfterWrite(), torch.randn(4, 4))
    child, parent = _inplace_slot_parent(trace)
    base = _op_for_validation_label(trace, child.call.parent_arg_positions["args"][0])
    parent.out_versions_by_child[child.layer_label] = _slot_value(child, 0).detach().t().clone()
    assert base is parent
    assert snapshot_binding_explained(
        child, parent, _saved_out_payload, lambda label: _op_for_validation_label(trace, label)
    )


def test_sibling_alias_snapshot_is_explained_only_by_its_capture_mark() -> None:
    """A write's reader of a sibling view needs capture's alias mark for exactly its slots."""

    from torchlens.validation._core import _saved_out_payload
    from torchlens.validation._snapshot_binding import snapshot_binding_explained

    trace = _trace(_SiblingViewAfterWrite(), torch.randn(3, 8))
    child = _op(trace, "sub")
    parent = _op_for_validation_label(trace, child.call.parent_arg_positions["args"][0])
    assert parent.is_inplace

    def explained(marks: Any) -> bool:
        """Run the tripwire on the sibling slot with the capture ``marks``."""

        return snapshot_binding_explained(
            child,
            parent,
            _saved_out_payload,
            lambda label: _op_for_validation_label(trace, label),
            marks,
        )

    journal = trace.__dict__["_foreign_departures"]
    moved = {key: frozenset({("args", 1)}) for key in journal.alias_slots}
    assert explained(journal)
    assert not explained(None)
    assert not explained(dataclasses.replace(journal, alias_slots={}))
    assert not explained(dataclasses.replace(journal, alias_slots=moved))
    versions = parent.out_versions_by_child
    key = next(k for k in (child.label, child.layer_label) if k in versions)
    versions[key] = versions[key].double()
    assert not explained(journal)


def test_returned_sibling_alias_is_explained_only_by_its_capture_mark() -> None:
    """A returned sibling view bound to the write needs capture's mark for its output node."""

    from torchlens.validation._core import _saved_out_payload
    from torchlens.validation._snapshot_binding import snapshot_binding_explained

    trace = _trace(_SiblingViewReturnedAfterWrite(), torch.randn(3, 8))
    parent = _op(trace, "__imul__")
    (child,) = [
        op
        for op in trace._layer_list
        if op.is_output and op.parents[0] == parent.layer_label and tuple(op.shape) == (3, 1)
    ]

    def explained(marks: Any) -> bool:
        """Run the tripwire on the returned sibling with the capture ``marks``."""

        return snapshot_binding_explained(
            child,
            parent,
            _saved_out_payload,
            lambda label: _op_for_validation_label(trace, label),
            marks,
        )

    journal = trace.__dict__["_foreign_departures"]
    key = (parent._label_raw, child._label_raw)
    assert journal.alias_slots[key] == frozenset({("args", 0)})
    assert explained(journal)
    assert not explained(None)
    others = {k: v for k, v in journal.alias_slots.items() if k != key}
    assert not explained(dataclasses.replace(journal, alias_slots=others))
    assert not explained(
        dataclasses.replace(journal, alias_slots={**others, key: frozenset({("args", 1)})})
    )
    versions = parent.out_versions_by_child
    label = next(k for k in (child.label, child.layer_label) if k in versions)
    versions[label] = versions[label].double()
    assert not explained(journal)

"""The faster label-rename and template walks return what the plain ladders returned.

Reference implementations below are the ``isinstance`` ladders that
``torchlens/_postprocess/_label_rename.py`` and
``torchlens/_data_classes/_template_walk.py`` replaced; every result must
match them in value, exact node type and order.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import pytest
import torch

from torchlens._data_classes._template_walk import map_template_leaves, template_has_leaf
from torchlens._postprocess._label_rename import (
    _rename_label_dataclass,
    _rename_template_parent_refs,
)
from torchlens.intervention._types import (
    CapturedArgTemplate,
    LiteralValue,
    ParentRef,
    Unsupported,
)


def _parent_refs(value: Any) -> list[ParentRef]:
    """Return every ``ParentRef`` leaf of ``value`` in walk order."""
    found: list[ParentRef] = []

    def collect(leaf: Any) -> Any:
        """Record a ``ParentRef`` leaf and keep every node unchanged."""
        if isinstance(leaf, ParentRef):
            found.append(leaf)
        return leaf

    map_template_leaves(value, collect)
    return found


# Reference implementations: the plain isinstance ladders the fast walks replaced.


def _ref_rename_template(value: Any, mapping: dict[str, str]) -> Any:
    """Reference template rename: the plain ``isinstance`` ladder."""
    if isinstance(value, ParentRef):
        return replace(value, parent_label=mapping.get(value.parent_label, value.parent_label))
    if isinstance(value, tuple):
        return tuple(_ref_rename_template(item, mapping) for item in value)
    if isinstance(value, list):
        return [_ref_rename_template(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _ref_rename_template(item, mapping) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _ref_rename_template(getattr(value, field.name), mapping)
            for field in fields(value)
            if hasattr(value, field.name)
        }
        return replace(value, **updates)
    return value


_LABELS = {"parent_label", "child_label", "target_label", "call_label", "site_label", "layer_label"}


def _ref_rename_record(value: Any, mapping: dict[str, str]) -> Any:
    """Reference record rename: the plain ``isinstance`` ladder."""
    if callable(value):
        return value
    if isinstance(value, tuple):
        return tuple(_ref_rename_record(item, mapping) for item in value)
    if isinstance(value, list):
        return [_ref_rename_record(item, mapping) for item in value]
    if isinstance(value, dict):
        return {
            _ref_rename_record(key, mapping): _ref_rename_record(item, mapping)
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        updates = {}
        for field in fields(value):
            if not hasattr(value, field.name):
                continue
            field_value = getattr(value, field.name)
            if field.name in _LABELS and isinstance(field_value, str):
                updates[field.name] = mapping.get(field_value, field_value)
            else:
                updates[field.name] = _ref_rename_record(field_value, mapping)
        return replace(value, **updates)
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


@dataclass(frozen=True)
class _SubRef(ParentRef):
    """A ``ParentRef`` subclass (renamed through ``replace``, not the constructor)."""

    extra: int = 0


@dataclass(frozen=True)
class _Record:
    """A label-bearing frozen record."""

    parent_label: str
    payload: Any = None
    note: str = "raw_1"


class _Label(str):
    """A ``str`` subclass (renamed like a plain string by the record walk)."""


_Pair = namedtuple("_Pair", "left right")
_MAP = {"raw_1": "fin_1", "raw_2": "fin_2"}


def _corpus() -> list[Any]:
    """Return templates and records covering every branch of both walks."""
    layout = object()
    ref = ParentRef("raw_1", layout)
    return [
        CapturedArgTemplate(
            args=(ref, LiteralValue(_Record("raw_2")), (ParentRef("missing"), 3.0)),
            kwargs=(("k", ParentRef("raw_2")),),
            notes=("raw_1",),
        ),
        _Pair(ref, [ParentRef("raw_2"), {"d": _SubRef("raw_1", extra=2)}]),
        {"a": [None, True, 1, b"x", 2j, "raw_1", _Label("raw_2")]},
        _Record("raw_1", payload=(_Record("raw_2"), len, _Pair("raw_1", "x"))),
        torch.zeros(1),
        _Record,
    ]


def _same(left: Any, right: Any) -> bool:
    """Return whether two values are equal with identical types at every node."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left is right
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(map(_same, left, right))
    if isinstance(left, dict):
        return list(left) == list(right) and all(_same(left[k], right[k]) for k in left)
    if is_dataclass(left) and not isinstance(left, type):
        return all(
            _same(getattr(left, f.name), getattr(right, f.name))
            and getattr(left, f.name).__class__ is getattr(right, f.name).__class__
            for f in fields(left)
        )
    return left == right and (not callable(left) or left is right)


@pytest.mark.parametrize("index", range(len(_corpus())))
def test_template_rename_matches_reference(index: int) -> None:
    """Fast template rename returns the reference result, layouts kept by identity."""

    value = _corpus()[index]
    got = _rename_template_parent_refs(value, _MAP, ParentRef)
    assert _same(got, _ref_rename_template(value, _MAP))
    for new, old in zip(_parent_refs(got), _parent_refs(value)):
        assert new._capture_layout is old._capture_layout


@pytest.mark.parametrize("index", range(len(_corpus())))
def test_record_rename_matches_reference(index: int) -> None:
    """Fast record rename returns the reference result."""

    value = _corpus()[index]
    assert _same(_rename_label_dataclass(value, _MAP), _ref_rename_record(value, _MAP))


@pytest.mark.parametrize("index", range(len(_corpus())))
def test_template_walks_match_reference(index: int) -> None:
    """``template_has_leaf`` and ``map_template_leaves`` agree with the reference ladder."""

    value = _corpus()[index]
    stale = Unsupported(reason="removed_parent_ref", value_type="ParentRef")

    def is_hit(leaf: Any) -> bool:
        """Return whether ``leaf`` is a ``ParentRef`` to ``raw_2``."""
        return isinstance(leaf, ParentRef) and leaf.parent_label == "raw_2"

    def swap(leaf: Any) -> Any:
        """Replace a hit with the stale marker; keep anything else."""
        return stale if is_hit(leaf) else leaf

    def ref_has(node: Any) -> bool:
        """Reference ``template_has_leaf``: the plain ladder."""
        if is_hit(node):
            return True
        if isinstance(node, (tuple, list)):
            return any(ref_has(item) for item in node)
        if isinstance(node, dict):
            return any(ref_has(item) for item in node.values())
        if is_dataclass(node) and not isinstance(node, type):
            return any(ref_has(getattr(node, f.name)) for f in fields(node))
        return False

    def ref_map(node: Any) -> Any:
        """Reference ``map_template_leaves``: the plain ladder."""
        if swap(node) is not node:
            return stale
        if isinstance(node, tuple):
            return tuple(ref_map(item) for item in node)
        if isinstance(node, list):
            return [ref_map(item) for item in node]
        if isinstance(node, dict):
            return {key: ref_map(item) for key, item in node.items()}
        if is_dataclass(node) and not isinstance(node, type):
            return replace(node, **{f.name: ref_map(getattr(node, f.name)) for f in fields(node)})
        return node

    assert template_has_leaf(value, is_hit) == ref_has(value)
    assert _same(map_template_leaves(value, swap), ref_map(value))

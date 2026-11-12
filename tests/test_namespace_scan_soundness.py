"""Soundness of the module-namespace slot scan in the session clear and snapshot walks.

The scan decides its inert-type and class tests once per distinct value type and
applies them in C. These tests pin that the slots it visits are exactly the ones
the per-item filter visits, in the same order, and that a stamped tensor in a
module namespace is still cleared when it replaces a value at constant length,
is deleted before session end, is reached across repeated captures, or was
stashed by a capture that raised.
"""

from __future__ import annotations

import collections
import enum
import math
import types
from collections import deque
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils._tensor_label import TorchLensMeta
from torchlens.backends.torch import _session_cleanup as sc

_CONTAINERS = (dict, list, tuple, set, frozenset, deque)


def _reference_visits(module: types.ModuleType) -> list[Any]:
    """Visit ``module``'s namespace slots with the per-item filter, as the scan did before."""

    visited: list[Any] = []
    seen = {id(module)}
    namespace = module.__dict__
    slot_names = tuple(
        name
        for name, item in namespace.items()
        if type(item) not in sc._NAMESPACE_INERT_TYPES
        and not issubclass(type(item), type)
        and (
            type(item).__name__ != "_reduce_op"
            or not sc._is_isinstance_hostile_deprecation_shim(item)
        )
        and isinstance(item, (torch.Tensor, *_CONTAINERS))
    )
    for name in slot_names:
        item = namespace.get(name)
        if isinstance(item, torch.Tensor) and not isinstance(item, nn.Parameter):
            visited.append(item)
        elif isinstance(item, _CONTAINERS):
            sc._clear_container_tree_tensor_metadata(item, seen, 1, visited.append)
    return visited


def _scan_visits(module: types.ModuleType) -> list[Any]:
    """Visit ``module``'s namespace slots with the scan under test."""

    visited: list[Any] = []
    sc._clear_module_namespace_tensor_metadata(module, set(), 0, visited.append, visited.append)
    return visited


class _Color(enum.Enum):
    RED = 1


class _TensorList(list):
    """A list subclass: a container slot by ``isinstance``, not by exact type."""


def _mixed_module() -> types.ModuleType:
    """Return a module whose namespace holds every kind of slot the scan decides."""

    module = types.ModuleType("tl_test_mixed_namespace")
    inner = types.ModuleType("tl_test_inner_namespace")
    inner.hidden = torch.ones(1)
    holder = types.SimpleNamespace(hidden=torch.ones(1))
    values: dict[str, Any] = {
        "plain": torch.zeros(2),
        "param": nn.Parameter(torch.zeros(2)),
        "in_list": [torch.ones(2), 3],
        "in_dict": {"k": torch.ones(3), "n": None},
        "in_tuple": (torch.ones(4),),
        "empty_set": frozenset(),
        "in_deque": deque([torch.ones(5)]),
        "nested": [{"deep": (torch.ones(6),)}],
        "subclassed": _TensorList([torch.ones(7)]),
        "number": 5,
        "text": "x",
        "function": _mixed_module,
        "klass": _TensorList,
        "enum_class": _Color,
        "enum_member": _Color.RED,
        "inner_module": inner,
        "holder": holder,
        "late": torch.full((2,), 9.0),
    }
    for name, value in values.items():
        setattr(module, name, value)
    return module


@pytest.mark.parametrize(
    "module",
    [torch, torch.nn.functional, math, collections, _mixed_module()],
    ids=["torch", "functional", "math", "collections", "mixed"],
)
def test_scan_visits_exactly_the_per_item_filter_slots(module: types.ModuleType) -> None:
    """The scan visits the same objects in the same order as the per-item filter."""

    expected = _reference_visits(module)
    actual = _scan_visits(module)
    assert [id(item) for item in actual] == [id(item) for item in expected]


def test_mixed_namespace_visits_tensors_and_container_members_only() -> None:
    """Parameters, nested modules and plain objects stay out of the namespace scan."""

    module = _mixed_module()
    visited = {id(item) for item in _scan_visits(module)}
    assert id(module.plain) in visited and id(module.late) in visited
    assert id(module.in_list[0]) in visited
    assert id(module.nested[0]["deep"][0]) in visited
    assert id(module.subclassed[0]) in visited
    assert id(module.param) not in visited
    assert id(module.inner_module.hidden) not in visited
    assert id(module.holder.hidden) not in visited


@pytest.mark.parametrize(
    "replacement_kind",
    ["tensor", "list_of_tensor", "other_tensor"],
)
def test_constant_length_replacement_is_visited(replacement_kind: str) -> None:
    """A value replaced at constant namespace length is scanned as its new value."""

    module = types.ModuleType("tl_test_replacement_namespace")
    original = torch.zeros(1)
    module.stash = 123 if replacement_kind != "other_tensor" else original
    first = _scan_visits(module)
    size = len(module.__dict__)
    replacement = torch.ones(3)
    module.stash = [replacement] if replacement_kind == "list_of_tensor" else replacement
    assert len(module.__dict__) == size
    second = _scan_visits(module)
    assert any(item is replacement for item in second)
    assert not any(item is original for item in second)
    if replacement_kind == "other_tensor":
        assert any(item is original for item in first)


def test_deleted_slot_is_not_visited() -> None:
    """A tensor deleted from the namespace before the scan is not reached."""

    module = types.ModuleType("tl_test_deleted_namespace")
    module.stash = torch.ones(2)
    module.keep = torch.zeros(2)
    gone = module.stash
    del module.stash
    visited = _scan_visits(module)
    assert not any(item is gone for item in visited)
    assert any(item is module.keep for item in visited)


# A module namespace the model's forward names, so the session walks reach it.
stash_ns = types.ModuleType("tl_test_stash_namespace")
stash_ns.stash = 0
stash_ns.raise_now = False


class _StashingModel(nn.Module):
    """Stores an intermediate in a module namespace, optionally raising after it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stash ``relu(x)`` in ``stash_ns``; raise afterwards when asked."""

        stash_ns.stash = torch.relu(x)
        if stash_ns.raise_now:
            raise RuntimeError("tl_test mid-capture failure")
        return stash_ns.stash * 2


def _stamped(value: Any) -> bool:
    """Return whether ``value`` carries a TorchLens-owned stamp."""

    return isinstance(getattr(value, "_tl", None), TorchLensMeta)


def test_repeated_captures_clear_the_namespace_stash() -> None:
    """Each capture's stash is reached and cleared; the graph never changes."""

    model = _StashingModel()
    stash_ns.raise_now = False
    labels = None
    previous: list[torch.Tensor] = []
    for _ in range(3):
        trace = tl.trace(model, torch.ones(4))
        assert not _stamped(stash_ns.stash)
        assert not any(_stamped(old) for old in previous)
        labels = labels or list(trace.layer_labels)
        assert list(trace.layer_labels) == labels
        previous.append(stash_ns.stash)
        trace.cleanup()


def test_capture_after_mid_capture_exception_is_clean() -> None:
    """A capture that raised after stashing leaves the next capture clean and unchanged."""

    model = _StashingModel()
    stash_ns.raise_now = False
    reference = list(tl.trace(model, torch.ones(4)).layer_labels)
    stash_ns.raise_now = True
    try:
        with pytest.raises(Exception):  # noqa: B017 - torchlens may wrap the forward's error
            tl.trace(model, torch.ones(4))
    finally:
        stash_ns.raise_now = False
    stashed_by_failure = stash_ns.stash
    trace = tl.trace(model, torch.ones(4))
    assert list(trace.layer_labels) == reference
    assert not _stamped(stash_ns.stash)
    assert not _stamped(stashed_by_failure)

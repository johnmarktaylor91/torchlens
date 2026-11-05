"""Value snapshots the validation deepcopy fallback uses for richer plain attributes.

When ``copy.deepcopy`` of a model fails, validation runs the live model and must
prove that every plain (non-registered) module attribute is back to its
pre-run value before the logged run. :mod:`torchlens._capture_state_helpers`
snapshots builtin scalars and containers; this module adds the kinds that
models commonly keep as plain attributes and that used to refuse validation:

* immutable torch descriptors (``torch.device``, ``torch.dtype``, ...) and other
  immutable values, kept as themselves;
* Python module objects and enum members, tracked by identity (they are
  process-wide singletons: reaching one through an attribute is the same as
  reaching it through a global name);
* tensors of any size, cloned whole (host memory for dense accelerator
  tensors), compared byte for byte and restored in place so other references
  to the same tensor see the restored value;
* NumPy arrays of non-object dtype, copied and restored the same way;
* torch, NumPy and ``random`` generators, through their own state APIs, and
  thread locks, through their held/free state (a lock whose state changed
  cannot be put back, so its restore raises instead);
* ``collections.deque`` values, item by item;
* instances of Python-defined classes whose whole state is their instance dict
  and ``__slots__`` (config objects, namespaces, buffers, helper objects),
  snapshotted entry by entry and restored in place. A class with a C-level base
  other than ``object`` (an extension or pybind type) can hold state no Python
  attribute shows, so it stays unsupported.

Each snapshot either captures the full value, so a later comparison proves the
attribute unchanged, or raises ``RuntimeError`` so the fallback names the
attribute as unsupported. Nothing here is a by-reference copy of mutable state.
"""

from __future__ import annotations

import collections
import dataclasses
import decimal
import enum
import fractions
import pathlib
import random
import re
import threading
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ._utils.tensor_leaves import flat_byte_view

_HOST_DEVICE = torch.device("cpu")
_IMMUTABLE_VALUE_TYPES: tuple[type, ...] = (
    torch.device,
    torch.dtype,
    torch.layout,
    torch.memory_format,
    range,
    slice,
    type(Ellipsis),
    type(NotImplemented),
    fractions.Fraction,
    decimal.Decimal,
    pathlib.PurePath,
)
_IDENTITY_VALUE_TYPES: tuple[type, ...] = (types.ModuleType, enum.Enum)
_LOCK_TYPE = type(threading.Lock())
_RLOCK_TYPE = type(threading.RLock())
_RLOCK_STATE = re.compile(r"^<(locked|unlocked) .* owner=(\d+) count=(\d+)")


_HEAPTYPE_FLAG = 1 << 9  # Py_TPFLAGS_HEAPTYPE: the class was defined by a class statement


class _UnsetSlot:
    """Snapshot marker for a ``__slots__`` entry that holds no value."""

    def __repr__(self) -> str:
        """Name the marker in diagnostics."""

        return "<unset slot>"


_UNSET_SLOT = _UnsetSlot()

Snapshotter = Callable[[Any, str], Any]
Comparer = Callable[[Any, Any, str], bool]


def is_immutable_value(value: Any) -> bool:
    """Return whether ``value`` is an immutable value kept as itself in a snapshot."""

    return isinstance(value, _IMMUTABLE_VALUE_TYPES)


def is_identity_value(value: Any) -> bool:
    """Return whether ``value`` is a process-wide singleton tracked by identity."""

    return isinstance(value, _IDENTITY_VALUE_TYPES)


@dataclasses.dataclass(frozen=True, eq=False)
class PlainAttrTensorSnapshot:
    """Whole-value snapshot of a plain tensor attribute.

    Attributes
    ----------
    value:
        The tensor object the attribute held, restored in place when possible.
    saved:
        Detached copy of its contents (host memory for dense accelerator tensors).
    """

    value: torch.Tensor
    saved: torch.Tensor


@dataclasses.dataclass(frozen=True, eq=False)
class PlainAttrArraySnapshot:
    """Whole-value snapshot of a plain NumPy array attribute."""

    value: np.ndarray
    saved: np.ndarray


@dataclasses.dataclass(frozen=True, eq=False)
class PlainAttrGeneratorSnapshot:
    """Snapshot of a random generator through its own state API."""

    value: Any
    state: Any


@dataclasses.dataclass(frozen=True, eq=False)
class PlainAttrObjectSnapshot:
    """Attribute-by-attribute snapshot of a plain Python object.

    Attributes
    ----------
    value:
        The object the attribute held; a reassignment restores this object.
    attrs:
        Snapshot of each entry of its instance dict.
    slots:
        Snapshot of each ``__slots__`` value (``_UNSET_SLOT`` for an unset slot).
    """

    value: Any
    attrs: dict[str, Any]
    slots: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, eq=False)
class PlainAttrDequeSnapshot:
    """Item-by-item snapshot of a ``collections.deque``, restored in place."""

    value: collections.deque[Any]
    items: list[Any]


RICH_SNAPSHOT_TYPES = (
    PlainAttrDequeSnapshot,
    PlainAttrTensorSnapshot,
    PlainAttrArraySnapshot,
    PlainAttrGeneratorSnapshot,
    PlainAttrObjectSnapshot,
)


def _holds_on_host(tensor: torch.Tensor) -> bool:
    """Return whether a tensor snapshot can live in host memory bit for bit."""

    return (
        type(tensor) is torch.Tensor
        and tensor.layout == torch.strided
        and tensor.device.type not in ("cpu", "meta")
        and not tensor.is_quantized
    )


def snapshot_tensor(value: torch.Tensor, attr_path: str) -> PlainAttrTensorSnapshot:
    """Snapshot a plain tensor attribute whole.

    Raises
    ------
    RuntimeError
        If the tensor's contents cannot be copied and compared byte for byte
        (meta, quantized, sparse or other non-strided layouts, or a subclass
        that refuses a byte view).
    """

    detached = value.detach()
    if detached.layout != torch.strided or detached.device.type == "meta" or detached.is_quantized:
        raise RuntimeError(
            "TorchLens validation deepcopy fallback cannot snapshot plain attribute "
            f"'{attr_path}': a {detached.layout} / {detached.device.type} tensor has no "
            "byte-comparable contents."
        )
    try:
        with torch.no_grad():
            saved = (
                detached.to(_HOST_DEVICE, copy=True)
                if _holds_on_host(detached)
                else detached.clone()
            )
            flat_byte_view(saved)
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        raise RuntimeError(
            "TorchLens validation deepcopy fallback cannot snapshot plain attribute "
            f"'{attr_path}': its tensor contents cannot be copied byte for byte ({exc})."
        ) from exc
    return PlainAttrTensorSnapshot(value=value, saved=saved)


def _tensor_snapshots_equal(left: PlainAttrTensorSnapshot, right: PlainAttrTensorSnapshot) -> bool:
    """Return whether two tensor snapshots name the same tensor holding the same bytes.

    A reassignment is a change even to an equal tensor: the restore puts the
    original object back, so every other reference to it agrees again.
    """

    a, b = left.saved, right.saved
    if left.value is not right.value or a.dtype != b.dtype or a.shape != b.shape:
        return False
    with torch.no_grad():
        return bool(torch.equal(flat_byte_view(a), flat_byte_view(b)))


def _restore_tensor(snapshot: PlainAttrTensorSnapshot) -> torch.Tensor:
    """Write the saved contents back into the original tensor, else return a fresh copy."""

    original, saved = snapshot.value, snapshot.saved
    if original.shape == saved.shape and original.dtype == saved.dtype:
        try:
            with torch.no_grad():
                original.copy_(saved.to(original.device))
            return original
        except (RuntimeError, TypeError, NotImplementedError):
            pass  # an inference tensor or a rebound storage: hand back a copy instead
    return saved.to(original.device, copy=True)


def snapshot_array(value: np.ndarray, attr_path: str) -> PlainAttrArraySnapshot:
    """Snapshot a NumPy array attribute whole; object arrays hold references and refuse."""

    if value.dtype.hasobject:
        raise RuntimeError(
            "TorchLens validation deepcopy fallback cannot snapshot plain attribute "
            f"'{attr_path}': a NumPy object array holds references, not values."
        )
    return PlainAttrArraySnapshot(value=value, saved=value.copy())


def _arrays_equal(left: PlainAttrArraySnapshot, right: PlainAttrArraySnapshot) -> bool:
    """Return whether two array snapshots name the same array holding the same bytes."""

    a, b = left.saved, right.saved
    return (
        left.value is right.value
        and a.dtype == b.dtype
        and a.shape == b.shape
        and a.tobytes() == b.tobytes()
    )


def _restore_array(snapshot: PlainAttrArraySnapshot) -> np.ndarray:
    """Copy the saved contents back into the original array, else return a copy."""

    original, saved = snapshot.value, snapshot.saved
    if original.shape == saved.shape and original.dtype == saved.dtype and original.flags.writeable:
        np.copyto(original, saved)
        return original
    return saved.copy()


def _lock_state(value: Any) -> Any:
    """Return a thread lock's held/free state, or ``None`` for any other value."""

    if isinstance(value, _LOCK_TYPE):
        return ("lock", value.locked())
    if isinstance(value, _RLOCK_TYPE):
        match = _RLOCK_STATE.match(repr(value))
        return None if match is None else ("rlock", *match.groups())
    return None


def _generator_state(value: Any) -> Any:
    """Return a detached copy of a supported generator's or lock's state, or ``None``."""

    if isinstance(value, torch.Generator):
        return value.get_state().clone()
    if isinstance(value, np.random.RandomState):
        return value.get_state(legacy=False)
    if isinstance(value, np.random.Generator):
        return value.bit_generator.state
    if isinstance(value, random.Random):
        return value.getstate()
    return _lock_state(value)


def _set_generator_state(value: Any, state: Any) -> None:
    """Restore a supported generator's state; a changed lock cannot be restored.

    Raises
    ------
    RuntimeError
        If ``value`` is a thread lock: releasing or taking it on the forward's
        behalf could break whoever holds it, so the fallback fails closed.
    """

    if _lock_state(value) is not None:
        raise RuntimeError(
            "TorchLens validation deepcopy fallback cannot restore a thread lock "
            "whose held/free state changed during the forward."
        )
    if isinstance(value, (torch.Generator, np.random.RandomState)):
        value.set_state(state)
    elif isinstance(value, np.random.Generator):
        value.bit_generator.state = state
    else:
        value.setstate(state)


def _states_equal(left: Any, right: Any) -> bool:
    """Compare generator states made of tensors, arrays, dicts, tuples and scalars."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and bool(torch.equal(left, right))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_states_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _states_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and bool(left == right)


def _slot_descriptors(kind: type) -> dict[str, Any] | None:
    """Return the slot descriptors of a Python-defined class whose layout is only slots.

    Parameters
    ----------
    kind:
        Class of the value being snapshotted.

    Returns
    -------
    dict[str, Any] | None
        Slot name to member descriptor (empty for a dict-only class), or ``None``
        when an instance may hold state outside its instance dict and slots: a
        C-level base other than ``object`` (an extension type, a pybind class) or
        a layout larger than the same slots would give a pure Python class.
    """

    descriptors: dict[str, Any] = {}
    for base in kind.__mro__:
        if base is object:
            continue
        if not base.__flags__ & _HEAPTYPE_FLAG:
            return None
        for name, attr in vars(base).items():
            if isinstance(attr, types.MemberDescriptorType) and attr.__objclass__ is base:
                descriptors.setdefault(name, attr)
    layout = tuple(descriptors)
    layout += ("__dict__",) if kind.__dictoffset__ else ()
    layout += ("__weakref__",) if kind.__weakrefoffset__ else ()
    probe = type("_LayoutProbe", (), {"__slots__": layout})
    return descriptors if probe.__basicsize__ == kind.__basicsize__ else None


def _object_descriptors(kind: type) -> dict[str, Any] | None:
    """Return the slot descriptors of a snapshottable object class, or ``None``.

    ``types.SimpleNamespace`` keeps its whole state in its instance dict, so it
    is snapshotted like a dict-only Python class.
    """

    if kind is types.SimpleNamespace:
        return {}
    return _slot_descriptors(kind)


def is_snapshot_object(value: Any) -> bool:
    """Return whether ``value`` is snapshotted entry by entry and restored in place.

    A second reference to such an object inside its own snapshot (a cycle, such
    as an element that points back at its group) is tracked by identity: the
    enclosing snapshot already holds, compares and restores its entries.
    """

    return not isinstance(value, (list, tuple, dict, set, frozenset)) and (
        _object_descriptors(type(value)) is not None
    )


def hashable_snapshot(snapshot: Any) -> Any:
    """Return a dict-key or set-item snapshot, refusing one that compares by identity.

    Raises
    ------
    TypeError
        If the snapshot is (or contains) a whole-value snapshot from this module:
        those compare entry by entry, not by hash, so two snapshots of one key
        would never match as keys.
    """

    if isinstance(snapshot, RICH_SNAPSHOT_TYPES):
        raise TypeError("a whole-value snapshot cannot serve as a dict key or set item")
    if isinstance(snapshot, (tuple, frozenset)):
        for item in snapshot:
            hashable_snapshot(item)
    hash(snapshot)
    return snapshot


def _snapshot_object(value: Any, attr_path: str, snapshot_item: Snapshotter) -> Any:
    """Snapshot a Python object entry by entry, or return ``NotImplemented``."""

    descriptors = _object_descriptors(type(value))
    if descriptors is None:
        return NotImplemented
    namespace = getattr(value, "__dict__", None) if type(value).__dictoffset__ else {}
    if not isinstance(namespace, dict) or namespace is value:
        return NotImplemented
    slots: dict[str, Any] = {}
    for name, descriptor in descriptors.items():
        try:
            item = descriptor.__get__(value, type(value))
        except AttributeError:  # an unset slot
            slots[name] = _UNSET_SLOT
            continue
        slots[name] = snapshot_item(item, f"{attr_path}.{name}")
    attrs = {name: snapshot_item(item, f"{attr_path}.{name}") for name, item in namespace.items()}
    return PlainAttrObjectSnapshot(value=value, attrs=attrs, slots=slots)


def snapshot_rich_value(value: Any, attr_path: str, snapshot_item: Snapshotter) -> Any:
    """Return a snapshot for a value outside the builtin scalar and container kinds.

    Parameters
    ----------
    value:
        Attribute value (or nested item) to snapshot.
    attr_path:
        Human-readable path for diagnostics.
    snapshot_item:
        The recursive snapshotter for nested values (``(value, path) -> snapshot``).

    Returns
    -------
    Any
        A snapshot from this module, or ``NotImplemented`` when the value is of
        no kind handled here (the caller then refuses it).
    """

    if isinstance(value, torch.Tensor):
        return snapshot_tensor(value, attr_path)
    if isinstance(value, np.ndarray):
        return snapshot_array(value, attr_path)
    state = _generator_state(value)
    if state is not None:
        return PlainAttrGeneratorSnapshot(value=value, state=state)
    if isinstance(value, collections.deque):
        items = [snapshot_item(item, f"{attr_path}[{index}]") for index, item in enumerate(value)]
        return PlainAttrDequeSnapshot(value=value, items=items)
    return _snapshot_object(value, attr_path, snapshot_item)


def _entries_equal(
    left: dict[str, Any], right: dict[str, Any], attr_path: str, compare_item: Comparer
) -> bool:
    """Return whether two name-keyed entry snapshots match name for name."""

    return left.keys() == right.keys() and all(
        left[name] is right[name] is _UNSET_SLOT
        or (
            left[name] is not _UNSET_SLOT
            and right[name] is not _UNSET_SLOT
            and compare_item(left[name], right[name], f"{attr_path}.{name}")
        )
        for name in left
    )


def rich_values_equal(left: Any, right: Any, attr_path: str, compare_item: Comparer) -> bool:
    """Return whether two snapshots from :func:`snapshot_rich_value` match.

    Every kind must also be the same object: a reassignment is a change even
    when the new object holds equal values.
    """

    if type(left) is not type(right):
        return False
    if isinstance(left, PlainAttrTensorSnapshot):
        return _tensor_snapshots_equal(left, right)
    if isinstance(left, PlainAttrArraySnapshot):
        return _arrays_equal(left, right)
    if left.value is not right.value:
        return False
    if isinstance(left, PlainAttrGeneratorSnapshot):
        return _states_equal(left.state, right.state)
    if isinstance(left, PlainAttrDequeSnapshot):
        return len(left.items) == len(right.items) and all(
            compare_item(a, b, f"{attr_path}[{index}]")
            for index, (a, b) in enumerate(zip(left.items, right.items, strict=True))
        )
    return _entries_equal(left.attrs, right.attrs, attr_path, compare_item) and _entries_equal(
        left.slots, right.slots, attr_path, compare_item
    )


def _entry_unchanged(
    value: Any, saved: Any, name: str, snapshot_item: Snapshotter, compare_item: Comparer
) -> bool:
    """Return whether one live entry still matches its snapshot."""

    try:
        current = snapshot_item(value, name)
    except RuntimeError:  # now unsnapshottable: changed
        return False
    return compare_item(current, saved, name)


def _restore_object(
    snapshot: PlainAttrObjectSnapshot,
    snapshot_item: Snapshotter,
    compare_item: Comparer,
    restore_item: Callable[[Any], Any],
) -> Any:
    """Put back each changed instance-dict entry and slot of the original object.

    Entries are written to the instance dict and through the slot descriptors,
    so property setters and ``__setattr__`` overrides do not run; entries added
    since the snapshot are removed and unset slots are cleared again.
    """

    value = snapshot.value
    namespace = vars(value) if snapshot.attrs or type(value).__dictoffset__ else {}
    for name in [name for name in namespace if name not in snapshot.attrs]:
        del namespace[name]
    for name, saved in snapshot.attrs.items():
        if name not in namespace or not _entry_unchanged(
            namespace[name], saved, name, snapshot_item, compare_item
        ):
            namespace[name] = restore_item(saved)
    descriptors = _object_descriptors(type(value)) or {}
    for name, saved in snapshot.slots.items():
        descriptor = descriptors[name]
        try:
            current = descriptor.__get__(value, type(value))
        except AttributeError:
            current = _UNSET_SLOT
        if saved is _UNSET_SLOT:
            if current is not _UNSET_SLOT:
                descriptor.__delete__(value)
        elif current is _UNSET_SLOT or not _entry_unchanged(
            current, saved, name, snapshot_item, compare_item
        ):
            descriptor.__set__(value, restore_item(saved))
    return value


def restore_rich_value(
    snapshot: Any,
    snapshot_item: Snapshotter,
    compare_item: Comparer,
    restore_item: Callable[[Any], Any],
) -> Any:
    """Restore a snapshot from :func:`snapshot_rich_value` and return the value to assign.

    Tensors and arrays are written back in place; generators get their state
    back; a deque gets its items back; an object gets back each instance-dict
    entry and slot that changed.
    """

    if isinstance(snapshot, PlainAttrTensorSnapshot):
        return _restore_tensor(snapshot)
    if isinstance(snapshot, PlainAttrArraySnapshot):
        return _restore_array(snapshot)
    if isinstance(snapshot, PlainAttrGeneratorSnapshot):
        _set_generator_state(snapshot.value, snapshot.state)
        return snapshot.value
    if isinstance(snapshot, PlainAttrDequeSnapshot):
        snapshot.value.clear()
        snapshot.value.extend(restore_item(item) for item in snapshot.items)
        return snapshot.value
    return _restore_object(snapshot, snapshot_item, compare_item, restore_item)

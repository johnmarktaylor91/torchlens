"""Plain-attribute leg of the capture observer bracket (U40.1, AGT-2360).

A default ``tl.trace`` is a full pure observer: after the capture every plain
instance attribute of every (non-lazy) module holds the object it held before
the forward, with the contents it had, and attributes the forward added are
gone. What the forward wrote is recorded per module as
:class:`~torchlens.data_classes.attribute_write.AttributeWrite` rows, read at
``trace.modules[address].attribute_writes``; ``CaptureOptions(carry_state=True)``
records the same rows and leaves the writes on the module.

The restore keeps identity: an object the forward rebound is put back in its
slot, and a builtin ``list``/``dict``/``set`` or a tensor the forward changed in
place is restored in place, so outside references to it see the restored
contents too. Builtin containers are snapshotted whatever their size, and a
tensor inside one is cloned whatever its size. Custom objects (and the items of a
container that are custom objects) are tracked by identity only. Containers nested
deeper than :data:`_MAX_DEPTH` levels are tracked by identity below that level, and
the capture names such an attribute in its ``capture_state_restore_incomplete``
warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .data_classes.attribute_write import AttributeWrite

__all__ = (
    "ModuleAttrSnapshot",
    "observer_attr_names",
    "settle_module_attrs",
    "snapshot_module_attrs",
)

#: Instance attributes the bracket never snapshots: the mode flag (restored
#: separately) and an instance-assigned ``forward`` (TorchLens's session-scoped
#: submodule wrappers live there during a capture).
_OBSERVER_SKIPPED_ATTRS = frozenset({"training", "forward"})

#: Failures a compare or restore step may raise; each is disclosed, never masked.
_RESTORE_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    KeyError,
    NotImplementedError,
)

_IMMUTABLE_SCALARS = (type(None), bool, int, float, complex, str, bytes)
_SCALAR_TYPES = frozenset(_IMMUTABLE_SCALARS)
_MISSING = object()
#: Container depth the snapshot, skeleton, compare, restore and record copy
#: descend. Each level costs at most two Python frames (a comprehension frame on
#: Python < 3.12), so 256 levels stay well inside the default 1000-frame
#: recursion limit; a ``RecursionError`` is still caught and disclosed.
_MAX_DEPTH = 256


@dataclass(frozen=True, slots=True)
class _AttrSlot:
    """One plain attribute as the forward found it.

    ``original`` is the object in the slot, ``snapshot`` its value snapshot and
    ``skeleton`` the identities inside it (see :func:`_skeleton`), so a restore
    puts back the same inner objects, not copies.
    """

    name: str
    attr_path: str
    original: Any
    snapshot: Any
    skeleton: Any
    #: The attribute nests deeper than ``_MAX_DEPTH`` (or overflowed the
    #: recursion limit): changes below that depth are neither restored nor recorded.
    incomplete: bool = False


@dataclass(frozen=True, slots=True)
class ModuleAttrSnapshot:
    """The plain attributes of one module before the forward."""

    module: Any
    cls: type
    names: frozenset[str]
    slots: tuple[_AttrSlot, ...]


def observer_attr_names(module: Any) -> frozenset[str]:
    """Return the plain instance attribute names the bracket snapshots."""

    from ._capture_state_helpers import _module_baseline_instance_attrs

    baseline = _module_baseline_instance_attrs()
    return frozenset(
        name
        for name in module.__dict__
        if name not in baseline
        and name not in _OBSERVER_SKIPPED_ATTRS
        and not name.startswith(("tl_", "_tl"))
    )


def _value_snapshot(module: Any, name: str, attr_path: str) -> tuple[Any, bool]:
    """Snapshot one attribute by value; return ``(snapshot, incomplete)``.

    Managed (parametrization) tensors, registered tensor aliases and e3nn
    ``Irreps`` keep the validation snapshotter's rules. Everything else goes
    through :func:`_observer_snapshot`: builtin containers of any size, tensors
    cloned whole, custom objects by identity. ``incomplete`` is true when part of
    the value nests past ``_MAX_DEPTH`` levels.
    """

    from ._capture_state_helpers import (
        _managed_plain_tensor_attr_snapshot,
        _PlainAttrIdentitySnapshot,
        _snapshot_module_plain_attr_value,
    )
    from ._plain_attr_fidelity import is_registered_tensor_alias

    value = module.__dict__[name]
    if _managed_plain_tensor_attr_snapshot(module, name, value) is not None or (
        is_registered_tensor_alias(module, value)
    ):
        return _snapshot_module_plain_attr_value(module, name, attr_path), False
    truncated: list[bool] = []
    try:
        return _observer_snapshot(value, 0, set(), truncated), bool(truncated)
    except RecursionError:
        return _PlainAttrIdentitySnapshot(value=value, value_type_name=type(value).__name__), True


def _snapshot_tensor(value: torch.Tensor) -> Any:
    """Clone a tensor whole (a meta tensor, or one that cannot be cloned, by identity)."""

    from ._capture_state_helpers import _PlainAttrIdentitySnapshot

    if not value.is_meta:
        try:
            return value.detach().clone()
        except _RESTORE_ERRORS:
            pass
    return _PlainAttrIdentitySnapshot(value=value, value_type_name=type(value).__name__)


def _snapshot_leaf(value: Any) -> Any:
    """Snapshot a non-container value: itself when immutable, else by identity."""

    from ._capture_state_helpers import _is_identity_stable_plain_attr, _PlainAttrIdentitySnapshot

    if isinstance(value, _IMMUTABLE_SCALARS):
        return value
    if isinstance(value, np.generic) and not isinstance(value, np.void):
        return value  # NumPy scalars are immutable; a structured np.void is a view
    if isinstance(value, torch.Tensor) and not _is_identity_stable_plain_attr(value):
        return _snapshot_tensor(value)
    return _PlainAttrIdentitySnapshot(value=value, value_type_name=type(value).__name__)


def _is_e3nn_tuple(value: Any) -> bool:
    """Return whether ``value`` is an e3nn ``Irreps`` tuple (validation snapshot rules)."""

    kind = type(value)
    return kind.__module__ == "e3nn.o3._irreps" and kind.__qualname__ in {
        "Irreps",
        "_MulIr",
        "Irrep",
    }


def _observer_snapshot(value: Any, depth: int, ancestors: set[int], truncated: list[bool]) -> Any:
    """Snapshot a plain-attribute value for the observer bracket.

    Builtin ``list``/``tuple``/``dict``/``set``/``frozenset`` containers are
    copied whatever their size, item by item; dict keys and set items are kept
    as they are (hashable). A container past ``_MAX_DEPTH`` levels, or one that contains
    itself, is tracked by identity at that point; past the depth bound
    ``truncated`` is marked, so the capture can name the attribute.
    """

    from ._capture_state_helpers import _PlainAttrIdentitySnapshot, _snapshot_plain_attr_value

    kind = type(value)
    if kind in _SCALAR_TYPES:
        return value
    if not isinstance(value, (list, tuple, dict, set, frozenset)):
        return _snapshot_leaf(value)
    if _is_e3nn_tuple(value):
        try:
            return _snapshot_plain_attr_value(value, "<e3nn>")
        except RuntimeError:
            return _PlainAttrIdentitySnapshot(value=value, value_type_name=kind.__name__)
    if depth >= _MAX_DEPTH or id(value) in ancestors:
        if depth >= _MAX_DEPTH:
            truncated.append(True)
        return _PlainAttrIdentitySnapshot(value=value, value_type_name=kind.__name__)
    if isinstance(value, (set, frozenset)):
        return frozenset(value)  # hashable items: a set changes by membership only
    ancestors.add(id(value))
    try:
        if isinstance(value, dict):
            snapshot: Any = {}
            for key, item in value.items():
                snapshot[key] = (
                    item
                    if type(item) in _SCALAR_TYPES
                    else _observer_snapshot(item, depth + 1, ancestors, truncated)
                )
            return snapshot
        items = [
            item
            if type(item) in _SCALAR_TYPES
            else _observer_snapshot(item, depth + 1, ancestors, truncated)
            for item in value
        ]
        # A list (tuple) subclass snapshots as a plain list (tuple) of its items.
        return items if isinstance(value, list) else tuple(items)
    finally:
        ancestors.discard(id(value))


def _observer_unchanged(value: Any, snapshot: Any, depth: int = 0) -> bool:
    """Return whether ``value`` still matches its observer snapshot (NaN-safe)."""

    from ._capture_state_helpers import _PlainAttrIdentitySnapshot
    from ._runnable_state import declared_slot_unchanged

    if isinstance(snapshot, _PlainAttrIdentitySnapshot):
        return value is snapshot.value
    if isinstance(snapshot, torch.Tensor):
        return isinstance(value, torch.Tensor) and declared_slot_unchanged(value, snapshot)
    if isinstance(snapshot, (_IMMUTABLE_SCALARS, np.generic)):
        return _equal_scalar(value, snapshot) or (
            isinstance(snapshot, np.generic)
            and type(value) is type(snapshot)
            and bool(value == snapshot)
        )
    if isinstance(snapshot, dict):
        if not isinstance(value, dict) or value.keys() != snapshot.keys():
            return False
        return all(_observer_unchanged(value[key], snapshot[key], depth + 1) for key in snapshot)
    if isinstance(snapshot, frozenset):
        return isinstance(value, (set, frozenset)) and bool(value == snapshot)
    if isinstance(snapshot, (list, tuple)):
        if not isinstance(value, type(snapshot)) or len(value) != len(snapshot):
            return False
        for item, item_snapshot in zip(value, snapshot):
            if item is item_snapshot and type(item) in _SCALAR_TYPES:
                continue
            if not _observer_unchanged(item, item_snapshot, depth + 1):
                return False
        return True
    return False


def _skeleton(value: Any, depth: int = 0) -> Any:
    """Record the identities inside a value-snapshotted attribute.

    A tensor records its version counter; a ``list``/``tuple`` its items and a
    ``dict`` its ``(key, item)`` pairs, each with the item's own skeleton; a
    ``set`` its items. Anything else needs no skeleton (immutable, or tracked
    by identity).
    """

    from ._runnable_state import declared_state_versions

    if isinstance(value, torch.Tensor):
        return declared_state_versions((value,)).get(id(value))
    if depth >= _MAX_DEPTH:
        return None
    if isinstance(value, (list, tuple)):
        return [
            (item, None if type(item) in _SCALAR_TYPES else _skeleton(item, depth + 1))
            for item in value
        ]
    if isinstance(value, dict):
        return [
            (key, item, None if type(item) in _SCALAR_TYPES else _skeleton(item, depth + 1))
            for key, item in value.items()
        ]
    if isinstance(value, (set, frozenset)):
        return list(value)
    return None


def snapshot_module_attrs(modules: list[tuple[str, Any]]) -> tuple[ModuleAttrSnapshot, ...]:
    """Snapshot the plain attributes of each distinct module (path order)."""

    snapshots: list[ModuleAttrSnapshot] = []
    seen: set[int] = set()
    for path, module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        names = observer_attr_names(module)
        slots = []
        for name in sorted(names):
            original = module.__dict__[name]
            attr_path = f"{path or '<root>'}.{name}"
            snapshot, incomplete = _value_snapshot(module, name, attr_path)
            skeleton = (
                None
                if _is_identity_only(snapshot) or _is_special(snapshot)
                else (_skeleton(original))
            )
            slots.append(_AttrSlot(name, attr_path, original, snapshot, skeleton, incomplete))
        snapshots.append(ModuleAttrSnapshot(module, type(module), names, tuple(slots)))
    return tuple(snapshots)


def _is_identity_only(snapshot: Any) -> bool:
    """Return whether the bracket tracks this attribute by identity only."""

    from ._capture_state_helpers import _PlainAttrIdentitySnapshot

    return isinstance(snapshot, _PlainAttrIdentitySnapshot)


def _is_special(snapshot: Any) -> bool:
    """Managed (parametrization) and e3nn snapshots keep the by-value restore path."""

    from ._capture_state_helpers import _PlainAttrE3nnTupleSnapshot, _PlainAttrManagedTensorSnapshot

    return isinstance(snapshot, (_PlainAttrManagedTensorSnapshot, _PlainAttrE3nnTupleSnapshot))


def _contents_unchanged(module: Any, slot: _AttrSlot) -> bool:
    """Return whether the ORIGINAL object still has its snapshot contents."""

    from ._capture_state_helpers import _plain_attr_values_equal, _snapshot_module_plain_attr_value

    snapshot = slot.snapshot
    if _is_identity_only(snapshot):
        return True  # interior invisible to the bracket
    if isinstance(slot.original, _IMMUTABLE_SCALARS):
        return True
    try:
        if _is_special(snapshot):
            current = _snapshot_module_plain_attr_value(module, slot.name, slot.attr_path)
            return _plain_attr_values_equal(current, snapshot, slot.attr_path)
        return _observer_unchanged(slot.original, snapshot)
    except _RESTORE_ERRORS:
        return False  # no longer comparable


def _original_tensor_ids(original: Any, skeleton: Any, depth: int = 0) -> set[int]:
    """Ids of the tensors inside an attribute before the forward (restored in place)."""

    if isinstance(original, torch.Tensor):
        return {id(original)}
    if skeleton is None or depth >= _MAX_DEPTH or not isinstance(skeleton, list):
        return set()
    ids: set[int] = set()
    for row in skeleton:
        if isinstance(row, tuple) and len(row) in (2, 3):
            ids |= _original_tensor_ids(row[-2], row[-1], depth + 1)
    return ids


def _copy_left_by_forward(value: Any, originals: set[int], depth: int = 0) -> Any:
    """Copy an object the forward changed in place (the trace keeps this copy).

    Builtin ``list``/``tuple``/``dict``/``set`` containers are copied
    recursively; a tensor that existed before the forward (``originals``; the
    restore rewrites it in place) is cloned, whatever its size; a tensor the
    forward created and any other object are kept by reference.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().clone() if id(value) in originals else value
    kind = type(value)
    if depth >= _MAX_DEPTH or not isinstance(value, (list, tuple, dict, set)):
        return value
    if isinstance(value, set):
        return set(value)
    if isinstance(value, dict):
        return {
            key: _copy_left_by_forward(item, originals, depth + 1) for key, item in value.items()
        }
    items = [_copy_left_by_forward(item, originals, depth + 1) for item in value]
    if isinstance(value, list):
        return items
    make = getattr(kind, "_make", None)  # a namedtuple keeps its type
    return tuple(items) if kind is tuple or make is None else make(items)


def _restore_tensor(obj: Any, snapshot: torch.Tensor, version: int | None) -> None:
    """Write a tensor's snapshot back in place (the buffer rules: only changed bytes)."""

    from ._runnable_state import write_back_slot

    if not isinstance(obj, torch.Tensor):
        return
    if obj.shape != snapshot.shape or obj.dtype != snapshot.dtype:
        obj.data = snapshot  # resized or retyped in place: swap the storage back
    else:
        write_back_slot(obj, snapshot, version)


def _restore_into(obj: Any, snapshot: Any, skeleton: Any) -> None:
    """Restore ``obj`` (an original object) to ``snapshot`` in place, inner identities kept."""

    if isinstance(snapshot, torch.Tensor):
        _restore_tensor(obj, snapshot, skeleton)
        return
    if skeleton is None or _is_identity_only(snapshot) or _is_special(snapshot):
        return
    if isinstance(obj, (list, tuple)) and isinstance(snapshot, (list, tuple)):
        for (item, child), item_snapshot in zip(skeleton, snapshot, strict=True):
            _restore_into(item, item_snapshot, child)
        if isinstance(obj, list):
            obj[:] = [item for item, _child in skeleton]
    elif isinstance(obj, dict) and isinstance(snapshot, dict):
        for (_key, item, child), item_snapshot in zip(skeleton, snapshot.values(), strict=True):
            _restore_into(item, item_snapshot, child)
        obj.clear()
        obj.update((key, item) for key, item, _child in skeleton)
    elif isinstance(obj, set):
        obj.clear()
        obj.update(skeleton)


def _restore_contents(slot: _AttrSlot) -> bool:
    """Restore the original object's contents in place; ``False`` when it cannot be."""

    if _is_special(slot.snapshot) or (
        slot.skeleton is None and not isinstance(slot.snapshot, torch.Tensor)
    ):
        return False
    _restore_into(slot.original, slot.snapshot, slot.skeleton)
    return True


def _put_back(module: Any, name: str, value: Any) -> None:
    """Bind ``value`` as the plain attribute ``name`` again (the slot it came from)."""

    for registry_name in ("_parameters", "_buffers", "_modules"):
        registry = module.__dict__.get(registry_name)
        if isinstance(registry, dict) and name in registry:
            del registry[name]  # the forward registered the name; it was plain before
    module.__dict__[name] = value


def _equal_scalar(left: Any, right: Any) -> bool:
    """Return whether two immutable scalars are the same value (NaN equals NaN)."""

    return (
        isinstance(left, _IMMUTABLE_SCALARS)
        and type(left) is type(right)
        and bool(left == right or (left != left and right != right))
    )


def _settle_slot(module: Any, slot: _AttrSlot, restore: bool) -> AttributeWrite | None:
    """Record what the forward did to one pre-existing attribute; undo it if asked."""

    from ._capture_state_helpers import _plain_attr_restore_value

    current = module.__dict__.get(slot.name, _MISSING)
    contents_changed = not _contents_unchanged(module, slot)
    if current is _MISSING:
        write: AttributeWrite | None = AttributeWrite("removed", None)
    elif current is not slot.original:
        write = (
            None if _equal_scalar(current, slot.original) else AttributeWrite("replaced", current)
        )
    elif contents_changed:
        originals = _original_tensor_ids(slot.original, slot.skeleton)
        write = AttributeWrite("replaced", _copy_left_by_forward(current, originals))
    else:
        write = None
    if not restore:
        return write
    if contents_changed and not _restore_contents(slot):
        # Not restorable in place (a tuple, a managed tensor): bind a fresh value copy.
        _put_back(module, slot.name, _plain_attr_restore_value(slot.snapshot))
        return write
    if current is not slot.original:
        _put_back(module, slot.name, slot.original)
    return write


def settle_module_attrs(
    snapshots: tuple[ModuleAttrSnapshot, ...], *, restore: bool
) -> tuple[dict[int, dict[str, AttributeWrite]], list[str]]:
    """Record each module's attribute writes and, when ``restore``, undo them.

    Returns the writes keyed by ``id(module)`` (an entry, possibly empty, for
    every module whose class the forward did not swap) and the attribute paths
    that could not be restored.
    """

    writes: dict[int, dict[str, AttributeWrite]] = {}
    failed: list[str] = []
    for snap in snapshots:
        module = snap.module
        if type(module) is not snap.cls:
            continue
        module_writes: dict[str, AttributeWrite] = {}
        for slot in snap.slots:
            if restore and slot.incomplete:
                failed.append(f"{slot.attr_path} (nested past {_MAX_DEPTH} levels)")
            try:
                write = _settle_slot(module, slot, restore)
            except _RESTORE_ERRORS:
                failed.append(slot.attr_path)
                continue
            if write is not None:
                module_writes[slot.name] = write
        for name in sorted(observer_attr_names(module) - snap.names):
            module_writes[name] = AttributeWrite("added", module.__dict__[name])
            if restore:
                try:
                    delattr(module, name)
                except _RESTORE_ERRORS:
                    failed.append(f"{type(module).__name__}.{name}")
        writes[id(module)] = dict(sorted(module_writes.items()))
    return writes, failed

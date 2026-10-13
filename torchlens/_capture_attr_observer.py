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
contents too. Objects without a value snapshot (custom objects, containers past
the snapshot bounds) are tracked by identity only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
_MISSING = object()
#: Container depth the skeleton, restore and record copy descend (the
#: validation snapshotter's own bound).
_MAX_DEPTH = 64


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


def _value_snapshot(module: Any, name: str, attr_path: str) -> Any:
    """Snapshot one attribute by value, or by identity when unsupported.

    The validation snapshotter covers scalars, strings, tensors up to 4096
    elements and builtin containers of them (128 items, 64 levels). A larger
    plain tensor is cloned whole, like a buffer; any other object is tracked by
    identity, so a forward that rebinds it is undone while its interior is
    outside the bracket.
    """

    from ._capture_state_helpers import (
        _PlainAttrIdentitySnapshot,
        _snapshot_module_plain_attr_value,
    )

    try:
        return _snapshot_module_plain_attr_value(module, name, attr_path)
    except RuntimeError:
        value = module.__dict__[name]
        if isinstance(value, torch.Tensor) and not value.is_meta:
            try:
                return value.detach().clone()
            except _RESTORE_ERRORS:
                pass
        return _PlainAttrIdentitySnapshot(value=value, value_type_name=type(value).__name__)


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
        return [(item, _skeleton(item, depth + 1)) for item in value]
    if type(value) is dict:
        return [(key, item, _skeleton(item, depth + 1)) for key, item in value.items()]
    if type(value) in (set, frozenset):
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
            snapshot = _value_snapshot(module, name, attr_path)
            skeleton = (
                None
                if _is_identity_only(snapshot) or _is_special(snapshot)
                else (_skeleton(original))
            )
            slots.append(_AttrSlot(name, attr_path, original, snapshot, skeleton))
        snapshots.append(ModuleAttrSnapshot(module, type(module), names, tuple(slots)))
    return tuple(snapshots)


def _is_identity_only(snapshot: Any) -> bool:
    from ._capture_state_helpers import _PlainAttrIdentitySnapshot

    return isinstance(snapshot, _PlainAttrIdentitySnapshot)


def _is_special(snapshot: Any) -> bool:
    """Managed (parametrization) and e3nn snapshots keep the by-value restore path."""

    from ._capture_state_helpers import _PlainAttrE3nnTupleSnapshot, _PlainAttrManagedTensorSnapshot

    return isinstance(snapshot, (_PlainAttrManagedTensorSnapshot, _PlainAttrE3nnTupleSnapshot))


def _contents_unchanged(module: Any, slot: _AttrSlot) -> bool:
    """Return whether the ORIGINAL object still has its snapshot contents."""

    from ._capture_state_helpers import (
        _plain_attr_values_equal,
        _snapshot_module_plain_attr_value,
        _snapshot_plain_attr_value,
    )
    from ._runnable_state import declared_slot_unchanged

    snapshot = slot.snapshot
    if _is_identity_only(snapshot):
        return True  # interior invisible to the bracket
    if isinstance(snapshot, torch.Tensor):
        return isinstance(slot.original, torch.Tensor) and declared_slot_unchanged(
            slot.original, snapshot
        )
    if isinstance(slot.original, _IMMUTABLE_SCALARS):
        return True
    try:
        if _is_special(snapshot):
            current = _snapshot_module_plain_attr_value(module, slot.name, slot.attr_path)
        else:
            current = _snapshot_plain_attr_value(slot.original, slot.attr_path)
        return _plain_attr_values_equal(current, snapshot, slot.attr_path)
    except _RESTORE_ERRORS:
        return False  # grew past the snapshot bounds or no longer comparable


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
    if depth >= _MAX_DEPTH or not (kind in (list, dict, set) or isinstance(value, tuple)):
        return value
    if kind is set:
        return set(value)
    if kind is dict:
        return {
            key: _copy_left_by_forward(item, originals, depth + 1) for key, item in value.items()
        }
    items = [_copy_left_by_forward(item, originals, depth + 1) for item in value]
    if kind is list:
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
        if type(obj) is list:
            obj[:] = [item for item, _child in skeleton]
    elif type(obj) is dict and isinstance(snapshot, dict):
        for (_key, item, child), item_snapshot in zip(skeleton, snapshot.values(), strict=True):
            _restore_into(item, item_snapshot, child)
        obj.clear()
        obj.update((key, item) for key, item, _child in skeleton)
    elif type(obj) is set:
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

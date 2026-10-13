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


@dataclass(frozen=True, slots=True)
class _AttrSlot:
    """One plain attribute as the forward found it: its object and a value snapshot."""

    name: str
    attr_path: str
    original: Any
    snapshot: Any
    version: int | None


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


def snapshot_module_attrs(modules: list[tuple[str, Any]]) -> tuple[ModuleAttrSnapshot, ...]:
    """Snapshot the plain attributes of each distinct module (path order)."""

    from ._runnable_state import declared_state_versions

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
            version = (
                declared_state_versions((original,)).get(id(original))
                if isinstance(original, torch.Tensor)
                else None
            )
            slots.append(
                _AttrSlot(
                    name, attr_path, original, _value_snapshot(module, name, attr_path), version
                )
            )
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


def _copy_left_by_forward(value: Any, attr_path: str) -> Any:
    """Copy an object the forward changed in place (the trace keeps this copy)."""

    from ._capture_state_helpers import _plain_attr_restore_value, _snapshot_plain_attr_value

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if type(value) in (list, dict, set):
        return type(value)(value)
    try:
        return _plain_attr_restore_value(_snapshot_plain_attr_value(value, attr_path))
    except _RESTORE_ERRORS:
        return value


def _restore_contents(module: Any, slot: _AttrSlot) -> bool:
    """Restore the original object's contents in place; ``False`` when it cannot be."""

    from ._capture_state_helpers import _plain_attr_restore_value
    from ._runnable_state import write_back_slot

    original, snapshot = slot.original, slot.snapshot
    if isinstance(snapshot, torch.Tensor) and isinstance(original, torch.Tensor):
        if original.shape != snapshot.shape or original.dtype != snapshot.dtype:
            original.data = snapshot  # resized or retyped in place: swap the storage back
        else:
            write_back_slot(original, snapshot, slot.version)
        return True
    if _is_special(snapshot):
        return False
    value = _plain_attr_restore_value(snapshot)
    if type(original) is list:
        original[:] = value
    elif type(original) in (dict, set):
        original.clear()
        original.update(value)
    else:
        return False
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
        write = AttributeWrite("replaced", _copy_left_by_forward(current, slot.attr_path))
    else:
        write = None
    if not restore:
        return write
    if contents_changed and not _restore_contents(module, slot):
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

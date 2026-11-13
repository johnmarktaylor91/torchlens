"""Record state restore for the 3.0 reader: one decoded row becomes one record object.

``restore_record`` runs the record's state hook on its row; ``true_up_record`` then puts
back what the hook's default fill and container coercion changed, so load then save is
lossless. Two causes, both reader-side: a saved ``null`` that the hook coerces to its typed
default (``_buffer_persistence`` ``null`` became ``{}``), and a column the save level
omitted (``forward_args`` at ``portable``) that leaves the attribute unset, so a property
over it (``has_saved_forward_args``) raises instead of answering ``False``. Nothing here
touches a class attribute or property: only instance slots the schema names.

``settle_tuple`` and ``settle_mapping`` rebuild a decoded container with the type the
decoder gave it once its members are settled; ``loaded_persistent_buffer_universe``
rebuilds, from the validated runnable descriptor, the capture-time persistent-buffer
universe a runnable resave of a loaded trace declares its slots from.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any, Final

from ..._data_classes._state_adapter import state_restore
from ..._data_classes.field_policy import StorageKind
from ..._utils._record_group_table import record_storage_attr
from .._replay_recipe_codec import attach_replay_recipes
from . import contract
from .decode import CELL_CODE, cell_error
from .schema import record_schema
from .values import tuple_subtypes

__all__ = [
    "computed_columns",
    "loaded_persistent_buffer_universe",
    "restore_record",
    "settle_mapping",
    "settle_tuple",
    "true_up_record",
]

TLSPEC_KEY: Final = "tlspec_version"
_UNSET: Final = object()


@cache
def computed_columns(kind: str) -> frozenset[str]:
    """Columns the writer fills from a read-only property; loaded objects recompute them."""

    return frozenset(
        name
        for name, column in record_schema(kind).items()
        if column.storage == StorageKind.COMPUTED.value
    )


@cache
def _stored_columns(kind: str) -> frozenset[str]:
    """Columns a loaded object stores itself (every column that is not computed)."""

    return frozenset(record_schema(kind)) - computed_columns(kind)


def restore_record(obj: Any, row: Mapping[str, Any], *, version: int, kind: str) -> None:
    """Run the record's state hook on its decoded row, computed columns dropped, then true up."""

    computed = computed_columns(kind)
    state = {name: value for name, value in row.items() if name not in computed}
    state[TLSPEC_KEY] = version
    setstate = getattr(type(obj), "__setstate__", None)
    if setstate is not None and setstate is not getattr(object, "__setstate__", None):
        setstate(obj, state)
    else:
        state.pop(TLSPEC_KEY)
        state_restore(obj, state)
    true_up_record(obj, row, kind)


def _writable_field(cls: type, name: str) -> bool:
    """Whether ``name`` is instance state the true-up may write: absent from the class, or
    a data descriptor that is not a property (a record's row-cell field). A property, a
    method or a class constant is never written: the writer recomputes those."""

    attr = inspect.getattr_static(cls, name, _UNSET)
    if attr is _UNSET:
        return True
    return not isinstance(attr, property) and hasattr(type(attr), "__set__")


def true_up_record(obj: Any, row: Mapping[str, Any], kind: str) -> None:
    """Restore the saved value of every stored column the state hook changed.

    A column saved ``null`` whose attribute the hook filled with a typed default goes back
    to ``None``; a stored column the save level omitted, left unset by the hook, is set to
    ``None`` so the properties over it answer. Row-cell fields (data descriptors) are
    written through the descriptor; a property, method or class constant is never written:
    the writer recomputes those.
    """

    if getattr(obj, "__dict__", None) is None:
        return
    cls = type(obj)
    for name in _stored_columns(kind):
        attr = record_storage_attr(obj, name)
        if not _writable_field(cls, attr):
            continue
        current = getattr(obj, attr, _UNSET)
        if name in row:
            if row[name] is None and current is not None and current is not _UNSET:
                setattr(obj, attr, None)
        elif current is _UNSET:
            setattr(obj, attr, None)


def settle_tuple(value: tuple[Any, ...], items: list[Any], at: str) -> tuple[Any, ...]:
    """Rebuild a decoded tuple with the type the decoder gave it: a named tuple through
    ``_make``, a ``tuple_subtypes`` constructor (torch.Size, torch.return_types) through the
    code-owned table, else ``tuple``. The type is the decoder's, never a name from the artifact.
    """

    cls = type(value)
    if cls is tuple:
        return tuple(items)
    maker = getattr(cls, "_make", None)
    if callable(maker):
        return maker(items)
    if cls in tuple_subtypes().values():
        return cls(tuple(items))
    raise cell_error(at, f"tuple of type {cls.__name__} cannot be rebuilt", code=CELL_CODE)


def settle_mapping(value: dict[Any, Any], items: dict[Any, Any]) -> dict[Any, Any]:
    """Rebuild a decoded dict with its builtin default factory when it is a defaultdict."""

    if isinstance(value, defaultdict):
        return defaultdict(value.default_factory, items)
    return items


def loaded_persistent_buffer_universe(descriptor: Any) -> dict[str, dict[str, Any]] | None:
    """Return ``name -> {shape, dtype, device}`` for the descriptor's persistent buffer slots.

    A live capture records this universe from the model; a loaded trace has no model, so a
    runnable resave reads the same facts from the descriptor the load validated (every
    persistent buffer is a tensor slot with a state binding). ``None`` without a descriptor.
    """

    if descriptor is None:
        return None
    universe: dict[str, dict[str, Any]] = {}
    for slot in descriptor.tensor_slots:
        binding = slot.state_binding
        role = getattr(slot.role, "value", slot.role)
        if binding is None or not binding.persistent or role != "buffer":
            continue
        device = slot.device_type
        if slot.device_index is not None:
            device = f"{device}:{slot.device_index}"
        universe[binding.state_dict_name] = {
            "shape": tuple(slot.shape),
            "dtype": slot.dtype,
            "device": device,
        }
    return universe


def _stamp_loaded_context(
    trace: Any,
    manifest: Mapping[str, Any],
    root: Path,
    parse_provenance: Callable[[dict[str, Any]], Any],
    file_digest: Callable[[Path], str],
) -> None:
    """Retain validated artifact origin and capture-time provenance on the loaded trace."""

    trace._loaded_from_bundle = True
    trace._source_bundle_manifest_sha256 = file_digest(root / contract.MANIFEST_FILE)
    trace._source_bundle_path = root
    trace._source_bundle_created_at = manifest["created_at"]
    # Keep the original certificate rather than re-deriving weaker capture facts.
    raw_provenance = manifest.get("provenance")
    trace._source_bundle_provenance = (
        parse_provenance(raw_provenance) if isinstance(raw_provenance, dict) else None
    )
    # Replay recipes (listed, digest-checked) are bound on the first validation read.
    attach_replay_recipes(trace, root, manifest)

"""Late buffer indexing for pre-forward model-owned tensors the held scan leaves unnamed.

Model preparation stamps registered buffers and the PUBLIC plain tensors a module
holds (``_held_tensor_scan``) as buffer sources. The pre-forward ownership snapshot
(``_module_boundary_adoption.collect_pre_forward_tensor_ids``) reaches further into
model state: private attributes (``self._grid``, a ``self._wh_pairs`` list), containers
inside plain helper objects (a position cache dict on a non-Module getter) and plain
holders nested in plain holders (``Independent(Normal(loc, scale))``). Those tensors
exist before the forward and are read from model state, so they are graph sources
exactly like a held buffer. Without a stamp, their first consumer was recorded as a
parentless op with an unattributed tensor argument.

At first use, :func:`late_index_model_owned_tensor` resolves such a tensor to an
address derived from where the model holds it; the caller
(``buffer_writes.resolve_or_late_index_buffer_address``) stamps it as a session buffer and
logs a buffer source. It admits only the exact object the snapshot pinned,
untagged, and not reachable only from a forward callable (those are closure
constants). A tensor first seen mid-forward is not in the snapshot and keeps the
unattributed witness.
"""

from __future__ import annotations

from collections import deque
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ... import _state
from ._held_tensor_scan import _HELD_KEY_ESCAPES, _render_held_key
from ._session_cleanup import _is_isinstance_hostile_deprecation_shim
from ._tl import get_module_meta, get_tensor_label

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

#: ``module_build_data`` key of the weak reference to the prepared root model.
ROOT_MODEL_KEY = "pre_forward_root_model"
#: ``module_build_data`` key of the lazily built ``id -> address`` map.
_ADDRESSES_KEY = "late_held_tensor_addresses"
#: Address stem for a snapshot tensor no longer reachable from model state.
_FALLBACK_STEM = "pre_forward_held"
#: Object levels below a module attribute the address walk descends (the
#: session walker's bound).
_MAX_DEPTH = 12
_SCALAR_TYPES = (str, bytes, int, float, bool, complex)


def late_index_model_owned_tensor(trace: Trace, tensor: torch.Tensor) -> str | None:
    """Return the buffer address for an unnamed pre-forward model-owned tensor.

    Parameters
    ----------
    trace:
        Active capture trace holding the pre-forward ownership snapshot.
    tensor:
        Tensor argument met at a wrapped call with no buffer address.

    Returns
    -------
    str | None
        The buffer address for the caller to stamp, or ``None`` when ``tensor`` is not an untagged
        pre-forward model-owned tensor.
    """

    # Snapshot membership first: it is one dict probe, and the wrapper ladder asks
    # about every non-buffer tensor argument, almost all of them labeled activations.
    build_data = trace._module_capture_ws.module_build_data
    if (build_data.get("model_owned_tensor_ids_at_entry") or {}).get(id(tensor)) is not tensor:
        return None
    if isinstance(tensor, nn.Parameter) or get_tensor_label(tensor) is not None:
        return None
    if trace._raw_transform_escape_detected or trace._raw_dynamo_region_detected:
        return None
    if id(tensor) in (build_data.get("forward_outside_tensor_ids_at_entry") or ()):
        return None
    taken = {
        str(getattr(stamp, "address", "")) for stamp in trace._session_buffer_identity.values()
    }
    addresses = build_data.get(_ADDRESSES_KEY)
    if addresses is None:
        root_ref = build_data.get(ROOT_MODEL_KEY)
        root = root_ref() if callable(root_ref) else None
        with _state.pause_logging():
            addresses = held_tensor_addresses(root, taken) if root is not None else {}
        build_data[_ADDRESSES_KEY] = addresses
    return addresses.get(id(tensor)) or _reserve(_FALLBACK_STEM, taken)


def held_tensor_addresses(root: nn.Module, taken: set[str]) -> dict[int, str]:
    """Map every plain tensor reachable from module state to a held address.

    Each module's ``__dict__`` is walked (submodules are visited as modules of
    their own, Python modules are not entered). The first path to a tensor names
    it: the module address, the attribute name, then one ``[segment]`` per level
    (list index, rendered dict key or plain-object attribute), so the last ``.``
    still splits the owning module from the held name.

    Parameters
    ----------
    root:
        The prepared root model.
    taken:
        Addresses already in use this session; reserved names are added to it.

    Returns
    -------
    dict[int, str]
        ``id(tensor) -> address`` for each reachable non-Parameter tensor.
    """

    found: dict[int, str] = {}
    seen: set[int] = set()
    for module in root.modules():
        meta = get_module_meta(module)
        prefix = f"{meta.address}." if meta is not None and meta.address else ""
        for name, value in list(vars(module).items()):
            _walk_held_value(value, f"{prefix}{name}", found, seen, taken)
    return found


def _walk_held_value(
    value: Any, path: str, found: dict[int, str], seen: set[int], taken: set[str]
) -> None:
    """Record the tensors under one module attribute, depth first, in item order."""

    stack: list[tuple[Any, str, int]] = [(value, path, 0)]
    while stack:
        item, where, level = stack.pop()
        if item is None or _is_isinstance_hostile_deprecation_shim(item):
            continue
        if isinstance(item, torch.Tensor):
            if not isinstance(item, nn.Parameter) and id(item) not in found:
                found[id(item)] = _reserve(where, taken)
            continue
        if (
            isinstance(item, (_SCALAR_TYPES, nn.Module, ModuleType, type))
            or level >= _MAX_DEPTH
            or id(item) in seen
        ):
            continue
        seen.add(id(item))
        children = [(child, f"{where}[{segment}]", level + 1) for segment, child in _children(item)]
        stack.extend(reversed(children))


def _children(item: Any) -> list[tuple[str, Any]]:
    """Return ``(rendered segment, child)`` pairs of a container or plain object."""

    if isinstance(item, dict):
        return [(_render_held_key(key), child) for key, child in item.items()]
    if isinstance(item, (list, tuple, deque, set, frozenset)):
        return [(str(index), child) for index, child in enumerate(item)]
    namespace = getattr(item, "__dict__", None)
    if not isinstance(namespace, dict):
        return []
    return [
        (_escape_attr_name(name), child)
        for name, child in namespace.items()
        if name != "__annotations__"
    ]


def _escape_attr_name(name: Any) -> str:
    """Render a plain-object attribute name as a dot-free, colon-free segment."""

    if not isinstance(name, str):
        return _render_held_key(name)
    for raw, escaped in _HELD_KEY_ESCAPES:
        name = name.replace(raw, escaped)
    return name


def _reserve(name: str, taken: set[str]) -> str:
    """Return ``name``, or ``name#<n>`` with the smallest free ``n >= 2``; reserve it."""

    unique = name
    ordinal = 2
    while unique in taken:
        unique = f"{name}#{ordinal}"
        ordinal += 1
    taken.add(unique)
    return unique

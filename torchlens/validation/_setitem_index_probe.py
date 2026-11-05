"""In-domain index probe for integer and bool-mask index components of ``__setitem__``.

``dest[index] = value`` reads an integer index component whose VALUES decide
which destination cells are written, inside a domain bounded by the destination
dim the component addresses. The default random draw and the uniform value
steps can leave such a write unchanged even though the edge is real:

- a permutation of the written rows (``[0, 1]`` drawn as ``[1, 0]``) writes the
  same cells with the same scalar;
- the neighbouring rows a step reaches may already hold the written value (an
  asynchronous ``state[i, mask] = -1`` update where rows ``i - 1`` and
  ``i + 1`` are already ``-1``).

This probe moves ONE in-domain entry of the component to another in-domain
position (for a bool mask component: flips one mask entry), and picks the first
such change (in a fixed, bounded order) whose plain write on the saved call
arguments differs from the unperturbed write. It is an
extra probe for the replay retry ladder, not a proof: the verdict still comes
from the replay, and a dead index edge (no in-domain position changes the
write, for example a destination that already holds the value everywhere) has
no such move, yields no probe and still fails.

This module never registers ``__setitem__`` in the index-domain slot table
(``_index_domain._INDEX_DOMAIN_ARG_SPECS``), so the exemptions that read that
table see exactly what they saw before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._utils._record_group_table import record_member
from .._utils.tensor_utils import tensor_nanequal
from ._saved_value_proofs import _index_component_dims

if TYPE_CHECKING:
    from .._data_classes.op import Op

#: Index dtypes probed: integer positions and bool masks (``uint8`` masks are never probed).
_PROBED_INDEX_DTYPES = frozenset({torch.int32, torch.int64, torch.bool})

#: Most single-entry moves simulated per probe.
_MAX_CANDIDATE_MOVES = 64

#: Most index entries whose moves are tried (the earliest in-domain ones).
_MAX_PROBED_ENTRIES = 8

#: Element budget for the simulated writes; a large destination gets fewer moves.
_SIMULATED_ELEMENT_BUDGET = 1 << 26


def setitem_index_component_key(layer: Op, parent_label: str) -> Any:
    """Return the index-slot key a ``__setitem__`` parent occupies, if unique.

    Parameters
    ----------
    layer:
        Captured op.
    parent_label:
        Perturbed parent label.

    Returns
    -------
    Any
        ``1`` for a bare index or ``(1, position)`` for one component of a tuple
        index, when the op is ``__setitem__`` and the parent occupies exactly
        that one argument slot; otherwise ``None``. A parent that fills several
        slots is not probed: each slot would be moved on its own and the moves
        could cancel.
    """

    if getattr(layer, "func_name", None) != "__setitem__":
        return None
    positions = record_member(layer, "parent_arg_positions", {}) or {}
    keys = [
        key for key, label in (positions.get("args", {}) or {}).items() if label == parent_label
    ]
    if any(label == parent_label for label in (positions.get("kwargs", {}) or {}).values()):
        return None
    if len(keys) != 1:
        return None
    key = keys[0]
    if key == 1 or (isinstance(key, tuple) and len(key) == 2 and key[0] == 1):
        return key
    return None


def layer_has_setitem_index_parent(layer: Op) -> bool:
    """Return whether a ``__setitem__`` op has a parent inside its index argument.

    Parameters
    ----------
    layer:
        Child op being validated.

    Returns
    -------
    bool
        True when some parent label occupies exactly one index slot.
    """

    if getattr(layer, "func_name", None) != "__setitem__":
        return False
    positions = record_member(layer, "parent_arg_positions", {}) or {}
    labels = {label for label in (positions.get("args", {}) or {}).values() if label}
    return any(setitem_index_component_key(layer, label) is not None for label in labels)


def setitem_index_probe_values(
    layer: Op,
    parent_label: str,
    parent_values: torch.Tensor,
) -> torch.Tensor | None:
    """Return the parent's index values with one entry changed so the write changes.

    Parameters
    ----------
    layer:
        Captured ``__setitem__`` op being replayed.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved parent tensor values for that index slot.

    Returns
    -------
    torch.Tensor or None
        The index tensor with exactly one entry moved to another position of
        its destination dim (one entry flipped for a bool mask), chosen as the
        first change whose write on the saved destination and value differs
        from the unperturbed write; ``None`` when the parent is not a lone
        integer or bool index component, the saved call cannot be rebuilt, or
        no bounded change alters the write.
    """

    call = _setitem_call(layer, parent_label, parent_values)
    if call is None:
        return None
    destination, components, position, value, domain_size = call
    baseline = _simulated_write(destination, components, value)
    if baseline is None:
        return None
    if parent_values.dtype == torch.bool:
        candidates = _single_flips(parent_values, destination.numel())
    else:
        candidates = _single_entry_moves(parent_values, domain_size, destination.numel())
    for candidate in candidates:
        moved = components[:position] + (candidate,) + components[position + 1 :]
        written = _simulated_write(destination, moved, value)
        if written is not None and not tensor_nanequal(written, baseline, allow_tolerance=False):
            return candidate
    return None


def _setitem_call(
    layer: Op, parent_label: str, parent_values: torch.Tensor
) -> tuple[torch.Tensor, tuple[Any, ...], int, Any, int] | None:
    """Rebuild the saved ``__setitem__`` call around one integer or bool index component.

    Parameters
    ----------
    layer:
        Captured op.
    parent_label:
        Perturbed parent label.
    parent_values:
        Saved parent tensor values for its index slot.

    Returns
    -------
    tuple or None
        ``(destination, components, position, value, domain_size)`` with the
        saved parent values at ``components[position]`` (``domain_size`` is the
        size of the first addressed dim), or ``None`` when the call is not a
        probeable integer or bool-mask index write.
    """

    key = setitem_index_component_key(layer, parent_label)
    args: tuple[Any, ...] = record_member(layer, "saved_args", None) or ()
    if key is None or len(args) < 3 or not isinstance(args[0], torch.Tensor):
        return None
    if not isinstance(parent_values, torch.Tensor):
        return None
    if parent_values.dtype not in _PROBED_INDEX_DTYPES or parent_values.numel() == 0:
        return None
    destination, index = args[0], args[1]
    if key == 1:
        if isinstance(index, tuple):
            return None
        components, position = (parent_values,), 0
    else:
        if not isinstance(index, tuple) or not 0 <= key[1] < len(index):
            return None
        position = key[1]
        components = index[:position] + (parent_values,) + index[position + 1 :]
    dims = _index_component_dims(destination.ndim, components)
    dim = None if dims is None else dims[position]
    if dim is None:
        return None
    domain_size = int(destination.shape[dim])
    if domain_size < 2 and parent_values.dtype != torch.bool:
        return None
    return destination, components, position, args[2], domain_size


def _single_entry_moves(
    index: torch.Tensor, domain_size: int, destination_numel: int
) -> list[torch.Tensor]:
    """Return bounded single-entry in-domain moves of an index tensor, in order.

    Parameters
    ----------
    index:
        Saved integer index component.
    domain_size:
        Size of the destination dim it addresses (at least 2).
    destination_numel:
        Destination element count; bounds how many writes are simulated.

    Returns
    -------
    list of torch.Tensor
        For each of the earliest in-domain entries (``-n <= v < n``), the index
        with only that entry moved forward by 1, 2, ... positions modulo ``n``.
        Shifts run in the outer loop, so every probed entry is tried at a
        distance before any entry is tried farther away.
    """

    budget = _SIMULATED_ELEMENT_BUDGET // max(destination_numel, 1)
    limit = max(1, min(_MAX_CANDIDATE_MOVES, budget))
    flat = index.reshape(-1)
    in_domain = (flat >= -domain_size) & (flat < domain_size)
    entries = torch.nonzero(in_domain).reshape(-1)[:_MAX_PROBED_ENTRIES].tolist()
    moves: list[torch.Tensor] = []
    for shift in range(1, domain_size):
        for entry in entries:
            if len(moves) >= limit:
                return moves
            moved = flat.clone()
            moved[entry] = (int(flat[entry]) % domain_size + shift) % domain_size
            moves.append(moved.reshape(index.shape))
    return moves


def _single_flips(mask: torch.Tensor, destination_numel: int) -> list[torch.Tensor]:
    """Return bounded single-entry flips of a bool mask component, in order.

    Parameters
    ----------
    mask:
        Saved bool mask component.
    destination_numel:
        Destination element count; bounds how many writes are simulated.

    Returns
    -------
    list of torch.Tensor
        The mask with only its first, second, ... entry flipped.
    """

    budget = _SIMULATED_ELEMENT_BUDGET // max(destination_numel, 1)
    flat = mask.reshape(-1)
    flips: list[torch.Tensor] = []
    for entry in range(min(flat.numel(), max(1, min(_MAX_CANDIDATE_MOVES, budget)))):
        flipped = flat.clone()
        flipped[entry] = not bool(flat[entry])
        flips.append(flipped.reshape(mask.shape))
    return flips


def _simulated_write(
    destination: torch.Tensor, components: tuple[Any, ...], value: Any
) -> torch.Tensor | None:
    """Return ``destination`` with ``value`` written at the rebuilt index.

    Parameters
    ----------
    destination:
        Saved destination tensor (never modified).
    components:
        Index components (a one-element tuple stands for a bare index).
    value:
        Saved replacement value.

    Returns
    -------
    torch.Tensor or None
        The written copy, or ``None`` when the write raises.
    """

    index = components[0] if len(components) == 1 else components
    with torch.no_grad():
        written = destination.detach().clone()
        try:
            written[index] = value.detach() if isinstance(value, torch.Tensor) else value
        except (IndexError, RuntimeError, TypeError, ValueError):
            return None
    return written

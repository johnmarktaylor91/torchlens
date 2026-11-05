"""Arg-slot locators for perturbed parents in the perturbation exemption checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.op import Op


def _perturbed_parent_is_arg_position(
    layer: Op,
    layers_to_perturb: list[str],
    position: int,
) -> bool:
    """Return whether the perturbed parent occupies positional arg ``position``.

    Reads ``layer.parent_arg_positions["args"]`` (arg index -> parent layer label)
    and checks the perturbed label is the parent registered at ``position``. Used to
    confirm a perturbed parent is the DESTINATION (slot 0) by structure rather than
    by tensor-content equality, which can collide when a value-parent's contents
    match the destination.
    """

    arg_positions = (record_member(layer, "parent_arg_positions", None) or {}).get("args", {})
    return arg_positions.get(position) == layers_to_perturb[0]


def _perturbed_parent_arg_positions(layer: Op, layers_to_perturb: list[str]) -> set[int]:
    """Return positional arg slots occupied by the perturbed parent.

    Parameters
    ----------
    layer:
        Captured op whose parent-argument map is being inspected.
    layers_to_perturb:
        Single perturbed parent label supplied by validation.

    Returns
    -------
    set[int]
        Positional arg indices whose parent label is the perturbed layer. Missing
        or ambiguous metadata returns an empty set so callers fail closed.
    """

    if len(layers_to_perturb) != 1:
        return set()
    arg_positions = (record_member(layer, "parent_arg_positions", None) or {}).get("args", {})
    return {
        position
        for position, parent_label in arg_positions.items()
        if parent_label == layers_to_perturb[0]
    }


def _perturbed_parent_occupies_arg_slot(
    layer: Op,
    layers_to_perturb: list[str],
    slot: int,
) -> bool:
    """Return whether the perturbed parent sits anywhere inside positional ``slot``.

    ``parent_arg_positions["args"]`` keys a parent that was nested inside a
    container argument by a TUPLE path -- a real ``lstm(input, (h0, c0))``
    capture registers ``(1, 0)`` and ``(1, 1)``, never a bare ``1``. Matching
    only bare integer keys therefore misses every genuinely nested structural
    argument. This resolves the OWNING positional slot for both flat and nested
    keys, so an exemption keyed on argument POSITION keeps the reach its
    content-equality predecessor had without ever consulting tensor VALUES.

    Parameters
    ----------
    layer:
        Captured op whose parent-argument map is inspected.
    layers_to_perturb:
        Single perturbed parent label supplied by validation.
    slot:
        Positional argument index the exemption is scoped to.

    Returns
    -------
    bool
        True when the perturbed parent occupies ``slot`` or any position nested
        inside it. Missing or ambiguous metadata returns False (fail closed).
    """

    if len(layers_to_perturb) != 1:
        return False
    arg_positions = (record_member(layer, "parent_arg_positions", None) or {}).get("args", {})
    for position, parent_label in arg_positions.items():
        if parent_label != layers_to_perturb[0]:
            continue
        if position == slot:
            return True
        if isinstance(position, tuple) and position and position[0] == slot:
            return True
    return False

"""Boundary-crossing parent perturbations for ``isclose`` and integer saturation.

``isclose(input, other)`` moves only when an element enters or leaves the
tolerance band around its comparand, and an integer ``clamp``/``maximum``/
``minimum`` saturated against a large constant moves only when the parent
crosses that constant. The generic integer draw (the saved range, or ``0..10``
for a one-valued tensor), the float draw and the bool battery rarely land on
the other side, so a real edge can look insensitive. The values built here
cross the boundary by construction and are checked to change the op's result
before they are returned; the perturbation check itself is unchanged and still
requires the replayed output to differ from the saved one, so a recorded parent
that does not feed the op still fails. Same style as
``_membership_perturbation.py`` for ``isin``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.op import Op

_ISCLOSE_SLOTS = ("input", "other")
_INT64_MAX = torch.iinfo(torch.int64).max
_INT64_MIN = torch.iinfo(torch.int64).min
_ISCLOSE_DEFAULTS: tuple[tuple[str, Any], ...] = (("rtol", 1e-05), ("atol", 1e-08))


def isclose_tolerance_crossing_values(
    layer: Op, parent_label: str, parent_values: torch.Tensor
) -> torch.Tensor | None:
    """Return a perturbation of an ``isclose`` operand that flips every closeness.

    Elements inside the tolerance band move to a value well outside it, and
    elements outside move onto the comparand itself; every flip is re-checked
    with ``torch.isclose`` under the recorded ``rtol``/``atol``/``equal_nan``.

    Parameters
    ----------
    layer:
        The ``isclose`` op being replayed.
    parent_label:
        Label of the parent selected for perturbation.
    parent_values:
        The saved value of that parent.

    Returns
    -------
    torch.Tensor or None
        The crossing values, or ``None`` when the slot is unknown, an operand
        is not a real numeric tensor or scalar, the comparand does not
        broadcast to the parent's shape, or some element cannot be flipped;
        the caller then uses the generic perturbation.
    """

    slot = _isclose_slot_of(layer, parent_label)
    if slot is None or parent_values.numel() == 0 or not _is_real_numeric(parent_values):
        return None
    other = _isclose_operand(layer, 1 - slot)
    if other is None or not _is_real_numeric(other):
        return None
    rtol, atol, equal_nan = _isclose_tolerances(layer)

    def close(candidate: torch.Tensor) -> torch.Tensor:
        """Return the elementwise ``isclose`` mask of ``candidate`` against the other operand."""

        pair = (candidate, other) if slot == 0 else (other, candidate)
        return torch.isclose(*pair, rtol=rtol, atol=atol, equal_nan=equal_nan)

    with torch.no_grad():
        try:
            target = other.to(parent_values).broadcast_to(parent_values.shape)
            was_close = close(parent_values)
            away = _outside_band(target, rtol, atol)
            if away is None:
                return None
            toggled = torch.where(was_close, away, target)
            flipped = close(toggled) != was_close
        except RuntimeError:
            return None
        if flipped.shape != parent_values.shape or not bool(flipped.all()):
            return None
    return toggled


def _is_real_numeric(value: torch.Tensor) -> bool:
    """Return whether ``value`` has a real, non-boolean numeric dtype."""

    return value.dtype != torch.bool and not value.is_complex()


def _isclose_slot_of(layer: Op, parent_label: str) -> int | None:
    """Return 0 for the ``input`` operand, 1 for ``other``, else None."""

    positions = record_member(layer, "parent_arg_positions", None) or {}
    for position in (0, 1):
        if positions.get("args", {}).get(position) == parent_label:
            return position
        if positions.get("kwargs", {}).get(_ISCLOSE_SLOTS[position]) == parent_label:
            return position
    return None


def _isclose_operand(layer: Op, slot: int) -> torch.Tensor | None:
    """Return the saved operand in ``slot`` as a tensor (a Python scalar is wrapped)."""

    args = tuple(record_member(layer, "saved_args", None) or ())
    kwargs = record_member(layer, "saved_kwargs", None) or {}
    value = args[slot] if slot < len(args) else kwargs.get(_ISCLOSE_SLOTS[slot])
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return torch.tensor(value)
    return None


def _isclose_tolerances(layer: Op) -> tuple[float, float, bool]:
    """Return the recorded ``rtol``, ``atol`` and ``equal_nan`` (positional or keyword)."""

    args = tuple(record_member(layer, "saved_args", None) or ())
    kwargs = record_member(layer, "saved_kwargs", None) or {}
    values: list[Any] = []
    for index, (name, default) in enumerate(_ISCLOSE_DEFAULTS, start=2):
        values.append(args[index] if index < len(args) else kwargs.get(name, default))
    equal_nan = args[4] if len(args) > 4 else kwargs.get("equal_nan", False)
    return float(values[0]), float(values[1]), bool(equal_nan)


def _outside_band(target: torch.Tensor, rtol: float, atol: float) -> torch.Tensor | None:
    """Return ``target`` moved well outside its tolerance band, or None if unrepresentable."""

    margin = 2.0 * (atol + rtol * target.double().abs()) + 1.0
    if target.is_floating_point():
        away = target + margin.to(target.dtype)
        return away if bool(torch.isfinite(away[torch.isfinite(target)]).all()) else None
    info = torch.iinfo(target.dtype)
    shifted = target.double() + torch.ceil(margin)
    shifted = torch.where(shifted > info.max, target.double() - torch.ceil(margin), shifted)
    if bool((shifted < info.min).any()) or bool((shifted > info.max).any()):
        return None
    return shifted.to(target.dtype)


def integer_clamp_crossing_values(
    parent_values: torch.Tensor, lower: float | None, upper: float | None
) -> torch.Tensor | None:
    """Return integer clamp inputs one step across each element's clamp edge.

    Saturated elements move inside the bounds and interior elements move one
    past the upper bound (or the lower bound when only that one is set). The
    values are checked to change every clamped element and to fit the dtype.

    Parameters
    ----------
    parent_values:
        Saved integer clamp input.
    lower, upper:
        Scalar clamp bounds, either of which may be absent.

    Returns
    -------
    torch.Tensor or None
        The crossing values, or ``None`` when no bound is known or a value
        would not fit the dtype or not change the clamped output.
    """

    if lower is None and upper is None:
        return None
    lo = None if lower is None else int(lower)
    hi = None if upper is None else int(upper)
    wide = parent_values.detach().long()
    try:
        if lo is not None and hi is not None:
            saturated = (wide <= lo) | (wide >= hi)
            candidate = torch.where(saturated, (lo + hi) // 2, hi + 1)
        elif lo is not None:
            candidate = torch.where(wide <= lo, lo + 1, lo - 1)
        else:
            assert hi is not None
            candidate = torch.where(wide >= hi, hi - 1, hi + 1)
        moved = torch.clamp(candidate, lo, hi) != torch.clamp(wide, lo, hi)
    except RuntimeError:
        # A bound at the edge of int64 has no representable value beyond it.
        return None
    return _fitting(candidate, moved, parent_values)


def integer_extrema_crossing_values(
    layer: Op, parent_label: str, parent_values: torch.Tensor
) -> torch.Tensor | None:
    """Return integer ``maximum``/``minimum`` operand values past the saved result.

    One step above the saved elementwise maximum (below the minimum) wins
    every element, so the output moves wherever the edge is real.

    Returns
    -------
    torch.Tensor or None
        The crossing values, or ``None`` when the parent is not one of the two
        positional tensor operands, the result does not have the parent's
        shape, or a value would leave the dtype's range.
    """

    other_operand = _extrema_other_operand(layer, parent_label)
    if other_operand is None:
        return None
    upward = layer.func_name == "maximum"
    wide = parent_values.detach().long()
    other = other_operand.detach().to(device=wide.device).long()
    try:
        result = torch.maximum(wide, other) if upward else torch.minimum(wide, other)
    except RuntimeError:
        return None
    if result.shape != parent_values.shape or bool(
        (result == (_INT64_MAX if upward else _INT64_MIN)).any()
    ):
        return None
    candidate = result + 1 if upward else result - 1
    return _fitting(candidate, candidate != result, parent_values)


def _extrema_other_operand(layer: Op, parent_label: str) -> torch.Tensor | None:
    """Return the other positional tensor operand when the parent fills exactly one of two."""

    args = tuple(record_member(layer, "saved_args", None) or ())
    positions = (record_member(layer, "parent_arg_positions", None) or {}).get("args", {})
    slots = [index for index in (0, 1) if positions.get(index) == parent_label]
    if len(args) < 2 or len(slots) != 1:
        return None
    other = args[1 - slots[0]]
    return other if isinstance(other, torch.Tensor) else None


def _fitting(
    candidate: torch.Tensor, moved: torch.Tensor, parent_values: torch.Tensor
) -> torch.Tensor | None:
    """Return ``candidate`` in the parent's dtype when it fits and moves every element."""

    info = torch.iinfo(parent_values.dtype)
    if not bool(moved.all()) or bool((candidate < info.min).any() | (candidate > info.max).any()):
        return None
    return candidate.to(parent_values.dtype)

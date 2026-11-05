"""Boundary-crossing parent perturbations for ``isclose``, saturation and clamp bounds.

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

A tensor ``clamp`` bound (``min=``/``max=`` given as a tensor) that never binds
moves the output only once it passes the value it bounds, so the bound is moved
one step past ``max(x, lo)`` (below ``min(max(x, lo), hi)``) at every element. A float
``scaled_dot_product_attention`` mask over saturated scores gets a margin, sized to the
recomputed scores, that pushes each row's winning key below the others.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from ._sdpa_value_proofs import _sdpa_param, sdpa_softmax_rows_well_defined

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

    positions = getattr(layer, "parent_arg_positions", None) or {}
    for position in (0, 1):
        if positions.get("args", {}).get(position) == parent_label:
            return position
        if positions.get("kwargs", {}).get(_ISCLOSE_SLOTS[position]) == parent_label:
            return position
    return None


def _isclose_operand(layer: Op, slot: int) -> torch.Tensor | None:
    """Return the saved operand in ``slot`` as a tensor (a Python scalar is wrapped)."""

    args = tuple(getattr(layer, "saved_args", None) or ())
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    value = args[slot] if slot < len(args) else kwargs.get(_ISCLOSE_SLOTS[slot])
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return torch.tensor(value)
    return None


def _isclose_tolerances(layer: Op) -> tuple[float, float, bool]:
    """Return the recorded ``rtol``, ``atol`` and ``equal_nan`` (positional or keyword)."""

    args = tuple(getattr(layer, "saved_args", None) or ())
    kwargs = getattr(layer, "saved_kwargs", None) or {}
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

    args = tuple(getattr(layer, "saved_args", None) or ())
    positions = (getattr(layer, "parent_arg_positions", None) or {}).get("args", {})
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


_CLAMP_RANGE = ((1, "min", "lower"), (2, "max", "upper"))
#: ``clamp``-family spellings -> their bound slots as (position, keyword, role).
CLAMP_BOUND_SLOTS: dict[str, tuple[tuple[int, str, str], ...]] = {
    **dict.fromkeys(("clamp", "clamp_", "clip", "clip_"), _CLAMP_RANGE),
    **dict.fromkeys(("clamp_min", "clamp_min_"), ((1, "min", "lower"),)),
    **dict.fromkeys(("clamp_max", "clamp_max_"), ((1, "max", "upper"),)),
}


def clamp_bound_crossing_values(
    layer: Op, parent_label: str, parent_values: torch.Tensor
) -> torch.Tensor | None:
    """Return a tensor clamp bound moved past the value it bounds, at every element.

    ``clamp(x, lo, hi)`` is ``min(max(x, lo), hi)``. A tensor bound that never binds
    (a ``0`` floor under positive inputs, an ``inf`` ceiling) stays inactive under the
    generic draw, which samples near the bound's own values. A lower bound set one
    step above ``max(x, lo)`` (an upper bound one step below ``min(max(x, lo), hi)``)
    binds at every element, so the output moves wherever the other bound does not
    already fix it. Where the other bound fixes the output everywhere the edge is
    dead, the output stays put and the perturbation check still fails.

    Parameters
    ----------
    layer:
        Clamp-family op being replayed.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved bound tensor.

    Returns
    -------
    torch.Tensor or None
        The moved bound, or ``None`` when the parent is not exactly one tensor bound,
        the bounded input is not a real numeric tensor, the bound cannot be reduced
        from the broadcast shape, or no element can move within the dtype.
    """

    role = _clamp_bound_role(layer, parent_label)
    if role is None or not _is_real_numeric(parent_values):
        return None
    bound = parent_values.detach()
    try:
        bounded = _clamp_bounded_value(layer, role)
        if bounded is None:
            return None
        if role == "lower":
            reached = _reduce_to_shape(torch.maximum(bounded, bound), bound.shape, torch.amax)
        else:
            reached = _reduce_to_shape(torch.minimum(bounded, bound), bound.shape, torch.amin)
    except RuntimeError:
        # Bounds on another device or of a dtype the extrema kernels refuse.
        return None
    if reached is None or not _integer_step_fits(reached, role):
        return None
    step = _bound_step(reached)
    candidate = (reached + step if role == "lower" else reached - step).to(bound.dtype)
    representable = torch.isfinite(candidate) if candidate.is_floating_point() else None
    if representable is not None:
        candidate = torch.where(representable, candidate, bound)
    if torch.equal(candidate, bound):
        return None
    return candidate


def _clamp_bound_role(layer: Op, parent_label: str) -> str | None:
    """Return ``"lower"``/``"upper"`` when the parent fills exactly one clamp bound slot only."""

    slots = CLAMP_BOUND_SLOTS.get(getattr(layer, "func_name", None) or "")
    if slots is None:
        return None
    positions = getattr(layer, "parent_arg_positions", None) or {}
    arg_slots = [k for k, v in positions.get("args", {}).items() if v == parent_label]
    kwarg_slots = [k for k, v in positions.get("kwargs", {}).items() if v == parent_label]
    roles = [role for pos, key, role in slots if pos in arg_slots or key in kwarg_slots]
    if len(roles) != 1 or len(arg_slots) + len(kwarg_slots) != 1:
        return None
    return roles[0]


def _clamp_saved_slot(layer: Op, position: int, keyword: str) -> Any:
    """Return the saved clamp argument at ``position`` or ``keyword`` (``None`` when absent)."""

    args = tuple(getattr(layer, "saved_args", None) or ())
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    if keyword in kwargs:
        return kwargs[keyword]
    return args[position] if len(args) > position else None


def _clamp_bounded_value(layer: Op, role: str) -> torch.Tensor | None:
    """Return the value the perturbed bound acts on: ``x``, or ``max(x, lo)`` for an upper bound."""

    bounded = _clamp_saved_slot(layer, 0, "input")
    if not isinstance(bounded, torch.Tensor) or not _is_real_numeric(bounded):
        return None
    bounded = bounded.detach()
    lower = _clamp_saved_slot(layer, 1, "min") if role == "upper" else None
    if role == "upper" and CLAMP_BOUND_SLOTS[layer.func_name] == _CLAMP_RANGE:
        if isinstance(lower, torch.Tensor):
            return torch.maximum(bounded, lower.detach().to(bounded.device))
        if isinstance(lower, (int, float)) and not isinstance(lower, bool):
            return torch.clamp(bounded, min=lower)
    return bounded


def _reduce_to_shape(values: torch.Tensor, shape: torch.Size, reduce: Any) -> torch.Tensor | None:
    """Reduce a broadcast result back to ``shape`` with ``reduce`` over the broadcast dims."""

    lead = values.ndim - len(shape)
    if lead < 0:
        return None
    if lead:
        values = reduce(values, dim=tuple(range(lead)))
    dims = tuple(i for i, size in enumerate(shape) if size == 1 and values.shape[i] != 1)
    if dims:
        values = reduce(values, dim=dims, keepdim=True)
    return values if values.shape == shape else None


def _integer_step_fits(reached: torch.Tensor, role: str) -> bool:
    """Return whether a one-unit integer step away from ``reached`` stays in its dtype."""

    if reached.is_floating_point():
        return True
    info = torch.iinfo(reached.dtype)
    edge = info.max if role == "lower" else info.min
    return not bool((reached == edge).any())


def _bound_step(reached: torch.Tensor) -> torch.Tensor | int:
    """Return a step that moves every element of ``reached`` in its own dtype."""

    if not reached.is_floating_point():
        return 1
    return torch.clamp(reached.abs() * 0.01, min=1.0)


def sdpa_mask_crossing_values(
    layer: Op, parent_label: str, parent_values: torch.Tensor
) -> torch.Tensor | None:
    """Return an additive sdpa mask that pushes every row's winning key below the rest.

    ``scaled_dot_product_attention`` adds a float ``attn_mask`` to the scores before the
    softmax. When the scores are large (``q`` and ``k`` of norm 1e3 give scores near 1e7)
    every softmax row is one-hot, and a mask perturbation sized to the output (the generic
    draw) cannot change which key wins, so a live mask edge looked dead. Here the saved
    scores are recomputed, each row's winning key gets ``-(4 * max|score| + 1)`` added at
    the mask element that reaches it, and every other mask element keeps its value: the
    winner drops below every other key, so the output moves unless the value rows that
    win are equal. A row whose every key is some broadcast row's winner would only shift
    uniformly, so a mask whose every row is like that is left to the generic draw.

    Parameters
    ----------
    layer:
        ``scaled_dot_product_attention`` op being replayed.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved float ``attn_mask``.

    Returns
    -------
    torch.Tensor or None
        The moved mask, or ``None`` when the parent is not only the float mask, a softmax
        row is not well defined, there is one key, or the moved mask is not finite.
    """

    if getattr(layer, "func_name", None) != _SDPA_FUNC or not parent_values.is_floating_point():
        return None
    args = tuple(getattr(layer, "saved_args", None) or ())
    if parent_values.ndim == 0 or len(args) < 3 or not _sdpa_mask_only_slot(layer, parent_label):
        return None
    if not sdpa_softmax_rows_well_defined(layer, args):
        return None
    try:
        scores = _sdpa_scores(layer, args, parent_values.detach())
    except RuntimeError:
        # Shapes the recomputation cannot broadcast: leave the edge to the generic draw.
        return None
    if scores is None or scores.shape[-1] < 2 or not bool(torch.isfinite(scores).all()):
        return None
    winners = torch.zeros_like(scores).scatter_(-1, scores.argmax(dim=-1, keepdim=True), 1.0)
    marked = _reduce_to_shape(winners, parent_values.shape, torch.amax)
    if marked is None or bool((marked.amin(dim=-1) == 1).all()):
        return None
    margin = 4.0 * float(scores.abs().max()) + 1.0
    candidate = (parent_values.detach().double() - margin * marked).to(parent_values.dtype)
    return candidate if bool(torch.isfinite(candidate).all()) else None


_SDPA_FUNC = "scaled_dot_product_attention"


def _sdpa_mask_only_slot(layer: Op, parent_label: str) -> bool:
    """Return whether the parent fills the sdpa ``attn_mask`` slot and no other slot."""

    positions = getattr(layer, "parent_arg_positions", None) or {}
    arg_slots = [k for k, v in positions.get("args", {}).items() if v == parent_label]
    kwarg_slots = [k for k, v in positions.get("kwargs", {}).items() if v == parent_label]
    return arg_slots + kwarg_slots in ([3], ["attn_mask"])


def _sdpa_scores(layer: Op, args: tuple[Any, ...], mask: torch.Tensor) -> torch.Tensor | None:
    """Recompute ``q @ k^T * scale + mask`` in float64 from the saved call (grouped keys expanded)."""

    query, key = args[0].detach().double(), args[1].detach().double()
    if query.ndim >= 3 and key.ndim >= 3 and query.shape[-3] != key.shape[-3]:
        if key.shape[-3] == 0 or query.shape[-3] % key.shape[-3]:
            return None
        key = key.repeat_interleave(query.shape[-3] // key.shape[-3], dim=-3)
    scale = _sdpa_param(layer, args, "scale")
    scale = 1.0 / math.sqrt(query.shape[-1]) if scale is None else float(scale)
    return query @ key.transpose(-2, -1) * scale + mask.double().to(query.device)

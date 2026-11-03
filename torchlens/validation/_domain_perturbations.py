"""Deterministic in-domain parent perturbations for domain-sensitive ops.

Naive range sampling can leave a perturbed parent inside a region where the
child op's output cannot move (a saturated activation, the flat side of a
rounding step, a zero base under a positive exponent, a Bernoulli probability
already at its draw). ``_perturb_domain_sensitive_parent_values`` returns a
valid alternate parent value for those ops so the perturbation check keeps
its meaning; the helpers below build each family's values. Split from
``validation/core.py`` (file-cap row), which calls the entry point from its
parent perturbation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.op import Op


def _perturb_domain_sensitive_parent_values(
    layer: Op,
    parent_label: str,
    parent_values: torch.Tensor,
) -> torch.Tensor | None:
    """Return deterministic valid perturbations for domain-sensitive ops.

    Parameters
    ----------
    layer:
        Child op being replayed.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved parent tensor values.

    Returns
    -------
    torch.Tensor or None
        A valid alternate parent value for ops whose naive range sampling can
        stay inside a saturated/invalid region, otherwise ``None``.
    """

    if not parent_values.is_floating_point():
        return None
    if layer.func_name in {"__mul__", "mul"} and _output_is_all_inf(layer.out):
        return _finite_fill_distinct_from(parent_values, 0.0, 1.0)
    if layer.func_name in {"bernoulli", "bernoulli_"} and _bernoulli_probability_slot_hit(
        layer, parent_label
    ):
        complement = _bernoulli_complement_probabilities(layer, parent_values)
        if complement is not None:
            return complement
    if layer.func_name in {"__pow__", "pow"} and _pow_zero_base_exponent_parent(
        layer, parent_label
    ):
        # ``0 ** e`` is 0 for every e > 0; only an exponent of 0 or below moves it.
        return _finite_fill_distinct_from(parent_values, 0.0, -1.0)
    if not _parent_label_occupies_arg_position(layer, parent_label, 0):
        return None
    if layer.func_name == "log":
        return _finite_fill_distinct_from(parent_values, 2.0, 3.0)
    if layer.func_name == "exp":
        return _finite_fill_distinct_from(parent_values, 0.0, 1.0)
    if layer.func_name in {"__pow__", "pow"}:
        exponent = _pow_exponent(layer)
        if isinstance(exponent, (int, float)) and exponent > 0:
            return _finite_fill_distinct_from(parent_values, 0.5, 2.0)
    boundary_values = _BOUNDARY_CROSSING_VALUES.get(layer.func_name)
    if boundary_values is not None:
        return boundary_values(parent_values)
    if layer.func_name == "clamp":
        return _clamp_boundary_crossing_values(layer, parent_values)
    if layer.func_name in {"hardtanh", "hardsigmoid"}:
        return _saturated_activation_boundary_values(layer, parent_values)
    if layer.func_name in {"relu", "relu_"}:
        return _relu_boundary_crossing_values(parent_values)
    return None


def _bernoulli_has_explicit_probability(layer: Op) -> bool:
    """Return whether a ``bernoulli_`` call carries an explicit probability arg.

    ``dest.bernoulli_(p)`` supplies probabilities at positional slot 1 or the
    ``p`` keyword; bare ``x.bernoulli_()`` / ``torch.bernoulli(x)`` draw from
    the slot-0 tensor's own values.

    Parameters
    ----------
    layer:
        Captured bernoulli-family op.

    Returns
    -------
    bool
        True when an explicit probability argument is present.
    """

    if len(record_member(layer, "saved_args", None) or ()) > 1:
        return True
    return "p" in (record_member(layer, "saved_kwargs", None) or {})


def _bernoulli_probability_slot_hit(layer: Op, parent_label: str) -> bool:
    """Return whether the perturbed parent feeds the bernoulli PROBABILITY slot.

    Parameters
    ----------
    layer:
        Captured bernoulli-family op being replayed.
    parent_label:
        Parent label selected for perturbation.

    Returns
    -------
    bool
        True when the parent occupies the probability argument. Out-of-place
        ``bernoulli`` reads probabilities from slot 0; ``bernoulli_(p)`` reads
        them from slot 1 / ``p=`` (slot 0 is the overwritten destination).
        Bare ``x.bernoulli_()`` has NO probability edge at all: it fills every
        element with Bernoulli(0.5) draws and IGNORES the destination's values
        (verified empirically -- ``zeros.bernoulli_()`` produces ones), so its
        slot-0 parent is a pure template handled by the posthoc exemption.
    """

    if layer.func_name == "bernoulli_":
        if not _bernoulli_has_explicit_probability(layer):
            return False
        kwarg_positions = (record_member(layer, "parent_arg_positions", None) or {}).get(
            "kwargs", {}
        )
        return _parent_label_occupies_arg_position(layer, parent_label, 1) or (
            kwarg_positions.get("p") == parent_label
        )
    return _parent_label_occupies_arg_position(layer, parent_label, 0)


def _bernoulli_complement_probabilities(
    layer: Op,
    parent_values: torch.Tensor,
) -> torch.Tensor | None:
    """Return complement probabilities that provably flip every drawn element.

    A small in-domain probability perturbation under restored RNG replays
    IDENTICAL samples (the draw only changes where the perturbation crosses
    the resampled uniforms), so the genuine values-as-probabilities edge
    looked ``perturbation_insensitive`` and real bernoulli models could not
    validate (deephunt L17). Probabilities at the deterministic extremes
    remove the RNG from the comparison entirely: ``bernoulli(1) == 1`` and
    ``bernoulli(0) == 0`` regardless of RNG state, so feeding
    ``1 - saved_draw`` forces the replay output to differ from the saved draw
    at EVERY element when the edge is live. A genuinely dropped edge still
    replays unchanged and still fails.

    Parameters
    ----------
    layer:
        Captured bernoulli-family op being replayed.
    parent_values:
        Saved probability-parent tensor values.

    Returns
    -------
    torch.Tensor | None
        Complement-of-saved-draw probabilities, or ``None`` when the saved
        output is not an elementwise 0/1 draw of the same shape (broadcast
        probabilities keep the generic perturbation).
    """

    saved_draw = getattr(layer, "out", None)
    if not isinstance(saved_draw, torch.Tensor) or not saved_draw.is_floating_point():
        return None
    if tuple(saved_draw.shape) != tuple(parent_values.shape):
        return None
    if not _tensor_is_binary_draw(saved_draw):
        return None
    return (1.0 - saved_draw.detach()).to(parent_values.dtype)


def _tensor_is_binary_draw(value: torch.Tensor) -> bool:
    """Return whether a tensor holds only 0/1 values (a bernoulli draw shape).

    Parameters
    ----------
    value:
        Tensor to classify.

    Returns
    -------
    bool
        True when every element is exactly 0 or 1 (NaN/Inf elements fail the
        comparison, so a corrupted buffer never classifies as a draw).
    """

    if value.numel() == 0:
        return False
    return bool(torch.all((value == 0) | (value == 1)))


def _sign_boundary_crossing_values(parent_values: torch.Tensor) -> torch.Tensor:
    """Return sign-op inputs across the zero boundary.

    Parameters
    ----------
    parent_values:
        Saved sign input tensor.

    Returns
    -------
    torch.Tensor
        Candidate values on the opposite side of zero.
    """

    parent_float = torch.nan_to_num(parent_values.detach().float(), nan=0.0)
    candidate = torch.where(
        parent_float >= 0, -torch.ones_like(parent_float), torch.ones_like(parent_float)
    )
    return candidate.to(parent_values.dtype)


def _ceil_boundary_crossing_values(parent_values: torch.Tensor) -> torch.Tensor:
    """Return ceil-op inputs across the nearest integer boundary.

    Parameters
    ----------
    parent_values:
        Saved ceil input tensor.

    Returns
    -------
    torch.Tensor
        Candidate values whose ceil should differ from the saved ceil result.
    """

    parent_float = parent_values.detach().float()
    finite = torch.isfinite(parent_float)
    floored = torch.floor(torch.where(finite, parent_float, torch.zeros_like(parent_float)))
    is_integer = finite & (parent_float == floored)
    candidate = torch.where(is_integer, parent_float + 1.0, floored)
    return torch.where(finite, candidate, torch.zeros_like(parent_float)).to(parent_values.dtype)


def _floor_boundary_crossing_values(parent_values: torch.Tensor) -> torch.Tensor:
    """Return floor-op inputs across the nearest integer boundary.

    Parameters
    ----------
    parent_values:
        Saved floor input tensor.

    Returns
    -------
    torch.Tensor
        Candidate values whose floor should differ from the saved floor result.
    """

    parent_float = parent_values.detach().float()
    finite = torch.isfinite(parent_float)
    ceiled = torch.ceil(torch.where(finite, parent_float, torch.zeros_like(parent_float)))
    is_integer = finite & (parent_float == ceiled)
    candidate = torch.where(is_integer, parent_float - 1.0, ceiled)
    return torch.where(finite, candidate, torch.zeros_like(parent_float)).to(parent_values.dtype)


def _round_boundary_crossing_values(parent_values: torch.Tensor) -> torch.Tensor:
    """Return round-op inputs across a rounding boundary.

    Parameters
    ----------
    parent_values:
        Saved round input tensor.

    Returns
    -------
    torch.Tensor
        Candidate values whose rounded result should differ from the saved
        rounded result.
    """

    parent_float = parent_values.detach().float()
    finite = torch.isfinite(parent_float)
    rounded = torch.round(torch.where(finite, parent_float, torch.zeros_like(parent_float)))
    candidate = rounded + 1.0
    return torch.where(finite, candidate, torch.zeros_like(parent_float)).to(parent_values.dtype)


def _relu_boundary_crossing_values(parent_values: torch.Tensor) -> torch.Tensor:
    """Return parent values on the other side of relu's kink: zeros for positives, ones elsewhere."""

    parent_float = torch.nan_to_num(parent_values.detach().float(), nan=-1.0)
    candidate = torch.where(
        parent_float > 0, torch.zeros_like(parent_float), torch.ones_like(parent_float)
    )
    return candidate.to(parent_values.dtype)


_BOUNDARY_CROSSING_VALUES = {
    "sign": _sign_boundary_crossing_values,
    "ceil": _ceil_boundary_crossing_values,
    "floor": _floor_boundary_crossing_values,
    "round": _round_boundary_crossing_values,
}
"""Unary ops whose output moves only when the parent crosses a value boundary."""


def _clamp_boundary_crossing_values(layer: Op, parent_values: torch.Tensor) -> torch.Tensor | None:
    """Return clamp inputs across a clamp edge when scalar bounds are known.

    Parameters
    ----------
    layer:
        Clamp op being replayed.
    parent_values:
        Saved clamp input tensor.

    Returns
    -------
    torch.Tensor or None
        Boundary-crossing candidate, or ``None`` when bounds are not scalar.
    """

    lower, upper = _clamp_scalar_bounds(layer)
    if lower is None and upper is None:
        return None
    return _bounded_activation_candidate(parent_values, lower, upper)


def _saturated_activation_boundary_values(
    layer: Op,
    parent_values: torch.Tensor,
) -> torch.Tensor | None:
    """Return inputs across known saturated activation edges.

    Parameters
    ----------
    layer:
        Saturating activation op being replayed.
    parent_values:
        Saved activation input tensor.

    Returns
    -------
    torch.Tensor or None
        Boundary-crossing candidate for known scalar-bound activations.
    """

    if layer.func_name == "hardsigmoid":
        return _bounded_activation_candidate(parent_values, -3.0, 3.0)
    lower, upper = _clamp_scalar_bounds(layer)
    return _bounded_activation_candidate(parent_values, lower, upper)


def _bounded_activation_candidate(
    parent_values: torch.Tensor,
    lower: float | None,
    upper: float | None,
) -> torch.Tensor | None:
    """Return candidate values crossing scalar lower/upper saturation edges.

    Parameters
    ----------
    parent_values:
        Saved activation input tensor.
    lower:
        Lower saturation edge, if any.
    upper:
        Upper saturation edge, if any.

    Returns
    -------
    torch.Tensor or None
        Candidate tensor, or ``None`` when no finite edge can be crossed.
    """

    if lower is None and upper is None:
        return None
    parent_float = parent_values.detach().float()
    finite = torch.isfinite(parent_float)
    inside = _inside_scalar_bounds(lower, upper)
    if lower is not None and upper is not None:
        below_candidate = torch.full_like(parent_float, inside)
        above_candidate = torch.full_like(parent_float, inside)
        interior_candidate = torch.full_like(parent_float, upper + 1.0)
        candidate = torch.where(parent_float <= lower, below_candidate, interior_candidate)
        candidate = torch.where(parent_float >= upper, above_candidate, candidate)
    elif lower is not None:
        below_candidate = torch.full_like(parent_float, lower + 1.0)
        interior_candidate = torch.full_like(parent_float, lower - 1.0)
        candidate = torch.where(parent_float <= lower, below_candidate, interior_candidate)
    else:
        assert upper is not None
        above_candidate = torch.full_like(parent_float, upper - 1.0)
        interior_candidate = torch.full_like(parent_float, upper + 1.0)
        candidate = torch.where(parent_float >= upper, above_candidate, interior_candidate)
    return torch.where(finite, candidate, torch.zeros_like(parent_float)).to(parent_values.dtype)


def _inside_scalar_bounds(lower: float | None, upper: float | None) -> float:
    """Return a finite value inside scalar saturation bounds.

    Parameters
    ----------
    lower:
        Lower edge, if any.
    upper:
        Upper edge, if any.

    Returns
    -------
    float
        A value inside the bounded region.
    """

    if lower is not None and upper is not None:
        return (lower + upper) / 2.0
    if lower is not None:
        return lower + 1.0
    if upper is not None:
        return upper - 1.0
    return 0.0


def _clamp_scalar_bounds(layer: Op) -> tuple[float | None, float | None]:
    """Return scalar clamp-style bounds from saved args/kwargs.

    Parameters
    ----------
    layer:
        Clamp-like op.

    Returns
    -------
    tuple[float or None, float or None]
        Lower and upper scalar bounds when available.
    """

    saved_args = cast(Sequence[Any], record_member(layer, "saved_args", ()) or ())
    saved_kwargs = record_member(layer, "saved_kwargs", {}) or {}
    lower = saved_kwargs.get("min", saved_kwargs.get("min_val"))
    upper = saved_kwargs.get("max", saved_kwargs.get("max_val"))
    if lower is None and len(saved_args) > 1:
        lower = saved_args[1]
    if upper is None and len(saved_args) > 2:
        upper = saved_args[2]
    return _scalar_float_or_none(lower), _scalar_float_or_none(upper)


def _scalar_float_or_none(value: Any) -> float | None:
    """Return ``value`` as a finite scalar float when possible.

    Parameters
    ----------
    value:
        Candidate scalar value.

    Returns
    -------
    float or None
        Finite scalar float, otherwise ``None``.
    """

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    if not isinstance(value, (int, float)):
        return None
    value_float = float(value)
    if not torch.isfinite(torch.tensor(value_float)):
        return None
    return value_float


def _output_is_all_inf(output: Any) -> bool:
    """Return whether ``output`` is a non-empty all-inf tensor.

    Parameters
    ----------
    output:
        Candidate operation output.

    Returns
    -------
    bool
        True when ``output`` is a tensor whose elements are all infinite.
    """

    return (
        isinstance(output, torch.Tensor)
        and output.numel() > 0
        and bool(torch.all(torch.isinf(output)).item())
    )


def _finite_fill_distinct_from(
    tensor: torch.Tensor,
    primary: float,
    fallback: float,
) -> torch.Tensor:
    """Return a finite fill tensor distinct from ``tensor`` when possible.

    Parameters
    ----------
    tensor:
        Tensor whose dtype/device/shape should be preserved.
    primary:
        Preferred fill value.
    fallback:
        Alternate fill value when the primary exactly matches ``tensor``.

    Returns
    -------
    torch.Tensor
        Filled tensor with the same dtype/device/shape as ``tensor``.
    """

    candidate = torch.full_like(tensor, primary)
    if torch.equal(candidate, tensor):
        return torch.full_like(tensor, fallback)
    return candidate


def _pow_zero_base_exponent_parent(layer: Op, parent_label: str) -> bool:
    """Return whether ``parent_label`` is only the exponent of a power whose saved base is zero."""

    saved_args = cast(Sequence[Any], record_member(layer, "saved_args", ()) or ())
    if len(saved_args) < 2 or _parent_label_occupies_arg_position(layer, parent_label, 0):
        return False
    if not _parent_label_occupies_arg_position(layer, parent_label, 1):
        return False
    base = saved_args[0]
    if isinstance(base, torch.Tensor):
        return base.numel() > 0 and bool(torch.all(torch.eq(base, 0)).item())
    return isinstance(base, (int, float)) and not isinstance(base, bool) and base == 0


def _pow_exponent(layer: Op) -> Any:
    """Return a captured exponent operand for ``pow``-style ops.

    Parameters
    ----------
    layer:
        Candidate power op.

    Returns
    -------
    Any
        Positional or keyword exponent value, if captured.
    """

    saved_args = cast(Sequence[Any], record_member(layer, "saved_args", ()) or ())
    if len(saved_args) > 1:
        return saved_args[1]
    return (record_member(layer, "saved_kwargs", {}) or {}).get("exponent")


def _parent_label_occupies_arg_position(layer: Op, parent_label: str, position: int) -> bool:
    """Return whether a parent label occupies a positional argument.

    Parameters
    ----------
    layer:
        Op whose parent-argument map should be inspected.
    parent_label:
        Parent label to find.
    position:
        Positional argument index.

    Returns
    -------
    bool
        True when ``parent_label`` is registered at ``args[position]``.
    """

    return (record_member(layer, "parent_arg_positions", None) or {}).get("args", {}).get(
        position
    ) == parent_label

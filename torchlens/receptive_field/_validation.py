"""Exact gradient-inside-geometric receptive-field validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

import torch

from ..backends import BackendUnsupportedError
from . import _engine
from ._errors import ReceptiveFieldError
from ._gradient import gradient_for_unit
from ._types import (
    GradientReceptiveField,
    ReceptiveField,
    ReceptiveFieldBox,
    ReceptiveFieldBoxAxis,
    ReceptiveFieldStatus,
    ReceptiveFieldValidation,
    ReceptiveFieldValidationStatus,
    ReceptiveFieldViolation,
)
from ._view import ReceptiveFieldView


if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


_TAINTED_STATUSES = {
    ReceptiveFieldStatus.DATA_DEPENDENT,
    ReceptiveFieldStatus.UNKNOWN,
    ReceptiveFieldStatus.UNSUPPORTED,
}
_MAX_LISTED_VIOLATIONS = 32


def _normalize_complete_unit(target: Op, unit: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize a complete target output-element index.

    Parameters
    ----------
    target:
        Target operation.
    unit:
        Complete output index.

    Returns
    -------
    tuple[int, ...]
        Validated output index.
    """

    normalized = tuple(unit)
    shape = tuple(int(extent) for extent in target.shape)
    if len(normalized) != len(shape):
        raise ReceptiveFieldError(
            f"unit for {target.label!r} requires {len(shape)} coordinates; got {len(normalized)}."
        )
    for coordinate, extent in zip(normalized, shape, strict=True):
        if isinstance(coordinate, bool) or not isinstance(coordinate, int):
            raise ReceptiveFieldError("unit coordinates must be integers.")
        if coordinate < 0 or coordinate >= extent:
            raise ReceptiveFieldError(
                f"unit coordinate {coordinate} is out of bounds for extent {extent}."
            )
    return normalized


def _selected_descriptors(
    view: ReceptiveFieldView, selected_input: Op | str | None
) -> Mapping[str, ReceptiveField]:
    """Select the geometric descriptors participating in one check.

    Parameters
    ----------
    view:
        Target operation receptive-field view.
    selected_input:
        Optional graph input or exact IO role.

    Returns
    -------
    collections.abc.Mapping
        Selected descriptors keyed by IO role.
    """

    if selected_input is None:
        return view.per_input
    descriptor = view._descriptor(selected_input)
    return MappingProxyType({descriptor.io_role: descriptor})


def _box_for_descriptor(
    view: ReceptiveFieldView,
    descriptor: ReceptiveField,
    complete_unit: tuple[int, ...],
) -> ReceptiveFieldBox:
    """Resolve a geometric box from a complete output index.

    Parameters
    ----------
    view:
        Target operation receptive-field view.
    descriptor:
        Eligible descriptor for one input role.
    complete_unit:
        Complete target output index.

    Returns
    -------
    ReceptiveFieldBox
        Clipped geometric box for containment checking.
    """

    assert descriptor.axes is not None
    windowed = tuple(
        complete_unit[cast(int, axis.output_axis)]
        for axis in descriptor.axes
        if axis.kind == "windowed"
    )
    if windowed:
        return view.at(windowed, input=descriptor.io_role, clip=True)
    axes = tuple(
        ReceptiveFieldBoxAxis(
            input_axis=axis.input_axis,
            kind=axis.kind,
            theoretical_start=None,
            theoretical_stop=None,
            index_start=0 if axis.kind == "full" else None,
            index_stop=axis.input_extent if axis.kind == "full" else None,
            clipped_start=0 if axis.kind == "full" else None,
            clipped_stop=axis.input_extent if axis.kind == "full" else None,
        )
        for axis in descriptor.axes
    )
    return ReceptiveFieldBox(
        op_label=descriptor.op_label,
        io_role=descriptor.io_role,
        unit=(),
        axes=axes,
        input_shape=descriptor.input_shape,
        status=descriptor.status,
        exact=all(axis.exact for axis in descriptor.axes),
        clipped=False,
        empty=False,
        covers_input=all(axis.kind in {"pointwise", "full"} for axis in axes),
    )


def _allowed_bounds(
    descriptor: ReceptiveField,
    box: ReceptiveFieldBox,
    complete_unit: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Return exact clipped integer bounds on every input axis.

    Parameters
    ----------
    descriptor:
        Geometric descriptor carrying pointwise output-axis mappings.
    box:
        Concrete clipped geometric box.
    complete_unit:
        Complete target output index.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Half-open allowed interval per input axis.
    """

    assert descriptor.axes is not None
    bounds: list[tuple[int, int]] = []
    for axis_descriptor, box_axis in zip(descriptor.axes, box.axes, strict=True):
        if box_axis.kind == "pointwise":
            if axis_descriptor.output_axis is None:
                raise ReceptiveFieldError(
                    f"Pointwise input axis {axis_descriptor.input_axis} has no output mapping."
                )
            coordinate = complete_unit[axis_descriptor.output_axis]
            bounds.append((coordinate, coordinate + 1))
        elif box_axis.clipped_start is None or box_axis.clipped_stop is None:
            raise ReceptiveFieldError(
                f"Geometric bounds are unavailable on input axis {box_axis.input_axis}."
            )
        else:
            bounds.append((box_axis.clipped_start, box_axis.clipped_stop))
    return tuple(bounds)


def _role_violations(
    gradient: GradientReceptiveField,
    descriptor: ReceptiveField,
    box: ReceptiveFieldBox,
    complete_unit: tuple[int, ...],
) -> tuple[list[ReceptiveFieldViolation], int, tuple[int, ...], tuple[int, ...]]:
    """Check one role's complete support mask against exact integer bounds.

    Parameters
    ----------
    gradient:
        Empirical full-input influence set.
    descriptor:
        Eligible geometric descriptor.
    box:
        Concrete clipped geometric box.
    complete_unit:
        Complete target output index.

    Returns
    -------
    tuple
        Listed violations, exact count, per-axis excess, and per-axis slack.
    """

    bounds = _allowed_bounds(descriptor, box, complete_unit)
    if tuple(gradient.support_mask.shape) != descriptor.input_shape:
        raise ReceptiveFieldError(
            f"Gradient support shape {tuple(gradient.support_mask.shape)} does not match "
            f"geometric input shape {descriptor.input_shape}."
        )
    supported = torch.nonzero(gradient.support_mask, as_tuple=False)
    excess = [0] * len(bounds)
    support_min = [stop for _, stop in bounds]
    support_max = [start - 1 for start, _ in bounds]
    listed: list[ReceptiveFieldViolation] = []
    count = 0
    for row in supported.tolist():
        index = tuple(int(coordinate) for coordinate in row)
        outside = False
        for axis, (coordinate, (start, stop)) in enumerate(zip(index, bounds, strict=True)):
            support_min[axis] = min(support_min[axis], coordinate)
            support_max[axis] = max(support_max[axis], coordinate)
            axis_excess = max(start - coordinate, coordinate - stop + 1, 0)
            excess[axis] = max(excess[axis], axis_excess)
            outside = outside or axis_excess > 0
        if outside:
            count += 1
            if len(listed) < _MAX_LISTED_VIOLATIONS:
                listed.append(
                    ReceptiveFieldViolation(
                        io_role=gradient.io_role,
                        index=index,
                        magnitude=float(gradient.grad[index].item()),
                        box=box,
                        reason="gradient support index lies outside the clipped geometric box",
                    )
                )
    slack = tuple(
        (stop - start)
        - (0 if maximum < minimum else min(maximum, stop - 1) - max(minimum, start) + 1)
        for (start, stop), minimum, maximum in zip(bounds, support_min, support_max, strict=True)
    )
    return listed, count, tuple(excess), slack


def _merge_diagnostic_rows(
    rows: Sequence[tuple[int, ...]], *, maximum: bool
) -> tuple[int, ...] | None:
    """Merge same-rank diagnostic rows across selected input roles.

    Parameters
    ----------
    rows:
        Per-role diagnostic tuples.
    maximum:
        Whether to select maxima instead of minima.

    Returns
    -------
    tuple[int, ...] or None
        Merged diagnostics, or ``None`` when ranks differ or no rows exist.
    """

    if not rows or len({len(row) for row in rows}) != 1:
        return None
    reducer = max if maximum else min
    return tuple(reducer(values) for values in zip(*rows, strict=True))


def _validation_result(
    *,
    target: Op,
    unit: tuple[int, ...],
    descriptors: Mapping[str, ReceptiveField],
    geometric: dict[str, ReceptiveFieldBox],
    gradients: Mapping[str, GradientReceptiveField],
    gradient_error: str | None,
) -> ReceptiveFieldValidation:
    """Assemble the tri-state result and its exact diagnostics.

    Parameters
    ----------
    target, unit:
        Validated target operation and complete output index.
    descriptors:
        Selected descriptors keyed by IO role.
    geometric, gradients:
        Successfully materialized geometric and empirical results.
    gradient_error:
        Gradient unavailability diagnostic, when probing failed.

    Returns
    -------
    ReceptiveFieldValidation
        Immutable tri-state validation result.
    """

    listed: list[ReceptiveFieldViolation] = []
    n_violations = 0
    excess_rows: list[tuple[int, ...]] = []
    slack_rows: list[tuple[int, ...]] = []
    indeterminate_roles: list[str] = []
    geometric_batch = False
    undeclared_batch = False
    nonfinite = False
    for role, descriptor in descriptors.items():
        gradient = gradients.get(role)
        if descriptor.status in _TAINTED_STATUSES or descriptor.axes is None:
            indeterminate_roles.append(role)
            nonfinite = nonfinite or (gradient is not None and gradient.nonfinite_count > 0)
            continue
        geometric_batch = geometric_batch or descriptor.batch_coupled
        box = geometric.get(role)
        if gradient is None or box is None:
            indeterminate_roles.append(role)
            continue
        nonfinite = nonfinite or gradient.nonfinite_count > 0
        undeclared_batch = undeclared_batch or (
            gradient.cross_batch_influence and not descriptor.batch_coupled
        )
        role_listed, role_count, role_excess, role_slack = _role_violations(
            gradient, descriptor, box, unit
        )
        listed.extend(role_listed[: _MAX_LISTED_VIOLATIONS - len(listed)])
        n_violations += role_count
        excess_rows.append(role_excess)
        slack_rows.append(role_slack)

    cross_batch: Literal["none", "geometric", "undeclared"]
    if undeclared_batch:
        cross_batch = "undeclared"
    elif geometric_batch:
        cross_batch = "geometric"
    else:
        cross_batch = "none"
    if n_violations or undeclared_batch:
        status = ReceptiveFieldValidationStatus.FAIL
        message = (
            f"Receptive-field validation failed for {target.label!r} at {unit}: "
            f"{n_violations} gradient-support indices lie outside geometric bounds."
        )
        if undeclared_batch:
            message += " Cross-batch influence was not declared geometrically."
        if listed:
            first = listed[0]
            descriptor = descriptors[first.io_role]
            bounds = tuple(
                (axis.clipped_start, axis.clipped_stop)
                if axis.kind != "pointwise"
                else ("same-index", "same-index")
                for axis in cast(ReceptiveFieldBox, first.box).axes
            )
            message += (
                f" First violation: role={first.io_role!r}, index={first.index}, "
                f"magnitude={first.magnitude}, rule={descriptor.rule!r}, bounds={bounds}."
            )
        per_axis_excess = _merge_diagnostic_rows(excess_rows, maximum=True)
        slack_per_axis = None
    elif gradient_error is not None or indeterminate_roles or nonfinite:
        status = ReceptiveFieldValidationStatus.INDETERMINATE
        details = gradient_error or (
            "non-finite gradient contamination"
            if nonfinite
            else f"ineligible geometry for roles {', '.join(indeterminate_roles)}"
        )
        message = f"Receptive-field validation is indeterminate for {target.label!r}: {details}."
        per_axis_excess = None
        slack_per_axis = None
    else:
        status = ReceptiveFieldValidationStatus.PASS
        message = (
            f"All gradient-support indices for {target.label!r} at {unit} are inside the "
            "clipped geometric boxes."
        )
        per_axis_excess = None
        slack_per_axis = _merge_diagnostic_rows(slack_rows, maximum=False)
    return ReceptiveFieldValidation(
        status=status,
        op_label=target.label,
        unit=unit,
        geometric=MappingProxyType(geometric),
        gradient=MappingProxyType(dict(gradients)),
        violations=tuple(listed),
        n_violations=n_violations,
        per_axis_excess=per_axis_excess,
        slack_per_axis=slack_per_axis,
        cross_batch=cross_batch,
        checked_input_roles=tuple(descriptors),
        message=message,
    )


def _check_for_unit(
    target: Op,
    unit: Sequence[int],
    *,
    input: Op | str | None,
    atol: float,
    rtol: float,
    retain_graph: bool,
) -> ReceptiveFieldValidation:
    """Run one validation with explicit graph-retention control."""

    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative.")
    complete_unit = _normalize_complete_unit(target, unit)
    solution = _engine.solve(target.source_trace)
    view = ReceptiveFieldView(target, solution)
    descriptors = _selected_descriptors(view, input)
    geometric: dict[str, ReceptiveFieldBox] = {}
    for role, descriptor in descriptors.items():
        if descriptor.status not in _TAINTED_STATUSES and descriptor.axes is not None:
            geometric[role] = _box_for_descriptor(view, descriptor, complete_unit)
    gradient_error: str | None = None
    gradients: Mapping[str, GradientReceptiveField]
    try:
        probed = gradient_for_unit(
            target,
            complete_unit,
            input=input,
            atol=atol,
            rtol=rtol,
            retain_graph=retain_graph,
        )
        if isinstance(probed, GradientReceptiveField):
            gradients = MappingProxyType({probed.io_role: probed})
        else:
            gradients = probed
    except (BackendUnsupportedError, ReceptiveFieldError) as exc:
        gradients = MappingProxyType({})
        gradient_error = str(exc)
    return _validation_result(
        target=target,
        unit=complete_unit,
        descriptors=descriptors,
        geometric=geometric,
        gradients=gradients,
        gradient_error=gradient_error,
    )


def check_for_unit(
    target: Op,
    unit: Sequence[int],
    *,
    input: Op | str | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> ReceptiveFieldValidation:
    """Cross-check one target unit for the entity-view delegation surface.

    Parameters
    ----------
    target, unit:
        Target operation and complete output-element index.
    input:
        Optional model-input operation or exact IO role.
    atol, rtol:
        Gradient support thresholds; these never relax geometric containment.

    Returns
    -------
    ReceptiveFieldValidation
        Tri-state exact-containment result.
    """

    return _check_for_unit(target, unit, input=input, atol=atol, rtol=rtol, retain_graph=False)


def check(
    view: ReceptiveFieldView,
    unit: Sequence[int],
    *,
    input: Op | str | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> ReceptiveFieldValidation:
    """Cross-check geometry and gradients through an explicit RF view.

    Parameters
    ----------
    view, unit:
        Receptive-field view and complete output-element index.
    input:
        Optional model-input operation or exact IO role.
    atol, rtol:
        Gradient support thresholds; these never relax geometric containment.

    Returns
    -------
    ReceptiveFieldValidation
        Tri-state exact-containment result.
    """

    return check_for_unit(view._op, unit, input=input, atol=atol, rtol=rtol)


def _resolve_ops(trace: Trace, ops: Sequence[Op | str] | None) -> tuple[Op, ...]:
    """Resolve explicit operations or the eligible default sweep."""

    if ops is not None:
        resolved: list[Op] = []
        for selected in ops:
            if not isinstance(selected, str):
                resolved.append(selected)
                continue
            matches = [op for op in trace.layer_list if selected in {op.label, op.layer_label}]
            if len(matches) != 1:
                raise ReceptiveFieldError(
                    f"Operation selector {selected!r} resolved to {len(matches)} operations."
                )
            resolved.append(matches[0])
        return tuple(resolved)
    solution = _engine.solve(trace)
    return tuple(
        op
        for op in trace.layer_list
        if not op.is_input
        and not op.is_output
        and op.has_saved_activation
        and any(
            descriptor.status not in _TAINTED_STATUSES
            for descriptor in solution.per_op.get(op.label, {}).values()
        )
    )


def _preset_units(
    view: ReceptiveFieldView,
    preset: Literal["center", "corners"],
    batch_index: int,
    selected_input: Op | str | None,
) -> tuple[tuple[int, ...], ...]:
    """Resolve a named unit preset to complete output indices."""

    center = view.center_unit(batch_index=batch_index, input=selected_input)
    if preset == "center":
        return (center,)
    descriptor = view._descriptor(selected_input)
    assert descriptor.axes is not None
    windowed_output_axes = tuple(
        cast(int, axis.output_axis) for axis in descriptor.axes if axis.kind == "windowed"
    )
    choices = tuple((0, int(view._op.shape[axis]) - 1) for axis in windowed_output_axes)
    units: list[tuple[int, ...]] = []
    for corner in product(*choices):
        resolved = list(center)
        for axis, coordinate in zip(windowed_output_axes, corner, strict=True):
            resolved[axis] = coordinate
        units.append(tuple(resolved))
    return tuple(dict.fromkeys(units)) or (center,)


def cross_validate(
    trace: Trace,
    *,
    ops: Sequence[Op | str] | None = None,
    units: Literal["center", "corners"] | Sequence[int] | Sequence[Sequence[int]] = "center",
    batch_index: int = 0,
    inputs: Op | str | Sequence[Op | str] | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    raise_on_failure: bool = False,
) -> list[ReceptiveFieldValidation]:
    """Sweep exact gradient-inside-geometric checks over a captured trace.

    Parameters
    ----------
    trace:
        Backward-ready captured trace.
    ops:
        Explicit operations or labels. ``None`` selects saved eligible compute ops.
    units:
        ``"center"``, ``"corners"``, one complete index, or complete indices.
    batch_index:
        Explicit seeded batch index used by named presets.
    inputs:
        Optional input selector or selectors.
    atol, rtol:
        Gradient support thresholds; these never relax geometric containment.
    raise_on_failure:
        Raise on ``FAIL`` only.

    Returns
    -------
    list[ReceptiveFieldValidation]
        One tri-state result per operation, unit, and explicit input selector.
    """

    if atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be non-negative.")
    targets = _resolve_ops(trace, ops)
    selected_inputs: tuple[Op | str | None, ...]
    if inputs is None or isinstance(inputs, (str,)) or hasattr(inputs, "source_trace"):
        selected_inputs = (cast("Op | str | None", inputs),)
    else:
        selected_inputs = tuple(inputs)
    results: list[ReceptiveFieldValidation] = []
    for target in targets:
        view = ReceptiveFieldView(target, _engine.solve(trace))
        for selected_input in selected_inputs:
            if isinstance(units, str):
                resolved_units = _preset_units(view, units, batch_index, selected_input)
            elif units and all(
                isinstance(value, int) and not isinstance(value, bool) for value in units
            ):
                resolved_units = (tuple(cast("Sequence[int]", units)),)
            else:
                resolved_units = tuple(
                    tuple(unit) for unit in cast("Sequence[Sequence[int]]", units)
                )
            for unit in resolved_units:
                result = _check_for_unit(
                    target,
                    unit,
                    input=selected_input,
                    atol=atol,
                    rtol=rtol,
                    retain_graph=True,
                )
                if raise_on_failure:
                    result.assert_valid()
                results.append(result)
    return results


__all__ = ["check", "check_for_unit", "cross_validate"]

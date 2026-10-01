"""Exact gradient-inside-geometric receptive-field validation.

Oracle scope (R74/75-7): the empirical gradients probed here flow through the
autograd graph recorded during the wrapped capture forward — the same root the
geometric solution was derived from. The cross-check is independent of the
geometric derivation (TL indexing/sampling/rule bugs fail it) but is
structurally blind to capture-time forward corruption, which would shift both
the geometry and the gradients together. Capture fidelity belongs to the
``torchlens.validation`` replay tripwire, not to this module; see
:func:`torchlens.receptive_field.verify` for the user-facing statement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from ..backends import BackendUnsupportedError
from . import _engine
from ._engine_forward import solve_projective
from ._errors import (
    ReceptiveFieldConfigurationError,
    ReceptiveFieldError,
    ReceptiveFieldUnavailableError,
)
from ._forward_query import box_for_source_unit
from ._gradient import gradient_for_unit
from ._gradient_forward import _select_targets, projective_gradient_for_unit
from ._path import resolve_graph_point
from ._query import _windowed_axes_in_unit_order, box_for_unit
from ._types import (
    GradientReceptiveField,
    ReceptiveField,
    ReceptiveFieldAlignment,
    ReceptiveFieldBox,
    ReceptiveFieldBoxAxis,
    ReceptiveFieldDirection,
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


def _refuse_poisoned_rf_verification(trace: Trace) -> None:
    """Refuse a poison-marked trace at an RF verdict-producing surface.

    Parameters
    ----------
    trace:
        Trace whose geometry/gradient agreement would otherwise be reported as
        a receptive-field validation verdict.
    """

    from ..runnable import refuse_poisoned_trace

    refuse_poisoned_trace(trace, "receptive field verification")


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
    solution: object,
    owner: Op,
    descriptor: ReceptiveField,
    complete_unit: tuple[int, ...],
    *,
    direction: ReceptiveFieldDirection,
    endpoint: Op | None,
) -> ReceptiveFieldBox:
    """Resolve a geometric box from a complete output index.

    Parameters
    ----------
    solution:
        Direction-specific geometric solution.
    owner:
        Operation owning the seeded unit.
    descriptor:
        Eligible descriptor for one input role.
    complete_unit:
        Complete seed-owner output index.
    direction:
        Influence direction selecting the geometric walker.
    endpoint:
        Explicit opposite endpoint, when one was supplied.

    Returns
    -------
    ReceptiveFieldBox
        Clipped geometric box for containment checking.
    """

    assert descriptor.axes is not None
    # Windowed unit coordinates are ordered by ascending output axis (the
    # seed operation's own grid order), not descriptor (input-axis) order;
    # the two differ whenever an axis permutation separates the endpoints.
    windowed = tuple(
        complete_unit[cast(int, axis.output_axis)]
        for axis in _windowed_axes_in_unit_order(descriptor)
    )
    if windowed:
        if direction is ReceptiveFieldDirection.RECEPTIVE:
            return box_for_unit(
                cast(Any, solution),
                owner,
                windowed,
                input=None if endpoint is not None else descriptor.io_role,
                source=endpoint,
                clip=True,
                complete_unit=complete_unit,
            )
        return box_for_source_unit(
            cast(Any, solution),
            owner,
            windowed,
            target=descriptor.io_role,
            clip=True,
            complete_unit=complete_unit,
        )
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
            sparse_possible=axis.sparse_possible,
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
        direction=direction,
        # ``shape`` is legitimately ``None`` for a non-tensor-valued op (see
        # the identical note in ``_indeterminate_unit``).
        unit_shape=tuple(owner.shape) if owner.shape is not None else (),
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
    if box.empty:
        return tuple((0, 0) for _ in descriptor.axes)
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


_ADJOINT_PROBE_RADIUS = 2


def _adjoint_probe_units(
    owner: Op,
    descriptors: Mapping[str, ReceptiveField],
    complete_unit: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return the sampled unit plus a small windowed-axis neighborhood.

    Exact-box corners of an HONEST box are always true influence members, so
    the corner cross-check on the sampled unit alone can never expose a
    spurious-nonempty claim: the lie lives at units OFF the true lattice
    (for example the odd sources of a step-2 slice). Probing each windowed
    output axis at ±1 and ±2 around the sampled unit covers the residue
    classes of the common stride-2/3 lattices at bounded rational-query cost.
    """

    axes: list[int] = []
    for descriptor in descriptors.values():
        for axis in descriptor.axes or ():
            if axis.kind == "windowed" and axis.output_axis is not None:
                axes.append(axis.output_axis)
    units: dict[tuple[int, ...], None] = {complete_unit: None}
    for output_axis in dict.fromkeys(axes):
        if output_axis >= len(complete_unit):
            continue
        extent = int(owner.shape[output_axis])
        for offset in range(-_ADJOINT_PROBE_RADIUS, _ADJOINT_PROBE_RADIUS + 1):
            coordinate = complete_unit[output_axis] + offset
            if 0 <= coordinate < extent:
                probe = list(complete_unit)
                probe[output_axis] = coordinate
                units[tuple(probe)] = None
    return tuple(units)


def _box_membership_contains(box: ReceptiveFieldBox, unit: tuple[int, ...]) -> bool:
    """Return whether a reverse-direction box contains one complete unit.

    Any axis with concrete clipped bounds must contain the coordinate;
    pointwise and unbounded axes are treated as containing, so an upper-bound
    or partially-unbounded reverse box can never produce a false violation.
    Non-windowed axes joined the comparison in the disputed-r2 b6/R20-3
    strengthening: walk-narrowed full axes (for example a scalar-selected
    getitem axis) carry sound over-approximate bounds, so a coordinate outside
    them is truly outside the influence set — the historical windowed-only
    test made over-approximated non-windowed axes tripwire-blind.
    """

    if box.empty:
        return False
    for axis in box.axes:
        if axis.kind not in {"windowed", "full"}:
            continue
        if axis.clipped_start is None or axis.clipped_stop is None:
            continue
        coordinate = unit[axis.input_axis]
        if not axis.clipped_start <= coordinate < axis.clipped_stop:
            return False
    return True


_ADJOINT_CORNER_BUDGET = 16


def _complete_corner_choices(
    descriptor: ReceptiveField,
    box: ReceptiveFieldBox,
    complete_unit: tuple[int, ...],
) -> tuple[tuple[int, ...], ...] | None:
    """Return per-axis claimed source coordinates for complete corner seeding.

    Every axis must contribute at least one CLAIMED member: windowed and
    concrete-bounded full axes their hull endpoints, pointwise axes the seed
    unit's own coordinate (the exact element the same-index claim asserts).
    ``None`` means some axis has no representable claimed coordinate and the
    caller falls back to the historical windowed-only enumeration. Full-axis
    endpoint pairs widen inside a fixed corner budget, narrowest claims
    first, so wide honest full axes (channel mixing) cannot blow up the probe
    count; a budget-trimmed axis still probes its start coordinate.
    """

    assert descriptor.axes is not None
    choices: list[tuple[int, ...]] = []
    widenable: list[tuple[int, int, int]] = []
    count = 1
    for position, (axis_descriptor, axis) in enumerate(zip(descriptor.axes, box.axes, strict=True)):
        axis_choices: tuple[int, ...]
        if axis.kind in {"windowed", "full"}:
            if (
                axis.clipped_start is None
                or axis.clipped_stop is None
                or axis.clipped_stop <= axis.clipped_start
            ):
                return None
            start, last = axis.clipped_start, axis.clipped_stop - 1
            if axis.kind == "windowed":
                axis_choices = (start, last) if last != start else (start,)
            else:
                axis_choices = (start,)
                if last != start:
                    widenable.append((last - start, position, last))
        elif axis.kind == "pointwise":
            output_axis = axis_descriptor.output_axis
            if output_axis is None or output_axis >= len(complete_unit):
                return None
            coordinate = complete_unit[output_axis]
            if not 0 <= coordinate < axis_descriptor.input_extent:
                return None
            axis_choices = (coordinate,)
        else:
            return None
        choices.append(axis_choices)
        count *= len(axis_choices)
    if count > _ADJOINT_CORNER_BUDGET:
        return None
    for _width, position, last in sorted(widenable):
        if count * 2 > _ADJOINT_CORNER_BUDGET:
            break
        choices[position] = (choices[position][0], last)
        count *= 2
    return tuple(choices)


def _exact_box_adjoint_violations(
    owner: Op,
    complete_unit: tuple[int, ...],
    descriptor: ReceptiveField,
    box: ReceptiveFieldBox,
    direction: ReceptiveFieldDirection,
) -> tuple[ReceptiveFieldViolation, ...]:
    """Cross-check an exact nonempty box's corners through the opposite engine.

    An exact hull's per-axis endpoints are true members of the claimed
    influence set, and influence is symmetric: a source element is in the
    receptive field of a target unit exactly when the target unit is in the
    source's projective field. Every corner of an exact claimed-nonempty box
    must therefore land in a SOUND opposite-direction box of that corner.
    A violated corner proves the two directions inconsistent — at least one
    claim is wrong — and fails validation without needing autograd. This
    catches spurious-nonempty exact claims that pure gradient containment is
    structurally unable to see (an empty true support is contained in any
    box) ONLY when the two directions disagree: both engines consult the same
    rule registry, so a rule that overclaims symmetrically in both directions
    passes this consistency check. It is a direction-coherence oracle, not a
    tightness oracle; tightness of the built-in rules is enforced by the
    saturating-model slack battery in tests. Reverse queries that refuse with
    a typed error are skipped: the check only ever adds failure power.

    Non-windowed axes join the corner enumeration (grind-p3 T7): when every
    axis of the box yields a claimed source coordinate — windowed hull
    endpoints, concrete-bounded FULL-axis endpoints, and the POINTWISE
    same-index coordinate — corners are seeded as COMPLETE far-grid units
    (``complete_unit=``), so an exact claim over a full or pointwise axis is
    actually probed instead of silently trusted. The historical windowed-only
    enumeration was tripwire-blind to asymmetric walk bugs on non-windowed
    axes (for example a receptive walk serving the full parent extent on an
    int-selected getitem axis while the projective walk correctly prunes
    off-index sources). Complete seeding asserts PRODUCT membership, which
    the documented EXACT contract (per-axis integer hulls) only implies for
    MERGE-FREE descriptors: along a single chain every axis maps
    independently, so the support is a product set and each joint corner is
    a true member. A merged (union) influence set attains its per-axis hull
    endpoints only axis-wise — the channel-cat projective union is the
    canonical case — so merged descriptors (``alignment`` other than
    ``NOT_APPLICABLE``, the enforced ``merge_seen`` mirror) keep the
    historical windowed-only enumeration and their non-windowed conservatism
    remains a disclosed residual. Boxes with no windowed axes remain
    unprobed — the reverse per-unit query refuses them typed — and symmetric
    two-direction overclaims stay out of scope as documented above.
    """

    if not box.exact or box.empty:
        return ()
    assert descriptor.axes is not None
    windowed_bounds: list[tuple[int, int]] = []
    for axis in box.axes:
        if (
            axis.kind == "windowed"
            and axis.clipped_start is not None
            and axis.clipped_stop is not None
            and axis.clipped_stop > axis.clipped_start
        ):
            windowed_bounds.append((axis.clipped_start, axis.clipped_stop))
    if not windowed_bounds or len(windowed_bounds) > 3:
        return ()
    trace = owner.source_trace
    far = next((op for op in trace.layer_list if op.label == descriptor.input_op_label), None)
    if far is None:
        return ()

    def _reverse_confirms(corner: tuple[int, ...], far_unit: tuple[int, ...] | None) -> bool | None:
        """Probe one claimed member; ``None`` means the reverse query refused."""

        reverse_solution: Any
        try:
            if direction is ReceptiveFieldDirection.RECEPTIVE:
                reverse_solution = solve_projective(trace, (owner,))
                reverse = box_for_source_unit(
                    reverse_solution,
                    far,
                    corner,
                    target=owner,
                    clip=True,
                    complete_unit=far_unit,
                )
            elif bool(getattr(owner, "is_input", False)):
                reverse_solution = _engine.solve(trace)
                reverse = box_for_unit(
                    cast(Any, reverse_solution),
                    far,
                    corner,
                    input=owner,
                    clip=True,
                    complete_unit=far_unit,
                )
            else:
                reverse_solution = _engine.solve_from(trace, owner)
                reverse = box_for_unit(
                    cast(Any, reverse_solution),
                    far,
                    corner,
                    source=owner,
                    clip=True,
                    complete_unit=far_unit,
                )
        except (ReceptiveFieldError, ValueError):
            return None
        return _box_membership_contains(reverse, complete_unit)

    def _violation(index: tuple[int, ...]) -> ReceptiveFieldViolation:
        """Build the violation record for a corner that fails reverse membership."""

        return ReceptiveFieldViolation(
            io_role=descriptor.io_role,
            index=index,
            magnitude=0.0,
            box=box,
            reason=(
                "exact box corner fails opposite-direction membership: the "
                "claimed influence is not confirmed by the reverse engine"
            ),
        )

    violations: list[ReceptiveFieldViolation] = []
    complete_choices = (
        _complete_corner_choices(descriptor, box, complete_unit)
        if descriptor.alignment is ReceptiveFieldAlignment.NOT_APPLICABLE
        else None
    )
    if complete_choices is not None:
        windowed_positions = tuple(
            index for index, axis in enumerate(box.axes) if axis.kind == "windowed"
        )
        for raw_unit in product(*complete_choices):
            far_unit = tuple(int(value) for value in raw_unit)
            corner = tuple(far_unit[position] for position in windowed_positions)
            if _reverse_confirms(corner, far_unit) is False:
                violations.append(_violation(far_unit))
        return tuple(violations)
    corner_choices = tuple(
        (start, stop - 1) if stop - 1 != start else (start,) for start, stop in windowed_bounds
    )
    for raw_corner in product(*corner_choices):
        corner = tuple(int(value) for value in raw_corner)
        if _reverse_confirms(corner, None) is False:
            violations.append(_violation(corner))
    return tuple(violations)


def _validation_result(
    *,
    target: Op,
    unit: tuple[int, ...],
    descriptors: Mapping[str, ReceptiveField],
    geometric: dict[str, ReceptiveFieldBox],
    gradients: Mapping[str, GradientReceptiveField],
    gradient_error: str | None,
    direction: ReceptiveFieldDirection,
    adjoint_violations: tuple[ReceptiveFieldViolation, ...] = (),
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
    direction:
        Direction shared by the paired geometric and empirical results.

    Returns
    -------
    ReceptiveFieldValidation
        Immutable tri-state validation result.
    """

    listed: list[ReceptiveFieldViolation] = list(adjoint_violations[:_MAX_LISTED_VIOLATIONS])
    n_violations = len(adjoint_violations)
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
        if adjoint_violations:
            message = (
                f"Receptive-field validation failed for {target.label!r} at {unit}: "
                f"{len(adjoint_violations)} exact-box corner(s) failed opposite-direction "
                f"membership and {n_violations - len(adjoint_violations)} gradient-support "
                "indices lie outside geometric bounds."
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
    elif gradient_error is not None or not descriptors or indeterminate_roles or nonfinite:
        status = ReceptiveFieldValidationStatus.INDETERMINATE
        details = gradient_error or (
            "non-finite gradient contamination"
            if nonfinite
            else (
                f"ineligible geometry for roles {', '.join(indeterminate_roles)}"
                if indeterminate_roles
                else "no eligible geometric endpoint"
            )
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
        direction=direction,
        unit_shape=tuple(target.shape),
    )


def _check_for_unit(
    target: Op,
    unit: Sequence[int],
    *,
    input: Op | str | None,
    source: object | None,
    direction: ReceptiveFieldDirection | str,
    result_target: object | None,
    atol: float,
    rtol: float,
    retain_graph: bool,
) -> ReceptiveFieldValidation:
    """Run one validation with explicit graph-retention control.

    Validation always probes exact finite nonzero support. ``atol`` and
    ``rtol`` remain accepted by the public compatibility surface but are not
    forwarded to the empirical probe, so they cannot suppress a violation.
    """

    if atol < 0 or rtol < 0:
        raise ReceptiveFieldConfigurationError("atol and rtol must be non-negative.")
    normalized_direction = ReceptiveFieldDirection(direction)
    if normalized_direction is ReceptiveFieldDirection.RECEPTIVE and result_target is not None:
        raise TypeError("target= is only valid with direction='projective'.")
    if normalized_direction is ReceptiveFieldDirection.PROJECTIVE and (
        input is not None or source is not None
    ):
        raise TypeError("input= and source= are only valid with direction='receptive'.")
    complete_unit = _normalize_complete_unit(target, unit)
    endpoint: Op | None = None
    solution: Any
    try:
        if normalized_direction is ReceptiveFieldDirection.RECEPTIVE:
            if input is not None and source is not None:
                raise TypeError("input= and source= cannot be used together.")
            if source is None:
                solution = _engine.solve(target.source_trace)
                view = ReceptiveFieldView(target, solution)
                descriptors = _selected_descriptors(view, input)
            else:
                endpoint = resolve_graph_point(target.source_trace, source)
                solution = _engine.solve_from(target.source_trace, endpoint)
                descriptors = solution.per_op.get(target.label, MappingProxyType({}))
        else:
            targets, _ = _select_targets(target, result_target)
            endpoint = targets[0] if len(targets) == 1 else None
            solution = solve_projective(target.source_trace, targets)
            descriptors = solution.per_op.get(target.label, MappingProxyType({}))
    except ReceptiveFieldUnavailableError as exc:
        return _validation_result(
            target=target,
            unit=complete_unit,
            descriptors=MappingProxyType({}),
            geometric={},
            gradients=MappingProxyType({}),
            gradient_error=str(exc),
            direction=normalized_direction,
        )
    geometric: dict[str, ReceptiveFieldBox] = {}
    for role, descriptor in descriptors.items():
        if descriptor.status not in _TAINTED_STATUSES and descriptor.axes is not None:
            geometric[role] = _box_for_descriptor(
                solution,
                target,
                descriptor,
                complete_unit,
                direction=normalized_direction,
                endpoint=endpoint,
            )
    adjoint_violations: list[ReceptiveFieldViolation] = []
    for probe_unit in _adjoint_probe_units(target, descriptors, complete_unit):
        for role, descriptor in descriptors.items():
            if descriptor.status in _TAINTED_STATUSES or descriptor.axes is None:
                continue
            if probe_unit == complete_unit:
                probe_box = geometric.get(role)
                if probe_box is None:
                    continue
            else:
                try:
                    probe_box = _box_for_descriptor(
                        solution,
                        target,
                        descriptor,
                        probe_unit,
                        direction=normalized_direction,
                        endpoint=endpoint,
                    )
                except (ReceptiveFieldError, ValueError):
                    continue
            adjoint_violations.extend(
                _exact_box_adjoint_violations(
                    target, probe_unit, descriptor, probe_box, normalized_direction
                )
            )
    gradient_error: str | None = None
    gradients: Mapping[str, GradientReceptiveField]
    try:
        if normalized_direction is ReceptiveFieldDirection.RECEPTIVE:
            probed = gradient_for_unit(
                target,
                complete_unit,
                input=input,
                source=source,
                atol=0.0,
                rtol=0.0,
                retain_graph=retain_graph,
            )
        else:
            probed = projective_gradient_for_unit(
                target,
                complete_unit,
                target=result_target,
                atol=0.0,
                rtol=0.0,
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
        direction=normalized_direction,
        adjoint_violations=tuple(adjoint_violations),
    )


def check_for_unit(
    owner: Op,
    unit: Sequence[int],
    *,
    input: Op | str | None = None,
    source: object | None = None,
    direction: ReceptiveFieldDirection | str = ReceptiveFieldDirection.RECEPTIVE,
    target: object | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    retain_graph: bool = False,
) -> ReceptiveFieldValidation:
    """Cross-check one target unit for the entity-view delegation surface.

    Parameters
    ----------
    owner, unit:
        Seed-owner operation and complete output-element index.
    input:
        Optional model-input operation or exact IO role.
    source:
        Optional receptive-direction ancestor graph point.
    direction:
        Receptive or projective containment direction.
    target:
        Optional projective-direction descendant graph point.
    atol, rtol:
        Accepted for compatibility and ignored. Validation always uses exact
        finite nonzero gradient support.
    retain_graph:
        Keep the armed autograd graph alive after the check. The default
        ``False`` FREES the graph -- disclosed here and on ``rf.check`` --
        so a later ``gradient(..., retain_graph=True)`` on the same armed
        capture needs ``retain_graph=True`` here first (b3 R14-N1).

    Returns
    -------
    ReceptiveFieldValidation
        Tri-state exact-containment result.
    """

    trace = owner.source_trace
    if trace is not None:
        _refuse_poisoned_rf_verification(trace)
    return _check_for_unit(
        owner,
        unit,
        input=input,
        source=source,
        direction=direction,
        result_target=target,
        atol=atol,
        rtol=rtol,
        retain_graph=retain_graph,
    )


def check(
    view: ReceptiveFieldView,
    unit: Sequence[int],
    *,
    input: Op | str | None = None,
    source: object | None = None,
    direction: ReceptiveFieldDirection | str = ReceptiveFieldDirection.RECEPTIVE,
    target: object | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    retain_graph: bool = False,
) -> ReceptiveFieldValidation:
    """Cross-check geometry and gradients through an explicit RF view.

    Parameters
    ----------
    view, unit:
        Receptive-field view and complete output-element index.
    input:
        Optional model-input operation or exact IO role.
    source:
        Optional receptive-direction ancestor graph point.
    direction:
        Receptive or projective containment direction.
    target:
        Optional projective-direction descendant graph point.
    atol, rtol:
        Accepted for compatibility and ignored. Validation always uses exact
        finite nonzero gradient support.
    retain_graph:
        Keep the armed autograd graph alive after the check (default frees
        it, like ``gradient()``'s default; b3 R14-N1).

    Returns
    -------
    ReceptiveFieldValidation
        Tri-state exact-containment result.
    """

    return check_for_unit(
        view._op,
        unit,
        input=input,
        source=source,
        direction=direction,
        target=target,
        atol=atol,
        rtol=rtol,
        retain_graph=retain_graph,
    )


def _resolve_ops(
    trace: Trace,
    ops: Sequence[Op | str] | None,
    direction: ReceptiveFieldDirection = ReceptiveFieldDirection.RECEPTIVE,
) -> tuple[Op, ...]:
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
        if (direction is ReceptiveFieldDirection.PROJECTIVE or not op.is_input)
        and not op.is_output
        and (op.is_input or op.has_saved_activation)
        and (
            op.is_input
            or any(
                descriptor.status not in _TAINTED_STATUSES
                for descriptor in solution.per_op.get(op.label, {}).values()
            )
        )
    )


def _preset_units(
    owner: Op,
    descriptor: ReceptiveField,
    preset: Literal["center", "corners"],
    batch_index: int,
) -> tuple[tuple[int, ...], ...]:
    """Resolve a named unit preset to complete output indices."""

    center_values = [int(extent) // 2 for extent in owner.shape]
    if center_values:
        batch_output_axis = next(
            (
                axis.output_axis
                for axis in descriptor.axes or ()
                if axis.input_axis == 0 and axis.output_axis is not None
            ),
            0,
        )
        if batch_index < 0 or batch_index >= int(owner.shape[batch_output_axis]):
            raise ReceptiveFieldError(
                f"batch_index {batch_index} is out of bounds for extent "
                f"{int(owner.shape[batch_output_axis])}."
            )
        center_values[batch_output_axis] = batch_index
    center = tuple(center_values)
    if preset == "center":
        return (center,)
    assert descriptor.axes is not None
    windowed_output_axes = tuple(
        cast(int, axis.output_axis)
        for axis in descriptor.axes
        if axis.kind == "windowed" and axis.output_axis is not None
    )
    choices = tuple((0, int(owner.shape[axis]) - 1) for axis in windowed_output_axes)
    units: list[tuple[int, ...]] = []
    for corner in product(*choices):
        resolved = list(center)
        for axis, coordinate in zip(windowed_output_axes, corner, strict=True):
            resolved[axis] = coordinate
        units.append(tuple(resolved))
    return tuple(dict.fromkeys(units)) or (center,)


def _selectors(value: object | Sequence[object] | None) -> tuple[object | None, ...]:
    """Normalize one optional endpoint selector or selector sequence."""

    if value is None or isinstance(value, str) or hasattr(value, "source_trace"):
        return (value,)
    if not isinstance(value, Sequence):
        return (value,)
    return tuple(value)


def _indeterminate_unit(owner: Op, batch_index: int) -> tuple[int, ...]:
    """Return a deterministic complete unit when grid metadata is unavailable."""

    # ``shape`` is legitimately ``None`` for a non-tensor-valued op (e.g. a
    # JAX while/cond decision pseudo-op); treat it as the op.py-documented
    # shapeless default (``()``) instead of crashing on an empty iteration.
    owner_shape = owner.shape if owner.shape is not None else ()
    unit = [int(extent) // 2 for extent in owner_shape]
    if unit and 0 <= batch_index < int(owner_shape[0]):
        unit[0] = batch_index
    return tuple(unit)


def _expected_status(descriptor: ReceptiveField) -> ReceptiveFieldStatus:
    """Derive the required certainty status from public geometric axes."""

    assert descriptor.axes is not None
    if any(axis.kind == "unknown" for axis in descriptor.axes):
        return ReceptiveFieldStatus.UNKNOWN
    non_pointwise = tuple(axis for axis in descriptor.axes if axis.kind != "pointwise")
    if non_pointwise and all(axis.kind == "full" and axis.exact for axis in non_pointwise):
        return ReceptiveFieldStatus.WHOLE_INPUT
    if descriptor.batch_coupled or any(not axis.exact for axis in descriptor.axes):
        return ReceptiveFieldStatus.UPPER_BOUND
    return ReceptiveFieldStatus.EXACT


def _check_solution_metadata(
    trace: Trace,
    solution: Any,
    direction: ReceptiveFieldDirection,
) -> None:
    """Check cheap descriptor and box invariants for one directional solution."""

    ops_by_label = {op.label: op for op in trace.layer_list}
    for (op_label, role), descriptor in solution.descriptors.items():
        state = solution.states[(op_label, role)]
        if state.taint is not None:
            if descriptor.status is not state.taint or descriptor.axes is not None:
                raise ReceptiveFieldError(
                    f"taint propagation failed for {op_label!r}, role {role!r}."
                )
            if descriptor.alignment is not ReceptiveFieldAlignment.NOT_APPLICABLE:
                raise ReceptiveFieldError(
                    f"tainted descriptor {op_label!r}, role {role!r} has an alignment."
                )
            continue
        if descriptor.axes is None:
            raise ReceptiveFieldError(
                f"untainted descriptor {op_label!r}, role {role!r} has no axes."
            )
        expected_status = _expected_status(descriptor)
        if descriptor.status is not expected_status:
            raise ReceptiveFieldError(
                f"status monotonicity failed for {op_label!r}, role {role!r}: "
                f"expected {expected_status.value}, got {descriptor.status.value}."
            )
        if descriptor.batch_coupled and descriptor.status is ReceptiveFieldStatus.EXACT:
            raise ReceptiveFieldError(
                f"batch-coupled descriptor {op_label!r}, role {role!r} is incorrectly exact."
            )
        expected_alignment = ReceptiveFieldAlignment.NOT_APPLICABLE
        if state.merge_seen:
            expected_alignment = (
                ReceptiveFieldAlignment.ALIGNED
                if all(axis.aligned for axis in descriptor.axes)
                else ReceptiveFieldAlignment.MISALIGNED
            )
        if descriptor.alignment is not expected_alignment:
            raise ReceptiveFieldError(
                f"alignment coherence failed for {op_label!r}, role {role!r}."
            )

        owner = ops_by_label[op_label]
        unit = _indeterminate_unit(owner, 0)
        try:
            box = _box_for_descriptor(
                solution,
                owner,
                descriptor,
                unit,
                direction=direction,
                endpoint=None,
            )
        except ReceptiveFieldError:
            continue
        for axis, extent in zip(box.axes, box.input_shape, strict=True):
            if axis.clipped_start is None or axis.clipped_stop is None:
                continue
            if not 0 <= axis.clipped_start <= axis.clipped_stop <= extent:
                raise ReceptiveFieldError(
                    f"box bounds for {op_label!r}, role {role!r}, axis "
                    f"{axis.input_axis} escape [0, {extent})."
                )


def check_geometric_metadata_invariants(trace: Trace) -> bool:
    """Run always-on, autograd-free receptive-field metadata invariants."""

    receptive_solution = _engine.solve(trace)
    _check_solution_metadata(trace, receptive_solution, ReceptiveFieldDirection.RECEPTIVE)
    output_ops = tuple(trace.output_ops)
    if output_ops:
        projective_solution = solve_projective(trace, output_ops)
        _check_solution_metadata(trace, projective_solution, ReceptiveFieldDirection.PROJECTIVE)
    return True


def validate_receptive_field_trace(
    trace: Trace,
    *,
    ops: Sequence[Op | str] | None = None,
    units: Literal["center", "corners"] | Sequence[int] | Sequence[Sequence[int]] = "center",
    batch_index: int = 0,
    inputs: Op | str | Sequence[Op | str] | None = None,
    source: object | Sequence[object] | None = None,
    direction: ReceptiveFieldDirection | str = ReceptiveFieldDirection.RECEPTIVE,
    target: object | Sequence[object] | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    raise_on_failure: bool = False,
) -> list[ReceptiveFieldValidation]:
    """Run the shared receptive-field validation scope on an existing trace."""

    if atol < 0 or rtol < 0:
        raise ReceptiveFieldConfigurationError("atol and rtol must be non-negative.")
    normalized_direction = ReceptiveFieldDirection(direction)
    if normalized_direction is ReceptiveFieldDirection.RECEPTIVE and target is not None:
        raise TypeError("target= is only valid with direction='projective'.")
    if normalized_direction is ReceptiveFieldDirection.PROJECTIVE and (
        inputs is not None or source is not None
    ):
        raise TypeError("inputs= and source= are only valid with direction='receptive'.")
    if inputs is not None and source is not None:
        raise TypeError("inputs= and source= cannot be used together.")

    owners = _resolve_ops(trace, ops, normalized_direction)
    receptive_endpoints = _selectors(source) if source is not None else _selectors(inputs)
    projective_endpoints = _selectors(target)
    endpoints = (
        receptive_endpoints
        if normalized_direction is ReceptiveFieldDirection.RECEPTIVE
        else projective_endpoints
    )
    results: list[ReceptiveFieldValidation] = []
    for owner in owners:
        for selected_endpoint in endpoints:
            solution: Any
            selected_input = (
                selected_endpoint
                if normalized_direction is ReceptiveFieldDirection.RECEPTIVE and source is None
                else None
            )
            selected_source = (
                selected_endpoint
                if normalized_direction is ReceptiveFieldDirection.RECEPTIVE and source is not None
                else None
            )
            if normalized_direction is ReceptiveFieldDirection.RECEPTIVE:
                if selected_source is None:
                    solution = _engine.solve(trace)
                    descriptors = _selected_descriptors(
                        ReceptiveFieldView(owner, solution), cast("Op | str | None", selected_input)
                    )
                else:
                    source_op = resolve_graph_point(trace, selected_source)
                    solution = _engine.solve_from(trace, source_op)
                    descriptors = solution.per_op.get(owner.label, MappingProxyType({}))
            else:
                try:
                    target_ops, _ = _select_targets(owner, selected_endpoint)
                    solution = solve_projective(trace, target_ops)
                    descriptors = solution.per_op.get(owner.label, MappingProxyType({}))
                except ReceptiveFieldUnavailableError:
                    descriptors = MappingProxyType({})

            if isinstance(units, str):
                eligible = next(
                    (
                        descriptor
                        for descriptor in descriptors.values()
                        if descriptor.axes is not None
                    ),
                    None,
                )
                resolved_units = (
                    (_indeterminate_unit(owner, batch_index),)
                    if eligible is None
                    else _preset_units(owner, eligible, units, batch_index)
                )
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
                    owner,
                    unit,
                    input=cast("Op | str | None", selected_input),
                    source=selected_source,
                    direction=normalized_direction,
                    result_target=(
                        selected_endpoint
                        if normalized_direction is ReceptiveFieldDirection.PROJECTIVE
                        else None
                    ),
                    atol=atol,
                    rtol=rtol,
                    retain_graph=True,
                )
                if raise_on_failure:
                    result.assert_valid()
                results.append(result)
    return results


def cross_validate(
    trace: Trace,
    *,
    ops: Sequence[Op | str] | None = None,
    units: Literal["center", "corners"] | Sequence[int] | Sequence[Sequence[int]] = "center",
    batch_index: int = 0,
    inputs: Op | str | Sequence[Op | str] | None = None,
    source: object | Sequence[object] | None = None,
    direction: ReceptiveFieldDirection | str = ReceptiveFieldDirection.RECEPTIVE,
    target: object | Sequence[object] | None = None,
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
    source:
        Optional receptive-direction ancestor selector or selectors.
    direction:
        Receptive or projective containment direction.
    target:
        Optional projective-direction descendant selector or selectors.
    atol, rtol:
        Accepted for compatibility and ignored. Validation always uses exact
        finite nonzero gradient support.
    raise_on_failure:
        Raise on ``FAIL`` only.

    Returns
    -------
    list[ReceptiveFieldValidation]
        One tri-state result per operation, unit, and explicit input selector.
    """

    _refuse_poisoned_rf_verification(trace)
    return validate_receptive_field_trace(
        trace,
        ops=ops,
        units=units,
        batch_index=batch_index,
        inputs=inputs,
        source=source,
        direction=direction,
        target=target,
        atol=atol,
        rtol=rtol,
        raise_on_failure=raise_on_failure,
    )


__all__ = [
    "check",
    "check_for_unit",
    "cross_validate",
    "validate_receptive_field_trace",
]

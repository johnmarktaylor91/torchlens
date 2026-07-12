"""Public descriptor materialization for geometric receptive-field states."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ._engine_geometry import (
    _AxisState,
    _Dissolved,
    _Full,
    _InputState,
    _maximum_integer_hull_size,
)
from ._types import (
    AxisKind,
    GridLayout,
    ReceptiveField,
    ReceptiveFieldAlignment,
    ReceptiveFieldAxis,
    ReceptiveFieldStatus,
)


if TYPE_CHECKING:
    from ..data_classes.op import Op


def _descriptor(op: Op, state: _InputState) -> ReceptiveField:
    """Materialize one public descriptor and derive its status."""

    if state.taint is not None or state.axes is None:
        status = state.taint or ReceptiveFieldStatus.UNKNOWN
        layout = GridLayout(
            axis_kinds=cast("tuple[AxisKind, ...]", ("unknown",) * len(state.input_shape)),
            windowed_axes=(),
            source="derived",
            provenance=(op.label,) * len(state.input_shape),
        )
        return ReceptiveField(
            op_label=op.label,
            io_role=state.io_role,
            input_op_label=state.input_op_label,
            input_shape=state.input_shape,
            layout=layout,
            axes=None,
            status=status,
            alignment=ReceptiveFieldAlignment.NOT_APPLICABLE,
            batch_coupled=False,
            rule=state.rule,
            notes=state.notes,
        )

    public_axes = tuple(
        _public_axis(axis_index, input_extent, axis, op)
        for axis_index, (input_extent, axis) in enumerate(zip(state.input_shape, state.axes))
    )
    kinds = cast("tuple[AxisKind, ...]", tuple(axis.kind for axis in public_axes))
    layout = GridLayout(
        axis_kinds=kinds,
        windowed_axes=tuple(index for index, kind in enumerate(kinds) if kind == "windowed"),
        source="derived",
        provenance=tuple(axis.provenance for axis in state.axes),
    )
    batch_coupled = state.batch_axis is not None and public_axes[state.batch_axis].kind == "full"
    status = _derive_status(public_axes, batch_coupled)
    alignment = ReceptiveFieldAlignment.NOT_APPLICABLE
    if state.merge_seen:
        alignment = (
            ReceptiveFieldAlignment.ALIGNED
            if all(axis.aligned for axis in public_axes)
            else ReceptiveFieldAlignment.MISALIGNED
        )
    notes = state.notes
    if batch_coupled and status == ReceptiveFieldStatus.UPPER_BOUND:
        notes += (f"{op.label}: batch coupling prevents an exact descriptor",)
    return ReceptiveField(
        op_label=op.label,
        io_role=state.io_role,
        input_op_label=state.input_op_label,
        input_shape=state.input_shape,
        layout=layout,
        axes=public_axes,
        status=status,
        alignment=alignment,
        batch_coupled=batch_coupled,
        rule=state.rule,
        notes=notes,
    )


def _public_axis(axis_index: int, extent: int, state: _AxisState, op: Op) -> ReceptiveFieldAxis:
    """Convert one internal axis to its validated public semantic form."""

    geometry = state.geometry
    if isinstance(geometry, _Full):
        return ReceptiveFieldAxis(
            axis_index, None, "full", extent, None, None, None, geometry.exact, True, False
        )
    if isinstance(geometry, _Dissolved) or state.kind == "unknown":
        return ReceptiveFieldAxis(
            axis_index, state.output_axis, "unknown", extent, None, None, None, False, False, False
        )
    if state.kind == "pointwise":
        return ReceptiveFieldAxis(
            axis_index,
            state.output_axis,
            "pointwise",
            extent,
            None,
            None,
            None,
            geometry.exact,
            geometry.aligned,
            geometry.sparse,
        )
    if geometry.lo.a != geometry.hi.a or state.output_axis is None:
        return ReceptiveFieldAxis(
            axis_index, state.output_axis, "unknown", extent, None, None, None, False, False, True
        )
    output_extent = int(op.shape[state.output_axis])
    return ReceptiveFieldAxis(
        input_axis=axis_index,
        output_axis=state.output_axis,
        kind="windowed",
        input_extent=extent,
        size=_maximum_integer_hull_size(geometry, output_extent),
        jump=geometry.lo.a,
        center0=(geometry.lo.b + geometry.hi.b) / 2,
        exact=geometry.exact,
        aligned=geometry.aligned,
        sparse_possible=geometry.sparse,
    )


def _derive_status(
    axes: tuple[ReceptiveFieldAxis, ...], batch_coupled: bool
) -> ReceptiveFieldStatus:
    """Derive the six-value public status from semantic axes."""

    if any(axis.kind == "unknown" for axis in axes):
        return ReceptiveFieldStatus.UNKNOWN
    non_pointwise = tuple(axis for axis in axes if axis.kind != "pointwise")
    all_global = bool(non_pointwise) and all(
        axis.kind == "full" and axis.exact for axis in non_pointwise
    )
    if all_global:
        return ReceptiveFieldStatus.WHOLE_INPUT
    if batch_coupled or any(not axis.exact for axis in axes):
        return ReceptiveFieldStatus.UPPER_BOUND
    return ReceptiveFieldStatus.EXACT


__all__: list[str] = []

"""Tabular summaries of trace-wide receptive-field solutions."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any, Literal

from ._engine import solve
from ._types import ReceptiveField, ReceptiveFieldProfile, ReceptiveFieldStatus


if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


ReceptiveFieldProfileLevel = Literal["op", "layer", "call", "module"]


def _require_pandas() -> Any:
    """Import pandas with the standard TorchLens tabular-extra guidance.

    Returns
    -------
    Any
        Imported pandas module.
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
        ) from exc
    return pd


def _require_input_role(trace: "Trace", input_op: "Op | None") -> str | None:
    """Validate a bulk-table input handle and return its IO role.

    Parameters
    ----------
    trace:
        Trace that owns the receptive-field solution.
    input_op:
        Optional model-input operation handle.

    Returns
    -------
    str | None
        The requested model-input IO role, or ``None`` when unfiltered.

    Raises
    ------
    TypeError
        If ``input_op`` is not an operation handle.
    ValueError
        If the operation is not a model input belonging to ``trace``.
    """

    if input_op is None:
        return None

    from ..data_classes.op import Op

    if not isinstance(input_op, Op):
        raise TypeError(
            "input must be a model-input Op handle; string layer names are not accepted."
        )
    if not input_op.is_input or input_op.io_role is None:
        raise ValueError("input must be a model-input Op handle.")
    if not any(op is input_op for op in trace.layer_list):
        raise ValueError("input must be a model-input Op belonging to this trace.")
    return input_op.io_role


def _resolved_output_label(trace: "Trace", output_label: str) -> str:
    """Resolve a boundary label to its pass-qualified operation label.

    Parameters
    ----------
    trace:
        Trace that owns the boundary metadata.
    output_label:
        Boundary label stored by a Layer, ModuleCall, or Module.

    Returns
    -------
    str
        Exact pass-qualified operation label when the boundary resolves.
    """

    try:
        return str(trace[output_label].label)
    except (KeyError, ValueError):
        return output_label


def _entity_outputs(
    trace: "Trace", level: ReceptiveFieldProfileLevel
) -> Iterable[tuple[str, str, int, str]]:
    """Yield profile entity labels and their represented boundary outputs.

    Parameters
    ----------
    trace:
        Trace containing completed operation and module metadata.
    level:
        Requested table granularity.

    Yields
    ------
    tuple[str, str, int, str]
        Entity name, entity kind, boundary-output index, and output-op label.
    """

    if level == "op":
        for op in trace.layer_list:
            yield op.label, "op", 0, op.label
    elif level == "layer":
        for layer in trace.layer_logs.values():
            for output_index, output_op in enumerate(layer.op_labels):
                yield layer.layer_label, "layer", output_index, output_op
    elif level == "call":
        for call in trace.module_calls.values():
            for output_index, output_op in enumerate(call.output_ops):
                yield (
                    call.call_label,
                    "call",
                    output_index,
                    _resolved_output_label(trace, output_op),
                )
    else:
        for module in trace.modules.values():
            for output_index, output_op in enumerate(module.output_ops):
                yield (
                    module.address,
                    "module",
                    output_index,
                    _resolved_output_label(trace, output_op),
                )


def _kind_summary(descriptor: ReceptiveField) -> str:
    """Format derived axis kinds without assuming semantic axis names.

    Parameters
    ----------
    descriptor:
        Per-input geometric receptive-field descriptor.

    Returns
    -------
    str
        Stable input-axis-index and short-kind summary.
    """

    abbreviations = {"windowed": "win", "pointwise": "pt", "full": "full", "unknown": "unk"}
    return " ".join(
        f"axis{axis}:{abbreviations[kind]}"
        for axis, kind in enumerate(descriptor.layout.axis_kinds)
    )


def _row(
    name: str,
    kind: str,
    output_index: int,
    output_op: str,
    descriptor: ReceptiveField,
) -> dict[str, Any]:
    """Build one receptive-field profile row from a solved descriptor.

    Parameters
    ----------
    name:
        Entity label at the requested level.
    kind:
        Entity granularity label.
    output_index:
        Index of the represented boundary output within the entity.
    output_op:
        Pass-qualified represented operation label.
    descriptor:
        Solved per-input receptive-field descriptor.

    Returns
    -------
    dict[str, Any]
        One dataframe-ready profile row.
    """

    axes = descriptor.axes
    windowed_axes = () if axes is None else tuple(axis for axis in axes if axis.kind == "windowed")
    return {
        "name": name,
        "kind": kind,
        "output_index": output_index,
        "output_op": output_op,
        "input_role": descriptor.io_role,
        "input_op": descriptor.input_op_label,
        "status": descriptor.status,
        "alignment": descriptor.alignment,
        "spatial_rank": len(windowed_axes),
        "size": None if axes is None else tuple(axis.size for axis in windowed_axes),
        "jump": None if axes is None else tuple(axis.jump for axis in windowed_axes),
        "center0": None if axes is None else tuple(axis.center0 for axis in windowed_axes),
        "kind_summary": _kind_summary(descriptor),
        "sparse_possible": bool(axes is not None and any(axis.sparse_possible for axis in axes)),
        "rule": descriptor.rule,
        "notes": descriptor.notes,
        "batch_coupled": descriptor.batch_coupled,
        "exact": bool(axes is not None and all(axis.exact for axis in axes)),
        "layout": descriptor.layout,
    }


def build_rf_profile(
    trace: "Trace",
    *,
    level: ReceptiveFieldProfileLevel = "op",
    input: "Op | None" = None,
    statuses: Collection[ReceptiveFieldStatus] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
) -> ReceptiveFieldProfile:
    """Build a geometric receptive-field table for a completed trace.

    Parameters
    ----------
    trace:
        Completed trace whose captured DAG is reported.
    level:
        ``"op"``, ``"layer"``, ``"call"``, or ``"module"`` granularity.
    input:
        Optional model-input operation handle. String layer names are rejected.
    statuses:
        Optional receptive-field statuses to retain.
    sort_by:
        Optional profile column used for stable sorting.
    ascending:
        Whether an explicit ``sort_by`` sorts in ascending order.

    Returns
    -------
    ReceptiveFieldProfile
        Frozen profile wrapper exposing a copied pandas dataframe.
    """

    if level not in {"op", "layer", "call", "module"}:
        raise ValueError("level must be 'op', 'layer', 'call', or 'module'.")
    requested_role = _require_input_role(trace, input)
    status_filter = None if statuses is None else frozenset(statuses)
    if status_filter is not None and not all(
        isinstance(status, ReceptiveFieldStatus) for status in status_filter
    ):
        raise TypeError("statuses must contain only ReceptiveFieldStatus values.")

    pd = _require_pandas()
    solution = solve(trace)
    rows: list[dict[str, Any]] = []
    for name, kind, output_index, output_op in _entity_outputs(trace, level):
        for role, descriptor in solution.per_op.get(output_op, {}).items():
            if requested_role is not None and role != requested_role:
                continue
            if status_filter is not None and descriptor.status not in status_filter:
                continue
            rows.append(_row(name, kind, output_index, output_op, descriptor))

    columns = [
        "name",
        "kind",
        "output_index",
        "output_op",
        "input_role",
        "input_op",
        "status",
        "alignment",
        "spatial_rank",
        "size",
        "jump",
        "center0",
        "kind_summary",
        "sparse_possible",
        "rule",
        "notes",
        "batch_coupled",
        "exact",
        "layout",
    ]
    if sort_by is not None and sort_by not in columns:
        raise ValueError(f"sort_by must be one of: {', '.join(columns)}.")
    frame = pd.DataFrame(rows, columns=columns)
    if sort_by is not None:
        frame = frame.sort_values(sort_by, ascending=ascending, na_position="last", kind="stable")
    frame = frame.reset_index(drop=True)
    frame.attrs.update(
        {
            "level": level,
            "input_role": requested_role,
            "statuses": status_filter,
            "sort_by": sort_by,
            "ascending": ascending,
        }
    )
    return ReceptiveFieldProfile(frame=frame, level=level)

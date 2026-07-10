"""Unified performance and resource profile tables for completed traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.trace import Trace


ProfileLevel = Literal["op", "module", "call"]
ProfileSort = Literal["time", "flops", "activation_memory", "param_count"]


@dataclass(frozen=True)
class TraceProfile:
    """A printable, tabular resource profile assembled from one trace.

    Parameters
    ----------
    frame:
        Profile rows in the requested granularity.
    level:
        Granularity used to construct ``frame``.

    Notes
    -----
    Operation wall-times are collected while TorchLens instrumentation is
    active. Use them to identify relative hotspots, not as clean benchmarks.
    """

    frame: "pd.DataFrame"
    level: ProfileLevel

    def to_pandas(self) -> "pd.DataFrame":
        """Return a copy of the underlying profile dataframe.

        Returns
        -------
        pandas.DataFrame
            Resource profile rows.
        """

        return self.frame.copy()

    def __repr__(self) -> str:
        """Render the profile as a compact notebook-friendly table.

        Returns
        -------
        str
            Human-readable table representation.
        """

        display_frame = self.frame.copy()
        display_frame["time"] = display_frame["time"].map(_format_duration)
        return display_frame.to_string(index=False)


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


def _ops_for_labels(trace: "Trace", labels: list[str]) -> list[Any]:
    """Resolve pass-qualified labels to operation records.

    Parameters
    ----------
    trace:
        Source trace.
    labels:
        Pass-qualified operation labels.

    Returns
    -------
    list[Any]
        Resolved operation records; stale labels are ignored.
    """

    ops: list[Any] = []
    for label in labels:
        try:
            ops.append(trace[label])
        except (KeyError, ValueError):
            continue
    return ops


def _sum_optional(ops: list[Any], field: str) -> int | float | None:
    """Sum a metric without turning wholly unavailable metadata into zero.

    Parameters
    ----------
    ops:
        Operations to aggregate.
    field:
        Numeric metadata field.

    Returns
    -------
    int | float | None
        Sum when at least one value is available, otherwise ``None``.
    """

    values = [getattr(op, field, None) for op in ops]
    if field == "func_duration":
        available = [float(value) for value in values if value is not None]
    else:
        available = [int(value) for value in values if value is not None]
    return sum(available) if available else None


def _format_duration(seconds: float | None) -> str:
    """Format an operation duration using a compact human-readable unit.

    Parameters
    ----------
    seconds:
        Duration in seconds, if instrumentation captured one.

    Returns
    -------
    str
        Duration rendered in seconds, milliseconds, or microseconds.
    """

    if seconds is None:
        return ""
    if seconds >= 1:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds * 1e6:.3f} us"


def _values(ops: list[Any], field: str) -> str | None:
    """Format the distinct non-null values of one operation metadata field.

    Parameters
    ----------
    ops:
        Operations to inspect.
    field:
        Metadata field name.

    Returns
    -------
    str | None
        One value or a comma-separated stable set, if available.
    """

    values = sorted({str(value) for op in ops if (value := getattr(op, field, None)) is not None})
    return ", ".join(values) if values else None


def _row(name: str, kind: str, ops: list[Any], *, param_count: int | None) -> dict[str, Any]:
    """Build one consistent profile row from operation metadata.

    Parameters
    ----------
    name:
        Stable row label.
    kind:
        Row granularity label.
    ops:
        Operations represented by the row.
    param_count:
        Parameter count owned by the corresponding module scope.

    Returns
    -------
    dict[str, Any]
        Profile row.
    """

    saved = [bool(getattr(op, "has_saved_activation", False)) for op in ops]
    saved_activation: bool | str | None
    if not saved:
        saved_activation = None
    elif all(saved):
        saved_activation = True
    elif not any(saved):
        saved_activation = False
    else:
        saved_activation = "partial"
    return {
        "name": name,
        "kind": kind,
        "op_count": len(ops),
        "time": _sum_optional(ops, "func_duration"),
        "flops": _sum_optional(ops, "flops_forward"),
        "activation_memory": _sum_optional(ops, "activation_memory"),
        "saved_activation": saved_activation,
        "param_count": param_count,
        "dtype": _values(ops, "dtype"),
        "device": _values(ops, "device"),
    }


def build_profile(
    trace: "Trace",
    *,
    level: ProfileLevel = "op",
    sort_by: ProfileSort = "time",
    ascending: bool = False,
) -> TraceProfile:
    """Build a unified op, module, or invocation resource profile.

    Parameters
    ----------
    trace:
        Completed trace whose existing metadata is reported.
    level:
        ``"op"`` for one row per operation, ``"module"`` for one row per
        module address, or ``"call"`` for one row per module invocation.
    sort_by:
        Column used for sorting. ``"time"`` sorts descending by default.
    ascending:
        Whether to reverse the default descending order.

    Returns
    -------
    TraceProfile
        Printable profile object exposing :meth:`TraceProfile.to_pandas`.

    Notes
    -----
    Operation wall-times are captured under instrumentation. They are relative
    hotspot guidance, not clean benchmark timings.
    """

    if level not in {"op", "module", "call"}:
        raise ValueError("level must be 'op', 'module', or 'call'.")
    if sort_by not in {"time", "flops", "activation_memory", "param_count"}:
        raise ValueError("sort_by must be 'time', 'flops', 'activation_memory', or 'param_count'.")

    pd = _require_pandas()
    rows: list[dict[str, Any]] = []
    if level == "op":
        for op in trace.layer_list:
            rows.append(
                _row(
                    str(getattr(op, "label", None) or getattr(op, "layer_label", "")),
                    "op",
                    [op],
                    param_count=getattr(op, "num_params", None),
                )
            )
    elif level == "call":
        for call in trace.module_calls.values():
            ops = _ops_for_labels(trace, list(getattr(call, "ops", ())))
            rows.append(
                _row(
                    str(getattr(call, "call_label", "")),
                    "call",
                    ops,
                    param_count=getattr(call, "num_params", None),
                )
            )
    else:
        for module in trace.modules.values():
            labels = [label for call in module.calls.values() for label in call.ops]
            ops = _ops_for_labels(trace, labels)
            rows.append(
                _row(
                    str(getattr(module, "address", "")),
                    "module",
                    ops,
                    param_count=getattr(module, "num_params", None),
                )
            )

    columns = [
        "name",
        "kind",
        "op_count",
        "time",
        "flops",
        "activation_memory",
        "saved_activation",
        "param_count",
        "dtype",
        "device",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame = frame.sort_values(sort_by, ascending=ascending, na_position="last", kind="stable")
    frame = frame.reset_index(drop=True)
    frame.attrs.update({"level": level, "sort_by": sort_by, "ascending": ascending})
    return TraceProfile(frame=frame, level=level)

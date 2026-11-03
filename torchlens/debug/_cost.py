"""Forward-cost ranking helpers for TorchLens traces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

from .._debug_report import Report, table_report, trace_subject
from .._utils._record_group_table import record_member
from ._common import _ordered_ops, _require_pandas, _source_line

CostMetric = Literal["flops", "memory", "duration", "device_time"]


def _metric_field(by: CostMetric) -> str:
    """Map a public cost metric alias to an Op field.

    Parameters
    ----------
    by:
        Cost metric alias.

    Returns
    -------
    str
        Op field name.
    """

    fields = {
        "flops": "flops_forward",
        "memory": "activation_memory",
        "duration": "func_duration",
        # device_time reads the session-time Kineto join, not an Op field;
        # _op_metric_reader owns that branch.
        "device_time": "func_duration",
    }
    return fields[by]


def _device_time_lookup(trace: Trace) -> dict[str, int]:
    """Return the per-op joined device-time table, refusing typed when absent.

    The Kineto join is session-time (torchnative W2.2): it exists only on a
    trace captured under ``torchlens.report.kernel_profile`` (or an
    owned session joined via ``join_session``) whose join reached
    ``joined``. Asking for ``by="device_time"`` anywhere else raises the
    typed ladder refusal (``device_time_unavailable``) instead of returning
    silent zeros.
    """

    from .._observability._join import require_availability
    from .._observability._native_profile import join_result_for

    result = join_result_for(trace)
    if result is None:
        from .._observability._errors import ProfilerSessionError

        raise ProfilerSessionError(
            "device_time is unavailable: this trace carries no Kineto join. "
            "Device time exists only for captures run under an owned "
            "profiler session.",
            code="device_time_unavailable",
            remedy=(
                "Capture through torchlens.report.kernel_profile "
                "(one execution, save-nothing tier) on a CUDA host, then "
                "rank with by='device_time'."
            ),
        )
    require_availability(result)
    return result.op_device_ns


def hot_path_rows(
    trace: Trace, by: CostMetric = "flops"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pandas-free core of :func:`hot_path` (agent stage-0 item 4).

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    by:
        Cost metric: ``"flops"``, ``"memory"``, ``"duration"``, or
        ``"device_time"`` (joined Kineto device nanoseconds; refuses typed
        without a joined session -- see torchnative W2.2/W2.3).

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Cost-ranked row dicts and the attrs mapping (excluded count, metric,
        capture-honesty facts).
    """

    field_name = _metric_field(by)
    device_times = _device_time_lookup(trace) if by == "device_time" else None
    rows: dict[str, dict[str, float | int | str]] = {}
    excluded = 0
    for op in _ordered_ops(trace):
        if int(getattr(op, "step_index", 0) or 0) <= 0:
            continue
        if device_times is not None:
            value = device_times.get(str(getattr(op, "label", "")), None)
        else:
            value = record_member(op, field_name, None)
        if value is None:
            excluded += 1
            continue
        numeric_value = float(value)
        source = _source_line(op) or "<unknown>"
        row = rows.setdefault(
            source,
            {"source_file:line": source, "op_count": 0, "total_cost": 0.0, "pct_total": 0.0},
        )
        row["op_count"] = int(row["op_count"]) + 1
        row["total_cost"] = float(row["total_cost"]) + numeric_value

    total = sum(float(row["total_cost"]) for row in rows.values())
    for row in rows.values():
        row["pct_total"] = 0.0 if total == 0 else float(row["total_cost"]) / total * 100.0

    from .._capture_honesty import capture_honesty_facts

    sorted_rows = sorted(rows.values(), key=lambda row: float(row["total_cost"]), reverse=True)
    attrs: dict[str, Any] = {
        "excluded_missing_metric_count": excluded,
        "metric": by,
        "torchlens_capture_honesty": capture_honesty_facts(trace),
    }
    return [dict(row) for row in sorted_rows], attrs


def hot_path(trace: Trace, by: CostMetric = "flops") -> Report:
    """Rank source lines by aggregate forward cost.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    by:
        Cost metric: ``"flops"``, ``"memory"``, or ``"duration"``.

    Returns
    -------
    Report
        Tabular ``tl.debug`` report; ``.to_pandas()`` returns the
        DataFrame described here.
        Columns are ``source_file:line``, ``op_count``, ``total_cost``, and
        ``pct_total``. The number of ops excluded for missing metrics is stored
        in ``df.attrs["excluded_missing_metric_count"]``. The optional view
        over :func:`hot_path_rows` (the pandas-free core).
    """

    rows, attrs = hot_path_rows(trace, by)
    pd = _require_pandas()
    frame = pd.DataFrame(
        rows,
        columns=["source_file:line", "op_count", "total_cost", "pct_total"],
    )
    frame.attrs.update(attrs)
    return table_report(
        "hot_path",
        frame,
        subject=trace_subject(trace),
        finding_columns=("source_file:line", "total_cost", "pct_total"),
    )

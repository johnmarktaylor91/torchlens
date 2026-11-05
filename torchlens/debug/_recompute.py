"""Activation recomputation candidate ranking helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

from .._debug_report import Report, table_report, trace_subject
from .._utils._record_group_table import record_member
from ._common import _compute_ops, _op_label, _require_pandas, _source_line


def _candidate_rows(trace: Trace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build one unranked candidate row per compute op with both cost facts.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, int]]
        Candidate rows (``suggested`` still False) and the per-reason counts of ops
        excluded for missing or nonpositive activation memory or forward FLOPs.
    """

    rows: list[dict[str, Any]] = []
    excluded = {
        "missing_activation_memory": 0,
        "missing_flops_forward": 0,
        "nonpositive_flops_forward": 0,
    }
    for op in _compute_ops(trace):
        activation_memory = record_member(op, "activation_memory", None)
        flops_forward = record_member(op, "flops_forward", None)
        if activation_memory is None:
            excluded["missing_activation_memory"] += 1
            continue
        if flops_forward is None:
            excluded["missing_flops_forward"] += 1
            continue
        memory_value = int(activation_memory)
        flops_value = float(flops_forward)
        if flops_value <= 0:
            excluded["nonpositive_flops_forward"] += 1
            continue
        rows.append(
            {
                "op": _op_label(op),
                "activation_memory": memory_value,
                "flops_forward": flops_value,
                "mem_per_flop": memory_value / flops_value,
                "suggested": False,
                "source_line": _source_line(op),
            }
        )
    return rows, excluded


def recompute_candidates(trace: Trace, *, budget_gb: float | None = None) -> Report:
    """Rank ops by activation memory per forward FLOP.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    budget_gb:
        Optional greedy freed-memory target in GiB.

    Returns
    -------
    Report
        Tabular ``tl.debug`` report; ``.to_pandas()`` returns the
        DataFrame described here.
        Candidate rows with exclusion counts in ``df.attrs``; ``source_line``
        is the ``"file:line"`` of the user code that ran the op.

    Examples
    --------
    >>> # requires: pandas
    >>> frame = recompute_candidates(tl.trace(model, x), budget_gb=1e-9).to_pandas()
    >>> len(frame) > 0, bool(frame["suggested"].any())
    (True, True)
    """

    pd = _require_pandas()
    rows, excluded = _candidate_rows(trace)
    rows.sort(key=lambda row: float(row["mem_per_flop"]), reverse=True)
    total_freeable = sum(int(row["activation_memory"]) for row in rows)
    if budget_gb is not None:
        target = budget_gb * 1024**3
        freed = 0
        for row in rows:
            if freed >= target:
                break
            row["suggested"] = True
            freed += int(row["activation_memory"])

    frame = pd.DataFrame(
        rows,
        columns=[
            "op",
            "activation_memory",
            "flops_forward",
            "mem_per_flop",
            "suggested",
            "source_line",
        ],
    )
    frame.attrs["total_freeable"] = total_freeable
    frame.attrs["budget_gb"] = budget_gb
    for key, value in excluded.items():
        frame.attrs[f"excluded_{key}_count"] = value
    from .._capture_honesty import attach_dataframe_honesty

    return table_report(
        "recompute_candidates",
        attach_dataframe_honesty(frame, trace),
        subject=trace_subject(trace),
        finding_columns=("op", "activation_memory", "mem_per_flop", "suggested", "source_line"),
    )

"""Activation recomputation candidate ranking helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.trace import Trace

from ._common import _compute_ops, _op_label, _require_pandas


def recompute_candidates(trace: Trace, *, budget_gb: float | None = None) -> "pd.DataFrame":
    """Rank ops by activation memory per forward FLOP.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    budget_gb:
        Optional greedy freed-memory target in GiB.

    Returns
    -------
    pandas.DataFrame
        Candidate rows with exclusion counts in ``df.attrs``.
    """

    pd = _require_pandas()
    rows: list[dict[str, Any]] = []
    excluded = {
        "missing_activation_memory": 0,
        "missing_flops_forward": 0,
        "nonpositive_flops_forward": 0,
    }
    for op in _compute_ops(trace):
        activation_memory = getattr(op, "activation_memory", None)
        flops_forward = getattr(op, "flops_forward", None)
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
            }
        )

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
        columns=["op", "activation_memory", "flops_forward", "mem_per_flop", "suggested"],
    )
    frame.attrs["total_freeable"] = total_freeable
    frame.attrs["budget_gb"] = budget_gb
    for key, value in excluded.items():
        frame.attrs[f"excluded_{key}_count"] = value
    return frame

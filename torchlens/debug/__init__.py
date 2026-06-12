"""Power-user debugging helpers for completed TorchLens traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.op import Op
    from torchlens.data_classes.trace import Trace


CostMetric = Literal["flops", "memory", "duration"]


@dataclass(frozen=True)
class BisectNanResult:
    """Result returned by :func:`bisect_nan`.

    Parameters
    ----------
    found:
        Whether a saved activation containing NaN or Inf was found.
    op:
        First offending op, or ``None`` when no saved offender was found.
    label:
        Offending op label, when available.
    source_line:
        Source location formatted as ``"file:line"``, when available.
    kind:
        ``"nan"``, ``"inf"``, ``"nan+inf"``, or ``"none"``.
    message:
        Actionable human-readable result summary.
    """

    found: bool
    op: Op | None
    label: str | None
    source_line: str | None
    kind: str
    message: str


def _ordered_ops(trace: Trace) -> list[Op]:
    """Return trace ops in forward execution order.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.

    Returns
    -------
    list[Op]
        Ops sorted by ``step_index`` with boundary ops kept after compute ops
        at the same step.
    """

    return sorted(
        trace.layer_list,
        key=lambda op: (
            int(getattr(op, "step_index", 0) or 0),
            bool(getattr(op, "is_input", False)),
            bool(getattr(op, "is_output", False)),
            str(getattr(op, "layer_label", "")),
        ),
    )


def _source_line(op: Op) -> str | None:
    """Return the first source location for an op.

    Parameters
    ----------
    op:
        TorchLens op.

    Returns
    -------
    str | None
        ``"file:line"`` or ``None`` when no source context is present.
    """

    context = getattr(op, "code_context", None) or ()
    if not context:
        return None
    location = context[0]
    file_name = getattr(location, "file", None)
    line_number = getattr(location, "line_number", None)
    if file_name is None or line_number is None:
        return None
    return f"{file_name}:{line_number}"


def _nonfinite_kind(tensor: torch.Tensor) -> str:
    """Classify non-finite values in a tensor.

    Parameters
    ----------
    tensor:
        Tensor to inspect.

    Returns
    -------
    str
        ``"nan"``, ``"inf"``, ``"nan+inf"``, or ``"none"``.
    """

    if not torch.is_floating_point(tensor) and not torch.is_complex(tensor):
        return "none"
    has_nan = bool(torch.isnan(tensor).any().item())
    has_inf = bool(torch.isinf(tensor).any().item())
    if has_nan and has_inf:
        return "nan+inf"
    if has_nan:
        return "nan"
    if has_inf:
        return "inf"
    return "none"


def bisect_nan(trace: Trace) -> BisectNanResult:
    """Locate the first saved op whose output contains NaN or Inf.

    Parameters
    ----------
    trace:
        Completed TorchLens trace with saved activations.

    Returns
    -------
    BisectNanResult
        Clear result object instead of raising when no non-finite saved
        activation is found.
    """

    unsaved_compute_ops = 0
    for op in _ordered_ops(trace):
        if bool(getattr(op, "is_input", False)):
            continue
        if not bool(getattr(op, "has_saved_activation", False)):
            if int(getattr(op, "step_index", 0) or 0) > 0:
                unsaved_compute_ops += 1
            continue
        try:
            out = op.out
        except ValueError:
            unsaved_compute_ops += 1
            continue
        if not isinstance(out, torch.Tensor):
            continue
        kind = _nonfinite_kind(out)
        if kind != "none":
            label = str(getattr(op, "layer_label", ""))
            source_line = _source_line(op)
            return BisectNanResult(
                found=True,
                op=op,
                label=label,
                source_line=source_line,
                kind=kind,
                message=f"First non-finite saved activation is {kind} at {label}.",
            )

    if unsaved_compute_ops:
        return BisectNanResult(
            found=False,
            op=None,
            label=None,
            source_line=None,
            kind="none",
            message=(
                "No NaN/Inf found in saved activations; no saved activation for the suspect "
                "region may be available. Re-trace with save=tl.func(...) for the suspect op, "
                "a wider predicate, or no selective save."
            ),
        )
    return BisectNanResult(
        found=False,
        op=None,
        label=None,
        source_line=None,
        kind="none",
        message="No NaN/Inf found in saved activations.",
    )


def _require_pandas() -> Any:
    """Import pandas with the TorchLens tabular-extra error message.

    Returns
    -------
    Any
        Imported pandas module.
    """

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
        ) from e
    return pd


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
    }
    return fields[by]


def hot_path(trace: Trace, by: CostMetric = "flops") -> "pd.DataFrame":
    """Rank source lines by aggregate forward cost.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    by:
        Cost metric: ``"flops"``, ``"memory"``, or ``"duration"``.

    Returns
    -------
    pandas.DataFrame
        Columns are ``source_file:line``, ``op_count``, ``total_cost``, and
        ``pct_total``. The number of ops excluded for missing metrics is stored
        in ``df.attrs["excluded_missing_metric_count"]``.
    """

    pd = _require_pandas()
    field_name = _metric_field(by)
    rows: dict[str, dict[str, float | int | str]] = {}
    excluded = 0
    for op in _ordered_ops(trace):
        if int(getattr(op, "step_index", 0) or 0) <= 0:
            continue
        value = getattr(op, field_name, None)
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

    frame = pd.DataFrame(
        sorted(rows.values(), key=lambda row: float(row["total_cost"]), reverse=True),
        columns=["source_file:line", "op_count", "total_cost", "pct_total"],
    )
    frame.attrs["excluded_missing_metric_count"] = excluded
    frame.attrs["metric"] = by
    return frame


__all__ = ["bisect_nan", "hot_path"]

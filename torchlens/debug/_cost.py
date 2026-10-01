"""Forward-cost ranking helpers for TorchLens traces."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.op import Op
    from torchlens.data_classes.trace import Trace

from ..quantities import Bytes
from ._common import _ordered_ops, _require_pandas, _source_line

CostMetric = Literal["flops", "memory", "duration"]

_NO_COPY_OPS = frozenset(
    {
        "adjoint",
        "alias",
        "as_strided",
        "conj",
        "detach",
        "expand",
        "flatten",
        "imag",
        "movedim",
        "narrow",
        "permute",
        "real",
        "reshape",
        "select",
        "slice",
        "squeeze",
        "swapaxes",
        "swapdims",
        "t",
        "transpose",
        "unflatten",
        "unsqueeze",
        "view",
    }
)


def _normalized_op_name(op: Op) -> str:
    """Return a normalized operation name for cost policy dispatch.

    Parameters
    ----------
    op:
        Operation metadata record.

    Returns
    -------
    str
        Lowercase name without leading/trailing double underscores.
    """

    return str(getattr(op, "func_name", "")).lower().strip("_")


def _metadata_bytes(shape: Any, dtype: Any) -> Bytes | None:
    """Compute dense tensor bytes from shape and native dtype metadata.

    Parameters
    ----------
    shape:
        Tensor shape metadata.
    dtype:
        Native torch dtype metadata.

    Returns
    -------
    Bytes | None
        Dense logical byte count, or ``None`` when metadata is insufficient.
    """

    if shape is None or not isinstance(dtype, torch.dtype):
        return None
    try:
        dimensions = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in dimensions):
            return None
        element_size = int(dtype.itemsize)
    except (AttributeError, TypeError, ValueError):
        return None
    return Bytes(math.prod(dimensions) * element_size)


def _input_bytes(op: Op) -> Bytes | None:
    """Return ideal read-once bytes for distinct graph-parent tensors.

    Parameters
    ----------
    op:
        Operation metadata record.

    Returns
    -------
    Bytes | None
        Sum from parent shape/dtype metadata, or ``None`` when any parent cannot
        be resolved or sized.
    """

    try:
        parents = list(op.input_ops.values())
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return None
    if len(parents) != len(getattr(op, "parents", ())):
        return None
    total = 0
    for parent in parents:
        amount = _metadata_bytes(getattr(parent, "shape", None), getattr(parent, "dtype", None))
        if amount is None:
            return None
        total += int(amount)
    return Bytes(total)


def _parameter_bytes(op: Op) -> Bytes | None:
    """Return ideal read-once bytes for parameters consumed by an operation.

    Parameters
    ----------
    op:
        Operation metadata record.

    Returns
    -------
    Bytes | None
        Sum from parameter shape/dtype metadata, or ``None`` when incomplete.
    """

    shapes = tuple(getattr(op, "param_shapes", ()) or ())
    try:
        dtypes = tuple(op.param_dtypes)
    except (AttributeError, RuntimeError, ValueError):
        return None
    if len(shapes) != len(dtypes):
        return None
    total = 0
    for shape, dtype in zip(shapes, dtypes, strict=True):
        amount = _metadata_bytes(shape, dtype)
        if amount is None:
            return None
        total += int(amount)
    return Bytes(total)


def theoretical_op_bytes(op: Op) -> tuple[Bytes | None, Bytes | None]:
    """Return theoretical ideal read-once/write-once traffic for an operation.

    View and alias operations return zero read and write traffic: this model
    treats them as logical metadata transformations, including ``reshape`` even
    though a particular non-contiguous eager execution can materialize a copy.
    This avoids presenting logical tensor size as measured traffic. For all
    other operations, distinct graph-parent and parameter tensors are read once
    and the dense output is written once. Kernel fusion, caches, allocator
    behavior, and hardware transactions are intentionally outside this model.

    Parameters
    ----------
    op:
        Operation metadata record.

    Returns
    -------
    tuple[Bytes | None, Bytes | None]
        Theoretical bytes read and written. Either side is ``None`` when its
        required shape/dtype metadata is insufficient.
    """

    if _normalized_op_name(op) in _NO_COPY_OPS:
        return Bytes(0), Bytes(0)
    input_bytes = _input_bytes(op)
    parameter_bytes = _parameter_bytes(op)
    bytes_read = (
        None
        if input_bytes is None or parameter_bytes is None
        else Bytes(int(input_bytes) + int(parameter_bytes))
    )
    bytes_written = _metadata_bytes(getattr(op, "shape", None), getattr(op, "dtype", None))
    return bytes_read, bytes_written


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


def hot_path_rows(
    trace: Trace, by: CostMetric = "flops"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pandas-free core of :func:`hot_path` (agent stage-0 item 4).

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    by:
        Cost metric: ``"flops"``, ``"memory"``, or ``"duration"``.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Cost-ranked row dicts and the attrs mapping (excluded count, metric,
        capture-honesty facts).
    """

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

    from .._capture_honesty import capture_honesty_facts

    sorted_rows = sorted(rows.values(), key=lambda row: float(row["total_cost"]), reverse=True)
    attrs: dict[str, Any] = {
        "excluded_missing_metric_count": excluded,
        "metric": by,
        "torchlens_capture_honesty": capture_honesty_facts(trace),
    }
    return [dict(row) for row in sorted_rows], attrs


def hot_path(trace: Trace, by: CostMetric = "flops") -> pd.DataFrame:
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
    return frame

"""Graph lineage, comparison, and activation-analysis helpers."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from torchlens._errors import ShapeInferenceError

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.op import Op
    from torchlens.data_classes.trace import Trace

from ._common import (
    _compute_ops,
    _op_from_label,
    _op_label,
    _resolve_op,
    _safe_out,
    _shape_dtype,
    _source_line,
    _tensor_unavailable_reason,
    _require_pandas,
)

LineageDirection = Literal["ancestors", "descendants", "both"]


@dataclass(frozen=True)
class LineageResult:
    """Result returned by :func:`lineage`.

    Parameters
    ----------
    start_label:
        Resolved starting op label, or the requested label when lookup failed.
    direction:
        Traversal direction.
    nodes:
        Tuples of ``(label, depth, source_line, shape, dtype)``.
    message:
        Human-readable status.
    """

    start_label: str | None
    direction: str
    nodes: list[tuple[str, int, str | None, tuple[int, ...] | None, str | None]]
    message: str


def lineage(
    trace: Trace,
    op_or_label: Any,
    *,
    direction: str = "ancestors",
    max_depth: int | None = None,
) -> LineageResult:
    """Walk graph lineage for a completed trace.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    op_or_label:
        Start op or lookup key accepted by ``trace.__getitem__``.
    direction:
        ``"ancestors"``, ``"descendants"``, or ``"both"``.
    max_depth:
        Optional maximum graph distance from the start op.

    Returns
    -------
    LineageResult
        Traversal result with unavailable lookups reported in ``message``.
    """

    start_op, error = _resolve_op(trace, op_or_label)
    if direction not in {"ancestors", "descendants", "both"}:
        requested = str(getattr(op_or_label, "label", op_or_label))
        return LineageResult(
            start_label=requested,
            direction=direction,
            nodes=[],
            message="unavailable: direction must be 'ancestors', 'descendants', or 'both'",
        )
    if start_op is None:
        return LineageResult(
            start_label=str(op_or_label),
            direction=direction,
            nodes=[],
            message=error or "unavailable",
        )

    def neighbor_labels(op: Op) -> Iterable[str]:
        """Return graph-neighbor labels for the requested direction."""

        if direction == "ancestors":
            return getattr(op, "parents", ()) or ()
        if direction == "descendants":
            return getattr(op, "children", ()) or ()
        return tuple(getattr(op, "parents", ()) or ()) + tuple(getattr(op, "children", ()) or ())

    start_label = _op_label(start_op)
    queue: list[tuple[Op, int]] = [(start_op, 0)]
    visited = {start_label}
    nodes: list[tuple[str, int, str | None, tuple[int, ...] | None, str | None]] = []
    while queue:
        op, depth = queue.pop(0)
        shape, dtype = _shape_dtype(op)
        nodes.append((_op_label(op), depth, _source_line(op), shape, dtype))
        if max_depth is not None and depth >= max_depth:
            continue
        for label in neighbor_labels(op):
            child_op = _op_from_label(trace, str(label))
            visit_label = _op_label(child_op) if child_op is not None else str(label)
            if visit_label in visited:
                continue
            visited.add(visit_label)
            if child_op is None:
                nodes.append((visit_label, depth + 1, None, None, None))
                continue
            queue.append((child_op, depth + 1))

    return LineageResult(
        start_label=start_label,
        direction=direction,
        nodes=nodes,
        message=f"{len(nodes)} node(s)",
    )


def _activation_row(label: str, status: str, reason: str) -> dict[str, Any]:
    """Build a minimal activation-unavailable comparison row.

    Parameters
    ----------
    label:
        Op label.
    status:
        Presence status.
    reason:
        Unavailable reason.

    Returns
    -------
    dict[str, Any]
        Row dictionary.
    """

    return {
        "op": label,
        "status": status,
        "shape_match": None,
        "dtype_match": None,
        "max_abs": None,
        "mean_abs": None,
        "allclose": None,
        "reason": reason,
    }


def compare(
    trace_a: Trace,
    trace_b: Trace,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> "pd.DataFrame":
    """Compare saved dense floating activations across two traces.

    Parameters
    ----------
    trace_a:
        First completed TorchLens trace.
    trace_b:
        Second completed TorchLens trace.
    rtol:
        Relative tolerance for ``torch.allclose``.
    atol:
        Absolute tolerance for ``torch.allclose``.

    Returns
    -------
    pandas.DataFrame
        One row per pass-qualified op with summary counts in ``df.attrs``.
    """

    pd = _require_pandas()
    ops_a = {_op_label(op): op for op in _compute_ops(trace_a)}
    ops_b = {_op_label(op): op for op in _compute_ops(trace_b)}
    labels = sorted(set(ops_a) | set(ops_b))
    rows: list[dict[str, Any]] = []
    summary = {
        "matched": 0,
        "shape_mismatch": 0,
        "only_a": 0,
        "only_b": 0,
        "value_diverged": 0,
        "activation_unavailable": 0,
    }

    for label in labels:
        op_a = ops_a.get(label)
        op_b = ops_b.get(label)
        if op_a is None:
            summary["only_b"] += 1
            rows.append(_activation_row(label, "only-b", "only-b"))
            continue
        if op_b is None:
            summary["only_a"] += 1
            rows.append(_activation_row(label, "only-a", "only-a"))
            continue

        out_a, reason_a = _safe_out(op_a)
        out_b, reason_b = _safe_out(op_b)
        reason_a = reason_a or _tensor_unavailable_reason(out_a)
        reason_b = reason_b or _tensor_unavailable_reason(out_b)
        shape_match = getattr(op_a, "shape", None) == getattr(op_b, "shape", None)
        dtype_match = getattr(op_a, "dtype", None) == getattr(op_b, "dtype", None)
        row = {
            "op": label,
            "status": "present-in-both",
            "shape_match": shape_match,
            "dtype_match": dtype_match,
            "max_abs": None,
            "mean_abs": None,
            "allclose": None,
            "reason": "",
        }
        if not shape_match:
            summary["shape_mismatch"] += 1
        if reason_a is not None or reason_b is not None:
            summary["activation_unavailable"] += 1
            row["reason"] = f"a={reason_a or 'ok'}; b={reason_b or 'ok'}"
            rows.append(row)
            continue
        if not isinstance(out_a, torch.Tensor) or not isinstance(out_b, torch.Tensor):
            summary["activation_unavailable"] += 1
            row["reason"] = "non-tensor/container"
            rows.append(row)
            continue
        if out_a.device != out_b.device:
            summary["activation_unavailable"] += 1
            row["reason"] = "device-mismatch"
            rows.append(row)
            continue
        if out_a.shape != out_b.shape or out_a.dtype != out_b.dtype:
            summary["activation_unavailable"] += 1
            row["reason"] = "shape-or-dtype-mismatch"
            rows.append(row)
            continue

        delta = torch.abs(out_a.detach() - out_b.detach())
        max_abs = float(delta.max().item()) if delta.numel() else 0.0
        mean_abs = float(delta.mean().item()) if delta.numel() else 0.0
        allclose = bool(torch.allclose(out_a, out_b, rtol=rtol, atol=atol))
        if allclose:
            summary["matched"] += 1
        else:
            summary["value_diverged"] += 1
        row.update({"max_abs": max_abs, "mean_abs": mean_abs, "allclose": allclose})
        rows.append(row)

    frame = pd.DataFrame(
        rows,
        columns=[
            "op",
            "status",
            "shape_match",
            "dtype_match",
            "max_abs",
            "mean_abs",
            "allclose",
            "reason",
        ],
    )
    frame.attrs.update(summary)
    frame.attrs["rtol"] = rtol
    frame.attrs["atol"] = atol
    return frame


def dead_neurons(trace: Trace, *, dim: int = 1, threshold: float = 0.0) -> "pd.DataFrame":
    """Find units that are inactive or zero-variance in one completed trace.

    A single trace is a single example; zero-variance here is an insufficient-sample
    signal, not dataset-level neuron death. Aggregate multiple traces when deciding
    whether units are dead over a dataset.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    dim:
        Feature dimension.
    threshold:
        Maximum activation value for post-activation death.

    Returns
    -------
    pandas.DataFrame
        Columns are ``op``, ``total_units``, ``dead_count``, ``dead_frac``,
        ``sample_dead_idx``, and ``reason``.
    """

    pd = _require_pandas()
    rows: list[dict[str, Any]] = []
    skipped = 0
    for op in _compute_ops(trace):
        label = _op_label(op)
        out, reason = _safe_out(op)
        reason = reason or _tensor_unavailable_reason(out)
        if reason is not None:
            skipped += 1
            rows.append(
                {
                    "op": label,
                    "total_units": None,
                    "dead_count": None,
                    "dead_frac": None,
                    "sample_dead_idx": [],
                    "reason": reason,
                }
            )
            continue
        if not isinstance(out, torch.Tensor):
            skipped += 1
            continue
        feature_dim = dim if dim >= 0 else out.ndim + dim
        if feature_dim < 0 or feature_dim >= out.ndim:
            skipped += 1
            rows.append(
                {
                    "op": label,
                    "total_units": None,
                    "dead_count": None,
                    "dead_frac": None,
                    "sample_dead_idx": [],
                    "reason": "invalid-dim",
                }
            )
            continue
        reduce_dims = tuple(index for index in range(out.ndim) if index != feature_dim)
        if reduce_dims:
            max_by_unit = out.detach().amax(dim=reduce_dims)
            var_by_unit = out.detach().var(dim=reduce_dims, unbiased=False)
        else:
            max_by_unit = out.detach()
            var_by_unit = torch.zeros_like(out.detach())
        dead_mask = (max_by_unit <= threshold) | (var_by_unit == 0)
        dead_indices = torch.nonzero(dead_mask, as_tuple=False).flatten().tolist()
        total_units = int(out.shape[feature_dim])
        dead_count = len(dead_indices)
        rows.append(
            {
                "op": label,
                "total_units": total_units,
                "dead_count": dead_count,
                "dead_frac": 0.0 if total_units == 0 else dead_count / total_units,
                "sample_dead_idx": dead_indices[:10],
                "reason": "",
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=["op", "total_units", "dead_count", "dead_frac", "sample_dead_idx", "reason"],
    )
    frame.attrs["skipped"] = skipped
    frame.attrs["threshold"] = threshold
    frame.attrs["dim"] = dim
    frame.attrs["note"] = "single-trace zero-variance is insufficient sample"
    return frame

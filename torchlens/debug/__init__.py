"""Power-user debugging helpers for completed TorchLens traces."""

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


CostMetric = Literal["flops", "memory", "duration"]
LineageDirection = Literal["ancestors", "descendants", "both"]


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


def _op_label(op: Op) -> str:
    """Return the stable pass-qualified label for an op.

    Parameters
    ----------
    op:
        TorchLens op.

    Returns
    -------
    str
        Op label.
    """

    return str(getattr(op, "label", None) or getattr(op, "layer_label", ""))


def _compute_ops(trace: Trace) -> list[Op]:
    """Return pass-qualified compute ops by the debug module's local convention.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.

    Returns
    -------
    list[Op]
        Ops whose ``step_index`` is positive.
    """

    return [op for op in _ordered_ops(trace) if int(getattr(op, "step_index", 0) or 0) > 0]


def _resolve_op(trace: Trace, op_or_label: Any) -> tuple[Op | None, str | None]:
    """Resolve an op-like object or lookup key without raising.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    op_or_label:
        Op object or key accepted by ``trace.__getitem__``.

    Returns
    -------
    tuple[Op | None, str | None]
        Resolved op and error message.
    """

    if hasattr(op_or_label, "parents") and hasattr(op_or_label, "children"):
        return op_or_label, None
    try:
        resolved = trace[op_or_label]
    except Exception as exc:  # noqa: BLE001 - debug helpers report odd inputs instead of raising.
        return None, f"unavailable: {exc}"
    if hasattr(resolved, "parents") and hasattr(resolved, "children"):
        return resolved, None
    ops = getattr(resolved, "ops", None)
    if ops is not None:
        try:
            first_op = next(iter(ops.values()))
        except (AttributeError, StopIteration):
            return None, f"unavailable: {op_or_label!r} did not resolve to an op"
        return first_op, None
    return None, f"unavailable: {op_or_label!r} did not resolve to an op"


def _safe_out(op: Op) -> tuple[Any | None, str | None]:
    """Read ``op.out`` and convert known unavailable cases into a reason.

    Parameters
    ----------
    op:
        TorchLens op.

    Returns
    -------
    tuple[Any | None, str | None]
        Payload and unavailable reason.
    """

    if not bool(getattr(op, "has_saved_activation", False)):
        return None, "unsaved"
    try:
        return op.out, None
    except ValueError as exc:
        return None, str(exc)


def _tensor_unavailable_reason(value: Any) -> str | None:
    """Return why a value is not a usable dense floating tensor.

    Parameters
    ----------
    value:
        Candidate activation or gradient payload.

    Returns
    -------
    str | None
        Reason string, or ``None`` when the tensor is usable.
    """

    if not isinstance(value, torch.Tensor):
        return "non-tensor/container"
    if bool(getattr(value, "is_meta", False)):
        return "meta"
    if bool(getattr(value, "is_sparse", False)):
        return "sparse"
    if bool(getattr(value, "is_quantized", False)):
        return "quantized"
    if torch.is_complex(value):
        return "complex"
    if not torch.is_floating_point(value):
        return "non-floating"
    return None


def _shape_dtype(op: Op) -> tuple[tuple[int, ...] | None, str | None]:
    """Return shape and dtype metadata without materializing payloads.

    Parameters
    ----------
    op:
        TorchLens op.

    Returns
    -------
    tuple[tuple[int, ...] | None, str | None]
        Shape tuple and dtype string when available.
    """

    shape = getattr(op, "shape", None)
    dtype = getattr(op, "dtype", None)
    return shape, None if dtype is None else str(dtype)


def _op_from_label(trace: Trace, label: str) -> Op | None:
    """Resolve an edge label to an op, returning ``None`` when unavailable.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    label:
        Edge label from ``Op.parents`` or ``Op.children``.

    Returns
    -------
    Op | None
        Resolved op.
    """

    op, _ = _resolve_op(trace, label)
    return op


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


def _empty_grad_frame(message: str, *, pd: Any, **attrs: Any) -> "pd.DataFrame":
    """Build an empty gradient audit frame with attrs.

    Parameters
    ----------
    message:
        Human-readable status.
    pd:
        Imported pandas module.
    attrs:
        Additional attrs.

    Returns
    -------
    pandas.DataFrame
        Empty audit result.
    """

    frame = pd.DataFrame(
        columns=["op", "grad_norm", "vanishing", "exploding", "dead", "severity", "reason"]
    )
    frame.attrs["message"] = message
    frame.attrs.update(attrs)
    return frame


def gradient_flow_audit(
    trace: Trace,
    *,
    bwd: int | None = None,
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 1e4,
) -> "pd.DataFrame":
    """Audit saved op gradients for vanishing, exploding, and zero gradients.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    bwd:
        One-based backward pass selector. Required when multiple backward passes
        are captured.
    vanishing_threshold:
        Norm below which a nonzero finite gradient is flagged vanishing.
    exploding_threshold:
        Norm above which a finite gradient is flagged exploding.

    Returns
    -------
    pandas.DataFrame
        Ranked audit rows with counts in ``df.attrs``.
    """

    pd = _require_pandas()
    try:
        backward_passes = trace.backward_passes
        saved_grad_ops = trace.saved_grad_ops
    except ValueError:
        return _empty_grad_frame("torch-only", pd=pd, torch_only=True)

    num_backward = len(backward_passes)
    if num_backward == 0 or len(saved_grad_ops) == 0:
        return _empty_grad_frame(
            "no saved gradients; re-trace backward_ready=True + trace.log_backward(loss)",
            pd=pd,
            vanishing=0,
            exploding=0,
            dead=0,
        )
    if num_backward > 1 and bwd is None:
        return _empty_grad_frame(
            "bwd is required for multi-backward-pass traces",
            pd=pd,
            backward_passes=num_backward,
        )
    selected_bwd = bwd if bwd is not None else 1

    rows: list[dict[str, Any]] = []
    counts = {"vanishing": 0, "exploding": 0, "dead": 0, "unavailable": 0}
    for op in saved_grad_ops:
        label = _op_label(op)
        try:
            grad = op.grad_for(bwd=selected_bwd)
        except (KeyError, ValueError) as exc:
            counts["unavailable"] += 1
            rows.append(
                {
                    "op": label,
                    "grad_norm": None,
                    "vanishing": False,
                    "exploding": False,
                    "dead": False,
                    "severity": 0,
                    "reason": str(exc),
                }
            )
            continue
        reason = _tensor_unavailable_reason(grad)
        if reason is not None:
            counts["unavailable"] += 1
            rows.append(
                {
                    "op": label,
                    "grad_norm": None,
                    "vanishing": False,
                    "exploding": False,
                    "dead": False,
                    "severity": 0,
                    "reason": reason,
                }
            )
            continue
        if not isinstance(grad, torch.Tensor):
            counts["unavailable"] += 1
            continue
        norm_tensor = torch.linalg.vector_norm(grad.detach())
        grad_norm = float(norm_tensor.item())
        finite = bool(torch.isfinite(norm_tensor).item())
        dead = finite and grad_norm == 0.0
        vanishing = finite and 0.0 < grad_norm < vanishing_threshold
        exploding = (not finite) or grad_norm > exploding_threshold
        counts["dead"] += int(dead)
        counts["vanishing"] += int(vanishing)
        counts["exploding"] += int(exploding)
        severity = int(exploding) * 3 + int(dead) * 2 + int(vanishing)
        rows.append(
            {
                "op": label,
                "grad_norm": grad_norm,
                "vanishing": vanishing,
                "exploding": exploding,
                "dead": dead,
                "severity": severity,
                "reason": "",
            }
        )

    frame = pd.DataFrame(
        sorted(rows, key=lambda row: (int(row["severity"]), str(row["op"])), reverse=True),
        columns=["op", "grad_norm", "vanishing", "exploding", "dead", "severity", "reason"],
    )
    frame.attrs.update(counts)
    frame.attrs["bwd"] = selected_bwd
    frame.attrs["vanishing_threshold"] = vanishing_threshold
    frame.attrs["exploding_threshold"] = exploding_threshold
    return frame


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


FailureReason = Literal[
    "non_shape_blocker",
    "exact_size_unreachable",
    "multi_input_unsupported",
    "budget_exhausted",
    "unknown_entry",
]


@dataclass(frozen=True)
class InferInputShapeResult:
    """Result returned by :func:`infer_input_shape`.

    Parameters
    ----------
    found:
        Whether a verified input was found.
    shape:
        Single tensor input shape, when applicable.
    shapes:
        Multi-input shapes, when applicable.
    dtype:
        Input dtype for the primary tensor.
    value_range:
        Synthetic value recipe: ``("uniform", 0, 1)`` or ``("randint", 0, vocab)``.
    flexible_dims:
        Dimensions that are expected to tolerate other sizes.
    constraining_module:
        Module qualname that constrained the inferred shape, when known.
    constraining_op:
        Operation name that constrained the inferred shape, when known.
    source_line:
        Source line for the constraining op, when known.
    example_input:
        Ready-to-use input object.
    strategy:
        Strategy that produced the winning input.
    reason:
        Failure reason when ``found`` is false.
    attempts:
        Probe diary as ``(shape, outcome)`` tuples.
    trace:
        Final verification trace when ``return_trace=True``.
    message:
        Actionable human-readable summary.
    """

    found: bool
    shape: tuple[int, ...] | None
    shapes: tuple[tuple[int, ...], ...] | None
    dtype: torch.dtype | None
    value_range: tuple[str, float, float] | None
    flexible_dims: tuple[int, ...]
    constraining_module: str | None
    constraining_op: str | None
    source_line: str | None
    example_input: Any | None
    strategy: str
    reason: str | None
    attempts: tuple[tuple[tuple[int, ...] | None, str], ...]
    trace: Trace | None
    message: str


@dataclass(frozen=True)
class _InputPrior:
    """Static facts inferred without running the model."""

    kind: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    value_range: tuple[str, float, float]
    flexible_dims: tuple[int, ...]
    constraining_module: str | None
    constraining_op: str | None
    strategy: str
    device: torch.device
    spatial_rank: int | None = None
    channels: int | None = None
    min_side: int = 1


@dataclass(frozen=True)
class _ProbeResult:
    """Outcome of one synthetic forward probe."""

    ok: bool
    outcome: str
    exception: Exception | None
    diary: tuple[tuple[str, tuple[tuple[int, ...], ...], tuple[str, ...]], ...]
    got_features: int | None
    target_features: int | None
    constraining_module: str | None


@dataclass(frozen=True)
class _ExecutedConstraint:
    """Constraint read from an executed TorchLens op."""

    kind: str
    label: str | None
    module: str | None
    source_line: str | None
    input_shape: tuple[int, ...] | None
    dtype: torch.dtype | None
    in_features: int | None = None
    in_channels: int | None = None
    spatial_rank: int | None = None
    kernel_size: tuple[int, ...] = ()
    flexible: bool = False


_LINEAR_RE = re.compile(
    r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)"
)
_CHANNEL_RE = re.compile(r"expected input\[[^\]]+\] to have (\d+) channels, but got (\d+) channels")
_BOOL_RE = re.compile(r"mask|bool|boolean", re.IGNORECASE)
_COMPLEX_RE = re.compile(r"complex|ComplexFloat|ComplexDouble", re.IGNORECASE)
_INTEGER_RE = re.compile(r"Long|Int|integer|indices", re.IGNORECASE)
_KERNEL_RE = re.compile(r"kernel size|calculated padded input size", re.IGNORECASE)
_SIZE_RE = re.compile(r"size of tensor|shape|shapes|Expected|expected|dimension|mat1 and mat2")
_POSITION_RE = re.compile(
    r"(pos_embed|position_embeddings|positional|position|grid)", re.IGNORECASE
)


def _shape_tuple(shape: Any) -> tuple[int, ...] | None:
    """Convert a Torch/TorchLens shape-like object to an integer tuple.

    Parameters
    ----------
    shape:
        Shape-like object.

    Returns
    -------
    tuple[int, ...] | None
        Converted shape, or ``None`` when conversion is not possible.
    """

    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _dtype_from_message(message: str) -> torch.dtype | None:
    """Infer a replacement dtype from a probe error message.

    Parameters
    ----------
    message:
        Exception message from a failed probe.

    Returns
    -------
    torch.dtype | None
        Suggested dtype, or ``None`` when the message is not dtype-specific.
    """

    if _BOOL_RE.search(message):
        return torch.bool
    if _COMPLEX_RE.search(message):
        return torch.complex64
    if _INTEGER_RE.search(message):
        return torch.long
    return None


def _is_skippable_shape_error(message: str) -> bool:
    """Return whether a failed probe should still allow other size probes.

    Parameters
    ----------
    message:
        Exception message.

    Returns
    -------
    bool
        Whether the error looks like a shape/rank/size miss.
    """

    return bool(
        _KERNEL_RE.search(message) or _SIZE_RE.search(message) or _CHANNEL_RE.search(message)
    )


def _rank_default(rank: int) -> int:
    """Return a conservative default side for a spatial rank.

    Parameters
    ----------
    rank:
        Spatial rank.

    Returns
    -------
    int
        Default side length.
    """

    return 16 if rank == 3 else 32


def _module_index(model: nn.Module) -> dict[nn.Module, int]:
    """Return definition-order indices for modules.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    dict[nn.Module, int]
        Module object to definition-order index.
    """

    return {module: index for index, module in enumerate(model.modules())}


def _module_device_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype | None]:
    """Return the first parameter or buffer device and dtype.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    tuple[torch.device, torch.dtype | None]
        Device and dtype, defaulting to CPU with no dtype when the model has no state.
    """

    for tensor in list(model.parameters(recurse=True)) + list(model.buffers(recurse=True)):
        return tensor.device, tensor.dtype
    return torch.device("cpu"), None


def _float_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Normalize a model dtype to a synthetic floating input dtype.

    Parameters
    ----------
    dtype:
        Model parameter dtype.

    Returns
    -------
    torch.dtype
        Floating dtype for generated inputs.
    """

    if dtype in {torch.float16, torch.bfloat16, torch.float64, torch.float32}:
        return dtype
    return torch.float32


def _first_module(
    model: nn.Module,
    classes: tuple[type[nn.Module], ...],
) -> tuple[str, nn.Module] | None:
    """Return the first named module matching any requested class.

    Parameters
    ----------
    model:
        Model to inspect.
    classes:
        Module classes to match.

    Returns
    -------
    tuple[str, nn.Module] | None
        Qualname and module, or ``None``.
    """

    for name, module in model.named_modules():
        if isinstance(module, classes):
            return name, module
    return None


def _has_adaptive_pool(model: nn.Module) -> bool:
    """Return whether the model contains an adaptive pooling module.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    bool
        Whether an adaptive pooling module is present.
    """

    return any(
        isinstance(
            module,
            (
                nn.AdaptiveAvgPool1d,
                nn.AdaptiveAvgPool2d,
                nn.AdaptiveAvgPool3d,
                nn.AdaptiveMaxPool1d,
                nn.AdaptiveMaxPool2d,
                nn.AdaptiveMaxPool3d,
            ),
        )
        for module in model.modules()
    )


def _shape_has_executed_adaptive_pool(trace: Trace) -> bool:
    """Return whether the executed trace used adaptive pooling.

    Parameters
    ----------
    trace:
        Successful TorchLens trace.

    Returns
    -------
    bool
        Whether an adaptive pooling op executed.
    """

    return any(
        "adaptive" in str(getattr(op, "func_name", "")).lower()
        and "pool" in str(getattr(op, "func_name", "")).lower()
        for op in trace.layers
    )


def _positional_side(model: nn.Module, stride: int) -> int | None:
    """Infer a square ViT image side from positional embeddings.

    Parameters
    ----------
    model:
        Model to inspect.
    stride:
        Patch stride.

    Returns
    -------
    int | None
        Inferred side length, or ``None``.
    """

    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.ndim < 2 or not _POSITION_RE.search(name):
            continue
        length = int(tensor.shape[1] if tensor.ndim >= 3 else tensor.shape[0])
        for offset in (2, 1, 0):
            patches = length - offset
            root = math.isqrt(patches)
            if root * root == patches and root > 0:
                return root * stride
    return None


def _positional_side_from_any_table(model: nn.Module, stride: int) -> int | None:
    """Infer a square ViT side from any plausible position table.

    Parameters
    ----------
    model:
        Model to inspect.
    stride:
        Patch stride.

    Returns
    -------
    int | None
        Inferred side length, or ``None``.
    """

    named_tensors = list(model.named_parameters()) + list(model.named_buffers())
    preferred = [(name, tensor) for name, tensor in named_tensors if _POSITION_RE.search(name)]
    fallback = [
        (name, tensor)
        for name, tensor in named_tensors
        if tensor.ndim == 3 and int(tensor.shape[0]) == 1 and int(tensor.shape[-1]) > 1
    ]
    for _name, tensor in [*preferred, *fallback]:
        length = int(tensor.shape[1] if tensor.ndim >= 3 else tensor.shape[0])
        for offset in (2, 1, 0):
            patches = length - offset
            root = math.isqrt(patches)
            if root * root == patches and root > 0:
                return root * stride
    return None


def _make_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    value_range: tuple[str, float, float],
) -> torch.Tensor:
    """Create a synthetic input tensor.

    Parameters
    ----------
    shape:
        Tensor shape.
    dtype:
        Tensor dtype.
    device:
        Tensor device.
    value_range:
        Synthetic value recipe.

    Returns
    -------
    torch.Tensor
        Generated tensor.
    """

    if dtype == torch.bool:
        return torch.rand(shape, device=device) > 0.5
    if dtype.is_complex:
        real = torch.randn(shape, device=device, dtype=torch.float32)
        imag = torch.randn(shape, device=device, dtype=torch.float32)
        return torch.complex(real, imag).to(dtype)
    if value_range[0] == "randint" or dtype in {
        torch.long,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
    }:
        return torch.randint(
            int(value_range[1]), int(value_range[2]), shape, device=device, dtype=dtype
        )
    return torch.rand(shape, device=device, dtype=dtype)


def _training_states(model: nn.Module) -> dict[nn.Module, bool]:
    """Capture training flags for all modules.

    Parameters
    ----------
    model:
        Model whose module states are captured.

    Returns
    -------
    dict[nn.Module, bool]
        Training flags keyed by module object.
    """

    return {module: module.training for module in model.modules()}


def _restore_training_states(states: dict[nn.Module, bool]) -> None:
    """Restore module training flags.

    Parameters
    ----------
    states:
        Training flags from :func:`_training_states`.
    """

    for module, training in states.items():
        module.train(training)


def _probe(model: nn.Module, example_input: Any, seed: int) -> _ProbeResult:
    """Run a no-grad forward probe with pre-hook shape diary.

    Parameters
    ----------
    model:
        Model to run.
    example_input:
        Input object passed to ``model``.
    seed:
        RNG seed.

    Returns
    -------
    _ProbeResult
        Probe outcome and diary.
    """

    diary: list[tuple[str, tuple[tuple[int, ...], ...], tuple[str, ...]]] = []
    handles: list[Any] = []

    def hook(name: str) -> Any:
        """Build a pre-hook that records incoming tensor metadata."""

        def record(_module: nn.Module, args: tuple[Any, ...]) -> None:
            """Record tensor shapes and dtypes from positional module inputs."""

            tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
            diary.append(
                (
                    name,
                    tuple(tuple(int(dim) for dim in tensor.shape) for tensor in tensors),
                    tuple(str(tensor.dtype) for tensor in tensors),
                )
            )

        return record

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_pre_hook(hook(name)))

    states = _training_states(model)
    got_features: int | None = None
    target_features: int | None = None
    constraining_module: str | None = None
    try:
        torch.manual_seed(seed)
        model.eval()
        with torch.no_grad():
            if isinstance(example_input, tuple):
                model(*example_input)
            elif isinstance(example_input, dict):
                model(**example_input)
            else:
                model(example_input)
    except Exception as exc:  # noqa: BLE001 - debug inference classifies arbitrary model failures.
        message = str(exc)
        match = _LINEAR_RE.search(message)
        if match is not None:
            got_features = int(match.group(2))
            target_features = int(match.group(3))
        for name, shapes, _dtypes in reversed(diary):
            module = dict(model.named_modules()).get(name)
            if isinstance(module, nn.Linear):
                constraining_module = name
                if shapes:
                    got_features = int(shapes[0][-1])
                target_features = int(module.in_features)
                break
        return _ProbeResult(
            ok=False,
            outcome=message,
            exception=exc,
            diary=tuple(diary),
            got_features=got_features,
            target_features=target_features,
            constraining_module=constraining_module,
        )
    finally:
        for handle in handles:
            handle.remove()
        _restore_training_states(states)

    return _ProbeResult(
        ok=True,
        outcome="ok",
        exception=None,
        diary=tuple(diary),
        got_features=None,
        target_features=None,
        constraining_module=None,
    )


def _shape_from_prior(
    model: nn.Module,
    batch_size: int,
    input_dtype: torch.dtype | None,
    channels: int | None,
    spatial_rank: int | Literal["auto"],
    seq_len: int | None,
    min_size: int,
    preferred_sizes: Sequence[int],
    device: torch.device,
) -> _InputPrior | None:
    """Build the best static single-input prior.

    Parameters
    ----------
    model:
        Model to inspect.
    batch_size:
        Batch dimension.
    input_dtype:
        Optional dtype override.
    channels:
        Optional channel override.
    spatial_rank:
        Optional spatial rank override.
    seq_len:
        Optional sequence length override.
    min_size:
        Minimum free dimension.
    preferred_sizes:
        Preferred side lengths.
    device:
        Target device.

    Returns
    -------
    _InputPrior | None
        Static prior, or ``None`` when no known entry is found.
    """

    _model_device, model_dtype = _module_device_dtype(model)
    emb = _first_module(model, (nn.Embedding,))
    if emb is not None:
        name, module = emb
        assert isinstance(module, nn.Embedding)
        cap = _sequence_cap(model)
        size = seq_len or min(16, cap or 16)
        return _InputPrior(
            kind="embedding",
            shape=(batch_size, size),
            dtype=input_dtype or torch.long,
            value_range=("randint", 0.0, float(module.num_embeddings)),
            flexible_dims=(1,),
            constraining_module=name,
            constraining_op="Embedding",
            strategy="introspection",
            device=device,
        )

    rnn = _first_module(model, (nn.RNN, nn.LSTM, nn.GRU))
    conv = _first_module(model, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
    linear = _first_module(model, (nn.Linear,))
    if rnn is not None and (
        conv is None or list(model.modules()).index(rnn[1]) < list(model.modules()).index(conv[1])
    ):
        name, module = rnn
        assert isinstance(module, (nn.RNN, nn.LSTM, nn.GRU))
        size = seq_len or 16
        shape = (batch_size, size, int(module.input_size))
        if not bool(module.batch_first):
            shape = (size, batch_size, int(module.input_size))
        return _InputPrior(
            kind="rnn",
            shape=shape,
            dtype=input_dtype or _float_dtype(model_dtype),
            value_range=("uniform", 0.0, 1.0),
            flexible_dims=(1 if bool(module.batch_first) else 0,),
            constraining_module=name,
            constraining_op=module.__class__.__name__,
            strategy="introspection",
            device=device,
        )

    if conv is not None:
        name, module = conv
        assert isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        rank = (
            int(spatial_rank)
            if spatial_rank != "auto"
            else {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}[type(module)]
        )
        channel_count = channels or int(module.in_channels)
        first_preferred = _rank_default(rank) if rank == 3 else next(iter(preferred_sizes), 32)
        side = max(min_size, first_preferred)
        stride = module.stride[0] if isinstance(module.stride, tuple) else int(module.stride)
        side = _positional_side(model, int(stride)) or side
        strategy = "fixed_size" if _positional_side(model, int(stride)) else "adaptive_default"
        flexible = tuple(range(2, 2 + rank)) if _has_adaptive_pool(model) else ()
        return _InputPrior(
            kind="conv",
            shape=(batch_size, channel_count, *([side] * rank)),
            dtype=input_dtype or _float_dtype(model_dtype),
            value_range=("uniform", 0.0, 1.0),
            flexible_dims=flexible,
            constraining_module=name,
            constraining_op=module.__class__.__name__,
            strategy=strategy,
            device=device,
            spatial_rank=rank,
            channels=channel_count,
        )

    if linear is not None:
        name, module = linear
        assert isinstance(module, nn.Linear)
        return _InputPrior(
            kind="linear",
            shape=(batch_size, int(module.in_features)),
            dtype=input_dtype or _float_dtype(model_dtype),
            value_range=("uniform", 0.0, 1.0),
            flexible_dims=(),
            constraining_module=name,
            constraining_op="Linear",
            strategy="introspection",
            device=device,
        )
    return None


def _parameter_priors(
    model: nn.Module,
    batch_size: int,
    input_dtype: torch.dtype | None,
    seq_len: int | None,
    min_size: int,
    preferred_sizes: Sequence[int],
    device: torch.device,
) -> list[_InputPrior]:
    """Build functional-op priors from raw parameter shapes.

    Parameters
    ----------
    model:
        Model to inspect.
    batch_size:
        Batch dimension.
    input_dtype:
        Optional dtype override.
    seq_len:
        Optional sequence length override.
    min_size:
        Minimum spatial side.
    preferred_sizes:
        Preferred spatial sizes.
    device:
        Target device.

    Returns
    -------
    list[_InputPrior]
        Candidate priors for functional models.
    """

    priors: list[_InputPrior] = []
    _model_device, model_dtype = _module_device_dtype(model)
    cap = _sequence_cap(model)
    for name, param in model.named_parameters():
        if param.ndim in {3, 4, 5}:
            rank = int(param.ndim - 2)
            channels = int(param.shape[1])
            kernel = tuple(int(dim) for dim in param.shape[2:])
            default_side = _rank_default(rank)
            first_preferred = next(iter(preferred_sizes), default_side)
            side = max(min_size, max(max(kernel), default_side if rank == 3 else first_preferred))
            if rank == 2:
                side = _positional_side_from_any_table(model, max(kernel)) or side
            priors.append(
                _InputPrior(
                    kind="conv",
                    shape=(batch_size, channels, *([side] * rank)),
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(),
                    constraining_module=None,
                    constraining_op=f"parameter:{name}",
                    strategy="op_seed",
                    device=device,
                    spatial_rank=rank,
                    channels=channels,
                    min_side=max(kernel),
                )
            )
        elif param.ndim == 2:
            rows, cols = int(param.shape[0]), int(param.shape[1])
            if rows > cols and ("embed" in name.lower() or "token" in name.lower()):
                size = seq_len or min(16, cap or 16)
                priors.append(
                    _InputPrior(
                        kind="embedding",
                        shape=(batch_size, size),
                        dtype=input_dtype or torch.long,
                        value_range=("randint", 0.0, float(rows)),
                        flexible_dims=(1,),
                        constraining_module=None,
                        constraining_op=f"parameter:{name}",
                        strategy="op_seed",
                        device=device,
                    )
                )
            priors.append(
                _InputPrior(
                    kind="linear",
                    shape=(batch_size, cols),
                    dtype=input_dtype
                    or _float_dtype(
                        param.dtype
                        if param.dtype.is_floating_point or param.dtype.is_complex
                        else model_dtype
                    ),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(),
                    constraining_module=None,
                    constraining_op=f"parameter:{name}",
                    strategy="op_seed",
                    device=device,
                )
            )
    return priors


def _input_priors(
    model: nn.Module,
    batch_size: int,
    input_dtype: torch.dtype | None,
    channels: int | None,
    spatial_rank: int | Literal["auto"],
    seq_len: int | None,
    min_size: int,
    preferred_sizes: Sequence[int],
    device: torch.device,
) -> list[_InputPrior]:
    """Build ordered candidate input priors.

    Parameters
    ----------
    model:
        Model to inspect.
    batch_size:
        Batch dimension.
    input_dtype:
        Optional dtype override.
    channels:
        Optional channel override.
    spatial_rank:
        Optional spatial rank override.
    seq_len:
        Optional sequence length override.
    min_size:
        Minimum spatial side.
    preferred_sizes:
        Preferred spatial sizes.
    device:
        Target device.

    Returns
    -------
    list[_InputPrior]
        Candidate priors, de-duplicated by shape and dtype.
    """

    priors: list[_InputPrior] = []
    static = _shape_from_prior(
        model,
        batch_size,
        input_dtype,
        channels,
        spatial_rank,
        seq_len,
        min_size,
        preferred_sizes,
        device,
    )
    if static is not None:
        priors.append(static)

    _model_device, model_dtype = _module_device_dtype(model)
    index = _module_index(model)
    modules = [(name, module) for name, module in model.named_modules() if name]
    for name, module in sorted(modules, key=lambda item: index[item[1]]):
        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            size = seq_len or 16
            shape = (batch_size, size, int(module.input_size))
            if not bool(module.batch_first):
                shape = (size, batch_size, int(module.input_size))
            priors.append(
                _InputPrior(
                    kind="rnn",
                    shape=shape,
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(1 if bool(module.batch_first) else 0,),
                    constraining_module=name,
                    constraining_op=module.__class__.__name__,
                    strategy="introspection",
                    device=device,
                )
            )
        elif isinstance(module, nn.Embedding):
            cap = _sequence_cap(model)
            size = seq_len or min(16, cap or 16)
            priors.append(
                _InputPrior(
                    kind="embedding",
                    shape=(batch_size, size),
                    dtype=input_dtype or torch.long,
                    value_range=("randint", 0.0, float(module.num_embeddings)),
                    flexible_dims=(1,),
                    constraining_module=name,
                    constraining_op="Embedding",
                    strategy="introspection",
                    device=device,
                )
            )
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            rank = (
                int(spatial_rank)
                if spatial_rank != "auto"
                else {nn.Conv1d: 1, nn.Conv2d: 2, nn.Conv3d: 3}[type(module)]
            )
            kernel = (
                module.kernel_size
                if isinstance(module.kernel_size, tuple)
                else (int(module.kernel_size),)
            )
            stride = module.stride[0] if isinstance(module.stride, tuple) else int(module.stride)
            side = _positional_side_from_any_table(model, int(stride)) or max(
                max(kernel),
                _rank_default(rank) if rank == 3 else next(iter(preferred_sizes), 32),
                min_size,
            )
            priors.append(
                _InputPrior(
                    kind="conv",
                    shape=(batch_size, channels or int(module.in_channels), *([side] * rank)),
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=tuple(range(2, 2 + rank)) if _has_adaptive_pool(model) else (),
                    constraining_module=name,
                    constraining_op=module.__class__.__name__,
                    strategy="adaptive_default" if _has_adaptive_pool(model) else "introspection",
                    device=device,
                    spatial_rank=rank,
                    channels=channels or int(module.in_channels),
                    min_side=max(kernel),
                )
            )
        elif isinstance(module, nn.TransformerEncoderLayer):
            size = seq_len or 16
            priors.append(
                _InputPrior(
                    kind="transformer",
                    shape=(batch_size, size, int(module.self_attn.embed_dim)),
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(1,),
                    constraining_module=name,
                    constraining_op=module.__class__.__name__,
                    strategy="introspection",
                    device=device,
                )
            )
        elif isinstance(module, nn.GroupNorm):
            priors.append(
                _InputPrior(
                    kind="norm",
                    shape=(batch_size, int(module.num_channels), 32, 32),
                    dtype=input_dtype or _float_dtype(model_dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(2, 3),
                    constraining_module=name,
                    constraining_op="GroupNorm",
                    strategy="introspection",
                    device=device,
                )
            )
        elif isinstance(module, nn.Linear):
            priors.append(
                _InputPrior(
                    kind="linear",
                    shape=(batch_size, int(module.in_features)),
                    dtype=input_dtype or _float_dtype(module.weight.dtype),
                    value_range=("uniform", 0.0, 1.0),
                    flexible_dims=(),
                    constraining_module=name,
                    constraining_op="Linear",
                    strategy="introspection",
                    device=device,
                )
            )
    priors.extend(
        _parameter_priors(
            model, batch_size, input_dtype, seq_len, min_size, preferred_sizes, device
        )
    )

    unique: list[_InputPrior] = []
    seen: set[tuple[tuple[int, ...], torch.dtype, str]] = set()
    for prior in priors:
        key = (prior.shape, prior.dtype, prior.kind)
        if key not in seen:
            seen.add(key)
            unique.append(prior)
    return unique


def _sequence_cap(model: nn.Module) -> int | None:
    """Infer a maximum sequence length from positional tables.

    Parameters
    ----------
    model:
        Model to inspect.

    Returns
    -------
    int | None
        Sequence cap, or ``None``.
    """

    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.ndim >= 2 and _POSITION_RE.search(name):
            return int(tensor.shape[1] if tensor.ndim >= 3 else tensor.shape[0])
    return None


def _candidate_sides(
    start: int,
    min_size: int,
    max_size: int,
    preferred_sizes: Sequence[int],
) -> list[int]:
    """Return ordered spatial side candidates.

    Parameters
    ----------
    start:
        Initial side from static prior.
    min_size:
        Minimum side.
    max_size:
        Maximum side.
    preferred_sizes:
        Preferred side lengths.

    Returns
    -------
    list[int]
        Unique side candidates in probe order.
    """

    values = [start, *preferred_sizes, 32, 28, 16, 8, min_size, max_size]
    return [
        side
        for index, side in enumerate(values)
        if min_size <= side <= max_size and side not in values[:index]
    ]


def _trace_model(model: nn.Module, example_input: Any) -> Trace:
    """Run a final TorchLens inference-only trace.

    Parameters
    ----------
    model:
        Model to trace.
    example_input:
        Input object.

    Returns
    -------
    Trace
        Completed TorchLens trace.
    """

    from torchlens.user_funcs import trace

    return trace(model, example_input, inference_only=True)


def _shape_op_label(op: Any) -> str | None:
    """Return an op label from a TorchLens layer/op object.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    str | None
        Label when present.
    """

    label = getattr(op, "layer_label", None) or getattr(op, "op_label", None)
    return str(label) if label is not None else None


def _shape_op_module(op: Any) -> str | None:
    """Return a module address from a TorchLens op-like object.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    str | None
        Module address when present.
    """

    module = getattr(op, "module_address", None) or getattr(op, "module", None)
    return str(module) if module is not None else None


def _shape_op_source_line(op: Any) -> str | None:
    """Return an op source-line string when TorchLens exposes one.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    str | None
        Source-line string, or ``None``.
    """

    source_line = getattr(op, "source_line", None)
    return str(source_line) if source_line is not None else None


def _shape_op_config_int(op: Any, key: str) -> int | None:
    """Read an integer from an op's ``func_config``.

    Parameters
    ----------
    op:
        TorchLens op-like object.
    key:
        Config key.

    Returns
    -------
    int | None
        Integer value when present.
    """

    config = getattr(op, "func_config", {}) or {}
    value = config.get(key)
    return int(value) if value is not None else None


def _executed_constraint(trace: Trace) -> _ExecutedConstraint | None:
    """Read the first executed input-consuming constraint op from a trace.

    Parameters
    ----------
    trace:
        Successful TorchLens trace.

    Returns
    -------
    _ExecutedConstraint | None
        Executed constraint metadata, or ``None``.
    """

    for op in trace.layers:
        func_name = str(getattr(op, "func_name", "")).lower()
        input_shapes = tuple(_shape_tuple(shape) for shape in getattr(op, "input_shapes", ()) or ())
        input_dtypes = tuple(getattr(op, "input_dtypes", ()) or ())
        first_shape = next((shape for shape in input_shapes if shape is not None), None)
        first_dtype = input_dtypes[0] if input_dtypes else None
        param_shapes = tuple(_shape_tuple(shape) for shape in getattr(op, "param_shapes", []) or [])
        if func_name in {"conv1d", "conv2d", "conv3d"}:
            weight_shape = next((shape for shape in param_shapes if shape is not None), None)
            kernel = weight_shape[2:] if weight_shape is not None and len(weight_shape) >= 3 else ()
            return _ExecutedConstraint(
                kind="conv",
                label=_shape_op_label(op),
                module=_shape_op_module(op),
                source_line=_shape_op_source_line(op),
                input_shape=first_shape,
                dtype=first_dtype,
                in_channels=_shape_op_config_int(op, "in_channels")
                or (
                    weight_shape[1] if weight_shape is not None and len(weight_shape) > 1 else None
                ),
                spatial_rank=len(kernel)
                or (len(first_shape) - 2 if first_shape is not None else None),
                kernel_size=kernel,
                flexible=_shape_has_executed_adaptive_pool(trace),
            )
        if func_name in {"linear", "addmm"}:
            weight_shape = next((shape for shape in param_shapes if shape is not None), None)
            return _ExecutedConstraint(
                kind="linear",
                label=_shape_op_label(op),
                module=_shape_op_module(op),
                source_line=_shape_op_source_line(op),
                input_shape=first_shape,
                dtype=first_dtype,
                in_features=_shape_op_config_int(op, "in_features")
                or (
                    weight_shape[1] if weight_shape is not None and len(weight_shape) > 1 else None
                ),
            )
        if func_name in {"matmul", "mm", "bmm"}:
            needed = None
            if len(param_shapes) >= 1 and param_shapes[0] is not None:
                needed = param_shapes[0][0]
            if first_shape is not None and len(first_shape) >= 1:
                needed = first_shape[-1]
            return _ExecutedConstraint(
                kind="matmul",
                label=_shape_op_label(op),
                module=_shape_op_module(op),
                source_line=_shape_op_source_line(op),
                input_shape=first_shape,
                dtype=first_dtype,
                in_features=needed,
            )
        if func_name == "embedding":
            vocab = None
            weight_shape = next((shape for shape in param_shapes if shape is not None), None)
            if weight_shape is not None:
                vocab = weight_shape[0]
            return _ExecutedConstraint(
                kind="embedding",
                label=_shape_op_label(op),
                module=_shape_op_module(op),
                source_line=_shape_op_source_line(op),
                input_shape=first_shape,
                dtype=first_dtype,
                in_features=vocab,
            )
    return None


def _verified_result(
    prior: _InputPrior,
    shape: tuple[int, ...],
    example: torch.Tensor,
    attempts: list[tuple[tuple[int, ...] | None, str]],
    trace_obj: Trace,
    return_trace: bool,
    strategy: str,
) -> InferInputShapeResult:
    """Build a successful result from a verified trace.

    Parameters
    ----------
    prior:
        Candidate prior that produced the input.
    shape:
        Verified input shape.
    example:
        Verified example tensor.
    attempts:
        Probe attempts.
    trace_obj:
        Verification trace.
    return_trace:
        Whether to retain the trace in the public result.
    strategy:
        Winning strategy label.

    Returns
    -------
    InferInputShapeResult
        Successful result.
    """

    constraint = _executed_constraint(trace_obj)
    flexible: tuple[int, ...] = ()
    if constraint is not None and constraint.kind == "conv" and constraint.flexible:
        rank = constraint.spatial_rank or prior.spatial_rank or max(0, len(shape) - 2)
        flexible = tuple(range(2, 2 + rank))
    elif constraint is not None and constraint.kind in {"embedding", "matmul", "linear"}:
        flexible = prior.flexible_dims if prior.kind in {"embedding", "rnn", "transformer"} else ()
    elif constraint is None:
        flexible = prior.flexible_dims
    return InferInputShapeResult(
        found=True,
        shape=shape,
        shapes=None,
        dtype=example.dtype,
        value_range=prior.value_range,
        flexible_dims=flexible,
        constraining_module=constraint.module
        if constraint is not None
        else prior.constraining_module,
        constraining_op=constraint.label if constraint is not None else prior.constraining_op,
        source_line=constraint.source_line if constraint is not None else None,
        example_input=example,
        strategy=strategy,
        reason=None,
        attempts=tuple(attempts),
        trace=trace_obj if return_trace else None,
        message=f"Found valid input shape {shape} using executed op metadata.",
    )


def _maybe_normalize_success(
    model: nn.Module,
    prior: _InputPrior,
    example: torch.Tensor,
    trace_obj: Trace,
    attempts: list[tuple[tuple[int, ...] | None, str]],
    seed: int,
) -> tuple[torch.Tensor, Trace, str]:
    """Prefer the minimal executed linear input over a decoy-shaped success.

    Parameters
    ----------
    model:
        Model being inferred.
    prior:
        Prior that produced the success.
    example:
        Successful example.
    trace_obj:
        Successful trace for ``example``.
    attempts:
        Probe attempt list to append to.
    seed:
        RNG seed.

    Returns
    -------
    tuple[torch.Tensor, Trace, str]
        Possibly normalized example, trace, and strategy.
    """

    constraint = _executed_constraint(trace_obj)
    if (
        constraint is None
        or constraint.kind not in {"linear", "matmul"}
        or constraint.input_shape is None
        or len(constraint.input_shape) <= 2
        or constraint.in_features is None
    ):
        return example, trace_obj, prior.strategy

    shape = (constraint.input_shape[0], constraint.in_features)
    normalized = _make_tensor(shape, example.dtype, example.device, prior.value_range)
    probe = _probe(model, normalized, seed)
    attempts.append((shape, probe.outcome))
    if not probe.ok:
        return example, trace_obj, prior.strategy
    normalized_trace = _trace_model(model, normalized)
    return normalized, normalized_trace, "executed_op_normalize"


def _failure_result(
    reason: FailureReason,
    attempts: list[tuple[tuple[int, ...] | None, str]],
    message: str,
) -> InferInputShapeResult:
    """Build a standardized failed inference result.

    Parameters
    ----------
    reason:
        Failure reason.
    attempts:
        Probe attempts.
    message:
        User-facing message.

    Returns
    -------
    InferInputShapeResult
        Failed result.
    """

    return InferInputShapeResult(
        found=False,
        shape=None,
        shapes=None,
        dtype=None,
        value_range=None,
        flexible_dims=(),
        constraining_module=None,
        constraining_op=None,
        source_line=None,
        example_input=None,
        strategy="probe_success",
        reason=reason,
        attempts=tuple(attempts),
        trace=None,
        message=message,
    )


def _maybe_raise(result: InferInputShapeResult, on_failure: Literal["return", "raise"]) -> None:
    """Raise for a failed result when requested.

    Parameters
    ----------
    result:
        Inference result.
    on_failure:
        Failure behavior.
    """

    if not result.found and on_failure == "raise":
        raise ShapeInferenceError(result.message)


def infer_input_shape(
    model: nn.Module,
    *,
    batch_size: int = 1,
    input_dtype: torch.dtype | None = None,
    channels: int | None = None,
    spatial_rank: int | Literal["auto"] = "auto",
    seq_len: int | None = None,
    square: bool = True,
    min_size: int = 1,
    max_size: int = 512,
    preferred_sizes: Sequence[int] = (224, 256, 384, 299, 128, 96, 64, 32, 28),
    max_probes: int = 64,
    device: torch.device | str | None = None,
    seed: int = 0,
    return_trace: bool = False,
    on_failure: Literal["return", "raise"] = "return",
    input_specs: Any = None,
) -> InferInputShapeResult:
    """Infer a verified synthetic input shape for a PyTorch module.

    Parameters
    ----------
    model:
        PyTorch module to probe.
    batch_size:
        Batch dimension for synthesized inputs.
    input_dtype:
        Optional primary input dtype override.
    channels:
        Optional channel count override for convolutional inputs.
    spatial_rank:
        Spatial rank override, or ``"auto"`` from the first convolution.
    seq_len:
        Optional sequence length override for token or recurrent inputs.
    square:
        Whether spatial search should use equal side lengths. Non-square exact inference is
        intentionally not attempted without a future aspect-ratio hint.
    min_size:
        Minimum spatial side considered.
    max_size:
        Maximum spatial side considered.
    preferred_sizes:
        Spatial candidates to try before measured fallback search.
    max_probes:
        Maximum forward probes.
    device:
        Optional device override; defaults to the model's first parameter or buffer device.
    seed:
        RNG seed used for deterministic probes.
    return_trace:
        Whether to include the final verification ``Trace``.
    on_failure:
        ``"return"`` for notebook-friendly diagnostics or ``"raise"`` for
        ``ShapeInferenceError``.
    input_specs:
        Reserved for explicit multi-input specs. Coupled multi-input inference is currently
        unsupported unless the caller supplies a ready-made tensor/container in future work.

    Returns
    -------
    InferInputShapeResult
        Verified input, diagnostic failure, and probe attempts.

    Notes
    -----
    The implementation combines static module introspection, forward pre-hook probe diaries,
    successful ``tl.trace(..., inference_only=True)`` verification, and torch exception parsing.
    It measures candidate shapes instead of deriving convolution and pooling formulas.

    Limitations
    -----------
    Inferring valid inputs for arbitrary Python ``forward`` code is undecidable in general. This
    helper targets common MLP, CNN, adaptive-pool, ViT patch-embedding, transformer-LM, RNN, and
    1D/3D convolution cases. It cannot reliably infer coupled multi-tensor or dict inputs,
    value-dependent/data-dependent-control-flow shapes, non-tensor kwargs such as masks, labels,
    or ``past_key_values``, tokenizer/image-processor preprocessing, stateful or buffer-mutating
    forwards, opaque C++/CUDA errors, non-square exact spatial sizes without an aspect hint, or
    ``torch.compile``/scripted/exported artifacts.
    """

    if not isinstance(model, nn.Module):
        raise ShapeInferenceError("infer_input_shape expects a torch.nn.Module.")
    if input_specs is not None:
        result = _failure_result(
            "multi_input_unsupported",
            [],
            "input_specs multi-input inference is not implemented yet; pass a single-input module.",
        )
        _maybe_raise(result, on_failure)
        return result
    if not square:
        result = _failure_result(
            "exact_size_unreachable",
            [],
            "Non-square spatial inference requires an explicit aspect-ratio hint, which is not supported.",
        )
        _maybe_raise(result, on_failure)
        return result

    base_device, _base_dtype = _module_device_dtype(model)
    resolved_device = torch.device(device) if device is not None else base_device
    priors = _input_priors(
        model,
        batch_size,
        input_dtype,
        channels,
        spatial_rank,
        seq_len,
        min_size,
        preferred_sizes,
        resolved_device,
    )
    attempts: list[tuple[tuple[int, ...] | None, str]] = []
    if not priors:
        result = _failure_result(
            "unknown_entry",
            attempts,
            "No supported executed-op seed was found; identity-like models are not inferred.",
        )
        _maybe_raise(result, on_failure)
        return result

    probes = 0
    delayed_blockers: list[str] = []
    for prior in priors:
        if probes >= max_probes:
            break
        example = _make_tensor(prior.shape, prior.dtype, prior.device, prior.value_range)
        probe = _probe(model, example, seed)
        probes += 1
        attempts.append((prior.shape, probe.outcome))
        if probe.ok:
            trace_obj = _trace_model(model, example)
            final_example, final_trace, strategy = _maybe_normalize_success(
                model, prior, example, trace_obj, attempts, seed
            )
            shape = tuple(int(dim) for dim in final_example.shape)
            return _verified_result(
                prior, shape, final_example, attempts, final_trace, return_trace, strategy
            )

        lower_outcome = probe.outcome.lower()
        suggested_dtype = _dtype_from_message(probe.outcome)
        if suggested_dtype is not None and suggested_dtype != prior.dtype and probes < max_probes:
            value_range = prior.value_range
            if suggested_dtype == torch.long and value_range[0] != "randint":
                value_range = ("randint", 0.0, 2.0)
            fixed = _make_tensor(prior.shape, suggested_dtype, prior.device, value_range)
            second = _probe(model, fixed, seed)
            probes += 1
            attempts.append((prior.shape, second.outcome))
            if second.ok:
                trace_obj = _trace_model(model, fixed)
                dtype_prior = _InputPrior(
                    kind=prior.kind,
                    shape=prior.shape,
                    dtype=suggested_dtype,
                    value_range=value_range,
                    flexible_dims=prior.flexible_dims,
                    constraining_module=prior.constraining_module,
                    constraining_op=prior.constraining_op,
                    strategy="dtype_corrected",
                    device=prior.device,
                    spatial_rank=prior.spatial_rank,
                    channels=prior.channels,
                    min_side=prior.min_side,
                )
                return _verified_result(
                    dtype_prior,
                    prior.shape,
                    fixed,
                    attempts,
                    trace_obj,
                    return_trace,
                    "dtype_corrected",
                )

        if prior.kind == "linear" and probe.target_features is not None and probes < max_probes:
            shape = (batch_size, probe.target_features)
            fixed = _make_tensor(shape, prior.dtype, prior.device, prior.value_range)
            second = _probe(model, fixed, seed)
            probes += 1
            attempts.append((shape, second.outcome))
            if second.ok:
                trace_obj = _trace_model(model, fixed)
                corrected = _InputPrior(
                    kind=prior.kind,
                    shape=shape,
                    dtype=prior.dtype,
                    value_range=prior.value_range,
                    flexible_dims=prior.flexible_dims,
                    constraining_module=second.constraining_module or prior.constraining_module,
                    constraining_op=prior.constraining_op,
                    strategy="executed_op_linear",
                    device=prior.device,
                )
                return _verified_result(
                    corrected, shape, fixed, attempts, trace_obj, return_trace, "executed_op_linear"
                )

        if prior.kind != "conv":
            if not _is_skippable_shape_error(probe.outcome):
                delayed_blockers.append(probe.outcome)
            continue

        rank = prior.spatial_rank or max(1, len(prior.shape) - 2)
        lower_bound = max(min_size, prior.min_side, 1)
        measured: list[tuple[int, int]] = []
        target = probe.target_features
        if probe.got_features is not None and probe.target_features is not None:
            measured.append((prior.shape[-1], probe.got_features))
        sides = _candidate_sides(prior.shape[-1], lower_bound, max_size, preferred_sizes)
        for side in sides:
            if probes >= max_probes:
                break
            if side == prior.shape[-1]:
                continue
            shape = (batch_size, prior.channels or prior.shape[1], *([side] * rank))
            example = _make_tensor(shape, prior.dtype, prior.device, prior.value_range)
            side_probe = _probe(model, example, seed)
            probes += 1
            attempts.append((shape, side_probe.outcome))
            if side_probe.ok:
                trace_obj = _trace_model(model, example)
                final_example, final_trace, strategy = _maybe_normalize_success(
                    model, prior, example, trace_obj, attempts, seed
                )
                final_shape = tuple(int(dim) for dim in final_example.shape)
                return _verified_result(
                    prior,
                    final_shape,
                    final_example,
                    attempts,
                    final_trace,
                    return_trace,
                    strategy if strategy != prior.strategy else "probe_success",
                )
            if side_probe.got_features is not None and side_probe.target_features is not None:
                measured.append((side, side_probe.got_features))
                target = side_probe.target_features
            elif not _is_skippable_shape_error(side_probe.outcome):
                delayed_blockers.append(side_probe.outcome)
                break

        if measured and target is not None and probes < max_probes:
            low = lower_bound
            high = max_size
            while low <= high and probes < max_probes:
                side = (low + high) // 2
                shape = (batch_size, prior.channels or prior.shape[1], *([side] * rank))
                example = _make_tensor(shape, prior.dtype, prior.device, prior.value_range)
                search_probe = _probe(model, example, seed)
                probes += 1
                attempts.append((shape, search_probe.outcome))
                if search_probe.ok:
                    trace_obj = _trace_model(model, example)
                    final_example, final_trace, strategy = _maybe_normalize_success(
                        model, prior, example, trace_obj, attempts, seed
                    )
                    final_shape = tuple(int(dim) for dim in final_example.shape)
                    return _verified_result(
                        prior,
                        final_shape,
                        final_example,
                        attempts,
                        final_trace,
                        return_trace,
                        strategy if strategy != prior.strategy else "binary_search",
                    )
                if search_probe.got_features is None:
                    if _is_skippable_shape_error(search_probe.outcome):
                        low = side + 1
                        continue
                    delayed_blockers.append(search_probe.outcome)
                    break
                if search_probe.got_features < target:
                    low = side + 1
                else:
                    high = side - 1
            for side in range(max(lower_bound, low - 4), min(max_size, low + 4) + 1):
                if probes >= max_probes:
                    break
                shape = (batch_size, prior.channels or prior.shape[1], *([side] * rank))
                example = _make_tensor(shape, prior.dtype, prior.device, prior.value_range)
                near_probe = _probe(model, example, seed)
                probes += 1
                attempts.append((shape, near_probe.outcome))
                if near_probe.ok:
                    trace_obj = _trace_model(model, example)
                    final_example, final_trace, strategy = _maybe_normalize_success(
                        model, prior, example, trace_obj, attempts, seed
                    )
                    final_shape = tuple(int(dim) for dim in final_example.shape)
                    return _verified_result(
                        prior,
                        final_shape,
                        final_example,
                        attempts,
                        final_trace,
                        return_trace,
                        strategy if strategy != prior.strategy else "binary_search",
                    )

        if lower_outcome and not _is_skippable_shape_error(probe.outcome):
            delayed_blockers.append(probe.outcome)

    if probes >= max_probes:
        result = _failure_result(
            "budget_exhausted", attempts, f"No valid shape was found within {max_probes} probes."
        )
        _maybe_raise(result, on_failure)
        return result
    if delayed_blockers:
        result = _failure_result(
            "non_shape_blocker",
            attempts,
            f"Shape inference was blocked by an unsupported forward error: {delayed_blockers[-1]}",
        )
        _maybe_raise(result, on_failure)
        return result
    result = _failure_result(
        "exact_size_unreachable",
        attempts,
        "No square valid input was found; pass an explicit size/aspect hint for rectangular cases.",
    )
    _maybe_raise(result, on_failure)
    return result


__all__ = [
    "BisectNanResult",
    "InferInputShapeResult",
    "LineageResult",
    "bisect_nan",
    "compare",
    "dead_neurons",
    "gradient_flow_audit",
    "hot_path",
    "infer_input_shape",
    "lineage",
    "recompute_candidates",
]

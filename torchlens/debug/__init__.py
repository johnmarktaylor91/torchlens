"""Power-user debugging helpers for completed TorchLens traces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

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


__all__ = [
    "BisectNanResult",
    "LineageResult",
    "bisect_nan",
    "compare",
    "dead_neurons",
    "gradient_flow_audit",
    "hot_path",
    "lineage",
    "recompute_candidates",
]

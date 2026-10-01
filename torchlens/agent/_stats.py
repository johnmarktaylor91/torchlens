"""payload_stats: bounded numbers over saved tensors, never the tensors.

Reads are scoped and NON-attaching (the A09 P0 primitive): verify, compute,
release -- sequential site scans hold one payload at a time. Byte budgets
refuse BEFORE materialization from declared shape/dtype facts. Determinism:
CPU, float64 accumulation, stable flat-index tie-breaks, algorithms named in
the record. An unsaved or unsupported site is a per-row typed status with the
exact recapture remedy -- it never fails the batch, and no purported global
statistic is ever computed from a truncated prefix.
"""

from __future__ import annotations

from typing import Any

from .._errors import InvalidArgumentError

#: Result schema id.
PAYLOAD_STATS_SCHEMA = "torchlens.agent.payload_stats.v1"

#: Closed v1 metric vocabulary.
METRICS = (
    "shape",
    "dtype",
    "numel",
    "bytes",
    "finite_count",
    "nan_count",
    "posinf_count",
    "neginf_count",
    "zero_count",
    "min",
    "max",
    "mean",
    "std",
    "l1_norm",
    "l2_norm",
    "top_k",
    "bottom_k",
)

#: Ceiling on requested top-k/bottom-k entries.
TOP_K_MAX = 64

#: Default k when extremes are requested without a k.
TOP_K_DEFAULT = 8

#: Closed target vocabulary.
TARGETS = ("out", "grad")


def _declared_payload_bytes(op: Any, target: str) -> int | None:
    """Return the payload's byte size WITHOUT materializing anything.

    Parameters
    ----------
    op:
        Op record.
    target:
        ``"out"`` or ``"grad"``.

    Returns
    -------
    int | None
        Exact bytes from resident tensor meta or lazy-ref declared
        shape/dtype; ``None`` when no payload exists.
    """

    slot = getattr(op, "_slot", None)
    resident = slot(target) if callable(slot) else getattr(op, target, None)
    if resident is not None and hasattr(resident, "element_size"):
        return int(resident.numel()) * int(resident.element_size())
    ref = slot(f"{target}_ref") if callable(slot) else getattr(op, f"{target}_ref", None)
    if ref is None:
        return None
    import torch

    numel = 1
    for dim in ref.shape:
        numel *= int(dim)
    element_size = torch.empty((), dtype=ref.dtype).element_size()
    return numel * element_size


def _read_payload(op: Any, target: str) -> Any:
    """Materialize one payload through the scoped non-attaching door.

    Parameters
    ----------
    op:
        Op record.
    target:
        ``"out"`` or ``"grad"``.

    Returns
    -------
    Any
        Tensor (caller's only reference) or ``None`` when absent.
    """

    if target == "out":
        from .._io.payload_reader import read_op_payload

        try:
            return read_op_payload(op)
        except Exception:  # noqa: BLE001 - any read failure is the per-row typed 'unsaved' status
            return None
    slot = getattr(op, "_slot", None)
    resident = slot("grad") if callable(slot) else getattr(op, "grad", None)
    if resident is not None:
        return resident
    ref = slot("grad_ref") if callable(slot) else getattr(op, "grad_ref", None)
    if ref is not None:
        return ref.materialize(map_location="cpu")
    return None


def _unsupported_reason(tensor: Any) -> str | None:
    """Name why a payload's metric subset is restricted, or ``None``."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        return f"non-tensor payload ({type(tensor).__name__}); described, not coerced"
    if tensor.is_sparse or getattr(tensor, "is_sparse_csr", False):
        return "sparse payload: ordered metrics undefined on implicit zeros"
    if tensor.is_complex():
        return "complex payload: ordered comparisons undefined; counts/norms only"
    if tensor.is_quantized:
        return "quantized payload: dequantize in Python for value metrics"
    if tensor.numel() == 0:
        return "empty payload: extremes/mean/std undefined on zero elements"
    return None


def _counts(values: Any) -> dict[str, int]:
    """Finite/NaN/inf/zero counts in overflow-safe integer arithmetic."""

    import torch

    return {
        "finite_count": int(torch.isfinite(values).sum().item()),
        "nan_count": int(torch.isnan(values).sum().item()),
        "posinf_count": int(torch.isposinf(values).sum().item()),
        "neginf_count": int(torch.isneginf(values).sum().item()),
        "zero_count": int((values == 0).sum().item()),
    }


def _scalar_stats(values: Any) -> dict[str, Any]:
    """min/max/mean/std/norms in float64 on CPU (population std, correction=0)."""

    values64 = values.detach().to("cpu", dtype=_accumulation_dtype(values)).flatten()
    float64 = values64.double() if values64.dtype != _torch().float64 else values64
    return {
        "min": values64.min().item(),
        "max": values64.max().item(),
        "mean": float64.mean().item(),
        "std": float64.std(correction=0).item(),
        "l1_norm": float64.abs().sum().item(),
        "l2_norm": float64.pow(2).sum().sqrt().item(),
    }


def _torch() -> Any:
    """Return the torch module (loaded traces imply torch is importable)."""

    import torch

    return torch


def _accumulation_dtype(values: Any) -> Any:
    """Float64 accumulation for every real dtype (named in the record)."""

    return _torch().float64


def _extremes(values: Any, k: int, *, largest: bool) -> list[dict[str, Any]]:
    """Deterministic top/bottom-k with FULL coordinates.

    NaNs never satisfy a criterion: they are masked to the losing infinity
    before ranking. Ties break by flat index (stable sort), so the result is
    byte-reproducible across processes.

    Parameters
    ----------
    values:
        Tensor payload.
    k:
        Number of entries.
    largest:
        Rank direction.

    Returns
    -------
    list[dict[str, Any]]
        ``{"value", "coordinates"}`` entries.
    """

    torch = _torch()
    flat = values.detach().to("cpu", dtype=torch.float64).flatten()
    fill = float("-inf") if largest else float("inf")
    masked = torch.nan_to_num(flat, nan=fill, posinf=None, neginf=None)
    order = torch.argsort(masked, descending=largest, stable=True)
    take = min(k, flat.numel())
    shape = tuple(int(dim) for dim in values.shape)
    entries: list[dict[str, Any]] = []
    for flat_index in order[:take].tolist():
        coords: list[int] = []
        remainder = flat_index
        for dim in reversed(shape):
            coords.append(remainder % dim)
            remainder //= dim
        entries.append({"value": flat[flat_index].item(), "coordinates": coords[::-1]})
    return entries


def _reduced_rows(  # noqa: PLR0913 - one reduction record's fields, never positional call sites
    values: Any,
    retain_dim: int,
    metrics: list[str],
    max_rows: int,
    rank_by: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Per-index rows along ONE retained dimension (memo 3.6's one axis).

    Full extent when it fits ``max_rows``; otherwise the TOP ``max_rows`` by
    the required ranking metric, so the interesting rows survive the cap.

    Parameters
    ----------
    values:
        Tensor payload.
    retain_dim:
        Dimension INDEX to retain (explicit; never a guessed role).
    metrics:
        Scalar metrics to compute per index.
    max_rows:
        Row cap.
    rank_by:
        Ranking metric, required when the extent exceeds ``max_rows``.

    Returns
    -------
    tuple[list[dict], int]
        Rows and the full extent.

    Raises
    ------
    InvalidArgumentError
        ``agent_reduction_invalid`` for a bad dim or a missing rank_by.
    """

    torch = _torch()
    ndim = values.dim()
    if not (-ndim <= retain_dim < ndim):
        raise InvalidArgumentError(
            f"retain_dim={retain_dim} is out of range for a rank-{ndim} payload",
            code="agent_reduction_invalid",
            remedy=f"pass a dimension index in [-{ndim}, {ndim - 1}]",
        )
    dim = retain_dim % ndim
    moved = values.detach().to("cpu", dtype=torch.float64).movedim(dim, 0)
    flat = moved.reshape(moved.shape[0], -1)
    extent = int(flat.shape[0])
    per_index: list[dict[str, Any]] = []
    for index in range(extent):
        row_values = flat[index]
        row: dict[str, Any] = {"index": index}
        row.update(
            {
                "min": row_values.min().item(),
                "max": row_values.max().item(),
                "mean": row_values.mean().item(),
                "std": row_values.std(correction=0).item(),
                "l2_norm": row_values.pow(2).sum().sqrt().item(),
                "nan_count": int(torch.isnan(row_values).sum().item()),
            }
        )
        per_index.append(row)
    if extent <= max_rows:
        return per_index, extent
    if rank_by is None or rank_by not in ("min", "max", "mean", "std", "l2_norm", "nan_count"):
        raise InvalidArgumentError(
            f"retained extent ({extent}) exceeds max_rows ({max_rows}) and no "
            "ranking metric was named",
            code="agent_reduction_invalid",
            remedy=(
                "pass rank_by= one of min, max, mean, std, l2_norm, nan_count; "
                "the top rows by that metric survive the cap"
            ),
        )

    def _rank_key(row: dict[str, Any]) -> tuple[float, int]:
        """Rank by the named metric descending; NaN loses; ties break on index."""

        value = row[rank_by]
        magnitude = value if value == value else float("-inf")
        return (-magnitude, row["index"])

    ranked = sorted(per_index, key=_rank_key)[:max_rows]
    return ranked, extent


def payload_stats_rows(  # noqa: PLR0913 - the registry-declared request record, keyword-threaded
    log: Any,
    *,
    labels: list[str] | None,
    query: dict[str, Any] | None,
    target: str,
    metrics: list[str] | None,
    reduction: dict[str, Any] | None,
    k: int | None,
    max_rows: int,
    rank_by: str | None = None,
    max_blob_bytes: int,
    max_call_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute bounded per-site statistic rows.

    Parameters
    ----------
    log:
        Loaded ``Trace``.
    labels:
        Explicit site labels, or ``None`` to use ``query``.
    query:
        Query AST selecting sites (both ``None`` = every retained site).
    target:
        ``"out"`` or ``"grad"``.
    metrics:
        Closed metric subset, or ``None`` for the scalar default set.
    reduction:
        ``None`` (whole payload) or ``{"retain_dim": int}``.
    k:
        top/bottom-k entry count.
    max_rows:
        Site-row cap (reduction rows are additionally capped per site).
    rank_by:
        Ranking metric for over-extent reductions.
    max_blob_bytes:
        Per-payload byte ceiling, checked BEFORE materialization.
    max_call_bytes:
        Aggregate byte ceiling across the call.

    Returns
    -------
    tuple[list[dict], dict]
        Per-site rows (typed statuses inline) and the record header
        (population, evidence, accumulation disclosures).

    Raises
    ------
    InvalidArgumentError
        ``agent_argument_invalid`` on vocabulary violations;
        ``agent_reduction_invalid`` per :func:`_reduced_rows`.
    """

    if target not in TARGETS:
        raise InvalidArgumentError(
            f"target={target!r} is not a payload target",
            code="agent_argument_invalid",
            remedy=f"pass one of {', '.join(TARGETS)}",
        )
    chosen = list(metrics) if metrics else [m for m in METRICS if m not in ("top_k", "bottom_k")]
    unknown = [m for m in chosen if m not in METRICS]
    if unknown:
        raise InvalidArgumentError(
            f"unknown metrics {unknown!r}",
            code="agent_argument_invalid",
            remedy=f"choose from the closed set: {', '.join(METRICS)}",
        )
    effective_k = k if k is not None else TOP_K_DEFAULT
    if (
        not isinstance(effective_k, int)
        or isinstance(effective_k, bool)
        or not (1 <= effective_k <= TOP_K_MAX)
    ):
        raise InvalidArgumentError(
            f"k={k!r} is outside [1, {TOP_K_MAX}]",
            code="agent_argument_invalid",
            remedy=f"pass 1 <= k <= {TOP_K_MAX}",
        )

    ops = _select_ops(log, labels=labels, query=query)
    total_selected = len(ops)
    ops = ops[:max_rows]
    rows: list[dict[str, Any]] = []
    spent_bytes = 0
    for op in ops:
        label = str(getattr(op, "label", "unknown"))
        row: dict[str, Any] = {"label": label, "target": target}
        declared = _declared_payload_bytes(op, target)
        if declared is None:
            row["status"] = "unsaved"
            row["remedy"] = (
                f"re-capture with save= covering this op, e.g. "
                f"tl.trace(model, x, save=tl.label({label.rsplit(':', 1)[0]!r}))"
                if target == "out"
                else "re-capture with backward_ready=True and save_grads covering this op"
            )
            rows.append(row)
            continue
        if declared > max_blob_bytes:
            row["status"] = "refused_blob_budget"
            row["declared_bytes"] = declared
            row["remedy"] = (
                f"payload declares {declared:,} bytes > per-payload ceiling "
                f"{max_blob_bytes:,}; read it in Python via trace[{label!r}].{target}"
            )
            rows.append(row)
            continue
        if spent_bytes + declared > max_call_bytes:
            row["status"] = "refused_call_budget"
            row["declared_bytes"] = declared
            row["remedy"] = (
                f"aggregate call budget {max_call_bytes:,} bytes exhausted; "
                "page the site list across calls"
            )
            rows.append(row)
            continue
        tensor = _read_payload(op, target)
        if tensor is None:
            row["status"] = "unsaved"
            row["remedy"] = "payload could not be materialized; see payload_state on the site row"
            rows.append(row)
            continue
        spent_bytes += declared
        row.update(_stat_one(tensor, chosen, reduction, effective_k, max_rows, rank_by))
        rows.append(row)
    header = {
        "population_total": total_selected,
        "rows_returned": len(rows),
        "target": target,
        "accumulation_dtype": "float64",
        "std_correction": 0,
        "ranking": "stable_flat_index_ties",
        "bytes_materialized": spent_bytes,
    }
    return rows, header


def _select_ops(log: Any, *, labels: list[str] | None, query: dict[str, Any] | None) -> list[Any]:
    """Resolve the site population for a stats call (execution-ordered)."""

    ops = list(getattr(log, "layer_list", []) or [])
    if labels is not None:
        wanted = {str(item) for item in labels}
        return [
            op
            for op in ops
            if str(getattr(op, "label", "")) in wanted
            or str(getattr(op, "layer_label", "")) in wanted
        ]
    if query is not None:
        from ._query import query_sites

        matched, _, _ = query_sites(log, query)
        matched_labels = {row["label"] for row in matched}
        return [op for op in ops if str(getattr(op, "label", "")) in matched_labels]
    return ops


def _stat_one(  # noqa: PLR0913 - one site's request facets, internal keyword threading
    tensor: Any,
    metrics: list[str],
    reduction: dict[str, Any] | None,
    k: int,
    max_rows: int,
    rank_by: str | None,
) -> dict[str, Any]:
    """Compute one site's requested metrics, with typed per-metric refusals."""

    result: dict[str, Any] = {"status": "ok"}
    reason = _unsupported_reason(tensor)
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        return {"status": "described", "description": reason}
    result["shape"] = [int(dim) for dim in tensor.shape]
    result["dtype"] = str(tensor.dtype)
    result["numel"] = int(tensor.numel())
    result["bytes"] = int(tensor.numel()) * int(tensor.element_size())
    value_metrics = [m for m in metrics if m not in ("shape", "dtype", "numel", "bytes")]
    if reason is not None and value_metrics:
        countable = tensor.is_complex() and not tensor.is_sparse and not tensor.is_quantized
        if countable and tensor.numel() > 0:
            magnitudes = tensor.detach().to("cpu").abs()
            result.update(_counts(magnitudes))
            result["metric_note"] = reason + " (counts computed on magnitudes)"
        else:
            result["status"] = "metrics_restricted"
            result["metric_note"] = reason
        return result
    working = tensor.detach().to("cpu")
    if any(
        m in value_metrics
        for m in ("finite_count", "nan_count", "posinf_count", "neginf_count", "zero_count")
    ):
        floatish = working.float() if working.dtype == torch.bool else working
        counts = _counts(floatish)
        result.update({key: counts[key] for key in counts if key in value_metrics})
    if reduction is not None:
        retain_dim = reduction.get("retain_dim")
        if not isinstance(retain_dim, int) or isinstance(retain_dim, bool):
            raise InvalidArgumentError(
                f"reduction={reduction!r} must name an integer retain_dim",
                code="agent_reduction_invalid",
                remedy='pass reduction={"retain_dim": <int>} (one explicit dimension INDEX)',
            )
        reduced, extent = _reduced_rows(
            working.float() if working.dtype == torch.bool else working,
            retain_dim,
            value_metrics,
            max_rows,
            rank_by,
        )
        result["reduction"] = {
            "retain_dim": retain_dim,
            "extent": extent,
            "rows": reduced,
            "reduced_axes": "all_but_retain_dim",
        }
        return result
    scalar_wanted = [
        m for m in value_metrics if m in ("min", "max", "mean", "std", "l1_norm", "l2_norm")
    ]
    if scalar_wanted:
        numeric = working.float() if working.dtype == torch.bool else working
        stats = _scalar_stats(numeric)
        result.update({key: stats[key] for key in scalar_wanted})
    if "top_k" in metrics:
        result["top_k"] = _extremes(
            working.float() if working.dtype == torch.bool else working, k, largest=True
        )
    if "bottom_k" in metrics:
        result["bottom_k"] = _extremes(
            working.float() if working.dtype == torch.bool else working, k, largest=False
        )
    return result

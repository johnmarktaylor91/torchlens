"""Parameter-level comparison between two models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._debug_report import DebugReport, table_report
from ._common import _require_pandas

if TYPE_CHECKING:
    pass


def _named_state(
    model: torch.nn.Module, include_buffers: bool
) -> dict[str, tuple[str, torch.Tensor]]:
    """Collect named parameters (and optionally buffers) from a model.

    Parameters
    ----------
    model:
        Source module.
    include_buffers:
        Whether registered buffers join the inventory.

    Returns
    -------
    dict[str, tuple[str, torch.Tensor]]
        Mapping of qualified name to ``(kind, tensor)`` with kind
        ``"parameter"`` or ``"buffer"``.
    """

    state: dict[str, tuple[str, torch.Tensor]] = {
        name: ("parameter", tensor) for name, tensor in model.named_parameters()
    }
    if include_buffers:
        for name, tensor in model.named_buffers():
            state.setdefault(name, ("buffer", tensor))
    return state


def _param_row(name: str, kind: str, status: str, reason: str) -> dict[str, Any]:
    """Build a placeholder comparison row for a one-sided or incomparable entry.

    Parameters
    ----------
    name:
        Qualified parameter name.
    kind:
        ``"parameter"`` or ``"buffer"``.
    status:
        Row status label.
    reason:
        Why no value comparison happened.

    Returns
    -------
    dict[str, Any]
        Row with value columns set to ``None``.
    """

    return {
        "name": name,
        "kind": kind,
        "status": status,
        "shape_match": None,
        "dtype_match": None,
        "max_abs": None,
        "mean_abs": None,
        "allclose": None,
        "reason": reason,
    }


def _comparable_values(
    tensor_a: torch.Tensor, tensor_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return detached same-device views of both tensors, or ``None``.

    Parameters
    ----------
    tensor_a:
        First tensor.
    tensor_b:
        Second tensor.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] | None
        Detached tensors on a shared device, or ``None`` when either side
        carries no data (meta tensors).
    """

    if tensor_a.is_meta or tensor_b.is_meta:
        return None
    value_a = tensor_a.detach()
    value_b = tensor_b.detach()
    if value_a.device != value_b.device:
        value_a = value_a.cpu()
        value_b = value_b.cpu()
    return value_a, value_b


def _value_comparison(
    value_a: torch.Tensor,
    value_b: torch.Tensor,
    rtol: float,
    atol: float,
) -> tuple[float, float, bool]:
    """Compare two same-shape, same-dtype detached tensors.

    Parameters
    ----------
    value_a:
        First tensor.
    value_b:
        Second tensor.
    rtol:
        Relative tolerance for ``torch.allclose``.
    atol:
        Absolute tolerance for ``torch.allclose``.

    Returns
    -------
    tuple[float, float, bool]
        ``(max_abs, mean_abs, allclose)``; integer and boolean tensors
        compare exactly.
    """

    if value_a.is_complex():
        delta = torch.abs(value_a.to(torch.complex128) - value_b.to(torch.complex128))
        allclose = bool(torch.allclose(value_a, value_b, rtol=rtol, atol=atol))
    elif value_a.is_floating_point():
        delta = torch.abs(value_a.to(torch.float64) - value_b.to(torch.float64))
        allclose = bool(torch.allclose(value_a, value_b, rtol=rtol, atol=atol))
    else:
        delta = torch.abs(value_a.long() - value_b.long())
        allclose = bool(torch.equal(value_a, value_b))
    max_abs = float(delta.max().item()) if delta.numel() else 0.0
    mean_abs = float(delta.float().mean().item()) if delta.numel() else 0.0
    return max_abs, mean_abs, allclose


def compare_params(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    include_buffers: bool = False,
) -> DebugReport:
    """Compare two models' parameters name by name.

    The counterpart of :func:`compare` for weights instead of activations:
    aligned on qualified parameter names, one row per name in either model,
    with aggregate counts in ``DataFrame.attrs``. Tensors on different
    devices are compared on CPU; meta tensors and dtype mismatches skip the
    value comparison with the reason recorded.

    Parameters
    ----------
    model_a:
        First model.
    model_b:
        Second model.
    rtol:
        Relative tolerance for ``torch.allclose``.
    atol:
        Absolute tolerance for ``torch.allclose``.
    include_buffers:
        Also compare registered buffers (rows carry ``kind="buffer"``).

    Returns
    -------
    DebugReport
        Tabular ``tl._debug`` report; ``.to_pandas()`` returns the
        DataFrame described here.
        One row per qualified name with columns ``name``, ``kind``,
        ``status``, ``shape_match``, ``dtype_match``, ``max_abs``,
        ``mean_abs``, ``allclose``, and ``reason``; ``attrs`` carries the
        summary counts.
    """

    pd = _require_pandas()
    state_a = _named_state(model_a, include_buffers)
    state_b = _named_state(model_b, include_buffers)
    names = sorted(set(state_a) | set(state_b))
    rows: list[dict[str, Any]] = []
    summary = {
        "matched": 0,
        "value_diverged": 0,
        "shape_mismatch": 0,
        "only_a": 0,
        "only_b": 0,
        "incomparable": 0,
    }

    for name in names:
        entry_a = state_a.get(name)
        entry_b = state_b.get(name)
        if entry_a is None or entry_b is None:
            present = entry_a if entry_a is not None else entry_b
            if present is None:  # pragma: no cover - names come from the union
                continue
            side = "only-a" if entry_b is None else "only-b"
            summary[side.replace("-", "_")] += 1
            rows.append(_param_row(name, present[0], side, side))
            continue

        kind_a, tensor_a = entry_a
        kind_b, tensor_b = entry_b
        kind = kind_a if kind_a == kind_b else f"{kind_a}/{kind_b}"
        shape_match = tuple(tensor_a.shape) == tuple(tensor_b.shape)
        dtype_match = tensor_a.dtype == tensor_b.dtype
        row = {
            "name": name,
            "kind": kind,
            "status": "present-in-both",
            "shape_match": shape_match,
            "dtype_match": dtype_match,
            "max_abs": None,
            "mean_abs": None,
            "allclose": None,
            "reason": "",
        }
        if not shape_match or not dtype_match:
            summary["shape_mismatch"] += 1
            row["reason"] = "shape-or-dtype-mismatch"
            rows.append(row)
            continue
        values = _comparable_values(tensor_a, tensor_b)
        if values is None:
            summary["incomparable"] += 1
            row["reason"] = "meta-tensor-has-no-data"
            rows.append(row)
            continue
        max_abs, mean_abs, allclose = _value_comparison(*values, rtol, atol)
        row["max_abs"] = max_abs
        row["mean_abs"] = mean_abs
        row["allclose"] = allclose
        if allclose:
            summary["matched"] += 1
        else:
            summary["value_diverged"] += 1
        rows.append(row)

    frame = pd.DataFrame(
        rows,
        columns=[
            "name",
            "kind",
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
    return table_report(
        "compare_params",
        frame,
        subject=f"{type(model_a).__name__} vs {type(model_b).__name__}",
        finding_columns=("name", "status", "max_abs", "reason"),
    )

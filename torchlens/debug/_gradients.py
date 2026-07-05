"""Gradient-flow audit helpers for TorchLens traces."""

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

from ._common import _compute_ops, _op_label, _safe_out, _tensor_unavailable_reason, _require_pandas


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

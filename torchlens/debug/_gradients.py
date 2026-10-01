"""Gradient-flow audit helpers for TorchLens traces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.trace import Trace

from ._common import _op_label, _require_pandas, _tensor_unavailable_reason

#: Column order shared by the rows core and the DataFrame view.
_GRAD_AUDIT_COLUMNS = ("op", "grad_norm", "vanishing", "exploding", "dead", "severity", "reason")


def _unavailable_row(label: str, reason: str) -> dict[str, Any]:
    """Build one audit row for a gradient that could not be measured.

    Parameters
    ----------
    label:
        Op label for the row.
    reason:
        Why the gradient payload was unavailable.

    Returns
    -------
    dict[str, Any]
        Zero-severity audit row carrying the reason.
    """

    return {
        "op": label,
        "grad_norm": None,
        "vanishing": False,
        "exploding": False,
        "dead": False,
        "severity": 0,
        "reason": reason,
    }


def _validate_grad_scale(grad_scale: float | None) -> float | None:
    """Normalize the AMP loss scale, refusing junk values typed.

    Parameters
    ----------
    grad_scale:
        User-supplied loss scale, or ``None``.

    Returns
    -------
    float | None
        The scale as a float, or ``None`` when not supplied.

    Raises
    ------
    InvalidArgumentError
        If the scale is not a positive finite number (code
        ``grad_scale_invalid``).
    """

    if grad_scale is None:
        return None
    from torchlens._errors import InvalidArgumentError

    scale_value = float(grad_scale)
    if not (0.0 < scale_value < float("inf")):
        raise InvalidArgumentError(
            f"grad_scale must be a positive finite number, got {grad_scale!r}",
            code="grad_scale_invalid",
            remedy=(
                "pass scaler.get_scale() from the GradScaler the captured "
                "backward ran under, or omit grad_scale to audit captured "
                "norms as-is"
            ),
        )
    return scale_value


def _audit_op_gradient(
    op: Any,
    *,
    bwd: int,
    grad_scale: float | None,
    vanishing_threshold: float,
    exploding_threshold: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Score one op's saved gradient for the flow audit.

    Parameters
    ----------
    op:
        Saved-gradient op record.
    bwd:
        One-based backward pass to read.
    grad_scale:
        Positive loss scale to divide out of the norm, or ``None``.
    vanishing_threshold:
        Norm below which a nonzero finite gradient is flagged vanishing.
    exploding_threshold:
        Norm above which a finite gradient is flagged exploding.

    Returns
    -------
    tuple[dict[str, Any] | None, str | None]
        ``(row, flag)`` where ``row`` is the audit row (``None`` when the
        payload is not a tensor) and ``flag`` is the counts key to increment
        (``"unavailable"``/``"dead"``/``"vanishing"``/``"exploding"``) or
        ``None`` for a clean measurable gradient.
    """

    label = _op_label(op)
    try:
        grad = op.grad_for(bwd=bwd)
    except (KeyError, ValueError) as exc:
        return _unavailable_row(label, str(exc)), "unavailable"
    reason = _tensor_unavailable_reason(grad)
    if reason is not None:
        return _unavailable_row(label, reason), "unavailable"
    if not isinstance(grad, torch.Tensor):
        return None, "unavailable"
    norm_tensor = torch.linalg.vector_norm(grad.detach())
    grad_norm = float(norm_tensor.item())
    if grad_scale is not None:
        grad_norm = grad_norm / grad_scale
    finite = bool(torch.isfinite(norm_tensor).item())
    dead = finite and grad_norm == 0.0
    vanishing = finite and 0.0 < grad_norm < vanishing_threshold
    exploding = (not finite) or grad_norm > exploding_threshold
    severity = int(exploding) * 3 + int(dead) * 2 + int(vanishing)
    flag = "exploding" if exploding else ("dead" if dead else ("vanishing" if vanishing else None))
    row = {
        "op": label,
        "grad_norm": grad_norm,
        "vanishing": vanishing,
        "exploding": exploding,
        "dead": dead,
        "severity": severity,
        "reason": "",
    }
    return row, flag


def gradient_flow_audit_rows(
    trace: Trace,
    *,
    bwd: int | None = None,
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 1e4,
    grad_scale: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pandas-free core of :func:`gradient_flow_audit` (agent stage-0 item 4).

    The wire path for agent/MCP consumers: plain row dicts plus the attrs
    mapping, importable and runnable without pandas.
    :func:`gradient_flow_audit` is the optional DataFrame view over this
    core. Empty-result statuses (torch-only subject, no saved gradients,
    missing ``bwd`` selector) return zero rows with the status under
    ``attrs["message"]``.

    AMP disclosure (list-A row 11): gradients captured from a backward run
    under ``torch.amp.GradScaler`` carry the loss scale (~2**16 at the default
    ``init_scale``) -- TorchLens records the gradients that REALLY flowed, and
    the scaler unscales only leaf ``param.grad`` during ``scaler.step``, never
    the intermediate gradients this audit reads. Unscaled thresholds then flag
    every op exploding. Pass ``grad_scale=scaler.get_scale()`` to audit in
    unscaled units; the applied scale is disclosed in the attrs mapping.

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
    grad_scale:
        Positive loss scale the captured backward ran under (AMP:
        ``scaler.get_scale()``). Every gradient norm is divided by it before
        thresholding and reported in unscaled units; ``attrs["grad_scale"]``
        discloses the value applied. ``None`` (default) audits the captured
        norms as-is.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Severity-ranked audit row dicts and the attrs mapping (counts,
        thresholds, capture-honesty facts on the audited path). When every
        measurable gradient flags exploding without ``grad_scale``,
        ``attrs["all_exploding_hint"]`` names the AMP scaled-gradients
        possibility.

    Raises
    ------
    InvalidArgumentError
        If ``grad_scale`` is not a positive finite number (code
        ``grad_scale_invalid``).
    """

    grad_scale = _validate_grad_scale(grad_scale)
    try:
        backward_passes = trace.backward_passes
        saved_grad_ops = trace.saved_grad_ops
    except ValueError:
        return [], {"message": "torch-only", "torch_only": True}

    num_backward = len(backward_passes)
    if num_backward == 0 or len(saved_grad_ops) == 0:
        return [], {
            "message": "no saved gradients; re-trace backward_ready=True + trace.log_backward(loss)",
            "vanishing": 0,
            "exploding": 0,
            "dead": 0,
        }
    if num_backward > 1 and bwd is None:
        return [], {
            "message": "bwd is required for multi-backward-pass traces",
            "backward_passes": num_backward,
        }
    selected_bwd = bwd if bwd is not None else 1

    rows: list[dict[str, Any]] = []
    counts = {"vanishing": 0, "exploding": 0, "dead": 0, "unavailable": 0}
    measurable = 0
    for op in saved_grad_ops:
        row, flag = _audit_op_gradient(
            op,
            bwd=selected_bwd,
            grad_scale=grad_scale,
            vanishing_threshold=vanishing_threshold,
            exploding_threshold=exploding_threshold,
        )
        if flag == "unavailable":
            counts["unavailable"] += 1
        else:
            measurable += 1
            if flag is not None:
                counts[flag] += 1
        if row is not None:
            rows.append(row)

    from .._capture_honesty import capture_honesty_facts

    sorted_rows = sorted(rows, key=lambda row: (int(row["severity"]), str(row["op"])), reverse=True)
    attrs: dict[str, Any] = {
        **counts,
        "bwd": selected_bwd,
        "vanishing_threshold": vanishing_threshold,
        "exploding_threshold": exploding_threshold,
        "torchlens_capture_honesty": capture_honesty_facts(trace),
    }
    if grad_scale is not None:
        attrs["grad_scale"] = grad_scale
    if grad_scale is None and measurable > 0 and counts["exploding"] == measurable:
        # Deterministic disclosure, not a guess: EVERY measurable gradient
        # exceeded the threshold, which is exactly the signature of a scaled
        # AMP backward (list-A row 11). The hint names the possibility and the
        # remedy; it never edits the rows.
        attrs["all_exploding_hint"] = (
            "every measurable gradient flags exploding; if this backward ran "
            "under torch.amp.GradScaler the captured gradients carry the loss "
            "scale (~2**16 at the default init_scale) -- pass "
            "grad_scale=scaler.get_scale() to audit in unscaled units"
        )
    return sorted_rows, attrs


def gradient_flow_audit(
    trace: Trace,
    *,
    bwd: int | None = None,
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 1e4,
    grad_scale: float | None = None,
) -> pd.DataFrame:
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
    grad_scale:
        Positive loss scale the captured backward ran under (AMP:
        ``scaler.get_scale()``); see :func:`gradient_flow_audit_rows`.

    Returns
    -------
    pandas.DataFrame
        Ranked audit rows with counts in ``df.attrs``. The optional view over
        :func:`gradient_flow_audit_rows` (the pandas-free core). When every
        measurable gradient flags exploding without ``grad_scale``,
        ``df.attrs["all_exploding_hint"]`` names the AMP scaled-gradients
        possibility.

    Raises
    ------
    InvalidArgumentError
        If ``grad_scale`` is not a positive finite number (code
        ``grad_scale_invalid``).
    """

    rows, attrs = gradient_flow_audit_rows(
        trace,
        bwd=bwd,
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
        grad_scale=grad_scale,
    )
    pd = _require_pandas()
    frame = pd.DataFrame(rows, columns=list(_GRAD_AUDIT_COLUMNS))
    frame.attrs.update(attrs)
    return frame

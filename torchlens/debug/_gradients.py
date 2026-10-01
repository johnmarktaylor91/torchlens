"""Gradient-flow audit helpers for TorchLens traces.

Fix pack (checks memo item 2 / D17), all ADDITIVE:

- ``module`` + ``is_frontier`` columns: a dead layer zeroes the gradient of
  everything upstream, so the naive report names ~all of the cone; the
  FRONTIER (a flagged op none of whose children carry the same flag) names
  the culprit (memo D3 -- the composition is load-bearing).
- ``grad_norm_raw`` / ``rms`` / ``numel`` columns: the fixed absolute
  threshold trips at a ~128x different per-element magnitude across tensors
  of one model (numel artifact); ``rms = norm / sqrt(numel)`` kills it.
- ONE batched host sync per device (the historical per-op ``.item()`` was a
  device sync per op).
- ``stage`` + ``scale_provenance`` on every row; per-backward scale STAMPS
  via ``grad_scales={bwd: scale}``; UNKNOWN provenance yields UNKNOWN
  magnitude verdicts in the new ``verdict`` column, never a silent
  pass-as-1.0.
- ``basis=`` selector: the legacy total-norm classification stays the
  DEFAULT for one deprecation cycle; the flip to ``basis="rms"`` is
  announced in ``attrs["basis_notice"]`` and gated on the multi-model
  calibration runs -- changing a shipped verdict semantics unannounced is
  the defect class the checks panel exists to prevent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.trace import Trace

from ._common import _op_label, _require_pandas, _tensor_unavailable_reason

#: Column order shared by the rows core and the DataFrame view. The first
#: seven are the historical columns (semantics unchanged under the default
#: basis); the rest are the D17 fix-pack additions.
_GRAD_AUDIT_COLUMNS = (
    "op",
    "grad_norm",
    "vanishing",
    "exploding",
    "dead",
    "severity",
    "reason",
    "module",
    "is_frontier",
    "grad_norm_raw",
    "rms",
    "numel",
    "verdict",
    "stage",
    "scale_provenance",
)

#: Closed classification bases (D17): legacy total-norm default, RMS opt-in.
_BASES = ("total_norm", "rms")

#: The planned-flip announcement (deprecation notice, D17).
_BASIS_NOTICE = (
    "basis='total_norm' is the legacy classification and remains the default "
    "for one deprecation cycle; the flip to basis='rms' (norm/sqrt(numel), "
    "killing the ~128x numel artifact) is announced and gated on the "
    "multi-model calibration runs. Opt in with basis='rms' today."
)


def _unavailable_row(label: str, module: str | None, reason: str) -> dict[str, Any]:
    """Build one audit row for a gradient that could not be measured.

    Parameters
    ----------
    label:
        Op label for the row.
    module:
        Atomic module address for the row, if known.
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
        "module": module,
        "is_frontier": False,
        "grad_norm_raw": None,
        "rms": None,
        "numel": None,
        "verdict": "unavailable",
        "stage": "pre_clip",
        "scale_provenance": "unknown",
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


def _validate_options(
    grad_scale: float | None,
    grad_scales: Mapping[int, float] | None,
    basis: str,
) -> tuple[float | None, dict[int, float] | None]:
    """Validate the fix-pack option surface, refusing typed on misuse.

    Raises
    ------
    InvalidArgumentError
        On an unknown ``basis`` (code ``grad_basis_invalid``) or a junk
        scale value (``grad_scale_invalid``).
    KeywordConflictError
        When ``grad_scale`` and ``grad_scales`` are both passed (code
        ``grad_scale_conflict``): one scalar for the selected pass and
        per-backward stamps are two spellings of the same fact.
    """

    from torchlens._errors import InvalidArgumentError, KeywordConflictError

    if basis not in _BASES:
        raise InvalidArgumentError(
            f"basis must be one of {_BASES}, got {basis!r}",
            code="grad_basis_invalid",
            remedy="pass basis='total_norm' (legacy default) or basis='rms'",
        )
    if grad_scale is not None and grad_scales is not None:
        raise KeywordConflictError(
            "pass grad_scale= (one scalar for the selected backward) OR "
            "grad_scales= (per-backward stamps), not both",
            code="grad_scale_conflict",
            remedy="drop one of the two spellings; grad_scales={bwd: scale} covers the multi-backward case",
        )
    validated_scales: dict[int, float] | None = None
    if grad_scales is not None:
        validated_scales = {}
        for key, value in grad_scales.items():
            coerced = _validate_grad_scale(value if value is not None else math.nan)
            # _validate_grad_scale refuses non-finite input typed, so the
            # fallback arm is unreachable; it exists to keep the narrow
            # explicit without an assert.
            validated_scales[int(key)] = coerced if coerced is not None else math.nan
    return _validate_grad_scale(grad_scale), validated_scales


def _bare_label(label: str) -> str:
    """Strip the ``:pass`` suffix from a pass-qualified op label."""

    head, _, tail = label.rpartition(":")
    return head if head and tail.isdigit() else label


def _mark_frontiers(
    rows: list[dict[str, Any]], children_by_label: dict[str, tuple[str, ...]]
) -> None:
    """Mark each flagged row whose children do not share its flag (D3).

    The gradient cone propagates a flag UPSTREAM (a dead layer zeroes every
    ancestor's gradient), so the culprit is the flagged op closest to the
    healthy side: a flagged row is a frontier when no child row carries the
    same flag. Children outside the audited set count as unflagged.
    """

    flags_by_bare: dict[str, set[str]] = {}
    for row in rows:
        flag = (
            row["verdict"]
            if row["verdict"] in ("dead", "vanishing", "exploding", "nonfinite")
            else None
        )
        if flag is None and (row["exploding"] or row["dead"] or row["vanishing"]):
            flag = "exploding" if row["exploding"] else ("dead" if row["dead"] else "vanishing")
        row["_flag"] = flag
        if flag is not None:
            flags_by_bare.setdefault(_bare_label(str(row["op"])), set()).add(flag)
    for row in rows:
        flag = row.pop("_flag")
        if flag is None:
            continue
        children = children_by_label.get(str(row["op"]), ())
        row["is_frontier"] = not any(
            flag in flags_by_bare.get(_bare_label(child), set()) for child in children
        )


@dataclass(frozen=True)
class _AuditScaleContext:
    """Resolved scale/basis facts one audit run classifies under (D17)."""

    scale: float | None
    provenance: str
    basis: str
    vanishing_threshold: float
    exploding_threshold: float


def _collect_norm_rows(
    saved_grad_ops: Any,
    selected_bwd: int,
    provenance: str,
    counts: dict[str, int],
) -> tuple[
    list[dict[str, Any]], list[tuple[dict[str, Any], torch.Tensor]], dict[str, tuple[str, ...]]
]:
    """Phase A: collect per-op 0-dim norm tensors -- NO host sync here.

    The historical per-op ``.item()`` was a device sync per op; this phase
    only stages 0-dim norm tensors for the phase-B batched sync.
    """

    rows: list[dict[str, Any]] = []
    measurable_entries: list[tuple[dict[str, Any], torch.Tensor]] = []
    children_by_label: dict[str, tuple[str, ...]] = {}
    for op in saved_grad_ops:
        label = _op_label(op)
        module = getattr(op, "atomic_module_address", None)
        try:
            grad = op.grad_for(bwd=selected_bwd)
        except (KeyError, ValueError) as exc:
            rows.append(_unavailable_row(label, module, str(exc)))
            counts["unavailable"] += 1
            continue
        reason = _tensor_unavailable_reason(grad)
        if reason is not None:
            rows.append(_unavailable_row(label, module, reason))
            counts["unavailable"] += 1
            continue
        if not isinstance(grad, torch.Tensor):
            counts["unavailable"] += 1
            continue
        row: dict[str, Any] = {
            "op": label,
            "module": module,
            "numel": grad.numel(),
            "stage": "pre_clip",
            "scale_provenance": provenance,
            "is_frontier": False,
            "reason": "",
        }
        children_by_label[label] = tuple(getattr(op, "children", ()) or ())
        measurable_entries.append((row, torch.linalg.vector_norm(grad.detach())))
        rows.append(row)
    return rows, measurable_entries, children_by_label


def _sync_raw_norms(measurable_entries: list[tuple[dict[str, Any], torch.Tensor]]) -> None:
    """Phase B: ONE batched host sync per device group (writes ``_raw_norm``)."""

    by_device: dict[str, list[tuple[dict[str, Any], torch.Tensor]]] = {}
    for row, norm in measurable_entries:
        by_device.setdefault(str(norm.device), []).append((row, norm))
    for group in by_device.values():
        stacked = torch.stack([norm.to(torch.float64) for _, norm in group])
        values = stacked.cpu().tolist()  # the ONE sync for this device group
        for (row, _), value in zip(group, values, strict=True):
            row["_raw_norm"] = value


def _verdict_for(
    finite: bool, dead: bool, vanishing: bool, exploding: bool, provenance: str
) -> str:
    """Return one row's verdict token under the D17 provenance rule.

    Magnitude claims are scale-dependent: unknown provenance yields UNKNOWN
    verdicts, never a silent pass-as-1.0. ``dead`` and ``nonfinite`` are
    scale-invariant and survive regardless of provenance.
    """

    if not finite:
        return "nonfinite"
    if dead:
        return "dead"
    if provenance == "unknown":
        return "unknown"
    if exploding:
        return "exploding"
    if vanishing:
        return "vanishing"
    return "ok"


def _classify_rows(
    measurable_entries: list[tuple[dict[str, Any], torch.Tensor]],
    ctx: _AuditScaleContext,
    counts: dict[str, int],
) -> int:
    """Phase C: classify each measurable row in place; return the count."""

    measurable = 0
    for row, _ in measurable_entries:
        raw_norm = row.pop("_raw_norm")
        grad_norm = raw_norm / ctx.scale if ctx.scale is not None else raw_norm
        numel = row["numel"]
        rms = grad_norm / math.sqrt(numel) if numel else None
        finite = math.isfinite(grad_norm)
        if ctx.basis == "total_norm":
            classified = grad_norm
        else:
            classified = rms if rms is not None else grad_norm
        dead = finite and grad_norm == 0.0
        vanishing = finite and 0.0 < classified < ctx.vanishing_threshold
        exploding = (not finite) or classified > ctx.exploding_threshold
        verdict = _verdict_for(finite, dead, vanishing, exploding, ctx.provenance)
        if verdict == "unknown":
            counts["unknown_verdicts"] += 1
        severity = int(exploding) * 3 + int(dead) * 2 + int(vanishing)
        # Carrier attribution repair (observe item 10): a per-op gradient is
        # w.r.t. the op's OUTPUT, so a NON-FINITE gradient observed on X was
        # BORN in the backward of one of X's CONSUMERS and merely LANDED here.
        # The row keeps its severity (the landing is real) but is labeled a
        # carrier observation, and the birth question routes to the
        # fire-order bisector.
        if not finite:
            row["reason"] = (
                "non-finite gradient LANDED on this op (carrier observation, not the "
                "birth site -- per-op grads are w.r.t. the op's OUTPUT, so the birth is "
                "in a CONSUMER's backward); run tl.debug.bisect_nan_backward(trace, "
                "bwd=N) to localize the birth"
            )
        row.update(
            {
                "grad_norm": grad_norm,
                "grad_norm_raw": raw_norm,
                "rms": rms,
                "vanishing": vanishing,
                "exploding": exploding,
                "dead": dead,
                "verdict": verdict,
                "severity": severity,
            }
        )
        measurable += 1
        if exploding:
            counts["exploding"] += 1
        elif dead:
            counts["dead"] += 1
        elif vanishing:
            counts["vanishing"] += 1
    return measurable


def gradient_flow_audit_rows(  # noqa: PLR0913 -- the D17 fix-pack public surface: shipped kwargs + scale stamps + basis selector, all additive by contract
    trace: Trace,
    *,
    bwd: int | None = None,
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 1e4,
    grad_scale: float | None = None,
    grad_scales: Mapping[int, float] | None = None,
    basis: str = "total_norm",
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
    every op exploding. Pass ``grad_scale=scaler.get_scale()`` (or per-backward
    ``grad_scales={bwd: scale}`` stamps captured AT each backward, never read
    off the scaler later) to audit in unscaled units; the applied scale and
    its provenance are disclosed per row and in the attrs mapping. UNKNOWN
    provenance yields UNKNOWN magnitude verdicts in the ``verdict`` column
    (the scale-invariant ``dead`` and ``nonfinite`` verdicts survive); the
    legacy boolean columns keep their shipped classification through the
    ``basis`` deprecation window (``attrs["basis_notice"]``).

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    bwd:
        One-based backward pass selector. Required when multiple backward passes
        are captured.
    vanishing_threshold:
        Norm below which a nonzero finite gradient is flagged vanishing
        (interpreted on the selected ``basis``).
    exploding_threshold:
        Norm above which a finite gradient is flagged exploding (interpreted
        on the selected ``basis``).
    grad_scale:
        Positive loss scale the captured backward ran under (AMP:
        ``scaler.get_scale()``). Every gradient norm is divided by it before
        thresholding and reported in unscaled units; ``attrs["grad_scale"]``
        discloses the value applied. ``None`` (default) audits the captured
        norms as-is with ``scale_provenance="unknown"``.
    grad_scales:
        Per-backward scale stamps ``{bwd: scale}`` for multi-backward traces
        whose passes ran under different scales; each audited pass applies
        ITS stamp. Mutually exclusive with ``grad_scale``.
    basis:
        Classification basis for the boolean columns and the ``verdict``
        column: ``"total_norm"`` (legacy default) or ``"rms"``
        (``norm / sqrt(numel)``).

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, Any]]
        Severity-ranked audit row dicts and the attrs mapping (counts,
        thresholds, basis + deprecation notice, capture-honesty facts on the
        audited path). When every measurable gradient flags exploding
        without a scale, ``attrs["all_exploding_hint"]`` names the AMP
        scaled-gradients possibility.

    Raises
    ------
    InvalidArgumentError
        If a scale is not a positive finite number (``grad_scale_invalid``)
        or ``basis`` is unknown (``grad_basis_invalid``).
    KeywordConflictError
        If both ``grad_scale`` and ``grad_scales`` are passed
        (``grad_scale_conflict``).
    """

    grad_scale, grad_scales = _validate_options(grad_scale, grad_scales, basis)
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
    scale = grad_scales.get(selected_bwd) if grad_scales is not None else grad_scale
    ctx = _AuditScaleContext(
        scale=scale,
        provenance="explicit" if scale is not None else "unknown",
        basis=basis,
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
    )

    counts = {"vanishing": 0, "exploding": 0, "dead": 0, "unavailable": 0, "unknown_verdicts": 0}
    rows, measurable_entries, children_by_label = _collect_norm_rows(
        saved_grad_ops, selected_bwd, ctx.provenance, counts
    )
    _sync_raw_norms(measurable_entries)
    measurable = _classify_rows(measurable_entries, ctx, counts)
    _mark_frontiers(rows, children_by_label)

    from .._capture_honesty import capture_honesty_facts

    sorted_rows = sorted(
        rows,
        key=lambda row: (int(row["severity"]), bool(row["is_frontier"]), str(row["op"])),
        reverse=True,
    )
    attrs: dict[str, Any] = {
        **counts,
        "bwd": selected_bwd,
        "vanishing_threshold": vanishing_threshold,
        "exploding_threshold": exploding_threshold,
        "basis": basis,
        "scale_provenance": ctx.provenance,
        "torchlens_capture_honesty": capture_honesty_facts(trace),
    }
    if basis == "total_norm":
        attrs["basis_notice"] = _BASIS_NOTICE
    if scale is not None:
        attrs["grad_scale"] = scale
    if grad_scales is not None:
        attrs["grad_scales"] = dict(grad_scales)
    if scale is None and measurable > 0 and counts["exploding"] == measurable:
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


def gradient_flow_audit(  # noqa: PLR0913 -- the D17 fix-pack public surface: shipped kwargs + scale stamps + basis selector, all additive by contract
    trace: Trace,
    *,
    bwd: int | None = None,
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 1e4,
    grad_scale: float | None = None,
    grad_scales: Mapping[int, float] | None = None,
    basis: str = "total_norm",
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
    grad_scales:
        Per-backward scale stamps ``{bwd: scale}``; see
        :func:`gradient_flow_audit_rows`.
    basis:
        ``"total_norm"`` (legacy default, deprecation window announced in
        ``attrs["basis_notice"]``) or ``"rms"``.

    Returns
    -------
    pandas.DataFrame
        Ranked audit rows with counts in ``df.attrs``. The optional view over
        :func:`gradient_flow_audit_rows` (the pandas-free core). New fix-pack
        columns: ``module``, ``is_frontier`` (the culprit localizer, memo
        D3), ``grad_norm_raw``, ``rms``, ``numel``, ``verdict``, ``stage``,
        ``scale_provenance``.

    Raises
    ------
    InvalidArgumentError
        If a scale is not a positive finite number (code
        ``grad_scale_invalid``) or ``basis`` is unknown
        (``grad_basis_invalid``).
    KeywordConflictError
        If both scale spellings are passed (``grad_scale_conflict``).
    """

    rows, attrs = gradient_flow_audit_rows(
        trace,
        bwd=bwd,
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
        grad_scale=grad_scale,
        grad_scales=grad_scales,
        basis=basis,
    )
    pd = _require_pandas()
    frame = pd.DataFrame(rows, columns=list(_GRAD_AUDIT_COLUMNS))
    frame.attrs.update(attrs)
    return frame

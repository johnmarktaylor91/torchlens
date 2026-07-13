"""Structured replay-failure diagnostics for forward validation.

This module is **ADD-ONLY relative to the pass/fail decision**: nothing here
changes whether ``validate_saved_outs`` returns ``True`` or ``False``. Its sole
purpose is to capture *why* a validation failed so the reason can reach a caller
(e.g. the model-menagerie ledger) instead of the bare ``repr(False) == "False"``.

The validation core (``core.py``) records a :class:`ValidationFailure` on a
side-channel attached to the live ``Trace`` at the FIRST mismatch it surfaces.
Recording is best-effort and wrapped so a diagnostics bug can never flip a
validation result or raise out of the tripwire path: the worst case is a missing
diagnostic, never a changed decision.

LOCKED: this never reads, relaxes, or short-circuits any ``allclose`` /
tolerance / equality check. It only *describes* a failure the core already
decided on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

# Attribute name used to stash the structured failure on a live Trace. Private,
# best-effort, and cleared at the start of each validation run so a stale
# failure from a prior run can never be reported.
TRACE_FAILURE_ATTR = "_last_validation_failure"
TRACE_DIAGNOSTICS_ATTR = "_validation_diagnostics"

# The named validation checks a failure can originate from. Mirrors the phases
# documented in ``core.validate_saved_outs``.
CHECK_GROUND_TRUTH = "ground_truth"
CHECK_OUTPUT_MISSING = "ground_truth_missing_out"
CHECK_REPLAY = "forward_replay"
CHECK_PERTURBATION = "perturbation"
CHECK_ARG_LOGGING = "argument_logging"
CHECK_COMPLETENESS = "bfs_completeness"
CHECK_METADATA_INVARIANT = "metadata_invariant"


@dataclass
class ValidationFailure:
    """A structured description of the first surfaced validation mismatch.

    Every field is optional so partial information is still useful. ``check`` is
    always populated and names which validation phase fired. The tensor-shaped
    fields (``saved_shape`` etc.) are filled in only for the numeric replay /
    ground-truth paths where a saved-vs-expected tensor pair exists.
    """

    check: str
    op_label: Optional[str] = None
    func_name: Optional[str] = None
    message: str = ""
    saved_shape: Optional[tuple[int, ...]] = None
    recomputed_shape: Optional[tuple[int, ...]] = None
    saved_dtype: Optional[str] = None
    recomputed_dtype: Optional[str] = None
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    nan_mismatch: Optional[bool] = None
    inf_mismatch: Optional[bool] = None
    reduction_depth: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a compact single-line human-readable summary.

        Returns
        -------
        str
            A ledger-friendly description, e.g.
            ``"forward_replay mismatch at conv2d_3_4 (conv2d): max_abs_diff=1.2e-01
            saved=(1,64,7,7) f32 vs recomputed=(1,64,7,7) f32"``.
        """

        parts: list[str] = [f"{self.check} mismatch"]
        if self.op_label is not None:
            loc = self.op_label
            if self.func_name:
                loc = f"{loc} ({self.func_name})"
            parts.append(f"at {loc}")
        metrics: list[str] = []
        if self.max_abs_diff is not None:
            metrics.append(f"max_abs_diff={self.max_abs_diff:.3e}")
        if self.max_rel_diff is not None:
            metrics.append(f"max_rel_diff={self.max_rel_diff:.3e}")
        if self.nan_mismatch:
            metrics.append("nan_pattern_differs")
        if self.inf_mismatch:
            metrics.append("inf_pattern_differs")
        if self.reduction_depth is not None:
            metrics.append(f"reduction_depth={self.reduction_depth}")
        if metrics:
            parts.append(": " + " ".join(metrics))
        shapes: list[str] = []
        if self.saved_shape is not None or self.saved_dtype is not None:
            shapes.append(f"saved={self.saved_shape} {self.saved_dtype or ''}".strip())
        if self.recomputed_shape is not None or self.recomputed_dtype is not None:
            shapes.append(
                f"recomputed={self.recomputed_shape} {self.recomputed_dtype or ''}".strip()
            )
        if shapes:
            parts.append("(" + " vs ".join(shapes) + ")")
        if self.message:
            parts.append(f"-- {self.message}")
        return " ".join(parts).replace(" : ", ": ")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view suitable for JSON/ledger serialization."""

        return {
            "check": self.check,
            "op_label": self.op_label,
            "func_name": self.func_name,
            "message": self.message,
            "saved_shape": list(self.saved_shape) if self.saved_shape is not None else None,
            "recomputed_shape": (
                list(self.recomputed_shape) if self.recomputed_shape is not None else None
            ),
            "saved_dtype": self.saved_dtype,
            "recomputed_dtype": self.recomputed_dtype,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "nan_mismatch": self.nan_mismatch,
            "inf_mismatch": self.inf_mismatch,
            "reduction_depth": self.reduction_depth,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class ValidationDiagnostic:
    """Structured non-failing diagnostic emitted by validation.

    Parameters
    ----------
    check:
        Stable name for the validation diagnostic source.
    message:
        Human-readable explanation of the diagnostic.
    extra:
        Additional JSON-friendly context for downstream reporting.
    """

    check: str
    message: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of this diagnostic.

        Returns
        -------
        dict[str, Any]
            Stable structured diagnostic payload.
        """

        return {"check": self.check, "message": self.message, "extra": dict(self.extra)}


def reset_validation_failure(trace: Any) -> None:
    """Clear any previously-recorded failure on a Trace (best-effort, no raise)."""

    try:
        setattr(trace, TRACE_FAILURE_ATTR, None)
    except Exception:
        pass


def record_validation_failure(trace: Any, failure: ValidationFailure) -> None:
    """Record the FIRST failure on a Trace; later calls are ignored.

    Recording is strictly first-write-wins so the reported reason is the first
    mismatch the BFS surfaced (matching the existing ``return False`` ordering in
    ``core.py``). Best-effort and exception-swallowing: a diagnostics failure
    must never alter a validation decision or raise out of the tripwire path.

    Parameters
    ----------
    trace:
        Live validation Trace acting as the side-channel carrier.
    failure:
        Structured failure description to stash.
    """

    try:
        if getattr(trace, TRACE_FAILURE_ATTR, None) is None:
            setattr(trace, TRACE_FAILURE_ATTR, failure)
    except Exception:
        pass


def get_validation_failure(trace: Any) -> Optional[ValidationFailure]:
    """Return the recorded structured failure on a Trace, if any."""

    try:
        failure = getattr(trace, TRACE_FAILURE_ATTR, None)
    except Exception:
        return None
    return failure if isinstance(failure, ValidationFailure) else None


def record_validation_diagnostic(trace: Any, diagnostic: ValidationDiagnostic) -> None:
    """Append a non-failing validation diagnostic to a live trace.

    Parameters
    ----------
    trace:
        Live validation trace that carries the diagnostic side channel.
    diagnostic:
        Diagnostic to retain for callers that need more than a warning string.

    Notes
    -----
    Recording is deliberately best-effort: diagnostics must never turn a
    validation warning into a validation failure.
    """

    try:
        existing = getattr(trace, TRACE_DIAGNOSTICS_ATTR, ())
        diagnostics = list(existing) if isinstance(existing, (list, tuple)) else []
        diagnostics.append(diagnostic)
        setattr(trace, TRACE_DIAGNOSTICS_ATTR, tuple(diagnostics))
    except Exception:
        pass


def get_validation_diagnostics(trace: Any) -> tuple[ValidationDiagnostic, ...]:
    """Return validation diagnostics retained on a live trace.

    Parameters
    ----------
    trace:
        Trace potentially carrying validation diagnostics.

    Returns
    -------
    tuple[ValidationDiagnostic, ...]
        Diagnostics in emission order, or an empty tuple when unavailable.
    """

    try:
        diagnostics = getattr(trace, TRACE_DIAGNOSTICS_ATTR, ())
    except Exception:
        return ()
    if not isinstance(diagnostics, (list, tuple)):
        return ()
    return tuple(item for item in diagnostics if isinstance(item, ValidationDiagnostic))


def _short_dtype(dtype: Any) -> Optional[str]:
    """Return a compact dtype string such as ``f32`` / ``i64`` (best-effort)."""

    if dtype is None:
        return None
    text = str(dtype).replace("torch.", "")
    return {
        "float32": "f32",
        "float64": "f64",
        "float16": "f16",
        "bfloat16": "bf16",
        "int64": "i64",
        "int32": "i32",
        "int16": "i16",
        "int8": "i8",
        "uint8": "u8",
        "bool": "bool",
        "complex64": "c64",
        "complex128": "c128",
    }.get(text, text)


def describe_tensor_mismatch(
    saved: Any,
    recomputed: Any,
    *,
    check: str,
    op_label: Optional[str] = None,
    func_name: Optional[str] = None,
    reduction_depth: Optional[int] = None,
    message: str = "",
) -> ValidationFailure:
    """Build a :class:`ValidationFailure` describing a saved-vs-recomputed mismatch.

    Computes shapes, dtypes, max abs/rel diff, and nan/inf pattern mismatches for
    a tensor pair. Every metric op is wrapped so a measurement error degrades to
    a partial diagnostic, never an exception. **Read-only on the tensors** -- it
    measures, it does not compare-to-decide.

    Parameters
    ----------
    saved:
        The tensor TorchLens captured for this op.
    recomputed:
        The tensor produced by isolated replay (or the ground-truth forward).
    check:
        Which named validation phase fired (one of the ``CHECK_*`` constants).
    op_label, func_name:
        Identifying metadata for the divergent op.
    reduction_depth:
        Per-output FP32 accumulation depth, when the caller already computed it.
    message:
        Optional extra context string.

    Returns
    -------
    ValidationFailure
        A populated failure record (always returned; never raises).
    """

    failure = ValidationFailure(
        check=check,
        op_label=op_label,
        func_name=func_name,
        message=message,
        reduction_depth=reduction_depth,
    )

    saved_t = saved if isinstance(saved, torch.Tensor) else None
    recomp_t = recomputed if isinstance(recomputed, torch.Tensor) else None

    try:
        if saved_t is not None:
            failure.saved_shape = tuple(saved_t.shape)
            failure.saved_dtype = _short_dtype(saved_t.dtype)
        else:
            failure.extra["saved_type"] = type(saved).__name__
        if recomp_t is not None:
            failure.recomputed_shape = tuple(recomp_t.shape)
            failure.recomputed_dtype = _short_dtype(recomp_t.dtype)
        else:
            failure.extra["recomputed_type"] = type(recomputed).__name__
    except Exception:
        return failure

    if saved_t is None or recomp_t is None:
        return failure
    if saved_t.shape != recomp_t.shape or saved_t.dtype != recomp_t.dtype:
        # Shape/dtype divergence is the whole story; element-wise diff undefined.
        return failure

    from .._state import pause_logging

    try:
        with pause_logging():
            if saved_t.is_meta or recomp_t.is_meta:
                return failure
            saved_f = saved_t.detach().to(torch.float64).reshape(-1)
            recomp_f = recomp_t.detach().to(torch.float64).reshape(-1)
            if saved_t.is_floating_point() or saved_t.is_complex():
                saved_nan = torch.isnan(saved_t.detach())
                recomp_nan = torch.isnan(recomp_t.detach())
                failure.nan_mismatch = not bool(torch.equal(saved_nan, recomp_nan))
                saved_inf = torch.isinf(saved_t.detach())
                recomp_inf = torch.isinf(recomp_t.detach())
                failure.inf_mismatch = not bool(torch.equal(saved_inf, recomp_inf))
                # Replace non-finite with a sentinel so the magnitude metrics are
                # finite and meaningful even when the nan/inf patterns differ.
                saved_f = torch.nan_to_num(saved_f, nan=0.0, posinf=0.0, neginf=0.0)
                recomp_f = torch.nan_to_num(recomp_f, nan=0.0, posinf=0.0, neginf=0.0)
            if saved_f.numel() > 0:
                abs_diff = (saved_f - recomp_f).abs()
                failure.max_abs_diff = float(abs_diff.max().item())
                denom = saved_f.abs().clamp_min(1e-12)
                failure.max_rel_diff = float((abs_diff / denom).max().item())
    except Exception:
        # Partial diagnostic is fine; never let measurement raise.
        pass
    return failure

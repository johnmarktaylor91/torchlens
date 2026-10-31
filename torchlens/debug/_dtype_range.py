"""Saved-activation numeric range and precision diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._debug_report import DebugReport, rows_of
from ._audit import AuditFinding, AuditSeverity
from ._common import _compute_ops, _op_label, _safe_out
from ._nan import _module_address, _nonfinite_kind

if TYPE_CHECKING:
    from torchlens.data_classes.trace import Trace


@dataclass(frozen=True)
class DTypeRangeAudit(DebugReport):
    """Coverage-bounded dtype range audit result.

    Attributes
    ----------
    findings:
        Evidence-backed findings following the :class:`AuditFinding` shape.
    n_ops_audited:
        Operations with one saved native tensor payload that was inspected.
    n_ops_total:
        Compute and output operations present in the trace.
    """

    findings: tuple[AuditFinding, ...]
    n_ops_audited: int
    n_ops_total: int

    _prints_repr = True

    def _report_status(self) -> str:
        """The verdict token on the card's lead line."""

        return f"{len(self.findings)} findings" if self.findings else "clean"

    def _report_coverage(self) -> str | None:
        """What this report did and did not examine."""

        return f"{self.n_ops_audited} of {self.n_ops_total} ops audited"

    def _report_findings(self) -> list[str]:
        """One-line findings for the card, most severe first."""

        return [f"[{item.severity}] {item.check}: {item.message}" for item in self.findings]

    def _report_rows(self) -> list[dict[str, Any]] | None:
        """The natural table rows behind ``to_pandas()``."""

        return rows_of(self.findings)

    @property
    def coverage(self) -> float:
        """Return the fraction of trace operations actually audited.

        Returns
        -------
        float
            ``n_ops_audited / n_ops_total``, or ``0.0`` for an empty trace.
        """

        if self.n_ops_total == 0:
            return 0.0
        return self.n_ops_audited / self.n_ops_total

    def __repr__(self) -> str:
        """Return a compact coverage-honest summary.

        Returns
        -------
        str
            Finding count and audited/total operation coverage.
        """

        return (
            f"DTypeRangeAudit(findings={len(self.findings)}, "
            f"coverage={self.n_ops_audited}/{self.n_ops_total})"
        )


def _finding(
    op: Any,
    *,
    severity: AuditSeverity,
    check: str,
    message: str,
    follow_up: str = "tl.debug.dtype_range_audit(trace)",
) -> AuditFinding:
    """Build one dtype finding with recorded operation provenance.

    Parameters
    ----------
    op:
        Offending operation.
    severity:
        Audit severity label.
    check:
        Diagnostic check name.
    message:
        Evidence-backed measured-statistic summary.
    follow_up:
        Suggested inspection call.

    Returns
    -------
    AuditFinding
        Structured finding.
    """

    module = _module_address(op)
    return AuditFinding(
        severity=severity,
        check=check,
        message=message,
        ops=(_op_label(op),),
        modules=(module,) if module is not None else (),
        follow_up=follow_up,
    )


def _finite_max_abs(tensor: torch.Tensor) -> float | None:
    """Return the largest finite magnitude in a tensor.

    Parameters
    ----------
    tensor:
        Numeric tensor to inspect.

    Returns
    -------
    float | None
        Maximum absolute finite value, or ``None`` when no finite value exists.
    """

    finite = tensor.detach()[torch.isfinite(tensor.detach())]
    if finite.numel() == 0:
        return None
    return float(finite.to(torch.complex128 if finite.is_complex() else torch.float64).abs().max())


def _dtype_limit(dtype: torch.dtype) -> float | None:
    """Return the positive finite limit for a native numeric dtype.

    Parameters
    ----------
    dtype:
        Native torch dtype.

    Returns
    -------
    float | None
        Floating or integer maximum, otherwise ``None``.
    """

    try:
        if dtype.is_floating_point or dtype.is_complex:
            return float(torch.finfo(dtype).max)
        return float(torch.iinfo(dtype).max)
    except TypeError:
        return None


def _subnormal_fraction(tensor: torch.Tensor) -> float | None:
    """Return the fraction of elements in the dtype's subnormal interval.

    Parameters
    ----------
    tensor:
        Saved activation tensor.

    Returns
    -------
    float | None
        Fraction of all elements with ``0 < abs(x) < finfo.tiny``, or ``None``
        for non-floating/complex or empty tensors.
    """

    if tensor.numel() == 0 or not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
        return None
    tiny = torch.finfo(tensor.dtype).tiny
    magnitude = tensor.detach().abs()
    count = ((magnitude > 0) & (magnitude < tiny) & torch.isfinite(magnitude)).sum()
    return float(count.item() / tensor.numel())


def _dtype_precision(dtype: Any) -> tuple[str, int] | None:
    """Return a comparable numeric category and precision width.

    Parameters
    ----------
    dtype:
        Candidate native torch dtype.

    Returns
    -------
    tuple[str, int] | None
        Category and precision bits, or ``None`` when not comparable.
    """

    if not isinstance(dtype, torch.dtype):
        return None
    try:
        if dtype.is_complex:
            return "complex", int(torch.finfo(dtype).bits)
        if dtype.is_floating_point:
            return "float", int(torch.finfo(dtype).bits)
        return "integer", int(torch.iinfo(dtype).bits)
    except TypeError:
        return None


def _wider_input_dtypes(op: Any, output_dtype: torch.dtype) -> tuple[torch.dtype, ...]:
    """Return recorded input dtypes wider than the saved output dtype.

    Parameters
    ----------
    op:
        Operation carrying ``input_dtypes`` metadata.
    output_dtype:
        Saved output tensor dtype.

    Returns
    -------
    tuple[torch.dtype, ...]
        Distinct wider dtypes in recorded order.
    """

    output_precision = _dtype_precision(output_dtype)
    if output_precision is None:
        return ()
    wider: list[torch.dtype] = []
    try:
        input_dtypes = tuple(parent.dtype for parent in op.input_ops.values())
    except (AttributeError, KeyError, RuntimeError, ValueError):
        input_dtypes = tuple(getattr(op, "input_dtypes", ()) or ())
    for input_dtype in input_dtypes:
        input_precision = _dtype_precision(input_dtype)
        if (
            input_precision is not None
            and input_precision[0] == output_precision[0]
            and input_precision[1] > output_precision[1]
            and input_dtype not in wider
        ):
            wider.append(input_dtype)
    return tuple(wider)


def dtype_range_audit(
    trace: Trace,
    *,
    max_fraction: float = 0.9,
    subnormal_fraction_threshold: float = 0.1,
) -> DTypeRangeAudit:
    """Audit saved activations for numeric-range and precision hazards.

    Non-finite classification reuses the same classifier as
    :func:`torchlens.report.find_nan`. Range proximity is measured against the
    saved output dtype's finite maximum. Downcast findings require recorded
    input dtypes that are wider within the same numeric category. Coverage is
    always reported because unsaved and non-tensor payloads are not inspected.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    max_fraction:
        Fraction of the dtype maximum at or above which a finite magnitude is
        flagged. Defaults to ``0.9``.
    subnormal_fraction_threshold:
        Minimum fraction of all tensor elements in the subnormal interval to
        flag. Defaults to ``0.1``.

    Returns
    -------
    DTypeRangeAudit
        Structured findings and audited/total operation coverage.

    Raises
    ------
    ValueError
        If either threshold is outside ``[0, 1]``.
    """

    if not 0.0 <= max_fraction <= 1.0:
        raise ValueError("max_fraction must be in [0, 1].")
    if not 0.0 <= subnormal_fraction_threshold <= 1.0:
        raise ValueError("subnormal_fraction_threshold must be in [0, 1].")

    ops = [op for op in _compute_ops(trace) if not bool(getattr(op, "is_output", False))]
    findings: list[AuditFinding] = []
    n_ops_audited = 0
    for op in ops:
        output, _reason = _safe_out(op)
        if not isinstance(output, torch.Tensor):
            continue
        n_ops_audited += 1
        label = _op_label(op)
        dtype = output.dtype
        nonfinite_kind = _nonfinite_kind(output)
        if nonfinite_kind != "none":
            findings.append(
                _finding(
                    op,
                    severity="critical",
                    check="dtype_nonfinite",
                    message=(
                        f"{label} dtype={dtype} contains measured non-finite kind={nonfinite_kind}."
                    ),
                    follow_up="trace.find_nan()",
                )
            )

        max_abs = _finite_max_abs(output)
        dtype_limit = _dtype_limit(dtype)
        if (
            max_abs is not None
            and dtype_limit is not None
            and max_abs >= max_fraction * dtype_limit
        ):
            findings.append(
                _finding(
                    op,
                    severity="warning",
                    check="dtype_near_max",
                    message=(
                        f"{label} dtype={dtype} measured max_abs={max_abs:.6g}, "
                        f"which is {max_abs / dtype_limit:.6g} of dtype_max={dtype_limit:.6g}."
                    ),
                )
            )

        subnormal_fraction = _subnormal_fraction(output)
        if (
            subnormal_fraction is not None
            and subnormal_fraction >= subnormal_fraction_threshold
            and subnormal_fraction > 0.0
        ):
            findings.append(
                _finding(
                    op,
                    severity="warning",
                    check="dtype_subnormal",
                    message=(
                        f"{label} dtype={dtype} measured subnormal_fraction="
                        f"{subnormal_fraction:.6g}."
                    ),
                )
            )

        wider_inputs = _wider_input_dtypes(op, dtype)
        if wider_inputs:
            input_names = ", ".join(str(input_dtype) for input_dtype in wider_inputs)
            findings.append(
                _finding(
                    op,
                    severity="warning",
                    check="dtype_downcast",
                    message=(
                        f"{label} recorded wider input dtype(s) {input_names} and saved "
                        f"output dtype={dtype}."
                    ),
                )
            )

    return DTypeRangeAudit(tuple(findings), n_ops_audited, len(ops))


__all__ = ["DTypeRangeAudit", "dtype_range_audit"]

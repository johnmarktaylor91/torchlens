"""The preprocessing verifier: field-level configuration audit (tvscope B3).

The verdict ORACLE is the configuration comparison against the user's own
authority (tvscope memo D3): tensor heuristics were measured to
false-positive on five of seven legitimate pipelines while missing both
wrong-preprocessing controls, so they live in the opt-in diagnostics module
and may NEVER return a match. This module compares DECLARED fields only.

Verdict law (memo D4): opaque callables and partial metadata never become
``match``; the TorchLens-authored ImageNet fallback never yields
``verified``; default is non-raising, and the explicit strict mode refuses
``mismatch`` AND ``unknown`` with DISTINCT typed codes. The audit never
returns a transform, never auto-fixes, and never consults a constants table
(memo D7).

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from torchlens._errors import _actionable_message, _ActionableErrorMixin
from torchlens.errors._base import ConfigurationError

from ._records import (
    COMPARABLE_FIELDS,
    STATUS_AUTHORITATIVE,
    DeclaredPreprocessing,
    Resolution,
    status_of,
)

__tl_layer__ = "L5"

#: Closed per-field verdicts.
FIELD_MATCH = "match"
FIELD_MISMATCH = "mismatch"
FIELD_UNKNOWN = "unknown"

#: Closed audit verdicts. ``verified`` requires every field matched AND an
#: authoritative authority; anything less is ``unknown`` or ``mismatch``.
VERDICT_VERIFIED = "verified"
VERDICT_MISMATCH = "mismatch"
VERDICT_UNKNOWN = "unknown"

#: Closed evidence-class ranking (memo D5): a property of OUR evidence.
#: ``contradiction`` is reserved for tensor diagnostics; a config-vs-config
#: disagreement is ``mismatch``; conventions-premised findings are
#: ``premised``; disclosures are ``info``.
EVIDENCE_CONTRADICTION = "contradiction"
EVIDENCE_MISMATCH = "mismatch"
EVIDENCE_PREMISED = "premised"
EVIDENCE_INFO = "info"

#: Relative tolerance for float field comparison (mean/std/value_range).
_FLOAT_RTOL = 1e-6

#: Consequence prose per comparable field (memo D5/D18): family-split, names
#: the stimulus set behind every number, never alarmist, never dismissive.
#: Measured on the seeded 128-image COCO test2017 sample ("diverse natural
#: photographs"; severity is a property of stimulus STRUCTURE).
_CONSEQUENCES: dict[str, str] = {
    "mean": (
        "Wrong or missing normalization constants materially perturb "
        "second-order analyses when normalization is skipped outright (CNN "
        "RDM Spearman ~0.70 on the 128-image diverse-photograph COCO set) "
        "and are nearly harmless at second order for wrong-FAMILY constants "
        "on an ImageNet CNN (RDM Pearson ~0.99 on the same set), while "
        "first-order analyses (decoding, retrieval, unit-level) degrade in "
        "every wrong condition (top-1 agreement ~0.52-0.90)."
    ),
    "std": (
        "Stds can differ while means agree; a wrong std rescales every "
        "channel and degrades first-order analyses even when second-order "
        "structure survives (CLIP ImageNet-constants NN retrieval agreement "
        "~0.55 on the 128-image diverse-photograph COCO set)."
    ),
    "resize_size": (
        "Wrong geometry materially perturbs RDMs (CNN RDM Spearman ~0.69 at "
        "128x128 vs the correct pipeline on the 128-image diverse-photograph "
        "COCO set) and first-order outputs (top-1 agreement ~0.39)."
    ),
    "crop_size": (
        "A wrong crop changes the model's effective field of view; measured "
        "geometry errors moved CNN RDM Spearman to ~0.69 on the 128-image "
        "diverse-photograph COCO set."
    ),
    "interpolation": (
        "Interpolation mode shifts are small per-pixel perturbations; they "
        "matter most for first-order, unit-level reads."
    ),
    "antialias": (
        "Antialias mismatches are small per-pixel perturbations; they matter "
        "most for first-order, unit-level reads."
    ),
    "channel_order": (
        "A channel-order swap (RGB/BGR) feeds every channel the wrong "
        "statistics; treat any downstream comparison as suspect."
    ),
    "value_range": (
        "A value-range mismatch (e.g. [0,255] into a [0,1]-scaled model) "
        "saturates activations; both first- and second-order analyses are "
        "affected."
    ),
}


class PreprocessingAuditError(_ActionableErrorMixin, ConfigurationError, RuntimeError):
    """Strict-mode refusal from the preprocessing audit (tvscope B3).

    Distinct stable codes separate the two strict failures: a measured
    disagreement (``preprocessing_audit_mismatch``) and inability to compare
    (``preprocessing_audit_unknown``). Consumers branch on
    ``exc.fields['code']``, never message text.
    """

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize a typed audit refusal.

        Parameters
        ----------
        problem:
            What disagreed or could not be compared, naming the fields.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured diagnostic context (field lists, verdicts).
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


@dataclass(frozen=True)
class AuditFinding:
    """One comparable field's audit row.

    Attributes
    ----------
    field:
        Comparable field name (closed vocabulary, memo B3).
    verdict:
        ``"match"`` / ``"mismatch"`` / ``"unknown"``.
    authority_value:
        The authority's declared value (``None`` = undeclared).
    applied_value:
        The applied pipeline's declared value (``None`` = undeclared).
    unknown_reason:
        Why the field could not be compared, when ``verdict="unknown"``.
    evidence_class:
        Evidence-class ranking token (memo D5).
    consequence:
        Family-split consequence prose with its stimulus set named.
    """

    field: str
    verdict: str
    authority_value: Any
    applied_value: Any
    unknown_reason: str | None
    evidence_class: str
    consequence: str

    def to_json(self) -> dict[str, Any]:
        """Serialize this finding to a JSON-portable row."""

        return {
            "field": self.field,
            "verdict": self.verdict,
            "authority_value": _jsonable(self.authority_value),
            "applied_value": _jsonable(self.applied_value),
            "unknown_reason": self.unknown_reason,
            "evidence_class": self.evidence_class,
            "consequence": self.consequence,
        }


@dataclass(frozen=True)
class PreprocessingAudit:
    """The field-level audit report (tvscope memo D4, record two of three).

    Attributes
    ----------
    authority:
        The resolution used as the reference side.
    applied_source:
        How the applied side was obtained: ``"resolution"`` /
        ``"declaration"`` / ``"parsed_transform"`` / ``"undeclared"`` /
        ``"opaque_transform"``.
    findings:
        One row per comparable field, in the closed field order.
    verdict:
        ``"verified"`` / ``"mismatch"`` / ``"unknown"``.
    unknown_reasons:
        Deduplicated reasons behind every unknown field, in field order.
    authority_status:
        The authority's standing (``authoritative`` / ``unverified_fallback``
        / ``unknown``); a non-authoritative reference caps the verdict below
        ``verified`` (memo D9).
    """

    authority: Resolution
    applied_source: str
    findings: tuple[AuditFinding, ...]
    verdict: str
    unknown_reasons: tuple[str, ...]
    authority_status: str

    @property
    def mismatched_fields(self) -> tuple[str, ...]:
        """Names of every mismatched field, in field order."""

        return tuple(f.field for f in self.findings if f.verdict == FIELD_MISMATCH)

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        """Names of every uncomparable field, in field order."""

        return tuple(f.field for f in self.findings if f.verdict == FIELD_UNKNOWN)

    def to_json(self) -> dict[str, Any]:
        """Serialize the full report (the manifest-block payload shape)."""

        record = self.authority.record
        return {
            "schema": "tl_preprocessing_audit_v1",
            "authority": {
                "source": record.source,
                "identifier": record.identifier,
                "verified": bool(record.verified),
                "status": self.authority_status,
                "config": _jsonable(record.config),
                "description": record.description,
            },
            "applied_source": self.applied_source,
            "verdict": self.verdict,
            "unknown_reasons": list(self.unknown_reasons),
            "findings": [finding.to_json() for finding in self.findings],
        }


def _jsonable(value: Any) -> Any:
    """Best-effort JSON coercion for record payloads (tuples -> lists)."""

    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _values_match(left: Any, right: Any) -> bool:
    """Compare two declared field values with float tolerance.

    Parameters
    ----------
    left, right:
        Normalized declared values (scalars or tuples).

    Returns
    -------
    bool
        True when the values agree (ints vs floats compare numerically;
        sequences compare elementwise; strings case-insensitively).
    """

    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _values_match(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        scale = max(abs(float(left)), abs(float(right)), 1e-12)
        return abs(float(left) - float(right)) <= _FLOAT_RTOL * scale
    if isinstance(left, str) and isinstance(right, str):
        return left.lower() == right.lower()
    return bool(left == right)


def _coerce_applied(applied: Any) -> tuple[DeclaredPreprocessing | None, str]:
    """Normalize the applied-side argument into declared fields.

    Parameters
    ----------
    applied:
        ``Resolution`` | ``DeclaredPreprocessing`` | declaration mapping |
        transform pipeline (Compose-shaped) | opaque callable | ``None``.

    Returns
    -------
    tuple[DeclaredPreprocessing | None, str]
        Declared fields (``None`` = nothing declared) and the applied-source
        token for the report.
    """

    from ._authorities import _adapt_explicit_mapping, _probe_compose, declared_from_compose

    if applied is None:
        return None, "undeclared"
    if isinstance(applied, Resolution):
        return applied.declared, "resolution"
    if isinstance(applied, DeclaredPreprocessing):
        return applied, "declaration"
    if isinstance(applied, Mapping):
        return _adapt_explicit_mapping(applied).declared, "declaration"
    if _probe_compose(applied):
        declared, _unparsed = declared_from_compose(applied)
        return declared, "parsed_transform"
    # An opaque callable declares nothing; it may NEVER become match.
    return None, "opaque_transform" if callable(applied) else "undeclared"


def _field_finding(
    name: str,
    authority_value: Any,
    applied_value: Any,
    applied_source: str,
) -> AuditFinding:
    """Build one field's audit row.

    Parameters
    ----------
    name:
        Comparable field name.
    authority_value:
        Authority-side declared value (``None`` = undeclared).
    applied_value:
        Applied-side declared value (``None`` = undeclared).
    applied_source:
        Applied-source token (for unknown reasons).

    Returns
    -------
    AuditFinding
        The audit row (match / mismatch / unknown-with-reason).
    """

    consequence = _CONSEQUENCES[name]
    if authority_value is None or applied_value is None:
        if authority_value is None and applied_value is None:
            reason = "undeclared_on_both_sides"
        elif authority_value is None:
            reason = "authority_undeclared"
        elif applied_source == "opaque_transform":
            reason = "opaque_transform"
        else:
            reason = "applied_undeclared"
        return AuditFinding(
            field=name,
            verdict=FIELD_UNKNOWN,
            authority_value=authority_value,
            applied_value=applied_value,
            unknown_reason=reason,
            evidence_class=EVIDENCE_INFO,
            consequence=consequence,
        )
    if _values_match(authority_value, applied_value):
        return AuditFinding(
            field=name,
            verdict=FIELD_MATCH,
            authority_value=authority_value,
            applied_value=applied_value,
            unknown_reason=None,
            evidence_class=EVIDENCE_INFO,
            consequence=consequence,
        )
    return AuditFinding(
        field=name,
        verdict=FIELD_MISMATCH,
        authority_value=authority_value,
        applied_value=applied_value,
        unknown_reason=None,
        evidence_class=EVIDENCE_MISMATCH,
        consequence=consequence,
    )


def _overall_verdict(findings: tuple[AuditFinding, ...], authority_status: str) -> str:
    """Fold field rows into the closed overall verdict.

    ``verified`` requires EVERY field matched and an AUTHORITATIVE authority:
    the demoted ImageNet fallback (and an unknown authority) can never anchor
    a ``verified`` verdict (memo D9), and partial metadata never becomes a
    match (memo D4).

    Parameters
    ----------
    findings:
        Per-field rows.
    authority_status:
        The authority's standing token.

    Returns
    -------
    str
        ``"verified"`` / ``"mismatch"`` / ``"unknown"``.
    """

    if any(f.verdict == FIELD_MISMATCH for f in findings):
        return VERDICT_MISMATCH
    if any(f.verdict == FIELD_UNKNOWN for f in findings):
        return VERDICT_UNKNOWN
    if authority_status != STATUS_AUTHORITATIVE:
        return VERDICT_UNKNOWN
    return VERDICT_VERIFIED


def audit(
    authority: Any,
    applied: Any = None,
    *,
    strict: bool = False,
) -> PreprocessingAudit:
    """Audit an applied preprocessing pipeline against an authority (B3).

    The comparison is field-by-field over the closed comparable vocabulary
    (resize, crop, interpolation, antialias, channel order, value range,
    mean AND std). It never returns a transform, never auto-fixes, and never
    consults a constants table; mismatch messages print measured values
    against the authority's values.

    Parameters
    ----------
    authority:
        The reference side: a :class:`Resolution`, a raw authority object
        (resolved through :func:`torchlens.preprocessing.resolve`), or a
        ``ResolvedPreprocessing`` record is NOT accepted directly -- resolve
        it first so declared fields are available.
    applied:
        What actually ran: a :class:`Resolution`, declared fields, a
        declaration mapping, a Compose-shaped pipeline (parsed), an opaque
        callable (every field ``unknown``, never match), or ``None``
        (nothing declared).
    strict:
        When True, refuse ``mismatch`` and ``unknown`` verdicts with
        DISTINCT typed codes (``preprocessing_audit_mismatch`` /
        ``preprocessing_audit_unknown``). Default is non-raising (memo D6:
        no automatic warnings in v1; the report is the product).

    Returns
    -------
    PreprocessingAudit
        The field-level report (JSON-serializable via ``to_json()``).

    Raises
    ------
    PreprocessingAuditError
        Strict mode only: on any mismatched field, or on an
        uncomparable/unauthoritative audit.
    """

    from ._authorities import resolve as _resolve

    resolution = authority if isinstance(authority, Resolution) else _resolve(authority)
    authority_status = status_of(resolution.record)
    authority_fields = (
        resolution.declared.comparable_fields()
        if resolution.declared is not None
        else dict.fromkeys(COMPARABLE_FIELDS)
    )
    applied_declared, applied_source = _coerce_applied(applied)
    applied_fields = (
        applied_declared.comparable_fields()
        if applied_declared is not None
        else dict.fromkeys(COMPARABLE_FIELDS)
    )
    findings = tuple(
        _field_finding(name, authority_fields[name], applied_fields[name], applied_source)
        for name in COMPARABLE_FIELDS
    )
    verdict = _overall_verdict(findings, authority_status)
    unknown_reasons: list[str] = []
    for finding in findings:
        if finding.unknown_reason and finding.unknown_reason not in unknown_reasons:
            unknown_reasons.append(finding.unknown_reason)
    if verdict == VERDICT_UNKNOWN and authority_status != STATUS_AUTHORITATIVE:
        reason = f"authority_status_{authority_status}"
        if reason not in unknown_reasons:
            unknown_reasons.append(reason)
    report = PreprocessingAudit(
        authority=resolution,
        applied_source=applied_source,
        findings=findings,
        verdict=verdict,
        unknown_reasons=tuple(unknown_reasons),
        authority_status=authority_status,
    )
    if strict:
        _strict_refuse(report)
    return report


def _strict_refuse(report: PreprocessingAudit) -> None:
    """Raise the strict-mode refusal for non-verified reports.

    Parameters
    ----------
    report:
        The completed audit report.

    Raises
    ------
    PreprocessingAuditError
        ``preprocessing_audit_mismatch`` on measured disagreement;
        ``preprocessing_audit_unknown`` when the audit could not verify.
    """

    if report.verdict == VERDICT_MISMATCH:
        rows = [
            f"{f.field}: authority={f.authority_value!r} applied={f.applied_value!r}"
            for f in report.findings
            if f.verdict == FIELD_MISMATCH
        ]
        raise PreprocessingAuditError(
            "strict preprocessing audit found mismatched fields -- " + "; ".join(rows),
            code="preprocessing_audit_mismatch",
            remedy=(
                "apply the authority's own transform (resolution.transform) or "
                "correct the mismatched fields; rerun with strict=False to get "
                "the full report without raising"
            ),
            mismatched_fields=list(report.mismatched_fields),
            verdict=report.verdict,
        )
    if report.verdict == VERDICT_UNKNOWN:
        raise PreprocessingAuditError(
            "strict preprocessing audit could not verify: "
            f"unknown fields {list(report.unknown_fields)!r}, authority status "
            f"{report.authority_status!r}, reasons {list(report.unknown_reasons)!r}.",
            code="preprocessing_audit_unknown",
            remedy=(
                "supply a declared authority (loader preset, processor, data "
                "config, or explicit mapping) AND a declared applied side; an "
                "opaque transform or a declares-nothing family cannot be "
                "verified -- rerun with strict=False to proceed on the honest "
                "unknown"
            ),
            unknown_fields=list(report.unknown_fields),
            unknown_reasons=list(report.unknown_reasons),
            authority_status=report.authority_status,
            verdict=report.verdict,
        )

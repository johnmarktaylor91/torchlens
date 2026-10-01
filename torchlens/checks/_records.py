"""Immutable finding / report records for the checks kit (checks memo 4.5).

Every record is a frozen dataclass with a versioned ``to_dict()`` so the
public JSON export (``schema_version=1``) exists from day 1 while the report
itself stays declared NON-PERSISTENT in bundles for v1 (checks memo D19):
the schema version and the immutable record shape are the forward-
compatibility hooks for a later bundle sidecar, never a migration.

Evidence vocabulary law (checks memo 4.6): every gradient/parameter scalar
that crosses any surface carries ``stage`` + ``scale_provenance``; unknown
provenance yields UNKNOWN verdicts, never a silent pass-as-1.0. Spellings
are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ._errors import CheckConfigError

__tl_layer__ = "L5"

#: Version stamped into every exported report/audit dict (memo D19).
CHECK_REPORT_SCHEMA_VERSION = 1

#: Severity trio -- deliberately the shipped ``AuditSeverity`` vocabulary
#: (torchlens.debug), gaining no sibling (memo D16).
SEVERITIES = ("critical", "warning", "info")

#: Orthogonal control-flow axis (memo 4.5); overridable per registration.
ACTIONS = ("raise", "warn", "collect")

#: Loop-phase stage of a gradient scalar (memo 4.6). S-A values are
#: pre-clip (scaled under a GradScaler); S-B/S-C values are post-clip and
#: already unscaled.
STAGES = ("pre_clip_scaled", "pre_clip_unscaled", "post_clip_applied")

#: Where the scale knowledge came from (memo exec 6). ``unknown`` never
#: silently becomes 1.0.
SCALE_PROVENANCE = ("unscaled", "gradscaler", "explicit", "unknown")

#: How the evidence was obtained. ``digest`` may never emit an exact pass
#: (memo D5); ``multi_grad_all`` marks the priced fallback mechanism (DR-1).
EVIDENCE_KINDS = (
    "clone",
    "digest",
    "counting_hook",
    "magnitude_pass",
    "multi_grad_all",
    "step_hook",
    "scan_kernel",
    "capture_hook",
)

#: Step-axis provenance for check findings (memo D10/D18): hook-only mode
#: labels attempt GROUPING ``inferred``; only the explicit boundary is
#: ``explicit`` and only it carries caller ``global_step``.
CHECK_STEP_PROVENANCE = ("explicit", "inferred")

#: Nonfinite attribution (memo D6): ``smeared`` is a field, not prose --
#: the NaN-smear case never names an innocent parameter.
ATTRIBUTIONS = ("exact", "smeared")


def _require_token(field_name: str, value: str | None, vocabulary: tuple[str, ...]) -> None:
    """Validate one closed-vocabulary token, refusing typed on violation."""

    if value is not None and value not in vocabulary:
        raise CheckConfigError(
            f"{field_name}={value!r} is not in the closed vocabulary {vocabulary}. "
            "Check records never carry invented tokens: report consumers and "
            "the trackers seam branch on these values.",
            code="check_vocab_invalid",
            field=field_name,
            value=value,
            vocabulary=vocabulary,
            remedy=f"Use one of {vocabulary}.",
        )


@dataclass(frozen=True)
class CheckFinding:
    """One immutable check finding (memo 4.5).

    ``values`` carries the measured numbers (already-reduced floats only);
    ``names`` the offending parameter/site names with every alias of a tied
    tensor preserved; ``follow_up`` the call that localizes the culprit
    (memo D3: the finding names a cone, the frontier names the culprit).
    """

    check: str
    code: str
    severity: str
    action: str
    message: str
    names: tuple[str, ...] = ()
    global_step: int | None = None
    backward_id: int | None = None
    attempted_step_id: int | None = None
    accepted_step_id: int | None = None
    step_provenance: str = "inferred"
    stage: str | None = None
    scale_provenance: str | None = None
    evidence: str | None = None
    coverage: str | None = None
    attribution: str | None = None
    collateral_zeroed: int | None = None
    zero_baseline: bool | None = None
    values: dict[str, float | None] = field(default_factory=dict)
    remedy: str = ""
    follow_up: str = ""

    def __post_init__(self) -> None:
        """Validate closed-vocabulary fields at construction."""

        _require_token("severity", self.severity, SEVERITIES)
        _require_token("action", self.action, ACTIONS)
        _require_token("stage", self.stage, STAGES)
        _require_token("scale_provenance", self.scale_provenance, SCALE_PROVENANCE)
        _require_token("evidence", self.evidence, EVIDENCE_KINDS)
        _require_token("step_provenance", self.step_provenance, CHECK_STEP_PROVENANCE)
        _require_token("attribution", self.attribution, ATTRIBUTIONS)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable payload for this finding."""

        payload = asdict(self)
        payload["names"] = list(self.names)
        return payload


@dataclass(frozen=True)
class CheckReport:
    """Deterministically finalized check report (memo 4.5 / D19).

    Finalizes on demand and on structured scope exit INCLUDING exceptional
    exit; kill -9 loses in-memory state under any design, so no atexit
    device work is owed. ``unavailable`` entries are mandatory vocabulary:
    a check that cannot run reports ``(check, reason)`` here, never silence.
    """

    findings: tuple[CheckFinding, ...] = ()
    checks_run: tuple[str, ...] = ()
    unavailable: tuple[tuple[str, str], ...] = ()
    disclosures: dict[str, Any] = field(default_factory=dict)
    ledgers: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    finalized_reason: str = "on_demand"
    schema_version: int = CHECK_REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-serializable report payload."""

        return {
            "schema_version": self.schema_version,
            "findings": [finding.to_dict() for finding in self.findings],
            "checks_run": list(self.checks_run),
            "unavailable": [list(entry) for entry in self.unavailable],
            "disclosures": self.disclosures,
            "ledgers": self.ledgers,
            "coverage": self.coverage,
            "counters": self.counters,
            "finalized_reason": self.finalized_reason,
        }

    def to_json(self, *, indent: int | None = 1) -> str:
        """Serialize :meth:`to_dict` as JSON text."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def __repr__(self) -> str:
        """Render a compact house-style summary (matches ``TraceAudit``)."""

        by_severity = dict.fromkeys(SEVERITIES, 0)
        for finding in self.findings:
            by_severity[finding.severity] += 1
        heading = (
            f"CheckReport: {len(self.findings)} finding(s) "
            f"({by_severity['critical']} critical, {by_severity['warning']} warning, "
            f"{by_severity['info']} info); {len(self.checks_run)} checks run, "
            f"{len(self.unavailable)} unavailable"
        )
        lines = [heading]
        lines.extend(
            f"- [{finding.severity}] {finding.check}: {finding.message}"
            for finding in self.findings[:20]
        )
        if len(self.findings) > 20:
            lines.append(f"- ... {len(self.findings) - 20} more finding(s)")
        lines.extend(f"- unavailable {check}: {reason}" for check, reason in self.unavailable)
        return "\n".join(lines)


def severity_sorted(findings: list[CheckFinding]) -> tuple[CheckFinding, ...]:
    """Return findings ordered critical -> warning -> info, stably."""

    order = {severity: rank for rank, severity in enumerate(SEVERITIES)}
    return tuple(sorted(findings, key=lambda finding: order[finding.severity]))


__all__ = [
    "ACTIONS",
    "ATTRIBUTIONS",
    "CHECK_REPORT_SCHEMA_VERSION",
    "CHECK_STEP_PROVENANCE",
    "EVIDENCE_KINDS",
    "SCALE_PROVENANCE",
    "SEVERITIES",
    "STAGES",
    "CheckFinding",
    "CheckReport",
    "severity_sorted",
]

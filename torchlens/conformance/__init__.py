"""``torchlens.conformance``: the capture conformance suite (MEMO 4, B4).

Ships in the base wheel; real framework/checkpoint packs stay behind extras.
The claim grammar is closed (MEMO 4.2):

- C0 provider API -> "TorchLens provider API compatible, C0" (never
  "capture-conformant").
- C1 real capture -> "TorchLens-conformant capture, C1" + profiles + scope.
- C2 durable artifact -> "TorchLens-conformant durable capture, C2".

TOYS EARN NOTHING: config-built / hermetic models are the smoke pack and
mutation substrate only, and the RUNNER enforces that no string containing
"conformant" is ever emitted for them -- not even suffixed; a suffix is a
detail a tweet drops. A C1/C2 claim requires at least TWO pinned pretrained
model families with checkpoint evidence. Four anti-vacuity mechanisms ship:
planted defects (the reserved ``"fake"`` provider,
``torchlens.conformance._adapters.FAKE_PLANTS``), executed-count floors,
named skips (a skip fails the requested cell), and canonical report bytes
attested with SHA-256. "Verified by TorchLens" is reserved for
TorchLens-controlled CI; third parties may claim "self-tested" only with
the unmodified report.

Every spelling is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import hashlib
import json
import platform as _platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigurationError
from ._adapters import FAKE_PLANTS, RosterModel, default_roster, resolve_adapter
from ._cases import CaseResult, run_c0_cases, run_c1_cases, run_c2_cases

__tl_layer__ = "L9"

__all__ = [
    "FAKE_PLANTS",
    "CaseResult",
    "ConformanceReport",
    "RosterModel",
    "claim_string",
    "default_roster",
    "run_conformance",
]

SUITE_VERSION = "torchlens.conformance.v1"

#: Minimum pretrained model FAMILIES a C1/C2 claim requires (MEMO 4.2).
CLAIM_MIN_PRETRAINED_FAMILIES = 2

#: Default executed-count floor: a cell that executed fewer real checks than
#: this fails outright (zero executed tests must never read as green).
DEFAULT_EXECUTED_FLOOR = 3


@dataclass(frozen=True)
class ConformanceReport:
    """Canonical, cwd-independent conformance report (MEMO 4.2).

    Parameters
    ----------
    suite_version:
        The suite/schema identifier.
    backend:
        Provider name under test.
    tiers:
        Tiers requested, in rung order.
    results:
        Every case row (the COMPLETE result funnel; skips included).
    funnel:
        Counts: requested/executed/passed/failed/skipped.
    realism:
        Families executed per realism class.
    earned_tier:
        Highest tier every executed case passed, or ``""`` when none.
    claim_eligible:
        ``True`` only when the pretrained-family floor is met; toys earn
        nothing regardless of pass rate.
    environment:
        Runtime receipt (torchlens/torch/python/platform).
    attestation_sha256:
        SHA-256 over the canonical report bytes (this field excluded).
    """

    suite_version: str
    backend: str
    tiers: tuple[str, ...]
    results: tuple[CaseResult, ...]
    funnel: dict[str, int]
    realism: dict[str, list[str]]
    earned_tier: str
    claim_eligible: bool
    environment: dict[str, str]
    attestation_sha256: str

    def to_canonical_json(self) -> str:
        """Render the attested canonical byte form (sorted keys, no cwd)."""

        return _canonical_report_json(self, include_attestation=True)


def _canonical_report_json(report: ConformanceReport, *, include_attestation: bool) -> str:
    """Serialize one report canonically; the digest hashes the digest-free form."""

    payload = {
        "suite_version": report.suite_version,
        "backend": report.backend,
        "tiers": list(report.tiers),
        "results": [
            {
                "case_id": row.case_id,
                "tier": row.tier,
                "outcome": row.outcome,
                "family": row.family,
                "realism": row.realism,
                "detail": row.detail,
            }
            for row in report.results
        ],
        "funnel": dict(sorted(report.funnel.items())),
        "realism": {key: sorted(value) for key, value in sorted(report.realism.items())},
        "earned_tier": report.earned_tier,
        "claim_eligible": report.claim_eligible,
        "environment": dict(sorted(report.environment.items())),
    }
    if include_attestation:
        payload["attestation_sha256"] = report.attestation_sha256
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _environment_receipt() -> dict[str, str]:
    """Runtime receipt rows for the report."""

    from importlib import metadata as _metadata

    from .. import __version__

    return {
        "torchlens_version": __version__,
        # Installed-distribution version, not the torch dunder: this package
        # carries no torch-privates license row (arch spine item 12).
        "torch_version": _metadata.version("torch"),
        "python_version": _platform.python_version(),
        "platform": _platform.platform(),
    }


def _adjudicate(
    results: list[CaseResult], tiers: tuple[str, ...], executed_floor: int
) -> tuple[str, dict[str, int]]:
    """Fold the result rows into the funnel and the earned tier."""

    funnel = {
        "requested": len(results),
        "executed": sum(1 for row in results if row.outcome != "skipped"),
        "passed": sum(1 for row in results if row.outcome == "passed"),
        "failed": sum(1 for row in results if row.outcome == "failed"),
        "skipped": sum(1 for row in results if row.outcome == "skipped"),
    }
    if funnel["executed"] < executed_floor:
        return "", funnel
    earned = ""
    for tier in ("C0", "C1", "C2"):
        if tier not in tiers:
            break
        tier_rows = [row for row in results if row.tier == tier]
        if not tier_rows or any(row.outcome != "passed" for row in tier_rows):
            break
        earned = tier
    return earned, funnel


def run_conformance(
    backend: str,
    *,
    tiers: tuple[str, ...] = ("C0",),
    roster: tuple[RosterModel, ...] | None = None,
    workdir: str | Path | None = None,
    executed_floor: int = DEFAULT_EXECUTED_FLOOR,
    _plant: str | None = None,
) -> ConformanceReport:
    """Run the conformance packs for one provider and attest the report.

    Parameters
    ----------
    backend:
        Provider name (``"torch"`` reference or the reserved ``"fake"``).
    tiers:
        Rungs to run, cumulative from ``"C0"`` (``("C0","C1","C2")`` runs
        all three; a higher rung without its predecessors refuses typed).
    roster:
        Model rows for C1/C2; defaults to the zero-network config-built
        smoke roster (which can NEVER earn a claim).
    workdir:
        Scratch directory for C2 artifacts; a temp dir when omitted.
    executed_floor:
        Executed-count floor; fewer executed checks than this earns no
        tier (zero executed tests must never read as green).
    _plant:
        Internal: plant selector for the reserved fake provider.

    Returns
    -------
    ConformanceReport
        The attested canonical report with the complete result funnel.

    Raises
    ------
    ConfigurationError
        ``conformance_scope_invalid`` for an unknown or non-cumulative tier
        request; ``conformance_backend_unknown`` per the adapter resolver.
    """

    order = ("C0", "C1", "C2")
    if any(tier not in order for tier in tiers) or list(tiers) != list(order[: len(tiers)]):
        raise ConfigurationError(
            f"Conformance tiers must be cumulative from C0 in rung order; got "
            f"{list(tiers)}. Remedy: request ('C0',), ('C0','C1'), or "
            "('C0','C1','C2').",
            code="conformance_scope_invalid",
            remedy="request a cumulative tier prefix from C0",
            requested_tiers=list(tiers),
        )
    adapter = resolve_adapter(backend, plant=_plant)
    model_rows = default_roster() if roster is None else roster
    results: list[CaseResult] = []
    results.extend(run_c0_cases(adapter))
    if "C1" in tiers:
        results.extend(run_c1_cases(adapter, model_rows))
    if "C2" in tiers:
        if workdir is None:
            with tempfile.TemporaryDirectory(prefix="tl_conformance_") as tmp:
                results.extend(run_c2_cases(adapter, model_rows, workdir=Path(tmp)))
        else:
            results.extend(run_c2_cases(adapter, model_rows, workdir=Path(workdir)))
    earned, funnel = _adjudicate(results, tiers, executed_floor)
    realism: dict[str, list[str]] = {}
    for row in results:
        if row.realism:
            realism.setdefault(row.realism, [])
            if row.family not in realism[row.realism]:
                realism[row.realism].append(row.family)
    pretrained_families = len(realism.get("pretrained", []))
    claim_eligible = (
        earned != ""
        and (earned == "C0" or pretrained_families >= CLAIM_MIN_PRETRAINED_FAMILIES)
        and adapter.claim_eligible
    )
    report = ConformanceReport(
        suite_version=SUITE_VERSION,
        backend=backend,
        tiers=tiers,
        results=tuple(results),
        funnel=funnel,
        realism=realism,
        earned_tier=earned,
        claim_eligible=claim_eligible,
        environment=_environment_receipt(),
        attestation_sha256="",
    )
    digest = hashlib.sha256(
        _canonical_report_json(report, include_attestation=False).encode("utf-8")
    ).hexdigest()
    return ConformanceReport(
        suite_version=report.suite_version,
        backend=report.backend,
        tiers=report.tiers,
        results=report.results,
        funnel=report.funnel,
        realism=report.realism,
        earned_tier=report.earned_tier,
        claim_eligible=report.claim_eligible,
        environment=report.environment,
        attestation_sha256=digest,
    )


def claim_string(report: ConformanceReport) -> str:
    """Render the earned claim string, refusing anything unearned.

    The claim string is what third parties quote, so the RUNNER -- not
    prose -- enforces the grammar: C0 earns "provider API compatible" and
    never "capture-conformant"; C1/C2 claims require the pretrained-family
    floor; toy-only runs and the reserved fake provider earn NO string
    containing "conformant", not even suffixed.

    Parameters
    ----------
    report:
        The attested report to render.

    Returns
    -------
    str
        The exact claim string for the earned tier.

    Raises
    ------
    ConfigurationError
        ``conformance_claim_unearned`` when the report earned no claim
        (failed cells, executed-floor miss, toy-only evidence, or the fake
        provider).
    """

    if not report.claim_eligible or not report.earned_tier:
        pretrained = len(report.realism.get("pretrained", []))
        raise ConfigurationError(
            f"Report for backend {report.backend!r} earned no claim: earned_tier="
            f"{report.earned_tier!r}, pretrained families={pretrained} (floor "
            f"{CLAIM_MIN_PRETRAINED_FAMILIES} for C1+), claim_eligible="
            f"{report.claim_eligible}. Toy/hermetic models are the smoke pack "
            "and mutation substrate only; they emit no claim. Remedy: run the "
            "real pretrained packs (pinned revisions) and pass every cell.",
            code="conformance_claim_unearned",
            remedy="run the real pretrained packs and pass every cell",
            earned_tier=report.earned_tier,
        )
    scope = ",".join(sorted(report.realism.get("pretrained", []))) or "none"
    if report.earned_tier == "C0":
        return "TorchLens provider API compatible, C0"
    if report.earned_tier == "C1":
        return f"TorchLens-conformant capture, C1 (scope: {scope})"
    return f"TorchLens-conformant durable capture, C2 (scope: {scope})"

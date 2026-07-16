"""Bounded block-at-write routing for immutable metadata and fidelity gates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from menagerie.crawler.author_dispatch import (
    AuthorResult,
    BlockedRecommendation,
    DeferRecommendation,
    ProposedAuthorResult,
    SkipRecommendation,
)
from menagerie.crawler.constants import (
    GATE_SCHEMA_VERSION_V3,
    AccuracyVerdict,
    FidelityVerdict,
    GateKind,
    GateRoute,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import JsonObject
from menagerie.crawler.schema import PayloadValidationError, validate_payload


class GateRoutingError(ValueError):
    """Raised when an immutable gate cannot be routed safely."""


@dataclass(frozen=True)
class MetadataRouteDecision:
    """Per-item block-at-write decision.

    Parameters
    ----------
    stable_id:
        Independently gated model.
    verdict:
        Checker metadata verdict.
    route:
        Accept, next-batch repair, or terminal human failure.
    canonical_write_allowed:
        True only for a current accurate item.
    repair_count:
        Failure count including this verdict.
    required_repairs:
        Checker-provided repair instructions.
    human_review_required:
        Whether bounded repair is exhausted.
    """

    stable_id: str
    verdict: AccuracyVerdict
    route: GateRoute
    canonical_write_allowed: bool
    repair_count: int
    required_repairs: tuple[str, ...]
    human_review_required: bool


@dataclass(frozen=True)
class FidelityRouteDecision:
    """Per-model five-way fidelity route.

    Parameters
    ----------
    stable_id:
        Checked model.
    verdict:
        One frozen five-way fidelity verdict.
    route:
        Accept for match/minor drift; block otherwise.
    accepted_for_fidelity:
        Whether fidelity permits later driver consideration.
    canonical_write_allowed:
        True only when both fidelity and source-rung checks permit a canonical write.
    permanent_scar:
        True only for slop or a checker-preserved prior scar.
    failure_reason_code:
        Closed fidelity failure reason when blocked.
    """

    stable_id: str
    verdict: FidelityVerdict
    route: GateRoute
    accepted_for_fidelity: bool
    canonical_write_allowed: bool
    permanent_scar: bool
    failure_reason_code: str | None


@dataclass(frozen=True)
class TerminalDispositionDecision:
    """Exact terminal-gate result over one advisory author-result arm.

    Parameters
    ----------
    stable_id, work_id, campaign_id:
        Exact author-result identities checked by the gate.
    gate_id:
        Immutable v3 gate identity.
    predicate:
        Closed independently checked terminal predicate.
    accepted:
        Whether the checker accepted the recommendation. This is not a terminal
        award; the authority kernel must still derive a :class:`TerminalProof`.
    source_ids, evidence_ids:
        Exact resolved status-candidate references.
    findings:
        Checker findings retained for repair or audit.
    """

    stable_id: str
    work_id: str
    campaign_id: str
    gate_id: str
    predicate: str
    accepted: bool
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    findings: tuple[str, ...]


def route_metadata_gate(
    gate: Mapping[str, Any],
    repair_counts: Mapping[str, int],
    *,
    max_repairs: int,
) -> tuple[MetadataRouteDecision, ...]:
    """Apply bounded per-item metadata block-at-write decisions.

    A failed item is scheduled for a different next batch until the bound is
    exhausted. No decision grants execution status.

    Parameters
    ----------
    gate:
        Schema-valid ``metadata_batch`` gate record.
    repair_counts:
        Prior failed-gate counts keyed by stable ID.
    max_repairs:
        Maximum repair requeues before terminal human failure.

    Returns
    -------
    tuple[MetadataRouteDecision, ...]
        One independent route per gate item.

    Raises
    ------
    GateRoutingError
        If the gate kind, schema, or repair counts are invalid.
    """

    _validate_gate(gate, GateKind.METADATA_BATCH)
    if max_repairs < 0:
        raise GateRoutingError("max_repairs must be non-negative")
    decisions: list[MetadataRouteDecision] = []
    for item in _items(gate):
        stable_id = str(item["stable_id"])
        prior = repair_counts.get(stable_id, 0)
        if prior < 0:
            raise GateRoutingError(f"negative repair count for {stable_id}")
        verdict = AccuracyVerdict(str(item["verdict"]))
        rung_check = item.get("rung_check")
        if not isinstance(rung_check, Mapping):
            raise GateRoutingError("metadata gate item must require a rung check")
        rung_verdict = AccuracyVerdict(str(rung_check.get("verdict")))
        if rung_verdict is AccuracyVerdict.ACCURATE and (
            rung_check.get("highest_applicable") != rung_check.get("selected_rung")
        ):
            raise GateRoutingError(
                "accurate rung check requires highest_applicable == selected_rung"
            )
        repairs = tuple(str(value) for value in item.get("required_repairs", []))
        if verdict is AccuracyVerdict.ACCURATE and rung_verdict is AccuracyVerdict.ACCURATE:
            route = GateRoute.ACCEPT
            count = prior
            allowed = True
            human = False
        else:
            count = prior + 1
            allowed = False
            human = count > max_repairs
            route = GateRoute.HUMAN_FAIL if human else GateRoute.REQUEUE_NEXT_BATCH
        decisions.append(
            MetadataRouteDecision(
                stable_id=stable_id,
                verdict=verdict,
                route=route,
                canonical_write_allowed=allowed,
                repair_count=count,
                required_repairs=repairs,
                human_review_required=human,
            )
        )
    return tuple(decisions)


def next_metadata_batch_ids(
    decisions: Iterable[MetadataRouteDecision],
) -> tuple[str, ...]:
    """Return failed item IDs that must enter the next fresh batch.

    Parameters
    ----------
    decisions:
        Per-item metadata routes.

    Returns
    -------
    tuple[str, ...]
        Stable IDs requiring bounded repair and rebatching.
    """

    return tuple(
        decision.stable_id
        for decision in decisions
        if decision.route is GateRoute.REQUEUE_NEXT_BATCH
    )


def route_fidelity_gate(
    gate: Mapping[str, Any], proposal: Mapping[str, Any]
) -> FidelityRouteDecision:
    """Route the frozen five-way fidelity verdict without editing its proposal.

    Parameters
    ----------
    gate:
        Schema-valid one-item fidelity gate.
    proposal:
        Immutable staged proposal checked for accidental mutation.

    Returns
    -------
    FidelityRouteDecision
        Deterministic fidelity route; never an execution award.

    Raises
    ------
    GateRoutingError
        If the gate is invalid or routing mutates the proposal.
    """

    _validate_gate(gate, GateKind.FIDELITY)
    before = stable_hash(proposal)
    item = _items(gate)[0]
    fidelity = item.get("fidelity")
    if not isinstance(fidelity, Mapping) or fidelity.get("required") is not True:
        raise GateRoutingError("fidelity gate item must require fidelity")
    rung_check = item.get("rung_check")
    if not isinstance(rung_check, Mapping):
        raise GateRoutingError("fidelity gate item must require a rung check")
    verdict = FidelityVerdict(str(fidelity.get("verdict")))
    rung_verdict = AccuracyVerdict(str(rung_check.get("verdict")))
    fidelity_accepted = verdict in {FidelityVerdict.MATCH, FidelityVerdict.MINOR_DRIFT}
    accepted = fidelity_accepted and rung_verdict is AccuracyVerdict.ACCURATE
    reason_codes = {
        FidelityVerdict.MAJOR_DRIFT: "major-drift-cap-exhausted",
        FidelityVerdict.SLOP: "slop-cap-exhausted",
        FidelityVerdict.CANNOT_VERIFY: "cannot-verify-cap-exhausted",
    }
    failure_reason_code: str | None = None
    if not fidelity_accepted:
        failure_reason_code = reason_codes[verdict]
    elif rung_verdict is AccuracyVerdict.INACCURATE:
        failure_reason_code = "slop-cap-exhausted"
    elif rung_verdict is AccuracyVerdict.CANNOT_VERIFY:
        failure_reason_code = "cannot-verify-cap-exhausted"
    permanent_scar = bool(fidelity.get("permanent_scar")) or verdict is FidelityVerdict.SLOP
    decision = FidelityRouteDecision(
        stable_id=str(item["stable_id"]),
        verdict=verdict,
        route=GateRoute.ACCEPT if accepted else GateRoute.BLOCK_FIDELITY,
        accepted_for_fidelity=accepted,
        canonical_write_allowed=accepted,
        permanent_scar=permanent_scar,
        failure_reason_code=failure_reason_code,
    )
    if stable_hash(proposal) != before:
        raise GateRoutingError("fidelity routing mutated the staged proposal")
    return decision


def emit_gate_records(gate: Mapping[str, Any]) -> tuple[JsonObject, ...]:
    """Return defensive immutable gate records for reducer append.

    Parameters
    ----------
    gate:
        Validated metadata or fidelity gate.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One append-only record. The API is tuple-shaped for later lane batching.

    Raises
    ------
    GateRoutingError
        If the proposed record is not a complete gate.
    """

    try:
        validate_payload(gate)
    except PayloadValidationError as exc:
        raise GateRoutingError(str(exc)) from exc
    return (deepcopy(dict(gate)),)


def validate_terminal_disposition_gate(
    gate: Mapping[str, Any],
    result: AuthorResult,
    *,
    source_manifest: Mapping[str, Any],
    evidence_pack: Mapping[str, Any],
    license_identity: str,
) -> TerminalDispositionDecision:
    """Resolve a v3 terminal gate to exact result/source/evidence/license facts.

    Parameters
    ----------
    gate:
        Single-item ``terminal_disposition`` gate.v3 record.
    result:
        Exact typed advisory author result under review.
    source_manifest:
        Exact private-staged source manifest bound by the result.
    evidence_pack:
        Exact evidence object containing ``evidence_identity`` and ``excerpts``.
    license_identity:
        Independently derived exact license-disposition identity.

    Returns
    -------
    TerminalDispositionDecision
        Exact checked candidate for the authority kernel. It cannot award a terminal.

    Raises
    ------
    GateRoutingError
        If any result, source, evidence, license, kind, or predicate reference differs.
    """

    if isinstance(result, ProposedAuthorResult):
        raise GateRoutingError("PROPOSED author results cannot use a terminal-disposition gate")
    _validate_gate(gate, GateKind.TERMINAL_DISPOSITION)
    if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3:
        raise GateRoutingError("terminal dispositions require gate.v3")
    item = _items(gate)[0]
    binding = result.binding
    expected_item = {
        "stable_id": binding.stable_id,
        "work_id": binding.work_id,
        "campaign_root_work_id": binding.campaign_id,
    }
    for field, value in expected_item.items():
        if item.get(field) != value:
            raise GateRoutingError(f"terminal gate item {field} does not match author result")
    terminal = item.get("terminal_disposition")
    if not isinstance(terminal, Mapping):
        raise GateRoutingError("terminal gate item lacks terminal_disposition")
    (
        expected_kind,
        expected_predicate,
        result_source_ids,
        result_evidence_ids,
        support_claims,
    ) = _terminal_result_references(result, evidence_pack)
    exact = {
        "author_result_id": binding.result_id,
        "author_result_sha256": binding.result_sha256,
        "kind": expected_kind,
        "predicate": expected_predicate,
        "source_manifest_identity": binding.source_manifest_identity,
        "evidence_identity": result.evidence_identity,
        "license_identity": result.license_identity,
    }
    for field, value in exact.items():
        if terminal.get(field) != value:
            raise GateRoutingError(f"terminal disposition {field} does not match author result")
    gate_source_ids = _unique_string_tuple(terminal.get("source_ids"), "terminal source_ids")
    if set(gate_source_ids) != set(result_source_ids):
        raise GateRoutingError("terminal disposition source IDs do not exactly match result")
    gate_evidence_ids = _unique_string_tuple(terminal.get("evidence_ids"), "terminal evidence_ids")
    if set(gate_evidence_ids) != set(result_evidence_ids):
        raise GateRoutingError("terminal disposition evidence IDs do not exactly match result")
    if _source_manifest_identity(source_manifest) != binding.source_manifest_identity:
        raise GateRoutingError("terminal source manifest identity does not match author result")
    manifest_source_ids = _manifest_source_ids(source_manifest)
    if not set(result_source_ids).issubset(manifest_source_ids):
        raise GateRoutingError("terminal recommendation references a source outside its manifest")
    if evidence_pack.get("evidence_identity") != result.evidence_identity:
        raise GateRoutingError("terminal evidence-pack identity does not match author result")
    _validate_terminal_evidence_references(
        evidence_pack,
        evidence_ids=result_evidence_ids,
        source_ids=result_source_ids,
        support_claims=support_claims,
    )
    if license_identity != result.license_identity:
        raise GateRoutingError("terminal license identity does not match staged license facts")
    verdict = str(terminal.get("verdict"))
    if verdict == "accepted":
        integrity = item.get("integrity")
        if (
            item.get("verdict") != "accurate"
            or not isinstance(integrity, Mapping)
            or (integrity.get("verdict") != "accurate")
        ):
            raise GateRoutingError("accepted terminal disposition requires accurate item integrity")
    findings = terminal.get("findings")
    if not isinstance(findings, list) or not all(isinstance(value, str) for value in findings):
        raise GateRoutingError("terminal disposition findings are invalid")
    return TerminalDispositionDecision(
        stable_id=binding.stable_id,
        work_id=binding.work_id,
        campaign_id=binding.campaign_id,
        gate_id=str(gate["gate_id"]),
        predicate=expected_predicate,
        accepted=verdict == "accepted",
        source_ids=result_source_ids,
        evidence_ids=result_evidence_ids,
        findings=tuple(findings),
    )


def _terminal_result_references(
    result: DeferRecommendation | SkipRecommendation | BlockedRecommendation,
    evidence_pack: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...], tuple[str, ...], frozenset[str]]:
    """Derive exact terminal gate references from one advisory union arm."""

    if isinstance(result, DeferRecommendation):
        predicate = f"needs-{result.platform}"
        return (
            "DEFER_RECOMMENDATION",
            predicate,
            result.source_ids,
            result.evidence_ids,
            frozenset({predicate, f"deferred:{predicate}", f"platform.{result.platform}"}),
        )
    if isinstance(result, SkipRecommendation):
        predicate = result.status_code.removeprefix("skipped:")
        return (
            "SKIP_RECOMMENDATION",
            predicate,
            result.source_ids,
            result.evidence_ids,
            frozenset({predicate, result.status_code}),
        )
    source_ids = _evidence_source_ids(evidence_pack, result.evidence_ids)
    return (
        "BLOCKED",
        "blocked-prerequisite",
        source_ids,
        result.evidence_ids,
        frozenset(
            {
                "blocked-prerequisite",
                f"blocked.{result.stage}.{result.reason_code}",
            }
        ),
    )


def _source_manifest_identity(source_manifest: Mapping[str, Any]) -> str:
    """Return the exact source-manifest identity used by author dispatch."""

    explicit = source_manifest.get("manifest_sha256") or source_manifest.get(
        "source_manifest_identity"
    )
    return (
        str(explicit) if explicit is not None else stable_hash(source_manifest.get("sources", []))
    )


def _manifest_source_ids(source_manifest: Mapping[str, Any]) -> frozenset[str]:
    """Return unique source IDs from an exact staged manifest."""

    sources = source_manifest.get("sources")
    if not isinstance(sources, list):
        raise GateRoutingError("terminal source manifest has no sources")
    source_ids = [source.get("source_id") for source in sources if isinstance(source, Mapping)]
    if (
        len(source_ids) != len(sources)
        or not all(isinstance(source_id, str) and source_id for source_id in source_ids)
        or len(source_ids) != len(set(source_ids))
    ):
        raise GateRoutingError("terminal source manifest IDs are incomplete or duplicated")
    return frozenset(str(source_id) for source_id in source_ids)


def _evidence_source_ids(
    evidence_pack: Mapping[str, Any], evidence_ids: tuple[str, ...]
) -> tuple[str, ...]:
    """Resolve exact source IDs for blocked-result evidence references."""

    excerpts = evidence_pack.get("excerpts")
    if not isinstance(excerpts, list):
        raise GateRoutingError("terminal evidence pack has no excerpts")
    by_id = {
        str(excerpt.get("evidence_id")): excerpt
        for excerpt in excerpts
        if isinstance(excerpt, Mapping) and isinstance(excerpt.get("evidence_id"), str)
    }
    if len(by_id) != len(excerpts):
        raise GateRoutingError("terminal evidence IDs are incomplete or duplicated")
    sources: set[str] = set()
    for evidence_id in evidence_ids:
        excerpt = by_id.get(evidence_id)
        if excerpt is None or not isinstance(excerpt.get("source_id"), str):
            raise GateRoutingError(f"terminal evidence reference is unresolved: {evidence_id}")
        sources.add(str(excerpt["source_id"]))
    return tuple(sorted(sources))


def _validate_terminal_evidence_references(
    evidence_pack: Mapping[str, Any],
    *,
    evidence_ids: tuple[str, ...],
    source_ids: tuple[str, ...],
    support_claims: frozenset[str],
) -> None:
    """Resolve terminal excerpts and require typed predicate support."""

    resolved_sources = _evidence_source_ids(evidence_pack, evidence_ids)
    if not set(resolved_sources).issubset(source_ids):
        raise GateRoutingError("terminal evidence resolves outside the checked source set")
    excerpts = evidence_pack.get("excerpts")
    assert isinstance(excerpts, list)
    by_id = {
        str(excerpt["evidence_id"]): excerpt
        for excerpt in excerpts
        if isinstance(excerpt, Mapping) and "evidence_id" in excerpt
    }
    for evidence_id in evidence_ids:
        supports = by_id[evidence_id].get("supports")
        if not isinstance(supports, list) or not support_claims.intersection(supports):
            raise GateRoutingError(
                f"terminal evidence {evidence_id} does not support its typed predicate"
            )


def _unique_string_tuple(value: object, field: str) -> tuple[str, ...]:
    """Return one duplicate-free string array from a schema-valid gate."""

    if (
        not isinstance(value, list)
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise GateRoutingError(f"{field} must contain unique strings")
    return tuple(value)


def _validate_gate(gate: Mapping[str, Any], expected_kind: GateKind) -> None:
    """Validate a gate schema and exact lane kind.

    Parameters
    ----------
    gate:
        Candidate gate record.
    expected_kind:
        Required lane.

    Raises
    ------
    GateRoutingError
        If schema or kind is invalid.
    """

    try:
        validate_payload(gate)
    except PayloadValidationError as exc:
        raise GateRoutingError(str(exc)) from exc
    if gate.get("gate_kind") != expected_kind.value:
        raise GateRoutingError(f"expected {expected_kind.value} gate")


def _items(gate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return gate items as validated mappings.

    Parameters
    ----------
    gate:
        Schema-valid gate.

    Returns
    -------
    list[Mapping[str, Any]]
        Gate items.

    Raises
    ------
    GateRoutingError
        If an item is unexpectedly not an object.
    """

    raw = gate.get("items")
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise GateRoutingError("gate items must be objects")
    return list(raw)

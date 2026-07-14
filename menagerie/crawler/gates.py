"""Bounded block-at-write routing for immutable metadata and fidelity gates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from menagerie.crawler.constants import (
    GATE_SCHEMA_VERSION,
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
        validate_payload(gate, GATE_SCHEMA_VERSION)
    except PayloadValidationError as exc:
        raise GateRoutingError(str(exc)) from exc
    return (deepcopy(dict(gate)),)


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
        validate_payload(gate, GATE_SCHEMA_VERSION)
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

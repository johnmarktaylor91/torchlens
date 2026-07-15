"""Fresh Codex checker envelopes, atomic results, and usage backoff classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from menagerie.crawler.constants import (
    CHECKER_PROMPT_NAME,
    GATE_SCHEMA_VERSION,
    METADATA_BATCH_MAX,
    METADATA_BATCH_MIN,
    METADATA_FINAL_TAIL_MIN,
    AccuracyVerdict,
    CheckerPauseReason,
    FidelityVerdict,
    GateKind,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.models import JsonObject
from menagerie.crawler.proposal import ProposalValidationError, required_verified_hash_keys
from menagerie.crawler.schema import PayloadValidationError, validate_payload

PROMPT_PATH = Path(__file__).with_name("prompts") / f"{CHECKER_PROMPT_NAME}.txt"


class CheckerDispatchError(ValueError):
    """Raised when a checker request or atomic result violates its contract."""


@dataclass(frozen=True)
class CheckerBackoffSignal:
    """Typed rate/quota pause signal for the later wakeup layer.

    Parameters
    ----------
    reason:
        Closed rate-limit or quota-exhaustion reason.
    retry_after_seconds:
        Provider retry delay when supplied.
    reset_at:
        Provider reset timestamp when supplied.
    response_excerpt:
        Bounded diagnostic response text.
    """

    reason: CheckerPauseReason
    retry_after_seconds: Optional[int]
    reset_at: Optional[str]
    response_excerpt: str


def build_metadata_vet_envelope(
    items: Sequence[Mapping[str, Any]],
    *,
    gate_round: int,
    output_path: Union[str, Path],
    checker_model: str,
    checker_version: str,
    request_nonce: str,
    final_tail: bool = False,
) -> JsonObject:
    """Build a fresh metadata-vetting envelope, including an explicit final tail.

    Parameters
    ----------
    items:
        Independent item packs containing proposal/source/evidence artifacts and
        the exact identity fields expected in each result.
    gate_round:
        Bounded metadata repair round.
    output_path:
        Exact final checker result path.
    checker_model, checker_version:
        Expected independent checker identity.
    request_nonce:
        Caller-created fresh request identity.
    final_tail:
        Whether this is the end-of-queue flush that may contain one to nine items.

    Returns
    -------
    dict[str, Any]
        Hash-bound metadata checker envelope.

    Raises
    ------
    CheckerDispatchError
        If batch cardinality or item identities are invalid.
    """

    minimum = METADATA_FINAL_TAIL_MIN if final_tail else METADATA_BATCH_MIN
    if not minimum <= len(items) <= METADATA_BATCH_MAX:
        raise CheckerDispatchError(
            f"metadata batch must contain {minimum}--{METADATA_BATCH_MAX} items"
        )
    return _build_envelope(
        GateKind.METADATA_BATCH,
        items,
        gate_round=gate_round,
        output_path=output_path,
        checker_model=checker_model,
        checker_version=checker_version,
        request_nonce=request_nonce,
        final_tail=final_tail,
    )


def build_fidelity_envelope(
    item: Mapping[str, Any],
    *,
    gate_round: int,
    output_path: Union[str, Path],
    checker_model: str,
    checker_version: str,
    request_nonce: str,
) -> JsonObject:
    """Build one fresh per-model fidelity checker envelope.

    Parameters
    ----------
    item:
        Complete per-model proposal/source/evidence/code pack.
    gate_round:
        Fidelity review round.
    output_path:
        Exact final checker result path.
    checker_model, checker_version:
        Expected independent checker identity.
    request_nonce:
        Caller-created fresh request identity.

    Returns
    -------
    dict[str, Any]
        Hash-bound fidelity checker envelope.
    """

    return _build_envelope(
        GateKind.FIDELITY,
        [item],
        gate_round=gate_round,
        output_path=output_path,
        checker_model=checker_model,
        checker_version=checker_version,
        request_nonce=request_nonce,
    )


def validate_checker_result(
    result_path: Union[str, Path], envelope: Mapping[str, Any]
) -> JsonObject:
    """Validate one complete gate result and every independently bound item.

    Parameters
    ----------
    result_path:
        Expected final result path.
    envelope:
        Metadata or fidelity request envelope.

    Returns
    -------
    dict[str, Any]
        Schema-valid immutable ``gate.v2`` record for reducer emission.

    Raises
    ------
    CheckerDispatchError
        If the result is partial, stale, mismatched, or logically inconsistent.
    """

    _validate_envelope_hash(envelope)
    path = Path(result_path).resolve()
    if (
        path != Path(str(envelope.get("required_output_path"))).resolve()
        or path.name != "result.json"
    ):
        raise CheckerDispatchError("checker result is not at the exact atomic output path")
    result = _read_json_object(path)
    try:
        validate_payload(result, GATE_SCHEMA_VERSION)
    except PayloadValidationError as exc:
        raise CheckerDispatchError(str(exc)) from exc
    if result.get("gate_kind") != envelope.get("gate_kind"):
        raise CheckerDispatchError("checker result gate_kind does not match its envelope")
    if result.get("gate_round") != envelope.get("gate_round"):
        raise CheckerDispatchError("checker result gate_round does not match its envelope")
    if result.get("gate_identity") != envelope.get("envelope_sha256"):
        raise CheckerDispatchError("checker result gate_identity does not bind its envelope")
    result_checker = _required_mapping(result.get("checker"), "result checker")
    expected_checker = _required_mapping(envelope.get("checker"), "envelope checker")
    for field in ("provider", "model", "version", "prompt_sha256"):
        if result_checker.get(field) != expected_checker.get(field):
            raise CheckerDispatchError(f"checker result {field} does not match its envelope")
    result_items = result.get("items")
    expected_items = envelope.get("items")
    if not isinstance(result_items, list) or not isinstance(expected_items, list):
        raise CheckerDispatchError("checker items are incomplete")
    if (
        result.get("gate_kind") == GateKind.METADATA_BATCH.value
        and len(expected_items) < METADATA_BATCH_MIN
        and envelope.get("final_tail") is not True
    ):
        raise CheckerDispatchError("short metadata result is not bound to a final-tail envelope")
    if result.get("batch_size") != len(expected_items) or len(result_items) != len(expected_items):
        raise CheckerDispatchError("checker result is partial or has extra items")
    expected_by_id = {
        str(item["stable_id"]): item for item in expected_items if isinstance(item, Mapping)
    }
    if len(expected_by_id) != len(expected_items):
        raise CheckerDispatchError("checker envelope contains duplicate stable IDs")
    seen: set[str] = set()
    for result_item in result_items:
        if not isinstance(result_item, Mapping):
            raise CheckerDispatchError("checker result item must be an object")
        stable_id = str(result_item.get("stable_id"))
        expected = expected_by_id.get(stable_id)
        if expected is None or stable_id in seen:
            raise CheckerDispatchError(f"unexpected or duplicate checker item: {stable_id}")
        seen.add(stable_id)
        _validate_item_binding(result_item, expected)
        _validate_item_decision(result_item, GateKind(str(result["gate_kind"])))
    expected_hash = stable_hash(
        {
            key: value
            for key, value in result.items()
            if key not in {"result_envelope_sha256", "payload_sha256"}
        }
    )
    if result.get("result_envelope_sha256") != expected_hash:
        raise CheckerDispatchError("result_envelope_sha256 does not bind the complete gate result")
    return result


def classify_checker_response(
    status_code: int,
    response_body: str,
    *,
    retry_after_seconds: Optional[int] = None,
    reset_at: Optional[str] = None,
) -> Optional[CheckerBackoffSignal]:
    """Classify rate/quota responses without treating them as gate verdicts.

    Parameters
    ----------
    status_code:
        Provider HTTP or process response status.
    response_body:
        Provider response text.
    retry_after_seconds:
        Parsed Retry-After delay, if supplied.
    reset_at:
        Provider reset timestamp, if supplied.

    Returns
    -------
    CheckerBackoffSignal | None
        Typed pause signal, or None for non-rate/quota responses.
    """

    lowered = response_body.lower()
    quota_markers = ("quota", "billing limit", "usage limit", "insufficient_quota")
    rate_markers = ("rate limit", "too many requests", "retry after", "tokens per minute")
    reason: Optional[CheckerPauseReason] = None
    if any(marker in lowered for marker in quota_markers):
        reason = CheckerPauseReason.QUOTA_EXHAUSTED
    elif status_code == 429 or any(marker in lowered for marker in rate_markers):
        reason = CheckerPauseReason.RATE_LIMIT
    if reason is None:
        return None
    return CheckerBackoffSignal(
        reason=reason,
        retry_after_seconds=retry_after_seconds,
        reset_at=reset_at,
        response_excerpt=response_body[:1_500],
    )


def compute_result_envelope_sha256(result: Mapping[str, Any]) -> str:
    """Compute the non-self-referential checker result-envelope digest.

    Parameters
    ----------
    result:
        Gate result with placeholder or absent result/payload hashes.

    Returns
    -------
    str
        Canonical result digest.
    """

    return stable_hash(
        {
            key: value
            for key, value in result.items()
            if key not in {"result_envelope_sha256", "payload_sha256"}
        }
    )


def _build_envelope(
    gate_kind: GateKind,
    items: Sequence[Mapping[str, Any]],
    *,
    gate_round: int,
    output_path: Union[str, Path],
    checker_model: str,
    checker_version: str,
    request_nonce: str,
    final_tail: bool = False,
) -> JsonObject:
    """Build the shared metadata/fidelity envelope body.

    Parameters
    ----------
    gate_kind:
        Metadata batch or fidelity.
    items:
        Independent item artifact packs.
    gate_round:
        Review round.
    output_path:
        Exact final result path.
    checker_model, checker_version:
        Expected checker identity.
    request_nonce:
        Fresh caller-created identity.
    final_tail:
        Whether a metadata request is the explicitly authorized final short batch.

    Returns
    -------
    dict[str, Any]
        Hash-bound request envelope.

    Raises
    ------
    CheckerDispatchError
        If item identity bindings are incomplete or duplicated.
    """

    if gate_round < 1 or not request_nonce:
        raise CheckerDispatchError("gate_round and request_nonce must be positive/non-empty")
    normalized_items: list[JsonObject] = []
    seen: set[str] = set()
    for item in items:
        required = (
            "work_id",
            "stable_id",
            "family_representative_id",
            "fidelity_identity",
            "vet_identity",
            "verified_hashes",
        )
        if any(field not in item for field in required):
            raise CheckerDispatchError("checker item is missing identity/hash bindings")
        verified_hashes = item.get("verified_hashes")
        proposal = item.get("proposal")
        if not isinstance(proposal, Mapping):
            raise CheckerDispatchError("checker item proposal must be a complete object")
        try:
            required_hash_keys = required_verified_hash_keys(proposal, include_proposal=True)
        except ProposalValidationError as exc:
            raise CheckerDispatchError(str(exc)) from exc
        if not isinstance(verified_hashes, Mapping) or set(verified_hashes) != required_hash_keys:
            raise CheckerDispatchError(
                "checker item verified_hashes must bind the exact proposal/artifact pack"
            )
        stable_id = str(item["stable_id"])
        if not stable_id or stable_id in seen:
            raise CheckerDispatchError("checker envelope stable IDs must be non-empty and unique")
        seen.add(stable_id)
        normalized_items.append(dict(item))
    prompt_sha256 = hash_bytes(_read_prompt())
    body: JsonObject = {
        "envelope_version": "menagerie.crawler.checker-envelope.v2",
        "gate_kind": gate_kind.value,
        "gate_round": gate_round,
        "request_nonce": request_nonce,
        "checker": {
            "provider": "openai",
            "model": checker_model,
            "version": checker_version,
            "prompt_sha256": prompt_sha256,
        },
        "prompt": {
            "name": CHECKER_PROMPT_NAME,
            "path": str(PROMPT_PATH),
            "sha256": prompt_sha256,
        },
        "items": normalized_items,
        "required_output_path": str(Path(output_path).resolve()),
        "allowed_output_root": str(Path(output_path).resolve().parent),
        "required_result_schema": GATE_SCHEMA_VERSION,
        "final_tail": final_tail,
    }
    return {**body, "envelope_sha256": stable_hash(body)}


def _validate_item_binding(result_item: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Validate one result item's independent request binding.

    Parameters
    ----------
    result_item:
        Checker-produced gate item.
    expected:
        Exact envelope item pack.

    Raises
    ------
    CheckerDispatchError
        If identity or artifact hashes differ.
    """

    for field in (
        "work_id",
        "stable_id",
        "family_representative_id",
        "fidelity_identity",
        "vet_identity",
        "verified_hashes",
    ):
        if result_item.get(field) != expected.get(field):
            raise CheckerDispatchError(
                f"checker item {expected.get('stable_id')} mismatched binding: {field}"
            )


def _validate_item_decision(result_item: Mapping[str, Any], gate_kind: GateKind) -> None:
    """Enforce top-level verdict precedence and fidelity lane separation.

    Parameters
    ----------
    result_item:
        Schema-valid gate item.
    gate_kind:
        Metadata or fidelity lane.

    Raises
    ------
    CheckerDispatchError
        If a top-level verdict contradicts its atomic checks.
    """

    verdict = AccuracyVerdict(str(result_item.get("verdict")))
    integrity = _required_mapping(result_item.get("integrity"), "integrity")
    checks = result_item.get("field_checks")
    fidelity = _required_mapping(result_item.get("fidelity"), "fidelity")
    if not isinstance(checks, list) or not checks:
        raise CheckerDispatchError("every gate item requires scoped field checks")
    component_verdicts = [str(integrity.get("verdict"))]
    component_verdicts.extend(
        str(check.get("verdict")) for check in checks if isinstance(check, Mapping)
    )
    if AccuracyVerdict.INACCURATE.value in component_verdicts:
        expected = AccuracyVerdict.INACCURATE
    elif AccuracyVerdict.CANNOT_VERIFY.value in component_verdicts:
        expected = AccuracyVerdict.CANNOT_VERIFY
    else:
        expected = AccuracyVerdict.ACCURATE
    if gate_kind is GateKind.METADATA_BATCH:
        if fidelity.get("required") is not False or fidelity.get("verdict") != "not-applicable":
            raise CheckerDispatchError("metadata gates cannot issue fidelity verdicts")
    else:
        if fidelity.get("required") is not True:
            raise CheckerDispatchError("fidelity envelope requires a fidelity decision")
        fidelity_verdict = FidelityVerdict(str(fidelity.get("verdict")))
        if fidelity_verdict in {FidelityVerdict.MAJOR_DRIFT, FidelityVerdict.SLOP}:
            expected = AccuracyVerdict.INACCURATE
        elif (
            fidelity_verdict is FidelityVerdict.CANNOT_VERIFY
            and expected is AccuracyVerdict.ACCURATE
        ):
            expected = AccuracyVerdict.CANNOT_VERIFY
    if verdict is not expected:
        raise CheckerDispatchError(
            f"top-level verdict {verdict.value!r} contradicts component verdict {expected.value!r}"
        )


def _read_prompt() -> bytes:
    """Read exact frozen checker prompt bytes.

    Returns
    -------
    bytes
        Frozen checker prompt.

    Raises
    ------
    CheckerDispatchError
        If the prompt is absent.
    """

    try:
        return PROMPT_PATH.read_bytes()
    except OSError as exc:
        raise CheckerDispatchError(f"cannot read frozen checker prompt: {exc}") from exc


def _validate_envelope_hash(envelope: Mapping[str, Any]) -> None:
    """Validate a checker envelope's self-hash.

    Parameters
    ----------
    envelope:
        Candidate envelope.

    Raises
    ------
    CheckerDispatchError
        If its exact request binding changed.
    """

    expected = stable_hash(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )
    if envelope.get("envelope_sha256") != expected:
        raise CheckerDispatchError("checker envelope hash mismatch")


def _read_json_object(path: Path) -> JsonObject:
    """Read one complete checker result object.

    Parameters
    ----------
    path:
        Final result path.

    Returns
    -------
    dict[str, Any]
        Parsed result.

    Raises
    ------
    CheckerDispatchError
        If the file is absent, partial, symlinked, or not one object.
    """

    try:
        raw = path.read_bytes()
        if not raw or path.is_symlink():
            raise CheckerDispatchError("checker result must be a non-empty regular file")
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckerDispatchError(f"partial or invalid checker result: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CheckerDispatchError("checker result must contain exactly one JSON object")
    return parsed


def _required_mapping(value: object, field: str) -> Mapping[str, Any]:
    """Return a required checker mapping.

    Parameters
    ----------
    value:
        Candidate object.
    field:
        Field name used in errors.

    Returns
    -------
    Mapping[str, Any]
        Valid mapping.

    Raises
    ------
    CheckerDispatchError
        If the value is not an object.
    """

    if not isinstance(value, Mapping):
        raise CheckerDispatchError(f"checker {field} must be an object")
    return value

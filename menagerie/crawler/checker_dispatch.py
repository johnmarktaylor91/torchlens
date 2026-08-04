"""Fresh Codex checker envelopes, atomic results, and usage backoff classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Union

from menagerie.crawler.constants import (
    AUTHOR_DISPATCHER_COMPONENT,
    AUTHOR_RESULT_SCHEMA_COMPONENT,
    CHECKER_PROMPT_NAME,
    GATE_SCHEMA_VERSION_V3,
    METADATA_BATCH_MAX,
    METADATA_BATCH_MIN,
    METADATA_FINAL_TAIL_MIN,
    AccuracyVerdict,
    CheckerPauseReason,
    FidelityVerdict,
    GateKind,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.models import JsonObject, bounded_json_repr
from menagerie.crawler.operator_protocol import build_operator_fields
from menagerie.crawler.proposal import (
    ProposalValidationError,
    required_metadata_field_checks,
    required_verified_hash_keys,
)
from menagerie.crawler.schema import (
    PayloadValidationError,
    RequiredFieldProjection,
    required_field_projection_spec,
    validate_payload,
)
from menagerie.crawler.terminal_evidence import (
    GROUNDED as TERMINAL_EVIDENCE_GROUNDED,
    SOURCE_CAS_DIRNAME,
    TERMINAL_EVIDENCE_RESOLUTIONS,
    UNRESOLVED as TERMINAL_EVIDENCE_UNRESOLVED,
)

PROMPT_PATH = Path(__file__).with_name("prompts") / f"{CHECKER_PROMPT_NAME}.txt"

#: Total severity order over the closed checker verdicts.
#:
#: The frozen checker prompt's DECISION RULE is a maximum over this order: any
#: inaccurate component makes the top level inaccurate, otherwise any
#: cannot-verify makes it cannot-verify, otherwise accurate. Naming the order
#: once lets every lane state that one rule instead of each spelling out a
#: different comparison.
VERDICT_SEVERITY: Mapping[AccuracyVerdict, int] = MappingProxyType(
    {
        AccuracyVerdict.ACCURATE: 0,
        AccuracyVerdict.CANNOT_VERIFY: 1,
        AccuracyVerdict.INACCURATE: 2,
    }
)

#: Terminal-lane verdict lockstep. On a ``terminal_disposition`` item the
#: disposition decides the item's top-level ``verdict``, which carries its exact
#: counterpart below. ``integrity`` is scored on its own evidence and is bounded,
#: not pinned: it may be no MORE severe than the verdict under
#: :data:`VERDICT_SEVERITY`.
#:
#: The integrity half used to be an equality, and that mandated a falsehood. An
#: item rejected purely on the merits, with zero hash mismatches, zero excerpt
#: discrepancies and zero locator failures, still had to declare
#: ``integrity: "inaccurate"``, erasing the only signal separating "rejected
#: because the evidence is forged" from "rejected because the argument fails".
#: The same 10-model pilot rung on 2026-07-30 lost two of six terminal verdicts
#: to that clause: both agreed with the disposition at the top level and both
#: scored ``integrity`` independently (once ``accurate``, once ``cannot-verify``)
#: -- which the monotone rule now ADMITS, because in both the verdict was at
#: least as severe as integrity. The equality had no safety content: it could
#: only force integrity UP to match a severe disposition, never restrain a
#: lenient one. What refuses an unsafe item is the monotone bound, which still
#: rejects every acceptance over degraded integrity -- as
#: :func:`menagerie.crawler.gates.validate_terminal_disposition_gate`
#: independently does.
#:
#: This is a named constant rather than a dict literal inside the check because
#: it is a CONTRACT THE CHECKER MUST BE TOLD; the pilot's rejections were not
#: defiance, they were a rule the prompt never stated.
#: ``test_terminal_verdict_lockstep_is_stated_in_the_frozen_prompt`` reads this
#: mapping and requires the prompt to spell every arrow, so changing the rule
#: here fails until the prompt is re-stated and re-pinned.
TERMINAL_VERDICT_LOCKSTEP: Mapping[str, AccuracyVerdict] = MappingProxyType(
    {
        "accepted": AccuracyVerdict.ACCURATE,
        "rejected": AccuracyVerdict.INACCURATE,
        "cannot-verify": AccuracyVerdict.CANNOT_VERIFY,
    }
)

# Machine-owned gate scaffold. ``gate_id`` and the two component identities are
# IDENTITIES, not judgments: the checker cannot observe them and has no authority
# over them, so asking a model to author them turns a complete, well-reasoned
# verdict into a discarded contract rejection the first time it forgets one. That
# is exactly what killed a live campaign on 2026-07-29 (``'gate_id' is a required
# property``). The machine derives them and the wrapper stamps them; the checker
# owns only its verdict.
GATE_ID_PREFIX = "gate-"

# LEDGER-assigned, not wrapper-stamped. gate.v3's own authority-ownership table
# says ``$.ledger_seq -> reducer-derived``, and ``payload_sha256`` binds the exact
# line the ledger writes. NEITHER value exists until the append happens, so the
# stamped gate OMITS both and ``JsonlLedger.append`` assigns them from the
# ledger's own state.
#
# This used to be a pair of constants stamped into the scaffold
# (``PLACEHOLDER_LEDGER_SEQ = 1``). ``payload_sha256`` survived that because the
# ledger overwrites it unconditionally; ``ledger_seq`` did not, because the ledger
# only fills it when ABSENT and otherwise refuses a value that is not the next
# local sequence. A constant 1 is the next sequence exactly once, so the defect
# was invisible until a campaign recorded its SECOND gate and died with
# ``LedgerConflictError: ledger_seq must be next local sequence 2`` (2026-07-30,
# rung 10, 10/10 authors published). The stamped constant also poisoned the FIRST
# gate: ``result_envelope_sha256`` was computed over a body containing
# ``ledger_seq``, while ``authority.load_current_gate_proof`` recomputes it with
# ``ledger_seq`` excluded, so every recorded gate failed its own self-hash with
# "current gate result-envelope self-hash is invalid". Omitting the fields makes
# both hashes agree by construction, because an absent key and an excluded key
# canonicalize identically.
LEDGER_ASSIGNED_GATE_FIELDS = ("ledger_seq", "payload_sha256")

# Schema-shape stand-ins used ONLY to satisfy gate.v3's ``required`` list when a
# not-yet-appended gate is validated. They are never published, never hashed, and
# never handed to the ledger -- see ``_with_pending_ledger_fields``.
_PENDING_LEDGER_GATE_FIELDS: Mapping[str, Any] = {
    "ledger_seq": 1,
    "payload_sha256": "sha256:" + "0" * 64,
}
DETERMINISTIC_GATE_SCAFFOLD_FIELDS = (
    "schema_version",
    "gate_id",
    "gate_kind",
    "batch_size",
    "gate_round",
    "gate_identity",
    "author_result_schema_identity",
    "dispatcher_identity",
)


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
    provider:
        Closed usage-limit provider vocabulary member owning the pause. The
        checker lane is Codex, so this stays ``"openai"``; it is explicit rather
        than assumed by the scheduler.
    """

    reason: CheckerPauseReason
    retry_after_seconds: Optional[int]
    reset_at: Optional[str]
    response_excerpt: str
    provider: str = "openai"


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


def build_terminal_disposition_envelope(
    item: Mapping[str, Any],
    *,
    gate_round: int,
    output_path: Union[str, Path],
    checker_model: str,
    checker_version: str,
    request_nonce: str,
) -> JsonObject:
    """Build one exact typed terminal-recommendation checker envelope.

    Parameters
    ----------
    item:
        Complete author-result/source/evidence/license pack.
    gate_round, output_path, checker_model, checker_version, request_nonce:
        Fresh checker request bindings.

    Returns
    -------
    dict[str, Any]
        Hash-bound terminal-disposition envelope.
    """

    return _build_envelope(
        GateKind.TERMINAL_DISPOSITION,
        [item],
        gate_round=gate_round,
        output_path=output_path,
        checker_model=checker_model,
        checker_version=checker_version,
        request_nonce=request_nonce,
    )


def machine_owned_gate_fields(envelope: Mapping[str, Any]) -> JsonObject:
    """Derive every gate field the machine owns and a checker must never invent.

    Parameters
    ----------
    envelope:
        Hash-bound metadata, fidelity, or terminal request envelope.

    Returns
    -------
    dict[str, Any]
        Deterministic gate scaffold. The two ``LEDGER_ASSIGNED_GATE_FIELDS`` are
        deliberately absent: the ledger owns them and assigns them at append time.

    Raises
    ------
    CheckerDispatchError
        If the envelope binding is invalid or incomplete.
    """

    _validate_envelope_hash(envelope)
    items = envelope.get("items")
    if not isinstance(items, list) or not items:
        raise CheckerDispatchError("checker envelope has no items")
    envelope_sha256 = str(envelope.get("envelope_sha256"))
    gate_seed = stable_hash(
        {"envelope_sha256": envelope_sha256, "request_nonce": envelope.get("request_nonce")}
    )
    return {
        "schema_version": str(envelope.get("required_result_schema")),
        "gate_id": GATE_ID_PREFIX + gate_seed.removeprefix("sha256:")[:32],
        "gate_kind": envelope.get("gate_kind"),
        "batch_size": len(items),
        "gate_round": envelope.get("gate_round"),
        "gate_identity": envelope_sha256,
        "author_result_schema_identity": component_identity(AUTHOR_RESULT_SCHEMA_COMPONENT),
        "dispatcher_identity": component_identity(AUTHOR_DISPATCHER_COMPONENT),
    }


def apply_machine_owned_gate_fields(
    result: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    started_at: str,
    finished_at: str,
) -> JsonObject:
    """Stamp the machine-owned scaffold onto one decoded checker verdict.

    The checker's authority is its verdict, its findings, and its per-field
    reasoning. Identities and wall timings are the machine's. Stamping happens
    BEFORE validation so a substantively complete verdict is never discarded for
    a scaffold field the model had no way to observe, and it is unconditional so
    a model-supplied identity can never be believed.

    OMITTING a machine-owned field is free and always has been. SUPPLYING one
    with a conflicting value is refused, because the stamp cannot both discard
    a fabricated identity and leave evidence that one was fabricated.

    The ``LEDGER_ASSIGNED_GATE_FIELDS`` are stricter still: they are refused on
    ANY value, because there is no value the checker could legitimately hold. The
    ledger has not run yet, so the correct sequence and payload hash do not exist
    at stamp time -- anything present in those slots was invented.

    Parameters
    ----------
    result:
        Decoded candidate gate carrying the checker's own verdict.
    envelope:
        Hash-bound request envelope owning the scaffold.
    started_at, finished_at:
        Wrapper-observed UTC timestamps for the checker round trip.

    Returns
    -------
    dict[str, Any]
        Scaffolded gate whose ``result_envelope_sha256`` binds the stamped body.

    Raises
    ------
    CheckerDispatchError
        If the envelope binding is invalid, the candidate is not an object, or
        the candidate supplies a machine-owned field with a conflicting value.
    """

    if not isinstance(result, Mapping):
        raise CheckerDispatchError("checker result must contain exactly one JSON object")
    stamped = dict(result)
    scaffold = machine_owned_gate_fields(envelope)
    expected_checker = _required_mapping(envelope.get("checker"), "envelope checker")
    checker_scaffold = {
        "provider": expected_checker.get("provider"),
        "model": expected_checker.get("model"),
        "version": expected_checker.get("version"),
        "prompt_sha256": expected_checker.get("prompt_sha256"),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    supplied_checker = stamped.get("checker")
    if supplied_checker is not None and not isinstance(supplied_checker, Mapping):
        raise CheckerDispatchError("checker result checker field must be one JSON object")
    # Refuse BEFORE stamping. Omission stays free -- that tolerance is the
    # 2026-07-29 fix and is not being narrowed. What is refused is a CONFLICTING
    # supplied value, and the distinction matters because the stamp below is
    # unconditional: it used to overwrite a model-supplied scaffold silently,
    # which meant the two checks that already exist downstream in
    # ``validate_checker_result_mapping`` -- the
    # ``DETERMINISTIC_GATE_SCAFFOLD_FIELDS`` comparison and the four-field
    # ``checker`` comparison -- could never fail, because they ran against
    # values the stamp had just made correct by construction. Those checks were
    # structurally dead. Refusing here restores them, and it also closes the
    # near-miss they were written for: a checker that templates its answer from
    # a repository test fixture arrives carrying that fixture's placeholder
    # verification values, and silently rewriting them to the machine's real
    # ones launders a fabricated gate into a well-formed one.
    _refuse_conflicting_machine_owned(scaffold, stamped, prefix="")
    if isinstance(supplied_checker, Mapping):
        _refuse_conflicting_machine_owned(checker_scaffold, supplied_checker, prefix="checker.")
    # AFTER the scaffold comparison, deliberately: that check names the offending
    # field and both values, and running the ledger refusal first would shadow it
    # for any candidate that happens to carry both kinds of fabrication.
    _refuse_ledger_assigned_fields(stamped, "checker result")
    stamped.update(scaffold)
    checker = dict(supplied_checker) if isinstance(supplied_checker, Mapping) else {}
    checker.update(checker_scaffold)
    stamped["checker"] = checker
    stamped["result_envelope_sha256"] = compute_result_envelope_sha256(stamped)
    return stamped


def _refuse_ledger_assigned_fields(supplied: Mapping[str, Any], origin: str) -> None:
    """Refuse one gate that carries a field only the ledger may assign.

    Parameters
    ----------
    supplied:
        Candidate gate object.
    origin:
        Human-readable description of where the candidate came from.

    Raises
    ------
    CheckerDispatchError
        If any ``LEDGER_ASSIGNED_GATE_FIELDS`` entry is present. Presence alone is
        the offence: the append has not happened, so no correct value exists yet
        and whatever is in the slot was invented. Refusing rather than overwriting
        keeps the ledger's own next-sequence check reachable -- silently rewriting
        the value is what made the constant-``1`` stamp survive five rungs.
    """

    for field in LEDGER_ASSIGNED_GATE_FIELDS:
        if field not in supplied:
            continue
        raise CheckerDispatchError(
            f"{origin} supplies the ledger-assigned field {field}="
            f"{bounded_json_repr(supplied[field])}; {field} is assigned by the ledger from its "
            "own state at append time and does not exist before then, so any value here was "
            "invented rather than derived"
        )


def _with_pending_ledger_fields(gate: Mapping[str, Any]) -> JsonObject:
    """Return one schema-shaped copy of a gate that has not been appended yet.

    Parameters
    ----------
    gate:
        Stamped gate without its ledger-assigned fields.

    Returns
    -------
    dict[str, Any]
        Copy whose ABSENT ledger fields carry stand-in values, so ``gate.v3``'s
        ``required`` list is satisfiable before the append that will supply the
        real ones. A field the candidate actually supplied is left as supplied, so
        an out-of-range value still fails its own schema constraint rather than
        being masked by the stand-in. The copy is validated and discarded; it is
        never hashed, published, or appended.
    """

    return {**_PENDING_LEDGER_GATE_FIELDS, **gate}


def _refuse_conflicting_machine_owned(
    scaffold: Mapping[str, Any], supplied: Mapping[str, Any], *, prefix: str
) -> None:
    """Refuse one checker result that supplies a conflicting machine-owned field.

    Parameters
    ----------
    scaffold:
        Machine-owned field values for this exact envelope.
    supplied:
        Model-authored candidate object at the same nesting level.
    prefix:
        Dotted path prefix used to name the offending field.

    Raises
    ------
    CheckerDispatchError
        If any machine-owned field is present with a value that is not the
        machine's. The message names the field and BOTH values, because the
        supplied value is the evidence: a fixture-templated gate carries the
        fixture's placeholders there, and a message that only said "invalid"
        would discard the one fact that identifies what went wrong.
    """

    for field, machine_value in scaffold.items():
        if field not in supplied:
            continue
        if supplied[field] == machine_value:
            continue
        raise CheckerDispatchError(
            f"checker supplied the machine-owned field {prefix}{field}="
            f"{bounded_json_repr(supplied[field])} but the machine-owned value for this "
            f"envelope is {bounded_json_repr(machine_value)}; the checker's authority is its "
            "verdict, not the gate scaffold, so a conflicting scaffold value is evidence the "
            "gate was templated rather than derived"
        )


def component_identity(relative: str) -> str:
    """Hash one exact shipped crawler component by its package-relative path.

    Parameters
    ----------
    relative:
        Package-relative path of a shipped authority component.

    Returns
    -------
    str
        Content identity, computed identically to the authority context.

    Raises
    ------
    CheckerDispatchError
        If the component is unavailable.
    """

    try:
        return hash_bytes((Path(__file__).parent / relative).read_bytes())
    except OSError as exc:
        raise CheckerDispatchError(f"checker authority component is unavailable: {relative}") from exc


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
    return validate_checker_result_mapping(_read_json_object(path), envelope)


def validate_checker_result_mapping(
    result: Mapping[str, Any], envelope: Mapping[str, Any]
) -> JsonObject:
    """Validate one in-memory gate result before atomic publication.

    Parameters
    ----------
    result:
        Candidate complete gate result.
    envelope:
        Metadata, fidelity, or terminal request envelope.

    Returns
    -------
    dict[str, Any]
        Schema-valid immutable gate record safe to publish.

    Raises
    ------
    CheckerDispatchError
        If the result is partial, stale, mismatched, or logically inconsistent.
    """

    _validate_envelope_hash(envelope)
    normalized = dict(result)
    try:
        # Validate against a PENDING copy: gate.v3 requires the ledger-assigned
        # fields, but this gate has not been appended, so the real values do not
        # exist yet. The stand-ins prove the shape; ``normalized`` -- the object
        # actually returned, hashed, and handed to the reducer -- stays free of
        # them so the ledger can assign from its own state.
        validate_payload(_with_pending_ledger_fields(normalized), GATE_SCHEMA_VERSION_V3)
    except PayloadValidationError as exc:
        raise CheckerDispatchError(str(exc)) from exc
    scaffold = machine_owned_gate_fields(envelope)
    for field in DETERMINISTIC_GATE_SCAFFOLD_FIELDS:
        if normalized.get(field) != scaffold[field]:
            raise CheckerDispatchError(
                f"checker result {field} is not the machine-owned value for its envelope"
            )
    # After the scaffold comparison, so a candidate carrying both kinds of
    # fabrication is still named by the more specific message first.
    _refuse_ledger_assigned_fields(normalized, "checker result")
    result_checker = _required_mapping(normalized.get("checker"), "result checker")
    expected_checker = _required_mapping(envelope.get("checker"), "envelope checker")
    for field in ("provider", "model", "version", "prompt_sha256"):
        if result_checker.get(field) != expected_checker.get(field):
            raise CheckerDispatchError(f"checker result {field} does not match its envelope")
    result_items = normalized.get("items")
    expected_items = envelope.get("items")
    if not isinstance(result_items, list) or not isinstance(expected_items, list):
        raise CheckerDispatchError("checker items are incomplete")
    if (
        normalized.get("gate_kind") == GateKind.METADATA_BATCH.value
        and len(expected_items) < METADATA_BATCH_MIN
        and envelope.get("final_tail") is not True
    ):
        raise CheckerDispatchError("short metadata result is not bound to a final-tail envelope")
    if normalized.get("batch_size") != len(expected_items) or len(result_items) != len(
        expected_items
    ):
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
        _validate_item_decision(result_item, GateKind(str(normalized["gate_kind"])))
    expected_hash = compute_result_envelope_sha256(normalized)
    if normalized.get("result_envelope_sha256") != expected_hash:
        raise CheckerDispatchError("result_envelope_sha256 does not bind the complete gate result")
    return normalized


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
    quota_markers = ("quota", "usage limit")
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
        provider="openai",
    )


def compute_result_envelope_sha256(result: Mapping[str, Any]) -> str:
    """Compute the non-self-referential checker result-envelope digest.

    Parameters
    ----------
    result:
        Gate result with absent result hash and absent ledger-assigned fields.

    Returns
    -------
    str
        Canonical result digest.

    Notes
    -----
    The excluded set is exactly the one ``authority.load_current_gate_proof`` and
    ``driver_models._normalize_gate_generation`` recompute over. It MUST stay
    aligned with them: this digest is written once, before the append, and
    re-derived afterwards from the persisted gate, so any key the ledger touches
    between those two moments has to be outside the digest or no gate can ever
    replay its own self-hash. ``ledger_seq`` was inside here and outside there,
    which made every recorded gate fail with "current gate result-envelope
    self-hash is invalid" the moment anything tried to read one back.
    """

    excluded = {"result_envelope_sha256", *LEDGER_ASSIGNED_GATE_FIELDS}
    return stable_hash({key: value for key, value in result.items() if key not in excluded})


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
        required = required_field_projection_spec(
            RequiredFieldProjection.GATE_ITEM_BINDING
        ).field_order
        if any(field not in item for field in required):
            raise CheckerDispatchError("checker item is missing identity/hash bindings")
        verified_hashes = item.get("verified_hashes")
        if gate_kind is GateKind.TERMINAL_DISPOSITION:
            if not isinstance(verified_hashes, Mapping):
                raise CheckerDispatchError("terminal checker item lacks verified hashes")
            if not isinstance(item.get("author_result"), Mapping):
                raise CheckerDispatchError("terminal checker item lacks its author result")
            if not isinstance(item.get("source_manifest"), Mapping):
                raise CheckerDispatchError("terminal checker item lacks its source manifest")
            evidence_pack = item.get("evidence_pack")
            if not isinstance(evidence_pack, Mapping):
                raise CheckerDispatchError("terminal checker item lacks its evidence pack")
            _validate_terminal_evidence_pack(evidence_pack)
            if not isinstance(item.get("license_disposition"), Mapping):
                raise CheckerDispatchError(
                    "terminal checker item lacks the license disposition its "
                    "license_identity binds"
                )
            if not isinstance(item.get("recommendation_preimage"), Mapping):
                raise CheckerDispatchError(
                    "terminal checker item lacks the recommendation preimage its "
                    "recommendation_sha256 binds"
                )
            stable_id = str(item["stable_id"])
            if not stable_id or stable_id in seen:
                raise CheckerDispatchError(
                    "checker envelope stable IDs must be non-empty and unique"
                )
            seen.add(stable_id)
            normalized_items.append(dict(item))
            continue
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
        normalized = dict(item)
        # The required-check inventory is MACHINE-DERIVED here, at the one
        # boundary every checker request passes through, from the same proposal
        # bytes ``verified_hashes`` binds. The checker was previously asked to
        # reconstruct the required coverage set by inspecting the proposal JSON,
        # which no model does reliably, so complete well-reasoned verdicts died
        # at canonical write on ``ungated authored facts``. Stamping is
        # unconditional; a caller-supplied conflicting value is refused rather
        # than overwritten, mirroring the gate-scaffold rule, because silently
        # rewriting it would hide the only evidence that a caller invented an
        # inventory instead of deriving one.
        facts = proposal.get("proposed_facts")
        if not isinstance(facts, Mapping):
            raise CheckerDispatchError("checker item proposal lacks proposed_facts")
        derived_checks = list(required_metadata_field_checks(facts))
        supplied_checks = normalized.get("required_field_checks")
        if supplied_checks is not None and supplied_checks != derived_checks:
            raise CheckerDispatchError(
                f"checker item {stable_id} supplies a required_field_checks list that is "
                "not the machine derivation for its own proposal; the inventory is "
                "machine-owned and cannot be authored"
            )
        normalized["required_field_checks"] = derived_checks
        normalized_items.append(normalized)
    prompt_sha256 = hash_bytes(_read_prompt())
    resolved_output_path = Path(output_path).resolve()
    operator_fields = build_operator_fields(
        work_generation_identity=request_nonce,
        model=checker_model,
        allowed_read_roots=(resolved_output_path.parent, *_source_read_roots(normalized_items)),
        allowed_write_root=resolved_output_path.parent,
        required_output_path=resolved_output_path,
    )
    body: JsonObject = {
        "envelope_version": "menagerie.crawler.checker-envelope.v3",
        **operator_fields,
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
        "required_output_path": str(resolved_output_path),
        "allowed_output_root": str(resolved_output_path.parent),
        "required_result_schema": GATE_SCHEMA_VERSION_V3,
        "final_tail": final_tail,
    }
    return {**body, "envelope_sha256": stable_hash(body)}


def _source_read_roots(items: Sequence[Mapping[str, Any]]) -> tuple[Path, ...]:
    """Return the content-addressed source roots the checker actually reads.

    The declaration is derived from the items themselves rather than passed in by
    a caller, because a read root that a call site can forget to declare is a
    declaration that will eventually be wrong. It is also the only honest
    declaration available: the checker's job is to re-derive every literal
    excerpt from FROZEN BYTES, and those bytes live in each item's author
    ``source-cas`` tree.

    The previous declaration named this module's own ``prompts`` directory and
    nothing else. That was wrong in both directions. The wrapper inlines the
    frozen prompt text into the argv, so the checker never opens ``prompts/``;
    and the one tree it must open to do its job -- ``source-cas`` -- was not
    declared at all. A declaration that names an unused directory while omitting
    the used one cannot be audited against behavior, which is exactly how a
    checker wandering the repository went unnoticed.

    Parameters
    ----------
    items:
        Normalized envelope items, each carrying its author ``model_dir``.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Deduplicated absolute ``source-cas`` roots in first-seen order.

    Raises
    ------
    CheckerDispatchError
        If an item does not name the author directory whose frozen bytes it
        expects the checker to dereference. Refusing is deliberate: silently
        declaring nothing would restore the untrue declaration this replaces.
    """

    roots: list[Path] = []
    for item in items:
        model_dir = item.get("model_dir")
        if not isinstance(model_dir, str) or not model_dir.strip():
            raise CheckerDispatchError(
                f"checker item {item.get('stable_id')!r} does not name its author model_dir, so "
                "the source-cas root the checker must read cannot be declared"
            )
        cas_root = Path(model_dir).parent / SOURCE_CAS_DIRNAME
        if cas_root not in roots:
            roots.append(cas_root)
    return tuple(roots)


def _validate_terminal_evidence_pack(evidence_pack: Mapping[str, Any]) -> None:
    """Require a terminal evidence pack to be inspectable or honestly unresolved.

    A checker asked to rule on a terminal recommendation must be able to read the
    literal excerpt behind every evidence ID it is handed. The envelope carries
    two separate things and they must never be confused: ``identity_preimage``
    is the machine-derived citation table whose hash IS ``evidence_identity``,
    and ``excerpts`` holds only text the machine re-derived from frozen bytes.

    Resolution is per record, so the structural invariant this enforces is a
    PARTITION: the IDs carrying excerpts and the IDs named unresolved are
    disjoint, and together they are exactly ``declared_evidence_ids``. Every
    declared ID therefore lands in exactly one bucket, and an unverified record
    has no shape in which it can be presented as verified -- it is either an
    excerpt the machine re-derived, or a named gap.

    Parameters
    ----------
    evidence_pack:
        Terminal evidence pack supplied by the driver.

    Raises
    ------
    CheckerDispatchError
        If the pack does not partition its declared evidence IDs into
        re-derived excerpts and explicitly named gaps.
    """

    resolution = evidence_pack.get("resolution")
    declared = evidence_pack.get("declared_evidence_ids")
    excerpts = evidence_pack.get("excerpts")
    if resolution not in TERMINAL_EVIDENCE_RESOLUTIONS:
        raise CheckerDispatchError("terminal evidence pack must declare its resolution")
    if not isinstance(declared, list):
        raise CheckerDispatchError("terminal evidence pack must declare its evidence IDs")
    if not isinstance(evidence_pack.get("identity_preimage"), list):
        raise CheckerDispatchError(
            "terminal evidence pack must carry the preimage of its evidence_identity"
        )
    if not isinstance(excerpts, list):
        raise CheckerDispatchError("terminal evidence pack excerpts must be a list")
    unresolved = evidence_pack.get("unresolved_evidence_ids")
    if resolution == TERMINAL_EVIDENCE_UNRESOLVED:
        if excerpts:
            raise CheckerDispatchError(
                "unresolved terminal evidence pack cannot ship excerpts it did not verify"
            )
        if not isinstance(unresolved, list):
            raise CheckerDispatchError(
                "unresolved terminal evidence pack must name its unresolved evidence IDs"
            )
        if not isinstance(evidence_pack.get("unresolved_reason"), str):
            raise CheckerDispatchError(
                "unresolved terminal evidence pack must explain why it is unresolved"
            )
        return
    if not declared:
        raise CheckerDispatchError("a terminal evidence pack citing nothing cannot be grounded")
    grounded = {
        str(excerpt.get("evidence_id")): excerpt
        for excerpt in excerpts
        if isinstance(excerpt, Mapping)
    }
    for evidence_id, excerpt in grounded.items():
        if any(
            not isinstance(excerpt.get(field), str) or not excerpt.get(field)
            for field in ("source_id", "locator", "text")
        ):
            raise CheckerDispatchError(
                f"grounded terminal evidence pack has no literal excerpt for {evidence_id}"
            )
    if resolution == TERMINAL_EVIDENCE_GROUNDED:
        missing = [str(value) for value in declared if str(value) not in grounded]
        if missing:
            raise CheckerDispatchError(
                f"grounded terminal evidence pack has no literal excerpt for {missing[0]}"
            )
        if unresolved:
            raise CheckerDispatchError(
                "a grounded terminal evidence pack cannot also name unresolved evidence IDs"
            )
        return
    if not grounded:
        raise CheckerDispatchError(
            "a partially grounded terminal evidence pack must carry at least one excerpt"
        )
    if not isinstance(unresolved, list) or not unresolved:
        raise CheckerDispatchError(
            "a partially grounded terminal evidence pack must name its unresolved evidence IDs"
        )
    if not isinstance(evidence_pack.get("unresolved_reason"), str):
        raise CheckerDispatchError(
            "a partially grounded terminal evidence pack must explain each gap"
        )
    named = {str(value) for value in unresolved}
    if named & set(grounded):
        raise CheckerDispatchError(
            "a terminal evidence pack cannot both ground and disclaim the same evidence ID"
        )
    if named | set(grounded) != {str(value) for value in declared}:
        raise CheckerDispatchError(
            "a partially grounded terminal evidence pack must account for every declared "
            "evidence ID exactly once"
        )


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

    # ``campaign_root_work_id`` is required by the gate.v3 item schema, so it is
    # always PRESENT, but its VALUE was verified only by
    # ``driver_models._require_gate_bindings`` -- which the metadata and fidelity
    # lanes reach and the terminal lane does not. On the terminal lane a checker
    # could therefore return any lineage it liked and nothing compared it to the
    # envelope. The prompt now tells the checker to copy this field verbatim from
    # its envelope item; a copy instruction is only worth giving when the copy is
    # checked, so it is checked here, for every gate kind, before publication.
    for field in (
        "work_id",
        "campaign_root_work_id",
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

    Every lane enforces the same rule: the top-level verdict is the maximum of its
    components under :data:`VERDICT_SEVERITY`. The metadata and fidelity lanes take that
    maximum over integrity plus every scoped field check. The terminal lane's components
    are integrity and the disposition, so the same rule reads as two clauses: the verdict
    equals the disposition restated (:data:`TERMINAL_VERDICT_LOCKSTEP`), and the verdict
    is at least as severe as integrity.

    The terminal lane previously required ``integrity == verdict``. That mandated a false
    statement: an item rejected purely on the merits, with zero hash mismatches, zero
    excerpt discrepancies and zero locator failures, still had to declare
    ``integrity: "inaccurate"``. It erased the only signal separating "rejected because the
    evidence is forged" from "rejected because the argument fails", and it contradicted how
    this same function treats integrity in the metadata lane. Dropping the equality removes
    no protection: the equality could only ever force integrity *up* to match a severe
    disposition, never restrain a lenient one. What protects the gate is the monotone
    clause, which still refuses every cell where the verdict is more lenient than integrity
    -- above all an acceptance over degraded integrity, which
    :func:`menagerie.crawler.gates.validate_terminal_disposition_gate` independently
    refuses as well.

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
    if gate_kind is GateKind.TERMINAL_DISPOSITION:
        terminal = _required_mapping(
            result_item.get("terminal_disposition"), "terminal_disposition"
        )
        expected_verdict = TERMINAL_VERDICT_LOCKSTEP[str(terminal.get("verdict"))]
        if verdict is not expected_verdict:
            raise CheckerDispatchError(
                "terminal top-level verdict contradicts the disposition"
            )
        integrity_verdict = AccuracyVerdict(str(integrity.get("verdict")))
        if VERDICT_SEVERITY[verdict] < VERDICT_SEVERITY[integrity_verdict]:
            raise CheckerDispatchError(
                "terminal verdict is less severe than its own integrity verdict"
            )
        return
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

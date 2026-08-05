"""The author lane names its own cause, and splits terminals by OWNERSHIP.

Ten genuine ``c1-mech`` models terminalized as ``failed:author`` / ``session-crashed``
with ``detail: null`` and no ``error_summary`` anywhere. No session crashed: every one
of the ten published a complete ``result.json`` and was then refused by
``_validate_proposal_binding`` because ``proposal.author.version`` said
``claude_crawler_author_v2`` (the PROMPT FILE NAME) where the driver's authority context
had hashed ``current`` (the campaign's ``author_version``). One field, ten models, and
the durable record said the sessions had died.

Two independent defects made that undiagnosable, and these tests pin both:

1. ``AuthorDispatchError`` had 33 raise sites owned by three different parties -- the
   engine's own envelope, the host's cache and prompt bytes, and the author's published
   result -- and the lane's blanket arm recorded ALL of them as ``session-crashed``.
   That reason is a field the reducer counts, so every unrelated cause it absorbed made
   the campaign's own failure census wrong.
2. A ``failed:`` terminal nulls ``status.detail`` on purpose and files the traceback in
   a gitignored local diagnostics sidecar. Safe, but the durable operational record then
   named no cause at all.

Every test here drives the REAL production functions -- ``_validate_proposal_binding``,
``_validate_envelope_hash``, ``build_author_envelope``, ``validate_author_result``,
``_author_lane_failure``, ``_model_lane_failure_event`` -- because a reimplementation of
a classifier proves only that the reimplementation agrees with itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorDispatchError,
    AuthorEngineFaultError,
    AuthorInfrastructureFaultError,
    AuthorResultBinding,
    AuthorResultMalformedError,
    _validate_envelope_hash,
    _validate_proposal_binding,
    validate_author_result,
)
from menagerie.crawler.constants import (
    FAILURE_REASON_CODES,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.driver_admission import (
    _author_lane_failure,
    _model_lane_failure_event,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.schema import validate_payload

pytestmark = pytest.mark.smoke

# The exact author block m3671 published, and the exact one the driver's authority
# context hashed. They differ in ONE field. `model` matched: the author guessed the
# campaign's tier model correctly and the prompt digest was disclosed to it in
# `expected_result.prompt_identity`. `version` is the undisclosed one.
_PROMPT_IDENTITY = "sha256:" + "d3" * 32
_AUTHORED_AUTHOR_BLOCK: dict[str, Any] = {
    "provider": "anthropic",
    "model": "claude-sonnet",
    "version": "claude_crawler_author_v2",
    "prompt_sha256": _PROMPT_IDENTITY,
}
_DRIVER_AUTHOR_BLOCK: dict[str, Any] = {
    "provider": "anthropic",
    "model": "claude-sonnet",
    "version": "current",
    "prompt_sha256": _PROMPT_IDENTITY,
}


def _binding(**overrides: Any) -> AuthorResultBinding:
    """Return a complete result binding with the driver-derived author identity."""

    fields: dict[str, Any] = {
        "result_id": "sha256:" + "01" * 32,
        "result_sha256": "sha256:" + "02" * 32,
        "stable_id": "m3671",
        "work_id": "work-m3671",
        "campaign_id": "campaign-m3671",
        "author_identity": stable_hash(_DRIVER_AUTHOR_BLOCK),
        "prompt_identity": _PROMPT_IDENTITY,
        "dispatcher_identity": "sha256:" + "a2" * 32,
        "source_manifest_identity": "sha256:" + "ad" * 32,
        "intake_snapshot_id": "intake-8b718fb4e6bfa5ede5bb",
        "intake_snapshot_sha256": "sha256:" + "8b" * 32,
        "intake_item_sha256": "sha256:" + "05" * 32,
        "created_at": "2026-07-31T10:19:06.434932Z",
        "raw_result": {},
    }
    fields.update(overrides)
    return AuthorResultBinding(**fields)


def _proposal(binding: AuthorResultBinding, author_block: dict[str, Any]) -> dict[str, Any]:
    """Return a proposal that repeats every binding field, self-hashed."""

    body: dict[str, Any] = {
        "stable_id": binding.stable_id,
        "work_id": binding.work_id,
        "campaign_id": binding.campaign_id,
        "intake_snapshot_id": binding.intake_snapshot_id,
        "intake_snapshot_sha256": binding.intake_snapshot_sha256,
        "intake_item_sha256": binding.intake_item_sha256,
        "source_manifest_identity": binding.source_manifest_identity,
        "dispatcher_identity": binding.dispatcher_identity,
        "author": dict(author_block),
    }
    return {**body, "proposal_sha256": stable_hash(body)}


def _envelope(tmp_path: Path) -> dict[str, Any]:
    """Return a self-hashed v3 author envelope."""

    envelope: dict[str, Any] = {
        "envelope_version": "menagerie.crawler.author-envelope.v3",
        "expected_result": {"stable_id": "m3671"},
        "source_manifest": {"sources": []},
        "allowed_model_dir": str(tmp_path),
        "required_output_path": str(tmp_path / "result.json"),
    }
    envelope["envelope_sha256"] = stable_hash(envelope)
    return envelope


def _raise_and_capture(callable_: Any, *args: Any) -> AuthorDispatchError:
    """Invoke a real validator expected to refuse, and return its exception."""

    with pytest.raises(AuthorDispatchError) as caught:
        callable_(*args)
    return caught.value


# --------------------------------------------------------------------------------
# 1. The measured cause of the ten.
# --------------------------------------------------------------------------------


def test_authored_version_string_is_the_exact_refusal_the_ten_models_hit() -> None:
    """One undisclosed field, and the real validator refuses -- as it should.

    This is a TRIPWIRE working, not a bug in the check. ``proposal.author`` is
    recomputed and compared because a proposal that lifted an identity from somewhere
    its own facts do not produce must be caught. The defect is that the author is
    never told the preimage: ``expected_result`` discloses the opaque
    ``author_identity`` DIGEST and the ``prompt_identity``, and nothing else. Nothing
    here loosens the comparison.
    """

    binding = _binding()

    # The two blocks differ in exactly one field, and it is `version`. Asserted
    # explicitly so this fixture cannot degenerate into comparing a value with itself.
    differing = {
        key
        for key in set(_AUTHORED_AUTHOR_BLOCK) | set(_DRIVER_AUTHOR_BLOCK)
        if _AUTHORED_AUTHOR_BLOCK.get(key) != _DRIVER_AUTHOR_BLOCK.get(key)
    }
    assert differing == {"version"}
    assert stable_hash(_AUTHORED_AUTHOR_BLOCK) != binding.author_identity

    exc = _raise_and_capture(
        _validate_proposal_binding,
        _proposal(binding, _AUTHORED_AUTHOR_BLOCK),
        binding,
    )
    assert str(exc) == "proposal author identity does not match its result binding"

    # The identical proposal with the driver's own author block is accepted, so the
    # refusal is attributable to that one field and to nothing else in the fixture.
    _validate_proposal_binding(_proposal(binding, _DRIVER_AUTHOR_BLOCK), binding)


# --------------------------------------------------------------------------------
# 2. The old code collapsed every owner into one reason. Proven, not asserted.
# --------------------------------------------------------------------------------


def test_old_classification_collapsed_every_owner_onto_session_crashed(
    tmp_path: Path,
) -> None:
    """Before the split, an engine fault and an author fault were the same terminal.

    Both real raise sites still produce a subclass of ``AuthorDispatchError`` -- which
    is exactly what the old blanket arm dispatched on -- so the collapse is reproduced
    from the production call paths rather than described. Constructing the BARE base,
    as all 33 sites once did, still routes to ``session-crashed`` for both messages,
    which is the disarmed behaviour this change removes.
    """

    binding = _binding()
    author_fault = _raise_and_capture(
        _validate_proposal_binding,
        _proposal(binding, _AUTHORED_AUTHOR_BLOCK),
        binding,
    )
    tampered = _envelope(tmp_path)
    tampered["allowed_model_dir"] = str(tmp_path / "elsewhere")
    engine_fault = _raise_and_capture(_validate_envelope_hash, tampered)

    # Old dispatch input: both are AuthorDispatchError, indistinguishable by base type.
    assert isinstance(author_fault, AuthorDispatchError)
    assert isinstance(engine_fault, AuthorDispatchError)

    # Old dispatch OUTPUT, reproduced through the real classifier by handing it the
    # bare base every site used to raise.
    for message in (str(author_fault), str(engine_fault)):
        assert _author_lane_failure(AuthorDispatchError(message)) == (
            "author",
            "session-crashed",
        )

    # New dispatch output: two different terminals, from the same two raise sites.
    assert _author_lane_failure(author_fault) == ("author", "malformed-result")
    assert _author_lane_failure(engine_fault) == ("runner", "internal-error")
    assert _author_lane_failure(author_fault) != _author_lane_failure(engine_fault)


# --------------------------------------------------------------------------------
# 3. The split, exercised over several real raise sites per owner.
# --------------------------------------------------------------------------------


def test_author_owned_raise_sites_record_malformed_result(tmp_path: Path) -> None:
    """Several distinct author-owned refusals, all on the author-content terminal.

    More than one site per owner on purpose: a single frozen fixture hides
    width-dependent defects, and a taxonomy that happens to be right at one call site
    proves nothing about the other thirty-two.
    """

    binding = _binding()
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(["not an object"]), encoding="utf-8")

    faults = {
        "proposal-author-identity": _raise_and_capture(
            _validate_proposal_binding,
            _proposal(binding, _AUTHORED_AUTHOR_BLOCK),
            binding,
        ),
        "proposal-binding-field": _raise_and_capture(
            _validate_proposal_binding,
            {**_proposal(binding, _DRIVER_AUTHOR_BLOCK), "work_id": "work-somebody-else"},
            binding,
        ),
        "result-not-one-object": _raise_and_capture(
            validate_author_result,
            result_path,
            _envelope(tmp_path),
        ),
    }
    for label, exc in faults.items():
        assert isinstance(exc, AuthorResultMalformedError), label
        assert _author_lane_failure(exc) == ("author", "malformed-result"), label

    # The recorded reason has to be a member of the stage vocabulary it is recorded
    # under, or the terminal is unrepresentable rather than merely wrong.
    assert "malformed-result" in FAILURE_REASON_CODES["author"]


def test_engine_owned_raise_sites_record_runner_internal_error(tmp_path: Path) -> None:
    """Engine-built envelope defects leave the author-stage census alone."""

    tampered_hash = _envelope(tmp_path)
    tampered_hash["expected_result"] = {"stable_id": "m-somebody-else"}
    wrong_version = _envelope(tmp_path)
    wrong_version["envelope_version"] = "menagerie.crawler.author-envelope.v2"

    misrouted = _envelope(tmp_path)
    misrouted_path = tmp_path / "elsewhere" / "result.json"
    misrouted_path.parent.mkdir()
    misrouted_path.write_text("{}", encoding="utf-8")

    faults = {
        "envelope-hash": _raise_and_capture(_validate_envelope_hash, tampered_hash),
        "envelope-version": _raise_and_capture(_validate_envelope_hash, wrong_version),
        "atomic-path": _raise_and_capture(
            validate_author_result, misrouted_path, misrouted
        ),
    }
    for label, exc in faults.items():
        assert isinstance(exc, AuthorEngineFaultError), label
        assert _author_lane_failure(exc) == ("runner", "internal-error"), label

    assert "internal-error" in FAILURE_REASON_CODES["runner"]


def test_infrastructure_faults_share_the_engine_terminal_but_not_its_type() -> None:
    """Host faults are a distinct TYPE even where the reason vocabulary is shared.

    The ``runner`` vocabulary has no retryable member, so an infrastructure fault
    lands on ``internal-error`` alongside an engine fault. Collapsing the two TYPES as
    well would repeat the mistake being fixed: the exception type is what an operator
    reads off ``error_summary`` to decide between fixing the host and fixing the code.
    """

    infra = AuthorInfrastructureFaultError("author-result cache identity mismatch")
    engine = AuthorEngineFaultError("author envelope hash mismatch")

    assert _author_lane_failure(infra) == _author_lane_failure(engine)
    assert type(infra) is not type(engine)
    assert not isinstance(infra, AuthorEngineFaultError)
    assert not isinstance(engine, AuthorInfrastructureFaultError)


def test_ownership_ambiguous_sites_still_reach_the_catch_all() -> None:
    """A site that cannot know its owner keeps the honest catch-all.

    ``session-crashed`` remains reachable and remains correct for the one observation
    consistent with it: nothing usable at the envelope's atomic path. A split that
    emptied the catch-all would have had to guess somewhere.
    """

    assert _author_lane_failure(
        AuthorDispatchError("author result must be a non-empty regular file")
    ) == ("author", "session-crashed")
    assert "session-crashed" in FAILURE_REASON_CODES["author"]


# --------------------------------------------------------------------------------
# 4. The terminal names its own cause.
# --------------------------------------------------------------------------------


def _event_for(exc: AuthorDispatchError) -> dict[str, Any]:
    """Build the real operational event for one classified lane failure."""

    stage, reason_code = _author_lane_failure(exc)
    return dict(
        _model_lane_failure_event(
            stable_id="m3671",
            work_id="work-m3671",
            status_code=f"failed:{stage}",
            reason_code=reason_code,
            exc=exc,
            run_id="run-pilot",
            machine_id="mymini",
            created_at="2026-07-31T10:19:06.504798Z",
        )
    )


def test_lane_failure_event_names_type_message_and_raise_site() -> None:
    """The cause is legible at the event's TOP level, not nested for excavation."""

    binding = _binding()
    exc = _raise_and_capture(
        _validate_proposal_binding,
        _proposal(binding, _AUTHORED_AUTHOR_BLOCK),
        binding,
    )
    event = _event_for(exc)

    assert event["error_summary"] == (
        "AuthorResultMalformedError: proposal author identity does not "
        "match its result binding"
    )
    assert event["event_kind"] == OperationalEventKind.MODEL_LANE_FAILED.value
    assert event["status"] == OperationalEventStatus.MODEL_LANE_FAILED.value
    details = event["details"]
    assert details["error_type"] == "AuthorResultMalformedError"
    assert details["error_message"] == (
        "proposal author identity does not match its result binding"
    )
    assert details["status_code"] == "failed:author"
    assert details["reason_code"] == "malformed-result"

    # The raise site is machine-derived from the traceback, and it names the file and
    # function that actually refused -- not the classifier, and not the test.
    raise_site = details["raise_site"]
    assert raise_site.startswith("author_dispatch.py:")
    assert raise_site.endswith(" in _validate_proposal_binding")

    # Top-level, not nested: the terminal-unrecordable ladder established that a cause
    # buried in a nested report is a cause nobody reads.
    assert "error_summary" in event
    validate_payload(
        {**event, "ledger_seq": 1, "payload_sha256": "sha256:" + "0f" * 32},
        OPERATIONAL_EVENT_SCHEMA_VERSION,
    )


def test_lane_failure_events_distinguish_the_two_owners(tmp_path: Path) -> None:
    """An author-content terminal is distinguishable from an engine-fault terminal.

    The failing direction is the point: two causes, two events, and every discriminating
    slot different. A test that only asserted the author case would pass just as well
    against the old code path that produced one undifferentiated ``session-crashed``.
    """

    binding = _binding()
    author_event = _event_for(
        _raise_and_capture(
            _validate_proposal_binding,
            _proposal(binding, _AUTHORED_AUTHOR_BLOCK),
            binding,
        )
    )
    tampered = _envelope(tmp_path)
    tampered["source_manifest"] = {"sources": [{"source_id": "impl-main"}]}
    engine_event = _event_for(_raise_and_capture(_validate_envelope_hash, tampered))

    assert author_event["details"]["status_code"] == "failed:author"
    assert engine_event["details"]["status_code"] == "failed:runner"
    assert author_event["details"]["reason_code"] == "malformed-result"
    assert engine_event["details"]["reason_code"] == "internal-error"
    assert author_event["details"]["error_type"] == "AuthorResultMalformedError"
    assert engine_event["details"]["error_type"] == "AuthorEngineFaultError"
    assert author_event["error_summary"] != engine_event["error_summary"]
    assert author_event["details"]["raise_site"] != engine_event["details"]["raise_site"]
    assert author_event["event_id"] != engine_event["event_id"]

    for event in (author_event, engine_event):
        validate_payload(
            {**event, "ledger_seq": 1, "payload_sha256": "sha256:" + "0f" * 32},
            OPERATIONAL_EVENT_SCHEMA_VERSION,
        )


def test_identity_mismatch_terminal_also_names_its_cause() -> None:
    """The NEXT wall for these same ten models is legible too.

    ``_validate_artifact_identities`` recomputes five author-gated identities and
    raises ``DriverIntegrationError`` carrying the per-field claimed/computed mismatch
    dict -- the entire diagnostic -- into a detail the terminal then nulls. All ten
    pilot proposals mismatch ``source_identity``, ``evidence_identity``, and
    ``recipe_revision`` as well, so fixing only the author block moves them here.

    Its reason code is NOT reclassified. ``DriverIntegrationError`` is an engine
    exception TYPE carrying an author-CONTENT fault, and blanket-routing the type to
    ``failed:runner`` would be exactly the wrong-census mistake in the other
    direction. Only the cause is surfaced.
    """

    from menagerie.crawler.driver_contracts import DriverIntegrationError

    exc = DriverIntegrationError(
        "author proposal identity mismatch: "
        "{'evidence_identity': {'claimed': 'sha256:aa', 'computed': 'sha256:bb'}}"
    )
    event = dict(
        _model_lane_failure_event(
            stable_id="m3671",
            work_id="work-m3671",
            status_code="failed:evidence",
            reason_code="coverage-incomplete",
            exc=exc,
            run_id="run-pilot",
            machine_id="mymini",
            created_at="2026-07-31T10:19:06.504798Z",
        )
    )

    assert event["error_summary"].startswith(
        "DriverIntegrationError: author proposal identity mismatch:"
    )
    assert event["details"]["error_type"] == "DriverIntegrationError"
    assert event["details"]["reason_code"] == "coverage-incomplete"
    assert "coverage-incomplete" in FAILURE_REASON_CODES["evidence"]
    validate_payload(
        {**event, "ledger_seq": 1, "payload_sha256": "sha256:" + "0f" * 32},
        OPERATIONAL_EVENT_SCHEMA_VERSION,
    )


def test_lane_failure_event_survives_an_exception_with_no_traceback() -> None:
    """A raise site is a best-effort discriminator; its absence never aborts a terminal.

    ``driver._blocked_terminal``'s totality has a sibling obligation here: the event
    that explains a failure must not itself be able to fail on an input it did not
    expect. An exception object that was never raised has no traceback at all.
    """

    event = _event_for(AuthorResultMalformedError("never raised"))

    assert event["details"]["raise_site"] is None
    assert event["error_summary"] == "AuthorResultMalformedError: never raised"
    validate_payload(
        {**event, "ledger_seq": 1, "payload_sha256": "sha256:" + "0f" * 32},
        OPERATIONAL_EVENT_SCHEMA_VERSION,
    )

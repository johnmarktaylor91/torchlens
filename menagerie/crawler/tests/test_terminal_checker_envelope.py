"""Terminal checker verdict routing and grounded-evidence envelope tripwires.

Two live campaign-killing defects are pinned here (rung 1d, 2026-07-29):

1. A legitimate checker verdict was discarded because the checker model omitted
   ``gate_id`` -- a machine-owned identity it could not observe -- and the
   resulting typed contract rejection propagated as an undifferentiated
   ``DriverIntegrationError`` that ended the whole run.
2. The frozen terminal envelope carried evidence *identifiers* plus a
   machine-derived citation table (source IDs paired by round-robin index,
   ``supports`` stamped with the typed predicate) presented as ``excerpts``.
   That table owns the evidence IDENTITY and is kept, under its own name; what
   the checker additionally needs, and now gets, is literal excerpt text the
   machine re-derived verbatim from the frozen source bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorResultBinding,
    BlockedRecommendation,
    derive_terminal_evidence_pack,
    derive_terminal_license_disposition,
)
from menagerie.crawler.checker_dispatch import (
    LEDGER_ASSIGNED_GATE_FIELDS,
    PROMPT_PATH,
    TERMINAL_VERDICT_LOCKSTEP,
    CheckerDispatchError,
    apply_machine_owned_gate_fields,
    build_terminal_disposition_envelope,
    machine_owned_gate_fields,
    validate_checker_result_mapping,
)
from menagerie.crawler.driver_admission import _raise_for_checker_exit
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    DriverIntegrationError,
    RetryableOperatorError,
)
from menagerie.crawler.driver_models import _terminal_checker_item
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.operator_checker import TERMINAL_CHECKER_MODEL
from menagerie.crawler.operator_protocol import status_sidecar_path
from menagerie.crawler.terminal_evidence import (
    GROUNDED,
    TERMINAL_EVIDENCE_FILENAME,
    UNRESOLVED,
    resolve_terminal_evidence,
)
from menagerie.crawler.tests.conftest import HASH, make_gate

SOURCE_ONE_BYTES = b"[convolutional]\nbatch_normalize=1\nfilters=16\nactivation=leaky\n"
SOURCE_TWO_BYTES = b"  [[-1, 1, Conv, [16, 3, 1]],  # 0\n   [-1, 1, nn.MaxPool2d, [2, 2, 0]],\n"
PREDICATE = "blocked-prerequisite"
DECLARED_EVIDENCE = ("ev-one", "ev-two")


def _excerpts() -> list[dict[str, Any]]:
    """Return two literal excerpt records present verbatim in their sources."""

    return [
        {
            "evidence_id": "ev-one",
            "source_id": "source-1",
            "locator": "cfg/yolov3-tiny.cfg lines 25-31",
            "text": "activation=leaky\n",
            "supports": ["source_resolution.rung"],
        },
        {
            "evidence_id": "ev-two",
            "source_id": "source-2",
            "locator": "models/yolov3-tiny.yaml lines 13-16",
            "text": "[-1, 1, nn.MaxPool2d, [2, 2, 0]]",
            "supports": ["fidelity.deviations"],
        },
    ]


def _stage_sources(author_root: Path) -> dict[str, Any]:
    """Write both frozen sources into the CAS and return their manifest."""

    cas_root = author_root / "source-cas"
    cas_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_id, content in (("source-1", SOURCE_ONE_BYTES), ("source-2", SOURCE_TWO_BYTES)):
        digest = hash_bytes(content)
        path = cas_root / f"{digest.removeprefix('sha256:')}.source"
        path.write_bytes(content)
        rows.append(
            {
                "source_id": source_id,
                "url": f"https://example.org/{source_id}",
                "content_sha256": digest,
                "cas_path": str(path),
            }
        )
    return {"manifest_sha256": HASH, "sources": rows}


def _blocked_artifact(
    tmp_path: Path,
    *,
    excerpts: list[dict[str, Any]] | None,
    evidence_ids: tuple[str, ...] = DECLARED_EVIDENCE,
) -> AuthorArtifact:
    """Stage one BLOCKED recommendation with optional frozen author excerpts.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory used as the campaign work root.
    excerpts:
        Frozen excerpt records to publish, or ``None`` to publish none.
    evidence_ids:
        Exact evidence identities the recommendation declares.

    Returns
    -------
    AuthorArtifact
        Privately staged terminal artifact whose identities are machine-derived.
    """

    author_root = tmp_path / "work" / "m_example" / "author"
    model_dir = author_root / "model"
    model_dir.mkdir(parents=True)
    source_manifest = _stage_sources(author_root)
    if excerpts is not None:
        (author_root / TERMINAL_EVIDENCE_FILENAME).write_text(
            json.dumps({"excerpts": excerpts}), encoding="utf-8"
        )
    source_ids = tuple(str(row["source_id"]) for row in source_manifest["sources"])
    evidence_pack = derive_terminal_evidence_pack(
        source_ids=source_ids, evidence_ids=evidence_ids, predicate=PREDICATE
    )
    license_disposition = derive_terminal_license_disposition(
        kind="BLOCKED", source_manifest_identity=HASH
    )
    payload = {
        "arm": "BLOCKED",
        "stage": "source",
        "reason_code": "missing-material-source",
        "prerequisite_ids": ["prereq-1"],
        "evidence_ids": list(evidence_ids),
        "evidence_identity": evidence_pack["evidence_identity"],
        "license_identity": stable_hash(license_disposition),
    }
    payload["recommendation_sha256"] = stable_hash(payload)
    raw_fields = {
        "result_id": "result-blocked",
        "result_sha256": HASH,
        "stable_id": "m_example",
        "work_id": "work-m_example",
        "campaign_id": "work-m_example",
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": HASH,
        "intake_snapshot_id": "intake-1",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": HASH,
        "created_at": "2026-07-29T00:00:00Z",
    }
    binding = AuthorResultBinding(
        raw_result={**raw_fields, "kind": "BLOCKED", "payload": payload}, **raw_fields
    )
    result = BlockedRecommendation(
        binding=binding,
        stage="source",
        reason_code="missing-material-source",
        prerequisite_ids=("prereq-1",),
        evidence_ids=evidence_ids,
        evidence_identity=str(payload["evidence_identity"]),
        license_identity=str(payload["license_identity"]),
        recommendation_sha256=str(payload["recommendation_sha256"]),
    )
    return AuthorArtifact(
        author_result=result, source_manifest=source_manifest, model_dir=model_dir
    )


def _terminal_envelope(item: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Build one terminal-disposition envelope around a checker item pack."""

    return build_terminal_disposition_envelope(
        item,
        gate_round=1,
        output_path=tmp_path / "checker" / "result.json",
        checker_model=TERMINAL_CHECKER_MODEL,
        checker_version="test",
        request_nonce="terminal-m_example",
    )


def _rejected_verdict_body() -> dict[str, Any]:
    """Return a complete terminal verdict body with no machine-owned scaffold."""

    gate = make_gate(["m_example"], gate_kind="terminal_disposition")
    gate["items"][0]["terminal_disposition"] = {
        "author_result_id": "result-blocked",
        "author_result_sha256": HASH,
        "handoff_proposal_id": None,
        "handoff_sha256": None,
        "kind": "BLOCKED",
        "predicate": PREDICATE,
        "verdict": "rejected",
        "source_manifest_identity": HASH,
        "source_ids": ["source-1", "source-2"],
        "evidence_identity": HASH,
        "evidence_ids": list(DECLARED_EVIDENCE),
        "license_identity": HASH,
        "findings": ["evidence excerpts are not inspectable"],
    }
    gate["items"][0]["verdict"] = "inaccurate"
    gate["items"][0]["integrity"]["verdict"] = "inaccurate"
    gate["items"][0]["required_repairs"] = ["supply the frozen evidence records"]
    return gate


# -- 1. a verdict is a per-model result, not an infrastructure failure ---------


def test_verdict_missing_machine_owned_scaffold_is_not_discarded(tmp_path: Path) -> None:
    """A complete rejection survives even when the checker omits every machine field.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=_excerpts())
    envelope = _terminal_envelope(_terminal_checker_item(artifact), tmp_path)
    verdict = _rejected_verdict_body()
    item = verdict["items"][0]
    for field in (
        "work_id",
        "stable_id",
        "family_representative_id",
        "fidelity_identity",
        "vet_identity",
        "verified_hashes",
    ):
        item[field] = envelope["items"][0][field]
    for machine_field in (*machine_owned_gate_fields(envelope), *LEDGER_ASSIGNED_GATE_FIELDS):
        # ``schema_version`` is the one machine field the checker can read off
        # the envelope, so the live omission was exactly the rest of them. The
        # ledger-assigned pair goes with them: the ledger assigns it at append
        # time, so a not-yet-appended gate never legitimately carries it.
        if machine_field != "schema_version":
            verdict.pop(machine_field, None)
    verdict.pop("checker", None)

    # Pre-fix behaviour: the raw verdict is refused for a field the checker
    # could not observe, and the whole campaign died on that refusal.
    with pytest.raises(CheckerDispatchError, match="gate_id"):
        validate_checker_result_mapping(verdict, envelope)

    stamped = apply_machine_owned_gate_fields(
        verdict,
        envelope,
        started_at="2026-07-29T00:00:00Z",
        finished_at="2026-07-29T00:05:00Z",
    )
    validated = validate_checker_result_mapping(stamped, envelope)
    assert validated["items"][0]["terminal_disposition"]["verdict"] == "rejected"
    assert validated["gate_identity"] == envelope["envelope_sha256"]


def test_machine_owned_gate_identity_is_never_taken_from_the_checker(tmp_path: Path) -> None:
    """A checker-supplied identity is refused outright, not quietly corrected.

    This assertion used to be the opposite: a fabricated ``gate_id`` was
    overwritten and the stamped gate proceeded. "Overwritten, never believed"
    is true but insufficient -- silently correcting a fabricated identity also
    ERASES the only evidence that the checker fabricated one, so a gate
    templated from a fixture (which necessarily carries that fixture's
    placeholder identities) was laundered into a well-formed one. A fabricated
    identity is now a contract rejection that names the field and the value.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=_excerpts())
    envelope = _terminal_envelope(_terminal_checker_item(artifact), tmp_path)
    verdict = _rejected_verdict_body()
    verdict["gate_id"] = "gate-fabricated"
    verdict["dispatcher_identity"] = "sha256:" + "f" * 64

    with pytest.raises(CheckerDispatchError) as excinfo:
        apply_machine_owned_gate_fields(
            verdict,
            envelope,
            started_at="2026-07-29T00:00:00Z",
            finished_at="2026-07-29T00:05:00Z",
        )

    message = str(excinfo.value)
    assert 'machine-owned field gate_id="gate-fabricated"' in message
    assert machine_owned_gate_fields(envelope)["gate_id"] in message

    # Omission remains free: the same verdict with every machine-owned and
    # ledger-assigned field omitted stamps exactly as before.
    for machine_field in (*machine_owned_gate_fields(envelope), *LEDGER_ASSIGNED_GATE_FIELDS):
        if machine_field != "schema_version":
            verdict.pop(machine_field, None)
    verdict.pop("checker", None)
    stamped = apply_machine_owned_gate_fields(
        verdict,
        envelope,
        started_at="2026-07-29T00:00:00Z",
        finished_at="2026-07-29T00:05:00Z",
    )
    expected = machine_owned_gate_fields(envelope)
    assert stamped["gate_id"] == expected["gate_id"] != "gate-fabricated"
    assert stamped["dispatcher_identity"] == expected["dispatcher_identity"]


# -- 2. only a genuine execution failure is an integration error ---------------


def test_retryable_and_unavailable_checker_exits_do_not_end_the_campaign(
    tmp_path: Path,
) -> None:
    """Declared-transient wrapper exits are bounded transport retries.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    request_path = tmp_path / "request.json"
    for returncode in (75, 78):
        with pytest.raises(RetryableOperatorError):
            _raise_for_checker_exit(
                returncode, "", "transport failed", request_path=request_path
            )


def test_genuine_checker_execution_failure_still_raises_integration_error(
    tmp_path: Path,
) -> None:
    """A crash, a missing binary, or a refused contract stays an integration error.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    request_path = tmp_path / "request.json"
    with pytest.raises(DriverIntegrationError) as crashed:
        _raise_for_checker_exit(
            127, "", "codex: command not found", request_path=request_path
        )
    assert not isinstance(crashed.value, RetryableOperatorError)
    with pytest.raises(DriverIntegrationError, match="violated the gate contract"):
        _raise_for_checker_exit(
            64, "", "no valid gate was produced", request_path=request_path
        )
    _raise_for_checker_exit(0, "", "", request_path=request_path)


# The verbatim leak recorded on 2026-07-30: a Codex ``command_execution`` event
# whose ``aggregated_output`` was this package's own ``tests/conftest.py``. The
# driver copied this into a durable campaign record as the failure "reason".
_LEAKED_EVENT_STREAM = (
    '{"type":"item.completed","item":{"type":"command_execution",'
    '"command":"/bin/zsh -lc \\"sed -n 2000,2095p '
    'menagerie/crawler/tests/conftest.py\\"","aggregated_output":'
    '"    proposal = {\\n        \\"gate_identity\\": HASH,\\n'
    '        \\"checker\\": {\\"version\\": \\"test\\"},\\n",'
    '"exit_code":0,"status":"completed"}}\n'
)


def test_checker_failure_evidence_is_the_wrapper_reason_not_the_event_stream(
    tmp_path: Path,
) -> None:
    """The wrapper's structured reason wins, and the raw stream never leaks.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    request_path = tmp_path / "request.json"
    status_sidecar_path(request_path).write_text(
        json.dumps(
            {
                "classification": "permanent-contract-rejection",
                "exit_code": 64,
                "detail": (
                    "menagerie.crawler.gate.v3 validation failed at items[0] "
                    "(schema path properties.items.items.required; constraint required): "
                    "'campaign_root_work_id' is a required property"
                ),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DriverIntegrationError) as excinfo:
        _raise_for_checker_exit(
            64, _LEAKED_EVENT_STREAM, "", request_path=request_path
        )

    message = str(excinfo.value)
    # The violating party is named, and the wrapper is named as the detector.
    assert "checker violated the gate contract" in message
    assert "refused by the operator wrapper" in message
    # The precise reason is carried as evidence.
    assert "'campaign_root_work_id' is a required property" in message
    assert "[permanent-contract-rejection]" in message
    # The failing direction: repository source from the event stream must not
    # reach a durable record. Before this guard the whole tail was the message.
    assert "conftest.py" not in message
    assert "aggregated_output" not in message
    assert "command_execution" not in message


def test_absent_wrapper_status_falls_back_and_says_it_is_a_transcript(
    tmp_path: Path,
) -> None:
    """Without a sidecar the tail is still shown, but never as a diagnosis.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    request_path = tmp_path / "request.json"

    with pytest.raises(DriverIntegrationError) as excinfo:
        _raise_for_checker_exit(
            64, _LEAKED_EVENT_STREAM, "", request_path=request_path
        )

    message = str(excinfo.value)
    assert "published no structured reason" in message
    assert "raw stream tail:" in message


# -- 3. the envelope carries literal excerpts and locators ---------------------


def test_terminal_envelope_carries_literal_excerpts_and_locators(tmp_path: Path) -> None:
    """Every referenced evidence ID arrives with verbatim text and its locator.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=_excerpts())
    item = _terminal_checker_item(artifact)
    pack = item["evidence_pack"]
    assert pack["resolution"] == GROUNDED
    by_id = {excerpt["evidence_id"]: excerpt for excerpt in pack["excerpts"]}
    assert set(by_id) == set(DECLARED_EVIDENCE)
    for excerpt in by_id.values():
        assert excerpt["locator"]
        assert excerpt["text"]
    # Real provenance re-derived from frozen bytes, not the round-robin pairing.
    assert by_id["ev-one"]["source_id"] == "source-1"
    assert by_id["ev-two"]["source_id"] == "source-2"
    assert by_id["ev-one"]["text"].encode("utf-8") in SOURCE_ONE_BYTES
    assert by_id["ev-two"]["text"].encode("utf-8") in SOURCE_TWO_BYTES
    # The identity preimage is kept, under its own name, and still owns the hash.
    assert pack["evidence_identity"] == stable_hash(pack["identity_preimage"])
    assert item["license_disposition"]["disposition"] == "not-applicable-no-license-claim"
    assert stable_hash(item["license_disposition"]) == item["license_identity"]
    preimage = item["recommendation_preimage"]
    assert stable_hash(preimage) == item["recommendation_sha256"]
    envelope = _terminal_envelope(item, tmp_path)
    assert envelope["items"][0]["evidence_pack"]["resolution"] == GROUNDED


def test_terminal_envelope_refuses_a_grounded_claim_without_literal_excerpts(
    tmp_path: Path,
) -> None:
    """A pack that claims grounding without inspectable text cannot be dispatched.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=_excerpts())
    item = _terminal_checker_item(artifact)
    # Exactly the shape the envelope used to ship as evidence: the citation
    # table, which carries no excerpt text at all.
    item["evidence_pack"]["excerpts"] = item["evidence_pack"]["identity_preimage"]
    with pytest.raises(CheckerDispatchError, match="no literal excerpt"):
        _terminal_envelope(item, tmp_path)


# -- 4. a genuinely incomplete evidence pack is still rejected -----------------


def test_excerpt_absent_from_its_frozen_source_is_not_evidence(tmp_path: Path) -> None:
    """Text that does not appear in the frozen bytes is never shown as grounded.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    excerpts = _excerpts()
    excerpts[1]["text"] = "activation=mish  # never appears in the frozen source\n"
    artifact = _blocked_artifact(tmp_path, excerpts=excerpts)
    item = _terminal_checker_item(artifact)
    pack = item["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert pack["excerpts"] == []
    assert set(pack["unresolved_evidence_ids"]) == set(DECLARED_EVIDENCE)
    assert "did not re-derive" in str(pack["unresolved_reason"])
    # The envelope still builds -- the gap is stated, not hidden -- so the
    # independent checker sees exactly what it cannot verify.
    shipped = _terminal_envelope(item, tmp_path)["items"][0]["evidence_pack"]
    assert shipped["resolution"] == UNRESOLVED
    assert shipped["excerpts"] == []
    assert shipped["unresolved_reason"]


def test_excerpt_citing_a_source_outside_the_manifest_is_not_evidence(
    tmp_path: Path,
) -> None:
    """An excerpt may only ground against a source the manifest actually froze.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    excerpts = _excerpts()
    excerpts[0]["source_id"] = "source-invented"
    artifact = _blocked_artifact(tmp_path, excerpts=excerpts)
    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert "outside the frozen source manifest" in str(pack["unresolved_reason"])


def test_absent_evidence_records_are_declared_not_invented(tmp_path: Path) -> None:
    """With no frozen records at all the envelope names every unresolved ID.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=None)
    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert pack["excerpts"] == []
    assert set(pack["unresolved_evidence_ids"]) == set(DECLARED_EVIDENCE)
    assert pack["checked_source_ids"] == ["source-1", "source-2"]


def test_partial_evidence_records_are_unresolved(tmp_path: Path) -> None:
    """A pack that omits one excerpt's locator is not grounded.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    excerpts = _excerpts()
    del excerpts[1]["locator"]
    artifact = _blocked_artifact(tmp_path, excerpts=excerpts)
    resolution = resolve_terminal_evidence(
        source_manifest=artifact.source_manifest,
        evidence_ids=DECLARED_EVIDENCE,
        predicate=PREDICATE,
        author_root=artifact.model_dir.parent,
    )
    assert resolution.resolution == UNRESOLVED
    assert "ev-two has no inspectable excerpt record" in str(resolution.reason)


# -- 3. the terminal verdict lockstep, and the prompt that must state it -------


def _validated_terminal_verdict(
    envelope: dict[str, Any],
    *,
    top_level: str,
    integrity: str,
    disposition: str,
) -> dict[str, Any]:
    """Run one complete terminal verdict through the real publication validator.

    Parameters
    ----------
    envelope:
        Terminal request envelope the verdict answers.
    top_level:
        Item ``verdict`` under test.
    integrity:
        Item ``integrity.verdict`` under test.
    disposition:
        ``terminal_disposition.verdict`` under test.

    Returns
    -------
    dict[str, Any]
        Validated gate result.
    """

    verdict = _rejected_verdict_body()
    item = verdict["items"][0]
    for field in (
        "work_id",
        "stable_id",
        "family_representative_id",
        "fidelity_identity",
        "vet_identity",
        "verified_hashes",
    ):
        item[field] = envelope["items"][0][field]
    item["verdict"] = top_level
    item["integrity"]["verdict"] = integrity
    item["terminal_disposition"]["verdict"] = disposition
    # ``make_gate`` supplies fixture placeholders for the machine-owned scaffold.
    # Leaving them would refuse every case at the scaffold comparison, long before
    # the verdict clause under test ever decided -- a probe that proves nothing.
    for machine_field in (*machine_owned_gate_fields(envelope), *LEDGER_ASSIGNED_GATE_FIELDS):
        if machine_field != "schema_version":
            verdict.pop(machine_field, None)
    verdict.pop("checker", None)
    stamped = apply_machine_owned_gate_fields(
        verdict,
        envelope,
        started_at="2026-07-30T00:00:00Z",
        finished_at="2026-07-30T00:05:00Z",
    )
    return validate_checker_result_mapping(stamped, envelope)


def test_terminal_verdict_lockstep_refuses_an_independently_scored_integrity_lane(
    tmp_path: Path,
) -> None:
    """A terminal item's three verdicts move together, and the two live shapes still fail.

    The 2026-07-30 pilot rung lost two of six terminal verdicts to this clause. Neither
    disagreed with the disposition at the TOP level -- both said ``inaccurate`` for a
    ``rejected`` disposition, which is exactly right. Both scored ``integrity`` as its own
    lane the way a metadata item does: once ``accurate`` (nothing was wrong with the
    hashes; the rejection was on the merits) and once ``cannot-verify``. So this pins the
    clause against the shapes that actually occurred, not an invented one.

    Every case differs from the passing control in exactly ONE slot, and the control runs
    first: a guard exercised only where it fails proves nothing, and a negative whose
    compared slots hold the same value would pass against a check that never ran.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=_excerpts())
    envelope = _terminal_envelope(_terminal_checker_item(artifact), tmp_path)

    # The passing control, through the same real entry point the driver uses.
    validated = _validated_terminal_verdict(
        envelope, top_level="inaccurate", integrity="inaccurate", disposition="rejected"
    )
    assert validated["items"][0]["verdict"] == "inaccurate"
    assert validated["items"][0]["integrity"]["verdict"] == "inaccurate"
    assert validated["items"][0]["terminal_disposition"]["verdict"] == "rejected"

    live_shapes = (
        # m_3c3c1e8d404047cb4bcb: clean integrity, rejected on the merits.
        ("inaccurate", "accurate", "rejected"),
        # m10551: integrity scored cannot-verify, still rejected.
        ("inaccurate", "cannot-verify", "rejected"),
        # The dangerous direction, which this clause must keep refusing: an
        # acceptance riding on an integrity lane that says the evidence is bad.
        ("accurate", "inaccurate", "accepted"),
        ("accurate", "cannot-verify", "accepted"),
        # Top-level disagreeing with the disposition outright.
        ("cannot-verify", "cannot-verify", "rejected"),
    )
    for top_level, integrity, disposition in live_shapes:
        # The slots being compared must genuinely differ, or the case would pass
        # against a check that never decided.
        assert (top_level, integrity) != ("inaccurate", "inaccurate")
        with pytest.raises(CheckerDispatchError) as excinfo:
            _validated_terminal_verdict(
                envelope,
                top_level=top_level,
                integrity=integrity,
                disposition=disposition,
            )
        # Exact equality, not a substring: an earlier clause refusing this item
        # for an unrelated reason would otherwise read as a pass.
        assert (
            str(excinfo.value)
            == "terminal top-level/integrity verdicts contradict the disposition"
        )


def test_terminal_verdict_lockstep_is_stated_in_the_frozen_prompt() -> None:
    """The rule the checker is judged by is a rule the checker was told.

    The pilot's two contract rejections were not defiance: the frozen prompt stated the
    metadata worst-of precedence rule and never stated this one, so the checker scored
    ``integrity`` independently because nothing said not to. This reads the mapping the
    check itself uses, so changing ``TERMINAL_VERDICT_LOCKSTEP`` fails here until the
    prompt is re-stated and its PLAN.md digest re-pinned.
    """

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    assert TERMINAL_VERDICT_LOCKSTEP, "an empty lockstep would make every arrow vacuous"
    for disposition, verdict in TERMINAL_VERDICT_LOCKSTEP.items():
        assert f"{disposition} -> {verdict.value}" in prompt
    # The mapping alone does not say the integrity lane moves with it, which is the
    # half the pilot got wrong.
    assert "integrity.verdict" in prompt


def test_machine_derived_envelope_objects_are_disclosed_to_the_checker() -> None:
    """The checker is told which terminal envelope objects it must not read as claims.

    All three of the pilot's recorded rejections rested on the ``identity_preimage``
    rows disagreeing with the literal excerpts about ``source_id``, read as forged
    evidence bindings. That table is machine-derived by ``derive_terminal_evidence_pack``
    -- the author cannot influence it -- and the disagreement is an artifact of the
    round-robin pairing, so the prompt now says so. ``source_to_code_map`` is the same
    class of defect: it is the hash of ``checked_source_ids``, and four of six calls
    reported it as an unverifiable digest with no frozen artifact.
    """

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    for disclosed in (
        "identity_preimage",
        "ROUND-ROBIN INDEX",
        "checked_source_ids",
        "source_to_code_map",
    ):
        assert disclosed in prompt

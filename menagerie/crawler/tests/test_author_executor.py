"""Author executor: headless round trips, machine effort, typed classification."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from menagerie.crawler.author_attempts import (
    latest_attempt,
    list_attempts,
    new_attempt,
    record_checker_findings,
)
from menagerie.crawler.author_dispatch import (
    AuthorEffortGrant,
    AuthorPauseReason,
    BlockedRecommendation,
    SkipRecommendation,
    _validate_author_result_mapping,
    classify_author_response,
    plausible_author_reset_at,
)
from menagerie.crawler.author_executor import (
    EXA_API_KEY_ENV,
    EXIT_BACKOFF,
    EXIT_OK,
    EXIT_PERMANENT,
    EXIT_RETRYABLE,
    RECEIPT_VERSION,
    SUPPLEMENT_VERSION,
    AuthorExecutorError,
    _author_result_from_author_payload,
    _stamp_machine_owned_proposal_fields,
    _discovery_envelope_from_author_payload,
    _supplement_request_from_author_payload,
    main,
    structured_limit_reset_at,
    structured_limit_signal,
)
from menagerie.crawler.driver_progress import _normalize_wake_reset
from menagerie.crawler.wakeup import (
    MIN_RETRY_INTERVAL_SECONDS,
    build_wake_episode,
)
from menagerie.crawler.capability_probe import canonical_tool_name
from menagerie.crawler.constants import (
    ACCESS_BLOCKED_REASON_CODE,
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_RESULT_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION_V3,
    EnvironmentPhase,
)
from menagerie.crawler.discovery import (
    AccessBlockedDiscovery,
    FoundDiscovery,
    HigherTierDiscovery,
    NegativeDiscovery,
    RetryableToolFailureDiscovery,
    SourceDiscovery,
    materialize_discovery_artifact,
    validate_source_discovery,
)
from menagerie.crawler.driver_admission import (
    CommandAuthorLane,
    DriverIntegrationError,
    _validate_artifact_identities,
    _verify_executor_receipt,
)
from menagerie.crawler.driver_contracts import AuthorArtifact, DriverConfig, WorkItem
from menagerie.crawler.driver_models import _terminal_checker_item
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.metadata import recompute_accepted_identities
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.source_broker import TransportResponse
from menagerie.crawler.tests.executor_test_support import (
    AUTHOR_IDENTITY_INPUTS,
    COMMITS_URL,
    DEFAULT_DISCOVERY,
    FABRICATED_SHA,
    RESOLVED_SHA,
    executor_environment,
    read_invocations,
    write_author_envelope,
    write_broker_fixtures,
    write_fake_claude,
    write_source_request,
)
from menagerie.crawler.tests.conftest import (
    HASH,
    make_author_proposal,
    make_authority_context,
    make_proposed_artifact,
)


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One configured executor rig: fake harness, fixtures, log, author root."""

    fake = write_fake_claude(tmp_path / "bin")
    fixtures = write_broker_fixtures(tmp_path / "fixtures")
    log_dir = tmp_path / "log"
    executor_environment(
        monkeypatch, fake_claude=fake, fixtures=fixtures, log_dir=log_dir
    )
    return {
        "root": tmp_path / "work" / "m1" / "author",
        "log": log_dir,
        "monkeypatch": monkeypatch,
        "tmp": tmp_path,
        "fixtures": fixtures,
    }


def _run_source_round(rig, stable_id: str = "m1") -> tuple[int, Path]:
    root = rig["root"]
    request = write_source_request(root, stable_id)
    return main([str(request)]), root


def _run_author_round(rig, stable_id: str = "m1") -> tuple[int, Path]:
    root = rig["root"]
    request = write_author_envelope(root, stable_id)
    return main([str(request)]), root


def _prompt_contract_fixture(path: Path, marker: str) -> dict[str, Any]:
    """Read the JSON fixture immediately following one prompt marker.

    Parameters
    ----------
    path:
        Prompt file containing the contract fixture.
    marker:
        Exact fixture marker name.

    Returns
    -------
    dict[str, Any]
        Parsed fixture object.
    """

    prompt = path.read_text(encoding="utf-8")
    marker_text = f"<!-- CONTRACT_FIXTURE: {marker} -->"
    marked = prompt.split(marker_text, maxsplit=1)[1]
    fenced = marked.split("```json", maxsplit=1)[1].split("```", maxsplit=1)[0]
    fixture = json.loads(fenced)
    assert isinstance(fixture, dict)
    return fixture


def _prompt_contract_markers(path: Path) -> tuple[str, ...]:
    """Return every contract-fixture marker a prompt ships, in prompt order.

    Parameters
    ----------
    path:
        Prompt file to scan.

    Returns
    -------
    tuple[str, ...]
        Exact marker names.
    """

    return tuple(
        re.findall(r"<!-- CONTRACT_FIXTURE: (\S+) -->", path.read_text(encoding="utf-8"))
    )


#: Exactly which typed stage-1 arm each shipped discovery fixture must parse to. The
#: mapping is total over the prompt's markers (asserted below), so a newly taught arm
#: cannot ship uncovered -- which is how the ``NEEDS_SOURCE_ACCESS`` fixture reached the
#: prompt without any test ever reading it.
_STAGE1_FIXTURE_ARMS: dict[str, type[SourceDiscovery]] = {
    "stage1-author-payload": FoundDiscovery,
    "stage1-no-usable-source-author-payload": NegativeDiscovery,
    "stage1-insufficient-description-author-payload": NegativeDiscovery,
    "stage1-not-a-model-author-payload": NegativeDiscovery,
    "stage1-needs-higher-tier-author-payload": HigherTierDiscovery,
    "stage1-needs-source-access-author-payload": AccessBlockedDiscovery,
    "stage1-retryable-tool-failure-author-payload": RetryableToolFailureDiscovery,
}

#: The terminal each materializable fixture is destined for. ``FOUND`` enters controlled
#: fetch instead and ``RETRYABLE_TOOL_FAILURE`` retries, so neither is a
#: ``NonFetchDiscovery`` and neither has a terminal to materialize.
_STAGE1_FIXTURE_TERMINALS: dict[str, str] = {
    "stage1-no-usable-source-author-payload": "skipped:no-description",
    "stage1-insufficient-description-author-payload": "skipped:insufficient-description",
    "stage1-not-a-model-author-payload": "skipped:not-a-real-NN",
    "stage1-needs-higher-tier-author-payload": "needs-higher-tier",
    "stage1-needs-source-access-author-payload": ACCESS_BLOCKED_REASON_CODE,
}


def _fixture_work_item(stable_id: str) -> WorkItem:
    """Return the minimal routed work item the discovery materializer reads."""

    return WorkItem(
        intake=IntakeItem(
            stable_id=stable_id,
            name="ExampleNet",
            zoo="crawler",
            variant="base",
            discovery_source="crawl_roster",
            legacy_row_sha256="0" * 64,
            preserved_legacy_flags=(),
            variant_scope="standalone",
            family_representative_id=stable_id,
        ),
        route=IntentRoute(
            stable_id=stable_id, intent="core", phase=EnvironmentPhase.PYTORCH
        ),
    )


def _refusing_probe_transport(
    url: str, *, max_bytes: int, timeout: float
) -> TransportResponse:
    """Return a deterministic hermetic refusal for any probed candidate locator."""

    del max_bytes, timeout
    return TransportResponse(
        status=403,
        final_url=url,
        redirect_chain=(url,),
        body=b"",
        truncated=False,
        error="fixture candidate refused",
    )


#: Proposal keys the executor now derives, so an author is never asked for them.
_MACHINE_OWNED_PROPOSAL_KEYS = (
    "schema_version",
    "proposal_sha256",
    "campaign_id",
    "stable_id",
    "work_id",
    "intake_snapshot_id",
    "intake_snapshot_sha256",
    "intake_item_sha256",
    "source_manifest_identity",
    "dispatcher_identity",
)


def _author_owned_proposal(stable_id: str) -> dict[str, Any]:
    """Return the shared proposal fixture stripped to what an author really owns.

    ``make_author_proposal`` fills every binding with a fixture placeholder --
    ``campaign-m-fixture``, an all-``a`` digest -- which is exactly the shape a
    model templating its answer from repository test data would produce. Nothing
    an author must transcribe is left in, so the round trip proves the executor
    supplies each one from the request rather than believing a supplied copy.

    Parameters
    ----------
    stable_id:
        Proposed model identity.

    Returns
    -------
    dict[str, Any]
        Proposal carrying only author-owned judgment.
    """

    proposal = make_author_proposal(stable_id)
    for key in _MACHINE_OWNED_PROPOSAL_KEYS:
        proposal.pop(key, None)
    proposal["proposed_facts"]["modes"].pop("per_mode_run", None)
    return proposal


def _assert_machine_stamped_proposal(
    proposal: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Assert every machine-owned proposal leaf came from the machine.

    Parameters
    ----------
    proposal:
        Materialized proposal from a stage-2 round trip.
    expected:
        Trusted ``expected_result`` bindings from the request envelope.
    """

    assert proposal["schema_version"] == AUTHOR_PROPOSAL_SCHEMA_VERSION_V3
    for key in (
        "campaign_id",
        "stable_id",
        "work_id",
        "intake_snapshot_id",
        "intake_snapshot_sha256",
        "intake_item_sha256",
        "source_manifest_identity",
        "dispatcher_identity",
    ):
        assert proposal[key] == expected[key], key
    assert proposal["proposed_facts"]["modes"]["per_mode_run"] == {}
    # The whole-object digest binds the STAMPED proposal, which is what the
    # driver re-derives and compares after publication.
    assert proposal["proposal_sha256"] == stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )


def test_prompt_contract_fixtures_materialize_against_registered_schemas(
    tmp_path: Path,
) -> None:
    """Prompt-taught inner shapes stay coupled to executor wrappers and schemas.

    Envelope validation alone is not what this test's name promises, and the gap was
    not theoretical: the insufficient-description fixture shipped a plain-``http://``
    locator that ``validate_source_discovery`` happily accepted while the
    materialization it was destined for threw, because the broker refuses a non-HTTPS
    descriptor for the whole batch. So every fixture that names a terminal is now
    carried all the way through ``materialize_discovery_artifact`` -- the same call the
    production author lane makes -- and its terminal is asserted.

    ``FOUND`` and ``RETRYABLE_TOOL_FAILURE`` stop at the typed arm on purpose: neither is
    a ``NonFetchDiscovery``, so neither has a terminal to materialize. ``FOUND`` enters
    controlled fetch (covered end-to-end by the executor round trips in this module) and
    ``RETRYABLE_TOOL_FAILURE`` retries rather than terminalizing.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory used as the per-fixture author custody root.
    """

    prompt_root = Path(__file__).parents[1] / "prompts" / "executor"
    request = {
        "stable_id": "m-fixture",
        "work_id": "work-m-fixture",
        "identity_inputs": deepcopy(AUTHOR_IDENTITY_INPUTS),
        "expected_result": {
            "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
            "stable_id": "m-fixture",
            "work_id": "work-m-fixture",
            "campaign_id": "campaign-fixture",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": "sha256:" + "6" * 64,
            "intake_snapshot_id": "intake-fixture",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "source_manifest": {
            "manifest_sha256": "sha256:" + "6" * 64,
            "sources": [{"source_id": "impl-main"}],
        },
    }

    discovery_prompt = prompt_root / "stage1_discovery.md"
    assert set(_prompt_contract_markers(discovery_prompt)) == set(_STAGE1_FIXTURE_ARMS)
    assert set(_STAGE1_FIXTURE_TERMINALS) <= set(_STAGE1_FIXTURE_ARMS)

    item = _fixture_work_item("m-fixture")
    context = make_authority_context(["m-fixture"])
    for marker, expected_arm in _STAGE1_FIXTURE_ARMS.items():
        discovery_payload = _prompt_contract_fixture(discovery_prompt, marker)
        discovery_envelope = _discovery_envelope_from_author_payload(
            discovery_payload,
            request,
        )
        discovery = validate_source_discovery(
            discovery_envelope,
            stable_id="m-fixture",
            work_id="work-m-fixture",
        )
        assert isinstance(discovery, expected_arm)
        terminal = _STAGE1_FIXTURE_TERMINALS.get(marker)
        if not isinstance(
            discovery, (NegativeDiscovery, HigherTierDiscovery, AccessBlockedDiscovery)
        ):
            assert terminal is None
            continue
        assert terminal is not None
        artifact = materialize_discovery_artifact(
            discovery,
            item=item,
            context=context,
            root=tmp_path / marker,
            probe_transport=_refusing_probe_transport,
        )
        terminal_result = artifact.author_result
        if isinstance(discovery, NegativeDiscovery):
            assert isinstance(terminal_result, SkipRecommendation)
            assert terminal_result.status_code == terminal
        else:
            assert isinstance(terminal_result, BlockedRecommendation)
            assert terminal_result.reason_code == terminal

    supplement_payload = _prompt_contract_fixture(
        prompt_root / "stage2_author.md",
        "supplement-author-payload",
    )
    supplement = _supplement_request_from_author_payload(supplement_payload, request)
    assert supplement["supplement_version"] == SUPPLEMENT_VERSION

    result_payload = _prompt_contract_fixture(
        prompt_root / "stage2_author.md",
        "stage2-author-payload",
    )
    result = _author_result_from_author_payload(result_payload, request)
    validate_payload(result, AUTHOR_RESULT_SCHEMA_VERSION)

    proposed_payload = _prompt_contract_fixture(
        prompt_root / "stage2_author.md",
        "stage2-proposed-author-payload",
    )
    proposed_fixture = _author_owned_proposal("m-fixture")
    proposed_payload["payload"]["proposal"] = proposed_fixture
    proposed_result = _author_result_from_author_payload(proposed_payload, request)
    validate_payload(proposed_result, AUTHOR_RESULT_SCHEMA_VERSION)
    expected_bindings = cast("Mapping[str, Any]", request["expected_result"])
    _assert_machine_stamped_proposal(proposed_result["payload"]["proposal"], expected_bindings)

    defer_payload = _prompt_contract_fixture(
        prompt_root / "stage2_author.md",
        "stage2-defer-author-payload",
    )
    defer_fixture = _author_owned_proposal("m-fixture")
    defer_fixture["proposed_facts"]["implementation"]["code_manifest"] = []
    defer_payload["payload"]["handoff_execution"]["proposal"] = defer_fixture
    defer_result = _author_result_from_author_payload(defer_payload, request)
    validate_payload(defer_result, AUTHOR_RESULT_SCHEMA_VERSION)
    handoff = defer_result["payload"]["handoff_execution"]
    _assert_machine_stamped_proposal(handoff["proposal"], expected_bindings)
    assert handoff["proposal_sha256"] == handoff["proposal"]["proposal_sha256"]

    skip_payload = _prompt_contract_fixture(
        prompt_root / "stage2_author.md",
        "stage2-skip-author-payload",
    )
    skip_result = _author_result_from_author_payload(skip_payload, request)
    validate_payload(skip_result, AUTHOR_RESULT_SCHEMA_VERSION)


def test_blocked_without_author_identities_materializes_machine_facts(
    tmp_path: Path,
) -> None:
    """BLOCKED transport omits hashes that only the executor can derive."""

    source_manifest_identity = "sha256:" + "6" * 64
    source_manifest: dict[str, Any] = {
        "manifest_sha256": source_manifest_identity,
        "sources": [{"source_id": "impl-main"}],
    }
    request = {
        "identity_inputs": deepcopy(AUTHOR_IDENTITY_INPUTS),
        "expected_result": {
            "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
            "stable_id": "m-blocked",
            "work_id": "work-m-blocked",
            "campaign_id": "campaign-blocked",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": source_manifest_identity,
            "intake_snapshot_id": "intake-blocked",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "source_manifest": source_manifest,
    }
    authored = {
        "kind": "BLOCKED",
        "payload": {
            "stage": "environment",
            "reason_code": "missing-runtime-dependency",
            "prerequisite_ids": ["runtime-dependency"],
            "evidence_ids": [],
        },
    }

    result = _author_result_from_author_payload(authored, request)

    validate_payload(result, AUTHOR_RESULT_SCHEMA_VERSION)
    assert result["payload"]["evidence_identity"] == stable_hash([])
    assert result["payload"]["license_identity"] == stable_hash(
        {
            "arm": "BLOCKED",
            "disposition": "not-applicable-no-license-claim",
            "source_manifest_identity": source_manifest_identity,
        }
    )
    envelope = {
        "envelope_version": "menagerie.crawler.author-envelope.v3",
        "expected_result": request["expected_result"],
        "source_manifest": source_manifest,
        "allowed_model_dir": str(tmp_path),
        "required_output_path": str(tmp_path / "result.json"),
    }
    envelope["envelope_sha256"] = stable_hash(envelope)
    parsed = _validate_author_result_mapping(result, envelope, cas_root=None)
    assert isinstance(parsed, BlockedRecommendation)
    pack = _terminal_checker_item(AuthorArtifact(parsed, source_manifest, tmp_path))
    assert pack["evidence_pack"]["evidence_identity"] == result["payload"][
        "evidence_identity"
    ]
    assert stable_hash(pack["license_disposition"]) == result["payload"][
        "license_identity"
    ]


def test_defer_without_author_identities_materializes_proposal_facts() -> None:
    """DEFER identities bind the retained proposal evidence and licenses."""

    request = {
        "identity_inputs": deepcopy(AUTHOR_IDENTITY_INPUTS),
        "expected_result": {
            "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
            "stable_id": "m-fixture",
            "work_id": "work-m-fixture",
            "campaign_id": "campaign-fixture",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": "sha256:" + "6" * 64,
            "intake_snapshot_id": "intake-fixture",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "source_manifest": {
            "manifest_sha256": "sha256:" + "6" * 64,
            "sources": [{"source_id": "source-1"}],
        },
    }
    proposal = _author_owned_proposal("m-fixture")
    proposal["proposed_facts"]["implementation"]["code_manifest"] = []
    licenses = proposal["proposed_facts"]["licenses"]
    authored = {
        "kind": "DEFER_RECOMMENDATION",
        "payload": {
            "platform": "cuda",
            "source_ids": ["source-1"],
            "evidence_ids": ["evidence-1"],
            "handoff_execution": {"proposal": proposal},
        },
    }

    result = _author_result_from_author_payload(authored, request)

    validate_payload(result, AUTHOR_RESULT_SCHEMA_VERSION)
    assert result["payload"]["evidence_identity"] == stable_hash(
        [
            {
                "evidence_id": "evidence-1",
                "source_id": "source-1",
                "supports": ["needs-cuda"],
            }
        ]
    )
    assert result["payload"]["license_identity"] == stable_hash(licenses)


def _proposed_request(stable_id: str = "m-fixture") -> dict[str, Any]:
    """Return one trusted author request for proposal round trips.

    Parameters
    ----------
    stable_id:
        Proposed model identity.

    Returns
    -------
    dict[str, Any]
        Request envelope carrying the machine's own expected bindings.
    """

    return {
        "identity_inputs": deepcopy(AUTHOR_IDENTITY_INPUTS),
        "expected_result": {
            "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
            "stable_id": stable_id,
            "work_id": f"work-{stable_id}",
            "campaign_id": "campaign-fixture",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": "sha256:" + "6" * 64,
            "intake_snapshot_id": "intake-fixture",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "source_manifest": {
            "manifest_sha256": "sha256:" + "6" * 64,
            "sources": [{"source_id": "source-1"}],
        },
    }


def test_proposal_omitting_machine_owned_identities_still_materializes() -> None:
    """The four live 2026-07-30 ``result-contract-invalid`` rejections are gone.

    Every one of them named a field the machine already held or could compute
    from what the author wrote: ``proposal_sha256`` (a self-hash),
    ``excerpts[].text_sha256`` (a self-hash of the sibling text, which the
    terminal ``evidence_records`` channel already refuses to accept from an
    author), and ``modes.per_mode_run`` (attempts that do not exist yet).
    Omitting all of them is now free.
    """

    request = _proposed_request()
    proposal = _author_owned_proposal("m-fixture")
    for key in _MACHINE_OWNED_PROPOSAL_KEYS:
        assert key not in proposal
    assert "per_mode_run" not in proposal["proposed_facts"]["modes"]

    result = _author_result_from_author_payload(
        {"kind": "PROPOSED", "payload": {"proposal": proposal}}, request
    )

    validate_payload(result, AUTHOR_RESULT_SCHEMA_VERSION)
    _assert_machine_stamped_proposal(result["payload"]["proposal"], request["expected_result"])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            lambda proposal: proposal.update({"campaign_id": "campaign-somewhere-else"}),
            "machine-owned proposal.campaign_id",
            id="binding-conflict",
        ),
        pytest.param(
            lambda proposal: proposal.update(
                {"source_manifest_identity": "sha256:" + "a" * 64}
            ),
            "machine-owned proposal.source_manifest_identity",
            id="fixture-placeholder-digest",
        ),
        pytest.param(
            lambda proposal: proposal["proposed_facts"]["modes"].update(
                {"per_mode_run": {"eval": {"attempt_id": "attempt-1", "status": "succeeded"}}}
            ),
            "no attempt has run",
            id="fabricated-per-mode-run",
        ),
    ],
)
def test_machine_owned_proposal_conflicts_are_refused_not_overwritten(
    mutate: Any, match: str
) -> None:
    """A conflicting authored value stays visible instead of being laundered.

    This is the direction the checker lane's unconditional ``stamped.update``
    got wrong: overwriting made a fabricated identity harmless AND invisible,
    and left the downstream comparison checking the machine against itself.
    Stamping here refuses first, so a proposal templated from repository test
    data -- an all-``a`` placeholder digest, a foreign campaign, a claimed run
    that never happened -- fails loudly at the boundary that noticed it.

    Parameters
    ----------
    mutate:
        Applies one conflicting authored value to the proposal.
    match:
        Substring the typed refusal must name.
    """

    proposal = _author_owned_proposal("m-fixture")
    mutate(proposal)

    with pytest.raises(AuthorExecutorError, match=match):
        _author_result_from_author_payload(
            {"kind": "PROPOSED", "payload": {"proposal": proposal}}, _proposed_request()
        )


def test_stamped_proposal_survives_driver_side_identity_recomputation() -> None:
    """Stamping must not disturb the identities the driver re-derives and compares.

    ``driver_admission`` recomputes ``source_identity``, ``evidence_identity``,
    ``recipe_revision``, ``vet_identity``, and ``fidelity_identity`` from the
    declared facts and refuses a mismatch. Those five are deliberately NOT
    stamped -- they are the live cross-check against a proposal whose claimed
    identity its own facts do not produce. This proves the fields that ARE
    stamped leave that recomputation exactly where it was, so filling in a
    machine-owned omission can never turn a good proposal into an identity
    mismatch downstream.
    """

    request = _proposed_request()
    proposal = _author_owned_proposal("m-fixture")
    before = recompute_accepted_identities(
        proposal["proposed_facts"],
        checker_prompt_hash="sha256:" + "9" * 64,
        checker_model="checker-model",
        checker_version="checker-version",
        schema_version=MODEL_SCHEMA_VERSION_V3,
    )

    result = _author_result_from_author_payload(
        {"kind": "PROPOSED", "payload": {"proposal": proposal}}, request
    )
    after = recompute_accepted_identities(
        result["payload"]["proposal"]["proposed_facts"],
        checker_prompt_hash="sha256:" + "9" * 64,
        checker_model="checker-model",
        checker_version="checker-version",
        schema_version=MODEL_SCHEMA_VERSION_V3,
    )

    assert after == before

    # The assertion above only bites while stamping leaves the evidence
    # projection alone, and a proposal that already carries its digests would
    # not notice a stamp being re-added. So check the decision directly: given
    # excerpts with no digest at all, stamping must not invent one.
    # ``compute_evidence_identity`` projects ``text_sha256``, so filling it in
    # here would move every evidence identity the driver recomputes.
    undigested = _author_owned_proposal("m-fixture")
    for excerpt in undigested["proposed_facts"]["evidence"]["excerpts"]:
        excerpt.pop("text_sha256", None)
    stamped = _stamp_machine_owned_proposal_fields(
        undigested,
        request["expected_result"],
        author_binding=cast("Mapping[str, Any]", AUTHOR_IDENTITY_INPUTS["author"]),
    )

    assert all(
        "text_sha256" not in excerpt
        for excerpt in stamped["proposed_facts"]["evidence"]["excerpts"]
    )


def test_author_identity_fields_cannot_override_machine_derivation() -> None:
    """Authored identity assertions are rejected instead of trusted or ignored."""

    request = {
        "identity_inputs": deepcopy(AUTHOR_IDENTITY_INPUTS),
        "expected_result": {
            "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
            "stable_id": "m-blocked",
            "work_id": "work-m-blocked",
            "campaign_id": "campaign-blocked",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": "sha256:" + "6" * 64,
            "intake_snapshot_id": "intake-blocked",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "source_manifest": {
            "manifest_sha256": "sha256:" + "6" * 64,
            "sources": [{"source_id": "impl-main"}],
        },
    }
    authored = {
        "kind": "BLOCKED",
        "payload": {
            "stage": "environment",
            "reason_code": "missing-runtime-dependency",
            "prerequisite_ids": ["runtime-dependency"],
            "evidence_ids": [],
            "evidence_identity": "sha256:" + "a" * 64,
            "license_identity": "sha256:" + "b" * 64,
        },
    }

    with pytest.raises(AuthorExecutorError, match="machine-owned"):
        _author_result_from_author_payload(authored, request)


def test_blocked_prerequisites_are_semantic_ids_not_schema_paths() -> None:
    """BLOCKED prerequisites name external needs, never omitted output fields."""

    request = {
        "identity_inputs": deepcopy(AUTHOR_IDENTITY_INPUTS),
        "expected_result": {
            "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
            "stable_id": "m-blocked",
            "work_id": "work-m-blocked",
            "campaign_id": "campaign-blocked",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": "sha256:" + "6" * 64,
            "intake_snapshot_id": "intake-blocked",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "source_manifest": {
            "manifest_sha256": "sha256:" + "6" * 64,
            "sources": [{"source_id": "impl-main"}],
        },
    }
    authored = {
        "kind": "BLOCKED",
        "payload": {
            "stage": "environment",
            "reason_code": "missing-runtime-dependency",
            "prerequisite_ids": ["payload.evidence_identity"],
            "evidence_ids": [],
        },
    }

    with pytest.raises(AuthorExecutorError, match="validation failed"):
        _author_result_from_author_payload(authored, request)


def test_needs_higher_tier_accepts_observed_http_candidate_locator() -> None:
    """Research summaries retain observed HTTP candidate locators verbatim."""

    payload = {
        "arm": "NEEDS_HIGHER_TIER",
        "research_summary": {
            "queries": ["RENet temporal knowledge graph"],
            "places": ["Author project page"],
            "candidate_links": [
                {
                    "url": "http://inklab.usc.edu/renet/",
                    "why_rejected": "Useful project context, but not executable source.",
                    "rejection_class": "not-this-model",
                }
            ],
            "languages": ["English"],
            "conclusion": "The exact upstream implementation needs a higher-tier audit.",
        },
    }
    request = {"stable_id": "m11695", "work_id": "work-m11695"}

    envelope = _discovery_envelope_from_author_payload(payload, request)

    validate_source_discovery(
        envelope,
        stable_id="m11695",
        work_id="work-m11695",
    )


def test_source_round_publishes_machine_derived_pack(rig, capsys) -> None:
    """Stage 1 + broker publish a pack whose exact strings are machine-derived."""

    code, root = _run_source_round(rig)
    assert code == EXIT_OK
    published = json.loads((root / "source-targets.json").read_text(encoding="utf-8"))
    row = published["sources"][0]
    assert row["revision"] == RESOLVED_SHA
    assert row["expected_sha256"].startswith("sha256:")
    # The receipt on stdout is attempt-bound and digest-matches the bytes.
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["receipt_version"] == RECEIPT_VERSION
    assert receipt["result_sha256"] == hash_bytes(
        (root / "source-targets.json").read_bytes()
    )
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.status == "sources-published"
    assert receipt["attempt_nonce"] == attempt.nonce


def test_source_round_retains_authored_architecture_observations(rig) -> None:
    """The live m10551 descriptor shape validates and carries observations forward."""

    notes = (
        "Observed forward definition: five DarknetConv2D_BN_Leaky(16,32,64,128,256) "
        "3x3 stages each followed by MaxPooling2D(2,2) stride 2 'same'; then "
        "MaxPooling2D stride 2, conv 512 3x3. LeakyReLU alpha=0.1 throughout, conv "
        "bias disabled under BatchNormalization."
    )
    discovery = {
        "arm": "FOUND",
        "sources": [
            {
                **cast(dict[str, Any], DEFAULT_DISCOVERY["sources"][0]),
                "notes": notes,
            }
        ],
    }
    rig["monkeypatch"].setenv("FAKE_CLAUDE_DISCOVERY", json.dumps(discovery))

    code, root = _run_source_round(rig)

    assert code == EXIT_OK
    published = json.loads((root / "source-targets.json").read_text(encoding="utf-8"))
    assert published["sources"][0]["notes"] == notes


def test_fabricated_sha_ref_is_bad_ref_before_publication(rig) -> None:
    """A plausible authored ref is dereferenced and cannot create a manifest row."""

    fabricated_sha = FABRICATED_SHA
    rig["monkeypatch"].setenv(
        "FAKE_CLAUDE_DISCOVERY",
        json.dumps(
            {
                "arm": "FOUND",
                "sources": [
                    {
                        "source_id": "impl-fabricated",
                        "kind": "forge-file",
                        "repo": "github.com/acme/widgets",
                        "path": "models/net.py",
                        "ref": fabricated_sha,
                        "requested_role": "implementation",
                        "basis": "A locator that still requires machine dereference.",
                    }
                ],
            }
        ),
    )

    code, root = _run_source_round(rig)

    assert code == EXIT_RETRYABLE
    assert not (root / "source-targets.json").exists()
    attempt = latest_attempt(root)
    assert attempt is not None
    receipts = json.loads((attempt.paths.broker / "receipts.json").read_text("utf-8"))
    assert receipts["sources"] == []
    assert receipts["broker"]["outcomes"][0]["outcome"] == "bad-ref"


def test_a_throttled_forge_is_not_reported_as_an_unfetchable_implementation(rig) -> None:
    """The whole point, end to end: nobody looked, so nothing may be said about the ref.

    ``primary-implementation-unfetchable`` asserts the implementation could not be
    found. Under a rate limit the forge never evaluated the request at all, so that
    reason is false and the retry it triggers goes to repair a reference that was
    never wrong. It also has to be a *distinct* reason, because the driver promotes
    a sustained streak of it to a campaign pause.
    """

    fixtures = Path(rig["fixtures"])
    index = json.loads((fixtures / "index.json").read_text(encoding="utf-8"))
    index[COMMITS_URL] = {
        "status": 403,
        "body_text": json.dumps({"message": "API rate limit exceeded"}),
        "headers": {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1785000000"},
    }
    (fixtures / "index.json").write_text(json.dumps(index), encoding="utf-8")

    code, root = _run_source_round(rig)

    assert code == EXIT_RETRYABLE
    attempt = latest_attempt(root)
    assert attempt is not None
    outcome = attempt.record["outcome"]
    assert outcome["failure_reason"] == "forge-rate-limited"
    assert outcome["failure_reason"] != "primary-implementation-unfetchable"
    assert outcome["detail"]["rate_limit_reset_epoch"] == 1785000000
    receipts = json.loads((attempt.paths.broker / "receipts.json").read_text("utf-8"))
    assert receipts["broker"]["outcomes"][0]["outcome"] == "rate-limited"


def test_author_cannot_supply_discovery_envelope_bindings(rig) -> None:
    """The executor rejects authored identities instead of trusting their echo."""

    discovery = json.loads(json.dumps(DEFAULT_DISCOVERY))
    discovery["stable_id"] = "different-model"
    discovery["work_id"] = "different-work"
    rig["monkeypatch"].setenv("FAKE_CLAUDE_DISCOVERY", json.dumps(discovery))

    code, root = _run_source_round(rig)

    assert code == EXIT_RETRYABLE
    assert not (root / "source-targets.json").exists()
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.record["outcome"]["failure_reason"] == "discovery-contract-invalid"
    error = attempt.record["outcome"]["detail"]["error"]
    # Pin the refusal MECHANISM, not the prose. The rejection must be the closed-schema
    # additionalProperties refusal naming both smuggled bindings -- an error that merely
    # mentions one of them could be some unrelated validation failure that happens to quote
    # the field.
    assert "Additional properties are not allowed" in error
    assert "stable_id" in error and "work_id" in error
    # The author's proposed VALUES must never be echoed back into the record. Quoting
    # attacker-controlled text into a stored error is its own defect, and an assertion that
    # DEMANDS the echo would entrench it -- which is what this assertion previously did.
    assert "different-model" not in error
    assert "different-work" not in error


def test_pinned_recipe_flags_are_load_bearing_and_present(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every flag of the verified web-tools recipe is on the session argv.

    The Exa credential is cleared first so this asserts the recipe's SHAPE rather than
    whatever happens to be in the ambient environment. Without this the assertion embeds a
    live secret in its failure output -- which is exactly how one got printed into a
    transcript. Authenticated-vs-anonymous endpoint selection is covered separately, with a
    sentinel, in ``test_exa_credential.py``.
    """

    monkeypatch.delenv(EXA_API_KEY_ENV, raising=False)
    code, _root = _run_source_round(rig)
    assert code == EXIT_OK
    argv = read_invocations(rig["log"])[0]["argv"]
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--mcp-config" in argv
    assert json.loads(argv[argv.index("--mcp-config") + 1]) == {
        "mcpServers": {"exa": {"type": "http", "url": "https://mcp.exa.ai/mcp"}}
    }
    tools_at = argv.index("--allowedTools")
    tools = argv[tools_at + 1 : argv.index("--output-format")]
    assert "WebSearch" in tools
    assert "mcp__exa__web_search_exa" in tools
    assert "mcp__exa__web_fetch_exa" in tools
    assert "ToolSearch" in tools
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--session-id" in argv


def test_author_round_resumes_the_stage1_session(rig) -> None:
    """Stage 2 runs with ``--resume <stage-1 session>`` -- no cold reread."""

    code, root = _run_source_round(rig)
    assert code == EXIT_OK
    stage1 = latest_attempt(root)
    assert stage1 is not None
    stage1_session = stage1.record["stage1"]["session_id"]

    code, root = _run_author_round(rig)
    assert code == EXIT_OK
    assert (root / "result.json").is_file()
    stage2_call = read_invocations(rig["log"])[1]
    assert stage2_call["stage"] == "stage2"
    argv = stage2_call["argv"]
    assert argv[argv.index("--resume") + 1] == stage1_session
    schema_root = Path(__file__).parents[1] / "schemas"
    assert f"- PROPOSAL schema, exact: `{schema_root / 'author-proposal-v3.schema.json'}`" in (
        stage2_call["prompt"]
    )
    tools = argv[argv.index("--allowedTools") + 1 : argv.index("--output-format")]
    assert f"Read(/{schema_root}/**)" in tools
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.status == "completed"
    assert attempt.record["stage2"]["resumed_from"] == stage1_session


def test_effort_records_match_the_harness_json(rig) -> None:
    """Effort is machine-counted from the harness JSON, not self-reported."""

    _run_source_round(rig)
    _run_author_round(rig)
    attempt = latest_attempt(rig["root"])
    assert attempt is not None
    for stage in ("stage1", "stage2"):
        effort = attempt.record[stage]["effort"]
        # These exact values are what the fake harness printed as its JSON.
        assert effort["duration_ms"] == 1234
        assert effort["duration_api_ms"] == 987
        assert effort["num_turns"] == 7
        assert effort["total_cost_usd"] == 0.0123
        assert effort["usage"] == {
            "input_tokens": 111,
            "output_tokens": 222,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 20,
        }
        assert effort["timed_out"] is False
        assert effort["wall_seconds_observed"] > 0


def test_structured_limit_signal_is_the_only_pause_authority(rig) -> None:
    """A structured harness limit is exit 76; free-text noise never is."""

    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "limit")
    code, _root = _run_source_round(rig)
    assert code == EXIT_BACKOFF


# -- positive limit recognition -------------------------------------------
#
# These are the direction the suite was missing. It previously proved only that
# a limit is *not* falsely detected; nothing proved a real one IS caught, and the
# constants being matched (``usage_limit``, ``error_usage_limit``, ...) existed
# nowhere in the shipped harness, so the campaign pause could never fire. Every
# payload replayed below is a real Claude Code result shape; see
# ``_LIMIT_TERMINAL_REASONS`` in ``author_executor.py`` for per-field citations.


@pytest.mark.parametrize(
    "shape",
    ["blocking_limit", "rapid_refill_breaker", "rate_limit_info_only", "api_error_status"],
)
def test_real_harness_limit_payloads_pause_the_campaign(rig, shape: str) -> None:
    """Each observed limit envelope is recognised and exits 76."""

    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "limit")
    rig["monkeypatch"].setenv("FAKE_CLAUDE_LIMIT_SHAPE", shape)
    code, _root = _run_source_round(rig)
    assert code == EXIT_BACKOFF


@pytest.mark.parametrize("window", ["five_hour", "seven_day"])
def test_five_hour_and_weekly_limits_both_pause_and_carry_their_reset(
    rig, window: str, capsys
) -> None:
    """Both limit windows pause, and the harness's own reset reaches the driver.

    The seven-day case is the expensive one: without the declared reset the
    driver guesses ``now + 1h`` and re-wakes for a week.
    """

    ahead = timedelta(hours=5) if window == "five_hour" else timedelta(days=7)
    resets_at = datetime.now(timezone.utc) + ahead
    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "limit")
    rig["monkeypatch"].setenv("FAKE_CLAUDE_LIMIT_SHAPE", window)
    rig["monkeypatch"].setenv(
        "FAKE_CLAUDE_LIMIT_RESETS_AT", str(int(resets_at.timestamp()))
    )
    code, _root = _run_source_round(rig)
    assert code == EXIT_BACKOFF

    # The executor announces the pause on stdout, which is the channel the driver
    # classifies. Round-trip it through the real driver-side classifier.
    notice = _last_json_line(capsys.readouterr().out)
    assert notice is not None, "executor emitted no structured backoff notice"
    signal = classify_author_response(EXIT_BACKOFF, json.dumps(notice))
    assert signal is not None
    assert signal.reason is AuthorPauseReason.QUOTA_EXHAUSTED
    assert signal.reset_at is not None, "declared reset was dropped before the driver"
    parsed = datetime.fromisoformat(signal.reset_at)
    assert abs((parsed - resets_at).total_seconds()) <= 1


def test_generic_session_error_is_not_a_campaign_pause(rig) -> None:
    """``error_during_execution`` without a limit terminal reason stays a retry.

    This is the over-widening guard. ``error_during_execution`` is the harness's
    *generic* failure subtype, so treating the subtype itself as a limit would
    convert every ordinary session crash into a campaign-wide outage.
    """

    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "limit")
    rig["monkeypatch"].setenv("FAKE_CLAUDE_LIMIT_SHAPE", "generic_crash")
    code, _root = _run_source_round(rig)
    assert code == EXIT_RETRYABLE


def test_fabricated_limit_subtypes_are_not_pause_authority() -> None:
    """The removed constants were fiction and must not silently return.

    None of these appear in the Claude Code 2.1.220 binary or in the
    claude-agent-sdk 0.3.211 ``subtype`` union. Re-adding one would restore a
    matcher that can never fire against a real harness while looking correct.
    """

    for subtype in (
        "usage_limit",
        "usage_limit_reached",
        "error_usage_limit",
        "error_rate_limit",
        "rate_limit",
    ):
        assert structured_limit_signal({"type": "result", "subtype": subtype}) is None


def test_real_success_payload_is_never_a_limit() -> None:
    """A verbatim captured clean result must not read as a pause."""

    assert (
        structured_limit_signal(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "terminal_reason": "completed",
                "api_error_status": None,
                "result": "ok",
            }
        )
        is None
    )


def test_limit_signal_ignores_free_text_entirely() -> None:
    """Limit words in payload prose never classify; only structured fields do."""

    assert (
        structured_limit_signal(
            {
                "type": "result",
                "subtype": "success",
                "terminal_reason": "completed",
                "result": "GitHub API rate limit exceeded; usage limit reached",
            }
        )
        is None
    )


def test_declared_reset_is_read_from_the_harness_field_only() -> None:
    """``resetsAt`` is unix seconds; junk and prose yield no reset."""

    moment = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    declared = structured_limit_reset_at(
        {"rate_limit_info": {"status": "rejected", "resetsAt": int(moment.timestamp())}}
    )
    # RFC 3339 ``Z`` form: ``wakeup._parse_utc`` refuses anything else, so an
    # offset-form reset would crash the pause it was meant to schedule.
    assert declared == "2026-08-04T12:00:00Z"
    assert structured_limit_reset_at({"rate_limit_info": {"status": "rejected"}}) is None
    assert (
        structured_limit_reset_at({"result": "resets at 2026-08-04T12:00:00+00:00"}) is None
    )
    assert (
        structured_limit_reset_at({"rate_limit_info": {"resetsAt": "soon"}}) is None
    )


def test_weekly_reset_survives_plausibility_validation() -> None:
    """A seven-day-out reset is admitted; an implausible one is still refused.

    A weekly limit resets up to seven days ahead. If the validator rejected that
    as implausible the pause would be silently discarded and the campaign would
    resume straight into the wall.
    """

    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    weekly = (now + timedelta(days=7)).isoformat()
    assert plausible_author_reset_at(weekly, now=now) == weekly
    five_hour = (now + timedelta(hours=5)).isoformat()
    assert plausible_author_reset_at(five_hour, now=now) == five_hour
    # The tripwire still holds at both ends.
    assert plausible_author_reset_at((now + timedelta(days=30)).isoformat(), now=now) is None
    assert plausible_author_reset_at((now - timedelta(hours=1)).isoformat(), now=now) is None


def test_observed_offset_form_reset_is_normalized_before_scheduling() -> None:
    """An observed ``+00:00`` reset must not crash the pause it should schedule.

    ``plausible_author_reset_at`` returns its candidate verbatim, so a provider or
    harness reset can reach the driver in offset form, while ``wakeup._parse_utc``
    accepts only ``Z``. Guessed resets already end in ``Z``, so only the *observed*
    path -- the one a real limit takes -- was exposed.
    """

    assert _normalize_wake_reset("2026-08-04T12:00:00+00:00") == "2026-08-04T12:00:00Z"
    assert _normalize_wake_reset("2026-08-04T12:00:00Z") == "2026-08-04T12:00:00Z"
    assert _normalize_wake_reset("2026-08-04T08:00:00-04:00") == "2026-08-04T12:00:00Z"
    # Never launder a malformed reset into a well-formed one; the wake layer
    # must still get its chance to refuse it.
    assert _normalize_wake_reset("not-a-timestamp") == "not-a-timestamp"
    assert _normalize_wake_reset("2026-08-04T12:00:00") == "2026-08-04T12:00:00"
    # End to end: what the executor declares is schedulable as-is.
    declared = structured_limit_reset_at(
        {"rate_limit_info": {"status": "rejected", "resetsAt": 1786000000}}
    )
    assert declared is not None
    build_wake_episode(
        provider="anthropic",
        reset_at=_normalize_wake_reset(declared),
        reset_observation="observed",
        callback_argv=["crawler", "resume"],
    )


def test_wake_episode_accepts_a_weekly_reset() -> None:
    """The wake machinery can carry a reset a full week out.

    Resumption is a *recurring* guarded poll, not one far-future timer: the
    episode fires on its retry cadence and the ``not_before`` guard suppresses
    every fire until the reset passes. So a seven-day reset needs no special
    scheduling capability -- but the episode must still accept it.
    """

    reset_at = _normalize_wake_reset(
        (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    )
    episode = build_wake_episode(
        provider="anthropic",
        reset_at=reset_at,
        reset_observation="observed",
        callback_argv=["crawler", "resume"],
    )
    assert episode.reset_at == reset_at
    assert MIN_RETRY_INTERVAL_SECONDS <= episode.retry_interval_seconds


def test_pause_during_stage2_resumes_without_losing_prior_work(rig) -> None:
    """A usage pause costs nothing already earned; resume is not a cold reread.

    The kill matrix proves this for SIGKILL, but a usage pause leaves a different
    durable shape: ``_fail`` marks the attempt ``failed`` with a typed
    ``provider-usage-pause`` reason rather than leaving it mid-stage. This asserts
    the recovery path is equally lossless -- stage 1 does not re-run, its session
    identity is inherited by the retry, and the pause is visible as prior-attempt
    feedback rather than being silently swallowed.
    """

    assert _run_source_round(rig)[0] == EXIT_OK
    stage1 = latest_attempt(rig["root"])
    assert stage1 is not None
    stage1_session = stage1.record["stage1"]["session_id"]
    stage1_status = stage1.status

    # Stage 2 hits a real five-hour limit envelope.
    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "limit")
    rig["monkeypatch"].setenv("FAKE_CLAUDE_LIMIT_SHAPE", "blocking_limit")
    assert _run_author_round(rig)[0] == EXIT_BACKOFF

    paused = latest_attempt(rig["root"])
    assert paused is not None
    assert paused.record["outcome"]["failure_reason"] == "provider-usage-pause"

    # Stage 1's durable product survives the pause untouched.
    assert stage1.record["stage1"]["session_id"] == stage1_session
    assert stage1.status == stage1_status

    # The provider comes back; the campaign resumes.
    rig["monkeypatch"].delenv("FAKE_CLAUDE_MODE", raising=False)
    rig["monkeypatch"].delenv("FAKE_CLAUDE_LIMIT_SHAPE", raising=False)
    before = len(read_invocations(rig["log"]))
    assert _run_author_round(rig)[0] == EXIT_OK

    resumed = latest_attempt(rig["root"])
    assert resumed is not None
    assert resumed.status == "completed"
    # No cold reread: the retry ran stage 2 only, resuming stage 1's session.
    replayed = read_invocations(rig["log"])[before:]
    assert [entry["stage"] for entry in replayed] == ["stage2"]
    assert resumed.record["inherited"]["stage1"]["session_id"] == stage1_session
    argv = replayed[-1]["argv"]
    assert argv[argv.index("--resume") + 1] == stage1_session


def _last_json_line(text: str) -> dict[str, Any] | None:
    """Return the last JSON object printed on a stream, mirroring the driver."""

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def test_rate_limit_stderr_noise_does_not_pause(rig) -> None:
    """GitHub rate-limit chatter on stderr never becomes a provider pause."""

    rig["monkeypatch"].setenv(
        "FAKE_CLAUDE_STDERR_NOISE", "GitHub API rate limit exceeded for 1.2.3.4"
    )
    code, _root = _run_source_round(rig)
    assert code == EXIT_OK


def test_session_crash_is_retryable_not_quota(rig) -> None:
    """A crashed session (with rate-limit words on stderr) exits 75, not 76."""

    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "crash")
    code, root = _run_source_round(rig)
    assert code == EXIT_RETRYABLE
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.record["outcome"]["failure_reason"] == "session-crashed"


def test_negative_discovery_arm_is_published_for_lane_materialization(rig) -> None:
    """A typed negative arm is published intact for the lane's checked R5 path."""

    rig["monkeypatch"].setenv(
        "FAKE_CLAUDE_DISCOVERY",
        json.dumps(
            {
                "arm": "NO_USABLE_SOURCE",
                "search_evidence": {
                    "queries": ["m1 architecture"],
                    "places": ["code hosts"],
                    "candidate_links": [],
                    "languages": ["en"],
                    "conclusion": "No usable source exists.",
                },
            }
        ),
    )
    code, root = _run_source_round(rig)
    assert code == EXIT_OK
    published = json.loads((root / "source-targets.json").read_text("utf-8"))
    assert published["arm"] == "NO_USABLE_SOURCE"
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.status == "completed"
    assert attempt.record["outcome"]["kind"] == "discovery-published"


def test_tool_failure_arm_fails_loudly_retryable(rig) -> None:
    """RETRYABLE_TOOL_FAILURE is the fail-loudly arm: exit 75, verbatim error kept."""

    rig["monkeypatch"].setenv(
        "FAKE_CLAUDE_DISCOVERY",
        json.dumps(
            {
                "arm": "RETRYABLE_TOOL_FAILURE",
                "tool_name": "Exa search",
                "tool_spelling": "mcp__exa__web_search_exa",
                "error": "tool not found",
            }
        ),
    )
    code, root = _run_source_round(rig)
    assert code == EXIT_RETRYABLE
    attempt = latest_attempt(root)
    assert attempt is not None
    outcome = attempt.record["outcome"]
    assert outcome["failure_reason"] == "research-tools-unavailable"
    assert outcome["detail"]["tool_spelling"] == "mcp__exa__web_search_exa"


def test_supplement_round_is_granted_exactly_once(rig) -> None:
    """A typed supplement request earns one broker pass and one resume."""

    _run_source_round(rig)
    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "supplement-request")
    code, root = _run_author_round(rig)
    assert code == EXIT_OK
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.status == "completed"
    assert attempt.record["supplement"]["manifest_path"]
    stages = [entry["stage"] for entry in read_invocations(rig["log"])]
    assert stages == ["stage1", "stage2", "supplement"]


def test_resume_failure_falls_back_cold_with_recorded_discovery(rig) -> None:
    """A dead provider session reruns cold with the recorded stage-1 output."""

    _run_source_round(rig)
    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "resume-fail")
    code, root = _run_author_round(rig)
    assert code == EXIT_OK
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.record["stage2"]["cold_start_reason"] == "resume-exit-5"
    calls = read_invocations(rig["log"])
    assert "--resume" in calls[1]["argv"]
    assert "--resume" not in calls[2]["argv"]
    assert "COLD START" in calls[2]["prompt"]


def test_prior_attempt_failure_is_rendered_into_the_next_brief(rig) -> None:
    """The feedback channel: a failed attempt's reason reaches the next brief."""

    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "crash")
    code, root = _run_source_round(rig)
    assert code == EXIT_RETRYABLE
    rig["monkeypatch"].delenv("FAKE_CLAUDE_MODE")
    code, root = _run_source_round(rig)
    assert code == EXIT_OK
    retry_brief = read_invocations(rig["log"])[1]["prompt"]
    assert "WHAT WENT WRONG LAST TIME" in retry_brief
    assert "session-crashed" in retry_brief


def test_schema_failure_detail_is_rendered_into_the_next_brief(rig) -> None:
    """A schema-rejected property reaches the retry brief as actionable feedback."""

    malformed = {
        "arm": "FOUND",
        "sources": [
            {
                **cast(dict[str, Any], DEFAULT_DISCOVERY["sources"][0]),
                "undeclared_observation": "No contract home.",
            }
        ],
    }
    rig["monkeypatch"].setenv("FAKE_CLAUDE_DISCOVERY", json.dumps(malformed))
    code, root = _run_source_round(rig)
    assert code == EXIT_RETRYABLE
    failed = latest_attempt(root)
    assert failed is not None
    diagnostic = failed.record["outcome"]["detail"]["error"]
    assert "undeclared_observation" in diagnostic
    assert "additionalProperties" in diagnostic

    rig["monkeypatch"].delenv("FAKE_CLAUDE_DISCOVERY")
    code, _ = _run_source_round(rig)
    assert code == EXIT_OK
    retry_brief = read_invocations(rig["log"])[1]["prompt"]
    assert diagnostic in retry_brief


def test_checker_findings_reach_the_repair_generation_brief(rig) -> None:
    """Checker-rejected leaves are named, verbatim, in generation 2's brief."""

    _run_source_round(rig)
    code, root = _run_author_round(rig)
    assert code == EXIT_OK
    # The checker rejects the accepted generation; the driver records findings.
    assert record_checker_findings(
        root,
        gate_kind="metadata_batch",
        generation=1,
        required_repairs=[
            "external_metadata.citation.title is unsupported by any excerpt",
            "taxonomy.family names a family the sources never state",
        ],
        root_cause_fingerprint="fp-123",
    )
    # The repair generation re-enters through the lane: source round + author.
    code, root = _run_source_round(rig)
    assert code == EXIT_OK
    repair_brief = read_invocations(rig["log"])[2]["prompt"]
    assert "WHAT WENT WRONG LAST TIME" in repair_brief
    assert "external_metadata.citation.title is unsupported" in repair_brief
    assert "taxonomy.family names a family" in repair_brief


def test_ten_model_rung_headless_with_zero_managing_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ten models complete source + author rounds with no managing session.

    No queue directory, no pool, no operator: the only components are the
    executor subprocess contract, the fake harness, and the hermetic broker.
    """

    fake = write_fake_claude(tmp_path / "bin")
    fixtures = write_broker_fixtures(tmp_path / "fixtures")
    log_dir = tmp_path / "log"
    executor_environment(
        monkeypatch, fake_claude=fake, fixtures=fixtures, log_dir=log_dir
    )
    monkeypatch.delenv("MENAGERIE_AUTHOR_QUEUE", raising=False)
    completed = []
    for index in range(10):
        stable_id = f"model-{index:02d}"
        root = tmp_path / "work" / stable_id / "author"
        assert main([str(write_source_request(root, stable_id))]) == EXIT_OK
        assert main([str(write_author_envelope(root, stable_id))]) == EXIT_OK
        assert (root / "result.json").is_file()
        attempt = latest_attempt(root)
        assert attempt is not None and attempt.status == "completed"
        completed.append(stable_id)
    assert len(completed) == 10
    # Zero managing session: nothing ever created a queue or claimed a lease.
    assert not list((tmp_path / "work").rglob("pending")), "no queue dirs may exist"
    stages = [entry["stage"] for entry in read_invocations(log_dir)]
    assert stages == ["stage1", "stage2"] * 10


def _write_probe_request(
    probe_dir: Path, nonce: str, *, drop_requested_at: bool = False
) -> Path:
    """Write one doctor-shaped capability request into ``probe_dir``."""

    probe_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "format": "menagerie.crawler.author-capability-probe.v1",
        "nonce": nonce,
        "requested_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "deadline_seconds": 300,
        "required_output_path": str(probe_dir / "receipt.json"),
    }
    if drop_requested_at:
        del request["requested_at"]
    request_path = probe_dir / "probe-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request_path


def test_capability_probe_publishes_doctor_shaped_receipt(rig, capsys) -> None:
    """A genuinely-exercised probe publishes the receipt shape the doctor accepts.

    The assertions below replicate the doctor's own acceptance filter
    (``doctor.author_tools``) field for field: top-level nonce, bounded
    ``completed_at``, and a ``receipts`` list whose entries only count when
    they echo the nonce, are ``exercised``, and carry a non-empty receipt
    string. The receipt is minted by ``validate_capability_evidence`` from the
    session's evidence — the fake session here produced genuine-shaped,
    nonce-echoing, digest-consistent observations, which is the ONLY reason
    ``exercised`` is true.
    """

    probe_dir = rig["tmp"] / "probe"
    nonce = "f" * 32
    request_path = _write_probe_request(probe_dir, nonce)
    assert main([str(request_path)]) == EXIT_OK
    published = probe_dir / "receipt.json"
    assert published.is_file()
    receipt = json.loads(published.read_text(encoding="utf-8"))
    assert receipt["nonce"] == nonce
    request = json.loads(request_path.read_text(encoding="utf-8"))
    requested_at = datetime.fromisoformat(
        str(request["requested_at"]).removesuffix("Z") + "+00:00"
    )
    completed_at = datetime.fromisoformat(
        str(receipt["completed_at"]).removesuffix("Z") + "+00:00"
    )
    assert requested_at <= completed_at <= requested_at + timedelta(seconds=300)
    accepted = {
        canonical_tool_name(value.get("tool"))
        for value in receipt["receipts"]
        if isinstance(value, dict)
        and value.get("nonce") == nonce
        and value.get("exercised") is True
        and isinstance(value.get("receipt"), str)
        and bool(str(value["receipt"]).strip())
    }
    assert accepted == {"WebSearch", "web_search_exa", "web_fetch_exa"}
    publication = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert publication["kind"] == "capability-probe"
    assert publication["attempt_nonce"] == nonce


def test_capability_probe_refuses_unproven_evidence(rig, capsys) -> None:
    """Evidence without genuine per-tool observations publishes NOTHING.

    A session that researched nothing must not become a passing receipt: the
    executor may stamp only what it witnessed (its clock, the nonce it was
    handed), never ``exercised`` for tools whose evidence does not prove the
    call happened. The refusal is retryable and the doctor's strict check
    fails, which is the probe working.
    """

    rig["monkeypatch"].setenv("FAKE_CLAUDE_PROBE", "hollow")
    probe_dir = rig["tmp"] / "probe"
    nonce = "f" * 32
    request_path = _write_probe_request(probe_dir, nonce)
    assert main([str(request_path)]) == EXIT_RETRYABLE
    assert not (probe_dir / "receipt.json").exists()
    assert "capability-probe" not in capsys.readouterr().out


def test_capability_probe_requires_requested_at(rig) -> None:
    """A request without ``requested_at`` is a permanent request defect.

    Without the request instant there is no freshness window to validate
    against, so the executor refuses before spending a session rather than
    minting a receipt whose window it cannot anchor.
    """

    probe_dir = rig["tmp"] / "probe"
    request_path = _write_probe_request(probe_dir, "f" * 32, drop_requested_at=True)
    assert main([str(request_path)]) == EXIT_PERMANENT
    assert not (probe_dir / "receipt.json").exists()


def test_lane_receipt_verification_rejects_tampered_bytes(tmp_path: Path) -> None:
    """The lane refuses a result whose bytes drifted from the receipt digest."""

    output = tmp_path / "result.json"
    output.write_text('{"kind": "PROPOSED"}', encoding="utf-8")
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "attempt_nonce": "n1",
        "result_sha256": hash_bytes(output.read_bytes()),
    }
    stdout = json.dumps(receipt)
    # Matching bytes pass.
    _verify_executor_receipt(stdout, output, stable_id="m1", required=True)
    # A late writer racing the published result is refused.
    output.write_text('{"kind": "STALE_OVERWRITE"}', encoding="utf-8")
    with pytest.raises(DriverIntegrationError, match="do not match the executor receipt"):
        _verify_executor_receipt(stdout, output, stable_id="m1", required=True)


def test_lane_requires_receipt_when_configured(tmp_path: Path) -> None:
    """`require_receipt` makes a receiptless success a typed integration failure."""

    output = tmp_path / "result.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(DriverIntegrationError, match="printed no publication receipt"):
        _verify_executor_receipt("all done", output, stable_id="m1", required=True)
    # Opportunistic mode tolerates legacy wrappers with no receipt.
    _verify_executor_receipt("all done", output, stable_id="m1", required=False)
    lane = CommandAuthorLane(["true"], effort_grant=AuthorEffortGrant())
    assert lane.require_receipt is False


def test_supersession_quarantines_stale_results_before_new_attempt(
    tmp_path: Path,
) -> None:
    """Sol's laundering repro, ported: a late old completion is quarantined.

    The old attempt's late ``result.json`` can never be stamped with a new
    attempt's nonce: attempts never share paths, and opening attempt N+1
    moves any stray result into quarantine, never into publication.
    """

    root = tmp_path / "author"
    old = new_attempt(root, stable_id="m1", campaign_id="c1-mech", kind="author")
    old.update(status="stage2-running")
    stale = old.paths.directory / "result.json"
    stale.write_text('{"kind": "STALE", "from": "old-attempt"}', encoding="utf-8")

    fresh = new_attempt(root, stable_id="m1", campaign_id="c1-mech", kind="author")
    assert not stale.exists(), "the stale result must leave the publishable path"
    superseded = list_attempts(root)[0]
    assert superseded.status == "superseded"
    assert superseded.record["superseded"]["by_nonce"] == fresh.nonce
    quarantined = superseded.record["quarantine"]
    assert len(quarantined) == 1
    payload = json.loads(Path(quarantined[0]["quarantined_to"]).read_text("utf-8"))
    assert payload["from"] == "old-attempt"


def _admissible_work_item(proposal: dict[str, Any]) -> WorkItem:
    """Return the trusted intake row the shared proposal fixture was built against.

    ``trusted_identity_fields`` designates a standalone row by its intake NAME,
    so the roster name is the fixture's ``identity.variant``. Getting this wrong
    short-circuits admission at the trusted-intake check, which silently stops
    any later assertion from being the one that decides.
    """

    identity = proposal["proposed_facts"]["identity"]
    return WorkItem(
        intake=IntakeItem(
            stable_id=proposal["stable_id"],
            name=identity["variant"],
            zoo="crawler",
            variant=identity["variant"],
            discovery_source="crawl_roster",
            legacy_row_sha256="0" * 64,
            preserved_legacy_flags=(),
            variant_scope=identity["variant_scope"],
            family_representative_id=identity["family_representative_id"],
        ),
        route=IntentRoute(
            stable_id=proposal["stable_id"], intent="core", phase=EnvironmentPhase.PYTORCH
        ),
    )


def _admit(proposal: dict[str, Any], tmp_path: Path) -> None:
    """Run the shared proposal fixture through artifact-identity admission."""

    artifact = make_proposed_artifact(
        proposal, {"manifest_sha256": proposal["source_manifest_identity"]}, tmp_path
    )
    _validate_artifact_identities(
        artifact,
        DriverConfig(checker_model="codex", checker_version="current"),
        item=_admissible_work_item(proposal),
    )


def test_admission_accepts_the_honest_shared_proposal_fixture(tmp_path: Path) -> None:
    """The shared fixture must be admissible, or the refusals below prove nothing."""

    _admit(make_author_proposal(), tmp_path)


def test_admission_binds_the_verified_source_manifest_hash(tmp_path: Path) -> None:
    """A verified hash the machine already holds may not be the author's word.

    ``artifact_transactions`` already refuses a ``verified_hashes.source_manifest``
    that disagrees with staging, but only at publication-authorization time --
    long after a checker has gated the proposal on it. Admission is the first
    point the machine can make the same refusal.
    """

    proposal = make_author_proposal()
    proposal["verified_hashes"]["source_manifest"] = "sha256:" + "9" * 64

    with pytest.raises(DriverIntegrationError, match="verified_hashes.source_manifest"):
        _admit(proposal, tmp_path)


def test_admission_refuses_a_fixture_placeholder_left_in_a_real_binding(
    tmp_path: Path,
) -> None:
    """The all-``a`` placeholder is only detectable once the binding is real.

    ``make_author_proposal`` ships the placeholder in BOTH
    ``source_manifest_identity`` and ``verified_hashes.source_manifest``, so the
    two agree trivially and no guard can see it. Give the proposal the real
    identity a dispatched author would have been handed and the stale
    fixture-templated digest beside it becomes visible -- which is exactly the
    shape a model templating its answer from repository test data emits.
    """

    proposal = make_author_proposal()
    proposal["source_manifest_identity"] = "sha256:" + "7" * 64
    assert proposal["verified_hashes"]["source_manifest"] == HASH, (
        "the fixture must still carry its placeholder for this to be the real case"
    )

    with pytest.raises(DriverIntegrationError, match="verified_hashes.source_manifest"):
        _admit(proposal, tmp_path)

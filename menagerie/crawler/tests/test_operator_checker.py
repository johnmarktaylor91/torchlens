"""Targeted checker-wrapper protocol, tiering, and fault-injection tests."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import pytest

from menagerie.crawler.checker_dispatch import (
    LEDGER_ASSIGNED_GATE_FIELDS,
    PROMPT_PATH,
    CheckerDispatchError,
    apply_machine_owned_gate_fields,
    build_fidelity_envelope,
    build_metadata_vet_envelope,
    compute_result_envelope_sha256,
    machine_owned_gate_fields,
    metadata_fidelity_state,
    validate_checker_result,
    validate_checker_result_mapping,
)
from menagerie.crawler.constants import GateKind
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.operator_checker import (
    CHECKER_MAX_ATTEMPTS,
    CHECKER_TIMEOUT_SECONDS,
    FIDELITY_CHECKER_MODEL,
    METADATA_CHECKER_MODEL,
    TERMINAL_CHECKER_MODEL,
    CodexAttempt,
    _invoke_codex,
    _native_output_schema,
    classify_codex_attempt,
    execute_checker_request,
    required_checker_model,
)
from menagerie.crawler.operator_protocol import (
    OPERATOR_ATTEMPT_TIMEOUT_SECONDS,
    OPERATOR_DEADLINE_SECONDS,
    OPERATOR_DISPATCH_SLACK_SECONDS,
    OPERATOR_INTER_ATTEMPT_BACKOFF_SECONDS,
    OPERATOR_MAX_ATTEMPTS,
    OperatorExitCode,
    status_sidecar_path,
    telemetry_path,
)
from menagerie.crawler.tests.conftest import make_gate
from menagerie.crawler.tools.checker_latency import build_report, iter_attempt_records


def _checker_item_pack(item: dict[str, Any]) -> dict[str, Any]:
    """Build the minimal exact checker item accepted by the envelope builder.

    Parameters
    ----------
    item:
        Expected result item from the shared gate fixture.

    Returns
    -------
    dict[str, Any]
        Complete request item.
    """

    return {
        "work_id": item["work_id"],
        "campaign_root_work_id": item["campaign_root_work_id"],
        "stable_id": item["stable_id"],
        "family_representative_id": item["family_representative_id"],
        "fidelity_identity": item["fidelity_identity"],
        "vet_identity": item["vet_identity"],
        "verified_hashes": deepcopy(item["verified_hashes"]),
        "proposal": {
            "description": "scoped checker wrapper test",
            "proposed_facts": {"implementation": {"code_path": None}},
        },
        "source_manifest": {"sources": []},
        "evidence": {"excerpts": []},
        # Every real envelope item names its author directory, because the
        # envelope derives the declared ``source-cas`` read root from it.
        "model_dir": f"/menagerie-checker-test/{item['stable_id']}/author/model",
    }


def _request_and_result(
    tmp_path: Path,
    *,
    gate_kind: GateKind = GateKind.METADATA_BATCH,
    model: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write one exact request and return its schema-valid candidate result.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    gate_kind:
        Metadata or fidelity gate.
    model:
        Optional model override for rejection tests.

    Returns
    -------
    tuple[pathlib.Path, dict[str, Any]]
        Absolute request path and final-message candidate.
    """

    stable_id = f"m_{gate_kind.value}"
    gate = make_gate([stable_id])
    item = _checker_item_pack(gate["items"][0])
    output_path = tmp_path / "result.json"
    selected_model = model or required_checker_model(gate_kind)
    if gate_kind is GateKind.METADATA_BATCH:
        envelope = build_metadata_vet_envelope(
            [item],
            gate_round=1,
            output_path=output_path,
            checker_model=selected_model,
            checker_version="current",
            request_nonce="wrapper-metadata",
            final_tail=True,
        )
    elif gate_kind is GateKind.FIDELITY:
        envelope = build_fidelity_envelope(
            item,
            gate_round=1,
            output_path=output_path,
            checker_model=selected_model,
            checker_version="current",
            request_nonce="wrapper-fidelity",
        )
        gate["items"][0]["fidelity"] = {
            "required": True,
            "verdict": "match",
            "material_checks": [],
            "unsupported_choices": [],
            "contradictions": [],
            "omissions": [],
            "permanent_scar": False,
        }
    else:
        raise AssertionError("test helper supports metadata and fidelity only")
    # Model a COMPLIANT checker: the frozen prompt tells it to omit every
    # machine-owned field, and omission is what the wrapper stamps over. This
    # helper previously handed back the shared fixture's scaffold verbatim --
    # ``gate_id="gate-1"``, ``payload_sha256`` of all ``a``s, ``checker.version
    # == "test"`` -- and relied on the stamp silently rewriting them. That made
    # the suite's own happy path a laundered gate, which is exactly the
    # near-miss the stamp guard now refuses, so the fixture has to stop
    # depending on it.
    #
    # The ledger-assigned pair goes too. It is not part of the machine-owned
    # scaffold -- the ledger assigns it at append time from its own state -- so a
    # compliant checker result cannot carry it either.
    for field in (*machine_owned_gate_fields(envelope), *LEDGER_ASSIGNED_GATE_FIELDS):
        gate.pop(field, None)
    gate.pop("checker", None)
    gate.pop("result_envelope_sha256", None)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
    return request_path, gate


def _native_final_message(gate: dict[str, Any]) -> bytes:
    """Serialize one gate the way the natively constrained checker returns it.

    The final message is the checker-authored fragment itself, not a string
    carrying a whole gate: the ``--output-schema`` now pins the gate's own item
    vocabulary, so the machine-owned scaffold is not merely omitted by
    convention, it is absent from what the model is able to emit.

    Parameters
    ----------
    gate:
        Candidate gate whose items the checker authored.

    Returns
    -------
    bytes
        Exact ``-o`` final-message bytes.
    """

    return canonical_json_bytes({"items": gate["items"]}) + b"\n"


def _status(request_path: Path) -> dict[str, Any]:
    """Read one wrapper status sidecar.

    Parameters
    ----------
    request_path:
        Exact wrapper request.

    Returns
    -------
    dict[str, Any]
        Parsed status authority.
    """

    value = json.loads(status_sidecar_path(request_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_locked_model_tiering_uses_full_identifiers() -> None:
    """Routine gates use Terra and fidelity alone uses Sol."""

    assert required_checker_model(GateKind.METADATA_BATCH) == METADATA_CHECKER_MODEL
    assert required_checker_model(GateKind.FIDELITY) == FIDELITY_CHECKER_MODEL
    assert required_checker_model(GateKind.TERMINAL_DISPOSITION) == TERMINAL_CHECKER_MODEL
    assert METADATA_CHECKER_MODEL == "gpt-5.6-terra"
    assert FIDELITY_CHECKER_MODEL == "gpt-5.6-sol"


def test_operator_effort_grant_fits_inside_the_published_deadline() -> None:
    """Every granted attempt fits under the deadline the same grant publishes.

    The wrapper refuses to start an attempt past ``deadline_at`` and clamps each
    attempt to the time remaining, so a deadline that does not cover
    ``OPERATOR_MAX_ATTEMPTS`` full attempts plus the wrapper's own inter-attempt
    backoff silently shortens or drops the last attempt while the effort grant
    still advertises it. That coupling is why the attempt cap cannot be raised on
    its own, and this pins it structurally rather than by remembered arithmetic.
    """

    consumed = (
        OPERATOR_MAX_ATTEMPTS * OPERATOR_ATTEMPT_TIMEOUT_SECONDS
        + OPERATOR_INTER_ATTEMPT_BACKOFF_SECONDS
    )
    assert consumed < OPERATOR_DEADLINE_SECONDS
    # The residual is dispatch latency, declared rather than accidental: the
    # deadline is minted at envelope build and the first attempt starts later.
    assert OPERATOR_DEADLINE_SECONDS - consumed == OPERATOR_DISPATCH_SLACK_SECONDS
    assert OPERATOR_DISPATCH_SLACK_SECONDS > 0
    # The backoff term must equal the sleeps the wrapper actually performs:
    # ``sleep(2 ** (attempt_number - 1))`` after every attempt but the last.
    assert OPERATOR_INTER_ATTEMPT_BACKOFF_SECONDS == sum(
        2 ** (attempt_number - 1) for attempt_number in range(1, OPERATOR_MAX_ATTEMPTS)
    )
    assert (CHECKER_MAX_ATTEMPTS, CHECKER_TIMEOUT_SECONDS) == (
        OPERATOR_MAX_ATTEMPTS,
        float(OPERATOR_ATTEMPT_TIMEOUT_SECONDS),
    )


def test_attempt_telemetry_records_completed_and_censored_durations(tmp_path: Path) -> None:
    """Both attempt outcomes carry a wall duration and the bound that censored it.

    A timed-out attempt is a RIGHT-CENSORED observation, not an absent one. Its
    duration and the bound it was held to are what make the sample analyzable, so
    both are recorded for timed-out and completed attempts alike.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, result = _request_and_result(tmp_path)
    calls = 0

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject one measurable timeout followed by one success."""

        del argv
        nonlocal calls
        calls += 1
        assert timeout == CHECKER_TIMEOUT_SECONDS
        if calls == 1:
            # Real elapsed time, so the recorded duration is a measurement rather
            # than a constant the wrapper could have invented.
            time.sleep(0.05)
            return CodexAttempt(-9, "", "", timed_out=True)
        last_message.write_bytes(_native_final_message(result))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.SUCCESS
    events = [
        json.loads(line)
        for line in telemetry_path(request_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attempts = [event for event in events if event["event"] == "codex-attempt"]
    assert [event["attempt"] for event in attempts] == [1, 2]
    censored, completed = attempts
    assert censored["timed_out"] is True
    assert completed["timed_out"] is False
    for event in attempts:
        assert event["attempt_timeout_seconds"] == CHECKER_TIMEOUT_SECONDS
        assert event["attempt_budget_seconds"] == CHECKER_TIMEOUT_SECONDS
        # The workload covariates: one flat cap spans a one-model fidelity call and
        # a twenty-model metadata batch, so a duration is only interpretable
        # alongside what the call was asked to do.
        assert event["gate_kind"] == GateKind.METADATA_BATCH.value
        assert event["item_count"] == 1
        assert event["started_at"] <= event["finished_at"]
        assert isinstance(event["duration_seconds"], float)
        assert event["duration_seconds"] >= 0.0
    assert censored["duration_seconds"] >= 0.05
    # The existing keys stay exactly as they were; duration is additive.
    assert {"attempt", "classification", "detail", "event", "returncode", "timed_out"} <= set(
        censored
    )
    finished = [event for event in events if event["event"] == "operator-finished"]
    assert len(finished) == 1
    assert finished[0]["wall_seconds"] >= censored["duration_seconds"]
    # The report must read what the wrapper actually WRITES, not a hand-written
    # fixture of it: a renamed field would otherwise leave the report silently
    # summarizing an empty sample.
    report = build_report(iter_attempt_records([request_path.parent]))
    assert report["attempts"] == 2
    assert report["completed"]["count"] == 1
    assert report["censored_lower_bounds"]["count"] == 1
    assert report["untimed_attempts"] == 0
    assert set(report["by_workload"]) == {f"{GateKind.METADATA_BATCH.value}[1]"}


def _flubbed_echo(result: dict[str, Any]) -> dict[str, Any]:
    """Return the result with one well-formed but mis-transcribed vet identity.

    This is the observed live fault shape (rung 3, batch ``0f44a52d3e324c4c``,
    item m4334): every other binding echoed exactly, and the flubbed value is
    still a syntactically valid ``sha256:`` string, so nothing upstream of the
    binding comparison can catch it.
    """

    flubbed = deepcopy(result)
    flubbed["items"][0]["vet_identity"] = "sha256:" + "0" * 64
    return flubbed


def test_binding_mismatch_is_refused_then_retried_not_terminal(tmp_path: Path) -> None:
    """One mis-transcribed identity echo costs an attempt, not the whole batch.

    The mismatched result is REFUSED -- the telemetry proves it never published
    -- and a fresh bounded attempt re-reads the same frozen envelope. Before
    this, the first flub was an immediate ``PERMANENT_CONTRACT_REJECTION`` that
    terminalized every batch member as ``failed:runner / protocol-violation``
    on a run-once system (rung 3: m4334 and m8245 both died on one flubbed
    string).

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, result = _request_and_result(tmp_path)
    calls = 0

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject one mis-echoed binding followed by one faithful echo."""

        del argv, timeout
        nonlocal calls
        calls += 1
        message = _flubbed_echo(result) if calls == 1 else result
        last_message.write_bytes(_native_final_message(message))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.SUCCESS
    assert calls == 2
    published = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert published["items"][0]["vet_identity"] == result["items"][0]["vet_identity"]
    events = [
        json.loads(line)
        for line in telemetry_path(request_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    refusals = [event for event in events if event["event"] == "binding-mismatch-refused"]
    assert [event["attempt"] for event in refusals] == [1]
    assert refusals[0]["retrying"] is True
    assert "mismatched binding: vet_identity" in refusals[0]["detail"]
    assert _status(request_path)["classification"] == "success"


def test_persistent_binding_mismatch_still_exits_permanent(tmp_path: Path) -> None:
    """A mismatch on every bounded attempt keeps the exact historical refusal.

    The retry never weakens the tripwire: a checker that cannot ever echo its
    envelope bindings -- the signature of a genuine builder/matcher divergence
    rather than a stochastic flub -- exhausts ``CHECKER_MAX_ATTEMPTS`` and
    exits with the same permanent contract rejection, the same sidecar
    classification, and no published result.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, result = _request_and_result(tmp_path)
    calls = 0

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject the same mis-echoed binding on every attempt."""

        del argv, timeout
        nonlocal calls
        calls += 1
        last_message.write_bytes(_native_final_message(_flubbed_echo(result)))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.PERMANENT_CONTRACT_REJECTION
    assert calls == CHECKER_MAX_ATTEMPTS
    assert not (tmp_path / "result.json").exists()
    status = _status(request_path)
    assert status["classification"] == "permanent-contract-rejection"
    assert "mismatched binding: vet_identity" in status["detail"]
    events = [
        json.loads(line)
        for line in telemetry_path(request_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    refusals = [event for event in events if event["event"] == "binding-mismatch-refused"]
    assert [event["retrying"] for event in refusals] == [True] * (CHECKER_MAX_ATTEMPTS - 1) + [
        False
    ]


@pytest.mark.parametrize(
    ("gate_kind", "expected_model"),
    [
        (GateKind.METADATA_BATCH, METADATA_CHECKER_MODEL),
        (GateKind.FIDELITY, FIDELITY_CHECKER_MODEL),
    ],
)
def test_success_uses_settled_argv_and_publishes_atomically(
    tmp_path: Path, gate_kind: GateKind, expected_model: str
) -> None:
    """A native final answer reaches the exact result path through the locked argv.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    gate_kind:
        Gate tier under test.
    expected_model:
        Exact locked model.
    """

    request_path, result = _request_and_result(tmp_path, gate_kind=gate_kind)
    observed: list[tuple[str, ...]] = []

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject one successful native structured-output response."""

        observed.append(tuple(argv))
        # The full granted cap, not a deadline-clamped remainder: a freshly minted
        # envelope must leave room for every attempt it grants.
        assert timeout == CHECKER_TIMEOUT_SECONDS
        last_message.write_bytes(_native_final_message(result))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "Reading additional input from stdin...\n",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.SUCCESS
    published = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    # The published gate is the checker's verdict plus the machine-owned
    # scaffold the wrapper stamps. Identities and wall timings are never taken
    # from the checker, so they differ from whatever it wrote.
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    scaffold = machine_owned_gate_fields(envelope)
    assert {key: published[key] for key in scaffold} == scaffold
    assert published["checker"]["started_at"] <= published["checker"]["finished_at"]
    checker_owned = set(scaffold) | {"checker", "result_envelope_sha256"}
    assert {key: value for key, value in published.items() if key not in checker_owned} == {
        key: value for key, value in result.items() if key not in checker_owned
    }
    argv = observed[0]
    assert argv[:6] == (
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
    )
    assert argv[argv.index("-m") + 1] == expected_model
    assert argv[argv.index("-c") + 1] == "model_reasoning_effort=high"
    assert "--output-schema" in argv
    assert "-o" in argv
    assert "--json" in argv
    assert _status(request_path)["classification"] == "success"


def test_injected_transient_exhaustion_never_becomes_permanent(tmp_path: Path) -> None:
    """Three injected stream failures retain retryable exit 75 per risk R8."""

    request_path, _result = _request_and_result(tmp_path)
    attempts = 0

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject a retryable stream disconnect."""

        nonlocal attempts
        del argv, last_message, timeout
        attempts += 1
        return CodexAttempt(
            1,
            '{"type":"error","message":"Reconnecting... 1/5 '
            '(stream disconnected before completion: error sending request)"}\n',
            "Reading additional input from stdin...\n",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.RETRYABLE_INFRASTRUCTURE
    assert attempts == 3
    status = _status(request_path)
    assert status["classification"] == "retryable-infrastructure"
    assert status["exit_code"] == 75
    assert not (tmp_path / "result.json").exists()


def test_agent_tool_content_cannot_forge_a_quota_pause() -> None:
    """Quota words in ordinary item payloads are not provider error authority."""

    classification = classify_codex_attempt(
        CodexAttempt(
            0,
            '{"type":"item.completed","item":{"type":"command_execution",'
            '"aggregated_output":"test the quota marker"}}\n'
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "Reading additional input from stdin...\n",
        )
    )

    assert classification.kind.value == "success"


def test_stdout_usage_limit_survives_nonempty_stderr_and_extracts_reset(tmp_path: Path) -> None:
    """Merged streams expose stdout quota text and publish its real reset."""

    request_path, _result = _request_and_result(tmp_path)
    diagnostics = StringIO()

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject measured Codex stream placement for a usage-limit failure."""

        del argv, last_message, timeout
        return CodexAttempt(
            1,
            '{"type":"turn.failed","error":{"message":"You have hit your usage limit. '
            'Purchase credits or try again at 2026-07-28T16:30:00Z."}}\n',
            "Reading additional input from stdin...\n",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        now=lambda: datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc),
        diagnostic_stream=diagnostics,
    )

    assert exit_code is OperatorExitCode.RATE_OR_QUOTA_PAUSE
    assert "Reading additional input" in diagnostics.getvalue()
    assert "usage limit" in diagnostics.getvalue()
    status = _status(request_path)
    assert status["reset_at"] == "2026-07-28T16:30:00Z"
    assert status["reset_observation"] == "observed"
    assert status["exit_code"] == 76


@pytest.mark.parametrize(
    "message",
    [
        "400 invalid_request_error: model is not supported",
        "Failed to read output schema file /missing/schema.json",
        "Model metadata for terra not found",
    ],
)
def test_permanent_configuration_faults_do_not_retry(tmp_path: Path, message: str) -> None:
    """Known permanent faults exit 64 on their first attempt.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    message:
        Permanent Codex configuration signature.
    """

    request_path, _result = _request_and_result(tmp_path)
    attempts = 0

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject one permanent configuration failure."""

        nonlocal attempts
        del argv, last_message, timeout
        attempts += 1
        return CodexAttempt(1, f'{{"type":"error","message":"{message}"}}\n', "")

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.PERMANENT_CONTRACT_REJECTION
    assert attempts == 1
    assert _status(request_path)["classification"] == "permanent-contract-rejection"


def test_wrong_tier_is_a_preflight_contract_rejection(tmp_path: Path) -> None:
    """A fidelity request cannot silently run under Terra."""

    request_path, _result = _request_and_result(
        tmp_path,
        gate_kind=GateKind.FIDELITY,
        model=METADATA_CHECKER_MODEL,
    )
    called = False

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Fail if preflight allows the wrong tier to reach Codex."""

        nonlocal called
        del argv, last_message, timeout
        called = True
        return CodexAttempt(0, "", "")

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.PERMANENT_CONTRACT_REJECTION
    assert called is False


def test_missing_codex_binary_is_typed_service_unavailable(tmp_path: Path) -> None:
    """An unavailable executable exits 78 without becoming a model failure."""

    request_path, _result = _request_and_result(tmp_path)

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject an executable-resolution failure."""

        del argv, last_message, timeout
        return CodexAttempt(127, "", "codex: command not found", unavailable=True)

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.SERVICE_UNAVAILABLE
    assert _status(request_path)["classification"] == "service-unavailable"


@pytest.mark.parametrize(
    "gate_kind", [GateKind.METADATA_BATCH, GateKind.FIDELITY, GateKind.TERMINAL_DISPOSITION]
)
def test_native_output_schema_constrains_the_gate_itself_not_an_opaque_string(
    gate_kind: GateKind,
) -> None:
    """The wrapper hands Codex the gate's own vocabulary, not a string transport.

    The one-field ``result_json`` STRING schema this replaces made the provider's
    structured-output constraint vacuous: the gate travelled as an opaque string,
    so nothing but prompt prose carried its shape.

    Parameters
    ----------
    gate_kind:
        Gate kind whose derived schema is checked.
    """

    schema = _native_output_schema(gate_kind)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"items"}
    assert schema["required"] == ["items"]
    # The gate is the constraint, not a payload inside one.
    assert "result_json" not in json.dumps(schema)
    assert schema["properties"]["items"]["items"]["type"] == "object"
    # The metadata schema is the one the rung-8 batch died on: the whole
    # fidelity block is machine-owned there and must be unrepresentable.
    item_properties = schema["properties"]["items"]["items"]["properties"]
    if gate_kind is GateKind.METADATA_BATCH:
        assert "fidelity" not in item_properties
    else:
        assert "fidelity" in item_properties


def test_external_timeout_kills_the_codex_process_group(tmp_path: Path) -> None:
    """The real subprocess boundary returns a typed transient timeout quickly."""

    attempt = _invoke_codex(
        (sys.executable, "-c", "import time; time.sleep(5)"),
        tmp_path / "unused.json",
        0.05,
    )

    assert attempt.timed_out is True
    assert attempt.returncode != 0


def _fixture_templated_candidate(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Return one request plus the RAW shared-fixture gate as a checker answer.

    ``make_gate`` is the exact ``conftest`` helper a live checker was observed
    reading out of this repository on 2026-07-30: the ``m11695``
    ``operator-telemetry.jsonl`` captured it printing ``conftest.py`` lines
    2000-2095. Unlike ``_request_and_result``, this helper does NOT repair the
    fixture's machine-owned scaffold, so the candidate carries the fixture's
    placeholder verification values verbatim -- precisely the artifact a checker
    produces when it templates its answer instead of deriving it.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.

    Returns
    -------
    tuple[pathlib.Path, dict[str, Any], dict[str, Any]]
        Request path, parsed envelope, and the untouched fixture gate.
    """

    gate = make_gate(["m_fixture_templated"])
    envelope = build_metadata_vet_envelope(
        [_checker_item_pack(gate["items"][0])],
        gate_round=1,
        output_path=tmp_path / "result.json",
        checker_model=required_checker_model(GateKind.METADATA_BATCH),
        checker_version="current",
        request_nonce="wrapper-fixture-templated",
        final_tail=True,
    )
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
    return request_path, envelope, gate


def test_unconditional_stamping_would_launder_a_fixture_templated_gate(tmp_path: Path) -> None:
    """The removed behavior is proven to have been a laundering path.

    This is the FAILING direction the guard exists for, asserted positively
    rather than assumed: the old stamp is reproduced verbatim and its output is
    shown to pass full gate validation. A guard demonstrated only where it
    passes proves nothing, so the hazard is demonstrated first.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    _request_path, envelope, gate = _fixture_templated_candidate(tmp_path)

    # The fixture's placeholder verification values, before any stamping.
    assert gate["gate_id"] == "gate-1"
    assert gate["payload_sha256"] == "sha256:" + "a" * 64
    assert gate["checker"]["version"] == "test"
    assert gate["checker"]["model"] == "codex"

    # Verbatim reproduction of the pre-fix stamp: an unconditional ``update``.
    # The pre-fix scaffold also carried the two ledger placeholders, so they are
    # reproduced here explicitly. They have since moved OUT of the scaffold and
    # into ``LEDGER_ASSIGNED_GATE_FIELDS`` -- reading them from the current
    # ``machine_owned_gate_fields`` would silently stop reproducing the old stamp
    # and turn the ``payload_sha256`` assertion below into a no-op.
    laundered = dict(gate)
    laundered.update(machine_owned_gate_fields(envelope))
    laundered.update({"ledger_seq": 1, "payload_sha256": "sha256:" + "0" * 64})
    checker = dict(laundered["checker"])
    checker.update(
        {
            "provider": envelope["checker"]["provider"],
            "model": envelope["checker"]["model"],
            "version": envelope["checker"]["version"],
            "prompt_sha256": envelope["checker"]["prompt_sha256"],
            "started_at": "2026-07-30T18:00:00Z",
            "finished_at": "2026-07-30T18:00:01Z",
        }
    )
    laundered["checker"] = checker
    laundered["result_envelope_sha256"] = compute_result_envelope_sha256(laundered)

    # Every fixture placeholder has been silently rewritten to the machine's
    # real value.
    assert laundered["gate_id"] != "gate-1"
    assert laundered["payload_sha256"] != "sha256:" + "a" * 64
    assert laundered["checker"]["version"] == "current"

    # ...and the fabricated gate validates clean. Today a gate carrying the
    # ledger-assigned pair at all is refused before this point, which is a
    # strictly stronger second gate that did not exist pre-fix; drop the pair so
    # the hazard being demonstrated here is the SCAFFOLD laundering and not that
    # later refusal standing in for it.
    pre_ledger_split = {
        key: value for key, value in laundered.items() if key not in LEDGER_ASSIGNED_GATE_FIELDS
    }
    pre_ledger_split["result_envelope_sha256"] = compute_result_envelope_sha256(pre_ledger_split)
    validated = validate_checker_result_mapping(pre_ledger_split, envelope)
    assert validated["gate_id"] == machine_owned_gate_fields(envelope)["gate_id"]


def test_fixture_templated_gate_is_refused_and_the_stamp_guard_still_bites(
    tmp_path: Path,
) -> None:
    """The same candidate is refused, and the attribution guard remains live.

    The refusal now happens EARLIER than it used to. A whole gate is no longer a
    shape the final message can take: the native output schema admits exactly the
    checker-authored ``items``, so a templated gate carrying a scaffold is refused
    as a transport violation before ``apply_machine_owned_gate_fields`` ever sees
    it. That is strictly stronger, but it would make a wrapper-only assertion a
    tautology, so the attributing guard is ALSO exercised at its own boundary
    here -- the earlier refusal must not be allowed to read as the later one
    having been removed.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, envelope, gate = _fixture_templated_candidate(tmp_path)

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject the fixture-templated whole gate as the final answer."""

        del argv, timeout
        last_message.write_bytes(canonical_json_bytes(gate) + b"\n")
        return CodexAttempt(0, '{"type":"turn.completed"}\n', "")

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.PERMANENT_CONTRACT_REJECTION
    # Fails CLOSED: nothing is published at the atomic output path.
    assert not (tmp_path / "result.json").exists()
    status = _status(request_path)
    assert status["classification"] == "permanent-contract-rejection"
    assert "sole key is 'items'" in status["detail"]

    # The guard the earlier refusal now pre-empts is still armed at its own
    # boundary, with the offending field and its value as the evidence.
    with pytest.raises(CheckerDispatchError) as excinfo:
        apply_machine_owned_gate_fields(
            gate,
            envelope,
            started_at="2026-07-30T18:00:00Z",
            finished_at="2026-07-30T18:00:01Z",
        )
    detail = str(excinfo.value)
    assert detail.startswith("checker supplied the machine-owned field ")
    assert "templated rather than derived" in detail
    assert any(
        f"field {field}=" in detail
        for field in ("gate_id", "payload_sha256", "gate_identity", "ledger_seq")
    )


@pytest.mark.parametrize(
    ("path", "supplied", "owner"),
    [
        ("payload_sha256", "sha256:" + "a" * 64, "ledger-assigned"),
        ("ledger_seq", 7, "ledger-assigned"),
        ("gate_id", "gate-1", "machine-owned"),
        ("checker.version", "test", "machine-owned"),
        ("checker.model", "codex", "machine-owned"),
        ("checker.started_at", "2026-07-14T12:00:00Z", "machine-owned"),
    ],
)
def test_each_fixture_placeholder_is_refused_with_its_own_value_as_evidence(
    tmp_path: Path, path: str, supplied: Any, owner: str
) -> None:
    """One field the checker does not own is enough, and it is named exactly.

    Two ownership classes reach this boundary. A ``machine-owned`` scaffold field
    is refused when it CONFLICTS with the value the machine derived. A
    ``ledger-assigned`` field is refused on any value at all, because the append
    has not happened and no correct value exists yet. Both must name the offending
    field and carry the supplied value as evidence, and the parametrization pins
    WHICH class each field belongs to so a field silently changing owners cannot
    pass unnoticed.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    path:
        Dotted field seeded with a fixture placeholder.
    supplied:
        The exact fixture placeholder value.
    owner:
        Expected ownership class named in the refusal.
    """

    request_path, result = _request_and_result(tmp_path)
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    candidate = deepcopy(result)
    if path.startswith("checker."):
        candidate["checker"] = {path.removeprefix("checker."): supplied}
    else:
        candidate[path] = supplied

    with pytest.raises(CheckerDispatchError) as excinfo:
        apply_machine_owned_gate_fields(
            candidate,
            envelope,
            started_at="2026-07-30T18:00:00Z",
            finished_at="2026-07-30T18:00:01Z",
        )

    message = str(excinfo.value)
    assert f"{owner} field {path}=" in message
    assert json.dumps(supplied) in message
    other = "machine-owned" if owner == "ledger-assigned" else "ledger-assigned"
    assert f"{other} field {path}=" not in message


def test_omitting_every_machine_owned_field_stays_free(tmp_path: Path) -> None:
    """The 2026-07-29 tolerance is intact: omission is still never an error.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, result = _request_and_result(tmp_path)
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    candidate = deepcopy(result)
    for field in machine_owned_gate_fields(envelope):
        candidate.pop(field, None)
    candidate.pop("checker", None)

    stamped = apply_machine_owned_gate_fields(
        candidate,
        envelope,
        started_at="2026-07-30T18:00:00Z",
        finished_at="2026-07-30T18:00:01Z",
    )

    assert validate_checker_result_mapping(stamped, envelope)["gate_id"] == (
        machine_owned_gate_fields(envelope)["gate_id"]
    )


def _reseal(request_path: Path, **changes: Any) -> Path:
    """Rewrite one request with changed fields and a repaired self-hash.

    The read-root checks sit downstream of the envelope-hash check, so a naive
    mutation would be rejected as tampering before the check under test ever
    ran. Resealing isolates the guard being exercised.

    Parameters
    ----------
    request_path:
        Exact wrapper request to rewrite in place.
    changes:
        Envelope fields to replace.

    Returns
    -------
    pathlib.Path
        The same request path, rewritten.
    """

    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    envelope.update(changes)
    envelope["envelope_sha256"] = stable_hash(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )
    request_path.write_text(json.dumps(envelope), encoding="utf-8")
    return request_path


def _reject_detail(request_path: Path) -> str:
    """Run one request that must never reach Codex and return its refusal detail.

    Parameters
    ----------
    request_path:
        Exact wrapper request.

    Returns
    -------
    str
        Status-sidecar detail explaining the refusal.
    """

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Fail loudly if preflight lets a bad declaration reach Codex."""

        del argv, last_message, timeout
        raise AssertionError("preflight must reject before invoking Codex")

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        diagnostic_stream=StringIO(),
    )
    assert exit_code is OperatorExitCode.PERMANENT_CONTRACT_REJECTION
    detail = _status(request_path)["detail"]
    assert isinstance(detail, str)
    return detail


def test_declared_read_roots_name_the_frozen_source_not_the_prompt(tmp_path: Path) -> None:
    """The declaration names what the checker reads, and only that.

    The checker re-derives every literal excerpt from frozen bytes under the
    author ``source-cas`` tree, so that root must be declared. It never opens the
    repository ``prompts`` directory, because ``_build_prompt`` inlines the frozen
    text into the argv, so that directory must not be declared.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, _result = _request_and_result(tmp_path)
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    read_roots = envelope["allowed_read_roots"]

    model_dir = Path(envelope["items"][0]["model_dir"])
    assert str(model_dir.parent / "source-cas") in read_roots
    assert str(PROMPT_PATH.parent.resolve()) not in read_roots
    assert str(request_path.parent.resolve()) in read_roots


def test_item_without_its_author_root_cannot_declare_a_source_root(tmp_path: Path) -> None:
    """An item naming no author directory is refused, not silently undeclared.

    Falling back to declaring nothing would reinstate the untrue declaration this
    replaces, so the builder refuses instead.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    gate = make_gate(["m_no_root"])
    item = _checker_item_pack(gate["items"][0])
    item.pop("model_dir")

    with pytest.raises(CheckerDispatchError) as excinfo:
        build_metadata_vet_envelope(
            [item],
            gate_round=1,
            output_path=tmp_path / "result.json",
            checker_model=METADATA_CHECKER_MODEL,
            checker_version="current",
            request_nonce="missing-root",
            final_tail=True,
        )

    assert "model_dir" in str(excinfo.value)
    assert "m_no_root" in str(excinfo.value)


def test_declaring_the_frozen_prompt_directory_is_refused(tmp_path: Path) -> None:
    """Re-adding the unread prompt root fails closed rather than passing quietly.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, _result = _request_and_result(tmp_path)
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    _reseal(
        request_path,
        allowed_read_roots=[*envelope["allowed_read_roots"], str(PROMPT_PATH.parent.resolve())],
    )

    assert "frozen prompt directory" in _reject_detail(request_path)


def test_declaring_no_frozen_source_root_is_refused(tmp_path: Path) -> None:
    """A request-directory-only declaration describes a checker that cannot work.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, _result = _request_and_result(tmp_path)
    _reseal(request_path, allowed_read_roots=[str(request_path.parent.resolve())])

    assert "no frozen source root" in _reject_detail(request_path)


def _frozen_prompt_text() -> str:
    """Return the frozen checker prompt with its hard wrapping normalized away.

    The prompt is hand-wrapped, so a phrase under test can straddle a newline.
    Collapsing runs of whitespace lets these checks assert on wording rather than
    on where a line happens to break.

    Returns
    -------
    str
        Prompt text with every whitespace run collapsed to one space.
    """

    return " ".join(PROMPT_PATH.read_text(encoding="utf-8").split())


def test_frozen_prompt_never_orders_a_write_the_sandbox_forbids() -> None:
    """The prompt describes the transport that exists, not one it cannot use.

    The wrapper runs Codex under ``--sandbox read-only`` and reads the verdict
    from the native structured final message. The prompt previously ordered the
    model to write ``result.json`` with a temporary file, fsync and atomic
    rename, and to "End after result.json is written" -- an unreachable
    termination condition inside a read-only sandbox. Reinstating any of that is a
    contract contradiction, so it fails here.
    """

    prompt = _frozen_prompt_text()

    for ordered_write in (
        "Write exactly one UTF-8 JSON object to",
        "atomic rename",
        "fsync",
        "End after result.json is written",
    ):
        assert ordered_write not in prompt, f"prompt re-orders a forbidden write: {ordered_write}"
    assert "Do not write, create, or rename any file" in prompt
    assert "sole `items` field" in prompt


def test_frozen_prompt_never_asks_for_machine_owned_identities() -> None:
    """The prompt claims only what the checker can legitimately judge.

    A locator the machine verifies by dereferencing may be model-supplied; an
    identity the machine derives must be machine-derived. Asking the model for an
    "exact gate identity" sent a live checker into this repository to learn what
    one looks like, where it found and templated a test fixture.
    """

    prompt = _frozen_prompt_text()

    assert "exact work/model/gate identities" not in prompt
    for field in ("gate_id", "gate_identity", "ledger_seq", "result_envelope_sha256"):
        assert field in prompt, f"{field} must be named as machine-owned so it is omitted"
    assert "MACHINE-OWNED -- OMIT these entirely" in prompt
    assert "copy each one VERBATIM from the same item in the envelope's `items` array" in prompt
    assert "Do NOT search the repository, the test suite, or any fixture" in prompt


def test_frozen_prompt_names_every_required_gate_item_field() -> None:
    """Every schema-required item field is named, so none is omitted by accident.

    Telling the checker to omit machine-owned scaffold is only safe while the
    prompt still accounts for everything the schema demands. ``fidelity`` and
    ``rung_check`` are required on EVERY item, including ``metadata_batch`` items
    that carry their not-applicable form, so wording that reads as "supply these
    only for a fidelity envelope" would induce a schema rejection -- discarding a
    complete verdict over a field the model was told to drop, which is the exact
    failure this whole contract exists to prevent. Binding the prompt to the
    schema means a future required field cannot be added without the prompt
    naming it.
    """

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1] / "schemas" / "gate-v3.schema.json"
        ).read_text(encoding="utf-8")
    )
    required = schema["$defs"]["item"]["required"]
    prompt = _frozen_prompt_text()

    missing = [field for field in required if field not in prompt]
    assert not missing, f"frozen prompt never names required gate item fields: {missing}"


def test_frozen_prompt_names_every_terminal_disposition_key() -> None:
    """The closed terminal block's key set is spelled out, not left to be guessed.

    ``terminal_disposition`` is the checker's own judgment, and the prompt forbids
    looking the gate's shape up in this repository -- so the prompt text is the only
    place the key set can come from. It previously said "the closed
    terminal_disposition item" without ever naming a key, and a 10-model pilot rung
    lost all nine terminal verdicts to that gap: each checker reached for the
    *author-result's* vocabulary instead (``arm`` for ``kind``, ``result_sha256`` for
    ``author_result_sha256``, ``reason`` for ``findings``) and the closed block
    correctly refused every one. The gate was right; the prompt was silent. Naming the
    keys here means a future required key cannot be added without the prompt teaching
    it.
    """

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1] / "schemas" / "gate-v3.schema.json"
        ).read_text(encoding="utf-8")
    )
    block = schema["$defs"]["terminal_disposition"]
    prompt = _frozen_prompt_text()

    missing = [field for field in block["required"] if field not in prompt]
    assert not missing, f"frozen prompt never names terminal_disposition keys: {missing}"

    # The block is closed, so "named" must mean the full set: a prompt that listed only
    # some keys would still leave the rest to be invented.
    assert set(block["required"]) == set(block["properties"])

    # The predicate is a closed enum the checker cannot derive from anywhere else.
    for predicate in block["properties"]["predicate"]["enum"]:
        assert predicate in prompt, f"prompt never names the {predicate!r} predicate"


def test_forged_campaign_lineage_is_refused_before_publication(tmp_path: Path) -> None:
    """A copied envelope-bound identity is checked, not taken on the model's word.

    ``campaign_root_work_id`` is schema-required so it is always present, but its
    value was compared to the envelope only by
    ``driver_models._require_gate_bindings``, which the metadata and fidelity
    lanes reach and the terminal lane does not. Telling the checker to copy a
    field verbatim is only meaningful while the copy is verified.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, result = _request_and_result(tmp_path)
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    forged = apply_machine_owned_gate_fields(
        deepcopy(result),
        envelope,
        started_at="2026-07-30T18:00:00Z",
        finished_at="2026-07-30T18:00:01Z",
    )
    assert validate_checker_result_mapping(deepcopy(forged), envelope)

    forged["items"][0]["campaign_root_work_id"] = "work-somebody-elses-campaign"
    forged["result_envelope_sha256"] = compute_result_envelope_sha256(forged)

    with pytest.raises(CheckerDispatchError) as excinfo:
        validate_checker_result_mapping(forged, envelope)

    assert "campaign_root_work_id" in str(excinfo.value)


def _decided(result: dict[str, Any], index: int = 0) -> dict[str, Any]:
    """Return the result with one item honestly decided ``inaccurate``.

    Parameters
    ----------
    result:
        Compliant candidate whose items all read ``accurate``.
    index:
        Item to flip.

    Returns
    -------
    dict[str, Any]
        Result whose flipped item satisfies the verdict-precedence rule.
    """

    decided = deepcopy(result)
    item = decided["items"][index]
    item["field_checks"][0]["verdict"] = "inaccurate"
    item["field_checks"][0]["required_repair"] = "ground or type-empty the field"
    item["verdict"] = "inaccurate"
    item["required_repairs"] = ["ground or type-empty the field"]
    return decided


def _events(request_path: Path) -> list[dict[str, Any]]:
    """Return every parsed telemetry event for one request.

    Parameters
    ----------
    request_path:
        Exact wrapper request.

    Returns
    -------
    list[dict[str, Any]]
        Telemetry events in emission order.
    """

    return [
        json.loads(line)
        for line in telemetry_path(request_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_metadata_fidelity_is_machine_filled_never_checker_authored(tmp_path: Path) -> None:
    """A schema-honoring metadata checker omits fidelity; the wrapper stamps it.

    The metadata output schema no longer contains the block at all, so omission
    is not merely tolerated -- it is the only representable answer -- and the
    published gate must still satisfy ``gate.v3``'s per-item requirement through
    the machine fill.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, result = _request_and_result(tmp_path)
    authored = deepcopy(result)
    del authored["items"][0]["fidelity"]

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject one schema-honoring metadata answer without a fidelity block."""

        del argv, timeout
        last_message.write_bytes(_native_final_message(authored))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.SUCCESS
    published = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert published["items"][0]["fidelity"] == metadata_fidelity_state()
    # A compliant omission is not a "normalization": nothing divergent was
    # supplied, so no confusion telemetry is emitted.
    assert not [
        event for event in _events(request_path) if event["event"] == "fidelity-normalized"
    ]


def test_contract_rejection_gets_one_bounded_retry_with_the_exact_error(
    tmp_path: Path,
) -> None:
    """A structural contract refusal is re-asked once, with the error fed back.

    Rung 8 (batch ``metadata-c001cad863d95a67``): the wrapper's only response to
    a refused complete verdict was a permanent exit that terminalized every
    batch member, and the retry that DID exist (binding mismatch) re-ran an
    identical prompt that named nothing. The re-ask mirrors the author-side echo
    retry: bounded to ONE, the refused result is never published, and the next
    attempt's prompt carries the machine validator's exact refusal.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, result = _request_and_result(tmp_path)
    contradictory = deepcopy(result)
    contradictory["items"][0]["field_checks"][0]["verdict"] = "inaccurate"
    # Top-level verdict left "accurate": the decision rule refuses the item.
    prompts: list[str] = []

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject one refused decision, then a compliant one."""

        del timeout
        prompts.append(str(argv[-1]))
        message = contradictory if len(prompts) == 1 else result
        last_message.write_bytes(_native_final_message(message))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.SUCCESS
    assert len(prompts) == 2
    assert "REFUSED BY THE MACHINE VALIDATOR" not in prompts[0]
    assert "REFUSED BY THE MACHINE VALIDATOR" in prompts[1]
    assert "contradicts component verdict" in prompts[1]
    refusals = [
        event for event in _events(request_path) if event["event"] == "contract-rejection-refused"
    ]
    assert [event["retrying"] for event in refusals] == [True]
    assert "contradicts component verdict" in refusals[0]["detail"]
    assert _status(request_path)["classification"] == "success"


def test_persistent_contract_rejection_stays_permanent_after_one_retry(
    tmp_path: Path,
) -> None:
    """The re-ask is bounded: a second refusal with the error named is terminal.

    A checker that violates its contract twice, the second time with the exact
    defect spelled out in its prompt, is systematically divergent -- the
    permanent classification, sidecar detail, and absent result are all exactly
    the historical refusal.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, result = _request_and_result(tmp_path)
    contradictory = deepcopy(result)
    contradictory["items"][0]["field_checks"][0]["verdict"] = "inaccurate"
    calls = 0

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject the same refused decision on every attempt."""

        del argv, timeout
        nonlocal calls
        calls += 1
        last_message.write_bytes(_native_final_message(contradictory))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.PERMANENT_CONTRACT_REJECTION
    assert calls == 2
    assert not (tmp_path / "result.json").exists()
    status = _status(request_path)
    assert status["classification"] == "permanent-contract-rejection"
    assert "contradicts component verdict" in status["detail"]
    refusals = [
        event for event in _events(request_path) if event["event"] == "contract-rejection-refused"
    ]
    assert [event["retrying"] for event in refusals] == [True, False]


def test_an_adverse_merits_verdict_is_published_never_reasked(tmp_path: Path) -> None:
    """The retry boundary is shape, not merits: ``inaccurate`` publishes at once.

    A validated adverse verdict is a SUCCESS publication. No code path may spend
    an attempt because the answer was unwelcome -- that boundary is what keeps
    the bounded re-ask from ever becoming verdict shopping.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, result = _request_and_result(tmp_path)
    adverse = _decided(result)
    calls = 0

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject one compliant adverse verdict."""

        del argv, timeout
        nonlocal calls
        calls += 1
        last_message.write_bytes(_native_final_message(adverse))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.SUCCESS
    assert calls == 1
    published = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert published["items"][0]["verdict"] == "inaccurate"
    events = _events(request_path)
    assert not [event for event in events if event["event"] == "contract-rejection-refused"]
    assert not [event for event in events if event["event"] == "binding-mismatch-refused"]


def test_rung8_metadata_batch_now_flows_through_both_recorded_slips(tmp_path: Path) -> None:
    """Replay of ``metadata-c001cad863d95a67``: m10517 and m9666 survive both slips.

    The frozen rung-8 telemetry records exactly two checker faults, one per
    attempt, and the second killed both models at 12:27:32 in the same second:

    1. attempt 1 -- ``checker item m9666 mismatched binding: fidelity_identity``
       (the envelope's value is ``null``; the checker echoed a stale identity
       from the prior gate round);
    2. attempt 2 -- ``gate.v3 validation failed at items[0].fidelity.verdict ...
       'not-applicable' was expected`` (a real fidelity verdict on a metadata
       gate), classified ``permanent-contract-rejection``, exit 64.

    Under the fixed wrapper the same two emissions produce a published gate:
    the binding flub is retried WITH the refusal named in the next prompt, and
    the machine-owned fidelity block is stamped so the illegal verdict never
    reaches ``gate.v3``. Both items' adverse merits verdicts survive verbatim --
    the fix moves the models forward to their repair round, it does not bless
    them.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    gate = make_gate(["m10517", "m9666"])
    packs = [_checker_item_pack(item) for item in gate["items"]]
    envelope = build_metadata_vet_envelope(
        packs,
        gate_round=1,
        output_path=tmp_path / "result.json",
        checker_model=required_checker_model(GateKind.METADATA_BATCH),
        checker_version="current",
        request_nonce="rung8-replay",
        final_tail=True,
    )
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
    for field in (*machine_owned_gate_fields(envelope), *LEDGER_ASSIGNED_GATE_FIELDS):
        gate.pop(field, None)
    gate.pop("checker", None)
    gate.pop("result_envelope_sha256", None)
    # Both models' recorded merits outcome: metadata `inaccurate` with concrete
    # repairs (m10517: unsupported lineage/paradigm; m9666: original_framework).
    honest = _decided(_decided(gate, 0), 1)

    # Attempt 1: the recorded stale-identity echo. The envelope's
    # ``fidelity_identity`` for a metadata item is null; the checker copied a
    # non-null identity out of the prior round's files.
    stale_echo = deepcopy(honest)
    stale_echo["items"][1]["fidelity_identity"] = (
        "sha256:96c9a32ece5c7f085d73a2d72436ee04e2c019ba6b95bf63eb7b7c27e751b1c3"
    )
    # Attempt 2: the recorded illegal fidelity verdict on items[0].
    illegal_fidelity = deepcopy(honest)
    illegal_fidelity["items"][0]["fidelity"] = {
        "required": False,
        "verdict": "cannot-verify",
        "material_checks": [],
        "unsupported_choices": [],
        "contradictions": [],
        "omissions": [],
        "permanent_scar": False,
    }
    prompts: list[str] = []

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject the two recorded rung-8 emissions in their recorded order."""

        del timeout
        prompts.append(str(argv[-1]))
        message = stale_echo if len(prompts) == 1 else illegal_fidelity
        last_message.write_bytes(_native_final_message(message))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    # The batch that died in one second now publishes.
    assert exit_code is OperatorExitCode.SUCCESS
    assert len(prompts) == 2
    # The retry carried the exact recorded refusal, not a blind re-roll.
    assert "REFUSED BY THE MACHINE VALIDATOR" in prompts[1]
    assert "mismatched binding: fidelity_identity" in prompts[1]
    # The published gate is what the DRIVER validates before routing; both
    # models flow to their repair rounds with their adverse verdicts intact.
    published = validate_checker_result(tmp_path / "result.json", envelope)
    by_id = {item["stable_id"]: item for item in published["items"]}
    assert set(by_id) == {"m10517", "m9666"}
    for stable_id, item in by_id.items():
        assert item["verdict"] == "inaccurate", stable_id
        assert item["fidelity"] == metadata_fidelity_state(), stable_id
    assert by_id["m9666"]["fidelity_identity"] is None
    events = _events(request_path)
    mismatches = [event for event in events if event["event"] == "binding-mismatch-refused"]
    assert [event["attempt"] for event in mismatches] == [1]
    assert "fidelity_identity" in mismatches[0]["detail"]
    normalized = [event for event in events if event["event"] == "fidelity-normalized"]
    assert [event["attempt"] for event in normalized] == [2]
    assert normalized[0]["stable_ids"] == ["m10517"]
    assert _status(request_path)["classification"] == "success"

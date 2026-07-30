"""Targeted checker-wrapper protocol, tiering, and fault-injection tests."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pytest

from menagerie.crawler.checker_dispatch import (
    PROMPT_PATH,
    CheckerDispatchError,
    apply_machine_owned_gate_fields,
    build_fidelity_envelope,
    build_metadata_vet_envelope,
    compute_result_envelope_sha256,
    machine_owned_gate_fields,
    validate_checker_result_mapping,
)
from menagerie.crawler.constants import GateKind
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.operator_checker import (
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
    OperatorExitCode,
    status_sidecar_path,
)
from menagerie.crawler.tests.conftest import make_gate


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
    for field in machine_owned_gate_fields(envelope):
        gate.pop(field, None)
    gate.pop("checker", None)
    gate.pop("result_envelope_sha256", None)
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
    return request_path, gate


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
        assert timeout == 180.0
        last_message.write_bytes(
            canonical_json_bytes({"result_json": canonical_json_bytes(result).decode("utf-8")})
            + b"\n"
        )
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


def test_native_output_schema_is_strict_one_field_transport() -> None:
    """Native coercion stays simple while full gate validation remains local."""

    assert _native_output_schema() == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "result_json": {
                "type": "string",
                "description": (
                    "Compact JSON serialization of the complete menagerie.crawler.gate.v3 "
                    "candidate."
                ),
            }
        },
        "required": ["result_json"],
    }


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
    laundered = dict(gate)
    laundered.update(machine_owned_gate_fields(envelope))
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
    # real value, and the fabricated gate now validates clean.
    assert laundered["gate_id"] != "gate-1"
    assert laundered["payload_sha256"] != "sha256:" + "a" * 64
    assert laundered["checker"]["version"] == "current"
    validated = validate_checker_result_mapping(laundered, envelope)
    assert validated["gate_id"] == machine_owned_gate_fields(envelope)["gate_id"]


def test_fixture_templated_gate_is_refused_and_attributed_to_the_checker(
    tmp_path: Path,
) -> None:
    """The same candidate is now refused, and the refusal names the checker.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    request_path, _envelope, gate = _fixture_templated_candidate(tmp_path)

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject the fixture-templated gate as the native final answer."""

        del argv, timeout
        last_message.write_bytes(
            canonical_json_bytes({"result_json": canonical_json_bytes(gate).decode("utf-8")})
            + b"\n"
        )
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
    detail = status["detail"]
    # Attribution: the CHECKER supplied it. Evidence: the field and the value.
    assert detail.startswith("checker supplied the machine-owned field ")
    assert "templated rather than derived" in detail
    assert any(
        f"field {field}=" in detail
        for field in ("gate_id", "payload_sha256", "gate_identity", "ledger_seq")
    )


@pytest.mark.parametrize(
    ("path", "supplied"),
    [
        ("payload_sha256", "sha256:" + "a" * 64),
        ("gate_id", "gate-1"),
        ("checker.version", "test"),
        ("checker.model", "codex"),
        ("checker.started_at", "2026-07-14T12:00:00Z"),
    ],
)
def test_each_fixture_placeholder_is_refused_with_its_own_value_as_evidence(
    tmp_path: Path, path: str, supplied: str
) -> None:
    """One conflicting machine-owned field is enough, and it is named exactly.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    path:
        Dotted machine-owned field seeded with a fixture placeholder.
    supplied:
        The exact fixture placeholder value.
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
    assert f"machine-owned field {path}=" in message
    assert json.dumps(supplied) in message


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
    assert "sole `result_json` field" in prompt


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

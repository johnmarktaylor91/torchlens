"""Headless Codex wrapper for immutable menagerie checker envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence, TextIO

from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    PROMPT_PATH,
    validate_checker_result_mapping,
)
from menagerie.crawler.constants import GateKind
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import JsonObject
from menagerie.crawler.operator_protocol import (
    OPERATOR_ATTEMPT_TIMEOUT_SECONDS,
    OPERATOR_MAX_ATTEMPTS,
    OPERATOR_PROTOCOL_VERSION,
    OPERATOR_REASONING_EFFORT,
    OperatorExitCode,
    OperatorProtocolError,
    append_telemetry,
    emit_combined_tail,
    load_absolute_request,
    publish_json_atomic,
    write_status,
)

CHECKER_TIMEOUT_SECONDS = float(OPERATOR_ATTEMPT_TIMEOUT_SECONDS)
CHECKER_MAX_ATTEMPTS = OPERATOR_MAX_ATTEMPTS
CHECKER_EFFORT = OPERATOR_REASONING_EFFORT
METADATA_CHECKER_MODEL = "gpt-5.6-terra"
FIDELITY_CHECKER_MODEL = "gpt-5.6-sol"
TERMINAL_CHECKER_MODEL = "gpt-5.6-terra"

_MODEL_METADATA_ERROR = "model metadata for"
_QUOTA_MARKERS = ("quota", "usage limit")
_TRANSIENT_MARKERS = (
    "reconnecting...",
    "stream disconnected",
    "error sending request",
)
_PERMANENT_MARKERS = (
    "invalid_request_error",
    "model is not supported",
    "failed to read output schema file",
)
_STARTED_MARKERS = ('"type":"thread.started"', '"type":"turn.started"')
_RESET_PATTERN = re.compile(r"try again at\s+([^\r\n]+?)\.", re.IGNORECASE)
_SERVER_ERROR_PATTERN = re.compile(r"(?:^|[^0-9])5[0-9]{2}(?:[^0-9]|$)")

InvokeCodex = Callable[[Sequence[str], Path, float], "CodexAttempt"]
Sleep = Callable[[float], None]
Now = Callable[[], datetime]


class FailureKind(str, Enum):
    """Closed wrapper classification of one Codex attempt."""

    SUCCESS = "success"
    TRANSIENT = "retryable-infrastructure"
    PERMANENT = "permanent-contract-rejection"
    QUOTA = "rate-or-quota-pause"
    UNAVAILABLE = "service-unavailable"


@dataclass(frozen=True)
class CodexAttempt:
    """Captured result of one process-group-contained Codex invocation."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    unavailable: bool = False


@dataclass(frozen=True)
class AttemptClassification:
    """Classification and optional reset facts for one Codex attempt."""

    kind: FailureKind
    reset_text: str | None = None


def required_checker_model(gate_kind: GateKind | str) -> str:
    """Return the locked full model string for one gate kind.

    Parameters
    ----------
    gate_kind:
        Metadata, fidelity, or terminal-disposition gate.

    Returns
    -------
    str
        Exact ``gpt-5.6-<tier>`` identifier.
    """

    kind = gate_kind if isinstance(gate_kind, GateKind) else GateKind(gate_kind)
    return {
        GateKind.METADATA_BATCH: METADATA_CHECKER_MODEL,
        GateKind.FIDELITY: FIDELITY_CHECKER_MODEL,
        GateKind.TERMINAL_DISPOSITION: TERMINAL_CHECKER_MODEL,
    }[kind]


def classify_codex_attempt(attempt: CodexAttempt) -> AttemptClassification:
    """Classify one Codex result by merged message content, not exit code alone.

    Parameters
    ----------
    attempt:
        Captured process result.

    Returns
    -------
    AttemptClassification
        Closed retry/permanent/quota classification.
    """

    failure_text = _codex_failure_text(attempt.stdout, attempt.stderr)
    lowered = failure_text.lower()
    reset_match = _RESET_PATTERN.search(failure_text)
    reset_text = reset_match.group(1).strip(" \\\"'") if reset_match is not None else None
    if _MODEL_METADATA_ERROR in lowered and "not found" in lowered:
        return AttemptClassification(FailureKind.PERMANENT)
    if any(marker in lowered for marker in _QUOTA_MARKERS):
        return AttemptClassification(FailureKind.QUOTA, reset_text)
    if attempt.timed_out or any(marker in lowered for marker in _TRANSIENT_MARKERS):
        return AttemptClassification(FailureKind.TRANSIENT)
    if _SERVER_ERROR_PATTERN.search(lowered):
        return AttemptClassification(FailureKind.TRANSIENT)
    if any(marker in lowered for marker in _PERMANENT_MARKERS):
        return AttemptClassification(FailureKind.PERMANENT)
    if attempt.unavailable:
        return AttemptClassification(FailureKind.UNAVAILABLE)
    if attempt.returncode == 0:
        return AttemptClassification(FailureKind.SUCCESS)
    compact_stdout = attempt.stdout.lower().replace(" ", "")
    if not any(marker in compact_stdout for marker in _STARTED_MARKERS):
        return AttemptClassification(FailureKind.PERMANENT)
    return AttemptClassification(FailureKind.UNAVAILABLE)


def _codex_failure_text(stdout: str, stderr: str) -> str:
    """Extract only wrapper-visible Codex errors from a JSONL event stream.

    Parameters
    ----------
    stdout, stderr:
        Captured Codex streams. Stderr remains authoritative for preflight
        failures; stdout contributes only explicit error events or non-JSON text.

    Returns
    -------
    str
        Failure-classification text excluding agent messages and tool payloads.
    """

    messages = [stderr] if stderr else []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            messages.append(line)
            continue
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("type")
        if event_type == "error" and isinstance(event.get("message"), str):
            messages.append(str(event["message"]))
        elif event_type == "turn.failed":
            error = event.get("error")
            if isinstance(error, Mapping) and isinstance(error.get("message"), str):
                messages.append(str(error["message"]))
    return "\n".join(messages)


def build_codex_argv(
    *,
    workdir: Path,
    model: str,
    schema_path: Path,
    last_message_path: Path,
    prompt: str,
) -> tuple[str, ...]:
    """Build the empirically settled headless Codex argv.

    Parameters
    ----------
    workdir:
        Read-only checker working directory.
    model:
        Exact locked full model identifier.
    schema_path:
        Native structured-output JSON Schema.
    last_message_path:
        Temporary final-message destination.
    prompt:
        Complete frozen prompt and request envelope.

    Returns
    -------
    tuple[str, ...]
        Non-shell argv.
    """

    return (
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "-C",
        str(workdir),
        "-m",
        model,
        "-c",
        f"model_reasoning_effort={CHECKER_EFFORT}",
        "--output-schema",
        str(schema_path),
        "-o",
        str(last_message_path),
        "--json",
        prompt,
    )


def execute_checker_request(
    request_path: Path,
    *,
    invoke: InvokeCodex | None = None,
    sleep: Sleep = time.sleep,
    now: Now | None = None,
    diagnostic_stream: TextIO = sys.stderr,
) -> OperatorExitCode:
    """Execute one absolute checker request under the shared operator protocol.

    Parameters
    ----------
    request_path:
        Sole absolute request path supplied by the driver.
    invoke:
        Injectable subprocess boundary used by fault tests.
    sleep:
        Injectable retry backoff.
    now:
        Injectable UTC clock for quota fallback.
    diagnostic_stream:
        Merged bounded diagnostic destination.

    Returns
    -------
    OperatorExitCode
        Closed wrapper outcome.
    """

    runner = invoke or _invoke_codex
    clock = now or (lambda: datetime.now(timezone.utc))
    try:
        envelope = load_absolute_request(request_path)
        request_sha256, output_path, model, deadline = _validate_request(envelope, request_path)
    except (OperatorProtocolError, CheckerDispatchError, OSError, ValueError) as exc:
        detail = str(exc)
        write_status(
            request_path,
            exit_code=OperatorExitCode.PERMANENT_CONTRACT_REJECTION,
            classification=FailureKind.PERMANENT.value,
            request_sha256=None,
            attempts=0,
            detail=detail,
        )
        append_telemetry(
            request_path,
            {"event": "request-rejected", "classification": FailureKind.PERMANENT.value},
        )
        diagnostic_stream.write(f"{detail}\n")
        return OperatorExitCode.PERMANENT_CONTRACT_REJECTION

    prompt = _build_prompt(envelope, request_path)
    with tempfile.TemporaryDirectory(prefix=".checker-", dir=output_path.parent) as temporary:
        temp_root = Path(temporary)
        schema_path = temp_root / "gate-output-schema.json"
        last_message_path = temp_root / "last-message.json"
        publish_json_atomic(schema_path, _native_output_schema())
        argv = build_codex_argv(
            workdir=request_path.parent,
            model=model,
            schema_path=schema_path,
            last_message_path=last_message_path,
            prompt=prompt,
        )
        last_detail = ""
        for attempt_number in range(1, CHECKER_MAX_ATTEMPTS + 1):
            remaining = (deadline - clock().astimezone(timezone.utc)).total_seconds()
            if remaining <= 0:
                return _finish(
                    request_path,
                    OperatorExitCode.RETRYABLE_INFRASTRUCTURE,
                    FailureKind.TRANSIENT,
                    request_sha256,
                    attempt_number - 1,
                    "checker operator deadline expired before the next attempt",
                )
            attempt = runner(argv, last_message_path, min(CHECKER_TIMEOUT_SECONDS, remaining))
            classification = classify_codex_attempt(attempt)
            last_detail = emit_combined_tail(
                attempt.stdout,
                attempt.stderr,
                stream=diagnostic_stream,
            )
            append_telemetry(
                request_path,
                {
                    "event": "codex-attempt",
                    "attempt": attempt_number,
                    "returncode": attempt.returncode,
                    "classification": classification.kind.value,
                    "timed_out": attempt.timed_out,
                    "detail": last_detail,
                },
            )
            if classification.kind is FailureKind.SUCCESS:
                try:
                    result = _load_last_message(last_message_path)
                    validate_checker_result_mapping(result, envelope)
                except (
                    CheckerDispatchError,
                    OSError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ) as exc:
                    return _finish(
                        request_path,
                        OperatorExitCode.PERMANENT_CONTRACT_REJECTION,
                        FailureKind.PERMANENT,
                        request_sha256,
                        attempt_number,
                        str(exc),
                    )
                publish_json_atomic(output_path, result)
                return _finish(
                    request_path,
                    OperatorExitCode.SUCCESS,
                    FailureKind.SUCCESS,
                    request_sha256,
                    attempt_number,
                    "checker result validated and published atomically",
                )
            if classification.kind is FailureKind.QUOTA:
                reset_at, observation = _quota_reset(classification.reset_text, clock())
                return _finish(
                    request_path,
                    OperatorExitCode.RATE_OR_QUOTA_PAUSE,
                    FailureKind.QUOTA,
                    request_sha256,
                    attempt_number,
                    last_detail,
                    reset_at=reset_at,
                    reset_observation=observation,
                )
            if classification.kind is FailureKind.PERMANENT:
                return _finish(
                    request_path,
                    OperatorExitCode.PERMANENT_CONTRACT_REJECTION,
                    FailureKind.PERMANENT,
                    request_sha256,
                    attempt_number,
                    last_detail,
                )
            if classification.kind is FailureKind.UNAVAILABLE:
                return _finish(
                    request_path,
                    OperatorExitCode.SERVICE_UNAVAILABLE,
                    FailureKind.UNAVAILABLE,
                    request_sha256,
                    attempt_number,
                    last_detail,
                )
            if attempt_number < CHECKER_MAX_ATTEMPTS:
                sleep(float(2 ** (attempt_number - 1)))
        return _finish(
            request_path,
            OperatorExitCode.RETRYABLE_INFRASTRUCTURE,
            FailureKind.TRANSIENT,
            request_sha256,
            CHECKER_MAX_ATTEMPTS,
            last_detail,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker wrapper CLI with exactly one absolute request argument.

    Parameters
    ----------
    argv:
        Optional argument vector excluding the program name.

    Returns
    -------
    int
        Closed operator exit code.
    """

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        sys.stderr.write(
            "usage: python -m menagerie.crawler.operator_checker /absolute/request.json\n"
        )
        return int(OperatorExitCode.PERMANENT_CONTRACT_REJECTION)
    return int(execute_checker_request(Path(arguments[0])))


def _validate_request(
    envelope: Mapping[str, Any], request_path: Path
) -> tuple[str, Path, str, datetime]:
    """Validate checker-specific request bindings and output confinement.

    Parameters
    ----------
    envelope:
        Parsed checker envelope.
    request_path:
        Exact absolute request path.

    Returns
    -------
    tuple[str, pathlib.Path, str, datetime.datetime]
        Request digest, exact output path, locked model, and UTC deadline.

    Raises
    ------
    CheckerDispatchError
        If any identity, tier, or path binding is invalid.
    """

    if envelope.get("envelope_version") != "menagerie.crawler.checker-envelope.v3":
        raise CheckerDispatchError("unsupported checker envelope version")
    if envelope.get("operator_protocol_version") != OPERATOR_PROTOCOL_VERSION:
        raise CheckerDispatchError("unsupported common operator protocol version")
    expected_hash = stable_hash(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )
    if envelope.get("envelope_sha256") != expected_hash:
        raise CheckerDispatchError("checker envelope hash mismatch")
    try:
        kind = GateKind(str(envelope["gate_kind"]))
    except (KeyError, ValueError) as exc:
        raise CheckerDispatchError("checker envelope has an invalid gate_kind") from exc
    checker = envelope.get("checker")
    if not isinstance(checker, Mapping):
        raise CheckerDispatchError("checker envelope lacks its exact checker identity")
    model = required_checker_model(kind)
    if checker.get("model") != model:
        raise CheckerDispatchError(
            f"{kind.value} requires exact checker model {model!r}, got {checker.get('model')!r}"
        )
    if checker.get("provider") != "openai":
        raise CheckerDispatchError("checker envelope provider must be openai")
    if envelope.get("operator_model") != model:
        raise CheckerDispatchError("operator model does not match checker identity")
    if envelope.get("work_generation_identity") != envelope.get("request_nonce"):
        raise CheckerDispatchError("operator work generation does not bind the request nonce")
    effort_grant = envelope.get("effort_grant")
    expected_effort = {
        "max_attempts": CHECKER_MAX_ATTEMPTS,
        "attempt_timeout_seconds": int(CHECKER_TIMEOUT_SECONDS),
        "reasoning_effort": CHECKER_EFFORT,
    }
    if effort_grant != expected_effort:
        raise CheckerDispatchError("checker effort grant does not match the locked cap")
    deadline_value = envelope.get("deadline_at")
    if not isinstance(deadline_value, str):
        raise CheckerDispatchError("checker envelope lacks an exact deadline")
    try:
        deadline = datetime.fromisoformat(deadline_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CheckerDispatchError("checker deadline is not an ISO timestamp") from exc
    if deadline.tzinfo is None:
        raise CheckerDispatchError("checker deadline must be timezone-aware")
    output_value = envelope.get("required_output_path")
    root_value = envelope.get("allowed_output_root")
    if not isinstance(output_value, str) or not isinstance(root_value, str):
        raise CheckerDispatchError("checker envelope lacks exact output paths")
    output_path = Path(output_value)
    output_root = Path(root_value)
    if not output_path.is_absolute() or not output_root.is_absolute():
        raise CheckerDispatchError("checker output paths must be absolute")
    if envelope.get("allowed_write_root") != str(output_root) or envelope.get(
        "operator_required_output_path"
    ) != str(output_path):
        raise CheckerDispatchError("common operator write paths do not match checker paths")
    read_roots = envelope.get("allowed_read_roots")
    if (
        not isinstance(read_roots, list)
        or any(not isinstance(value, str) or not Path(value).is_absolute() for value in read_roots)
        or str(request_path.parent.resolve()) not in read_roots
        or str(PROMPT_PATH.parent.resolve()) not in read_roots
    ):
        raise CheckerDispatchError("checker allowed read roots are incomplete or non-absolute")
    if (
        output_path.resolve().parent != output_root.resolve()
        or output_path.name != "result.json"
        or output_root.resolve() != request_path.parent.resolve()
    ):
        raise CheckerDispatchError("checker output must be request-sibling result.json")
    if output_root.is_symlink():
        raise CheckerDispatchError("checker output root cannot be a symlink")
    return expected_hash, output_path.resolve(), model, deadline.astimezone(timezone.utc)


def _build_prompt(envelope: Mapping[str, Any], request_path: Path) -> str:
    """Bind the frozen checker instructions to one complete request object.

    Parameters
    ----------
    envelope:
        Validated checker envelope.
    request_path:
        Exact request path used for audit identity.

    Returns
    -------
    str
        One-shot prompt argument.
    """

    frozen = PROMPT_PATH.read_text(encoding="utf-8")
    return (
        f"{frozen}\n\n"
        f"WORK_ENVELOPE_PATH={request_path}\n"
        "The outer read-only wrapper publishes result.json. Do not write files. "
        "Read the exact envelope at WORK_ENVELOPE_PATH, construct the complete gate.v3 "
        "object, serialize it as compact JSON, and return it in the output schema's sole "
        "result_json string field."
    )


def _native_output_schema() -> JsonObject:
    """Return the strict native structured-output transport schema.

    Returns
    -------
    dict[str, Any]
        One-field transport schema. The decoded gate still passes the complete
        repository schema and semantic validator before publication.
    """

    return {
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


def _invoke_codex(
    argv: Sequence[str], last_message_path: Path, timeout_seconds: float
) -> CodexAttempt:
    """Run Codex in a fresh process group with an external wall-clock timeout.

    Parameters
    ----------
    argv:
        Exact non-shell Codex argv.
    last_message_path:
        Expected native final-message output, used only for interface symmetry.
    timeout_seconds:
        External process-group wall limit.

    Returns
    -------
    CodexAttempt
        Captured streams and typed timeout/unavailability facts.
    """

    del last_message_path
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return CodexAttempt(127, "", str(exc), unavailable=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return CodexAttempt(process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        del exc
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return CodexAttempt(
            process.returncode if process.returncode is not None else -signal.SIGKILL,
            stdout,
            stderr,
            timed_out=True,
        )


def _load_last_message(path: Path) -> JsonObject:
    """Load the native schema-coerced final answer as exactly one object.

    Parameters
    ----------
    path:
        Codex ``-o`` destination.

    Returns
    -------
    dict[str, Any]
        Parsed checker result.

    Raises
    ------
    CheckerDispatchError
        If the output is absent, symlinked, empty, or not one object.
    """

    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise CheckerDispatchError("Codex did not publish a non-empty final answer")
    transport = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(transport, dict) or set(transport) != {"result_json"}:
        raise CheckerDispatchError("Codex final answer must match the native transport schema")
    result_json = transport.get("result_json")
    if not isinstance(result_json, str):
        raise CheckerDispatchError("Codex result_json must be a string")
    value = json.loads(result_json)
    if not isinstance(value, dict):
        raise CheckerDispatchError("decoded Codex gate must be exactly one JSON object")
    return value


def _quota_reset(reset_text: str | None, now: datetime) -> tuple[str, str]:
    """Normalize an extracted Codex reset string or use the one-hour fallback.

    Parameters
    ----------
    reset_text:
        Free-text value captured after ``try again at``.
    now:
        Current aware instant.

    Returns
    -------
    tuple[str, str]
        UTC reset timestamp and ``observed``/``guessed`` provenance.
    """

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if reset_text is not None:
        parsed = _parse_reset_text(reset_text, now)
        if parsed is not None:
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), "observed"
    fallback = now.astimezone(timezone.utc) + timedelta(hours=1)
    return fallback.isoformat().replace("+00:00", "Z"), "guessed"


def _parse_reset_text(value: str, now: datetime) -> datetime | None:
    """Parse known Codex free-text reset formats without guessing missing fields.

    Parameters
    ----------
    value:
        Extracted reset text.
    now:
        Aware instant supplying the machine-local timezone.

    Returns
    -------
    datetime.datetime | None
        Parsed aware timestamp, or ``None`` for the explicit fallback path.
    """

    normalized = value.strip().replace('\\"', "").strip("\"'")
    iso_candidate = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
    except ValueError:
        parsed = None
    if parsed is not None:
        return (
            parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=now.astimezone().tzinfo)
        )
    for format_string in (
        "%b %d, %Y %I:%M %p",
        "%B %d, %Y %I:%M %p",
        "%b %d, %Y, %I:%M %p",
        "%B %d, %Y, %I:%M %p",
        "%m/%d/%Y, %I:%M %p",
    ):
        try:
            parsed = datetime.strptime(normalized, format_string)
        except ValueError:
            continue
        return parsed.replace(tzinfo=now.astimezone().tzinfo)
    return None


def _finish(
    request_path: Path,
    exit_code: OperatorExitCode,
    classification: FailureKind,
    request_sha256: str,
    attempts: int,
    detail: str,
    *,
    reset_at: str | None = None,
    reset_observation: str | None = None,
) -> OperatorExitCode:
    """Publish final status and telemetry for one wrapper outcome.

    Parameters
    ----------
    request_path:
        Exact request path.
    exit_code:
        Closed process outcome.
    classification:
        Structured classification authority.
    request_sha256:
        Validated request digest.
    attempts:
        External attempts consumed.
    detail:
        Bounded diagnostic.
    reset_at, reset_observation:
        Optional quota wake facts.

    Returns
    -------
    OperatorExitCode
        The supplied exit for direct CLI return.
    """

    write_status(
        request_path,
        exit_code=exit_code,
        classification=classification.value,
        request_sha256=request_sha256,
        attempts=attempts,
        detail=detail,
        reset_at=reset_at,
        reset_observation=reset_observation,
    )
    append_telemetry(
        request_path,
        {
            "event": "operator-finished",
            "exit_code": int(exit_code),
            "classification": classification.value,
            "attempts": attempts,
            "reset_at": reset_at,
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

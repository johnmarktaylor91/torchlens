"""Shared atomic publication and diagnostics for crawler operator wrappers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import IntEnum
import json
from pathlib import Path
import re
from typing import Any, Mapping, TextIO

from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes
from menagerie.crawler.models import JsonObject

OPERATOR_PROTOCOL_VERSION = "menagerie.crawler.operator-protocol.v1"
OPERATOR_STATUS_VERSION = "menagerie.crawler.operator-status.v1"
OPERATOR_DEADLINE_SECONDS = 600
OPERATOR_MAX_ATTEMPTS = 3
OPERATOR_ATTEMPT_TIMEOUT_SECONDS = 180
OPERATOR_REASONING_EFFORT = "high"
TELEMETRY_MAX_BYTES = 64 * 1024
TELEMETRY_RECORD_MAX_CHARS = 4_000

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(
        r"""(?ix)
        ("?(?:api[_-]?key|authorization|password|secret|token)"?\s*[:=]\s*)
        ("?)[^",\s}]+
        """
    ),
    re.compile(r"\b(?:sk|sess)-[A-Za-z0-9_-]{12,}\b"),
)


class OperatorExitCode(IntEnum):
    """Closed process exit codes for crawler operator wrappers."""

    SUCCESS = 0
    PERMANENT_CONTRACT_REJECTION = 64
    RETRYABLE_INFRASTRUCTURE = 75
    RATE_OR_QUOTA_PAUSE = 76
    SERVICE_UNAVAILABLE = 78


class OperatorProtocolError(ValueError):
    """Raised when an operator request violates the shared argv contract."""


def build_operator_fields(
    *,
    work_generation_identity: str,
    model: str,
    allowed_read_roots: tuple[Path, ...],
    allowed_write_root: Path,
    required_output_path: Path,
    now: datetime | None = None,
) -> JsonObject:
    """Build the common request fields shared by operator wrappers.

    Parameters
    ----------
    work_generation_identity:
        Exact request nonce or generation identity.
    model:
        Full provider model identifier.
    allowed_read_roots:
        Declarative absolute read roots for the request.
    allowed_write_root:
        Sole absolute write root.
    required_output_path:
        Exact final result path.
    now:
        Optional aware clock for deterministic tests.

    Returns
    -------
    dict[str, Any]
        Common protocol fields ready to merge into a request body.
    """

    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise OperatorProtocolError("operator request clock must be timezone-aware")
    if not work_generation_identity or not model:
        raise OperatorProtocolError("operator work identity and model must be non-empty")
    read_roots = tuple(path.resolve() for path in allowed_read_roots)
    write_root = allowed_write_root.resolve()
    output_path = required_output_path.resolve()
    if not read_roots or any(not path.is_absolute() for path in read_roots):
        raise OperatorProtocolError("operator read roots must be non-empty and absolute")
    if output_path.parent != write_root:
        raise OperatorProtocolError("operator output must be directly under its write root")
    deadline = instant.astimezone(timezone.utc) + timedelta(seconds=OPERATOR_DEADLINE_SECONDS)
    return {
        "operator_protocol_version": OPERATOR_PROTOCOL_VERSION,
        "work_generation_identity": work_generation_identity,
        "deadline_at": deadline.isoformat().replace("+00:00", "Z"),
        "effort_grant": {
            "max_attempts": OPERATOR_MAX_ATTEMPTS,
            "attempt_timeout_seconds": OPERATOR_ATTEMPT_TIMEOUT_SECONDS,
            "reasoning_effort": OPERATOR_REASONING_EFFORT,
        },
        "allowed_read_roots": [str(path) for path in read_roots],
        "allowed_write_root": str(write_root),
        "operator_model": model,
        "operator_required_output_path": str(output_path),
    }


def load_absolute_request(path: Path) -> JsonObject:
    """Load one non-symlinked JSON object from an absolute request path.

    Parameters
    ----------
    path:
        Exact request path supplied as the wrapper's sole argument.

    Returns
    -------
    dict[str, Any]
        Parsed request object.

    Raises
    ------
    OperatorProtocolError
        If the request path or JSON object is invalid.
    """

    if not path.is_absolute():
        raise OperatorProtocolError("operator request path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise OperatorProtocolError("operator request must be a non-symlinked regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorProtocolError(f"cannot read operator request: {exc}") from exc
    if not isinstance(value, dict):
        raise OperatorProtocolError("operator request must contain exactly one JSON object")
    return value


def publish_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish one canonical JSON object with file and directory synchronization.

    Parameters
    ----------
    path:
        Exact final path.
    value:
        JSON-compatible object to publish.
    """

    atomic_replace_bytes(path, canonical_json_bytes(value) + b"\n")


def status_sidecar_path(request_path: Path) -> Path:
    """Return the deterministic structured-status path for a request.

    Parameters
    ----------
    request_path:
        Exact operator request path.

    Returns
    -------
    pathlib.Path
        Sibling status-sidecar path.
    """

    return request_path.with_name("operator-status.json")


def telemetry_path(request_path: Path) -> Path:
    """Return the deterministic bounded-telemetry path for a request.

    Parameters
    ----------
    request_path:
        Exact operator request path.

    Returns
    -------
    pathlib.Path
        Sibling JSONL telemetry path.
    """

    return request_path.with_name("operator-telemetry.jsonl")


def redact_text(value: str) -> str:
    """Redact common credential forms from diagnostic text.

    Parameters
    ----------
    value:
        Untrusted subprocess or exception text.

    Returns
    -------
    str
        Bounded redacted text.
    """

    redacted = value
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(r"\1\2[REDACTED]", redacted)
        elif pattern.groups == 1:
            redacted = pattern.sub(r"\1 [REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted[-TELEMETRY_RECORD_MAX_CHARS:]


def emit_combined_tail(
    stdout: str,
    stderr: str,
    *,
    stream: TextIO,
) -> str:
    """Merge both subprocess streams and re-emit one bounded redacted tail.

    Parameters
    ----------
    stdout, stderr:
        Captured Codex streams.
    stream:
        Wrapper diagnostic stream, normally ``sys.stderr``.

    Returns
    -------
    str
        Exact merged, bounded, redacted text used for classification.
    """

    combined = "\n".join(part for part in (stderr, stdout) if part)
    tail = redact_text(combined)
    if tail:
        stream.write(tail)
        if not tail.endswith("\n"):
            stream.write("\n")
        stream.flush()
    return tail


def write_status(
    request_path: Path,
    *,
    exit_code: OperatorExitCode,
    classification: str,
    request_sha256: str | None,
    attempts: int,
    detail: str,
    reset_at: str | None = None,
    reset_observation: str | None = None,
) -> JsonObject:
    """Atomically publish the classification-authority sidecar.

    Parameters
    ----------
    request_path:
        Exact request path.
    exit_code:
        Closed operator exit.
    classification:
        Machine-readable outcome class.
    request_sha256:
        Request identity when validation reached it.
    attempts:
        Number of external tool attempts consumed.
    detail:
        Bounded human diagnostic.
    reset_at, reset_observation:
        Extracted or fallback quota reset facts.

    Returns
    -------
    dict[str, Any]
        Published status object.
    """

    status: JsonObject = {
        "protocol_version": OPERATOR_PROTOCOL_VERSION,
        "status_version": OPERATOR_STATUS_VERSION,
        "exit_code": int(exit_code),
        "classification": classification,
        "request_sha256": request_sha256,
        "attempts": attempts,
        "detail": redact_text(detail),
        "reset_at": reset_at,
        "reset_observation": reset_observation,
    }
    publish_json_atomic(status_sidecar_path(request_path), status)
    return status


def append_telemetry(request_path: Path, event: Mapping[str, Any]) -> None:
    """Append one redacted JSONL event while bounding the complete telemetry file.

    Parameters
    ----------
    request_path:
        Exact request path.
    event:
        JSON-compatible telemetry fields.
    """

    path = telemetry_path(request_path)
    redacted_event = _redact_value(dict(event))
    record = canonical_json_bytes(redacted_event) + b"\n"
    try:
        prior = path.read_bytes() if path.is_file() and not path.is_symlink() else b""
    except OSError:
        prior = b""
    retained = (prior + record)[-TELEMETRY_MAX_BYTES:]
    first_newline = retained.find(b"\n")
    if len(prior) + len(record) > TELEMETRY_MAX_BYTES and first_newline >= 0:
        retained = retained[first_newline + 1 :]
    atomic_replace_bytes(path, retained)


def _redact_value(value: Any) -> Any:
    """Recursively redact strings in one telemetry value.

    Parameters
    ----------
    value:
        JSON-compatible candidate.

    Returns
    -------
    Any
        Redacted JSON-compatible value.
    """

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact_value(item) for key, item in value.items()}
    return value

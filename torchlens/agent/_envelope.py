"""Canonical serializer and the one agent-surface envelope (agent memo 3.9).

Every machine payload the agent surface emits -- tool results, errors, CLI
JSON -- rides ONE envelope with a schema string, the package version, a
content-digest artifact identity, request/limits echoes, and an always-present
``truncation`` key. Serialization is CANONICAL: UTF-8, ASCII-safe escaping,
sorted keys, compact separators, ``allow_nan=False`` with non-finite floats as
tagged records -- same artifact + version + arguments -> byte-identical output.

DOCUMENTED-UNSTABLE spellings (naming ratification pending); consumers branch
on the ``schema`` string, never on field presence.
"""

from __future__ import annotations

import json
import math
from typing import Any

#: Envelope stability disclosure carried on every payload.
SCHEMA_STABILITY = "documented-unstable"

#: Schema id of the error envelope.
ERROR_SCHEMA = "torchlens.agent.error.v1"


def _torchlens_version() -> str:
    """Return the installed torchlens version without importing torch."""

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("torchlens")
    except PackageNotFoundError:  # pragma: no cover - editable-install fallback
        return "unknown"


def json_safe(value: Any) -> Any:
    """Recursively project a value into strict-JSON-safe form.

    Non-finite floats become tagged records (``{"nonfinite": "nan"}``) so the
    canonical serializer's ``allow_nan=False`` never sees a bare ``NaN`` --
    the defect the agent memo (3.6) names as otherwise ARRIVING with the
    stats tool. Dict keys are coerced to strings; tuples become lists.

    Parameters
    ----------
    value:
        Arbitrary JSON-adjacent value.

    Returns
    -------
    Any
        Strict-JSON-serializable projection.
    """

    if isinstance(value, float):
        if math.isnan(value):
            return {"nonfinite": "nan"}
        return {"nonfinite": "inf" if value > 0 else "-inf"} if math.isinf(value) else value
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return repr(value)


def canonical_dumps(payload: Any) -> str:
    """Serialize a payload under the determinism contract (agent memo 3.9).

    Sorted keys, compact separators, ASCII-safe escaping, and
    ``allow_nan=False`` so a bare ``NaN``/``Infinity`` literal is a crash
    here, never a silently non-standard document downstream.

    Parameters
    ----------
    payload:
        JSON-safe payload (run through :func:`json_safe` first when the value
        may contain non-finite floats or tuples).

    Returns
    -------
    str
        Canonical JSON text.
    """

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def build_envelope(  # noqa: PLR0913 - the envelope's declared keys, keyword-only by design
    *,
    schema: str,
    data: Any,
    artifact: dict[str, Any] | None = None,
    request: dict[str, Any] | None = None,
    limits: dict[str, Any] | None = None,
    truncation: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    """Assemble the one agent-surface envelope.

    Parameters
    ----------
    schema:
        Versioned payload schema id (``torchlens.agent.<name>.v1``).
    data:
        Tool-specific result record.
    artifact:
        Artifact identity block (``None`` for environment-only results).
    request:
        Normalized request echo.
    limits:
        Applied limits echo.
    truncation:
        Truncation disclosure, or ``None`` when nothing was dropped.
    warnings:
        Human-readable warnings (never load-bearing).
    status:
        Result status token (``"ok"`` unless a floor degradation applies).

    Returns
    -------
    dict[str, Any]
        JSON-safe envelope.
    """

    return json_safe(
        {
            "schema": schema,
            "schema_stability": SCHEMA_STABILITY,
            "torchlens_version": _torchlens_version(),
            "status": status,
            "artifact": artifact,
            "request": request or {},
            "limits": limits or {},
            "data": data,
            "truncation": truncation,
            "warnings": warnings or [],
        }
    )


def build_error_envelope(exc: BaseException) -> dict[str, Any]:
    """Project one exception into the ``torchlens.agent.error.v1`` envelope.

    Exposes the EXISTING typed fields (class name, stable code, remedy,
    severity, structured fields) -- no traceback, no filesystem paths beyond
    what the message already carries.

    Parameters
    ----------
    exc:
        Raised exception (typed TorchLens refusals carry ``fields``).

    Returns
    -------
    dict[str, Any]
        JSON-safe error envelope.
    """

    fields = dict(getattr(exc, "fields", {}) or {})
    code = fields.pop("code", None)
    remedy = fields.pop("remedy", None)
    return json_safe(
        {
            "schema": ERROR_SCHEMA,
            "schema_stability": SCHEMA_STABILITY,
            "torchlens_version": _torchlens_version(),
            "status": "error",
            "error": {
                "class": type(exc).__name__,
                "code": code,
                "remedy": remedy,
                "severity": getattr(exc, "severity", None),
                "message": str(exc),
                "fields": fields,
            },
        }
    )

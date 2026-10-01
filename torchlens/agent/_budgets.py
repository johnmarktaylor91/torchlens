"""Budgets, truncation, floor semantics, and the continuation struct.

Three resources, three names, identical semantics on every transport (agent
memo 3.8): response TOKENS protect the context window, result ROWS protect
evidence legibility, and payload BYTES protect memory -- refused BEFORE
materialization from manifest-declared numbers. All numerals here are
CONFIGURATION (memo D4), ratified by the stage-D corpus render, never API law.

Continuation is a transparent, mandatory-checked struct -- no opaque cursor,
no server state; artifact, request, or schema drift refuses typed.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from .._errors import InvalidArgumentError
from ._envelope import canonical_dumps, json_safe

#: Served token default for orientation tools (folded overview measured ~2.2k).
ORIENTATION_MAX_TOKENS = 4_000

#: Served token BACKSTOP on row tools (the row cap is primary there).
ROW_TOOL_MAX_TOKENS = 25_000

#: Served row default on row tools (PRIMARY control; 46-116 tokens/row measured).
DEFAULT_MAX_ROWS = 200

#: Per-blob payload byte ceiling, checked against manifest-DECLARED bytes.
MAX_BLOB_BYTES = 64 * 1024 * 1024

#: Aggregate per-call payload byte ceiling.
MAX_CALL_BYTES = 128 * 1024 * 1024

#: Token estimator: ~chars/4, named in-band as approximate.
TOKEN_ESTIMATOR = "approx_chars_div_4"  # gitleaks:allow (estimator name)


def estimate_tokens(text: str) -> int:
    """Estimate the token cost of a serialized payload (chars/4, ceil)."""

    return -(-len(text) // 4)


def resolve_limit(
    requested: Any,
    served_default: int,
    *,
    name: str,
) -> int:
    """Resolve one limit: requests LOWER the served ceiling, never raise it.

    Parameters
    ----------
    requested:
        Caller-requested value, or ``None`` for the served default.
    served_default:
        The served default and ceiling for this tool class.
    name:
        Limit name for refusal text.

    Returns
    -------
    int
        Effective limit.

    Raises
    ------
    InvalidArgumentError
        ``agent_limit_invalid`` when the request is not a positive integer or
        exceeds the served ceiling (no ``0``-as-infinity sentinel exists).
    """

    if requested is None:
        return served_default
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise InvalidArgumentError(
            f"{name}={requested!r} is not a positive integer",
            code="agent_limit_invalid",
            remedy=f"pass a positive integer {name} (served ceiling {served_default})",
            limit_name=name,
        )
    if requested > served_default:
        raise InvalidArgumentError(
            f"{name}={requested} exceeds the served ceiling ({served_default})",
            code="agent_limit_invalid",
            remedy=(
                f"requests lower the ceiling, never raise it; use {name}<="
                f"{served_default} or run the unbounded spelling in Python"
            ),
            limit_name=name,
        )
    return requested


def truncation_block(
    *,
    included: int,
    omitted: int,
    policy: str,
    how_to_get_more: str,
) -> dict[str, Any] | None:
    """Build the always-present truncation object (``None`` when nothing dropped).

    Parameters
    ----------
    included:
        Rows/blocks included in the response.
    omitted:
        Rows/blocks dropped.
    policy:
        One-line statement of the drop rule applied.
    how_to_get_more:
        A CONCRETE next call, never prose.

    Returns
    -------
    dict[str, Any] | None
        Truncation disclosure, or ``None``.
    """

    if omitted <= 0:
        return None
    return {
        "included": included,
        "omitted": omitted,
        "policy": policy,
        "how_to_get_more": how_to_get_more,
    }


def request_hash(request: dict[str, Any]) -> str:
    """Hash a normalized request for continuation drift checks.

    The offset and continuation keys are excluded so page N+1's echo matches
    page N's issuing request.

    Parameters
    ----------
    request:
        Normalized request record.

    Returns
    -------
    str
        ``sha256:...`` digest of the canonical request minus paging keys.
    """

    stripped = {
        key: value for key, value in request.items() if key not in ("offset", "continuation")
    }
    digest = sha256(canonical_dumps(json_safe(stripped)).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def continuation_struct(
    *,
    offset: int,
    artifact_id: str,
    request: dict[str, Any],
    schema: str,
) -> dict[str, Any]:
    """Build the transparent continuation struct for the next page.

    Parameters
    ----------
    offset:
        Row offset the next request must start at.
    artifact_id:
        Content-digest artifact identity (``sha256:...``).
    request:
        The normalized request that produced this page.
    schema:
        Result schema id the next page must match.

    Returns
    -------
    dict[str, Any]
        ``{"offset", "artifact_id", "request_hash", "schema"}``.
    """

    return {
        "offset": offset,
        "artifact_id": artifact_id,
        "request_hash": request_hash(request),
        "schema": schema,
    }


def check_continuation(
    continuation: dict[str, Any] | None,
    *,
    artifact_id: str,
    request: dict[str, Any],
    schema: str,
) -> int:
    """Validate an echoed continuation struct and return the page offset.

    Parameters
    ----------
    continuation:
        The ``next`` struct echoed by the caller, or ``None`` for page one.
    artifact_id:
        Current artifact identity.
    request:
        Current normalized request.
    schema:
        Current result schema id.

    Returns
    -------
    int
        Offset to resume at (``0`` when no continuation was passed).

    Raises
    ------
    InvalidArgumentError
        ``agent_continuation_drift`` when the artifact, request, ordering, or
        schema changed between pages -- a stale page is refused, never
        silently reinterpreted.
    """

    if continuation is None:
        return 0
    expected = {
        "artifact_id": artifact_id,
        "request_hash": request_hash(request),
        "schema": schema,
    }
    for key, value in expected.items():
        got = continuation.get(key)
        if got != value:
            raise InvalidArgumentError(
                f"continuation {key} mismatch: page issued under {got!r}, "
                f"request now resolves {value!r}",
                code="agent_continuation_drift",
                remedy=(
                    "re-issue the query from page one; the artifact, query, "
                    "or schema changed since the continuation was minted"
                ),
                drift_key=key,
            )
    offset = continuation.get("offset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise InvalidArgumentError(
            f"continuation offset {offset!r} is not a non-negative integer",
            code="agent_continuation_drift",
            remedy="echo the continuation struct verbatim from the prior page",
            drift_key="offset",
        )
    return offset

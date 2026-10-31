"""Version-neutral shapes of ``torch._dynamo.explain`` results (pure, no torch access).

Split out of ``_torch_compat.py``, which re-exports every name here; the
explain call itself (``run_dynamo_explain``) stays in the compat layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class _NormalizedDynamoBreak:
    """Version-neutral graph-break evidence from ``torch._dynamo.explain``."""

    reason: str
    source_file: str | None
    line_number: int | None


class _DynamoExplainOutputError(RuntimeError):
    """Raised when a Dynamo explain result has an unsupported runtime shape."""


def _dynamo_break_source(reason: Any) -> tuple[str | None, int | None]:
    """Extract one source location from a version-specific break reason.

    Parameters
    ----------
    reason:
        Dynamo break-reason object or mapping.

    Returns
    -------
    tuple[str | None, int | None]
        Source file and one-indexed line number when supplied by Dynamo.
    """

    user_stack = (
        reason.get("user_stack")
        if isinstance(reason, dict)
        else getattr(reason, "user_stack", None)
    )
    for frame in user_stack or ():
        source_file = (
            frame.get("filename")
            if isinstance(frame, dict)
            else getattr(frame, "filename", getattr(frame, "file", None))
        )
        line_number = (
            frame.get("lineno", frame.get("line"))
            if isinstance(frame, dict)
            else getattr(frame, "lineno", getattr(frame, "line", None))
        )
        if source_file is not None and isinstance(line_number, int):
            return str(source_file), line_number
    source_file = (
        reason.get("filename") if isinstance(reason, dict) else getattr(reason, "filename", None)
    )
    line_number = (
        reason.get("lineno", reason.get("line"))
        if isinstance(reason, dict)
        else getattr(reason, "lineno", getattr(reason, "line", None))
    )
    if source_file is not None and isinstance(line_number, int):
        return str(source_file), line_number
    return None, None


def normalize_dynamo_explain_output(output: Any) -> tuple[_NormalizedDynamoBreak, ...]:
    """Normalize supported Dynamo explain result shapes without version parsing.

    Supported shapes are attribute objects and mappings carrying
    ``break_reasons``, plus the historical tuple whose fourth element contains
    break reasons. Individual reasons may be objects, mappings, or strings.

    Parameters
    ----------
    output:
        Raw result from :func:`run_dynamo_explain`.

    Returns
    -------
    tuple[_NormalizedDynamoBreak, ...]
        Version-neutral graph-break evidence.

    Raises
    ------
    _DynamoExplainOutputError
        If the runtime result shape cannot be interpreted safely.
    """

    break_count: Any = None
    if isinstance(output, dict):
        reasons = output.get("break_reasons")
        break_count = output.get("graph_break_count")
    elif hasattr(output, "break_reasons"):
        reasons = getattr(output, "break_reasons")
        break_count = getattr(output, "graph_break_count", None)
    elif isinstance(output, tuple) and len(output) >= 4:
        reasons = output[3]
        break_count = output[2]
    else:
        raise _DynamoExplainOutputError(
            f"unsupported torch._dynamo.explain output type {type(output).__name__}"
        )
    if reasons is None:
        if break_count == 0:
            return ()
        raise _DynamoExplainOutputError("Dynamo explain output omitted break_reasons")
    if not isinstance(reasons, (list, tuple)):
        raise _DynamoExplainOutputError("Dynamo break_reasons is not a sequence")

    normalized: list[_NormalizedDynamoBreak] = []
    for item in reasons:
        reason_text: object | None
        if isinstance(item, str):
            reason_text = item
        elif isinstance(item, dict):
            reason_text = item.get("reason")
        else:
            reason_text = getattr(item, "reason", None)
        if reason_text is None:
            raise _DynamoExplainOutputError(
                f"unsupported Dynamo break reason type {type(item).__name__}"
            )
        source_file, line_number = _dynamo_break_source(item)
        normalized.append(_NormalizedDynamoBreak(str(reason_text), source_file, line_number))
    return tuple(normalized)

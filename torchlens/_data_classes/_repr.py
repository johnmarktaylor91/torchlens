"""Shared record representation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .._utils.fail_open import fail_open


def format_summary_lines(header: str, lines: Iterable[str]) -> str:
    """Join a record summary header and indented detail lines.

    Parameters
    ----------
    header:
        First line of the representation.
    lines:
        Subsequent lines, already formatted with their intended indentation.

    Returns
    -------
    str
        Newline-joined representation text.
    """

    return "\n".join([header, *lines])


def format_config_items(config: Mapping[str, Any]) -> str:
    """Format function configuration key/value pairs.

    Parameters
    ----------
    config:
        Function configuration mapping.

    Returns
    -------
    str
        Comma-separated ``key=value`` text.
    """

    return ", ".join(f"{key}={value}" for key, value in config.items())


def format_shape_list(shapes: Iterable[Any]) -> str:
    """Format shape values as comma-separated text.

    Parameters
    ----------
    shapes:
        Shape-like values.

    Returns
    -------
    str
        Comma-separated shape text.
    """

    return ", ".join(str(shape) for shape in shapes)


def trace_honesty_suffix(trace: Any) -> str:
    """Return the outcome and health suffix tokens for a trace's identity repr (F10)."""

    tokens: list[str] = []
    outcome = fail_open(lambda: trace.outcome, lambda _error: None)
    status = getattr(getattr(outcome, "status", None), "value", None)
    if status and status != "complete":
        tokens.append(f"outcome={status}")
    verdict = fail_open(lambda: trace.nonfinite_verdict, lambda _error: None)
    if verdict == "checked_and_clean":
        tokens.append("no NaN/Inf (exact)")
    elif verdict == "found":
        tokens.append("NaN/Inf FOUND (.nonfinite_ops)")
    armed = fail_open(trace._armed_summary_line, lambda _error: None)
    if armed:
        tokens.append(armed)
    return f" [{'; '.join(tokens)}]" if tokens else ""

"""Shared record representation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


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

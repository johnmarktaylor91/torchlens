"""Shared record representation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class OpCardHtmlMixin:
    """Notebook-card ``_repr_html_`` for ``Op`` (housed here for the R43 op.py ceiling)."""

    __slots__ = ()

    def _repr_html_(self) -> str:
        """Return the notebook Op card (treescope memo B2).

        Identity / value / context zones with the PROOF row; cost wrappers
        pass through ``str()`` untouched; the budgeted array view rides the
        ported truncation protocol. Never raises: any internal failure
        degrades to a one-line ``card unavailable`` fragment, and the
        treescope-bridge suppression sentinel replaces the card exactly
        when the bridge just rendered this object (memo 3.6).
        """

        from ..notebook.cards import op_repr_html

        return op_repr_html(self)


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

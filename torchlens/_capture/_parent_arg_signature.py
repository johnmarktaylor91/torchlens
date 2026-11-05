"""Deterministic parent-argument routing signatures for refresh graph comparison."""

from __future__ import annotations

from typing import Any


def _signature_sort_key(value: Any) -> tuple[str, str]:
    """Return a stable comparable token for graph-signature sort keys.

    Parameters
    ----------
    value
        Parent-position key to normalize for signature comparison.

    Returns
    -------
    tuple[str, str]
        Type-qualified token that keeps incomparable key types sortable.
    """

    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


def _normalized_parent_arg_positions(
    parent_arg_positions: dict[str, dict[Any, str]] | None,
) -> tuple[tuple[str, tuple[tuple[Any, str], ...]], ...]:
    """Return a deterministic signature view of parent argument positions.

    Parameters
    ----------
    parent_arg_positions
        Layer parent routing metadata keyed by argument domain.

    Returns
    -------
    tuple[tuple[str, tuple[tuple[Any, str], ...]], ...]
        Stable, comparable representation of argument and keyword parent routing.
    """

    positions = parent_arg_positions or {}
    return tuple(
        (
            arg_domain,
            tuple(
                sorted(
                    (positions.get(arg_domain, {}) or {}).items(),
                    key=lambda item: _signature_sort_key(item[0]),
                )
            ),
        )
        for arg_domain in ("args", "kwargs")
    )


def _format_parent_arg_positions(
    parent_arg_positions: tuple[tuple[str, tuple[tuple[Any, str], ...]], ...],
) -> str:
    """Format normalized parent routing for graph-drift error details.

    Parameters
    ----------
    parent_arg_positions
        Normalized parent routing metadata from ``_normalized_parent_arg_positions``.

    Returns
    -------
    str
        Compact human-readable routing summary.
    """

    rendered_parts: list[str] = []
    for arg_domain, entries in parent_arg_positions:
        rendered_parts.append(f"{arg_domain}={list(entries)!r}")
    return ", ".join(rendered_parts)

"""Small naming and sanitizing helpers shared by the timeline and flamegraph exporters."""

from __future__ import annotations

from typing import Any

from .._record_type import record_type


def _duration_us(layer: Any) -> int:
    """Return a positive microsecond duration for a layer.

    Parameters
    ----------
    layer:
        Layer entry.

    Returns
    -------
    int
        Duration in microseconds.
    """

    duration = float(getattr(layer, "func_duration", 0.0) or 0.0)
    return max(1, int(duration * 1_000_000))


def _layer_display_name(layer: Any) -> str:
    """Return a human-readable layer name.

    Parameters
    ----------
    layer:
        Layer entry.

    Returns
    -------
    str
        Display name.
    """

    label = str(getattr(layer, "layer_label", ""))
    func_name = str(getattr(layer, "func_name", "") or record_type(layer, ""))
    return f"{label} ({func_name})" if func_name and func_name != label else label


def _sanitize_flamegraph_frame(frame: str) -> str:
    """Return a folded-stack-safe frame name.

    Parameters
    ----------
    frame:
        Raw frame name.

    Returns
    -------
    str
        Sanitized frame name.
    """

    return frame.replace(";", "_").replace("\n", " ").strip() or "<unknown>"


def _safe_id(value: str) -> str:
    """Return a CSS/SVG-safe identifier fragment.

    Parameters
    ----------
    value:
        Raw identifier.

    Returns
    -------
    str
        Sanitized identifier.
    """

    return "".join(char if char.isalnum() else "-" for char in value).strip("-")

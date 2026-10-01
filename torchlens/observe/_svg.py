"""Deterministic, dependency-free SVG renderer for the timeline v2 artifact.

Observe item 12: matplotlib is a TEST extra, so a matplotlib renderer would
mint a new runtime dependency -- the required static renderer is plain SVG
text with a byte-exact golden. Execution ordinal on x, logical bytes on y,
closed category bands stacked as CUMULATIVE PRODUCED bytes (named as such),
the persistent baseline flat underneath, and the saved-for-backward band as
an ANNOTATION line, never a stacked area (the parameter-phantom rule). Large
traces bin contiguous ordinals to a documented pixel budget with the bin
count disclosed in the caption -- time is never truncated or top-k reordered.
Palette and cosmetics are [UI-SPRINT] placeholders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["render_timeline_svg", "write_timeline_svg"]

#: Stacked band order (bottom-up) and placeholder fills.
_BANDS: tuple[tuple[str, str], ...] = (
    ("parameter", "#4878a8"),
    ("buffer", "#76a5af"),
    ("input", "#b3b3b3"),
    ("activation", "#e8a33d"),
    ("newly_saved_activation", "#c25454"),
)

_WIDTH = 800
_HEIGHT = 360
_MARGIN_LEFT = 60
_MARGIN_RIGHT = 16
_MARGIN_TOP = 40
_MARGIN_BOTTOM = 78
_DEFAULT_COLUMN_BUDGET = 320


def _cumulative_columns(
    artifact: dict[str, Any], column_budget: int
) -> tuple[list[dict[str, int]], int, int]:
    """Build per-column cumulative band values with disclosed binning.

    Returns
    -------
    tuple[list[dict[str, int]], int, int]
        ``(columns, bin_size, n_ordinals)`` where each column maps band name
        to its cumulative produced bytes at the column's LAST ordinal.
    """

    persistent = {"parameter": 0, "buffer": 0}
    per_ordinal: dict[int, dict[str, int]] = {}
    max_ordinal = 0
    for row in artifact["events"]:
        category = row["category"]
        if row["phase"] == "persistent":
            if category in persistent:
                persistent[category] += row["bytes"]
            continue
        if row["phase"] != "forward":
            continue
        ordinal = int(row["ordinal"] or 0)
        max_ordinal = max(max_ordinal, ordinal)
        bucket = per_ordinal.setdefault(ordinal, {"input": 0, "activation": 0, "newly": 0})
        if category in ("input", "activation"):
            bucket[category] += row["bytes"]
        band = row.get("decomposition")
        if isinstance(band, dict):
            bucket["newly"] += int(band.get("newly_saved_activation", 0))
    n_ordinals = max(max_ordinal, 1)
    bin_size = max(1, -(-n_ordinals // column_budget))
    columns: list[dict[str, int]] = []
    running = {"input": 0, "activation": 0, "newly": 0}
    for start in range(1, n_ordinals + 1, bin_size):
        for ordinal in range(start, min(start + bin_size, n_ordinals + 1)):
            ordinal_bucket = per_ordinal.get(ordinal)
            if ordinal_bucket:
                running["input"] += ordinal_bucket["input"]
                running["activation"] += ordinal_bucket["activation"]
                running["newly"] += ordinal_bucket["newly"]
        columns.append(
            {
                "parameter": persistent["parameter"],
                "buffer": persistent["buffer"],
                "input": running["input"],
                "activation": running["activation"],
                "newly_saved_activation": running["newly"],
            }
        )
    return columns, bin_size, n_ordinals


def render_timeline_svg(
    artifact: dict[str, Any], *, column_budget: int = _DEFAULT_COLUMN_BUDGET
) -> str:
    """Render one timeline v2 artifact as deterministic standalone SVG text.

    Parameters
    ----------
    artifact:
        A ``torchlens.memory_timeline.v2`` artifact.
    column_budget:
        Maximum rendered columns; larger traces bin contiguous ordinals and
        the caption discloses the bin size.

    Returns
    -------
    str
        Complete SVG document. Identical artifacts render byte-identically
        (fixed band order, fixed number formatting, no unordered iteration).
    """

    columns, bin_size, n_ordinals = _cumulative_columns(artifact, column_budget)
    peak_total = max((sum(column.values()) for column in columns), default=0)
    y_scale = (_HEIGHT - _MARGIN_TOP - _MARGIN_BOTTOM) / float(peak_total) if peak_total else 0.0
    plot_width = _WIDTH - _MARGIN_LEFT - _MARGIN_RIGHT
    column_width = plot_width / float(len(columns) or 1)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" '
        f'viewBox="0 0 {_WIDTH} {_HEIGHT}">',
        f'<rect x="0" y="0" width="{_WIDTH}" height="{_HEIGHT}" fill="#ffffff"/>',
        f'<text x="{_MARGIN_LEFT}" y="18" font-family="monospace" font-size="13">'
        "memory timeline v2 (cumulative produced logical bytes)</text>",
    ]
    for band_index, (band, fill) in enumerate(_BANDS):
        for column_index, column in enumerate(columns):
            below = sum(column[name] for name, _fill in _BANDS[:band_index])
            value = column[band]
            if value <= 0 or y_scale == 0.0:
                continue
            x = _MARGIN_LEFT + column_index * column_width
            height = value * y_scale
            y = _HEIGHT - _MARGIN_BOTTOM - (below * y_scale) - height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{column_width:.2f}" '
                f'height="{height:.2f}" fill="{fill}"/>'
            )
    axis_y = _HEIGHT - _MARGIN_BOTTOM
    parts.append(
        f'<line x1="{_MARGIN_LEFT}" y1="{axis_y}" x2="{_WIDTH - _MARGIN_RIGHT}" '
        f'y2="{axis_y}" stroke="#000000"/>'
    )
    parts.append(
        f'<line x1="{_MARGIN_LEFT}" y1="{_MARGIN_TOP}" x2="{_MARGIN_LEFT}" '
        f'y2="{axis_y}" stroke="#000000"/>'
    )
    parts.append(
        f'<text x="{_MARGIN_LEFT}" y="{axis_y + 16}" font-family="monospace" '
        f'font-size="11">execution ordinal 1..{n_ordinals}'
        + (f" (binned x{bin_size}, {len(columns)} columns)" if bin_size > 1 else "")
        + "</text>"
    )
    legend_x = _MARGIN_LEFT
    for band, fill in _BANDS:
        parts.append(
            f'<rect x="{legend_x}" y="{axis_y + 24}" width="10" height="10" fill="{fill}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 14}" y="{axis_y + 33}" font-family="monospace" '
            f'font-size="10">{band}</text>'
        )
        legend_x += 14 + 8 * len(band) + 16
    saved_band = artifact.get("saved_band", {})
    saved_parameter_total = saved_band.get("saved_parameter_total")
    if saved_parameter_total is not None:
        annotation = (
            f"parameter band: of which {saved_parameter_total} B retained for "
            "backward (annotation, never stacked)"
        )
    else:
        annotation = (
            "saved-for-backward band: gross attribution only "
            "(decomposition unavailable on this trace); never stacked"
        )
    parts.append(
        f'<text x="{_MARGIN_LEFT}" y="{axis_y + 50}" font-family="monospace" '
        f'font-size="10">{annotation}</text>'
    )
    parts.append(
        f'<text x="{_MARGIN_LEFT}" y="{axis_y + 64}" font-family="monospace" '
        f'font-size="10">logical recorded tensor bytes; not allocator residency</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_timeline_svg(
    artifact: dict[str, Any],
    path: str | Path,
    *,
    column_budget: int = _DEFAULT_COLUMN_BUDGET,
) -> Path:
    """Write the deterministic SVG rendering of one timeline artifact.

    Parameters
    ----------
    artifact:
        A ``torchlens.memory_timeline.v2`` artifact.
    path:
        Destination ``.svg`` path.
    column_budget:
        Maximum rendered columns (disclosed binning above it).

    Returns
    -------
    Path
        The written path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_timeline_svg(artifact, column_budget=column_budget))
    return destination

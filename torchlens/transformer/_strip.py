"""Colored-token strips: matplotlib paper path + zero-dependency emitters.

The memo's D3 split: matplotlib lays out the proportional-width,
token-boundary-wrapped PAPER strip (real font metrics; measured 0.090 s at
66 tokens; SVG with zero ``<image>`` elements; PDF text extracts in token
order), while the small zero-dependency SVG/HTML emitter here keeps notebook
duty and first contact alive on a bare install (no matplotlib, no cairosvg,
no network, no JS). Both read the SAME :class:`TokenScores` record.

Multi-row strips serve the NMF factor view and per-method comparisons.
Footer disclosure lines are rendered by BOTH paths, always.

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import html as _html
import math
from pathlib import Path
from typing import Any

from ._mpl import figure, save_figure
from ._records import Artifact, TokenScoreRow, TokenScores

__all__ = ["render_token_strip", "token_strip_html", "token_strip_svg"]

#: Tokens per wrapped line in the paper strip (R0 tuning candidate).
TOKENS_PER_LINE = 16

#: Positive/negative/neutral fills for the zero-dependency emitter
#: (diverging, zero-centered; colorblind-safe blue/orange pair).
_POSITIVE_RGB = (0, 114, 178)  # Okabe-Ito blue
_NEGATIVE_RGB = (213, 94, 0)  # Okabe-Ito vermillion
_NA_FILL = "#f0f0f0"


def _row_limit(row: TokenScoreRow) -> float:
    """Return the row's normalization limit (max |score|, never zero)."""

    finite = [abs(score) for score in row.scores if score is not None]
    return max(finite) if finite else 1.0


def _cell_rgb(score: float | None, limit: float, domain: str) -> tuple[int, int, int] | None:
    """Return the 0-255 RGB fill for one token cell (``None`` = N/A fill)."""

    if score is None:
        return None
    intensity = min(1.0, abs(score) / limit) if limit > 0.0 else 0.0
    if domain == "zero_centered_diverging" and score < 0:
        base = _NEGATIVE_RGB
    else:
        base = _POSITIVE_RGB
    # White -> full base color; alpha-free so the HTML is self-contained.
    red = round(255 - (255 - base[0]) * intensity)
    green = round(255 - (255 - base[1]) * intensity)
    blue = round(255 - (255 - base[2]) * intensity)
    return red, green, blue


def _cell_color(score: float | None, limit: float, domain: str) -> str:
    """Return the CSS color for one token cell (emitter paths)."""

    rgb = _cell_rgb(score, limit, domain)
    if rgb is None:
        return _NA_FILL
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _cell_hex(score: float | None, limit: float, domain: str) -> str:
    """Return the hex color for one token cell (matplotlib path)."""

    rgb = _cell_rgb(score, limit, domain)
    if rgb is None:
        return _NA_FILL
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def token_strip_html(record: TokenScores) -> str:
    """Emit the zero-dependency, self-contained HTML token strip.

    Parameters
    ----------
    record:
        The token-scores record (single- or multi-row).

    Returns
    -------
    str
        Escaped, network-free HTML: one table row per score row plus every
        footer disclosure line. N/A cells render "n/a", never zero-colored.
    """

    parts = ['<table role="presentation" style="border-collapse:collapse">']
    for row in record.rows:
        limit = _row_limit(row)
        cells = [
            f'<td style="padding:1px 3px;border:1px solid #ddd;font-size:11px">'
            f"{_html.escape(row.label)}</td>"
        ]
        for token, score in zip(record.tokens, row.scores, strict=True):
            title = "n/a" if score is None else f"{score:+.6g}"
            cells.append(
                f'<td style="background:{_cell_color(score, limit, record.domain)};'
                f'padding:1px 3px;font-size:11px" title="{title}">'
                f"{_html.escape(token)}</td>"
            )
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</table>")
    for line in record.footer_lines:
        parts.append(f'<div style="font-size:10px;color:#444">{_html.escape(line)}</div>')
    parts.append(
        f'<div style="font-size:10px;color:#444">provenance: '
        f"{_html.escape(record.provenance)}</div>"
    )
    return "".join(parts)


def _svg_text_width(token: str) -> float:
    """Return a deterministic monospace-advance width estimate in px."""

    return 4.0 + 6.5 * len(token)


def token_strip_svg(record: TokenScores) -> str:
    """Emit the zero-dependency SVG token strip (no fonts embedded).

    Parameters
    ----------
    record:
        The token-scores record.

    Returns
    -------
    str
        Self-contained SVG with real ``<text>`` elements and zero
        ``<image>`` elements; token widths use a deterministic monospace
        estimate (the paper-format path with real font metrics is
        :func:`render_token_strip`).
    """

    row_height = 20.0
    footer_height = 14.0 * (len(record.footer_lines) + 1)
    widths = [_svg_text_width(token) for token in record.tokens]
    label_width = max(_svg_text_width(row.label) for row in record.rows)
    total_width = label_width + sum(widths) + 4.0
    total_height = row_height * len(record.rows) + footer_height + 4.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width:.0f}" '
        f'height="{total_height:.0f}" font-family="monospace" font-size="11">'
    ]
    for row_index, row in enumerate(record.rows):
        limit = _row_limit(row)
        y = 2.0 + row_index * row_height
        parts.append(f'<text x="2" y="{y + 14:.1f}">{_html.escape(row.label)}</text>')
        x = label_width + 2.0
        for token, score, width in zip(record.tokens, row.scores, widths, strict=True):
            fill = _cell_color(score, limit, record.domain)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
                f'height="{row_height - 2:.1f}" fill="{fill}"/>'
            )
            parts.append(f'<text x="{x + 2:.1f}" y="{y + 14:.1f}">{_html.escape(token)}</text>')
            x += width
    y = row_height * len(record.rows) + 12.0
    for line in list(record.footer_lines) + [f"provenance: {record.provenance}"]:
        parts.append(f'<text x="2" y="{y:.1f}" font-size="9">{_html.escape(line)}</text>')
        y += 14.0
    parts.append("</svg>")
    return "".join(parts)


def render_token_strip(
    record: TokenScores,
    path: Path | str,
    *,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render the paper-format token strip via matplotlib (memo D3).

    Proportional-width tokens with real font metrics, wrapped at token
    boundaries, one band per score row, every footer line rendered.

    Parameters
    ----------
    record:
        The token-scores record (single- or multi-row).
    path:
        Output path (PNG/SVG/PDF by suffix).
    svg_fonttype:
        ``'path'`` (default) or ``'none'``.

    Returns
    -------
    Artifact
        The written file plus rendered disclosure lines.
    """

    n_lines = math.ceil(len(record.tokens) / TOKENS_PER_LINE)
    n_bands = n_lines * len(record.rows)
    fig = figure(width=8.0, height=0.42 * n_bands + 0.22 * (len(record.footer_lines) + 2) + 0.5)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    renderer = fig.canvas.get_renderer()
    band = 0
    for row in record.rows:
        limit = _row_limit(row)
        for line_index in range(n_lines):
            start = line_index * TOKENS_PER_LINE
            chunk_tokens = record.tokens[start : start + TOKENS_PER_LINE]
            chunk_scores = row.scores[start : start + TOKENS_PER_LINE]
            _draw_strip_line(
                ax,
                fig,
                renderer,
                tokens=chunk_tokens,
                scores=chunk_scores,
                domain=record.domain,
                limit=limit,
                y=-band,
                label=row.label if line_index == 0 else "",
            )
            band += 1
    ax.set_xlim(0, 1)
    ax.set_ylim(-band + 0.2, 1.0)
    disclosures = list(record.footer_lines) + [f"provenance: {record.provenance}"]
    fig.text(0.01, 0.01, "\n".join(disclosures), fontsize=6, va="bottom")
    written = save_figure(fig, path, svg_fonttype=svg_fonttype)
    return Artifact(
        paths=(written,),
        format=Path(path).suffix.lstrip(".").lower(),
        title=record.rows[0].label,
        disclosure_lines=tuple(disclosures),
        provenance=record.provenance,
        fingerprint="",
        svg_fonttype=svg_fonttype,
        pages_total=1,
    )


def _draw_strip_line(  # noqa: PLR0913 -- explicit line geometry beats hidden state
    ax: Any,
    fig: Any,
    renderer: Any,
    *,
    tokens: tuple[str, ...],
    scores: tuple[float | None, ...],
    domain: str,
    limit: float,
    y: float,
    label: str,
) -> None:
    """Draw one wrapped strip line with measured proportional widths."""

    if label:
        ax.text(0.0, y + 0.25, label, fontsize=7, style="italic", transform=ax.transData)
    x = 0.12
    for token, score in zip(tokens, scores, strict=True):
        text = ax.text(x, y, token, fontsize=8, va="bottom", transform=ax.transData)
        bbox = text.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        pad = 0.004
        width = bbox.width + 2 * pad
        text.remove()
        from matplotlib.patches import Rectangle

        ax.add_patch(
            Rectangle(
                (x - pad, y - 0.08),
                width,
                0.5,
                facecolor=_cell_hex(score, limit, domain),
                edgecolor="none",
            )
        )
        ax.text(x, y, token, fontsize=8, va="bottom", transform=ax.transData)
        x += width + 0.004

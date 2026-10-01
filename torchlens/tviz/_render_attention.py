"""Attention picture renderers: single head, grids, atlas (tviz roster).

Every panel is a rect mesh (``pcolormesh``), never ``imshow``; the fixed
shared ``[0, 1]`` probability domain is the default color scale; masked cells
fill with a color OUTSIDE the data colormap plus a legend line; cropping is
disclosed, never renormalized; the routing-not-causal footer is permanent.
Annotations follow the D16 grammar: a whole-head effect is a printed number
plus its own diverging border channel -- effects NEVER color attention cells.

File output paginates and drops nothing (D21); the atlas emits an all-member
overview plus numbered detail pages. Atlas constants below are R0 tuning
candidates frozen through the themes gate, not memo constants.

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._errors import refuse
from ._mpl import (
    ATTENTION_CMAP,
    DIVERGING_CMAP,
    MASKED_CELL_COLOR,
    colorbar,
    figure,
    paged_paths,
    require_matplotlib,
    save_figure,
)
from ._records import (
    Annotation,
    Artifact,
    AttentionView,
    tensor_fingerprint,
)
from ._wording import (
    AXES_WORDING,
    MEASURED_N_OF_M,
    PANELS_NOT_COMPARABLE,
    ROUTING_FOOTER,
    UNMARKED_MASK_WORDING,
)

__all__ = ["render_attention", "render_attention_atlas"]

#: Panels per page in head-grid output (R0 tuning candidate).
PANELS_PER_PAGE = 12

#: Maximum token labels drawn per axis before decimation (R0 candidate).
MAX_TICK_LABELS = 32

#: Kind glyph prefixes (ASCII; D16 -- four visually distinct annotations).
_KIND_GLYPHS = {
    "descriptive_head_score": "score",
    "additive_logit_contribution": "logit+",
    "screening_estimate": "screen~",
    "intervention_effect": "ablate",
}


def _tick_positions(n_items: int) -> tuple[list[int], int]:
    """Return decimated tick indices and the honest omitted count."""

    if n_items <= MAX_TICK_LABELS:
        return list(range(n_items)), 0
    step = math.ceil(n_items / MAX_TICK_LABELS)
    kept = list(range(0, n_items, step))
    return kept, n_items - len(kept)


def _panel_mesh(ax: Any, view: AttentionView, head_row: int, vmin: float, vmax: float) -> Any:
    """Draw one head's pattern as a rect mesh with masked-cell fill."""

    matplotlib = require_matplotlib()
    data = view.pattern[head_row].detach().to("cpu", dtype=torch.float32).numpy()
    if view.mask is not None:
        data = np.ma.masked_array(data, mask=view.mask.mask.cpu().numpy())
    cmap = matplotlib.colormaps[ATTENTION_CMAP].copy()
    cmap.set_bad(MASKED_CELL_COLOR)
    mesh = ax.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=False)
    ax.invert_yaxis()  # row 0 (first query) at the top, the reading order
    ax.set_aspect("equal")
    return mesh


def _panel_ticks(ax: Any, view: AttentionView) -> int:
    """Label the panel's token axes with decimation; return omitted count."""

    key_idx, key_omitted = _tick_positions(len(view.key_tokens))
    query_idx, query_omitted = _tick_positions(len(view.query_tokens))
    ax.set_xticks([index + 0.5 for index in key_idx])
    ax.set_xticklabels(
        [view.key_tokens.tokens[index] for index in key_idx], rotation=90, fontsize=6
    )
    ax.set_yticks([index + 0.5 for index in query_idx])
    ax.set_yticklabels([view.query_tokens.tokens[index] for index in query_idx], fontsize=6)
    ax.tick_params(length=0)
    return key_omitted + query_omitted


def _annotation_for(annotation: Annotation | None, head: int) -> tuple[str, float | None]:
    """Return the header suffix and value for one head's annotation."""

    if annotation is None or head not in annotation.heads:
        return "", None
    value = annotation.values[annotation.heads.index(head)]
    if value is None:
        return "", None  # unmeasured: blank, never zero
    return f"  [{_KIND_GLYPHS[annotation.kind]} {value:+.3g}]", value


def _effect_border(ax: Any, value: float | None, scale: float) -> None:
    """Color the panel border from the diverging channel (D16 grammar).

    Effects never color attention cells; the border/header IS the effect
    channel.
    """

    if value is None or scale <= 0.0:
        return
    matplotlib = require_matplotlib()
    cmap = matplotlib.colormaps[DIVERGING_CMAP]
    color = cmap(0.5 + 0.5 * max(-1.0, min(1.0, value / scale)))
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(3.0)


def _disclosure_lines(view: AttentionView, annotation: Annotation | None) -> list[str]:
    """Assemble the figure's honesty footer lines, in render order."""

    lines = [AXES_WORDING, ROUTING_FOOTER, f"pattern: {view.provenance_wording}"]
    if view.domain == "per_panel":
        lines.append(PANELS_NOT_COMPARABLE)
    if view.mask is None:
        lines.append(UNMARKED_MASK_WORDING)
    else:
        lines.append(f"grey cells: masked (mask source: {view.mask.source})")
    if view.crop is not None:
        lines.append(view.crop.disclosure())
    if view.gqa is not None:
        lines.append(
            f"grouped-query attention: {view.gqa.n_query_heads} query heads share "
            f"{view.gqa.n_kv_heads} kv groups; each panel header names its kv group "
            "(shared storage is never presented as independent heads)"
        )
    if view.episode is not None:
        lines.append(
            f"episode step {view.episode.step} ({view.episode.role}, {view.episode.completion})"
        )
    if annotation is not None:
        measured = sum(1 for value in annotation.values if value is not None)
        lines.append(
            f"{_KIND_GLYPHS[annotation.kind]}: {annotation.legend or annotation.kind} -- "
            f"{annotation.source} -- " + MEASURED_N_OF_M.format(n=measured, m=len(annotation.heads))
        )
        if annotation.receipt is not None:
            receipt = annotation.receipt
            lines.append(
                f"controls: negative {receipt.negative_control:+.2e} "
                f"(tol {receipt.negative_control_tol:.1e}), positive "
                f"{receipt.positive_control:+.3g}; fires: {receipt.fires}"
            )
    return lines


def _head_title(view: AttentionView, head: int, suffix: str) -> str:
    """Return one panel's title with the GQA header when applicable."""

    title = f"{view.layer} / head {head}{suffix}"
    if view.gqa is not None:
        title += f"\n{view.gqa.header(head)}"
    return title


def _render_pages(
    view: AttentionView,
    annotation: Annotation | None,
    path: Path | str,
    svg_fonttype: str,
) -> Artifact:
    """Render every head of a view across numbered pages."""

    heads = list(view.heads)
    n_pages = math.ceil(len(heads) / PANELS_PER_PAGE)
    targets = paged_paths(path, n_pages)
    effect_scale = 0.0
    if annotation is not None:
        effect_scale = max(
            (abs(value) for value in annotation.values if value is not None), default=0.0
        )
    disclosures = _disclosure_lines(view, annotation)
    vmin, vmax = (
        (0.0, 1.0)
        if view.domain == "probability"
        else (
            float(view.pattern.min()),
            float(view.pattern.max()),
        )
    )
    written: list[Path] = []
    for page, target in enumerate(targets):
        page_heads = heads[page * PANELS_PER_PAGE : (page + 1) * PANELS_PER_PAGE]
        written.append(
            _render_one_page(
                view,
                annotation,
                page_heads,
                target,
                disclosures=disclosures,
                vmin=vmin,
                vmax=vmax,
                effect_scale=effect_scale,
                page=page,
                n_pages=n_pages,
                svg_fonttype=svg_fonttype,
            )
        )
    return Artifact(
        paths=tuple(written),
        format=Path(path).suffix.lstrip(".").lower(),
        title=view.layer,
        disclosure_lines=tuple(disclosures),
        provenance=view.provenance_wording,
        fingerprint=view.fingerprint,
        svg_fonttype=svg_fonttype,
        pages_total=n_pages,
    )


def _render_one_page(  # noqa: PLR0913 -- page geometry is explicit, never module state
    view: AttentionView,
    annotation: Annotation | None,
    page_heads: list[int],
    target: Path,
    *,
    disclosures: list[str],
    vmin: float,
    vmax: float,
    effect_scale: float,
    page: int,
    n_pages: int,
    svg_fonttype: str,
) -> Path:
    """Render one page of head panels and save it."""

    columns = min(4, len(page_heads))
    rows = math.ceil(len(page_heads) / columns)
    n_src = len(view.key_tokens)
    panel_inches = max(2.2, min(6.0, n_src * 0.14))
    fig = figure(
        width=columns * (panel_inches + 0.9),
        height=rows * (panel_inches + 1.2) + 0.25 * len(disclosures) + 0.6,
    )
    axes = fig.subplots(rows, columns, squeeze=False)
    mesh = None
    for slot, head in enumerate(page_heads):
        ax = axes[slot // columns][slot % columns]
        mesh = _panel_mesh(ax, view, view.heads.index(head), vmin, vmax)
        _panel_ticks(ax, view)
        suffix, value = _annotation_for(annotation, head)
        _effect_border(ax, value, effect_scale)
        ax.set_title(_head_title(view, head, suffix), fontsize=7)
    for slot in range(len(page_heads), rows * columns):
        axes[slot // columns][slot % columns].set_visible(False)
    if mesh is not None:
        label = "attention weight [0, 1]" if view.domain == "probability" else "value (per panel)"
        colorbar(fig, mesh, [a for row in axes for a in row if a.get_visible()], label=label)
    footer = "\n".join(disclosures)
    if n_pages > 1:
        footer = f"page {page + 1} of {n_pages}\n" + footer
    fig.suptitle(view.layer, fontsize=9)
    fig.text(0.01, 0.01, footer, fontsize=6, va="bottom")
    return save_figure(fig, target, svg_fonttype=svg_fonttype)


def render_attention(
    view: AttentionView,
    path: Path | str,
    *,
    heads: tuple[int, ...] | None = None,
    annotation: Annotation | None = None,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render an attention view to paper-ready PNG/SVG/PDF pages.

    One head renders a single square panel; several heads render the
    per-head grid with deterministic pagination (nothing is silently
    dropped; every page path is returned on the artifact).

    Parameters
    ----------
    view:
        The attention view record.
    path:
        Output path; the suffix picks the format. Multi-page output writes
        ``name-pageNN.ext`` files.
    heads:
        Optional subset of ORIGINAL head indices to render.
    annotation:
        Optional typed annotation (D16); ``intervention_effect`` requires a
        valid receipt and renders as printed numbers plus a diverging
        border -- never recolored attention cells.
    svg_fonttype:
        ``'path'`` (default) or ``'none'`` for editable text.

    Returns
    -------
    Artifact
        Written paths plus every rendered disclosure line.
    """

    if heads is not None:
        missing = [head for head in heads if head not in view.heads]
        if missing:
            refuse(
                code="tv_record_invalid",
                message=f"Requested heads {missing} are not in this view ({view.heads}).",
                remedy="pass original head indices present on the view",
            )
        rows = [view.heads.index(head) for head in heads]
        view = AttentionView(
            pattern=view.pattern[rows],
            query_tokens=view.query_tokens,
            key_tokens=view.key_tokens,
            heads=tuple(heads),
            layer=view.layer,
            coordinate=view.coordinate,
            domain=view.domain,
            provenance=view.provenance,
            mask=view.mask,
            crop=view.crop,
            gqa=view.gqa,
            episode=view.episode,
        )
    return _render_pages(view, annotation, path, svg_fonttype)


def render_attention_atlas(
    views: list[AttentionView],
    path: Path | str,
    *,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render the whole-model attention atlas (BertViz model view, matched).

    Page 1 is the all-member overview (every layer x head tile admitted by
    the frozen resource budget -- with today's constants, all of them);
    subsequent numbered detail pages restore full token labels per layer.

    Parameters
    ----------
    views:
        One attention view per layer, in execution order.
    path:
        Output path; pages write ``name-pageNN.ext``.
    svg_fonttype:
        ``'path'`` (default) or ``'none'``.

    Returns
    -------
    Artifact
        Overview + detail page paths and the rendered disclosures.
    """

    if not views:
        refuse(
            code="tv_record_invalid",
            message="The atlas needs at least one attention view.",
            remedy="pass the per-layer views from tviz.attention_views(trace)",
        )
    n_pages = 1 + len(views)
    targets = paged_paths(path, n_pages)
    overview = _render_atlas_overview(views, targets[0], svg_fonttype)
    disclosures = list(overview[1])
    written = [overview[0]]
    for view, target in zip(views, targets[1:], strict=True):
        artifact = _render_pages(view, None, target, svg_fonttype)
        written.extend(artifact.paths)
    return Artifact(
        paths=tuple(written),
        format=Path(path).suffix.lstrip(".").lower(),
        title="attention atlas",
        disclosure_lines=tuple(disclosures),
        provenance=views[0].provenance_wording,
        fingerprint=tensor_fingerprint(views[0].pattern),
        svg_fonttype=svg_fonttype,
        pages_total=n_pages,
    )


def _render_atlas_overview(
    views: list[AttentionView], target: Path, svg_fonttype: str
) -> tuple[Path, list[str]]:
    """Render the atlas overview page: one small tile per (layer, head)."""

    n_layers = len(views)
    n_heads = max(len(view.heads) for view in views)
    tile = 0.62  # inches per tile; R0 floor candidate (cell edge >= 2x seq len px)
    fig = figure(width=max(4.0, n_heads * tile + 1.6), height=max(3.0, n_layers * tile + 1.6))
    axes = fig.subplots(n_layers, n_heads, squeeze=False)
    vmin, vmax = 0.0, 1.0
    for layer_row, view in enumerate(views):
        for head_col in range(n_heads):
            ax = axes[layer_row][head_col]
            if head_col >= len(view.heads):
                ax.set_visible(False)
                continue
            _panel_mesh(ax, view, head_col, vmin, vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if layer_row == 0:
                ax.set_title(f"h{view.heads[head_col]}", fontsize=6)
            if head_col == 0:
                ax.set_ylabel(view.layer.split(".")[-1] or view.layer, fontsize=5, rotation=0)
    disclosures = _disclosure_lines(views[0], None)
    disclosures.append(
        f"overview: all {sum(len(view.heads) for view in views)} heads across "
        f"{n_layers} layers; detail pages follow with full labels"
    )
    fig.suptitle("attention atlas (overview)", fontsize=9)
    fig.text(0.01, 0.01, "\n".join(disclosures), fontsize=5, va="bottom")
    return save_figure(fig, target, svg_fonttype=svg_fonttype), disclosures

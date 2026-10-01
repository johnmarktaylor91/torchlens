"""Prediction pictures: ribbon, answer trajectory, rank panel, top-k table.

These consume the STREAMING lens result (:func:`torchlens.semantic.logit_lens.
logit_lens_predictions`, the FIX-L extractor): probabilities always use the
full-vocabulary denominator, ranks are one-based with the tie convention
recorded, and provenance wording travels row by row -- only a row numerically
equal to the captured native logits says "native output"; every other row
says "projected through final norm/head" (UNVALIDATED-LENS-AS-NATIVE and
PROJECTION-AS-PREDICTION are gate sentinels, so the wording is load-bearing).

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._errors import refuse
from ._mpl import ATTENTION_CMAP, colorbar, figure, require_matplotlib, save_figure
from ._records import Artifact, PredictionTable, PredictionTrajectory

__all__ = [
    "prediction_table",
    "prediction_table_html",
    "prediction_trajectory",
    "render_answer_trajectory",
    "render_prediction_ribbon",
    "render_prediction_table",
]

#: Positions per page in table output (whole positions, never split; R0).
POSITIONS_PER_PAGE = 24


def _decode(tokenizer: Any | None, token_id: int) -> str:
    """Decode one token id, falling back to the raw id string."""

    if tokenizer is None:
        return str(token_id)
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return str(tokenizer.convert_ids_to_tokens([token_id])[0])
    return str(tokenizer.decode([token_id]))


def _row_position_index(row: Any, position: int) -> int:
    """Return the index of ``position`` inside a lens row's positions."""

    if position < 0:
        return len(row.positions) + position
    if position in row.positions:
        return row.positions.index(position)
    refuse(
        code="tv_record_invalid",
        message=f"Position {position} is not covered by the lens row for {row.address!r} "
        f"(covered: {row.positions}).",
        remedy="request one of the extracted positions, or re-extract with the position included",
    )


def prediction_trajectory(  # noqa: PLR0913 -- position/batch/tokenizer/target are the addressing axes
    predictions: Any,
    *,
    position: int = -1,
    batch_index: int = 0,
    tokenizer: Any | None = None,
    target_token_id: int | None = None,
) -> PredictionTrajectory:
    """Build the trajectory record from a streaming lens result.

    Parameters
    ----------
    predictions:
        A :class:`torchlens.semantic.logit_lens.LogitLensPredictions`.
    position:
        Sequence position to inspect (negative = from the end).
    batch_index:
        Batch element to inspect.
    tokenizer:
        Optional tokenizer for display strings.
    target_token_id:
        Optional tracked answer token; must be one of the ids the extractor
        retained (``token_ids``).

    Returns
    -------
    PredictionTrajectory
        The typed per-layer trajectory record.
    """

    layers: list[str] = []
    provenance: list[str] = []
    top_tokens: list[tuple[str, ...]] = []
    top_probs: list[tuple[float, ...]] = []
    target_probs: list[float] = []
    target_ranks: list[int] = []
    for row in predictions.rows:
        index = _row_position_index(row, position)
        layers.append(row.address)
        provenance.append(row.provenance)
        top_tokens.append(
            tuple(_decode(tokenizer, int(tid)) for tid in row.top_ids[batch_index, index])
        )
        top_probs.append(tuple(float(p) for p in row.top_probs[batch_index, index]))
        if target_token_id is not None:
            prob, rank = _target_values(row, target_token_id, batch_index, index)
            target_probs.append(prob)
            target_ranks.append(rank)
    target_token = None if target_token_id is None else _decode(tokenizer, target_token_id)
    return PredictionTrajectory(
        layers=tuple(layers),
        provenance=tuple(provenance),
        position=position,
        top_tokens=tuple(top_tokens),
        top_probs=tuple(top_probs),
        target_token=target_token,
        target_probs=tuple(target_probs) if target_token_id is not None else None,
        target_ranks=tuple(target_ranks) if target_token_id is not None else None,
        lens_source=predictions.lens_source,
        validated=predictions.validated,
        tie_convention=predictions.tie_convention,
    )


def _target_values(
    row: Any, target_token_id: int, batch_index: int, index: int
) -> tuple[float, int]:
    """Return one layer's tracked-token probability and one-based rank."""

    if target_token_id not in row.token_ids:
        refuse(
            code="tv_record_invalid",
            message=f"Token id {target_token_id} was not retained by the lens extraction "
            f"(retained: {row.token_ids}).",
            remedy="pass tokens= to logit_lens_predictions so the streaming extractor "
            "keeps the answer token's values",
        )
    slot = row.token_ids.index(target_token_id)
    return (
        float(row.token_probs[batch_index, index, slot]),
        int(row.token_ranks[batch_index, index, slot]),
    )


def _lens_footer(trajectory: PredictionTrajectory) -> list[str]:
    """Return the trajectory's honesty footer lines."""

    validated = "numerically validated" if trajectory.validated else "UNVALIDATED"
    lines = [
        f"lens: {trajectory.lens_source} ({validated}); probabilities use the "
        "full-vocabulary denominator",
    ]
    if trajectory.tie_convention:
        lines.append(f"ranks: one-based; {trajectory.tie_convention}")
    native_rows = sum(1 for p in trajectory.provenance if p == "native output")
    lines.append(
        f"{len(trajectory.layers) - native_rows} projected rows "
        f"(projected through final norm/head), {native_rows} native-output rows"
    )
    return lines


def render_prediction_ribbon(
    trajectory: PredictionTrajectory,
    path: Path | str,
    *,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render the layer-prediction ribbon (Ecco layer predictions, matched).

    Layers on the y axis in execution order, top-k token slots on the x
    axis; each cell prints the token and colors by its full-vocabulary
    probability on the fixed [0, 1] scale. Native-output rows are marked.

    Parameters
    ----------
    trajectory:
        The trajectory record.
    path:
        Output path (PNG/SVG/PDF by suffix).
    svg_fonttype:
        ``'path'`` (default) or ``'none'``.

    Returns
    -------
    Artifact
        The written file plus rendered disclosure lines.
    """

    matplotlib = require_matplotlib()
    n_layers = len(trajectory.layers)
    k = len(trajectory.top_tokens[0]) if trajectory.top_tokens else 0
    fig = figure(width=1.1 * k + 3.4, height=0.28 * n_layers + 1.6)
    ax = fig.add_subplot(111)
    import numpy as np

    probs = np.asarray(trajectory.top_probs, dtype=float)
    mesh = ax.pcolormesh(probs, cmap=ATTENTION_CMAP, vmin=0.0, vmax=1.0, rasterized=False)
    ax.invert_yaxis()
    for layer_row in range(n_layers):
        for slot in range(k):
            prob = probs[layer_row, slot]
            color = "white" if prob > 0.55 else "black"
            ax.text(
                slot + 0.5,
                layer_row + 0.5,
                trajectory.top_tokens[layer_row][slot],
                ha="center",
                va="center",
                fontsize=6,
                color=color,
            )
    labels = [
        layer + ("  [native]" if prov == "native output" else "")
        for layer, prov in zip(trajectory.layers, trajectory.provenance, strict=True)
    ]
    ax.set_yticks([row + 0.5 for row in range(n_layers)])
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xticks([slot + 0.5 for slot in range(k)])
    ax.set_xticklabels([f"top {slot + 1}" for slot in range(k)], fontsize=7)
    ax.tick_params(length=0)
    colorbar(fig, mesh, ax, label="probability (full-vocabulary denominator)")
    disclosures = _lens_footer(trajectory)
    fig.suptitle(f"layer predictions @ position {trajectory.position}", fontsize=9)
    fig.text(0.01, 0.01, "\n".join(disclosures), fontsize=6, va="bottom")
    written = save_figure(fig, path, svg_fonttype=svg_fonttype)
    del matplotlib
    return Artifact(
        paths=(written,),
        format=Path(path).suffix.lstrip(".").lower(),
        title="layer prediction ribbon",
        disclosure_lines=tuple(disclosures),
        provenance=f"lens: {trajectory.lens_source}",
        fingerprint="",
        svg_fonttype=svg_fonttype,
        pages_total=1,
    )


def render_answer_trajectory(
    trajectory: PredictionTrajectory,
    path: Path | str,
    *,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render the tracked answer token's probability + rank panels.

    Parameters
    ----------
    trajectory:
        The trajectory record; must carry a tracked target token.
    path:
        Output path (PNG/SVG/PDF by suffix).
    svg_fonttype:
        ``'path'`` (default) or ``'none'``.

    Returns
    -------
    Artifact
        The written file plus rendered disclosure lines.
    """

    if trajectory.target_probs is None or trajectory.target_ranks is None:
        refuse(
            code="tv_record_invalid",
            message="This trajectory tracks no target token, so there is no answer "
            "trajectory to draw.",
            remedy="build the trajectory with target_token_id= (and extract with "
            "tokens= so the streaming lens retains it)",
        )
    n_layers = len(trajectory.layers)
    fig = figure(width=7.0, height=4.6)
    prob_ax, rank_ax = fig.subplots(2, 1, sharex=True)
    xs = list(range(n_layers))
    prob_ax.plot(xs, trajectory.target_probs, marker="o", markersize=3)
    prob_ax.set_ylabel(f"p({trajectory.target_token!r})", fontsize=8)
    prob_ax.set_ylim(0.0, 1.0)
    rank_ax.plot(xs, trajectory.target_ranks, marker="o", markersize=3, color="#d55e00")
    rank_ax.set_yscale("log")
    rank_ax.invert_yaxis()  # rank 1 (best) at the top
    rank_ax.set_ylabel("rank (one-based, log scale)", fontsize=8)
    rank_ax.set_xticks(xs)
    rank_ax.set_xticklabels(trajectory.layers, rotation=90, fontsize=6)
    disclosures = _lens_footer(trajectory)
    fig.suptitle(
        f"answer trajectory: {trajectory.target_token!r} @ position {trajectory.position}",
        fontsize=9,
    )
    fig.text(0.01, 0.01, "\n".join(disclosures), fontsize=6, va="bottom")
    written = save_figure(fig, path, svg_fonttype=svg_fonttype)
    return Artifact(
        paths=(written,),
        format=Path(path).suffix.lstrip(".").lower(),
        title="answer trajectory",
        disclosure_lines=tuple(disclosures),
        provenance=f"lens: {trajectory.lens_source}",
        fingerprint="",
        svg_fonttype=svg_fonttype,
        pages_total=1,
    )


def prediction_table(
    predictions: Any,
    *,
    batch_index: int = 0,
    tokenizer: Any | None = None,
    context_tokens: tuple[str, ...] | None = None,
) -> PredictionTable:
    """Build the per-position top-k table from the NATIVE lens row.

    Parameters
    ----------
    predictions:
        A ``LogitLensPredictions`` whose final row is the native-output row
        (the default extraction includes it).
    batch_index:
        Batch element to inspect.
    tokenizer:
        Optional tokenizer for display strings.
    context_tokens:
        Optional input tokens per position (defaults to position numbers).

    Returns
    -------
    PredictionTable
        The typed positions x top-k record.
    """

    native_rows = [row for row in predictions.rows if row.provenance == "native output"]
    if not native_rows:
        refuse(
            code="tv_record_invalid",
            message="No native-output row in this lens result; a projected row must not "
            "silently serve the prediction table (PROJECTION-AS-PREDICTION).",
            remedy="extract with the native row included, or build the table from "
            "captured output logits directly",
        )
    row = native_rows[-1]
    positions = tuple(int(p) for p in row.positions)
    tokens = context_tokens or tuple(str(p) for p in positions)
    top_tokens = tuple(
        tuple(_decode(tokenizer, int(tid)) for tid in row.top_ids[batch_index, index])
        for index in range(len(positions))
    )
    top_probs = tuple(
        tuple(float(p) for p in row.top_probs[batch_index, index])
        for index in range(len(positions))
    )
    return PredictionTable(
        positions=positions,
        context_tokens=tokens,
        top_tokens=top_tokens,
        top_probs=top_probs,
        provenance=row.provenance,
    )


def prediction_table_html(table: PredictionTable) -> str:
    """Emit the zero-dependency HTML top-k table (inspectus, matched)."""

    import html as _html

    parts = ['<table style="border-collapse:collapse;font-size:11px">']
    parts.append(
        "<tr><th>position</th><th>token</th>"
        + "".join(f"<th>top {slot + 1}</th>" for slot in range(len(table.top_tokens[0])))
        + "</tr>"
    )
    for index, position in enumerate(table.positions):
        cells = [f"<td>{position}</td><td>{_html.escape(table.context_tokens[index])}</td>"]
        for token, prob in zip(table.top_tokens[index], table.top_probs[index], strict=True):
            cells.append(f'<td title="{prob:.4f}">{_html.escape(token)} ({prob:.2f})</td>')
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.append("</table>")
    parts.append(f'<div style="font-size:10px">source: {_html.escape(table.provenance)}</div>')
    return "".join(parts)


def render_prediction_table(
    table: PredictionTable,
    path: Path | str,
    *,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render the top-k table to paginated paper pages (whole positions).

    Parameters
    ----------
    table:
        The prediction-table record.
    path:
        Output path; pages write ``name-pageNN.ext``.
    svg_fonttype:
        ``'path'`` (default) or ``'none'``.

    Returns
    -------
    Artifact
        Written pages plus rendered disclosure lines.
    """

    import math as _math

    from ._mpl import paged_paths

    n_pages = _math.ceil(len(table.positions) / POSITIONS_PER_PAGE)
    targets = paged_paths(path, n_pages)
    written: list[Path] = []
    disclosures = [f"source: {table.provenance}", "probabilities: full-vocabulary denominator"]
    for page, target in enumerate(targets):
        start = page * POSITIONS_PER_PAGE
        rows = list(range(start, min(start + POSITIONS_PER_PAGE, len(table.positions))))
        fig = figure(width=1.3 * len(table.top_tokens[0]) + 2.4, height=0.26 * len(rows) + 1.4)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        cell_text = [
            [table.context_tokens[index]]
            + [
                f"{token} ({prob:.2f})"
                for token, prob in zip(table.top_tokens[index], table.top_probs[index], strict=True)
            ]
            for index in rows
        ]
        table_artist = ax.table(
            cellText=cell_text,
            rowLabels=[str(table.positions[index]) for index in rows],
            colLabels=["token"] + [f"top {slot + 1}" for slot in range(len(table.top_tokens[0]))],
            loc="center",
        )
        table_artist.auto_set_font_size(False)
        table_artist.set_fontsize(7)
        footer = "\n".join(disclosures)
        if n_pages > 1:
            footer = f"page {page + 1} of {n_pages}\n" + footer
        fig.text(0.01, 0.01, footer, fontsize=6, va="bottom")
        written.append(save_figure(fig, target, svg_fonttype=svg_fonttype))
    return Artifact(
        paths=tuple(written),
        format=Path(path).suffix.lstrip(".").lower(),
        title="top-k tokens per position",
        disclosure_lines=tuple(disclosures),
        provenance=table.provenance,
        fingerprint="",
        svg_fonttype=svg_fonttype,
        pages_total=n_pages,
    )

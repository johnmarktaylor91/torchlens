"""Layers-by-positions logit-lens heatmap over :class:`LogitLensResult`.

The nostalgebraist picture: one row per projected block, one column per
sequence position; each cell prints that layer's top-1 token at that
position and is colored by a probability on the fixed [0, 1] scale (the
top-1 probability, or a tracked target token's probability). Every row is
a projection through the final norm and head, never the model's own
output, and the footer says so together with the lens validation state.

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._errors import refuse
from ._mpl import ATTENTION_CMAP, colorbar, figure, require_matplotlib, save_figure
from ._records import Artifact

__all__ = ["render_logit_lens_heatmap"]

#: Widest column count drawn on one figure; wider requests name positions.
MAX_HEATMAP_POSITIONS = 48

#: Characters of a decoded token printed in a cell before clipping.
TOKEN_CLIP = 7

#: Font size (points) for cell text, tick labels and the footer.
_FONT = 10

_VALUES = ("top_prob", "target_prob")


def _position_logits(logits: torch.Tensor, batch_index: int) -> torch.Tensor:
    """Return one batch element's ``[positions, vocab]`` logits."""

    if logits.ndim == 3:
        return logits[batch_index]
    refuse(
        code="tv_record_invalid",
        message=f"The lens logits have shape {tuple(logits.shape)}; a layers-by-positions "
        "heatmap needs [batch, positions, vocab].",
        remedy="use render_prediction_ribbon for single-position lenses",
        shape=list(logits.shape),
    )


def _resolve_positions(n_positions: int, positions: Sequence[int] | None) -> list[int]:
    """Return absolute column positions, refusing silent crops."""

    if positions is None:
        if n_positions > MAX_HEATMAP_POSITIONS:
            refuse(
                code="tv_record_invalid",
                message=f"The lens covers {n_positions} positions; one heatmap draws at most "
                f"{MAX_HEATMAP_POSITIONS} columns and never crops silently.",
                remedy="pass positions= with the columns to draw",
                n_positions=n_positions,
            )
        return list(range(n_positions))
    resolved = [int(p) + n_positions if int(p) < 0 else int(p) for p in positions]
    bad = [p for p in resolved if not 0 <= p < n_positions]
    if bad or not resolved or len(resolved) > MAX_HEATMAP_POSITIONS:
        refuse(
            code="tv_record_invalid",
            message=f"positions={list(positions)!r} is empty, too wide, or outside the "
            f"{n_positions} captured positions.",
            remedy=f"pass 1 to {MAX_HEATMAP_POSITIONS} positions in range",
        )
    return resolved


def _decode(tokenizer: Any | None, token_id: int) -> str:
    """Decode one token id, falling back to the raw id string."""

    if tokenizer is None:
        return str(token_id)
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        return str(tokenizer.convert_ids_to_tokens([token_id])[0])
    return str(tokenizer.decode([token_id]))


def _heatmap_cells(
    result: Any,
    *,
    columns: list[int],
    batch_index: int,
    target_token_id: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(color values, top-1 ids)``, each ``[n_layers, n_columns]``.

    Probabilities use the full-vocabulary softmax, one layer at a time, so
    no ``[layers, positions, vocab]`` probability tensor is ever held.
    """

    values = np.zeros((len(result.entries), len(columns)), dtype=float)
    top_ids = np.zeros((len(result.entries), len(columns)), dtype=np.int64)
    index = torch.tensor(columns, dtype=torch.long)
    for row, entry in enumerate(result.entries):
        logits = _position_logits(entry.logits, batch_index).index_select(0, index)
        probs = torch.softmax(logits.detach().float(), dim=-1)
        top_prob, top_id = probs.max(dim=-1)
        top_ids[row] = top_id.cpu().numpy()
        if target_token_id is None:
            values[row] = top_prob.cpu().numpy()
        else:
            values[row] = probs[:, target_token_id].cpu().numpy()
    return values, top_ids


def _clip(token: str) -> tuple[str, bool]:
    """Clip one decoded token for a cell; report whether it was clipped.

    Surrounding spaces are not drawn (the footer says so); a whitespace-only
    token prints quoted with escapes (``'\\n'``) so its cell never looks
    empty.
    """

    token = token.strip() or repr(token)
    if len(token) <= TOKEN_CLIP:
        return token, False
    return token[: TOKEN_CLIP - 1] + "~", True


def _validate_request(
    result: Any, value: str, target_token_id: int | None, tokens: Sequence[str] | None
) -> int:
    """Validate the request; return the captured position count."""

    if value not in _VALUES:
        refuse(
            code="tv_record_invalid",
            message=f"value={value!r} is not one of {_VALUES}.",
            remedy="use 'top_prob' or 'target_prob' with target_token_id=",
        )
    if (value == "target_prob") != (target_token_id is not None):
        refuse(
            code="tv_record_invalid",
            message="value='target_prob' and target_token_id= go together.",
            remedy="pass both, or neither for the top-1 probability",
        )
    if not result.entries:
        refuse(
            code="tv_record_invalid",
            message="The lens result has no projected layers.",
            remedy="run logit_lens on a trace whose blocks expose the lens facet",
        )
    first = _position_logits(result.entries[0].logits, 0)
    n_positions, vocab = int(first.shape[0]), int(first.shape[-1])
    if target_token_id is not None and not 0 <= target_token_id < vocab:
        refuse(
            code="tv_record_invalid",
            message=f"target_token_id={target_token_id} is outside the vocabulary ({vocab}).",
            remedy="pass a token id the head can emit",
        )
    if tokens is not None and len(tokens) != n_positions:
        refuse(
            code="tv_record_invalid",
            message=f"tokens= has {len(tokens)} labels for {n_positions} positions.",
            remedy="pass one input-token label per captured position",
        )
    return n_positions


def _footer(
    result: Any,
    *,
    value: str,
    target: str | None,
    batch_index: int,
    clipped: bool,
) -> list[str]:
    """Return the heatmap's honesty footer lines."""

    validated = "numerically validated" if result.validated else "UNVALIDATED"
    color = (
        "color: probability of each cell's top-1 token"
        if value == "top_prob"
        else f"color: probability of target token {target!r}"
    )
    lines = [
        f"lens: {result.lens_source} ({validated}); probabilities use the "
        "full-vocabulary denominator",
        f"{len(result.entries)} rows projected through final norm/head from facet "
        f"{result.facet!r}; none is the model's own output",
        f"{color}; text: each cell's top-1 token, surrounding spaces not shown; "
        f"batch element {batch_index}",
    ]
    if clipped:
        lines.append(f"token text clipped to {TOKEN_CLIP} characters (~ marks a clip)")
    return lines


def render_logit_lens_heatmap(  # noqa: PLR0913 -- addressing, labeling and output axes
    result: Any,
    path: Path | str,
    *,
    value: str = "top_prob",
    target_token_id: int | None = None,
    positions: Sequence[int] | None = None,
    tokens: Sequence[str] | None = None,
    tokenizer: Any | None = None,
    batch_index: int = 0,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render the layers-by-positions logit-lens heatmap.

    Parameters
    ----------
    result:
        A :class:`torchlens.semantic.logit_lens.LogitLensResult` whose
        per-layer logits are ``[batch, positions, vocab]``.
    path:
        Output path (PNG/SVG/PDF by suffix).
    value:
        ``"top_prob"`` (color = the top-1 token's probability) or
        ``"target_prob"`` (color = ``target_token_id``'s probability).
    target_token_id:
        Tracked token id; required with ``value="target_prob"``.
    positions:
        Column positions to draw (negative counts from the end); default
        all, refused above ``MAX_HEATMAP_POSITIONS`` rather than cropped.
    tokens:
        Optional input-token labels, one per captured position.
    tokenizer:
        Optional tokenizer used to decode the predicted token ids.
    batch_index:
        Batch element to draw.
    svg_fonttype:
        ``'path'`` (default) or ``'none'``.

    Returns
    -------
    Artifact
        The written file plus every rendered disclosure line.
    """

    n_positions = _validate_request(result, value, target_token_id, tokens)
    columns = _resolve_positions(n_positions, positions)
    values, top_ids = _heatmap_cells(
        result, columns=columns, batch_index=batch_index, target_token_id=target_token_id
    )
    n_layers = len(result.entries)
    cells = [
        [_clip(_decode(tokenizer, int(top_ids[row, col]))) for col in range(len(columns))]
        for row in range(n_layers)
    ]
    target = None if target_token_id is None else _decode(tokenizer, target_token_id)
    disclosures = _footer(
        result,
        value=value,
        target=target,
        batch_index=batch_index,
        clipped=any(clipped for line in cells for _, clipped in line),
    )
    require_matplotlib()
    labels = [_clip(str(tokens[p]))[0] if tokens is not None else str(p) for p in columns]
    longest = max(len(label) for label in labels)
    label_room = 0.5 + (0.09 * longest if tokens is not None else 0.0)
    footer_room = 0.18 * len(disclosures) + 0.05
    height = 0.36 * n_layers + 0.65 + label_room + footer_room
    fig = figure(width=0.68 * len(columns) + 2.9, height=height)
    fig.subplots_adjust(bottom=(label_room + footer_room) / height, top=1 - 0.45 / height)
    ax = fig.add_subplot(111)
    mesh = ax.pcolormesh(values, cmap=ATTENTION_CMAP, vmin=0.0, vmax=1.0, rasterized=False)
    ax.invert_yaxis()
    for row, line in enumerate(cells):
        for col, (text, _) in enumerate(line):
            ax.text(
                col + 0.5,
                row + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=_FONT,
                color="black" if values[row, col] > 0.55 else "white",
            )
    ax.set_yticks([row + 0.5 for row in range(n_layers)])
    ax.set_yticklabels(
        [f"{entry.layer_index}: {entry.address}" for entry in result.entries], fontsize=_FONT
    )
    ax.set_xticks([col + 0.5 for col in range(len(columns))])
    ax.set_xticklabels(labels, fontsize=_FONT, rotation=90 if tokens is not None else 0)
    ax.set_xlabel(
        "input position" + (" (input token)" if tokens is not None else ""), fontsize=_FONT
    )
    ax.set_ylabel("projected layer", fontsize=_FONT)
    ax.tick_params(length=0)
    bar = colorbar(fig, mesh, ax, label="probability (full-vocabulary denominator)")
    bar.ax.tick_params(labelsize=_FONT)
    bar.ax.yaxis.label.set_size(_FONT)
    fig.suptitle("logit lens: top-1 token per layer and position", fontsize=10)
    fig.text(0.01, 0.01, "\n".join(disclosures), fontsize=_FONT - 0.5, va="bottom")
    written = save_figure(fig, path, svg_fonttype=svg_fonttype)
    return Artifact(
        paths=(written,),
        format=Path(path).suffix.lstrip(".").lower(),
        title="logit lens heatmap",
        disclosure_lines=tuple(disclosures),
        provenance=f"lens: {result.lens_source}",
        fingerprint="",
        svg_fonttype=svg_fonttype,
        pages_total=1,
    )

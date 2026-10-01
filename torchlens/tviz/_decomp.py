"""Score decomposition: the BertViz neuron-view DATA, term-complete (D19).

For one selected (layer, head, destination, source) pair, the record carries
the post-transform q/k vectors, their per-dimension products, the scale, and
every additive term, under a HARD sum-to-score invariant: if the terms do not
reproduce the captured scores facet within tolerance, the record REFUSES
rather than mis-decomposing. gpt2/bert-class families close today; RoPE
families refuse per-family until term extraction covers them. The full
neuron BROWSER is skipped (per-architecture reverse engineering is the
treadmill that killed it); the selected-pair static card ships wherever the
terms close.

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from ._errors import refuse
from ._mpl import DIVERGING_CMAP, figure, require_matplotlib, save_figure
from ._records import Artifact, ScoreDecomposition

__all__ = ["render_neuron_card", "score_decomposition"]


def _facet_value(module: Any, name: str, address: str) -> torch.Tensor:
    """Read one facet tensor or refuse with the capture remedy."""

    view = getattr(module, "facets", None)
    if view is None or name not in view:
        refuse(
            code="tv_facet_missing",
            message=f"Module {address!r} exposes no {name!r} facet.",
            remedy="capture an attention module with q/k/scores facets (eager "
            "attention; see tl.compat.report)",
            address=address,
        )
    value = view[name].value
    if not isinstance(value, torch.Tensor):
        refuse(
            code="tv_payload_missing",
            message=f"The {name!r} facet at {address!r} has no captured payload.",
            remedy="recapture with the attention ops saved",
            address=address,
        )
    return value.detach().to(torch.float32)


def score_decomposition(  # noqa: PLR0913 -- one keyword per addressed coordinate (D19)
    trace: Any,
    layer: str,
    *,
    head: int,
    destination: int,
    source: int,
    batch_index: int = 0,
    tolerance: float = 1e-3,
) -> ScoreDecomposition:
    """Decompose one attention score into its per-dimension terms (D19).

    Reads the ``q``/``k`` facets (position-major ``[batch, pos, head,
    d_head]``) and the ``scores`` facet (head-major ``[batch, head, dst,
    src]``), reproduces ``score = q . k / scale (+ additive terms)``, and
    returns the term-complete record -- or refuses
    ``tv_decomposition_unclosed`` when the family's terms do not close
    (RoPE-class families, by design).

    Parameters
    ----------
    trace:
        A finished TorchLens trace.
    layer:
        Attention module address.
    head:
        Query head index.
    destination:
        Query (row) position.
    source:
        Key (column) position.
    batch_index:
        Batch element.
    tolerance:
        Relative closure tolerance for the sum-to-score invariant.

    Returns
    -------
    ScoreDecomposition
        The validated term-complete record.
    """

    if layer not in getattr(trace, "modules", {}):
        refuse(
            code="tv_facet_missing",
            message=f"No module at address {layer!r} in this trace.",
            remedy="pass a pattern-bearing attention module address",
        )
    module = trace.modules[layer]
    q = _facet_value(module, "q", layer)[batch_index, destination, head]
    k = _facet_value(module, "k", layer)[batch_index, source, head]
    scores = _facet_value(module, "scores", layer)[batch_index, head]
    reference = float(scores[destination, source])
    products = q * k
    d_head = q.shape[-1]
    scale = math.sqrt(d_head)
    additive: dict[str, float] = {}
    dot_term = float(products.sum()) / scale
    remainder = reference - dot_term
    if abs(remainder) > tolerance * max(1.0, abs(reference)):
        # One additive-mask term may close the gap exactly (an additively
        # masked position); anything else stays open and refuses inside the
        # record constructor.
        floor = torch.finfo(torch.float32).min
        if remainder <= floor / 4:
            additive["mask"] = remainder
    return ScoreDecomposition(
        layer=layer,
        head=head,
        destination=destination,
        source=source,
        query_vector=q,
        key_vector=k,
        products=products,
        scale=scale,
        additive_terms=additive,
        reference_score=reference,
        tolerance=tolerance,
    )


def render_neuron_card(
    decomposition: ScoreDecomposition,
    path: Path | str,
    *,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render the selected-pair neuron card (family-gated static card, D19).

    Three aligned bands -- q vector, k vector, and their per-dimension
    products -- plus the printed term accounting ending at the captured
    score. Only constructible from a CLOSED decomposition (the record
    refuses otherwise), so the card can never mis-decompose.

    Parameters
    ----------
    decomposition:
        A validated :class:`ScoreDecomposition`.
    path:
        Output path (PNG/SVG/PDF by suffix).
    svg_fonttype:
        ``'path'`` (default) or ``'none'``.

    Returns
    -------
    Artifact
        The written file plus the printed term accounting.
    """

    matplotlib = require_matplotlib()
    bands = [
        ("q (post-transform)", decomposition.query_vector),
        ("k (post-transform)", decomposition.key_vector),
        ("q_i * k_i", decomposition.products),
    ]
    fig = figure(width=8.0, height=3.4)
    axes = fig.subplots(len(bands), 1, sharex=True)
    cmap = matplotlib.colormaps[DIVERGING_CMAP]
    for ax, (label, vector) in zip(axes, bands, strict=True):
        data = vector.unsqueeze(0).numpy()
        scale = max(1e-12, float(vector.abs().max()))
        ax.pcolormesh(data, cmap=cmap, vmin=-scale, vmax=scale, rasterized=False)
        ax.set_yticks([0.5])
        ax.set_yticklabels([label], fontsize=7)
        ax.tick_params(length=0)
    axes[-1].set_xlabel(f"dimension (d_head = {len(decomposition.products)})", fontsize=7)
    terms = " + ".join(
        [f"sum(q_i*k_i)/{decomposition.scale:.3f} = {decomposition.total:.4f}"]
        + [f"{name} {value:+.4f}" for name, value in decomposition.additive_terms.items()]
    )
    disclosures = [
        f"{decomposition.layer} / head {decomposition.head}: query position "
        f"{decomposition.destination} <- key position {decomposition.source}",
        f"terms: {terms}; captured score: {decomposition.reference_score:.4f} "
        f"(closed within tol {decomposition.tolerance:g})",
    ]
    fig.suptitle("score decomposition (term-complete)", fontsize=9)
    fig.text(0.01, 0.01, "\n".join(disclosures), fontsize=6, va="bottom")
    written = save_figure(fig, path, svg_fonttype=svg_fonttype)
    return Artifact(
        paths=(written,),
        format=Path(path).suffix.lstrip(".").lower(),
        title="neuron card",
        disclosure_lines=tuple(disclosures),
        provenance="captured q/k/scores facets",
        fingerprint="",
        svg_fonttype=svg_fonttype,
        pages_total=1,
    )

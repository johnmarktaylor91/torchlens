"""repeng bridge helpers.

Builds a real ``repeng.ControlVector`` from contrastive saved TorchLens
activations. ``ControlVector.train`` runs the model itself and then computes the
direction in ``repeng.extract.read_representations``; this bridge replicates that
direction math on the saved last-token activations instead of re-running the
model. The private helper ``_read_directions`` is shared with the dialz bridge,
whose ``read_representations`` is a fork of repeng's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ._contrastive import (
    _contrastive_rows,
    _ContrastiveRead,
    _interleave,
    _model_type,
    _read_directions,
)


def control_vector(  # noqa: PLR0913 -- mirrors upstream repeng signature
    log: Any,
    positive_site: Any,
    negative_site: Any | None = None,
    *,
    layer: int,
    negative_log: Any | None = None,
    read_token_index: int | Sequence[int] | None = -1,
    attention_mask: torch.Tensor | None = None,
    negative_attention_mask: torch.Tensor | None = None,
    method: str = "pca_diff",
    model_type: str | None = None,
) -> dict[str, Any]:
    """Build a ``repeng.ControlVector`` from contrastive saved TorchLens outs.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` of the positive prompts (one prompt per batch row).
    positive_site:
        Site whose saved out holds the positive hidden states. repeng's layer
        ``i`` reads ``hidden_states[i + 1]``: the output of decoder layer ``i``
        (``"model.layers.<i>"``), except for the last layer, where Hugging Face
        returns the final-norm output (``"model.norm"``).
    negative_site:
        Site holding the negative hidden states. Defaults to ``positive_site``
        when ``negative_log`` is given.
    layer:
        Layer index the direction is keyed by in ``ControlVector.directions``;
        ``ControlModel`` applies it to that decoder layer.
    negative_log:
        Optional TorchLens ``Trace`` of the negative prompts. A layer-object
        site is re-resolved there by its label.
    read_token_index:
        Token position read per prompt (default ``-1``, the last token, which is
        what ``ControlVector.train`` reads); a sequence gives one position per
        prompt; ``None`` passes ``[n_prompts, hidden]`` outs unsliced.
    attention_mask:
        ``[n_prompts, n_tokens]`` mask of the positive prompts (1 for real
        tokens). Read from the trace's saved ``attention_mask`` input when
        omitted. With a mask, ``read_token_index`` counts within each prompt's
        unpadded tokens (``-1`` is the last real token), so right- or
        left-padded batches of unequal-length prompts read the right token.
    negative_attention_mask:
        Mask of the negative prompts. Defaults to ``attention_mask`` when both
        sites live in ``log``, else to ``negative_log``'s saved mask.
    method:
        repeng's training method: ``"pca_diff"`` (default), ``"pca_center"``, or
        ``"umap"`` (needs the ``umap`` package).
    model_type:
        ``ControlVector.model_type``. Defaults to the traced model's
        ``config.model_type``.

    Returns
    -------
    dict[str, Any]
        Payload with ``control_vector`` (a ``repeng.ControlVector``) and the
        ``positive``/``negative`` rows.

    Raises
    ------
    ImportError
        If repeng is unavailable.
    ValueError
        If no negative activations are given, the rows do not line up or are
        identical, the method is unknown, or no model type can be found.
    """

    try:
        import repeng as repeng_module
    except ImportError as exc:
        raise ImportError(
            "repeng bridge requires the `repeng` extra: install torchlens[repeng]."
        ) from exc

    positive, negative = _contrastive_rows(
        log,
        positive_site,
        negative_site,
        _ContrastiveRead(
            negative_log=negative_log,
            read_token_index=read_token_index,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
        ),
    )
    hiddens = _interleave(positive, negative)
    direction = _read_directions(
        hiddens, method, diff_methods=("pca_diff",), center_in_place=True, mean_diff=False
    )
    result = repeng_module.ControlVector(
        model_type=_model_type(log, model_type), directions={int(layer): direction}
    )
    return {
        "schema": "torchlens.repeng.v1",
        "control_vector": result,
        "positive": positive,
        "negative": negative,
    }


__all__ = ["control_vector"]

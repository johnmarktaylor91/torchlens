"""steering-vectors bridge helpers.

Builds a steering-vectors ``SteeringVector`` from contrastive saved TorchLens
activations, doing what ``steering_vectors.train_steering_vector`` does after its
own forward passes: read one token per prompt, then apply an aggregator (default
``steering_vectors.aggregators.mean_aggregator()``) to the positive and negative
rows.

The row reading is shared with the repeng and dialz bridges
(``torchlens.bridge._contrastive``), which train from the same layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ._contrastive import _contrastive_rows


def vector(
    log: Any,
    positive_site: Any,
    negative_site: Any | None = None,
    *,
    negative_log: Any | None = None,
    read_token_index: int | Sequence[int] | None = -1,
    attention_mask: torch.Tensor | None = None,
    negative_attention_mask: torch.Tensor | None = None,
    trainer: Any | None = None,
    layer: int | None = None,
    layer_type: str = "decoder_block",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a steering vector from contrastive saved TorchLens outs.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` of the positive prompts (one prompt per batch row).
    positive_site:
        Site whose saved out holds the positive activations, shaped
        ``[n_prompts, n_tokens, hidden]`` (or ``[n_prompts, hidden]`` with
        ``read_token_index=None``).
    negative_site:
        Site holding the negative activations. Defaults to ``positive_site``
        when ``negative_log`` is given.
    negative_log:
        Optional TorchLens ``Trace`` of the negative prompts. Contrastive prompts
        usually live in two traces; without it both sites resolve in ``log``. A
        layer-object site is re-resolved in ``negative_log`` by its label.
    read_token_index:
        Token position read from every prompt before the trainer runs (default
        ``-1``, the last token, as in ``train_steering_vector``). A sequence gives
        one position per prompt; ``None`` passes the outs unsliced.
    attention_mask:
        ``[n_prompts, n_tokens]`` mask of the positive prompts (1 for real
        tokens). Read from the trace's saved ``attention_mask`` input when
        omitted. With a mask, ``read_token_index`` counts within each prompt's
        unpadded tokens (``-1`` is the last real token), so right- or
        left-padded batches of unequal-length prompts read the right token.
    negative_attention_mask:
        Mask of the negative prompts. Defaults to ``attention_mask`` when both
        sites live in ``log``, else to ``negative_log``'s saved mask.
    trainer:
        Aggregator called as ``trainer(positive_rows, negative_rows, **kwargs)``
        with ``[n_prompts, hidden]`` rows. Defaults to
        ``steering_vectors.aggregators.mean_aggregator()``; any steering-vectors
        aggregator (``pca_aggregator()``, ``logistic_aggregator()``) fits.
    layer:
        Optional layer number the site corresponds to. When given, the payload
        also carries a real ``steering_vectors.SteeringVector`` keyed by it,
        ready for ``patch_activations`` / ``apply``.
    layer_type:
        steering-vectors layer type for that ``SteeringVector`` (default
        ``"decoder_block"``).
    **kwargs:
        Additional keyword arguments forwarded to the trainer.

    Returns
    -------
    dict[str, Any]
        Payload with ``vector`` (the aggregated tensor), ``steering_vector``
        (a ``SteeringVector`` or ``None``), and the ``positive``/``negative`` rows.

    Raises
    ------
    ImportError
        If steering-vectors is unavailable.
    ValueError
        If no negative activations are given, the rows do not line up, or the
        positive and negative rows are identical.
    """

    try:
        import steering_vectors as steering_module
    except ImportError as exc:
        raise ImportError(
            "steering-vectors bridge requires the `steering` extra: install torchlens[steering]."
        ) from exc

    positive, negative = _contrastive_rows(
        log,
        positive_site,
        negative_site,
        negative_log=negative_log,
        read_token_index=read_token_index,
        attention_mask=attention_mask,
        negative_attention_mask=negative_attention_mask,
    )
    train = trainer if trainer is not None else steering_module.mean_aggregator()
    result = train(positive, negative, **kwargs)
    steering_vector = None
    if layer is not None:
        steering_vector = steering_module.SteeringVector({int(layer): result}, layer_type)
    return {
        "schema": "torchlens.steering_vectors.v1",
        "vector": result,
        "steering_vector": steering_vector,
        "positive": positive,
        "negative": negative,
    }


__all__ = ["vector"]

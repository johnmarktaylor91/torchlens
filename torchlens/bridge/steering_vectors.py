"""steering-vectors bridge helpers.

Builds a steering-vectors ``SteeringVector`` from contrastive saved TorchLens
activations, doing what ``steering_vectors.train_steering_vector`` does after its
own forward passes: read one token per prompt, then apply an aggregator (default
``steering_vectors.aggregators.mean_aggregator()``) to the positive and negative
rows.

The private helper ``_contrastive_rows`` is shared with the repeng and dialz
bridges, which train from the same positive/negative row layout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ._utils import out_at


def vector(
    log: Any,
    positive_site: Any,
    negative_site: Any | None = None,
    *,
    negative_log: Any | None = None,
    read_token_index: int | Sequence[int] | None = -1,
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
        usually live in two traces; without it both sites resolve in ``log``.
    read_token_index:
        Token position read from every prompt before the trainer runs (default
        ``-1``, the last token, as in ``train_steering_vector``). A sequence gives
        one position per prompt (for padded batches); ``None`` passes the outs
        unsliced.
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
        If no negative activations are given or the rows do not line up.
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


def _contrastive_rows(
    log: Any,
    positive_site: Any,
    negative_site: Any | None,
    *,
    negative_log: Any | None,
    read_token_index: int | Sequence[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned ``[n_prompts, hidden]`` positive and negative rows.

    Parameters
    ----------
    log:
        Trace holding the positive site.
    positive_site:
        Positive site.
    negative_site:
        Negative site; defaults to ``positive_site`` when ``negative_log`` is set.
    negative_log:
        Optional trace holding the negative site.
    read_token_index:
        Token position(s) to read, or ``None`` for unsliced outs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Positive and negative rows, detached.

    Raises
    ------
    ValueError
        If no negative site is given, or the two sides disagree in shape.
    """

    if negative_site is None:
        if negative_log is None:
            raise ValueError(
                "Contrastive steering needs negative activations: pass negative_site=, "
                "or negative_log= (a trace of the negative prompts; the site then "
                "defaults to positive_site)."
            )
        negative_site = positive_site
    negative_trace = log if negative_log is None else negative_log
    positive = _read_rows(out_at(log, positive_site), read_token_index, "positive")
    negative = _read_rows(out_at(negative_trace, negative_site), read_token_index, "negative")
    if positive.shape != negative.shape:
        raise ValueError(
            f"Positive rows {tuple(positive.shape)} and negative rows "
            f"{tuple(negative.shape)} must match: trace one negative prompt per "
            "positive prompt, padded to the same layout."
        )
    return positive, negative


def _read_rows(
    out: torch.Tensor, read_token_index: int | Sequence[int] | None, side: str
) -> torch.Tensor:
    """Slice one token per prompt out of a ``[n, tokens, hidden]`` out.

    Parameters
    ----------
    out:
        Saved out tensor.
    read_token_index:
        Token position, per-prompt positions, or ``None`` for no slicing.
    side:
        ``"positive"`` or ``"negative"`` for error messages.

    Returns
    -------
    torch.Tensor
        Detached rows.

    Raises
    ------
    ValueError
        If the out has too few dimensions or the positions do not match the rows.
    """

    out = out.detach()
    if read_token_index is None:
        return out
    if out.dim() < 3:
        raise ValueError(
            f"The {side} out has shape {tuple(out.shape)}; reading a token needs "
            "[n_prompts, n_tokens, hidden]. Pass read_token_index=None for outs "
            "that are already one row per prompt."
        )
    if isinstance(read_token_index, int):
        return out[:, read_token_index]
    indices = torch.as_tensor(list(read_token_index), dtype=torch.long, device=out.device)
    if indices.numel() != out.shape[0]:
        raise ValueError(
            f"read_token_index lists {indices.numel()} positions for {out.shape[0]} {side} prompts."
        )
    return out[torch.arange(out.shape[0], device=out.device), indices]


__all__ = ["vector"]

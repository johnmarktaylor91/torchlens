"""Shared helpers for the contrastive steering bridges.

``steering_vectors``, ``repeng`` and ``dialz`` all train from the same layout:
one positive and one negative row per prompt pair, read at one token per
prompt. This module reads those rows from saved TorchLens outs (padding-aware)
and replicates the direction math of repeng's ``read_representations``, which
dialz forked.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ._utils import out_at, resolve_one_site, source_model

ReadIndex = int | Sequence[int] | None


def _contrastive_rows(
    log: Any,
    positive_site: Any,
    negative_site: Any | None,
    *,
    negative_log: Any | None,
    read_token_index: ReadIndex,
    attention_mask: torch.Tensor | None = None,
    negative_attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned, distinct ``[n_prompts, hidden]`` positive and negative rows.

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
    attention_mask:
        ``[n_prompts, n_tokens]`` mask of the positive prompts; read from the
        trace's ``attention_mask`` input when omitted.
    negative_attention_mask:
        Mask of the negative prompts; defaults to ``attention_mask`` when both
        sides live in ``log``, else is read from ``negative_log``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Positive and negative rows, detached.

    Raises
    ------
    ValueError
        If no negative site is given, the two sides disagree in shape, or the
        rows are identical (the vector would be all zeros).
    """

    negative_trace, negative_site = _negative_source(
        log, positive_site, negative_site, negative_log
    )
    positive_mask = attention_mask if attention_mask is not None else _captured_mask(log)
    if negative_attention_mask is not None:
        negative_mask: torch.Tensor | None = negative_attention_mask
    elif negative_log is None:
        negative_mask = positive_mask
    else:
        negative_mask = _captured_mask(negative_log)
    positive = _read_rows(out_at(log, positive_site), read_token_index, positive_mask, "positive")
    negative = _read_rows(
        out_at(negative_trace, negative_site), read_token_index, negative_mask, "negative"
    )
    if positive.shape != negative.shape:
        raise ValueError(
            f"Positive rows {tuple(positive.shape)} and negative rows "
            f"{tuple(negative.shape)} must match: trace one negative prompt per "
            "positive prompt."
        )
    if torch.equal(positive, negative):
        raise ValueError(
            "The positive and negative rows are identical, so the steering vector "
            "would be all zeros. Check that the negative site resolves in the "
            "negative prompts' trace (negative_log=), not the positive one."
        )
    return positive, negative


def _negative_source(
    log: Any, positive_site: Any, negative_site: Any | None, negative_log: Any | None
) -> tuple[Any, Any]:
    """Return the trace and site that hold the negative activations.

    A layer object belongs to the trace it came from, so when ``negative_log``
    is given a layer-object site is re-resolved there by its ``layer_label``.

    Parameters
    ----------
    log:
        Positive trace.
    positive_site:
        Positive site.
    negative_site:
        Explicit negative site, if any.
    negative_log:
        Negative trace, if any.

    Returns
    -------
    tuple[Any, Any]
        ``(trace, site)`` for the negative side.

    Raises
    ------
    ValueError
        If neither a negative site nor a negative trace is given.
    """

    if negative_site is None:
        if negative_log is None:
            raise ValueError(
                "Contrastive steering needs negative activations: pass negative_site=, "
                "or negative_log= (a trace of the negative prompts; the site then "
                "defaults to positive_site)."
            )
        negative_site = positive_site
    if negative_log is None:
        return log, negative_site
    if hasattr(negative_site, "out") and hasattr(negative_site, "layer_label"):
        negative_site = resolve_one_site(negative_log, negative_site.layer_label)
    return negative_log, negative_site


def _captured_mask(log: Any) -> torch.Tensor | None:
    """Return the saved ``attention_mask`` input of a trace, if any.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.

    Returns
    -------
    torch.Tensor | None
        The saved mask, or ``None`` when the forward pass had none (or it was
        not saved).
    """

    for layer in getattr(log, "layer_list", []):
        role = getattr(layer, "io_role", None)
        if not getattr(layer, "is_input", False) or not isinstance(role, str):
            continue
        out = getattr(layer, "out", None)
        if role.split(".")[-1] == "attention_mask" and isinstance(out, torch.Tensor):
            return out
    return None


def _read_rows(
    out: torch.Tensor,
    read_token_index: ReadIndex,
    attention_mask: torch.Tensor | None,
    side: str,
) -> torch.Tensor:
    """Slice one token per prompt out of a ``[n, tokens, hidden]`` out.

    With a mask, positions count within each prompt's unpadded tokens, as
    steering-vectors' ``adjust_read_indices_for_padding`` does: ``-1`` is the
    last real token (what repeng and dialz read), ``0`` the first.

    Parameters
    ----------
    out:
        Saved out tensor.
    read_token_index:
        Token position, per-prompt positions, or ``None`` for no slicing.
    attention_mask:
        ``[n, tokens]`` mask (1 for real tokens), or ``None`` for absolute positions.
    side:
        ``"positive"`` or ``"negative"`` for error messages.

    Returns
    -------
    torch.Tensor
        Detached rows.

    Raises
    ------
    ValueError
        If the out has too few dimensions or the positions or mask do not match.
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
        indices = torch.full((out.shape[0],), read_token_index, dtype=torch.long, device=out.device)
    else:
        indices = torch.as_tensor(list(read_token_index), dtype=torch.long, device=out.device)
    if indices.numel() != out.shape[0]:
        raise ValueError(
            f"read_token_index lists {indices.numel()} positions for {out.shape[0]} {side} prompts."
        )
    if attention_mask is not None:
        indices = _adjust_for_padding(indices, attention_mask, out.shape[:2], side)
    return out[torch.arange(out.shape[0], device=out.device), indices]


def _adjust_for_padding(
    indices: torch.Tensor, attention_mask: torch.Tensor, rows_tokens: Any, side: str
) -> torch.Tensor:
    """Map positions within each prompt's real tokens to absolute positions.

    Parameters
    ----------
    indices:
        ``[n]`` positions relative to each prompt's unpadded tokens.
    attention_mask:
        ``[n, tokens]`` mask.
    rows_tokens:
        ``(n, tokens)`` of the out being read.
    side:
        ``"positive"`` or ``"negative"`` for error messages.

    Returns
    -------
    torch.Tensor
        Absolute ``[n]`` positions.

    Raises
    ------
    ValueError
        If the mask shape does not match the out or a prompt has no real token.
    """

    mask = attention_mask.to(indices.device) == 1
    if tuple(mask.shape) != tuple(rows_tokens):
        raise ValueError(
            f"The {side} attention_mask has shape {tuple(mask.shape)}; the out "
            f"reads [n_prompts, n_tokens] = {tuple(rows_tokens)}."
        )
    if not bool(mask.any(dim=1).all()):
        raise ValueError(f"A {side} prompt has no unmasked token in its attention_mask.")
    positions = torch.arange(mask.shape[1], device=indices.device).expand_as(mask)
    start = torch.where(mask, positions, mask.shape[1]).min(dim=1).values
    end = torch.where(mask, positions, -1).max(dim=1).values
    lengths = end - start + 1
    return torch.where(indices < 0, indices + lengths, indices) + start


def _interleave(positive: Any, negative: Any) -> np.ndarray:
    """Stack rows as ``[pos0, neg0, pos1, neg1, ...]`` float32, as repeng does.

    Parameters
    ----------
    positive:
        ``[n, hidden]`` positive rows.
    negative:
        ``[n, hidden]`` negative rows.

    Returns
    -------
    np.ndarray
        ``[2n, hidden]`` float32 array.
    """

    pos = positive.cpu().float().numpy()
    neg = negative.cpu().float().numpy()
    hiddens = np.empty((pos.shape[0] * 2, *pos.shape[1:]), dtype=np.float32)
    hiddens[::2] = pos
    hiddens[1::2] = neg
    return hiddens


def _read_directions(
    hiddens: np.ndarray,
    method: str,
    *,
    diff_methods: tuple[str, ...],
    center_in_place: bool,
    mean_diff: bool,
) -> np.ndarray:
    """Replicate the per-layer direction math of ``read_representations``.

    Parameters
    ----------
    hiddens:
        ``[2n, hidden]`` interleaved positive/negative rows.
    method:
        Training method name.
    diff_methods:
        Names that mean PCA over positive-minus-negative differences
        (repeng: ``pca_diff``; dialz 1.x: ``pca``; dialz 0.2: ``pca_diff``).
    center_in_place:
        repeng centers ``h`` in place for ``pca_center`` (so the sign check
        projects the centered rows); dialz centers a copy.
    mean_diff:
        Whether ``mean_diff`` (dialz only) is accepted.

    Returns
    -------
    np.ndarray
        The signed direction.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """

    h = hiddens
    if method in diff_methods or (mean_diff and method == "mean_diff"):
        train = h[::2] - h[1::2]
    elif method == "pca_center":
        center = (h[::2] + h[1::2]) / 2
        train = h if center_in_place else h.copy()
        train[::2] -= center
        train[1::2] -= center
    elif method == "umap":
        train = h
    else:
        known = [*diff_methods, "pca_center", "umap", *(["mean_diff"] if mean_diff else [])]
        raise ValueError(f"Unknown method {method!r}; expected one of {known}.")
    direction = _fit_direction(train, method)
    projected = (h @ direction) / np.linalg.norm(direction)
    pairs = range(0, h.shape[0], 2)
    smaller = np.mean([projected[i] < projected[i + 1] for i in pairs])
    larger = np.mean([projected[i] > projected[i + 1] for i in pairs])
    if smaller > larger:
        direction *= -1
    return direction


def _fit_direction(train: np.ndarray, method: str) -> np.ndarray:
    """Fit the unsigned direction exactly as repeng/dialz do.

    Parameters
    ----------
    train:
        Training rows.
    method:
        Training method name.

    Returns
    -------
    np.ndarray
        Unsigned direction.
    """

    if method == "mean_diff":
        return np.mean(train, axis=0).astype(np.float32)
    if method == "umap":
        import umap

        embedding = umap.UMAP(n_components=1).fit_transform(train).astype(np.float32)
        return np.sum(train * embedding, axis=0) / np.sum(embedding)
    from sklearn.decomposition import PCA

    pca_model = PCA(n_components=1, whiten=False).fit(train)
    return pca_model.components_.astype(np.float32).squeeze(axis=0)


def _model_type(log: Any, model_type: str | None) -> str:
    """Return the vector's ``model_type``, defaulting to the traced model's.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    model_type:
        Explicit model type, if any.

    Returns
    -------
    str
        Model type string.

    Raises
    ------
    ValueError
        If no explicit type is given and the traced model has no
        ``config.model_type``.
    """

    if model_type is not None:
        return model_type
    try:
        found = getattr(getattr(source_model(log), "config", None), "model_type", None)
    except ValueError:
        found = None
    if not isinstance(found, str):
        raise ValueError(
            "Could not read config.model_type from the traced model; pass model_type=."
        )
    return found


__all__: list[str] = []

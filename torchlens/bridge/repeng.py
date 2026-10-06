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

import numpy as np

from ._utils import source_model
from .steering_vectors import _contrastive_rows


def control_vector(
    log: Any,
    positive_site: Any,
    negative_site: Any | None = None,
    *,
    layer: int,
    negative_log: Any | None = None,
    read_token_index: int | Sequence[int] | None = -1,
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
        Optional TorchLens ``Trace`` of the negative prompts.
    read_token_index:
        Token position read per prompt (default ``-1``, the last token, which is
        what ``ControlVector.train`` reads); a sequence gives one position per
        prompt; ``None`` passes ``[n_prompts, hidden]`` outs unsliced.
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
        If no negative activations are given, the rows do not line up, the
        method is unknown, or no model type can be found.
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
        negative_log=negative_log,
        read_token_index=read_token_index,
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


__all__ = ["control_vector"]

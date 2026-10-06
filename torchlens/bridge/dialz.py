"""dialz bridge helpers.

Builds a real ``dialz.SteeringVector`` from contrastive saved TorchLens
activations. ``SteeringVector.train`` runs a ``SteeringModel`` itself and then
computes the direction in ``dialz.vector.read_representations`` (a fork of
repeng's); this bridge replicates that direction math on the saved last-token
activations instead of re-running the model. Works with dialz 0.2 and 1.x.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from .repeng import _interleave, _model_type, _read_directions
from .steering_vectors import _contrastive_rows


def vector(
    log: Any,
    positive_site: Any,
    negative_site: Any | None = None,
    *,
    layer: int,
    negative_log: Any | None = None,
    read_token_index: int | Sequence[int] | None = -1,
    method: str | None = None,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Build a ``dialz.SteeringVector`` from contrastive saved TorchLens outs.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` of the positive prompts (one prompt per batch row).
    positive_site:
        Site whose saved out holds the positive hidden states. dialz's layer
        ``i`` reads ``hidden_states[i + 1]``: the output of decoder layer ``i``
        (``"model.layers.<i>"``), except for the last layer, where Hugging Face
        returns the final-norm output (``"model.norm"``).
    negative_site:
        Site holding the negative hidden states. Defaults to ``positive_site``
        when ``negative_log`` is given.
    layer:
        Layer index the direction is keyed by in ``SteeringVector.directions``;
        ``SteeringModel.set_control`` applies it to that decoder layer.
    negative_log:
        Optional TorchLens ``Trace`` of the negative prompts.
    read_token_index:
        Token position read per prompt (default ``-1``, the last token, which is
        what ``SteeringVector.train`` reads); a sequence gives one position per
        prompt; ``None`` passes ``[n_prompts, hidden]`` outs unsliced.
    method:
        dialz training method. Defaults to the installed dialz's own default
        (``"pca"`` in dialz 1.x, ``"pca_diff"`` in 0.2; both spellings are
        accepted), or ``"pca_center"``, ``"mean_diff"``, ``"umap"``.
    model_type:
        ``SteeringVector.model_type``. Defaults to the traced model's
        ``config.model_type``.

    Returns
    -------
    dict[str, Any]
        Payload with ``steering_vector`` (a ``dialz.SteeringVector``) and the
        ``positive``/``negative`` rows.

    Raises
    ------
    ImportError
        If dialz is unavailable.
    ValueError
        If no negative activations are given, the rows do not line up, the
        method is unknown, or no model type can be found.
    """

    try:
        import dialz as dialz_module
    except ImportError as exc:
        raise ImportError(
            "dialz bridge requires the `dialz` extra: install torchlens[dialz]."
        ) from exc

    positive, negative = _contrastive_rows(
        log,
        positive_site,
        negative_site,
        negative_log=negative_log,
        read_token_index=read_token_index,
    )
    chosen = method if method is not None else _default_method(dialz_module)
    direction = _read_directions(
        _interleave(positive, negative),
        chosen,
        diff_methods=("pca", "pca_diff"),
        center_in_place=False,
        mean_diff=True,
    )
    result = dialz_module.SteeringVector(
        model_type=_model_type(log, model_type), directions={int(layer): direction}
    )
    return {
        "schema": "torchlens.dialz.v2",
        "steering_vector": result,
        "positive": positive,
        "negative": negative,
    }


def _default_method(module: Any) -> str:
    """Return the installed dialz's default training method name.

    Parameters
    ----------
    module:
        Imported ``dialz`` module.

    Returns
    -------
    str
        ``read_representations``'s ``method`` default (``"pca"`` in 1.x,
        ``"pca_diff"`` in 0.2), or ``"pca"`` when it cannot be read.
    """

    reader = getattr(getattr(module, "vector", None), "read_representations", None)
    if not callable(reader):
        return "pca"
    try:
        default = inspect.signature(reader).parameters["method"].default
    except (TypeError, ValueError, KeyError):
        return "pca"
    return default if isinstance(default, str) else "pca"


__all__ = ["vector"]

"""rsatoolbox bridge helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def dataset(log: Any) -> Any:
    """Convert a TorchLens log into an ``rsatoolbox`` Dataset.

    The LAUNCH bridge uses the final tensor output as the measurement matrix:
    batch items become observations and flattened output units become channels.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` containing saved output activations.

    Returns
    -------
    Any
        ``rsatoolbox.data.Dataset`` with presentation and neuroid descriptors.

    Raises
    ------
    ImportError
        If rsatoolbox is unavailable.
    ValueError
        If the log does not contain a tensor output activation.
    """

    try:
        import rsatoolbox as rsa
    except ImportError as exc:
        raise ImportError(
            "rsatoolbox bridge requires the `neuro` extra: install torchlens[neuro]."
        ) from exc

    activation = _final_output_activation(log)
    measurements = activation.detach().cpu().reshape(activation.shape[0], -1).numpy()
    return rsa.data.Dataset(
        measurements=measurements,
        obs_descriptors={"presentation": np.arange(measurements.shape[0])},
        channel_descriptors={"neuroid": np.arange(measurements.shape[1])},
        descriptors={"source": "torchlens"},
    )


def _final_output_activation(log: Any) -> torch.Tensor:
    """Return the first final output tensor activation.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog``.

    Returns
    -------
    torch.Tensor
        Final output activation.

    Raises
    ------
    ValueError
        If no tensor output activation is available.
    """

    for label in getattr(log, "output_layers", []) or []:
        layer = log[label]
        activation = getattr(layer, "activation", None)
        if isinstance(activation, torch.Tensor):
            return activation
    for layer in reversed(getattr(log, "layer_list", [])):
        activation = getattr(layer, "activation", None)
        if getattr(layer, "is_output_layer", False) and isinstance(activation, torch.Tensor):
            return activation
    raise ValueError("Could not find a tensor output activation for rsatoolbox export.")


__all__ = ["dataset"]

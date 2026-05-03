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
        TorchLens ``Trace`` containing saved output outs.

    Returns
    -------
    Any
        ``rsatoolbox.data.Dataset`` with presentation and neuroid descriptors.

    Raises
    ------
    ImportError
        If rsatoolbox is unavailable.
    ValueError
        If the log does not contain a tensor output out.
    """

    try:
        import rsatoolbox as rsa
    except ImportError as exc:
        raise ImportError(
            "rsatoolbox bridge requires the `neuro` extra: install torchlens[neuro]."
        ) from exc

    out = _final_output_out(log)
    measurements = out.detach().cpu().reshape(out.shape[0], -1).numpy()
    return rsa.data.Dataset(
        measurements=measurements,
        obs_descriptors={"presentation": np.arange(measurements.shape[0])},
        channel_descriptors={"neuroid": np.arange(measurements.shape[1])},
        descriptors={"source": "torchlens"},
    )


def _final_output_out(log: Any) -> torch.Tensor:
    """Return the first final output tensor out.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.

    Returns
    -------
    torch.Tensor
        Final output out.

    Raises
    ------
    ValueError
        If no tensor output out is available.
    """

    for label in getattr(log, "output_layers", []) or []:
        layer = log[label]
        out = getattr(layer, "out", None)
        if isinstance(out, torch.Tensor):
            return out
    for layer in reversed(getattr(log, "layer_list", [])):
        out = getattr(layer, "out", None)
        if getattr(layer, "is_output", False) and isinstance(out, torch.Tensor):
            return out
    raise ValueError("Could not find a tensor output out for rsatoolbox export.")


__all__ = ["dataset"]

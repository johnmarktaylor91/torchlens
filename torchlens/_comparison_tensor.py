"""Host tensor conversion for saved-payload comparisons without artifact imports."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _comparison_tensor(value: Any, codec: Any) -> torch.Tensor | None:
    """Read a backend payload as a host tensor, preserving torch tensors unchanged.

    Parameters
    ----------
    value:
        Saved output or gradient from the producing trace.
    codec:
        Payload codec of that trace's provenance backend. Passed by the caller
        to keep these shared debug helpers independent of artifact I/O imports.

    Returns
    -------
    torch.Tensor | None
        Original torch tensor, an independent CPU copy of a preview array, or
        ``None`` for payloads the codec or torch cannot convert.
    """

    if isinstance(value, torch.Tensor):
        return value
    try:
        encoded = codec.to_numpy(value)
        array = encoded.array
        # TF and Paddle encode bf16 as storage bits, not integer values.
        if str(encoded.logical_dtype) in {"tf.bfloat16", "paddle.bfloat16"}:
            tensor = torch.from_numpy(np.array(array, copy=True, order="C")).view(torch.bfloat16)
        else:
            if str(array.dtype) == "bfloat16":
                array = array.astype(np.float32)
            tensor = torch.from_numpy(np.array(array, copy=True, order="C"))
        shape = (encoded.codec_metadata or {}).get("logical_shape")
        return tensor.reshape(shape) if shape is not None else tensor
    except (TypeError, ValueError):
        return None

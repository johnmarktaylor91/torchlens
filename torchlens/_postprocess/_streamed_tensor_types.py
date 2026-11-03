"""Tensor dtype lookup for streamed-reference finalization."""

import torch

from .._io.format_errors import ArtifactError
from .._utils._torch_symbols import torch_dtype_named


def _dtype_from_manifest_string(dtype_name: str) -> torch.dtype:
    """Resolve a manifest dtype string back into a ``torch.dtype``.

    Parameters
    ----------
    dtype_name:
        Manifest dtype name without the ``torch.`` prefix.

    Returns
    -------
    torch.dtype
        Resolved dtype object.

    Raises
    ------
    ArtifactError
        If the dtype name is unknown to the current runtime.
    """

    dtype_obj = torch_dtype_named(dtype_name)
    if dtype_obj is None:
        raise ArtifactError(f"Unsupported dtype string in manifest: {dtype_name}.")
    return dtype_obj

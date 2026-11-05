"""Wire tensor identity and dtype lookup for streamed-reference finalization."""

from dataclasses import replace
from typing import Any

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
        raise ArtifactError(
            f"Unsupported dtype string in manifest: {dtype_name}.",
            code="record_schema_invalid",
            remedy="re-save with a supported tensor dtype",
        )
    return dtype_obj


def _streamed_entry(writer: Any, blob_id: str) -> tuple[Any, str]:
    """Return the published chunk identity and key of a capture-time stream entry.

    The writer uses numeric ids while capturing. Published format 3 tensor identities
    are chunk-path plus key, exactly as the reader's manifest index derives them.
    """

    entry = writer.get_entry(blob_id)
    reference = writer.get_tensor_reference(blob_id)
    key = reference["key"]
    return replace(entry, blob_id=f"{reference['blob']}#{key}"), key

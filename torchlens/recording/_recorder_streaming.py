"""Rank-local streaming-option helper for the fastlog ``Recorder``."""

from __future__ import annotations

from pathlib import Path

import torch

from .._deprecations import MissingType
from ..options import StorageOptions


def _rank_prefixed_streaming_options(
    streaming: StorageOptions | None | MissingType,
) -> StorageOptions | None | MissingType:
    """Return streaming options with a rank-local directory prefix.

    Parameters
    ----------
    streaming:
        Caller-supplied streaming options.

    Returns
    -------
    StorageOptions | None | MissingType
        Options with ``path`` rewritten to include ``rank_NN`` when a
        bundle path is configured.
    """

    if isinstance(streaming, MissingType) or streaming is None or streaming.path is None:
        return streaming
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    bundle_path = Path(streaming.path)
    # Rewrite ONLY bundle_path: rebuilding from defaults silently dropped the
    # R62 include_* opt-outs (and now the async-write routing) for rank-split
    # recordings.
    return StorageOptions(
        path=bundle_path.parent / f"rank_{rank:02d}" / bundle_path.name,
        retain_in_memory=streaming.retain_in_memory,
        on_save=streaming.on_save,
        include_custom_attributes=streaming.include_custom_attributes,
        include_buffer_values=streaming.include_buffer_values,
        async_writes=streaming.async_writes,
        max_pending_bytes=streaming.max_pending_bytes,
    )

"""MPS headroom query behind the ``HAS_MPS_MEMORY_QUERY`` capability flag."""

from __future__ import annotations

import torch

from . import _torch_compat


def mps_headroom_bytes() -> int | None:
    """Return bytes the MPS allocator may still hand out, or ``None`` when torch cannot say.

    Reads ``torch.mps.recommended_max_memory`` minus ``torch.mps.driver_allocated_memory``;
    both exist only where :data:`~torchlens._utils._torch_compat.HAS_MPS_MEMORY_QUERY` is
    true (torch 2.5+). A query that raises answers ``None`` so the caller falls back to the
    host number rather than a guess.
    """

    if not _torch_compat.HAS_MPS_MEMORY_QUERY:
        return None
    try:
        headroom = int(torch.mps.recommended_max_memory()) - int(
            torch.mps.driver_allocated_memory()
        )
    except Exception:  # noqa: BLE001 - any allocator failure means "unmeasurable", never a crash
        return None
    return max(headroom, 0)

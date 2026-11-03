"""Compatibility re-export; the helper lives in ``torchlens._utils._tl_collective``."""

# layering: re-export kept for record importers; remove after the record-tier fold
from ..._utils._tl_collective import (  # noqa: F401
    _async_collective_elem as _async_collective_elem,
)

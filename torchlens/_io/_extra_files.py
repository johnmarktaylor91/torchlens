"""The shared advisory emission for ignored, unlisted artifact files."""

from __future__ import annotations

import warnings

from ..errors._base import TorchLensWarning


def _warn_extra_files(message: str, *, stacklevel: int) -> None:
    """Reuse the existing advisory site at the caller's original warning depth."""

    warnings.warn(message, TorchLensWarning, stacklevel=stacklevel + 1)

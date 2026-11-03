"""Bundle save: the one late-bound ``_io`` door the bundle package keeps."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Bundle

__all__ = [
    "save_bundle",
]


def save_bundle(
    self: Bundle,
    path: str | Path,
    *,
    level: str = "portable",
    overwrite: bool = False,
) -> None:
    """Save this bundle as a unified ``.tlspec`` directory."""

    from .._io.tlspec import _TlSpecWriter

    _TlSpecWriter.write_bundle(
        bundle=self,
        path=path,
        save_level=level,
        overwrite=overwrite,
    )

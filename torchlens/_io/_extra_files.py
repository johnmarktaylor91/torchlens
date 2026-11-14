"""The shared advisory emission for ignored, unlisted artifact files."""

from __future__ import annotations

import warnings

from ..errors._base import TorchLensWarning
from ._staging import is_superseded_record_name


def is_unnamed_extra_file(relpath: str) -> bool:
    """Return whether an unlisted artifact file at ``relpath`` stays out of the advisory.

    Only the root ``NEWER_SAVE_UNFINISHED.json`` record (and a temp file left while writing
    it) qualifies: it is outside the artifact by design, written by a failed or killed
    overwrite and read by ``tl.load`` before any verification runs.
    """

    return "/" not in relpath and is_superseded_record_name(relpath)


def _warn_extra_files(message: str, *, stacklevel: int) -> None:
    """Reuse the existing advisory site at the caller's original warning depth."""

    warnings.warn(message, TorchLensWarning, stacklevel=stacklevel + 1)

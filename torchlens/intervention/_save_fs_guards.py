"""Filesystem guards shared by intervention spec save and recipe io."""

from __future__ import annotations

from pathlib import Path

from .._io._durability import fsync_dir
from .._io.paths import reject_symlink_path
from .errors import ReplayPreconditionError


def _reject_symlink_path(path: Path, *, context: str) -> None:
    """Reject symlink paths using the intervention replay error contract."""

    reject_symlink_path(
        path,
        context=context,
        exc_type=ReplayPreconditionError,
        message_prefix="Refusing to use symlink",
        trailing_period=False,
    )


def _fsync_directory(path: Path) -> None:
    """Fsync a directory when the platform allows it.

    Failures propagate where directory handles exist; Windows has none and is
    skipped (``fsync_dir(strict=True)``).

    Parameters
    ----------
    path:
        Directory path.
    """

    fsync_dir(path, strict=True)

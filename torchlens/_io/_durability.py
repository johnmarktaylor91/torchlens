"""Crash-durability helpers for atomic artifact publishes.

The temp-write + rename pattern used by the ``.tlspec`` writers survives a
*process* crash, but not a power loss / OS crash: without ``fsync`` the rename
can be journaled before the renamed files' data blocks reach disk, leaving a
"successfully saved" artifact holding zero-length or partial files while the
previous version is already gone. These helpers make the publish durable:
fsync every written file, then the directories that contain them, then the
parent directory that observed the rename.

File fsync failures propagate (a save that cannot be made durable must fail
while its backup/restore machinery is still armed); directory fsyncs are
best-effort because some platforms/filesystems cannot open or fsync a
directory handle (for example Windows), where the rename itself is the best
available guarantee.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["fsync_file", "fsync_dir", "fsync_tree"]


def _fsync_open_flags(os_name: str) -> int:
    """Return the ``os.open`` flags for a descriptor that ``os.fsync`` accepts.

    Parameters
    ----------
    os_name:
        ``os.name`` of the running platform.

    Returns
    -------
    int
        ``O_RDWR`` on Windows, whose ``FlushFileBuffers`` rejects a read-only
        handle with ``EBADF``; ``O_RDONLY`` elsewhere.
    """

    return os.O_RDWR if os_name == "nt" else os.O_RDONLY


def fsync_file(path: Path) -> None:
    """Flush one regular file's data to stable storage.

    Parameters
    ----------
    path:
        File whose contents must be durable. Failures propagate as ``OSError``.
    """

    fd = os.open(path, _fsync_open_flags(os.name))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: Path, *, strict: bool = False) -> None:
    """Flush a directory's entries (new/renamed children) to disk.

    Best-effort by default; ``strict=True`` propagates failures where possible.

    Parameters
    ----------
    path:
        Directory whose entries (created files, completed renames) should be
        durable. A platform that cannot open or fsync directories is skipped.
    strict:
        When True, open/fsync failures propagate on platforms that expose
        directory handles (``os.O_DIRECTORY`` exists); platforms without them
        (Windows) are still skipped, since the rename is all they offer.
    """

    if strict:
        if not hasattr(os, "O_DIRECTORY"):
            return
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        return
    finally:
        os.close(fd)


def fsync_tree(root: Path) -> None:
    """Flush every regular file and directory under ``root``, bottom-up.

    Parameters
    ----------
    root:
        Directory tree about to be renamed into its published location.
        Symlinks are skipped (the writers never create them; a hostile one
        must not open an arbitrary target).
    """

    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        directory = Path(dirpath)
        for filename in filenames:
            file_path = directory / filename
            if file_path.is_symlink() or not file_path.is_file():
                continue
            fsync_file(file_path)
        fsync_dir(directory)

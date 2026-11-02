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

import io
import os
import warnings
from pathlib import Path
from typing import Any

from .._utils.hashing import sha256_file
from ..errors import TorchLensWarning

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


def _stream_bytes_left(stream: Any) -> int | None:
    """Return the bytes after ``stream``'s position, or ``None`` when it cannot tell."""

    try:
        if not stream.seekable():
            return None
        position = stream.tell()
        end = stream.seek(0, io.SEEK_END)
        stream.seek(position)
    except (AttributeError, OSError, ValueError):
        return None
    return int(end - position)


def _restrict_mode(path: Path, mode: int) -> None:
    """Best-effort tighten a streamed bundle path's permissions (POSIX only).

    Shared by the eager and streaming writers: ``mkdir``/``open`` honor the
    ambient umask, so under umask 002 a bundle's directories and metadata
    sidecars were left group-writable (B8-10). Best-effort: a filesystem that
    ignores mode bits is not a save failure.

    Parameters
    ----------
    path:
        Bundle directory or file to tighten.
    mode:
        Target permission bits (``0o700`` for directories, ``0o600`` for files).
    """

    if os.name != "posix":
        return
    try:
        path.chmod(mode)
    except OSError:
        pass


def _restrict_tree_modes(root: Path) -> None:
    """Tighten every directory (0700) and file (0600) under a staged artifact.

    Parameters
    ----------
    root:
        Staged artifact directory.
    """

    for path in root.rglob("*"):
        _restrict_mode(path, 0o700 if path.is_dir() else 0o600)


def _directories_content_equal(left: Path, right: Path) -> bool:
    """Return whether two directory trees hold byte-identical file contents.

    Used by :func:`cleanup_tmp` to decide whether an orphaned ``.bak.*``
    bundle directory is a provable duplicate of the live bundle (safe to
    delete) versus genuinely distinct data that must not be silently
    destroyed. Compares the set of relative file paths and, for each,
    the file's SHA-256 digest via :func:`sha256_file`.

    Parameters
    ----------
    left:
        First directory to compare.
    right:
        Second directory to compare.

    Returns
    -------
    bool
        ``True`` if both directories contain the same relative file paths
        with byte-identical contents, ``False`` otherwise (including on
        any I/O error while comparing, to fail closed toward "not proven
        redundant").
    """

    try:
        left_files = sorted(p.relative_to(left) for p in left.rglob("*") if p.is_file())
        right_files = sorted(p.relative_to(right) for p in right.rglob("*") if p.is_file())
    except OSError:
        return False
    if left_files != right_files:
        return False
    try:
        return all(sha256_file(left / rel) == sha256_file(right / rel) for rel in left_files)
    except OSError:
        return False


def _restore_backup(backup_path: Path, bundle_path: Path, *, warn_on_failure: bool = True) -> bool:
    """Best-effort restore an overwritten bundle after a failed replacement.

    On a DOUBLE fault (the save failed AND this restore also fails -- permissions,
    disk-full, a cross-device backup), the user's previous artifact is gone from its
    canonical ``bundle_path`` but still exists under the ``.bak.<uuid>`` backup name.
    Silently swallowing that left the prior artifact stranded under a hidden name the
    error never mentioned; disclose the backup path so it is recoverable (matching the
    save handler's stated intent). No behavior change on the single-fault path, where
    the restore succeeds.

    Parameters
    ----------
    backup_path:
        Backup path holding the previous bundle contents.
    bundle_path:
        Final bundle path to restore.
    warn_on_failure:
        Emit a warning naming the stranded backup when the restore fails. Callers that
        do their own disclosure (``cleanup_tmp``) pass ``False``.

    Returns
    -------
    bool
        ``True`` when the backup was restored onto ``bundle_path``.
    """

    try:
        backup_path.rename(bundle_path)
    except OSError as exc:
        if warn_on_failure:
            warnings.warn(
                f"Could not restore the previous bundle from its backup after a failed "
                f"save ({exc}). Your prior artifact is NOT lost: it remains at "
                f"{backup_path}. Move it back to {bundle_path} to recover it.",
                TorchLensWarning,
                stacklevel=2,
            )
        return False
    return True

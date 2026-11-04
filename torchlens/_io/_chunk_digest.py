"""Digest of one payload chunk file, computed once per file version.

Shared by the lazy tensor layer and the runnable payload binder; both verify a chunk
against the manifest before reading one key of it, and a chunk holding many tensors is
digested once, not once per tensor.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .._utils.hashing import sha256_file


def _file_identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    """Return the ``(st_dev, st_ino, st_size, st_mtime_ns)`` identity of a file.

    Two stat results with the same tuple describe the same file content at the
    same version: an atomic rename-replace changes ``st_ino`` and an in-place
    rewrite changes ``st_size``/``st_mtime_ns``.

    Parameters
    ----------
    stat_result:
        Stat result to summarize.

    Returns
    -------
    tuple[int, int, int, int]
        Device, inode, size, and nanosecond mtime identity.
    """

    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


@lru_cache(maxsize=64)
def _chunk_digest(path: str, identity: tuple[int, int, int, int]) -> str:
    """Return the sha256 of one payload chunk, computed once per file version.

    Parameters
    ----------
    path:
        Absolute chunk path.
    identity:
        :func:`_file_identity` of the chunk; a rewrite or replace changes it
        and so misses the cache.

    Returns
    -------
    str
        Hex digest of the chunk file.
    """

    return sha256_file(path)


def chunk_sha256(path: Path) -> str:
    """Return the sha256 of one payload chunk, cached per file version.

    Parameters
    ----------
    path:
        Chunk path; its current stat identity keys the cache.

    Returns
    -------
    str
        Hex digest of the chunk file.

    Raises
    ------
    OSError
        If the chunk cannot be stat'ed or read.
    """

    return _chunk_digest(str(path), _file_identity(path.stat()))

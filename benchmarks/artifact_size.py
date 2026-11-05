"""Byte size of a saved benchmark artifact, whether it is a file or a directory.

A ``.tlspec`` artifact is a directory, and ``Path.stat().st_size`` on a directory
reports the directory entry (4096 on ext4), not the bytes written. Benchmark size
metrics go through :func:`artifact_size_bytes` so a file and a bundle are measured
the same way: the regular-file bytes the save produced.
"""

from __future__ import annotations

from pathlib import Path


def artifact_size_bytes(path: Path) -> int:
    """Return the bytes a saved artifact occupies.

    Parameters
    ----------
    path:
        A regular file or a directory tree written by a save.

    Returns
    -------
    int
        ``st_size`` of ``path`` when it is a regular file, else the sum of
        ``st_size`` over every regular file under ``path`` (symlinks excluded,
        empty directory 0).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """

    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise FileNotFoundError(f"artifact path does not exist: {path}")
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file() and not child.is_symlink()
    )

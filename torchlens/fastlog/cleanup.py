"""Cleanup helpers for partial fastlog bundles."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

from .._io import TorchLensIOError
from .._io.streaming import PARTIAL_SENTINEL


def cleanup_partial(path: str | Path, *, force: bool = False) -> list[Path]:
    """Remove partial fastlog bundles and sibling temp directories.

    Parameters
    ----------
    path:
        Fastlog bundle path or final path whose ``.tmp.*`` siblings should be inspected.
    force:
        Whether candidates without ``PARTIAL`` sentinels may be removed.

    Returns
    -------
    list[Path]
        Removed paths.

    Raises
    ------
    TorchLensIOError
        If any cleanup candidate is a symlink.
    """

    bundle_path = Path(path)
    if bundle_path.is_symlink():
        raise TorchLensIOError(f"Refusing to clean symlink path {bundle_path}.")
    removed: list[Path] = []
    candidates = []
    if bundle_path.exists():
        candidates.append(bundle_path)
    candidates.extend(bundle_path.parent.glob(f"{bundle_path.name}.tmp.*"))
    for candidate in candidates:
        if candidate.is_symlink():
            raise TorchLensIOError(f"Refusing to clean symlink temp directory {candidate}.")
        if not candidate.is_dir():
            continue
        if force or (candidate / PARTIAL_SENTINEL).exists():
            shutil.rmtree(candidate)
            removed.append(candidate)
        else:
            warnings.warn(
                f"Leaving non-partial fastlog directory {candidate}; pass force=True to remove it.",
                UserWarning,
                stacklevel=2,
            )
    return removed

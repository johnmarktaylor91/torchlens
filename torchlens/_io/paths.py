"""Path validation helpers for portable TorchLens bundles."""

from __future__ import annotations

from pathlib import Path

from . import TorchLensIOError


def resolve_bundle_blob_path(bundle_root: Path, relative_path: str) -> Path:
    """Resolve one manifest-supplied blob path under ``<bundle>/blobs``.

    Parameters
    ----------
    bundle_root:
        Root directory of the portable bundle.
    relative_path:
        Manifest-provided blob path relative to the bundle root.

    Returns
    -------
    Path
        Absolute resolved blob path.

    Raises
    ------
    TorchLensIOError
        If the path is absolute, contains ``".."``, or resolves outside the
        bundle's ``blobs/`` directory.
    """

    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        raise TorchLensIOError(f"Bundle rejected absolute relative_path {relative_path!r}.")
    if ".." in candidate_path.parts:
        raise TorchLensIOError(
            f"Bundle rejected parent traversal in relative_path {relative_path!r}."
        )

    candidate = (bundle_root / candidate_path).resolve()
    allowed_root = (bundle_root / "blobs").resolve()
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise TorchLensIOError(
            f"Bundle rejected path traversal outside blobs/: {relative_path!r}."
        ) from exc
    return candidate

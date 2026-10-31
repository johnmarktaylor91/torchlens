"""Path validation helpers for portable TorchLens bundles."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from . import ArtifactError

ExceptionType: TypeAlias = type[Exception]


def reject_symlink_path(
    path: Path,
    *,
    context: str,
    exc_type: ExceptionType = ArtifactError,
    message_prefix: str = "Refusing symlinked",
    trailing_period: bool = True,
    code: str | None = None,
) -> None:
    """Reject symlink paths with a caller-chosen exception policy.

    Parameters
    ----------
    path:
        Path to validate.
    context:
        Human-readable path role included in the exception message.
    exc_type:
        Exception type raised when ``path`` is a symlink.
    message_prefix:
        Prefix text used before the contextual path description.
    trailing_period:
        Whether to end the message with ``"."``.
    code:
        Optional stable ``fields["code"]`` for callers that branch on the cause
        (R65). Attached only when ``exc_type`` is a ``TorchLensError`` subclass,
        which accepts structured payload; ignored for foreign exception types.

    Returns
    -------
    None
        Returns when ``path`` is not a symlink.

    Raises
    ------
    Exception
        Raised as ``exc_type`` when ``path`` is a symlink.
    """

    if path.is_symlink():
        suffix = "." if trailing_period else ""
        message = f"{message_prefix} {context}: {path}{suffix}"
        from ..errors._base import TorchLensError

        if code is not None and isinstance(exc_type, type) and issubclass(exc_type, TorchLensError):
            raise exc_type(message, code=code)
        raise exc_type(message)


def resolve_bundle_blobs_dir(bundle_root: Path) -> Path:
    """Validate and resolve the blob containment root for one bundle operation.

    Parameters
    ----------
    bundle_root:
        Root directory of the portable bundle.

    Returns
    -------
    Path
        Canonical path to the bundle's ``blobs/`` directory.

    Raises
    ------
    ArtifactError
        If the ``blobs/`` directory is a symlink.
    """

    blobs_dir = bundle_root / "blobs"
    if blobs_dir.is_symlink():
        raise ArtifactError(f"Refusing symlinked blobs directory: {blobs_dir}.")
    return blobs_dir.resolve()


def resolve_bundle_blob_path(
    bundle_root: Path,
    relative_path: str,
    *,
    resolved_blobs_dir: Path | None = None,
) -> Path:
    """Resolve one manifest-supplied blob path under ``<bundle>/blobs``.

    Parameters
    ----------
    bundle_root:
        Root directory of the portable bundle.
    relative_path:
        Manifest-provided blob path relative to the bundle root.
    resolved_blobs_dir:
        Canonical blob containment root already resolved for this bundle
        operation. When omitted, it is validated and resolved here.

    Returns
    -------
    Path
        Absolute resolved blob path.

    Raises
    ------
    ArtifactError
        If the path is absolute, contains ``".."``, resolves outside the
        bundle's ``blobs/`` directory, or the ``blobs/`` directory itself is a
        symlink.
    """

    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        raise ArtifactError(f"Bundle rejected absolute relative_path {relative_path!r}.")
    if ".." in candidate_path.parts:
        raise ArtifactError(f"Bundle rejected parent traversal in relative_path {relative_path!r}.")

    blobs_dir = bundle_root / "blobs"
    if resolved_blobs_dir is not None and blobs_dir.is_symlink():
        # Keep the per-candidate guard when a caller reuses its canonical root:
        # replacing blobs/ with a symlink during an operation must still fail.
        raise ArtifactError(f"Refusing symlinked blobs directory: {blobs_dir}.")
    try:
        candidate = (bundle_root / candidate_path).resolve()
    except (ValueError, OSError) as exc:
        # A NUL byte (or another OS-unrepresentable component) in a hostile
        # manifest path raised a bare ValueError from ``Path.resolve``; every
        # other hostile shape refuses typed, so this one must too.
        raise ArtifactError(
            f"Bundle rejected unresolvable relative_path {relative_path!r}."
        ) from exc
    allowed_root = (
        resolve_bundle_blobs_dir(bundle_root) if resolved_blobs_dir is None else resolved_blobs_dir
    )
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise ArtifactError(
            f"Bundle rejected path traversal outside blobs/: {relative_path!r}."
        ) from exc
    return candidate

"""Hugging Face Hub publishing helpers."""

from __future__ import annotations

import io
import pickle
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def push_to_hub(
    log_or_bundle_or_spec: Any,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool | None = None,
    path_in_repo: str = "torchlens_artifact.pkl",
    commit_message: str = "Add TorchLens artifact",
    create_repo: bool = True,
    dry_run: bool = False,
    api: Any | None = None,
    save_level: str = "portable",
) -> dict[str, Any]:
    """Upload a real TorchLens artifact to the Hugging Face Hub.

    Parameters
    ----------
    log_or_bundle_or_spec:
        ``Trace``, ``Bundle``, or ``InterventionSpec``-like object to publish.
    repo_id:
        Target Hugging Face repository ID.
    token:
        Optional Hub token.
    private:
        Optional repository privacy flag used when creating the repo.
    path_in_repo:
        Destination filename inside the repository.
    commit_message:
        Commit message for the upload.
    create_repo:
        Whether to create the repo before upload.
    dry_run:
        If True, serialize locally and return planned upload metadata without
        contacting the Hub.
    api:
        Optional ``HfApi``-compatible object for tests or advanced callers.
    save_level:
        Public ``.tlspec`` save level used when the artifact must be
        serialized through the portable-bundle scrub path (``"audit"``,
        ``"executable_with_callables"``, or ``"portable"``). Only consulted
        when ``log_or_bundle_or_spec`` is a ``Trace``/``Bundle`` that cannot
        be pickled directly (see ``_artifact_bytes``).

    Returns
    -------
    dict[str, Any]
        Upload metadata including ``repo_id`` and ``path_in_repo``.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is unavailable.
    TorchLensIOError
        If the artifact cannot be serialized at all. ``push_to_hub`` never
        silently substitutes a metadata-only stub for genuine artifact
        content -- either the real artifact is uploaded, or this raises.
    """

    if api is None and not dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(
                "Hugging Face publishing requires the `hf` extra: install torchlens[hf]."
            ) from exc
        api = HfApi(token=token)

    if dry_run:
        payload = _artifact_bytes(log_or_bundle_or_spec, save_level=save_level)
        return {
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "size_bytes": len(payload),
            "dry_run": True,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / Path(path_in_repo).name
        artifact_path.write_bytes(_artifact_bytes(log_or_bundle_or_spec, save_level=save_level))
        size_bytes = artifact_path.stat().st_size
        if api is None:
            raise RuntimeError("A Hugging Face API object is required when dry_run=False.")
        if create_repo:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        upload_result = api.upload_file(
            path_or_fileobj=str(artifact_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
        )

    return {
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "size_bytes": size_bytes,
        "dry_run": False,
        "upload_result": upload_result,
    }


def _artifact_bytes(log_or_bundle_or_spec: Any, *, save_level: str = "portable") -> bytes:
    """Serialize an artifact for Hub upload.

    Parameters
    ----------
    log_or_bundle_or_spec:
        Artifact object.
    save_level:
        Public ``.tlspec`` save level used for the portable-bundle fallback
        path (see below).

    Returns
    -------
    bytes
        Pickle bytes when the object pickles directly. When direct pickling
        fails (for example a ``Trace``/``Bundle`` retaining live ``grad_fn``
        references, which is the default for any backward-eligible capture),
        this scrubs the artifact through the same real ``.tlspec`` portable
        bundle path used by :func:`torchlens.save`/``Bundle.save`` -- i.e. the
        grad_fn/live-callable scrub is real, not a metadata stand-in -- and
        returns the resulting bundle directory packed as a gzipped tar
        archive. This never silently substitutes a metadata-only stub for
        genuine artifact content.

    Raises
    ------
    TorchLensIOError
        If the artifact cannot be serialized at all (direct pickling fails
        and no portable-bundle path applies, or the portable-bundle path
        itself fails).
    """

    try:
        return pickle.dumps(log_or_bundle_or_spec)
    except Exception as direct_pickle_error:
        from .._io import TorchLensIOError

        saver = _resolve_bundle_saver(log_or_bundle_or_spec)
        if saver is None:
            raise TorchLensIOError(
                "push_to_hub could not serialize this "
                f"{type(log_or_bundle_or_spec).__name__} artifact for upload "
                f"({direct_pickle_error!r}) and it has no portable `.tlspec` bundle "
                "save path to fall back to. Refusing to silently upload a metadata-only "
                "stub instead of the real artifact."
            ) from direct_pickle_error
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                saver(bundle_dir, level=save_level, overwrite=True)
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                    tar.add(bundle_dir, arcname=bundle_dir.name)
                return buffer.getvalue()
        except Exception as bundle_error:
            raise TorchLensIOError(
                "push_to_hub could not serialize this "
                f"{type(log_or_bundle_or_spec).__name__} artifact for upload: direct "
                f"pickling failed ({direct_pickle_error!r}) and the portable `.tlspec` "
                f"bundle path also failed ({bundle_error!r}). Refusing to silently "
                "upload a metadata-only stub instead of the real artifact."
            ) from bundle_error


def _resolve_bundle_saver(log_or_bundle_or_spec: Any) -> Any | None:
    """Return a ``save(path, *, level, overwrite)`` callable for real objects.

    Parameters
    ----------
    log_or_bundle_or_spec:
        Candidate artifact object.

    Returns
    -------
    Any | None
        A bound ``save`` callable matching the ``.tlspec`` bundle contract
        (``Trace``/``Bundle`` both expose one), or ``None`` if the object has
        no known portable-bundle save path.
    """

    from ..data_classes.trace import Trace

    if isinstance(log_or_bundle_or_spec, Trace):
        from .._io.bundle import save as _save_trace_bundle

        def _save_trace(path: Path, *, level: str, overwrite: bool) -> None:
            _save_trace_bundle(log_or_bundle_or_spec, path, level=level, overwrite=overwrite)

        return _save_trace

    from ..bundle import Bundle

    if isinstance(log_or_bundle_or_spec, Bundle):
        return log_or_bundle_or_spec.save

    return None


__all__ = ["push_to_hub"]

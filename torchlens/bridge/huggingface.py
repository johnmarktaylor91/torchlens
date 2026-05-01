"""Hugging Face Hub publishing helpers."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
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
) -> dict[str, Any]:
    """Upload a TorchLens artifact pickle to the Hugging Face Hub.

    Parameters
    ----------
    log_or_bundle_or_spec:
        ``ModelLog``, ``Bundle``, or ``InterventionSpec``-like object to publish.
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

    Returns
    -------
    dict[str, Any]
        Upload metadata including ``repo_id`` and ``path_in_repo``.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is unavailable.
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
        payload = _artifact_bytes(log_or_bundle_or_spec)
        return {
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "size_bytes": len(payload),
            "dry_run": True,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / Path(path_in_repo).name
        artifact_path.write_bytes(_artifact_bytes(log_or_bundle_or_spec))
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


def _artifact_bytes(log_or_bundle_or_spec: Any) -> bytes:
    """Serialize an artifact for Hub upload.

    Parameters
    ----------
    log_or_bundle_or_spec:
        Artifact object.

    Returns
    -------
    bytes
        Pickle bytes when possible, otherwise a JSON metadata manifest.
    """

    try:
        return pickle.dumps(log_or_bundle_or_spec)
    except Exception:
        metadata = {
            "schema": "torchlens.hub_artifact_manifest.v1",
            "kind": type(log_or_bundle_or_spec).__name__,
            "model_name": getattr(log_or_bundle_or_spec, "model_name", None),
            "name": getattr(log_or_bundle_or_spec, "name", None),
            "num_layers": len(getattr(log_or_bundle_or_spec, "layer_list", []) or []),
            "note": "Full pickle serialization failed; this manifest is inspection metadata only.",
        }
        return json.dumps(metadata, indent=2).encode("utf-8")


__all__ = ["push_to_hub"]

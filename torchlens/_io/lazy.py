"""Lazy out reference placeholders for portable TorchLens bundles.

This module defines the lightweight references attached to lazy-loaded
outs and grads. Each ref stores the bundle path, tensor metadata,
and expected checksum so materialization can open a blob file on demand,
verify integrity, return a tensor, and close the file immediately.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import SafetensorError
from safetensors.torch import load, load_file

from . import ArtifactError, PayloadLoadHints
from .manifest import sha256_of_file
from .paths import resolve_bundle_blob_path
from .payload_codec import materialize_transport_tensor

_INLINE_LOAD_MAX_BYTES = 500 * 1024 * 1024
_TORCH_BACKEND_NAME = "torch"


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


@dataclass(frozen=True)
class LazyActivationRef:
    """Reference to one out or grad blob stored in a bundle.

    Parameters
    ----------
    blob_id:
        Opaque zero-padded blob identifier.
    shape:
        Tensor shape recorded at save time.
    dtype:
        Transport tensor dtype recorded at save time.
    device_at_save:
        Original device string recorded at save time.
    source_bundle_path:
        Final bundle directory containing the blob.
    relative_path:
        Bundle-relative path to the persisted safetensors blob.
    kind:
        Logical tensor kind.
    expected_sha256:
        Expected blob checksum from the manifest.
    requires_grad:
        Whether the materialized torch tensor should require gradients.
    logical_backend:
        Backend that produced the logical payload before transport.
    codec:
        Payload codec declared by the manifest entry.
    logical_dtype:
        Backend-native dtype string before transport conversion.
    logical_device:
        Backend-native device string before transport conversion.
    codec_metadata:
        Optional JSON-ready codec metadata.
    payload_hints:
        Optional backend payload hints captured from lazy load.
    """

    blob_id: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device_at_save: str
    source_bundle_path: Path
    relative_path: str
    kind: Literal["out", "grad"]
    expected_sha256: str
    requires_grad: bool = False
    logical_backend: str = "torch"
    codec: str = "torch_safetensors_v1"
    logical_dtype: str | None = None
    logical_device: str | None = None
    codec_metadata: dict[str, Any] | None = None
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None

    def blob_path(self, *, resolved_blobs_dir: Path | None = None) -> Path:
        """Return the absolute path to the referenced blob file.

        Parameters
        ----------
        resolved_blobs_dir:
            Canonical blob containment root already resolved for the current
            bundle operation.

        Returns
        -------
        Path
            Absolute safetensors blob path.
        """

        return resolve_bundle_blob_path(
            self.source_bundle_path,
            self.relative_path,
            resolved_blobs_dir=resolved_blobs_dir,
        )

    def materialize(
        self,
        *,
        map_location: Any = "cpu",
        payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
        resolved_blobs_dir: Path | None = None,
    ) -> Any:
        """Materialize the referenced payload from disk.

        Parameters
        ----------
        map_location:
            Requested target device for the materialized tensor.
        payload_hints:
            Optional backend payload hints. When omitted, hints captured during
            ``torchlens.load(..., lazy=True)`` are used.
        resolved_blobs_dir:
            Canonical blob containment root already resolved for the current
            bundle operation.

        Returns
        -------
        Any
            Materialized payload for the logical backend.

        Raises
        ------
        ArtifactError
            If the referenced blob is missing, corrupt, or checksum-drifted.
        """

        blob_path = self.blob_path(resolved_blobs_dir=resolved_blobs_dir)

        try:
            blob_size = blob_path.stat().st_size
        except FileNotFoundError as exc:
            raise ArtifactError(f"Tensor blob not found at {blob_path}.") from exc
        except OSError as exc:
            raise ArtifactError(f"Failed to access blob at {blob_path}.") from exc

        if blob_size <= _INLINE_LOAD_MAX_BYTES:
            try:
                # Stat-sized bounded read (R33-1, same fix as
                # _json._bounded_read_bytes): reading the CEILING+1 made
                # BufferedReader pre-allocate a ~500 MiB bytes object for
                # every inline blob regardless of its actual size. Read the
                # stat size plus one growth sentinel byte instead, and refuse
                # any growth between stat and read.
                with blob_path.open("rb") as handle:
                    blob_bytes = handle.read(blob_size + 1)
            except FileNotFoundError as exc:
                raise ArtifactError(f"Tensor blob not found at {blob_path}.") from exc
            except OSError as exc:
                raise ArtifactError(f"Failed to materialize blob at {blob_path}.") from exc
            if len(blob_bytes) > blob_size:
                raise ArtifactError(f"blob at {blob_path} grew between its size check and read.")

            observed_sha256 = sha256(blob_bytes).hexdigest()
            if observed_sha256 != self.expected_sha256:
                raise ArtifactError(
                    f"blob at {blob_path} sha256 mismatch; expected {self.expected_sha256} "
                    f"got {observed_sha256}"
                )

            try:
                tensor_map = load(blob_bytes)
            except ImportError as exc:
                raise ArtifactError(
                    "Portable bundle load requires the safetensors backend. Install safetensors>=0.4."
                ) from exc
            except (OSError, SafetensorError, ValueError) as exc:
                raise ArtifactError(f"Failed to materialize blob at {blob_path}.") from exc
        else:
            # Large blobs are stream-hashed and memory-mapped for load to keep
            # peak memory bounded (IO-S9), which unavoidably means two reads of
            # the same path -- unlike the single-read small-blob branch above,
            # whose hash and load observe identical in-memory bytes. Bracket the
            # hash with the file identity and re-check it immediately before the
            # load so a rename-replace or in-place rewrite that occurs between
            # the integrity check and the load is refused rather than silently
            # loading content that was never hashed.
            try:
                pre_hash_identity = _file_identity(blob_path.stat())
            except OSError as exc:
                raise ArtifactError(f"Failed to access blob at {blob_path}.") from exc

            observed_sha256 = sha256_of_file(blob_path)
            if observed_sha256 != self.expected_sha256:
                raise ArtifactError(
                    f"blob at {blob_path} sha256 mismatch; expected {self.expected_sha256} "
                    f"got {observed_sha256}"
                )

            try:
                pre_load_identity = _file_identity(blob_path.stat())
            except OSError as exc:
                raise ArtifactError(f"Failed to access blob at {blob_path}.") from exc
            if pre_load_identity != pre_hash_identity:
                raise ArtifactError(
                    f"blob at {blob_path} changed between integrity check and load."
                )

            try:
                device = str(map_location) if self.logical_backend == _TORCH_BACKEND_NAME else "cpu"
                tensor_map = load_file(blob_path, device=device)
            except ImportError as exc:
                raise ArtifactError(
                    "Portable bundle load requires the safetensors backend. Install safetensors>=0.4."
                ) from exc
            except (OSError, SafetensorError, ValueError) as exc:
                raise ArtifactError(f"Failed to materialize blob at {blob_path}.") from exc

        if len(tensor_map) != 1:
            raise ArtifactError(f"Expected a single tensor in blob file {blob_path}.")
        tensor = next(iter(tensor_map.values()))
        effective_hints = self.payload_hints if payload_hints is None else payload_hints
        payload = materialize_transport_tensor(
            tensor,
            self,
            map_location=map_location,
            payload_hints=effective_hints,
        )
        if self.requires_grad and isinstance(payload, torch.Tensor):
            payload.requires_grad_(True)
        return payload

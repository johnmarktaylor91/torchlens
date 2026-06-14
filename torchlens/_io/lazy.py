"""Lazy out reference placeholders for portable TorchLens bundles.

This module defines the lightweight references attached to lazy-loaded
outs and grads. Each ref stores the bundle path, tensor metadata,
and expected checksum so materialization can open a blob file on demand,
verify integrity, return a tensor, and close the file immediately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import SafetensorError
from safetensors.torch import load, load_file

from . import PayloadLoadHints, TorchLensIOError
from .manifest import sha256_of_file
from .payload_codec import materialize_transport_tensor
from .paths import resolve_bundle_blob_path

_INLINE_LOAD_MAX_BYTES = 500 * 1024 * 1024
_TORCH_BACKEND_NAME = "torch"


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
    logical_backend: str = "torch"
    codec: str = "torch_safetensors_v1"
    logical_dtype: str | None = None
    logical_device: str | None = None
    codec_metadata: dict[str, Any] | None = None
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None

    def blob_path(self) -> Path:
        """Return the absolute path to the referenced blob file.

        Returns
        -------
        Path
            Absolute safetensors blob path.
        """

        return resolve_bundle_blob_path(self.source_bundle_path, self.relative_path)

    def materialize(
        self,
        *,
        map_location: Any = "cpu",
        payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
    ) -> Any:
        """Materialize the referenced payload from disk.

        Parameters
        ----------
        map_location:
            Requested target device for the materialized tensor.
        payload_hints:
            Optional backend payload hints. When omitted, hints captured during
            ``torchlens.load(..., lazy=True)`` are used.

        Returns
        -------
        Any
            Materialized payload for the logical backend.

        Raises
        ------
        TorchLensIOError
            If the referenced blob is missing, corrupt, or checksum-drifted.
        """

        blob_path = self.blob_path()

        try:
            blob_size = blob_path.stat().st_size
        except FileNotFoundError as exc:
            raise TorchLensIOError(f"Tensor blob not found at {blob_path}.") from exc
        except OSError as exc:
            raise TorchLensIOError(f"Failed to access blob at {blob_path}.") from exc

        if blob_size <= _INLINE_LOAD_MAX_BYTES:
            try:
                blob_bytes = blob_path.read_bytes()
            except FileNotFoundError as exc:
                raise TorchLensIOError(f"Tensor blob not found at {blob_path}.") from exc
            except OSError as exc:
                raise TorchLensIOError(f"Failed to materialize blob at {blob_path}.") from exc

            observed_sha256 = sha256(blob_bytes).hexdigest()
            if observed_sha256 != self.expected_sha256:
                raise TorchLensIOError(
                    f"blob at {blob_path} sha256 mismatch; expected {self.expected_sha256} "
                    f"got {observed_sha256}"
                )

            try:
                tensor_map = load(blob_bytes)
            except ImportError as exc:
                raise TorchLensIOError(
                    "Portable bundle load requires the safetensors backend. Install safetensors>=0.4."
                ) from exc
            except (OSError, SafetensorError, ValueError) as exc:
                raise TorchLensIOError(f"Failed to materialize blob at {blob_path}.") from exc
        else:
            observed_sha256 = sha256_of_file(blob_path)
            if observed_sha256 != self.expected_sha256:
                raise TorchLensIOError(
                    f"blob at {blob_path} sha256 mismatch; expected {self.expected_sha256} "
                    f"got {observed_sha256}"
                )

            try:
                device = str(map_location) if self.logical_backend == _TORCH_BACKEND_NAME else "cpu"
                tensor_map = load_file(blob_path, device=device)
            except ImportError as exc:
                raise TorchLensIOError(
                    "Portable bundle load requires the safetensors backend. Install safetensors>=0.4."
                ) from exc
            except (OSError, SafetensorError, ValueError) as exc:
                raise TorchLensIOError(f"Failed to materialize blob at {blob_path}.") from exc

        if len(tensor_map) != 1:
            raise TorchLensIOError(f"Expected a single tensor in blob file {blob_path}.")
        tensor = next(iter(tensor_map.values()))
        effective_hints = self.payload_hints if payload_hints is None else payload_hints
        return materialize_transport_tensor(
            tensor,
            self,
            map_location=map_location,
            payload_hints=effective_hints,
        )

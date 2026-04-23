"""Lazy activation reference placeholders for portable TorchLens bundles.

This module defines the lightweight references attached to lazy-loaded
activations and gradients. Each ref stores the bundle path, tensor metadata,
and expected checksum so materialization can open a blob file on demand,
verify integrity, return a tensor, and close the file immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file

from . import TorchLensIOError
from .manifest import sha256_of_file


@dataclass(frozen=True)
class LazyActivationRef:
    """Reference to one activation or gradient blob stored in a bundle.

    Parameters
    ----------
    blob_id:
        Opaque zero-padded blob identifier.
    shape:
        Tensor shape recorded at save time.
    dtype:
        Tensor dtype recorded at save time.
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
    """

    blob_id: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device_at_save: str
    source_bundle_path: Path
    relative_path: str
    kind: Literal["activation", "gradient"]
    expected_sha256: str

    def blob_path(self) -> Path:
        """Return the absolute path to the referenced blob file.

        Returns
        -------
        Path
            Absolute safetensors blob path.
        """

        return self.source_bundle_path / self.relative_path

    def materialize(self, *, map_location: str | torch.device = "cpu") -> torch.Tensor:
        """Materialize the referenced tensor from disk.

        Parameters
        ----------
        map_location:
            Requested target device for the materialized tensor.

        Returns
        -------
        torch.Tensor
            Materialized tensor payload.

        Raises
        ------
        TorchLensIOError
            If the referenced blob is missing, corrupt, or checksum-drifted.
        """

        blob_path = self.blob_path()
        if not blob_path.exists():
            raise TorchLensIOError(f"Tensor blob not found at {blob_path}.")

        observed_sha256 = sha256_of_file(blob_path)
        if observed_sha256 != self.expected_sha256:
            raise TorchLensIOError(
                f"blob at {blob_path} sha256 mismatch; expected {self.expected_sha256} "
                f"got {observed_sha256}"
            )

        try:
            tensor_map = load_file(blob_path, device=str(map_location))
        except ImportError as exc:
            raise TorchLensIOError(
                "Portable bundle load requires the safetensors backend. Install safetensors>=0.4."
            ) from exc
        except (OSError, SafetensorError, ValueError) as exc:
            raise TorchLensIOError(f"Failed to materialize blob at {blob_path}.") from exc

        if len(tensor_map) != 1:
            raise TorchLensIOError(f"Expected a single tensor in blob file {blob_path}.")
        return next(iter(tensor_map.values()))

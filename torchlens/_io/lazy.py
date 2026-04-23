"""Lazy activation reference placeholders for portable TorchLens bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


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
    kind: Literal["activation", "gradient"]
    expected_sha256: str

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
        NotImplementedError
            S6 will wire lazy materialization.
        """

        raise NotImplementedError("S6 will wire this")

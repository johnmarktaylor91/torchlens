"""Pure naming helpers mapping manifest blob kinds to lazy-reference fields."""

from __future__ import annotations

from typing import Literal

import torch


def _lazy_ref_field_name(field_name: str) -> str | None:
    """Return the corresponding lazy-ref field for a direct blob field.

    Parameters
    ----------
    field_name:
        Direct blob field name on the owning data class.

    Returns
    -------
    str | None
        Matching lazy-ref field name, or ``None`` when not applicable.
    """

    if field_name == "out":
        return "out_ref"
    if field_name == "grad":
        return "grad_ref"
    return None


def _lazy_ref_kind(blob_kind: str) -> Literal["out", "grad"]:
    """Normalize a blob kind into a lazy direct-tensor kind.

    Parameters
    ----------
    blob_kind:
        Blob kind stored in the portable manifest.

    Returns
    -------
    str
        Direct tensor kind expected by ``LazyActivationRef``.
    """

    if blob_kind == "grad":
        return "grad"
    return "out"


def _normalize_map_location(map_location: str | torch.device) -> str:
    """Normalize ``map_location`` to the string form expected by safetensors."""

    return str(map_location)

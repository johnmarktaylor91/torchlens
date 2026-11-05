"""Variation-marker and mirrored-field lookups for rolled multi-pass Layers."""

from __future__ import annotations

from typing import Any

from .._data_classes._layer_spec import _LAYER_MIRROR_SPEC


def _varying_marker(node: Any) -> dict[str, Any]:
    """Return the reconciler's variation marker for a rolled Layer."""

    annotations = getattr(node, "annotations", None)
    if isinstance(annotations, dict):
        marker = annotations.get("varying_across_passes")
        if isinstance(marker, dict):
            return marker
    return {}


def _mirror_backed_op_field(field_name: str) -> str | None:
    """Return the Op field backing a mirrored Layer name, if any."""

    entry = _LAYER_MIRROR_SPEC.get(field_name)
    return entry[0] if entry is not None else None

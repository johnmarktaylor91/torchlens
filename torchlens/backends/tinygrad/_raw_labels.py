"""Resolve materialized tinygrad op labels back to their raw capture labels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _final_to_raw_labels(ops_by_raw_label: Mapping[str, Any]) -> dict[str, str]:
    """Map each materialized op's final labels to its raw capture label.

    The pass-qualified ``label`` always maps; the bare ``layer_label`` maps
    only for a single-pass op, where it is unambiguous.

    Parameters
    ----------
    ops_by_raw_label
        Materialized operations keyed by raw label.

    Returns
    -------
    dict[str, str]
        Final label to raw label.
    """

    final_to_raw: dict[str, str] = {}
    for known_op in ops_by_raw_label.values():
        raw_label = getattr(known_op, "_label_raw", None)
        if not isinstance(raw_label, str):
            continue
        label = getattr(known_op, "label", None)
        if isinstance(label, str):
            final_to_raw[label] = raw_label
        layer_label = getattr(known_op, "layer_label", None)
        if isinstance(layer_label, str) and int(getattr(known_op, "num_passes", 1)) == 1:
            final_to_raw[layer_label] = raw_label
    return final_to_raw

"""Metadata-only join keys for the summary's executed operation rows."""

from __future__ import annotations

from typing import Any


def _op_metadata(trace: Any) -> dict[tuple[str, int], tuple[int, tuple[str, ...], Any, int, Any]]:
    """Index each stored operation by its canonical label and occurrence."""

    entries: dict[tuple[str, int], tuple[int, tuple[str, ...], Any, int, Any]] = {}
    seen: dict[str, int] = {}
    for index, op in enumerate(trace._layer_list or ()):
        label = str(getattr(op, "layer_label", "?"))
        occurrence = seen.get(label, 0)
        seen[label] = occurrence + 1
        shape = getattr(op, "shape", None)
        entries[(label, occurrence)] = (
            index,
            tuple(getattr(op, "module_call_stack", ()) or ()),
            None if shape is None else tuple(int(dim) for dim in shape),
            int(getattr(op, "pass_number", occurrence + 1) or (occurrence + 1)),
            op,
        )
    return entries

"""Rolled conditional edge-pass indices rebuilt after Step 15.5."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ._column_passes_finalize import _strip_pass_suffix

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _rebuild_conditional_edge_call_indices(self: Trace) -> None:
    """Recompute rolled conditional edge-pass metadata from arm-entry edges.

    Parameters
    ----------
    self:
        Model log being finalized.
    """
    if self._graph_conditional_edge_call_indices:
        self._graph_conditional_edge_call_indices = {
            edge_key: sorted(dict.fromkeys(call_indexs))
            for edge_key, call_indexs in self._graph_conditional_edge_call_indices.items()
        }
        return

    conditional_edge_call_indices: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)
    for (conditional_id, branch_kind), edge_list in self._graph_conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            parent_no_pass = _strip_pass_suffix(parent_label)
            child_no_pass = _strip_pass_suffix(child_label)
            child_call_index = _get_label_call_index(child_label)
            edge_key = (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
            )
            if child_call_index not in conditional_edge_call_indices[edge_key]:
                conditional_edge_call_indices[edge_key].append(child_call_index)

    self._graph_conditional_edge_call_indices = {
        edge_key: sorted(call_indexs)
        for edge_key, call_indexs in conditional_edge_call_indices.items()
    }


def _get_label_call_index(layer_label: str) -> int:
    """Extract the pass number encoded in a layer label.

    Parameters
    ----------
    layer_label:
        Layer label that may include a ``:call_index`` suffix.

    Returns
    -------
    int
        Parsed pass number, or ``1`` for pass-free labels.
    """
    label_parts = layer_label.rsplit(":", 1)
    if len(label_parts) == 2 and label_parts[1].isdigit():
        return int(label_parts[1])
    return 1

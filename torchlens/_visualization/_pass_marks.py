"""Pass marks: how a drawn node or edge says which passes of a recurrent layer it shows.

The default node title carries them (``linear_1_1:2`` on an unrolled pass,
``linear_1_1 (x3)`` or ``linear_1_1:1-2,4`` on a rolled layer), and so do the
``"label"`` and ``"pass"`` tokens of ``node_label_fields``, so a picker that drops
the wide parameter row keeps the loop marks. Rolled edges carry ``In``/``Out``
pass ranges; on a conditional edge they share the branch label's midpoint slot,
and beside an argument label they share its midpoint slot, which Graphviz reserves
layout space for, instead of floating at the endpoint over arrows and other labels.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .._utils._multipass_access import get_multipass_attr

__all__ = (
    "marked_layer_label",
    "pass_label_fits",
    "pass_mark_suffix",
    "pass_rows",
    "stack_labels",
    "with_branch",
)

#: The shape ``_render_leaf._format_branch_edge_label_html`` emits (IF, THEN, ELIF n, ...).
_BRANCH_LABEL = re.compile(r'<<FONT POINT-SIZE="[^"]*"><b><u>.*</u></b></FONT>>', re.DOTALL)


def _num_passes(layer_log: Any) -> int:
    """Return the node's pass count (1 for nodes that do not record one)."""

    return int(getattr(layer_log, "num_passes", 1) or 1)


def pass_mark_suffix(
    layer_log: Any, node_address: str, vis_mode: str, *, module_row: bool = True
) -> str:
    """Return the pass mark the default node title appends to the node's name.

    ``":2"`` on an unrolled pass of a multi-pass layer; on a rolled multi-pass
    layer the rolling call-group suffix (``":1-2,4"``) or the ``" (x3)"`` count;
    ``""`` on single-pass nodes. A rolled atomic module whose ``@module`` row
    carries the call range keeps its title clean while that row is shown
    (``module_row``).
    """

    # Lazy: both renderer modules import this one at module load.
    from ._render_leaf import _rolled_visual_num_passes
    from ._render_nodes import _format_rolling_suffix, _rolling_annotation

    if _num_passes(layer_log) <= 1:
        return ""
    if vis_mode == "unrolled":
        return f":{layer_log.pass_number}"
    if vis_mode != "rolled":
        return ""
    if module_row and getattr(layer_log, "is_atomic_module", False) and ":" in node_address:
        return ""
    rolling_suffix = _format_rolling_suffix(_rolling_annotation(layer_log, vis_mode))
    return rolling_suffix or f" (x{_rolled_visual_num_passes(layer_log)})"


def marked_layer_label(
    layer_log: Any, node_address: str, vis_mode: str, fields: Sequence[str]
) -> str:
    """Return the ``"label"`` token row: the layer label with the default's pass mark."""

    suffix = pass_mark_suffix(layer_log, node_address, vis_mode, module_row="module" in fields)
    return f"{getattr(layer_log, 'layer_label', '')}{suffix}"


def pass_rows(layer_log: Any, vis_mode: str) -> list[str]:
    """Return the ``"pass"`` token rows: none on a single-pass node.

    An unrolled node shows its real 1-based pass (``pass_number``); a rolled
    node shows how many passes it stands for.
    """

    if _num_passes(layer_log) <= 1:
        return []
    if vis_mode == "unrolled":
        return [str(get_multipass_attr(layer_log, "pass_number", 1, multipass=1))]
    return [str(_num_passes(layer_log))]


def pass_label_fits(edge_dict: Mapping[str, Any]) -> bool:
    """Return whether an edge's midpoint slot can take its ``In``/``Out`` pass label."""

    label = edge_dict.get("label")
    return label is None or bool(_BRANCH_LABEL.fullmatch(str(label)))


def with_branch(edge_dict: Mapping[str, Any], pass_label: str) -> str:
    """Return ``pass_label``, stacked under the edge's branch label when it has one.

    Both are Graphviz HTML labels (``<...>`` wrapped once more in ``<>``); the
    result is one borderless two-row table, so the branch word and the pass
    range share the reserved midpoint space.
    """

    branch = edge_dict.get("label")
    if branch is None or not _BRANCH_LABEL.fullmatch(str(branch)):
        return pass_label
    return stack_labels(str(branch), pass_label)


def stack_labels(top: str, bottom: str) -> str:
    """Return two Graphviz HTML labels as one borderless two-row label.

    Each argument is an HTML label (``<...>`` wrapped once more in ``<>``); the
    result keeps one HTML root, so it is a valid single ``label``.
    """

    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f"<TR><TD>{top[1:-1]}</TD></TR><TR><TD>{bottom[1:-1]}</TD></TR></TABLE>>"
    )

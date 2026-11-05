"""Pass marks: how a drawn node or edge says which passes of a recurrent layer it shows.

The default node title carries them (``linear_1_1:2`` on an unrolled pass,
``linear_1_1 (x3)`` or ``linear_1_1:1-2,4`` on a rolled layer), and so do the
``"label"`` and ``"pass"`` tokens of ``node_label_fields``, so a picker that drops
the wide parameter row keeps the loop marks. Rolled edges carry ``In``/``Out``
pass ranges as head and tail labels, each beside the node it describes
(``_render_edges._label_rolled_call_indexs``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .._utils._multipass_access import get_multipass_attr

__all__ = (
    "pass_mark_suffix",
    "pass_rows",
)

#: The shape ``_render_leaf._format_branch_edge_label_html`` emits (IF, THEN, ELIF n, ...).
_BRANCH_LABEL = re.compile(
    r'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0"><TR>'
    r'<TD WIDTH="\d+"></TD><TD><FONT POINT-SIZE="[^"]*"><b><u>.*</u></b></FONT></TD>'
    r'<TD WIDTH="\d+"></TD></TR></TABLE>>',
    re.DOTALL,
)


def _num_passes(layer_log: Any) -> int:
    """Return the node's pass count (1 for nodes that do not record one)."""

    return int(getattr(layer_log, "num_passes", 1) or 1)


def pass_mark_suffix(
    layer_log: Any,
    node_address: str,
    vis_mode: str,
    rolled_mark: Callable[[Any, str], str],
    *,
    module_row: bool = True,
) -> str:
    """Return the pass mark the default node title appends to the node's name.

    ``":2"`` on an unrolled pass of a multi-pass layer; on a rolled multi-pass
    layer ``rolled_mark(layer_log, vis_mode)`` (the renderer's call-group suffix
    ``":1-2,4"`` or ``" (x3)"`` count); ``""`` on single-pass nodes. A rolled
    atomic module whose ``@module`` row carries the call range keeps its title
    clean while that row is shown (``module_row``).
    """

    if _num_passes(layer_log) <= 1:
        return ""
    if vis_mode == "unrolled":
        return f":{layer_log.pass_number}"
    if vis_mode != "rolled":
        return ""
    if module_row and getattr(layer_log, "is_atomic_module", False) and ":" in node_address:
        return ""
    return rolled_mark(layer_log, vis_mode)


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

"""Policy-independent site keys for preview-backend ops, minted before grouping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._data_classes._site_key import SiteKeyMinter
from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _mint_preview_site_keys(trace: Trace) -> None:
    """Mint the policy-independent ``site_key_v1`` on every retained preview op.

    Runs before recurrence assignments are computed (site keys are structural
    facts from raw records, identical whether grouping runs or not -- P4), so
    the preview node builder copies the minted key into its
    ``RecurrenceNode`` and the ungrouped path carries keys all the same.
    Orphan ops consume no ordinals and keep ``site_key=None``.

    Parameters
    ----------
    trace:
        Trace whose ``_raw_graph_ws.raw_layer_dict`` holds materialized
        preview ops in execution order.
    """

    minter = SiteKeyMinter()
    for op_log in trace._raw_graph_ws.raw_layer_dict.values():
        if record_member(op_log, "is_orphan", False):
            continue
        op_log._graph_site_key = minter.mint(
            getattr(op_log, "modules", None) or (),
            str(getattr(op_log, "type", "") or ""),
            (
                getattr(op_log, "multi_output_index", None)
                if getattr(op_log, "in_multi_output", False)
                else None
            ),
        )

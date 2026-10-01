"""Backward-view style inventory and pass-row helpers (vizmech item 17).

Split from ``_render_leaf.py`` (the R43 file-size ratchet): the per-render
inventory of painted backward styles feeds the in-frame key and the
uniform-constant-row suppression; the pass-row helper spells the ``bwd N``
label line under that suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.grad_fn import GradFn
    from ..data_classes.trace import Trace
    from ._render_common import BackwardPassFilter

__all__ = [
    "BackwardStyleInventory",
    "_backward_pass_row",
    "compute_backward_style_inventory",
]


@dataclass(frozen=True)
class BackwardStyleInventory:
    """Per-render inventory of the backward styles ACTUALLY painted.

    Computed ONCE over the visible grad_fns (vizmech item 17): it feeds the
    in-frame backward key (a key row for an unused style would itself be an
    unexplained claim) and the uniform-row suppression (a row carrying the
    same degenerate value on every node -- ``grad N/A`` x36 on the memo's
    WGAN-GP render -- is ink, not information).
    """

    has_higher_order: bool
    has_intervening: bool
    has_accumulation: bool
    has_custom: bool
    num_backward_passes: int
    suppress_grad_row: bool
    suppress_order_row: bool
    suppress_bwd_row: bool


def compute_backward_style_inventory(
    trace: Trace,
    pass_filter: BackwardPassFilter = None,
) -> BackwardStyleInventory:
    """Build the style inventory over the pass-filtered visible grad_fns.

    Parameters
    ----------
    trace:
        Trace with a captured backward graph.
    pass_filter:
        Normalized backward-pass filter (visibility must match the render).
    """

    # Call-time import: the filter and grad-shape helpers live in
    # ``_render_leaf``, which imports THIS module at load time.
    from ._render_leaf import _format_backward_output_shape, _grad_fn_matches_backward_filter

    visible = [
        grad_fn_handle
        for grad_fn_handle in trace.grad_fns
        if _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter)
    ]
    orders = [getattr(grad_fn_handle, "order", None) for grad_fn_handle in visible]
    has_higher_order = any(order is not None and order > 1 for order in orders)
    grad_shapes = [_format_backward_output_shape(grad_fn_handle) for grad_fn_handle in visible]
    num_passes = int(getattr(trace, "num_backward_passes", 1) or 1)
    return BackwardStyleInventory(
        has_higher_order=has_higher_order,
        has_intervening=any(not grad_fn_handle.has_op for grad_fn_handle in visible),
        has_accumulation=any(grad_fn_handle.type == "accumulategrad" for grad_fn_handle in visible),
        has_custom=any(grad_fn_handle.is_custom for grad_fn_handle in visible),
        num_backward_passes=num_passes,
        # "N/A" on EVERY node is a uniform constant row (pure ink); a mixed
        # render keeps N/A rows -- there they are informative.
        suppress_grad_row=bool(visible) and all(shape == "N/A" for shape in grad_shapes),
        suppress_order_row=not has_higher_order,
        suppress_bwd_row=num_passes <= 1,
    )


def _backward_pass_row(
    grad_fn_handle: GradFn,
    call: Any | None,
    pass_filter: BackwardPassFilter,
    inventory: BackwardStyleInventory | None,
) -> list[str]:
    """Return the ``bwd N`` label row (empty when suppressed or unknown).

    Suppressed as a uniform constant row on single-pass captures (vizmech
    item 17); otherwise rolled nodes list their passes compactly and
    unrolled calls carry their one pass index.
    """

    from ..utils.display import int_list_to_compact_str

    if inventory is not None and inventory.suppress_bwd_row:
        return []
    if call is None:
        pass_indices = sorted(
            {
                int(pass_index)
                for pass_index in (
                    getattr(grad_fn_call, "backward_pass_index", None)
                    for grad_fn_call in grad_fn_handle.calls.values()
                )
                if pass_index is not None
            }
        )
        if pass_filter is not None:
            pass_indices = [pass_index for pass_index in pass_indices if pass_index in pass_filter]
        if pass_indices:
            return [f"bwd {int_list_to_compact_str(pass_indices)}"]
        return []
    pass_index = getattr(call, "backward_pass_index", None)
    if pass_index is not None:
        return [f"bwd {pass_index}"]
    return []

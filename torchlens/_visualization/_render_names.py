"""Render names shared by the renderer and the lens and surgery layers above it.

A leaf: the lens stages and the surgery mark family read the rendered-node
slot, the bare-draw stem and the optimizer ceiling, and import them from
here so they never import the render modules (which reach the records).
``node_spec``, ``_render_utils`` and ``collapse_optimizer`` bind the same
objects, so their spellings are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .._utils._render_slots import _SPEC_SLOT_BOX_OPS, _SPEC_SLOT_RENDERED_NODE

__all__ = (
    "COLLAPSE_OPTIMIZER_MAX_OPS",
    "DEFAULT_DRAW_STEM",
    "_SPEC_SLOT_BOX_OPS",
    "_SPEC_SLOT_RENDERED_NODE",
    "call_collapsed_spec_fn",
)

# The two node-spec slots are defined in ``_utils._render_slots`` so repgeom,
# below this layer, reads the same objects without importing the renderer.


def call_collapsed_spec_fn(
    fn: Callable[[Any, Any], Any], module: Any, spec: Any, op_labels: Iterable[str]
) -> Any:
    """Call a collapsed-box spec fn with the box's op labels in ``_SPEC_SLOT_BOX_OPS``."""

    token = _SPEC_SLOT_BOX_OPS.set(tuple(op_labels))
    try:
        return fn(module, spec)
    finally:
        _SPEC_SLOT_BOX_OPS.reset(token)


#: Stem a bare ``Trace.draw()`` (no ``vis_outpath``) writes under.
DEFAULT_DRAW_STEM = "modelgraph"

#: Preflight compute ceiling for the v2 collapse selection. The
#: frontier DP is measured superlinear (~n^1.75) in rendered op count with no
#: internal time budget, so above this many ops the optimizer DECLINES with a
#: disclosed warning instead of silently burning CPU-hours: ``draw`` renders
#: uncollapsed, ``Drawing.collapse_plan`` refuses typed
#: (``collapse_plan_unavailable``), and the schedule degrades to its single
#: full-graph step. Calibration: the largest shipped-suite models
#: (densenet201 at 1,517 ops, maxvit_t at 1,307) must stay admitted -- their
#: schedules are the suite's slow cells at tens of seconds -- while the
#: extrapolated multi-thousand-op cost (an hour-class compute at ~5,000 ops)
#: is exactly what the ceiling exists to refuse.
COLLAPSE_OPTIMIZER_MAX_OPS = 2000

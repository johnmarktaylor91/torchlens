"""Module frame border widths, shared by the Trace and bundle renderers.

Outermost modules get the thickest border; deeper modules thin out by depth
fraction so the nesting reads at a glance. This module imports nothing from
the renderer, so a helper that only needs the widths stays out of the
renderer's import cycle.
"""

from __future__ import annotations

MAX_MODULE_PENWIDTH = 5
MIN_MODULE_PENWIDTH = 2
PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH


def compute_module_penwidth(call_depth: int, max_call_depth: int) -> float:
    """Return the cluster border width for a module at ``call_depth``.

    ``call_depth`` is 0-based (outermost is depth 0).  Outermost modules
    get the maximum penwidth; deepest modules get the minimum.  When the
    overall hierarchy has only one level (``max_call_depth == 0`` or
    ``1``) we still return a sensible value so callers don't have to
    special-case shallow models.
    """

    if max_call_depth <= 0:
        return float(MIN_MODULE_PENWIDTH + PENWIDTH_RANGE)
    nesting_fraction = (max_call_depth - call_depth) / max_call_depth
    nesting_fraction = max(0.0, min(1.0, nesting_fraction))
    return MIN_MODULE_PENWIDTH + nesting_fraction * PENWIDTH_RANGE

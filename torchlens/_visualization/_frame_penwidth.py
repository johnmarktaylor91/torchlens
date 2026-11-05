"""Module frame border width, shared by the Trace and bundle renderers.

Every module frame is drawn with the same stroke, whatever its nesting depth:
a frame then looks the same in every picture of the model (a focused draw of
one submodule, a collapsed overview, the whole graph), and nesting reads from
the boxes themselves and their titles. The stroke stays a notch heavier than a
node outline so a frame reads as a container and survives downscaling. This
module imports nothing from the renderer, so a helper that only needs the
width stays out of the renderer's import cycle.
"""

from __future__ import annotations

#: Stroke of every module frame, in points (a node outline is 1 pt).
MODULE_PENWIDTH = 1.5
#: Kept for callers that quote the range: both ends are the one stroke.
MAX_MODULE_PENWIDTH = MODULE_PENWIDTH
MIN_MODULE_PENWIDTH = MODULE_PENWIDTH
PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH


def compute_module_penwidth(call_depth: int, max_call_depth: int) -> float:
    """Return the cluster border width for a module at ``call_depth``.

    The width no longer depends on depth; the arguments stay so every caller
    that knows a module's depth keeps working unchanged.
    """

    del call_depth, max_call_depth
    return MODULE_PENWIDTH

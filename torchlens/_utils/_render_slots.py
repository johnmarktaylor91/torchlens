"""Context slots a node-spec callback reads while the renderer calls it.

A leaf below every layer that reads them: the renderer sets the slots around
a callback, and the lens stages, the surgery marks and the repgeom node
visuals read them. ``_visualization._render_names`` binds the same objects,
so its spellings are unchanged.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

__all__ = ("_SPEC_SLOT_BOX_OPS", "_SPEC_SLOT_RENDERED_NODE")

#: The rendered node (per-pass Op on unrolled nodes, else the Layer) while a
#: node-spec callback runs; the callback itself receives the aggregate Layer.
#: Defined in a leaf, not in ``_render_nodes``, so building the intervention
#: callback never imports the render modules: they bind ``node_spec``'s
#: helpers by value at import time, and a first import while a helper is
#: patched would keep the patch.
_SPEC_SLOT_RENDERED_NODE: ContextVar[Any] = ContextVar("spec_slot_rendered_node", default=None)
#: The pass-qualified op labels a collapsed box stands for, set around the
#: collapsed-box callback so TorchLens-internal stages can read the box scope.
_SPEC_SLOT_BOX_OPS: ContextVar[tuple[str, ...] | None] = ContextVar(
    "spec_slot_box_ops", default=None
)

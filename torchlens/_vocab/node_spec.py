"""NodeSpec value vocabulary (ratified V2 relocation; C01 item 6).

The node-presentation VALUE vocabulary consumed by strata below PRESENT:
``options`` at the engine boundary and the viz/repgeom/receptive-field
appliances all type against :class:`NodeSpec` and the node-callback aliases.
Only vocabulary lives here (Rule V3: no behavior smuggling) -- the render
behavior helpers stay in :mod:`torchlens.visualization.node_spec`, which
re-exports these names so no public spelling changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.grad_fn import GradFn
    from ..data_classes.layer import Layer
    from ..data_classes.module import Module

__tl_layer__ = "L0"
__tl_vocabulary__ = True

INTERVENTION_SITE_COLOR = "#FF00FF"
INTERVENTION_CONE_COLOR = "#FFB3FF"
INTERVENTION_HOOK_FILL_COLOR = "#FFE6FF"
INTERVENTION_HOOK_BORDER_COLOR = "#CC00CC"
INTERVENTION_OVERRIDE_KEYS = frozenset(
    {
        "intervention_site_color",
        "intervention_cone_color",
        "intervention_site_penwidth",
        "intervention_cone_penwidth",
        "intervention_hook_fillcolor",
        "intervention_hook_color",
        "intervention_hook_penwidth",
    }
)


@dataclass
class NodeSpec:
    """Graphviz node attributes produced by TorchLens before user customization.

    The dataclass is intentionally mutable because visualization callbacks are
    user ergonomics APIs: mutating and returning the supplied default spec is a
    natural pattern for small display tweaks.

    Attributes
    ----------
    lines:
        Plain-text rows to render in the node label.
    shape:
        Graphviz node shape.
    fillcolor:
        Optional fill color.
    fontcolor:
        Optional font color.
    style:
        Graphviz node style.
    color:
        Optional border color.
    penwidth:
        Optional border width.
    tooltip:
        Optional node tooltip.
    image:
        Optional image path to embed in the node.
    width:
        Optional node width minimum in inches (size encoding channel). With
        ``fixedsize="false"`` the box can only GROW from the label's natural
        size, so a label can never be truncated by an encoding. Dropped by
        the spec funnel when ``image`` is set: an image node's size is
        pixel-derived, and a width minimum would become a live scaling
        floor (``extra_attrs`` remains the power-valve override).
    height:
        Optional node height minimum in inches (see ``width``).
    fixedsize:
        Optional Graphviz ``fixedsize`` value emitted with the size fields.
    extra_attrs:
        Additional Graphviz node attributes.
    """

    lines: list[str]
    shape: str = "box"
    fillcolor: str | None = None
    fontcolor: str | None = None
    style: str = "filled,rounded"
    color: str | None = None
    penwidth: float | None = None
    tooltip: str | None = None
    image: str | None = None
    width: float | None = None
    height: float | None = None
    fixedsize: str | None = None
    extra_attrs: dict[str, str] = field(default_factory=dict)

    def replace(self, **kwargs: Any) -> NodeSpec:
        """Return a copy of this spec with selected fields replaced.

        Parameters
        ----------
        **kwargs:
            Dataclass fields to replace.

        Returns
        -------
        NodeSpec
            A copied ``NodeSpec`` with the requested field changes.
        """

        return dataclass_replace(self, **kwargs)


# S5 contract (C4): the three node-callback aliases have ONE declaration home
# (this module); ``_render_common`` re-exports them for internal consumers.
NodeSpecFn = Callable[["Layer", NodeSpec], NodeSpec | None]
BackwardNodeSpecFn = Callable[["GradFn", NodeSpec], NodeSpec | None]
CollapsedNodeSpecFn = Callable[["Module", NodeSpec], NodeSpec | None]


# Pickle-visible identity stays the historical path (move-compat law).
NodeSpec.__module__ = "torchlens.visualization.node_spec"

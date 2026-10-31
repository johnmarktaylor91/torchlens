"""Node visualizer factories: callbacks that draw one graph node's picture.

Pass one to ``trace.draw(node_spec_fn=...)``. Each factory returns a callable
that maps a node's tensor (or record) to an image or a node spec.
Spellings DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

from typing import Any

from ._node_visualizers import channel_grid, heatmap, histogram
from .batch_summary import text_table
from .feature_maps import feature_map_node_spec as feature_map

#: Representational-geometry node visualizers, resolved on first access so
#: importing this module does not import ``torchlens.repgeom``.
_REPGEOM_NODES = {
    "mds_scatter": "mds_scatter_node_spec",
    "rdm": "rdm_node_spec",
    "scree": "scree_node_spec",
}

__all__ = [
    "channel_grid",
    "feature_map",
    "heatmap",
    "histogram",
    "mds_scatter",  # noqa: F822 -- resolved by __getattr__
    "rdm",  # noqa: F822 -- resolved by __getattr__
    "scree",  # noqa: F822 -- resolved by __getattr__
    "text_table",
]


def __getattr__(name: str) -> Any:
    """Resolve the representational-geometry node visualizers lazily.

    Parameters
    ----------
    name:
        Attribute requested from ``torchlens.viz.nodes``.

    Returns
    -------
    Any
        The requested node visualizer factory.

    Raises
    ------
    AttributeError
        If ``name`` is not a node visualizer.
    """

    if name in _REPGEOM_NODES:
        from ..repgeom import _node_visuals

        value = getattr(_node_visuals, _REPGEOM_NODES[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module 'torchlens.viz.nodes' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the public node visualizer names."""

    return sorted(__all__)

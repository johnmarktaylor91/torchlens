"""Experimental Dagua rendering bridge exports.

Import this module before selecting ``vis_renderer="dagua"`` from the core
visualization dispatcher.
"""

from ..._visualization.node_spec import NodeSpec
from ._bridge import (
    trace_to_dagua_graph,
)

__torchlens_dagua_opted_in__ = True

__all__ = [
    "NodeSpec",
    "trace_to_dagua_graph",
]

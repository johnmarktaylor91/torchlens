"""Graphviz-based computational graph rendering for Trace objects.

Renders the computational graph captured by TorchLens as a Graphviz Digraph,
supporting two visualization modes:

- **unrolled** (default): every pass of every layer is a separate node.
  Uses ``layer_dict_main_keys`` as the node source.
- **rolled**: layers with multiple ops are collapsed into a single node
  with edge labels showing which ops an edge applies to.  Uses
  ``layer_logs`` (Layer objects) as the node source.

Key mechanisms:

- **Collapsed modules**: when ``vis_call_depth`` is set, layers nested
  deeper than the threshold are collapsed into ``box3d`` module summary
  nodes.  ``_is_collapsed_module`` is the gatekeeper; ``_build_collapsed_module_node``
  renders the summary.  Intra-module edges between layers in the same
  collapsed module are skipped to avoid clutter.

- **Edge deduplication**: ``edges_used`` (set of (tail, head) tuples) prevents
  duplicate edges when multiple layers map to the same collapsed module node.

- **Override system**: six override dicts (graph, node, nested_node, edge,
  grad_edge, module) allow callers to customize any Graphviz attribute.
  Values can be static strings or callables receiving ``(trace, node)``
  for dynamic computation.

- **_layers_logged guard**: rendering requires all layers to be present in the
  Trace. This check prevents IndexError crashes when nodes reference absent layers.
"""

from ._render_common import *  # noqa: F403
from ._render_leaf import *  # noqa: F403
from ._render_edges import *  # noqa: F403
from ._render_nodes import *  # noqa: F403
from ._render_flow import *  # noqa: F403
from ._render_dot import *  # noqa: F403
from ._render_entrypoints import *  # noqa: F403

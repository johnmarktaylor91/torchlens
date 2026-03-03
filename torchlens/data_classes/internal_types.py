"""Internal dataclasses for parameter bundles passed between functions.

These types consolidate groups of parameters that always travel together
through the call chain, replacing loose positional arguments with named fields.
They are internal to TorchLens and not part of the public API.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(slots=True)
class FuncExecutionContext:
    """Execution context captured around a single torch function call.

    Bundles the timing, RNG state, and autocast state recorded during
    function execution in ``torch_func_decorator``.  Threaded through
    ``log_function_output_tensors`` → ``_build_shared_fields_dict``.
    """

    time_elapsed: float
    rng_states: Dict
    autocast_state: Dict


@dataclass(slots=True)
class VisualizationOverrides:
    """User-supplied graphviz attribute overrides for ``render_graph``.

    Bundles the six override dicts accepted by the public ``render_graph``
    method and threads them to internal helpers (``_add_node_to_graphviz``,
    ``_add_edges_for_node``, etc.) as a single object.
    """

    graph: Optional[Dict] = None
    node: Optional[Dict] = None
    nested_node: Optional[Dict] = None
    edge: Optional[Dict] = None
    gradient_edge: Optional[Dict] = None
    module: Optional[Dict] = None

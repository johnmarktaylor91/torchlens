"""Internal dataclasses for parameter bundles passed between functions.

These types consolidate groups of parameters that always travel together
through the call chain, replacing loose positional arguments with named fields.
They are internal to TorchLens and not part of the public API.

Both use ``@dataclass(slots=True)`` for memory efficiency and faster
attribute access (no per-instance ``__dict__``).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(slots=True)
class FuncExecutionContext:
    """Execution context captured around a single torch function call.

    Bundles the timing, RNG state, and autocast state recorded during
    function execution in ``torch_func_decorator``.  Threaded through
    ``log_function_output_tensors`` -> ``_build_shared_fields_dict``.

    Attributes:
        time_elapsed: Wall-clock time for the function call (seconds).
        rng_states: Snapshot of CPU/GPU RNG states before the call,
            used for deterministic replay during validation.
        autocast_state: Snapshot of autocast settings (enabled, dtype)
            active during the call.
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

    Each dict, if provided, is merged into the graphviz attributes for
    the corresponding element type.  ``None`` means "use defaults".

    Attributes:
        graph: Overrides for the top-level graph (e.g. rankdir, bgcolor).
        node: Overrides for regular layer nodes (e.g. shape, color).
        nested_node: Overrides for nodes inside module clusters.
        edge: Overrides for data-flow edges.
        gradient_edge: Overrides for gradient-flow edges.
        module: Overrides for module subgraph clusters (e.g. label style).
    """

    graph: Optional[Dict] = None
    node: Optional[Dict] = None
    nested_node: Optional[Dict] = None
    edge: Optional[Dict] = None
    gradient_edge: Optional[Dict] = None
    module: Optional[Dict] = None

"""Graphviz-based computational graph rendering for ModelLog objects.

Renders the computational graph captured by TorchLens as a Graphviz Digraph,
supporting two visualization modes:

- **unrolled** (default): every pass of every layer is a separate node.
  Uses ``layer_dict_main_keys`` as the node source.
- **rolled**: layers with multiple passes are collapsed into a single node
  with edge labels showing which passes an edge applies to.  Uses
  ``layer_logs`` (LayerLog objects) as the node source.

Key mechanisms:

- **Collapsed modules**: when ``vis_nesting_depth`` is set, layers nested
  deeper than the threshold are collapsed into ``box3d`` module summary
  nodes.  ``_is_collapsed_module`` is the gatekeeper; ``_build_collapsed_module_node``
  renders the summary.  Intra-module edges between layers in the same
  collapsed module are skipped to avoid clutter.

- **Edge deduplication**: ``edges_used`` (set of (tail, head) tuples) prevents
  duplicate edges when multiple layers map to the same collapsed module node.

- **Override system**: six override dicts (graph, node, nested_node, edge,
  gradient_edge, module) allow callers to customize any Graphviz attribute.
  Values can be static strings or callables receiving ``(model_log, node)``
  for dynamic computation.

- **_all_layers_logged guard**: rendering requires all layers to be present
  in the ModelLog (either saved or kept-unsaved).  This check prevents
  IndexError crashes when ``keep_unsaved_layers=False`` was used and nodes
  reference absent layers.
"""

import subprocess
import warnings
from collections import defaultdict
from typing import Any, Optional, Dict, List, Set, TYPE_CHECKING, Tuple, Union

import graphviz

from ..data_classes.internal_types import VisualizationOverrides
from ..utils.display import in_notebook, int_list_to_compact_str, _vprint
from ..data_classes.layer_pass_log import LayerPassLog
from ..data_classes.layer_log import LayerLog

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog

# -- Color palette for node types --
INPUT_COLOR = "#98FB98"  # Light green
OUTPUT_COLOR = "#ff9999"  # Light red/salmon
PARAMS_NODE_BG_COLOR = "#E6E6E6"  # Generic param (no ParamLog available)
TRAINABLE_PARAMS_BG_COLOR = "#D9D9D9"  # Light gray for trainable params
FROZEN_PARAMS_BG_COLOR = "#B0B0B0"  # Darker gray for frozen params
BUFFER_NODE_COLOR = "#888888"  # Medium gray for buffer nodes
GRADIENT_ARROW_COLOR = "#9197F6"  # Light blue/purple for backward edges
DEFAULT_BG_COLOR = "white"
BOOL_NODE_COLOR = "#F7D460"  # Yellow for terminal boolean layers

# -- Module subgraph border widths --
MAX_MODULE_PENWIDTH = 5
MIN_MODULE_PENWIDTH = 2
PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH

# Commutative functions: argument order doesn't matter, so we skip arg-position
# labels on their incoming edges to reduce visual clutter.
COMMUTE_FUNCS = ["add", "mul", "cat", "eq", "ne"]


def render_graph(
    self: "ModelLog",
    vis_opt: str = "unrolled",
    vis_nesting_depth: int = 1000,
    vis_outpath: str = "modelgraph",
    vis_graph_overrides: Optional[Dict] = None,
    vis_node_overrides: Optional[Dict] = None,
    vis_nested_node_overrides: Optional[Dict] = None,
    vis_edge_overrides: Optional[Dict] = None,
    vis_gradient_edge_overrides: Optional[Dict] = None,
    vis_module_overrides: Optional[Dict] = None,
    save_only: bool = False,
    vis_fileformat: str = "pdf",
    show_buffer_layers: bool = False,
    direction: str = "bottomup",
    vis_node_placement: str = "auto",
) -> str:
    """Render the computational graph as a Graphviz Digraph.

    Orchestrates the full rendering pipeline:
    1. Validates that all layers are logged (``_all_layers_logged`` guard).
    2. Iterates over entries_to_plot, building nodes and edges.
    3. Groups edges into module subgraph clusters.
    4. Renders to file and optionally displays.

    Args:
        vis_opt: ``'unrolled'`` (each pass is a separate node) or ``'rolled'``
            (multi-pass layers collapsed into one node with pass annotations).
        vis_nesting_depth: Maximum module nesting levels to show before
            collapsing deeper layers into ``box3d`` module summary nodes.
            Use 0 to show all layers without collapsing.
        vis_outpath: Output file path (extension auto-stripped).
        vis_graph_overrides: Graphviz graph-level attribute overrides.
        vis_node_overrides: Overrides for standard (non-collapsed) nodes.
        vis_nested_node_overrides: Overrides for collapsed module nodes.
        vis_edge_overrides: Overrides for forward edges.
        vis_gradient_edge_overrides: Overrides for backward (gradient) edges.
        vis_module_overrides: Overrides for module subgraph boxes.
        save_only: If True, save without opening a viewer.
        vis_fileformat: Output format (pdf, png, svg, etc.).
        show_buffer_layers: Whether to include buffer layers in the graph.
        direction: Layout direction: ``'bottomup'``, ``'topdown'``, or ``'leftright'``.
        vis_node_placement: Layout engine: ``'auto'`` (default), ``'dot'``, ``'elk'``,
            or ``'sfdp'``.  ``'auto'`` uses dot for small graphs and ELK (or sfdp
            fallback) for large ones.

    Returns:
        The Graphviz DOT source string.

    Raises:
        ValueError: If ``_all_layers_logged`` is False (layers were discarded
            by ``keep_unsaved_layers=False``).
    """
    overrides = VisualizationOverrides(
        graph=vis_graph_overrides or {},
        node=vis_node_overrides or {},
        nested_node=vis_nested_node_overrides or {},
        edge=vis_edge_overrides or {},
        gradient_edge=vis_gradient_edge_overrides or {},
        module=vis_module_overrides or {},
    )

    # THE _all_layers_logged guard: prevents IndexError crashes that would
    # occur when edges reference layers that were discarded by
    # keep_unsaved_layers=False.  This is the single chokepoint that
    # protects all downstream rendering code from missing-layer lookups.
    if not self._all_layers_logged:
        raise ValueError(
            "Must have all layers logged in order to render the graph; either save all layers,"
            "set keep_unsaved_layers to True, or use show_model_graph."
        )

    # Fix the filename if need be, to remove the extension:
    split_outpath = vis_outpath.split(".")
    if split_outpath[-1] in [
        "pdf",
        "png",
        "jpg",
        "svg",
        "jpeg",
        "bmp",
        "pic",
        "tif",
        "tiff",
    ]:
        vis_outpath = ".".join(split_outpath[:-1])

    # Unrolled: iterate LayerPassLog objects (one node per pass).
    # Rolled: iterate LayerLog objects (one node per logical layer, multi-pass
    # collapsed into a single node with edge annotations).
    if vis_opt == "unrolled":
        entries_to_plot = self.layer_dict_main_keys
    elif vis_opt == "rolled":
        entries_to_plot = self.layer_logs  # type: ignore[assignment]
    else:
        raise ValueError("vis_opt must be either 'rolled' or 'unrolled'")

    if direction == "bottomup":
        rankdir = "BT"
    elif direction == "leftright":
        rankdir = "LR"
    elif direction == "topdown":
        rankdir = "TB"
    else:
        raise ValueError("direction must be either 'bottomup', 'topdown', or 'leftright'.")

    # Resolve the layout engine early to potentially skip graphviz.Digraph construction.
    from .elk_layout import get_node_placement_engine

    num_nodes = len(entries_to_plot)
    engine = get_node_placement_engine(vis_node_placement, num_nodes)
    _vprint(self, f"Rendering {vis_opt} graph ({num_nodes} nodes, format={vis_fileformat})")
    _vprint(self, f"Layout engine: {engine}")

    if self.total_params == 0:
        params_detail = "0 params"
    elif self.total_params_frozen == 0:
        params_detail = (
            f"{self.total_params} params (all trainable, {self.total_params_fsize_nice})"
        )
    elif self.total_params_trainable == 0:
        params_detail = f"{self.total_params} params (all frozen, {self.total_params_fsize_nice})"
    else:
        params_detail = (
            f"{self.total_params} params "
            f"({self.total_params_trainable}/{self.total_params} trainable, "
            f"{self.total_params_fsize_nice})"
        )

    graph_caption = (
        f"<<B>{self.model_name}</B><br align='left'/>{self.num_tensors_total} "
        f"tensors total ({self.tensor_fsize_total_nice})"
        f"<br align='left'/>{params_detail}<br align='left'/>>"
    )

    # ELK fast path: skip graphviz.Digraph construction entirely.
    # Generates DOT directly with ELK positions and cluster subgraphs (module boxes).
    if engine == "elk":
        from .elk_layout import render_elk_direct, render_with_sfdp

        try:
            result = render_elk_direct(
                self,
                entries_to_plot,
                vis_opt,
                vis_nesting_depth,
                show_buffer_layers,
                overrides,
                vis_outpath,
                vis_fileformat,
                save_only,
                graph_caption,
                rankdir,
            )
            _vprint(self, f"Graph saved to {vis_outpath}.{vis_fileformat}")
            return result
        except RuntimeError as e:
            warnings.warn(f"ELK layout failed ({e}), falling back to sfdp.")
            engine = "sfdp"  # fall through to build graphviz.Digraph for sfdp

    dot = graphviz.Digraph(
        name=self.model_name,
        comment="Computational graph for the feedforward sweep",
        format=vis_fileformat,
    )

    graph_args = {
        "rankdir": rankdir,
        "label": graph_caption,
        "labelloc": "t",
        "labeljust": "left",
        "ordering": "out",
    }

    # Override system: callers can pass dicts of Graphviz attributes to
    # customize rendering.  Values can be static (str) or dynamic (callable
    # receiving the ModelLog, evaluated at render time).
    for arg_name, arg_val in overrides.graph.items():  # type: ignore[union-attr]
        if callable(arg_val):
            graph_args[arg_name] = str(arg_val(self))
        else:
            graph_args[arg_name] = str(arg_val)

    dot.graph_attr.update(graph_args)
    dot.node_attr.update({"ordering": "out"})

    # Accumulate edges per module cluster; actual Graphviz subgraphs are
    # created at the end in _setup_subgraphs to ensure proper nesting.
    module_cluster_dict: Dict[str, Any] = defaultdict(
        lambda: {"edges": [], "has_input_ancestor": False}
    )
    # Track which collapsed module nodes have been added to avoid duplicates
    # (multiple layers in the same collapsed module would otherwise each try
    # to create the same box3d node).
    collapsed_modules: Set[str] = set()
    # Edge deduplication: (tail_name, head_name) pairs already added.
    # Critical when collapsed modules cause many layers to map to the same
    # node name -- without this, we'd get duplicate edges.
    edges_used: Set[str] = set()

    for node_barcode, node in entries_to_plot.items():
        if node.is_buffer_layer and not show_buffer_layers:
            continue
        _add_node_to_graphviz(
            self,
            node,
            dot,
            module_cluster_dict,
            edges_used,
            vis_opt,
            collapsed_modules,
            vis_nesting_depth,
            show_buffer_layers,
            overrides,
        )

    # Finally, set up the subgraphs.
    _setup_subgraphs(self, dot, vis_opt, module_cluster_dict, overrides)

    if in_notebook() and not save_only:
        from IPython.display import display  # #72: lazy import

        display(dot)

    # ELK was already handled above (early return). Only dot/sfdp reach here.
    from .elk_layout import render_with_sfdp

    _RENDER_TIMEOUT = 120  # seconds
    source_path = dot.save(vis_outpath)
    try:
        if engine == "sfdp":
            render_with_sfdp(source_path, vis_outpath, vis_fileformat, save_only)
        else:
            # dot engine (default for small graphs)
            rendered_path = f"{vis_outpath}.{vis_fileformat}"
            cmd = [dot.engine, f"-T{vis_fileformat}", "-o", rendered_path, source_path]
            subprocess.run(cmd, timeout=_RENDER_TIMEOUT, check=True, capture_output=True)
            if not save_only:
                graphviz.backend.viewing.view(rendered_path)
        _vprint(self, f"Graph saved to {vis_outpath}.{vis_fileformat}")
    except subprocess.TimeoutExpired:
        warnings.warn(
            f"Graphviz render timed out ({_RENDER_TIMEOUT}s) for graph with "
            f"{self.num_tensors_total} nodes. DOT source saved to "
            f"'{source_path}'. Consider using vis_node_placement='sfdp' or "
            f"vis_nesting_depth to collapse modules."
        )
    except subprocess.CalledProcessError as e:
        warnings.warn(f"Graphviz render failed: {e.stderr.decode()}")
    finally:
        import os

        if os.path.exists(source_path):
            os.remove(source_path)
    return dot.source


def _add_node_to_graphviz(
    self: "ModelLog",
    node: Union["LayerPassLog", "LayerLog"],
    graphviz_graph,
    module_edge_dict: Dict,
    edges_used: Set,
    vis_opt: str,
    collapsed_modules: Set,
    vis_nesting_depth: int = 1000,
    show_buffer_layers: bool = False,
    overrides: Optional[VisualizationOverrides] = None,
) -> None:
    """Adds a node and its relevant edges to the graphviz figure.

    Args:
        node: node to add
        graphviz_graph: The graphviz object to add the node to.
        module_edge_dict: Dictionary of the module clusters.
        vis_opt: Whether to roll the graph or not
        vis_nesting_depth: How many levels of nested modules to show
        collapsed_modules: Labels of collapsed module nodes that have been made so far.
        show_buffer_layers: Whether to show the buffer layers
        overrides: Graphviz attribute overrides for nodes, edges, etc.
    """
    is_collapsed_module = _is_collapsed_module(node, vis_nesting_depth)

    if is_collapsed_module:
        _build_collapsed_module_node(
            self,
            node,
            graphviz_graph,
            collapsed_modules,
            vis_opt,
            vis_nesting_depth,
            overrides,  # type: ignore[arg-type]
        )
        node_color = "black"
    else:
        node_color = _build_layer_node(
            self,
            node,
            graphviz_graph,
            show_buffer_layers,
            vis_opt,
            overrides,  # type: ignore[arg-type]
        )

    _add_edges_for_node(
        self,
        node,
        is_collapsed_module,
        vis_nesting_depth,
        node_color,
        module_edge_dict,
        edges_used,
        graphviz_graph,
        vis_opt,
        show_buffer_layers,
        overrides,
    )


def _is_collapsed_module(node, vis_nesting_depth: int) -> bool:
    """THE IndexError guard for collapsed module rendering.

    Returns True if the node is nested deep enough to be rendered as a
    collapsed ``box3d`` module summary node instead of an individual layer.

    This function is the single decision point that determines whether a node
    gets its own graphviz node or is absorbed into a module box.  Getting this
    wrong causes IndexError when ``_build_collapsed_module_node`` tries to
    access ``containing_modules_origin_nested[vis_nesting_depth - 1]``.

    Special cases:
    - ``vis_nesting_depth == 0``: show all layers, never collapse (#94).
    - ``is_bottom_level_submodule_output``: the node represents the output of
      its innermost module, so its effective nesting depth is one less (it
      visually "belongs" to the parent scope).

    Args:
        node: The LayerPassLog or LayerLog node to check.
        vis_nesting_depth: Maximum nesting depth before collapsing into a module box.
    """
    if vis_nesting_depth == 0:
        return False  # #94: depth 0 means show all layers, never collapse

    node_nesting_depth = len(node.containing_modules_origin_nested)
    # Bottom-level submodule outputs are rendered at the parent nesting level,
    # not their own, so subtract 1 from their effective depth.
    if getattr(node, "is_bottom_level_submodule_output", False):
        node_nesting_depth -= 1

    if node_nesting_depth >= vis_nesting_depth:
        return True
    else:
        return False


def _build_layer_node(
    self: "ModelLog",
    node,
    graphviz_graph,
    show_buffer_layers,
    vis_opt,
    overrides: VisualizationOverrides,
) -> str:
    """Builds and adds a standard (non-collapsed) layer node to the graphviz graph.

    Args:
        node: The LayerPassLog or LayerLog node to render.
        graphviz_graph: The graphviz Digraph object to add the node to.
        show_buffer_layers: Whether buffer layers are shown.
        vis_opt: 'unrolled' or 'rolled'.
        overrides: Graphviz attribute overrides.

    Returns:
        The node color string used for this node.
    """
    # Get the address, shape, color, and line style:

    node_address, node_shape, node_color = _get_node_address_shape_color(
        self, node, show_buffer_layers
    )
    node_bg_color = _get_node_bg_color(self, node)

    if node.has_input_ancestor:
        line_style = "solid"
    else:
        line_style = "dashed"

    # Get the text for the node label:

    node_label = _make_node_label(node, node_address, vis_opt)

    # Graphviz node names can't contain colons (used for port syntax), so
    # replace ":" with "pass" in pass-qualified labels (e.g., "relu_1:2" -> "relu_1pass2").
    node_args = {
        "name": node.layer_label.replace(":", "pass"),
        "label": node_label,
        "fontcolor": node_color,
        "color": node_color,
        "style": f"filled,{line_style}",
        "fillcolor": node_bg_color,
        "shape": node_shape,
        "ordering": "out",
    }
    # Colon in bg_color means it's a gradient fill (e.g.,
    # "#D9D9D9:#B0B0B0" for mixed trainable/frozen params).
    # Graphviz requires gradientangle to render gradients.
    if ":" in node_bg_color:
        node_args["gradientangle"] = "0"

    for arg_name, arg_val in overrides.node.items():  # type: ignore[union-attr]
        if callable(arg_val):
            node_args[arg_name] = str(arg_val(self, node))
        else:
            node_args[arg_name] = str(arg_val)

    graphviz_graph.node(**node_args)

    if node.is_last_output_layer:
        with graphviz_graph.subgraph() as s:
            s.attr(rank="sink")
            s.node(node.layer_label.replace(":", "pass"))

    return node_color


def _build_collapsed_module_node(
    self: "ModelLog",
    node,
    graphviz_graph,
    collapsed_modules,
    vis_opt,
    vis_nesting_depth,
    overrides: VisualizationOverrides,
) -> None:
    """Builds and adds a collapsed module box node to the graphviz graph.

    Args:
        node: The LayerPassLog or LayerLog node triggering the collapse.
        graphviz_graph: The graphviz Digraph object to add the node to.
        collapsed_modules: Set of collapsed module names already added; updated in place.
        vis_opt: 'unrolled' or 'rolled'.
        vis_nesting_depth: Maximum nesting depth; nodes at this depth are collapsed.
        overrides: Graphviz attribute overrides.
    """
    # Access the module at the collapse threshold depth.  This index is safe
    # because _is_collapsed_module already verified the node is deep enough.
    module_address_w_pass = node.containing_modules_origin_nested[vis_nesting_depth - 1]
    # rsplit with maxsplit=1 handles module names containing colons (#104).
    module_tuple = module_address_w_pass.rsplit(":", 1)
    module_output_layer = self[module_address_w_pass]
    module_output_shape = module_output_layer.tensor_shape or ()
    module_output_fsize = module_output_layer.tensor_fsize_nice
    module_address, pass_num = module_tuple
    ml = self.modules[module_address]
    module_type = ml.module_class_name  # type: ignore[union-attr]
    module_num_passes = ml.num_passes  # type: ignore[union-attr]
    module_nparams = ml.num_params  # type: ignore[union-attr]

    # In unrolled mode, each pass of a module is a separate collapsed node
    # (e.g., "encoder.layer.0pass1").  In rolled mode, all passes share one
    # node (e.g., "encoder.layer.0").
    if vis_opt == "unrolled":
        node_name = "pass".join(module_tuple)
        mpl = self.modules[module_address_w_pass]
        module_num_tensors = mpl.num_layers
        module_has_input_ancestor = any(self[layer].has_input_ancestor for layer in mpl.layers)
    else:
        node_name = module_tuple[0]
        module_num_tensors = ml.num_layers
        module_has_input_ancestor = any(self[layer].has_input_ancestor for layer in ml.all_layers)  # type: ignore[union-attr]

    # Deduplicate: multiple layers in the same collapsed module will each
    # trigger this function, but the node should only be added once.
    if node_name in collapsed_modules:
        return

    if module_num_passes == 1:
        node_title = f"<b>@{module_address}</b>"
    elif vis_opt == "unrolled" and (module_num_passes > 1):
        node_title = f"<b>@{module_address}:{pass_num}</b>"
    else:
        node_title = f"<b>@{module_address} (x{module_num_passes})</b>"

    if len(module_output_shape) > 1:
        tensor_shape_str = "x".join([str(x) for x in module_output_shape])
    elif len(module_output_shape) == 1:  # #100: use module_output_shape, not node.tensor_shape
        tensor_shape_str = f"x{module_output_shape[0]}"  # type: ignore[misc]
    else:
        tensor_shape_str = "x1"

    module_nparams_trainable = ml.num_params_trainable  # type: ignore[union-attr]
    module_nparams_frozen = ml.num_params_frozen  # type: ignore[union-attr]

    if module_nparams > 0:
        if module_nparams_frozen == 0:
            bg_color = TRAINABLE_PARAMS_BG_COLOR
        elif module_nparams_trainable == 0:
            bg_color = FROZEN_PARAMS_BG_COLOR
        else:
            bg_color = TRAINABLE_PARAMS_BG_COLOR + ":" + FROZEN_PARAMS_BG_COLOR
    else:
        bg_color = DEFAULT_BG_COLOR

    if module_has_input_ancestor:
        line_style = "solid"
    else:
        line_style = "dashed"

    # Build param detail string for collapsed module
    if module_nparams == 0:
        param_detail = "0 parameters"
    elif module_nparams_frozen == 0:
        param_detail = f"{module_nparams} params (all trainable)"
    elif module_nparams_trainable == 0:
        param_detail = f"{module_nparams} params (all frozen)"
    else:
        param_detail = (
            f"{module_nparams} params ({module_nparams_trainable} trainable, "
            f"{module_nparams_frozen} frozen)"
        )

    node_label = (
        f"<{node_title}<br/>"
        f"{module_type}<br/>"
        f"{tensor_shape_str} ({module_output_fsize})<br/>"
        f"{module_num_tensors} layers total<br/>"
        f"{param_detail}>"
    )

    node_args = {
        "name": node_name,
        "label": node_label,
        "fontcolor": "black",
        "color": "black",
        "style": f"filled,{line_style}",
        "fillcolor": bg_color,
        "shape": "box3d",
        "ordering": "out",
    }
    if ":" in bg_color:
        node_args["gradientangle"] = "0"

    for arg_name, arg_val in overrides.nested_node.items():  # type: ignore[union-attr]
        if callable(arg_val):
            node_args[arg_name] = str(arg_val(self, node))
        else:
            node_args[arg_name] = str(arg_val)

    graphviz_graph.node(**node_args)
    collapsed_modules.add(node_name)


def _get_node_address_shape_color(
    self: "ModelLog",
    node: Union["LayerPassLog", "LayerLog"],
    show_buffer_layers: bool,
) -> Tuple[str, str, str]:
    """Gets the node shape, address, and color for the graphviz figure.

    Args:
        node: node to add

    Returns:
        node_address: address of the node
        node_shape: shape of the node
        node_color: color of the node
    """
    if not show_buffer_layers:
        only_non_buffer_layer = _is_only_non_buffer_in_module(self, node)
    else:
        only_non_buffer_layer = False

    if (node.is_bottom_level_submodule_output or only_non_buffer_layer) and (
        len(node.containing_modules_origin_nested) > 0
    ):
        if isinstance(node, LayerPassLog):
            module_pass_exited = node.containing_modules_origin_nested[-1]
            module, _ = module_pass_exited.split(":")
            if self.modules[module].num_passes == 1:  # type: ignore[union-attr]
                node_address = module
            else:
                node_address = module_pass_exited
        else:
            sample_module_pass = node.containing_modules_origin_nested[-1]
            module = sample_module_pass.split(":")[0]
            node_address = module

        node_address = "<br/>@" + node_address
        node_shape = "box"
        node_color = "black"
    elif node.is_buffer_layer:
        if (self.buffer_num_passes[node.buffer_address] == 1) or (
            isinstance(node, LayerLog) and node.layer_passes_total > 1
        ):
            buffer_address = node.buffer_address
        else:
            buffer_address = f"{node.buffer_address}:{node.buffer_pass}"
        node_address = "<br/>@" + buffer_address
        node_shape = "box"
        node_color = BUFFER_NODE_COLOR
    elif node.is_output_layer or node.is_input_layer:
        node_address = "<br/>@" + node.input_output_address
        node_shape = "oval"
        node_color = "black"
    else:
        node_address = ""
        node_shape = "oval"
        node_color = "black"

    return node_address, node_shape, node_color


def _is_only_non_buffer_in_module(
    self: "ModelLog", node: Union["LayerPassLog", "LayerLog"]
) -> bool:
    """Returns True if a layer is the only non-buffer layer in a leaf module.

    Leaf modules are those with no child submodules. Container modules with
    functional ops at the end should NOT match — those ops are rendered as
    ovals, not boxes (issue #48).

    Args:
        node: The LayerPassLog or LayerLog node to check.
    """
    # Check whether it leaves its module:
    if not (
        (len(node.modules_exited) > 0)
        and (len(node.containing_modules_origin_nested) > 0)
        and (node.containing_modules_origin_nested[-1].split(":")[0] in node.modules_exited)
    ):
        return False

    # Only apply box rendering for leaf modules (no child submodules).
    exited_module = node.containing_modules_origin_nested[-1].split(":")[0]
    if exited_module in self.modules and len(self.modules[exited_module].call_children) > 0:
        return False

    # Now check whether all of its parents are either buffers, or are outside the module.
    # If any aren't, return False.

    for parent_layer_label in node.parent_layers:
        if isinstance(node, LayerPassLog):
            parent_layer = self[parent_layer_label]
        else:
            parent_layer = self.layer_logs[parent_layer_label]  # type: ignore[assignment]
        if (not parent_layer.is_buffer_layer) and (
            (len(parent_layer.containing_modules_origin_nested) > 0)
            and parent_layer.containing_modules_origin_nested[-1]
            == node.containing_modules_origin_nested[-1]
        ):
            return False

    return True


def _get_node_bg_color(self: "ModelLog", node: Union["LayerPassLog", "LayerLog"]) -> str:
    """Returns the background color hex string for a graph node based on its type.

    Maps node types to colors: input=green, output=red, boolean=orange,
    parameterized layers=blue (trainable) or gray (frozen), default=white.

    Args:
        node: node to add

    Returns:
        node_bg_color: background color of the node
    """
    if node.is_input_layer:
        bg_color = INPUT_COLOR
    elif node.is_output_layer:
        bg_color = OUTPUT_COLOR
    elif node.is_terminal_bool_layer:
        bg_color = BOOL_NODE_COLOR
    elif node.computed_with_params:
        param_logs = getattr(node, "parent_param_logs", [])
        if param_logs:
            trainable_flags = [pl.trainable for pl in param_logs]
            all_trainable = all(trainable_flags)
            all_frozen = not any(trainable_flags)
            if all_trainable:
                bg_color = TRAINABLE_PARAMS_BG_COLOR
            elif all_frozen:
                bg_color = FROZEN_PARAMS_BG_COLOR
            else:
                bg_color = TRAINABLE_PARAMS_BG_COLOR + ":" + FROZEN_PARAMS_BG_COLOR
        else:
            bg_color = PARAMS_NODE_BG_COLOR
    else:
        bg_color = DEFAULT_BG_COLOR
    return bg_color


def _make_node_label(
    node: Union["LayerPassLog", "LayerLog"],
    node_address: str,
    vis_opt: str,
) -> str:
    """Builds an HTML-table label string for a graphviz node.

    Assembles rows for the layer name, tensor shape, operation type, and other
    metadata into an HTML table used as the node label in graphviz rendering.
    """
    # Pass info:

    if (node.layer_passes_total > 1) and (vis_opt == "unrolled"):
        pass_label = f":{node.pass_num}"
    elif (node.layer_passes_total > 1) and (vis_opt == "rolled"):
        pass_label = f" (x{node.layer_passes_total})"
    else:
        pass_label = ""

    # Tensor shape info:

    if len(node.tensor_shape) > 1:
        tensor_shape_str = "x".join([str(x) for x in node.tensor_shape])
    elif len(node.tensor_shape) == 1:
        tensor_shape_str = f"x{node.tensor_shape[0]}"
    else:
        tensor_shape_str = "x1"

    # Layer param info:

    param_label = _make_param_label(node)

    tensor_fsize = node.tensor_fsize_nice
    if node.layer_type in ["input", "output", "buffer"]:
        node_title = f"<b>{node.layer_type}_{node.layer_type_num}{pass_label}</b>"
    else:
        node_title = (
            f"<b>{node.layer_type}_{node.layer_type_num}_{node.layer_total_num}{pass_label}</b>"
        )

    if node.is_terminal_bool_layer:
        label_text = str(node.atomic_bool_val).upper()
        bool_label = f"<b><u>{label_text}:</u></b><br/><br/>"
    else:
        bool_label = ""

    node_label = (
        f"<{bool_label}{node_title}<br/>{tensor_shape_str} "
        f"({tensor_fsize}){param_label}{node_address}>"
    )

    return node_label


def _format_shape_str(shape: tuple) -> str:
    """Formats a shape tuple as a compact string like '3x3x64'."""
    if len(shape) > 1:
        return "x".join(str(s) for s in shape)
    elif len(shape) == 1:
        return f"x{shape[0]}"
    return "x1"


def _make_param_label(node: Union["LayerPassLog", "LayerLog"]) -> str:
    """Makes the label for parameters of a node.

    Uses param names and bracket convention when ParamLog objects are available:
    round brackets () for trainable, square brackets [] for frozen.
    """
    if node.num_param_tensors == 0:
        return ""

    param_logs = getattr(node, "parent_param_logs", [])
    if param_logs:
        parts = []
        for pl in param_logs:
            shape_str = _format_shape_str(pl.shape)
            if pl.trainable:
                parts.append(f"{pl.name}: ({shape_str})")
            else:
                parts.append(f"{pl.name}: [{shape_str}]")
        param_label = "<br/>params: " + ", ".join(parts)
    else:
        each_param_shape = [_format_shape_str(s) for s in node.parent_param_shapes]
        param_label = "<br/>params: " + ", ".join(each_param_shape)
    return param_label


def _add_edges_for_node(
    self: "ModelLog",
    parent_node: Union["LayerPassLog", "LayerLog"],
    parent_is_collapsed_module: bool,
    vis_nesting_depth: int,
    node_color: str,
    module_edge_dict: Dict,
    edges_used: Set,
    graphviz_graph,
    vis_opt: str = "unrolled",
    show_buffer_layers: bool = False,
    overrides: Optional[VisualizationOverrides] = None,
) -> None:
    """Add forward (and optionally gradient) edges from a parent node to all its children.

    Handles several complex cases:

    - **Collapsed module nodes**: when parent or child is collapsed, the edge
      endpoint is the module box name, not the individual layer name.
    - **Intra-module edge skip**: when both parent and child map to the SAME
      collapsed module box AND share the same module nesting prefix up to
      ``vis_nesting_depth``, the edge is internal to the collapsed module
      and should not be drawn.
    - **Edge deduplication**: ``edges_used`` prevents duplicate edges that
      arise when multiple layers map to the same collapsed module node.
    - **Argument labels**: for non-commutative ops with multiple parents,
      edge labels show which argument position each parent occupies.
      Note: uses substring matching on layer_label for arg_label lookup,
      which has a theoretical false-positive risk if one label is a
      substring of another (extremely rare in practice).
    - **Pass annotations** (rolled mode): ``_label_rolled_pass_nums`` adds
      tail/head labels showing which passes an edge applies to.

    Args:
        parent_node: The node to add edges for.
        parent_is_collapsed_module: Whether the node is a collapsed module node.
        vis_nesting_depth: How many levels of module nesting to show.
        node_color: Color of the node.
        module_edge_dict: Dict mapping each cluster to its edges.
        edges_used: Set of (tail, head) pairs already added.
        graphviz_graph: The graphviz graph object.
        vis_opt: ``'unrolled'`` or ``'rolled'``.
        show_buffer_layers: Whether buffer layers are shown.
        overrides: Graphviz attribute overrides.
    """
    for child_layer_label in parent_node.child_layers:
        if vis_opt == "unrolled":
            child_node = self.layer_dict_main_keys[child_layer_label]
        elif vis_opt == "rolled":
            child_node = self.layer_logs[child_layer_label]  # type: ignore[assignment]
        else:
            raise ValueError(f"vis_opt must be 'unrolled' or 'rolled', not {vis_opt}")

        if child_node.is_buffer_layer and not show_buffer_layers:
            continue

        if parent_node.has_input_ancestor:
            edge_style = "solid"
        else:
            edge_style = "dashed"

        if parent_is_collapsed_module:
            module_name_w_pass = parent_node.containing_modules_origin_nested[vis_nesting_depth - 1]
            module_tuple = module_name_w_pass.split(":")
            if vis_opt == "unrolled":
                tail_name = "pass".join(module_tuple)
            else:
                tail_name = module_tuple[0]
        else:
            tail_name = parent_node.layer_label.replace(":", "pass")

        child_is_collapsed_module = _is_collapsed_module(child_node, vis_nesting_depth)

        if child_is_collapsed_module:
            module_name_w_pass = child_node.containing_modules_origin_nested[vis_nesting_depth - 1]
            module_tuple = module_name_w_pass.split(":")
            if vis_opt == "unrolled":
                head_name = "pass".join(module_tuple)
            else:
                head_name = module_tuple[0]
        else:
            head_name = child_node.layer_label.replace(":", "pass")

        both_nodes_collapsed_modules = parent_is_collapsed_module and child_is_collapsed_module

        # Collapsed module intra-edge skip: if both nodes are collapsed AND
        # they share the same module path up to vis_nesting_depth, the edge
        # is internal to the collapsed module box and should not be drawn.
        # The tail_name != head_name check handles the case where they map to
        # different collapsed modules (cross-module edge, should be drawn).
        if both_nodes_collapsed_modules and (tail_name != head_name):
            child_containing_modules = child_node.containing_modules_origin_nested[:]
            parent_containing_modules = parent_node.containing_modules_origin_nested[:]
            # Adjust for bottom-level submodule outputs (they belong to parent scope).
            if child_node.is_bottom_level_submodule_output:
                child_containing_modules = child_containing_modules[:-1]
            if parent_node.is_bottom_level_submodule_output:
                parent_containing_modules = parent_containing_modules[:-1]
            if (
                child_containing_modules[:vis_nesting_depth]
                == parent_containing_modules[:vis_nesting_depth]
            ):
                continue

        # Edge deduplication: multiple layers mapping to the same collapsed
        # module node would produce duplicate edges without this check.
        if (tail_name, head_name) in edges_used:
            continue
        edges_used.add((tail_name, head_name))

        edge_dict = {
            "tail_name": tail_name,
            "head_name": head_name,
            "color": node_color,
            "fontcolor": node_color,
            "style": edge_style,
            "arrowsize": ".7",
            "labelfontsize": "8",
        }

        # Mark with "if" in the case edge starts a cond branch
        cond_children = parent_node.cond_branch_start_children
        if vis_opt == "rolled":
            cond_children = [label.split(":")[0] for label in cond_children]
        if (child_layer_label in cond_children) and (not child_is_collapsed_module):
            edge_dict["label"] = '<<FONT POINT-SIZE="18"><b><u>IF</u></b></FONT>>'

        # Label the arguments to the next node if multiple inputs
        if not child_is_collapsed_module:
            _label_node_arguments_if_needed(
                self, parent_node, child_node, edge_dict, show_buffer_layers
            )

        # Annotate passes for rolled node edge if it varies across passes
        if vis_opt == "rolled":
            _label_rolled_pass_nums(child_node, parent_node, edge_dict)  # type: ignore[arg-type]

        for arg_name, arg_val in overrides.edge.items():  # type: ignore[union-attr]
            if callable(arg_val):
                edge_dict[arg_name] = str(arg_val(self, parent_node, child_node))
            else:
                edge_dict[arg_name] = str(arg_val)

        # Add it to the appropriate module cluster (most nested one containing both nodes)
        containing_module = _get_lowest_containing_module_for_two_nodes(
            parent_node, child_node, both_nodes_collapsed_modules, vis_nesting_depth
        )
        if containing_module != -1:
            module_edge_dict[containing_module]["edges"].append(edge_dict)
            if parent_node.has_input_ancestor or child_node.has_input_ancestor:
                module_edge_dict[containing_module]["has_input_ancestor"] = True
                for module in parent_node.containing_modules_origin_nested:
                    module_key = module.split(":")[0] if vis_opt == "rolled" else module
                    module_edge_dict[module_key]["has_input_ancestor"] = True
                    if module_key == containing_module:
                        break
                for module in child_node.containing_modules_origin_nested:
                    module_key = module.split(":")[0] if vis_opt == "rolled" else module
                    module_edge_dict[module_key]["has_input_ancestor"] = True
                    if module_key == containing_module:
                        break
        else:
            graphviz_graph.edge(**edge_dict)

        # Finally, add a backwards edge if both tensors have stored gradients.
        if vis_opt == "unrolled":
            _add_gradient_edge(
                self,
                parent_node,
                child_node,
                edge_style,
                containing_module,
                module_edge_dict,
                graphviz_graph,
                overrides,  # type: ignore[arg-type]
            )


def _label_node_arguments_if_needed(
    self: "ModelLog",
    parent_node: Union["LayerPassLog", "LayerLog"],
    child_node: Union["LayerPassLog", "LayerLog"],
    edge_dict: Dict,
    show_buffer_layers: bool = False,
) -> None:
    """Add argument position labels to an edge when the child has multiple non-commutative parents.

    For nodes like ``sub(a, b)`` where argument order matters, labels like
    ``"arg 0"`` / ``"arg 1"`` are added to distinguish which parent feeds
    which argument.

    Note on substring false-positive risk: the lookup ``parent_node.layer_label == arg_label``
    uses exact equality, so substring matching is not an issue here.  However, the
    ``parent_layer_arg_locs`` keys are positional and the check iterates all of them,
    so a parent appearing in multiple arg positions will get multiple labels joined
    with ``<br/>``.

    Args:
        parent_node: The parent node whose edge is being labeled.
        child_node: The child node receiving the edge.
        edge_dict: Mutable dict of edge attributes; ``"label"`` may be added/appended.
        show_buffer_layers: Whether buffer layers are visible (affects parent count).
    """
    if not _should_mark_arguments_on_edge(self, child_node, show_buffer_layers):
        return

    arg_labels = []
    for arg_type in ["args", "kwargs"]:
        for arg_loc, arg_label in child_node.parent_layer_arg_locs[arg_type].items():
            if parent_node.layer_label == arg_label:
                arg_labels.append(f"{arg_type[:-1]} {str(arg_loc)}")

    arg_labels = "<br/>".join(arg_labels)  # type: ignore[assignment]
    if not arg_labels:
        return
    arg_label = f"<<FONT POINT-SIZE='10'><b>{arg_labels}</b></FONT>>"
    if "label" not in edge_dict:
        edge_dict["label"] = arg_label
    else:
        edge_dict["label"] = edge_dict["label"][:-1] + "<br/>" + arg_label[1:]


def _should_mark_arguments_on_edge(
    self: "ModelLog",
    child_node: Union["LayerPassLog", "LayerLog"],
    show_buffer_layers: bool = False,
) -> bool:
    """Returns True if argument position labels should be shown on the edge to child_node.

    Skips commutative functions (add, mul, cat, eq, ne) where arg order is
    interchangeable -- showing "arg 0" vs "arg 1" would be misleading.
    For non-commutative ops, labels are shown when the child has multiple
    visible parents.

    Args:
        child_node: The child node whose incoming edge is being considered.
        show_buffer_layers: Whether buffer layers are shown in the graph.
    """
    # Commutative ops: argument order doesn't matter, skip labels.
    if child_node.layer_type in COMMUTE_FUNCS:
        return False

    if isinstance(child_node, LayerPassLog):
        return _should_mark_arguments_on_unrolled_edge(self, child_node, show_buffer_layers)
    elif isinstance(child_node, LayerLog):
        return _should_mark_arguments_on_rolled_edge(self, child_node, show_buffer_layers)


def _should_mark_arguments_on_unrolled_edge(
    self, child_node: "LayerPassLog", show_buffer_layers: bool = False
) -> bool:
    """Returns True if argument labels should be shown on an unrolled graph edge.

    Args:
        child_node: The child LayerPassLog node whose incoming edge is being considered.
        show_buffer_layers: Whether buffer layers are shown in the graph.
    """
    num_parents_shown = len(child_node.parent_layers)

    if not show_buffer_layers:
        num_parents_shown -= sum(
            [int(self[parent].is_buffer_layer) for parent in child_node.parent_layers]
        )

    if num_parents_shown > 1:
        return True
    else:
        return False


def _should_mark_arguments_on_rolled_edge(
    self: "ModelLog", child_node: "LayerLog", show_buffer_layers: bool = False
) -> bool:
    """Returns True if argument labels should be shown on a rolled graph edge.

    Args:
        child_node: The child LayerLog node whose incoming edge is being considered.
        show_buffer_layers: Whether buffer layers are shown in the graph.
    """
    for pass_num, pass_parents in child_node.parent_layers_per_pass.items():
        num_parents_shown = len(pass_parents)
        if not show_buffer_layers:
            num_parents_shown -= sum(
                [int(self.layer_logs[parent].is_buffer_layer) for parent in pass_parents]
            )
        if num_parents_shown > 1:
            return True

    return False


def _label_rolled_pass_nums(
    child_node: "LayerLog",
    parent_node: "LayerLog",
    edge_dict: Dict,
) -> None:
    """Add pass-number annotations to edges in rolled mode.

    In rolled mode, a single edge may represent connections from different
    passes.  When edges vary across passes (``edges_vary_across_passes``),
    tail and head labels show which passes the edge applies to, e.g.,
    ``"Out 1,3"`` / ``"In 2,4"``.  Uses ``int_list_to_compact_str`` for
    concise range notation (e.g., ``"1-3"`` instead of ``"1,2,3"``).

    Args:
        child_node: The child LayerLog node.
        parent_node: The parent LayerLog node.
        edge_dict: Mutable dict of edge attributes; taillabel/headlabel may be added.
    """
    parent_pass_nums = parent_node.child_passes_per_layer[child_node.layer_label]
    child_pass_nums = child_node.parent_passes_per_layer[parent_node.layer_label]
    if parent_node.edges_vary_across_passes:
        edge_dict["taillabel"] = f"  Out {int_list_to_compact_str(parent_pass_nums)}  "

    # Mark the head label with the argument if need be:
    if child_node.edges_vary_across_passes:
        edge_dict["headlabel"] = f"  In {int_list_to_compact_str(child_pass_nums)}  "


def _get_lowest_containing_module_for_two_nodes(
    node1: Union["LayerPassLog", "LayerLog"],
    node2: Union["LayerPassLog", "LayerLog"],
    both_nodes_collapsed_modules: bool,
    vis_nesting_depth: int,
) -> Union[str, int]:
    """Find the deepest module subgraph that contains both nodes.

    Used to place an edge into the correct Graphviz cluster (subgraph).
    Edges between nodes in the same module cluster are drawn inside that
    cluster; edges crossing module boundaries are drawn at the level of
    the lowest common ancestor module.

    Returns -1 when no module contains both nodes (the edge belongs to the
    top-level graph, not any subgraph).

    Special handling:
    - ``is_bottom_level_submodule_output`` nodes are adjusted to their parent
      scope (they represent the module's output, rendered one level up).
    - Rolled mode: pass suffixes are stripped from module names so that all
      passes share the same cluster.
    - Both-collapsed case: when both nodes are collapsed module boxes, the
      containing module must be at least one level above the collapse depth.

    Args:
        node1: The first node.
        node2: The second node.
        both_nodes_collapsed_modules: Whether both nodes are collapsed module boxes.
        vis_nesting_depth: How many levels deep to visualize.

    Returns:
        Module name (str) for the containing cluster, or -1 for top-level.
    """
    node1_modules = node1.containing_modules_origin_nested[:]
    node2_modules = node2.containing_modules_origin_nested[:]

    if isinstance(node1, LayerLog) or isinstance(node2, LayerLog):
        node1_modules = [module.split(":")[0] for module in node1_modules]
        node2_modules = [module.split(":")[0] for module in node2_modules]

    if node1.is_bottom_level_submodule_output:
        node1_nested_modules = node1_modules[:-1]
    else:
        node1_nested_modules = node1_modules[:]

    if (
        (len(node1_modules) == 0)
        or (len(node2_modules) == 0)
        or (node1_modules[0] != node2_modules[0])
    ):
        return -1  # no submodule contains them both.

    if node1 == node2:
        if node1.is_bottom_level_submodule_output and (len(node1_modules) == 1):
            return -1
        elif node1.is_bottom_level_submodule_output and (len(node1_modules) > 1):
            containing_module = node1_modules[-2]
        else:
            containing_module = node1_modules[-1]
        return containing_module

    if both_nodes_collapsed_modules:
        if (vis_nesting_depth == 1) or (len(node1_nested_modules) == 1):
            return -1
        if node1_modules[vis_nesting_depth - 1] == node2_modules[vis_nesting_depth - 1]:
            containing_module = node1_modules[vis_nesting_depth - 2]
            return containing_module

    containing_module = node1_modules[0]
    for m in range(min(len(node1_modules), len(node2_modules))):
        if node1_modules[m] != node2_modules[m]:
            break
        containing_module = node1_modules[m]

    return containing_module


def _add_gradient_edge(
    self: "ModelLog",
    parent_layer,
    child_layer,
    edge_style,
    containing_module,
    module_edge_dict,
    graphviz_graph,
    overrides: VisualizationOverrides,
) -> None:
    """Add a backward (gradient) edge if both layers have saved gradients.

    Gradient edges flow child -> parent (opposite of data flow), drawn in
    ``GRADIENT_ARROW_COLOR`` to distinguish from forward edges.  Only added
    in unrolled mode (rolled mode doesn't show gradients).

    Args:
        parent_layer: The parent LayerPassLog (gradient destination).
        child_layer: The child LayerPassLog (gradient source).
        edge_style: ``'solid'`` or ``'dashed'`` (matches the forward edge style).
        containing_module: Module cluster name, or -1 for top-level.
        module_edge_dict: Dict mapping each module cluster to its edges.
        graphviz_graph: The graphviz Digraph object.
        overrides: Graphviz attribute overrides for gradient edges.
    """
    if parent_layer.has_saved_grad and child_layer.has_saved_grad:
        edge_dict = {
            "tail_name": child_layer.layer_label.replace(":", "pass"),
            "head_name": parent_layer.layer_label.replace(":", "pass"),
            "color": GRADIENT_ARROW_COLOR,
            "fontcolor": GRADIENT_ARROW_COLOR,
            "style": edge_style,
            "arrowsize": ".7",
            "labelfontsize": "8",
        }
        for arg_name, arg_val in overrides.gradient_edge.items():  # type: ignore[union-attr]
            if callable(arg_val):
                edge_dict[arg_name] = str(arg_val(self, parent_layer, child_layer))
            else:
                edge_dict[arg_name] = str(arg_val)

        if containing_module != -1:
            module_edge_dict[containing_module]["edges"].append(edge_dict)
        else:
            graphviz_graph.edge(**edge_dict)


def _setup_subgraphs(
    self: "ModelLog",
    graphviz_graph,
    vis_opt: str,
    module_edge_dict: Dict,
    overrides: Optional[VisualizationOverrides] = None,
) -> None:
    """Build nested Graphviz subgraphs for module clusters.

    Creates the module hierarchy as nested Graphviz subgraphs (clusters),
    placing edges into the appropriate depth level.  Uses a BFS-like
    approach: starts from top-level modules, builds each subgraph via
    ``_setup_subgraphs_recurse``, and pushes child modules onto a stack.

    In **unrolled** mode, each module pass is a separate subgraph (keyed by
    ``"module_addr:pass_num"``).  In **rolled** mode, all passes share one
    subgraph (keyed by ``"module_addr"``).

    Subgraph names are prefixed with ``"cluster_"`` (Graphviz convention to
    draw a border box around them).

    Args:
        graphviz_graph: The top-level Graphviz Digraph.
        vis_opt: ``'rolled'`` or ``'unrolled'``.
        module_edge_dict: Dict mapping each module cluster name to
            ``{"edges": [...], "has_input_ancestor": bool}``.
        overrides: Graphviz attribute overrides for module subgraphs.
    """
    if vis_opt == "unrolled":
        module_submodule_dict = defaultdict(list)
        for pass_label, mpl in self.modules._pass_dict.items():
            module_submodule_dict[pass_label] = list(mpl.call_children)
        subgraphs = list(self.modules["self"].passes[1].call_children)  # type: ignore[union-attr]
    else:
        module_submodule_dict = defaultdict(list)
        for ml in self.modules:
            if ml.address != "self":
                module_submodule_dict[ml.address] = list(ml.call_children)
        subgraphs = list(self.modules["self"].call_children)

    # Get the max module nesting depth:

    max_nesting_depth = _get_max_nesting_depth(subgraphs, module_edge_dict, module_submodule_dict)

    subgraph_stack = [[subgraph] for subgraph in subgraphs]
    nesting_depth = 0
    while len(subgraph_stack) > 0:
        parent_graph_list = subgraph_stack.pop(0)
        _setup_subgraphs_recurse(
            self,
            graphviz_graph,
            parent_graph_list,
            module_edge_dict,
            module_submodule_dict,
            subgraph_stack,
            nesting_depth,
            max_nesting_depth,
            vis_opt,
            overrides,  # type: ignore[arg-type]
        )


def _setup_subgraphs_recurse(
    self: "ModelLog",
    starting_subgraph,
    parent_graph_list: List,
    module_edge_dict,
    module_submodule_dict,
    subgraph_stack,
    nesting_depth,
    max_nesting_depth,
    vis_opt,
    overrides: VisualizationOverrides,
) -> None:
    """Recursively build a single branch of the module subgraph hierarchy.

    Walks down ``parent_graph_list`` (a path from root to leaf module),
    creating nested Graphviz context managers at each level.  When the
    leaf is reached, adds all accumulated edges and pushes child modules
    onto ``subgraph_stack`` for later processing.

    Module border width scales inversely with nesting depth (deeper modules
    get thinner borders) to provide visual hierarchy.

    Args:
        starting_subgraph: The parent Graphviz subgraph to nest into.
        parent_graph_list: Path of module names from root to current target.
        module_edge_dict: Dict mapping each cluster to its edges.
        module_submodule_dict: Dict mapping each cluster to its subclusters.
        subgraph_stack: BFS work queue for remaining branches.
        nesting_depth: Current position in ``parent_graph_list``.
        max_nesting_depth: Maximum depth across all branches (for penwidth scaling).
        vis_opt: ``'rolled'`` or ``'unrolled'``.
        overrides: Graphviz attribute overrides.
    """
    subgraph_name_w_pass = parent_graph_list[nesting_depth]
    subgraph_module = subgraph_name_w_pass.split(":")[0]
    if vis_opt == "unrolled":
        cluster_name = f"cluster_{subgraph_name_w_pass.replace(':', '_pass')}"
        subgraph_name = subgraph_name_w_pass
    elif vis_opt == "rolled":
        cluster_name = f"cluster_{subgraph_module}"
        subgraph_name = subgraph_module
    else:
        raise ValueError("vis_opt must be 'rolled' or 'unrolled'")
    sg_ml = self.modules[subgraph_module]
    module_type = sg_ml.module_class_name  # type: ignore[union-attr]
    if (sg_ml.num_passes > 1) and (vis_opt == "unrolled"):  # type: ignore[union-attr]
        subgraph_title = subgraph_name_w_pass
    elif (sg_ml.num_passes > 1) and (vis_opt == "rolled"):  # type: ignore[union-attr]
        subgraph_title = f"{subgraph_module} (x{sg_ml.num_passes})"  # type: ignore[union-attr]
    else:
        subgraph_title = subgraph_module

    if (
        nesting_depth < len(parent_graph_list) - 1
    ):  # we haven't gotten to the bottom yet, keep going.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            _setup_subgraphs_recurse(
                self,
                s,
                parent_graph_list,
                module_edge_dict,
                module_submodule_dict,
                subgraph_stack,
                nesting_depth + 1,
                max_nesting_depth,
                vis_opt,
                overrides,
            )

    else:  # Leaf of this branch: create the subgraph and add all edges.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            # Penwidth scales with depth: outermost modules get thickest
            # borders, deepest modules get thinnest.  Provides visual hierarchy.
            nesting_fraction = (max_nesting_depth - nesting_depth) / max_nesting_depth
            pen_width = MIN_MODULE_PENWIDTH + nesting_fraction * PENWIDTH_RANGE
            if module_edge_dict[subgraph_name]["has_input_ancestor"]:
                line_style = "solid"
            else:
                line_style = "dashed"

            module_args = {
                "label": f"<<B>@{subgraph_title}</B><br align='left'/>({module_type})<br align='left'/>>",
                "labelloc": "b",
                "style": f"filled,{line_style}",
                "fillcolor": "white",
                "penwidth": str(pen_width),
            }

            for arg_name, arg_val in overrides.module.items():  # type: ignore[union-attr]
                if callable(arg_val):
                    module_args[arg_name] = str(arg_val(self, subgraph_name))
                else:
                    module_args[arg_name] = str(arg_val)
            s.attr(**module_args)
            subgraph_edges = module_edge_dict[subgraph_name]["edges"]
            for edge_dict in subgraph_edges:
                s.edge(**edge_dict)
            subgraph_children = module_submodule_dict[subgraph_name_w_pass]
            for subgraph_child in subgraph_children:  # it's weird but have to go in reverse order.
                subgraph_stack.append(parent_graph_list[:] + [subgraph_child])


def _get_max_nesting_depth(top_modules, module_edge_dict, module_submodule_dict) -> int:
    """Recursively computes the maximum module nesting depth in the model hierarchy.

    Used to determine subgraph layout depth for graphviz rendering. Works by
    crawling down the stack of modules till it hits one with no children and at least one edge.

    Args:
        top_modules: modules at highest level of nesting
        module_edge_dict: Edges in each module.
        module_submodule_dict: Mapping from each module to any children.

    Returns:
        Max nesting depth.
    """
    max_nesting_depth = 1
    module_stack = [(graph, 1) for graph in top_modules]

    while len(module_stack) > 0:
        module, module_depth = module_stack.pop()
        module_edges = module_edge_dict[module]["edges"]
        module_submodules = module_submodule_dict[module]

        if (len(module_edges) == 0) and (
            len(module_submodules) == 0
        ):  # can ignore if no edges and no children.
            continue
        elif (len(module_edges) > 0) and (len(module_submodules) == 0):
            max_nesting_depth = max([module_depth, max_nesting_depth])
        elif (len(module_edges) == 0) and (len(module_submodules) > 0):
            module_stack.extend(
                [(module_child, module_depth + 1) for module_child in module_submodules]
            )
        else:
            max_nesting_depth = max([module_depth, max_nesting_depth])
            module_stack.extend(
                [(module_child, module_depth + 1) for module_child in module_submodules]
            )
    return max_nesting_depth

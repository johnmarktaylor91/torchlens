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

import copy
import subprocess
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import graphviz
import torch

from .._literals import (
    BufferVisibilityLiteral,
    VisDirectionLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from ..data_classes.internal_types import VisualizationOverrides
from ..utils.display import in_notebook, int_list_to_compact_str, _vprint
from ..data_classes.layer_pass_log import LayerPassLog
from ..data_classes.layer_log import LayerLog
from .modes import COLLAPSED_MODE_REGISTRY, MODE_REGISTRY
from .node_spec import NodeSpec, render_lines_to_html
from .code_panel import CodePanelOption, render_code_panel_subgraph, resolve_code_panel_source
from ._render_utils import (
    compute_module_penwidth,
    direction_to_rankdir,
    make_module_cluster_attrs,
)

if TYPE_CHECKING:
    from ..data_classes.grad_fn_log import GradFnLog
    from ..data_classes.model_log import ModelLog
    from ..data_classes.module_log import ModuleLog

BaseGraphNode = Union["LayerPassLog", "LayerLog"]


@dataclass
class FocusNode:
    """Mutable render proxy for a focused LayerLog or LayerPassLog.

    Parameters
    ----------
    original:
        Source graph node whose metadata should be rendered.
    parent_layers:
        Focus-rewritten incoming labels.
    child_layers:
        Focus-rewritten outgoing labels.
    containing_modules:
        Copied module path for cluster placement.
    """

    original: BaseGraphNode
    parent_layers: list[str]
    child_layers: list[str]
    containing_modules: list[str]

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the source node."""

        return getattr(self.original, name)


@dataclass
class BoundaryNode:
    """Synthetic node representing a focused module boundary.

    Parameters
    ----------
    layer_label:
        DOT-safe synthetic label.
    display_label:
        Human-readable label shown in the node.
    boundary_kind:
        ``"input"`` for external upstreams, ``"output"`` for external sinks.
    child_layers:
        Outgoing rendered labels.
    parent_layers:
        Incoming rendered labels.
    containing_modules:
        Module path used for cluster placement.
    """

    layer_label: str
    display_label: str
    boundary_kind: str
    child_layers: list[str]
    parent_layers: list[str]
    containing_modules: list[str]
    is_buffer_layer: bool = False
    has_input_ancestor: bool = True
    is_final_output: bool = False
    is_leaf_module_output: bool = False
    modules_exited: list[str] = field(default_factory=list)
    is_input_layer: bool = False
    is_output_layer: bool = False
    is_terminal_bool_layer: bool = False
    uses_params: bool = False
    num_param_tensors: int = 0
    parent_param_logs: list[Any] = field(default_factory=list)
    parent_param_shapes: list[tuple[Any, ...]] = field(default_factory=list)
    num_passes: int = 1
    pass_num: int = 1
    layer_type_num: int = 1
    layer_total_num: int = 1
    tensor_shape: tuple[Any, ...] = ()
    tensor_memory_str: str = "0 B"
    io_role: str = ""
    layer_type: str = "input"

    def __post_init__(self) -> None:
        """Fill mutable defaults and role flags."""

        self.is_input_layer = self.boundary_kind == "input"
        self.is_output_layer = self.boundary_kind == "output"
        self.layer_type = self.boundary_kind
        self.io_role = self.boundary_kind


GraphNode = Union[BaseGraphNode, BoundaryNode, FocusNode]
NodeSpecFn = Callable[["LayerLog", NodeSpec], NodeSpec | None]
BackwardNodeSpecFn = Callable[["GradFnLog", NodeSpec], NodeSpec | None]
CollapsedNodeSpecFn = Callable[["ModuleLog", NodeSpec], NodeSpec | None]
CollapseFn = Callable[["ModuleLog"], bool]
SkipFn = Callable[["LayerLog"], bool]

# -- Color palette for node types --
INPUT_COLOR = "#98FB98"  # Light green
OUTPUT_COLOR = "#ff9999"  # Light red/salmon
PARAMS_NODE_BG_COLOR = "#E6E6E6"  # Generic param (no ParamLog available)
TRAINABLE_PARAMS_BG_COLOR = "#D9D9D9"  # Light gray for trainable params
FROZEN_PARAMS_BG_COLOR = "#B0B0B0"  # Darker gray for frozen params
GRADIENT_ARROW_COLOR = "#9197F6"  # Light blue/purple for backward edges
BACKWARD_NODE_COLOR = "#F2F3FF"  # Very light blue/purple for backward grad_fn nodes
BACKWARD_NODE_BORDER_COLOR = GRADIENT_ARROW_COLOR
DEFAULT_BG_COLOR = "white"
BOOL_NODE_COLOR = "#F7D460"  # Yellow for terminal boolean layers
_NOISE_BUFFER_NAMES = frozenset({"running_mean", "running_var", "num_batches_tracked"})

# Module subgraph border widths live in ._render_utils -- both this file
# and ``multi_trace/visualization.py`` use ``compute_module_penwidth`` so
# bundle and ModelLog clusters scale identically by depth.

# Commutative functions: argument order doesn't matter, so we skip arg-position
# labels on their incoming edges to reduce visual clutter.
COMMUTE_FUNCS = ["add", "mul", "cat", "eq", "ne"]


def _normalize_buffer_visibility(
    show_buffer_layers: BufferVisibilityLiteral | bool,
) -> BufferVisibilityLiteral:
    """Normalize buffer visibility values accepted by the render path.

    Parameters
    ----------
    show_buffer_layers:
        Tri-state buffer visibility mode or legacy bool.

    Returns
    -------
    BufferVisibilityLiteral
        Canonical tri-state mode.

    Raises
    ------
    ValueError
        If ``show_buffer_layers`` is not supported.
    """

    if show_buffer_layers is True:
        return "always"
    if show_buffer_layers is False:
        return "never"
    if show_buffer_layers in {"never", "meaningful", "always"}:
        return show_buffer_layers
    raise ValueError("show_buffer_layers must be 'never', 'meaningful', 'always', or a bool.")


def _buffer_name_segment(buffer_address: str | None) -> str:
    """Return the last dotted segment of a buffer address.

    Parameters
    ----------
    buffer_address:
        Fully qualified buffer address, if available.

    Returns
    -------
    str
        Final dotted address segment, or an empty string for missing addresses.
    """

    if buffer_address is None:
        return ""
    return buffer_address.split(".")[-1]


def _is_noise_buffer(node: GraphNode) -> bool:
    """Return whether ``node`` is a hardcoded noisy buffer.

    Parameters
    ----------
    node:
        Candidate graph node.

    Returns
    -------
    bool
        True when the node is a buffer whose last address segment is filtered in
        ``"meaningful"`` mode.
    """

    source_node = _unwrap_focus_node(node)
    if not source_node.is_buffer_layer:
        return False
    buffer_address = getattr(source_node, "buffer_address", None)
    return _buffer_name_segment(buffer_address) in _NOISE_BUFFER_NAMES


def _is_buffer_visible(node: GraphNode, show_buffer_layers: BufferVisibilityLiteral) -> bool:
    """Return whether a buffer node should be visible in the current mode.

    Parameters
    ----------
    node:
        Candidate graph node.
    show_buffer_layers:
        Canonical tri-state visibility mode.

    Returns
    -------
    bool
        True when the node is visible. Non-buffer nodes are always visible.
    """

    if not node.is_buffer_layer:
        return True
    if show_buffer_layers == "always":
        return True
    if show_buffer_layers == "never":
        return False
    return not _is_noise_buffer(node)


def _get_hidden_parent_buffer_addresses(
    model_log: "ModelLog",
    node: GraphNode,
    show_buffer_layers: BufferVisibilityLiteral,
) -> list[str]:
    """Return hidden buffer addresses attached as parents of ``node``.

    Parameters
    ----------
    model_log:
        Owning ModelLog.
    node:
        Non-buffer node to inspect.
    show_buffer_layers:
        Canonical tri-state visibility mode.

    Returns
    -------
    list[str]
        Hidden buffer addresses in parent order, de-duplicated.
    """

    if show_buffer_layers == "always" or node.is_buffer_layer:
        return []

    hidden_addresses: list[str] = []
    seen_addresses: set[str] = set()
    source_node = _unwrap_focus_node(node)
    for parent_label in node.parent_layers:
        if parent_label.startswith("__module_focus_"):
            continue
        parent_node: BaseGraphNode
        if isinstance(source_node, LayerPassLog):
            parent_node = model_log[parent_label]
        else:
            parent_node = model_log.layer_logs[parent_label]
        if not parent_node.is_buffer_layer or _is_buffer_visible(parent_node, show_buffer_layers):
            continue
        buffer_address = parent_node.buffer_address
        if buffer_address is None or buffer_address in seen_addresses:
            continue
        hidden_addresses.append(buffer_address)
        seen_addresses.add(buffer_address)
    return hidden_addresses


@dataclass(frozen=True)
class RenderEdge:
    """Skip-aware edge between two rendered graph nodes.

    Attributes
    ----------
    target:
        Non-skipped edge target.
    metadata_child:
        Original first child edge to use for labels and override callbacks. ``None``
        means multiple skipped paths disagreed, so optional labels are dropped.
    """

    target: GraphNode
    metadata_child: Optional[GraphNode]


def render_graph(
    self: "ModelLog",
    vis_mode: VisModeLiteral = "unrolled",
    vis_nesting_depth: int = 1000,
    vis_outpath: str = "modelgraph",
    vis_graph_overrides: Optional[Dict] = None,
    module: "ModuleLog | str | None" = None,
    node_mode: VisNodeModeLiteral = "default",
    node_spec_fn: NodeSpecFn | None = None,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    collapse_fn: CollapseFn | None = None,
    skip_fn: SkipFn | None = None,
    vis_edge_overrides: Optional[Dict] = None,
    vis_gradient_edge_overrides: Optional[Dict] = None,
    vis_module_overrides: Optional[Dict] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
    direction: VisDirectionLiteral = "bottomup",
    vis_node_placement: VisNodePlacementLiteral = "auto",
    vis_renderer: VisRendererLiteral = "graphviz",
    vis_theme: str = "torchlens",
    code_panel: CodePanelOption = False,
) -> str:
    """Render the computational graph as a Graphviz Digraph.

    Orchestrates the full rendering pipeline:
    1. Validates that all layers are logged (``_all_layers_logged`` guard).
    2. Iterates over entries_to_plot, building nodes and edges.
    3. Groups edges into module subgraph clusters.
    4. Renders to file and optionally displays.

    Args:
        vis_mode: ``'unrolled'`` (each pass is a separate node) or ``'rolled'``
            (multi-pass layers collapsed into one node with pass annotations).
        vis_nesting_depth: Maximum module nesting levels to show before
            collapsing deeper layers into ``box3d`` module summary nodes.
            Use 0 to show all layers without collapsing.
        vis_outpath: Output file path (extension auto-stripped).
        vis_graph_overrides: Graphviz graph-level attribute overrides.
        module: Optional module focus. A ModuleLog focuses that module; a string
            is interpreted as a module address.
        node_mode: Preset applied to default ``NodeSpec`` objects before
            user callbacks run.
        node_spec_fn: Optional callback receiving ``(layer_log, default_spec)``.
            In unrolled mode, ``layer_log`` is the parent aggregate LayerLog for
            the rendered LayerPassLog.
        collapsed_node_spec_fn: Optional callback receiving
            ``(module_log, default_spec)`` for collapsed module nodes.
        collapse_fn: Optional predicate receiving a ModuleLog. When provided,
            it replaces ``vis_nesting_depth`` collapse decisions.
        skip_fn: Optional predicate receiving a LayerLog. Skipped nodes are
            elided and edges are chained through them.
        vis_edge_overrides: Overrides for forward edges.
        vis_gradient_edge_overrides: Overrides for backward (gradient) edges.
        vis_module_overrides: Overrides for module subgraph boxes.
        vis_save_only: If True, save without opening a viewer.
        vis_fileformat: Output format (pdf, png, svg, etc.).
        show_buffer_layers: Buffer visibility mode. ``"never"`` hides all
            buffers, ``"meaningful"`` hides hardcoded BatchNorm running-stat
            noise buffers, and ``"always"`` shows all buffers. Legacy bools are
            deprecated but supported: ``True`` maps to ``"always"`` and
            ``False`` maps to ``"never"``.
        direction: Layout direction: ``'bottomup'``, ``'topdown'``, or ``'leftright'``.
        vis_node_placement: Layout engine: ``'auto'`` (default), ``'dot'``, ``'elk'``,
            or ``'sfdp'``.  ``'auto'`` uses dot for small graphs and ELK (or sfdp
            fallback) for large ones.
        code_panel: Optional source-code panel. ``True`` is equivalent to
            ``"forward"``; callable values receive the live model object when
            it is still available.

    Returns:
        The Graphviz DOT source string.

    Raises:
        ValueError: If ``_all_layers_logged`` is False (layers were discarded
            by ``keep_unsaved_layers=False``).
    """
    if node_mode not in MODE_REGISTRY:
        raise ValueError(
            "Visualization node_mode must be one of 'default', 'profiling', "
            "'vision', or 'attention'."
        )
    show_buffer_layers = _normalize_buffer_visibility(show_buffer_layers)

    if vis_renderer == "dagua":
        from .dagua_bridge import render_model_log_with_dagua

        return render_model_log_with_dagua(
            self,
            vis_mode=vis_mode,
            vis_nesting_depth=vis_nesting_depth,
            vis_outpath=vis_outpath,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            vis_buffer_layers=show_buffer_layers == "always",
            vis_direction=direction,
            vis_theme=vis_theme,
        )
    if vis_renderer not in {"graphviz", "dagua"}:
        raise ValueError("vis_renderer must be 'graphviz' or 'dagua'")

    overrides = VisualizationOverrides(
        graph=vis_graph_overrides or {},
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
    if vis_mode == "unrolled":
        entries_to_plot: dict[str, GraphNode] = dict(self.layer_dict_main_keys)
    elif vis_mode == "rolled":
        entries_to_plot = dict(self.layer_logs)
    else:
        raise ValueError("vis_mode must be either 'rolled' or 'unrolled'")

    if module is not None:
        target_module = _resolve_focus_module(self, module)
        entries_to_plot = _build_module_focus_entries(
            self,
            entries_to_plot,
            target_module,
            vis_mode=vis_mode,
        )

    rankdir = direction_to_rankdir(direction)

    # Resolve the layout engine early to potentially skip graphviz.Digraph construction.
    from .elk_layout import get_node_placement_engine

    edge_map, skipped_labels = _build_skip_filtered_edge_map(
        self,
        entries_to_plot,
        vis_mode=vis_mode,
        show_buffer_layers=show_buffer_layers,
        skip_fn=skip_fn,
    )
    source_text = resolve_code_panel_source(
        code_panel,
        getattr(self, "_source_code_blob", {}),
        getattr(self, "_source_model_ref", None),
    )
    num_nodes = len(entries_to_plot) - len(skipped_labels)
    engine = get_node_placement_engine(vis_node_placement, num_nodes)
    if source_text is not None and engine == "elk":
        # The code panel is implemented in pure Graphviz so the graph and source
        # remain in one output file. ELK's direct renderer bypasses Digraph
        # construction, so panel renders stay on the Graphviz path.
        engine = "dot"
    _vprint(self, f"Rendering {vis_mode} graph ({num_nodes} nodes, format={vis_fileformat})")
    _vprint(self, f"Layout engine: {engine}")

    if self.total_params == 0:
        params_detail = "0 params"
    elif self.total_params_frozen == 0:
        params_detail = (
            f"{self.total_params} params (all trainable, {self.total_params_memory_str})"
        )
    elif self.total_params_trainable == 0:
        params_detail = f"{self.total_params} params (all frozen, {self.total_params_memory_str})"
    else:
        params_detail = (
            f"{self.total_params} params "
            f"({self.total_params_trainable}/{self.total_params} trainable, "
            f"{self.total_params_memory_str})"
        )

    graph_caption = (
        f"<<B>{self.model_name}</B><br align='left'/>{self.num_tensors_total} "
        f"tensors total ({self.total_activation_memory_str})"
        f"<br align='left'/>{params_detail}<br align='left'/>>"
    )

    # ELK fast path: skip graphviz.Digraph construction entirely.
    # Generates DOT directly with ELK positions and cluster subgraphs (module boxes).
    # If ELK layout fails (OOM, timeout), render_elk_direct falls back internally
    # to sfdp — still using the fast DOT-text path, never graphviz.Digraph.
    if engine == "elk":
        from .elk_layout import render_elk_direct

        result = render_elk_direct(
            self,
            entries_to_plot,
            vis_mode,
            vis_nesting_depth,
            show_buffer_layers == "always",
            overrides,
            node_mode,
            node_spec_fn,
            collapsed_node_spec_fn,
            collapse_fn,
            skip_fn,
            edge_map,
            skipped_labels,
            vis_outpath,
            vis_fileformat,
            vis_save_only,
            graph_caption,
            rankdir,
        )
        _vprint(self, f"Graph saved to {vis_outpath}.{vis_fileformat}")
        return result

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
        if node.layer_label in skipped_labels:
            continue
        if node.is_buffer_layer and not _is_buffer_visible(node, show_buffer_layers):
            continue
        _add_node_to_graphviz(
            self,
            node,
            dot,
            module_cluster_dict,
            edges_used,
            vis_mode,
            collapsed_modules,
            vis_nesting_depth,
            show_buffer_layers,
            overrides,
            node_mode,
            node_spec_fn,
            collapsed_node_spec_fn,
            collapse_fn,
            edge_map,
        )

    # Finally, set up the subgraphs.
    _setup_subgraphs(self, dot, vis_mode, module_cluster_dict, overrides)
    if source_text is not None:
        render_code_panel_subgraph(dot, source_text)

    if in_notebook() and not vis_save_only:
        from IPython.display import display  # #72: lazy import

        display(dot)

    # ELK was already handled above (early return). Only dot/sfdp reach here.
    from .elk_layout import render_with_sfdp

    _RENDER_TIMEOUT = 120  # seconds
    source_path = dot.save(vis_outpath)
    try:
        if engine == "sfdp":
            render_with_sfdp(source_path, vis_outpath, vis_fileformat, vis_save_only)
        else:
            # dot engine (default for small graphs)
            rendered_path = f"{vis_outpath}.{vis_fileformat}"
            cmd = [dot.engine, f"-T{vis_fileformat}", "-o", rendered_path, source_path]
            subprocess.run(cmd, timeout=_RENDER_TIMEOUT, check=True, capture_output=True)
            if not vis_save_only:
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


def render_backward_graph(
    self: "ModelLog",
    vis_outpath: str = "backward_modelgraph",
    vis_graph_overrides: Optional[Dict] = None,
    node_spec_fn: BackwardNodeSpecFn | None = None,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    vis_node_mode: VisNodeModeLiteral = "default",
    vis_edge_overrides: Optional[Dict] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    direction: VisDirectionLiteral = "topdown",
    code_panel: CodePanelOption = False,
) -> str:
    """Render the captured backward grad_fn DAG as a Graphviz graph.

    Intervening grad_fns use a ``[i]`` label prefix. Custom autograd grad_fns
    use a ``[custom]`` label suffix so the two cues compose on the same node.

    Parameters
    ----------
    self:
        ModelLog containing captured backward metadata.
    vis_outpath:
        Output path for the rendered graph.
    vis_graph_overrides:
        Graphviz graph-level overrides.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_log, default_spec)``.
    collapsed_node_spec_fn:
        Accepted for API symmetry with forward visualization. Not applied
        because backward graphs do not render collapsed module nodes.
    vis_node_mode:
        Accepted for API symmetry with forward visualization. Not applied to
        grad_fn nodes.
    vis_edge_overrides:
        Graphviz edge-level overrides.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output format.
    direction:
        Layout direction: ``'bottomup'``, ``'topdown'``, or ``'leftright'``.
    code_panel:
        Optional source-code panel mode.

    Returns
    -------
    str
        Graphviz DOT source.

    Raises
    ------
    ValueError
        If no explicit backward graph has been captured.
    """

    if not self.has_backward_log or not self.grad_fn_logs:
        raise ValueError("No backward graph is available; call log_backward(loss) first.")
    _ = collapsed_node_spec_fn, vis_node_mode

    rankdir = direction_to_rankdir(direction)

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

    graph_caption = (
        f"<<B>{self.model_name} backward graph</B><br align='left'/>"
        f"{self.num_grad_fns} grad_fn nodes"
        f"<br align='left'/>{self.backward_num_passes} backward pass(es)<br align='left'/>>"
    )
    dot = graphviz.Digraph(
        name=f"{self.model_name}_backward",
        comment="Backward grad_fn graph",
        format=vis_fileformat,
    )
    graph_args = {
        "rankdir": rankdir,
        "label": graph_caption,
        "labelloc": "t",
        "labeljust": "left",
        "ordering": "out",
    }
    for arg_name, arg_val in (vis_graph_overrides or {}).items():
        if callable(arg_val):
            graph_args[arg_name] = str(arg_val(self))
        else:
            graph_args[arg_name] = str(arg_val)

    edge_args = {"color": GRADIENT_ARROW_COLOR, "fontcolor": GRADIENT_ARROW_COLOR}
    for arg_name, arg_val in (vis_edge_overrides or {}).items():
        if callable(arg_val):
            edge_args[arg_name] = str(arg_val(self))
        else:
            edge_args[arg_name] = str(arg_val)

    dot.graph_attr.update(graph_args)
    dot.node_attr.update({"ordering": "out"})
    dot.edge_attr.update(edge_args)

    for grad_fn in self.grad_fns:
        _add_backward_node_to_graphviz(grad_fn, dot, node_spec_fn)

    visible_ids = set(self.grad_fn_logs)
    for grad_fn in self.grad_fns:
        tail_name = _backward_dot_node_name(grad_fn)
        for next_grad_fn_id in grad_fn.next_grad_fn_ids:
            if next_grad_fn_id not in visible_ids:
                continue
            head_name = _backward_dot_node_name(self.grad_fn_logs[next_grad_fn_id])
            dot.edge(tail_name, head_name)

    source_text = resolve_code_panel_source(
        code_panel,
        getattr(self, "_source_code_blob", {}),
        getattr(self, "_source_model_ref", None),
    )
    if source_text is not None:
        render_code_panel_subgraph(dot, source_text)

    if in_notebook() and not vis_save_only:
        from IPython.display import display

        display(dot)

    _RENDER_TIMEOUT = 120
    source_path = dot.save(vis_outpath)
    try:
        rendered_path = f"{vis_outpath}.{vis_fileformat}"
        cmd = [dot.engine, f"-T{vis_fileformat}", "-o", rendered_path, source_path]
        subprocess.run(cmd, timeout=_RENDER_TIMEOUT, check=True, capture_output=True)
        if not vis_save_only:
            graphviz.backend.viewing.view(rendered_path)
        _vprint(self, f"Backward graph saved to {vis_outpath}.{vis_fileformat}")
    except subprocess.TimeoutExpired:
        warnings.warn(
            f"Graphviz render timed out ({_RENDER_TIMEOUT}s) for backward graph with "
            f"{self.num_grad_fns} grad_fn nodes. DOT source saved to '{source_path}'."
        )
    except subprocess.CalledProcessError as e:
        warnings.warn(f"Graphviz render failed: {e.stderr.decode()}")
    finally:
        import os

        if os.path.exists(source_path):
            os.remove(source_path)
    return dot.source


def _backward_dot_node_name(grad_fn: "GradFnLog") -> str:
    """Return a DOT-safe node name for a grad_fn log.

    Parameters
    ----------
    grad_fn:
        GradFnLog to name.

    Returns
    -------
    str
        DOT-safe node identifier.
    """

    return f"grad_fn_{grad_fn.grad_fn_id}"


def _add_backward_node_to_graphviz(
    grad_fn: "GradFnLog",
    graphviz_graph,
    node_spec_fn: BackwardNodeSpecFn | None,
) -> None:
    """Add one backward grad_fn node to a Graphviz graph.

    Parameters
    ----------
    grad_fn:
        GradFnLog to render.
    graphviz_graph:
        Graphviz Digraph object.
    node_spec_fn:
        Optional callback receiving ``(grad_fn, default_spec)``.
    """

    default_spec = NodeSpec(
        lines=_compute_backward_node_lines(grad_fn),
        shape="oval",
        fillcolor=BACKWARD_NODE_COLOR,
        fontcolor="black",
        color=BACKWARD_NODE_BORDER_COLOR,
        style="filled,solid",
        penwidth=1.8,
        extra_attrs={"ordering": "out"},
    )
    if node_spec_fn is not None:
        result = node_spec_fn(grad_fn, default_spec)
        spec = default_spec if result is None else result
    else:
        spec = default_spec
    node_args = _node_spec_to_graphviz_args(spec)
    node_args["name"] = _backward_dot_node_name(grad_fn)
    graphviz_graph.node(**node_args)


def _compute_backward_node_lines(grad_fn: "GradFnLog") -> list[str]:
    """Build default label rows for a backward grad_fn node.

    Parameters
    ----------
    grad_fn:
        GradFnLog to render.

    Returns
    -------
    list[str]
        Plain-text rows for ``NodeSpec.lines``.
    """

    title = grad_fn.label
    if grad_fn.is_intervening:
        title = f"[i] {title}"
    if grad_fn.is_custom:
        title = f"{title} [custom]"

    lines = [title]
    if grad_fn.corresponding_layer is not None:
        lines.append(f"@{grad_fn.corresponding_layer.layer_label}")
    lines.append(f"grad {_format_backward_output_shape(grad_fn)}")
    return lines


def _format_backward_output_shape(grad_fn: "GradFnLog") -> str:
    """Return the first captured output-gradient shape for a grad_fn.

    Parameters
    ----------
    grad_fn:
        GradFnLog to inspect.

    Returns
    -------
    str
        Compact shape string, or ``"unknown"`` when no tensor was captured.
    """

    for grad_fn_pass in reversed(list(grad_fn.passes.values())):
        tensor = _first_tensor_in_obj(grad_fn_pass.grad_outputs)
        if tensor is not None:
            return _format_shape_str(tuple(tensor.shape))
    return "unknown"


def _first_tensor_in_obj(value: Any) -> torch.Tensor | None:
    """Return the first tensor found in a nested value.

    Parameters
    ----------
    value:
        Arbitrarily nested hook payload.

    Returns
    -------
    torch.Tensor | None
        First tensor in traversal order, if present.
    """

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor_in_obj(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor_in_obj(item)
            if tensor is not None:
                return tensor
    return None


def _build_skip_filtered_edge_map(
    model_log: "ModelLog",
    entries_to_plot: Mapping[str, GraphNode],
    *,
    vis_mode: str,
    show_buffer_layers: BufferVisibilityLiteral,
    skip_fn: SkipFn | None,
) -> tuple[dict[str, list[RenderEdge]], set[str]]:
    """Build skip-aware outgoing edges for each rendered node.

    Parameters
    ----------
    model_log:
        Owning ModelLog.
    entries_to_plot:
        Candidate nodes for the current visualization mode.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
        show_buffer_layers:
        Buffer visibility mode.
    skip_fn:
        Optional user predicate receiving aggregate LayerLog objects.

    Returns
    -------
    tuple[dict[str, list[RenderEdge]], set[str]]
        Outgoing edge map and skipped node labels.
    """

    visible_entries = {
        label: node
        for label, node in entries_to_plot.items()
        if not node.is_buffer_layer or _is_buffer_visible(node, show_buffer_layers)
    }
    skipped_labels: set[str] = set()
    if skip_fn is not None:
        for node in visible_entries.values():
            if isinstance(node, BoundaryNode):
                continue
            layer_log = _layer_log_for_node(model_log, node)
            if not skip_fn(layer_log):
                continue
            if layer_log.is_input_layer or layer_log.is_output_layer:
                raise ValueError(
                    f"skip_fn cannot skip input or output layer '{layer_log.layer_label}'."
                )
            skipped_labels.add(node.layer_label)

    edge_map: dict[str, list[RenderEdge]] = {}
    for node in visible_entries.values():
        if node.layer_label in skipped_labels:
            continue
        edge_map[node.layer_label] = _expand_edges_through_skipped(
            model_log,
            node,
            visible_entries,
            skipped_labels,
            vis_mode,
        )
    return edge_map, skipped_labels


def _resolve_focus_module(
    model_log: "ModelLog",
    module: "ModuleLog | str",
) -> "ModuleLog":
    """Resolve and validate a module focus argument.

    Parameters
    ----------
    model_log:
        Model log being rendered.
    module:
        ModuleLog instance or module address string.

    Returns
    -------
    ModuleLog
        Module to focus.

    Raises
    ------
    ValueError
        If the module cannot be found or belongs to a different ModelLog.
    """

    from ..data_classes.module_log import ModuleLog

    if isinstance(module, str):
        if module not in model_log.modules:
            raise ValueError(f"Module address '{module}' was not found in this ModelLog.")
        resolved = model_log.modules[module]
        if not isinstance(resolved, ModuleLog):
            raise ValueError(
                f"Module address '{module}' resolved to a module pass, not a ModuleLog."
            )
        return resolved
    if not isinstance(module, ModuleLog):
        raise ValueError("module must be a ModuleLog, module address string, or None.")
    if module._source_model_log is not model_log:
        raise ValueError("ModuleLog focus must belong to the ModelLog being rendered.")
    return module


def _build_module_focus_entries(
    model_log: "ModelLog",
    entries_to_plot: Mapping[str, GraphNode],
    target_module: "ModuleLog",
    *,
    vis_mode: str,
) -> dict[str, GraphNode]:
    """Return render entries focused on one module plus synthetic boundaries.

    Parameters
    ----------
    model_log:
        ModelLog being rendered.
    entries_to_plot:
        Original entries for the current render mode.
    target_module:
        Module whose internal forward operations should be shown.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    dict[str, GraphNode]
        Focused entries with boundary nodes inserted.

    Raises
    ------
    ValueError
        If the module contains no rendered layers.
    """

    focus_labels = {
        node.layer_label
        for node in entries_to_plot.values()
        if _node_is_inside_module(node, target_module.address)
    }
    if not focus_labels:
        raise ValueError(
            f"Module '{target_module.address}' has no layers to render. "
            "Empty modules cannot be focused."
        )

    focused_entries: dict[str, GraphNode] = {
        label: _copy_focus_node(node)
        for label, node in entries_to_plot.items()
        if node.layer_label in focus_labels
    }
    input_boundaries: dict[str, BoundaryNode] = {}
    output_boundaries: dict[str, BoundaryNode] = {}

    for render_node in list(focused_entries.values()):
        node = cast(FocusNode, render_node)
        new_parents: list[str] = []
        for parent_label in node.parent_layers:
            if parent_label in focus_labels:
                new_parents.append(parent_label)
                continue
            parent_node = entries_to_plot.get(parent_label)
            if parent_node is None:
                continue
            boundary = _get_or_create_boundary_node(
                input_boundaries,
                parent_node,
                target_module,
                vis_mode=vis_mode,
                boundary_kind="input",
                child_label=node.layer_label,
            )
            if node.layer_label not in boundary.child_layers:
                boundary.child_layers.append(node.layer_label)
            new_parents.append(boundary.layer_label)
        node.parent_layers = new_parents

        new_children: list[str] = []
        for child_label in node.child_layers:
            if child_label in focus_labels:
                new_children.append(child_label)
                continue
            child_node = entries_to_plot.get(child_label)
            if child_node is None:
                continue
            boundary = _get_or_create_boundary_node(
                output_boundaries,
                child_node,
                target_module,
                vis_mode=vis_mode,
                boundary_kind="output",
                parent_label=node.layer_label,
            )
            if node.layer_label not in boundary.parent_layers:
                boundary.parent_layers.append(node.layer_label)
            new_children.append(boundary.layer_label)
        node.child_layers = new_children

    _simplify_boundary_labels(input_boundaries, "input")
    _simplify_boundary_labels(output_boundaries, "output")
    for boundary_dict in (input_boundaries, output_boundaries):
        for label, boundary in boundary_dict.items():
            focused_entries[label] = boundary

    return focused_entries


def _node_is_inside_module(node: GraphNode, module_address: str) -> bool:
    """Return whether ``node`` ran inside ``module_address``."""

    return any(module.split(":", 1)[0] == module_address for module in node.containing_modules)


def _unwrap_focus_node(node: GraphNode) -> GraphNode:
    """Return the source node behind a focus proxy."""

    if isinstance(node, FocusNode):
        return node.original
    return node


def _base_node_for_metadata(node: GraphNode) -> BaseGraphNode:
    """Return a non-boundary graph node for metadata helpers."""

    unwrapped = _unwrap_focus_node(node)
    if isinstance(unwrapped, BoundaryNode):
        raise ValueError("Boundary nodes do not carry edge metadata.")
    return cast(BaseGraphNode, unwrapped)


def _copy_focus_node(node: GraphNode) -> GraphNode:
    """Return a shallow render copy whose edge lists can be rewritten."""

    if isinstance(node, BoundaryNode):
        return copy.copy(node)
    if isinstance(node, FocusNode):
        original = node.original
    else:
        original = node
    return FocusNode(
        original=original,
        parent_layers=list(node.parent_layers),
        child_layers=list(node.child_layers),
        containing_modules=list(node.containing_modules),
    )


def _get_or_create_boundary_node(
    boundary_nodes: dict[str, BoundaryNode],
    external_node: GraphNode,
    target_module: "ModuleLog",
    *,
    vis_mode: str,
    boundary_kind: str,
    child_label: str | None = None,
    parent_label: str | None = None,
) -> BoundaryNode:
    """Create or return a boundary node for one external layer."""

    external_label = external_node.layer_label.replace(":", "pass")
    boundary_label = f"__module_focus_{boundary_kind}_{external_label}"
    boundary = boundary_nodes.get(boundary_label)
    if boundary is not None:
        return boundary

    module_path = _boundary_module_path(target_module, vis_mode)
    boundary = BoundaryNode(
        layer_label=boundary_label,
        display_label=f"ext: {external_node.layer_label}",
        boundary_kind=boundary_kind,
        child_layers=[] if child_label is None else [child_label],
        parent_layers=[] if parent_label is None else [parent_label],
        containing_modules=module_path,
    )
    boundary_nodes[boundary_label] = boundary
    return boundary


def _boundary_module_path(target_module: "ModuleLog", vis_mode: str) -> list[str]:
    """Return a module path for focus boundary node cluster placement."""

    module_path = []
    parts = target_module.address.split(".") if target_module.address != "self" else ["self"]
    for idx in range(len(parts)):
        address = ".".join(parts[: idx + 1])
        module_path.append(address if vis_mode == "rolled" else f"{address}:1")
    return module_path


def _simplify_boundary_labels(
    boundary_nodes: dict[str, BoundaryNode],
    fallback_label: str,
) -> None:
    """Use simple labels when a focus side has exactly one boundary."""

    if len(boundary_nodes) != 1:
        return
    only_boundary = next(iter(boundary_nodes.values()))
    only_boundary.display_label = fallback_label


def _expand_edges_through_skipped(
    model_log: "ModelLog",
    parent_node: GraphNode,
    visible_entries: dict[str, GraphNode],
    skipped_labels: set[str],
    vis_mode: str,
) -> list[RenderEdge]:
    """Expand one node's outgoing edges through skipped successor chains.

    Parameters
    ----------
    model_log:
        Owning ModelLog.
    parent_node:
        Source node whose outgoing edges should be expanded.
    visible_entries:
        Visible nodes before applying ``skip_fn``.
    skipped_labels:
        Labels elided by ``skip_fn``.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    list[RenderEdge]
        Deduplicated non-skipped targets.
    """

    by_target: dict[str, RenderEdge] = {}
    for child_label in parent_node.child_layers:
        child_node = visible_entries.get(child_label)
        if child_node is None:
            continue
        reached = _walk_skipped_successors(
            model_log,
            child_node,
            visible_entries,
            skipped_labels,
            vis_mode,
            seen={parent_node.layer_label},
        )
        for target_node in reached:
            first_child = child_node
            existing = by_target.get(target_node.layer_label)
            if existing is None:
                by_target[target_node.layer_label] = RenderEdge(target_node, first_child)
            elif existing.metadata_child is not first_child:
                by_target[target_node.layer_label] = RenderEdge(target_node, None)
    return list(by_target.values())


def _walk_skipped_successors(
    model_log: "ModelLog",
    node: GraphNode,
    visible_entries: dict[str, GraphNode],
    skipped_labels: set[str],
    vis_mode: str,
    seen: set[str],
) -> list[GraphNode]:
    """Return non-skipped descendants reached through skipped chains.

    Parameters
    ----------
    model_log:
        Owning ModelLog.
    node:
        Current node in the traversal.
    visible_entries:
        Visible nodes before applying ``skip_fn``.
    skipped_labels:
        Labels elided by ``skip_fn``.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    seen:
        Labels already visited on this traversal branch.

    Returns
    -------
    list[GraphNode]
        Non-skipped reachable nodes.
    """

    if node.layer_label in seen:
        return []
    seen.add(node.layer_label)
    if node.layer_label not in skipped_labels:
        return [node]
    reached: list[GraphNode] = []
    for child_label in node.child_layers:
        child_node = visible_entries.get(child_label)
        if child_node is None:
            continue
        reached.extend(
            _walk_skipped_successors(
                model_log,
                child_node,
                visible_entries,
                skipped_labels,
                vis_mode,
                seen=set(seen),
            )
        )
    return reached


def _get_node_by_label(model_log: "ModelLog", label: str, vis_mode: str) -> GraphNode:
    """Return a render node by label for the active visualization mode."""

    if vis_mode == "unrolled":
        return model_log.layer_dict_main_keys[label]
    if vis_mode == "rolled":
        return model_log.layer_logs[label]
    raise ValueError(f"vis_mode must be 'unrolled' or 'rolled', not {vis_mode}")


def _add_node_to_graphviz(
    self: "ModelLog",
    node: GraphNode,
    graphviz_graph,
    module_edge_dict: Dict,
    edges_used: Set,
    vis_mode: str,
    collapsed_modules: Set,
    vis_nesting_depth: int = 1000,
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
    overrides: Optional[VisualizationOverrides] = None,
    node_mode: VisNodeModeLiteral = "default",
    node_spec_fn: NodeSpecFn | None = None,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    collapse_fn: CollapseFn | None = None,
    edge_map: Optional[dict[str, list[RenderEdge]]] = None,
) -> None:
    """Adds a node and its relevant edges to the graphviz figure.

    Args:
        node: node to add
        graphviz_graph: The graphviz object to add the node to.
        module_edge_dict: Dictionary of the module clusters.
        vis_mode: Whether to roll the graph or not
        vis_nesting_depth: How many levels of nested modules to show
        collapsed_modules: Labels of collapsed module nodes that have been made so far.
        show_buffer_layers: Buffer visibility mode.
        overrides: Graphviz attribute overrides for nodes, edges, etc.
    """
    collapse_address = _collapse_module_address_for_node(
        self,
        node,
        collapse_fn=collapse_fn,
        max_module_depth=vis_nesting_depth,
    )
    is_collapsed_module = collapse_address is not None

    if is_collapsed_module:
        _build_collapsed_module_node(
            self,
            node,
            graphviz_graph,
            collapsed_modules,
            vis_mode,
            vis_nesting_depth,
            collapse_address,
            overrides,  # type: ignore[arg-type]
            node_mode,
            collapsed_node_spec_fn,
        )
        node_color = "black"
    else:
        node_color = _build_layer_node(
            self,
            node,
            graphviz_graph,
            show_buffer_layers,
            vis_mode,
            overrides,  # type: ignore[arg-type]
            node_mode,
            node_spec_fn,
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
        vis_mode,
        show_buffer_layers,
        overrides,
        collapse_fn,
        edge_map,
    )


def _should_collapse_module(
    module_log: "ModuleLog",
    *,
    collapse_fn: CollapseFn | None,
    max_module_depth: int,
) -> bool:
    """Return whether ``module_log`` should render as a collapsed module node.

    Parameters
    ----------
    module_log:
        Module metadata to check.
    collapse_fn:
        Optional user predicate. When supplied, it overrides depth logic.
    max_module_depth:
        Legacy nesting-depth threshold.

    Returns
    -------
    bool
        True if the module should be collapsed.
    """

    if collapse_fn is not None:
        return bool(collapse_fn(module_log))
    if max_module_depth == 0:
        return False
    return module_log.address_depth >= max_module_depth


def _collapse_module_address_for_node(
    model_log: "ModelLog",
    node: GraphNode,
    *,
    collapse_fn: CollapseFn | None,
    max_module_depth: int,
) -> Optional[str]:
    """Return the module-pass address that should absorb ``node``, if any.

    Parameters
    ----------
    model_log:
        Owning ModelLog.
    node:
        Layer node being rendered.
    collapse_fn:
        Optional user collapse predicate.
    max_module_depth:
        Legacy nesting-depth threshold.

    Returns
    -------
    Optional[str]
        Pass-qualified module address for unrolled lookup, or ``None``.
    """

    if isinstance(node, BoundaryNode):
        return None

    containing_modules = list(node.containing_modules)
    if getattr(node, "is_leaf_module_output", False):
        containing_modules = containing_modules[:-1]
    if not containing_modules:
        return None

    if collapse_fn is None:
        if max_module_depth == 0 or len(containing_modules) < max_module_depth:
            return None
        return containing_modules[max_module_depth - 1]

    for module_address_w_pass in containing_modules:
        module_address = module_address_w_pass.rsplit(":", 1)[0]
        if _should_collapse_module(
            cast("ModuleLog", model_log.modules[module_address]),
            collapse_fn=collapse_fn,
            max_module_depth=max_module_depth,
        ):
            return module_address_w_pass
    return None


def _is_collapsed_module(
    node: GraphNode,
    vis_nesting_depth: int,
    model_log: Optional["ModelLog"] = None,
    collapse_fn: CollapseFn | None = None,
) -> bool:
    """THE IndexError guard for collapsed module rendering.

    Returns True if the node is nested deep enough to be rendered as a
    collapsed ``box3d`` module summary node instead of an individual layer.

    This function is the single decision point that determines whether a node
    gets its own graphviz node or is absorbed into a module box.  Getting this
    wrong causes IndexError when ``_build_collapsed_module_node`` tries to
    access ``containing_modules[vis_nesting_depth - 1]``.

    Special cases:
    - ``vis_nesting_depth == 0``: show all layers, never collapse (#94).
    - ``is_leaf_module_output``: the node represents the output of
      its innermost module, so its effective nesting depth is one less (it
      visually "belongs" to the parent scope).

    Args:
        node: The LayerPassLog or LayerLog node to check.
        vis_nesting_depth: Maximum nesting depth before collapsing into a module box.
    """
    if model_log is not None:
        return (
            _collapse_module_address_for_node(
                model_log,
                node,
                collapse_fn=collapse_fn,
                max_module_depth=vis_nesting_depth,
            )
            is not None
        )
    if vis_nesting_depth == 0:
        return False  # #94: depth 0 means show all layers, never collapse

    node_nesting_depth = len(node.containing_modules)
    # Bottom-level submodule outputs are rendered at the parent nesting level,
    # not their own, so subtract 1 from their effective depth.
    if getattr(node, "is_leaf_module_output", False):
        node_nesting_depth -= 1

    if node_nesting_depth >= vis_nesting_depth:
        return True
    else:
        return False


def _build_layer_node(
    self: "ModelLog",
    node: GraphNode,
    graphviz_graph,
    show_buffer_layers: BufferVisibilityLiteral,
    vis_mode: str,
    overrides: VisualizationOverrides,
    node_mode: VisNodeModeLiteral,
    node_spec_fn: NodeSpecFn | None = None,
) -> str:
    """Builds and adds a standard (non-collapsed) layer node to the graphviz graph.

    Args:
        node: The LayerPassLog or LayerLog node to render.
        graphviz_graph: The graphviz Digraph object to add the node to.
        show_buffer_layers: Buffer visibility mode.
        vis_mode: 'unrolled' or 'rolled'.
        overrides: Graphviz attribute overrides.

    Returns:
        The node color string used for this node.
    """
    if isinstance(node, BoundaryNode):
        fillcolor = INPUT_COLOR if node.boundary_kind == "input" else OUTPUT_COLOR
        spec = NodeSpec(
            lines=[node.display_label],
            shape="oval",
            fillcolor=fillcolor,
            fontcolor="black",
            color="black",
            style="filled,solid",
            extra_attrs={"ordering": "out"},
        )
        node_args = _node_spec_to_graphviz_args(spec)
        node_args["name"] = node.layer_label
        graphviz_graph.node(**node_args)
        return "black"

    # Get the address, shape, color, and line style:

    node_address, node_shape, node_color = _get_node_address_shape_color(
        self, node, show_buffer_layers
    )
    node_bg_color = _get_node_bg_color(self, node)

    if node.has_input_ancestor:
        line_style = "solid"
    else:
        line_style = "dashed"

    default_spec = NodeSpec(
        lines=compute_default_node_lines(node, node_address, vis_mode),
        shape=node_shape,
        fillcolor=node_bg_color,
        fontcolor=node_color,
        style=f"filled,{line_style}",
        color=node_color,
        extra_attrs={"ordering": "out"},
    )
    spec = _apply_node_spec_fn(self, node, default_spec, node_mode, node_spec_fn)

    # Graphviz node names can't contain colons (used for port syntax), so
    # replace ":" with "pass" in pass-qualified labels (e.g., "relu_1:2" -> "relu_1pass2").
    node_args = _node_spec_to_graphviz_args(spec)
    node_args["name"] = node.layer_label.replace(":", "pass")
    hidden_buffer_addresses = _get_hidden_parent_buffer_addresses(self, node, show_buffer_layers)
    if hidden_buffer_addresses and not (
        node.is_input_layer or node.is_output_layer or node.is_buffer_layer
    ):
        node_args["peripheries"] = "2"
        node_args["tooltip"] = f"Hidden buffers: {', '.join(hidden_buffer_addresses)}"
    # Colon in bg_color means it's a gradient fill (e.g.,
    # "#D9D9D9:#B0B0B0" for mixed trainable/frozen params).
    # Graphviz requires gradientangle to render gradients.
    if spec.fillcolor is not None and ":" in spec.fillcolor:
        node_args["gradientangle"] = "0"

    graphviz_graph.node(**node_args)

    if node.is_final_output:
        with graphviz_graph.subgraph() as s:
            s.attr(rank="sink")
            s.node(node.layer_label.replace(":", "pass"))

    return node_color


def _build_collapsed_module_node(
    self: "ModelLog",
    node: GraphNode,
    graphviz_graph,
    collapsed_modules,
    vis_mode: str,
    vis_nesting_depth: int,
    collapse_address: str | None,
    overrides: VisualizationOverrides,
    node_mode: VisNodeModeLiteral,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
) -> None:
    """Builds and adds a collapsed module box node to the graphviz graph.

    Args:
        node: The LayerPassLog or LayerLog node triggering the collapse.
        graphviz_graph: The graphviz Digraph object to add the node to.
        collapsed_modules: Set of collapsed module names already added; updated in place.
        vis_mode: 'unrolled' or 'rolled'.
        vis_nesting_depth: Maximum nesting depth; nodes at this depth are collapsed.
        overrides: Graphviz attribute overrides.
    """
    # Access the module at the collapse threshold depth.  This index is safe
    # because _is_collapsed_module already verified the node is deep enough.
    module_address_w_pass = (
        collapse_address
        if collapse_address is not None
        else node.containing_modules[vis_nesting_depth - 1]
    )
    # rsplit with maxsplit=1 handles module names containing colons (#104).
    module_tuple = module_address_w_pass.rsplit(":", 1)
    module_output_layer = self[module_address_w_pass]
    module_output_shape = module_output_layer.tensor_shape or ()
    module_output_fsize = module_output_layer.tensor_memory_str
    module_address, pass_num = module_tuple
    ml = self.modules[module_address]
    module_type = ml.module_class_name  # type: ignore[union-attr]
    module_num_passes = ml.num_passes  # type: ignore[union-attr]
    module_nparams = ml.num_params  # type: ignore[union-attr]

    # In unrolled mode, each pass of a module is a separate collapsed node
    # (e.g., "encoder.layer.0pass1").  In rolled mode, all passes share one
    # node (e.g., "encoder.layer.0").
    if vis_mode == "unrolled":
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
    elif vis_mode == "unrolled" and (module_num_passes > 1):
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

    default_spec = NodeSpec(
        lines=[
            node_title.replace("<b>", "").replace("</b>", ""),
            module_type,
            f"{tensor_shape_str} ({module_output_fsize})",
            f"{module_num_tensors} layers total",
            param_detail,
        ],
        shape="box3d",
        fillcolor=bg_color,
        fontcolor="black",
        color="black",
        style=f"filled,{line_style}",
        extra_attrs={"ordering": "out"},
    )
    mode_fn = COLLAPSED_MODE_REGISTRY[node_mode]
    mode_result = mode_fn(ml, default_spec)  # type: ignore[arg-type]
    mode_spec = default_spec if mode_result is None else mode_result
    if collapsed_node_spec_fn is not None:
        result = collapsed_node_spec_fn(ml, mode_spec)  # type: ignore[arg-type]
        spec = mode_spec if result is None else result
    else:
        spec = mode_spec

    node_args = _node_spec_to_graphviz_args(spec)
    node_args["name"] = node_name
    if spec.fillcolor is not None and ":" in spec.fillcolor:
        node_args["gradientangle"] = "0"

    graphviz_graph.node(**node_args)
    collapsed_modules.add(node_name)


def _get_node_address_shape_color(
    self: "ModelLog",
    node: GraphNode,
    show_buffer_layers: BufferVisibilityLiteral | bool,
) -> Tuple[str, str, str]:
    """Gets the node shape, address, and color for the graphviz figure.

    Args:
        node: node to add

    Returns:
        node_address: address of the node
        node_shape: shape of the node
        node_color: color of the node
    """
    source_node = _unwrap_focus_node(node)
    if isinstance(source_node, BoundaryNode):
        raise ValueError("Boundary nodes are rendered by the boundary path.")
    show_buffer_layers = _normalize_buffer_visibility(show_buffer_layers)
    if show_buffer_layers != "always":
        only_non_buffer_layer = _is_only_non_buffer_in_module(self, node, show_buffer_layers)
    else:
        only_non_buffer_layer = False

    if (node.is_leaf_module_output or only_non_buffer_layer) and (len(node.containing_modules) > 0):
        if isinstance(source_node, LayerPassLog):
            module_pass_exited = node.containing_modules[-1]
            module, _ = module_pass_exited.split(":")
            if self.modules[module].num_passes == 1:  # type: ignore[union-attr]
                node_address = module
            else:
                node_address = module_pass_exited
        else:
            sample_module_pass = node.containing_modules[-1]
            module = sample_module_pass.split(":")[0]
            node_address = module

        node_address = "<br/>@" + node_address
        node_shape = "box"
        node_color = "black"
    elif node.is_buffer_layer:
        if (self.buffer_num_passes[source_node.buffer_address] == 1) or (
            isinstance(source_node, LayerLog) and node.num_passes > 1
        ):
            buffer_address = source_node.buffer_address
        else:
            buffer_address = f"{source_node.buffer_address}:{source_node.buffer_pass}"
        node_address = "<br/>@" + buffer_address
        node_shape = "cylinder"
        node_color = "black"
    elif node.is_output_layer or node.is_input_layer:
        node_address = "<br/>@" + node.io_role
        node_shape = "oval"
        node_color = "black"
    else:
        node_address = ""
        node_shape = "oval"
        node_color = "black"

    return node_address, node_shape, node_color


def _is_only_non_buffer_in_module(
    self: "ModelLog", node: GraphNode, show_buffer_layers: BufferVisibilityLiteral
) -> bool:
    """Returns True if a layer is the only non-buffer layer in a leaf module.

    Leaf modules are those with no child submodules. Container modules with
    functional ops at the end should NOT match — those ops are rendered as
    ovals, not boxes (issue #48).

    Args:
        node: The LayerPassLog or LayerLog node to check.
        show_buffer_layers: Buffer visibility mode.
    """
    # Check whether it leaves its module:
    if not (
        (len(node.modules_exited) > 0)
        and (len(node.containing_modules) > 0)
        and (node.containing_modules[-1].split(":")[0] in node.modules_exited)
    ):
        return False

    # Only apply box rendering for leaf modules (no child submodules).
    exited_module = node.containing_modules[-1].split(":")[0]
    if exited_module in self.modules and len(self.modules[exited_module].call_children) > 0:
        return False

    # Now check whether all of its parents are either buffers, or are outside the module.
    # If any aren't, return False.

    for parent_layer_label in node.parent_layers:
        if parent_layer_label.startswith("__module_focus_"):
            continue
        source_node = _unwrap_focus_node(node)
        if isinstance(source_node, LayerPassLog):
            parent_layer = self[parent_layer_label]
        else:
            parent_layer = self.layer_logs[parent_layer_label]  # type: ignore[assignment]
        if (
            (not parent_layer.is_buffer_layer)
            or _is_buffer_visible(parent_layer, show_buffer_layers)
        ) and (
            (len(parent_layer.containing_modules) > 0)
            and parent_layer.containing_modules[-1] == node.containing_modules[-1]
        ):
            return False

    return True


def _get_node_bg_color(self: "ModelLog", node: GraphNode) -> str:
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
    elif node.uses_params:
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


def _apply_node_spec_fn(
    model_log: "ModelLog",
    node: GraphNode,
    default_spec: NodeSpec,
    node_mode: VisNodeModeLiteral,
    node_spec_fn: NodeSpecFn | None,
) -> NodeSpec:
    """Apply a layer node callback to a default spec.

    Parameters
    ----------
    model_log:
        Owning ModelLog.
    node:
        Rendered LayerPassLog or LayerLog.
    default_spec:
        Default node spec.
    node_mode:
        Preset to apply before the optional user callback.
    node_spec_fn:
        Optional user callback. Unrolled nodes are represented to the callback
        by their parent LayerLog.

    Returns
    -------
    NodeSpec
        Callback result, or ``default_spec`` when the callback returns ``None``.
    """

    layer_log = _layer_log_for_node(model_log, node)
    mode_fn = MODE_REGISTRY[node_mode]
    mode_result = mode_fn(layer_log, default_spec)
    mode_spec = default_spec if mode_result is None else mode_result
    if node_spec_fn is None:
        return mode_spec
    result = node_spec_fn(layer_log, mode_spec)
    return mode_spec if result is None else result


def _layer_log_for_node(model_log: "ModelLog", node: GraphNode) -> "LayerLog":
    """Return the aggregate LayerLog for ``node``.

    Parameters
    ----------
    model_log:
        Owning ModelLog.
    node:
        LayerPassLog or LayerLog.

    Returns
    -------
    LayerLog
        Aggregate layer log for callbacks.
    """

    node = _unwrap_focus_node(node)
    if isinstance(node, BoundaryNode):
        raise ValueError("Synthetic boundary nodes do not have LayerLog metadata.")
    if isinstance(node, LayerLog):
        return node
    if node.parent_layer_log is not None:
        return node.parent_layer_log
    return model_log.layer_logs[node.layer_label_no_pass]


def _node_spec_to_graphviz_args(spec: NodeSpec) -> dict[str, str]:
    """Convert a ``NodeSpec`` to Graphviz node keyword arguments.

    Parameters
    ----------
    spec:
        Node spec to convert.

    Returns
    -------
    dict[str, str]
        Graphviz keyword arguments except for ``name``.
    """

    node_args: dict[str, str] = {
        "label": render_lines_to_html(spec.lines),
        "shape": spec.shape,
        "style": spec.style,
    }
    optional_attrs: dict[str, object | None] = {
        "fillcolor": spec.fillcolor,
        "fontcolor": spec.fontcolor,
        "color": spec.color,
        "penwidth": spec.penwidth,
        "tooltip": spec.tooltip,
    }
    for attr_name, attr_value in optional_attrs.items():
        if attr_value is not None:
            node_args[attr_name] = str(attr_value)
    node_args.update(spec.extra_attrs)
    return node_args


def compute_default_node_lines(
    layer_log: GraphNode,
    node_address: str = "",
    vis_mode: str = "unrolled",
) -> list[str]:
    """Build default plain-text rows for a layer node.

    Parameters
    ----------
    layer_log:
        LayerPassLog or LayerLog to render.
    node_address:
        Existing address suffix from TorchLens node address logic.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    list[str]
        Plain-text rows for ``NodeSpec.lines``.
    """

    layer_log = _unwrap_focus_node(layer_log)
    if isinstance(layer_log, BoundaryNode):
        return [layer_log.display_label]

    if (layer_log.num_passes > 1) and (vis_mode == "unrolled"):
        pass_label = f":{layer_log.pass_num}"
    elif (layer_log.num_passes > 1) and (vis_mode == "rolled"):
        pass_label = f" (x{layer_log.num_passes})"
    else:
        pass_label = ""

    if layer_log.layer_type in ["input", "output", "buffer"]:
        title = f"{layer_log.layer_type}_{layer_log.layer_type_num}{pass_label}"
    else:
        title = (
            f"{layer_log.layer_type}_{layer_log.layer_type_num}_"
            f"{layer_log.layer_total_num}{pass_label}"
        )

    lines: list[str] = []
    if layer_log.is_terminal_bool_layer:
        lines.append(str(layer_log.scalar_bool_value).upper())
    lines.append(title)
    lines.append(f"{_format_shape_str(layer_log.tensor_shape)} ({layer_log.tensor_memory_str})")

    important_args = _format_important_args(layer_log)
    if important_args:
        lines.append(important_args)

    param_line = _make_param_line(layer_log)
    if param_line:
        lines.append(param_line)

    address_line = node_address.replace("<br/>", "")
    if address_line:
        lines.append(address_line)
    return lines


def _make_node_label(
    node: Union["LayerPassLog", "LayerLog"],
    node_address: str,
    vis_mode: str,
) -> str:
    """Builds an HTML-table label string for a graphviz node.

    Assembles rows for the layer name, tensor shape, operation type, and other
    metadata into an HTML table used as the node label in graphviz rendering.
    """
    # Pass info:

    if (node.num_passes > 1) and (vis_mode == "unrolled"):
        pass_label = f":{node.pass_num}"
    elif (node.num_passes > 1) and (vis_mode == "rolled"):
        pass_label = f" (x{node.num_passes})"
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

    tensor_memory = node.tensor_memory_str
    if node.layer_type in ["input", "output", "buffer"]:
        node_title = f"<b>{node.layer_type}_{node.layer_type_num}{pass_label}</b>"
    else:
        node_title = (
            f"<b>{node.layer_type}_{node.layer_type_num}_{node.layer_total_num}{pass_label}</b>"
        )

    if node.is_terminal_bool_layer:
        label_text = str(node.scalar_bool_value).upper()
        bool_label = f"<b><u>{label_text}:</u></b><br/><br/>"
    else:
        bool_label = ""

    node_label = (
        f"<{bool_label}{node_title}<br/>{tensor_shape_str} "
        f"({tensor_memory}){param_label}{node_address}>"
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


def _make_param_line(node: GraphNode) -> str:
    """Make a plain-text parameter summary row for a node label.

    Parameters
    ----------
    node:
        Node being rendered.

    Returns
    -------
    str
        Plain-text parameter summary, or ``""`` when no parameters are used.
    """

    if node.num_param_tensors == 0:
        return ""

    param_logs = getattr(node, "parent_param_logs", [])
    if param_logs:
        parts = []
        for param_log in param_logs:
            shape_str = _format_shape_str(param_log.shape)
            wrapper = ("(", ")") if param_log.trainable else ("[", "]")
            parts.append(f"{param_log.name}: {wrapper[0]}{shape_str}{wrapper[1]}")
        return "params: " + ", ".join(parts)

    each_param_shape = [_format_shape_str(shape) for shape in node.parent_param_shapes]
    return "params: " + ", ".join(each_param_shape)


def _format_important_args(node: GraphNode) -> str:
    """Format a compact important-argument row for known operation types.

    Parameters
    ----------
    node:
        Node whose ``func_config`` should be summarized.

    Returns
    -------
    str
        Compact argument summary, or ``""`` for unrecognized/no-op configs.
    """

    config = getattr(node, "func_config", {}) or {}
    layer_type = node.layer_type.lower().replace("_", "")
    if layer_type in {
        "conv1d",
        "conv2d",
        "conv3d",
        "convolution",
        "convtranspose1d",
        "convtranspose2d",
        "convtranspose3d",
    }:
        return _format_config_fields(
            config,
            [
                ("kernel_size", "k", None),
                ("stride", "s", 1),
                ("padding", "p", 0),
                ("dilation", "d", 1),
                ("groups", "g", 1),
            ],
        )
    if layer_type == "linear":
        return _format_config_fields(
            config,
            [("in_features", "in", None), ("out_features", "out", None)],
        )
    if "pool" in layer_type:
        return _format_config_fields(
            config,
            [
                ("kernel_size", "k", None),
                ("stride", "s", None),
                ("padding", "p", 0),
                ("output_size", "out", None),
            ],
        )
    if layer_type == "layernorm":
        return _format_config_fields(config, [("normalized_shape", "shape", None)])
    if "batchnorm" in layer_type or "instancenorm" in layer_type:
        num_features = _num_features_from_params(node)
        if num_features is None:
            return ""
        return f"features={num_features}"
    if layer_type == "groupnorm":
        return _format_config_fields(config, [("num_groups", "groups", None)])
    if "multiheadattention" in layer_type or layer_type == "scaleddotproductattention":
        return _format_config_fields(
            config,
            [
                ("num_heads", "heads", None),
                ("embed_dim", "dim", None),
                ("dropout", "dropout", 0),
                ("dropout_p", "dropout", 0),
            ],
        )
    if "dropout" in layer_type:
        return _format_config_fields(config, [("p", "p", 0)])
    if layer_type == "embedding":
        return _format_config_fields(
            config,
            [("num_embeddings", "n", None), ("embedding_dim", "dim", None)],
        )
    return ""


def _format_config_fields(
    config: dict[str, Any],
    fields: list[tuple[str, str, Any]],
) -> str:
    """Format selected config fields with short display names.

    Parameters
    ----------
    config:
        Function config captured for a node.
    fields:
        ``(field_name, display_name, default_value)`` tuples.

    Returns
    -------
    str
        Space-separated compact summary.
    """

    parts = []
    for key, display, default in fields:
        if key not in config:
            continue
        value = config[key]
        if default is not None and _value_matches_default(value, default):
            continue
        parts.append(f"{display}={_format_config_value(value)}")
    return " ".join(parts)


def _value_matches_default(value: Any, default: Any) -> bool:
    """Return whether ``value`` equals a scalar default, including repeated tuples."""

    if value == default:
        return True
    if isinstance(value, tuple) and all(item == default for item in value):
        return True
    return False


def _format_config_value(value: Any) -> str:
    """Format a captured config value compactly for a node label."""

    if isinstance(value, tuple):
        if len(value) > 0 and all(item == value[0] for item in value):
            return str(value[0])
        return "x".join(str(item) for item in value)
    return str(value)


def _num_features_from_params(node: GraphNode) -> Optional[int]:
    """Infer normalization feature count from parameter shape metadata."""

    if not node.parent_param_shapes:
        return None
    first_shape = node.parent_param_shapes[0]
    if len(first_shape) == 0:
        return None
    return int(first_shape[0])


def _add_edges_for_node(
    self: "ModelLog",
    parent_node: GraphNode,
    parent_is_collapsed_module: bool,
    vis_nesting_depth: int,
    node_color: str,
    module_edge_dict: Dict,
    edges_used: Set,
    graphviz_graph,
    vis_mode: str = "unrolled",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
    overrides: Optional[VisualizationOverrides] = None,
    collapse_fn: CollapseFn | None = None,
    edge_map: Optional[dict[str, list[RenderEdge]]] = None,
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
        vis_mode: ``'unrolled'`` or ``'rolled'``.
        show_buffer_layers: Buffer visibility mode.
        overrides: Graphviz attribute overrides.
    """
    if edge_map is None:
        render_edges = [
            RenderEdge(
                target=_get_node_by_label(self, child_layer_label, vis_mode), metadata_child=None
            )
            for child_layer_label in parent_node.child_layers
        ]
    else:
        render_edges = edge_map.get(parent_node.layer_label, [])

    for render_edge in render_edges:
        child_node = render_edge.target
        metadata_child = render_edge.metadata_child

        if child_node.is_buffer_layer and not _is_buffer_visible(child_node, show_buffer_layers):
            continue

        if parent_node.has_input_ancestor:
            edge_style = "solid"
        else:
            edge_style = "dashed"

        if parent_is_collapsed_module:
            module_name_w_pass = _collapse_module_address_for_node(
                self,
                parent_node,
                collapse_fn=collapse_fn,
                max_module_depth=vis_nesting_depth,
            )
            if module_name_w_pass is None:
                continue
            module_tuple = module_name_w_pass.split(":")
            if vis_mode == "unrolled":
                tail_name = "pass".join(module_tuple)
            else:
                tail_name = module_tuple[0]
        else:
            tail_name = parent_node.layer_label.replace(":", "pass")

        child_collapse_address = _collapse_module_address_for_node(
            self,
            child_node,
            collapse_fn=collapse_fn,
            max_module_depth=vis_nesting_depth,
        )
        child_is_collapsed_module = child_collapse_address is not None

        if child_is_collapsed_module:
            module_name_w_pass = child_collapse_address
            if module_name_w_pass is None:
                continue
            module_tuple = module_name_w_pass.split(":")
            if vis_mode == "unrolled":
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
            child_containing_modules = child_node.containing_modules[:]
            parent_containing_modules = parent_node.containing_modules[:]
            # Adjust for bottom-level submodule outputs (they belong to parent scope).
            if child_node.is_leaf_module_output:
                child_containing_modules = child_containing_modules[:-1]
            if parent_node.is_leaf_module_output:
                parent_containing_modules = parent_containing_modules[:-1]
            if (
                child_containing_modules[:vis_nesting_depth]
                == parent_containing_modules[:vis_nesting_depth]
            ):
                continue

        # Edge deduplication: multiple layers mapping to the same collapsed
        # module node would produce duplicate edges without this check.
        if tail_name == head_name:
            continue
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

        edge_has_boundary = isinstance(parent_node, BoundaryNode) or isinstance(
            child_node, BoundaryNode
        )

        if not child_is_collapsed_module and not edge_has_boundary:
            metadata_base = (
                _base_node_for_metadata(metadata_child) if metadata_child is not None else None
            )
            edge_label = (
                _compute_edge_label(
                    _base_node_for_metadata(parent_node),
                    metadata_base,
                    self,
                    vis_mode,
                )
                if metadata_base is not None
                else None
            )
            if edge_label is not None:
                edge_dict["label"] = edge_label

        # Annotate passes for rolled node edge if it varies across passes
        if vis_mode == "rolled" and metadata_child is not None and not edge_has_boundary:
            metadata_base_for_pass = _base_node_for_metadata(metadata_child)
            parent_base_for_pass = _base_node_for_metadata(parent_node)
            if isinstance(metadata_base_for_pass, LayerLog) and isinstance(
                parent_base_for_pass, LayerLog
            ):
                _label_rolled_pass_nums(
                    metadata_base_for_pass,
                    parent_base_for_pass,
                    edge_dict,
                )

        # Label the arguments to the next node if multiple inputs
        if not child_is_collapsed_module and metadata_child is not None and not edge_has_boundary:
            _label_node_arguments_if_needed(
                self,
                _base_node_for_metadata(parent_node),
                _base_node_for_metadata(metadata_child),
                edge_dict,
                show_buffer_layers,
            )

        for arg_name, arg_val in overrides.edge.items():  # type: ignore[union-attr]
            if callable(arg_val):
                edge_dict[arg_name] = str(arg_val(self, parent_node, metadata_child or child_node))
            else:
                edge_dict[arg_name] = str(arg_val)

        # Add it to the appropriate module cluster (most nested one containing both nodes)
        if edge_has_boundary:
            containing_module = _get_lowest_containing_module_for_two_render_nodes(
                parent_node,
                child_node,
                both_nodes_collapsed_modules,
                vis_nesting_depth,
            )
        else:
            containing_module = _get_lowest_containing_module_for_two_nodes(
                _base_node_for_metadata(parent_node),
                _base_node_for_metadata(child_node),
                both_nodes_collapsed_modules,
                vis_nesting_depth,
            )
        if containing_module != -1:
            module_edge_dict[containing_module]["edges"].append(edge_dict)
            if parent_node.has_input_ancestor or child_node.has_input_ancestor:
                module_edge_dict[containing_module]["has_input_ancestor"] = True
                for module in parent_node.containing_modules:
                    module_key = module.split(":")[0] if vis_mode == "rolled" else module
                    module_edge_dict[module_key]["has_input_ancestor"] = True
                    if module_key == containing_module:
                        break
                for module in child_node.containing_modules:
                    module_key = module.split(":")[0] if vis_mode == "rolled" else module
                    module_edge_dict[module_key]["has_input_ancestor"] = True
                    if module_key == containing_module:
                        break
        else:
            graphviz_graph.edge(**edge_dict)

        # Finally, add a backwards edge if both tensors have stored gradients.
        if vis_mode == "unrolled" and not (
            isinstance(parent_node, BoundaryNode) or isinstance(child_node, BoundaryNode)
        ):
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


def _compute_edge_label(
    parent_node: Union["LayerPassLog", "LayerLog"],
    child_node: Union["LayerPassLog", "LayerLog"],
    model_log: "ModelLog",
    vis_mode: str,
) -> Optional[str]:
    """Return the highest-priority semantic label for an edge.

    Precedence matches the Phase 7 conditional rendering spec:

    1. Arm-entry labels from ``ModelLog.conditional_arm_edges`` /
       ``ModelLog.conditional_edge_passes``.
    2. ``IF`` labels from ``ModelLog.conditional_branch_edges``.
    3. ``None`` when the edge has no branch semantics.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        model_log:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    Optional[str]
        Graphviz HTML label string, or ``None`` if no semantic label applies.
    """
    arm_label = _compute_arm_entry_edge_label(parent_node, child_node, model_log, vis_mode)
    if arm_label is not None:
        return _format_branch_edge_label_html(arm_label)

    if _edge_is_conditional_branch(parent_node, child_node, model_log, vis_mode):
        return _format_branch_edge_label_html("IF")

    return None


def _compute_arm_entry_edge_label(
    parent_node: Union["LayerPassLog", "LayerLog"],
    child_node: Union["LayerPassLog", "LayerLog"],
    model_log: "ModelLog",
    vis_mode: str,
) -> Optional[str]:
    """Return the arm-entry text for an edge, without Graphviz HTML wrapping.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        model_log:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    Optional[str]
        Plain-text arm label, or ``None`` if the edge is not an arm-entry edge.
    """
    arm_entries = _get_arm_edge_entries(parent_node, child_node, model_log, vis_mode)
    if not arm_entries:
        return None

    if vis_mode == "rolled":
        return _format_rolled_arm_entry_label(arm_entries, model_log)

    if len(arm_entries) == 1:
        conditional_id, branch_kind, _ = arm_entries[0]
        return _format_arm_entry_text(conditional_id, branch_kind, model_log)

    return " · ".join(
        [
            _format_arm_entry_text(
                conditional_id,
                branch_kind,
                model_log,
                include_conditional_reference=True,
            )
            for conditional_id, branch_kind, _ in arm_entries
        ]
    )


def _get_arm_edge_entries(
    parent_node: Union["LayerPassLog", "LayerLog"],
    child_node: Union["LayerPassLog", "LayerLog"],
    model_log: "ModelLog",
    vis_mode: str,
) -> List[Tuple[int, str, Optional[Tuple[int, ...]]]]:
    """Collect conditional-arm metadata for one rendered edge.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        model_log:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    List[Tuple[int, str, Optional[Tuple[int, ...]]]]
        Sorted ``(conditional_id, branch_kind, pass_nums)`` tuples. Unrolled
        edges use ``pass_nums=None``.
    """
    arm_entries: List[Tuple[int, str, Optional[Tuple[int, ...]]]] = []
    if vis_mode == "unrolled":
        edge_key = (parent_node.layer_label, child_node.layer_label)
        for (conditional_id, branch_kind), edge_list in model_log.conditional_arm_edges.items():
            if edge_key in edge_list:
                arm_entries.append((conditional_id, branch_kind, None))
    elif vis_mode == "rolled":
        parent_no_pass = parent_node.layer_label_no_pass
        child_no_pass = child_node.layer_label_no_pass
        for (
            edge_parent,
            edge_child,
            conditional_id,
            branch_kind,
        ), pass_nums in model_log.conditional_edge_passes.items():
            if (edge_parent, edge_child) == (parent_no_pass, child_no_pass):
                arm_entries.append((conditional_id, branch_kind, tuple(pass_nums)))
    else:
        raise ValueError(f"vis_mode must be 'unrolled' or 'rolled', not {vis_mode}")

    return sorted(arm_entries, key=lambda entry: _arm_entry_sort_key(entry[0], entry[1], model_log))


def _format_rolled_arm_entry_label(
    arm_entries: List[Tuple[int, str, Optional[Tuple[int, ...]]]],
    model_log: "ModelLog",
) -> str:
    """Format a rolled-mode arm-entry label with pass-awareness.

    Args:
        arm_entries:
            Sorted ``(conditional_id, branch_kind, pass_nums)`` tuples for one
            rolled edge.
        model_log:
            Owning model log containing conditional metadata.

    Returns
    -------
    str
        Plain-text arm label for the rolled edge.
    """
    if len(arm_entries) == 1:
        conditional_id, branch_kind, _ = arm_entries[0]
        return _format_arm_entry_text(conditional_id, branch_kind, model_log)

    pass_sets = [set(pass_nums or ()) for _, _, pass_nums in arm_entries]
    if pass_sets and len({tuple(sorted(pass_set)) for pass_set in pass_sets}) == 1:
        return " · ".join(
            [
                _format_arm_entry_text(
                    conditional_id,
                    branch_kind,
                    model_log,
                    include_conditional_reference=True,
                )
                for conditional_id, branch_kind, _ in arm_entries
            ]
        )

    pass_counts: Dict[int, int] = defaultdict(int)
    for _, _, pass_nums in arm_entries:
        for pass_num in pass_nums or ():
            pass_counts[pass_num] += 1

    if pass_counts and all(pass_count == 1 for pass_count in pass_counts.values()):
        return " / ".join(
            [
                _format_rolled_pass_arm_text(
                    conditional_id,
                    branch_kind,
                    pass_nums,
                    model_log,
                    include_conditional_reference=_rolled_labels_need_disambiguation(arm_entries),
                )
                for conditional_id, branch_kind, pass_nums in arm_entries
            ]
        )

    return "mixed"


def _rolled_labels_need_disambiguation(
    arm_entries: List[Tuple[int, str, Optional[Tuple[int, ...]]]],
) -> bool:
    """Return True when rolled branch labels need conditional disambiguation.

    Args:
        arm_entries:
            Sorted ``(conditional_id, branch_kind, pass_nums)`` tuples for one
            rolled edge.

    Returns
    -------
    bool
        True when multiple entries would otherwise share the same branch label.
    """
    base_labels = [_format_branch_kind_text(branch_kind) for _, branch_kind, _ in arm_entries]
    return len(base_labels) != len(set(base_labels))


def _format_rolled_pass_arm_text(
    conditional_id: int,
    branch_kind: str,
    pass_nums: Optional[Tuple[int, ...]],
    model_log: "ModelLog",
    include_conditional_reference: bool,
) -> str:
    """Format one rolled arm label with its pass list.

    Args:
        conditional_id:
            Dense conditional id.
        branch_kind:
            Branch kind such as ``"then"`` or ``"elif_2"``.
        pass_nums:
            Sorted pass numbers for this rolled edge/arm tuple.
        model_log:
            Owning model log containing conditional metadata.
        include_conditional_reference:
            Whether to append a conditional line-number reference.

    Returns
    -------
    str
        Plain-text label like ``"THEN(1,3)"``.
    """
    branch_text = _format_arm_entry_text(
        conditional_id,
        branch_kind,
        model_log,
        include_conditional_reference=include_conditional_reference,
    )
    if not pass_nums:
        return branch_text
    return f"{branch_text}({int_list_to_compact_str(list(pass_nums))})"


def _format_arm_entry_text(
    conditional_id: int,
    branch_kind: str,
    model_log: "ModelLog",
    include_conditional_reference: bool = False,
) -> str:
    """Format one arm-entry label as plain text.

    Args:
        conditional_id:
            Dense conditional id.
        branch_kind:
            Branch kind such as ``"then"`` or ``"elif_2"``.
        model_log:
            Owning model log containing conditional metadata.
        include_conditional_reference:
            Whether to append ``@L...`` to identify the conditional event.

    Returns
    -------
    str
        Plain-text arm label.
    """
    branch_text = _format_branch_kind_text(branch_kind)
    if not include_conditional_reference:
        return branch_text
    return f"{branch_text}@{_get_conditional_reference_text(conditional_id, model_log)}"


def _format_branch_kind_text(branch_kind: str) -> str:
    """Format a branch-kind token as display text.

    Args:
        branch_kind:
            Stored branch kind such as ``"then"``, ``"elif_1"``, or ``"else"``.

    Returns
    -------
    str
        Display label such as ``"THEN"`` or ``"ELIF 1"``.

    Raises
    ------
    ValueError
        If ``branch_kind`` is not recognized.
    """
    if branch_kind == "then":
        return "THEN"
    if branch_kind == "else":
        return "ELSE"
    if branch_kind.startswith("elif_"):
        return f"ELIF {int(branch_kind.split('_', 1)[1])}"
    raise ValueError(f"Unrecognized branch kind: {branch_kind}")


def _get_conditional_reference_text(conditional_id: int, model_log: "ModelLog") -> str:
    """Return a readable conditional identifier for composite edge labels.

    Args:
        conditional_id:
            Dense conditional id.
        model_log:
            Owning model log containing conditional metadata.

    Returns
    -------
    str
        Line-based conditional reference when available, otherwise ``"C{id}"``.
    """
    for conditional_event in model_log.conditional_events:
        if conditional_event.id == conditional_id:
            return f"L{conditional_event.if_stmt_span[0]}"
    return f"C{conditional_id}"


def _arm_entry_sort_key(
    conditional_id: int,
    branch_kind: str,
    model_log: "ModelLog",
) -> Tuple[int, int, int]:
    """Return a stable sort key for multi-arm edge labels.

    Args:
        conditional_id:
            Dense conditional id.
        branch_kind:
            Branch kind such as ``"then"`` or ``"elif_2"``.
        model_log:
            Owning model log containing conditional metadata.

    Returns
    -------
    Tuple[int, int, int]
        Sort key ordered by source line, branch rank, then conditional id.
    """
    source_line = 10**9
    for conditional_event in model_log.conditional_events:
        if conditional_event.id == conditional_id:
            source_line = conditional_event.if_stmt_span[0]
            break
    return (source_line, _branch_kind_sort_key(branch_kind), conditional_id)


def _branch_kind_sort_key(branch_kind: str) -> int:
    """Return an ordering key for branch kinds.

    Args:
        branch_kind:
            Stored branch kind such as ``"then"``, ``"elif_1"``, or ``"else"``.

    Returns
    -------
    int
        Sort rank for the branch kind.
    """
    if branch_kind == "then":
        return 0
    if branch_kind.startswith("elif_"):
        return int(branch_kind.split("_", 1)[1])
    if branch_kind == "else":
        return 10**6
    return 10**9


def _edge_is_conditional_branch(
    parent_node: Union["LayerPassLog", "LayerLog"],
    child_node: Union["LayerPassLog", "LayerLog"],
    model_log: "ModelLog",
    vis_mode: str,
) -> bool:
    """Return True when an edge is an ``IF`` branch-entry edge.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        model_log:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    bool
        True when the edge appears in ``conditional_branch_edges``.
    """
    if vis_mode == "unrolled":
        return (
            parent_node.layer_label,
            child_node.layer_label,
        ) in model_log.conditional_branch_edges
    if vis_mode == "rolled":
        edge_key = (parent_node.layer_label_no_pass, child_node.layer_label_no_pass)
        return any(
            (branch_parent.split(":")[0], branch_child.split(":")[0]) == edge_key
            for branch_parent, branch_child in model_log.conditional_branch_edges
        )
    raise ValueError(f"vis_mode must be 'unrolled' or 'rolled', not {vis_mode}")


def _format_branch_edge_label_html(label_text: str) -> str:
    """Wrap plain branch-label text in the Graphviz HTML used by TorchLens.

    Args:
        label_text:
            Plain text to display on the edge.

    Returns
    -------
    str
        Graphviz HTML edge-label string.
    """
    return f'<<FONT POINT-SIZE="18"><b><u>{label_text}</u></b></FONT>>'


def _label_node_arguments_if_needed(
    self: "ModelLog",
    parent_node: Union["LayerPassLog", "LayerLog"],
    child_node: Union["LayerPassLog", "LayerLog"],
    edge_dict: Dict,
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
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
        edge_dict: Mutable dict of edge attributes; ``"headlabel"`` or ``"xlabel"``
            may be added.
        show_buffer_layers: Buffer visibility mode (affects parent count).
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
    _set_argument_edge_label(edge_dict, arg_label)


def _set_argument_edge_label(edge_dict: Dict, arg_label: str) -> None:
    """Attach an argument-position label without overwriting semantic edge labels.

    Args:
        edge_dict:
            Mutable Graphviz edge attribute dict.
        arg_label:
            HTML label string describing edge argument positions.
    """
    if "headlabel" not in edge_dict:
        edge_dict["headlabel"] = arg_label
        return
    if "xlabel" not in edge_dict:
        edge_dict["xlabel"] = arg_label
        return
    if edge_dict["xlabel"] == arg_label:
        return
    edge_dict["xlabel"] = edge_dict["xlabel"][:-1] + "<br/>" + arg_label[1:]


def _should_mark_arguments_on_edge(
    self: "ModelLog",
    child_node: Union["LayerPassLog", "LayerLog"],
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument position labels should be shown on the edge to child_node.

    Skips commutative functions (add, mul, cat, eq, ne) where arg order is
    interchangeable -- showing "arg 0" vs "arg 1" would be misleading.
    For non-commutative ops, labels are shown when the child has multiple
    visible parents.

    Args:
        child_node: The child node whose incoming edge is being considered.
        show_buffer_layers: Buffer visibility mode.
    """
    # Commutative ops: argument order doesn't matter, skip labels.
    if child_node.layer_type in COMMUTE_FUNCS:
        return False

    if isinstance(child_node, LayerPassLog):
        return _should_mark_arguments_on_unrolled_edge(self, child_node, show_buffer_layers)
    elif isinstance(child_node, LayerLog):
        return _should_mark_arguments_on_rolled_edge(self, child_node, show_buffer_layers)


def _should_mark_arguments_on_unrolled_edge(
    self: "ModelLog",
    child_node: "LayerPassLog",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument labels should be shown on an unrolled graph edge.

    Args:
        child_node: The child LayerPassLog node whose incoming edge is being considered.
        show_buffer_layers: Buffer visibility mode.
    """
    num_parents_shown = len(child_node.parent_layers)

    if show_buffer_layers != "always":
        num_parents_shown -= sum(
            [
                int(
                    self[parent].is_buffer_layer
                    and not _is_buffer_visible(self[parent], show_buffer_layers)
                )
                for parent in child_node.parent_layers
            ]
        )

    if num_parents_shown > 1:
        return True
    else:
        return False


def _should_mark_arguments_on_rolled_edge(
    self: "ModelLog",
    child_node: "LayerLog",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument labels should be shown on a rolled graph edge.

    Args:
        child_node: The child LayerLog node whose incoming edge is being considered.
        show_buffer_layers: Buffer visibility mode.
    """
    for pass_num, pass_parents in child_node.parent_layers_per_pass.items():
        num_parents_shown = len(pass_parents)
        if show_buffer_layers != "always":
            num_parents_shown -= sum(
                [
                    int(
                        self.layer_logs[parent].is_buffer_layer
                        and not _is_buffer_visible(self.layer_logs[parent], show_buffer_layers)
                    )
                    for parent in pass_parents
                ]
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


def _get_lowest_containing_module_for_two_render_nodes(
    node1: GraphNode,
    node2: GraphNode,
    both_nodes_collapsed_modules: bool,
    vis_nesting_depth: int,
) -> Union[str, int]:
    """Find the deepest module subgraph for render nodes including boundaries."""

    return _get_lowest_containing_module_for_two_nodes(
        cast(Union["LayerPassLog", "LayerLog"], node1),
        cast(Union["LayerPassLog", "LayerLog"], node2),
        both_nodes_collapsed_modules,
        vis_nesting_depth,
    )


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
    - ``is_leaf_module_output`` nodes are adjusted to their parent
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
    node1_modules = node1.containing_modules[:]
    node2_modules = node2.containing_modules[:]

    if isinstance(node1, LayerLog) or isinstance(node2, LayerLog):
        node1_modules = [module.split(":")[0] for module in node1_modules]
        node2_modules = [module.split(":")[0] for module in node2_modules]

    if node1.is_leaf_module_output:
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
        if node1.is_leaf_module_output and (len(node1_modules) == 1):
            return -1
        elif node1.is_leaf_module_output and (len(node1_modules) > 1):
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
    if parent_layer.has_gradient and child_layer.has_gradient:
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
    vis_mode: str,
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
        vis_mode: ``'rolled'`` or ``'unrolled'``.
        module_edge_dict: Dict mapping each module cluster name to
            ``{"edges": [...], "has_input_ancestor": bool}``.
        overrides: Graphviz attribute overrides for module subgraphs.
    """
    if vis_mode == "unrolled":
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
            vis_mode,
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
    vis_mode,
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
        vis_mode: ``'rolled'`` or ``'unrolled'``.
        overrides: Graphviz attribute overrides.
    """
    subgraph_name_w_pass = parent_graph_list[nesting_depth]
    subgraph_module = subgraph_name_w_pass.split(":")[0]
    if vis_mode == "unrolled":
        cluster_name = f"cluster_{subgraph_name_w_pass.replace(':', '_pass')}"
        subgraph_name = subgraph_name_w_pass
    elif vis_mode == "rolled":
        cluster_name = f"cluster_{subgraph_module}"
        subgraph_name = subgraph_module
    else:
        raise ValueError("vis_mode must be 'rolled' or 'unrolled'")
    sg_ml = self.modules[subgraph_module]
    module_type = sg_ml.module_class_name  # type: ignore[union-attr]
    if (sg_ml.num_passes > 1) and (vis_mode == "unrolled"):  # type: ignore[union-attr]
        subgraph_title = subgraph_name_w_pass
    elif (sg_ml.num_passes > 1) and (vis_mode == "rolled"):  # type: ignore[union-attr]
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
                vis_mode,
                overrides,
            )

    else:  # Leaf of this branch: create the subgraph and add all edges.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            # Penwidth + cluster attrs come from ``_render_utils`` so the
            # bundle renderer in ``multi_trace/visualization.py`` can build
            # equivalent clusters with the same formula and label format.
            pen_width = compute_module_penwidth(nesting_depth, max_nesting_depth)
            if module_edge_dict[subgraph_name]["has_input_ancestor"]:
                line_style = "solid"
            else:
                line_style = "dashed"

            # ``title_already_escaped=True`` preserves the existing byte-level
            # output: ModelLog subgraph titles never contain HTML specials, so
            # the title is fed through verbatim, matching the legacy format.
            module_args = make_module_cluster_attrs(
                title=subgraph_title,
                module_type=module_type,
                line_style=line_style,
                penwidth=pen_width,
                title_already_escaped=True,
            )

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

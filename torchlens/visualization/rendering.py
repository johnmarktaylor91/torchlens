"""Graphviz-based computational graph rendering for Trace objects.

Renders the computational graph captured by TorchLens as a Graphviz Digraph,
supporting two visualization modes:

- **unrolled** (default): every pass of every layer is a separate node.
  Uses ``layer_dict_main_keys`` as the node source.
- **rolled**: layers with multiple ops are collapsed into a single node
  with edge labels showing which ops an edge applies to.  Uses
  ``layer_logs`` (LayerLog objects) as the node source.

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

- **_layers_logged guard**: rendering requires all layers to be present
  in the Trace (either saved or kept-unsaved).  This check prevents
  IndexError crashes when ``keep_unsaved_layers=False`` was used and nodes
  reference absent layers.
"""

import copy
import os
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
from PIL import Image

from .._literals import (
    BufferVisibilityLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from ..data_classes.internal_types import VisualizationOverrides
from ..utils.display import in_notebook, int_list_to_compact_str, _vprint
from ..data_classes.op_log import OpLog
from ..data_classes.layer_log import LayerLog
from ..viz import batch_summary
from .modes import COLLAPSED_MODE_REGISTRY, DOMAIN_NODE_MODES, MODE_REGISTRY
from .node_spec import (
    INTERVENTION_HOOK_BORDER_COLOR,
    INTERVENTION_HOOK_FILL_COLOR,
    NodeSpec,
    graphviz_graph_overrides,
    intervention_graph_override,
    intervention_site_and_cone_labels,
    make_intervention_node_spec_fn,
    render_lines_to_html,
)
from .overlays import OverlayScores, overlay_border_attrs, overlay_line
from .themes import (
    VisualizationTheme,
    apply_theme_to_spec,
    legend_lines,
    resolve_theme,
    theme_edge_attrs,
    theme_graph_attrs,
    theme_node_attrs,
)
from .code_panel import CodePanelOption, render_code_panel_subgraph, resolve_code_panel_source
from ._render_utils import (
    compute_module_penwidth,
    direction_to_rankdir,
    make_module_cluster_attrs,
)

if TYPE_CHECKING:
    from ..data_classes.grad_fn_log import GradFnLog
    from ..data_classes.model_log import Trace
    from ..data_classes.module_log import ModuleLog

BaseGraphNode = Union["OpLog", "LayerLog"]


@dataclass
class FocusNode:
    """Mutable render proxy for a focused LayerLog or OpLog.

    Parameters
    ----------
    original:
        Source graph node whose metadata should be rendered.
    parents:
        Focus-rewritten incoming labels.
    children:
        Focus-rewritten outgoing labels.
    modules:
        Copied module path for cluster placement.
    """

    original: BaseGraphNode
    parents: list[str]
    children: list[str]
    modules: list[str]

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
    children:
        Outgoing rendered labels.
    parents:
        Incoming rendered labels.
    modules:
        Module path used for cluster placement.
    """

    layer_label: str
    display_label: str
    boundary_kind: str
    children: list[str]
    parents: list[str]
    modules: list[str]
    is_buffer: bool = False
    has_input_ancestor: bool = True
    is_final_output: bool = False
    is_atomic_module_op: bool = False
    output_of_modules: list[str] = field(default_factory=list)
    is_input: bool = False
    is_output: bool = False
    is_terminal_bool: bool = False
    uses_params: bool = False
    num_param_tensors: int = 0
    _param_logs: list[Any] = field(default_factory=list)
    param_shapes: list[tuple[Any, ...]] = field(default_factory=list)
    num_calls: int = 1
    call_index: int = 1
    type_index: int = 1
    trace_index: int = 1
    shape: tuple[Any, ...] = ()
    memory_str: str = "0 B"
    io_role: str = ""
    layer_type: str = "input"

    def __post_init__(self) -> None:
        """Fill mutable defaults and role flags."""

        self.is_input = self.boundary_kind == "input"
        self.is_output = self.boundary_kind == "output"
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
# bundle and Trace clusters scale identically by depth.

# Commutative functions: argument order doesn't matter, so we skip arg-position
# labels on their incoming edges to reduce visual clutter.
COMMUTE_FUNCS = ["add", "mul", "cat", "eq", "ne"]


def _is_headless_linux() -> bool:
    """Return whether the current process is a Linux shell without a display.

    Returns
    -------
    bool
        ``True`` when running on Linux with no ``DISPLAY`` environment
        variable, which means Graphviz viewer dispatch would usually fall
        through to a failing ``xdg-open`` call.
    """

    return sys.platform.startswith("linux") and not os.environ.get("DISPLAY")


def _view_rendered_file(filepath: str) -> None:
    """Open a rendered visualization file when a local viewer is available.

    Parameters
    ----------
    filepath:
        Rendered artifact path.
    """

    if _is_headless_linux():
        print(f"[headless detected; saved to {filepath}; pass vis_save_only=False to force open]")
        return

    try:
        graphviz.backend.viewing.view(filepath)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[viewer open failed; saved to {filepath}: {exc}]")


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
    if not source_node.is_buffer:
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

    if not node.is_buffer:
        return True
    if show_buffer_layers == "always":
        return True
    if show_buffer_layers == "never":
        return False
    return not _is_noise_buffer(node)


def _get_hidden_parent_buffer_addresses(
    trace: "Trace",
    node: GraphNode,
    show_buffer_layers: BufferVisibilityLiteral,
) -> list[str]:
    """Return hidden buffer addresses attached as parents of ``node``.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Non-buffer node to inspect.
    show_buffer_layers:
        Canonical tri-state visibility mode.

    Returns
    -------
    list[str]
        Hidden buffer addresses in parent order, de-duplicated.
    """

    if show_buffer_layers == "always" or node.is_buffer:
        return []

    hidden_addresses: list[str] = []
    seen_addresses: set[str] = set()
    source_node = _unwrap_focus_node(node)
    for parent_label in node.parents:
        if parent_label.startswith("__module_focus_"):
            continue
        parent_node: BaseGraphNode
        if isinstance(source_node, OpLog):
            parent_node = trace[parent_label]
        else:
            parent_node = trace.layer_logs[parent_label]
        if not parent_node.is_buffer or _is_buffer_visible(parent_node, show_buffer_layers):
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


def draw(
    self: "Trace",
    vis_mode: VisModeLiteral = "unrolled",
    vis_call_depth: int = 1000,
    vis_outpath: str = "modelgraph",
    vis_graph_overrides: Optional[Dict[str, Any]] = None,
    module: "ModuleLog | str | None" = None,
    node_mode: VisNodeModeLiteral = "default",
    node_spec_fn: NodeSpecFn | None = None,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    collapse_fn: CollapseFn | None = None,
    skip_fn: SkipFn | None = None,
    vis_edge_overrides: Optional[Dict[str, Any]] = None,
    vis_grad_edge_overrides: Optional[Dict[str, Any]] = None,
    vis_module_overrides: Optional[Dict[str, Any]] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
    direction: VisDirectionLiteral = "bottomup",
    vis_node_placement: VisNodePlacementLiteral = "auto",
    vis_renderer: VisRendererLiteral = "graphviz",
    vis_theme: str = "torchlens",
    vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
    vis_show_cone: bool = True,
    code_panel: CodePanelOption = False,
    node_overlay: str | OverlayScores | None = None,
    node_label_fields: list[str] | None = None,
    show_legend: bool = False,
    font_size: int | None = None,
    dpi: int | None = None,
    for_paper: bool = False,
    return_graph: bool = False,
) -> Any:
    """Render the computational graph as a Graphviz Digraph.

    Orchestrates the full rendering pipeline:
    1. Validates that all layers are logged (``_layers_logged`` guard).
    2. Iterates over entries_to_plot, building nodes and edges.
    3. Groups edges into module subgraph clusters.
    4. Renders to file and optionally displays.

    Args:
        vis_mode: ``'unrolled'`` (each pass is a separate node) or ``'rolled'``
            (multi-pass layers collapsed into one node with pass annotations).
        vis_call_depth: Maximum module nesting levels to show before
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
            the rendered OpLog.
        collapsed_node_spec_fn: Optional callback receiving
            ``(module_log, default_spec)`` for collapsed module nodes.
        collapse_fn: Optional predicate receiving a ModuleLog. When provided,
            it replaces ``vis_call_depth`` collapse decisions.
        skip_fn: Optional predicate receiving a LayerLog. Skipped nodes are
            elided and edges are chained through them.
        vis_edge_overrides: Overrides for forward edges.
        vis_grad_edge_overrides: Overrides for backward (grad) edges.
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
            or ``'sfdp'``. ``'elk'`` remains accepted as an internal backend escape
            hatch; the public API default is ``'auto'``.
        vis_renderer: Renderer backend: ``'graphviz'`` or experimental
            ``'dagua'``. Import ``torchlens.experimental.dagua`` before using
            the Dagua renderer.
        vis_theme: Renderer theme name for backends that support themes.
        vis_intervention_mode: Intervention overlay mode. ``"node_mark"``
            marks sites and cones; ``"as_node"`` inserts hook nodes after
            intervention sites.
        vis_show_cone: Whether ``"node_mark"`` mode marks downstream cone
            members.
        code_panel: Optional source-code panel. ``True`` is equivalent to
            ``"forward"``; callable values receive the live model object when
            it is still available.
        node_overlay: Built-in overlay name or external mapping from node label
            to score. Supported built-ins include ``"flops"``, ``"time"``,
            ``"bytes"``, ``"magnitude"``, ``"grad_norm"``, ``"nan"``,
            ``"intervention"``, and ``"bundle_delta"``.
        node_label_fields: Optional label field picker. When omitted, the
            default TorchLens label rows are used.
        show_legend: Whether to render a compact colorblind-safe legend with
            the graph.
        font_size: Optional Graphviz font size.
        dpi: Optional Graphviz output DPI.
        for_paper: Whether to force the paper theme preset.
        return_graph: If True, return the underlying ``graphviz.Digraph`` on
            the Graphviz path or DOT text for direct text renderers.

    Returns:
        The Graphviz DOT source string.

    Raises:
        ValueError: If ``_layers_logged`` is False (layers were discarded
            by ``keep_unsaved_layers=False``).
    """
    if node_mode not in MODE_REGISTRY:
        raise ValueError(
            "Visualization node_style/node_mode must be one of 'default', "
            "'profiling', 'vision', or 'attention'."
        )
    if node_mode in DOMAIN_NODE_MODES:
        warnings.warn(
            f"node_style={node_mode!r} is moving out of core; use the equivalent "
            f"recipe at examples/recipes/{node_mode}.py or wait for the "
            f"torchlens.{node_mode} plugin",
            DeprecationWarning,
            stacklevel=2,
        )
    if vis_intervention_mode not in {"node_mark", "as_node"}:
        raise ValueError("vis_intervention_mode must be either 'node_mark' or 'as_node'.")
    show_buffer_layers = _normalize_buffer_visibility(show_buffer_layers)
    site_labels, _ = intervention_site_and_cone_labels(self, show_cone=vis_show_cone)
    intervention_node_spec_fn = make_intervention_node_spec_fn(
        self,
        show_cone=vis_show_cone,
        graph_overrides=vis_graph_overrides,
        user_node_spec_fn=node_spec_fn,
    )

    if vis_renderer == "dagua":
        opted_in_module = sys.modules.get("torchlens.experimental.dagua")
        if not getattr(opted_in_module, "__torchlens_dagua_opted_in__", False):
            raise RuntimeError(
                "dagua renderer is experimental; opt in via "
                "`from torchlens.experimental import dagua` first"
            )
        from ..experimental.dagua import render_trace_with_dagua

        return render_trace_with_dagua(
            self,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            vis_outpath=vis_outpath,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            vis_buffers=show_buffer_layers == "always",
            vis_direction=direction,
            vis_theme=vis_theme,
        )
    if vis_renderer not in {"graphviz", "dagua"}:
        raise ValueError("vis_renderer must be 'graphviz' or 'dagua'")
    theme = resolve_theme(vis_theme, for_paper=for_paper)
    if node_overlay is None:
        node_overlay = getattr(self, "_node_overlay_scores", None)

    overrides = VisualizationOverrides(
        graph=graphviz_graph_overrides(vis_graph_overrides),
        edge=vis_edge_overrides or {},
        grad_edge=vis_grad_edge_overrides or {},
        module=vis_module_overrides or {},
    )

    # THE _layers_logged guard: prevents IndexError crashes that would
    # occur when edges reference layers that were discarded by
    # keep_unsaved_layers=False.  This is the single chokepoint that
    # protects all downstream rendering code from missing-layer lookups.
    if not self._layers_logged:
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

    # Unrolled: iterate OpLog objects (one node per pass).
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
    from ._elk_internal.layout import get_node_placement_engine

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
    if vis_intervention_mode == "as_node" and engine == "elk":
        engine = "dot"
    if source_text is not None and engine == "elk":
        # The code panel is implemented in pure Graphviz so the graph and source
        # remain in one output file. ELK's direct renderer byops Digraph
        # construction, so panel renders stay on the Graphviz path.
        engine = "dot"
    _vprint(self, f"Rendering {vis_mode} graph ({num_nodes} nodes, format={vis_fileformat})")
    _vprint(self, f"Layout engine: {engine}")

    if self.num_params == 0:
        params_detail = "0 params"
    elif self.num_params_frozen == 0:
        params_detail = f"{self.num_params} params (all trainable, {self.param_memory_str})"
    elif self.num_params_trainable == 0:
        params_detail = f"{self.num_params} params (all frozen, {self.param_memory_str})"
    else:
        params_detail = (
            f"{self.num_params} params "
            f"({self.num_params_trainable}/{self.num_params} trainable, "
            f"{self.param_memory_str})"
        )

    graph_caption = (
        f"<<B>{self.model_name}</B><br align='left'/>{self.num_tensors} "
        f"tensors total ({self.total_out_memory_str})"
        f"<br align='left'/>{params_detail}<br align='left'/>>"
    )
    if getattr(self, "_has_direct_writes", False):
        graph_caption = graph_caption[:-2] + (
            "Direct writes detected - recipe propagation will overlay<br align='left'/>>"
        )

    # ELK fast path: skip graphviz.Digraph construction entirely.
    # Generates DOT directly with ELK positions and cluster subgraphs (module boxes).
    # If ELK layout fails (OOM, timeout), render_elk_direct falls back internally
    # to sfdp — still using the fast DOT-text path, never graphviz.Digraph.
    if engine == "elk":
        from ._elk_internal.layout import render_elk_direct

        result = render_elk_direct(
            self,
            entries_to_plot,
            vis_mode,
            vis_call_depth,
            show_buffer_layers == "always",
            overrides,
            node_mode,
            intervention_node_spec_fn,
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
    graph_args.update(theme_graph_attrs(theme, font_size=font_size, dpi=dpi))

    # Override system: callers can pass dicts of Graphviz attributes to
    # customize rendering.  Values can be static (str) or dynamic (callable
    # receiving the Trace, evaluated at render time).
    for arg_name, arg_val in overrides.graph.items():  # type: ignore[union-attr]
        if callable(arg_val):
            graph_args[arg_name] = str(arg_val(self))
        else:
            graph_args[arg_name] = str(arg_val)

    dot.graph_attr.update(graph_args)
    dot.node_attr.update({"ordering": "out", **theme_node_attrs(theme, font_size=font_size)})
    dot.edge_attr.update(theme_edge_attrs(theme, font_size=font_size))

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
    edges_used: Set[tuple[str, str]] = set()

    for node_barcode, node in entries_to_plot.items():
        if node.layer_label in skipped_labels:
            continue
        if node.is_buffer and not _is_buffer_visible(node, show_buffer_layers):
            continue
        _add_node_to_graphviz(
            self,
            node,
            dot,
            module_cluster_dict,
            edges_used,
            vis_mode,
            collapsed_modules,
            vis_call_depth,
            show_buffer_layers,
            overrides,
            node_mode,
            intervention_node_spec_fn,
            collapsed_node_spec_fn,
            collapse_fn,
            edge_map,
            vis_intervention_mode,
            site_labels,
            theme,
            node_overlay,
            node_label_fields,
        )

    if vis_intervention_mode == "as_node":
        _add_intervention_hook_nodes(dot, site_labels, vis_graph_overrides)

    # Finally, set up the subgraphs.
    _setup_subgraphs(self, dot, vis_mode, module_cluster_dict, overrides)
    if show_legend:
        _add_legend_to_graphviz(dot, theme)
    if source_text is not None:
        render_code_panel_subgraph(dot, source_text)

    if in_notebook() and not vis_save_only:
        try:
            from IPython.display import display  # #72: lazy import
        except ImportError as e:
            raise ImportError(
                "IPython is required for this feature. Install with "
                "`pip install torchlens[notebook]`."
            ) from e

        display_fn = cast(Any, display)
        display_fn(dot)

    # ELK was already handled above (early return). Only dot/sfdp reach here.
    from ._elk_internal.layout import render_with_sfdp

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
                _view_rendered_file(rendered_path)
        _vprint(self, f"Graph saved to {vis_outpath}.{vis_fileformat}")
    except subprocess.TimeoutExpired:
        warnings.warn(
            f"Graphviz render timed out ({_RENDER_TIMEOUT}s) for graph with "
            f"{self.num_tensors} nodes. DOT source saved to "
            f"'{source_path}'. Consider using vis_node_placement='sfdp' or "
            f"vis_call_depth to collapse modules."
        )
    except subprocess.CalledProcessError as e:
        warnings.warn(f"Graphviz render failed: {e.stderr.decode()}")
    finally:
        import os

        if os.path.exists(source_path):
            os.remove(source_path)
    if return_graph:
        return dot
    return dot.source


def _add_legend_to_graphviz(dot: graphviz.Digraph, theme: VisualizationTheme) -> None:
    """Add a compact color legend subgraph to a Graphviz graph.

    Parameters
    ----------
    dot:
        Graphviz graph being rendered.
    theme:
        Resolved visualization theme.
    """

    with dot.subgraph(name="cluster_torchlens_legend") as legend:
        legend.attr(
            label="TorchLens legend",
            labelloc="t",
            color=theme.default_border,
            fontcolor=theme.default_font,
            style="rounded",
        )
        for index, line in enumerate(legend_lines(theme)):
            label, color = line.split(": ", 1)
            legend.node(
                f"tl_legend_{index}",
                label=label,
                shape="box",
                style="filled,rounded",
                fillcolor=color,
                fontcolor="black",
                color=theme.default_border,
            )


def render_backward_graph(
    self: "Trace",
    vis_outpath: str = "backward_modelgraph",
    vis_graph_overrides: Optional[Dict[str, Any]] = None,
    node_spec_fn: BackwardNodeSpecFn | None = None,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    vis_node_mode: VisNodeModeLiteral = "default",
    vis_edge_overrides: Optional[Dict[str, Any]] = None,
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
        Trace containing captured backward metadata.
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

    if not self.has_backward_pass or not self.grad_fn_logs:
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
        f"<br align='left'/>{self.backward_num_calls} backward pass(es)<br align='left'/>>"
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
        try:
            from IPython.display import display
        except ImportError as e:
            raise ImportError(
                "IPython is required for this feature. Install with "
                "`pip install torchlens[notebook]`."
            ) from e

        display_fn = cast(Any, display)
        display_fn(dot)

    _RENDER_TIMEOUT = 120
    source_path = dot.save(vis_outpath)
    try:
        rendered_path = f"{vis_outpath}.{vis_fileformat}"
        cmd = [dot.engine, f"-T{vis_fileformat}", "-o", rendered_path, source_path]
        subprocess.run(cmd, timeout=_RENDER_TIMEOUT, check=True, capture_output=True)
        if not vis_save_only:
            _view_rendered_file(rendered_path)
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
    return cast(str, dot.source)


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
    graphviz_graph: graphviz.Digraph,
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
    if grad_fn.has_op:
        title = f"[i] {title}"
    if grad_fn.is_custom:
        title = f"{title} [custom]"

    lines = [title]
    if grad_fn.op is not None:
        lines.append(f"@{grad_fn.op.layer_label}")
    lines.append(f"grad {_format_backward_output_shape(grad_fn)}")
    return lines


def _format_backward_output_shape(grad_fn: "GradFnLog") -> str:
    """Return the first captured output-grad shape for a grad_fn.

    Parameters
    ----------
    grad_fn:
        GradFnLog to inspect.

    Returns
    -------
    str
        Compact shape string, or ``"N/A"`` when no tensor was captured
        (typical for intervening grad_fns that have no forward counterpart).
    """

    for grad_fn_pass in reversed(list(grad_fn.ops.values())):
        tensor = _first_tensor_in_obj(grad_fn_pass.grad_outputs)
        if tensor is not None:
            return _format_shape_str(tuple(tensor.shape))
    return "N/A"


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
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    *,
    vis_mode: str,
    show_buffer_layers: BufferVisibilityLiteral,
    skip_fn: SkipFn | None,
) -> tuple[dict[str, list[RenderEdge]], set[str]]:
    """Build skip-aware outgoing edges for each rendered node.

    Parameters
    ----------
    trace:
        Owning Trace.
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
        if not node.is_buffer or _is_buffer_visible(node, show_buffer_layers)
    }
    skipped_labels: set[str] = set()
    if skip_fn is not None:
        for node in visible_entries.values():
            if isinstance(node, BoundaryNode):
                continue
            layer_log = _layer_log_for_node(trace, node)
            if not skip_fn(layer_log):
                continue
            if layer_log.is_input or layer_log.is_output:
                raise ValueError(
                    f"skip_fn cannot skip input or output layer '{layer_log.layer_label}'."
                )
            skipped_labels.add(node.layer_label)

    edge_map: dict[str, list[RenderEdge]] = {}
    for node in visible_entries.values():
        if node.layer_label in skipped_labels:
            continue
        edge_map[node.layer_label] = _expand_edges_through_skipped(
            trace,
            node,
            visible_entries,
            skipped_labels,
            vis_mode,
        )
    return edge_map, skipped_labels


def _resolve_focus_module(
    trace: "Trace",
    module: "ModuleLog | str",
) -> "ModuleLog":
    """Resolve and validate a module focus argument.

    Parameters
    ----------
    trace:
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
        If the module cannot be found or belongs to a different Trace.
    """

    from ..data_classes.module_log import ModuleLog

    if isinstance(module, str):
        if module not in trace.modules:
            raise ValueError(f"Module address '{module}' was not found in this Trace.")
        resolved = trace.modules[module]
        if not isinstance(resolved, ModuleLog):
            raise ValueError(
                f"Module address '{module}' resolved to a module pass, not a ModuleLog."
            )
        return resolved
    if not isinstance(module, ModuleLog):
        raise ValueError("module must be a ModuleLog, module address string, or None.")
    if module._source_trace is not trace:
        raise ValueError("ModuleLog focus must belong to the Trace being rendered.")
    return module


def _build_module_focus_entries(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    target_module: "ModuleLog",
    *,
    vis_mode: str,
) -> dict[str, GraphNode]:
    """Return render entries focused on one module plus synthetic boundaries.

    Parameters
    ----------
    trace:
        Trace being rendered.
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
        for parent_label in node.parents:
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
            if node.layer_label not in boundary.children:
                boundary.children.append(node.layer_label)
            new_parents.append(boundary.layer_label)
        node.parents = new_parents

        new_children: list[str] = []
        for child_label in node.children:
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
            if node.layer_label not in boundary.parents:
                boundary.parents.append(node.layer_label)
            new_children.append(boundary.layer_label)
        node.children = new_children

    _simplify_boundary_labels(input_boundaries, "input")
    _simplify_boundary_labels(output_boundaries, "output")
    for boundary_dict in (input_boundaries, output_boundaries):
        for label, boundary in boundary_dict.items():
            focused_entries[label] = boundary

    return focused_entries


def _node_is_inside_module(node: GraphNode, address: str) -> bool:
    """Return whether ``node`` ran inside ``address``."""

    return any(module.split(":", 1)[0] == address for module in node.modules)


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
        parents=list(node.parents),
        children=list(node.children),
        modules=list(node.modules),
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
        children=[] if child_label is None else [child_label],
        parents=[] if parent_label is None else [parent_label],
        modules=module_path,
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
    trace: "Trace",
    parent_node: GraphNode,
    visible_entries: dict[str, GraphNode],
    skipped_labels: set[str],
    vis_mode: str,
) -> list[RenderEdge]:
    """Expand one node's outgoing edges through skipped successor chains.

    Parameters
    ----------
    trace:
        Owning Trace.
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
    for child_label in parent_node.children:
        child_node = visible_entries.get(child_label)
        if child_node is None:
            continue
        reached = _walk_skipped_successors(
            trace,
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
    trace: "Trace",
    node: GraphNode,
    visible_entries: dict[str, GraphNode],
    skipped_labels: set[str],
    vis_mode: str,
    seen: set[str],
) -> list[GraphNode]:
    """Return non-skipped descendants reached through skipped chains.

    Parameters
    ----------
    trace:
        Owning Trace.
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
    for child_label in node.children:
        child_node = visible_entries.get(child_label)
        if child_node is None:
            continue
        reached.extend(
            _walk_skipped_successors(
                trace,
                child_node,
                visible_entries,
                skipped_labels,
                vis_mode,
                seen=set(seen),
            )
        )
    return reached


def _get_node_by_label(trace: "Trace", label: str, vis_mode: str) -> GraphNode:
    """Return a render node by label for the active visualization mode."""

    if vis_mode == "unrolled":
        return trace.layer_dict_main_keys[label]
    if vis_mode == "rolled":
        return trace.layer_logs[label]
    raise ValueError(f"vis_mode must be 'unrolled' or 'rolled', not {vis_mode}")


def _add_node_to_graphviz(
    self: "Trace",
    node: GraphNode,
    graphviz_graph: graphviz.Digraph,
    module_edge_dict: Dict[str, Any],
    edges_used: Set[tuple[str, str]],
    vis_mode: str,
    collapsed_modules: Set[str],
    vis_call_depth: int = 1000,
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
    overrides: Optional[VisualizationOverrides] = None,
    node_mode: VisNodeModeLiteral = "default",
    node_spec_fn: NodeSpecFn | None = None,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    collapse_fn: CollapseFn | None = None,
    edge_map: Optional[dict[str, list[RenderEdge]]] = None,
    vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
    intervention_site_labels: set[str] | None = None,
    theme: VisualizationTheme | None = None,
    node_overlay: str | OverlayScores | None = None,
    node_label_fields: list[str] | None = None,
) -> None:
    """Adds a node and its relevant edges to the graphviz figure.

    Args:
        node: node to add
        graphviz_graph: The graphviz object to add the node to.
        module_edge_dict: Dictionary of the module clusters.
        vis_mode: Whether to roll the graph or not
        vis_call_depth: How many levels of nested modules to show
        collapsed_modules: Labels of collapsed module nodes that have been made so far.
        show_buffer_layers: Buffer visibility mode.
        overrides: Graphviz attribute overrides for nodes, edges, etc.
    """
    collapse_address = _collapse_address_for_node(
        self,
        node,
        collapse_fn=collapse_fn,
        max_module_depth=vis_call_depth,
    )
    is_collapsed_module = collapse_address is not None

    if is_collapsed_module:
        _build_collapsed_module_node(
            self,
            node,
            graphviz_graph,
            collapsed_modules,
            vis_mode,
            vis_call_depth,
            collapse_address,
            overrides,  # type: ignore[arg-type]
            node_mode,
            collapsed_node_spec_fn,
            theme,
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
            theme,
            node_overlay,
            node_label_fields,
        )

    _add_edges_for_node(
        self,
        node,
        is_collapsed_module,
        vis_call_depth,
        node_color,
        module_edge_dict,
        edges_used,
        graphviz_graph,
        vis_mode,
        show_buffer_layers,
        overrides,
        collapse_fn,
        edge_map,
        vis_intervention_mode,
        intervention_site_labels,
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


def _collapse_address_for_node(
    trace: "Trace",
    node: GraphNode,
    *,
    collapse_fn: CollapseFn | None,
    max_module_depth: int,
) -> Optional[str]:
    """Return the module-pass address that should absorb ``node``, if any.

    Parameters
    ----------
    trace:
        Owning Trace.
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

    modules = list(node.modules)
    if getattr(node, "is_atomic_module_op", False):
        modules = modules[:-1]
    if not modules:
        return None

    if collapse_fn is None:
        if max_module_depth == 0 or len(modules) < max_module_depth:
            return None
            return cast(str, modules[max_module_depth - 1])

    for address_w_pass in modules:
        address = address_w_pass.rsplit(":", 1)[0]
        if _should_collapse_module(
            cast("ModuleLog", trace.modules[address]),
            collapse_fn=collapse_fn,
            max_module_depth=max_module_depth,
        ):
            return str(address_w_pass)
    return None


def _is_collapsed_module(
    node: GraphNode,
    vis_call_depth: int,
    trace: Optional["Trace"] = None,
    collapse_fn: CollapseFn | None = None,
) -> bool:
    """THE IndexError guard for collapsed module rendering.

    Returns True if the node is nested deep enough to be rendered as a
    collapsed ``box3d`` module summary node instead of an individual layer.

    This function is the single decision point that determines whether a node
    gets its own graphviz node or is absorbed into a module box.  Getting this
    wrong causes IndexError when ``_build_collapsed_module_node`` tries to
    access ``modules[vis_call_depth - 1]``.

    Special cases:
    - ``vis_call_depth == 0``: show all layers, never collapse (#94).
    - ``is_atomic_module_op``: the node represents the output of
      its innermost module, so its effective nesting depth is one less (it
      visually "belongs" to the parent scope).

    Args:
        node: The OpLog or LayerLog node to check.
        vis_call_depth: Maximum nesting depth before collapsing into a module box.
    """
    if trace is not None:
        return (
            _collapse_address_for_node(
                trace,
                node,
                collapse_fn=collapse_fn,
                max_module_depth=vis_call_depth,
            )
            is not None
        )
    if vis_call_depth == 0:
        return False  # #94: depth 0 means show all layers, never collapse

    node_call_depth = len(node.modules)
    # Bottom-level submodule outputs are rendered at the parent nesting level,
    # not their own, so subtract 1 from their effective depth.
    if getattr(node, "is_atomic_module_op", False):
        node_call_depth -= 1

    if node_call_depth >= vis_call_depth:
        return True
    else:
        return False


def _build_layer_node(
    self: "Trace",
    node: GraphNode,
    graphviz_graph: graphviz.Digraph,
    show_buffer_layers: BufferVisibilityLiteral,
    vis_mode: str,
    overrides: VisualizationOverrides,
    node_mode: VisNodeModeLiteral,
    node_spec_fn: NodeSpecFn | None = None,
    theme: VisualizationTheme | None = None,
    node_overlay: str | OverlayScores | None = None,
    node_label_fields: list[str] | None = None,
) -> str:
    """Builds and adds a standard (non-collapsed) layer node to the graphviz graph.

    Args:
        node: The OpLog or LayerLog node to render.
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
        if theme is not None:
            spec = apply_theme_to_spec(spec, theme)
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
        lines=compute_default_node_lines(
            node,
            node_address,
            vis_mode,
            node_label_fields=node_label_fields,
            node_overlay=node_overlay,
        ),
        shape=node_shape,
        fillcolor=node_bg_color,
        fontcolor=node_color,
        style=f"filled,{line_style}",
        color=node_color,
        extra_attrs={"ordering": "out"},
    )
    visualizer_path = getattr(node, "visualizer_path", None)
    if isinstance(visualizer_path, str) and visualizer_path.lower().endswith(".png"):
        default_spec = default_spec.replace(
            image=visualizer_path,
            shape="none",
            style="",
            fillcolor=None,
            color=None,
            fontcolor=node_color,
            extra_attrs={
                **default_spec.extra_attrs,
                "imagescale": "true",
                "labelloc": "b",
                "fixedsize": "false",
            },
        )
    if theme is not None:
        default_spec = apply_theme_to_spec(default_spec, theme)
    spec = _apply_node_spec_fn(self, node, default_spec, node_mode, node_spec_fn)

    # Graphviz node names can't contain colons (used for port syntax), so
    # replace ":" with "pass" in pass-qualified labels (e.g., "relu_1:2" -> "relu_1pass2").
    node_args = _node_spec_to_graphviz_args(spec)
    if node.is_input:
        raw_input_attrs = _render_raw_input(
            self,
            getattr(self, "raw_input", None),
            batch_render=getattr(self, "batch_render", "auto"),
        )
        if raw_input_attrs is not None:
            node_args.update(raw_input_attrs)
    elif node.is_output:
        raw_output_attrs = _render_raw_output(getattr(self, "raw_output", None))
        if raw_output_attrs is not None:
            node_args.update(raw_output_attrs)
    node_args["name"] = node.layer_label.replace(":", "pass")
    hidden_buffer_addresses = _get_hidden_parent_buffer_addresses(self, node, show_buffer_layers)
    if hidden_buffer_addresses and not (node.is_input or node.is_output or node.is_buffer):
        node_args["peripheries"] = "2"
        node_args["tooltip"] = f"Hidden buffers: {', '.join(hidden_buffer_addresses)}"
    # Colon in bg_color means it's a grad fill (e.g.,
    # "#D9D9D9:#B0B0B0" for mixed trainable/frozen params).
    # Graphviz requires gradangle to render grads.
    if spec.fillcolor is not None and ":" in spec.fillcolor:
        node_args["gradangle"] = "0"
    node_args.update(overlay_border_attrs(node, node_overlay))

    graphviz_graph.node(**node_args)

    if node.is_final_output:
        with graphviz_graph.subgraph() as s:
            s.attr(rank="sink")
            s.node(node.layer_label.replace(":", "pass"))

    return node_color


def _render_raw_input(
    trace: "Trace",
    value: Any,
    *,
    batch_render: str = "auto",
) -> dict[str, str] | None:
    """Return Graphviz attributes for a renderable raw input value.

    Parameters
    ----------
    trace:
        Trace that owns the rendered input node.
    value:
        Raw user input stored on the owning ``Trace``.
    batch_render:
        Batch rendering policy.

    Returns
    -------
    dict[str, str] | None
        Node attributes to merge into an input-node spec, or ``None`` when the
        default tensor-shape rendering should be used.
    """

    max_items = _batch_render_limit(batch_render)
    if max_items == 0:
        return None
    if isinstance(value, str):
        text = _truncate_raw_input_text(value, limit=80)
        return {
            "label": render_lines_to_html(["input", text]),
            "tooltip": value,
        }
    include_more = batch_render != "first"
    if isinstance(value, torch.Tensor):
        return _render_raw_input_tensor_batch(
            trace,
            value,
            max_items=max_items,
            include_more=include_more,
        )
    sequence = _raw_input_sequence(value)
    if sequence is None:
        return None
    if len(sequence) == 1:
        return _render_raw_input(trace, sequence[0], batch_render="first")
    if all(isinstance(item, str) for item in sequence):
        strings = cast(Sequence[str], sequence)
        if not include_more:
            strings = strings[:max_items]
        return {
            "label": batch_summary.text_table(strings, max_items),
            "tooltip": repr(strings),
        }
    if all(isinstance(item, Image.Image) for item in sequence):
        images = cast(Sequence[Image.Image], sequence)
        total = len(images) if include_more else min(len(images), max_items)
        return _render_raw_input_image_batch(trace, images, max_items=max_items, total=total)
    return None


def _batch_render_limit(batch_render: str) -> int:
    """Return the maximum number of raw-input batch items to render.

    Parameters
    ----------
    batch_render:
        Batch rendering policy string.

    Returns
    -------
    int
        Maximum number of items to render; zero means shape-only fallback.

    Raises
    ------
    ValueError
        If ``batch_render`` is unsupported.
    """

    if batch_render == "auto":
        return 4
    if batch_render == "all":
        return 16
    if batch_render == "first":
        return 1
    if batch_render == "shape_only":
        return 0
    if batch_render.startswith("first_n:"):
        raw_n = batch_render.removeprefix("first_n:")
        try:
            n_items = int(raw_n)
        except ValueError as exc:
            raise ValueError("batch_render first_n value must be an integer.") from exc
        if n_items < 1:
            raise ValueError("batch_render first_n value must be at least 1.")
        return min(n_items, 16)
    raise ValueError("batch_render must be 'auto', 'all', 'first', 'first_n:<N>', or 'shape_only'.")


def _raw_input_sequence(value: Any) -> Sequence[Any] | None:
    """Return a concrete raw-input sequence when ``value`` is a batch container.

    Parameters
    ----------
    value:
        Candidate raw input.

    Returns
    -------
    Sequence[Any] | None
        Concrete sequence for batch rendering, or ``None`` for fallback.
    """

    if isinstance(value, str | bytes | bytearray | Mapping):
        return None
    if isinstance(value, Sequence):
        return value
    if isinstance(value, Iterable) and hasattr(value, "__len__"):
        return tuple(value)
    return None


def _render_raw_input_tensor_batch(
    trace: "Trace",
    tensor: torch.Tensor,
    *,
    max_items: int,
    include_more: bool,
) -> dict[str, str] | None:
    """Return render attributes for a batched raw-input tensor.

    Parameters
    ----------
    trace:
        Trace that owns the rendered input node.
    tensor:
        Candidate raw-input tensor.
    max_items:
        Maximum number of batch items to render.
    include_more:
        Whether to annotate hidden batch items.

    Returns
    -------
    dict[str, str] | None
        Graphviz attributes or ``None`` for shape fallback.
    """

    if tensor.dim() < 2 or int(tensor.shape[0]) <= 1:
        return None
    images = _tensor_batch_to_images(tensor, max_items=max_items)
    if images is None:
        return None
    return _render_raw_input_image_batch(
        trace,
        images,
        max_items=max_items,
        total=int(tensor.shape[0]) if include_more else len(images),
    )


def _render_raw_input_image_batch(
    trace: "Trace",
    images: Sequence[Image.Image],
    *,
    max_items: int,
    total: int,
) -> dict[str, str] | None:
    """Return Graphviz attributes for a PIL image batch.

    Parameters
    ----------
    trace:
        Trace that owns the rendered input node.
    images:
        PIL images to summarize.
    max_items:
        Maximum number of images to render.
    total:
        Total batch size before sampling.

    Returns
    -------
    dict[str, str] | None
        Graphviz attributes or ``None`` for shape fallback.
    """

    if not images:
        return None
    image_dir = _raw_input_visualizer_dir(trace)
    image_path = image_dir / "input_batch_montage.png"
    batch_summary.montage(images, max_items).save(image_path)
    label_lines = ["input"]
    more_count = total - min(total, max_items)
    if more_count > 0:
        label_lines.append(f"+{more_count} more")
    return {
        "image": str(image_path),
        "imagescale": "true",
        "label": render_lines_to_html(label_lines),
        "labelloc": "b",
        "shape": "none",
        "tooltip": f"{total} input images",
    }


def _raw_input_visualizer_dir(trace: "Trace") -> Path:
    """Return a directory for raw-input visualization artifacts.

    Parameters
    ----------
    trace:
        Trace that owns the rendered input node.

    Returns
    -------
    Path
        Directory where image artifacts can be written.
    """

    output_dir = getattr(trace, "_visualizer_dir", None)
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="torchlens_visualizers_")
        trace._visualizer_dir = str(output_dir)
    input_dir = Path(output_dir) / "raw_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir


def _tensor_batch_to_images(tensor: torch.Tensor, *, max_items: int) -> list[Image.Image] | None:
    """Convert a 4D image tensor batch into PIL images.

    Parameters
    ----------
    tensor:
        Candidate tensor with shape ``(B, C, H, W)``.
    max_items:
        Maximum number of batch items to convert.

    Returns
    -------
    list[Image.Image] | None
        Converted images, or ``None`` for non-image tensors.
    """

    if tensor.dim() != 4 or int(tensor.shape[1]) not in {1, 3}:
        return None
    shown = tensor.detach().cpu()[:max_items].float()
    images = []
    for item in shown:
        item = _normalize_image_tensor(item)
        if item.shape[0] == 1:
            array = (item.squeeze(0).numpy() * 255).astype("uint8")
            images.append(Image.fromarray(array, mode="L").convert("RGB"))
        else:
            array = (item.permute(1, 2, 0).numpy() * 255).astype("uint8")
            images.append(Image.fromarray(array, mode="RGB"))
    return images


def _normalize_image_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize an image tensor into the ``[0, 1]`` display range.

    Parameters
    ----------
    tensor:
        Image tensor with shape ``(C, H, W)``.

    Returns
    -------
    torch.Tensor
        Float tensor clipped or min-max normalized to ``[0, 1]``.
    """

    if float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0:
        return tensor.clamp(0.0, 1.0)
    min_value = tensor.min()
    max_value = tensor.max()
    if bool(torch.isclose(max_value, min_value)):
        return torch.zeros_like(tensor)
    return ((tensor - min_value) / (max_value - min_value)).clamp(0.0, 1.0)


def _render_raw_output(value: Any) -> dict[str, str] | None:
    """Return Graphviz attributes for a renderable raw output value.

    Parameters
    ----------
    value:
        Human-readable output metadata stored on the owning ``Trace``.

    Returns
    -------
    dict[str, str] | None
        Node attributes to merge into an output-node spec, or ``None`` when the
        default tensor-shape rendering should be used.
    """

    if value is None:
        return None
    if isinstance(value, str):
        text = _truncate_raw_input_text(value, limit=80)
        return {
            "label": render_lines_to_html(["output", text]),
            "tooltip": value,
        }
    if _is_label_score_sequence(value):
        lines = ["output", *[_format_label_score_row(label, score) for label, score in value]]
        return {
            "label": render_lines_to_html(lines),
            "tooltip": repr(value),
        }
    if isinstance(value, Mapping):
        rows = list(value.items())[:5]
        lines = [
            "output",
            *[f"{key}: {_truncate_raw_input_text(str(item), limit=60)}" for key, item in rows],
        ]
        return {
            "label": render_lines_to_html(lines),
            "tooltip": repr(value),
        }
    return None


def _is_label_score_sequence(value: Any) -> bool:
    """Return whether ``value`` is a flat list of label-score pairs.

    Parameters
    ----------
    value:
        Candidate raw output value.

    Returns
    -------
    bool
        Whether ``value`` can be rendered as prediction rows.
    """

    return isinstance(value, list) and all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str | int | float)
        and isinstance(item[1], int | float)
        for item in value
    )


def _format_label_score_row(label: Any, score: int | float) -> str:
    """Format one label-score prediction row.

    Parameters
    ----------
    label:
        Prediction label.
    score:
        Prediction confidence or score.

    Returns
    -------
    str
        Display row for the output node.
    """

    label_text = _truncate_raw_input_text(str(label), limit=48)
    if 0 <= float(score) <= 1:
        score_text = f"{float(score):.0%}"
    else:
        score_text = f"{float(score):.3g}"
    return f"{label_text} {score_text}"


def _truncate_raw_input_text(text: str, *, limit: int) -> str:
    """Return ``text`` truncated to a display-safe length.

    Parameters
    ----------
    text:
        Text to truncate.
    limit:
        Maximum displayed character count including the ellipsis.

    Returns
    -------
    str
        Original text or a shortened form ending in ``...``.
    """

    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _intervention_hook_node_name(site_label: str) -> str:
    """Return the Graphviz node name for an intervention hook node.

    Parameters
    ----------
    site_label:
        Layer-pass label for the intervention site.

    Returns
    -------
    str
        Graphviz-safe hook node identifier.
    """

    return f"intervention_hook_{site_label.replace(':', 'pass')}"


def _add_intervention_hook_nodes(
    graphviz_graph: graphviz.Digraph,
    site_labels: set[str],
    graph_overrides: dict[str, Any] | None,
) -> None:
    """Add standalone hook nodes for ``vis_intervention_mode='as_node'``.

    Parameters
    ----------
    graphviz_graph:
        Graphviz graph being rendered.
    site_labels:
        Layer-pass labels with intervention specs.
    graph_overrides:
        Graph override dictionary, including optional intervention style keys.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    fillcolor = str(
        intervention_graph_override(
            graph_overrides,
            "intervention_hook_fillcolor",
            INTERVENTION_HOOK_FILL_COLOR,
        )
    )
    color = str(
        intervention_graph_override(
            graph_overrides,
            "intervention_hook_color",
            INTERVENTION_HOOK_BORDER_COLOR,
        )
    )
    penwidth = str(intervention_graph_override(graph_overrides, "intervention_hook_penwidth", 2.0))
    for site_label in sorted(site_labels):
        spec = NodeSpec(
            lines=["intervention", site_label],
            shape="diamond",
            fillcolor=fillcolor,
            fontcolor="black",
            style="filled,solid",
            color=color,
            penwidth=float(penwidth),
            tooltip=f"Intervention hook after {site_label}",
            extra_attrs={"ordering": "out", "width": "0.35", "height": "0.25"},
        )
        node_args = _node_spec_to_graphviz_args(spec)
        node_args["name"] = _intervention_hook_node_name(site_label)
        graphviz_graph.node(**node_args)


def _build_collapsed_module_node(
    self: "Trace",
    node: GraphNode,
    graphviz_graph: graphviz.Digraph,
    collapsed_modules: set[str],
    vis_mode: str,
    vis_call_depth: int,
    collapse_address: str | None,
    overrides: VisualizationOverrides,
    node_mode: VisNodeModeLiteral,
    collapsed_node_spec_fn: CollapsedNodeSpecFn | None = None,
    theme: VisualizationTheme | None = None,
) -> None:
    """Builds and adds a collapsed module box node to the graphviz graph.

    Args:
        node: The OpLog or LayerLog node triggering the collapse.
        graphviz_graph: The graphviz Digraph object to add the node to.
        collapsed_modules: Set of collapsed module names already added; updated in place.
        vis_mode: 'unrolled' or 'rolled'.
        vis_call_depth: Maximum nesting depth; nodes at this depth are collapsed.
        overrides: Graphviz attribute overrides.
    """
    # Access the module at the collapse threshold depth.  This index is safe
    # because _is_collapsed_module already verified the node is deep enough.
    address_w_pass = (
        collapse_address if collapse_address is not None else node.modules[vis_call_depth - 1]
    )
    # rsplit with maxsplit=1 handles module names containing colons (#104).
    module_tuple = address_w_pass.rsplit(":", 1)
    module_output_layer = self[address_w_pass]
    module_output_shape = module_output_layer.shape or ()
    module_output_fsize = module_output_layer.memory_str
    address, call_index = module_tuple
    ml = self.modules[address]
    module_type = ml.class_name  # type: ignore[union-attr]
    module_num_calls = ml.num_calls  # type: ignore[union-attr]
    module_nparams = ml.num_params  # type: ignore[union-attr]

    # In unrolled mode, each pass of a module is a separate collapsed node
    # (e.g., "encoder.layer.0pass1").  In rolled mode, all ops share one
    # node (e.g., "encoder.layer.0").
    if vis_mode == "unrolled":
        node_name = "pass".join(module_tuple)
        mpl = self.modules[address_w_pass]
        module_num_tensors = mpl.num_layers
        module_has_input_ancestor = any(self[layer].has_input_ancestor for layer in mpl.layers)
    else:
        node_name = module_tuple[0]
        module_num_tensors = ml.num_layers
        module_has_input_ancestor = any(self[layer].has_input_ancestor for layer in ml.layers)  # type: ignore[union-attr]

    # Deduplicate: multiple layers in the same collapsed module will each
    # trigger this function, but the node should only be added once.
    if node_name in collapsed_modules:
        return

    if module_num_calls == 1:
        node_title = f"<b>@{address}</b>"
    elif vis_mode == "unrolled" and (module_num_calls > 1):
        node_title = f"<b>@{address}:{call_index}</b>"
    else:
        node_title = f"<b>@{address} (x{module_num_calls})</b>"

    if len(module_output_shape) > 1:
        shape_str = "x".join([str(x) for x in module_output_shape])
    elif len(module_output_shape) == 1:  # #100: use module_output_shape, not node.shape
        shape_str = f"x{module_output_shape[0]}"  # type: ignore[misc]
    else:
        shape_str = "x1"

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
            f"{shape_str} ({module_output_fsize})",
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
    if theme is not None:
        default_spec = apply_theme_to_spec(default_spec, theme)
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
        node_args["gradangle"] = "0"

    graphviz_graph.node(**node_args)
    collapsed_modules.add(node_name)


def _get_node_address_shape_color(
    self: "Trace",
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

    if (node.is_atomic_module_op or only_non_buffer_layer) and (len(node.modules) > 0):
        if isinstance(source_node, OpLog):
            module_pass_exited = node.modules[-1]
            module, _ = module_pass_exited.split(":")
            if self.modules[module].num_calls == 1:  # type: ignore[union-attr]
                node_address = module
            else:
                node_address = module_pass_exited
        else:
            sample_module_pass = node.modules[-1]
            module = sample_module_pass.split(":")[0]
            node_address = module

        node_address = "<br/>@" + node_address
        node_shape = "box"
        node_color = "black"
    elif node.is_buffer:
        if (self.buffer_num_calls[source_node.buffer_address] == 1) or (
            isinstance(source_node, LayerLog) and node.num_calls > 1
        ):
            buffer_address = source_node.buffer_address
        else:
            buffer_address = f"{source_node.buffer_address}:{source_node.buffer_pass}"
        node_address = "<br/>@" + buffer_address
        node_shape = "cylinder"
        node_color = "black"
    elif node.is_output or node.is_input:
        node_address = "<br/>@" + node.io_role
        node_shape = "oval"
        node_color = "black"
    else:
        node_address = ""
        node_shape = "oval"
        node_color = "black"

    return node_address, node_shape, node_color


def _is_only_non_buffer_in_module(
    self: "Trace", node: GraphNode, show_buffer_layers: BufferVisibilityLiteral
) -> bool:
    """Returns True if a layer is the only non-buffer layer in a leaf module.

    Leaf modules are those with no child submodules. Container modules with
    functional ops at the end should NOT match — those ops are rendered as
    ovals, not boxes (issue #48).

    Args:
        node: The OpLog or LayerLog node to check.
        show_buffer_layers: Buffer visibility mode.
    """
    # Check whether it leaves its module:
    if not (
        (len(node.output_of_modules) > 0)
        and (len(node.modules) > 0)
        and (node.modules[-1].split(":")[0] in node.output_of_modules)
    ):
        return False

    # Only apply box rendering for leaf modules (no child submodules).
    exited_module = node.modules[-1].split(":")[0]
    if exited_module in self.modules and len(self.modules[exited_module].call_children) > 0:
        return False

    # Now check whether all of its parents are either buffers, or are outside the module.
    # If any aren't, return False.

    for parent_layer_label in node.parents:
        if parent_layer_label.startswith("__module_focus_"):
            continue
        source_node = _unwrap_focus_node(node)
        if isinstance(source_node, OpLog):
            parent_layer = self[parent_layer_label]
        else:
            parent_layer = self.layer_logs[parent_layer_label]
        if (
            (not parent_layer.is_buffer) or _is_buffer_visible(parent_layer, show_buffer_layers)
        ) and ((len(parent_layer.modules) > 0) and parent_layer.modules[-1] == node.modules[-1]):
            return False

    return True


def _get_node_bg_color(self: "Trace", node: GraphNode) -> str:
    """Returns the background color hex string for a graph node based on its type.

    Maps node types to colors: input=green, output=red, boolean=orange,
    parameterized layers=blue (trainable) or gray (frozen), default=white.

    Args:
        node: node to add

    Returns:
        node_bg_color: background color of the node
    """
    if node.is_input:
        bg_color = INPUT_COLOR
    elif node.is_output:
        bg_color = OUTPUT_COLOR
    elif node.is_terminal_bool:
        bg_color = BOOL_NODE_COLOR
    elif node.uses_params:
        param_logs = getattr(node, "_param_logs", [])
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
    trace: "Trace",
    node: GraphNode,
    default_spec: NodeSpec,
    node_mode: VisNodeModeLiteral,
    node_spec_fn: NodeSpecFn | None,
) -> NodeSpec:
    """Apply a layer node callback to a default spec.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Rendered OpLog or LayerLog.
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

    layer_log = _layer_log_for_node(trace, node)
    mode_fn = MODE_REGISTRY[node_mode]
    mode_result = mode_fn(layer_log, default_spec)
    mode_spec = default_spec if mode_result is None else mode_result
    if node_spec_fn is None:
        return mode_spec
    result = node_spec_fn(layer_log, mode_spec)
    return mode_spec if result is None else result


def _layer_log_for_node(trace: "Trace", node: GraphNode) -> "LayerLog":
    """Return the aggregate LayerLog for ``node``.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        OpLog or LayerLog.

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
    return trace.layer_logs[node.layer_label_no_pass]


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
        "image": spec.image,
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
    *,
    node_label_fields: list[str] | None = None,
    node_overlay: str | OverlayScores | None = None,
) -> list[str]:
    """Build default plain-text rows for a layer node.

    Parameters
    ----------
    layer_log:
        OpLog or LayerLog to render.
    node_address:
        Existing address suffix from TorchLens node address logic.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    node_label_fields:
        Optional label fields to render instead of the default field set.
    node_overlay:
        Optional overlay to append as an additional label row.

    Returns
    -------
    list[str]
        Plain-text rows for ``NodeSpec.lines``.
    """

    layer_log = _unwrap_focus_node(layer_log)
    if isinstance(layer_log, BoundaryNode):
        return [layer_log.display_label]

    if node_label_fields is not None:
        selected_lines = _compute_selected_node_lines(
            layer_log, node_address, vis_mode, node_label_fields
        )
        overlay = overlay_line(layer_log, node_overlay)
        if overlay is not None:
            selected_lines.append(overlay)
        return selected_lines

    if (layer_log.num_calls > 1) and (vis_mode == "unrolled"):
        call_label = f":{layer_log.call_index}"
    elif (layer_log.num_calls > 1) and (vis_mode == "rolled"):
        call_label = f" (x{layer_log.num_calls})"
    else:
        call_label = ""

    if layer_log.layer_type in ["input", "output", "buffer"]:
        title = f"{layer_log.layer_type}_{layer_log.type_index}{call_label}"
    else:
        title = f"{layer_log.layer_type}_{layer_log.type_index}_{layer_log.trace_index}{call_label}"

    lines: list[str] = []
    if layer_log.is_terminal_bool:
        lines.append(str(layer_log.bool_value).upper())
    lines.append(title)
    lines.append(f"{_format_shape_str(layer_log.shape)} ({layer_log.memory_str})")

    important_args = _format_important_args(layer_log)
    if important_args:
        lines.append(important_args)

    param_line = _make_param_line(layer_log)
    if param_line:
        lines.append(param_line)

    address_line = node_address.replace("<br/>", "")
    if address_line:
        lines.append(address_line)
    overlay = overlay_line(layer_log, node_overlay)
    if overlay is not None:
        lines.append(overlay)
    return lines


def _compute_selected_node_lines(
    layer_log: GraphNode,
    node_address: str,
    vis_mode: str,
    node_label_fields: list[str],
) -> list[str]:
    """Build node-label rows from an explicit field picker.

    Parameters
    ----------
    layer_log:
        OpLog or LayerLog to render.
    node_address:
        Existing address suffix from TorchLens node address logic.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    node_label_fields:
        Requested field names.

    Returns
    -------
    list[str]
        Selected label rows.

    Raises
    ------
    ValueError
        If an unknown field is requested.
    """

    rows: list[str] = []
    for field_name in node_label_fields:
        if field_name in {"label", "name"}:
            rows.append(str(getattr(layer_log, "layer_label", "")))
        elif field_name in {"type", "op", "operation"}:
            rows.append(str(getattr(layer_log, "func_name", None) or layer_log.layer_type))
        elif field_name == "shape":
            rows.append(_format_shape_str(layer_log.shape))
        elif field_name in {"memory", "bytes"}:
            rows.append(str(getattr(layer_log, "memory_str", "")))
        elif field_name == "module":
            rows.append(node_address.replace("<br/>", "") or "@root")
        elif field_name == "params":
            param_line = _make_param_line(layer_log)
            if param_line:
                rows.append(param_line)
        elif field_name == "pass":
            rows.append(
                str(
                    getattr(layer_log, "call_index", 1)
                    if vis_mode == "unrolled"
                    else getattr(layer_log, "num_calls", 1)
                )
            )
        elif field_name == "flops":
            rows.append(str(getattr(layer_log, "flops_forward", 0) or 0))
        elif field_name == "time":
            rows.append(f"{float(getattr(layer_log, 'func_duration', 0.0) or 0.0) * 1000:.3g} ms")
        else:
            raise ValueError(f"Unsupported node label field: {field_name!r}.")
    return rows or compute_default_node_lines(layer_log, node_address, vis_mode)


def _make_node_label(
    node: Union["OpLog", "LayerLog"],
    node_address: str,
    vis_mode: str,
) -> str:
    """Builds an HTML-table label string for a graphviz node.

    Assembles rows for the layer name, tensor shape, operation type, and other
    metadata into an HTML table used as the node label in graphviz rendering.
    """
    # Pass info:

    if (node.num_calls > 1) and (vis_mode == "unrolled"):
        call_label = f":{node.call_index}"
    elif (node.num_calls > 1) and (vis_mode == "rolled"):
        call_label = f" (x{node.num_calls})"
    else:
        call_label = ""

    # Tensor shape info:

    if len(node.shape) > 1:
        shape_str = "x".join([str(x) for x in node.shape])
    elif len(node.shape) == 1:
        shape_str = f"x{node.shape[0]}"
    else:
        shape_str = "x1"

    # Layer param info:

    param_label = _make_param_label(node)

    memory = node.memory_str
    if node.layer_type in ["input", "output", "buffer"]:
        node_title = f"<b>{node.layer_type}_{node.type_index}{call_label}</b>"
    else:
        node_title = f"<b>{node.layer_type}_{node.type_index}_{node.trace_index}{call_label}</b>"

    if node.is_terminal_bool:
        label_text = str(node.bool_value).upper()
        bool_label = f"<b><u>{label_text}:</u></b><br/><br/>"
    else:
        bool_label = ""

    node_label = f"<{bool_label}{node_title}<br/>{shape_str} ({memory}){param_label}{node_address}>"

    return node_label


def _format_shape_str(shape: tuple[Any, ...]) -> str:
    """Formats a shape tuple as a compact string like '3x3x64'."""
    if len(shape) > 1:
        return "x".join(str(s) for s in shape)
    elif len(shape) == 1:
        return f"x{shape[0]}"
    return "x1"


def _make_param_label(node: Union["OpLog", "LayerLog"]) -> str:
    """Makes the label for parameters of a node.

    Uses param names and bracket convention when ParamLog objects are available:
    round brackets () for trainable, square brackets [] for frozen.
    """
    if node.num_param_tensors == 0:
        return ""

    param_logs = getattr(node, "_param_logs", [])
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
        each_param_shape = [_format_shape_str(s) for s in node.param_shapes]
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

    param_logs = getattr(node, "_param_logs", [])
    if param_logs:
        parts = []
        for param_log in param_logs:
            shape_str = _format_shape_str(param_log.shape)
            wrapper = ("(", ")") if param_log.trainable else ("[", "]")
            parts.append(f"{param_log.name}: {wrapper[0]}{shape_str}{wrapper[1]}")
        return "params: " + ", ".join(parts)

    each_param_shape = [_format_shape_str(shape) for shape in node.param_shapes]
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

    if not node.param_shapes:
        return None
    first_shape = node.param_shapes[0]
    if len(first_shape) == 0:
        return None
    return int(first_shape[0])


def _add_edges_for_node(
    self: "Trace",
    parent_node: GraphNode,
    parent_is_collapsed_module: bool,
    vis_call_depth: int,
    node_color: str,
    module_edge_dict: Dict[str, Any],
    edges_used: Set[tuple[str, str]],
    graphviz_graph: graphviz.Digraph,
    vis_mode: str = "unrolled",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
    overrides: Optional[VisualizationOverrides] = None,
    collapse_fn: CollapseFn | None = None,
    edge_map: Optional[dict[str, list[RenderEdge]]] = None,
    vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
    intervention_site_labels: set[str] | None = None,
) -> None:
    """Add forward (and optionally grad) edges from a parent node to all its children.

    Handles several complex cases:

    - **Collapsed module nodes**: when parent or child is collapsed, the edge
      endpoint is the module box name, not the individual layer name.
    - **Intra-module edge skip**: when both parent and child map to the SAME
      collapsed module box AND share the same module nesting prefix up to
      ``vis_call_depth``, the edge is internal to the collapsed module
      and should not be drawn.
    - **Edge deduplication**: ``edges_used`` prevents duplicate edges that
      arise when multiple layers map to the same collapsed module node.
    - **Argument labels**: for non-commutative ops with multiple parents,
      edge labels show which argument position each parent occupies.
      Note: uses substring matching on layer_label for arg_label lookup,
      which has a theoretical false-positive risk if one label is a
      substring of another (extremely rare in practice).
    - **Pass annotations** (rolled mode): ``_label_rolled_call_indexs`` adds
      tail/head labels showing which ops an edge applies to.

    Args:
        parent_node: The node to add edges for.
        parent_is_collapsed_module: Whether the node is a collapsed module node.
        vis_call_depth: How many levels of module nesting to show.
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
            for child_layer_label in parent_node.children
        ]
    else:
        render_edges = edge_map.get(parent_node.layer_label, [])

    for render_edge in render_edges:
        child_node = render_edge.target
        metadata_child = render_edge.metadata_child

        if child_node.is_buffer and not _is_buffer_visible(child_node, show_buffer_layers):
            continue

        if parent_node.has_input_ancestor:
            edge_style = "solid"
        else:
            edge_style = "dashed"

        if parent_is_collapsed_module:
            module_name_w_pass = _collapse_address_for_node(
                self,
                parent_node,
                collapse_fn=collapse_fn,
                max_module_depth=vis_call_depth,
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

        child_collapse_address = _collapse_address_for_node(
            self,
            child_node,
            collapse_fn=collapse_fn,
            max_module_depth=vis_call_depth,
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
        # they share the same module path up to vis_call_depth, the edge
        # is internal to the collapsed module box and should not be drawn.
        # The tail_name != head_name check handles the case where they map to
        # different collapsed modules (cross-module edge, should be drawn).
        if both_nodes_collapsed_modules and (tail_name != head_name):
            child_modules = child_node.modules[:]
            parent_modules = parent_node.modules[:]
            # Adjust for bottom-level submodule outputs (they belong to parent scope).
            if child_node.is_atomic_module_op:
                child_modules = child_modules[:-1]
            if parent_node.is_atomic_module_op:
                parent_modules = parent_modules[:-1]
            if child_modules[:vis_call_depth] == parent_modules[:vis_call_depth]:
                continue

        # Edge deduplication: multiple layers mapping to the same collapsed
        # module node would produce duplicate edges without this check.
        if (
            vis_intervention_mode == "as_node"
            and intervention_site_labels is not None
            and parent_node.layer_label in intervention_site_labels
        ):
            hook_name = _intervention_hook_node_name(parent_node.layer_label)
            if (tail_name, hook_name) not in edges_used:
                edges_used.add((tail_name, hook_name))
                graphviz_graph.edge(
                    tail_name=tail_name,
                    head_name=hook_name,
                    color=node_color,
                    fontcolor=node_color,
                    style=edge_style,
                    arrowsize=".7",
                    labelfontsize="8",
                )
            tail_name = hook_name

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

        # Annotate ops for rolled node edge if it varies across ops
        if vis_mode == "rolled" and metadata_child is not None and not edge_has_boundary:
            metadata_base_for_pass = _base_node_for_metadata(metadata_child)
            parent_base_for_pass = _base_node_for_metadata(parent_node)
            if isinstance(metadata_base_for_pass, LayerLog) and isinstance(
                parent_base_for_pass, LayerLog
            ):
                _label_rolled_call_indexs(
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
            module = _get_lowest_module_for_two_render_nodes(
                parent_node,
                child_node,
                both_nodes_collapsed_modules,
                vis_call_depth,
            )
        else:
            module = _get_lowest_module_for_two_nodes(
                _base_node_for_metadata(parent_node),
                _base_node_for_metadata(child_node),
                both_nodes_collapsed_modules,
                vis_call_depth,
            )
        if module != -1:
            module_key = cast(str, module)
            module_edge_dict[module_key]["edges"].append(edge_dict)
            if parent_node.has_input_ancestor or child_node.has_input_ancestor:
                module_edge_dict[module_key]["has_input_ancestor"] = True
                for module in parent_node.modules:
                    module_key = module.split(":")[0] if vis_mode == "rolled" else module
                    module_edge_dict[module_key]["has_input_ancestor"] = True
                    if module_key == module:
                        break
                for module in child_node.modules:
                    module_key = module.split(":")[0] if vis_mode == "rolled" else module
                    module_edge_dict[module_key]["has_input_ancestor"] = True
                    if module_key == module:
                        break
        else:
            graphviz_graph.edge(**edge_dict)

        # Finally, add a backwards edge if both tensors have stored grads.
        if not (isinstance(parent_node, BoundaryNode) or isinstance(child_node, BoundaryNode)):
            _add_grad_edge(
                self,
                parent_node,
                child_node,
                edge_style,
                module,
                module_edge_dict,
                graphviz_graph,
                overrides,  # type: ignore[arg-type]
            )


def _compute_edge_label(
    parent_node: Union["OpLog", "LayerLog"],
    child_node: Union["OpLog", "LayerLog"],
    trace: "Trace",
    vis_mode: str,
) -> Optional[str]:
    """Return the highest-priority semantic label for an edge.

    Precedence matches the Phase 7 conditional rendering spec:

    1. Arm-entry labels from ``Trace.conditional_arm_entry_edges`` /
       ``Trace.conditional_edge_call_indices``.
    2. ``IF`` labels from ``Trace.conditional_branch_edges``.
    3. ``None`` when the edge has no branch semantics.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        trace:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    Optional[str]
        Graphviz HTML label string, or ``None`` if no semantic label applies.
    """
    arm_label = _compute_arm_entry_edge_label(parent_node, child_node, trace, vis_mode)
    if arm_label is not None:
        return _format_branch_edge_label_html(arm_label)

    if _edge_is_conditional_branch(parent_node, child_node, trace, vis_mode):
        return _format_branch_edge_label_html("IF")

    return None


def _compute_arm_entry_edge_label(
    parent_node: Union["OpLog", "LayerLog"],
    child_node: Union["OpLog", "LayerLog"],
    trace: "Trace",
    vis_mode: str,
) -> Optional[str]:
    """Return the arm-entry text for an edge, without Graphviz HTML wrapping.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        trace:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    Optional[str]
        Plain-text arm label, or ``None`` if the edge is not an arm-entry edge.
    """
    arm_entries = _get_arm_edge_entries(parent_node, child_node, trace, vis_mode)
    if not arm_entries:
        return None

    if vis_mode == "rolled":
        return _format_rolled_arm_entry_label(arm_entries, trace)

    if len(arm_entries) == 1:
        conditional_id, branch_kind, _ = arm_entries[0]
        return _format_arm_entry_text(conditional_id, branch_kind, trace)

    return " · ".join(
        [
            _format_arm_entry_text(
                conditional_id,
                branch_kind,
                trace,
                include_conditional_reference=True,
            )
            for conditional_id, branch_kind, _ in arm_entries
        ]
    )


def _get_arm_edge_entries(
    parent_node: Union["OpLog", "LayerLog"],
    child_node: Union["OpLog", "LayerLog"],
    trace: "Trace",
    vis_mode: str,
) -> List[Tuple[int, str, Optional[Tuple[int, ...]]]]:
    """Collect conditional-arm metadata for one rendered edge.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        trace:
            Owning model log containing conditional metadata.
        vis_mode:
            ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    List[Tuple[int, str, Optional[Tuple[int, ...]]]]
        Sorted ``(conditional_id, branch_kind, call_indexs)`` tuples. Unrolled
        edges use ``call_indexs=None``.
    """
    arm_entries: List[Tuple[int, str, Optional[Tuple[int, ...]]]] = []
    if vis_mode == "unrolled":
        edge_key = (parent_node.layer_label, child_node.layer_label)
        for (conditional_id, branch_kind), edge_list in trace.conditional_arm_entry_edges.items():
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
        ), call_indexs in trace.conditional_edge_call_indices.items():
            if (edge_parent, edge_child) == (parent_no_pass, child_no_pass):
                arm_entries.append((conditional_id, branch_kind, tuple(call_indexs)))
    else:
        raise ValueError(f"vis_mode must be 'unrolled' or 'rolled', not {vis_mode}")

    return sorted(arm_entries, key=lambda entry: _arm_entry_sort_key(entry[0], entry[1], trace))


def _format_rolled_arm_entry_label(
    arm_entries: List[Tuple[int, str, Optional[Tuple[int, ...]]]],
    trace: "Trace",
) -> str:
    """Format a rolled-mode arm-entry label with pass-awareness.

    Args:
        arm_entries:
            Sorted ``(conditional_id, branch_kind, call_indexs)`` tuples for one
            rolled edge.
        trace:
            Owning model log containing conditional metadata.

    Returns
    -------
    str
        Plain-text arm label for the rolled edge.
    """
    if len(arm_entries) == 1:
        conditional_id, branch_kind, _ = arm_entries[0]
        return _format_arm_entry_text(conditional_id, branch_kind, trace)

    pass_sets = [set(call_indexs or ()) for _, _, call_indexs in arm_entries]
    if pass_sets and len({tuple(sorted(pass_set)) for pass_set in pass_sets}) == 1:
        return " · ".join(
            [
                _format_arm_entry_text(
                    conditional_id,
                    branch_kind,
                    trace,
                    include_conditional_reference=True,
                )
                for conditional_id, branch_kind, _ in arm_entries
            ]
        )

    pass_counts: Dict[int, int] = defaultdict(int)
    for _, _, call_indexs in arm_entries:
        for call_index in call_indexs or ():
            pass_counts[call_index] += 1

    if pass_counts and all(pass_count == 1 for pass_count in pass_counts.values()):
        return " / ".join(
            [
                _format_rolled_pass_arm_text(
                    conditional_id,
                    branch_kind,
                    call_indexs,
                    trace,
                    include_conditional_reference=_rolled_labels_need_disambiguation(arm_entries),
                )
                for conditional_id, branch_kind, call_indexs in arm_entries
            ]
        )

    return "mixed"


def _rolled_labels_need_disambiguation(
    arm_entries: List[Tuple[int, str, Optional[Tuple[int, ...]]]],
) -> bool:
    """Return True when rolled branch labels need conditional disambiguation.

    Args:
        arm_entries:
            Sorted ``(conditional_id, branch_kind, call_indexs)`` tuples for one
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
    call_indexs: Optional[Tuple[int, ...]],
    trace: "Trace",
    include_conditional_reference: bool,
) -> str:
    """Format one rolled arm label with its pass list.

    Args:
        conditional_id:
            Dense conditional id.
        branch_kind:
            Branch kind such as ``"then"`` or ``"elif_2"``.
        call_indexs:
            Sorted pass numbers for this rolled edge/arm tuple.
        trace:
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
        trace,
        include_conditional_reference=include_conditional_reference,
    )
    if not call_indexs:
        return branch_text
    return f"{branch_text}({int_list_to_compact_str(list(call_indexs))})"


def _format_arm_entry_text(
    conditional_id: int,
    branch_kind: str,
    trace: "Trace",
    include_conditional_reference: bool = False,
) -> str:
    """Format one arm-entry label as plain text.

    Args:
        conditional_id:
            Dense conditional id.
        branch_kind:
            Branch kind such as ``"then"`` or ``"elif_2"``.
        trace:
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
    return f"{branch_text}@{_get_conditional_reference_text(conditional_id, trace)}"


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


def _get_conditional_reference_text(conditional_id: int, trace: "Trace") -> str:
    """Return a readable conditional identifier for composite edge labels.

    Args:
        conditional_id:
            Dense conditional id.
        trace:
            Owning model log containing conditional metadata.

    Returns
    -------
    str
        Line-based conditional reference when available, otherwise ``"C{id}"``.
    """
    for conditional_event in trace.conditional_records:
        if conditional_event.id == conditional_id:
            return f"L{conditional_event.if_stmt_span[0]}"
    return f"C{conditional_id}"


def _arm_entry_sort_key(
    conditional_id: int,
    branch_kind: str,
    trace: "Trace",
) -> Tuple[int, int, int]:
    """Return a stable sort key for multi-arm edge labels.

    Args:
        conditional_id:
            Dense conditional id.
        branch_kind:
            Branch kind such as ``"then"`` or ``"elif_2"``.
        trace:
            Owning model log containing conditional metadata.

    Returns
    -------
    Tuple[int, int, int]
        Sort key ordered by source line, branch rank, then conditional id.
    """
    source_line = 10**9
    for conditional_event in trace.conditional_records:
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
    parent_node: Union["OpLog", "LayerLog"],
    child_node: Union["OpLog", "LayerLog"],
    trace: "Trace",
    vis_mode: str,
) -> bool:
    """Return True when an edge is an ``IF`` branch-entry edge.

    Args:
        parent_node:
            Source node for the edge.
        child_node:
            Destination node for the edge.
        trace:
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
        ) in trace.conditional_branch_edges
    if vis_mode == "rolled":
        edge_key = (parent_node.layer_label_no_pass, child_node.layer_label_no_pass)
        return any(
            (branch_parent.split(":")[0], branch_child.split(":")[0]) == edge_key
            for branch_parent, branch_child in trace.conditional_branch_edges
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
    self: "Trace",
    parent_node: Union["OpLog", "LayerLog"],
    child_node: Union["OpLog", "LayerLog"],
    edge_dict: Dict[str, Any],
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> None:
    """Add argument position labels to an edge when the child has multiple non-commutative parents.

    For nodes like ``sub(a, b)`` where argument order matters, labels like
    ``"arg 0"`` / ``"arg 1"`` are added to distinguish which parent feeds
    which argument.

    Note on substring false-positive risk: the lookup ``parent_node.layer_label == arg_label``
    uses exact equality, so substring matching is not an issue here.  However, the
    ``parent_arg_positions`` keys are positional and the check iterates all of them,
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
        for arg_loc, arg_label in child_node.parent_arg_positions[arg_type].items():
            if parent_node.layer_label == arg_label:
                arg_labels.append(f"{arg_type[:-1]} {str(arg_loc)}")

    arg_labels = "<br/>".join(arg_labels)  # type: ignore[assignment]
    if not arg_labels:
        return
    arg_label = f"<<FONT POINT-SIZE='10'><b>{arg_labels}</b></FONT>>"
    _set_argument_edge_label(edge_dict, arg_label)


def _set_argument_edge_label(edge_dict: Dict[str, Any], arg_label: str) -> None:
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
    self: "Trace",
    child_node: Union["OpLog", "LayerLog"],
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

    if isinstance(child_node, OpLog):
        return _should_mark_arguments_on_unrolled_edge(self, child_node, show_buffer_layers)
    elif isinstance(child_node, LayerLog):
        return _should_mark_arguments_on_rolled_edge(self, child_node, show_buffer_layers)


def _should_mark_arguments_on_unrolled_edge(
    self: "Trace",
    child_node: "OpLog",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument labels should be shown on an unrolled graph edge.

    Args:
        child_node: The child OpLog node whose incoming edge is being considered.
        show_buffer_layers: Buffer visibility mode.
    """
    num_parents_shown = len(child_node.parents)

    if show_buffer_layers != "always":
        num_parents_shown -= sum(
            [
                int(
                    self[parent].is_buffer
                    and not _is_buffer_visible(self[parent], show_buffer_layers)
                )
                for parent in child_node.parents
            ]
        )

    if num_parents_shown > 1:
        return True
    else:
        return False


def _should_mark_arguments_on_rolled_edge(
    self: "Trace",
    child_node: "LayerLog",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument labels should be shown on a rolled graph edge.

    Args:
        child_node: The child LayerLog node whose incoming edge is being considered.
        show_buffer_layers: Buffer visibility mode.
    """
    for call_index, pass_parents in child_node.parents_per_pass.items():
        num_parents_shown = len(pass_parents)
        if show_buffer_layers != "always":
            num_parents_shown -= sum(
                [
                    int(
                        self.layer_logs[parent].is_buffer
                        and not _is_buffer_visible(self.layer_logs[parent], show_buffer_layers)
                    )
                    for parent in pass_parents
                ]
            )
        if num_parents_shown > 1:
            return True

    return False


def _label_rolled_call_indexs(
    child_node: "LayerLog",
    parent_node: "LayerLog",
    edge_dict: Dict[str, Any],
) -> None:
    """Add pass-number annotations to edges in rolled mode.

    In rolled mode, a single edge may represent connections from different
    ops.  When edges vary across ops (``edges_vary_across_ops``),
    tail and head labels show which ops the edge applies to, e.g.,
    ``"Out 1,3"`` / ``"In 2,4"``.  Uses ``int_list_to_compact_str`` for
    concise range notation (e.g., ``"1-3"`` instead of ``"1,2,3"``).

    Args:
        child_node: The child LayerLog node.
        parent_node: The parent LayerLog node.
        edge_dict: Mutable dict of edge attributes; taillabel/headlabel may be added.
    """
    parent_call_indexs = parent_node.child_ops_per_layer[child_node.layer_label]
    child_call_indexs = child_node.parent_ops_per_layer[parent_node.layer_label]
    if parent_node.edges_vary_across_ops:
        edge_dict["taillabel"] = f"  Out {int_list_to_compact_str(parent_call_indexs)}  "

    # Mark the head label with the argument if need be:
    if child_node.edges_vary_across_ops:
        edge_dict["headlabel"] = f"  In {int_list_to_compact_str(child_call_indexs)}  "


def _get_lowest_module_for_two_render_nodes(
    node1: GraphNode,
    node2: GraphNode,
    both_nodes_collapsed_modules: bool,
    vis_call_depth: int,
) -> Union[str, int]:
    """Find the deepest module subgraph for render nodes including boundaries."""

    return _get_lowest_module_for_two_nodes(
        cast(Union["OpLog", "LayerLog"], node1),
        cast(Union["OpLog", "LayerLog"], node2),
        both_nodes_collapsed_modules,
        vis_call_depth,
    )


def _get_lowest_module_for_two_nodes(
    node1: Union["OpLog", "LayerLog"],
    node2: Union["OpLog", "LayerLog"],
    both_nodes_collapsed_modules: bool,
    vis_call_depth: int,
) -> Union[str, int]:
    """Find the deepest module subgraph that contains both nodes.

    Used to place an edge into the correct Graphviz cluster (subgraph).
    Edges between nodes in the same module cluster are drawn inside that
    cluster; edges crossing module boundaries are drawn at the level of
    the lowest common ancestor module.

    Returns -1 when no module contains both nodes (the edge belongs to the
    top-level graph, not any subgraph).

    Special handling:
    - ``is_atomic_module_op`` nodes are adjusted to their parent
      scope (they represent the module's output, rendered one level up).
    - Rolled mode: pass suffixes are stripped from module names so that all
      ops share the same cluster.
    - Both-collapsed case: when both nodes are collapsed module boxes, the
      containing module must be at least one level above the collapse depth.

    Args:
        node1: The first node.
        node2: The second node.
        both_nodes_collapsed_modules: Whether both nodes are collapsed module boxes.
        vis_call_depth: How many levels deep to visualize.

    Returns:
        Module name (str) for the containing cluster, or -1 for top-level.
    """
    node1_modules = node1.modules[:]
    node2_modules = node2.modules[:]

    if isinstance(node1, LayerLog) or isinstance(node2, LayerLog):
        node1_modules = [module.split(":")[0] for module in node1_modules]
        node2_modules = [module.split(":")[0] for module in node2_modules]

    if node1.is_atomic_module_op:
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
        if node1.is_atomic_module_op and (len(node1_modules) == 1):
            return -1
        elif node1.is_atomic_module_op and (len(node1_modules) > 1):
            module = node1_modules[-2]
        else:
            module = node1_modules[-1]
        return cast(str, module)

    if both_nodes_collapsed_modules:
        if (vis_call_depth == 1) or (len(node1_nested_modules) == 1):
            return -1
        if node1_modules[vis_call_depth - 1] == node2_modules[vis_call_depth - 1]:
            module = node1_modules[vis_call_depth - 2]
            return cast(str, module)

    module = node1_modules[0]
    for m in range(min(len(node1_modules), len(node2_modules))):
        if node1_modules[m] != node2_modules[m]:
            break
        module = node1_modules[m]

    return cast(str, module)


def _add_grad_edge(
    self: "Trace",
    parent_layer: GraphNode,
    child_layer: GraphNode,
    edge_style: str,
    module: str | int,
    module_edge_dict: Dict[str, Any],
    graphviz_graph: graphviz.Digraph,
    overrides: VisualizationOverrides,
) -> None:
    """Add a backward (grad) edge if both layers have saved grads.

    Gradient edges flow child -> parent (opposite of data flow), drawn in
    ``GRADIENT_ARROW_COLOR`` to distinguish from forward edges.  In rolled
    mode, an aggregate edge is shown when either rolled endpoint has a grad
    on any pass.

    Args:
        parent_layer: The parent OpLog or LayerLog (grad destination).
        child_layer: The child OpLog or LayerLog (grad source).
        edge_style: ``'solid'`` or ``'dashed'`` (matches the forward edge style).
        module: Module cluster name, or -1 for top-level.
        module_edge_dict: Dict mapping each module cluster to its edges.
        graphviz_graph: The graphviz Digraph object.
        overrides: Graphviz attribute overrides for grad edges.
    """
    if _node_has_grad(parent_layer) and _node_has_grad(child_layer):
        edge_dict = {
            "tail_name": _grad_node_name(child_layer),
            "head_name": _grad_node_name(parent_layer),
            "color": GRADIENT_ARROW_COLOR,
            "fontcolor": GRADIENT_ARROW_COLOR,
            "style": edge_style,
            "arrowsize": ".7",
            "labelfontsize": "8",
        }
        for arg_name, arg_val in overrides.grad_edge.items():  # type: ignore[union-attr]
            if callable(arg_val):
                edge_dict[arg_name] = str(arg_val(self, parent_layer, child_layer))
            else:
                edge_dict[arg_name] = str(arg_val)

        if module != -1:
            module_edge_dict[cast(str, module)]["edges"].append(edge_dict)
        else:
            graphviz_graph.edge(**edge_dict)


def _node_has_grad(layer: Any) -> bool:
    """Return whether a rendered node has any saved grad.

    Parameters
    ----------
    layer:
        ``OpLog`` or rolled ``LayerLog``.

    Returns
    -------
    bool
        True if the node has at least one saved grad tensor.
    """

    ops = getattr(layer, "ops", None)
    if ops is not None and hasattr(ops, "values"):
        return any(bool(getattr(pass_log, "has_grad", False)) for pass_log in ops.values())
    return bool(getattr(layer, "has_grad", False))


def _grad_node_name(layer: Any) -> str:
    """Return the Graphviz node name for a grad edge endpoint.

    Parameters
    ----------
    layer:
        Rendered graph node.

    Returns
    -------
    str
        Graphviz-safe node name.
    """

    return str(layer.layer_label).replace(":", "pass")


def _setup_subgraphs(
    self: "Trace",
    graphviz_graph: graphviz.Digraph,
    vis_mode: str,
    module_edge_dict: Dict[str, Any],
    overrides: Optional[VisualizationOverrides] = None,
) -> None:
    """Build nested Graphviz subgraphs for module clusters.

    Creates the module hierarchy as nested Graphviz subgraphs (clusters),
    placing edges into the appropriate depth level.  Uses a BFS-like
    approach: starts from top-level modules, builds each subgraph via
    ``_setup_subgraphs_recurse``, and pushes child modules onto a stack.

    In **unrolled** mode, each module pass is a separate subgraph (keyed by
    ``"module_addr:call_index"``).  In **rolled** mode, all ops share one
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
        for call_label, mpl in self.modules._pass_dict.items():
            module_submodule_dict[call_label] = list(mpl.call_children)
        subgraphs = list(self.modules["self"].ops[1].call_children)  # type: ignore[union-attr]
    else:
        module_submodule_dict = defaultdict(list)
        for ml in self.modules:
            if ml.address != "self":
                module_submodule_dict[ml.address] = list(ml.call_children)
        subgraphs = list(self.modules["self"].call_children)

    # Get the max module nesting depth:

    max_call_depth = _get_max_call_depth(subgraphs, module_edge_dict, module_submodule_dict)

    subgraph_stack = [[subgraph] for subgraph in subgraphs]
    call_depth = 0
    while len(subgraph_stack) > 0:
        parent_graph_list = subgraph_stack.pop(0)
        _setup_subgraphs_recurse(
            self,
            graphviz_graph,
            parent_graph_list,
            module_edge_dict,
            module_submodule_dict,
            subgraph_stack,
            call_depth,
            max_call_depth,
            vis_mode,
            overrides,  # type: ignore[arg-type]
        )


def _setup_subgraphs_recurse(
    self: "Trace",
    starting_subgraph: graphviz.Digraph,
    parent_graph_list: List[str],
    module_edge_dict: Dict[str, Any],
    module_submodule_dict: Dict[str, list[str]],
    subgraph_stack: list[list[str]],
    call_depth: int,
    max_call_depth: int,
    vis_mode: str,
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
        call_depth: Current position in ``parent_graph_list``.
        max_call_depth: Maximum depth across all branches (for penwidth scaling).
        vis_mode: ``'rolled'`` or ``'unrolled'``.
        overrides: Graphviz attribute overrides.
    """
    subgraph_name_w_pass = parent_graph_list[call_depth]
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
    module_type = sg_ml.class_name  # type: ignore[union-attr]
    if (sg_ml.num_calls > 1) and (vis_mode == "unrolled"):  # type: ignore[union-attr]
        subgraph_title = subgraph_name_w_pass
    elif (sg_ml.num_calls > 1) and (vis_mode == "rolled"):  # type: ignore[union-attr]
        subgraph_title = f"{subgraph_module} (x{sg_ml.num_calls})"  # type: ignore[union-attr]
    else:
        subgraph_title = subgraph_module

    if call_depth < len(parent_graph_list) - 1:  # we haven't gotten to the bottom yet, keep going.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            _setup_subgraphs_recurse(
                self,
                s,
                parent_graph_list,
                module_edge_dict,
                module_submodule_dict,
                subgraph_stack,
                call_depth + 1,
                max_call_depth,
                vis_mode,
                overrides,
            )

    else:  # Leaf of this branch: create the subgraph and add all edges.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            # Penwidth + cluster attrs come from ``_render_utils`` so the
            # bundle renderer in ``multi_trace/visualization.py`` can build
            # equivalent clusters with the same formula and label format.
            pen_width = compute_module_penwidth(call_depth, max_call_depth)
            if module_edge_dict[subgraph_name]["has_input_ancestor"]:
                line_style = "solid"
            else:
                line_style = "dashed"

            # ``title_already_escaped=True`` preserves the existing byte-level
            # output: Trace subgraph titles never contain HTML specials, so
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


def _get_max_call_depth(
    top_modules: list[str],
    module_edge_dict: Dict[str, Any],
    module_submodule_dict: Dict[str, list[str]],
) -> int:
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
    max_call_depth = 1
    module_depth_stack = [(graph, 1) for graph in top_modules]

    while len(module_depth_stack) > 0:
        module, module_depth = module_depth_stack.pop()
        module_edges = module_edge_dict[module]["edges"]
        module_submodules = module_submodule_dict[module]

        if (len(module_edges) == 0) and (
            len(module_submodules) == 0
        ):  # can ignore if no edges and no children.
            continue
        elif (len(module_edges) > 0) and (len(module_submodules) == 0):
            max_call_depth = max([module_depth, max_call_depth])
        elif (len(module_edges) == 0) and (len(module_submodules) > 0):
            module_depth_stack.extend(
                [(module_child, module_depth + 1) for module_child in module_submodules]
            )
        else:
            max_call_depth = max([module_depth, max_call_depth])
            module_depth_stack.extend(
                [(module_child, module_depth + 1) for module_child in module_submodules]
            )
    return max_call_depth

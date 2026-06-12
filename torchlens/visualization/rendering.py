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

import copy
import os
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Literal,
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
from graphviz.quoting import quote as quote_dot_id
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
from ..data_classes.op import Op
from ..data_classes.layer import Layer
from ..viz import batch_summary
from .modes import COLLAPSED_MODE_REGISTRY, DOMAIN_NODE_MODES, MODE_REGISTRY
from ._label_format import (
    format_memory,
    format_module_kwargs,
    format_module_path,
    format_param_list,
    format_shape,
)
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
from ._render_utils import _open_file_quietly
from .themes import (
    VisualizationTheme,
    apply_theme_to_spec,
    legend_lines,
    resolve_theme,
    theme_edge_attrs,
    theme_graph_attrs,
    theme_node_attrs,
)
from .code_panel import (
    CodePanelOption,
    compose_graph_with_code_panel,
    render_code_panel_subgraph,
    resolve_code_panel_source,
)
from ._render_utils import (
    compute_module_penwidth,
    direction_to_rankdir,
    make_module_cluster_attrs,
)

if TYPE_CHECKING:
    from ..data_classes.grad_fn import GradFn
    from ..data_classes.trace import Trace
    from ..data_classes.module import Module

BaseGraphNode = Union["Op", "Layer"]


@dataclass
class FocusNode:
    """Mutable render proxy for a focused Layer or Op.

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


@dataclass(frozen=True)
class RollingAnnotation:
    """Internal rolled-view annotation for one aggregate layer node.

    Parameters
    ----------
    call_groups:
        Optional grouped module-call partitions to show on the face.
    buffer_versions:
        Flat buffer version indices represented by this rolled buffer layer.
    """

    call_groups: tuple[tuple[int, ...], ...] = ()
    buffer_versions: tuple[int, ...] = ()


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
    is_atomic_module: bool = False
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
    step_index: int = 1
    shape: tuple[Any, ...] = ()
    activation_memory: str = "0 B"
    io_role: str = ""
    layer_type: str = "input"

    def __post_init__(self) -> None:
        """Fill mutable defaults and role flags."""

        self.is_input = self.boundary_kind == "input"
        self.is_output = self.boundary_kind == "output"
        self.layer_type = self.boundary_kind
        self.io_role = self.boundary_kind


GraphNode = Union[BaseGraphNode, BoundaryNode, FocusNode]
NodeSpecFn = Callable[["Layer", NodeSpec], NodeSpec | None]
BackwardNodeSpecFn = Callable[["GradFn", NodeSpec], NodeSpec | None]
CollapsedNodeSpecFn = Callable[["Module", NodeSpec], NodeSpec | None]
CollapseFn = Callable[["Module"], bool]
SkipFn = Callable[["Layer"], bool]
InterveningClusterMode = Literal["upstream", "outside", "downstream", "own"]
BackwardPassFilter = set[int] | None

# -- Color palette for node types --
INPUT_COLOR = "#98FB98"  # Light green
OUTPUT_COLOR = "#ff9999"  # Light red/salmon
PARAMS_NODE_BG_COLOR = "#E6E6E6"  # Generic param (no Param available)
TRAINABLE_PARAMS_BG_COLOR = "#D9D9D9"  # Light gray for trainable params
FROZEN_PARAMS_BG_COLOR = "#B0B0B0"  # Darker gray for frozen params
GRADIENT_ARROW_COLOR = "#9197F6"  # Light blue/purple for backward edges
BACKWARD_NODE_COLOR = "#F2F3FF"  # Very light blue/purple for backward grad_fn_handle nodes
BACKWARD_NODE_BORDER_COLOR = GRADIENT_ARROW_COLOR
BACKWARD_HIGHER_ORDER_COLOR = "#FFF4D6"
BACKWARD_ACCUMULATION_EDGE_STYLE = "dotted"
DEFAULT_BG_COLOR = "white"
BOOL_NODE_COLOR = "#F7D460"  # Yellow for terminal boolean layers
_NOISE_BUFFER_NAMES = frozenset({"running_mean", "running_var", "num_batches_tracked"})

# Module subgraph border widths live in ._render_utils -- both this file
# and ``multi_trace/visualization.py`` use ``compute_module_penwidth`` so
# bundle and Trace clusters scale identically by depth.

# Commutative functions: argument order doesn't matter, so we skip arg-position
# labels on their incoming edges to reduce visual clutter.
COMMUTE_FUNCS = ["add", "mul", "cat", "eq", "ne"]
SIBLING_ORDER_NODE_CAP = 2000
SIBLING_ORDER_STRETCH_CAP = 4.5
SIBLING_ORDER_EPSILON = 1e-9


def _view_rendered_file(filepath: str) -> None:
    """Open a rendered visualization file when a local viewer is available.

    Parameters
    ----------
    filepath:
        Rendered artifact path.
    """

    _open_file_quietly(filepath, announce_headless=True)


def _strip_render_extension(vis_outpath: str) -> str:
    """Return output path without a Graphviz-rendered file extension.

    Parameters
    ----------
    vis_outpath:
        User-provided render output path.

    Returns
    -------
    str
        Path without a recognized render extension.
    """

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
        "dot",
    ]:
        return ".".join(split_outpath[:-1])
    return vis_outpath


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


def _buffer_name_segment(address: str | None) -> str:
    """Return the last dotted segment of a buffer address.

    Parameters
    ----------
    address:
        Fully qualified buffer address, if available.

    Returns
    -------
    str
        Final dotted address segment, or an empty string for missing addresses.
    """

    if address is None:
        return ""
    return address.split(".")[-1]


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
    address = getattr(source_node, "address", None)
    return _buffer_name_segment(address) in _NOISE_BUFFER_NAMES


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
        if isinstance(source_node, Op):
            parent_node = trace[parent_label]
        else:
            parent_node = trace.layer_logs[parent_label]
        if not parent_node.is_buffer or _is_buffer_visible(parent_node, show_buffer_layers):
            continue
        address = parent_node.address
        if address is None or address in seen_addresses:
            continue
        hidden_addresses.append(address)
        seen_addresses.add(address)
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


@dataclass(frozen=True)
class CapturedForwardEdge:
    """Rendered forward edge captured at Graphviz edge-emission time.

    Parameters
    ----------
    source_label:
        TorchLens label for the parent node.
    target_label:
        TorchLens label for the child node.
    tail_name:
        Rendered Graphviz tail node name.
    head_name:
        Rendered Graphviz head node name.
    source_step:
        Execution step index for the parent node.
    target_step:
        Execution step index for the child node.
    source_node:
        Parent render node.
    target_node:
        Child render node.
    module_key:
        Cluster key where the real edge is emitted, or ``-1`` for top level.
    """

    source_label: str
    target_label: str
    tail_name: str
    head_name: str
    source_step: int
    target_step: int
    source_node: GraphNode
    target_node: GraphNode
    module_key: str | int


@dataclass(frozen=True)
class SiblingOrderChain:
    """Candidate same-rank sibling chain for one rendered fanout.

    Parameters
    ----------
    source_label:
        TorchLens label for the fanout source.
    source_name:
        Rendered Graphviz source node name.
    targets:
        Rendered child node names in execution order.
    target_labels:
        TorchLens child labels in execution order.
    lca_key:
        Cluster key where the rank group should be emitted, or ``-1`` for top level.
    """

    source_label: str
    source_name: str
    targets: tuple[str, ...]
    target_labels: tuple[str, ...]
    lca_key: str | int


@dataclass(frozen=True)
class PlainLayout:
    """Subset of ``dot -Tplain`` layout data needed by the verifier.

    Parameters
    ----------
    nodes:
        Mapping from rendered node name to ``(x, y)`` coordinates.
    edge_spans:
        Mapping from rendered real edge ``(tail, head)`` to flow-axis span.
    """

    nodes: dict[str, tuple[float, float]]
    edge_spans: dict[tuple[str, str], float]


@dataclass(frozen=True)
class SiblingOrderDecision:
    """Recorded sibling-ordering decision for one draw call."""

    candidate_count: int
    survivor_count: int
    ratios: dict[tuple[str, tuple[str, ...]], float]
    surviving_keys: tuple[tuple[str, tuple[str, ...]], ...]


def draw(
    self: "Trace",
    vis_mode: VisModeLiteral = "unrolled",
    vis_call_depth: int = 1000,
    vis_outpath: str = "modelgraph",
    vis_graph_overrides: Optional[Dict[str, Any]] = None,
    module: "Module | str | None" = None,
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
    order_siblings: bool = True,
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
        module: Optional module focus. A Module focuses that module; a string
            is interpreted as a module address.
        node_mode: Preset applied to default ``NodeSpec`` objects before
            user callbacks run.
        node_spec_fn: Optional callback receiving ``(layer_log, default_spec)``.
            In unrolled mode, ``layer_log`` is the parent aggregate Layer for
            the rendered Op.
        collapsed_node_spec_fn: Optional callback receiving
            ``(module_log, default_spec)`` for collapsed module nodes.
        collapse_fn: Optional predicate receiving a Module. When provided,
            it replaces ``vis_call_depth`` collapse decisions.
        skip_fn: Optional predicate receiving a Layer. Skipped nodes are
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
        vis_node_placement: Layout engine: ``'auto'`` (default), ``'dot'``,
            or ``'rank'``.
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
        order_siblings: Whether Graphviz ``dot`` renders should add verified invisible
            rank constraints so true parallel sibling fanouts follow execution order.

    Returns:
        The Graphviz DOT source string.

    Raises:
        ValueError: If ``_layers_logged`` is False (layers were discarded
            by missing final lookup containers).
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

    # THE _layers_logged guard: protects all downstream rendering code from missing-layer lookups.
    if not self._layers_logged:
        raise ValueError(
            "Must have all layers logged in order to render the graph; use show_model_graph."
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

    # Unrolled: iterate Op objects (one node per pass).
    # Rolled: iterate Layer objects (one node per logical layer, multi-pass
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
    from ._rank_layout_internal.layout import (
        RANK_LAYOUT_COST_THRESHOLD,
        RANK_LAYOUT_NOTICE,
        estimate_rank_layout_cost,
        get_node_placement_engine,
    )

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
    cost_node_labels, cost_edges = _rank_layout_cost_inputs(
        self,
        entries_to_plot,
        edge_map,
        vis_mode=vis_mode,
        vis_call_depth=vis_call_depth,
        collapse_fn=collapse_fn,
    )
    layout_cost = estimate_rank_layout_cost(cost_node_labels, cost_edges)
    engine = get_node_placement_engine(vis_node_placement, layout_cost)
    # The sibling-ordering post-pass only runs on the dot engine; set the
    # trivial decision up front so the attribute exists on every path.
    self._last_sibling_ordering_decision = SiblingOrderDecision(0, 0, {}, ())
    if vis_node_placement == "auto" and engine == "rank":
        warnings.warn(
            RANK_LAYOUT_NOTICE.format(
                cost=layout_cost,
                threshold=RANK_LAYOUT_COST_THRESHOLD,
            )
        )
    _vprint(self, f"Rendering {vis_mode} graph ({num_nodes} nodes, format={vis_fileformat})")
    _vprint(self, f"Layout engine: {engine} (estimated cost={layout_cost})")

    if self.num_params == 0:
        params_detail = "0 params"
    elif self.num_params_frozen == 0:
        params_detail = f"{self.num_params} params (all trainable, {self.total_param_memory})"
    elif self.num_params_trainable == 0:
        params_detail = f"{self.num_params} params (all frozen, {self.total_param_memory})"
    else:
        params_detail = (
            f"{self.num_params} params "
            f"({self.num_params_trainable}/{self.num_params} trainable, "
            f"{self.total_param_memory})"
        )

    graph_caption = (
        f"<<B>{self.model_class_name}</B><br align='left'/>{self.num_tensors} "
        f"tensors total ({self.total_activation_memory})"
        f"<br align='left'/>{params_detail}<br align='left'/>>"
    )
    if getattr(self, "_has_direct_writes", False):
        graph_caption = graph_caption[:-2] + (
            "Direct writes detected - recipe propagation will overlay<br align='left'/>>"
        )

    # Rank fast path: skip graphviz.Digraph construction entirely.
    # Generates DOT directly with topological-rank positions and cluster boxes.
    if engine == "rank":
        from ._rank_layout_internal.layout import render_rank_layout

        result = render_rank_layout(
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
        name=self.model_class_name,
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
        lambda: {"edges": [], "has_input_ancestor": False, "rank_groups": []}
    )
    top_level_sibling_rank_groups: list[SiblingOrderChain] = []
    # Track which collapsed module nodes have been added to avoid duplicates
    # (multiple layers in the same collapsed module would otherwise each try
    # to create the same box3d node).
    collapsed_modules: Set[str] = set()
    # Edge deduplication: (tail_name, head_name) pairs already added.
    # Critical when collapsed modules cause many layers to map to the same
    # node name -- without this, we'd get duplicate edges.
    edges_used: Set[tuple[str, str]] = set()
    captured_forward_edges: list[CapturedForwardEdge] = []

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
            captured_forward_edges,
            rankdir,
        )

    if vis_intervention_mode == "as_node":
        _add_intervention_hook_nodes(dot, site_labels, vis_graph_overrides)

    sibling_order_chains: tuple[SiblingOrderChain, ...] = ()
    if _should_order_siblings(
        order_siblings=order_siblings,
        engine=engine,
        vis_mode=vis_mode,
        num_nodes=num_nodes,
        module=module,
        vis_intervention_mode=vis_intervention_mode,
        collapse_fn=collapse_fn,
        vis_call_depth=vis_call_depth,
    ):
        sibling_order_chains = _build_sibling_order_chains(captured_forward_edges)
        if sibling_order_chains:
            for chain in sibling_order_chains:
                _queue_sibling_rank_group(
                    module_cluster_dict,
                    top_level_sibling_rank_groups,
                    chain,
                )

    # Finally, set up the subgraphs.
    _setup_subgraphs(
        self,
        dot,
        vis_mode,
        module_cluster_dict,
        overrides,
        top_level_sibling_rank_groups,
    )
    if show_legend:
        _add_legend_to_graphviz(dot, theme)
    # A code panel is composed side by side (separate render) when the output
    # format supports it, so the code never distorts the graph's layout. Only
    # fall back to an in-graph subgraph for formats we cannot compose.
    compose_code_panel = source_text is not None and _code_panel_composition_available(
        vis_fileformat, engine
    )
    if source_text is not None and not compose_code_panel:
        render_code_panel_subgraph(dot, source_text)

    if in_notebook() and not vis_save_only:
        try:
            from IPython.display import SVG, display  # #72: lazy import
        except ImportError as e:
            raise ImportError(
                "IPython is required for this feature. Install with "
                "`pip install torchlens[notebook]`."
            ) from e

        display_fn = cast(Any, display)
        if compose_code_panel:
            # Compose the graph SVG beside a standalone code panel so the inline
            # preview matches the saved output and leaves the graph undistorted.
            graph_svg = dot.pipe(format="svg").decode("utf-8")
            combined_svg = compose_graph_with_code_panel(graph_svg, cast(str, source_text))
            display_fn(SVG(combined_svg))
        else:
            display_fn(dot)

    # Rank was already handled above (early return). Only dot reaches here.
    _RENDER_TIMEOUT = 120  # seconds
    source_override = None
    self._last_sibling_ordering_decision = SiblingOrderDecision(0, 0, {}, ())
    if sibling_order_chains:
        source_override, decision = _verify_and_apply_sibling_ordering(
            dot.source,
            sibling_order_chains,
            captured_forward_edges,
            rankdir,
        )
        self._last_sibling_ordering_decision = decision

    final_source = source_override if source_override is not None else dot.source
    source_path = dot.save(vis_outpath)
    with open(source_path, "w", encoding="utf-8") as source_file:
        source_file.write(final_source)
    try:
        # dot engine (default for local-topology graphs)
        rendered_path = f"{vis_outpath}.{vis_fileformat}"
        if compose_code_panel:
            _write_composed_code_panel(
                dot.engine,
                source_path,
                cast(str, source_text),
                rendered_path,
                vis_fileformat,
                _RENDER_TIMEOUT,
            )
        else:
            cmd = [dot.engine, f"-T{vis_fileformat}", "-o", rendered_path, source_path]
            subprocess.run(cmd, timeout=_RENDER_TIMEOUT, check=True, capture_output=True)
        if not vis_save_only:
            _view_rendered_file(rendered_path)
        _vprint(self, f"Graph saved to {vis_outpath}.{vis_fileformat}")
    except subprocess.TimeoutExpired:
        warnings.warn(
            f"Graphviz render timed out ({_RENDER_TIMEOUT}s) for graph with "
            f"{self.num_tensors} nodes. DOT source saved to "
            f"'{source_path}'. Consider using vis_node_placement='rank' or "
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
    return final_source


_CODE_PANEL_COMPOSED_FORMATS = frozenset({"svg", "pdf", "png"})


def _code_panel_composition_available(file_format: str, engine: str) -> bool:
    """Return whether a code panel can be composed side by side for this output.

    Side-by-side composition renders the graph and the code panel as separate
    SVGs and joins them, which keeps the graph's proportions untouched (the code
    no longer participates in Graphviz layout). It needs a vector-capable target
    format and the ``cairosvg`` rasterizer for non-SVG outputs.
    """

    if engine == "rank" or file_format not in _CODE_PANEL_COMPOSED_FORMATS:
        return False
    if file_format == "svg":
        return True
    try:
        import cairosvg  # noqa: F401
    except Exception:
        return False
    return True


def _render_graph_only_svg(engine: str, source_path: str, timeout: int) -> str:
    """Render a saved DOT source to an SVG string (no code panel)."""

    completed = subprocess.run(
        [engine, "-Tsvg", source_path],
        timeout=timeout,
        check=True,
        capture_output=True,
    )
    return completed.stdout.decode("utf-8")


def _write_composed_code_panel(
    engine: str,
    source_path: str,
    source_text: str,
    rendered_path: str,
    file_format: str,
    timeout: int,
) -> None:
    """Render the graph and code panel separately and write the joined output.

    The graph is rendered to SVG without any code subgraph, composed beside a
    standalone code-panel SVG, then written in ``file_format``. SVG is written
    directly; PDF and PNG are converted from the composed SVG with ``cairosvg``
    so vectors are preserved.
    """

    graph_svg = _render_graph_only_svg(engine, source_path, timeout)
    combined_svg = compose_graph_with_code_panel(graph_svg, source_text)
    if file_format == "svg":
        with open(rendered_path, "w", encoding="utf-8") as svg_file:
            svg_file.write(combined_svg)
        return
    import cairosvg

    svg_bytes = combined_svg.encode("utf-8")
    if file_format == "pdf":
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=rendered_path)
    else:  # png, rendered at 2x for crispness
        cairosvg.svg2png(bytestring=svg_bytes, write_to=rendered_path, scale=2.0)


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
    vis_mode: VisModeLiteral = "rolled",
    bwd: int | Iterable[int] | None = None,
) -> str:
    """Render the captured backward grad_fn_handle DAG as a Graphviz graph.

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
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    collapsed_node_spec_fn:
        Accepted for API symmetry with forward visualization. Not applied
        because backward graphs do not render collapsed module nodes.
    vis_node_mode:
        Accepted for API symmetry with forward visualization. Not applied to
        grad_fn_handle nodes.
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
    vis_mode:
        ``"rolled"`` renders one node per GradFn. ``"unrolled"`` renders one
        node per GradFnCall, grouped into backward-pass clusters.
    bwd:
        Optional one-based backward pass number or numbers to render.

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
    if vis_mode not in {"rolled", "unrolled"}:
        raise ValueError("vis_mode must be either 'rolled' or 'unrolled'")
    pass_filter = _normalize_backward_pass_filter(bwd)

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
        f"<<B>{self.model_class_name} backward graph</B><br align='left'/>"
        f"{self.num_grad_fns} grad_fn_handle nodes"
        f"<br align='left'/>{self.num_backward_passes} backward pass(es)"
        f"{_format_backward_filter_caption(pass_filter)}"
        f"<br align='left'/>mode: {vis_mode}<br align='left'/>>"
    )
    dot = graphviz.Digraph(
        name=f"{self.model_class_name}_backward",
        comment="Backward grad_fn_handle graph",
        format=vis_fileformat,
    )
    graph_args = {
        "rankdir": rankdir,
        "label": graph_caption,
        "labelloc": "t",
        "labeljust": "left",
        "ordering": "out",
        "compound": "true",
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

    if vis_mode == "rolled":
        visible_ids = {
            grad_fn_handle.grad_fn_object_id
            for grad_fn_handle in self.grad_fns
            if _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter)
        }
        for grad_fn_handle in self.grad_fns:
            if grad_fn_handle.grad_fn_object_id in visible_ids:
                _add_backward_node_to_graphviz(
                    grad_fn_handle,
                    dot,
                    node_spec_fn,
                    pass_filter=pass_filter,
                )

        for grad_fn_handle in self.grad_fns:
            if grad_fn_handle.grad_fn_object_id not in visible_ids:
                continue
            tail_name = _backward_dot_node_name(grad_fn_handle)
            for next_grad_fn_id in grad_fn_handle.next_grad_fn_ids:
                if next_grad_fn_id not in visible_ids:
                    continue
                head_name = _backward_dot_node_name(self.grad_fn_logs[next_grad_fn_id])
                dot.edge(
                    tail_name,
                    head_name,
                    **_backward_edge_attrs(grad_fn_handle, self.grad_fn_logs[next_grad_fn_id]),
                )
    else:
        _add_unrolled_backward_pass_clusters(self, dot, node_spec_fn, pass_filter)

    source_text = resolve_code_panel_source(
        code_panel,
        getattr(self, "_source_code_blob", {}),
        getattr(self, "_source_model_ref", None),
    )
    # Compose the code panel side by side when the format supports it so it never
    # distorts the backward graph; otherwise fall back to an in-graph subgraph.
    compose_code_panel = source_text is not None and _code_panel_composition_available(
        vis_fileformat, dot.engine
    )
    if source_text is not None and not compose_code_panel:
        render_code_panel_subgraph(dot, source_text)

    if in_notebook() and not vis_save_only:
        try:
            from IPython.display import SVG, display
        except ImportError as e:
            raise ImportError(
                "IPython is required for this feature. Install with "
                "`pip install torchlens[notebook]`."
            ) from e

        display_fn = cast(Any, display)
        if compose_code_panel:
            graph_svg = dot.pipe(format="svg").decode("utf-8")
            combined_svg = compose_graph_with_code_panel(graph_svg, cast(str, source_text))
            display_fn(SVG(combined_svg))
        else:
            display_fn(dot)

    _RENDER_TIMEOUT = 120
    source_path = dot.save(vis_outpath)
    try:
        rendered_path = f"{vis_outpath}.{vis_fileformat}"
        if compose_code_panel:
            _write_composed_code_panel(
                dot.engine,
                source_path,
                cast(str, source_text),
                rendered_path,
                vis_fileformat,
                _RENDER_TIMEOUT,
            )
        else:
            cmd = [dot.engine, f"-T{vis_fileformat}", "-o", rendered_path, source_path]
            subprocess.run(cmd, timeout=_RENDER_TIMEOUT, check=True, capture_output=True)
        if not vis_save_only:
            _view_rendered_file(rendered_path)
        _vprint(self, f"Backward graph saved to {vis_outpath}.{vis_fileformat}")
    except subprocess.TimeoutExpired:
        warnings.warn(
            f"Graphviz render timed out ({_RENDER_TIMEOUT}s) for backward graph with "
            f"{self.num_grad_fns} grad_fn_handle nodes. DOT source saved to '{source_path}'."
        )
    except subprocess.CalledProcessError as e:
        warnings.warn(f"Graphviz render failed: {e.stderr.decode()}")
    finally:
        import os

        if os.path.exists(source_path):
            os.remove(source_path)
    return cast(str, dot.source)


def render_combined_graph(
    self: "Trace",
    vis_outpath: str = "combined_modelgraph",
    vis_graph_overrides: Optional[Dict[str, Any]] = None,
    node_spec_fn: NodeSpecFn | None = None,
    backward_node_spec_fn: BackwardNodeSpecFn | None = None,
    vis_edge_overrides: Optional[Dict[str, Any]] = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
    direction: VisDirectionLiteral = "leftright",
    vis_mode: VisModeLiteral = "unrolled",
    intervening_cluster: InterveningClusterMode = "upstream",
    show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
    bwd: int | Iterable[int] | None = None,
) -> str:
    """Render one Graphviz graph containing forward ops and backward grad_fns.

    Parameters
    ----------
    self:
        Trace containing forward and explicit backward metadata.
    vis_outpath:
        Output path for the rendered graph.
    vis_graph_overrides:
        Graphviz graph-level overrides.
    node_spec_fn:
        Optional callback receiving ``(layer_log, default_spec)`` for forward nodes.
    backward_node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)`` for backward nodes.
    vis_edge_overrides:
        Graphviz edge-level overrides applied to forward edges.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output format.
    direction:
        Layout direction.
    vis_mode:
        Combined rendering currently supports only ``"unrolled"``.
    intervening_cluster:
        Placement mode for grad_fns that have no corresponding forward op.
    show_buffer_layers:
        Buffer visibility mode for the forward side.
    bwd:
        Optional one-based backward pass number or numbers to render.

    Returns
    -------
    str
        Graphviz DOT source.

    Raises
    ------
    NotImplementedError
        If ``vis_mode="rolled"`` is requested.
    ValueError
        If no explicit backward graph has been captured.
    """

    if vis_mode == "rolled":
        raise NotImplementedError("draw_combined does not support vis_mode='rolled' yet.")
    if vis_mode != "unrolled":
        raise ValueError("vis_mode must be either 'unrolled' or 'rolled'")
    if not self.has_backward_pass or not self.grad_fn_logs:
        raise ValueError("No backward graph is available; call log_backward(loss) first.")
    if not self._layers_logged:
        raise ValueError(
            "Must have all layers logged in order to render the graph; use show_model_graph."
        )
    pass_filter = _normalize_backward_pass_filter(bwd)

    show_buffer_layers = _normalize_buffer_visibility(show_buffer_layers)
    vis_outpath = _strip_render_extension(vis_outpath)
    rankdir = direction_to_rankdir(direction)
    overrides = VisualizationOverrides(
        graph=graphviz_graph_overrides(vis_graph_overrides),
        edge=vis_edge_overrides or {},
        grad_edge={},
        module={},
    )

    graph_caption = (
        f"<<B>{self.model_class_name} combined forward/backward graph</B><br align='left'/>"
        f"{self.num_tensors} forward nodes, {self.num_grad_fns} grad_fn_handle nodes"
        f"<br align='left'/>{self.num_backward_passes} backward pass(es)"
        f"{_format_backward_filter_caption(pass_filter)}<br align='left'/>>"
    )
    dot = graphviz.Digraph(
        name=f"{self.model_class_name}_combined",
        comment="Combined forward and backward graph",
        format=vis_fileformat,
    )
    graph_args = {
        "rankdir": rankdir,
        "label": graph_caption,
        "labelloc": "t",
        "labeljust": "left",
        "ordering": "out",
    }
    for arg_name, arg_val in overrides.graph.items():  # type: ignore[union-attr]
        graph_args[arg_name] = str(arg_val(self) if callable(arg_val) else arg_val)
    dot.graph_attr.update(graph_args)
    dot.node_attr.update({"ordering": "out"})
    dot.edge_attr.update({"ordering": "out"})

    module_cluster_dict: Dict[str, Any] = defaultdict(
        lambda: {"edges": [], "nodes": [], "has_input_ancestor": False}
    )
    entries_to_plot: dict[str, GraphNode] = dict(self.layer_dict_main_keys)
    edge_map, skipped_labels = _build_skip_filtered_edge_map(
        self,
        entries_to_plot,
        vis_mode="unrolled",
        show_buffer_layers=show_buffer_layers,
        skip_fn=None,
    )
    edges_used: Set[tuple[str, str]] = set()
    collapsed_modules: Set[str] = set()
    for node in entries_to_plot.values():
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
            "unrolled",
            collapsed_modules,
            show_buffer_layers=show_buffer_layers,
            overrides=overrides,
            node_spec_fn=node_spec_fn,
            edge_map=edge_map,
        )

    _add_combined_backward_nodes(
        self,
        module_cluster_dict,
        dot,
        backward_node_spec_fn,
        intervening_cluster,
        pass_filter,
    )
    _add_combined_backward_edges(self, dot, pass_filter)
    _add_combined_correspondence_edges(self, dot, intervening_cluster, pass_filter)
    _setup_subgraphs(self, dot, "unrolled", module_cluster_dict, overrides)
    _setup_combined_special_clusters(dot, module_cluster_dict)

    _RENDER_TIMEOUT = 120
    source_path = dot.save(vis_outpath)
    try:
        rendered_path = f"{vis_outpath}.{vis_fileformat}"
        cmd = [dot.engine, f"-T{vis_fileformat}", "-o", rendered_path, source_path]
        subprocess.run(cmd, timeout=_RENDER_TIMEOUT, check=True, capture_output=True)
        if not vis_save_only:
            _view_rendered_file(rendered_path)
        _vprint(self, f"Combined graph saved to {vis_outpath}.{vis_fileformat}")
    except subprocess.TimeoutExpired:
        warnings.warn(
            f"Graphviz render timed out ({_RENDER_TIMEOUT}s) for combined graph with "
            f"{self.num_tensors + self.num_grad_fns} nodes. DOT source saved to '{source_path}'."
        )
    except subprocess.CalledProcessError as e:
        warnings.warn(f"Graphviz render failed: {e.stderr.decode()}")
    finally:
        if os.path.exists(source_path):
            os.remove(source_path)
    return cast(str, dot.source)


def _backward_dot_node_name(grad_fn_handle: "GradFn") -> str:
    """Return a DOT-safe node name for a grad_fn_handle log.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to name.

    Returns
    -------
    str
        DOT-safe node identifier.
    """

    return f"grad_fn_{grad_fn_handle.grad_fn_object_id}"


def _backward_dot_call_node_name(grad_fn_handle: "GradFn", call: Any) -> str:
    """Return a DOT-safe node name for one GradFnCall.

    Parameters
    ----------
    grad_fn_handle:
        GradFn owning the call.
    call:
        GradFnCall-like record.

    Returns
    -------
    str
        DOT-safe node identifier.
    """

    return (
        f"grad_fn_{grad_fn_handle.grad_fn_object_id}_"
        f"bwd{getattr(call, 'backward_pass_index', 0)}_call{getattr(call, 'call_index', 0)}"
    )


def _normalize_backward_pass_filter(bwd: int | Iterable[int] | None) -> BackwardPassFilter:
    """Normalize a backward-pass render filter.

    Parameters
    ----------
    bwd:
        Optional one-based backward pass number or iterable of pass numbers.

    Returns
    -------
    set[int] | None
        Pass numbers to render, or ``None`` for all passes.
    """

    if bwd is None:
        return None
    if isinstance(bwd, int):
        pass_indices = {bwd}
    else:
        pass_indices = {int(pass_index) for pass_index in bwd}
    if any(pass_index < 1 for pass_index in pass_indices):
        raise ValueError("bwd pass filters use one-based positive backward pass numbers.")
    return pass_indices


def _format_backward_filter_caption(pass_filter: BackwardPassFilter) -> str:
    """Return a compact caption suffix for a backward-pass filter.

    Parameters
    ----------
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    str
        Human-readable caption suffix.
    """

    if pass_filter is None:
        return ""
    return f" shown: {int_list_to_compact_str(sorted(pass_filter))}"


def _grad_fn_call_matches_backward_filter(call: Any, pass_filter: BackwardPassFilter) -> bool:
    """Return whether a GradFnCall should be visible for a pass filter.

    Parameters
    ----------
    call:
        GradFnCall-like record.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    bool
        ``True`` when the call participates in a requested pass.
    """

    if pass_filter is None:
        return True
    return getattr(call, "backward_pass_index", None) in pass_filter


def _grad_fn_matches_backward_filter(
    grad_fn_handle: "GradFn",
    pass_filter: BackwardPassFilter,
) -> bool:
    """Return whether a GradFn has at least one visible call.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to inspect.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    bool
        ``True`` when any call participates in the selected passes.
    """

    return any(
        _grad_fn_call_matches_backward_filter(call, pass_filter)
        for call in grad_fn_handle.calls.values()
    )


def _add_backward_node_to_graphviz(
    grad_fn_handle: "GradFn",
    graphviz_graph: graphviz.Digraph,
    node_spec_fn: BackwardNodeSpecFn | None,
    pass_filter: BackwardPassFilter = None,
) -> None:
    """Add one backward grad_fn_handle node to a Graphviz graph.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to render.
    graphviz_graph:
        Graphviz Digraph object.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    pass_filter:
        Normalized backward-pass filter.
    """

    node_args = _backward_node_graphviz_args(
        grad_fn_handle,
        node_spec_fn,
        pass_filter=pass_filter,
    )
    graphviz_graph.node(**node_args)


def _backward_node_graphviz_args(
    grad_fn_handle: "GradFn",
    node_spec_fn: BackwardNodeSpecFn | None,
    call: Any | None = None,
    pass_filter: BackwardPassFilter = None,
) -> dict[str, Any]:
    """Build Graphviz node arguments for one backward grad_fn_handle.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to render.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    call:
        Optional GradFnCall when rendering in unrolled mode.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    dict[str, Any]
        Keyword arguments accepted by ``graphviz.Digraph.node``.
    """

    default_spec = NodeSpec(
        lines=_compute_backward_node_lines(
            grad_fn_handle,
            call=call,
            pass_filter=pass_filter,
        ),
        shape="oval",
        fillcolor=_backward_node_fillcolor(grad_fn_handle),
        fontcolor="black",
        color=BACKWARD_NODE_BORDER_COLOR,
        style="filled,solid",
        penwidth=1.8,
        extra_attrs={"ordering": "out"},
    )
    if node_spec_fn is not None:
        result = node_spec_fn(grad_fn_handle, default_spec)
        spec = default_spec if result is None else result
    else:
        spec = default_spec
    node_args = _node_spec_to_graphviz_args(spec)
    node_args["name"] = (
        _backward_dot_node_name(grad_fn_handle)
        if call is None
        else _backward_dot_call_node_name(grad_fn_handle, call)
    )
    return node_args


def _backward_node_fillcolor(grad_fn_handle: "GradFn") -> str:
    """Return the fill color for a backward node.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to style.

    Returns
    -------
    str
        Graphviz fill color.
    """

    order = getattr(grad_fn_handle, "order", None)
    if order is not None and order > 1:
        return BACKWARD_HIGHER_ORDER_COLOR
    return BACKWARD_NODE_COLOR


def _backward_edge_attrs(tail: "GradFn", head: "GradFn") -> dict[str, str]:
    """Return Graphviz attributes for a backward GradFn edge.

    Parameters
    ----------
    tail:
        Edge tail GradFn.
    head:
        Edge head GradFn.

    Returns
    -------
    dict[str, str]
        Graphviz edge attributes.
    """

    edge_attrs = {"color": GRADIENT_ARROW_COLOR, "fontcolor": GRADIENT_ARROW_COLOR}
    if tail.type == "accumulategrad" or head.type == "accumulategrad":
        edge_attrs["style"] = BACKWARD_ACCUMULATION_EDGE_STYLE
        edge_attrs["label"] = "accum"
        edge_attrs["labelfontsize"] = "8"
    return edge_attrs


def _add_unrolled_backward_pass_clusters(
    trace: "Trace",
    graphviz_graph: graphviz.Digraph,
    node_spec_fn: BackwardNodeSpecFn | None,
    pass_filter: BackwardPassFilter,
) -> None:
    """Render GradFnCall nodes grouped by backward pass.

    Parameters
    ----------
    trace:
        Trace containing backward projections.
    graphviz_graph:
        Graphviz graph being rendered.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    pass_filter:
        Normalized backward-pass filter.
    """

    calls_by_pass = _visible_backward_calls_by_pass(trace, pass_filter)
    for pass_index, grad_fn_calls in calls_by_pass.items():
        with graphviz_graph.subgraph(name=f"cluster_backward_pass_{pass_index}") as subgraph:
            subgraph.attr(
                label=f"backward pass {pass_index}",
                color=GRADIENT_ARROW_COLOR,
                fontcolor="black",
                style="rounded,dashed",
            )
            for grad_fn_handle, call in grad_fn_calls:
                node_args = _backward_node_graphviz_args(grad_fn_handle, node_spec_fn, call=call)
                subgraph.node(**node_args)

    for pass_index, grad_fn_calls in calls_by_pass.items():
        calls_for_grad_fn: dict[int, list[tuple["GradFn", Any]]] = defaultdict(list)
        for grad_fn_handle, call in grad_fn_calls:
            calls_for_grad_fn[grad_fn_handle.grad_fn_object_id].append((grad_fn_handle, call))
        for grad_fn_handle, call in grad_fn_calls:
            tail_name = _backward_dot_call_node_name(grad_fn_handle, call)
            for next_grad_fn_id in grad_fn_handle.next_grad_fn_ids:
                for head_grad_fn, head_call in calls_for_grad_fn.get(next_grad_fn_id, []):
                    head_name = _backward_dot_call_node_name(head_grad_fn, head_call)
                    graphviz_graph.edge(
                        tail_name,
                        head_name,
                        **_backward_edge_attrs(grad_fn_handle, head_grad_fn),
                    )


def _visible_backward_calls_by_pass(
    trace: "Trace",
    pass_filter: BackwardPassFilter,
) -> dict[int, list[tuple["GradFn", Any]]]:
    """Group visible GradFnCalls by backward pass.

    Parameters
    ----------
    trace:
        Trace containing backward projections.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    dict[int, list[tuple[GradFn, Any]]]
        Visible calls keyed by one-based backward pass number.
    """

    calls_by_pass: dict[int, list[tuple["GradFn", Any]]] = defaultdict(list)
    for grad_fn_handle in trace.grad_fns:
        for call in grad_fn_handle.calls.values():
            if not _grad_fn_call_matches_backward_filter(call, pass_filter):
                continue
            pass_index = getattr(call, "backward_pass_index", None)
            if pass_index is None:
                continue
            calls_by_pass[int(pass_index)].append((grad_fn_handle, call))
    return dict(sorted(calls_by_pass.items()))


def _add_combined_backward_nodes(
    trace: "Trace",
    module_cluster_dict: Dict[str, Any],
    graphviz_graph: graphviz.Digraph,
    node_spec_fn: BackwardNodeSpecFn | None,
    intervening_cluster: InterveningClusterMode,
    pass_filter: BackwardPassFilter,
) -> None:
    """Add backward nodes to the combined graph and module clusters.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    module_cluster_dict:
        Shared module cluster accumulator.
    graphviz_graph:
        Graphviz graph being rendered.
    node_spec_fn:
        Optional backward node callback.
    intervening_cluster:
        Placement mode for intervening grad_fns.
    pass_filter:
        Normalized backward-pass filter.
    """

    for grad_fn_handle in trace.grad_fns:
        if not _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter):
            continue
        node_args = _backward_node_graphviz_args(
            grad_fn_handle,
            node_spec_fn,
            pass_filter=pass_filter,
        )
        module_key = _module_key_for_grad_fn(trace, grad_fn_handle, intervening_cluster)
        if module_key is None:
            graphviz_graph.node(**node_args)
            continue
        module_cluster_dict[module_key]["nodes"].append(node_args)
        module_cluster_dict[module_key]["has_input_ancestor"] = True


def _add_combined_backward_edges(
    trace: "Trace",
    graphviz_graph: graphviz.Digraph,
    pass_filter: BackwardPassFilter,
) -> None:
    """Add backward grad_fn_handle edges to a combined graph.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    graphviz_graph:
        Graphviz graph being rendered.
    pass_filter:
        Normalized backward-pass filter.
    """

    visible_ids = {
        grad_fn_handle.grad_fn_object_id
        for grad_fn_handle in trace.grad_fns
        if _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter)
    }
    for grad_fn_handle in trace.grad_fns:
        if grad_fn_handle.grad_fn_object_id not in visible_ids:
            continue
        tail_name = _backward_dot_node_name(grad_fn_handle)
        for next_grad_fn_id in grad_fn_handle.next_grad_fn_ids:
            if next_grad_fn_id not in visible_ids:
                continue
            head_name = _backward_dot_node_name(trace.grad_fn_logs[next_grad_fn_id])
            graphviz_graph.edge(
                tail_name,
                head_name,
                **_backward_edge_attrs(grad_fn_handle, trace.grad_fn_logs[next_grad_fn_id]),
            )


def _add_combined_correspondence_edges(
    trace: "Trace",
    graphviz_graph: graphviz.Digraph,
    intervening_cluster: InterveningClusterMode,
    pass_filter: BackwardPassFilter,
) -> None:
    """Add dashed forward-to-backward correspondence edges.

    Parameters
    ----------
    trace:
        Trace containing paired forward and grad_fn_handle metadata.
    graphviz_graph:
        Graphviz graph being rendered.
    intervening_cluster:
        Placement mode used to infer optional cluster boundary attributes.
    pass_filter:
        Normalized backward-pass filter.
    """

    for grad_fn_handle in trace.grad_fns:
        if not grad_fn_handle.has_op:
            continue
        if not _grad_fn_matches_backward_filter(grad_fn_handle, pass_filter):
            continue
        edge_attrs = {
            "color": GRADIENT_ARROW_COLOR,
            "fontcolor": GRADIENT_ARROW_COLOR,
            "style": "dashed",
            "constraint": "false",
            "arrowsize": ".6",
        }
        module_key = _module_key_for_grad_fn(trace, grad_fn_handle, intervening_cluster)
        if module_key is not None:
            cluster_name = f"cluster_{module_key.replace(':', '_pass')}"
            edge_attrs["ltail"] = cluster_name
            edge_attrs["lhead"] = cluster_name
        op = grad_fn_handle.op
        if op is not None:
            graphviz_graph.edge(
                op.layer_label,
                _backward_dot_node_name(grad_fn_handle),
                **edge_attrs,
            )


def _setup_combined_special_clusters(
    graphviz_graph: graphviz.Digraph,
    module_cluster_dict: Dict[str, Any],
) -> None:
    """Render non-module combined graph clusters.

    Parameters
    ----------
    graphviz_graph:
        Graphviz graph being rendered.
    module_cluster_dict:
        Shared module cluster accumulator.
    """

    cluster_data = module_cluster_dict.get("__intervening__")
    if cluster_data is None:
        return
    with graphviz_graph.subgraph(name="cluster___intervening__") as subgraph:
        subgraph.attr(
            label="intervening grad_fns",
            color=GRADIENT_ARROW_COLOR,
            fontcolor="black",
            style="dashed",
        )
        for node_args in cluster_data.get("nodes", []):
            subgraph.node(**node_args)
        for edge_dict in cluster_data.get("edges", []):
            subgraph.edge(**edge_dict)


def _module_key_for_grad_fn(
    trace: "Trace",
    grad_fn_handle: "GradFn",
    mode: InterveningClusterMode,
) -> str | None:
    """Return the module cluster key for a grad_fn_handle in combined rendering.

    Parameters
    ----------
    trace:
        Trace containing forward, backward, and parameter metadata.
    grad_fn_handle:
        GradFn to place.
    mode:
        Placement mode for intervening grad_fns.

    Returns
    -------
    str | None
        Unrolled module-call key, special cluster key, or None for top level.
    """

    op = grad_fn_handle.op
    if op is not None:
        return _module_key_for_forward_op(op)
    if grad_fn_handle.type == "accumulategrad":
        param_key = _param_module_for_accumulate_grad(trace, grad_fn_handle)
        if param_key is not None:
            return param_key
    if mode == "outside":
        return None
    if mode == "own":
        return "__intervening__"
    if mode == "upstream":
        return _infer_intervening_module_upstream(trace, grad_fn_handle)
    if mode == "downstream":
        return _infer_intervening_module_downstream(trace, grad_fn_handle)
    raise ValueError("intervening_cluster must be 'upstream', 'outside', 'downstream', or 'own'.")


def _module_key_for_forward_op(op: "Layer") -> str | None:
    """Return the unrolled module cluster key for a forward op.

    Parameters
    ----------
    op:
        Forward operation or layer log associated with a grad_fn_handle.

    Returns
    -------
    str | None
        Module-call key or None for top-level ops.
    """

    output_modules = list(getattr(op, "output_of_modules", []) or [])
    if getattr(op, "is_module_output", False) and output_modules:
        output_module = str(output_modules[0])
        output_calls = list(getattr(op, "output_of_module_calls", []) or [])
        for output_call in output_calls:
            if str(output_call).split(":", 1)[0] == output_module:
                return str(output_call)
        return f"{output_module}:1"
    modules = list(getattr(op, "modules", []) or [])
    if not modules:
        return None
    return str(modules[-1])


def _param_module_for_accumulate_grad(trace: "Trace", grad_fn_handle: "GradFn") -> str | None:
    """Return an unambiguous owning module for an AccumulateGrad node.

    Parameters
    ----------
    trace:
        Trace containing parameter metadata and grad_fn_handle parameter refs.
    grad_fn_handle:
        AccumulateGrad log.

    Returns
    -------
    str | None
        Owning module-call key, or None when attribution is missing or ambiguous.
    """

    param_address = trace._grad_fn_param_refs.get(grad_fn_handle.label)
    if param_address is None:
        return None
    param_log = trace.params[param_address]
    if param_log.co_parent_params:
        return None
    module_address = param_log.module_address
    if module_address is None:
        return None
    return f"{module_address}:1"


def _infer_intervening_module_upstream(trace: "Trace", grad_fn_handle: "GradFn") -> str | None:
    """Infer an intervening grad_fn_handle module from downstream autograd edges.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    grad_fn_handle:
        Intervening GradFn to place.

    Returns
    -------
    str | None
        Inherited module key, if a paired grad_fn_handle is reachable.
    """

    return _infer_intervening_module_bfs(trace, [grad_fn_handle.grad_fn_object_id], reverse=False)


def _infer_intervening_module_downstream(trace: "Trace", grad_fn_handle: "GradFn") -> str | None:
    """Infer an intervening grad_fn_handle module from reverse autograd edges.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    grad_fn_handle:
        Intervening GradFn to place.

    Returns
    -------
    str | None
        Inherited module key, if a paired grad_fn_handle is reachable.
    """

    reverse_edges: dict[int, list[int]] = defaultdict(list)
    for candidate in trace.grad_fns:
        for next_grad_fn_id in candidate.next_grad_fn_ids:
            reverse_edges[next_grad_fn_id].append(candidate.grad_fn_object_id)
    return _infer_intervening_module_bfs(
        trace,
        reverse_edges.get(grad_fn_handle.grad_fn_object_id, []),
        reverse=True,
    )


def _infer_intervening_module_bfs(
    trace: "Trace",
    start_ids: Iterable[int],
    *,
    reverse: bool,
) -> str | None:
    """Find the nearest module-anchored grad_fn_handle by breadth-first search.

    Parameters
    ----------
    trace:
        Trace containing grad_fn_handle metadata.
    start_ids:
        Initial grad_fn_handle ids to inspect.
    reverse:
        Whether traversal uses reverse edges.

    Returns
    -------
    str | None
        Module key for the nearest paired grad_fn_handle, if found.
    """

    queue = list(start_ids)
    seen: set[int] = set()
    reverse_edges: dict[int, list[int]] = defaultdict(list)
    if reverse:
        for candidate in trace.grad_fns:
            for next_grad_fn_id in candidate.next_grad_fn_ids:
                reverse_edges[next_grad_fn_id].append(candidate.grad_fn_object_id)
    while queue:
        grad_fn_object_id = queue.pop(0)
        if grad_fn_object_id in seen or grad_fn_object_id not in trace.grad_fn_logs:
            continue
        seen.add(grad_fn_object_id)
        candidate = trace.grad_fn_logs[grad_fn_object_id]
        candidate_op = candidate.op
        if candidate_op is not None:
            module_key = _module_key_for_forward_op(candidate_op)
            if module_key is not None:
                return module_key
        if reverse:
            queue.extend(reverse_edges.get(grad_fn_object_id, []))
        else:
            queue.extend(candidate.next_grad_fn_ids)
    return None


def _compute_backward_node_lines(
    grad_fn_handle: "GradFn",
    call: Any | None = None,
    pass_filter: BackwardPassFilter = None,
) -> list[str]:
    """Build default label rows for a backward grad_fn_handle node.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to render.
    call:
        Optional GradFnCall when rendering an unrolled backward graph.
    pass_filter:
        Normalized backward-pass filter.

    Returns
    -------
    list[str]
        Plain-text rows for ``NodeSpec.lines``.
    """

    title = grad_fn_handle.label
    if call is not None:
        call_index = getattr(call, "call_index", getattr(call, "ordinal", 0))
        title = getattr(call, "call_label", f"{grad_fn_handle.label}:{call_index}")
    if not grad_fn_handle.has_op:
        title = f"[i] {title}"
    if grad_fn_handle.is_custom:
        title = f"{title} [custom]"

    lines = [title]
    order = getattr(grad_fn_handle, "order", None)
    if order is not None:
        lines.append(f"order {order}")
    if call is None:
        pass_indices = sorted(
            {
                int(pass_index)
                for pass_index in (
                    getattr(grad_fn_call, "backward_pass_index", None)
                    for grad_fn_call in grad_fn_handle.calls.values()
                )
                if pass_index is not None
            }
        )
        if pass_filter is not None:
            pass_indices = [pass_index for pass_index in pass_indices if pass_index in pass_filter]
        if pass_indices:
            lines.append(f"bwd {int_list_to_compact_str(pass_indices)}")
    else:
        pass_index = getattr(call, "backward_pass_index", None)
        if pass_index is not None:
            lines.append(f"bwd {pass_index}")
    if grad_fn_handle.op is not None:
        lines.append(f"@{grad_fn_handle.op.layer_label}")
    lines.append(f"grad {_format_backward_output_shape(grad_fn_handle)}")
    return lines


def _format_backward_output_shape(grad_fn_handle: "GradFn") -> str:
    """Return the first captured output-grad shape for a grad_fn_handle.

    Parameters
    ----------
    grad_fn_handle:
        GradFn to inspect.

    Returns
    -------
    str
        Compact shape string, or ``"N/A"`` when no tensor was captured
        (typical for intervening grad_fns that have no forward counterpart).
    """

    for grad_fn_pass in reversed(list(grad_fn_handle.calls.values())):
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
        Optional user predicate receiving aggregate Layer objects.

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
            skipped_labels.add(_render_node_label(node, vis_mode))

    edge_map: dict[str, list[RenderEdge]] = {}
    for node in visible_entries.values():
        if _render_node_label(node, vis_mode) in skipped_labels:
            continue
        edge_map[_render_node_label(node, vis_mode)] = _expand_edges_through_skipped(
            trace,
            node,
            visible_entries,
            skipped_labels,
            vis_mode,
        )
    return edge_map, skipped_labels


def _rank_layout_cost_inputs(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    edge_map: Mapping[str, Sequence[RenderEdge]],
    *,
    vis_mode: str,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
) -> tuple[set[str], list[tuple[str, str]]]:
    """Convert rendered edges into node and edge labels for rank-cost estimation.

    Parameters
    ----------
    trace:
        Owning Trace.
    entries_to_plot:
        Candidate nodes for the current visualization mode.
    edge_map:
        Skip-filtered render edge map.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Module nesting depth for collapsed modules.
    collapse_fn:
        Optional user collapse predicate.

    Returns
    -------
    tuple[set[str], list[tuple[str, str]]]
        Render-node labels and directed render edges.
    """

    nodes_by_render_label = {
        _render_node_label(node, vis_mode): node for node in entries_to_plot.values()
    }
    node_labels: set[str] = set()
    edges: list[tuple[str, str]] = []
    for source_label, render_edges in edge_map.items():
        source_node = nodes_by_render_label.get(source_label)
        if source_node is None:
            continue
        source_name = _rank_cost_node_name(
            trace,
            source_node,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            collapse_fn=collapse_fn,
        )
        node_labels.add(source_name)
        for render_edge in render_edges:
            target_name = _rank_cost_node_name(
                trace,
                render_edge.target,
                vis_mode=vis_mode,
                vis_call_depth=vis_call_depth,
                collapse_fn=collapse_fn,
            )
            node_labels.add(target_name)
            if source_name != target_name:
                edges.append((source_name, target_name))
    return node_labels, edges


def _rank_cost_node_name(
    trace: "Trace",
    node: GraphNode,
    *,
    vis_mode: str,
    vis_call_depth: int,
    collapse_fn: CollapseFn | None,
) -> str:
    """Return the render node name used for rank-cost estimation.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Render node.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Module nesting depth for collapsed modules.
    collapse_fn:
        Optional user collapse predicate.

    Returns
    -------
    str
        Name matching the final rendered node after collapse decisions.
    """

    collapse_address = _collapse_address_for_node(
        trace,
        node,
        collapse_fn=collapse_fn,
        max_module_depth=vis_call_depth,
    )
    if collapse_address is None:
        return node.layer_label.replace(":", "pass")
    parts = collapse_address.rsplit(":", 1)
    return "pass".join(parts) if vis_mode == "unrolled" else parts[0]


def _render_node_label(node: GraphNode, vis_mode: str) -> str:
    """Return the graph node label for the active visualization mode.

    Parameters
    ----------
    node:
        Render node.
    vis_mode:
        ``"unrolled"`` renders individual Ops, while ``"rolled"`` renders
        aggregate Layers.

    Returns
    -------
    str
        Stable label used as the DOT node identifier before Graphviz escaping.
    """
    if vis_mode == "unrolled" and isinstance(node, Op):
        return node.label
    return node.layer_label


def _resolve_focus_module(
    trace: "Trace",
    module: "Module | str",
) -> "Module":
    """Resolve and validate a module focus argument.

    Parameters
    ----------
    trace:
        Model log being rendered.
    module:
        Module instance or module address string.

    Returns
    -------
    Module
        Module to focus.

    Raises
    ------
    ValueError
        If the module cannot be found or belongs to a different Trace.
    """

    from ..data_classes.module import Module

    if isinstance(module, str):
        if module not in trace.modules:
            raise ValueError(f"Module address '{module}' was not found in this Trace.")
        resolved = trace.modules[module]
        if not isinstance(resolved, Module):
            raise ValueError(f"Module address '{module}' resolved to a module pass, not a Module.")
        return resolved
    if not isinstance(module, Module):
        raise ValueError("module must be a Module, module address string, or None.")
    if module._source_trace is not trace:
        raise ValueError("Module focus must belong to the Trace being rendered.")
    return module


def _build_module_focus_entries(
    trace: "Trace",
    entries_to_plot: Mapping[str, GraphNode],
    target_module: "Module",
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
    entries_by_layer_label = {node.layer_label: node for node in entries_to_plot.values()}
    input_boundaries: dict[str, BoundaryNode] = {}
    output_boundaries: dict[str, BoundaryNode] = {}

    for render_node in list(focused_entries.values()):
        node = cast(FocusNode, render_node)
        new_parents: list[str] = []
        for parent_label in node.parents:
            if parent_label in focus_labels:
                new_parents.append(parent_label)
                continue
            parent_node = entries_to_plot.get(parent_label) or entries_by_layer_label.get(
                parent_label
            )
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
            child_node = entries_to_plot.get(child_label) or entries_by_layer_label.get(child_label)
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
    target_module: "Module",
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


def _boundary_module_path(target_module: "Module", vis_mode: str) -> list[str]:
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

    visible_entries_by_layer = {node.layer_label: node for node in visible_entries.values()}
    by_target: dict[str, RenderEdge] = {}
    for child_label in parent_node.children:
        child_node = visible_entries.get(child_label) or visible_entries_by_layer.get(child_label)
        if child_node is None and vis_mode == "unrolled":
            child_node = trace.layer_dict_all_keys.get(child_label)
        if child_node is None:
            continue
        parent_label = _render_node_label(parent_node, vis_mode)
        child_render_label = _render_node_label(child_node, vis_mode)
        if child_render_label == parent_label and child_render_label not in skipped_labels:
            reached = [child_node]
        else:
            reached = _walk_skipped_successors(
                trace,
                child_node,
                visible_entries,
                skipped_labels,
                vis_mode,
                seen={parent_label},
            )
        for target_node in reached:
            first_child = child_node
            target_label = _render_node_label(target_node, vis_mode)
            existing = by_target.get(target_label)
            if existing is None:
                by_target[target_label] = RenderEdge(target_node, first_child)
            elif existing.metadata_child is not first_child:
                by_target[target_label] = RenderEdge(target_node, None)
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

    node_label = _render_node_label(node, vis_mode)
    if node_label in seen:
        return []
    seen.add(node_label)
    if node_label not in skipped_labels:
        return [node]
    reached: list[GraphNode] = []
    visible_entries_by_layer = {entry.layer_label: entry for entry in visible_entries.values()}
    for child_label in node.children:
        child_node = visible_entries.get(child_label) or visible_entries_by_layer.get(child_label)
        if child_node is None and vis_mode == "unrolled":
            child_node = trace.layer_dict_all_keys.get(child_label)
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
        return trace.layer_dict_main_keys.get(label, trace.layer_dict_all_keys[label])
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
    captured_forward_edges: list[CapturedForwardEdge] | None = None,
    rankdir: str = "BT",
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
        vis_mode=vis_mode,
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
        captured_forward_edges,
        rankdir,
    )


def _should_collapse_module(
    module_log: "Module",
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


def _module_has_single_rendered_op(module_log: "Module") -> bool:
    """Return whether ``module_log`` contains exactly one rendered op.

    Parameters
    ----------
    module_log:
        Module metadata to inspect.

    Returns
    -------
    bool
        True when the module contains one op and should keep op rendering.
    """

    return int(getattr(module_log, "num_layers", 0) or 0) == 1


def _single_op_module_should_keep_op_render(trace: "Trace", address: str) -> bool:
    """Return whether a one-op module should render as its op rather than collapse.

    Parameters
    ----------
    trace:
        Owning trace.
    address:
        Module address without call suffix.

    Returns
    -------
    bool
        True when the module has one op and no split call ranges to show.
    """

    module_log = cast("Module", trace.modules[address])
    return _module_has_single_rendered_op(module_log) and not _collapsed_module_rolling_suffix(
        trace, address
    )


def _collapse_address_for_node(
    trace: "Trace",
    node: GraphNode,
    *,
    vis_mode: str = "unrolled",
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
    vis_mode:
        ``"unrolled"`` or ``"rolled"`` visualization mode.
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
    # An atomic (single-op) module is already maximally collapsed: it renders as
    # its own rectangle and is never absorbed into a box3d collapse on its own
    # account, even at the top level or when reused across split call sites. Drop
    # its innermost (own) module address so only genuinely-collapsible ancestor
    # modules remain eligible to absorb it.
    if getattr(node, "is_atomic_module", False) and modules:
        modules = modules[:-1]
    if not modules:
        return None

    if collapse_fn is None:
        if max_module_depth == 0 or len(modules) < max_module_depth:
            return None
        address_w_pass = cast(str, modules[max_module_depth - 1])
        address = address_w_pass.rsplit(":", 1)[0]
        if vis_mode == "rolled" and _single_op_module_should_keep_op_render(trace, address):
            return None
        return address_w_pass

    for address_w_pass in modules:
        address = address_w_pass.rsplit(":", 1)[0]
        if vis_mode == "rolled" and _single_op_module_should_keep_op_render(trace, address):
            continue
        if _should_collapse_module(
            cast("Module", trace.modules[address]),
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
    - ``is_atomic_module``: the node represents the output of
      its innermost module, so its effective nesting depth is one less (it
      visually "belongs" to the parent scope).

    Args:
        node: The Op or Layer node to check.
        vis_call_depth: Maximum nesting depth before collapsing into a module box.
    """
    if trace is not None:
        return (
            _collapse_address_for_node(
                trace,
                node,
                vis_mode="unrolled",
                collapse_fn=collapse_fn,
                max_module_depth=vis_call_depth,
            )
            is not None
        )
    if vis_call_depth == 0:
        return False  # #94: depth 0 means show all layers, never collapse

    node_call_depth = len(node.modules)
    # Bottom-level submodule outputs are rendered at the parent nesting level,
    # not their own. Top-level atomic leaves have no module parent to bubble
    # up to, so they remain eligible for top-level collapse.
    if getattr(node, "is_atomic_module", False) and node_call_depth > 1:
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
        node: The Op or Layer node to render.
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
    node_args["name"] = _render_node_label(node, vis_mode).replace(":", "pass")
    hidden_buffer_addresses = _get_hidden_parent_buffer_addresses(self, node, show_buffer_layers)
    if hidden_buffer_addresses and not (node.is_input or node.is_output or node.is_buffer):
        node_args["peripheries"] = "2"
        hidden_tooltip = f"Hidden buffers: {', '.join(hidden_buffer_addresses)}"
        if "tooltip" in node_args:
            node_args["tooltip"] = f"{node_args['tooltip']}; {hidden_tooltip}"
        else:
            node_args["tooltip"] = hidden_tooltip
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
            s.node(_render_node_label(node, vis_mode).replace(":", "pass"))

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
        node: The Op or Layer node triggering the collapse.
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
    module_output_shape = getattr(module_output_layer, "shape", None)
    if module_output_shape is None:
        module_output_shape = getattr(module_output_layer, "out_shape", None)
    module_output_shape = module_output_shape or ()
    module_output_fsize = getattr(module_output_layer, "activation_memory", None)
    if module_output_fsize is None:
        module_output_fsize = "0 B"
    address, call_index = module_tuple
    ml = self.modules[address]
    module_type = ml.class_name  # type: ignore[union-attr]
    module_num_calls = ml.num_calls  # type: ignore[union-attr]
    module_nparams = ml.num_params  # type: ignore[union-attr]

    # In unrolled mode, each pass of a module is a separate collapsed node
    # (e.g., "encoder.layer.0pass1").  In rolled mode, all ops share one
    # node (e.g., "encoder.layer.0").
    if vis_mode == "unrolled":
        graph_node_label = "pass".join(module_tuple)
        mpl = self.modules[address_w_pass]
        module_num_tensors = mpl.num_layers
        module_has_input_ancestor = any(self[layer].has_input_ancestor for layer in mpl.ops)
    else:
        graph_node_label = module_tuple[0]
        module_num_tensors = ml.num_layers
        module_has_input_ancestor = any(self[layer].has_input_ancestor for layer in ml.layer_labels)  # type: ignore[union-attr]

    # Deduplicate: multiple layers in the same collapsed module will each
    # trigger this function, but the node should only be added once.
    if graph_node_label in collapsed_modules:
        return

    module_suffix = _collapsed_module_rolling_suffix(self, address)
    if module_num_calls == 1:
        node_title = f"<b>@{address}</b>"
    elif vis_mode == "unrolled" and (module_num_calls > 1):
        node_title = f"<b>@{address}:{call_index}</b>"
    elif module_suffix:
        node_title = f"<b>@{address}{module_suffix}</b>"
    else:
        node_title = f"<b>@{address} (x{module_num_calls})</b>"

    shape_str = format_shape(module_output_shape)

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
            f"{shape_str}, {format_memory(module_output_fsize)}",
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
    node_args["name"] = graph_node_label
    if spec.fillcolor is not None and ":" in spec.fillcolor:
        node_args["gradangle"] = "0"

    graphviz_graph.node(**node_args)
    collapsed_modules.add(graph_node_label)


def _compact_int_ranges(values: Sequence[int]) -> str:
    """Return sorted integers in compact range notation.

    Parameters
    ----------
    values:
        Integer values to format.

    Returns
    -------
    str
        Comma-separated values and ranges, for example ``"1,2-4"``.
    """

    if not values:
        return ""
    sorted_values = sorted(set(values))
    ranges: list[str] = []
    start = sorted_values[0]
    previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _module_address_and_call(module_call: str) -> tuple[str, int] | None:
    """Parse a pass-qualified module call label.

    Parameters
    ----------
    module_call:
        Module call label of the form ``"address:call_index"``.

    Returns
    -------
    tuple[str, int] | None
        Parsed address and call index, or ``None`` if the suffix is not an integer.
    """

    address, separator, call_index_text = module_call.rpartition(":")
    if not separator:
        return None
    try:
        return address, int(call_index_text)
    except ValueError:
        return None


def _node_for_label(trace: "Trace", label: str) -> GraphNode | None:
    """Return an op or layer-like graph node for ``label`` when available.

    Parameters
    ----------
    trace:
        Trace containing graph nodes.
    label:
        Layer or op label to resolve.

    Returns
    -------
    GraphNode | None
        Matching node, or ``None`` if the label is not present.
    """

    try:
        return cast(GraphNode, trace.layer_dict_all_keys[label])
    except KeyError:
        try:
            return cast(GraphNode, trace.ops[label])
        except KeyError:
            return None


def _same_layer_reachability(layer_log: "Layer") -> dict[int, set[int]]:
    """Compute transitive same-layer reachability among passes.

    Parameters
    ----------
    layer_log:
        Rolled layer whose same-layer pass reachability is needed.

    Returns
    -------
    dict[int, set[int]]
        Mapping from pass index to reachable same-layer pass indices.
    """

    trace = layer_log.source_trace
    same_layer_labels = {op.label for op in layer_log.ops.values()}
    label_to_pass = {op.label: pass_index for pass_index, op in layer_log.ops.items()}
    reachability: dict[int, set[int]] = {pass_index: set() for pass_index in layer_log.ops}

    for pass_index, op in layer_log.ops.items():
        seen: set[str] = set()
        stack = list(op.children)
        while stack:
            label = stack.pop()
            if label in seen:
                continue
            seen.add(label)
            if label in same_layer_labels:
                reachability[pass_index].add(label_to_pass[label])
            child = _node_for_label(trace, label)
            if child is not None:
                stack.extend(child.children)
    return reachability


def _common_module_call_indices(layer_log: "Layer") -> dict[str, list[int]]:
    """Return module call indices for module addresses present on every pass.

    Parameters
    ----------
    layer_log:
        Layer whose per-pass module stacks should be inspected.

    Returns
    -------
    dict[str, list[int]]
        Address to call indices in pass order, limited to common addresses.
    """

    per_op: list[dict[str, int]] = []
    for op in layer_log.ops.values():
        parsed: dict[str, int] = {}
        for module_call in op.modules:
            parsed_call = _module_address_and_call(module_call)
            if parsed_call is not None:
                address, call_index = parsed_call
                parsed[address] = call_index
        per_op.append(parsed)
    if not per_op:
        return {}
    common_addresses = set(per_op[0])
    for parsed in per_op[1:]:
        common_addresses &= set(parsed)
    return {address: [parsed[address] for parsed in per_op] for address in sorted(common_addresses)}


def _same_layer_dependency_components(layer_log: "Layer") -> tuple[tuple[int, ...], ...]:
    """Return weak components in the same-layer dependency graph.

    Parameters
    ----------
    layer_log:
        Layer whose passes should be partitioned.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Pass-index components, sorted by first pass.
    """

    reachability = _same_layer_reachability(layer_log)
    adjacency: dict[int, set[int]] = {pass_index: set() for pass_index in layer_log.ops}
    for source, targets in reachability.items():
        for target in targets:
            adjacency[source].add(target)
            adjacency[target].add(source)

    components: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for pass_index in sorted(layer_log.ops):
        if pass_index in seen:
            continue
        stack = [pass_index]
        component: set[int] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(sorted(adjacency[current] - seen, reverse=True))
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda values: values[0]))


def _call_groups_for_layer(layer_log: "Layer") -> tuple[tuple[int, ...], ...]:
    """Return grouped module calls for disjoint same-layer regions.

    Parameters
    ----------
    layer_log:
        Layer to inspect.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Module call-index groups. Empty when there is only one dependency component or
        no single common module address.
    """

    common_calls = _common_module_call_indices(layer_log)
    if len(common_calls) != 1:
        return ()
    pass_to_call_index = {
        pass_index: call_index
        for pass_index, call_index in zip(
            layer_log.ops,
            next(iter(common_calls.values())),
            strict=True,
        )
    }
    components = _same_layer_dependency_components(layer_log)
    if len(components) <= 1:
        return ()
    groups: list[tuple[int, ...]] = []
    for component in components:
        groups.append(tuple(pass_to_call_index[pass_index] for pass_index in component))
    return tuple(groups)


def _format_call_groups(call_groups: Sequence[Sequence[int]]) -> str:
    """Format grouped module call partitions.

    Parameters
    ----------
    call_groups:
        Call-index groups to format.

    Returns
    -------
    str
        Comma-separated compact ranges, preserving group boundaries.
    """

    return ",".join(_compact_int_ranges(group) for group in call_groups)


def _atomic_module_split_range(trace: "Trace", layer_log: GraphNode, address: str) -> str:
    """Return the call-range an atomic module rectangle should mark, or ``""``.

    An atomic (single-op) module renders as a rectangle per call site. When the
    module is reused across split sites the rectangle's ``@module`` marking carries
    the call range that distinguishes it, e.g. ``@shared:1-2,3-5`` for a single
    rolled node spanning two loops, or ``@relu:1`` / ``@relu:2-4`` for two separate
    rectangles. A module used at a single contiguous site needs no range (the
    title's ``(xN)`` count suffices), so this returns ``""``.

    Parameters
    ----------
    trace:
        Owning trace.
    layer_log:
        Atomic module layer being rendered.
    address:
        The atomic module's address.

    Returns
    -------
    str
        Compact call range (no leading colon), or ``""`` when not split.
    """

    if not isinstance(layer_log, Layer):
        return ""
    groups = _call_groups_for_layer(layer_log)
    if groups:
        return _format_call_groups(groups)
    sibling_atomic_layers = sum(
        1
        for other in trace.layer_logs.values()
        if isinstance(other, Layer)
        and getattr(other, "is_atomic_module", False)
        and other.modules
        and other.modules[-1].rsplit(":", 1)[0] == address
    )
    if sibling_atomic_layers > 1:
        calls = _common_module_call_indices(layer_log).get(address, [])
        if calls:
            return _compact_int_ranges(calls)
    return ""


def _buffer_versions_for_layer(layer_log: "Layer") -> tuple[int, ...]:
    """Return flat buffer versions represented by a layer.

    Parameters
    ----------
    layer_log:
        Layer to inspect.

    Returns
    -------
    tuple[int, ...]
        Sorted buffer pass indices, excluding ``None``.
    """

    versions = {
        int(op.buffer_pass)
        for op in layer_log.ops.values()
        if getattr(op, "buffer_pass", None) is not None
    }
    return tuple(sorted(versions))


def _rolling_annotation(layer_log: GraphNode, vis_mode: str) -> RollingAnnotation | None:
    """Build a rolled-view annotation for a layer node.

    Parameters
    ----------
    layer_log:
        Layer or Op being rendered.
    vis_mode:
        Active visualization mode.

    Returns
    -------
    RollingAnnotation | None
        Annotation for rolled multi-pass layers, otherwise ``None``.
    """

    layer_log = _unwrap_focus_node(layer_log)
    if vis_mode != "rolled" or not isinstance(layer_log, Layer) or layer_log.num_passes <= 1:
        return None

    call_groups = _call_groups_for_layer(layer_log)
    buffer_versions = _buffer_versions_for_layer(layer_log) if layer_log.is_buffer else ()
    if not call_groups and not buffer_versions:
        return None
    return RollingAnnotation(call_groups=call_groups, buffer_versions=buffer_versions)


def _collapsed_module_rolling_suffix(trace: "Trace", address: str) -> str:
    """Return a face suffix for a collapsed module's hidden call partitions.

    Parameters
    ----------
    trace:
        Trace containing the rendered module.
    address:
        Collapsed module address.

    Returns
    -------
    str
        Suffix beginning with ``":"`` or an empty string.
    """

    candidate_groups: tuple[tuple[int, ...], ...] = ()
    for layer_log in trace.layer_logs.values():
        if not isinstance(layer_log, Layer) or layer_log.num_passes <= 1:
            continue
        layer_addresses = {
            parsed[0]
            for op in layer_log.ops.values()
            for module_call in op.modules
            if (parsed := _module_address_and_call(module_call)) is not None
        }
        if address not in layer_addresses:
            continue
        groups = _call_groups_for_layer(layer_log)
        if len(groups) > len(candidate_groups):
            candidate_groups = groups
    if not candidate_groups:
        return ""
    return f":{_format_call_groups(candidate_groups)}"


def _format_rolling_suffix(annotation: RollingAnnotation | None) -> str:
    """Return the face suffix for a rolled annotation.

    Parameters
    ----------
    annotation:
        Annotation to render.

    Returns
    -------
    str
        Suffix beginning with a colon, or an empty string.
    """

    if annotation is None or not annotation.call_groups:
        return ""
    return f":{_format_call_groups(annotation.call_groups)}"


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

    if (node.is_atomic_module or only_non_buffer_layer) and (len(node.modules) > 0):
        if isinstance(source_node, Op):
            module_pass_exited = node.modules[-1]
            module, _ = module_pass_exited.split(":")
            if self.modules[module].num_calls == 1:  # type: ignore[union-attr]
                node_address = module
            else:
                node_address = module_pass_exited
        else:
            sample_module_pass = node.modules[-1]
            module = sample_module_pass.split(":")[0]
            split_range = _atomic_module_split_range(self, source_node, module)
            node_address = f"{module}:{split_range}" if split_range else module

        node_address = "<br/>@" + node_address
        node_shape = "box"
        node_color = "black"
    elif node.is_buffer:
        annotation = (
            _rolling_annotation(source_node, "rolled") if isinstance(source_node, Layer) else None
        )
        if annotation is not None and annotation.buffer_versions:
            address = f"{source_node.address}:{_compact_int_ranges(annotation.buffer_versions)}"
        elif self.buffer_num_calls[source_node.address] == 1:
            address = source_node.address
        else:
            address = f"{source_node.address}:{source_node.buffer_pass}"
        node_address = "<br/>@" + address
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
        node: The Op or Layer node to check.
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
        if isinstance(source_node, Op):
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
            trainable_flags = [pl.is_trainable for pl in param_logs]
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
        Rendered Op or Layer.
    default_spec:
        Default node spec.
    node_mode:
        Preset to apply before the optional user callback.
    node_spec_fn:
        Optional user callback. Unrolled nodes are represented to the callback
        by their parent Layer.

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


def _layer_log_for_node(trace: "Trace", node: GraphNode) -> "Layer":
    """Return the aggregate Layer for ``node``.

    Parameters
    ----------
    trace:
        Owning Trace.
    node:
        Op or Layer.

    Returns
    -------
    Layer
        Aggregate layer log for callbacks.
    """

    node = _unwrap_focus_node(node)
    if isinstance(node, BoundaryNode):
        raise ValueError("Synthetic boundary nodes do not have Layer metadata.")
    if isinstance(node, Layer):
        return node
    return trace.layer_logs[node.layer_label]


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
        Op or Layer to render.
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

    annotation = _rolling_annotation(layer_log, vis_mode)
    # An atomic module rectangle carries its split call range on the ``@module``
    # marking (e.g. ``@shared:1-2,3-5``); keep the title's count clean and
    # non-redundant in that case, and fall back to a plain ``(xN)`` count.
    atomic_marking_has_range = getattr(layer_log, "is_atomic_module", False) and (
        ":" in node_address
    )
    if (layer_log.num_passes > 1) and (vis_mode == "unrolled"):
        call_label = f":{layer_log.pass_index}"
    elif (layer_log.num_passes > 1) and (vis_mode == "rolled"):
        if atomic_marking_has_range:
            call_label = ""
        else:
            rolling_suffix = _format_rolling_suffix(annotation)
            if rolling_suffix:
                call_label = rolling_suffix
            else:
                call_label = f" (x{layer_log.num_passes})"
    else:
        call_label = ""

    if layer_log.layer_type in ["input", "output", "buffer"]:
        title = f"{layer_log.layer_type}_{layer_log.type_index}{call_label}"
    else:
        title = f"{layer_log.layer_type}_{layer_log.type_index}_{layer_log.step_index}{call_label}"

    lines: list[str] = []
    if layer_log.is_terminal_bool:
        lines.append(str(layer_log.bool_value).upper())
    lines.append(title)
    lines.append(f"{format_shape(layer_log.shape)}, {format_memory(layer_log.activation_memory)}")

    module_kwargs = format_module_kwargs(layer_log)
    if module_kwargs is not None:
        lines.append(module_kwargs)

    param_line = format_param_list(layer_log)
    if param_line is not None:
        lines.append(param_line)

    address_line = format_module_path(node_address)
    if address_line is not None:
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
        Op or Layer to render.
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
            rows.append(format_shape(layer_log.shape))
        elif field_name in {"memory", "bytes"}:
            rows.append(str(getattr(layer_log, "activation_memory", "")))
        elif field_name == "module":
            rows.append(format_module_path(node_address) or "@root")
        elif field_name == "params":
            param_line = format_param_list(layer_log)
            if param_line is not None:
                rows.append(param_line)
        elif field_name == "pass":
            rows.append(
                str(
                    getattr(layer_log, "call_index", 1)
                    if vis_mode == "unrolled"
                    else getattr(layer_log, "num_passes", 1)
                )
            )
        elif field_name == "flops":
            rows.append(str(getattr(layer_log, "flops_forward", 0) or 0))
        elif field_name == "time":
            rows.append(f"{float(getattr(layer_log, 'func_duration', 0.0) or 0.0) * 1000:.3g} ms")
        else:
            raise ValueError(f"Unsupported node label field: {field_name!r}.")
    return rows or compute_default_node_lines(layer_log, node_address, vis_mode)


def _format_shape_str(shape: tuple[Any, ...]) -> str:
    """Format a shape tuple in Python tuple notation."""

    return format_shape(shape)


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
    captured_forward_edges: list[CapturedForwardEdge] | None = None,
    rankdir: str = "BT",
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
        render_edges = edge_map.get(_render_node_label(parent_node, vis_mode), [])

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
                vis_mode=vis_mode,
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
            tail_name = _render_node_label(parent_node, vis_mode).replace(":", "pass")

        child_collapse_address = _collapse_address_for_node(
            self,
            child_node,
            vis_mode=vis_mode,
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
            head_name = _render_node_label(child_node, vis_mode).replace(":", "pass")

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
            if child_node.is_atomic_module:
                child_modules = child_modules[:-1]
            if parent_node.is_atomic_module:
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

        edge_is_self_loop = tail_name == head_name
        if edge_is_self_loop and not _is_rolled_loop_carried_self_edge(
            parent_node,
            child_node,
            vis_mode,
        ):
            continue

        dedupe_key = (tail_name, head_name)
        if dedupe_key in edges_used:
            continue
        edges_used.add(dedupe_key)

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

        if not edge_is_self_loop and not child_is_collapsed_module and not edge_has_boundary:
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
            if isinstance(metadata_base_for_pass, Layer) and isinstance(
                parent_base_for_pass, Layer
            ):
                # A recurrence back-edge may only merge its In/Out annotations
                # into a midpoint ``label`` if no conditional label was set above
                # and the argument labeler below (which adds headlabel/xlabel)
                # cannot fire for this edge; otherwise it keeps head/tail labels.
                arg_labeler_may_fire = not child_is_collapsed_module and bool(
                    _should_mark_arguments_on_edge(self, metadata_base_for_pass, show_buffer_layers)
                )
                _label_rolled_call_indexs(
                    metadata_base_for_pass,
                    parent_base_for_pass,
                    edge_dict,
                    is_self_loop=edge_is_self_loop,
                    rankdir=rankdir,
                    allow_midpoint_merge="label" not in edge_dict and not arg_labeler_may_fire,
                )

        # Label the arguments to the next node if multiple inputs
        if (
            not edge_is_self_loop
            and not child_is_collapsed_module
            and metadata_child is not None
            and not edge_has_boundary
        ):
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
        if edge_is_self_loop and _self_loop_is_single_op_module(self, parent_node):
            module = -1
        # Preserve the edge's LCA cluster key BEFORE the has_input_ancestor loops
        # below clobber the ``module`` loop variable (they reassign it to each
        # node's own module path). Without this, the captured forward edge would
        # record the wrong cluster key and sibling rank-groups would never resolve
        # to a common cluster (always falling back to top-level emission).
        edge_module_key: str | int = module
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

        if captured_forward_edges is not None:
            captured_forward_edges.append(
                CapturedForwardEdge(
                    source_label=parent_node.layer_label,
                    target_label=child_node.layer_label,
                    tail_name=tail_name,
                    head_name=head_name,
                    source_step=int(getattr(parent_node, "step_index", 0) or 0),
                    target_step=int(getattr(child_node, "step_index", 0) or 0),
                    source_node=parent_node,
                    target_node=child_node,
                    module_key=edge_module_key,
                )
            )

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


def _is_rolled_loop_carried_self_edge(
    parent_node: GraphNode,
    child_node: GraphNode,
    vis_mode: str,
) -> bool:
    """Return whether a same-endpoint edge represents rolled loop-carried flow.

    Parameters
    ----------
    parent_node:
        Rendered source node before endpoint collapsing.
    child_node:
        Rendered target node before endpoint collapsing.
    vis_mode:
        Active visualization mode.

    Returns
    -------
    bool
        True when the rolled edge advances from an earlier pass to a later pass.
    """

    if vis_mode != "rolled":
        return False
    parent_base = _base_node_for_metadata(parent_node)
    child_base = _base_node_for_metadata(child_node)
    if not isinstance(parent_base, Layer) or not isinstance(child_base, Layer):
        return False
    parent_passes = parent_base.child_ops_per_layer.get(child_base.layer_label, [])
    child_passes = child_base.parent_ops_per_layer.get(parent_base.layer_label, [])
    return any(
        child_pass > parent_pass
        for parent_pass, child_pass in zip(parent_passes, child_passes, strict=False)
    )


def _self_loop_is_single_op_module(trace: "Trace", node: GraphNode) -> bool:
    """Return whether a self-loop belongs to an atomic one-op module.

    Parameters
    ----------
    trace:
        Owning trace.
    node:
        Rendered node carrying the self-loop.

    Returns
    -------
    bool
        True when the node's innermost module contains exactly one rendered op.
    """

    modules = list(getattr(node, "modules", []) or [])
    if not modules:
        return False
    address = str(modules[-1]).rsplit(":", 1)[0]
    if address not in trace.modules:
        return False
    return _module_has_single_rendered_op(cast("Module", trace.modules[address]))


def _compute_edge_label(
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
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
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
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
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
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
        parent_no_pass = parent_node.layer_label
        child_no_pass = child_node.layer_label
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
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
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
        edge_key = (parent_node.layer_label, child_node.layer_label)
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
    parent_node: Union["Op", "Layer"],
    child_node: Union["Op", "Layer"],
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
    child_node: Union["Op", "Layer"],
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

    if isinstance(child_node, Op):
        return _should_mark_arguments_on_unrolled_edge(self, child_node, show_buffer_layers)
    elif isinstance(child_node, Layer):
        return _should_mark_arguments_on_rolled_edge(self, child_node, show_buffer_layers)


def _should_mark_arguments_on_unrolled_edge(
    self: "Trace",
    child_node: "Op",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument labels should be shown on an unrolled graph edge.

    Args:
        child_node: The child Op node whose incoming edge is being considered.
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
    child_node: "Layer",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
) -> bool:
    """Returns True if argument labels should be shown on a rolled graph edge.

    Args:
        child_node: The child Layer node whose incoming edge is being considered.
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


_EDGE_LABEL_FONT_SIZE = 8
_EDGE_LABEL_PAD = 4  # points of transparent margin on every side of a head/tail label
_SELF_LOOP_LABEL_HGAP = 8  # points of blank spacer left/right of a self-loop label


def _html_edge_label(text: str) -> str:
    """Return an HTML head/tail edge label with an even transparent margin.

    graphviz does not allocate layout space for head/tail labels, so plain text
    touches the node, arrowhead, or edge it sits beside (an arrowhead can even
    clip the first letter). Wrapping the text in a borderless one-cell table with
    ``CELLPADDING`` gives it a small, even margin on every side -- enough to read
    as belonging to that endpoint without crowding it, and far less than the full
    blank text line a ``\\n`` pad would add.
    """

    return (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="{_EDGE_LABEL_PAD}">'
        f'<TR><TD><FONT POINT-SIZE="{_EDGE_LABEL_FONT_SIZE}">{text}</FONT></TD></TR></TABLE>>'
    )


def _html_combined_recurrence_label(top: str, bottom: Optional[str] = None) -> str:
    """Return an HTML combined recurrence label with left/right spacers.

    Used for a node's self-loop arc, a merged recurrence back-edge, and merged
    buffer-edge annotations; the combined ``In``/``Out`` label sits beside the
    edge, and wide side spacer cells keep it clear of the node border and the
    edge line.  When ``bottom`` is ``None`` the label is a single line (a
    buffer edge may carry only one of ``In``/``Out``).
    """

    text = top if bottom is None else f"{top}<BR/>{bottom}"
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD WIDTH="{_SELF_LOOP_LABEL_HGAP}"></TD>'
        f'<TD><FONT POINT-SIZE="{_EDGE_LABEL_FONT_SIZE}">{text}</FONT></TD>'
        f'<TD WIDTH="{_SELF_LOOP_LABEL_HGAP}"></TD></TR></TABLE>>'
    )


def _is_rolled_recurrence_back_edge(child_node: "Layer", parent_node: "Layer") -> bool:
    """Returns True if a rolled edge runs backwards in execution order.

    A recurrence back-edge feeds a layer whose first op executes EARLIER than
    the source layer's first op (e.g. ``tanh_1_2 -> linear_1_1`` in an RNN
    cell).  ``Layer.step_index`` records exactly that first-op position in the
    global execution order, so a structural comparison suffices -- no label
    parsing.  Input/buffer layers (``step_index`` 0) and output layers
    (``step_index`` is the final op count, never smaller than a real parent's)
    are excluded by requiring both indices to be positive, so input/output and
    buffer-write edges are never flagged.

    Args:
        child_node: The destination Layer of the edge.
        parent_node: The source Layer of the edge.
    """
    child_step = child_node.step_index
    parent_step = parent_node.step_index
    if not isinstance(child_step, int) or not isinstance(parent_step, int):
        return False
    return 0 < child_step < parent_step


def _layer_has_rolled_self_loop(node: "Layer") -> bool:
    """Returns True if a Layer feeds itself across passes (rolled self-loop edge).

    Args:
        node: The Layer to check.
    """
    return node.layer_label in node.child_ops_per_layer


def _is_rolled_congested_recurrence_forward_edge(child_node: "Layer", parent_node: "Layer") -> bool:
    """Returns True for the forward partner of a recurrence back-edge whose band
    is congested by a self-loop.

    The forward edge and its merged back-edge run anti-parallel; when either
    endpoint ALSO carries a recurrence self-loop, three-plus near-parallel
    curves crowd that band and the forward edge's head/tail labels collide with
    the back-edge's spline and arrowhead.  Merging the forward edge's
    annotations into a midpoint ``label`` (which Graphviz reserves dummy-node
    space for) pushes the curves apart, the same cure as the back-edge merge.
    Plain two-node recurrences (no self-loop) keep head/tail labels: their band
    holds only two curves and lays out cleanly.

    Args:
        child_node: The destination Layer of the edge.
        parent_node: The source Layer of the edge.
    """
    return (
        _is_rolled_recurrence_back_edge(parent_node, child_node)
        and parent_node.layer_label in child_node.child_ops_per_layer
        and (_layer_has_rolled_self_loop(parent_node) or _layer_has_rolled_self_loop(child_node))
    )


def _is_rolled_buffer_edge(child_node: "Layer", parent_node: "Layer") -> bool:
    """Returns True if either endpoint of a rolled edge is a buffer layer.

    Buffer layers carry the structural ``is_buffer`` flag (glossary:
    ``is_buffer_source``); their ``step_index`` is 0, so
    ``_is_rolled_recurrence_back_edge`` deliberately never matches them.  A
    buffer's read edge and write edge run anti-parallel in a narrow band (the
    same geometry the back-edge midpoint merge solves), so buffer-incident
    edges get the same midpoint treatment for their ``In``/``Out`` annotations.

    Args:
        child_node: The destination Layer of the edge.
        parent_node: The source Layer of the edge.
    """
    return bool(child_node.is_buffer) or bool(parent_node.is_buffer)


# Placement attrs (labeldistance, labelangle) for head/tail pass labels on
# structurally at-risk rolled edges.  Setting EITHER attr switches graphviz
# from its default endpoint-label placement to ``place_portlabel``, which
# positions the label relative to the edge's END TANGENT -- so the label
# follows an oblique or bowed spline instead of clipping it.  The default
# placement is kept for ordinary (straight) edges: explicitly setting the
# documented "defaults" (1.0, -25) is NOT a no-op and measurably worsens them.
# Values chosen by an offline audit-scored sweep over the 16-model rolled
# inspection set (dot 7.0.5, 55 configs, exact per-label geometry audit):
# each clears the listed failure class to zero hard violations while keeping
# labels within 9pt of their endpoint node.
#
# Heads of >=3-op cycle body edges: the cycle's merged back-edge midpoint
# label bows the whole forward chain; labels otherwise clip their own
# spline/arrowhead.
_ROLLED_CYCLE_HEAD_LABEL_PLACEMENT = ("1.6", "-90")
# Heads of adjacent forward edges into a self-loop-bearing layer: the
# self-loop arc invades the default head-label spot.
_ROLLED_SELF_LOOP_HEAD_LABEL_PLACEMENT = ("1.6", "-65")
# Tails of >=3-op cycle body edges, and either label of a multi-step edge
# touching a self-loop layer (long bowed skip edges, e.g. input -> loop op).
_ROLLED_OBLIQUE_LABEL_PLACEMENT = ("2.0", "-45")


def _rolled_pass_label_placement(
    child_node: "Layer",
    parent_node: "Layer",
    out_label: Optional[str],
    in_label: Optional[str],
) -> Optional[Tuple[str, str]]:
    """Pick explicit (labeldistance, labelangle) for an at-risk head/tail label.

    Returns None (keep graphviz's default endpoint placement) for edges that
    lay out straight; returns tuned tangent-relative placement attrs for the
    structural classes whose splines bow near the labeled endpoint (see the
    ``_ROLLED_*_LABEL_PLACEMENT`` constants).  Only single-annotation edges are
    conditioned: the attrs are per-edge and move BOTH endpoint labels, and
    dual-annotation edges lay out straight (their recurrence partners merge to
    midpoint labels instead).

    Args:
        child_node: The destination Layer of the edge.
        parent_node: The source Layer of the edge.
        out_label: The ``Out ...`` annotation, if the edge carries one.
        in_label: The ``In ...`` annotation, if the edge carries one.
    """
    has_head = in_label is not None
    has_tail = out_label is not None
    if has_head == has_tail:
        return None
    p_step = parent_node.step_index
    c_step = child_node.step_index
    if not isinstance(p_step, int) or not isinstance(c_step, int):
        return None
    span = c_step - p_step
    forward = 0 < p_step < c_step
    # Body edge of a >=3-op cycle: both endpoints recurrent, no direct
    # reverse edge (a direct reverse edge means a two-op cycle, which lays
    # out straight and is handled by the back-edge midpoint merge).
    if (
        forward
        and parent_node.num_passes > 1
        and child_node.num_passes > 1
        and parent_node.layer_label not in child_node.child_ops_per_layer
    ):
        return _ROLLED_CYCLE_HEAD_LABEL_PLACEMENT if has_head else _ROLLED_OBLIQUE_LABEL_PLACEMENT
    # Multi-step skip edge attached to a self-loop layer (bowed long curve);
    # edges into output layers run straight to the top rank and stay default.
    if (
        0 <= p_step < c_step
        and 2 <= span <= 4
        and (_layer_has_rolled_self_loop(parent_node) or _layer_has_rolled_self_loop(child_node))
        and child_node.layer_type != "output"
    ):
        return _ROLLED_OBLIQUE_LABEL_PLACEMENT
    # Adjacent forward edge into a self-loop layer: the self-loop arc sits
    # where the default head label would go.
    if has_head and forward and span == 1 and _layer_has_rolled_self_loop(child_node):
        return _ROLLED_SELF_LOOP_HEAD_LABEL_PLACEMENT
    return None


def _label_rolled_call_indexs(
    child_node: "Layer",
    parent_node: "Layer",
    edge_dict: Dict[str, Any],
    *,
    is_self_loop: bool = False,
    rankdir: str = "BT",
    allow_midpoint_merge: bool = False,
) -> None:
    """Add pass-number annotations to edges in rolled mode.

    In rolled mode, a single edge may represent connections from different
    ops.  When edges vary across ops (``edges_vary_across_ops``),
    tail and head labels show which ops the edge applies to, e.g.,
    ``"Out 1,3"`` / ``"In 2,4"``.  Uses ``int_list_to_compact_str`` for
    concise range notation (e.g., ``"1-3"`` instead of ``"1,2,3"``).

    Labels are emitted as HTML-like tables with small spacer cells so the text
    keeps an even gap from the node and the arrowhead without crowding it (a plain
    head/tail label is not allocated layout space and would touch the node).

    Self-loops are special-cased: their head/tail labels crowd against the node
    and (unlike forward edges) a recurrence self-edge never carries argument or
    conditional midpoint labels.  So the ``In``/``Out`` annotations are merged
    into a single midpoint ``label``, which Graphviz reserves layout space for
    (it is modeled as a dummy node), eliminating the overlap.  The ``In`` line is
    placed above ``Out`` for bottom-up graphs (flow points up) and flipped for
    top-down ones.

    Recurrence back-edges between distinct nodes get the same merge when the
    caller marks it safe (``allow_midpoint_merge``): a back-edge runs
    anti-parallel to the forward edge between the same two nodes, so four
    head/tail labels would fight for the narrow gap between the two near-parallel
    edges and collide with the arrowheads.  Merging into one midpoint ``label``
    both clears that gap and -- because Graphviz reserves a dummy-node spot for
    midpoint labels -- pushes the anti-parallel edges apart.  Forward edges keep
    head/tail labels: they may carry argument or conditional midpoint labels
    that a combined label would collide with, which is also why the caller must
    vouch (no existing ``label``, argument labeler cannot fire) before a
    back-edge merges.

    Buffer-incident edges (either endpoint has ``is_buffer``) merge under the
    same caller gate, including when only ONE of ``In``/``Out`` is present
    (rendered as a one-line midpoint label).  Buffer layers have
    ``step_index`` 0, so the back-edge check never matches them, yet a
    buffer's read and write edges run anti-parallel in the same narrow band --
    and the read edge curves enough that even its own head label collides with
    its own spline.

    Head/tail labels that stay (no merge) may get explicit per-edge
    ``labeldistance``/``labelangle`` attrs when the edge belongs to a
    structural class whose spline bows near the labeled endpoint; see
    ``_rolled_pass_label_placement``.

    Args:
        child_node: The child Layer node.
        parent_node: The parent Layer node.
        edge_dict: Mutable dict of edge attributes; taillabel/headlabel/label may be added.
        is_self_loop: Whether this edge is a node's recurrence self-loop.
        rankdir: Graphviz rank direction, used to order a combined label's lines.
        allow_midpoint_merge: Whether a non-self-loop edge may safely take a
            midpoint ``label`` (no conditional label present, no argument label
            coming); only such recurrence back-edges merge their annotations.
    """
    parent_call_indexs = parent_node.child_ops_per_layer[child_node.layer_label]
    child_call_indexs = child_node.parent_ops_per_layer[parent_node.layer_label]
    out_label = (
        f"Out {int_list_to_compact_str(parent_call_indexs)}"
        if parent_node.edges_vary_across_ops
        else None
    )
    in_label = (
        f"In {int_list_to_compact_str(child_call_indexs)}"
        if child_node.edges_vary_across_ops
        else None
    )

    is_buffer_edge = not is_self_loop and _is_rolled_buffer_edge(child_node, parent_node)
    merge_into_midpoint = is_self_loop or (
        allow_midpoint_merge
        and (
            is_buffer_edge
            or _is_rolled_recurrence_back_edge(child_node, parent_node)
            or _is_rolled_congested_recurrence_forward_edge(child_node, parent_node)
        )
    )
    if merge_into_midpoint and out_label is not None and in_label is not None:
        top, bottom = (out_label, in_label) if rankdir == "TB" else (in_label, out_label)
        edge_dict["label"] = _html_combined_recurrence_label(top, bottom)
        return
    single_label = out_label if out_label is not None else in_label
    if is_buffer_edge and allow_midpoint_merge and single_label is not None:
        # A buffer edge with a single annotation still merges: its read and
        # write edges run anti-parallel in a narrow band, where a head/tail
        # label collides with the opposing edge's spline and arrowhead.
        edge_dict["label"] = _html_combined_recurrence_label(single_label)
        return

    if out_label is not None:
        edge_dict["taillabel"] = _html_edge_label(out_label)
    if in_label is not None:
        edge_dict["headlabel"] = _html_edge_label(in_label)
    if (out_label is not None or in_label is not None) and allow_midpoint_merge:
        # ``allow_midpoint_merge`` doubles as the caller's promise that the
        # argument labeler cannot add a headlabel later, so the per-edge
        # placement attrs below can only ever move OUR pass labels.
        placement = _rolled_pass_label_placement(child_node, parent_node, out_label, in_label)
        if placement is not None:
            edge_dict["labeldistance"], edge_dict["labelangle"] = placement


def _get_lowest_module_for_two_render_nodes(
    node1: GraphNode,
    node2: GraphNode,
    both_nodes_collapsed_modules: bool,
    vis_call_depth: int,
) -> Union[str, int]:
    """Find the deepest module subgraph for render nodes including boundaries."""

    return _get_lowest_module_for_two_nodes(
        cast(Union["Op", "Layer"], node1),
        cast(Union["Op", "Layer"], node2),
        both_nodes_collapsed_modules,
        vis_call_depth,
    )


def _get_lowest_module_for_two_nodes(
    node1: Union["Op", "Layer"],
    node2: Union["Op", "Layer"],
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
    - ``is_atomic_module`` nodes are adjusted to their parent
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

    if isinstance(node1, Layer) or isinstance(node2, Layer):
        node1_modules = [module.split(":")[0] for module in node1_modules]
        node2_modules = [module.split(":")[0] for module in node2_modules]

    if node1.is_atomic_module:
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
        if node1.is_atomic_module and (len(node1_modules) == 1):
            return -1
        elif node1.is_atomic_module and (len(node1_modules) > 1):
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
        parent_layer: The parent Op or Layer (grad destination).
        child_layer: The child Op or Layer (grad source).
        edge_style: ``'solid'`` or ``'dashed'`` (matches the forward edge style).
        module: Module cluster name, or -1 for top-level.
        module_edge_dict: Dict mapping each module cluster to its edges.
        graphviz_graph: The graphviz Digraph object.
        overrides: Graphviz attribute overrides for grad edges.
    """
    if _node_has_grad(parent_layer) and _node_has_grad(child_layer):
        grad_passes = _shared_gradient_passes(parent_layer, child_layer)
        edge_dict = {
            "tail_name": _grad_node_name(child_layer),
            "head_name": _grad_node_name(parent_layer),
            "color": GRADIENT_ARROW_COLOR,
            "fontcolor": GRADIENT_ARROW_COLOR,
            "style": edge_style,
            "arrowsize": ".7",
            "labelfontsize": "8",
        }
        if (
            grad_passes
            and self.num_backward_passes > 1
            and grad_passes != set(range(1, self.num_backward_passes + 1))
        ):
            edge_dict["label"] = f"bwd {int_list_to_compact_str(sorted(grad_passes))}"
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
        ``Op`` or rolled ``Layer``.

    Returns
    -------
    bool
        True if the node has at least one saved grad tensor.
    """

    ops = getattr(layer, "ops", None)
    if ops is not None and hasattr(ops, "values"):
        return any(bool(getattr(pass_log, "has_grad", False)) for pass_log in ops.values())
    return bool(getattr(layer, "has_grad", False))


def _node_gradient_passes(layer: Any) -> set[int]:
    """Return backward pass numbers with saved gradients for a rendered node.

    Parameters
    ----------
    layer:
        ``Op`` or rolled ``Layer``.

    Returns
    -------
    set[int]
        One-based backward pass numbers.
    """

    ops = getattr(layer, "ops", None)
    if ops is not None and hasattr(ops, "values"):
        pass_indices: set[int] = set()
        for pass_log in ops.values():
            pass_indices.update(_node_gradient_passes(pass_log))
        return pass_indices
    grads = getattr(layer, "grads", None)
    if grads is None:
        return set()
    return {
        int(record.backward_pass_index)
        for record in grads
        if getattr(record, "backward_pass_index", None) is not None and record.is_saved
    }


def _shared_gradient_passes(parent_layer: GraphNode, child_layer: GraphNode) -> set[int]:
    """Return backward pass numbers shared by both gradient-edge endpoints.

    Parameters
    ----------
    parent_layer:
        Forward parent node.
    child_layer:
        Forward child node.

    Returns
    -------
    set[int]
        One-based backward pass numbers shared by both endpoints.
    """

    return _node_gradient_passes(parent_layer) & _node_gradient_passes(child_layer)


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


def _should_order_siblings(
    *,
    order_siblings: bool,
    engine: str,
    vis_mode: str,
    num_nodes: int,
    module: "Module | str | None",
    vis_intervention_mode: VisInterventionModeLiteral,
    collapse_fn: CollapseFn | None,
    vis_call_depth: int,
) -> bool:
    """Return whether sibling ordering is in scope for this render."""

    return (
        order_siblings
        and engine == "dot"
        and vis_mode == "unrolled"
        and num_nodes <= SIBLING_ORDER_NODE_CAP
        and module is None
        and vis_intervention_mode == "node_mark"
        and collapse_fn is None
        and vis_call_depth >= 1000
    )


def _build_sibling_order_chains(
    captured_edges: list[CapturedForwardEdge],
) -> tuple[SiblingOrderChain, ...]:
    """Build candidate sibling chains from captured rendered edges."""

    if not _has_rendered_fanout(captured_edges):
        return ()

    rendered_parents: dict[str, set[str]] = defaultdict(set)
    by_source: dict[tuple[str, str], list[CapturedForwardEdge]] = defaultdict(list)
    for edge in captured_edges:
        rendered_parents[edge.head_name].add(edge.tail_name)
        by_source[(edge.source_label, edge.tail_name)].append(edge)

    chains: list[SiblingOrderChain] = []
    for (source_label, source_name), source_edges in sorted(by_source.items()):
        distinct_targets: dict[str, CapturedForwardEdge] = {}
        source_node = source_edges[0].source_node
        if _has_conditional_fanout(source_node):
            continue
        for edge in source_edges:
            distinct_targets.setdefault(edge.head_name, edge)
        if len(distinct_targets) < 2:
            continue
        kept_edges = [
            edge
            for target_name, edge in distinct_targets.items()
            if rendered_parents[target_name] == {source_name}
        ]
        if len(kept_edges) < 2:
            continue
        kept_edges.sort(key=lambda edge: (edge.target_step, edge.head_name))
        chains.append(
            SiblingOrderChain(
                source_label=source_label,
                source_name=source_name,
                targets=tuple(edge.head_name for edge in kept_edges),
                target_labels=tuple(edge.target_label for edge in kept_edges),
                lca_key=_sibling_chain_lca_key(kept_edges),
            )
        )
    return tuple(chains)


def _has_rendered_fanout(captured_edges: list[CapturedForwardEdge]) -> bool:
    """Return whether any rendered source has at least two distinct children."""

    children_by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge in captured_edges:
        key = (edge.source_label, edge.tail_name)
        children_by_source[key].add(edge.head_name)
        if len(children_by_source[key]) >= 2:
            return True
    return False


def _has_conditional_fanout(node: GraphNode) -> bool:
    """Return whether a fanout source has conditional branch-child metadata."""

    for field_name in (
        "conditional_entry_children",
        "conditional_then_children",
        "conditional_elif_children",
        "conditional_else_children",
        "conditional_arm_children",
    ):
        if getattr(node, field_name, None):
            return True
    return False


def _sibling_chain_lca_key(edges: list[CapturedForwardEdge]) -> str | int:
    """Return the rendered module key shared by all sibling edges."""

    if not edges:
        return -1
    first_key = edges[0].module_key
    if all(edge.module_key == first_key for edge in edges):
        return first_key
    return -1


def _queue_sibling_rank_group(
    module_edge_dict: Dict[str, Any],
    top_level_rank_groups: list[SiblingOrderChain],
    chain: SiblingOrderChain,
) -> None:
    """Queue a sibling rank group in the cluster dictionary."""

    if chain.lca_key == -1:
        top_level_rank_groups.append(chain)
    else:
        module_edge_dict[cast(str, chain.lca_key)]["rank_groups"].append(chain)


def _verify_and_apply_sibling_ordering(
    source: str,
    chains: tuple[SiblingOrderChain, ...],
    captured_edges: list[CapturedForwardEdge],
    rankdir: str,
) -> tuple[str, SiblingOrderDecision]:
    """Verify sibling rank chains and return final DOT source."""

    baseline_source = _strip_sibling_rank_groups(source)
    baseline = _layout_dot_plain(baseline_source, rankdir, captured_edges)
    injected = _layout_dot_plain(source, rankdir, captured_edges)
    _assert_sibling_backstops(baseline, injected, chains, captured_edges)

    ratios = {
        _sibling_chain_key(chain): _sibling_chain_stretch_ratio(
            chain, captured_edges, baseline, injected
        )
        for chain in chains
    }
    survivors = tuple(
        chain for chain in chains if ratios[_sibling_chain_key(chain)] <= SIBLING_ORDER_STRETCH_CAP
    )
    current_source = (
        source if survivors == chains else _inject_sibling_rank_groups(baseline_source, survivors)
    )
    current_layout = (
        injected
        if survivors == chains
        else _layout_dot_plain(current_source, rankdir, captured_edges)
    )

    for _ in range(2):
        bad_chains = tuple(
            chain
            for chain in survivors
            if _sibling_chain_stretch_ratio(chain, captured_edges, baseline, current_layout)
            > SIBLING_ORDER_STRETCH_CAP
        )
        if not bad_chains:
            return current_source, _sibling_order_decision(chains, survivors, ratios)
        survivors = tuple(chain for chain in survivors if chain not in bad_chains)
        current_source = _inject_sibling_rank_groups(baseline_source, survivors)
        current_layout = _layout_dot_plain(current_source, rankdir, captured_edges)
    return current_source, _sibling_order_decision(chains, survivors, ratios)


def _sibling_chain_key(chain: SiblingOrderChain) -> tuple[str, tuple[str, ...]]:
    """Return a stable key for decision reporting."""

    return chain.source_name, chain.targets


def _sibling_order_decision(
    chains: tuple[SiblingOrderChain, ...],
    survivors: tuple[SiblingOrderChain, ...],
    ratios: dict[tuple[str, tuple[str, ...]], float],
) -> SiblingOrderDecision:
    """Build a sibling-order decision record."""

    return SiblingOrderDecision(
        candidate_count=len(chains),
        survivor_count=len(survivors),
        ratios=ratios,
        surviving_keys=tuple(_sibling_chain_key(chain) for chain in survivors),
    )


def _layout_dot_plain(
    source: str,
    rankdir: str,
    captured_edges: list[CapturedForwardEdge],
) -> PlainLayout:
    """Run ``dot -Tplain`` and parse coordinates and real-edge spans."""

    real_edges = {(edge.tail_name, edge.head_name) for edge in captured_edges}
    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as source_file:
        source_file.write(source)
        source_path = source_file.name
    try:
        proc = subprocess.run(
            ["dot", "-Tplain", source_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        os.remove(source_path)

    nodes: dict[str, tuple[float, float]] = {}
    pending_edges: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "node" and len(parts) >= 4:
            nodes[parts[1]] = (float(parts[2]), float(parts[3]))
        elif parts[0] == "edge" and len(parts) >= 4:
            edge_key = (parts[1], parts[2])
            if edge_key in real_edges:
                pending_edges.append(edge_key)

    edge_spans: dict[tuple[str, str], float] = {}
    for edge_key in pending_edges:
        if edge_key[0] in nodes and edge_key[1] in nodes:
            edge_spans[edge_key] = _flow_span(nodes[edge_key[0]], nodes[edge_key[1]], rankdir)
    return PlainLayout(nodes=nodes, edge_spans=edge_spans)


def _flow_span(tail_xy: tuple[float, float], head_xy: tuple[float, float], rankdir: str) -> float:
    """Return flow-axis span between two node coordinates."""

    if rankdir in {"LR", "RL"}:
        return abs(tail_xy[0] - head_xy[0])
    return abs(tail_xy[1] - head_xy[1])


def _sibling_chain_stretch_ratio(
    chain: SiblingOrderChain,
    captured_edges: list[CapturedForwardEdge],
    baseline: PlainLayout,
    candidate: PlainLayout,
) -> float:
    """Return the local incident-edge stretch ratio for ``chain``."""

    local_nodes = {chain.source_name, *chain.targets}
    ratios: list[float] = []
    for edge in captured_edges:
        edge_key = (edge.tail_name, edge.head_name)
        if edge.tail_name not in local_nodes and edge.head_name not in local_nodes:
            continue
        if edge_key not in baseline.edge_spans or edge_key not in candidate.edge_spans:
            continue
        ratios.append(
            candidate.edge_spans[edge_key]
            / max(SIBLING_ORDER_EPSILON, baseline.edge_spans[edge_key])
        )
    return max(ratios, default=1.0)


def _assert_sibling_backstops(
    baseline: PlainLayout,
    injected: PlainLayout,
    chains: tuple[SiblingOrderChain, ...],
    captured_edges: list[CapturedForwardEdge],
) -> None:
    """Assert sibling-ordering structural backstops."""

    real_edges = {(edge.tail_name, edge.head_name) for edge in captured_edges}
    assert len(baseline.nodes) == len(injected.nodes)
    for chain in chains:
        for target in chain.targets:
            assert target in baseline.nodes
        for left, right in zip(chain.targets, chain.targets[1:]):
            assert (left, right) not in real_edges
            assert (right, left) not in real_edges


def _strip_sibling_rank_groups(source: str) -> str:
    """Remove TorchLens sibling-order rank-group blocks from DOT source."""

    lines = source.splitlines()
    stripped: list[str] = []
    skipping = False
    for line in lines:
        if "tl:sibling-order:start" in line:
            skipping = True
            continue
        if "tl:sibling-order:end" in line:
            skipping = False
            continue
        if not skipping:
            stripped.append(line)
    return "\n".join(stripped) + "\n"


def _inject_sibling_rank_groups(source: str, chains: tuple[SiblingOrderChain, ...]) -> str:
    """Inject surviving sibling rank groups into baseline DOT source."""

    result = source
    top_level_groups = [chain for chain in chains if chain.lca_key == -1]
    if top_level_groups:
        result = _insert_before_final_brace(result, _rank_group_lines(top_level_groups, ""))

    by_cluster: dict[str, list[SiblingOrderChain]] = defaultdict(list)
    for chain in chains:
        if chain.lca_key != -1:
            by_cluster[cast(str, chain.lca_key)].append(chain)
    fallback_chains: list[SiblingOrderChain] = []
    for cluster_key, cluster_chains in by_cluster.items():
        cluster_name = f"cluster_{cluster_key.replace(':', '_pass')}"
        result, did_emit = _insert_into_cluster(
            result,
            cluster_name,
            _rank_group_lines(cluster_chains, "    "),
        )
        if not did_emit:
            # Cluster-name string surgery missed (e.g. an unexpected rendered key).
            # Fall back to top-level emission, which is verify-safe (a distorting
            # chain is dropped by the per-chain stretch check), rather than crash
            # the render with a bare assert.
            fallback_chains.extend(cluster_chains)
    if fallback_chains:
        result = _insert_before_final_brace(result, _rank_group_lines(fallback_chains, ""))
    return result


def _rank_group_lines(chains: Sequence[SiblingOrderChain], indent: str) -> str:
    """Return DOT lines for sibling rank groups."""

    lines: list[str] = []
    for chain in chains:
        lines.append(f"{indent}// tl:sibling-order:start")
        lines.append(f"{indent}{{")
        lines.append(f"{indent}    rank=same")
        for target in chain.targets:
            lines.append(f"{indent}    {quote_dot_id(target)}")
        for left, right in zip(chain.targets, chain.targets[1:]):
            lines.append(
                f"{indent}    {quote_dot_id(left)} -> {quote_dot_id(right)} "
                '[style=invis weight=100 comment="tl:sibling-order"]'
            )
        lines.append(f"{indent}}}")
        lines.append(f"{indent}// tl:sibling-order:end")
    return "\n".join(lines) + "\n"


def _emit_sibling_rank_group(graph: graphviz.Digraph, chain: SiblingOrderChain) -> None:
    """Emit one sibling rank group into ``graph``."""

    # Bracket the markers AROUND the whole subgraph (in the parent body) so that
    # ``_strip_sibling_rank_groups`` removes the entire ``{ rank=same ... }`` block,
    # not just its interior (which would leave an orphan empty wrapper).
    graph.body.append("\t// tl:sibling-order:start\n")
    with graph.subgraph() as rank_group:
        rank_group.attr(rank="same")
        for target in chain.targets:
            rank_group.node(target)
        for left, right in zip(chain.targets, chain.targets[1:]):
            rank_group.edge(
                left,
                right,
                style="invis",
                weight="100",
                comment="tl:sibling-order",
            )
    graph.body.append("\t// tl:sibling-order:end\n")


def _insert_before_final_brace(source: str, insertion: str) -> str:
    """Insert text before the final top-level DOT brace."""

    index = source.rfind("}")
    if index == -1:
        return source + insertion
    return source[:index] + insertion + source[index:]


def _insert_into_cluster(source: str, cluster_name: str, insertion: str) -> tuple[str, bool]:
    """Insert text immediately after a cluster's opening brace."""

    markers = (f"subgraph {cluster_name} {{", f"subgraph {quote_dot_id(cluster_name)} {{")
    index = -1
    for marker in markers:
        index = source.find(marker)
        if index != -1:
            break
    if index == -1:
        return source, False
    insert_at = source.find("\n", index)
    if insert_at == -1:
        return source, False
    return source[: insert_at + 1] + insertion + source[insert_at + 1 :], True


def _setup_subgraphs(
    self: "Trace",
    graphviz_graph: graphviz.Digraph,
    vis_mode: str,
    module_edge_dict: Dict[str, Any],
    overrides: Optional[VisualizationOverrides] = None,
    top_level_rank_groups: Sequence[SiblingOrderChain] = (),
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
    if "self" not in self.modules:
        return
    if vis_mode == "unrolled":
        module_submodule_dict = defaultdict(list)
        for call_label, mpl in self.modules._pass_dict.items():
            module_submodule_dict[call_label] = list(mpl.call_children)
        subgraphs = list(self.modules["self"].ops[0].call_children)  # type: ignore[union-attr]
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
    emitted_rank_groups = 0
    while len(subgraph_stack) > 0:
        parent_graph_list = subgraph_stack.pop(0)
        emitted_rank_groups += _setup_subgraphs_recurse(
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
    for chain in top_level_rank_groups:
        _emit_sibling_rank_group(graphviz_graph, chain)
        emitted_rank_groups += 1
    queued_rank_groups = len(top_level_rank_groups) + sum(
        len(data.get("rank_groups", [])) for data in module_edge_dict.values()
    )
    assert queued_rank_groups == emitted_rank_groups


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
) -> int:
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
        subgraph_title = (
            f"{subgraph_module} (x{sg_ml.num_calls}"
            f"{_collapsed_module_rolling_suffix(self, subgraph_module)})"
        )
    else:
        subgraph_title = subgraph_module

    if call_depth < len(parent_graph_list) - 1:  # we haven't gotten to the bottom yet, keep going.
        with starting_subgraph.subgraph(name=cluster_name) as s:
            return _setup_subgraphs_recurse(
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
        emitted_rank_groups = 0
        cluster_payload = module_edge_dict[subgraph_name]
        if (
            sg_ml.num_layers <= 1  # type: ignore[union-attr]
            and not module_submodule_dict[subgraph_name_w_pass]
            and not cluster_payload.get("nodes")
            and not cluster_payload.get("edges")
            and not cluster_payload.get("rank_groups")
        ):
            return emitted_rank_groups
        with starting_subgraph.subgraph(name=cluster_name) as s:
            # Penwidth + cluster attrs come from ``_render_utils`` so the
            # bundle renderer in ``multi_trace/visualization.py`` can build
            # equivalent clusters with the same formula and label format.
            pen_width = compute_module_penwidth(call_depth, max_call_depth)
            if cluster_payload["has_input_ancestor"]:
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
            for chain in cluster_payload.get("rank_groups", []):
                _emit_sibling_rank_group(s, chain)
                emitted_rank_groups += 1
            subgraph_nodes = cluster_payload.get("nodes", [])
            for node_args in subgraph_nodes:
                s.node(**node_args)
            subgraph_edges = cluster_payload["edges"]
            for edge_dict in subgraph_edges:
                s.edge(**edge_dict)
            subgraph_children = module_submodule_dict[subgraph_name_w_pass]
            for subgraph_child in subgraph_children:  # it's weird but have to go in reverse order.
                subgraph_stack.append(parent_graph_list[:] + [subgraph_child])
        return emitted_rank_groups


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

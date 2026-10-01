"""Shared render types, constants, and imports for Graphviz rendering."""

import base64
import copy
import html
import os
import re
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# UP035 rationale (suppressed on the import line below): `Dict`/`List`/`Set`/
# `Tuple` are deliberately re-exported through this module's __all__ for the
# sibling renderers that do `from ._render_common import *`, which still
# annotate with them. Modernizing the alias here means modernizing every
# star-import consumer first. (Reworded so this prose line no longer LEXES as
# a blanket noqa directive — SF-24.)
from typing import (  # noqa: UP035
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
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

from .._errors import _actionable_message, _ActionableErrorMixin
from .._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    FoldRepeatsLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from ..data_classes.internal_types import VisualizationOverrides
from ..data_classes.layer import Layer
from ..data_classes.op import Op
from ..errors._base import CompatibilityError
from ..ir.container import (
    ContainerSpec,
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
)
from ..ir.container_registry import ContainerRecord, ContainerSnapshot, Role
from ..quantities import Duration
from ..utils.display import _timed_phase, _vprint, in_notebook, int_list_to_compact_str
from ..utils.env_flags import closed_bool_env
from ._label_format import (
    format_memory,
    format_module_kwargs,
    format_module_path,
    format_param_list,
    format_shape,
    saved_for_backward_line,
)
from ._render_utils import (
    _open_file_quietly,
    compute_module_penwidth,
    direction_to_rankdir,
    make_module_cluster_attrs,
    relativize_visualizer_image,
)
from .code_panel import (
    CodePanelOption,
    compose_graph_with_code_panel,
    render_code_panel_subgraph,
    resolve_code_panel_source,
)
from .collapse_plan import CollapsePlan, RawOp, SegmentDescriptor
from .modes import COLLAPSED_MODE_REGISTRY, MODE_REGISTRY
from .node_spec import (
    INTERVENTION_CONE_COLOR,
    INTERVENTION_HOOK_BORDER_COLOR,
    INTERVENTION_HOOK_FILL_COLOR,
    INTERVENTION_SITE_COLOR,
    # S5 contract (C4): the three node-callback aliases have ONE declaration
    # home (node_spec.py); this module re-exports them for internal consumers.
    BackwardNodeSpecFn as BackwardNodeSpecFn,
    CollapsedNodeSpecFn as CollapsedNodeSpecFn,
    NodeSpec,
    NodeSpecFn as NodeSpecFn,
    graphviz_graph_overrides,
    intervention_graph_override,
    intervention_site_and_cone_labels,
    make_intervention_node_spec_fn,
    render_lines_to_html,
)
from .overlays import OverlayScores, overlay_border_attrs, overlay_line
from .render_ir import (
    RenderIRDotStatement,
    RenderIROrderingConstraint,
    build_render_ir,
    finalize_forward_regions,
    projected_antiparallel_endpoint_pairs,
)
from .request import RenderContext
from .themes import (
    VisualizationTheme,
    apply_theme_to_spec,
    resolve_theme,
    theme_edge_attrs,
    theme_graph_attrs,
    theme_node_attrs,
)


def strict_collapse_checks_enabled() -> bool:
    """Return whether collapse/sibling-order verification failures should raise.

    THE one parser of ``TORCHLENS_COLLAPSE_STRICT`` (r-b7 R42-9: it used to be
    duplicated byte-for-byte in ``auto_collapse`` and ``_render_dot``, a
    divergence hazard for a verification-arming knob).

    Keyed ONLY on the torchlens-owned knob (r3 b7-opus R47-A): the old
    ``PYTEST_CURRENT_TEST in os.environ`` arm meant a DOWNSTREAM project's
    test suite rendering a TorchLens graph got an ``AssertionError`` on a
    render that works in a script, from a knob it never set. The TorchLens
    suite arms the tripwire explicitly in ``tests/conftest.py``.

    Returns
    -------
    bool
        True when ``TORCHLENS_COLLAPSE_STRICT`` is set to an affirmative
        value (``1/true/yes/on``).

    Raises
    ------
    InvalidArgumentError
        When the variable is set to an unrecognized value (round-7 b7 R47:
        the historical exact-``"1"`` parse silently left this
        verification-arming tripwire OFF on ``=true`` or a typo while the
        exporter believed it was armed).
    """

    return closed_bool_env("TORCHLENS_COLLAPSE_STRICT")


def format_collapsed_module_contents(num_layers: int, num_buffer_layers: int) -> str:
    """Format the operation and buffer counts for a collapsed module.

    Parameters
    ----------
    num_layers:
        Total tensor-layer count already computed for the module.
    num_buffer_layers:
        Number of buffer tensor layers within ``num_layers``.

    Returns
    -------
    str
        An honest collapsed-module summary such as ``"2 ops + 6 buffers"``.
    """

    num_ops = max(0, num_layers - num_buffer_layers)
    parts: list[str] = []
    if num_ops:
        parts.append(f"{num_ops} {'op' if num_ops == 1 else 'ops'}")
    if num_buffer_layers:
        parts.append(f"{num_buffer_layers} {'buffer' if num_buffer_layers == 1 else 'buffers'}")
    return " + ".join(parts)


if TYPE_CHECKING:
    from ..data_classes.module import Module
    from .auto_collapse import ModuleRepeatFold

BaseGraphNode = Union["Op", "Layer"]
ShowContainersLiteral = Literal[False, "labels", "cluster", "collapsed", "auto", "nodes"]


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


GraphNode = BaseGraphNode | BoundaryNode | FocusNode
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
# Worst-case number of full ``dot -Tplain`` layouts the sibling-ordering
# verifier can run on top of the final render: baseline, injected, the
# post-filter re-layout, and up to two rejection retries.
SIBLING_ORDER_VERIFY_LAYOUT_BUDGET = 5
SIBLING_ORDER_COST_NOTICE = (
    "TorchLens skipped sibling-order verification: its up-to-{budget} extra "
    "layout passes would exceed the layout budget (estimated layout "
    "cost={cost} x {budget} > threshold={threshold}). Siblings render in "
    "Graphviz's default order. Reduce graph complexity with vis_call_depth, "
    "rolled mode, or module= focus, or disable this pass explicitly with "
    "order_siblings=False."
)
_SIBLING_ORDER_WARNING_EMITTED = False


class GraphvizRenderError(_ActionableErrorMixin, CompatibilityError, RuntimeError):
    """Raised when Graphviz fails to produce a usable rendered artifact.

    Keeps its historical ``RuntimeError`` base while joining the taxonomy
    with a stable code and a default remedy, so single-message raise sites
    stay valid.
    """

    code: str = "graphviz_render_failed"
    default_remedy: str = (
        "lower dpi, render direct SVG with vis_fileformat='svg', or reduce the "
        "graph with a node cap such as vis_call_depth"
    )

    def __init__(
        self,
        problem: str,
        *,
        remedy: str | None = None,
        **context: object,
    ) -> None:
        """Initialize an actionable Graphviz render failure.

        Parameters
        ----------
        problem:
            Description of the failed render and any saved DOT source path.
        remedy:
            Concrete caller action. The class default is used when omitted.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        resolved_remedy = remedy or type(self).default_remedy
        super().__init__(
            _actionable_message(problem, resolved_remedy),
            code=type(self).code,
            remedy=resolved_remedy,
            **cast(dict[str, Any], context),
        )


class GraphvizUnavailableError(GraphvizRenderError):
    """Raised when the Graphviz executable cannot be found on PATH.

    The most common cold-user visualization failure: the ``graphviz``
    Python package is installed but the system binary is not, so the
    spawn seam's ``exec`` fails. Distinct from ``graphviz_render_failed``
    (a present binary producing no usable artifact) because the remedy is
    installation, not render tuning.
    """

    code: str = "graphviz_binary_unavailable"
    default_remedy: str = (
        "install the Graphviz system package and ensure its binaries are on "
        "PATH (Debian/Ubuntu: apt install graphviz; macOS: brew install graphviz)"
    )


_GRAPHVIZ_ESCAPE_HINT = (
    "Try lowering dpi, rendering direct SVG with vis_fileformat='svg', or reducing the graph "
    "with a node cap such as vis_call_depth."
)


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
    occurrence_key:
        Stable key for the specific parent-to-child argument occurrence represented
        by this rendered edge.
    argument_label:
        Optional per-occurrence argument label for non-commutative child ops.
    """

    target: GraphNode
    metadata_child: GraphNode | None
    occurrence_key: tuple[Any, ...]
    argument_label: str | None = None


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
    occurrence_key:
        Stable key for the rendered edge occurrence. Parallel same-parent edges
        have distinct keys but the same rendered endpoints.
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
    occurrence_key: tuple[Any, ...]
    attrs: tuple[tuple[str, Any], ...]


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
class ContainerClusterSpec:
    """Graphviz cluster request for one single-owner output container."""

    cluster_id: str
    owner_key: str | int
    node_names: tuple[str, ...]
    title: str
    kind: str


@dataclass(frozen=True)
class ContainerOverlayEdge:
    """Container node or association edge emitted outside dataflow lists.

    Parameters
    ----------
    tail_name:
        Graphviz tail node name.
    head_name:
        Graphviz head node name.
    attrs:
        Graphviz edge attributes.
    """

    tail_name: str
    head_name: str
    attrs: dict[str, str]


@dataclass(frozen=True)
class ContainerOverlayNode:
    """Container node emitted outside dataflow node construction.

    Parameters
    ----------
    args:
        Graphviz node arguments.
    owner_key:
        Module cluster key for local placement, or ``None`` for top-level.
    """

    args: dict[str, str]
    owner_key: str | None


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


@dataclass(frozen=True)
class RenderedNodeEmission:
    """Renderer-faithful visible node emitted by the forward DOT path.

    Parameters
    ----------
    name:
        Graphviz node name after collapse and repeat-fold remapping.
    kind:
        Diagnostic node kind.
    node:
        Source render node when one exists.
    op_label:
        Source op or layer label for raw nodes.
    module_address:
        Pass-free module address for collapsed module boxes.
    call:
        Pass-qualified module call for collapsed module boxes.
    boundary_kind:
        Boundary kind for synthetic focus boundary nodes.
    fold:
        Repeat-fold represented or touched by this node.
    """

    name: str
    kind: Literal["raw_op", "module_box", "boundary", "run_fold_ellipsis", "hidden_run_member"]
    node: GraphNode | None = None
    op_label: str | None = None
    module_address: str | None = None
    call: str | None = None
    boundary_kind: str | None = None
    fold: "ModuleRepeatFold | None" = None


class _RenderIRDecisionBuilder:
    """Graph-like adapter used only while resolving immutable IR decisions."""

    def __init__(self) -> None:
        """Initialize an empty recorder."""

        self.calls: list[RenderIRDotStatement] = []
        self.body = _RenderIRBody(self.calls)

    def node(self, *args: Any, **kwargs: Any) -> None:
        """Record a Graphviz node call."""

        self.calls.append(RenderIRDotStatement("node", tuple(args), tuple(kwargs.items())))

    def edge(self, *args: Any, **kwargs: Any) -> None:
        """Record a Graphviz edge call."""

        self.calls.append(RenderIRDotStatement("edge", tuple(args), tuple(kwargs.items())))

    def attr(self, *args: Any, **kwargs: Any) -> None:
        """Record a Graphviz attribute statement."""

        self.calls.append(RenderIRDotStatement("attr", tuple(args), tuple(kwargs.items())))

    def subgraph(self, *args: Any, **kwargs: Any) -> "_RenderIRSubgraphDecisionBuilder":
        """Record a nested Graphviz subgraph."""

        return _RenderIRSubgraphDecisionBuilder(self.calls, tuple(args), dict(kwargs))


class _RenderIRSubgraphDecisionBuilder:
    """Context manager resolving a nested immutable IR statement."""

    def __init__(
        self,
        parent_calls: list[RenderIRDotStatement],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Initialize the nested recorder."""

        self._parent_calls = parent_calls
        self._args = args
        self._kwargs = kwargs
        self._children: list[RenderIRDotStatement] = []
        self.body = _RenderIRBody(self._children)

    def __enter__(self) -> "_RenderIRSubgraphDecisionBuilder":
        """Return the active nested recorder."""

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Append the subgraph call if the context exits successfully."""

        if exc_type is None:
            self._parent_calls.append(
                RenderIRDotStatement(
                    "subgraph",
                    self._args,
                    tuple(self._kwargs.items()),
                    children=tuple(self._children),
                )
            )

    def attr(self, *args: Any, **kwargs: Any) -> None:
        """Record a Graphviz attr call inside the subgraph."""

        self._children.append(RenderIRDotStatement("attr", tuple(args), tuple(kwargs.items())))

    def node(self, *args: Any, **kwargs: Any) -> None:
        """Record a Graphviz node call inside the subgraph."""

        self._children.append(RenderIRDotStatement("node", tuple(args), tuple(kwargs.items())))

    def edge(self, *args: Any, **kwargs: Any) -> None:
        """Record a Graphviz edge call inside the subgraph."""

        self._children.append(RenderIRDotStatement("edge", tuple(args), tuple(kwargs.items())))

    def subgraph(self, *args: Any, **kwargs: Any) -> "_RenderIRSubgraphDecisionBuilder":
        """Record a nested Graphviz subgraph."""

        return _RenderIRSubgraphDecisionBuilder(self._children, tuple(args), dict(kwargs))


class _RenderIRBody:
    """Minimal Graphviz body adapter recording already-resolved raw statements."""

    def __init__(self, calls: list[RenderIRDotStatement]) -> None:
        """Initialize the adapter over a statement list."""

        self._calls = calls

    def append(self, value: str) -> None:
        """Record one raw Graphviz body fragment."""

        self._calls.append(RenderIRDotStatement("raw", (value,)))


_CODE_PANEL_COMPOSED_FORMATS = frozenset({"svg", "pdf", "png"})


_SVG_IMAGE_TAG_RE = re.compile(r"<image\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_SVG_ATTR_RE = re.compile(r"""(?P<name>[\w:.-]+)\s*=\s*(?P<quote>["'])(?P<value>.*?)(?P=quote)""")
_SVG_ROOT_RE = re.compile(r"<svg\b(?P<attrs>[^>]*)>", re.IGNORECASE | re.DOTALL)
_SVG_VIEWBOX_RE = re.compile(r"""viewBox=(?P<quote>["'])(?P<value>[^"']+)(?P=quote)""")


_EDGE_LABEL_FONT_SIZE = 8
_EDGE_LABEL_PAD = 4  # points of transparent margin on every side of a head/tail label
_SELF_LOOP_LABEL_HGAP = 8  # points of blank spacer left/right of a self-loop label


# Placement attrs (labeldistance, labelangle) for head/tail pass labels on
# structurally at-risk rolled edges.  Setting EITHER attr switches graphviz
# from its default endpoint-label placement to ``place_portlabel``, which
# positions the label relative to the edge's END TANGENT -- so the label
# follows an oblique or bowed spline instead of clipping it.  The default
# placement is kept for ordinary (straight) edges: explicitly setting the
# documented "defaults" (1.0, -25) is NOT a no-op and measurably worsens them.
# Values chosen by offline audit-scored sweeps over the 16-model rolled
# inspection set (exact per-label geometry audit): each clears the listed
# failure class to zero hard violations while keeping labels within 9pt of
# their endpoint node.
#
# PORTABILITY WARNING: these placements are FONT-METRIC-SENSITIVE.  Graphviz
# resolves the default serif to different fonts per platform (macOS Times,
# Linux Liberation Serif), and sub-point glyph-width differences move the
# label bbox enough to flip a tight clearance into a penetration.  The
# original macOS/Times-tuned oblique pair ("2.0", "-45") had ALWAYS clipped
# 4 of the 16 models under Linux font metrics.  Current values were re-tuned
# 2026-08 on Linux (Liberation Serif) against dot 2.43 / 7.0.5 / 14.1
# simultaneously, selecting for the widest worst-case margin (>=2pt to the
# nearest spline/arrowhead/node, >=1pt of headroom under the 9pt orphan
# band) rather than bare passes.  If a class cannot reach robust margins on
# BOTH font stacks, do NOT re-tune it on your box (that re-breaks the other
# platform) -- move the class to a midpoint-merged label instead, which gets
# reserved layout space and is font-metric-independent (see the multi-step
# skip-edge merge in ``_render_edges._label_rolled_call_indexs``).
#
# Heads of >=3-op cycle body edges: the cycle's merged back-edge midpoint
# label bows the whole forward chain; labels otherwise clip their own
# spline/arrowhead.
_ROLLED_CYCLE_HEAD_LABEL_PLACEMENT = ("1.6", "-90")
# Heads of adjacent forward edges into a self-loop-bearing layer: the
# self-loop arc invades the default head-label spot.
_ROLLED_SELF_LOOP_HEAD_LABEL_PLACEMENT = ("1.6", "-65")
# Tails of >=3-op cycle body edges (heads use the steeper pair above).
# Multi-step skip edges touching a self-loop layer used to share this pair
# ("oblique" class); they midpoint-merge now -- no clean placement exists
# for them across engines and font stacks.
_ROLLED_CYCLE_TAIL_LABEL_PLACEMENT = ("2.6", "-45")

__all__ = [
    "Any",
    "BACKWARD_ACCUMULATION_EDGE_STYLE",
    "BACKWARD_HIGHER_ORDER_COLOR",
    "BACKWARD_NODE_BORDER_COLOR",
    "BACKWARD_NODE_COLOR",
    "BOOL_NODE_COLOR",
    "BackwardNodeSpecFn",
    "BackwardPassFilter",
    "BaseGraphNode",
    "BoundaryNode",
    "BufferVisibilityLiteral",
    "COLLAPSED_MODE_REGISTRY",
    "COMMUTE_FUNCS",
    "CapturedForwardEdge",
    "CodePanelOption",
    "CollapseFn",
    "CollapseLiteral",
    "CollapsePlan",
    "CollapsedNodeSpecFn",
    "ContainerClusterSpec",
    "ContainerOverlayEdge",
    "ContainerOverlayNode",
    "ContainerRecord",
    "ContainerSnapshot",
    "ContainerSpec",
    "DEFAULT_BG_COLOR",
    "DataclassField",
    "Dict",
    "DictKey",
    "Duration",
    "FROZEN_PARAMS_BG_COLOR",
    "FocusNode",
    "FoldRepeatsLiteral",
    "GRADIENT_ARROW_COLOR",
    "GraphNode",
    "GraphvizRenderError",
    "HFKey",
    "INPUT_COLOR",
    "INTERVENTION_HOOK_BORDER_COLOR",
    "INTERVENTION_HOOK_FILL_COLOR",
    "INTERVENTION_CONE_COLOR",
    "INTERVENTION_SITE_COLOR",
    "Image",
    "InterveningClusterMode",
    "Iterable",
    "Layer",
    "List",
    "Literal",
    "MODE_REGISTRY",
    "Mapping",
    "NamedField",
    "NodeSpec",
    "NodeSpecFn",
    "OUTPUT_COLOR",
    "Op",
    "Optional",
    "OutputPathComponent",
    "OverlayScores",
    "PARAMS_NODE_BG_COLOR",
    "Path",
    "PlainLayout",
    "RawOp",
    "RenderContext",
    "RenderEdge",
    "RenderedNodeEmission",
    "RenderIROrderingConstraint",
    "Role",
    "RollingAnnotation",
    "SIBLING_ORDER_COST_NOTICE",
    "SIBLING_ORDER_EPSILON",
    "SIBLING_ORDER_NODE_CAP",
    "SIBLING_ORDER_STRETCH_CAP",
    "SIBLING_ORDER_VERIFY_LAYOUT_BUDGET",
    "SegmentDescriptor",
    "Sequence",
    "Set",
    "ShowContainersLiteral",
    "SiblingOrderChain",
    "SiblingOrderDecision",
    "SkipFn",
    "TRAINABLE_PARAMS_BG_COLOR",
    "TYPE_CHECKING",
    "Tuple",
    "TupleIndex",
    "Union",
    "VisDirectionLiteral",
    "VisInterventionModeLiteral",
    "VisModeLiteral",
    "VisNodeModeLiteral",
    "VisNodePlacementLiteral",
    "VisRendererLiteral",
    "VisualizationOverrides",
    "VisualizationTheme",
    "_CODE_PANEL_COMPOSED_FORMATS",
    "_EDGE_LABEL_FONT_SIZE",
    "_EDGE_LABEL_PAD",
    "_RenderIRDecisionBuilder",
    "_GRAPHVIZ_ESCAPE_HINT",
    "_NOISE_BUFFER_NAMES",
    "_ROLLED_CYCLE_HEAD_LABEL_PLACEMENT",
    "_ROLLED_CYCLE_TAIL_LABEL_PLACEMENT",
    "_ROLLED_SELF_LOOP_HEAD_LABEL_PLACEMENT",
    "_SELF_LOOP_LABEL_HGAP",
    "_SIBLING_ORDER_WARNING_EMITTED",
    "_SVG_ATTR_RE",
    "_SVG_IMAGE_TAG_RE",
    "_SVG_ROOT_RE",
    "_SVG_VIEWBOX_RE",
    "_open_file_quietly",
    "relativize_visualizer_image",
    "_timed_phase",
    "_vprint",
    "apply_theme_to_spec",
    "base64",
    "build_render_ir",
    "finalize_forward_regions",
    "cast",
    "compose_graph_with_code_panel",
    "compute_module_penwidth",
    "copy",
    "dataclass",
    "defaultdict",
    "direction_to_rankdir",
    "field",
    "format_memory",
    "format_collapsed_module_contents",
    "strict_collapse_checks_enabled",
    "format_module_kwargs",
    "format_module_path",
    "format_param_list",
    "format_shape",
    "saved_for_backward_line",
    "graphviz",
    "graphviz_graph_overrides",
    "html",
    "in_notebook",
    "int_list_to_compact_str",
    "intervention_graph_override",
    "intervention_site_and_cone_labels",
    "make_intervention_node_spec_fn",
    "make_module_cluster_attrs",
    "os",
    "overlay_border_attrs",
    "overlay_line",
    "projected_antiparallel_endpoint_pairs",
    "quote_dot_id",
    "re",
    "render_code_panel_subgraph",
    "render_lines_to_html",
    "resolve_code_panel_source",
    "resolve_theme",
    "subprocess",
    "sys",
    "tempfile",
    "theme_edge_attrs",
    "theme_graph_attrs",
    "theme_node_attrs",
    "torch",
    "warnings",
]

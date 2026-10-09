"""Source nodes for in-place-mutated ``nn.Parameter`` objects in the forward render.

Capture records an in-place op on a prepared Parameter (MIX-HIC's
``with torch.no_grad(): self.temp.clamp_(...)``) as an ordinary op whose
parameter input is the Parameter; reads after it bind to the op as graph
parents. The Parameter itself has no op of its own, so the render adds one
synthetic source node per mutated Parameter: a cylinder (the buffer shape)
filled with the parameter-bearing grey, placed in its owning module's cluster
(the placement buffers get), with an edge to every op that reads the
Parameter's pre-mutation value (the reads before the first mutation and the
first mutation itself). Later mutations and reads already chain on the
captured graph.

The scan is read-only over captured metadata. A trace with no mutated
Parameter emits nothing, so its DOT stays byte-identical.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import quote as _percent_quote

from ._render_common import (
    FROZEN_PARAMS_BG_COLOR,
    TRAINABLE_PARAMS_BG_COLOR,
)

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ._render_dot import _ForwardRenderContext
    from .themes import VisualizationTheme

__all__ = [
    "MutatedParameterEmission",
    "MutatedParameterSource",
    "add_mutated_parameter_nodes",
    "find_mutated_parameter_sources",
    "mutated_parameter_fill",
    "mutated_parameter_node_name",
]

#: Unit kinds whose identifier is the emitted DOT node name of a reader.
_DRAWABLE_READER_KINDS = frozenset({"raw_op", "module_box"})

_NODE_NAME_PREFIX = "mutatedparam_"


@dataclass(frozen=True)
class MutatedParameterSource:
    """One Parameter mutated in place during the captured forward.

    Parameters
    ----------
    param:
        The ``Param`` record of the mutated Parameter.
    first_mutation:
        The first in-place op whose receiver is the Parameter.
    readers:
        Ops that consume the pre-mutation value, in execution order: every
        direct parameter read up to and including ``first_mutation``.
    """

    param: Any
    first_mutation: Any
    readers: tuple[Any, ...] = field(default_factory=tuple)


@dataclass
class _ParamNodeStub:
    """Module-path stand-in for the synthetic node in the cluster LCA helper."""

    modules: list[str]
    is_atomic_module: bool = False


def _is_inplace_func_name(func_name: object) -> bool:
    """Return whether ``func_name`` names an in-place tensor operation.

    Delegates to the capture wrapper's own receiver-mutation predicate so the
    render cannot drift from what capture treats as a mutation.

    Parameters
    ----------
    func_name:
        Captured ``Op.func_name``.

    Returns
    -------
    bool
        True for trailing-underscore methods (``clamp_``), augmented-assignment
        dunders, and item assignment.
    """

    if not isinstance(func_name, str) or not func_name:
        return False
    from ..backends.torch.wrappers import _func_mutates_receiver

    return _func_mutates_receiver(func_name)


def _receiver_param(op: Any) -> Any | None:
    """Return the Parameter an in-place op mutates, if its receiver is one.

    The receiver is argument position zero. A tensor parent in that slot means
    the receiver is a traced tensor (an already-mutated Parameter included),
    so only an op with no tensor parent there and a parameter input mutates a
    prepared Parameter directly. Parameter inputs are recorded in argument
    order, so the receiver is the first one.

    Parameters
    ----------
    op:
        Captured op.

    Returns
    -------
    Any | None
        The receiver ``Param``, or ``None``.
    """

    if not _is_inplace_func_name(getattr(op, "func_name", None)):
        return None
    param_logs = tuple(getattr(op, "_param_logs", ()) or ())
    if not param_logs:
        return None
    positions = getattr(op, "parent_arg_positions", None) or {}
    arg_positions = positions.get("args", {}) if isinstance(positions, Mapping) else {}
    if 0 in arg_positions:
        return None
    return param_logs[0]


def find_mutated_parameter_sources(trace: Trace) -> tuple[MutatedParameterSource, ...]:
    """Return every Parameter that an in-place op mutated during the forward.

    Parameters
    ----------
    trace:
        Finished trace.

    Returns
    -------
    tuple[MutatedParameterSource, ...]
        One record per mutated Parameter, in order of first mutation; empty
        when nothing was mutated.
    """

    ops = list(getattr(trace, "layer_list", ()) or ())
    first: dict[str, tuple[Any, int]] = {}
    for index, op in enumerate(ops):
        param = _receiver_param(op)
        if param is not None and param.address not in first:
            first[param.address] = (param, index)
    if not first:
        return ()
    readers: dict[str, list[Any]] = {address: [] for address in first}
    for index, op in enumerate(ops):
        for param in getattr(op, "_param_logs", ()) or ():
            entry = first.get(param.address)
            if entry is not None and index <= entry[1]:
                readers[param.address].append(op)
    return tuple(
        MutatedParameterSource(param, ops[index], tuple(readers[address]))
        for address, (param, index) in first.items()
    )


def mutated_parameter_node_name(address: str) -> str:
    """Return the DOT node name for a mutated Parameter's source node.

    Parameters
    ----------
    address:
        Fully qualified Parameter address.

    Returns
    -------
    str
        Identifier unique per address: every character outside
        ``[A-Za-z0-9_.~-]`` is percent-encoded (``%`` itself included), so the
        mapping is injective (``a.b`` and ``a_b`` stay distinct) and the name
        carries no DOT port separator or quote. Graphviz quotes it on emission,
        the same as the dotted module-cluster names.
    """

    return _NODE_NAME_PREFIX + _percent_quote(address, safe="")


def mutated_parameter_fill(param: Any) -> str:
    """Return the parameter-bearing fill for a Parameter's source node.

    Parameters
    ----------
    param:
        ``Param`` record.

    Returns
    -------
    str
        The trainable or frozen parameter grey a node bearing only this
        Parameter is painted with.
    """

    if getattr(param, "is_trainable", True) is False:
        return FROZEN_PARAMS_BG_COLOR
    return TRAINABLE_PARAMS_BG_COLOR


def _natural_label(node: Any, vis_mode: str) -> str:
    """Return the trace label identifying ``node`` independent of focus rewriting.

    Parameters
    ----------
    node:
        Captured op (or, in rolled mode, its layer).
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    str
        The op's pass-qualified label (unrolled) or its layer label (rolled).
    """

    if vis_mode == "unrolled" and hasattr(node, "label"):
        return str(node.label)
    return str(node.layer_label)


def _render_label_index(entries_to_plot: Mapping[str, Any], vis_mode: str) -> dict[str, str]:
    """Map each plotted op's natural label to the label the render keys it by.

    A plain render keys unrolled entries by the op's pass-qualified label; a
    module-focused render rewraps entries as focus nodes keyed by layer label,
    so the op label is resolved through the wrapped original.

    Parameters
    ----------
    entries_to_plot:
        Source-graph entries (focus-rewritten when a module focus is active).
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    dict[str, str]
        Natural label to render label; focus boundary nodes are omitted.
    """

    from ._render_common import BoundaryNode, FocusNode, _render_node_label

    index: dict[str, str] = {}
    for node in entries_to_plot.values():
        if isinstance(node, BoundaryNode):
            continue
        original = node.original if isinstance(node, FocusNode) else node
        index[_natural_label(original, vis_mode)] = _render_node_label(node, vis_mode)
    return index


def _owner_module_path(entry: Any, param: Any, vis_mode: str) -> tuple[str, ...] | None:
    """Return the owning module's cluster path on ``entry``'s rendered module path.

    When the owner is the module the op is drawn AS (an atomic-module box),
    the path stops at the enclosing scope.

    Parameters
    ----------
    entry:
        Render entry of an op drawn as its own node.
    param:
        ``Param`` record.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    tuple[str, ...] | None
        Module path (pass-qualified when unrolled), empty for a root-owned
        Parameter; ``None`` when the owner is not on the op's module path.
    """

    owner = str(getattr(param, "module_address", "") or "")
    if not owner:
        return ()
    modules = [str(module) for module in (getattr(entry, "modules", ()) or ())]
    if vis_mode == "rolled":
        modules = [module.split(":")[0] for module in modules]
    rendered_scope = modules[:-1] if getattr(entry, "is_atomic_module", False) else modules
    for depth, module in enumerate(rendered_scope):
        if module.split(":")[0] == owner:
            return tuple(rendered_scope[: depth + 1])
    if any(module.split(":")[0] == owner for module in modules):
        return tuple(rendered_scope)
    return None


def _param_node_args(param: Any, theme: VisualizationTheme | None) -> dict[str, str]:
    """Return Graphviz node arguments for a mutated Parameter's source node.

    Parameters
    ----------
    param:
        ``Param`` record.
    theme:
        Active theme.

    Returns
    -------
    dict[str, str]
        Node arguments, name included.
    """

    from ._label_format import _shape_with_trainability
    from ._render_leaf import _node_spec_to_graphviz_args
    from .node_spec import NodeSpec
    from .themes import apply_theme_to_spec

    trainable = getattr(param, "is_trainable", None)
    spec = NodeSpec(
        lines=[
            f"parameter {param.name}",
            _shape_with_trainability(getattr(param, "shape", ()), trainable),
        ],
        shape="cylinder",
        fillcolor=mutated_parameter_fill(param),
        fontcolor="black",
        color="black",
        # A Parameter has no input ancestor: dashed, like buffer sources.
        style="filled,dashed",
        extra_attrs={"ordering": "out"},
    )
    if theme is not None:
        spec = apply_theme_to_spec(spec, theme)
    node_args = _node_spec_to_graphviz_args(spec)
    node_args["name"] = mutated_parameter_node_name(str(param.address))
    return node_args


@dataclass(frozen=True)
class MutatedParameterEmission:
    """One drawn mutated-Parameter node and its read edges, engine-neutral.

    Parameters
    ----------
    cluster_key:
        Module-cluster key the node sits in, or ``None`` for top level.
    node_args:
        Graphviz node arguments, ``name`` included.
    edges:
        ``(cluster_key, edge_args)`` per read edge; ``None`` is top level.
    """

    cluster_key: str | None
    node_args: Mapping[str, Any]
    edges: tuple[tuple[str | None, Mapping[str, Any]], ...]


def _queue(
    module_clusters: MutableMapping[str, Any],
    builder: Any,
    cluster_key: str | None,
    kind: str,
    args: Mapping[str, Any],
) -> None:
    """Queue a node or edge in its module cluster, or at top level.

    Parameters
    ----------
    module_clusters:
        Module-cluster accumulator.
    builder:
        Top-level graph builder.
    cluster_key:
        Cluster key, or ``None`` for top level.
    kind:
        ``"nodes"`` or ``"edges"``.
    args:
        Graphviz arguments.
    """

    if cluster_key is None:
        if kind == "nodes":
            builder.node(**args)
        else:
            builder.edge(**args)
        return
    module_clusters[cluster_key].setdefault(kind, []).append(dict(args))


@dataclass(frozen=True)
class _EmitScope:
    """Per-draw lookups shared by every mutated-Parameter emission.

    Parameters
    ----------
    entries:
        Render entries keyed by render label.
    render_labels:
        Natural op label to render label (see ``_render_label_index``).
    projection:
        Render label to visible unit identifier.
    unit_kinds:
        Visible unit identifier to unit kind.
    skipped:
        Render labels removed by the skip predicate.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Active module depth.
    focus_address:
        Address of the focused module, or ``None`` without a module focus.
    cluster_keys:
        Module-cluster keys the render drew before the Parameter nodes.
    theme:
        Active theme.
    """

    entries: Mapping[str, Any]
    render_labels: Mapping[str, str]
    projection: Mapping[str, str]
    unit_kinds: Mapping[str, str]
    skipped: frozenset[str]
    vis_mode: str
    vis_call_depth: int
    focus_address: str | None
    cluster_keys: frozenset[str]
    theme: VisualizationTheme | None


def _drawn_unit(op: Any, scope: _EmitScope, kinds: frozenset[str]) -> tuple[str, str] | None:
    """Return the render label and visible unit drawing ``op``, if of an allowed kind.

    Parameters
    ----------
    op:
        Captured op.
    scope:
        Per-draw lookups.
    kinds:
        Accepted unit kinds.

    Returns
    -------
    tuple[str, str] | None
        ``(render_label, unit_id)``; the unit id is the emitted DOT node name.
        ``None`` when the op is not plotted, skipped, or drawn as another kind.
    """

    raw = scope.render_labels.get(_natural_label(op, scope.vis_mode))
    if raw is None or raw in scope.skipped or raw not in scope.entries:
        return None
    unit = scope.projection.get(raw)
    if unit is None or scope.unit_kinds.get(unit) not in kinds:
        return None
    return raw, unit


def _owner_in_focus(param: Any, focus_address: str | None) -> bool:
    """Return whether the Parameter's owner lies inside the module focus.

    Parameters
    ----------
    param:
        ``Param`` record.
    focus_address:
        Focused module address, or ``None`` without a focus.

    Returns
    -------
    bool
        True without a focus, or when the owner is the focused module or one
        of its descendants.
    """

    if not focus_address:
        return True
    owner = str(getattr(param, "module_address", "") or "")
    return owner == focus_address or owner.startswith(focus_address + ".")


def _node_cluster_path(source: MutatedParameterSource, scope: _EmitScope) -> tuple[str, ...]:
    """Return the module path the Parameter's node is drawn in.

    The owner's cluster on the first mutation's path wins; a Parameter mutated
    outside its owner (a shared Parameter) takes the owner's drawn cluster on
    the first drawn reader's module path (the pass that read it first);
    otherwise the node sits at top level.

    Parameters
    ----------
    source:
        Mutated Parameter.
    scope:
        Per-draw lookups.

    Returns
    -------
    tuple[str, ...]
        Module path; empty for top level.
    """

    drawn = _drawn_unit(source.first_mutation, scope, frozenset({"raw_op"}))
    if drawn is not None:
        path = _owner_module_path(scope.entries[drawn[0]], source.param, scope.vis_mode)
        if path is not None:
            return path
    owner = str(getattr(source.param, "module_address", "") or "")
    for reader in source.readers:
        drawn = _drawn_unit(reader, scope, frozenset({"raw_op"}))
        if drawn is None:
            continue
        modules = [
            str(module) for module in (getattr(scope.entries[drawn[0]], "modules", ()) or ())
        ]
        if scope.vis_mode == "rolled":
            modules = [module.split(":")[0] for module in modules]
        for depth, module in enumerate(modules):
            # Only a cluster the render already drew (an atomic module is a box).
            if module.split(":")[0] == owner and module in scope.cluster_keys:
                return tuple(modules[: depth + 1])
    return ()


def _edge_args(node_name: str, head: str) -> dict[str, Any]:
    """Return Graphviz arguments for one dashed Parameter read edge.

    Parameters
    ----------
    node_name:
        Parameter node name.
    head:
        Reader unit identifier.

    Returns
    -------
    dict[str, Any]
        Edge arguments.
    """

    from ._typography import DEFAULT_TYPOGRAPHY

    return {
        "tail_name": node_name,
        "head_name": head,
        "color": "black",
        "fontcolor": "black",
        "style": "dashed",
        "arrowsize": ".7",
        "labelfontsize": DEFAULT_TYPOGRAPHY.annotation_pt,
    }


def _emission_for(
    source: MutatedParameterSource, scope: _EmitScope
) -> MutatedParameterEmission | None:
    """Resolve one Parameter node and its read edges.

    Parameters
    ----------
    source:
        Mutated Parameter to draw.
    scope:
        Per-draw lookups.

    Returns
    -------
    MutatedParameterEmission | None
        The emission, or ``None`` when the first mutation is not drawn as its
        own node or the owner lies outside the module focus.
    """

    from ._render_edges import _get_lowest_module_for_two_nodes
    from ._render_leaf import _base_node_for_metadata

    if not _owner_in_focus(source.param, scope.focus_address):
        return None
    if _drawn_unit(source.first_mutation, scope, frozenset({"raw_op"})) is None:
        return None
    node_args = _param_node_args(source.param, scope.theme)
    node_name = node_args["name"]
    module_path = _node_cluster_path(source, scope)
    stub = _ParamNodeStub(modules=list(module_path))
    heads: set[str] = set()
    edges: list[tuple[str | None, Mapping[str, Any]]] = []
    for reader in source.readers:
        drawn = _drawn_unit(reader, scope, _DRAWABLE_READER_KINDS)
        if drawn is None or drawn[1] in heads:
            continue
        heads.add(drawn[1])
        edge_key = _get_lowest_module_for_two_nodes(
            stub,  # type: ignore[arg-type]
            _base_node_for_metadata(scope.entries[drawn[0]]),
            False,
            scope.vis_call_depth,
        )
        edges.append((None if edge_key == -1 else str(edge_key), _edge_args(node_name, drawn[1])))
    return MutatedParameterEmission(
        cluster_key=module_path[-1] if module_path else None,
        node_args=node_args,
        edges=tuple(edges),
    )


def _focus_address(trace: Trace, context: _ForwardRenderContext) -> str | None:
    """Return the focused module's address, or ``None`` without a module focus.

    Parameters
    ----------
    trace:
        Trace being rendered.
    context:
        Resolved forward render context.

    Returns
    -------
    str | None
        Focus address.
    """

    if context.request.module is None:
        return None
    from .source_graph import _resolve_focus_module

    return str(_resolve_focus_module(trace, context.request.module).address)


def add_mutated_parameter_nodes(
    trace: Trace,
    context: _ForwardRenderContext,
    builder: Any,
    module_clusters: MutableMapping[str, Any],
) -> tuple[MutatedParameterEmission, ...]:
    """Emit one source node per mutated Parameter plus its read edges.

    A node is drawn only when the Parameter's first mutation is itself drawn
    as its own node; a mutation hidden in a collapsed module or skipped stays
    hidden with it. Under a module focus the node is drawn when its owner is
    the focused module or inside it.

    Parameters
    ----------
    trace:
        Trace being rendered.
    context:
        Resolved forward render context (node universe, request, theme).
    builder:
        Top-level graph builder.
    module_clusters:
        Module-cluster accumulator.

    Returns
    -------
    tuple[MutatedParameterEmission, ...]
        The drawn nodes with their edges (the rank engine re-emits them);
        empty when none was drawn.
    """

    sources = find_mutated_parameter_sources(trace)
    if not sources:
        return ()
    from ._render_common import _render_node_label

    universe = context.node_universe
    vis_mode = context.request.vis_mode
    entries_to_plot = universe.source_graph.entries_to_plot
    scope = _EmitScope(
        entries={_render_node_label(node, vis_mode): node for node in entries_to_plot.values()},
        render_labels=_render_label_index(entries_to_plot, vis_mode),
        projection=universe.endpoint_projection,
        unit_kinds={unit.unit_id: unit.kind for unit in universe.units},
        skipped=frozenset(universe.source_graph.skipped_labels),
        vis_mode=vis_mode,
        vis_call_depth=context.request.vis_call_depth,
        focus_address=_focus_address(trace, context),
        cluster_keys=frozenset(str(key) for key in module_clusters),
        theme=context.theme,
    )
    emissions: list[MutatedParameterEmission] = []
    for source in sources:
        emission = _emission_for(source, scope)
        if emission is None:
            continue
        emissions.append(emission)
        _queue(module_clusters, builder, emission.cluster_key, "nodes", emission.node_args)
        for edge_key, edge_args in emission.edges:
            _queue(module_clusters, builder, edge_key, "edges", edge_args)
    return tuple(emissions)

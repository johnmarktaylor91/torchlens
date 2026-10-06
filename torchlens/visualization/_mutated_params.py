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

import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._render_common import (
    FROZEN_PARAMS_BG_COLOR,
    TRAINABLE_PARAMS_BG_COLOR,
)

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ._render_dot import _ForwardRenderContext
    from .themes import VisualizationTheme

__all__ = [
    "MutatedParameterSource",
    "add_mutated_parameter_nodes",
    "find_mutated_parameter_sources",
    "mutated_parameter_fill",
    "mutated_parameter_node_name",
]

#: In-place dunder operators: augmented assignment plus item assignment.
_INPLACE_DUNDERS = frozenset(
    {
        "__iadd__",
        "__isub__",
        "__imul__",
        "__itruediv__",
        "__ifloordiv__",
        "__imod__",
        "__ipow__",
        "__imatmul__",
        "__iand__",
        "__ior__",
        "__ixor__",
        "__ilshift__",
        "__irshift__",
        "__setitem__",
    }
)

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

    Parameters
    ----------
    func_name:
        Captured ``Op.func_name``.

    Returns
    -------
    bool
        True for trailing-underscore methods (``clamp_``) and in-place dunders.
    """

    if not isinstance(func_name, str) or not func_name:
        return False
    if func_name.startswith("__"):
        return func_name in _INPLACE_DUNDERS
    return func_name.endswith("_")


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
        DOT-safe identifier that cannot collide with op node names.
    """

    return _NODE_NAME_PREFIX + re.sub(r"\W", "_", address)


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


def _raw_label(op: Any, vis_mode: str) -> str:
    """Return the raw render label of ``op`` in the active mode.

    Parameters
    ----------
    op:
        Captured op.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    str
        The label keyed by the node universe's endpoint projection.
    """

    return str(op.label) if vis_mode == "unrolled" else str(op.layer_label)


def _owner_module_path(entry: Any, param: Any, vis_mode: str) -> tuple[str, ...]:
    """Return the module-cluster path the Parameter's node is drawn in.

    The node sits in its owning module's cluster on the first mutation's
    rendered module path. When the owner is the module the op is drawn AS (an
    atomic-module box), the node sits beside it in the enclosing scope; when
    the owner is not on the path at all, at top level.

    Parameters
    ----------
    entry:
        Render entry of the first mutation.
    param:
        ``Param`` record.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    tuple[str, ...]
        Module path (pass-qualified when unrolled); empty for top level.
    """

    modules = [str(module) for module in (getattr(entry, "modules", ()) or ())]
    if vis_mode == "rolled":
        modules = [module.split(":")[0] for module in modules]
    rendered_scope = modules[:-1] if getattr(entry, "is_atomic_module", False) else modules
    owner = str(getattr(param, "module_address", "") or "")
    for depth, module in enumerate(rendered_scope):
        if module.split(":")[0] == owner:
            return tuple(rendered_scope[: depth + 1])
    if any(module.split(":")[0] == owner for module in modules):
        return tuple(rendered_scope)
    return ()


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


def _queue(
    module_clusters: MutableMapping[str, Any],
    builder: Any,
    module_key: str | int,
    kind: str,
    args: dict[str, Any],
) -> None:
    """Queue a node or edge in its module cluster, or at top level.

    Parameters
    ----------
    module_clusters:
        Module-cluster accumulator.
    builder:
        Top-level graph builder.
    module_key:
        Cluster key, or ``-1`` for top level.
    kind:
        ``"nodes"`` or ``"edges"``.
    args:
        Graphviz arguments.
    """

    if module_key == -1:
        if kind == "nodes":
            builder.node(**args)
        else:
            builder.edge(**args)
        return
    module_clusters[str(module_key)].setdefault(kind, []).append(args)


@dataclass(frozen=True)
class _EmitScope:
    """Per-draw lookups shared by every mutated-Parameter emission.

    Parameters
    ----------
    entries:
        Render entries keyed by raw render label.
    projection:
        Raw render label to visible unit identifier.
    unit_kinds:
        Visible unit identifier to unit kind.
    skipped:
        Raw render labels removed by the skip predicate.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    vis_call_depth:
        Active module depth.
    theme:
        Active theme.
    builder:
        Top-level graph builder.
    module_clusters:
        Module-cluster accumulator.
    """

    entries: Mapping[str, Any]
    projection: Mapping[str, str]
    unit_kinds: Mapping[str, str]
    skipped: frozenset[str]
    vis_mode: str
    vis_call_depth: int
    theme: VisualizationTheme | None
    builder: Any
    module_clusters: MutableMapping[str, Any]


def _drawn_unit(raw: str, scope: _EmitScope, kinds: frozenset[str]) -> str | None:
    """Return the visible unit drawing raw label ``raw``, if it has an allowed kind.

    Parameters
    ----------
    raw:
        Raw render label.
    scope:
        Per-draw lookups.
    kinds:
        Accepted unit kinds.

    Returns
    -------
    str | None
        The unit identifier (the emitted DOT node name), or ``None``.
    """

    if raw in scope.skipped or raw not in scope.entries:
        return None
    unit = scope.projection.get(raw)
    if unit is None or scope.unit_kinds.get(unit) not in kinds:
        return None
    return unit


def _emit_source(source: MutatedParameterSource, scope: _EmitScope) -> str | None:
    """Queue one Parameter node and its read edges; return the node name.

    Parameters
    ----------
    source:
        Mutated Parameter to draw.
    scope:
        Per-draw lookups.

    Returns
    -------
    str | None
        The node name, or ``None`` when the first mutation is not drawn as its
        own node.
    """

    from ._render_edges import _get_lowest_module_for_two_nodes
    from ._render_leaf import _base_node_for_metadata
    from ._typography import DEFAULT_TYPOGRAPHY

    first_raw = _raw_label(source.first_mutation, scope.vis_mode)
    if _drawn_unit(first_raw, scope, frozenset({"raw_op"})) is None:
        return None
    node_args = _param_node_args(source.param, scope.theme)
    node_name = node_args["name"]
    module_path = _owner_module_path(scope.entries[first_raw], source.param, scope.vis_mode)
    owner_key: str | int = module_path[-1] if module_path else -1
    _queue(scope.module_clusters, scope.builder, owner_key, "nodes", node_args)
    stub = _ParamNodeStub(modules=list(module_path))
    heads: set[str] = set()
    for reader in source.readers:
        raw = _raw_label(reader, scope.vis_mode)
        head = _drawn_unit(raw, scope, _DRAWABLE_READER_KINDS)
        if head is None or head in heads:
            continue
        heads.add(head)
        edge_key = _get_lowest_module_for_two_nodes(
            stub,  # type: ignore[arg-type]
            _base_node_for_metadata(scope.entries[raw]),
            False,
            scope.vis_call_depth,
        )
        edge_args = {
            "tail_name": node_name,
            "head_name": head,
            "color": "black",
            "fontcolor": "black",
            "style": "dashed",
            "arrowsize": ".7",
            "labelfontsize": DEFAULT_TYPOGRAPHY.annotation_pt,
        }
        _queue(scope.module_clusters, scope.builder, edge_key, "edges", edge_args)
    return node_name


def add_mutated_parameter_nodes(
    trace: Trace,
    context: _ForwardRenderContext,
    builder: Any,
    module_clusters: MutableMapping[str, Any],
) -> tuple[str, ...]:
    """Emit one source node per mutated Parameter plus its read edges.

    A node is drawn only when the Parameter's first mutation is itself drawn
    as its own node; a mutation hidden in a collapsed module, skipped, or
    outside a module focus stays hidden with it.

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
    tuple[str, ...]
        Names of the emitted Parameter nodes; empty when none was drawn.
    """

    sources = find_mutated_parameter_sources(trace)
    if not sources:
        return ()
    from ._render_edges import _render_node_label

    universe = context.node_universe
    vis_mode = context.request.vis_mode
    scope = _EmitScope(
        entries={
            _render_node_label(node, vis_mode): node
            for node in universe.source_graph.entries_to_plot.values()
        },
        projection=universe.endpoint_projection,
        unit_kinds={unit.unit_id: unit.kind for unit in universe.units},
        skipped=frozenset(universe.source_graph.skipped_labels),
        vis_mode=vis_mode,
        vis_call_depth=context.request.vis_call_depth,
        theme=context.theme,
        builder=builder,
        module_clusters=module_clusters,
    )
    emitted = (_emit_source(source, scope) for source in sources)
    return tuple(name for name in emitted if name is not None)

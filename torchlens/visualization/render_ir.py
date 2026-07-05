"""Resolved render-IR adapters for TorchLens graph visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .collapse_plan import RenderContext

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ..data_classes.module import Module
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRunFold
    from .rendering import RenderedNodeEmission


@dataclass(frozen=True)
class RenderIRNode:
    """Resolved render node independent of Graphviz mutation.

    Parameters
    ----------
    name:
        Rendered node identifier.
    kind:
        Semantic node kind from the renderer-faithful node universe.
    owner_cluster:
        Pass-free module cluster that owns the node when known.
    source_label:
        Source op/layer label or module call backing the node.
    hidden_originals:
        Original module addresses hidden by this rendered node.
    """

    name: str
    kind: Literal["raw_op", "module_box", "boundary", "run_fold_ellipsis", "hidden_run_member"]
    owner_cluster: str | None
    source_label: str | None
    hidden_originals: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderIR:
    """Resolved render description slice used before DOT emission.

    Parameters
    ----------
    context:
        Render context used to resolve visibility.
    nodes:
        Resolved nodes in deterministic renderer emission order.
    node_emissions:
        Legacy node emission adapter kept during migration.
    """

    context: RenderContext
    nodes: tuple[RenderIRNode, ...]
    node_emissions: tuple["RenderedNodeEmission", ...]


def build_render_ir(
    trace: "Trace",
    *,
    collapse_fn: "Callable[[Module], bool] | None",
    run_folds: "Mapping[str, ModuleRunFold] | None",
    context: RenderContext | None = None,
) -> RenderIR:
    """Build the first render-IR slice from current renderer-faithful emissions.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate.
    run_folds:
        Active run-fold descriptors.
    context:
        Render context. Defaults to :class:`RenderContext`.

    Returns
    -------
    RenderIR
        Node-level render IR whose emission order matches the current renderer.
    """

    resolved_context = RenderContext() if context is None else context
    from .rendering import rendered_node_universe_from_v1

    emissions = rendered_node_universe_from_v1(
        trace,
        collapse_fn=collapse_fn,
        run_folds=run_folds,
        context=resolved_context,
    )
    nodes = tuple(_node_from_emission(emission) for emission in emissions)
    return RenderIR(context=resolved_context, nodes=nodes, node_emissions=emissions)


def _node_from_emission(emission: "RenderedNodeEmission") -> RenderIRNode:
    """Convert a legacy node emission into a render-IR node."""

    hidden_originals: tuple[str, ...] = ()
    if emission.fold is not None and emission.kind in {"module_box", "run_fold_ellipsis"}:
        hidden_originals = tuple(emission.fold.addresses)
    source_label = emission.op_label or emission.call or emission.boundary_kind
    return RenderIRNode(
        name=emission.name,
        kind=emission.kind,
        owner_cluster=emission.module_address,
        source_label=source_label,
        hidden_originals=hidden_originals,
    )

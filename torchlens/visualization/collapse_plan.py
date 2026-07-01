"""Diagnostic collapse-plan AST and renderer-faithful count helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .._literals import BufferVisibilityLiteral, VisModeLiteral, VisNodePlacementLiteral

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRunFold


@dataclass(frozen=True)
class RenderContext:
    """Visualization context bits that affect collapse planning.

    Parameters
    ----------
    vis_mode:
        Render granularity, ``"unrolled"`` by default.
    show_buffer_layers:
        Buffer visibility policy used by the renderer.
    show_containers:
        Container overlay policy. R1a parity is scoped to ``False``.
    engine:
        Graph layout engine. R1a parity is scoped to Graphviz ``dot``.
    """

    vis_mode: VisModeLiteral = "unrolled"
    show_buffer_layers: BufferVisibilityLiteral = "meaningful"
    show_containers: Literal[False, "labels", "cluster", "collapsed", "auto", "nodes"] = False
    engine: VisNodePlacementLiteral = "dot"


@dataclass(frozen=True)
class ModuleBox:
    """Collapsed module-call unit in a collapse plan.

    Parameters
    ----------
    call:
        Pass-qualified rendered module call.
    """

    call: str


@dataclass(frozen=True)
class RawOp:
    """Exposed operation node in a collapse plan.

    Parameters
    ----------
    op:
        Operation represented by the rendered node.
    """

    op: "Op | str"


@dataclass(frozen=True)
class EllipsisNode:
    """Rendered node that stands in for hidden run-fold members.

    Parameters
    ----------
    members:
        Hidden module addresses represented by the ellipsis.
    """

    members: tuple[str, ...]


@dataclass(frozen=True)
class RunFold:
    """Collapsed sibling run represented by a module box plus ellipsis.

    Parameters
    ----------
    rep:
        Representative module box.
    members:
        All folded module addresses, including the representative.
    ellipsis:
        Ellipsis node representing ``members[1:]``.
    """

    rep: ModuleBox
    members: tuple[str, ...]
    ellipsis: EllipsisNode


@dataclass(frozen=True)
class OpSegment:
    """Condensed operation chain placeholder for later v2 phases.

    Parameters
    ----------
    ops:
        Operation labels in the segment.
    """

    ops: tuple[str, ...]


@dataclass(frozen=True)
class ChildSegment:
    """Condensed child-module chain placeholder for later v2 phases.

    Parameters
    ----------
    members:
        Child module addresses in the segment.
    """

    members: tuple[str, ...]


@dataclass(frozen=True)
class Boundary:
    """Renderer boundary node.

    Parameters
    ----------
    kind:
        Boundary kind, such as ``"input"`` or ``"output"``.
    """

    kind: str


PlanNode = ModuleBox | RawOp | RunFold | OpSegment | ChildSegment | Boundary


@dataclass(frozen=True)
class CollapsePlan:
    """Renderer-faithful collapse plan.

    Parameters
    ----------
    nodes:
        Top-level plan nodes in deterministic render-enumeration order.
    context:
        Render context used to build the plan.
    """

    nodes: tuple[PlanNode, ...]
    context: RenderContext


def count(plan: CollapsePlan) -> int:
    """Return the rendered node count implied by ``plan``.

    Parameters
    ----------
    plan:
        Collapse plan to count.

    Returns
    -------
    int
        Number of rendered Graphviz node groups represented by the plan.
    """

    total = 0
    for node in plan.nodes:
        total += 2 if isinstance(node, RunFold) else 1
    return total


def plan_from_v1(
    trace: "Trace",
    collapse_fn: Callable[["Module"], bool] | None,
    run_folds: Mapping[str, "ModuleRunFold"] | None,
    context: RenderContext | None = None,
) -> CollapsePlan:
    """Reconstruct the plan implied by current v1 renderer decisions.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active v1 collapse predicate.
    run_folds:
        Current v1 run-fold mapping.
    context:
        Render context. Defaults to the scoped S7 parity context.

    Returns
    -------
    CollapsePlan
        Plan whose count matches the renderer's SVG node count in the default
        unrolled/dot/default-buffer/container-off context.
    """

    resolved_context = RenderContext() if context is None else context
    from .rendering import rendered_node_universe_from_v1

    emissions = rendered_node_universe_from_v1(
        trace,
        collapse_fn=collapse_fn,
        run_folds=run_folds,
        context=resolved_context,
    )
    nodes: list[PlanNode] = []
    consumed_ellipsis: set[str] = set()
    for emission in emissions:
        if emission.kind in {"run_fold_ellipsis", "hidden_run_member"}:
            continue
        if emission.kind == "boundary":
            nodes.append(Boundary(emission.boundary_kind or "boundary"))
            continue
        if emission.kind == "module_box":
            fold = emission.fold
            if fold is not None and emission.module_address == fold.representative:
                ellipsis_name = f"{emission.name}___runfoldellipsis"
                if any(
                    candidate.name == ellipsis_name and candidate.kind == "run_fold_ellipsis"
                    for candidate in emissions
                ):
                    consumed_ellipsis.add(ellipsis_name)
                    nodes.append(
                        RunFold(
                            rep=ModuleBox(emission.call or emission.name),
                            members=tuple(fold.addresses),
                            ellipsis=EllipsisNode(tuple(fold.addresses[1:])),
                        )
                    )
                    continue
            nodes.append(ModuleBox(emission.call or emission.name))
            continue
        nodes.append(RawOp(emission.op_label or emission.name))
    for emission in emissions:
        if emission.kind == "run_fold_ellipsis" and emission.name not in consumed_ellipsis:
            nodes.append(Boundary("run_fold_ellipsis"))
    return CollapsePlan(nodes=tuple(nodes), context=resolved_context)

"""Diagnostic collapse-plan AST and renderer-faithful count helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .._errors import InvalidArgumentError
from .request import RenderContext

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .auto_collapse import ModuleRepeatFold
    from .source_graph import SourceGraph


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

    op: Op | str


@dataclass(frozen=True)
class EllipsisNode:
    """Rendered node that stands in for hidden repeat-fold members.

    Parameters
    ----------
    members:
        Hidden module addresses represented by the ellipsis.
    """

    members: tuple[str, ...]


@dataclass(frozen=True)
class RepeatFold:
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
class SegmentDescriptor:
    """Render-time metadata for a condensed segment node.

    Parameters
    ----------
    name:
        Deterministic Graphviz node name.
    kind:
        Segment kind, either ``"child"`` or ``"op"``.
    label:
        Human-readable range label.
    members:
        Child module addresses represented by a child segment.
    ops:
        Operation labels represented by an op segment.
    owner:
        Module-cluster owner key, or ``None`` for top-level emission.
    num_ops:
        Number of operations represented by the segment.
    num_buffers:
        Number of buffer layers represented by the segment.
    num_params:
        Number of parameters represented by the segment.
    """

    name: str
    kind: Literal["child", "op"]
    label: str
    members: tuple[str, ...] = ()
    ops: tuple[str, ...] = ()
    owner: str | None = None
    num_ops: int = 0
    num_buffers: int = 0
    num_params: int = 0


@dataclass(frozen=True)
class Boundary:
    """Renderer boundary node.

    Parameters
    ----------
    kind:
        Boundary kind, such as ``"input"`` or ``"output"``.
    """

    kind: str


PlanNode = ModuleBox | RawOp | RepeatFold | OpSegment | ChildSegment | Boundary


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

    @property
    def total(self) -> int:
        """Return the rendered node count represented by this plan."""

        return count(self)

    def __len__(self) -> int:
        """Return the rendered node count represented by this plan."""

        return self.total

    def __repr__(self) -> str:
        """Return a compact diagnostic summary by rendered node kind.

        Returns
        -------
        str
            Summary containing the total node count and per-kind counts.
        """

        counts = Counter(_plan_node_kind(node) for node in self.nodes)
        kind_summary = ", ".join(
            f"{kind}={counts[kind]}"
            for kind in (
                "module_box",
                "raw_op",
                "run_fold",
                "op_segment",
                "child_segment",
                "boundary",
            )
            if counts[kind]
        )
        return f"CollapsePlan(total={self.total}, {kind_summary})"


@dataclass(frozen=True)
class CollapseScheduleStep:
    """One point on the public float collapse schedule.

    Parameters
    ----------
    t:
        Collapse level represented by this step.
    target_count:
        Linear target visible-node count for ``t``.
    visible_count:
        Renderer-faithful visible-node count for ``plan``.
    collapsed_addresses:
        Hidden-unit witnesses that remain collapsed at this and all later
        steps: module addresses for boxes, run folds, and child segments,
        plus pass-qualified op labels for ops hidden by operation segments.
    plan:
        Renderer-faithful collapse plan for this step.
    events:
        Typed condensation events this step applies on top of the previous
        step (F11 ladder, collapse memo D8): coalesced no-op events ride the
        next effective stop, so every event appears on exactly one step.
        Session-time records (:class:`~.collapse_ladder.CollapseEvent`);
        never persisted.
    """

    t: float
    target_count: int
    visible_count: int
    collapsed_addresses: frozenset[str]
    plan: CollapsePlan
    events: tuple[Any, ...] = ()


@dataclass(frozen=True)
class CollapseSchedule:
    """Inspectable public float collapse schedule.

    Parameters
    ----------
    steps:
        Ordered monotone schedule points from ``t=0.0`` to ``t=1.0``.
    """

    steps: tuple[CollapseScheduleStep, ...]

    def at(self, t: float) -> CollapseScheduleStep:
        """Return the selected schedule step for collapse level ``t``.

        Parameters
        ----------
        t:
            Collapse level in ``[0.0, 1.0]``.

        Returns
        -------
        CollapseScheduleStep
            The deterministic schedule step selected for ``t``. ``t == 0.0``
            always selects the first, fully expanded step, matching the
            public contract that ``0.0`` preserves the full graph even when
            later steps share ``t == 0.0`` after rounding.

        Raises
        ------
        ValueError
            If ``t`` is outside ``[0.0, 1.0]``.
        """

        if not 0.0 <= t <= 1.0:
            raise InvalidArgumentError(
                f"collapse float level must be in [0.0, 1.0]; received {t!r}",
                code="collapse_level_invalid",
                remedy="pass a collapse level between 0.0 and 1.0",
                argument="t",
            )
        if t == 0.0:
            return self.steps[0]
        for step in reversed(self.steps):
            if t >= step.t:
                return step
        return self.steps[0]


def _plan_node_kind(node: PlanNode) -> str:
    """Return the public diagnostic kind for a collapse-plan node.

    Parameters
    ----------
    node:
        Plan node to classify.

    Returns
    -------
    str
        One of ``"module_box"``, ``"raw_op"``, ``"run_fold"``,
        ``"op_segment"``, ``"child_segment"``, or ``"boundary"``.
    """

    if isinstance(node, ModuleBox):
        return "module_box"
    if isinstance(node, RawOp):
        return "raw_op"
    if isinstance(node, RepeatFold):
        return "run_fold"
    if isinstance(node, OpSegment):
        return "op_segment"
    if isinstance(node, ChildSegment):
        return "child_segment"
    return "boundary"


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
        total += 2 if isinstance(node, RepeatFold) else 1
    return total


def collapse_plan_for_trace(
    trace: Trace,
    collapse_fn: Callable[[Module], bool] | None,
    repeat_folds: Mapping[str, ModuleRepeatFold] | None,
    context: RenderContext | None = None,
) -> CollapsePlan:
    """Build a collapse plan through the shared node-universe entry point.

    Parameters
    ----------
    trace:
        Trace being projected.
    collapse_fn:
        Active collapse predicate.
    repeat_folds:
        Active repeat-fold mapping.
    context:
        Resolved render context.

    Returns
    -------
    CollapsePlan
        Renderer-faithful structural plan.
    """

    from .source_graph import build_source_graph

    resolved_context = RenderContext() if context is None else context
    return collapse_plan_for_source_graph(
        build_source_graph(trace, resolved_context), collapse_fn, repeat_folds
    )


def collapse_plan_for_source_graph(
    source_graph: SourceGraph,
    collapse_fn: Callable[[Module], bool] | None,
    repeat_folds: Mapping[str, ModuleRepeatFold] | None,
    node_pool: dict[PlanNode, PlanNode] | None = None,
) -> CollapsePlan:
    """Build a collapse plan from one normalized source graph.

    Parameters
    ----------
    source_graph:
        Normalized source graph shared by one or more plan projections.
    collapse_fn:
        Active collapse predicate.
    repeat_folds:
        Active repeat-fold mapping.
    node_pool:
        Optional schedule-local value interner for immutable plan nodes.

    Returns
    -------
    CollapsePlan
        Renderer-faithful structural plan.
    """

    from .node_universe import build_node_universe

    universe = build_node_universe(source_graph, collapse_fn, repeat_folds)
    return collapse_plan_from_universe(universe, node_pool=node_pool)


def collapse_plan_from_universe(
    universe: Any,
    node_pool: dict[PlanNode, PlanNode] | None = None,
) -> CollapsePlan:
    """Convert visible structural units into the stable collapse-plan AST.

    Parameters
    ----------
    universe:
        Presentation-free node universe.
    node_pool:
        Optional schedule-local value interner for immutable plan nodes.

    Returns
    -------
    CollapsePlan
        Existing plan AST with unchanged count and representation semantics.
    """

    emissions = universe.emissions
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
                        RepeatFold(
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
    if node_pool is not None:
        nodes = [node_pool.setdefault(node, node) for node in nodes]
    return CollapsePlan(nodes=tuple(nodes), context=universe.source_graph.request)


# ---------------------------------------------------------------------------
# Persistable collapse plans (leverage B13): session artifacts that reapply
# under the guarded site join, refusing or reporting changed cohorts — never
# label failure, ordinal guessing, or silent replanning.
# ---------------------------------------------------------------------------

COLLAPSE_PLAN_PAYLOAD_KEY = "tl_collapse_plan_v1"


@dataclass(frozen=True)
class CollapsePlanReapply:
    """The settled reapply verdict for one persisted plan on one capture.

    ``applicable`` means every structural cohort of the SOURCE capture joined
    the target capture (corroborated or positional); ``changed_cohorts`` are
    the refused / one-sided structural keys with their dispositions.
    """

    applicable: bool
    changed_cohorts: tuple[tuple[str, str], ...]
    collapsed_addresses: tuple[str, ...]
    graph_shape_match: bool


def export_collapse_plan(trace: Any, mode: str = "auto") -> dict[str, Any]:
    """Export one capture's collapse plan as a JSON-safe persistable payload.

    The payload carries the plan's collapsed/visible addresses AND the source
    capture's full site profile (label->key map, per-call-instance cohort
    cardinalities, source witnesses), so a later
    :func:`reapply_collapse_plan` can run the guarded join against a NEW
    capture without the source trace. Session artifact — its home is a plain
    JSON file the caller owns, never the tlspec.
    """

    from ..postprocess._site_join import site_profile

    plan = trace.collapse_plan(mode=mode)
    profile = site_profile(trace)
    per_call = [
        {
            "module_site": list(cohort[0]),
            "layer_type": cohort[1],
            "output_slot": cohort[2],
            "counts": dict(counts),
        }
        for cohort, counts in profile.per_call_counts.items()
    ]
    witnesses = {
        key: [list(witness) if witness is not None else None for witness in values]
        for key, values in profile.witnesses.items()
    }
    addresses: list[str] = []
    for node in plan.nodes:
        if isinstance(node, ModuleBox):
            addresses.append(node.call)
        elif isinstance(node, RawOp):
            addresses.append(node.op if isinstance(node.op, str) else str(node.op.label))
        elif isinstance(node, RepeatFold):
            addresses.extend(node.members)
        elif isinstance(node, (OpSegment, ChildSegment)):
            addresses.extend(node.ops if isinstance(node, OpSegment) else node.members)
    return {
        COLLAPSE_PLAN_PAYLOAD_KEY: {
            "mode": mode,
            "plan_addresses": addresses,
            "graph_shape_hash": getattr(trace, "graph_shape_hash", None),
            "profile": {
                "keys": dict(profile.keys),
                "per_call_counts": per_call,
                "witnesses": witnesses,
            },
        }
    }


def _profile_from_payload(payload: dict[str, Any]) -> Any:
    """Rebuild the persisted source SiteProfile, fail-closed."""

    from ..postprocess._site_join import SiteProfile

    try:
        profile = payload["profile"]
        keys = {str(label): str(key) for label, key in profile["keys"].items()}
        per_call = {
            (
                tuple(entry["module_site"]),
                str(entry["layer_type"]),
                entry["output_slot"],
            ): {str(instance): int(count) for instance, count in entry["counts"].items()}
            for entry in profile["per_call_counts"]
        }
        witnesses = {
            str(key): frozenset(
                tuple(witness) if witness is not None else None for witness in values
            )
            for key, values in profile["witnesses"].items()
        }
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise InvalidArgumentError(
            "malformed tl_collapse_plan_v1 payload: the persisted site "
            "profile did not parse, and a torn plan never half-applies.",
            code="collapse_plan_payload_invalid",
            remedy="re-export the plan with export_collapse_plan(trace)",
        ) from exc
    return SiteProfile(keys=keys, per_call_counts=per_call, witnesses=witnesses)


def reapply_collapse_plan(
    trace: Any, exported: dict[str, Any], *, strict: bool = True
) -> CollapsePlanReapply:
    """Reapply one persisted plan on a NEW capture under the guarded join.

    Every structural cohort of the persisted source profile is joined against
    ``trace``'s live profile. Refused and one-sided cohorts become the
    CHANGED SET: with ``strict=True`` (default) any change refuses typed
    (``collapse_plan_cohorts_changed``); ``strict=False`` returns the report
    for the caller to adjudicate. Labels never enter the decision, ordinals
    are never guessed, and the plan is never silently replanned.
    """

    payload = exported.get(COLLAPSE_PLAN_PAYLOAD_KEY)
    if not isinstance(payload, dict):
        raise InvalidArgumentError(
            "not a tl_collapse_plan_v1 payload (missing the versioned key); "
            "foreign or torn payloads never half-apply.",
            code="collapse_plan_payload_invalid",
            remedy="pass the dict returned by export_collapse_plan(trace)",
        )
    from ..postprocess._site_join import join_site_profiles, site_profile

    source_profile = _profile_from_payload(payload)
    target_profile = site_profile(trace)
    rows = join_site_profiles(source_profile, target_profile)
    changed: list[tuple[str, str]] = []
    for key in sorted(set(source_profile.keys.values())):
        row = rows.get(key)
        if row is None:
            changed.append((key, "removed_on_target"))
        elif not row.joined:
            changed.append((key, row.verdict.value))
    for key in sorted(set(target_profile.keys.values()) - set(source_profile.keys.values())):
        changed.append((key, "added_on_target"))
    graph_shape_match = payload.get("graph_shape_hash") is not None and payload[
        "graph_shape_hash"
    ] == getattr(trace, "graph_shape_hash", None)
    report = CollapsePlanReapply(
        applicable=not changed,
        changed_cohorts=tuple(changed),
        collapsed_addresses=tuple(payload.get("plan_addresses", ())),
        graph_shape_match=graph_shape_match,
    )
    if strict and changed:
        raise InvalidArgumentError(
            f"the persisted collapse plan does not reapply: {len(changed)} "
            f"structural cohort(s) changed between the captures (first 5: "
            f"{changed[:5]}). The guarded join refused or one-sided them; "
            "reapplying anyway would silently fold different structure. "
            "Re-export the plan on the new capture, or pass strict=False to "
            "read the changed-set report.",
            code="collapse_plan_cohorts_changed",
            remedy="re-export on the new capture, or strict=False for the report",
        )
    return report

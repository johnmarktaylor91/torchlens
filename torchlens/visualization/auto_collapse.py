"""Smart module-collapse scoring and selection for graph rendering."""

from __future__ import annotations

import bisect
import functools
import hashlib
import math
import time
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from .._errors import InvalidArgumentError
from .._literals import CollapseLiteral, FoldRepeatsLiteral, VisModeLiteral
from ..errors._base import TorchLensWarning

# Condensed-flow-graph construction: split to _condensed_flow.py under the R43
# file-size ratchet. The dataclasses and helpers re-export here (historical
# surface); call sites below that tests monkeypatch on _condensed_flow go
# through the module attribute so the patch seam has one home.
from . import _condensed_flow
from ._collapse_disclosures import _warn_undisclosed_floor
from ._collapse_runs import (  # noqa: F401
    RUN_FOLD_MIN_LENGTH,
    _chain_connector_nodes,
    _paired_external_connector,
    _run_fold_is_chain_interval,
    _run_fold_is_legal,
    _run_fold_is_parallel_fan,
    _run_is_flow_consecutive,
)
from ._collapse_signatures import (  # noqa: F401
    _MEMORY_ADDRESS_PATTERN,
    MemberFingerprintCache,
    _exterior_bindings_consistent,
    _func_config_digest,
    _module_exterior_bindings,
    _module_structural_signature,
    _module_wiring_digest,
    _module_wiring_walk,
    _non_tensor_args_digest,
    fingerprints_for,
)
from ._condensed_flow import (  # noqa: F401
    JUNCTION_FUNC_NAMES,
    ChildCondensedFlowGraph,
    FlowIntervalFlags,
    ModuleCollapseSignals,
    _compute_child_condensed_flow_graphs,
    _condensed_owner_for_op,
    _condensed_owner_map,
    _count_landmark_edges,
    _count_passthrough_edges,
    _empty_signal,
    _flow_interval_flags,
    _module_address_stack,
    _op_func_name,
    _output_junctions,
)
from ._render_common import strict_collapse_checks_enabled
from .collapse_plan import RenderContext, collapse_plan_for_trace, count

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .collapse_optimizer import OptimizerResult


GENERIC_CONTAINER_CLASSES = frozenset({"Sequential", "ModuleList", "ModuleDict", "ParameterList"})
_COUNT_MISMATCH_WARNING_EMITTED = False


def _indexed_child_stem(name: str) -> str | None:
    """Return the stem of an indexed child name, or ``None`` when unindexed.

    Manual, linear-time equivalent of the historical
    ``^(?P<stem>.*?)(?:\\.?\\d+|_?\\d+[a-z]?)$`` regex, whose lazy stem plus
    digit-run alternation backtracked QUADRATICALLY on artifact-supplied
    names like ``"9"*n + "!!"`` (measured 26s at 40k chars). The lazy stem
    means the LONGEST valid suffix wins: trailing decimal digits reaching
    the end (optionally preceded by one ``.`` or ``_``), or reaching one
    final ``a``-``z`` letter for the underscore form. ``str.isdecimal`` is
    exactly the ``\\d`` character class. One deliberate tightening: the
    regex ``$`` also matched before a trailing newline; a newline-bearing
    name now reads as unindexed.

    Parameters
    ----------
    name:
        Leaf child name.

    Returns
    -------
    str | None
        Stem before the numeric suffix, or ``None`` for unindexed names.
    """

    end = len(name)
    suffix_starts: list[int] = []
    # Form A: optional "." + decimal digits running to the end.
    cut = end
    while cut > 0 and name[cut - 1].isdecimal():
        cut -= 1
    if cut < end:
        suffix_starts.append(cut - 1 if cut > 0 and name[cut - 1] == "." else cut)
    # Form B: optional "_" + decimal digits + optional ONE final a-z letter.
    tail = end - 1 if end and "a" <= name[end - 1] <= "z" else end
    cut = tail
    while cut > 0 and name[cut - 1].isdecimal():
        cut -= 1
    if cut < tail:
        suffix_starts.append(cut - 1 if cut > 0 and name[cut - 1] == "_" else cut)
    if not suffix_starts:
        return None
    return name[: min(suffix_starts)]


# RUN_FOLD_MIN_LENGTH moved to ._collapse_runs (re-exported above).


@dataclass(frozen=True)
class ModuleRepeatFold:
    """Consecutive structurally-identical module run selected for render folding.

    Parameters
    ----------
    representative:
        First module address in the folded run.
    addresses:
        Consecutive sibling module addresses included in the run.
    class_name:
        Module class name shared by the folded run (named in the elision label).
    num_layers:
        Aggregate recursive layer count across the run.
    num_params:
        Aggregate recursive parameter count across the run.
    num_params_trainable:
        Aggregate trainable parameter count across the run.
    num_params_frozen:
        Aggregate frozen parameter count across the run.
    shape_summary:
        Short first-to-last output-shape summary when shapes vary, else ``None``.
    hidden_member_composition:
        Metadata describing hidden members represented by the ellipsis.
    hidden_calls:
        Total forward calls made by the hidden members ``addresses[1:]``, or
        ``None`` when unknown. Multi-call members hide more forward calls than
        addresses, and the elision label must disclose that mass.
    """

    representative: str
    addresses: tuple[str, ...]
    class_name: str
    num_layers: int
    num_params: int
    num_params_trainable: int
    num_params_frozen: int
    shape_summary: str | None
    hidden_member_composition: Mapping[str, int]
    hidden_calls: int | None = None

    @property
    def multiplicity(self) -> int:
        """Return the number of folded sibling modules."""

        return len(self.addresses)


@dataclass(frozen=True)
class CollapseAnalysis:
    """Trace-local module-collapse analysis.

    Parameters
    ----------
    signals:
        Signals keyed by module address.
    scores:
        Canonical rounded scores keyed by module address.
    peer_groups:
        Structural or relaxed peer groups keyed by signature and sibling parent
        address.
    child_flow_graphs:
        Child-condensed flow graph artifacts keyed by parent module address.
    elapsed_ms:
        Signal and score computation time in milliseconds.
    """

    signals: Mapping[str, ModuleCollapseSignals]
    scores: Mapping[str, float]
    peer_groups: Mapping[tuple[str, str | None], tuple[str, ...]]
    child_flow_graphs: Mapping[str, ChildCondensedFlowGraph]
    elapsed_ms: float


_ANALYSIS_CACHE: weakref.WeakKeyDictionary[Any, tuple[tuple[object, ...], CollapseAnalysis]] = (
    weakref.WeakKeyDictionary()
)
_OP_ADJACENCY_INDEX_CACHE: weakref.WeakKeyDictionary[
    Any, tuple[tuple[object, ...], Mapping[str, str]]
] = weakref.WeakKeyDictionary()


#: Active revision scope: ``(trace, cell)`` for the public collapse call
#: currently on the stack, or ``None``; ``cell`` holds the lazily-computed
#: snapshot (empty until the first interior read). B1 hot-path fix (collapse
#: memo item 6): the honest M2a box pricing revalidates the box-units cache
#: through ``_collapse_graph_revision`` once per priced box, which alone cost
#: ~95% of a 1x150 max selection (3,786 O(ops) snapshots). One public entry
#: serves ONE snapshot by identity to every interior revalidation; selection
#: never mutates collapse-relevant trace state, so the served snapshot cannot
#: go stale inside the scope.
_ACTIVE_REVISION_SCOPE: ContextVar[tuple[Any, list[tuple[object, ...]]] | None] = ContextVar(
    "torchlens_collapse_revision_scope", default=None
)


def _revision_scoped(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Serve one shared revision snapshot for a public collapse entry point.

    Wraps a function whose first positional argument is the trace. The scope
    is LAZY: the first interior ``_collapse_graph_revision`` read computes
    the snapshot and every later read is served by identity, so decline
    paths that never fingerprint pay nothing (r8 R60-13, strengthened -- the
    historical eager guard skipped the scope whenever raw ops exceeded
    ``COLLAPSE_OPTIMIZER_MAX_OPS``, a stale mirror of the pre-U-gate
    admission rule: an ADMITTED over-raw-ceiling trace ran the whole quality
    planner unscoped, recomputing the O(ops) snapshot per priced box.
    Train-mode densenet201, 2,120 raw ops at U=713, measured 119x slower
    than eval mode this way and tripped the selection watchdog). Nested
    scoped entries on the same trace reuse the outer scope.
    """

    @functools.wraps(fn)
    def wrapper(trace: Trace, *args: Any, **kwargs: Any) -> Any:
        """Open (or reuse) the lazy revision scope around one public entry."""

        active = _ACTIVE_REVISION_SCOPE.get()
        if active is not None and active[0] is trace:
            return fn(trace, *args, **kwargs)
        token = _ACTIVE_REVISION_SCOPE.set((trace, []))
        try:
            return fn(trace, *args, **kwargs)
        finally:
            _ACTIVE_REVISION_SCOPE.reset(token)

    return wrapper


def _collapse_graph_revision(trace: Trace) -> tuple[object, ...]:
    """Return a by-value graph fingerprint for visualization cache invalidation.

    Inside an active :func:`_revision_scoped` entry the one shared snapshot
    is served by identity (computed here on the scope's first read);
    otherwise the fingerprint is recomputed.

    Parameters
    ----------
    trace:
        Trace whose mutable graph and module relations are fingerprinted.

    Returns
    -------
    tuple[object, ...]
        Stable snapshot of collapse-relevant operation and module metadata.
    """

    active = _ACTIVE_REVISION_SCOPE.get()
    if active is not None and active[0] is trace:
        cell = active[1]
        if not cell:
            cell.append(_compute_collapse_graph_revision(trace))
        return cell[0]
    return _compute_collapse_graph_revision(trace)


def _compute_collapse_graph_revision(trace: Trace) -> tuple[object, ...]:
    """Compute the uncached collapse-relevant graph fingerprint."""

    op_revision = tuple(
        (
            op.label,
            op.label_short,
            op._label_raw,
            op.layer_label,
            op.layer_label_short,
            tuple(op.parents),
            tuple(op.children),
            tuple(str(module) for module in (op.modules or ())),
            op.func_name,
            tuple(op.shape),
            op.io_role,
        )
        for op in trace.ops
    )
    module_revision = tuple(
        (
            module.address,
            getattr(module, "address_parent", None),
            tuple(getattr(module, "address_children", ()) or ()),
            getattr(module, "num_calls", None),
            getattr(module, "num_params", None),
        )
        for module in trace.modules
    )
    return (op_revision, module_revision)


def _op_adjacency_index(
    trace: Trace, revision: tuple[object, ...] | None = None
) -> Mapping[str, str]:
    """Return unambiguous relationship labels mapped to canonical Op labels.

    Parameters
    ----------
    trace:
        Trace whose operation relationships are being indexed.
    revision:
        Already-computed graph fingerprint for this probe. ``None`` computes
        it here; entry points that just fingerprinted the trace pass it in so
        validation stays one O(ops) walk per public call, not one per probe.

    Returns
    -------
    Mapping[str, str]
        Unambiguous accessor label forms mapped to canonical operation labels.
    """

    if revision is None:
        revision = _collapse_graph_revision(trace)
    cached = _OP_ADJACENCY_INDEX_CACHE.get(trace)
    # Identity first: a walk threads ONE revision object through every
    # resolve, so repeat probes within that walk are O(1), not a full
    # tuple-equality pass over the fingerprint.
    if cached is not None and (cached[0] is revision or cached[0] == revision):
        return cached[1]
    unique_ops: dict[str, Op] = {}
    ambiguous_forms: set[str] = set()
    for op in trace.ops:
        forms = (
            op.label,
            op.label_short,
            op._label_raw,
            op.raw_label,
            op.layer_label,
            op.layer_label_short,
        )
        for form in forms:
            if not form:
                continue
            existing = unique_ops.get(form)
            if existing is None:
                unique_ops[form] = op
            elif existing is not op:
                ambiguous_forms.add(form)
    index = {form: op.label for form, op in unique_ops.items() if form not in ambiguous_forms}
    _OP_ADJACENCY_INDEX_CACHE[trace] = (revision, index)
    return index


def _resolve_relationship_op(
    trace: Trace, label: str, revision: tuple[object, ...] | None = None
) -> Op:
    """Resolve a parent/child relationship label without changing accessor semantics.

    Parameters
    ----------
    trace:
        Trace that owns the operation relationship.
    label:
        Label stored in an operation's ``parents`` or ``children`` collection.
    revision:
        Graph fingerprint already computed by the calling walk. ``None``
        fingerprints here; per-edge callers must thread the walk-level
        revision or every edge pays a full O(ops) fingerprint just to probe
        the adjacency cache. The index itself still builds lazily, on the
        first resolve that actually needs it.

    Returns
    -------
    Op
        The same operation returned by the public trace accessor.
    """

    canonical_label = _op_adjacency_index(trace, revision).get(label)
    if canonical_label is None:
        return cast("Op", trace.ops[label])
    return cast("Op", trace.ops[canonical_label])


def analyze_collapse(trace: Trace) -> CollapseAnalysis:
    """Return cached module-collapse signals and canonical scores for ``trace``.

    Parameters
    ----------
    trace:
        Trace to analyze.

    Returns
    -------
    CollapseAnalysis
        Cached signal, digest, peer, and score data.
    """

    revision = _collapse_graph_revision(trace)
    cached = _ANALYSIS_CACHE.get(trace)
    if cached is not None and cached[0] == revision:
        return cached[1]
    start = time.perf_counter()
    signals_without_peers = _compute_signal_skeleton(trace, revision)
    digests = _compute_structural_digests(trace, signals_without_peers)
    peer_groups = _group_structural_peers(trace, digests)
    child_flow_graphs = _compute_child_condensed_flow_graphs(trace, signals_without_peers, revision)
    peer_count_by_address: dict[str, int] = {}
    for group in peer_groups.values():
        for address in group:
            peer_count_by_address[address] = max(peer_count_by_address.get(address, 1), len(group))
    signals = {
        address: ModuleCollapseSignals(
            address=signal.address,
            subtree_ops=signal.subtree_ops,
            own_func_names=signal.own_func_names,
            internal_edges=signal.internal_edges,
            input_edges=signal.input_edges,
            output_edges=signal.output_edges,
            landmark_edges=signal.landmark_edges,
            passthrough_edges=signal.passthrough_edges,
            output_junctions=signal.output_junctions,
            params=signal.params,
            depth=signal.depth,
            num_calls=signal.num_calls,
            structural_digest=digests[address],
            peer_count=peer_count_by_address.get(address, 1),
            hidden_ops=signal.hidden_ops,
            eligible=signal.eligible,
        )
        for address, signal in signals_without_peers.items()
    }
    scores = _signal_size_scores(signals)
    analysis = CollapseAnalysis(
        signals=signals,
        scores=scores,
        peer_groups=peer_groups,
        child_flow_graphs=child_flow_graphs,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )
    _ANALYSIS_CACHE[trace] = (revision, analysis)
    return analysis


def _child_condensed_flow_graphs(trace: Trace) -> Mapping[str, ChildCondensedFlowGraph]:
    """Return cached child-condensed flow graphs for tests and v2 downstream work.

    Parameters
    ----------
    trace:
        Trace to analyze.

    Returns
    -------
    Mapping[str, ChildCondensedFlowGraph]
        Flow graph artifacts keyed by parent module address.
    """

    return analyze_collapse(trace).child_flow_graphs


def collapse_order(
    trace: Trace,
    mode: Literal["auto", "max"] = "auto",
) -> list[tuple[str, float]]:
    """Return v2 collapse diagnostics sorted by score for a policy.

    The documented-inert ``weights=`` parameter is REMOVED (collapse memo
    D9; clean-v2 hard-rename posture: no warn shim). Scores were never
    weight-sensitive on this surface; see MIGRATIONS.md.

    Parameters
    ----------
    trace:
        Trace to rank.
    mode:
        ``"auto"`` or ``"max"`` landmark policy.

    Returns
    -------
    list[tuple[str, float]]
        ``(module_address, rounded_score)`` sorted by ``(-score, address)``.
    """

    if mode not in {"auto", "max"}:
        raise InvalidArgumentError(
            f"mode must be 'auto' or 'max'; received {mode!r}",
            code="collapse_mode_invalid",
            remedy="pass mode='auto' or 'max'",
            argument="mode",
        )
    from .collapse_optimizer import select_collapse_plan

    result = select_collapse_plan(trace, RenderContext(), mode=mode)
    if result.declined:
        # Over-ceiling decline (r8 R60-8): the old order ran the full
        # O(N*D) ``analyze_collapse`` only to zero every score afterwards.
        # An empty table is the honest degraded surface -- every consumer
        # reads through ``.get(address, 0.0)``, so the observable scores
        # are identical without the wasted analysis.
        return []
    analysis = analyze_collapse(trace)
    scores = _v2_selected_module_scores(trace, analysis, mode=mode, result=result)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _signal_size_scores(signals: Mapping[str, ModuleCollapseSignals]) -> dict[str, float]:
    """Return non-selector signal-size diagnostics for cached analysis."""

    max_hidden = max((signal.hidden_ops for signal in signals.values()), default=0)
    if max_hidden <= 0:
        return dict.fromkeys(signals, 0.0)
    return {
        address: round(signal.hidden_ops / max_hidden, 6) if signal.eligible else 0.0
        for address, signal in signals.items()
    }


def _v2_selected_module_scores(
    trace: Trace,
    analysis: CollapseAnalysis,
    *,
    mode: Literal["auto", "max"],
    result: OptimizerResult | None = None,
) -> dict[str, float]:
    """Return same-shape scores derived from the v2 selected module set."""

    from .collapse_optimizer import select_collapse_plan

    if result is None:
        result = select_collapse_plan(trace, RenderContext(), mode=mode)
    selected = result.selected if not result.declined else frozenset()
    hidden_max = max(
        (
            analysis.signals[address].hidden_ops
            for address in selected
            if address in analysis.signals
        ),
        default=0,
    )
    scores = dict.fromkeys(analysis.signals, 0.0)
    for address in selected:
        signal = analysis.signals.get(address)
        if signal is None or hidden_max <= 0:
            continue
        scores[address] = round(max(signal.hidden_ops / hidden_max, 1e-6), 6)
    return scores


def resolve_collapse_fn(
    trace: Trace,
    collapse: CollapseLiteral,
    vis_mode: VisModeLiteral,
    context: RenderContext | None = None,
) -> Callable[[Module], bool] | None:
    """Resolve a public collapse option to a renderer predicate.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse:
        Public collapse mode.
    vis_mode:
        Current visualization mode.
    context:
        Render context for v2 instrumentation. Defaults preserve v1 behavior.

    Returns
    -------
    Callable[[Module], bool] | None
        Collapse predicate, or ``None`` for ``"none"``.
    """

    resolved_context = RenderContext(vis_mode=vis_mode) if context is None else context
    if isinstance(collapse, float):
        if not 0.0 <= collapse <= 1.0:
            raise InvalidArgumentError(
                f"collapse float level must be in [0.0, 1.0]; received {collapse!r}",
                code="collapse_level_invalid",
                remedy="pass a collapse level between 0.0 and 1.0",
                argument="collapse",
            )
        if collapse == 0.0:
            return None
        from .collapse_optimizer import select_collapse_level

        result = select_collapse_level(trace, resolved_context, collapse)
        if not result.declined:

            def v2_collapse_fn(module: Module) -> bool:
                """Return whether ``module`` is selected by the v2 optimizer."""

                return module.address in result.selected

            setattr(v2_collapse_fn, "_torchlens_v2_repeat_folds", result.repeat_folds)
            setattr(v2_collapse_fn, "_torchlens_v2_segments", result.segments or {})
            setattr(v2_collapse_fn, "_torchlens_v2_plan", result.plan)
            setattr(v2_collapse_fn, "_torchlens_v2_result", result)
            setattr(v2_collapse_fn, "_torchlens_v2_mode", "level")
            return v2_collapse_fn
    if collapse == "none":
        return None
    if collapse not in {"auto", "max"}:
        raise InvalidArgumentError(
            "collapse must be 'none', 'auto', 'max', or a float in [0.0, 1.0]; "
            f"received {collapse!r}",
            code="collapse_mode_invalid",
            remedy="pass collapse='none', 'auto', 'max', or an in-range float",
            argument="collapse",
        )
    if collapse in {"auto", "max"}:
        from .collapse_optimizer import select_collapse_plan

        result = select_collapse_plan(trace, resolved_context, mode=collapse)
        if not result.declined:
            _warn_undisclosed_floor(trace, collapse, result, resolved_context)

            def v2_collapse_fn(module: Module) -> bool:
                """Return whether ``module`` is selected by the v2 optimizer."""

                return module.address in result.selected

            setattr(v2_collapse_fn, "_torchlens_v2_repeat_folds", result.repeat_folds)
            setattr(v2_collapse_fn, "_torchlens_v2_segments", result.segments or {})
            setattr(v2_collapse_fn, "_torchlens_v2_plan", result.plan)
            setattr(v2_collapse_fn, "_torchlens_v2_result", result)
            setattr(v2_collapse_fn, "_torchlens_v2_mode", collapse)
            return v2_collapse_fn
    return None


@_revision_scoped
def resolve_repeat_folds(
    trace: Trace,
    collapse_fn: Callable[[Module], bool] | None,
    context: RenderContext | None = None,
    fold_repeats: FoldRepeatsLiteral = None,
) -> dict[str, ModuleRepeatFold]:
    """Return render-time folds for consecutive collapsed sibling runs.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate. ``None`` disables run folding.
    context:
        Render context for v2 instrumentation. Defaults preserve v1 behavior.
    fold_repeats:
        Repeat-fold policy override. ``None`` preserves the current band-gated
        policy, ``True`` folds every eligible repeated run, and ``False``
        disables run folding.

    Returns
    -------
    dict[str, ModuleRepeatFold]
        Mapping from each folded module address to its run descriptor.
    """

    resolved_context = RenderContext() if context is None else context
    if fold_repeats not in {None, True, False}:
        raise InvalidArgumentError(
            f"fold_repeats must be None, True, or False; received {fold_repeats!r}",
            code="fold_repeats_invalid",
            remedy="pass fold_repeats=None, True, or False",
            argument="fold_repeats",
        )
    if fold_repeats is False:
        return {}
    if collapse_fn is None and fold_repeats is not True:
        return {}
    from .collapse_optimizer import COLLAPSE_OPTIMIZER_MAX_OPS

    if len(trace.ops) > COLLAPSE_OPTIMIZER_MAX_OPS:
        # Compute-ceiling parity (r8 R60-9): ``draw(fold_repeats=True)`` with
        # ``collapse="none"`` entered run folding directly -- full
        # ``analyze_collapse`` plus uncached per-candidate structural digests
        # and three extra plan builds -- without ever consulting the one
        # op-count ceiling the collapse engine has. Decline DISCLOSED, same
        # policy as the optimizer: the graph renders without run folds.
        warnings.warn(
            f"TorchLens is skipping repeat-run folding: this trace has "
            f"{len(trace.ops)} ops, above the collapse engine's compute "
            f"ceiling COLLAPSE_OPTIMIZER_MAX_OPS={COLLAPSE_OPTIMIZER_MAX_OPS}. "
            "The graph renders without folds; reduce the rendered graph "
            "first with module= focus, vis_call_depth, or rolled mode.",
            TorchLensWarning,
            stacklevel=2,
        )
        return {}
    eligibility_collapse_fn = collapse_fn if collapse_fn is not None else _always_collapse_module
    render_collapse_fn = collapse_fn
    v2_repeat_folds = getattr(collapse_fn, "_torchlens_v2_repeat_folds", None)
    if fold_repeats is True and v2_repeat_folds is not None:
        # Explicit ``fold_repeats=True`` folds eligible repeated runs even
        # inside the readable band (the documented "even with
        # collapse='none'" contract). A MODE-resolved selection (v2-marked
        # fn) is a presentation choice, not a fold scope: the D8 ladder can
        # legally choose a near-full auto point whose selection would
        # otherwise silently veto the explicit fold request. An explicit
        # user collapse_fn (no marker) keeps scoping discovery to the
        # user's modules.
        eligibility_collapse_fn = _always_collapse_module
    if fold_repeats is None and v2_repeat_folds is not None:
        return dict(v2_repeat_folds)
    projected_count = count(
        collapse_plan_for_trace(trace, render_collapse_fn, None, resolved_context)
    )
    if fold_repeats is None and projected_count <= _readable_band_high(trace):
        _assert_plan_count(
            trace,
            render_collapse_fn,
            None,
            resolved_context,
            projected_count,
        )
        return {}
    hidden_member_contributions = _run_fold_hidden_member_contributions(
        trace,
        render_collapse_fn,
        resolved_context,
    )
    analysis = analyze_collapse(trace)
    # B1 (collapse memo item 6): one fingerprint memo per analysis revision,
    # shared by every uniformity check in this sweep.
    fingerprints = fingerprints_for(trace, analysis)
    candidate_folds: list[ModuleRepeatFold] = []
    candidate_addresses: set[str] = set()
    # One selected-module index per discovery sweep (r8 R29): the per-group
    # rebuild inside the iterators evaluated collapse_fn over every module
    # once per sibling group -- Theta(M^2) predicate calls on module-heavy
    # models.
    sweep_selected_index = _selected_address_index(trace, eligibility_collapse_fn)
    for parent_address, child_addresses in _sibling_address_groups(trace).items():
        graph = _flow_graph_for_sibling_group(
            trace,
            str(parent_address),
            child_addresses,
            analysis,
        )
        flow_addresses = _flow_ordered_child_addresses(child_addresses, graph)
        for run in _iter_collapsible_runs(
            trace, flow_addresses, eligibility_collapse_fn, fingerprints=fingerprints
        ):
            if not _run_fold_is_legal(run, graph):
                continue
            if not _run_fold_members_uniform(trace, run, fingerprints):
                continue
            fold = _make_run_fold(trace, run)
            candidate_folds.append(fold)
            candidate_addresses.update(run)
        for run in _iter_collapsible_child_path_runs(
            trace,
            flow_addresses,
            eligibility_collapse_fn,
            selected_index=sweep_selected_index,
            fingerprints=fingerprints,
        ):
            if any(address in candidate_addresses for address in run):
                continue
            run_parent = _common_parent_address(run)
            run_graph = _flow_graph_for_sibling_group(
                trace,
                str(run_parent),
                list(run),
                analysis,
            )
            if not _run_fold_is_legal(run, run_graph):
                continue
            if not _run_fold_members_uniform(trace, run, fingerprints):
                continue
            fold = _make_run_fold(trace, run)
            candidate_folds.append(fold)
            candidate_addresses.update(run)
        for run in _iter_collapsible_runs(
            trace,
            flow_addresses,
            eligibility_collapse_fn,
            allow_selected_descendant=True,
            selected_index=sweep_selected_index,
            fingerprints=fingerprints,
        ):
            if any(address in candidate_addresses for address in run):
                continue
            if not _run_fold_is_legal(run, graph):
                continue
            if not _run_fold_members_uniform(trace, run, fingerprints):
                continue
            fold = _make_run_fold(trace, run)
            candidate_folds.append(fold)
            candidate_addresses.update(run)
    folds_by_address: dict[str, ModuleRepeatFold] = {}
    for fold in sorted(candidate_folds, key=lambda item: (-item.multiplicity, item.representative)):
        if any(address in folds_by_address for address in fold.addresses):
            continue
        for address in fold.addresses:
            folds_by_address[address] = fold
        projected_count += _run_fold_delta(fold, hidden_member_contributions)
        if fold_repeats is None and projected_count <= _readable_band_high(trace):
            break
    if fold_repeats is True:
        projected_count = count(
            collapse_plan_for_trace(trace, render_collapse_fn, folds_by_address, resolved_context)
        )
    _assert_plan_count(
        trace,
        render_collapse_fn,
        folds_by_address,
        resolved_context,
        projected_count,
    )
    return folds_by_address


def _always_collapse_module(module: Module) -> bool:
    """Return ``True`` for standalone repeat-fold eligibility checks.

    Parameters
    ----------
    module:
        Module being considered.

    Returns
    -------
    bool
        Always ``True``.
    """

    _ = module
    return True


def _sibling_address_groups(trace: Trace) -> dict[str | None, list[str]]:
    """Return ordered sibling module addresses grouped by parent address.

    Parameters
    ----------
    trace:
        Trace owning the module hierarchy.

    Returns
    -------
    dict[str | None, list[str]]
        Module addresses keyed by their recorded parent address.
    """

    groups: dict[str | None, list[str]] = defaultdict(list)
    for module in trace.modules:
        if module.address == "self":
            continue
        groups[getattr(module, "address_parent", None)].append(module.address)
    return groups


def _flow_ordered_child_addresses(
    child_addresses: list[str],
    graph: ChildCondensedFlowGraph | None,
) -> list[str]:
    """Return child addresses in flow order with deterministic fallback.

    Parameters
    ----------
    child_addresses:
        Recorded sibling addresses.
    graph:
        Optional child-condensed flow graph for the sibling parent.

    Returns
    -------
    list[str]
        Sibling addresses ordered by first-op flow position when available.
    """

    if graph is None:
        return list(child_addresses)
    seen = set(graph.flow_children)
    ordered = [address for address in graph.flow_children if address in child_addresses]
    ordered.extend(address for address in child_addresses if address not in seen)
    return ordered


def _flow_graph_for_sibling_group(
    trace: Trace,
    parent_address: str,
    child_addresses: list[str],
    analysis: CollapseAnalysis,
) -> ChildCondensedFlowGraph | None:
    """Return the child-flow graph for a concrete or synthetic sibling scope.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    parent_address:
        Recorded parent address for the sibling group.
    child_addresses:
        Sibling child addresses.
    analysis:
        Cached collapse analysis for the trace.

    Returns
    -------
    ChildCondensedFlowGraph | None
        Precomputed graph when available, otherwise a graph synthesized for
        recorded non-module scopes such as ``ModuleList`` containers.
    """

    graph = analysis.child_flow_graphs.get(parent_address)
    if graph is not None:
        return graph
    if not child_addresses:
        return None
    return _synthetic_child_condensed_flow_graph(
        trace,
        parent_address,
        child_addresses,
        analysis.signals,
    )


def _synthetic_child_condensed_flow_graph(
    trace: Trace,
    parent_address: str,
    child_addresses: list[str],
    signals: Mapping[str, ModuleCollapseSignals],
) -> ChildCondensedFlowGraph:
    """Build a child-condensed graph for recorded non-module sibling scopes.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    parent_address:
        Synthetic parent address.
    child_addresses:
        Direct child addresses in the synthetic scope.
    signals:
        Precomputed module signals.

    Returns
    -------
    ChildCondensedFlowGraph
        Flow graph with external source/sink sentinels for boundary edges.
    """

    op_order = {op.label: index for index, op in enumerate(trace.ops)}
    child_sets = {
        child: set(signals[child].subtree_ops) for child in child_addresses if child in signals
    }
    flow_children = tuple(
        sorted(
            child_sets,
            key=lambda child: (
                _condensed_flow._first_flow_op_order(trace, child_sets[child], op_order),
                child,
            ),
        )
    )
    owner_by_label: dict[str, str] = {}
    for child_address, labels in child_sets.items():
        for label in labels:
            owner_by_label[label] = child_address
    edges: set[tuple[str, str]] = set()
    # Reachable outside analyze_collapse (optimizer synthetic scopes), so this
    # walk fingerprints once here rather than per edge.
    revision = _collapse_graph_revision(trace)
    for op in trace.ops:
        source = owner_by_label.get(op.label)
        for child_label in getattr(op, "children", ()) or ():
            child_op = _resolve_relationship_op(trace, child_label, revision)
            target_label = child_op.label
            if not _condensed_flow._is_forward_dataflow_edge(trace, op.label, target_label):
                continue
            target = owner_by_label.get(target_label)
            if source is None and target is None:
                continue
            if source is None:
                if target is None:
                    continue
                edges.add((f"external_source:{op.label}", target))
            elif target is None:
                edges.add((source, f"external_sink:{target_label}"))
            elif target != source:
                edges.add((source, target))
    ordered_nodes = (
        *flow_children,
        *sorted({node for edge in edges for node in edge if ":" in node}),
    )
    ordered = {node: index for index, node in enumerate(ordered_nodes)}
    sorted_edges = tuple(
        sorted(
            edges, key=lambda edge: (ordered.get(edge[0], 10**9), ordered.get(edge[1], 10**9), edge)
        )
    )
    return ChildCondensedFlowGraph(
        parent=parent_address,
        flow_children=flow_children,
        parent_owned_ops=(),
        nodes=ordered_nodes,
        edges=sorted_edges,
        child_external_endpoint_counts=_condensed_flow._child_external_endpoint_counts(
            sorted_edges, flow_children
        ),
        interval_flags=_condensed_flow._flow_interval_flags(
            trace, flow_children, child_sets, sorted_edges
        ),
    )


def module_collapse_score(module: Module) -> float:
    """Return the canonical default collapse score for a module.

    Parameters
    ----------
    module:
        Module metadata entry.

    Returns
    -------
    float
        Rounded canonical score, or ``0.0`` for ineligible/unbound modules.
    """

    trace = module.trace
    if trace is None:
        return 0.0
    return dict(collapse_order(trace)).get(module.address, 0.0)


# The structural-fingerprint family moved to ._collapse_signatures with
# the B1 fingerprint memo cache (collapse memo item 6); the names
# re-export from this module's imports for existing readers.


def _run_fold_members_uniform(
    trace: Trace,
    addresses: Sequence[str],
    fingerprints: MemberFingerprintCache | None = None,
) -> bool:
    """Return whether every run member shares one structural signature.

    A run fold renders the first member (``addresses[0]``) as a visible
    representative box and elides the rest behind a ``... +N more <class>``
    ellipsis. That ellipsis claims the hidden members are interchangeable
    with the visible representative, so the fold is only honest when EVERY
    member — representative included — has the same structural fingerprint
    (:func:`_module_structural_signature`).

    T9 (grind-p3, HIGH): the comparison previously spanned only
    ``addresses[1:]`` on the theory that the representative's own stats stay
    visible. That left the reverse direction unproven: a plateau uniformly
    different from its representative (e.g. every hidden block swapping ReLU
    for Tanh) folded anyway, and the hidden structure appeared NOWHERE in
    the render — two different models drew byte-identical DOT. Requiring
    the representative to match closes that hole; a run with an odd first
    member splits (both run assemblers retry shorter sub-runs), so the
    plateau re-folds from its own structurally-matching representative.
    The former separate trainability screen is subsumed: exact trainable and
    frozen parameter counts are components of the structural fingerprint.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Folded run addresses, representative first.
    fingerprints:
        Optional shared B1 fingerprint memo (collapse memo item 6). ``None``
        builds a call-local cache, which still shares one wiring walk per
        member between the signature and bindings reads.

    Returns
    -------
    bool
        Whether all members are structurally uniform.
    """

    if len(addresses) <= 1:
        return True
    cache = MemberFingerprintCache(trace) if fingerprints is None else fingerprints
    signatures = {cache.signature(address) for address in addresses}
    if len(signatures) != 1:
        return False
    # r4 b6-sol R19-1: equal per-member signatures are not enough when the
    # members SHARE exterior sources — the shared source must occupy the
    # same operand slot in every member (a - b vs b - a must never fold).
    merged: dict[str, int] = {}
    return all(
        _exterior_bindings_consistent(merged, cache.bindings(address)) for address in addresses
    )


def _split_run_by_member_uniformity(
    trace: Trace,
    run: tuple[str, ...],
    fingerprints: MemberFingerprintCache | None = None,
) -> Iterator[tuple[str, ...]]:
    """Split one grouped run into its maximal member-uniform sub-runs.

    :func:`_iter_collapsible_runs` groups addresses into a run purely by
    class, stem, and flow/shape adjacency -- that grouping says nothing
    about whether the fold members are structurally uniform. Without this
    retry, a single structurally-odd module anywhere inside an
    otherwise-eligible run would cause the caller to reject the *entire*
    run wholesale the moment :func:`_run_fold_members_uniform` failed on
    it, even though the legal sub-runs on either side of the odd member are
    still independently foldable.

    This mirrors :func:`collapse_optimizer._maximal_legal_runs`'s
    retry-shorter-subrun approach: grow a candidate window from each
    unconsumed starting position, keep the longest member-uniform prefix,
    emit it, and resume scanning from the next unconsumed address. A
    structurally-odd module stays visible on its own (every fold member,
    representative included, must match the fingerprint) instead of
    silently sinking every run it happens to sit inside.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    run:
        One flow-consecutive, same-class/stem/shape run as assembled by
        :func:`_iter_collapsible_runs`.
    fingerprints:
        Optional shared B1 fingerprint memo (collapse memo item 6). ``None``
        builds a call-local cache for this run's signature/bindings reads.

    Yields
    ------
    tuple[str, ...]
        Maximal sub-runs of at least :data:`RUN_FOLD_MIN_LENGTH` addresses,
        each with structurally uniform members.
    """

    # Uniformity is "all members share ONE structural signature", so the
    # maximal uniform window starting at any index is exactly the run of
    # consecutive equal signatures from that index. Computing one signature
    # per member and grouping equal neighbours is output-identical to the
    # historical grow-every-window scan, which recomputed signatures for
    # every (start, end) pair — near-cubic on long runs (b6/b4-sol
    # instrumented) — while this is linear in run length.
    total = len(run)
    if total < RUN_FOLD_MIN_LENGTH:
        return
    cache = MemberFingerprintCache(trace) if fingerprints is None else fingerprints
    signatures = [cache.signature(address) for address in run]
    bindings = [cache.bindings(address) for address in run]
    index = 0
    while index < total:
        end = index + 1
        # r4 b6-sol R19-1: a window member must both share the signature AND
        # bind any exterior source it shares with earlier window members to
        # the same operand slot (see _exterior_bindings_consistent).
        merged: dict[str, int] = {}
        if not _exterior_bindings_consistent(merged, bindings[index]):
            index = end
            continue
        while (
            end < total
            and signatures[end] == signatures[index]
            and _exterior_bindings_consistent(merged, bindings[end])
        ):
            end += 1
        if end - index >= RUN_FOLD_MIN_LENGTH:
            yield run[index:end]
        # Windows inside a shorter-than-minimum equal-signature block can
        # never reach the minimum length, so skipping the whole block is
        # NOT always output-identical to the historical index += 1 rescan
        # once binding consistency joins the constraint: a member rejected
        # for a binding conflict can open its own consistent window, and
        # `end` stopped exactly at the first such member.
        index = end


def _iter_collapsible_runs(
    trace: Trace,
    child_addresses: list[str],
    collapse_fn: Callable[[Module], bool],
    run_stem: str | None = None,
    allow_selected_descendant: bool = False,
    selected_index: tuple[str, ...] | None = None,
    fingerprints: MemberFingerprintCache | None = None,
) -> Iterator[tuple[str, ...]]:
    """Yield flow-consecutive same-class runs with equal adjacent output shapes.

    Each assembled group is further split by
    :func:`_split_run_by_member_uniformity` into its maximal member-uniform
    sub-runs before being yielded, so a single structurally-odd module
    anywhere inside an otherwise-eligible run only knocks out the sub-run(s)
    that would have hidden it -- the legal sub-runs on either side still
    fold, matching the "folds every eligible repeated run" contract that the
    default v2 engine (:func:`collapse_optimizer._maximal_legal_runs`)
    already honors.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    child_addresses:
        Direct children for one parent module in flow order.
    collapse_fn:
        Active collapse predicate.
    run_stem:
        Optional precomputed sibling-run stem for descendant-path folds.
    allow_selected_descendant:
        Whether selected descendants allow a sibling ancestor to stand in as
        the folded member.
    selected_index:
        Optional precomputed selected-address index shared across sibling
        groups (r8 R29); ``None`` rebuilds it for this group.
    fingerprints:
        Optional shared B1 fingerprint memo (collapse memo item 6), threaded
        through to :func:`_split_run_by_member_uniformity` so one discovery
        sweep prices each member once; ``None`` builds a cache per split.

    Yields
    ------
    tuple[str, ...]
        One run of at least :data:`RUN_FOLD_MIN_LENGTH` addresses, with
        structurally uniform members.
    """

    # One shared sorted index instead of a full trace.modules scan per child:
    # the per-child rescans made descendant-aware discovery Theta(S*M) in
    # sibling count x module count (hunt-6 R29-3). r8 R29: callers walking
    # MANY sibling groups pass the index in, so it builds once per discovery
    # sweep instead of once per group (the cross-group rebuild was Theta(M^2)
    # in module count).
    if selected_index is None and allow_selected_descendant:
        selected_index = _selected_address_index(trace, collapse_fn)
    if not allow_selected_descendant:
        selected_index = None
    current_key: tuple[str, str] | None = None
    current_descendant_only_num_layers: int | None = None
    current_has_direct_selection = False
    current_run: list[str] = []
    for address in child_addresses:
        module = cast("Module", trace.modules[address])
        directly_selected = collapse_fn(module)
        descendant_selected = selected_index is not None and bool(
            _selected_descendants_in_index(selected_index, address)
        )
        selected = directly_selected or descendant_selected
        if not selected:
            if allow_selected_descendant:
                continue
            yield from _split_run_by_member_uniformity(trace, tuple(current_run), fingerprints)
            current_key = None
            current_descendant_only_num_layers = None
            current_has_direct_selection = False
            current_run = []
            continue
        key = (
            str(getattr(module, "class_name", "")),
            run_stem or _indexed_parent_stem(address),
        )
        num_layers = int(getattr(module, "num_layers", 0) or 0)
        descendant_only_depth_matches = (
            directly_selected
            or current_has_direct_selection
            or current_descendant_only_num_layers in {None, num_layers}
        )
        extends_current = (
            key == current_key
            and bool(current_run)
            and _module_output_shapes_equal(trace, current_run[-1], address)
            and descendant_only_depth_matches
        )
        if extends_current:
            current_run.append(address)
            current_has_direct_selection = current_has_direct_selection or directly_selected
            if not current_has_direct_selection:
                current_descendant_only_num_layers = num_layers
            continue
        yield from _split_run_by_member_uniformity(trace, tuple(current_run), fingerprints)
        current_key = key
        current_descendant_only_num_layers = None if directly_selected else num_layers
        current_has_direct_selection = directly_selected
        current_run = [address]
    yield from _split_run_by_member_uniformity(trace, tuple(current_run), fingerprints)


def _iter_collapsible_child_path_runs(
    trace: Trace,
    sibling_addresses: list[str],
    collapse_fn: Callable[[Module], bool],
    selected_index: tuple[str, ...] | None = None,
    fingerprints: MemberFingerprintCache | None = None,
) -> Iterator[tuple[str, ...]]:
    """Yield repeated selected child paths under consecutive sibling parents.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    sibling_addresses:
        Ordered direct children for one parent module.
    collapse_fn:
        Active collapse predicate.
    selected_index:
        Optional precomputed selected-address index shared across sibling
        groups (r8 R29); ``None`` rebuilds it for this group.
    fingerprints:
        Optional shared B1 fingerprint memo (collapse memo item 6), forwarded
        to the per-stem :func:`_iter_collapsible_runs` sweeps; ``None`` builds
        a cache per split.

    Yields
    ------
    tuple[str, ...]
        One run of selected descendant modules sharing the same relative path.
    """

    # Shared index: the former per-sibling _selected_descendants call scanned
    # the whole module table once per sibling (hunt-6 R29-3). r8 R29: callers
    # walking many sibling groups pass it in (once per sweep, not per group).
    if selected_index is None:
        selected_index = _selected_address_index(trace, collapse_fn)
    relative_paths = sorted(
        {
            selected_address.removeprefix(f"{sibling}.")
            for sibling in sibling_addresses
            for selected_address in _selected_descendants_in_index(selected_index, sibling)
        }
    )
    for relative_path in relative_paths:
        candidate_addresses = [
            f"{sibling}.{relative_path}" if f"{sibling}.{relative_path}" in trace.modules else ""
            for sibling in sibling_addresses
        ]
        current_stem: str | None = None
        current_candidates: list[str] = []
        for sibling, candidate_address in zip(
            sibling_addresses,
            candidate_addresses,
            strict=True,
        ):
            stem = _indexed_parent_stem(sibling)
            if stem == current_stem:
                if candidate_address:
                    current_candidates.append(candidate_address)
                continue
            if current_candidates:
                yield from _iter_collapsible_runs(
                    trace,
                    current_candidates,
                    collapse_fn,
                    current_stem,
                    fingerprints=fingerprints,
                )
            current_stem = stem
            current_candidates = [candidate_address] if candidate_address else []
        if current_candidates:
            yield from _iter_collapsible_runs(
                trace,
                current_candidates,
                collapse_fn,
                current_stem,
                fingerprints=fingerprints,
            )


def _indexed_parent_stem(address: str) -> str:
    """Return a stem that keeps long indexed sibling runs together.

    Parameters
    ----------
    address:
        Module address.

    Returns
    -------
    str
        Address stem before a trailing numeric component or suffix.
    """

    if "." in address:
        parent, name = address.rsplit(".", 1)
    else:
        parent, name = "", address
    indexed_stem = _indexed_child_stem(name)
    if indexed_stem is None:
        stem = name
    else:
        stem = indexed_stem.rstrip("._") or ""
    return f"{parent}.{stem}" if parent and stem else parent or stem or name


def _common_parent_address(addresses: tuple[str, ...]) -> str | None:
    """Return the shared parent address for a run.

    Parameters
    ----------
    addresses:
        Candidate module addresses.

    Returns
    -------
    str | None
        Shared parent address, or ``None`` when the run is empty or mixed.
    """

    parents = {address.rsplit(".", 1)[0] if "." in address else "self" for address in addresses}
    if len(parents) != 1:
        return None
    return next(iter(parents))


def _module_output_shapes_equal(trace: Trace, left: str, right: str) -> bool:
    """Return whether two modules have exactly equal known output shapes.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    left:
        First module address.
    right:
        Second module address.

    Returns
    -------
    bool
        True only when both primary output shapes are known and all dimensions
        match exactly.
    """

    left_shape = _module_output_shape_tuple(trace, left)
    right_shape = _module_output_shape_tuple(trace, right)
    return left_shape is not None and left_shape == right_shape


# The run-fold legality grammar (_run_fold_is_legal and friends) moved to
# ._collapse_runs with the B2 indexed-adjacency rewrite (collapse memo
# item 7); the names re-export from this module's imports for existing
# readers.


def _selected_address_index(
    trace: Trace,
    collapse_fn: Callable[[Module], bool],
) -> tuple[str, ...]:
    """Return the sorted addresses of every module selected by ``collapse_fn``.

    One pass over the module table shared by all descendant lookups in a
    discovery sweep; the former per-child/per-sibling full scans made
    repeat-fold discovery Theta(S*M) (hunt-6 R29-3).

    Parameters
    ----------
    trace:
        Trace owning the modules.
    collapse_fn:
        Active collapse predicate.

    Returns
    -------
    tuple[str, ...]
        Selected module addresses in lexical order.
    """

    return tuple(sorted(module.address for module in trace.modules if collapse_fn(module)))


def _selected_descendants_in_index(
    selected_index: tuple[str, ...],
    address: str,
) -> tuple[str, ...]:
    """Return the selected addresses strictly under ``address``.

    Parameters
    ----------
    selected_index:
        Sorted selected addresses from :func:`_selected_address_index`.
    address:
        Parent module address.

    Returns
    -------
    tuple[str, ...]
        Selected descendant addresses in lexical order.
    """

    prefix = f"{address}."
    start = bisect.bisect_left(selected_index, prefix)
    end = start
    while end < len(selected_index) and selected_index[end].startswith(prefix):
        end += 1
    return selected_index[start:end]


def _make_run_fold(trace: Trace, addresses: tuple[str, ...]) -> ModuleRepeatFold:
    """Build aggregate metadata for one folded run.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Consecutive sibling addresses in the run.

    Returns
    -------
    ModuleRepeatFold
        Aggregate repeat-fold descriptor.
    """

    modules = [cast("Module", trace.modules[address]) for address in addresses]
    return ModuleRepeatFold(
        representative=addresses[0],
        addresses=addresses,
        class_name=str(getattr(modules[0], "class_name", "") or "blocks"),
        num_layers=sum(int(getattr(module, "num_layers", 0) or 0) for module in modules),
        num_params=sum(int(getattr(module, "num_params", 0) or 0) for module in modules),
        num_params_trainable=sum(
            int(getattr(module, "num_params_trainable", 0) or 0) for module in modules
        ),
        num_params_frozen=sum(
            int(getattr(module, "num_params_frozen", 0) or 0) for module in modules
        ),
        shape_summary=_run_shape_summary(trace, addresses),
        hidden_member_composition=_hidden_member_composition(trace, addresses),
        hidden_calls=sum(int(getattr(module, "num_calls", 1) or 1) for module in modules[1:]),
    )


def _hidden_member_composition(trace: Trace, addresses: tuple[str, ...]) -> Mapping[str, int]:
    """Return residual/passthrough composition for hidden run members.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Folded run addresses, including the representative.

    Returns
    -------
    Mapping[str, int]
        Counts for hidden members with and without residual or join-style
        passthrough operations.
    """

    analysis = analyze_collapse(trace)
    composition = {
        "hidden_with_residual_join": 0,
        "hidden_without_residual_join": 0,
    }
    for address in addresses[1:]:
        signal = analysis.signals.get(address)
        has_join = signal is not None and (
            signal.passthrough_edges > 0
            or any(
                _op_func_name(cast("Op", trace.ops[label])) in JUNCTION_FUNC_NAMES
                for label in signal.subtree_ops
            )
        )
        key = "hidden_with_residual_join" if has_join else "hidden_without_residual_join"
        composition[key] += 1
    return composition


def _run_shape_summary(trace: Trace, addresses: tuple[str, ...]) -> str | None:
    """Return a compact first-to-last output shape summary for a folded run.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Consecutive sibling addresses in the run.

    Returns
    -------
    str | None
        Shape summary when first and last output shapes differ, else ``None``.
    """

    shapes = [_module_output_shape(trace, address) for address in addresses]
    first = shapes[0]
    last = shapes[-1]
    if first is None or last is None or first == last:
        return None
    return f"{first}->{last}"


def _module_output_shape_tuple(trace: Trace, address: str) -> tuple[int, ...] | None:
    """Return the primary output shape tuple for a module address.

    Parameters
    ----------
    trace:
        Trace owning the module.
    address:
        Pass-free module address.

    Returns
    -------
    tuple[int, ...] | None
        Output shape as integers, or ``None`` when unavailable.
    """

    module_output_layer = _module_call_output_op(trace, f"{address}:1")
    if module_output_layer is None:
        return None
    shape = getattr(module_output_layer, "shape", None)
    if shape is None:
        shape = getattr(module_output_layer, "out_shape", None)
    if not shape:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _module_call_output_op(trace: Trace, call_label: str) -> Op | None:
    """Return the primary output Op for a module-call label.

    Parameters
    ----------
    trace:
        Trace owning the module call.
    call_label:
        Pass-qualified module-call label such as ``"encoder:1"``.

    Returns
    -------
    Op | None
        Last output op for the module call, or ``None`` when unavailable.
    """

    try:
        module_call = trace.module_calls[call_label]
    except (KeyError, IndexError):
        return None
    if not module_call.output_ops:
        return None
    try:
        return cast("Op", trace.ops[module_call.output_ops[-1]])
    except (KeyError, IndexError):
        return None


def _shape_spatial_dims(shape: tuple[int, ...]) -> tuple[int, ...] | None:
    """Return spatial dimensions for common image/video tensor shapes.

    Parameters
    ----------
    shape:
        Output tensor shape.

    Returns
    -------
    tuple[int, ...] | None
        Spatial dimensions, or ``None`` for non-spatial ranks.
    """

    if len(shape) == 4:
        return shape[2:]
    if len(shape) == 5:
        return shape[2:]
    return None


def _shape_channel_dim(shape: tuple[int, ...]) -> int | None:
    """Return the channel-like dimension for common tensor shapes.

    Parameters
    ----------
    shape:
        Output tensor shape.

    Returns
    -------
    int | None
        Channel dimension, or ``None`` when no stable convention applies.
    """

    if len(shape) in {2, 4, 5}:
        return shape[1]
    if len(shape) == 3:
        return shape[2]
    return None


def _module_output_shape(trace: Trace, address: str) -> str | None:
    """Return the primary output shape string for a module address.

    Parameters
    ----------
    trace:
        Trace owning the module.
    address:
        Pass-free module address.

    Returns
    -------
    str | None
        Formatted shape string, or ``None`` when unavailable.
    """

    shape = _module_output_shape_tuple(trace, address)
    if shape is None:
        return None
    return str(tuple(shape))


def _compute_signal_skeleton(
    trace: Trace, revision: tuple[object, ...]
) -> dict[str, ModuleCollapseSignals]:
    """Compute all non-peer module signals in one shared traversal."""

    op_labels_by_module: dict[str, list[str]] = defaultdict(list)
    own_func_names_by_module: dict[str, list[str]] = defaultdict(list)
    internal_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    input_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    output_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)

    ops = list(trace.ops)
    op_by_label = {op.label: op for op in ops}
    stack_by_label = {op.label: _module_address_stack(op) for op in ops}
    labels_by_stack: dict[tuple[str, ...], list[str]] = defaultdict(list)

    for op in ops:
        stack = stack_by_label[op.label]
        labels_by_stack[stack].append(op.label)
        if stack:
            own_func_names_by_module[stack[-1]].append(_op_func_name(op))

    # Most ops share an enclosing module stack. Expand each distinct stack once,
    # rather than repeating the same ancestry walk for every op in the module.
    for stack, labels in labels_by_stack.items():
        for address in stack:
            op_labels_by_module[address].extend(labels)

    stack_sets = {stack: frozenset(stack) for stack in labels_by_stack}

    for parent in ops:
        parent_stack = stack_by_label[parent.label]
        parent_set = stack_sets.setdefault(parent_stack, frozenset(parent_stack))
        for child_label in parent.children:
            child = op_by_label.get(child_label)
            if child is None:
                child = _resolve_relationship_op(trace, child_label, revision)
                op_by_label[child_label] = child
                op_by_label[child.label] = child
                stack_by_label[child.label] = _module_address_stack(child)
            child_stack = stack_by_label[child.label]
            child_set = stack_sets.setdefault(child_stack, frozenset(child_stack))
            edge = (parent.label, child.label)
            for address in parent_set & child_set:
                internal_edges[address].add(edge)
            for address in child_set - parent_set:
                input_edges[address].add(edge)
            for address in parent_set - child_set:
                output_edges[address].add(edge)

    signals: dict[str, ModuleCollapseSignals] = {}
    for module in trace.modules:
        address = module.address
        subtree_ops = tuple(dict.fromkeys(op_labels_by_module.get(address, ())))
        hidden_ops = max(len(subtree_ops) - 1, 0)
        signals[address] = ModuleCollapseSignals(
            address=address,
            subtree_ops=subtree_ops,
            own_func_names=tuple(own_func_names_by_module.get(address, ())),
            internal_edges=len(internal_edges.get(address, ())),
            input_edges=len(input_edges.get(address, ())),
            output_edges=len(output_edges.get(address, ())),
            landmark_edges=_count_landmark_edges(
                trace,
                module,
                subtree_ops,
                input_edges.get(address, set()) | output_edges.get(address, set()),
                revision,
            ),
            passthrough_edges=_count_passthrough_edges(trace, module, subtree_ops, revision),
            output_junctions=_output_junctions(
                trace,
                module,
                subtree_ops,
                output_edges.get(address, set()),
            ),
            params=int(getattr(module, "num_params", 0) or 0),
            depth=int(getattr(module, "address_depth", 0) or 0),
            num_calls=int(getattr(module, "num_calls", 1) or 1),
            structural_digest="",
            peer_count=1,
            hidden_ops=hidden_ops,
            eligible=_gate_module(module, hidden_ops, signals),
        )
    return signals


def _gate_module(
    module: Module,
    hidden_ops: int,
    partial_signals: Mapping[str, ModuleCollapseSignals],
) -> bool:
    """Return whether a module mirrors renderer collapse eligibility."""

    if module.address in {"", "self"}:
        return False
    if int(getattr(module, "num_layers", 0) or 0) <= 1:
        return False
    child_addresses = list(getattr(module, "address_children", ()) or ())
    if len(child_addresses) == 1:
        child_signal = partial_signals.get(child_addresses[0])
        if child_signal is not None and child_signal.hidden_ops == hidden_ops:
            return False
    return True


def _compute_structural_digests(
    trace: Trace,
    signals: Mapping[str, ModuleCollapseSignals],
) -> dict[str, str]:
    """Compute structural digests bottom-up for every module."""

    digests: dict[str, str] = {}
    modules = sorted(trace.modules, key=lambda module: module.address_depth, reverse=True)
    for module in modules:
        signal = signals[module.address]
        child_sigs = tuple(
            digests[child_address]
            for child_address in getattr(module, "address_children", ()) or ()
            if child_address in digests
        )
        payload = repr(
            (
                getattr(module, "class_name", ""),
                signal.own_func_names,
                child_sigs,
                round(math.log10(1 + max(signal.params, 0))),
            )
        ).encode("utf-8")
        digests[module.address] = hashlib.sha1(payload).hexdigest()
    return digests


def _group_structural_peers(
    trace: Trace,
    digests: Mapping[str, str],
) -> dict[tuple[str, str | None], tuple[str, ...]]:
    """Group trace-local structural peers by exact and relaxed sibling signatures."""

    groups: dict[tuple[str, str | None], list[str]] = defaultdict(list)
    for module in trace.modules:
        parent = str(getattr(module, "address_parent", None))
        exact_key = (f"exact:{digests[module.address]}", _peer_scope_key(trace, module))
        class_key = (f"class:{getattr(module, 'class_name', '')}", parent)
        stem_key = (f"stem:{_sibling_stem(module.address)}", parent)
        groups[exact_key].append(module.address)
        groups[class_key].append(module.address)
        groups[stem_key].append(module.address)
    return {
        key: tuple(sorted(dict.fromkeys(addresses)))
        for key, addresses in groups.items()
        if len(set(addresses)) >= 2 and key[0] not in {"class:", "stem:"}
    }


def _sibling_stem(address: str) -> str:
    """Return a relaxed sibling-address stem for stage-like module names."""

    name = address.rsplit(".", 1)[-1]
    indexed_stem = _indexed_child_stem(name)
    if indexed_stem is None:
        return name
    stem = indexed_stem.rstrip("._")
    return stem or name


def _peer_scope_key(trace: Trace, module: Module) -> str | None:
    """Return the sibling scope key used for repeated structural peers.

    Parameters
    ----------
    trace:
        Trace that owns the module hierarchy.
    module:
        Module whose peer grouping scope is being resolved.

    Returns
    -------
    str | None
        Stable scope key shared by repeated siblings or cousins under repeated
        parents.
    """

    parent_address = getattr(module, "address_parent", None)
    if parent_address is None:
        return None
    try:
        parent = cast("Module", trace.modules[parent_address])
    except KeyError:
        return str(parent_address)
    grandparent = getattr(parent, "address_parent", None)
    parent_class = str(getattr(parent, "class_name", ""))
    return f"{grandparent}:{parent_class}"


def _readable_band_high(trace: Trace) -> int:
    """Return the high watermark for a readable auto-collapsed render.

    Parameters
    ----------
    trace:
        Trace being rendered.

    Returns
    -------
    int
        Upper readable node-count budget for auto collapse.
    """

    return 25 if len(trace.ops) > 100 else 40


def _is_trunk_collapse(trace: Trace, signal: ModuleCollapseSignals) -> bool:
    """Return whether a module would collapse nearly the whole input-output trunk."""

    visible_after = max(1, len(trace.ops) - signal.hidden_ops)
    if visible_after >= 4:
        return False
    op_set = set(signal.subtree_ops)
    has_input = any(cast("Op", trace.ops[label]).is_input for label in op_set)
    has_output = any(cast("Op", trace.ops[label]).is_output for label in op_set)
    return has_input or has_output


def _rendered_module_hidden_counts(trace: Trace, context: RenderContext) -> dict[str, int]:
    """Return rendered-node counts hidden by selecting each module alone.

    Parameters
    ----------
    trace:
        Trace being rendered.
    context:
        Render context used by the caller's render.

    Returns
    -------
    dict[str, int]
        Per-module rendered hidden contribution. A module replacing ``n``
        rendered nodes with one box contributes ``n - 1``.
    """

    from ._render_common import BoundaryNode
    from ._render_edges import _is_buffer_visible
    from ._render_flow import _entries_to_plot_for_context
    from ._render_nodes import _normalize_buffer_visibility

    absorbed_counts: dict[str, int] = defaultdict(int)
    show_buffer_layers = _normalize_buffer_visibility(context.show_buffer_layers)
    entries_to_plot = _entries_to_plot_for_context(trace, context.vis_mode)
    for node in entries_to_plot.values():
        if isinstance(node, BoundaryNode):
            continue
        if node.is_buffer and not _is_buffer_visible(node, show_buffer_layers):
            continue
        modules = list(getattr(node, "modules", ()) or ())
        if getattr(node, "is_atomic_module", False) and modules:
            modules = modules[:-1]
        addresses = tuple(dict.fromkeys(str(module).rsplit(":", 1)[0] for module in modules))
        for address in addresses:
            absorbed_counts[address] += 1
    return {
        address: max(absorbed_count - 1, 0)
        for address, absorbed_count in absorbed_counts.items()
        if absorbed_count > 1
    }


def _assert_plan_count(
    trace: Trace,
    collapse_fn: Callable[[Module], bool] | None,
    repeat_folds: Mapping[str, ModuleRepeatFold] | None,
    context: RenderContext,
    running_count: int,
) -> None:
    """Assert that incremental count maintenance matches full planning.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate.
    repeat_folds:
        Active repeat-fold mapping.
    context:
        Render context used for planning.
    running_count:
        Incrementally maintained rendered node count.
    """

    planned_count = count(collapse_plan_for_trace(trace, collapse_fn, repeat_folds, context))
    if running_count == planned_count:
        return
    message = (
        "incremental collapse count mismatch: "
        f"running_count={running_count}, planned_count={planned_count}"
    )
    if _strict_count_checks_enabled():
        raise AssertionError(message)
    _warn_count_mismatch_once(message)


# r-b7 R42-9: one shared TORCHLENS_COLLAPSE_STRICT parser (_render_common).
_strict_count_checks_enabled = strict_collapse_checks_enabled


def _warn_count_mismatch_once(message: str) -> None:
    """Emit a single production warning for incremental count mismatches.

    Parameters
    ----------
    message:
        Diagnostic mismatch message.
    """

    global _COUNT_MISMATCH_WARNING_EMITTED
    if _COUNT_MISMATCH_WARNING_EMITTED:
        return
    _COUNT_MISMATCH_WARNING_EMITTED = True
    warnings.warn(
        f"{message}; using authoritative CollapsePlan count for rendering.",
        RuntimeWarning,
        stacklevel=3,
    )


def _run_fold_hidden_member_contributions(
    trace: Trace,
    collapse_fn: Callable[[Module], bool] | None,
    context: RenderContext,
) -> dict[str, int]:
    """Return pre-fold rendered contribution under each module address.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate before run folding, or ``None`` for an
        uncollapsed render.
    context:
        Render context used for the caller's render.

    Returns
    -------
    dict[str, int]
        Count of currently rendered node contributions contained by each
        module address. Boundary nodes with no module ancestry are excluded.
    """

    from .node_universe import build_node_universe
    from .source_graph import build_source_graph

    contributions: dict[str, int] = defaultdict(int)
    emissions = build_node_universe(build_source_graph(trace, context), collapse_fn, None).emissions
    for emission in emissions:
        if emission.kind in {"hidden_run_member", "run_fold_ellipsis"}:
            continue
        addresses = _emission_module_ancestors(emission)
        for address in addresses:
            contributions[address] += 1
    return contributions


def _emission_module_ancestors(emission: Any) -> tuple[str, ...]:
    """Return pass-free module ancestors for a rendered emission.

    Parameters
    ----------
    emission:
        Diagnostic rendered-node emission from the renderer.

    Returns
    -------
    tuple[str, ...]
        Pass-free module addresses enclosing the emitted node.
    """

    addresses: list[str] = []
    if emission.module_address is not None:
        addresses.append(emission.module_address)
    node = emission.node
    if node is None:
        return tuple(dict.fromkeys(addresses))
    addresses.extend(str(module).rsplit(":", 1)[0] for module in getattr(node, "modules", ()) or ())
    return tuple(dict.fromkeys(addresses))


def _run_fold_delta(fold: ModuleRepeatFold, hidden_member_contributions: Mapping[str, int]) -> int:
    """Return rendered-node delta for accepting ``fold``.

    Parameters
    ----------
    fold:
        Candidate run fold.
    hidden_member_contributions:
        Pre-fold contribution count keyed by module address.

    Returns
    -------
    int
        Incremental count change after replacing all member contributions with
        the representative box plus one ellipsis node.
    """

    removed = sum(hidden_member_contributions.get(address, 0) for address in fold.addresses)
    return 2 - removed

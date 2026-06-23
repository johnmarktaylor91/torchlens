"""Smart module-collapse scoring and selection for graph rendering."""

from __future__ import annotations

import hashlib
import math
import time
import weakref
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from .._literals import CollapseLiteral, VisModeLiteral

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


GENERIC_CONTAINER_CLASSES = frozenset({"Sequential", "ModuleList", "ModuleDict", "ParameterList"})


@dataclass(frozen=True)
class CollapseWeights:
    """Weights for module collapse scoring.

    Parameters
    ----------
    size:
        Weight for hidden rendered operation count.
    tangle:
        Weight for internal-edge density.
    repeat:
        Weight for structural peer repetition.
    named:
        Weight for non-container class names.
    landmark:
        Penalty weight for fan-in/fan-out landmarks.
    trunk:
        Penalty weight for collapsing too much of the input-output trunk.
    mass:
        Optional parameter-mass weight.
    grain:
        Cut-level grain variance penalty weight.
    peer:
        Cut-level partial peer-group penalty weight.
    """

    size: float = 1.0
    tangle: float = 0.6
    repeat: float = 0.5
    named: float = 0.15
    landmark: float = 1.2
    trunk: float = 1.0
    mass: float = 0.0
    grain: float = 0.4
    peer: float = 0.8


@dataclass(frozen=True)
class ModuleCollapseSignals:
    """Precomputed structural signals for one module.

    Parameters
    ----------
    address:
        Primary module address.
    subtree_ops:
        Pass-qualified operation labels in module scope.
    own_func_names:
        Function names for ops directly owned by the module, in call order.
    internal_edges:
        Distinct op-graph edges with both endpoints in the module.
    input_edges:
        Distinct op-graph edges entering the module from outside.
    output_edges:
        Distinct op-graph edges leaving the module.
    params:
        Number of recursive parameters for the module.
    depth:
        Address-tree depth.
    num_calls:
        Number of module calls.
    structural_digest:
        Trace-local structural digest.
    peer_count:
        Number of modules in the same address-keyed peer group.
    hidden_ops:
        Rendered op count hidden by collapsing this module.
    eligible:
        Whether renderer-faithful hard gating allows collapse.
    """

    address: str
    subtree_ops: tuple[str, ...]
    own_func_names: tuple[str, ...]
    internal_edges: int
    input_edges: int
    output_edges: int
    params: int
    depth: int
    num_calls: int
    structural_digest: str
    peer_count: int
    hidden_ops: int
    eligible: bool


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
        Structural peer groups keyed by digest and all-addresses tuple.
    elapsed_ms:
        Signal and score computation time in milliseconds.
    """

    signals: Mapping[str, ModuleCollapseSignals]
    scores: Mapping[str, float]
    peer_groups: Mapping[tuple[str, tuple[str, ...]], tuple[str, ...]]
    elapsed_ms: float


_ANALYSIS_CACHE: weakref.WeakKeyDictionary[Any, CollapseAnalysis] = weakref.WeakKeyDictionary()


def analyze_collapse(trace: "Trace") -> CollapseAnalysis:
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

    cached = _ANALYSIS_CACHE.get(trace)
    if cached is not None:
        return cached
    start = time.perf_counter()
    signals_without_peers = _compute_signal_skeleton(trace)
    digests = _compute_structural_digests(trace, signals_without_peers)
    peer_groups = _group_structural_peers(trace, digests)
    peer_count_by_address = {
        address: len(group) for group in peer_groups.values() for address in group
    }
    signals = {
        address: ModuleCollapseSignals(
            address=signal.address,
            subtree_ops=signal.subtree_ops,
            own_func_names=signal.own_func_names,
            internal_edges=signal.internal_edges,
            input_edges=signal.input_edges,
            output_edges=signal.output_edges,
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
    scores = {
        address: round(_score_module(trace, signal, CollapseWeights(), mode="auto"), 6)
        for address, signal in signals.items()
    }
    analysis = CollapseAnalysis(
        signals=signals,
        scores=scores,
        peer_groups=peer_groups,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )
    _ANALYSIS_CACHE[trace] = analysis
    return analysis


def collapse_order(
    trace: "Trace",
    weights: CollapseWeights | Mapping[str, float] | None = None,
    mode: Literal["auto", "max"] = "auto",
) -> list[tuple[str, float]]:
    """Return collapse candidates sorted by score for a policy.

    Parameters
    ----------
    trace:
        Trace to rank.
    weights:
        Optional scoring weights. A mapping overrides matching
        :class:`CollapseWeights` fields.
    mode:
        ``"auto"`` or ``"max"`` landmark policy.

    Returns
    -------
    list[tuple[str, float]]
        ``(module_address, rounded_score)`` sorted by ``(-score, address)``.
    """

    resolved_weights = _resolve_weights(weights)
    analysis = analyze_collapse(trace)
    scores = {
        address: round(_score_module(trace, signal, resolved_weights, mode=mode), 6)
        for address, signal in analysis.signals.items()
    }
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def resolve_collapse_fn(
    trace: "Trace",
    collapse: CollapseLiteral,
    vis_mode: VisModeLiteral,
) -> Callable[["Module"], bool] | None:
    """Resolve a public collapse option to a renderer predicate.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse:
        Public collapse mode.
    vis_mode:
        Current visualization mode.

    Returns
    -------
    Callable[[Module], bool] | None
        Collapse predicate, or ``None`` for ``"none"``.
    """

    if collapse == "none":
        return None
    if collapse not in {"auto", "max"}:
        raise ValueError("collapse must be one of 'none', 'auto', or 'max'.")
    selected = _select_modules(trace, collapse=collapse, vis_mode=vis_mode)

    def collapse_fn(module: "Module") -> bool:
        """Return whether ``module`` is selected for smart collapse."""

        return module.address in selected

    return collapse_fn


def module_collapse_score(module: "Module") -> float:
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
    analysis = analyze_collapse(trace)
    signal = analysis.signals.get(module.address)
    if signal is None or not signal.eligible:
        return 0.0
    return analysis.scores.get(module.address, 0.0)


def _resolve_weights(weights: CollapseWeights | Mapping[str, float] | None) -> CollapseWeights:
    """Return a complete weight object from optional overrides."""

    if weights is None:
        return CollapseWeights()
    if isinstance(weights, CollapseWeights):
        return weights
    values = CollapseWeights().__dict__.copy()
    for key, value in weights.items():
        if key not in values:
            raise ValueError(f"Unknown collapse weight: {key}")
        values[key] = float(value)
    return CollapseWeights(**values)


def _compute_signal_skeleton(trace: "Trace") -> dict[str, ModuleCollapseSignals]:
    """Compute all non-peer module signals in one shared traversal."""

    op_labels_by_module: dict[str, list[str]] = defaultdict(list)
    own_func_names_by_module: dict[str, list[str]] = defaultdict(list)
    internal_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    input_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)
    output_edges: dict[str, set[tuple[str, str]]] = defaultdict(set)

    ops = list(trace.ops)
    op_by_label = {op.label: op for op in ops}
    stack_by_label = {op.label: _module_address_stack(op) for op in ops}

    for op in ops:
        stack = stack_by_label[op.label]
        for address in stack:
            op_labels_by_module[address].append(op.label)
        if stack:
            own_func_names_by_module[stack[-1]].append(_op_func_name(op))

    for parent in ops:
        parent_stack = stack_by_label[parent.label]
        parent_set = set(parent_stack)
        for child_label in parent.children:
            child = op_by_label.get(child_label)
            if child is None:
                child = cast("Op", trace.ops[child_label])
                op_by_label[child_label] = child
                op_by_label[child.label] = child
                stack_by_label[child.label] = _module_address_stack(child)
            child_stack = stack_by_label[child.label]
            child_set = set(child_stack)
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
            params=int(getattr(module, "num_params", 0) or 0),
            depth=int(getattr(module, "address_depth", 0) or 0),
            num_calls=int(getattr(module, "num_calls", 1) or 1),
            structural_digest="",
            peer_count=1,
            hidden_ops=hidden_ops,
            eligible=_gate_module(module, hidden_ops, signals),
        )
    return signals


def _module_address_stack(op: "Op") -> tuple[str, ...]:
    """Return pass-free module addresses enclosing an op."""

    return tuple(str(module).rsplit(":", 1)[0] for module in getattr(op, "modules", ()) or ())


def _op_func_name(op: "Op") -> str:
    """Return a stable operation function name for digesting."""

    return str(getattr(op, "func_name", None) or getattr(op, "layer_type", "") or "")


def _gate_module(
    module: "Module",
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
    trace: "Trace",
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
    trace: "Trace",
    digests: Mapping[str, str],
) -> dict[tuple[str, tuple[str, ...]], tuple[str, ...]]:
    """Group trace-local structural peers by digest and all addresses."""

    groups: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for module in trace.modules:
        key = (digests[module.address], tuple(getattr(module, "all_addresses", ()) or ()))
        groups[key].append(module.address)
    return {key: tuple(sorted(addresses)) for key, addresses in groups.items()}


def _score_module(
    trace: "Trace",
    signal: ModuleCollapseSignals,
    weights: CollapseWeights,
    mode: Literal["auto", "max"],
) -> float:
    """Return unrounded collapse benefit for one module."""

    if not signal.eligible:
        return 0.0
    n = signal.hidden_ops
    size = _clip((math.log2(1 + n) - math.log2(1 + 3)) / (math.log2(1 + 64) - math.log2(1 + 3)))
    edge_total = signal.internal_edges + signal.input_edges + signal.output_edges
    tangle = (
        min(signal.internal_edges / edge_total, 1.0)
        if signal.internal_edges >= 2 and edge_total > 0
        else 0.0
    )
    repeat = 0.0 if mode == "max" else min((max(signal.peer_count, 1) - 1) / 5.0, 1.0)
    module = cast("Module", trace.modules[signal.address])
    named = 0.0 if module.class_name in GENERIC_CONTAINER_CLASSES else 1.0
    fan_threshold = 2 if mode == "auto" else 3
    max_boundary = max(signal.input_edges, signal.output_edges)
    landmark = min(max(0, max_boundary - (fan_threshold - 1)) / 2.0, 1.0)
    trunk = 1.0 if _is_trunk_collapse(trace, signal) else 0.0
    mass = _clip(math.log10(1 + max(signal.params, 0)) / 6.0)
    return (
        weights.size * size
        + weights.tangle * tangle
        + weights.repeat * repeat
        + weights.named * named
        + weights.mass * mass
        - weights.landmark * landmark
        - weights.trunk * trunk
    )


def _clip(value: float) -> float:
    """Clip a value to the unit interval."""

    return max(0.0, min(value, 1.0))


def _is_trunk_collapse(trace: "Trace", signal: ModuleCollapseSignals) -> bool:
    """Return whether a module would over-collapse the top-level backbone."""

    visible_after = max(1, len(trace.ops) - signal.hidden_ops)
    if visible_after >= 4:
        return False
    op_set = set(signal.subtree_ops)
    has_input = any(cast("Op", trace.ops[label]).is_input for label in op_set)
    has_output = any(cast("Op", trace.ops[label]).is_output for label in op_set)
    return has_input or has_output


def _select_modules(
    trace: "Trace",
    *,
    collapse: Literal["auto", "max"],
    vis_mode: VisModeLiteral,
) -> frozenset[str]:
    """Select an antichain of module addresses by greedy score."""

    _ = vis_mode
    analysis = analyze_collapse(trace)
    band = (8, 40, 28) if collapse == "auto" else (4, 12, 7)
    floor = 1 if collapse == "auto" else 3
    selected: list[str] = []
    visible_count = len(trace.ops)
    ordered = collapse_order(trace, mode=collapse)
    for address, score in ordered:
        if score <= 0.0:
            continue
        signal = analysis.signals[address]
        if not signal.eligible:
            continue
        if _inside_selected(address, selected):
            continue
        next_visible = visible_count - signal.hidden_ops
        if next_visible < floor:
            continue
        selected.append(address)
        visible_count = next_visible
        if band[0] <= visible_count <= band[1]:
            break
    if collapse == "max":
        return frozenset(selected)
    if band[0] <= visible_count <= band[1]:
        return frozenset(selected)
    return frozenset(selected)


def _inside_selected(address: str, selected: list[str]) -> bool:
    """Return whether ``address`` is already hidden by a selected ancestor."""

    return any(address.startswith(f"{ancestor}.") for ancestor in selected)

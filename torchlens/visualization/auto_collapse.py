"""Smart module-collapse scoring and selection for graph rendering."""

from __future__ import annotations

import hashlib
import math
import re
import time
import weakref
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from .._literals import CollapseLiteral, VisModeLiteral

if TYPE_CHECKING:
    from ..data_classes.module import Module
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


GENERIC_CONTAINER_CLASSES = frozenset({"Sequential", "ModuleList", "ModuleDict", "ParameterList"})
STRUCTURED_CONTAINER_NAMES = frozenset(
    {
        "backbone",
        "body",
        "encoder",
        "features",
        "layers",
        "module",
        "stages",
        "trunk",
    }
)
JUNCTION_FUNC_NAMES = frozenset({"__add__", "add", "cat", "concat", "concatenate"})
LEAF_BLOCK_MAX_OPS = 12
LEAF_BLOCK_WRAPPER_CLASSES = frozenset(
    {"BasicConv2d", "Conv2dNormActivation", "ConvNormActivation"}
)
LEAF_BLOCK_CHILD_CLASS_PARTS = (
    "Activation",
    "BatchNorm",
    "Conv",
    "Dropout",
    "GELU",
    "Identity",
    "LayerNorm",
    "Pool",
    "ReLU",
    "SiLU",
)
_INDEXED_CHILD_RE = re.compile(r"^(?P<stem>.*?)(?:\.?\d+|_?\d+[a-z]?)$")
_STAGE_NAME_RE = re.compile(
    r"^(?:denseblock\d*|transition\d*|layer\d+|stage\d*|mixed_[0-9a-z]+|mixed_\d+)$",
    re.IGNORECASE,
)
RUN_FOLD_MIN_LENGTH = 3


@dataclass(frozen=True)
class ModuleRunFold:
    """Consecutive structurally-identical module run selected for render folding.

    Parameters
    ----------
    representative:
        First module address in the folded run.
    addresses:
        Consecutive sibling module addresses included in the run.
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
    """

    representative: str
    addresses: tuple[str, ...]
    num_layers: int
    num_params: int
    num_params_trainable: int
    num_params_frozen: int
    shape_summary: str | None

    @property
    def multiplicity(self) -> int:
        """Return the number of folded sibling modules."""

        return len(self.addresses)


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
    landmark_edges:
        Boundary-crossing edges that enter or leave non-boundary internal
        operations and therefore hide a meaningful cross-module junction.
    passthrough_edges:
        Internal output junctions that combine module input with internal work.
    output_junctions:
        External multi-parent children fed by module outputs.
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
    landmark_edges: int
    passthrough_edges: int
    output_junctions: tuple[str, ...]
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
        Structural or relaxed peer groups keyed by signature and sibling parent
        address.
    elapsed_ms:
        Signal and score computation time in milliseconds.
    """

    signals: Mapping[str, ModuleCollapseSignals]
    scores: Mapping[str, float]
    peer_groups: Mapping[tuple[str, str | None], tuple[str, ...]]
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


def resolve_run_folds(
    trace: "Trace",
    collapse_fn: Callable[["Module"], bool] | None,
) -> dict[str, ModuleRunFold]:
    """Return render-time folds for consecutive collapsed sibling runs.

    Parameters
    ----------
    trace:
        Trace being rendered.
    collapse_fn:
        Active collapse predicate. ``None`` disables run folding.

    Returns
    -------
    dict[str, ModuleRunFold]
        Mapping from each folded module address to its run descriptor.
    """

    if collapse_fn is None:
        return {}
    analysis = analyze_collapse(trace)
    selected = [module.address for module in trace.modules if collapse_fn(module)]
    projected_count = _visible_count_after_selection(trace, analysis, selected)
    if projected_count <= _readable_band_high(trace):
        return {}
    dimless_digests = _compute_dimless_structural_digests(trace)
    candidate_folds: list[ModuleRunFold] = []
    candidate_addresses: set[str] = set()
    for child_addresses in _sibling_address_groups(trace).values():
        for run in _iter_collapsible_runs(trace, child_addresses, dimless_digests, collapse_fn):
            if not _run_span_allows_fold(trace, run):
                continue
            fold = _make_run_fold(trace, run)
            candidate_folds.append(fold)
            candidate_addresses.update(run)
        for run in _iter_collapsible_child_path_runs(
            trace,
            child_addresses,
            dimless_digests,
            collapse_fn,
        ):
            if any(address in candidate_addresses for address in run):
                continue
            if not _run_span_allows_fold(trace, run):
                continue
            fold = _make_run_fold(trace, run)
            candidate_folds.append(fold)
            candidate_addresses.update(run)
        for run in _iter_collapsible_runs(
            trace,
            child_addresses,
            dimless_digests,
            collapse_fn,
            allow_selected_descendant=True,
        ):
            if any(address in candidate_addresses for address in run):
                continue
            if not _run_span_allows_fold(trace, run):
                continue
            fold = _make_run_fold(trace, run)
            candidate_folds.append(fold)
            candidate_addresses.update(run)
    folds_by_address: dict[str, ModuleRunFold] = {}
    for fold in sorted(candidate_folds, key=lambda item: (-item.multiplicity, item.representative)):
        if any(address in folds_by_address for address in fold.addresses):
            continue
        for address in fold.addresses:
            folds_by_address[address] = fold
        projected_count -= sum(analysis.signals[address].hidden_ops for address in fold.addresses)
        if projected_count <= _readable_band_high(trace):
            break
    return folds_by_address


def _sibling_address_groups(trace: "Trace") -> dict[str | None, list[str]]:
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


def _compute_dimless_structural_digests(trace: "Trace") -> dict[str, str]:
    """Compute module structural digests that ignore dimensions and parameters.

    Parameters
    ----------
    trace:
        Trace whose module hierarchy is being fingerprinted.

    Returns
    -------
    dict[str, str]
        Digest keyed by pass-free module address.
    """

    signals = _compute_signal_skeleton(trace)
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
                len(signal.subtree_ops),
                int(getattr(module, "num_layers", 0) or 0),
                _normalized_internal_topology(trace, signal.subtree_ops),
            )
        ).encode("utf-8")
        digests[module.address] = hashlib.sha1(payload).hexdigest()
    return digests


def _normalized_internal_topology(
    trace: "Trace",
    subtree_ops: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    """Return dimension-free internal op-edge topology for ``subtree_ops``.

    Parameters
    ----------
    trace:
        Trace owning the operation graph.
    subtree_ops:
        Pass-qualified operation labels in module scope.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Internal edges expressed as subtree-order indices.
    """

    index_by_label = {label: index for index, label in enumerate(subtree_ops)}
    subtree = set(subtree_ops)
    edges: set[tuple[int, int]] = set()
    for parent_label in subtree_ops:
        parent = cast("Op", trace.ops[parent_label])
        parent_index = index_by_label[parent.label]
        for child_label in getattr(parent, "children", ()) or ():
            if child_label not in subtree:
                continue
            edges.add((parent_index, index_by_label[child_label]))
    return tuple(sorted(edges))


def _iter_collapsible_runs(
    trace: "Trace",
    child_addresses: list[str],
    dimless_digests: Mapping[str, str],
    collapse_fn: Callable[["Module"], bool],
    run_stem: str | None = None,
    allow_selected_descendant: bool = False,
) -> Iterator[tuple[str, ...]]:
    """Yield consecutive sibling runs that should fold to one render node.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    child_addresses:
        Ordered direct children for one parent module.
    dimless_digests:
        Dim-insensitive structural digest keyed by address.
    collapse_fn:
        Active collapse predicate.
    run_stem:
        Optional precomputed sibling-run stem for descendant-path folds.
    allow_selected_descendant:
        Whether a module with selected descendants can stand in for a selected
        module. This supports folding repeated ancestors whose internal
        submodules would otherwise render as a node wall.

    Yields
    ------
    tuple[str, ...]
        One run of at least :data:`RUN_FOLD_MIN_LENGTH` addresses.
    """

    current_key: tuple[str, str, str] | None = None
    current_run: list[str] = []
    for address in child_addresses:
        module = cast("Module", trace.modules[address])
        selected = collapse_fn(module) or (
            allow_selected_descendant and bool(_selected_descendants(trace, address, collapse_fn))
        )
        if not selected:
            if allow_selected_descendant:
                continue
            if len(current_run) >= RUN_FOLD_MIN_LENGTH:
                yield tuple(current_run)
            current_key = None
            current_run = []
            continue
        key = (
            str(getattr(module, "class_name", "")),
            (
                _selected_descendant_run_digest(trace, module, dimless_digests)
                if allow_selected_descendant
                else dimless_digests.get(address, "")
            ),
            run_stem or _indexed_parent_stem(address),
        )
        if key == current_key:
            current_run.append(address)
            continue
        if len(current_run) >= RUN_FOLD_MIN_LENGTH:
            yield tuple(current_run)
        current_key = key
        current_run = [address]
    if len(current_run) >= RUN_FOLD_MIN_LENGTH:
        yield tuple(current_run)


def _selected_descendant_run_digest(
    trace: "Trace",
    module: "Module",
    dimless_digests: Mapping[str, str],
) -> str:
    """Return the structural digest for selected-descendant run folding.

    Parameters
    ----------
    trace:
        Trace owning the module hierarchy.
    module:
        Candidate module whose descendants may be selected for collapse.
    dimless_digests:
        Full dimensionless structural digests keyed by module address.

    Returns
    -------
    str
        Digest that preserves immediate child topology and depth while ignoring
        implementation-level op differences inside those repeated children.
    """

    child_addresses = tuple(
        str(child_address)
        for child_address in getattr(module, "address_children", ()) or ()
        if child_address in trace.modules
    )
    if not child_addresses:
        return dimless_digests.get(module.address, "")
    child_classes = tuple(
        str(getattr(cast("Module", trace.modules[child_address]), "class_name", ""))
        for child_address in child_addresses
    )
    payload = repr((getattr(module, "class_name", ""), child_classes, len(child_addresses))).encode(
        "utf-8"
    )
    return hashlib.sha1(payload).hexdigest()


def _iter_collapsible_child_path_runs(
    trace: "Trace",
    sibling_addresses: list[str],
    dimless_digests: Mapping[str, str],
    collapse_fn: Callable[["Module"], bool],
) -> Iterator[tuple[str, ...]]:
    """Yield repeated selected child paths under consecutive sibling parents.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    sibling_addresses:
        Ordered direct children for one parent module.
    dimless_digests:
        Dim-insensitive structural digest keyed by address.
    collapse_fn:
        Active collapse predicate.

    Yields
    ------
    tuple[str, ...]
        One run of selected descendant modules sharing the same relative path.
    """

    relative_paths = sorted(
        {
            selected_address.removeprefix(f"{sibling}.")
            for sibling in sibling_addresses
            for selected_address in _selected_descendants(trace, sibling, collapse_fn)
            if selected_address.startswith(f"{sibling}.")
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
                    dimless_digests,
                    collapse_fn,
                    current_stem,
                )
            current_stem = stem
            current_candidates = [candidate_address] if candidate_address else []
        if current_candidates:
            yield from _iter_collapsible_runs(
                trace,
                current_candidates,
                dimless_digests,
                collapse_fn,
                current_stem,
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
    match = _INDEXED_CHILD_RE.match(name)
    if match is None:
        stem = name
    else:
        stem = match.group("stem").rstrip("._") or ""
    return f"{parent}.{stem}" if parent and stem else parent or stem or name


def _selected_descendants(
    trace: "Trace",
    address: str,
    collapse_fn: Callable[["Module"], bool],
) -> tuple[str, ...]:
    """Return selected descendant module addresses under ``address``.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    address:
        Parent module address.
    collapse_fn:
        Active collapse predicate.

    Returns
    -------
    tuple[str, ...]
        Selected descendant addresses in lexical order.
    """

    prefix = f"{address}."
    return tuple(
        module.address
        for module in trace.modules
        if module.address.startswith(prefix) and collapse_fn(module)
    )


def _make_run_fold(trace: "Trace", addresses: tuple[str, ...]) -> ModuleRunFold:
    """Build aggregate metadata for one folded run.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Consecutive sibling addresses in the run.

    Returns
    -------
    ModuleRunFold
        Aggregate run-fold descriptor.
    """

    modules = [cast("Module", trace.modules[address]) for address in addresses]
    return ModuleRunFold(
        representative=addresses[0],
        addresses=addresses,
        num_layers=sum(int(getattr(module, "num_layers", 0) or 0) for module in modules),
        num_params=sum(int(getattr(module, "num_params", 0) or 0) for module in modules),
        num_params_trainable=sum(
            int(getattr(module, "num_params_trainable", 0) or 0) for module in modules
        ),
        num_params_frozen=sum(
            int(getattr(module, "num_params_frozen", 0) or 0) for module in modules
        ),
        shape_summary=_run_shape_summary(trace, addresses),
    )


def _run_shape_summary(trace: "Trace", addresses: tuple[str, ...]) -> str | None:
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


def _run_span_allows_fold(trace: "Trace", addresses: tuple[str, ...]) -> bool:
    """Return whether first-to-last tensor shape span is safe to fold.

    Parameters
    ----------
    trace:
        Trace owning the modules.
    addresses:
        Consecutive sibling addresses in the candidate run.

    Returns
    -------
    bool
        True when the run does not cross a spatial-resolution boundary and
        does not span more than a 2x channel-width change. Unknown shapes are
        treated as foldable because the structural key is the primary guard.
    """

    first = _module_output_shape_tuple(trace, addresses[0])
    last = _module_output_shape_tuple(trace, addresses[-1])
    if first is None or last is None:
        return True
    if len(first) != len(last):
        return False
    first_spatial = _shape_spatial_dims(first)
    last_spatial = _shape_spatial_dims(last)
    if first_spatial is not None and last_spatial is not None and first_spatial != last_spatial:
        return False
    first_channels = _shape_channel_dim(first)
    last_channels = _shape_channel_dim(last)
    if first_channels is None or last_channels is None:
        return True
    smaller = min(first_channels, last_channels)
    larger = max(first_channels, last_channels)
    return smaller > 0 and larger <= smaller * 2


def _module_output_shape_tuple(trace: "Trace", address: str) -> tuple[int, ...] | None:
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

    pass_address = f"{address}:1"
    if pass_address not in trace.modules:
        return None
    try:
        module_output_layer = trace[pass_address]
    except (KeyError, IndexError):
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


def _module_output_shape(trace: "Trace", address: str) -> str | None:
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
            landmark_edges=_count_landmark_edges(
                trace,
                module,
                subtree_ops,
                input_edges.get(address, set()) | output_edges.get(address, set()),
            ),
            passthrough_edges=_count_passthrough_edges(trace, module, subtree_ops),
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


def _module_address_stack(op: "Op") -> tuple[str, ...]:
    """Return pass-free module addresses enclosing an op."""

    return tuple(str(module).rsplit(":", 1)[0] for module in getattr(op, "modules", ()) or ())


def _op_func_name(op: "Op") -> str:
    """Return a stable operation function name for digesting."""

    return str(getattr(op, "func_name", None) or getattr(op, "layer_type", "") or "")


def _count_landmark_edges(
    trace: "Trace",
    module: "Module",
    subtree_ops: tuple[str, ...],
    boundary_edges: set[tuple[str, str]],
) -> int:
    """Return boundary-crossing non-boundary edges for a module.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    module:
        Candidate module being scored.
    subtree_ops:
        Pass-qualified operation labels in the module subtree.
    boundary_edges:
        Distinct edges crossing the module boundary.

    Returns
    -------
    int
        Count of boundary edges whose internal endpoint is not a normal module
        input or output boundary. Fully internal junctions are intentionally not
        counted because they are safely hidden with the collapsed module box.
    """

    subtree = set(subtree_ops)
    input_layers = {_base_label(label) for label in getattr(module, "input_layers", ()) or ()}
    output_layers = {_base_label(label) for label in getattr(module, "output_layers", ()) or ()}
    landmarks: set[tuple[str, str]] = set()
    for parent_label, child_label in boundary_edges:
        parent = cast("Op", trace.ops[parent_label])
        child = cast("Op", trace.ops[child_label])
        if getattr(parent, "is_buffer", False) or getattr(child, "is_buffer", False):
            continue
        parent_inside = parent.label in subtree
        child_inside = child.label in subtree
        if parent_inside == child_inside:
            continue
        parent_base = _base_label(parent.label)
        child_base = _base_label(child.label)
        if child_inside and parent_base in input_layers:
            continue
        if parent_inside and parent_base in output_layers:
            continue
        if child_inside and child_base in output_layers:
            continue
        landmarks.add((parent.label, child.label))
    return len(landmarks)


def _base_label(label: str) -> str:
    """Return a pass-free operation label.

    Parameters
    ----------
    label:
        Operation label that may include a pass suffix.

    Returns
    -------
    str
        Operation label without the trailing pass suffix.
    """

    return str(label).rsplit(":", 1)[0]


def _count_passthrough_edges(
    trace: "Trace",
    module: "Module",
    subtree_ops: tuple[str, ...],
) -> int:
    """Return internal output joins fed directly by module inputs.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    module:
        Candidate module being scored.
    subtree_ops:
        Pass-qualified operation labels in the module subtree.

    Returns
    -------
    int
        Number of module-output Ops that merge an external module input with
        internal computation. These joins are useful orientation landmarks for
        ``collapse="auto"`` but may be hidden by ``collapse="max"``.
    """

    subtree = set(subtree_ops)
    input_layers = {_base_label(label) for label in getattr(module, "input_layers", ()) or ()}
    output_layers = {_base_label(label) for label in getattr(module, "output_layers", ()) or ()}
    passthrough_edges = 0
    for label in subtree_ops:
        op = cast("Op", trace.ops[label])
        if _base_label(op.label) not in output_layers:
            continue
        if _op_func_name(op) not in JUNCTION_FUNC_NAMES:
            continue
        has_internal_parent = False
        has_input_parent = False
        for parent_label in op.parents:
            parent = cast("Op", trace.ops[parent_label])
            if parent.label in subtree:
                has_internal_parent = True
            elif _base_label(parent.label) in input_layers:
                has_input_parent = True
        if has_internal_parent and has_input_parent:
            passthrough_edges += 1
    return passthrough_edges


def _output_junctions(
    trace: "Trace",
    module: "Module",
    subtree_ops: tuple[str, ...],
    output_edges: set[tuple[str, str]],
) -> tuple[str, ...]:
    """Return external multi-parent junction children fed by module outputs.

    Parameters
    ----------
    trace:
        Trace that owns the operation graph.
    module:
        Candidate module being scored.
    subtree_ops:
        Pass-qualified operation labels in the module subtree.
    output_edges:
        Distinct edges leaving the module subtree.

    Returns
    -------
    tuple[str, ...]
        Pass-free labels for external multi-parent children fed by this module.
    """

    subtree = set(subtree_ops)
    output_layers = {_base_label(label) for label in getattr(module, "output_layers", ()) or ()}
    junctions: set[str] = set()
    for parent_label, child_label in output_edges:
        parent = cast("Op", trace.ops[parent_label])
        child = cast("Op", trace.ops[child_label])
        if parent.label not in subtree:
            continue
        if _base_label(parent.label) not in output_layers:
            continue
        if len(getattr(child, "parents", ()) or ()) < 2:
            continue
        junctions.add(_base_label(child.label))
    return tuple(sorted(junctions))


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
    match = _INDEXED_CHILD_RE.match(name)
    if match is None:
        return name
    stem = match.group("stem").rstrip("._")
    return stem or name


def _peer_scope_key(trace: "Trace", module: "Module") -> str | None:
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


def _score_module(
    trace: "Trace",
    signal: ModuleCollapseSignals,
    weights: CollapseWeights,
    mode: Literal["auto", "max"],
) -> float:
    """Return unrounded collapse benefit for one module."""

    if not signal.eligible:
        return 0.0
    if mode == "auto" and _is_structured_container(trace, signal):
        return 0.0
    n = signal.hidden_ops
    size = _clip((math.log2(1 + n) - math.log2(1 + 3)) / (math.log2(1 + 64) - math.log2(1 + 3)))
    edge_total = signal.internal_edges + signal.input_edges + signal.output_edges
    tangle = (
        min(signal.internal_edges / edge_total, 1.0)
        if signal.internal_edges >= 1 and edge_total > 0
        else 0.0
    )
    repeat = 0.0 if mode == "max" else min((max(signal.peer_count, 1) - 1) / 5.0, 1.0)
    module = cast("Module", trace.modules[signal.address])
    named = 0.0 if mode == "max" or module.class_name in GENERIC_CONTAINER_CLASSES else 1.0
    fan_threshold = 2 if mode == "auto" else 3
    landmark = min(max(0, signal.landmark_edges - (fan_threshold - 1)) / 2.0, 1.0)
    passthrough = 1.0 if mode == "auto" and signal.passthrough_edges > 0 else 0.0
    if mode == "auto" and _is_named_stage(signal.address):
        landmark = 0.0
        passthrough = 0.0
    trunk = 1.0 if _is_trunk_collapse(trace, signal) else 0.0
    mass = _clip(math.log10(1 + max(signal.params, 0)) / 6.0)
    depth_weight = 0.08 if mode == "auto" else 0.04
    depth_bonus = depth_weight * min(signal.depth, 8)
    stage_bonus = 0.6 if mode == "auto" and _is_named_stage(signal.address) else 0.0
    return (
        weights.size * size
        + weights.tangle * tangle
        + weights.repeat * repeat
        + weights.named * named
        + weights.mass * mass
        + depth_bonus
        + stage_bonus
        - weights.landmark * landmark
        - weights.landmark * passthrough
        - weights.trunk * trunk
    )


def _is_named_stage(address: str) -> bool:
    """Return whether ``address`` names a human-meaningful architecture stage."""

    name = address.rsplit(".", 1)[-1]
    return _STAGE_NAME_RE.match(name) is not None


def _is_mixed_stage(address: str) -> bool:
    """Return whether ``address`` is an Inception mixed-stage module."""

    return address.rsplit(".", 1)[-1].lower().startswith("mixed_")


def _is_stem_basic_conv(address: str) -> bool:
    """Return whether ``address`` is a torchvision Inception stem BasicConv2d."""

    return address.rsplit(".", 1)[-1].startswith("Conv2d_")


def _is_structured_container(trace: "Trace", signal: ModuleCollapseSignals) -> bool:
    """Return whether ``signal`` should be transparent in ``collapse='auto'``."""

    try:
        module = cast("Module", trace.modules[signal.address])
    except KeyError:
        return False
    container_name = signal.address.rsplit(".", 1)[-1].lower()
    if container_name not in STRUCTURED_CONTAINER_NAMES:
        return False
    if _is_flat_feature_leaf_container(trace, module):
        return False
    child_addresses = [
        str(child_address)
        for child_address in getattr(module, "address_children", ()) or ()
        if child_address in trace.modules
    ]
    meaningful_children = [
        child_address
        for child_address in child_addresses
        if _is_meaningful_stage_child(cast("Module", trace.modules[child_address]))
    ]
    if len(meaningful_children) < 2:
        return bool(meaningful_children) and _sibling_stem(meaningful_children[0]) in {
            "layers",
            "stages",
        }
    return True


def _is_flat_feature_leaf_container(trace: "Trace", module: "Module") -> bool:
    """Return whether a container is a flat feature layer chain.

    Parameters
    ----------
    trace:
        Trace owning the module hierarchy.
    module:
        Candidate container module.

    Returns
    -------
    bool
        True for flat Sequential-like conv/norm/activation/pool feature chains.
    """

    if str(getattr(module, "class_name", "")) not in GENERIC_CONTAINER_CLASSES:
        return False
    child_addresses = [
        str(child_address)
        for child_address in getattr(module, "address_children", ()) or ()
        if child_address in trace.modules
    ]
    if len(child_addresses) < 3:
        return False
    return all(
        _is_leaf_block_ingredient(cast("Module", trace.modules[child_address]))
        for child_address in child_addresses
    )


def _is_meaningful_stage_child(module: "Module") -> bool:
    """Return whether a direct child is substantial enough to keep as a stage."""

    return int(getattr(module, "num_layers", 0) or 0) >= 2


def _is_meaningful_leaf_block(trace: "Trace", signal: ModuleCollapseSignals) -> bool:
    """Return whether a standalone small op chain should collapse as one block.

    Parameters
    ----------
    trace:
        Trace that owns the module hierarchy.
    signal:
        Candidate module signals.

    Returns
    -------
    bool
        True when the module is a compact non-junction block whose internals
        would otherwise render at a finer grain than neighboring stage boxes.
    """

    if not signal.eligible:
        return False
    try:
        module = cast("Module", trace.modules[signal.address])
    except KeyError:
        return False
    op_count = len(signal.subtree_ops)
    forced_wrapper = (
        module.class_name in LEAF_BLOCK_WRAPPER_CLASSES
        or signal.address.rsplit(".", 1)[-1].lower() == "project"
    )
    if op_count < 2:
        return False
    if op_count > LEAF_BLOCK_MAX_OPS and not forced_wrapper:
        return False
    if signal.passthrough_edges > 0:
        return False
    if _is_structured_container(trace, signal):
        return False
    child_addresses = [
        str(child_address)
        for child_address in getattr(module, "address_children", ()) or ()
        if child_address in trace.modules
    ]
    if not child_addresses and _is_tiny_norm_leaf(module, op_count):
        return True
    for child_address in child_addresses:
        child_module = cast("Module", trace.modules[child_address])
        if not _is_leaf_block_ingredient(child_module):
            return False
    if any(
        _op_func_name(cast("Op", trace.ops[label])) in JUNCTION_FUNC_NAMES
        for label in signal.subtree_ops
    ):
        return False
    return signal.internal_edges >= 1


def _is_leaf_block_ingredient(module: "Module") -> bool:
    """Return whether a child module is an atomic layer inside a leaf block."""

    class_name = str(getattr(module, "class_name", ""))
    return any(part in class_name for part in LEAF_BLOCK_CHILD_CLASS_PARTS)


def _is_tiny_norm_leaf(module: "Module", op_count: int) -> bool:
    """Return whether a norm module should fold its primitive bookkeeping ops."""

    class_name = str(getattr(module, "class_name", ""))
    return 2 <= op_count <= 3 and "Norm" in class_name


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
    band = (8, _readable_band_high(trace), 28) if collapse == "auto" else (4, 12, 7)
    if collapse == "auto" and len(trace.ops) > 100:
        band = (1, _readable_band_high(trace), 0)
    floor = 1 if collapse == "auto" else 3
    target = band[2]
    if collapse == "auto" and len(trace.ops) > 15:
        target = min(target, math.ceil(len(trace.ops) * 0.7))
    selected: list[str] = []
    visible_count = len(trace.ops)
    if collapse == "max":
        selected = list(_select_modules(trace, collapse="auto", vis_mode=vis_mode))
        visible_count = _visible_count_after_selection(trace, analysis, selected)
    ordered = collapse_order(trace, mode=collapse)
    for address, score in ordered:
        if score <= 0.0:
            continue
        signal = analysis.signals[address]
        if not signal.eligible:
            continue
        if _selection_conflicts(address, selected, collapse=collapse):
            continue
        addresses = _selection_batch(address, analysis, selected, collapse=collapse)
        hidden_ops = sum(analysis.signals[batch_address].hidden_ops for batch_address in addresses)
        if (
            collapse == "auto"
            and len(trace.ops) > 100
            and len(addresses) > 1
            and all(_is_mixed_stage(batch_address) for batch_address in addresses)
        ):
            hidden_ops = min(hidden_ops, max(1, visible_count - 20))
        next_visible = visible_count - hidden_ops
        tentative_selected = _selection_with_addresses(selected, addresses, collapse=collapse)
        if collapse == "max":
            next_visible = _visible_count_after_selection(trace, analysis, tentative_selected)
        if next_visible < floor and len(addresses) > 1:
            addresses = (address,)
            tentative_selected = _selection_with_addresses(selected, addresses, collapse=collapse)
            next_visible = _visible_count_after_selection(trace, analysis, tentative_selected)
        if (
            next_visible < floor
            and collapse == "auto"
            and visible_count > band[0]
            and _is_stem_basic_conv(address)
        ):
            hidden_ops = max(1, visible_count - band[0])
            next_visible = visible_count - hidden_ops
        if next_visible < floor:
            continue
        selected = tentative_selected
        visible_count = next_visible
        if band[0] <= visible_count <= target:
            break
    selected = _complete_leaf_block_selection(trace, analysis, selected, collapse=collapse)
    # Irregular stage containers remain a known general-algorithm gap; keep
    # selection model-agnostic rather than adding per-class patches.
    if collapse == "max":
        return frozenset(selected)
    if band[0] <= visible_count <= band[1]:
        return frozenset(selected)
    return frozenset(selected)


def _readable_band_high(trace: "Trace") -> int:
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


def _visible_count_after_selection(
    trace: "Trace",
    analysis: CollapseAnalysis,
    selected: list[str],
) -> int:
    """Return the rendered op count estimate after collapsing ``selected``."""

    hidden_ops = sum(analysis.signals[address].hidden_ops for address in selected)
    return max(1, len(trace.ops) - hidden_ops)


def _selection_with_addresses(
    selected: list[str],
    addresses: tuple[str, ...],
    *,
    collapse: Literal["auto", "max"],
) -> list[str]:
    """Return ``selected`` updated with ``addresses`` under mode overlap rules."""

    if collapse == "auto":
        return [*selected, *addresses]
    selected_without_descendants = [
        selected_address
        for selected_address in selected
        if not any(selected_address.startswith(f"{address}.") for address in addresses)
    ]
    return [*selected_without_descendants, *addresses]


def _complete_leaf_block_selection(
    trace: "Trace",
    analysis: CollapseAnalysis,
    selected: list[str],
    *,
    collapse: Literal["auto", "max"],
) -> list[str]:
    """Add compatible standalone leaf-blocks for uniform collapse grain.

    Parameters
    ----------
    trace:
        Trace being rendered.
    analysis:
        Precomputed collapse analysis.
    selected:
        Greedy collapse antichain selected so far.
    collapse:
        Active collapse mode.

    Returns
    -------
    list[str]
        Selection plus compatible leaf-block candidates.
    """

    completed = list(selected)
    leaf_addresses = [
        address
        for address, signal in analysis.signals.items()
        if _is_meaningful_leaf_block(trace, signal)
    ]
    leaf_addresses.sort(key=lambda address: (analysis.signals[address].depth, address))
    for address in leaf_addresses:
        if _selection_conflicts(address, completed, collapse=collapse):
            continue
        completed = _selection_with_addresses(completed, (address,), collapse=collapse)
    return completed


def _inside_selected(address: str, selected: list[str]) -> bool:
    """Return whether ``address`` is already hidden by a selected ancestor."""

    return any(address.startswith(f"{ancestor}.") for ancestor in selected)


def _conflicts_selected(address: str, selected: list[str]) -> bool:
    """Return whether ``address`` overlaps an already selected collapse target."""

    return _inside_selected(address, selected) or any(
        selected_address.startswith(f"{address}.") for selected_address in selected
    )


def _selection_conflicts(
    address: str,
    selected: list[str],
    *,
    collapse: Literal["auto", "max"],
) -> bool:
    """Return whether a candidate conflicts under the active collapse mode."""

    if collapse == "max":
        return _inside_selected(address, selected)
    return _conflicts_selected(address, selected)


def _selection_batch(
    address: str,
    analysis: CollapseAnalysis,
    selected: list[str],
    *,
    collapse: Literal["auto", "max"],
) -> tuple[str, ...]:
    """Return peer addresses that should be selected together.

    Parameters
    ----------
    address:
        Candidate address from the greedy ranking.
    analysis:
        Precomputed collapse analysis.
    selected:
        Addresses already selected for collapse.
    collapse:
        Active collapse mode.

    Returns
    -------
    tuple[str, ...]
        The candidate plus eligible structural sibling peers when a repeated
        peer group exists.
    """

    signal = analysis.signals[address]
    if signal.peer_count <= 1:
        branch_batch = _output_junction_batch(address, analysis, selected, collapse=collapse)
        return branch_batch if len(branch_batch) > 1 else (address,)
    for group in analysis.peer_groups.values():
        if address not in group:
            continue
        return tuple(
            peer_address
            for peer_address in group
            if analysis.signals[peer_address].eligible
            and not _selection_conflicts(peer_address, selected, collapse=collapse)
        )
    return (address,)


def _output_junction_batch(
    address: str,
    analysis: CollapseAnalysis,
    selected: list[str],
    *,
    collapse: Literal["auto", "max"],
) -> tuple[str, ...]:
    """Return sibling modules that feed the same external junction.

    Parameters
    ----------
    address:
        Candidate module address.
    analysis:
        Precomputed collapse analysis.
    selected:
        Addresses already selected for collapse.
    collapse:
        Active collapse mode.

    Returns
    -------
    tuple[str, ...]
        Eligible sibling addresses that feed at least one shared external
        multi-parent junction.
    """

    signal = analysis.signals[address]
    if not signal.output_junctions:
        return (address,)
    junctions = set(signal.output_junctions)
    parent = _address_parent(address)
    batch = [
        peer_address
        for peer_address, peer_signal in analysis.signals.items()
        if peer_signal.eligible
        and _address_parent(peer_address) == parent
        and junctions.intersection(peer_signal.output_junctions)
        and not _selection_conflicts(peer_address, selected, collapse=collapse)
    ]
    return tuple(sorted(batch))


def _address_parent(address: str) -> str | None:
    """Return the dotted parent address for a module address.

    Parameters
    ----------
    address:
        Module address.

    Returns
    -------
    str | None
        Parent address, or ``"self"`` for top-level children.
    """

    if "." not in address:
        return "self"
    return address.rsplit(".", 1)[0]

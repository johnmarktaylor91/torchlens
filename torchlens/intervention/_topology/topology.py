"""Topology comparison + supergraph construction for multi-trace bundles.

Given N ``Trace`` instances of the same architecture (or close variants)
this module produces:

* :func:`compare_topology` -- a pairwise structural diff between two Traces.
* :class:`TopologyDiff` -- the result type for the pairwise comparison.
* :class:`Supergraph` (and :class:`SupergraphNode`) -- the union of N graphs,
  with each node carrying which traces traversed it plus per-trace Layer
  pointers.
* :func:`build_supergraph` -- the constructor used by intervention ``Bundle``.

Matching is intentionally simple: a linear scan in topological order with a
greedy fingerprint match. The fingerprint is ``(module, func_name)``
-- the same module address and the same function under the hood. Topological
position breaks ties when a fingerprint repeats (e.g. multiple ``relu`` calls
in the same block).

This catches the common cases (same model, different inputs, conditional
branches that fire or not) without needing graph-isomorphism machinery.
Models whose graphs disagree in subtle ways -- e.g. operands swapped on a
commutative op, identical fingerprints in a different order -- will not match
perfectly; we document the limitation and let downstream errors surface
clearly.
"""

from __future__ import annotations

import heapq
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...data_classes.layer import Layer
    from ...data_classes.trace import Trace


# A fingerprint is (module, func_name).  Both fields normalize to
# strings; ``module`` is None when the op was logged outside of a
# named submodule (we map that to the empty string so equality works cleanly).
Fingerprint = tuple[str, str]

# ``func_name`` of the op a module-boundary edit inserts between the edited op
# and its consumers (the validation invariants key on the same value).
_EDIT_FUNC_NAME = "intervention_replacement"


def _fingerprint(layer: Layer) -> Fingerprint:
    """Return the structural fingerprint used for cross-trace node matching.

    Composed of ``(module or "", func_name or "")`` -- both
    coerced to strings.  Module pass labels (e.g. ``"fc1:1"``) include the
    pass index so multi-call modules disambiguate naturally.
    """

    mod = layer.module if layer.module is not None else ""
    func = layer.func_name if layer.func_name is not None else ""
    return (str(mod), str(func))


def _shape_excluding_batch(layer: Layer) -> tuple[int, ...] | None:
    """Return the layer's tensor shape excluding the leading (batch) dim.

    Returns ``None`` if the shape is unavailable. 0-d tensors yield ``()``;
    1-d tensors yield ``()`` because the single dim is treated as the batch
    dim. (This matches the bundle's stacking semantics.)
    """

    shape = getattr(layer, "shape", None)
    if shape is None:
        return None
    shape_t = tuple(shape)
    if len(shape_t) == 0:
        return ()
    return shape_t[1:]


def _ordered_layers(trace: Trace) -> list[Layer]:
    """Return layers in the trace's stored topological order.

    Parameters
    ----------
    trace:
        Trace whose ordered layers should be returned.

    Returns
    -------
    list[Layer]
        Ordered layer list.
    """

    return list(trace.layer_logs.values())


def _structural_parent_labels(layer: Layer, layers_by_label: dict[str, Layer]) -> set[str]:
    """Return a layer's parent labels, seeing through intervention-replacement ops.

    A module-boundary edit is recorded as its own ``intervention_replacement``
    op between the edited op and its consumers, and every later op is
    renumbered. The edit changes a value, not the structure, so for alignment
    a consumer's parent is the op the edit replaced (the edit op's own
    parents), the same folding the rerun divergence hash applies to value-only
    edit nodes. Without it the consumer of an edited module output never
    aligns with its counterpart in an unedited trace.

    Parameters
    ----------
    layer:
        Layer whose parents should be resolved.
    layers_by_label:
        The layer's own trace, keyed by layer label.

    Returns
    -------
    set[str]
        Parent labels with every intervention-replacement parent replaced by
        its own parents.
    """

    parents: set[str] = set()
    pending = [str(parent) for parent in getattr(layer, "parents", ())]
    seen: set[str] = set()
    while pending:
        label = pending.pop()
        if label in seen:
            continue
        seen.add(label)
        parent = layers_by_label.get(label)
        if getattr(parent, "func_name", None) == _EDIT_FUNC_NAME:
            pending.extend(str(grandparent) for grandparent in getattr(parent, "parents", ()))
            continue
        parents.add(label)
    return parents


def _matched_parent_labels(
    layer: Layer,
    *,
    matched_labels: set[str],
    layers_by_label: dict[str, Layer],
) -> set[str]:
    """Return parent labels that have already been aligned.

    Parameters
    ----------
    layer:
        Layer whose parents should be filtered.
    matched_labels:
        Parent labels already aligned earlier in topological order.
    layers_by_label:
        The layer's own trace, keyed by layer label (see
        :func:`_structural_parent_labels`).

    Returns
    -------
    set[str]
        Parent labels that are already aligned.
    """

    return {
        parent
        for parent in _structural_parent_labels(layer, layers_by_label)
        if parent in matched_labels
    }


def _find_alignment(
    reference_layers: list[Layer],
    candidate_layers: list[Layer],
) -> list[tuple[int, int]]:
    """Return monotone, parent-compatible layer alignments.

    Parameters
    ----------
    reference_layers:
        Canonical reference-layer sequence.
    candidate_layers:
        Candidate layer sequence to align against the reference.

    Returns
    -------
    list[tuple[int, int]]
        Matched ``(reference_index, candidate_index)`` pairs.
    """

    reference_by_label = {str(layer.layer_label): layer for layer in reference_layers}
    candidate_by_label = {str(layer.layer_label): layer for layer in candidate_layers}
    fingerprint_to_candidate_indices: dict[Fingerprint, deque[int]] = {}
    for idx, layer in enumerate(candidate_layers):
        fingerprint_to_candidate_indices.setdefault(_fingerprint(layer), deque()).append(idx)

    matched_pairs: list[tuple[int, int]] = []
    matched_reference_to_candidate: dict[str, str] = {}
    matched_candidate_labels: set[str] = set()
    last_candidate_index = -1

    for reference_idx, reference_layer in enumerate(reference_layers):
        candidates = fingerprint_to_candidate_indices.get(_fingerprint(reference_layer))
        if not candidates:
            continue
        while candidates and candidates[0] <= last_candidate_index:
            candidates.popleft()
        if not candidates:
            continue

        mapped_reference_parents = {
            matched_reference_to_candidate[parent]
            for parent in _matched_parent_labels(
                reference_layer,
                matched_labels=set(matched_reference_to_candidate),
                layers_by_label=reference_by_label,
            )
        }
        selected_candidate_index: int | None = None
        for candidate_index in candidates:
            candidate_layer = candidate_layers[candidate_index]
            candidate_parent_labels = _matched_parent_labels(
                candidate_layer,
                matched_labels=matched_candidate_labels,
                layers_by_label=candidate_by_label,
            )
            if candidate_parent_labels == mapped_reference_parents:
                selected_candidate_index = candidate_index
                break
        if selected_candidate_index is None:
            continue
        while candidates and candidates[0] <= selected_candidate_index:
            candidates.popleft()
        last_candidate_index = selected_candidate_index
        candidate_layer = candidate_layers[selected_candidate_index]
        matched_pairs.append((reference_idx, selected_candidate_index))
        matched_reference_to_candidate[str(reference_layer.layer_label)] = str(
            candidate_layer.layer_label
        )
        matched_candidate_labels.add(str(candidate_layer.layer_label))
    return matched_pairs


def _insert_preferred_name(
    order: list[str],
    positions: dict[str, int],
    name: str,
    *,
    insertion_pos: int,
) -> None:
    """Insert a node into the preferred-order list and refresh cached positions.

    Parameters
    ----------
    order:
        Mutable preferred-order list.
    positions:
        Cached node-position mapping for ``order``.
    name:
        Canonical node name to insert.
    insertion_pos:
        Target insertion index.
    """

    order.insert(insertion_pos, name)
    for index in range(insertion_pos, len(order)):
        positions[order[index]] = index


def _preferential_topological_order(
    node_names: set[str],
    edges: dict[tuple[str, str], set[str]],
    preferred_order: list[str],
) -> list[str]:
    """Return an acyclic topological order nearest to the preferred ordering.

    Parameters
    ----------
    node_names:
        Canonical node names in the supergraph.
    edges:
        Supergraph edge map.
    preferred_order:
        Stable ordering preference accumulated from the member traces.

    Returns
    -------
    list[str]
        Valid topological ordering.

    Raises
    ------
    ValueError
        If the merged supergraph is cyclic.
    """

    indegree = dict.fromkeys(node_names, 0)
    adjacency: dict[str, set[str]] = {name: set() for name in node_names}
    for parent, child in edges:
        if child not in adjacency[parent]:
            adjacency[parent].add(child)
            indegree[child] += 1
    preferred_positions = {name: idx for idx, name in enumerate(preferred_order)}
    fallback_start = len(preferred_positions)
    queue: list[tuple[int, str]] = []
    for name in node_names:
        if indegree[name] == 0:
            heapq.heappush(queue, (preferred_positions.get(name, fallback_start), name))

    ordered: list[str] = []
    while queue:
        _position, name = heapq.heappop(queue)
        ordered.append(name)
        for child in adjacency[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(queue, (preferred_positions.get(child, fallback_start), child))

    if len(ordered) != len(node_names):
        cyclic_edges = sorted(f"{parent}->{child}" for parent, child in edges)
        raise ValueError(
            "Merged bundle supergraph is cyclic; refusing to emit a false topological order. "
            f"Edges: {cyclic_edges}"
        )
    return ordered


def _unique_canonical_name(
    base_name: str,
    *,
    trace_name: str,
    existing_names: set[str],
) -> str:
    """Return a unique canonical node name for an unmatched member node.

    Parameters
    ----------
    base_name:
        Original member-layer label.
    trace_name:
        Bundle member name supplying the unmatched node.
    existing_names:
        Canonical names already claimed in the supergraph.

    Returns
    -------
    str
        Unique canonical name.
    """

    if base_name not in existing_names:
        return base_name
    candidate = f"{base_name}@{trace_name}"
    suffix = 2
    while candidate in existing_names:
        candidate = f"{base_name}@{trace_name}:{suffix}"
        suffix += 1
    return candidate


@dataclass(frozen=True)
class TopologyDiff:
    """Result of :func:`compare_topology` for two Traces.

    Attributes
    ----------
    matched:
        Pairs of node names that aligned across the two traces, in
        topological-order on ``a``.
    unmatched_a:
        Node names that appear only in trace ``a``.
    unmatched_b:
        Node names that appear only in trace ``b``.
    is_identical:
        ``True`` iff both unmatched lists are empty.
    """

    matched: list[tuple[str, str]]
    unmatched_a: list[str]
    unmatched_b: list[str]
    is_identical: bool


def compare_topology(a: Trace, b: Trace) -> TopologyDiff:
    """Compare two Traces structurally.

    Two nodes match when their :func:`_fingerprint` values agree -- i.e.
    they are inside the same module pass and run the same torch function.
    The match algorithm is a greedy linear scan in topological order: for
    each node in ``a``, we consume the next ``b`` node whose fingerprint
    matches and which has not been claimed yet.  Anything not consumed on
    either side becomes ``unmatched_a`` / ``unmatched_b``.

    Shapes (excluding the batch dim) at matched nodes are checked for
    consistency.  A mismatch emits a :func:`warnings.warn` but still leaves
    the nodes paired -- the bundle will raise a clearer error later if
    stacked accessors hit the disagreement.

    Limitations: the simple fingerprint match does not solve graph
    isomorphism.  If the same fingerprint appears multiple times in
    different orders across traces (rare in practice for the same model
    architecture), the greedy scan can mis-pair them.  In those cases the
    user should treat the bundle as divergent and rely on per-trace
    accessors (``.outs``, not ``.out``).
    """

    a_layers = _ordered_layers(a)
    b_layers = _ordered_layers(b)

    matched: list[tuple[str, str]] = []
    for a_idx, b_idx in _find_alignment(a_layers, b_layers):
        a_layer = a_layers[a_idx]
        b_layer = b_layers[b_idx]
        a_shape = _shape_excluding_batch(a_layer)
        b_shape = _shape_excluding_batch(b_layer)
        if a_shape is not None and b_shape is not None and a_shape != b_shape:
            warnings.warn(
                f"Shape mismatch at matched node '{a_layer.layer_label}' "
                f"(a={a_shape}, b={b_shape}); pairing kept but stacked accessors"
                " will raise.",
                stacklevel=2,
            )
        matched.append((a_layer.layer_label, b_layer.layer_label))

    matched_a_names = {pair[0] for pair in matched}
    matched_b_names = {pair[1] for pair in matched}
    unmatched_a = [
        layer.layer_label for layer in a_layers if layer.layer_label not in matched_a_names
    ]
    unmatched_b = [
        layer.layer_label for layer in b_layers if layer.layer_label not in matched_b_names
    ]

    return TopologyDiff(
        matched=matched,
        unmatched_a=unmatched_a,
        unmatched_b=unmatched_b,
        is_identical=(not unmatched_a and not unmatched_b),
    )


@dataclass
class SupergraphNode:
    """One node in the union supergraph.

    Attributes
    ----------
    name:
        Canonical supergraph node name. Equal to one of the per-trace
        ``Layer.layer_label`` values (chosen from the first trace that
        contributed the node).
    fingerprint:
        ``(module, func_name)`` tuple used for matching.
    traces:
        Ordered list of trace names that traversed this node, preserving
        bundle order.
    layer_refs:
        Maps each trace name to the ``Layer`` for that trace at this node.
    op_type:
        Representative function name (taken from the first contributing
        trace's Layer).
    module_path:
        Representative ``module`` (or ``None`` if not in a module).
    module_type:
        Representative module class name when available from ``Trace.modules``.
    comparisons:
        Named per-node comparison values stamped by
        ``Bundle.store_comparison`` (comparison name -> member name ->
        distance). Session-level: stored on the built supergraph so queries
        and renders read instead of recomputing; never serialized.
    """

    name: str
    fingerprint: Fingerprint
    traces: list[str] = field(default_factory=list)
    layer_refs: dict[str, Layer] = field(default_factory=dict)
    op_type: str = ""
    module_path: str | None = None
    module_type: str | None = None
    comparisons: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class Supergraph:
    """Union of N Trace graphs.

    Attributes
    ----------
    nodes:
        Maps canonical node name -> :class:`SupergraphNode`.
    edges:
        Maps an edge (parent_name, child_name) -> set of trace names that
        traversed it. Stored as ``edge_key -> set[str]`` rather than a multi-
        graph for compactness.
    topological_order:
        The canonical node names in a stable order compatible with all
        contributing traces.  Constructed by overlaying each trace's order;
        unmatched nodes are placed where they fit relative to their nearest
        matched neighbour, falling back to "after the last matched node"
        otherwise.
    """

    nodes: dict[str, SupergraphNode] = field(default_factory=dict)
    edges: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    topological_order: list[str] = field(default_factory=list)


def _module_type_for_layer(trace: Trace, layer: Layer) -> str | None:
    """Return the module class name for a layer's containing module.

    Parameters
    ----------
    trace:
        Model log that owns ``layer``.
    layer:
        Layer whose module class should be resolved.

    Returns
    -------
    str | None
        Module class name when the module accessor has the containing module.
    """

    module_path = getattr(layer, "module", None)
    if module_path is None:
        return None
    modules = getattr(trace, "modules", None)
    if modules is None:
        return None
    module_key = str(module_path).split(":", maxsplit=1)[0]
    try:
        module_log = modules[module_key]
    except (KeyError, TypeError):
        return None
    module_type = getattr(module_log, "class_name", None)
    return str(module_type) if module_type else None


def build_supergraph(traces: list[Trace], names: list[str]) -> Supergraph:
    """Build the union supergraph from N Traces.

    The construction proceeds in three ops:

    1. **Canonical-name assignment.** For every (trace, layer) we compute a
       fingerprint occurrence index.  The same (fingerprint,
       occurrence-index) across traces resolves to a single canonical node
       name (the layer_label from the first trace that contributed it).

    2. **Node payloads.** Walk each trace's layer_logs in order, populating
       :class:`SupergraphNode` entries with the per-trace Layer refs and
       the trace coverage list.

    3. **Edges + topological order.** Merge per-trace adjacency and per-
       trace ordering into a single canonical sequence.  Unmatched nodes
       (those traversed by only some traces) keep their relative position
       from the trace that produced them.
    """

    if len(traces) != len(names):
        raise ValueError(
            f"build_supergraph expected len(traces)==len(names), got {len(traces)} vs {len(names)}"
        )

    super_g = Supergraph()
    ordered_layers_by_trace = [_ordered_layers(trace) for trace in traces]
    reference_layers = ordered_layers_by_trace[0] if ordered_layers_by_trace else []
    canonical_by_trace: list[dict[str, str]] = []
    preferred_order: list[str] = []
    preferred_positions: dict[str, int] = {}

    # Pass 1: resolve each trace's layers onto canonical names.
    for trace_idx, trace in enumerate(traces):
        trace_name = names[trace_idx]
        layers = ordered_layers_by_trace[trace_idx]
        canonical_for_layer: dict[str, str] = {}

        if trace_idx == 0:
            for layer in layers:
                canonical_for_layer[str(layer.layer_label)] = str(layer.layer_label)
        else:
            matched_pairs = _find_alignment(reference_layers, layers)
            matched_candidate_indices = {
                candidate_idx for _reference_idx, candidate_idx in matched_pairs
            }
            for reference_idx, candidate_idx in matched_pairs:
                canonical_for_layer[str(layers[candidate_idx].layer_label)] = str(
                    reference_layers[reference_idx].layer_label
                )
            existing_names = set(super_g.nodes)
            existing_names.update(canonical_for_layer.values())
            for layer_idx, layer in enumerate(layers):
                if layer_idx in matched_candidate_indices:
                    continue
                canonical = _unique_canonical_name(
                    str(layer.layer_label),
                    trace_name=trace_name,
                    existing_names=existing_names,
                )
                canonical_for_layer[str(layer.layer_label)] = canonical
                existing_names.add(canonical)
        canonical_by_trace.append(canonical_for_layer)

        # Pass 2: node payloads + preferred stable ordering.
        last_preferred_index = -1
        for layer in layers:
            canonical = canonical_for_layer[str(layer.layer_label)]
            node = super_g.nodes.get(canonical)
            if node is None:
                node = SupergraphNode(
                    name=canonical,
                    fingerprint=_fingerprint(layer),
                    op_type=str(layer.func_name) if layer.func_name is not None else "",
                    module_path=(str(layer.module) if layer.module is not None else None),
                    module_type=_module_type_for_layer(trace, layer),
                )
                super_g.nodes[canonical] = node
            if trace_name not in node.layer_refs:
                node.layer_refs[trace_name] = layer
                node.traces.append(trace_name)

            if canonical in preferred_positions:
                last_preferred_index = preferred_positions[canonical]
                continue
            insertion_pos = last_preferred_index + 1
            _insert_preferred_name(
                preferred_order,
                preferred_positions,
                canonical,
                insertion_pos=insertion_pos,
            )
            last_preferred_index = insertion_pos

    # Pass 3a: edges. Rebuild each trace's adjacency through canonical names.
    for trace_idx, trace in enumerate(traces):
        trace_name = names[trace_idx]
        canonical_for_layer = canonical_by_trace[trace_idx]
        for layer in ordered_layers_by_trace[trace_idx]:
            child_canonical = canonical_for_layer.get(str(layer.layer_label))
            if child_canonical is None:
                continue
            for parent_label in layer.parents:
                parent_layer = trace.layer_logs.get(parent_label)
                if parent_layer is None:
                    try:
                        ref = trace[parent_label]
                    except (KeyError, IndexError):
                        continue
                    parent_no_pass = getattr(ref, "layer_label", None)
                    if parent_no_pass is None:
                        continue
                    parent_layer = trace.layer_logs.get(parent_no_pass)
                if parent_layer is None:
                    continue
                parent_canonical = canonical_for_layer.get(str(parent_layer.layer_label))
                if parent_canonical is None:
                    continue
                super_g.edges.setdefault((parent_canonical, child_canonical), set()).add(trace_name)

    # Pass 3b: derive a valid topological order or fail loudly on cycles.
    super_g.topological_order = _preferential_topological_order(
        set(super_g.nodes),
        super_g.edges,
        preferred_order,
    )
    return super_g

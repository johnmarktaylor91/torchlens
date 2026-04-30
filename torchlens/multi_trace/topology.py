"""Topology comparison + supergraph construction for multi-trace bundles.

Given N ``ModelLog`` instances of the same architecture (or close variants)
this module produces:

* :func:`compare_topology` -- a pairwise structural diff between two ModelLogs.
* :class:`TopologyDiff` -- the result type for the pairwise comparison.
* :class:`Supergraph` (and :class:`SupergraphNode`) -- the union of N graphs,
  with each node carrying which traces traversed it plus per-trace LayerLog
  pointers.
* :func:`build_supergraph` -- the constructor used by intervention ``Bundle``.

Matching is intentionally simple: a linear scan in topological order with a
greedy fingerprint match. The fingerprint is ``(containing_module, func_name)``
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

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..data_classes.layer_log import LayerLog
    from ..data_classes.model_log import ModelLog


# A fingerprint is (containing_module, func_name).  Both fields normalize to
# strings; ``containing_module`` is None when the op was logged outside of a
# named submodule (we map that to the empty string so equality works cleanly).
Fingerprint = Tuple[str, str]


def _fingerprint(layer: "LayerLog") -> Fingerprint:
    """Return the structural fingerprint used for cross-trace node matching.

    Composed of ``(containing_module or "", func_name or "")`` -- both
    coerced to strings.  Module pass labels (e.g. ``"fc1:1"``) include the
    pass index so multi-call modules disambiguate naturally.
    """

    mod = layer.containing_module if layer.containing_module is not None else ""
    func = layer.func_name if layer.func_name is not None else ""
    return (str(mod), str(func))


def _shape_excluding_batch(layer: "LayerLog") -> Optional[Tuple[int, ...]]:
    """Return the layer's tensor shape excluding the leading (batch) dim.

    Returns ``None`` if the shape is unavailable. 0-d tensors yield ``()``;
    1-d tensors yield ``()`` because the single dim is treated as the batch
    dim. (This matches the bundle's stacking semantics.)
    """

    shape = getattr(layer, "tensor_shape", None)
    if shape is None:
        return None
    shape_t = tuple(shape)
    if len(shape_t) == 0:
        return ()
    return shape_t[1:]


@dataclass(frozen=True)
class TopologyDiff:
    """Result of :func:`compare_topology` for two ModelLogs.

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

    matched: List[Tuple[str, str]]
    unmatched_a: List[str]
    unmatched_b: List[str]
    is_identical: bool


def compare_topology(a: "ModelLog", b: "ModelLog") -> TopologyDiff:
    """Compare two ModelLogs structurally.

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
    accessors (``.activations``, not ``.activation``).
    """

    a_layers = list(a.layer_logs.values())
    b_layers = list(b.layer_logs.values())

    matched: List[Tuple[str, str]] = []
    consumed_b: Set[int] = set()

    # For each fingerprint we maintain a queue of available indices in b.
    fingerprint_to_b_indices: Dict[Fingerprint, List[int]] = {}
    for idx, layer in enumerate(b_layers):
        fingerprint_to_b_indices.setdefault(_fingerprint(layer), []).append(idx)

    for a_layer in a_layers:
        fp = _fingerprint(a_layer)
        candidates = fingerprint_to_b_indices.get(fp)
        if not candidates:
            continue
        # Take the earliest unconsumed b match.
        b_idx = candidates.pop(0)
        consumed_b.add(b_idx)
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
        ``LayerLog.layer_label`` values (chosen from the first trace that
        contributed the node).
    fingerprint:
        ``(containing_module, func_name)`` tuple used for matching.
    traces:
        Ordered list of trace names that traversed this node, preserving
        bundle order.
    layer_refs:
        Maps each trace name to the ``LayerLog`` for that trace at this node.
    op_type:
        Representative function name (taken from the first contributing
        trace's LayerLog).
    module_path:
        Representative ``containing_module`` (or ``None`` if not in a module).
    """

    name: str
    fingerprint: Fingerprint
    traces: List[str] = field(default_factory=list)
    layer_refs: Dict[str, "LayerLog"] = field(default_factory=dict)
    op_type: str = ""
    module_path: Optional[str] = None


@dataclass
class Supergraph:
    """Union of N ModelLog graphs.

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

    nodes: Dict[str, SupergraphNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    topological_order: List[str] = field(default_factory=list)


def _per_trace_fingerprint_to_canonical(
    canonical_assignments: Dict[Tuple[str, Fingerprint, int], str],
    trace_name: str,
    layer: "LayerLog",
    fp_seen_count: Dict[Tuple[str, Fingerprint], int],
) -> str:
    """Compute the canonical node name for a per-trace LayerLog.

    Each fingerprint may legitimately repeat (e.g. multiple ``relu`` calls
    in the same module pass).  We disambiguate by pairing the trace name
    with the (fingerprint, occurrence-index) tuple.  ``canonical_assignments``
    persists those decisions across the build pass.
    """

    fp = _fingerprint(layer)
    occurrence = fp_seen_count.get((trace_name, fp), 0)
    fp_seen_count[(trace_name, fp)] = occurrence + 1
    key = (trace_name, fp, occurrence)
    return canonical_assignments[key]


def build_supergraph(traces: List["ModelLog"], names: List[str]) -> Supergraph:
    """Build the union supergraph from N ModelLogs.

    The construction proceeds in three passes:

    1. **Canonical-name assignment.** For every (trace, layer) we compute a
       fingerprint occurrence index.  The same (fingerprint,
       occurrence-index) across traces resolves to a single canonical node
       name (the layer_label from the first trace that contributed it).

    2. **Node payloads.** Walk each trace's layer_logs in order, populating
       :class:`SupergraphNode` entries with the per-trace LayerLog refs and
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

    # Pass 1a: collect per-trace ordered (layer_label, fingerprint, occurrence)
    per_trace_layers: List[List[Tuple[str, Fingerprint, int]]] = []
    fp_occurrence_per_trace: List[Dict[Fingerprint, int]] = []
    for trace in traces:
        ordered: List[Tuple[str, Fingerprint, int]] = []
        seen: Dict[Fingerprint, int] = {}
        for layer in trace.layer_logs.values():
            fp = _fingerprint(layer)
            occ = seen.get(fp, 0)
            seen[fp] = occ + 1
            ordered.append((layer.layer_label, fp, occ))
        per_trace_layers.append(ordered)
        fp_occurrence_per_trace.append(seen)

    # Pass 1b: deterministic canonical names. For each (fingerprint, occ)
    # we use the first trace's layer_label that produced it.
    canonical_name: Dict[Tuple[Fingerprint, int], str] = {}
    for trace_idx, ordered in enumerate(per_trace_layers):
        for layer_label, fp, occ in ordered:
            key = (fp, occ)
            if key not in canonical_name:
                canonical_name[key] = layer_label

    super_g = Supergraph()

    # Pass 2: node payloads
    for trace_idx, trace in enumerate(traces):
        trace_name = names[trace_idx]
        for layer in trace.layer_logs.values():
            fp = _fingerprint(layer)
            # Recompute occurrence in scan order for this trace.
            # NB: per_trace_layers[trace_idx] preserves the order, so we
            # zip-match by position.
            pass  # actual zip below
        # Walk per_trace_layers[trace_idx] alongside the layer_logs
        # iteration order (they're constructed from the same iteration).
        for (_, fp, occ), layer in zip(per_trace_layers[trace_idx], trace.layer_logs.values()):
            cname = canonical_name[(fp, occ)]
            node = super_g.nodes.get(cname)
            if node is None:
                node = SupergraphNode(
                    name=cname,
                    fingerprint=fp,
                    op_type=str(layer.func_name) if layer.func_name is not None else "",
                    module_path=(
                        str(layer.containing_module)
                        if layer.containing_module is not None
                        else None
                    ),
                )
                super_g.nodes[cname] = node
            if trace_name not in node.layer_refs:
                node.layer_refs[trace_name] = layer
                node.traces.append(trace_name)

    # Pass 3a: edges. We rebuild per-trace edges using each trace's
    # parent_layers structure, mapped through canonical names.
    for trace_idx, trace in enumerate(traces):
        trace_name = names[trace_idx]
        # Build trace-local layer_label -> (fingerprint, occurrence) lookup
        local_lookup: Dict[str, Tuple[Fingerprint, int]] = {
            label: (fp, occ) for label, fp, occ in per_trace_layers[trace_idx]
        }
        for layer in trace.layer_logs.values():
            child_label = layer.layer_label
            child_key = local_lookup.get(child_label)
            if child_key is None:
                continue
            child_canonical = canonical_name[child_key]
            for parent_label in layer.parent_layers:
                parent_layer = trace.layer_logs.get(parent_label)
                if parent_layer is None:
                    # parent_layers are typically pass-qualified strings; map
                    # back to no_pass labels via the source ModelLog index.
                    try:
                        ref = trace[parent_label]
                    except (KeyError, IndexError):
                        continue
                    parent_no_pass = getattr(ref, "layer_label_no_pass", None)
                    if parent_no_pass is None:
                        continue
                    parent_layer = trace.layer_logs.get(parent_no_pass)
                if parent_layer is None:
                    continue
                parent_key = local_lookup.get(parent_layer.layer_label)
                if parent_key is None:
                    continue
                parent_canonical = canonical_name[parent_key]
                edge_key = (parent_canonical, child_canonical)
                edge_traces = super_g.edges.setdefault(edge_key, set())
                edge_traces.add(trace_name)

    # Pass 3b: stable topological order.  Start from the first trace's
    # canonical-name sequence; for each subsequent trace, insert any
    # canonical names not yet placed by anchoring on the nearest already-
    # placed neighbour.
    placed: List[str] = []
    placed_set: Set[str] = set()
    if per_trace_layers:
        for _, fp, occ in per_trace_layers[0]:
            cname = canonical_name[(fp, occ)]
            if cname not in placed_set:
                placed.append(cname)
                placed_set.add(cname)

    for trace_idx in range(1, len(traces)):
        # Walk this trace's canonical names; for each not-yet-placed one,
        # insert it just after the nearest previously-placed canonical name
        # that came before it in this trace, or append if none exists.
        last_placed_idx_in_overall = -1
        for _, fp, occ in per_trace_layers[trace_idx]:
            cname = canonical_name[(fp, occ)]
            if cname in placed_set:
                last_placed_idx_in_overall = placed.index(cname)
                continue
            insertion_pos = last_placed_idx_in_overall + 1
            placed.insert(insertion_pos, cname)
            placed_set.add(cname)
            last_placed_idx_in_overall = insertion_pos

    super_g.topological_order = placed
    return super_g

"""Freeze-time conversion of relation staging containers to immutable views.

The M6 relations wave (trace_core_design.md sections 3.2/3.5, FORK-1
decided 2026-08-12). During postprocess (steps 0-20) relation fields are real
mutable builtins — the staging plane. At the core freeze point each family
converts exactly once:

* ``parents``/``children`` (the dataflow family) project into the core's ONE
  canonical edge-occurrence table — one edge occurrence per attributed
  argument position (from ``parent_arg_positions``), parents-major insertion
  order, CSR indexes over edge ids. The per-row cells become the ``_CSR``
  sentinel; the facade descriptors rematerialize an interned ``tuple`` view
  on first access and cache it back into the row.
* The remaining label-sequence families convert in place to per-trace
  interned ``tuple`` views (one shared object per distinct value — empties
  collapse to THE shared empty tuple).
* Label-set families convert to interned ``frozenset`` views. The two big
  ancestor closures stay bitset-encoded (``backends/torch/ops.py``) and
  materialize cached frozensets through their compatibility properties.

The conversion is differential by construction: the dataflow family is
rebuilt from the edge table and compared against the staging containers
BEFORE the staging cells die. A row whose rebuilt view does not byte-match
its staging container (children order is a chronological fact the
parents-major edge order cannot always reproduce; a reference label that
does not resolve into the live row domain) keeps an explicit interned view
cell instead — the edge table stays authoritative only where it provably
reproduces the public surface. The exception count is reported on the
returned stats so gates can assert it stays at zero for the oracle axes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .groups import GroupRef, MembershipGroups
from .op_store import _CSR, _MISSING

if TYPE_CHECKING:
    from .core import TraceCore
    from .op_store import OpRowStore

#: Op label-sequence relation fields stored as interned tuple views at freeze.
#: ``parents``/``children`` are NOT here — they clear into the edge table.
OP_TUPLE_VIEW_FIELDS: tuple[str, ...] = (
    "internal_source_parents",
    "in_conditionals",
    "conditional_branch_stack",
    "conditional_entry_children",
    "conditional_then_children",
    "conditional_else_children",
    "modules",
    "module_call_stack",
    "input_to_module_calls",
    "output_of_modules",
    "output_of_module_calls",
    "parent_params",
    "_param_barcodes",
    "_param_logs",
    "_edge_uses",
)

#: Op label-set relation fields normalized to interned frozenset views at
#: freeze. Since r8 R60-2 these are ALSO bitset-compacted right before the
#: seal (see ``OP_BITSET_VIEW_FIELDS``); this conversion is the belt for any
#: cell still holding a raw staging set at freeze time (bitset cells are
#: skipped by the exact-type check).
OP_FROZENSET_VIEW_FIELDS: tuple[str, ...] = (
    "input_ancestors",
    "output_descendants",
)

#: The two CSR-backed dataflow fields (cells become ``_CSR`` at freeze).
OP_DATAFLOW_FIELDS: tuple[str, ...] = ("parents", "children")

#: The bitset-backed ancestor closures (frozen materialization lives with the
#: bitset machinery in ``backends/torch/ops.py``; listed here so the view
#: universe is declared in one place). r8 R60-2 widened the pair to all four
#: label-set closures: per-op frozensets kept ``input_ancestors``/
#: ``output_descendants`` as the dominant O(N*L) retained-memory term.
OP_BITSET_VIEW_FIELDS: tuple[str, ...] = (
    "root_ancestors",
    "internal_source_ancestors",
    "input_ancestors",
    "output_descendants",
)

#: Edge family name in ``TraceCore.edges`` for forward dataflow.
DATAFLOW_FAMILY = "dataflow"

#: Group-membership relation fields (M7): each cell becomes THE one shared
#: ``GroupRef`` of its group; reads resolve to the group's cached immutable
#: view (LIVE through removal scrub, which rebinds the group row once).
OP_GROUP_VIEW_FIELDS: dict[str, type] = {
    "equivalent_ops": frozenset,
    "recurrent_ops": tuple,
}

#: Layer-side stored relation fields converted to the same view types (the
#: aggregate builds at step 15.5 alias or copy the op staging containers, so
#: they predate the op-cell conversion). ``parents``/``children`` on Layer
#: are computed properties, not storage, and are handled in ``layer.py``.
LAYER_TUPLE_VIEW_FIELDS: tuple[str, ...] = (
    "internal_source_parents",
    "in_conditionals",
    "conditional_role_stacks",
    "conditional_branch_stack_ops",
    "conditional_entry_children",
    "conditional_then_children",
    "conditional_else_children",
    "modules",
    "module_call_stack",
    "input_to_module_calls",
    "output_of_modules",
    "output_of_module_calls",
    "parent_params",
    "_param_barcodes",
    "_param_logs",
)

LAYER_FROZENSET_VIEW_FIELDS: tuple[str, ...] = (
    "input_ancestors",
    "output_descendants",
)


@dataclass
class RelationFreezeStats:
    """Outcome accounting for one relation freeze pass."""

    edges: int = 0
    csr_cleared_cells: int = 0
    explicit_view_cells: int = 0
    unresolved_labels: int = 0


def intern_view(value: Any, pool: dict[Any, Any]) -> Any:
    """Return the pooled canonical instance for one immutable view.

    Unhashable views (tuples carrying mutable records such as
    ``ConditionalRoleRef``) are returned unpooled — still immutable at the
    container level, just not shared.
    """

    try:
        canonical = pool.get(value)
    except TypeError:
        return value
    if canonical is None:
        pool[value] = value
        return value
    return canonical


def tuple_view(value: Any, pool: dict[Any, Any]) -> Any:
    """Convert one staging sequence to an interned tuple view."""

    if value.__class__ is not tuple:
        value = tuple(value)
    return intern_view(value, pool)


def frozenset_view(value: Any, pool: dict[Any, Any]) -> Any:
    """Convert one staging set to an interned frozenset view."""

    if value.__class__ is not frozenset:
        value = frozenset(value)
    return intern_view(value, pool)


def convert_row_cells(
    rows: list[list[Any]],
    fid_by_name: dict[str, int],
    pool: dict[Any, Any],
    *,
    tuple_fields: tuple[str, ...] = OP_TUPLE_VIEW_FIELDS,
    frozenset_fields: tuple[str, ...] = OP_FROZENSET_VIEW_FIELDS,
) -> None:
    """Convert the non-dataflow relation families of row-major cells."""

    for name in tuple_fields:
        fid = fid_by_name.get(name)
        if fid is None:
            continue
        for row_cells in rows:
            value = row_cells[fid]
            if value.__class__ is list or value.__class__ is tuple:
                row_cells[fid] = tuple_view(value, pool)
    for name in frozenset_fields:
        fid = fid_by_name.get(name)
        if fid is None:
            continue
        for row_cells in rows:
            value = row_cells[fid]
            if value.__class__ is set or value.__class__ is frozenset:
                row_cells[fid] = frozenset_view(value, pool)


def freeze_op_relation_views(
    core: TraceCore,
    store: OpRowStore,
    resolve_row: Callable[[str], int | None],
    pool: dict[Any, Any] | None = None,
) -> RelationFreezeStats:
    """Convert every relation family of a building-phase op store.

    Parameters
    ----------
    core:
        Owning trace core (receives the dataflow edge table).
    store:
        Building-phase ``OpRowStore`` (no-op if already frozen).
    resolve_row:
        Resolver from one relation reference label (the exact string staging
        containers use) to the live row id, or ``None`` when the label does
        not resolve to a row of this store.
    pool:
        Optional shared view-intern pool, so sibling record families (the
        Layer aggregates) share one canonical instance per distinct value.

    Returns
    -------
    RelationFreezeStats
        Edge and exception accounting for gate assertions.
    """

    stats = RelationFreezeStats()
    rows = store.rows_building()
    if rows is None:
        return stats
    fid_by_name = store.layout.fid_by_name
    if pool is None:
        pool = {}

    convert_row_cells(rows, fid_by_name, pool)

    parents_fid = fid_by_name["parents"]
    children_fid = fid_by_name["children"]
    pap_fid = fid_by_name["parent_arg_positions"]

    # Pass 1: resolve every distinct reference label once; build the
    # row -> reference-label table used for view rematerialization.
    row_of_label: dict[str, int | None] = {}
    ref_labels: dict[int, str] = {}
    ambiguous_rows: set[int] = set()
    for row_cells in rows:
        for fid in (parents_fid, children_fid):
            value = row_cells[fid]
            if value is _MISSING:
                continue
            for label in value:
                if label in row_of_label:
                    continue
                row = resolve_row(label)
                row_of_label[label] = row
                if row is None:
                    stats.unresolved_labels += 1
                elif row in ref_labels:
                    # Two distinct reference strings for one row: views for
                    # rows touching it cannot be rebuilt unambiguously.
                    ambiguous_rows.add(row)
                else:
                    ref_labels[row] = label

    edges = core.edge_table(DATAFLOW_FAMILY)
    if len(edges):
        raise RuntimeError("dataflow edge table already populated")

    # Pass 2: emit edge occurrences child-major, parents-order within each
    # child, one occurrence per attributed argument position.
    n_rows = len(store)
    for row in range(n_rows):
        row_cells = rows[row]
        parents = row_cells[parents_fid]
        if parents is _MISSING or not parents:
            continue
        pap = row_cells[pap_fid]
        positions_by_parent: dict[str, list[Any]] = {}
        if isinstance(pap, dict):
            for domain in ("args", "kwargs"):
                domain_map = pap.get(domain)
                if not isinstance(domain_map, dict):
                    continue
                for position, parent_label in domain_map.items():
                    positions_by_parent.setdefault(parent_label, []).append((domain, position))
        for parent_label in parents:
            source = row_of_label.get(parent_label)
            if source is None:
                continue
            occurrences = positions_by_parent.get(parent_label)
            if occurrences:
                for occurrence in occurrences:
                    edges.add(source, row, arg_position=occurrence)
            else:
                edges.add(source, row, arg_position=None)
    stats.edges = len(edges)
    edges.freeze(n_rows, n_rows)

    # Pass 3: rebuild both views from the CSR, verify against staging, then
    # clear derivable cells to the _CSR sentinel; keep explicit interned
    # views where the rebuild does not byte-match.
    label_of = ref_labels.get
    rebuilt_parents: dict[int, list[str]] = {row: [] for row in range(n_rows)}
    rebuilt_children: dict[int, list[str]] = {row: [] for row in range(n_rows)}
    for edge_id in range(len(edges)):
        edge = edges.edge(edge_id)
        source_label = label_of(edge.source)
        target_label = label_of(edge.target)
        if source_label is not None and edge.source not in ambiguous_rows:
            target_list = rebuilt_parents[edge.target]
            if not target_list or source_label not in target_list:
                target_list.append(source_label)
        if target_label is not None and edge.target not in ambiguous_rows:
            source_list = rebuilt_children[edge.source]
            if target_label not in source_list:
                source_list.append(target_label)

    for row in range(n_rows):
        row_cells = rows[row]
        for fid, rebuilt in (
            (parents_fid, rebuilt_parents[row]),
            (children_fid, rebuilt_children[row]),
        ):
            staging = row_cells[fid]
            if staging is _MISSING:
                continue
            if not (staging.__class__ is list or staging.__class__ is tuple):
                continue
            if row not in ambiguous_rows and list(staging) == rebuilt:
                row_cells[fid] = _CSR
                stats.csr_cleared_cells += 1
            else:
                row_cells[fid] = tuple_view(staging, pool)
                stats.explicit_view_cells += 1

    store.dataflow_edges = edges
    store.ref_labels = ref_labels
    return stats


def materialize_dataflow_view(store: Any, row: int, field_name: str) -> tuple:
    """Rematerialize one CSR-backed dataflow view for a facade read."""

    edges = store.dataflow_edges
    ref_labels = store.ref_labels
    if edges is None or ref_labels is None:
        return ()
    label_of = ref_labels.get
    seen: list[str] = []
    if field_name == "parents":
        for edge in edges.in_edges(row):
            label = label_of(edge.source)
            if label is not None and label not in seen:
                seen.append(label)
    else:
        for edge in edges.out_edges(row):
            label = label_of(edge.target)
            if label is not None and label not in seen:
                seen.append(label)
    return tuple(seen)


def convert_record_dict_views(
    record_dict: dict[str, Any],
    pool: dict[Any, Any],
    *,
    tuple_fields: tuple[str, ...],
    frozenset_fields: tuple[str, ...],
) -> None:
    """Convert one dict-backed record's stored relation values in place."""

    for name in tuple_fields:
        value = record_dict.get(name)
        if value is None:
            continue
        if value.__class__ is list or value.__class__ is tuple:
            record_dict[name] = tuple_view(value, pool)
    for name in frozenset_fields:
        value = record_dict.get(name)
        if value is None:
            continue
        if value.__class__ is set or value.__class__ is frozenset:
            record_dict[name] = frozenset_view(value, pool)


def convert_group_cells(
    core: TraceCore,
    store: OpRowStore,
    pool: dict[Any, Any],
    refs_by_identity: dict[int, GroupRef],
    keepalive: list[Any],
) -> None:
    """Convert equivalence/recurrence staging cells to shared ``GroupRef``s.

    One group row per distinct membership; every member cell holds THE one
    shared ref. Empty staging containers collapse to the interned empty view
    directly (no group row). ``refs_by_identity`` maps ``id(raw container)``
    to its ref for the caller's Layer pass; ``keepalive`` pins every keyed raw
    container so an id can never be reused while the map is consulted.
    """

    rows = store.rows_building()
    if rows is None:
        return
    fid_by_name = store.layout.fid_by_name
    for name, view_type in OP_GROUP_VIEW_FIELDS.items():
        fid = fid_by_name.get(name)
        if fid is None:
            continue
        family = core.groups.get(name)
        if family is None:
            family = core.groups[name] = MembershipGroups(view_type)
        make_view = tuple_view if view_type is tuple else frozenset_view
        refs_by_value: dict[Any, GroupRef] = {}
        for row_cells in rows:
            value = row_cells[fid]
            cls = value.__class__
            if cls is not set and cls is not list and cls is not tuple and cls is not frozenset:
                continue
            if not value:
                row_cells[fid] = make_view(value, pool)
                continue
            ref = refs_by_identity.get(id(value))
            if ref is None:
                # The rename memo shares ONE canonical container per group by
                # identity; the value key coalesces legacy equal-but-distinct
                # members (pre-memo loads, hand-built records).
                value_key = (name, view_type(value))
                ref = refs_by_value.get(value_key)
                if ref is None:
                    ref = GroupRef(family, family.add(value))
                    refs_by_value[value_key] = ref
                refs_by_identity[id(value)] = ref
                keepalive.append(value)
            row_cells[fid] = ref


def freeze_trace_relation_views(trace: Any) -> RelationFreezeStats | None:
    """Run the whole relation freeze for one finished trace.

    The single entry point shared by the torch postprocess freeze point and
    the preview-backend finalize seams: converts the op store's relation
    families (dataflow into the core edge table, the rest into interned
    views) and the Layer aggregates' stored relation containers. Idempotent —
    a store whose dataflow family is already bound (or that is already
    sealed) is left untouched, so a fork/reload can never double-convert.

    Returns ``None`` when the trace has no core-backed op store or the
    conversion already ran; otherwise the freeze stats.
    """

    core = trace.__dict__.get("_trace_core")
    if core is None or core.ops is None:
        return None
    store = core.ops
    if store.dataflow_edges is not None or store.frozen:
        return None

    # Row resolver: reference labels in staging containers are exactly the
    # final ``layer_label`` of each live entry. ``layer_dict_all_keys`` holds
    # every lookup alias, so dedup by object identity; entries not backed by
    # THIS store (detached copies, foreign records) never resolve.
    label_rows: dict[str, int] = {}
    seen_ids: set[int] = set()
    for entry in getattr(trace, "layer_dict_all_keys", {}).values():
        if id(entry) in seen_ids:
            continue
        seen_ids.add(id(entry))
        if getattr(entry, "_core", None) is not store:
            continue
        label = getattr(entry, "layer_label", None)
        if label is not None:
            label_rows[label] = entry._row

    # Bind the canonical label -> row index on the core (M8): the freeze-time
    # resolver IS the core-side lookup index the Trace-side label maps
    # progressively delegate to (M10 decomposition).
    core.label_rows = label_rows

    pool: dict[Any, Any] = {}
    stats = freeze_op_relation_views(core, store, label_rows.get, pool)

    # M7 group families: one shared GroupRef per membership group. The
    # keepalive list pins the raw containers so the id-keyed map stays sound
    # through the Layer pass below.
    group_refs: dict[int, GroupRef] = {}
    group_keepalive: list[Any] = []
    convert_group_cells(core, store, pool, group_refs, group_keepalive)

    # M7 shared-fact blocks: call-level facts once per FunctionCall group,
    # param facts once per distinct value (the ParamAlias block).
    from .fact_blocks import convert_fact_cells

    convert_fact_cells(store, pool)

    for layer_log in (getattr(trace, "layer_logs", None) or {}).values():
        record_dict = getattr(layer_log, "__dict__", None)
        if record_dict is None:
            continue
        convert_record_dict_views(
            record_dict,
            pool,
            tuple_fields=LAYER_TUPLE_VIEW_FIELDS,
            frozenset_fields=LAYER_FROZENSET_VIEW_FIELDS,
        )
        # Layer aggregates share the ops' canonical equivalence containers;
        # store the group's ONE cached view (Layer is dict-backed until M8,
        # so the value itself is the public read).
        group_value = record_dict.get("equivalent_ops")
        if group_value.__class__ is set:
            ref = group_refs.get(id(group_value))
            record_dict["equivalent_ops"] = (
                ref.view() if ref is not None else frozenset_view(group_value, pool)
            )
    del group_keepalive
    return stats

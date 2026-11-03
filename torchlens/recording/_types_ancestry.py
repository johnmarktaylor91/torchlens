"""Ancestry backfill for cook-owned recording replay projections."""

from __future__ import annotations

__tl_layer__ = "L1"

from typing import Any

from .._ir.op_record import AncestryFacet, OpRecord


def _backfill_cooked_ancestry(events: Any) -> None:
    """Derive the per-op ancestry closures the sparse recorder never tracked.

    Predicate-mode capture appends journal records without ancestry facts
    (the live exhaustive path computes them incrementally per op at capture
    time). A cooked Trace materializes its Op rows straight from these
    records, so without a backfill every cooked row carries empty
    ``root_ancestors`` / ``internal_source_ancestors`` and the
    ``ancestry_closure`` metadata invariant correctly fails on the first
    input layer. Recompute the exact closure the invariant checks, in journal
    order (parents precede children within one sealed pass), over the AMENDED
    view (graph-edge-insertion amendments contribute parents):

    * ``input_ancestors``           = own label for input rows, else parent union;
    * ``internal_source_ancestors`` = ``{self}`` for parentless non-input rows
      (matching the live source-minting convention), else parent union;
    * ``root_ancestors``            = ``input_ancestors | internal_source_ancestors``.

    Mutates ``events`` in place: this runs only on the ``_copy_for_replay``
    projection a cook owns (the sanctioned mutation surface -- postprocess
    graph traversal replaces events on the same projection), never on the
    sealed Recording stream. The fold cache keys on lane lengths, so it is
    explicitly invalidated after the in-place replacement.

    Parameters
    ----------
    events
        Replay-projection ``CaptureEvents`` whose op lane should be
        ancestry-backfilled before Trace postprocessing.
    """

    _backfill_cooked_op_records(events)


def _backfill_cooked_op_records(events: Any) -> None:
    """Replace replay-projection op records with derived ancestry facts.

    Parameters
    ----------
    events
        Cook-owned working projection whose op lane and fold cache are updated.
    """
    from dataclasses import replace as _dc_replace

    folded = events.amended_op_records()
    closures: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    by_raw_label = events.live_index.by_raw_label
    mutated = False
    for index, record in enumerate(folded):
        # Journal op records (OpRecord / compat events) carry layer_type; never an Op.
        layer_type = getattr(record, "layer_type", None)
        if layer_type in (None, "module_enter", "module_exit"):
            continue
        label_raw = record.label_raw
        parent_labels = [edge.parent_label_raw for edge in record.parents]
        if layer_type == "input":
            input_ancestors = frozenset((label_raw,))
            internal_source_ancestors: frozenset[str] = frozenset()
        elif not parent_labels:
            input_ancestors = frozenset()
            internal_source_ancestors = frozenset((label_raw,))
        else:
            input_ancestors = frozenset().union(
                *(closures[parent][0] for parent in parent_labels if parent in closures)
            )
            internal_source_ancestors = frozenset().union(
                *(closures[parent][1] for parent in parent_labels if parent in closures)
            )
        closures[label_raw] = (input_ancestors, internal_source_ancestors)
        ancestry = AncestryFacet(
            input_ancestors=input_ancestors,
            internal_source_ancestors=internal_source_ancestors,
            root_ancestors=input_ancestors | internal_source_ancestors,
            has_internal_source_ancestor=bool(internal_source_ancestors),
        )
        raw_record = events.op_events[index]
        if isinstance(raw_record, OpRecord):
            updated = _dc_replace(raw_record, ancestry=ancestry)
        else:
            updated = _dc_replace(
                raw_record,
                input_ancestors=input_ancestors,
                internal_source_ancestors=internal_source_ancestors,
                root_ancestors=ancestry.root_ancestors,
                has_internal_source_ancestor=ancestry.has_internal_source_ancestor,
            )
        events.op_events[index] = updated
        mutated = True
        if by_raw_label.get(label_raw) is raw_record:
            by_raw_label[label_raw] = updated
    if mutated:
        events._amended_fold_cache = None

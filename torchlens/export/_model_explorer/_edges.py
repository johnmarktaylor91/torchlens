"""Occurrence-preserving ported edges through an explicit label map (D4).

The capture relation mixes spellings within one graph: ``parents`` entries
and ``parent_arg_positions`` refs are bare ``layer_label`` for single-pass
parents and pass-suffixed ``label`` for multi-pass ones (measured 158/158
bare on ResNet-18). The shipped v2 ``if parent in node_ids`` filter survived
only by accident of construction; with site-key-derived ids it dropped every
edge on the model silently, into a viewer that also drops dangling edges
silently. Here every graph builds ONE explicit map keyed by BOTH spellings,
and an unresolvable parent is a typed exporter failure -- never a bare
``continue``.
"""

from __future__ import annotations

from typing import Any

from ._errors import ModelExplorerExportError

__tl_layer__ = "L8"


def build_label_map(entries: list[Any], node_ids: list[str]) -> dict[str, str]:
    """Build the two-spelling label -> node-id map for one graph.

    Every op registers its pass-qualified ``label``; single-pass ops also
    register their bare ``layer_label``. A bare ref to a multi-pass op stays
    deliberately unresolvable: picking a pass silently would fabricate an
    edge (exactly the bug class this map exists to kill).
    """

    label_map: dict[str, str] = {}
    for entry, node_id in zip(entries, node_ids, strict=True):
        label = str(getattr(entry, "label", "") or "")
        if label:
            label_map[label] = node_id
        if int(getattr(entry, "num_passes", 1) or 1) <= 1:
            layer_label = str(getattr(entry, "layer_label", "") or "")
            if layer_label:
                label_map[layer_label] = node_id
    return label_map


def incoming_edges(
    entry: Any,
    label_map: dict[str, str],
    *,
    resolve_missing: str = "raise",
) -> tuple[list[dict[str, str]], list[dict[str, Any]], int]:
    """Build one node's ``incomingEdges`` + ``inputsMetadata``, ports intact.

    Occurrence-preserving from ``parent_arg_positions``: ``x + x`` emits two
    edges; positional args use their recorded position, keyword args their
    recorded kwarg name or path (inventing an order would be fabrication).
    Falls back to the deduplicated ``parents`` relation only when no
    positions were recorded, disclosed as structural (no invented slots).

    Parameters
    ----------
    entry:
        Layer-pass entry (the edge target).
    label_map:
        The graph's two-spelling label -> node-id map.
    resolve_missing:
        ``"raise"`` (default) refuses typed on an unresolvable parent;
        ``"skip"`` counts the omission for cross-graph callers (episode step
        graphs, where the boundary machinery owns the disclosure).

    Returns
    -------
    tuple[list, list, int]
        Edge rows, inputs-metadata rows, and the skipped-parent count.
    """

    # Essential complexity (CC>10 named): the ported and structural edge
    # regimes are ONE decision table over the same recorded relation; two
    # functions would duplicate the resolve/skip/count discipline.
    positions = getattr(entry, "parent_arg_positions", None) or {}
    argument_rows = list((positions.get("args") or {}).items())
    keyword_rows = list((positions.get("kwargs") or {}).items())
    edges: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    skipped = 0
    if argument_rows or keyword_rows:
        argument_rows.sort(key=lambda item: _slot_sort_key(item[0]))
        keyword_rows.sort(key=lambda item: _slot_sort_key(item[0]))
        for slot_key, parent_ref in argument_rows + keyword_rows:
            source_id = _resolve(entry, str(parent_ref), label_map, resolve_missing)
            if source_id is None:
                skipped += 1
                continue
            slot = _slot_string(slot_key)
            edges.append(
                {"sourceNodeId": source_id, "sourceNodeOutputId": "0", "targetNodeInputId": slot}
            )
            metadata.append(
                {"id": slot, "attrs": [{"key": "__tensor_tag", "value": _tensor_tag(slot_key)}]}
            )
        return edges, metadata, skipped
    for parent_ref in getattr(entry, "parents", ()) or ():
        source_id = _resolve(entry, str(parent_ref), label_map, resolve_missing)
        if source_id is None:
            skipped += 1
            continue
        # Structural provenance: no recorded slot, so no inputsMetadata row
        # and no invented targetNodeInputId ordering (memo D4).
        edges.append({"sourceNodeId": source_id, "sourceNodeOutputId": "0"})
    return edges, metadata, skipped


def _resolve(
    entry: Any, parent_ref: str, label_map: dict[str, str], resolve_missing: str
) -> str | None:
    """Resolve one parent ref through the map, refusing typed on a miss."""

    source_id = label_map.get(parent_ref)
    if source_id is not None:
        return source_id
    if resolve_missing == "skip":
        return None
    raise ModelExplorerExportError(
        f"Parent reference {parent_ref!r} of op {getattr(entry, 'label', '<unknown>')!r} "
        "does not resolve against either label spelling in this graph; emitting the "
        "graph would silently drop the edge in Model Explorer",
        code="model_explorer_parent_unresolvable",
        remedy=(
            "this indicates a TorchLens capture-relation bug, not a user error; "
            "report it with the trace's summary() and the offending op label"
        ),
        parent_ref=parent_ref,
    )


def _slot_sort_key(slot_key: Any) -> tuple[Any, ...]:
    """Return a deterministic sort key for recorded arg positions/paths."""

    if isinstance(slot_key, tuple):
        return tuple(str(part) for part in slot_key)
    return (str(slot_key),)


def _slot_string(slot_key: Any) -> str:
    """Spell one recorded position/path as a stable port id (``0``, ``0.1``, ``cond``)."""

    if isinstance(slot_key, tuple):
        return ".".join(str(part) for part in slot_key)
    return str(slot_key)


def _tensor_tag(slot_key: Any) -> str:
    """Return the ``__tensor_tag`` input name for one recorded slot."""

    if isinstance(slot_key, tuple):
        return str(slot_key[0])
    return str(slot_key)

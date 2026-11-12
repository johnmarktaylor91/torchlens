"""Step 6 body: connect buffer sources, merge re-logged reads, number buffer passes.

``control_flow._fix_buffer_layers`` (the step's patchable entry) calls
:func:`fix_buffer_layers`. The common case, every buffer read once from a
registered address with no recorded source, is a column pass over the raw row
cells (see ``_seal_columns``): it writes ``buffer_pass`` and the trace's
per-address call counts and nothing else, as the per-buffer body does when no
buffer is rewired and every merge group has one member. Any other trace runs
the per-buffer body, which wires sources, merges same-version duplicates and
re-derives the affected ancestry.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch

from .._data_classes._op_out_read import _synthesized_out_read_due
from .._trace_core.cell_states import LazyCell
from ._buffer_source_wiring import _wire_buffer_to_source
from ._seal_columns import immutable_markers, markers, plain_column, seal_rows

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace

__all__ = ("MergeHelpers", "fix_buffer_layers")


class MergeHelpers(NamedTuple):
    """The per-buffer body's merge and repair steps (they live in ``control_flow``)."""

    merge_same_version: Callable[..., None]
    finish_removals: Callable[..., None]
    repair_ancestry: Callable[..., None]
    repair_descendants: Callable[..., None]


def fix_buffer_layers(trace: Trace, helpers: MergeHelpers) -> None:
    """Run step 6: the column pass when it applies, else the per-buffer body."""

    if not _fix_by_column(trace):
        _fix_by_facade(trace, helpers)


def _out_reads_inert(trace: Trace) -> bool:
    """Whether a buffer's ``Op.out`` read during this pass has no effect beyond its value.

    The per-buffer body reads each buffer's saved output while bucketing merge
    candidates. With a tensor in the cell, no lazy or predicate payload, the
    postprocess flag set (no reference re-validation) and no quickstart
    warning owed, the read only returns the tensor, so skipping it for a
    one-member group changes nothing.

    ``_predicate_save_options`` exists only when a predicate is configured, so
    it is read from the instance dict; ``_postprocessing_active`` is always set
    by ``Trace._postprocess``, the only entry to this step, so it is read
    directly.
    """

    return (
        trace.__dict__.get("_predicate_save_options") is None
        and getattr(trace, "payload_load_status", None) != "loaded_lazy"
        and bool(trace._postprocessing_active)
        and not _synthesized_out_read_due(trace)
    )


def _answered(column: list[object]) -> list[object]:
    """Return ``column`` with each immutable default marker replaced by its answer."""

    return [cell.source if cell.__class__ is LazyCell else cell for cell in column]


def _unwired(column: list[object], classes: set[type]) -> bool:
    """Whether every ``buffer_source`` cell answers ``None`` (no buffer to wire)."""

    if not classes <= {type(None), LazyCell} or not immutable_markers(column):
        return False
    return all(marker.source is None for marker in markers(column))


def _fix_by_column(trace: Trace) -> bool:
    """Number buffer passes over raw cells; return ``False`` (nothing written) to decline.

    Applies when every buffer label resolves in the raw layer dict to an op of
    one plain row store, no buffer has a recorded source (nothing to wire),
    every address is set and distinct (so every merge group has one member
    and nothing merges), and the skipped saved-output reads are inert.
    """

    labels = list(trace.buffer_layers)
    if not labels:
        return True
    view = _buffer_rows(trace, labels)
    if view is None:
        return False
    store, buffer_rows = view
    address_values = _distinct_addresses(store, buffer_rows)
    if address_values is None or not _tensor_outs(store, buffer_rows):
        return False
    cells = store._rows
    f_pass = int(store.layout.fid_by_name["buffer_pass"])
    num_calls = trace._buffer_num_calls
    for row, address in zip(buffer_rows, address_values, strict=True):
        cells[row][f_pass] = 1
        num_calls[address] = 1
    return True


def _buffer_rows(trace: Trace, labels: list[str]) -> tuple[Any, list[int]] | None:
    """Return the plain row store and the buffer rows, or ``None`` to decline."""

    if trace.__dict__.get("capture_events") is not None or not _out_reads_inert(trace):
        return None
    raw = trace._raw_graph_ws.raw_layer_dict
    if not raw.keys() >= set(labels):
        return None
    view = seal_rows(labels, list(map(raw.__getitem__, labels)))
    if view is None:
        return None
    return view.store, [view.rows[label] for label in labels]


def _distinct_addresses(store: Any, buffer_rows: list[int]) -> list[str] | None:
    """Return each buffer's address when none has a source and all are distinct strings."""

    sources = plain_column(store, "buffer_source", buffer_rows)
    addresses = plain_column(store, "address", buffer_rows)
    if sources is None or addresses is None or not _unwired(sources[0], sources[1]):
        return None
    if not addresses[1] <= {str, LazyCell} or not immutable_markers(addresses[0]):
        return None
    values = _answered(addresses[0])
    if set(map(type, values)) != {str} or len(set(values)) != len(buffer_rows):
        return None
    return cast("list[str]", values)


def _tensor_outs(store: Any, buffer_rows: list[int]) -> bool:
    """Whether every buffer's raw ``out`` cell holds a tensor (the facade returns it as is)."""

    f_out = int(store.layout.fid_by_name["out"])
    cells = store._rows
    out_classes = {cells[row][f_out].__class__ for row in buffer_rows}
    return all(issubclass(cls, torch.Tensor) for cls in out_classes)


def _fix_by_facade(trace: Trace, helpers: MergeHelpers) -> None:
    """Run step 6 per buffer through the facade.

    1. Connects each buffer to its buffer_source (the tensor that produced the
       buffer's value), updating parent/child links and ancestry.
    2. Merges re-logged reads of one buffer version: within a group sharing the
       hash (modules + buffer_source + address), a node merges into an earlier
       node only when capture recorded the same buffer version for both (see
       ``control_flow._merge_same_version_buffers``). Values never decide it.
    3. Assigns sequential buffer_pass numbers per address.
    """

    buffer_counter: dict[str, int] = defaultdict(lambda: 1)
    buffer_hash_groups: dict[str, list[str]] = defaultdict(list)
    # Buffer rows whose edges this step changes; their descendant cones need ancestry
    # re-derived (see _repropagate_ancestry_after_buffer_wiring).
    rewired_buffers: list[str] = []

    for layer_label in trace.buffer_layers:
        layer = trace[layer_label]
        if layer._graph_buffer_source is not None:
            rewired_buffers.append(layer._label_raw)
            _wire_buffer_to_source(trace, layer, layer_label)

        if layer.buffer_address is None:
            equivalence_class = str(getattr(layer, "_graph_equivalence_class", ""))
            if equivalence_class.startswith("buffer_"):
                recovered_address = equivalence_class.removeprefix("buffer_")
                if recovered_address and recovered_address != "None":
                    layer.buffer_address = recovered_address
        if layer.buffer_address is None:
            layer.buffer_address = f"anonymous_buffer_{layer._label_raw}"
        buffer_hash = str(layer.modules) + str(layer._graph_buffer_source) + layer.buffer_address
        buffer_hash_groups[buffer_hash].append(layer_label)

    deferred_buffer_removals: dict[str, tuple[Op, Op]] = {}
    # Per-survivor membership sets shared across the whole sweep: the
    # membership guards in _merge_buffer_entries otherwise rescan the
    # survivor's growing edge lists once per merged duplicate.
    survivor_edge_shadows: dict[str, dict[str, set[str]]] = {}
    for buffers_orig in buffer_hash_groups.values():
        helpers.merge_same_version(
            trace, buffers_orig, deferred_buffer_removals, survivor_edge_shadows, rewired_buffers
        )
    helpers.finish_removals(trace, deferred_buffer_removals)

    helpers.repair_ancestry(trace, rewired_buffers)
    helpers.repair_descendants(trace, rewired_buffers)

    # And relabel the buffer ops.

    for layer_label in trace.buffer_layers:
        layer = trace[layer_label]
        address = layer.buffer_address
        layer._internal_set("buffer_pass", buffer_counter[address])
        trace._buffer_num_calls[address] = buffer_counter[address]
        buffer_counter[address] += 1

"""Which outputs validation's reproducibility re-trace saves, and the check that it saved enough.

The re-trace exists only to compare graph-shape hashes (``compute_graph_shape_hash``
with ``include_module_address=False``) of two forwards. That hash reads layer
metadata only; the one place a saved output reaches a hashed field is buffer
postprocessing (``_postprocess/_buffer_source_wiring.py`` and
``_postprocess/control_flow.py``), which reads the saved output of every buffer
node (to merge re-logged reads of one buffer version) and of each buffer's source
op (to replay the buffer as ``identity``). So the re-trace saves those outputs and
drops the rest, instead of holding a second full set of activations beside the
first trace.

The save set is chosen from the first trace by raw operation index: every index
the first trace does NOT show as a kept, non-buffer op that is no buffer's source
(buffer nodes, buffer sources, and every index that left the layer list, such as
merged buffer reads and orphans). After the re-trace, ``retrace_save_coverage_gap``
checks that every buffer node and buffer source of the SECOND trace was in that
set; when one was not (the forwards ran different op sequences), the slim hash is
not trusted and ``compare_retrace_graphs`` reports a mismatch (fail closed).
``tests/test_validation_retrace_saves.py`` proves the hash reads nothing the slim
re-trace drops.
"""

from __future__ import annotations

from typing import Any

from ._utils.hashing import _record_is_buffer, compute_graph_shape_hash
from ._validation_divergence import _first_reproducibility_divergence


def _raw_index(record: Any) -> int | None:
    """Return a record's raw operation index, or ``None`` when it has none."""

    index = getattr(record, "raw_index", None)
    return index if isinstance(index, int) else None


def _buffer_source_record(trace: Any, buffer_record: Any) -> Any:
    """Return the record of ``buffer_record``'s source op, ``None`` when it has no source.

    Raises
    ------
    LookupError
        When the buffer names a source label the finished trace cannot resolve.
    """

    source_label = getattr(buffer_record, "_graph_buffer_source", None)
    if source_label is None:
        return None
    source = trace._layer_dict_all_keys.get(source_label)
    if source is None:
        raise LookupError(source_label)
    return source


def retrace_save_indices(first_trace: Any) -> tuple[int, ...]:
    """Return the raw operation indices the slim re-trace saves outputs for.

    Parameters
    ----------
    first_trace:
        Postprocessed validation trace (captured with every output saved).

    Returns
    -------
    tuple[int, ...]
        Every raw index up to the first trace's largest kept index except those of
        kept, non-buffer ops that are no buffer's source.
    """

    droppable: set[int] = set()
    needed: set[int] = set()
    largest = -1
    for record in first_trace._layer_list:
        index = _raw_index(record)
        if index is None:
            continue
        largest = max(largest, index)
        if not _record_is_buffer(record):
            droppable.add(index)
            continue
        needed.add(index)
        try:
            source = _buffer_source_record(first_trace, record)
        except LookupError:
            continue
        source_index = None if source is None else _raw_index(source)
        if source_index is not None:
            needed.add(source_index)
    return tuple(index for index in range(largest + 1) if index in needed or index not in droppable)


def retrace_save_coverage_gap(second_trace: Any, saved_indices: frozenset[int]) -> str | None:
    """Return why the slim re-trace's hash cannot be trusted, or ``None`` when it can.

    Every buffer node left in the second trace's layer list and each one's source
    op must have had its output saved. A buffer read the selection missed is never
    merged (an unsaved read is skipped), so it stays in the layer list and this
    check sees it.

    Parameters
    ----------
    second_trace:
        Postprocessed slim re-trace.
    saved_indices:
        Raw indices the re-trace saved (``retrace_save_indices`` of the first trace).

    Returns
    -------
    str | None
        A one-line reason naming the first unsaved buffer node or source, else ``None``.
    """

    for record in second_trace._layer_list:
        if not _record_is_buffer(record):
            continue
        label = getattr(record, "layer_label", None)
        index = _raw_index(record)
        if index not in saved_indices:
            return f"buffer op {label!r} ran at raw index {index}, unlike the first forward"
        try:
            source = _buffer_source_record(second_trace, record)
        except LookupError as exc:
            return f"buffer op {label!r} names unresolvable source {exc.args[0]!r}"
        if source is not None and _raw_index(source) not in saved_indices:
            return (
                f"source of buffer op {label!r} ran at raw index {_raw_index(source)}, "
                "unlike the first forward"
            )
    return None


def compare_retrace_graphs(
    first_trace: Any, second_trace: Any
) -> tuple[str, str, str | None] | None:
    """Compare the first validation trace with its slim re-trace.

    Parameters
    ----------
    first_trace:
        Postprocessed validation trace (every output saved).
    second_trace:
        Postprocessed slim re-trace, saved per ``retrace_save_indices(first_trace)``.

    Returns
    -------
    tuple[str, str, str | None] | None
        ``None`` when the structure hashes match and the save set covered every buffer
        node and source of the re-trace; else ``(first_hash, second_hash, hint)``, where
        the hint names the coverage gap (hashes equal) or the first divergence.
    """

    first_hash = compute_graph_shape_hash(first_trace, include_module_address=False)
    second_hash = compute_graph_shape_hash(second_trace, include_module_address=False)
    saved_indices = frozenset(retrace_save_indices(first_trace))
    coverage_gap = retrace_save_coverage_gap(second_trace, saved_indices)
    if first_hash == second_hash and coverage_gap is None:
        return None
    if first_hash == second_hash:
        return first_hash, second_hash, coverage_gap
    return first_hash, second_hash, _first_reproducibility_divergence(first_trace, second_trace)

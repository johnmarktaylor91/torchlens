"""Step 8 by column: label parts as integers, label strings formed in one pass.

Each op's label is built from four integers and a type: the type index (its
rank among first passes of that type), the step index (its rank among first
passes that are numbered; inputs and buffers are not), its pass number and,
for a later pass, the parts of the group's first pass. ``label_parts`` finds
those integers for every op from the raw columns in one loop;
``map_labels_by_column`` writes them and the four label strings into the
building rows and builds the raw-to-final maps, with no call per op.

The strings and maps are those of ``map_labels_by_facade``, the same work
through the ``Op`` facade, which runs when the ops are not one building store
or a column holds a cell this pass cannot read as the facade would. Every
label string is still formed here: the lookup-key table (step 11) and the
reference renames of step 9 read every op's labels during the same seal, so
forming them on first read would move that work, not remove it.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from .._record_type import record_type
from ._final_info_columns import _NotColumnar, _open_table

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Types whose labels carry no step index.
_UNSTEPPED_LABEL = frozenset({"input", "output", "buffer"})
#: Types that take step index 0 instead of a number.
_UNNUMBERED = frozenset({"input", "buffer"})
_READ = ("type", "pass_number", "num_passes", "_label_raw")
_WRITTEN = ("type_index", "step_index", "label", "layer_label", "short_label", "short_layer_label")
_STR = frozenset({str})
_INT = frozenset({int})


def label_parts(
    kinds: list[str], passes: list[int], raws: list[Any], groups: list[Any]
) -> tuple[list[str], list[int], list[int]]:
    """Return each op's label type, type index and step index, in op order.

    A later pass inherits the type and both indices of its group's first
    pass (``groups[i][0]``), which must come earlier in op order, as the
    per-op loop requires to read the values it has just written.

    Raises
    ------
    _NotColumnar
        A later pass whose group head is missing or not earlier: the per-op
        loop decides what happens to it.
    """

    position = {raw: index for index, raw in enumerate(raws)}
    count = len(kinds)
    out_kinds = list(kinds)
    type_indices = [0] * count
    step_indices = [0] * count
    # A missing type starts at 0; the default factory runs in C, so no call per op.
    type_counter: defaultdict[str, int] = defaultdict(int)
    next_step = 1
    for index, kind, pass_number, group in zip(range(count), kinds, passes, groups, strict=True):
        if pass_number == 1:
            type_counter[kind] += 1
            type_index = type_counter[kind]
            if kind in _UNNUMBERED:
                step_index = 0
            else:
                step_index = next_step
                next_step += 1
        else:
            head_label = group[0] if group else None
            head = position.get(head_label, count)
            if head >= index:
                raise _NotColumnar("recurrent_ops")
            kind = out_kinds[index] = out_kinds[head]
            type_index = type_indices[head]
            step_index = 0 if kind in _UNNUMBERED else step_indices[head]
        type_indices[index] = type_index
        step_indices[index] = step_index
    return out_kinds, type_indices, step_indices


def map_labels_by_column(trace: Trace) -> bool:
    """Write every op's label parts and labels and the trace's label maps by column.

    Parameters
    ----------
    trace:
        Trace being postprocessed.

    Returns
    -------
    bool
        ``False`` when the trace is not columnar; nothing was written and the
        caller runs ``map_labels_by_facade``.
    """

    table = _open_table(list(trace), (*_READ, *_WRITTEN), ("recurrent_ops",))
    if table is None:
        return False
    try:
        kinds = table.read("type", False, _STR)
        passes = table.read("pass_number", False, _INT)
        num_passes = table.read("num_passes", False, _INT)
        raws = table.read("_label_raw", False)
        groups = table.read("recurrent_ops", True)
        out_kinds, type_indices, step_indices = label_parts(kinds, passes, raws, groups)
    except _NotColumnar:
        return False
    _write_labels(trace, table, (out_kinds, type_indices, step_indices, passes, num_passes, raws))
    return True


def _write_labels(trace: Trace, table: Any, columns: tuple[list[Any], ...]) -> None:
    """Write the label cells of every row and set the trace's four label maps.

    ``columns`` holds, in op order, each op's label type, type index, step
    index, pass number, pass count and raw label.
    """

    fids = table.fids
    f_type, f_step = fids["type_index"], fids["step_index"]
    f_label, f_layer = fids["label"], fids["layer_label"]
    f_short, f_short_layer = fids["short_label"], fids["short_layer_label"]
    to_layer: dict[Any, str] = {}
    to_parent_layer: dict[Any, str] = {}
    to_op: dict[Any, str] = {}
    to_raw: dict[str, Any] = {}
    for row, kind, type_index, step_index, pass_number, passes_of_layer, raw in zip(
        table.rows, *columns, strict=True
    ):
        row[f_type] = type_index
        row[f_step] = step_index
        if kind in _UNSTEPPED_LABEL:
            label = row[f_label] = f"{kind}_{type_index}:{pass_number}"
            layer_label = row[f_layer] = f"{kind}_{type_index}"
        else:
            label = row[f_label] = f"{kind}_{type_index}_{step_index}:{pass_number}"
            layer_label = row[f_layer] = f"{kind}_{type_index}_{step_index}"
        row[f_short] = f"{kind}_{type_index}:{pass_number}"
        row[f_short_layer] = f"{kind}_{type_index}"
        lookup_label = layer_label if passes_of_layer == 1 else label
        to_layer[raw] = lookup_label
        to_parent_layer[raw] = layer_label
        to_op[raw] = label
        to_raw[lookup_label] = raw
    _set_maps(trace, to_layer, to_parent_layer, to_op, to_raw)


def _set_maps(
    trace: Trace,
    to_layer: dict[Any, str],
    to_parent_layer: dict[Any, str],
    to_op: dict[Any, str],
    to_raw: dict[str, Any],
) -> None:
    """Set the trace's label maps and clear its backward projection state."""

    trace._raw_to_final_layer_labels = to_layer
    trace._raw_to_final_parent_layer_labels = to_parent_layer
    trace._raw_to_final_op_labels = to_op
    trace._backward_projection_event_count = None
    trace._backward_projection_revision = None
    trace._backward_projection_fold_state = None
    trace._final_to_raw_layer_labels = to_raw


def map_labels_by_facade(self: Trace) -> None:
    """Assign every op's label parts and labels through the ``Op`` facade.

    The per-op form of ``map_labels_by_column``. Regular labels are
    ``{type}_{type_index}_{step_index}:{pass}``; input, output and buffer
    labels drop the step index. A later pass inherits the type and both
    indices from its group's first pass, so every member of a recurrence
    group shares one layer label.
    """

    to_layer = {}
    to_parent_layer = {}
    to_op = {}
    to_raw = {}
    type_counter: dict[str, int] = {}
    next_step = 1  # Sequential counter for non-input/buffer layers.
    for op in self:
        layer_type = record_type(op)
        pass_index = op.pass_number
        if pass_index == 1:
            type_index = type_counter.get(layer_type, 1)
            type_counter[layer_type] = type_index + 1
            if layer_type in _UNNUMBERED:
                step_index = 0
            else:
                step_index = next_step
                next_step += 1
        else:
            # Raw slot read: the public read is copy-on-read.
            first_pass = self[op._slot("recurrent_ops")[0]]
            layer_type = record_type(first_pass)
            type_index = first_pass.type_index
            step_index = 0 if layer_type in _UNNUMBERED else first_pass.step_index
        op.type_index = type_index
        op.step_index = step_index
        if layer_type not in _UNSTEPPED_LABEL:
            op.label = f"{layer_type}_{type_index}_{step_index}:{pass_index}"
            op.layer_label = f"{layer_type}_{type_index}_{step_index}"
        else:
            op.label = f"{layer_type}_{type_index}:{pass_index}"
            op.layer_label = f"{layer_type}_{type_index}"
        op.short_label = f"{layer_type}_{type_index}:{pass_index}"
        op.short_layer_label = f"{layer_type}_{type_index}"
        lookup_label = op.layer_label if op.num_passes == 1 else op.label
        to_layer[op._label_raw] = lookup_label
        to_parent_layer[op._label_raw] = op.layer_label
        to_op[op._label_raw] = op.label
        to_raw[lookup_label] = op._label_raw
    _set_maps(self, to_layer, to_parent_layer, to_op, to_raw)

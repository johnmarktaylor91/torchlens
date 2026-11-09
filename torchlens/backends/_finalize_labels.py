"""Torch's final-label rule for preview-backend ops.

Preview backends mint raw labels ``{type}_{type_count}_{raw_count}_raw`` in which the
raw count includes the input ops (and a few backends name inputs by address, such as
``input.arg_0``). Torch's rule (``_postprocess/labeling.py``,
``_map_raw_labels_to_final_labels``) numbers the same graph differently, and the
preview finalizers follow it so the labels of matching ops agree across backends:

- type and step numbers are assigned once per layer, at its first pass, in execution
  order; later passes of a recurrent layer inherit them;
- input and buffer ops take step 0 and no step in their label (``input_1``), output
  ops have no step in their label (``output_1``);
- every other layer takes the next step number from 1 (``linear_1_1``);
- the short label drops the step (``linear_1``);
- the ``step_index`` field is set afterwards per op, like torch's
  ``_log_final_info_for_layers``: 0 for inputs and buffers, else a count from 1 over every
  other op, so later passes of a recurrent layer get their own (unique) step.

The raw label stays a lookup key, so an address-named input (``input.arg_0``)
still resolves.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .._record_type import record_type

#: Op types that take step 0 (torch: ``input`` and ``buffer``).
_UNSTEPPED_TYPES = frozenset({"input", "buffer"})
#: Op types whose label carries no step number.
_STEPLESS_LABEL_TYPES = frozenset({"input", "buffer", "output"})


@dataclass(frozen=True)
class PreviewFinalLabel:
    """One layer's final numbering under torch's rule.

    Attributes
    ----------
    layer_label:
        ``{type}_{type_index}_{step_index}``, or ``{type}_{type_index}`` for input,
        buffer and output ops.
    short_layer_label:
        ``{type}_{type_index}``.
    type_index:
        1-based count of this op type's layers so far.
    step_index:
        The op's own ``step_index``: 1-based over ops that are not inputs or buffers,
        0 for those. Equal to the label's step for an op at its layer's first pass
        when no recurrent pass came before it.
    """

    layer_label: str
    short_layer_label: str
    type_index: int
    step_index: int


def torch_rule_final_labels(
    raw_ops: Mapping[str, Any], assignments: Mapping[str, Any] | None = None
) -> dict[str, PreviewFinalLabel]:
    """Number preview ops by torch's rule, keyed by raw label.

    Parameters
    ----------
    raw_ops:
        Raw label to op, in execution order (``trace._raw_graph_ws.raw_layer_dict``).
    assignments:
        Recurrence assignments by raw label; an op's layer is its assignment's
        ``layer_label``. ``None``, or an op without one, is its own layer.

    Returns
    -------
    dict[str, PreviewFinalLabel]
        The final numbering of every raw op; passes of one layer share their labels.
    """

    by_layer: dict[str, PreviewFinalLabel] = {}
    type_counts: defaultdict[str, int] = defaultdict(int)
    step = 0
    op_step = 0
    finals: dict[str, PreviewFinalLabel] = {}
    for raw_label, op_log in raw_ops.items():
        assignment = assignments.get(raw_label) if assignments is not None else None
        layer_key = assignment.layer_label if assignment is not None else raw_label
        final = by_layer.get(layer_key)
        if final is None:
            op_type = str(record_type(op_log))
            type_counts[op_type] += 1
            type_index = type_counts[op_type]
            step_index = 0
            if op_type not in _UNSTEPPED_TYPES:
                step += 1
                step_index = step
            short = f"{op_type}_{type_index}"
            layer_label = short if op_type in _STEPLESS_LABEL_TYPES else f"{short}_{step_index}"
            final = PreviewFinalLabel(layer_label, short, type_index, step_index)
            by_layer[layer_key] = final
        op_step_index = 0
        if str(record_type(op_log)) not in _UNSTEPPED_TYPES:
            op_step += 1
            op_step_index = op_step
        finals[raw_label] = replace(final, step_index=op_step_index)
    return finals


def apply_final_label(op_log: Any, final: PreviewFinalLabel, pass_index: int) -> str:
    """Write one op's final labels and numbers; return its pass-qualified label.

    Parameters
    ----------
    op_log:
        The preview op being finalized.
    final:
        Its layer's numbering from :func:`torch_rule_final_labels`.
    pass_index:
        The op's pass within its layer (1 for a single-pass layer).

    Returns
    -------
    str
        ``{layer_label}:{pass_index}``, the op's ``label``.
    """

    pass_label = f"{final.layer_label}:{pass_index}"
    op_log.label = pass_label
    op_log.layer_label = final.layer_label
    op_log.short_label = f"{final.short_layer_label}:{pass_index}"
    op_log.short_layer_label = final.short_layer_label
    op_log.type_index = final.type_index
    op_log.step_index = final.step_index
    return pass_label

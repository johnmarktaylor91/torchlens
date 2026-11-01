"""Label minting and field copying for synthesized untagged-tensor boundary ops.

Split out of ``model_prep.py`` under its file-size ceiling; used by
``model_prep._ensure_module_output_tensor_logged``.
"""

import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


def _next_untagged_tensor_label(trace: "Trace", layer_type: str) -> tuple[str, int, int]:
    """Return a fresh raw label for an untagged tensor surfaced mid-forward.

    Two distinct kinds of untagged tensors reach this helper:

    * ``"interventionreplacement"`` -- a genuine raw ``register_forward_hook``
      output replacement injected by the user; the tensor lacks traceable
      provenance because the user substituted it for a module's real output.
    * ``"internalsource"`` -- an internally generated tensor (e.g. an attention
      mask built inside ``torch.vmap``, whose construction TorchLens cannot
      trace) that enters a module untagged during plain capture. It is a
      legitimate graph source, not a user intervention.

    Parameters
    ----------
    trace:
        Active model log whose raw layer counters should be advanced.
    layer_type:
        ``"interventionreplacement"`` or ``"internalsource"``.

    Returns
    -------
    tuple[str, int, int]
        Raw label, capture index, and per-type index.
    """

    trace._raw_graph_ws.layer_counter += 1
    trace._raw_graph_ws.raw_layer_type_counter[layer_type] += 1
    raw_index = trace._raw_graph_ws.layer_counter
    type_index = trace._raw_graph_ws.raw_layer_type_counter[layer_type]
    return f"{layer_type}_{type_index}_{raw_index}_raw", raw_index, type_index


def _copy_field_value_for_replacement(value: Any) -> Any:
    """Copy mutable Op field values without cloning tensors.

    Parameters
    ----------
    value:
        Field value from a parent layer entry.

    Returns
    -------
    Any
        A structurally independent copy for container fields, or the original
        immutable/scalar/tensor reference otherwise.
    """

    if isinstance(value, (list, dict, set, defaultdict)):
        return copy.copy(value)
    return value

"""Lazy source-tensor logging for a lean plain ``tl.module`` read.

Split out of ``sources.py`` (file cap): ``log_source_tensor`` calls
``log_source_tensor_lazy`` when the pass defers its record contexts.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from ..._capture._lazy_contexts import LazyContextHistory, tensor_fields
from ._tl import set_tensor_label

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


def log_source_tensor_lazy(
    self: Trace,
    history: LazyContextHistory,
    t: torch.Tensor,
    source: str,
    extra_addr: str | None,
) -> str:
    """Label an input or buffer of a lean module read and defer its context.

    Matches ``log_source_tensor_predicate`` on a pass with deferred contexts:
    source events are not selected there (no ``include_source_events``), halt
    is off and a lean journal appends no shell event for an unkept source, so
    the lasting effects are the counters, the label, the input/buffer lists,
    and the context, which is deferred with the same inputs. Returns the
    label; the caller journals a buffer's version key (``buffer_writes`` sits
    in the package import cycle, which this module stays out of).
    """

    if source not in {"input", "buffer"}:
        raise ValueError("source must be either 'input' or 'buffer'")
    state = history.state
    graph_ws = self._raw_graph_ws
    graph_ws.layer_counter += 1
    graph_ws.raw_layer_type_counter[source] += 1
    state.event_index += 1
    type_index = graph_ws.raw_layer_type_counter[source]
    tensor_label = f"{source}_{type_index}_raw"
    set_tensor_label(t, tensor_label)
    (self.input_layers if source == "input" else self.buffer_layers).append(tensor_label)
    module_frame = state.module_stack[-1] if state.module_stack else None
    data = {
        "label": tensor_label,
        "raw_label": tensor_label,
        "_label_raw": tensor_label,
        "raw_index": graph_ws.layer_counter,
        "type": source,
        "type_index": type_index,
        "func_name": None,
        "input_output_address": extra_addr,
        "address": module_frame.address if module_frame else None,
        "module_type": module_frame.module_type if module_frame else None,
        "module_pass_number": module_frame.pass_number if module_frame else None,
        **tensor_fields(t),
    }
    elapsed = time.time() - self._provenance_start_time
    history.defer(
        "input" if source == "input" else "buffer", data, state.event_index, None, elapsed
    )
    return tensor_label

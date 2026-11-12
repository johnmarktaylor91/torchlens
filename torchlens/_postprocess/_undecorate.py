"""Step 12 body: strip TorchLens tensor metadata from every saved tensor.

The live ops' fields are read once per field as columns (``_step_views``);
orphans keep the facade reads. Each tensor is undecorated in the order the
per-op walk always produced: per op its saved output, its transformed output,
then the tensor leaves of its saved positional and keyword arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._ir.container_registry import _iter_tensor_leaves
from .._utils._record_group_table import record_member
from .._utils._tensor_label import clear_meta
from ._step_views import ABSENT, bind_ops, read_field, row_views

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = ("undecorate_all_saved_tensors",)


def _entry_tensors(entry: Any, out: Any, saved_args: Any, saved_kwargs: Any) -> list[Any]:
    """Return one op's tensors to undecorate, given its saved output and arguments."""

    tensors = []
    if out is not None:
        tensors.append(out)
    if record_member(entry, "transformed_out", None) is not None:
        tensors.append(entry._payload_transformed_out)
    if saved_args:
        tensors.extend(_iter_tensor_leaves(saved_args))
    if saved_kwargs:
        tensors.extend(_iter_tensor_leaves(saved_kwargs))
    return tensors


def undecorate_all_saved_tensors(trace: Trace) -> None:
    """Remove TorchLens metadata from saved outputs and saved argument tensors.

    During logging, tensors are tagged with private ``_tl`` metadata for
    tracking. This strips it from saved outs and any tensors embedded in
    saved_args/saved_kwargs, so tensors returned to the user are clean.
    """

    ops = trace._layer_list
    views = row_views(ops, ("has_saved_activation", "_call_saved_args", "_call_saved_kwargs"))
    saved = [op for op, view in zip(ops, views, strict=True) if view.has_saved_activation]
    out_of = dict(zip(map(id, saved), read_field(saved, "out", bind_ops(saved)), strict=True))
    tensors: list[Any] = []
    for op, view in zip(ops, views, strict=True):
        out = out_of.get(id(op))
        if out is ABSENT:
            out = op.out  # raises the facade's AttributeError, as the per-op read did
        tensors += _entry_tensors(op, out, view._call_saved_args, view._call_saved_kwargs)
    for entry in trace.graph.orphans.values():
        out = entry.out if entry.has_saved_activation else None
        tensors += _entry_tensors(entry, out, entry._call_saved_args, entry._call_saved_kwargs)
    for tensor in tensors:
        clear_meta(tensor)

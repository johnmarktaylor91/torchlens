"""Param barcodes on JAX pytree-module ops, stamped before recurrence grouping.

Recurrence grouping (the neutral ``group_recurrent_nodes`` torch also uses) only folds ops
into one multi-pass layer when their param barcodes agree, so two distinct modules of the
same shape (two ``Linear`` layers with their own weights) stay separate layers, as in torch.
In pytree-module mode the full param attachment (``JAXBackend._attach_pytree_op_params``)
runs after grouping, because it records final layer labels; this step gives each raw op its
barcodes first, with the same equation-to-capture pairing, so grouping sees them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def stamp_pytree_param_barcodes(trace: Any, tree: Any, captures: Sequence[Any]) -> None:
    """Set ``_param_barcodes`` on each raw op whose equation read a module parameter.

    Parameters
    ----------
    trace
        Trace with materialized raw ops and ``trace._param_logs`` already attached.
    tree
        Equinox or NNX module tree (``param_address_by_value_id``).
    captures
        Interpreted JAX equation captures in event order.
    """

    raw_dict = trace._raw_graph_ws.raw_layer_dict
    equation_labels = [
        label for label in trace._raw_graph_ws.raw_layer_labels_list if not raw_dict[label].is_input
    ]
    for label, capture in zip(equation_labels, captures, strict=False):
        addresses: list[str] = []
        for value in capture.input_values:
            address = tree.param_address_by_value_id.get(id(value))
            if address is not None and address not in addresses:
                addresses.append(address)
        if addresses:
            raw_dict[label]._param_barcodes = [
                trace._param_logs[address].barcode for address in addresses
            ]

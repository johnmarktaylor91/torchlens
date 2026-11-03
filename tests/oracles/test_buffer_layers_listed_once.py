"""``buffer_layers`` names each buffer Layer once, in first-seen order.

Hand count from ``LoopedBatchNorm.forward`` (``test_legacy_graph_counts.py``):
the loop runs three times; each step reads and writes the counter ``count``
(``add_``), and the training ``BatchNorm1d`` bumps ``num_batches_tracked``
and updates ``running_mean`` and ``running_var``. Four buffers, each read
once from its initial value and written three times, are 16 buffer passes
over 4 buffer Layers. The first step touches them in the order ``count``,
``bn.num_batches_tracked``, ``bn.running_mean``, ``bn.running_var``, so that
is the list's order; the root Module owns all four, ``bn`` the last three
and ``fc`` none.
"""

from __future__ import annotations

import torch

import torchlens as tl
from tests.oracles.test_legacy_graph_counts import LoopedBatchNorm

FIRST_SEEN = ["count", "bn.num_batches_tracked", "bn.running_mean", "bn.running_var"]


def test_buffer_layers_list_each_buffer_layer_once_in_first_seen_order() -> None:
    torch.manual_seed(0)
    trace = tl.trace(LoopedBatchNorm().train(), torch.randn(5, 4))
    try:
        labels = list(trace.buffer_layers)
        assert len(labels) == len(set(labels)) == 4
        assert [trace.layers[label].address for label in labels] == FIRST_SEEN
        assert int(trace.num_buffer_layers) == 4
        modules = {module.address: module for module in trace.modules}
        assert list(modules["self"].buffer_layers) == labels
        assert list(modules["bn"].buffer_layers) == labels[1:]
        assert list(modules["fc"].buffer_layers) == []
        # Each buffer Layer still carries its four passes (one read, three writes).
        assert [int(trace.layers[label].num_passes) for label in labels] == [4, 4, 4, 4]
    finally:
        trace.cleanup()

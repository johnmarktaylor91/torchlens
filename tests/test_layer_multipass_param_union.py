"""A multi-pass Layer whose passes run over different parameters.

One call site run over a ``ParameterList`` is one layer under call-site
grouping. Its parameter fields must be the union over the passes (each pass
keeps its own parameter on its ``Op``), and the per-pass addresses must be
disclosed under ``annotations["varying_across_passes"]["params"]``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _WeightLoop(nn.Module):
    """Apply three distinct weight matrices from one ``F.linear`` call site."""

    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for weight in self.weights:
            x = torch.tanh(F.linear(x, weight))
        return x


def _linear_layer(trace: Any) -> Any:
    layers = [
        layer
        for layer in trace.layer_logs.values()
        if layer.layer_label.startswith("linear_") and len(layer.ops) > 1
    ]
    assert len(layers) == 1, [layer.layer_label for layer in trace.layer_logs.values()]
    return layers[0]


def test_layer_params_are_the_union_over_passes_with_per_pass_disclosure() -> None:
    torch.manual_seed(0)
    model = _WeightLoop()
    trace = tl.trace(model, torch.randn(2, 4))

    layer = _linear_layer(trace)
    assert len(layer.ops) == 3
    assert layer.num_passes == 3

    # Union over the three passes, deduplicated by parameter identity.
    assert layer.param_shapes == [(4, 4), (4, 4), (4, 4)]
    assert layer.num_params == 3 * 16
    assert layer.num_params_trainable == 3 * 16
    assert layer.num_params_frozen == 0
    assert int(layer.total_param_memory) == sum(p.numel() * p.element_size() for p in model.weights)

    # Each pass keeps exactly its own weight.
    per_pass = [[param.address for param in (op._param_logs or ())] for op in layer.ops]
    assert per_pass == [["weights.0"], ["weights.1"], ["weights.2"]]
    assert layer.annotations["varying_across_passes"]["params"] == per_pass

    # The tanh after it runs over equal (empty) parameters, so it carries no marker.
    tanh_layer = next(
        layer for layer in trace.layer_logs.values() if layer.layer_label.startswith("tanh_")
    )
    assert tanh_layer.num_passes == 3
    assert "params" not in tanh_layer.annotations.get("varying_across_passes", {})

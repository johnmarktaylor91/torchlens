"""Regression coverage for fast-pass output population."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _SelectiveSaveModel(nn.Module):
    """Small stable graph with an explicit output node."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear, relu, and passthrough module."""

        return self.identity(torch.relu(self.linear(x)))


def _saved_tensor_by_label(trace: tl.Trace, labels: list[str]) -> dict[str, torch.Tensor]:
    """Return saved output tensors for the selected labels."""

    saved: dict[str, torch.Tensor] = {}
    for label in labels:
        layer = trace[label]
        assert layer.has_saved_activation
        assert layer.out is not None
        saved[label] = layer.out.detach().clone()
    return saved


def test_fast_pass_selective_save_matches_fresh_trace_outputs() -> None:
    """Fast selective save keeps postprocessed outputs identical to a fresh trace."""

    torch.manual_seed(7)
    model = _SelectiveSaveModel()
    initial_input = torch.randn(2, 3)
    replacement_input = torch.randn(2, 3)

    fast_trace = tl.trace(model, initial_input, random_seed=11)
    fresh_trace = None
    try:
        relu_label = next(layer.layer_label for layer in fast_trace if layer.layer_type == "relu")
        selected_labels = [relu_label, fast_trace.output_layers[0]]

        fast_trace.save_new_outs(
            model,
            replacement_input,
            layers_to_save=selected_labels,
            random_seed=11,
        )
        fresh_trace = tl.trace(model, replacement_input, layers_to_save="all", random_seed=11)

        fast_saved = _saved_tensor_by_label(fast_trace, selected_labels)
        fresh_saved = _saved_tensor_by_label(fresh_trace, selected_labels)
        assert fast_saved.keys() == fresh_saved.keys()
        for label, fast_out in fast_saved.items():
            torch.testing.assert_close(fast_out, fresh_saved[label])
    finally:
        fast_trace.cleanup()
        if fresh_trace is not None:
            fresh_trace.cleanup()

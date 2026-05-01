"""Phase 4 multi-pass tests."""

from __future__ import annotations

import torch

import torchlens as tl


def test_log_forward_pass_streaming_on_iterable_inputs() -> None:
    """Capture iterable inputs into one stacked root log."""

    model = torch.nn.Linear(2, 2)
    root = tl.utils.log_forward_pass_streaming(
        model,
        [torch.ones(1, 2), torch.zeros(1, 2)],
        layers_to_save="none",
    )
    assert root.num_streamed_passes == 2
    assert len(root.streaming_pass_logs) == 2


def test_multi_output_module_smoke() -> None:
    """Modules returning multiple outputs retain output-path metadata."""

    class MultiOut(torch.nn.Module):
        """Tiny multi-output module."""

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Return two outputs."""

            return x + 1, x * 2

    log = tl.log_forward_pass(MultiOut(), torch.ones(1, 2), intervention_ready=True)
    output_paths = [layer.output_path for layer in log.layer_list if layer.is_output_layer]
    assert output_paths

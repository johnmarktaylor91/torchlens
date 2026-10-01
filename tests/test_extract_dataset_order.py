"""Ordering contract tests for :func:`torchlens.extract_dataset`."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class _OrderRevealingModel(nn.Module):
    """Return a deterministic affine transform that preserves row identity."""

    def __init__(self) -> None:
        """Initialize the order-revealing layer."""

        super().__init__()
        self.reveal = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Expose each input row unchanged.

        Parameters
        ----------
        inputs:
            Batched distinctive rows.

        Returns
        -------
        torch.Tensor
            Unchanged rows.
        """

        return self.reveal(inputs)


def test_extract_dataset_preserves_stimulus_order_in_memory_and_on_disk(
    tmp_path: Path,
) -> None:
    """Both output modes should preserve order through a non-divisible final batch."""

    model = _OrderRevealingModel().eval()
    stimuli = torch.arange(7, dtype=torch.float32).unsqueeze(1).repeat(1, 3)

    in_memory = tl.extract_dataset(
        model,
        stimuli,
        {"rows": "identity"},
        batch_size=3,
        progress=False,
    )
    batch_paths = tl.extract_dataset(
        model,
        stimuli,
        {"rows": "identity"},
        batch_size=3,
        output_dir=tmp_path,
        progress=False,
    )

    assert torch.equal(in_memory["rows"], stimuli)
    assert [path.name for path in batch_paths] == [
        "batch_00000.safetensors",
        "batch_00001.safetensors",
        "batch_00002.safetensors",
    ]
    from safetensors.torch import load_file

    disk_rows = torch.cat([load_file(str(path))["rows"] for path in batch_paths])
    assert torch.equal(disk_rows, stimuli)

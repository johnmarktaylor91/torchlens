"""Classic clean-versus-corrupted activation patching.

What this demonstrates
----------------------
Capture a clean run and a corrupted run, then patch the corrupted ReLU
activation with the clean activation.

How to run
----------
``python examples/intervention/03_activation_patching_paired_prompt.py``

Runnable by default. The "prompts" are small tensors so the example stays fast.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small model for paired-input patching."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Patch a corrupted run with the clean ReLU activation."""

    torch.manual_seed(3)
    model = TinyMLP().eval()
    clean_x = torch.randn(2, 8)
    corrupted_x = clean_x + 0.75

    clean = tl.log_forward_pass(model, clean_x, vis_opt="none", intervention_ready=True)
    corrupted = tl.log_forward_pass(model, corrupted_x, vis_opt="none", intervention_ready=True)
    clean_relu = clean.find_sites(tl.func("relu")).first().activation

    patched = corrupted.fork("patched")
    patched.set(tl.func("relu"), clean_relu, confirm_mutation=True)
    patched.replay()

    clean_out = clean.layer_list[-1].activation
    corrupted_out = corrupted.layer_list[-1].activation
    patched_out = patched.layer_list[-1].activation
    assert torch.linalg.vector_norm(patched_out - clean_out) < torch.linalg.vector_norm(
        corrupted_out - clean_out
    )


if __name__ == "__main__":
    main()

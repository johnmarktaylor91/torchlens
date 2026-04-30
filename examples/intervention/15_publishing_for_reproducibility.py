"""Publish an intervention recipe with ``.tlspec``.

What this demonstrates
----------------------
Save a portable intervention spec, load it back, and check that it still
targets a fresh compatible capture.

How to run
----------
``python examples/intervention/15_publishing_for_reproducibility.py``

Runnable by default.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class TinyMLP(nn.Module):
    """Small model for portable save/load."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.in_proj = nn.Linear(8, 8)
        self.out_proj = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.out_proj(torch.relu(self.in_proj(x)))


def main() -> None:
    """Round-trip a portable built-in helper intervention."""

    torch.manual_seed(15)
    model = TinyMLP().eval()
    x = torch.randn(2, 8)
    log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
    log.attach_hooks(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "zero_relu.tlspec"
        log.save_intervention(path, level="portable")
        spec = tl.load_intervention_spec(path)
        fresh = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
        compat = tl.check_spec_compat(spec, fresh)

    assert spec.metadata["save_level"] == "portable"
    assert compat.outcome in {"EXACT", "COMPATIBLE_WITH_CONFIRMATION"}
    assert compat.targets_resolve_identically is True


if __name__ == "__main__":
    main()

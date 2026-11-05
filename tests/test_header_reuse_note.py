"""The overview header's module-reuse note uses the public spelling and stays short.

An unrolled overview of a model that calls a module more than once says how
many modules were reused and that ``rolled=True`` draws the loop. It never
names the private ``vis_mode`` request field, and it stays short enough not to
clip in a narrow drawing.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import torchlens as tl


class _Loop(nn.Module):
    """One linear cell called three times."""

    def __init__(self) -> None:
        """Build the cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the cell three times."""

        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x


def test_reuse_note_names_rolled_true(tmp_path: Path) -> None:
    """The note reads ``1 reused module (3 calls); rolled=True draws the loop``."""

    torch.manual_seed(0)
    trace = tl.trace(_Loop().eval(), torch.randn(2, 4))
    source = trace.draw(
        path=str(tmp_path / "loop"), file_format="svg", collapse="auto", rolled=False
    ).dot
    note = "1 reused module (3 calls); rolled=True draws the loop"
    assert note in source
    assert len(note) < 60
    assert "vis_mode" not in source

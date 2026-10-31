"""The ratified collapse items (collapse 1 to 6) that need their own pin.

collapse 5: on a model that reuses a module, an unrolled overview render
discloses the reuse and names rolled mode, and never switches to rolled mode
itself.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class _SharedCell(nn.Module):
    """One weight-shared cell called four times."""

    def __init__(self) -> None:
        """Initialize the shared cell and a head."""
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the shared cell four times, then the head."""
        for _ in range(4):
            x = self.cell(x)
        return self.head(x)


def _draw(trace: tl.Trace, tmp_path: Path, name: str, **kwargs: object) -> str:
    """Render to DOT and return the source."""
    return trace.draw(path=str(tmp_path / name), vis_save_only=True, file_format="dot", **kwargs)


def test_collapse5_overview_discloses_reuse_and_never_switches(tmp_path: Path) -> None:
    """Unrolled overview names the reuse; the render stays unrolled; rolled has no note."""
    trace = tl.trace(_SharedCell(), torch.randn(2, 4))
    try:
        overview = _draw(trace, tmp_path, "overview", collapse="auto")
        plain = _draw(trace, tmp_path, "plain")
        rolled = _draw(trace, tmp_path, "rolled", collapse="auto", vis_mode="rolled")
    finally:
        trace.cleanup()
    plain_note = "this model reuses 1 module across 4 calls"
    assert plain_note in overview
    assert "shows the loop" in overview
    assert plain_note not in plain, "the note belongs to overview (smart collapse) renders"
    assert plain_note not in rolled, "rolled mode already shows the loop"
    # Never auto-switches: the overview is still the unrolled picture (no rolled (xN) titles).
    assert "(x4)" not in overview

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


class _TwoLineRelu(nn.Module):
    """A ResNet-style block: one stateless ``nn.ReLU`` called on two lines."""

    def __init__(self) -> None:
        """Build the block."""

        super().__init__()
        self.l1, self.l2, self.relu = nn.Linear(4, 4), nn.Linear(4, 4), nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the relu after each linear."""

        x = self.relu(self.l1(x))
        return self.relu(self.l2(x))


class _Siamese(nn.Module):
    """One stateful encoder called on two lines."""

    def __init__(self) -> None:
        """Build the encoder."""

        super().__init__()
        self.enc = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode two views and subtract."""

        return self.enc(x) - self.enc(x * 2)


def _overview(model: nn.Module, tmp_path: Path) -> str:
    """Overview DOT source of an unrolled capture."""

    torch.manual_seed(0)
    trace = tl.trace(model.eval(), torch.randn(2, 4))
    return trace.draw(path=str(tmp_path / "m"), file_format="dot", collapse="auto").dot


def test_stateless_module_called_from_two_lines_is_not_a_loop(tmp_path: Path) -> None:
    """A stateless module's calls from two lines are separate layers: rolling folds nothing."""

    assert "rolled=True draws the loop" not in _overview(_TwoLineRelu(), tmp_path)


def test_shared_stateful_module_keeps_the_note(tmp_path: Path) -> None:
    """A stateful encoder called twice still says rolling draws the loop."""

    assert "1 reused module (2 calls); rolled=True draws the loop" in _overview(
        _Siamese(), tmp_path
    )

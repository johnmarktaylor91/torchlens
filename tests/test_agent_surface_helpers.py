"""Shared fixture builders for the F29 agent-surface suites (no tests here).

Every builder is DETERMINISTIC (explicit seeds, eval mode) so the determinism
and transcript suites can assert byte-identical output across processes.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class RepeatedBlockNet(nn.Module):
    """Three identical blocks plus a head: the smallest recurrence-fold case."""

    def __init__(self) -> None:
        """Build the deterministic block stack."""

        super().__init__()
        torch.manual_seed(0)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.ReLU()) for _ in range(3)])
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks then the head."""

        for block in self.blocks:
            x = block(x)
        return self.head(x)


def deterministic_input() -> torch.Tensor:
    """The one seeded input every fixture artifact shares."""

    generator = torch.Generator().manual_seed(1234)
    return torch.randn(2, 8, generator=generator)


def save_clean_artifact(directory: Path) -> Path:
    """Capture and save the clean fixture artifact.

    Parameters
    ----------
    directory:
        Directory to write into.

    Returns
    -------
    Path
        The saved ``.tlspec`` path.
    """

    model = RepeatedBlockNet().eval()
    log = tl.trace(model, deterministic_input(), save=tl.func("relu"))
    path = directory / "clean.tlspec"
    tl.save(log, str(path))
    return path


def save_ablated_artifact(directory: Path) -> Path:
    """Capture and save the middle-block zero-ablated fixture artifact.

    Parameters
    ----------
    directory:
        Directory to write into.

    Returns
    -------
    Path
        The saved ``.tlspec`` path.
    """

    model = RepeatedBlockNet().eval()
    log = tl.trace(
        model,
        deterministic_input(),
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu") & tl.in_module("blocks.1"), tl.zero_ablate()),
    )
    path = directory / "ablated.tlspec"
    tl.save(log, str(path))
    return path


class NaNHead(nn.Module):
    """A head that injects one NaN, for the triage-transcript fixture."""

    def __init__(self) -> None:
        """Build the deterministic linear plus NaN injection."""

        super().__init__()
        torch.manual_seed(0)
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Divide by a zero scale to plant deterministic non-finites."""

        out = self.fc(x)
        return out / torch.zeros(())


def save_nan_artifact(directory: Path) -> Path:
    """Capture and save the deterministically NaN-poisoned fixture artifact.

    Parameters
    ----------
    directory:
        Directory to write into.

    Returns
    -------
    Path
        The saved ``.tlspec`` path.
    """

    log = tl.trace(NaNHead().eval(), deterministic_input())
    path = directory / "nonfinite.tlspec"
    tl.save(log, str(path))
    return path

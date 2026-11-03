"""The identity-partition invariant suite (A1, accuracy pyramid top).

Lane A07 (2026-08-27). Spec: the summary design memo 3.1: every
parameter identity, executed op event, and tracked tensor identity is owned by
exactly ONE accounting row. Alias (input/output) rows display but own nothing
-- input rows own only the external inputs' bytes.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class BranchCNN(nn.Module):
    """Small branching CNN with BN buffers and a functional add."""

    def __init__(self) -> None:
        """Initialize two branches and a head."""

        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.left = nn.Conv2d(4, 4, 1)
        self.right = nn.Conv2d(4, 4, 1)
        self.head = nn.Linear(4 * 4 * 4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both branches and merge with a bare functional add."""

        x = self.norm(self.stem(x))
        x = self.left(x) + self.right(x)
        return self.head(x.flatten(1))


class TiedRecurrent(nn.Module):
    """Tied embedding + a 3-pass reused linear."""

    def __init__(self) -> None:
        """Initialize tied and reused layers."""

        super().__init__()
        self.emb = nn.Embedding(12, 6)
        self.mix = nn.Linear(6, 6, bias=False)
        self.head = nn.Linear(6, 12, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed, mix three times, project back."""

        h = self.emb(x)
        for _ in range(3):
            h = self.mix(h)
        return self.head(h)


def _fixture_traces() -> list[tuple[str, tl.Trace]]:
    """Capture the partition fixtures (metadata-only)."""

    fixtures: list[tuple[str, tl.Trace]] = []
    cnn = BranchCNN()
    cnn.eval()
    fixtures.append(
        (
            "branch_cnn",
            tl.trace(cnn, torch.randn(2, 3, 4, 4), save=None),
        )
    )
    tied = TiedRecurrent()
    fixtures.append(
        (
            "tied_recurrent",
            tl.trace(
                tied,
                torch.tensor([[1, 2, 3]]),
                save=None,
            ),
        )
    )
    return fixtures


def test_boundary_rows_render_dash_in_every_additive_cell() -> None:
    """The costreport D7 CI plant: a numeric additive cell on a boundary row fails."""

    model = BranchCNN()
    model.eval()
    log = tl.trace(model, torch.randn(2, 3, 4, 4), save=None)
    try:
        frame = log.profile().to_pandas()
        boundary = frame[frame["kind"] == "boundary"]
        assert not boundary.empty
        assert boundary["flops"].isna().all()
        assert boundary["time"].isna().all()
        assert boundary["param_count"].isna().all()
        outputs = boundary[boundary["name"].str.startswith("output")]
        assert outputs["activation_memory"].isna().all()
    finally:
        log.cleanup()

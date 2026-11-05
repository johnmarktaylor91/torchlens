"""W051-STOCH / AUD-CODE 4.5: ``resample_rows_from`` audit counts mean what they say."""

from __future__ import annotations

import torch
from support.sampling import sampling_draws
from torch import nn

import torchlens as tl
from torchlens.intervention import population, resample_rows_from
from torchlens.intervention.population import PerRowDatums


class _OneBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.a(x))


def _trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        _OneBlock(),
        torch.randn(3, 4),
        capture=tl.options.CaptureOptions(replayable=True, seed=0),
    )


def test_eligible_count_is_the_eligible_class_size() -> None:
    """Unconditioned: eligible == population; the distinct-donor tally has its own key."""

    log = _trace()
    row = log["relu_1_2"].out[0]
    pool = population(torch.stack([row + k for k in range(7)]), origin="rows")
    fork = log.fork()
    fork.do(tl.label("relu_1_2"), resample_rows_from(pool, seed=1, axis=0))
    record = sampling_draws(fork)[-1]
    assert record.population_count == 7
    assert record.eligible_count == 7, "eligible_count is the class size, not the draw"
    assert len(record.draw_indices) == 3


def test_eligible_count_under_agreement_is_the_union_of_row_classes() -> None:
    """Conditioned per row: eligible == members eligible for at least one subject row."""

    log = _trace()
    row = log["relu_1_2"].out[0]
    pool = population(
        torch.stack([row + k for k in range(6)]),
        origin="rows",
        data=["A", "A", "A", "B", "B", "C"],
    )
    fork = log.fork()
    fork.do(
        tl.label("relu_1_2"),
        resample_rows_from(
            pool,
            seed=1,
            axis=0,
            group_by=lambda datum: datum,
            matching=PerRowDatums(["A", "B", "A"]),
        ),
    )
    record = sampling_draws(fork)[-1]
    assert record.eligible_count == 5, "A (3) + B (2); C never eligible"
    assert all(index in (0, 1, 2) for index in (record.draw_indices[0], record.draw_indices[2]))
    assert record.draw_indices[1] in (3, 4)

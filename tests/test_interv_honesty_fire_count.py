"""Intervention honesty: fire records count the CURRENT value's fires
(edits memo row 0e -- the 2-edits-3-records over-count).

A replay push recomputes each site from capture truth through exactly this
push's hook fires, so the push REPLACES the site's replay-minted node-hook
records instead of extending them. Live-door records (capture facts) and
edge-substitution records (tier-(ii) corroboration) survive.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


class _ConvRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c1(x))


def _log(**capture_kwargs) -> tl.Trace:
    torch.manual_seed(0)
    capture = tl.options.CaptureOptions(intervention_ready=True, **capture_kwargs)
    return tl.trace(_ConvRelu().eval(), torch.randn(2, 1, 4, 4), capture=capture)


def test_two_sequential_edits_yield_two_records_and_the_chained_value() -> None:
    """The measured defect: 2 edits at one site produced 3 records."""

    log = _log()
    fork = log.fork()
    fork.do("relu_1_2", tl.scale(2.0))
    assert len(fork["relu_1_2"].interventions) == 1

    fork.do("relu_1_2", tl.scale(0.5))
    records = fork["relu_1_2"].interventions
    assert len(records) == 2
    assert all(record.engine == "replay" for record in records)
    # Both sticky hooks fired on the final push: 2.0 * 0.5 == identity.
    assert torch.allclose(fork["relu_1_2"].out, log["relu_1_2"].out)


def test_unrelated_upstream_edit_does_not_grow_downstream_fire_count() -> None:
    """A cone recompute re-fires sticky hooks; the count must not inflate."""

    log = _log()
    fork = log.fork()
    fork.do("relu_1_2", tl.scale(3.0))
    assert len(fork["relu_1_2"].interventions) == 1

    fork.do("conv2d_1_1", tl.scale(1.0))
    # The conv edit recomputed the relu site and re-fired its sticky hook:
    # still ONE record describing the current value's single fire.
    assert len(fork["relu_1_2"].interventions) == 1
    assert len(fork["conv2d_1_1"].interventions) == 1


def test_live_capture_records_survive_a_replay_push() -> None:
    """Live-door records are capture facts; replay replacement keeps them."""

    torch.manual_seed(0)
    model = _ConvRelu().eval()
    log = tl.trace(
        model,
        torch.randn(2, 1, 4, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
        intervene=tl.when(tl.func("relu"), tl.scale(2.0)),
    )
    live_before = [r for r in log["relu_1_2"].interventions if r.engine == "live"]
    assert live_before

    fork = log.fork()
    fork.do("relu_1_2", tl.scale(0.5))
    records = fork["relu_1_2"].interventions
    assert [r for r in records if r.engine == "live"] == live_before
    assert sum(1 for r in records if r.engine == "replay") >= 1

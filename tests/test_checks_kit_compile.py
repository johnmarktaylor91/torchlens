"""Checks kit: the torch.compile composition leg (heavy tier).

The mount decider (memo D1): Recorder refuses compiled models behind a
deliberate guard while the public optimizer/tensor hooks fire normally, so
the step-check family must run under torch.compile. This is the small CPU
leg; the AT-SCALE compiled leg stays an unrun gate (memo section 5) and no
default flips on it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens.checks as tc

pytestmark = pytest.mark.heavy


def _net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_step_family_runs_on_compiled_models() -> None:
    """The mount decider (memo D1): step checks run under torch.compile."""

    model = _net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer).attach()
    compiled = torch.compile(model)
    try:
        for _ in range(2):
            optimizer.zero_grad()
            compiled(torch.randn(2, 4)).sum().backward()
            optimizer.step()
    finally:
        report = session.report()
        session.detach()

    assert report.counters["backwards"] == 2
    assert report.counters["accepted_steps"] == 2
    assert report.coverage["fire_counts_nonzero"] == 4

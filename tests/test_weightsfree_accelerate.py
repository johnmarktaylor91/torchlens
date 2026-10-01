"""Accelerate on-ramp rows (memo item 20 / Tier 2).

``init_empty_weights()`` and literal ``from_pretrained(...,
device_map='meta')`` are advertised ONLY after these rows pass —
``device_map='meta'`` is the one DIGEST-AUDIT-named spelling no panel lab
could run in four rounds (accelerate absent from the checkout); absence of
evidence is not an entry-path claim. These rows block ADVERTISING those
spellings, never the D8 flip; the accelerate test-extra request rides
the packaging-request ledger.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

accelerate = pytest.importorskip("accelerate")

pytestmark = [pytest.mark.smoke, pytest.mark.optional]


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_init_empty_weights_onramp() -> None:
    """The accelerate construction idiom admits and passes the parity gate."""

    torch.manual_seed(0)
    real = Toy()
    real.eval()
    with accelerate.init_empty_weights():
        meta = Toy()
    meta.eval()
    x = torch.randn(2, 8)
    real_trace = tl.trace(real, x)
    meta_trace = tl.trace(
        meta, torch.empty_like(x, device="meta"), capture=CaptureOptions(structure_only=True)
    )
    assert tl.hash.trace(real_trace) == tl.hash.trace(meta_trace)
    assert meta_trace.discharge_against(real_trace).verdict.value == "corroborated"
    assert meta_trace.structure_evidence["substrate"] == "meta"

"""L5 M4 pins: checked suppression of redundant constructor-arg rows (DEFAULT-ON).

Covers the design-memo sec-5 contract: the closed candidate table (both
directions per row — suppresses on PROVEN equality, survives on mismatch,
including a deliberately-lying func_config), the never-candidate set, the
input-side single-edge rule, grouped convs, the deliberate rolled-vs-
unrolled divergence, the detached-record degrade rule, and the
``show_redundant_args=`` opt-out.

HONESTY: the equality check is the LICENSE — a mismatch is exactly the
interesting case and must stay VISIBLE. The rule can only reveal more,
never hide a discrepancy. Never weaken it to make a label shorter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._arg_suppression import (
    compute_suppressed_arg_keys,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MixedNet(nn.Module):
    """One member per major candidate family."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 8)
        self.ln = nn.LayerNorm(8)
        self.conv = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.emb = nn.Embedding(10, 5)

    def forward(self, x: torch.Tensor, img: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return self.ln(self.fc(x)).sum() + self.conv(img).sum() + self.emb(idx).sum()


class VaryingRecurrent(nn.Module):
    """Variable-length recurrence: rolled aggregate has no single shape."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for n in (4, 3, 2):
            x = torch.relu(self.fc(x[:n]))
        return x.sum()


@pytest.fixture(scope="module")
def varying_log() -> Any:
    log = tl.trace(VaryingRecurrent(), torch.randn(4, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _draw(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    ).dot


# ---------------------------------------------------------------------------
# Table rows: suppresses on equality, per family
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The check is the license: mismatch/unavailable stays VISIBLE
# ---------------------------------------------------------------------------


def test_non_torch_backend_never_candidates() -> None:
    class StubTrace:
        _provenance_backend = "tf"
        _layer_list = ()

    class StubUniverse:
        units = ()

    assert compute_suppressed_arg_keys(StubTrace(), StubUniverse()) == {}


# ---------------------------------------------------------------------------
# Rolled-vs-unrolled divergence (deliberate, pinned in BOTH modes)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Detached-record degrade rule (pinned, not inferred)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Table shape pins
# ---------------------------------------------------------------------------

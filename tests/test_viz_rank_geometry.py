"""Fixwave-2 FW2-POLISH pins for R19-2: rank-layout cluster geometry honesty.

The b6 probe found two coupled defects in the rank path:

- pass-qualified region keys ("cell:2") were collapsed to their pass-free
  address for the ``compound_bboxes`` mapping, so EVERY pass cluster received
  the same last-write-wins bbox, and
- the write order was set-iteration order, so WHICH pass's geometry survived
  was PYTHONHASHSEED-dependent (3 distinct DOT sha256 across seeds 0-9).

Bboxes are now keyed by the full pass-qualified region key on both the
producer and consumer sides, with sorted iteration for the remaining ties.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


class _Recurrent(nn.Module):
    """One cell called three times, producing pass-qualified module regions."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.cell(x)
        return x


_SWEEP_SCRIPT = r"""
import hashlib, sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import torchlens as tl
import torchlens._visualization._rank_layout_internal.layout as layout_mod

class Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    def forward(self, x):
        for _ in range(3):
            x = self.cell(x)
        return x

captured = {}

def fake_neato(**kwargs):
    captured["source"] = Path(str(kwargs["source_path"])).read_text()
    Path(str(kwargs["rendered_path"])).write_text("<svg></svg>")
    return SimpleNamespace(returncode=0, stderr="", stdout="")

layout_mod._run_neato_with_fallbacks = fake_neato
torch.manual_seed(0)
trace = tl.trace(Recurrent(), torch.randn(1, 4))
out = Path(sys.argv[1]) / "sweep"
trace.draw(
    rolled=False,
    layout="rank",
    file_format="svg",
    path=str(out),
    show_containers=False,
)
normalized = captured["source"].replace(str(out), "OUT")
print(hashlib.sha256(normalized.encode()).hexdigest())
"""


@pytest.mark.heavy
def test_rank_dot_source_is_hash_seed_independent(tmp_path: Path) -> None:
    """The rank DOT byte stream is identical across PYTHONHASHSEED values.

    This is the b6 probe re-run as a pin: pre-fix, seeds 0-9 produced 3
    distinct DOT sha256 values because set-iteration order picked which
    pass's bbox survived the key collapse.
    """

    digests: set[str] = set()
    for seed in ("0", "1", "2"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        workdir = tmp_path / f"seed_{seed}"
        workdir.mkdir()
        result = subprocess.run(
            [sys.executable, "-c", _SWEEP_SCRIPT, str(workdir)],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        digests.add(result.stdout.strip().splitlines()[-1])
    assert len(digests) == 1, f"rank DOT varies with PYTHONHASHSEED: {digests}"

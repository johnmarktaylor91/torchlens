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
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.visualization._rank_layout_internal.layout as layout_mod


class _Recurrent(nn.Module):
    """One cell called three times, producing pass-qualified module regions."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.cell(x)
        return x


def _rank_dot_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Draw the recurrent model through the rank path; return its DOT source.

    neato is stubbed out (the pin is about the geometry TorchLens emits, not
    about neato), and the stub captures the written source before cleanup.
    """

    captured: dict[str, str] = {}

    def _fake_neato(**kwargs: object) -> SimpleNamespace:
        captured["source"] = Path(str(kwargs["source_path"])).read_text()
        Path(str(kwargs["rendered_path"])).write_text("<svg></svg>")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(layout_mod, "_run_neato_with_fallbacks", _fake_neato)
    trace = tl.trace(_Recurrent(), torch.randn(1, 4))
    trace.draw(
        vis_mode="unrolled",
        vis_node_placement="rank",
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "rank_geometry"),
        show_containers=False,
    )
    return captured["source"]


def _cluster_bboxes(dot_source: str) -> dict[str, str]:
    """Return ``cluster name -> bb attribute`` from raw rank-path DOT."""

    bboxes: dict[str, str] = {}
    current: str | None = None
    for line in dot_source.splitlines():
        stripped = line.strip()
        cluster_match = re.match(r'subgraph "?cluster_([^" {]+)"? \{', stripped)
        if cluster_match:
            current = cluster_match.group(1)
            continue
        bb_match = re.match(r'bb="([^"]+)"', stripped)
        if bb_match and current is not None:
            bboxes[current] = bb_match.group(1)
            current = None
    return bboxes


def test_rank_pass_clusters_get_distinct_bboxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each recurrent pass cluster carries ITS OWN geometry (R19-2).

    Pre-fix, the pass-free key collapse gave every ``cell:N`` cluster the
    same bbox — geometry that is wrong for all but (at most) one pass.
    """

    bboxes = _cluster_bboxes(_rank_dot_source(tmp_path, monkeypatch))
    passes_by_module: dict[str, dict[str, str]] = {}
    for name, bb in bboxes.items():
        match = re.match(r"(.+)_pass(\d+)$", name)
        if match:
            passes_by_module.setdefault(match.group(1), {})[match.group(2)] = bb
    recurrent = {base: bbs for base, bbs in passes_by_module.items() if len(bbs) >= 2}
    assert recurrent, f"expected recurrent pass clusters, got {sorted(bboxes)}"
    for base, per_pass in recurrent.items():
        assert len(set(per_pass.values())) > 1, (
            f"every pass of {base!r} shares one bbox — the pass-free key collapse "
            f"(last-write-wins geometry) is back: {per_pass}"
        )


def test_rank_dot_source_is_deterministic_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two identical draws emit byte-identical rank DOT."""

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _rank_dot_source(tmp_path / "a", monkeypatch)
    second = _rank_dot_source(tmp_path / "b", monkeypatch)
    assert first.replace(str(tmp_path / "a"), "") == second.replace(str(tmp_path / "b"), "")


_SWEEP_SCRIPT = r"""
import hashlib, sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import torchlens as tl
import torchlens.visualization._rank_layout_internal.layout as layout_mod

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
    vis_mode="unrolled",
    vis_node_placement="rank",
    vis_save_only=True,
    vis_fileformat="svg",
    vis_outpath=str(out),
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

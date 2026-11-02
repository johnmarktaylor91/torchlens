"""A ``save="stats"`` capture must not re-enter capture logging through the alias check.

Saving a buffer-backed op (BatchNorm's running statistics) under ``save="stats"``
applies the summary transform and then the reduce-only alias check
(``ensure_summary_output_owns_storage``). Unpaused, that check's
``untyped_storage()`` call ran the torch wrapper's buffer-registration preamble
on the very buffer being logged, which applied the transform again and
re-entered the check: a six-block CNN nested 135 levels deep and stopped within
ten frames of the interpreter limit, and a Jupyter kernel (deeper stack base)
died with a ``RecursionError`` that the check's fail-safe ``except RuntimeError``
swallowed. The op graph was unchanged, so validation never saw it. Each child
here runs in a fresh process so the measurement does not depend on captures an
earlier test made.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_CHILD = r"""
import json, sys
import torch
from torch import nn
import torchlens as tl
from torchlens._ir import summary_role

MODE = sys.argv[1]
torch.manual_seed(0)


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        return torch.relu(self.norm(self.conv(x)))


class TinyCNN(nn.Module):
    def __init__(self, channels=8, num_blocks=6, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ConvBlock(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.blocks(x)
        return self.head(torch.flatten(self.pool(x), 1))


model = TinyCNN().eval()
x = torch.randn(2, 1, 16, 16)
if MODE == "plain-first":
    tl.trace(model, x)

nesting = {"now": 0, "max": 0, "calls": 0}
_orig = summary_role.ensure_summary_output_owns_storage


def counted(output, live):
    nesting["now"] += 1
    nesting["calls"] += 1
    nesting["max"] = max(nesting["max"], nesting["now"])
    try:
        return _orig(output, live)
    finally:
        nesting["now"] -= 1


summary_role.ensure_summary_output_owns_storage = counted
trace = tl.trace(model, x, save="stats")


def _num(value):
    return None if value is None else repr(float(value))


site = tl.stats.site_stats(trace)
results = {
    "ops": len(trace.ops),
    "layers": len(trace.layers),
    "stats": {
        label: [_num(s.mean), _num(s.sum_squares), _num(s.num_zero), _num(s.numel)]
        for label, s in ((label, site[label].compute()) for label in sorted(site))
    },
}
print("RESULT " + json.dumps({"nesting": nesting, "results": results}))
"""


def _run_child(mode: str) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, mode],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
    assert completed.returncode == 0 and lines, (
        f"{mode} child failed rc={completed.returncode}\n{completed.stderr[-3000:]}"
    )
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def stats_first() -> dict:
    return _run_child("stats-first")


@pytest.fixture(scope="module")
def plain_first() -> dict:
    return _run_child("plain-first")


@pytest.mark.heavy
def test_stats_capture_never_reenters_alias_check(stats_first: dict, plain_first: dict) -> None:
    """The alias check runs once per saved op; it never nests inside itself."""

    assert stats_first["nesting"]["max"] == 1, stats_first["nesting"]
    assert plain_first["nesting"]["max"] == 1, plain_first["nesting"]
    assert stats_first["nesting"]["calls"] == plain_first["nesting"]["calls"]


@pytest.mark.heavy
def test_stats_first_capture_matches_plain_first_run(stats_first: dict, plain_first: dict) -> None:
    """Op count, layer count and every site statistic agree with a plain-first run."""

    assert stats_first["results"] == plain_first["results"]
    assert stats_first["results"]["ops"] == 23
    assert len(stats_first["results"]["stats"]) == 23

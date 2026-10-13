"""Execution check for the big-model render default on a flat 800-layer model (U07.4).

The ruling asks for one execution check before launch copy claims a readable
default on big models. The check runs the default route (the quickstart
``render`` facade, ``collapse="auto"``) on a flat ``nn.Sequential`` of 800
layers (400 ``Linear``/``ReLU`` pairs, no nesting), the shape that gives the
collapse engine no module boxes to fold.

Measured at the branch point (pinned CI venv, one worker core): the render
takes about 5 s end to end and shows 805 nodes -- the whole graph, far above
the readable band of 25. The default is therefore NOT readable on a flat model
yet; what it does guarantee today is a finished render inside the time budget
with the fallback disclosure and the FK4 expansion manifest beside the file.
The node budget is tightened once the K_CAP flood fix (collapse item 6) lands.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens.errors._base import TorchLensWarning
from torchlens.quickstart import render

#: Wall budget for the whole default render (capture included), seconds.
FLAT_800_TIME_BUDGET_S = 60.0


def _flat_model(n_layers: int = 800, width: int = 16) -> nn.Sequential:
    """Return a flat Sequential of ``n_layers`` alternating Linear/ReLU layers."""

    layers: list[nn.Module] = []
    for _ in range(n_layers // 2):
        layers += [nn.Linear(width, width), nn.ReLU()]
    return nn.Sequential(*layers)


@pytest.mark.slow
def test_flat_800_layer_default_render_within_budget(tmp_path: Path) -> None:
    """The default render finishes in budget and discloses its over-band floor."""
    torch.manual_seed(0)
    target = tmp_path / "flat800.svg"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        started = time.perf_counter()
        result = render(_flat_model(), torch.randn(2, 16), file=target, format="svg")
        elapsed = time.perf_counter() - started
    assert elapsed < FLAT_800_TIME_BUDGET_S, f"default render took {elapsed:.1f} s"
    assert target.is_file() and target.stat().st_size > 0
    assert result.total_ops is not None and result.total_ops >= 800
    codes = {
        str(getattr(record.message, "fields", {}).get("code"))
        for record in records
        if isinstance(record.message, TorchLensWarning)
    }
    disclosures = {
        "collapse_floor_fallback",
        "collapse_budget_fallback",
        "collapse_near_uncollapsed",
    }
    assert codes & disclosures, f"an over-band default must disclose its fallback: {codes}"
    manifest_path = tmp_path / "flat800.expansion.json"
    assert manifest_path.is_file(), "the over-ceiling overview must ship its expansion manifest"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["visible_count"] > manifest["readable_band_high"]

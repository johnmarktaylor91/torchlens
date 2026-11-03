"""Anything older than the current format refuses as a file saved by torchlens 2.x.

Artifacts written at the current ``tlspec_version`` load; any older stamp refuses with the
typed ``ArtifactTooOldError`` that names the 2.x origin, instead of being resurrected through
legacy field-alias ladders.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


def _build_trace() -> tl.Trace:
    """Capture a small deterministic trace."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    return tl.trace(model, torch.randn(2, 4), save="all")


def test_current_artifacts_load_clean(tmp_path: Path) -> None:
    """A current save round-trips with no floor refusal and no version warning."""

    trace = _build_trace()
    path = tmp_path / "current.tlspec"
    trace.save(path)

    loaded = tl.load(path)

    assert isinstance(loaded, tl.Trace)
    assert [layer.layer_label for layer in loaded._layer_list] == [
        layer.layer_label for layer in trace._layer_list
    ]

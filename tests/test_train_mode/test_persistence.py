"""Persistence tests for train-mode session state."""

from __future__ import annotations

from pathlib import Path

import torch

import torchlens as tl
from .conftest import TwoLayerMlp


def test_train_mode_dropped_on_load(tmp_path: Path, two_layer_mlp: TwoLayerMlp) -> None:
    """Portable bundle loading restores the session-time backward_ready default."""

    trace = tl.trace(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        backward_ready=True,
        random_seed=0,
    )
    bundle_path = tmp_path / "backward_ready.tl"

    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path)

    assert trace.backward_ready is True
    assert loaded.backward_ready is False
    trace.cleanup()
    loaded.cleanup()

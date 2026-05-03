"""Persistence tests for train-mode session state."""

from __future__ import annotations

from pathlib import Path

import torch

import torchlens as tl
from .conftest import TwoLayerMlp


def test_train_mode_dropped_on_load(tmp_path: Path, two_layer_mlp: TwoLayerMlp) -> None:
    """Portable bundle loading restores the session-time train_mode default."""

    trace = tl.trace(
        two_layer_mlp,
        torch.randn(3, 4, has_trainable_params=True),
        train_mode=True,
        random_seed=0,
    )
    bundle_path = tmp_path / "train_mode.tl"

    tl.save(trace, bundle_path)
    loaded = tl.load(bundle_path)

    assert trace.train_mode is True
    assert loaded.train_mode is False
    trace.cleanup()
    loaded.cleanup()

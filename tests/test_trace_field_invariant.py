"""Regression tests for final Trace field ownership."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


def test_public_cached_property_read_does_not_poison_save(tmp_path) -> None:
    """Assert reading ``Trace.recipe`` never breaks ``Trace.save``.

    ``recipe`` is a ``functools.cached_property``, so a read caches
    its value under its own name in ``Trace.__dict__``. The tlspec scrub walks
    the live ``__dict__`` fail-closed, and before B3R4-R10-1 the undeclared
    cache key made every later ``Trace.save`` raise ``ArtifactError`` -- a
    read-only public accessor permanently poisoning the artifact path.

    Parameters
    ----------
    tmp_path:
        Pytest-provided temporary directory.

    Returns
    -------
    None
        Fails if the post-read save raises or the artifact does not load.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))

    spec = trace.recipe
    assert spec is not None
    assert "recipe" in trace.__dict__

    save_path = tmp_path / "after_read.tlspec"
    trace.save(str(save_path))

    loaded = tl.load(str(save_path))
    # The cache is derived state: it must rebuild on the loaded trace, not
    # round-trip as a portable field.
    assert "recipe" not in loaded.__dict__
    assert loaded.recipe is not None

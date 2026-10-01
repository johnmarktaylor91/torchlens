"""FLOORLEAK regression pins: test fixtures must not outlive the tests they serve.

Two session-pollution leaks flipped later tests red in the W051 backstops:

* ``tests/real_model/r0/conftest.py`` cached every realism-sweep model and its
  finished ``Trace`` in a SESSION-scoped dict, so the traces' saved activations
  (which carry autograd history) pinned every family's live forward graph for
  the rest of the process and the process-wide live-holder tripwires in
  ``tests/test_brainpipe_capture_floor.py`` read red for every later test.
* ``tests/test_fastlog/test_ddp_unwrap.py`` captured under an initialized
  process group, which lazily ARMS distributed capture; the fixture destroyed
  the group but never disarmed, so the completeness dispatch witness rode every
  later capture in the session (its own pin lives beside that fixture).

These pins exercise the fixture lifetimes directly, in-process, so a future
scope widening fails here rather than three thousand tests later.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
import torch.nn as nn

import torchlens as tl

pytestmark = [pytest.mark.smoke]


class _TinyConv(nn.Module):
    """Two-conv stack whose activations match the floor pins' census shape."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Conv2d(16, 16, 3, padding=1)
        self.b = nn.Conv2d(16, 16, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.b(torch.relu(self.a(x))))


def test_realism_capture_cache_is_package_scoped() -> None:
    """The realism-sweep cache must close with its package, never the session."""

    from tests.real_model.r0 import conftest as r0_conftest

    definition = r0_conftest.r0_capture_cache
    marker = getattr(definition, "_fixture_function_marker", None)
    scope = getattr(marker, "scope", None)
    assert scope == "package", (
        f"r0_capture_cache is {scope!r}-scoped; a cache that outlives the R0 package"
        " pins every family's Trace (and its live autograd graph) for the rest of"
        " the session (FLOORLEAK)"
    )


def test_realism_cache_lifetime_releases_cached_traces_on_close() -> None:
    """Closing the cache lifetime frees the cached Trace and its live graph."""

    from tests.real_model.r0.conftest import r0_cache_lifetime

    lifetime = r0_cache_lifetime()
    cache = next(lifetime)

    model = _TinyConv()
    x = torch.randn(2, 16, 8, 8)
    trace = tl.trace(model, x)
    cache[("tiny", "eager")] = trace
    trace_ref = weakref.ref(trace)
    # A saved activation under grad keeps its producer's graph alive: this is
    # exactly the population the floor pins count after a leaked capture.
    live_ref = weakref.ref(trace["conv2d_1_1"].out)
    del trace

    gc.collect()
    assert trace_ref() is not None, "the cache must keep the Trace while open"

    lifetime.close()

    assert cache == {}, "closing the lifetime must empty the cache"
    assert trace_ref() is None, "the cached Trace outlived the cache lifetime"
    assert live_ref() is None, "the cached Trace's saved activation outlived the cache"

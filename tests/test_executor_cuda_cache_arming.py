"""Liveness killer for the CUDA-cache step of the postprocess executor.

The executor mutation family neuters ``_run_clear_cuda_cache`` to ``return None``
(the step silently skips). On a CPU runner no capture touches CUDA, so no suite
test could tell the neutered step from the real one and the mutant survived.
These tests drive the step directly with recorders standing in for the CUDA
availability probe, ``capture_touched_cuda`` and ``torch.cuda.empty_cache``,
and pin the gate: the allocator is flushed exactly when CUDA is available AND
the capture touched CUDA, and never otherwise.

This file is enrolled in the mutation driver's ``SUITE`` list; removing it from
there resurrects the survivor.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from torchlens._postprocess import _executor


@pytest.mark.parametrize(
    ("cuda_available", "touched_cuda", "expect_flush"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_clear_cuda_cache_flushes_only_when_cuda_available_and_capture_touched_it(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    touched_cuda: bool,
    expect_flush: bool,
) -> None:
    """``empty_cache`` runs once when both facts hold, zero times otherwise."""

    flushes: list[tuple[Any, ...]] = []
    touched_queries: list[Any] = []

    def touched(trace: Any) -> bool:
        touched_queries.append(trace)
        return touched_cuda

    monkeypatch.setattr(_executor._cuda_probe, "_is_cuda_available", lambda: cuda_available)
    monkeypatch.setattr(_executor._tensor_utils, "capture_touched_cuda", touched)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *args: flushes.append(args))

    trace = object()
    _executor._run_clear_cuda_cache(SimpleNamespace(trace=trace))

    assert len(flushes) == (1 if expect_flush else 0)
    # Whenever the touched-CUDA fact is consulted it is asked about this trace.
    assert all(queried is trace for queried in touched_queries)

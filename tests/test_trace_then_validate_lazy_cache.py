"""Trace-then-validate on a model that builds a cache lazily in its forward.

A default ``tl.trace`` is a pure observer: it puts back every plain attribute
the forward wrote, so a lazily built cache is empty again after the trace.
Validating right after therefore starts from the pristine model, whose first
forward is a one-time execution, and the fresh re-trace check reports it
(``retrace_mismatch``). After one plain forward the model is in its steady
state and validates. ``tests/test_classics_corpus.py`` relies on this chain.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TraceNotReproducibleWarning
from torchlens.validation import last_validation_failure


class LazyCacheModel(nn.Module):
    """Linear layer plus an offset cache the forward builds on its first call."""

    def __init__(self) -> None:
        """Initialize the layer and the empty cache slot."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.cache: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Build the cache on first use, then add it to the linear output.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Linear output plus the cached offset.
        """

        if self.cache is None:
            self.cache = torch.zeros(x.shape[-1])
        return self.linear(x) + self.cache


def test_default_trace_restores_lazy_cache_so_validate_sees_the_first_forward() -> None:
    """Trace restores the cache, validate reports the one-time forward, warm-up fixes it."""

    model = LazyCacheModel()
    x = torch.randn(2, 3)

    tl.trace(model, x)
    assert model.cache is None

    with pytest.warns(TraceNotReproducibleWarning, match="stateful/non-reproducible"):
        assert tl.validate(model, x, scope="forward") is False
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check_name == "retrace_mismatch"

    with torch.no_grad():
        model(x)
    assert model.cache is not None
    assert tl.validate(model, x, scope="forward") is True

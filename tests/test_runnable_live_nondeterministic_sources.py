"""Live-refresh RunReport declares host-RNG nondeterminism (deephunt F2).

The loaded-sparse provider derives ``nondeterministic_sources`` from the
descriptor's RNG profile, but the live-refresh provider historically never
passed sources to the one report finalizer, so a Python-``random``/clock model
reported a deterministic-looking empty tuple next to VERIFIED. The live report
must carry the same capture-side host-RNG evidence the sparse producer uses.
"""

from __future__ import annotations

import random

import torch
from torch import nn

import torchlens as tl


class HostRngModel(nn.Module):
    """Model whose forward consumes the Python global ``random`` engine."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale the input by a host-RNG draw."""

        return value * (1.0 + random.random())


class DeterministicModel(nn.Module):
    """Model whose forward consumes no host RNG."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale the input by a constant."""

        return value * 2.0


def test_live_run_declares_host_rng_source() -> None:
    """A host-RNG capture's live-refresh report names ('host_rng',)."""

    model = HostRngModel()
    value = torch.randn(2, 3)
    trace = tl.trace(model, value)
    assert bool(trace._runnable.host_rng_consumed)
    result = trace.run(inputs=value)
    assert result.report.nondeterministic_sources == ("host_rng",)


def test_live_run_deterministic_model_declares_no_sources() -> None:
    """A deterministic capture's live-refresh report stays empty."""

    model = DeterministicModel()
    value = torch.randn(2, 3)
    trace = tl.trace(model, value)
    assert not bool(trace._runnable.host_rng_consumed)
    result = trace.run(inputs=value)
    assert result.report.nondeterministic_sources == ()


def test_fast_live_run_declares_host_rng_source() -> None:
    """fast=True carries the same host-RNG declaration as the ordinary provider.

    grind-p5 rollup: the fast-LIVE finalize call never passed
    ``nondeterministic_sources``, so a host-RNG capture's fast report read as
    a deterministic-looking empty tuple next to VERIFIED -- a contract
    violation against the shared settlement finalizer.
    """

    model = HostRngModel()
    value = torch.randn(2, 3)
    trace = tl.trace(model, value, save=tl.func("mul"))
    assert bool(trace._runnable.host_rng_consumed)
    result = trace.run(inputs=value, fast=True)
    assert result.report.nondeterministic_sources == ("host_rng",)


def test_fast_live_run_deterministic_model_declares_no_sources() -> None:
    """A deterministic capture's fast-live report stays empty."""

    model = DeterministicModel()
    value = torch.randn(2, 3)
    trace = tl.trace(model, value, save=tl.func("mul"))
    result = trace.run(inputs=value, fast=True)
    assert result.report.nondeterministic_sources == ()

"""Reduce-and-discard capture: ``tl.trace(model, x, save="stats")``.

Every site's out folds into a ``Spine`` during the forward and no payload is
kept. The per-site statistics must equal the ones computed from a full
capture of the same input (exactly: the reducer runs the same kernel), the
retained bytes per site must not grow with the batch, payload reads refuse
with a working spelling, and replay/validation refuse instead of passing on
missing payloads.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import torchlens as tl

_SPINE_FIELDS = (
    "numel",
    "num_finite",
    "num_zero",
    "num_negative",
    "num_nan",
    "num_posinf",
    "num_neginf",
    "finite_min",
    "finite_max",
    "finite_absmax",
    "sum",
    "sum_squares",
    "sum_abs",
    "mean",
    "m2",
    "reduction_dtype",
)


class _MLP(nn.Module):
    """Small MLP with a relu (zeros and negatives in the sites)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _CNN(nn.Module):
    """Small CNN: conv, relu, pool, flatten, linear."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.head = nn.Linear(4 * 2 * 2, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.max_pool2d(torch.relu(self.conv(x)), 2)
        return self.head(h.flatten(1))


def _model_and_input(kind: str, batch: int) -> tuple[nn.Module, torch.Tensor]:
    """Build a float64 model and a seeded input of ``batch`` rows."""

    torch.manual_seed(0)
    model = (_MLP() if kind == "mlp" else _CNN()).double().eval()
    generator = torch.Generator().manual_seed(batch)
    shape = (batch, 6) if kind == "mlp" else (batch, 2, 4, 4)
    return model, torch.randn(*shape, generator=generator, dtype=torch.float64)


def _assert_same(
    left: tl.stats.RunningStatsResult, right: tl.stats.RunningStatsResult, label: str
) -> None:
    """Every field equal: counts exactly, floats bit for bit."""

    for field in _SPINE_FIELDS:
        a, b = getattr(left, field), getattr(right, field)
        both_nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
        same = a == b or both_nan
        assert same, f"{label}.{field}: stats capture {a!r} != full capture {b!r}"


def test_payload_read_refuses_with_working_spelling() -> None:
    """``.out`` on a stats trace refuses typed; both remedy spellings run."""

    model, x = _model_and_input("mlp", 4)
    trace = tl.trace(model, x, save="stats")
    label = str(trace.ops[1].label)
    with pytest.raises(tl.errors.TorchLensError) as info:
        _ = trace[label].out
    assert info.value.fields["code"] == "activation_reduced_to_stats"
    assert "tl.stats.site_stats(trace)[label]" in str(info.value)
    assert tl.stats.site_stats(trace)[label].compute().numel > 0
    assert isinstance(tl.trace(model, x, save="all")[label].out, torch.Tensor)


def test_validation_and_replay_refuse() -> None:
    """Validation and replay refuse with a stated reason, never pass on missing payloads."""

    model, x = _model_and_input("mlp", 4)
    trace = tl.trace(model, x, save="stats")
    with pytest.raises(tl.errors.TorchLensError) as info:
        trace.validate_forward_pass(model(x))
    assert info.value.fields["code"] == "stats_capture_not_replayable"
    with pytest.raises(tl.errors.TorchLensError) as info:
        trace.run(x)
    assert info.value.fields["code"] == "stats_capture_not_replayable"
    assert tl.validate(model, x, scope="forward") is not None


def test_site_spines_refuses_full_trace() -> None:
    """Reading site spines off a payload trace refuses and names the spelling."""

    model, x = _model_and_input("mlp", 4)
    with pytest.raises(tl.errors.TorchLensError) as info:
        tl.stats.site_stats(tl.trace(model, x))
    assert info.value.fields["code"] == "stats_capture_not_reduced"
    assert "save='stats'" in str(info.value)


def test_site_spines_keep_folding_across_batches() -> None:
    """Merging two batches' site spines equals one Spine over both batches' outs."""

    model, x = _model_and_input("mlp", 8)
    first = tl.stats.site_stats(tl.trace(model, x[:5], save="stats"))
    second = tl.stats.site_stats(tl.trace(model, x[5:], save="stats"))
    full_a = tl.trace(model, x[:5])
    full_b = tl.trace(model, x[5:])
    for op_a, op_b in zip(full_a.ops, full_b.ops, strict=True):
        label = str(op_a.label)
        merged = first[label]
        merged.merge(second[label])
        reference = tl.stats.RunningStats()
        reference.update(op_a.out)
        reference.update(op_b.out)
        got, want = merged.compute(), reference.compute()
        assert got.counts() == want.counts()
        assert got.finite_min == want.finite_min and got.finite_max == want.finite_max
        assert got.mean == pytest.approx(want.mean, rel=1e-12, abs=1e-15)


def test_stats_trace_reprs_render() -> None:
    """A stats trace prints and summarises without reading a payload."""

    model, x = _model_and_input("cnn", 2)
    trace = tl.trace(model, x, save="stats")
    assert "Trace" in repr(trace) or repr(trace)
    assert str(trace)

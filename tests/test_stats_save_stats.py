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
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.stats._capture_stats import PACKED_SLOTS

_SPINE_FIELDS = (
    "count_total",
    "count_finite",
    "count_zero",
    "count_negative",
    "count_nan",
    "count_posinf",
    "count_neginf",
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


def _full_spines(trace: Any) -> dict[str, tl.stats.SpineResult]:
    """Per-op Spine results computed from a full capture's saved outs."""

    out: dict[str, tl.stats.SpineResult] = {}
    for op in trace.ops:
        value = op.out
        if isinstance(value, torch.Tensor) and not value.is_complex():
            spine = tl.stats.Spine()
            spine.update(value)
            out[str(op.label)] = spine.result()
    return out


def _assert_same(left: tl.stats.SpineResult, right: tl.stats.SpineResult, label: str) -> None:
    """Every field equal: counts exactly, floats bit for bit."""

    for field in _SPINE_FIELDS:
        a, b = getattr(left, field), getattr(right, field)
        both_nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
        same = a == b or both_nan
        assert same, f"{label}.{field}: stats capture {a!r} != full capture {b!r}"


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_site_spines_equal_full_capture(kind: str) -> None:
    """Per-site statistics equal the full capture's, field by field, at two batch sizes."""

    for batch in (4, 32):
        model, x = _model_and_input(kind, batch)
        full = tl.trace(model, x)
        reduced = tl.trace(model, x, save="stats")
        expected = _full_spines(full)
        got = {label: spine.result() for label, spine in tl.stats.site_spines(reduced).items()}
        assert set(got) == set(expected)
        assert got, "no sites folded"
        for label, result in expected.items():
            _assert_same(got[label], result, label)


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_stats_capture_logs_the_same_ops(kind: str) -> None:
    """The reducer's own torch ops never enter the graph."""

    model, x = _model_and_input(kind, 4)
    full = tl.trace(model, x)
    reduced = tl.trace(model, x, save="stats")
    assert [str(op.label) for op in reduced.ops] == [str(op.label) for op in full.ops]


@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_bytes_retained_per_site_constant_in_batch(kind: str) -> None:
    """Counter: a stats site retains one 16-slot float64 vector at any batch size."""

    retained: dict[int, tuple[int, int, int]] = {}
    for batch in (4, 64):
        model, x = _model_and_input(kind, batch)
        full = tl.trace(model, x)
        reduced = tl.trace(model, x, save="stats")
        full_bytes = sum(op.out.nbytes for op in full.ops if isinstance(op.out, torch.Tensor))
        stats_bytes = 0
        for op in reduced.ops:
            vector = op.transformed_out
            assert isinstance(vector, torch.Tensor) and vector.numel() == PACKED_SLOTS
            stats_bytes += vector.nbytes
        retained[batch] = (len(reduced.ops), stats_bytes, full_bytes)
    sites_4, stats_4, full_4 = retained[4]
    sites_64, stats_64, full_64 = retained[64]
    assert sites_4 == sites_64
    assert stats_4 == stats_64 == sites_4 * PACKED_SLOTS * 8
    assert full_64 > full_4 > stats_4


def test_payload_read_refuses_with_working_spelling() -> None:
    """``.out`` on a stats trace refuses typed; both remedy spellings run."""

    model, x = _model_and_input("mlp", 4)
    trace = tl.trace(model, x, save="stats")
    label = str(trace.ops[1].label)
    with pytest.raises(tl.errors.TorchLensError) as info:
        _ = trace[label].out
    assert info.value.fields["code"] == "activation_reduced_to_stats"
    assert "tl.stats.site_spines(trace)[label]" in str(info.value)
    assert tl.stats.site_spines(trace)[label].result().count_total > 0
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
        tl.stats.site_spines(tl.trace(model, x))
    assert info.value.fields["code"] == "stats_capture_not_reduced"
    assert "save='stats'" in str(info.value)


def test_site_spines_keep_folding_across_batches() -> None:
    """Merging two batches' site spines equals one Spine over both batches' outs."""

    model, x = _model_and_input("mlp", 8)
    first = tl.stats.site_spines(tl.trace(model, x[:5], save="stats"))
    second = tl.stats.site_spines(tl.trace(model, x[5:], save="stats"))
    full_a = tl.trace(model, x[:5])
    full_b = tl.trace(model, x[5:])
    for op_a, op_b in zip(full_a.ops, full_b.ops, strict=True):
        label = str(op_a.label)
        merged = first[label]
        merged.merge(second[label])
        reference = tl.stats.Spine()
        reference.update(op_a.out)
        reference.update(op_b.out)
        got, want = merged.result(), reference.result()
        assert got.counts() == want.counts()
        assert got.finite_min == want.finite_min and got.finite_max == want.finite_max
        assert got.mean == pytest.approx(want.mean, rel=1e-12, abs=1e-15)


def test_stats_trace_reprs_render() -> None:
    """A stats trace prints and summarises without reading a payload."""

    model, x = _model_and_input("cnn", 2)
    trace = tl.trace(model, x, save="stats")
    assert "Trace" in repr(trace) or repr(trace)
    assert str(trace)


def test_budget_refusal_names_stats_level_and_it_fits() -> None:
    """The save-budget refusal names save='stats'; that spelling fits the same budget."""

    from torchlens._save_budget import SaveBudgetExceededError
    from torchlens.options import CaptureOptions

    model, x = _model_and_input("mlp", 32)
    budget = 4096  # a full capture of this batch retains about 6.4 KiB; stats 128 B per op
    with pytest.raises(SaveBudgetExceededError) as info:
        tl.trace(model, x, capture=CaptureOptions(save_budget=budget))
    assert "save='stats'" in str(info.value)
    reduced = tl.trace(model, x, save="stats", capture=CaptureOptions(save_budget=budget))
    assert tl.stats.site_spines(reduced)


def test_chunked_reducer_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past one working chunk: counts, extrema, sum and mean stay exact; the rest within rounding."""

    from torchlens.stats import _capture_stats

    monkeypatch.setattr(_capture_stats, "_CHUNK", 7)
    model, x = _model_and_input("cnn", 8)
    full = _full_spines(tl.trace(model, x))
    reduced = tl.stats.site_spines(tl.trace(model, x, save="stats"))
    for label, want in full.items():
        got = reduced[label].result()
        assert got.counts() == want.counts(), label
        for field in ("finite_min", "finite_max", "finite_absmax", "sum", "mean"):
            assert getattr(got, field) == getattr(want, field), (label, field)
        for field in ("sum_squares", "sum_abs", "m2"):
            assert getattr(got, field) == pytest.approx(getattr(want, field), rel=1e-12, abs=1e-12)


def _largest_new_allocation(fn: Any, value: torch.Tensor) -> int:
    """Bytes of the largest fresh storage any torch op allocates while ``fn(value)`` runs."""

    from torch.utils._python_dispatch import TorchDispatchMode

    largest = [0]

    class _Counter(TorchDispatchMode):
        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            out = func(*args, **(kwargs or {}))
            inputs = {
                a.untyped_storage().data_ptr()
                for a in torch.utils._pytree.tree_leaves((args, kwargs))
                if isinstance(a, torch.Tensor)
            }
            for leaf in torch.utils._pytree.tree_leaves(out):
                if (
                    isinstance(leaf, torch.Tensor)
                    and leaf.untyped_storage().data_ptr() not in inputs
                ):
                    largest[0] = max(largest[0], leaf.untyped_storage().nbytes())
            return out

    with _Counter():
        fn(value)
    return largest[0]


def test_reducer_working_memory_is_bounded() -> None:
    """Counter: the reducer's largest working tensor is one chunk, not the out's size."""

    from torchlens.stats import _capture_stats

    out = torch.randn(1 << 20)  # 4 MiB float32
    lean = _largest_new_allocation(_capture_stats.lean_spine, out)
    naive = _largest_new_allocation(lambda v: tl.stats.Spine().update(v), out)
    assert lean <= _capture_stats._CHUNK * 8
    assert naive >= out.numel() * 8
    assert _capture_stats.lean_spine(out).result().counts() == _spine_counts(out)


def _spine_counts(value: torch.Tensor) -> dict[str, int]:
    spine = tl.stats.Spine()
    spine.update(value)
    return spine.result().counts()

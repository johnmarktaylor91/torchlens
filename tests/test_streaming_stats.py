"""Phase 4 streaming statistics tests."""

from __future__ import annotations

import tracemalloc

import pytest
import torch

import torchlens as tl


def test_mean_streams_10k_batches_under_memory_bound() -> None:
    """Stream many batches without retaining them."""

    stat = tl.stats.Mean(name="x")
    tracemalloc.start()
    for index in range(10_000):
        stat.update(torch.tensor([float(index % 10)]))
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert stat.result() == pytest.approx(4.5)
    assert peak < 200 * 1024 * 1024


def test_streaming_aggregators_smoke() -> None:
    """Exercise all Phase 4 stat accumulators."""

    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    stats = [
        tl.stats.Mean(),
        tl.stats.Quantile([0.5]),
        tl.stats.TopK(3),
        tl.stats.Covariance(),
        tl.stats.PCA(2),
        tl.stats.Aggregator(tl.stats.Mean(name="mean"), tl.stats.TopK(1, name="top")),
    ]
    for stat in stats:
        stat.update(values)
        assert stat.result() is not None


def test_aggregate_smoke_on_small_dataloader() -> None:
    """Aggregate one output metric through a tiny model."""

    model = torch.nn.Linear(2, 1)
    batches = [torch.ones(1, 2), torch.zeros(1, 2)]
    result = tl.aggregate(model, batches, {"output": tl.stats.Mean()})
    assert "output" in result

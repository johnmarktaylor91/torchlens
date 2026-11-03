"""Phase 4 streaming statistics tests."""

from __future__ import annotations

import tracemalloc

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning


def _manual_covariance(matrix: torch.Tensor) -> torch.Tensor:
    """Return the sample covariance of a row-wise feature matrix.

    Parameters
    ----------
    matrix:
        Two-dimensional tensor with observations in rows.

    Returns
    -------
    torch.Tensor
        Sample covariance matrix in ``float64``.
    """

    centered = matrix.to(dtype=torch.float64) - matrix.to(dtype=torch.float64).mean(
        dim=0, keepdim=True
    )
    return centered.T @ centered / (matrix.shape[0] - 1)


class _TwoLinearOutputs(torch.nn.Module):
    """Tiny model with two saved linear sites for ambiguity tests."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3)
        self.fc2 = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.fc2(torch.relu(self.fc1(x)))


def test_mean_streams_10k_batches_under_memory_bound() -> None:
    """Stream many batches without retaining them."""

    stat = tl.stats.Mean(name="x")
    tracemalloc.start()
    for index in range(10_000):
        stat.update(torch.tensor([float(index % 10)]))
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert stat.compute() == pytest.approx(4.5)
    assert peak < 200 * 1024 * 1024


def test_streaming_aggregators_smoke() -> None:
    """Exercise all Phase 4 stat accumulators against explicit references."""

    values = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 3.0],
            [2.0, 1.0, 1.0],
            [4.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    mean = tl.stats.Mean()
    quantile = tl.stats.Quantile([0.5])
    topk = tl.stats.TopK(3)
    covariance = tl.stats.Covariance()
    pca = tl.stats.PCA(2)
    aggregator = tl.stats.Aggregator(tl.stats.Mean(name="mean"), tl.stats.TopK(1, name="top"))

    for stat in (mean, quantile, topk, covariance, pca, aggregator):
        stat.update(values)

    expected_covariance = _manual_covariance(values)
    expected_eigenvalues, _expected_eigenvectors = torch.linalg.eigh(expected_covariance)
    expected_explained_variance = torch.flip(expected_eigenvalues, dims=(0,))[:2]
    pca_result = pca.compute()

    assert mean.compute() == pytest.approx(float(values.to(dtype=torch.float64).mean().item()))
    assert quantile.compute() == {0.5: pytest.approx(1.0)}
    assert topk.compute() == [4.0, 3.0, 2.0]
    assert torch.allclose(covariance.compute(), expected_covariance)
    assert torch.allclose(pca_result["explained_variance"], expected_explained_variance)
    assert pca_result["components"].shape == (2, values.shape[1])
    projected_covariance = (
        pca_result["components"] @ expected_covariance @ pca_result["components"].T
    )
    assert torch.allclose(
        torch.diag(projected_covariance),
        expected_explained_variance,
    )
    assert aggregator.compute() == {"mean": pytest.approx(1.4166666666666667), "top": [4.0]}


def test_aggregate_smoke_on_small_dataloader() -> None:
    """Aggregate one output metric through a tiny model with an exact reference."""

    model = torch.nn.Linear(2, 1)
    batches = [torch.ones(1, 2), torch.zeros(1, 2)]
    result = tl.stats.aggregate(model, batches, {"output": tl.stats.Mean()})
    with torch.no_grad():
        expected_values = torch.cat([model(batch).reshape(-1) for batch in batches])
    assert result["output"] == pytest.approx(
        float(expected_values.to(dtype=torch.float64).mean().item())
    )


def test_aggregate_warns_on_ambiguous_saved_output_selector_and_uses_first_match() -> None:
    """Ambiguous substring matches should warn and preserve first-match behavior."""

    model = _TwoLinearOutputs()
    batch = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    trace = tl.trace(model, batch, save=tl.func("linear"))
    matches = [
        layer
        for layer in trace._layer_list
        if "linear" in str(layer.layer_label) and layer.has_saved_activation
    ]
    expected = float(matches[0].out.detach().to(dtype=torch.float64).mean().item())
    trace.cleanup()

    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        result = tl.stats.aggregate(model, [batch], {"linear": tl.stats.Mean()})

    assert result["linear"] == pytest.approx(expected)


def test_covariance_rejects_feature_dimension_changes() -> None:
    """Covariance should raise a clear error when feature width changes."""

    stat = tl.stats.Covariance()
    stat.update(torch.ones(2, 3))

    with pytest.raises(ValueError, match="feature dimensions cannot change"):
        stat.update(torch.ones(2, 4))


def test_pca_rejects_feature_dimension_changes() -> None:
    """PCA should inherit the covariance feature-width guard."""

    stat = tl.stats.PCA(2)
    stat.update(torch.ones(2, 3))

    with pytest.raises(ValueError, match="feature dimensions cannot change"):
        stat.update(torch.ones(2, 4))


def test_norm_is_mean_of_per_update_tensor_norms() -> None:
    """Norm documents per-update behavior rather than batch-size invariance."""

    rows = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    batched = tl.stats.Norm()
    batched.update(rows)

    split = tl.stats.Norm()
    split.update(rows[:1])
    split.update(rows[1:])

    expected_batched = float(
        torch.linalg.vector_norm(rows.reshape(-1).to(dtype=torch.float64)).item()
    )
    expected_split = float(torch.linalg.vector_norm(rows[0].to(dtype=torch.float64)).item())

    assert batched.compute() == pytest.approx(expected_batched)
    assert split.compute() == pytest.approx(expected_split)
    assert batched.compute() != pytest.approx(split.compute())


def test_quantile_uses_private_seeded_rng_not_the_global_stream() -> None:
    """R13-7: Quantile sampling is seedable and decoupled from global random.

    The reservoir used to draw from the process-global ``random`` module --
    the same engine the capture pipeline snapshots/restores as protected
    state -- so identical inputs under two global seeds produced medians ~19%
    apart and estimates were irreproducible. Sampling now uses a private
    ``random.Random(seed)``: identical streams give identical estimates
    regardless of the global seed, and updating never advances the global
    engine.
    """

    import random as global_random

    data = torch.arange(100_000, dtype=torch.float64)

    def run_with_global_seed(seed: int) -> float:
        global_random.seed(seed)
        stat = tl.stats.Quantile(quantiles=(0.5,), reservoir_size=256)
        stat.update(data)
        return stat.compute()[0.5]

    assert run_with_global_seed(1) == run_with_global_seed(2)

    # The private stream never perturbs the protected global engine.
    global_random.seed(1234)
    checkpoint = global_random.getstate()
    stat = tl.stats.Quantile(quantiles=(0.5,), reservoir_size=16)
    stat.update(data[:4096])
    assert global_random.getstate() == checkpoint

    # Distinct explicit seeds give distinct (but internally deterministic)
    # subsamples; the same seed reproduces exactly.
    first = tl.stats.Quantile(quantiles=(0.5,), reservoir_size=256, seed=7)
    second = tl.stats.Quantile(quantiles=(0.5,), reservoir_size=256, seed=7)
    first.update(data)
    second.update(data)
    assert first.compute() == second.compute()


def test_topk_batched_update_matches_reference_semantics() -> None:
    """R29-5: the torch.topk-based update keeps exact top-k union semantics."""

    stat = tl.stats.TopK(k=5)
    chunks = [
        torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0]),
        torch.tensor([9.0, 2.0, 6.0]),
        torch.tensor([5.0, 3.0, 5.0]),  # duplicates retained like the old heap
        torch.tensor([float("inf"), -1.0, float("nan")]),  # NaN has no order
    ]
    for chunk in chunks:
        stat.update(chunk)
    everything = torch.cat(chunks)
    expected = sorted((float(v) for v in everything.tolist() if v == v), reverse=True)[:5]
    assert stat.compute() == expected


def test_covariance_batched_update_matches_torch_cov_across_uneven_chunks() -> None:
    """R29-5: the Chan batch combine equals torch.cov over any chunking."""

    torch.manual_seed(0)
    data = torch.randn(37, 8, dtype=torch.float64)
    stat = tl.stats.Covariance()
    for chunk in torch.split(data, [1, 4, 17, 2, 13]):
        stat.update(chunk)
    assert torch.allclose(stat.compute(), torch.cov(data.T), atol=1e-12)

    cross = tl.stats.CrossCovariance()
    other = torch.randn(37, 5, dtype=torch.float64)
    for chunk_a, chunk_b in zip(
        torch.split(data, [1, 4, 17, 2, 13]), torch.split(other, [1, 4, 17, 2, 13])
    ):
        cross.update(chunk_a, chunk_b)
    centered_a = data - data.mean(dim=0)
    centered_b = other - other.mean(dim=0)
    expected_cross = centered_a.T @ centered_b / (data.shape[0] - 1)
    assert torch.allclose(cross.compute(), expected_cross, atol=1e-12)

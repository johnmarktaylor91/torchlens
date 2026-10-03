"""Behavioral tests for streaming-stat edge semantics and the aggregate fast path.

The compiled sparse ``aggregate(target="out")`` plan only arms on fully-eval
models, so these tests pin the load-bearing contract: fast-path results are
identical to the exact per-batch full-trace path, including the fingerprint
drift fallback on models whose op stream changes between batches.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import torchlens as tl


@pytest.mark.smoke
def test_mean_and_norm_handle_empty_updates_and_report_nan_when_unfed() -> None:
    """Empty batches are ignored and unfed accumulators report NaN."""

    mean = tl.stats.Mean()
    assert math.isnan(mean.result())
    mean.update(torch.empty(0))
    assert math.isnan(mean.result())
    mean.update([1.0, 3.0])
    assert mean.result() == pytest.approx(2.0)

    norm = tl.stats.Norm()
    norm.update(torch.empty(0))
    assert math.isnan(norm.result())


@pytest.mark.smoke
def test_quantile_reservoir_stays_bounded_and_estimates_within_range() -> None:
    """A tiny reservoir keeps streaming and estimates stay inside the data range."""

    quantile = tl.stats.Quantile(quantiles=(0.5,), reservoir_size=4)
    assert math.isnan(quantile.result()[0.5])

    values = [float(i) for i in range(50)]
    quantile.update(torch.tensor(values))
    estimate = quantile.result()[0.5]
    assert min(values) <= estimate <= max(values)


def test_topk_with_nonpositive_k_stays_empty() -> None:
    """k=0 disables tracking without error."""

    top = tl.stats.TopK(k=0)
    top.update(torch.tensor([5.0, 1.0]))
    assert top.result() == []


def test_covariance_accepts_one_dimensional_rows_and_degenerate_counts() -> None:
    """1-d updates count as single rows; <2 rows produce empty/zero results."""

    empty = tl.stats.Covariance()
    assert empty.result().shape == (0, 0)

    single = tl.stats.Covariance()
    single.update(torch.tensor([1.0, 2.0]))
    assert torch.equal(single.result(), torch.zeros((2, 2), dtype=torch.float64))

    streamed = tl.stats.Covariance()
    streamed.update(torch.tensor([1.0, 2.0]))
    streamed.update(torch.tensor([3.0, 5.0]))
    stacked = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64)
    assert torch.allclose(streamed.result(), torch.cov(stacked.T))


@pytest.mark.smoke
def test_cross_covariance_edge_semantics() -> None:
    """Scalar inputs refuse, 1-d rows unsqueeze, and feature widths are pinned."""

    cross = tl.stats.CrossCovariance()
    assert cross.result().shape == (0, 0)

    with pytest.raises(ValueError, match="at least one dimension"):
        cross.update(torch.tensor(1.0), torch.tensor(2.0))

    cross.update(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))
    assert torch.equal(cross.result(), torch.zeros((2, 3), dtype=torch.float64))

    with pytest.raises(ValueError, match="feature dimensions cannot change"):
        cross.update(torch.ones(1, 5), torch.ones(1, 3))
    with pytest.raises(ValueError, match="feature dimensions cannot change"):
        cross.update(torch.ones(1, 2), torch.ones(1, 7))


@pytest.mark.smoke
def test_streaming_cka_zero_variance_is_nan_and_one_shot_rejects_mismatched_rows() -> None:
    """Zero-variance representations report NaN; row mismatch raises."""

    streaming = tl.stats.CKA()
    streaming.update(torch.ones(3, 2), torch.ones(3, 4))
    streaming.update(torch.ones(3, 2), torch.ones(3, 4))
    assert math.isnan(streaming.result())

    with pytest.raises(ValueError, match="matched row counts"):
        tl.stats.cka(torch.ones(3, 2), torch.ones(4, 2))


def test_pca_without_updates_returns_empty_shapes() -> None:
    """An unfed PCA reports empty components and variances."""

    pca = tl.stats.PCA(n_components=2)
    result = pca.result()
    assert result["components"].shape == (0, 0)
    assert result["explained_variance"].shape == (0,)


@pytest.mark.smoke
def test_aggregator_disambiguates_duplicate_stat_keys() -> None:
    """Two unnamed stats of one class get index-suffixed result keys."""

    aggregator = tl.stats.Aggregator(tl.stats.Mean(), tl.stats.Mean())
    aggregator.update(torch.tensor([2.0]))
    result = aggregator.result()
    assert set(result) == {"Mean", "Mean_1"}
    assert result["Mean"] == pytest.approx(2.0)
    assert result["Mean_1"] == pytest.approx(2.0)


def test_aggregate_rejects_unknown_target() -> None:
    """aggregate() refuses targets other than 'out' and 'grad'."""

    with pytest.raises(ValueError, match="'out' or 'grad'"):
        tl.aggregate(nn.Linear(2, 1), [torch.ones(1, 2)], {"output": tl.stats.Mean()}, target="w")


@pytest.mark.smoke
def test_aggregate_unmatched_selector_raises_key_error() -> None:
    """A selector that matches no saved out fails with a clear KeyError."""

    model = nn.Linear(2, 1)
    # The capture layer discloses the zero-match layers_to_save request with a
    # UserWarning (cdcf7d89) before aggregate's lookup raises the KeyError.
    with (
        pytest.raises(KeyError, match="no_such_layer"),
        pytest.warns(UserWarning, match="matched zero layers"),
    ):
        tl.aggregate(model, [torch.ones(1, 2)], {"no_such_layer": tl.stats.Mean()})


def test_aggregate_eval_fast_path_matches_exact_full_trace_path() -> None:
    """The compiled sparse plan on eval models reproduces the exact path results.

    The same model and batches run once in train mode (exact full-trace every
    batch) and once in eval mode (plan compiled on batch one, sparse record
    after); a parameter-free model makes the two modes semantically identical,
    so any fast-path divergence is a real bug.
    """

    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    batches = [torch.randn(2, 3) for _ in range(4)]

    def _run() -> dict[str, float]:
        metrics = {"relu": tl.stats.Mean(), "output": tl.stats.Norm()}
        return tl.aggregate(model, list(batches), metrics)

    model.train()
    exact = _run()
    model.eval()
    fast = _run()

    assert fast["relu"] == pytest.approx(exact["relu"], rel=1e-9)
    assert fast["output"] == pytest.approx(exact["output"], rel=1e-9)

    with torch.no_grad():
        relu_values = torch.cat([torch.relu(model[0](batch)).reshape(-1) for batch in batches]).to(
            dtype=torch.float64
        )
    assert fast["relu"] == pytest.approx(float(relu_values.mean().item()))


class _BranchingModel(nn.Module):
    """Parameter-free model whose branches emit equal-length, reordered op streams.

    Both branches run one relu and one sigmoid, but in swapped order, so the
    compiled plan's stream positions still exist on the other branch while
    holding semantically different ops. Only the fingerprint comparison can
    detect the drift; positional bookkeeping alone cannot.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            y = torch.relu(x)
            z = torch.sigmoid(x)
        else:
            z = torch.sigmoid(x)
            y = torch.relu(x)
        return y + z


@pytest.mark.smoke
def test_aggregate_fast_path_falls_back_exactly_on_fingerprint_drift() -> None:
    """A branch change after plan compilation re-traces the batch exactly."""

    model = _BranchingModel().eval()
    positive = torch.ones(1, 3)
    negative = -torch.ones(1, 3)
    batches = [positive, positive, negative, positive]

    result = tl.aggregate(model, batches, {"relu": tl.stats.Mean(), "output": tl.stats.Mean()})

    expected_relu = torch.cat([torch.relu(batch).reshape(-1) for batch in batches]).to(
        dtype=torch.float64
    )
    expected_output = torch.cat([model(batch).reshape(-1) for batch in batches]).to(
        dtype=torch.float64
    )
    assert result["relu"] == pytest.approx(float(expected_relu.mean().item()))
    assert result["output"] == pytest.approx(float(expected_output.mean().item()))


@pytest.mark.smoke
def test_aggregate_grad_accepts_list_batches_and_unmatched_grad_selector_raises() -> None:
    """List-shaped (input, target) batches split for the loss; bad grad selectors refuse."""

    torch.manual_seed(7)
    model = nn.Linear(3, 1)
    batch = [torch.randn(2, 3), torch.zeros(2, 1)]

    result = tl.aggregate(
        model,
        [batch],
        {"linear": tl.stats.Norm()},
        target="grad",
        loss_fn=lambda output, target: torch.nn.functional.mse_loss(output, target),
    )
    assert math.isfinite(result["linear"])

    # save_grads= validates selectors at capture entry, so an unmatched grad
    # metric fails there with the typed lookup error naming the selector.
    with pytest.raises((KeyError, ValueError), match="no_such_layer"):
        tl.aggregate(
            model,
            [batch],
            {"no_such_layer": tl.stats.Mean()},
            target="grad",
            loss_fn=lambda output, target: torch.nn.functional.mse_loss(output, target),
        )

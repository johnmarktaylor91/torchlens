"""Tests for streaming cross-covariance and linear CKA."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.stats import CKA, CrossCovariance, cka


def _gram_cka(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return a direct centered-Gram linear CKA reference value.

    Parameters
    ----------
    a:
        First representation matrix.
    b:
        Second representation matrix.

    Returns
    -------
    torch.Tensor
        Scalar linear CKA value.
    """

    centered_a = a - a.mean(dim=0)
    centered_b = b - b.mean(dim=0)
    gram_a = centered_a @ centered_a.T
    gram_b = centered_b @ centered_b.T
    return (gram_a * gram_b).sum() / torch.sqrt(gram_a.square().sum() * gram_b.square().sum())


def test_streaming_cka_matches_one_shot() -> None:
    """Chunked updates produce the same result as a one-shot update."""

    generator = torch.Generator().manual_seed(789)
    a = torch.randn(29, 5, generator=generator, dtype=torch.float64)
    b = torch.randn(29, 9, generator=generator, dtype=torch.float64)
    accumulator = CKA()
    boundaries = (0, 3, 14, 20, 29)
    for start, stop in zip(boundaries, boundaries[1:]):
        accumulator.update(a[start:stop], b[start:stop])

    assert accumulator.compute() == pytest.approx(cka(a, b), abs=1e-12)


def test_cross_covariance_refuses_mismatched_rows() -> None:
    """Paired batches must contain the same number of observations."""

    accumulator = CrossCovariance()
    with pytest.raises(ValueError, match="matched row counts.*3 and 4"):
        accumulator.update(torch.randn(3, 2), torch.randn(4, 5))


class _AggregateModel(nn.Module):
    """Minimal model exposing one hidden activation for aggregate tests."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.linear(x).relu()


def _single_batch_loader() -> Iterable[torch.Tensor]:
    """Yield one deterministic batch for aggregate tests."""

    yield torch.ones(2, 3)


def test_aggregate_grad_output_metric_raises_clear_key_error() -> None:
    """A `tl.grad("output")` metric key fails clearly: the model output has no saved grad."""

    with pytest.raises(KeyError, match="No saved grad matched metric 'output'"):
        tl.stats.aggregate(
            _AggregateModel(),
            _single_batch_loader(),
            {tl.grad("output"): tl.stats.Mean()},
            loss_fn=lambda output: output.sum(),
        )


def test_cka_results_name_the_measure_variant_they_compute() -> None:
    """One-shot cka and CKA.compute carry the similarity-repository id (U56.3)."""

    import json
    import pickle

    from torchlens.repgeom._similarity_ids import LINEAR_CKA_ID
    from torchlens.stats import SimilarityScore

    torch.manual_seed(0)
    a, b = torch.randn(12, 5), torch.randn(12, 4)
    one_shot = cka(a, b)
    streaming = CKA()
    streaming.update(a, b)
    streamed = streaming.compute()
    for score in (one_shot, streamed):
        assert isinstance(score, SimilarityScore) and isinstance(score, float)
        assert score.similarity_repository_id == LINEAR_CKA_ID == CKA.similarity_repository_id
        assert score.to_dict() == {"score": float(score), "similarity_repository_id": LINEAR_CKA_ID}
        assert json.loads(json.dumps(score)) == float(score)
        restored = pickle.loads(pickle.dumps(score))
        assert restored == score and restored.similarity_repository_id == LINEAR_CKA_ID
    assert math.isclose(one_shot, streamed, rel_tol=1e-9)
    with pytest.warns(Warning, match="zero variance"):
        degenerate = cka(torch.ones(4, 2), torch.randn(4, 2))
    assert math.isnan(degenerate) and degenerate.similarity_repository_id == LINEAR_CKA_ID

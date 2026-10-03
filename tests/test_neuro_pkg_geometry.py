"""Lane F22 canonical-home riders (neuro MEMO D18, items 12-14).

Pins:

- The Gaussian/RBF RDM metric matches thingsvision's published convention
  EXACTLY (reference implementation transcribed below, credited): squared
  euclidean distances, bandwidth = mean(D) over the full matrix, kernel
  ``exp(-D / (2 * mean(D)))``, dissimilarity ``1 - kernel``. All-identical
  stimuli refuse typed where the reference would silently NaN.
- ``rank_transform_rdm``: the Nili et al. 2014 display convention --
  average ranks for ties, percentile scaling to exactly 100, symmetric
  zero-diagonal output, display-only validation refusals.
- The ``rdm_node_spec(display=)`` render kwarg validates its vocabulary.
- The tl.stats.cka method contract (item 14): linear, BIASED estimator,
  CPU float64, matched observations, NaN on zero variance -- pinned with
  literal reference values so any estimator change fails loudly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from torchlens.repgeom import rank_transform_rdm, rdm
from torchlens.stats import cka


def _thingsvision_gaussian_rdm(features: np.ndarray) -> np.ndarray:
    """Reference implementation transcribed from thingsvision (credited).

    ``thingsvision.core.rsa.helpers`` computes the RSM as
    ``exp(-squared_dists(F) / (2 * mean(squared_dists(F))))`` and the RDM
    as ``1 - rsm``; the bandwidth denominator is the mean over the FULL
    matrix including the zero diagonal.
    """

    squared = ((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=-1)
    return 1.0 - np.exp(-squared / (2.0 * squared.mean()))


def test_gaussian_metric_matches_thingsvision_reference() -> None:
    """Item 12: reference equivalence against the credited convention."""

    rng = np.random.default_rng(7)
    features = rng.normal(size=(9, 13))
    ours = rdm(features, metric="gaussian")
    reference = _thingsvision_gaussian_rdm(features)
    assert np.allclose(ours, reference, atol=1e-12)
    assert np.allclose(ours, ours.T)
    assert np.allclose(np.diagonal(ours), 0.0)
    assert float(ours.max()) < 1.0


@pytest.mark.smoke
def test_gaussian_metric_refuses_identical_stimuli() -> None:
    """A zero data-derived bandwidth refuses typed, never NaN."""

    features = np.ones((4, 5))
    with pytest.raises(ValueError, match="bandwidth"):
        rdm(features, metric="gaussian")


@pytest.mark.smoke
def test_rank_transform_percentile_and_ties() -> None:
    """Item 13: average ranks for ties; percentile max is exactly 100."""

    square = np.array(
        [
            [0.0, 1.0, 2.0, 2.0],
            [1.0, 0.0, 3.0, 4.0],
            [2.0, 3.0, 0.0, 5.0],
            [2.0, 4.0, 5.0, 0.0],
        ]
    )
    ranks = rank_transform_rdm(square, output="rank")
    # Pairs (0,2) and (0,3) tie at value 2.0 -> average rank (2+3)/2 = 2.5.
    assert ranks[0, 2] == 2.5 and ranks[0, 3] == 2.5
    assert ranks[0, 1] == 1.0
    assert ranks[2, 3] == 6.0
    percentiles = rank_transform_rdm(square, output="percentile")
    assert percentiles.max() == 100.0
    assert np.allclose(percentiles, ranks / 6.0 * 100.0)
    assert np.allclose(percentiles, percentiles.T)
    assert np.allclose(np.diagonal(percentiles), 0.0)


def test_rank_transform_refusals() -> None:
    """Display-only validation: bad tokens and invalid matrices refuse."""

    square = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="output"):
        rank_transform_rdm(square, output="zscore")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        rank_transform_rdm(np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError):
        rank_transform_rdm(np.array([[0.0, np.nan], [np.nan, 0.0]]))
    with pytest.raises(ValueError):
        rank_transform_rdm(np.zeros((1, 1)))


@pytest.mark.smoke
def test_rdm_node_spec_display_vocabulary() -> None:
    """The render kwarg accepts raw/rank/percentile and refuses others."""

    from torchlens.repgeom import rdm_node_spec

    for display in ("raw", "rank", "percentile"):
        assert callable(rdm_node_spec(display=display))
    with pytest.raises(ValueError, match="display"):
        rdm_node_spec(display="log")


@pytest.mark.smoke
def test_cka_method_contract_pins() -> None:
    """Item 14: linear/biased/CPU-f64 contract with literal reference values.

    The pinned values were computed with the shipped estimator on fixed
    integer matrices; they hold for Kornblith et al. (2019)'s BIASED linear
    CKA (gram-matrix form, no unbiased-HSIC correction) in float64. An
    estimator change (e.g. silently going unbiased) fails these literals.
    """

    a = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 1.0], [0.0, 2.0]])
    b = np.array([[2.0, 0.0, 1.0], [1.0, 1.0, 4.0], [5.0, 2.0, 2.0], [2.0, 3.0, 0.0]])

    # Self-alignment is exactly 1; orthogonal transforms leave it there.
    assert cka(a, a) == pytest.approx(1.0, abs=1e-12)
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert cka(a, a @ rotation) == pytest.approx(1.0, abs=1e-12)
    assert cka(a, 3.5 * a) == pytest.approx(1.0, abs=1e-12)

    # Literal cross-representation reference value (biased linear CKA).
    reference = _reference_biased_linear_cka(a, b)
    value = cka(a, b)
    assert value == pytest.approx(reference, abs=1e-12)
    assert value == pytest.approx(0.8735487150980951, abs=1e-9)

    # Degenerate zero-variance input -> NaN, matched rows required.
    assert math.isnan(cka(a, np.ones((4, 2))))
    with pytest.raises(ValueError, match="matched row counts"):
        cka(a, b[:3])


def _reference_biased_linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """Independent float64 reimplementation of biased linear CKA."""

    centered_a = a - a.mean(axis=0, keepdims=True)
    centered_b = b - b.mean(axis=0, keepdims=True)
    gram_a = centered_a @ centered_a.T
    gram_b = centered_b @ centered_b.T
    numerator = float((gram_a * gram_b).sum())
    denominator = float(np.linalg.norm(gram_a) * np.linalg.norm(gram_b))
    return numerator / denominator

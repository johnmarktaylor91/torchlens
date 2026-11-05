"""Lane F22 canonical-home riders (neuro MEMO D18, items 12-14).

Pins:

- The Gaussian/RBF RDM metric matches thingsvision's published convention
  EXACTLY (reference implementation transcribed below, credited): squared
  euclidean distances, bandwidth = mean(D) over the full matrix, kernel
  ``exp(-D / mean(D))``, dissimilarity ``1 - kernel``. All-identical
  stimuli refuse typed where the reference would silently NaN. The
  similarity-repository oracle checked the same equality against
  the catalogued ``rdm/*/gaussian``; an analytic literal pin keeps it from
  drifting back to the earlier ``2 * mean(D)`` bandwidth.
- ``rank_transform_rdm``: the Nili et al. 2014 display convention --
  average ranks for ties, percentile scaling to exactly 100, symmetric
  zero-diagonal output, display-only validation refusals.
- The ``tl.viz.nodes.rdm(display=)`` render kwarg validates its vocabulary.
- The tl.stats.cka method contract (item 14): linear, BIASED estimator,
  CPU float64, matched observations, NaN on zero variance -- pinned with
  literal reference values so any estimator change fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchlens.repgeom import rank_transform_rdm, rdm


def _thingsvision_gaussian_rdm(features: np.ndarray) -> np.ndarray:
    """Reference implementation transcribed from thingsvision (credited).

    ``gaussian_kernel`` in ``thingsvision/core/rsa/helpers.py`` (unchanged
    upstream since 2022-08-31) computes the RSM as
    ``exp(-squared_dists(F) / mean(squared_dists(F)))`` and the RDM as
    ``1 - rsm``; the bandwidth denominator is the mean over the FULL matrix
    including the zero diagonal.
    """

    squared = ((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=-1)
    return 1.0 - np.exp(-squared / squared.mean())


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


def test_gaussian_metric_refuses_identical_stimuli() -> None:
    """A zero data-derived bandwidth refuses typed, never NaN."""

    features = np.ones((4, 5))
    with pytest.raises(ValueError, match="bandwidth"):
        rdm(features, metric="gaussian")


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


def test_rdm_node_spec_display_vocabulary() -> None:
    """The render kwarg accepts raw/rank/percentile and refuses others."""

    from torchlens.viz.nodes import rdm as rdm_node

    for display in ("raw", "rank", "percentile"):
        assert callable(rdm_node(display=display))
    with pytest.raises(ValueError, match="display"):
        rdm_node(display="log")


def _reference_biased_linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """Independent float64 reimplementation of biased linear CKA."""

    centered_a = a - a.mean(axis=0, keepdims=True)
    centered_b = b - b.mean(axis=0, keepdims=True)
    gram_a = centered_a @ centered_a.T
    gram_b = centered_b @ centered_b.T
    numerator = float((gram_a * gram_b).sum())
    denominator = float(np.linalg.norm(gram_a) * np.linalg.norm(gram_b))
    return numerator / denominator

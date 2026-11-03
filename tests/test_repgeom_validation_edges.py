"""Behavioral tests for repgeom input-validation and refusal semantics."""

from __future__ import annotations

import numpy as np
import pytest

import torchlens as tl


def _distances(n: int = 8) -> np.ndarray:
    """Return a valid Euclidean distance matrix over ``n`` distinct points."""

    rng = np.random.default_rng(9)
    points = rng.normal(size=(n, 3))
    deltas = points[:, None, :] - points[None, :, :]
    return np.sqrt((deltas**2).sum(axis=-1))


def test_rdm_refuses_nonfinite_empty_and_unknown_metric_inputs() -> None:
    """rdm() rejects NaN activations, empty stimulus axes, and bad metrics."""

    with pytest.raises(ValueError, match="finite"):
        tl.repgeom.rdm(np.array([[1.0, np.nan], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="non-empty leading stimulus"):
        tl.repgeom.rdm(np.empty((0, 4)))
    # F20 D-5 made "manhattan" a real metric; the unknown-metric refusal is
    # pinned with a genuinely unknown token.
    with pytest.raises(ValueError, match="Unsupported activation distance metric"):
        tl.repgeom.rdm(np.ones((4, 2)), metric="mahalanobis")


def test_classical_mds_refuses_malformed_distance_matrices() -> None:
    """Explicit distances must be square, symmetric, zero-diagonal, non-negative."""

    good = _distances(8)

    with pytest.raises(ValueError, match="square"):
        tl.repgeom.classical_mds(np.ones((3, 4)), input_kind="distances")

    asymmetric = good.copy()
    asymmetric[0, 1] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        tl.repgeom.classical_mds(asymmetric, input_kind="distances")

    dirty_diagonal = good.copy()
    np.fill_diagonal(dirty_diagonal, 0.5)
    with pytest.raises(ValueError, match="zero diagonal"):
        tl.repgeom.classical_mds(dirty_diagonal, input_kind="distances")

    negative = good.copy()
    negative[0, 1] = negative[1, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        tl.repgeom.classical_mds(negative, input_kind="distances")


def test_classical_mds_validates_option_values_and_stimulus_counts() -> None:
    """n_components, min_stimuli, input_kind, and stimulus floors are all enforced."""

    good = _distances(8)

    with pytest.raises(ValueError, match="num_components"):
        tl.repgeom.classical_mds(good, 0, input_kind="distances")
    with pytest.raises(ValueError, match="min_stimuli"):
        tl.repgeom.classical_mds(good, min_stimuli=2, input_kind="distances")
    with pytest.raises(ValueError, match="input_kind"):
        tl.repgeom.classical_mds(good, input_kind="magic")
    with pytest.raises(ValueError, match="at least 3 stimuli"):
        tl.repgeom.classical_mds(_distances(2), input_kind="distances", min_stimuli=3)
    with pytest.raises(ValueError, match="too few stimuli"):
        tl.repgeom.classical_mds(_distances(5), input_kind="distances", min_stimuli=6)


def test_classical_mds_refuses_rank_deficient_input() -> None:
    """Collinear points cannot support a 2-d embedding."""

    line = np.arange(8, dtype=np.float64).reshape(8, 1) * np.ones((1, 3))
    with pytest.raises(ValueError, match="rank-deficient"):
        tl.repgeom.classical_mds(line, 2, input_kind="features")


def test_scree_and_effective_dimensionality_validate_options() -> None:
    """scree min_stimuli floor and variance_threshold bounds are enforced."""

    with pytest.raises(ValueError, match="min_stimuli"):
        tl.repgeom.scree(np.ones((4, 2)), min_stimuli=2)
    with pytest.raises(ValueError, match="variance_threshold"):
        tl.repgeom.effective_dimensionality(_distances(6), variance_threshold=1.5)
    with pytest.raises(ValueError, match="variance_threshold"):
        tl.repgeom.effective_dimensionality(_distances(6), variance_threshold=float("nan"))

    info = tl.repgeom.effective_dimensionality(_distances(6), variance_threshold=0.9)
    assert info["n_components_for_threshold"] >= 1
    assert np.isfinite(info["participation_ratio"])


def test_procrustes_align_validates_point_clouds() -> None:
    """Procrustes inputs must be finite [N>=3, 2] clouds of matching shape and rank."""

    rng = np.random.default_rng(3)
    cloud = rng.normal(size=(5, 2))

    with pytest.raises(ValueError, match="shape \\[N, 2\\]"):
        tl.repgeom.procrustes_align(rng.normal(size=(5, 3)), cloud)
    with pytest.raises(ValueError, match="at least 3 points"):
        tl.repgeom.procrustes_align(cloud[:2], cloud[:2])
    with pytest.raises(ValueError, match="finite"):
        bad = cloud.copy()
        bad[0, 0] = np.inf
        tl.repgeom.procrustes_align(bad, cloud)
    with pytest.raises(ValueError, match="same shape"):
        tl.repgeom.procrustes_align(cloud, cloud[:4])

    collinear = np.stack([np.arange(5.0), np.arange(5.0)], axis=1)
    with pytest.raises(ValueError, match="rank-deficient"):
        tl.repgeom.procrustes_align(collinear, cloud)

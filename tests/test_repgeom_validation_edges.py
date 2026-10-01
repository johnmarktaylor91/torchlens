"""Behavioral tests for repgeom input-validation and refusal semantics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchlens as tl

pytestmark = pytest.mark.smoke


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


def test_rdm_metrics_produce_valid_distance_matrices() -> None:
    """cosine/correlation metrics return symmetric zero-diagonal matrices."""

    torch.manual_seed(1)
    activations = torch.randn(6, 5)
    for metric in ("euclidean", "cosine", "correlation"):
        matrix = tl.repgeom.rdm(activations, metric=metric)
        assert matrix.shape == (6, 6)
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0, atol=1e-8)
        assert np.all(matrix >= -1e-8)


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
    """n_components, min_n, input_kind, and stimulus floors are all enforced."""

    good = _distances(8)

    with pytest.raises(ValueError, match="n_components"):
        tl.repgeom.classical_mds(good, 0, input_kind="distances")
    with pytest.raises(ValueError, match="min_n"):
        tl.repgeom.classical_mds(good, min_n=2, input_kind="distances")
    with pytest.raises(ValueError, match="input_kind"):
        tl.repgeom.classical_mds(good, input_kind="magic")
    with pytest.raises(ValueError, match="at least 3 stimuli"):
        tl.repgeom.classical_mds(_distances(2), input_kind="distances", min_n=3)
    with pytest.raises(ValueError, match="too few stimuli"):
        tl.repgeom.classical_mds(_distances(5), input_kind="distances", min_n=6)


@pytest.mark.parametrize("scale", [1e-12, 1e-6, 1.0, 1e6, 1e12])
def test_symmetry_gate_is_relative_to_matrix_scale(scale: float) -> None:
    """The symmetry gate judges RELATIVE asymmetry at every matrix scale.

    Two-sided pin for the scale-aware `_symmetry_tolerance` fix. The former
    fixed absolute atol=1e-10 was broken in both directions: 66% relative
    asymmetry at tiny scales read as symmetric (total structural corruption
    blessed), while ~1e-15-relative float64 round-off at scale 1e7 was
    rejected (honest matrices refused). A fixed 66% relative asymmetry must
    FAIL at every scale and a fixed ~1e-14 relative asymmetry (float64
    round-off class) must PASS at every scale. Genuine Euclidean distance
    matrices keep the zero-diagonal/non-negativity gates out of the way.
    """

    from torchlens.repgeom import _check_square_distances

    base = _distances(8) * scale
    upper = np.triu_indices(8, k=1)

    corrupt = base.copy()
    corrupt[upper] *= 1.66
    with pytest.raises(ValueError, match="symmetric"):
        _check_square_distances(corrupt)

    noisy = base.copy()
    noisy[upper] *= 1.0 + 1e-14
    _check_square_distances(noisy)  # float64 round-off passes at every scale.


def test_classical_mds_accepts_float64_noise_at_large_scale_end_to_end() -> None:
    """Round-off-level asymmetry on a large-scale matrix embeds successfully.

    The measured false-fail regression: 1e-15-relative float64 round-off at
    scale ~1e7 was rejected by the fixed absolute symmetry atol. The full
    classical_mds pipeline (symmetry, diagonal, non-negativity, duplicate,
    and rank gates) must accept it now.
    """

    base = _distances(8) * 1e7
    upper = np.triu_indices(8, k=1)
    noisy = base.copy()
    noisy[upper] *= 1.0 + 1e-15

    embedding, info = tl.repgeom.classical_mds(noisy, input_kind="distances")
    assert embedding.shape == (8, 2)
    assert info["input_kind"] == "distances"


def test_classical_mds_refuses_rank_deficient_input() -> None:
    """Collinear points cannot support a 2-d embedding."""

    line = np.arange(8, dtype=np.float64).reshape(8, 1) * np.ones((1, 3))
    with pytest.raises(ValueError, match="rank-deficient"):
        tl.repgeom.classical_mds(line, 2, input_kind="features")


def test_scree_and_effective_dimensionality_validate_options() -> None:
    """scree min_n floor and variance_threshold bounds are enforced."""

    with pytest.raises(ValueError, match="min_n"):
        tl.repgeom.scree(np.ones((4, 2)), min_n=2)
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


def test_procrustes_align_recovers_rotation_without_reflection() -> None:
    """A rotated cloud aligns back onto the target exactly; no reflections."""

    rng = np.random.default_rng(4)
    target = rng.normal(size=(6, 2))
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    source = target @ rotation.T + 3.0

    aligned = tl.repgeom.procrustes_align(source, target)
    assert np.allclose(aligned, target, atol=1e-8)

    reflected = target.copy()
    reflected[:, 0] *= -1.0
    aligned_reflection = tl.repgeom.procrustes_align(reflected, target)
    # Rotation-only alignment cannot undo a reflection, so the result must
    # differ from the target (a full Procrustes with reflections would match).
    assert not np.allclose(aligned_reflection, target, atol=1e-6)

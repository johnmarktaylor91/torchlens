"""Tests for representation-geometry numeric helpers."""

from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import pytest

import torchlens as tl
from torchlens import repgeom


ANALYTIC_POINTS = np.array(
    [
        [-3.0, 0.5],
        [-1.7, -2.2],
        [-0.4, 1.8],
        [0.9, -0.8],
        [2.6, 2.4],
        [3.8, -1.5],
        [5.1, 0.7],
        [6.4, 3.1],
    ],
    dtype=np.float64,
)


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Return Euclidean pairwise distances for rows of ``points``.

    Parameters
    ----------
    points:
        Row-wise point coordinates.

    Returns
    -------
    np.ndarray
        Square pairwise distance matrix.
    """

    differences = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=-1))


def _reflection_allowed_procrustes_residual(source: np.ndarray, target: np.ndarray) -> float:
    """Return the normalized residual after unrestricted orthogonal alignment.

    Parameters
    ----------
    source:
        Source row-wise coordinates.
    target:
        Target row-wise coordinates.

    Returns
    -------
    float
        Frobenius residual after centering and reflection-allowed rotation.
    """

    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered, full_matrices=False)
    aligned = source_centered @ (u @ vt)
    denominator = np.linalg.norm(target_centered)
    return float(np.linalg.norm(aligned - target_centered) / denominator)


def test_classical_mds_recovers_closed_form_asymmetric_fixture() -> None:
    """MDS should recover a nondegenerate asymmetric 2D fixture analytically."""

    centered = ANALYTIC_POINTS - ANALYTIC_POINTS.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values * singular_values
    assert eigenvalues[0] > eigenvalues[1] > 0.0

    distances = _pairwise_distances(ANALYTIC_POINTS)
    embedding, info = repgeom.classical_mds(distances, n_components=2)

    assert info["effective_rank"] == 2
    assert np.allclose(_pairwise_distances(embedding), distances, atol=1e-9)
    assert _reflection_allowed_procrustes_residual(embedding, ANALYTIC_POINTS) < 1e-6


def test_procrustes_align_recovers_known_rotation_without_scaling() -> None:
    """No-reflection Procrustes should recover a known proper rotation."""

    theta = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    target = ANALYTIC_POINTS @ rotation + np.array([3.0, -4.0])

    aligned = repgeom.procrustes_align(ANALYTIC_POINTS, target)
    source_centered = ANALYTIC_POINTS - ANALYTIC_POINTS.mean(0)
    aligned_centered = aligned - aligned.mean(0)
    least_squares_rotation = np.linalg.lstsq(
        source_centered,
        aligned_centered,
        rcond=None,
    )[0]

    assert np.linalg.det(least_squares_rotation) > 0
    assert np.linalg.norm(aligned - target) < 1e-9


def test_classical_mds_min_n_gate_raises_clear_error() -> None:
    """MDS should reject too-small visualization batches."""

    with pytest.raises(ValueError, match="too few stimuli"):
        repgeom.classical_mds(ANALYTIC_POINTS[:7], min_n=8)


def test_classical_mds_rank_deficient_duplicate_gate_raises_clear_error() -> None:
    """MDS should reject duplicate stimuli as rank-deficient for display."""

    points = ANALYTIC_POINTS.copy()
    points[3] = points[0]

    with pytest.raises(ValueError, match="rank-deficient|duplicate"):
        repgeom.classical_mds(points, min_n=8)


def test_classical_mds_negative_eigenvalue_clip_reports_discarded_fraction() -> None:
    """Non-PSD dissimilarities should report clipped negative eigenvalue mass."""

    distances = _pairwise_distances(ANALYTIC_POINTS)
    distances[0, 1] = 25.0
    distances[1, 0] = 25.0

    embedding, info = repgeom.classical_mds(distances, min_n=8)

    assert embedding.shape == (8, 2)
    assert info["negative_eigenvalue_count"] > 0
    assert info["discarded_variance_fraction"] > 0.0


def test_classical_mds_sign_convention_is_deterministic() -> None:
    """Repeated MDS calls should use identical coordinate signs."""

    distances = _pairwise_distances(ANALYTIC_POINTS)

    embedding_a, _ = repgeom.classical_mds(distances)
    embedding_b, _ = repgeom.classical_mds(distances)

    assert np.array_equal(embedding_a, embedding_b)


def test_activation_distance_matrix_metrics() -> None:
    """Activation distance helper should flatten rows and support core metrics."""

    activations = np.array([[[1.0, 0.0, 2.0]], [[0.0, 1.0, 3.0]], [[1.0, 1.0, 4.0]]])

    euclidean = repgeom.activation_distance_matrix(activations)
    cosine = repgeom.activation_distance_matrix(activations, metric="cosine")
    correlation = repgeom.activation_distance_matrix(activations, metric="correlation")

    assert euclidean.shape == (3, 3)
    assert np.allclose(np.diag(cosine), 0.0)
    assert np.allclose(np.diag(correlation), 0.0)
    assert np.allclose(euclidean, euclidean.T)


def test_repgeom_reachable_as_top_level_submodule() -> None:
    """``tl.repgeom`` should be exposed without expanding ``tl.__all__``."""

    assert tl.repgeom is repgeom
    assert "repgeom" not in tl.__all__


def test_repgeom_import_clean_without_optional_neuro_or_plotting_deps() -> None:
    """Importing repgeom should not import optional plotting, sklearn, or neuro deps."""

    script = """
import sys
import torchlens.repgeom
blocked = {'scipy', 'sklearn', 'matplotlib', 'rsatoolbox', 'brainscore', 'brainscore_core'}
loaded = sorted(name for name in blocked if name in sys.modules)
if loaded:
    raise SystemExit('unexpected imports: ' + ', '.join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_repgeom_direct_import_clean() -> None:
    """Direct import of the submodule should succeed in the current interpreter."""

    assert importlib.import_module("torchlens.repgeom") is repgeom

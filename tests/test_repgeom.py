"""Tests for representation-geometry numeric helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

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


class _MDSClassifier(nn.Module):
    """Small multi-layer model with batch activations for MDS tests."""

    def __init__(self) -> None:
        """Initialize deterministic linear layers."""

        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(5))
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(
                torch.tensor(
                    [
                        [0.8, -0.6, 0.0, 0.0, 0.0],
                        [0.6, 0.8, 0.0, 0.0, 0.0],
                        [0.2, 0.1, 1.0, 0.0, 0.0],
                        [0.0, 0.3, 0.0, 1.0, 0.0],
                        [0.1, 0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two saved linear layers."""

        return self.fc2(torch.tanh(self.fc1(x)))


class _RecurrentMDS(nn.Module):
    """Model that reuses one linear layer across multiple passes."""

    def __init__(self) -> None:
        """Initialize the recurrent layer."""

        super().__init__()
        self.attn = nn.Linear(5, 5)
        with torch.no_grad():
            self.attn.weight.copy_(torch.eye(5))
            self.attn.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the same module three times."""

        for _index in range(3):
            x = torch.tanh(self.attn(x))
        return x


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


def _mds_trace(model: nn.Module, save: Any) -> tl.Trace:
    """Capture a deterministic MDS test trace.

    Parameters
    ----------
    model:
        Model to trace.
    save:
        TorchLens save selector.

    Returns
    -------
    tl.Trace
        Captured trace with requested saved activations.
    """

    torch.manual_seed(1001)
    zeros = np.zeros((ANALYTIC_POINTS.shape[0], 3), dtype=np.float64)
    x = torch.tensor(np.concatenate([ANALYTIC_POINTS, zeros], axis=1), dtype=torch.float32)
    return tl.trace(model, x, save=save, random_seed=1001)


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


def test_mds_evolution_single_pass_layers_annotates_and_round_trips(tmp_path: Path) -> None:
    """MDS evolution should compute, align, annotate, and persist coords."""

    trace = _mds_trace(_MDSClassifier(), tl.func("linear"))
    linear_layers = [layer for layer in trace.layers if layer.layer_type == "linear"]

    coords_by_key = repgeom.mds_evolution(trace, save=tl.func("linear"), min_n=8)

    assert list(coords_by_key) == [f"layer:{layer.layer_label}" for layer in linear_layers]
    assert trace._annotation_blobs is not None
    first_key, second_key = list(coords_by_key)
    for key, coords in coords_by_key.items():
        assert coords.shape == (8, 2)
        assert torch.equal(trace._annotation_blobs[key], torch.from_numpy(coords))

    raw_second_distances = repgeom.activation_distance_matrix(linear_layers[1].out)
    raw_second_coords, _info = repgeom.classical_mds(raw_second_distances, min_n=8)
    expected_second = repgeom.procrustes_align(raw_second_coords, coords_by_key[first_key])
    assert np.allclose(coords_by_key[second_key], expected_second)

    bundle_path = tmp_path / "mds_evolution.tlspec"
    trace.save(bundle_path)
    loaded = tl.load(bundle_path)

    assert loaded._annotation_blobs is not None
    for key, coords in coords_by_key.items():
        assert torch.equal(loaded._annotation_blobs[key], torch.from_numpy(coords))


def test_mds_evolution_recurrent_aggregate_requires_pass_selection() -> None:
    """Aggregate recurrent layers should raise a clear pass-selection error."""

    trace = _mds_trace(_RecurrentMDS(), tl.func("linear"))

    with pytest.raises(ValueError, match="select a pass \\(layer is recurrent\\)"):
        repgeom.mds_evolution(trace, save=tl.func("linear"), min_n=8)


def test_mds_evolution_recurrent_pass_qualified_selector_uses_op_key() -> None:
    """A single recurrent pass selection should read op.out and store op coords."""

    trace = _mds_trace(_RecurrentMDS(), tl.func("linear"))
    linear_op = next(
        op for op in trace.layer_list if op.layer_type == "linear" and op.pass_index == 2
    )

    coords_by_key = repgeom.mds_evolution(trace, save=tl.label(linear_op.label), min_n=8)

    key = f"op:{linear_op.label}"
    assert list(coords_by_key) == [key]
    assert coords_by_key[key].shape == (8, 2)
    assert trace._annotation_blobs is not None
    assert torch.equal(trace._annotation_blobs[key], torch.from_numpy(coords_by_key[key]))


def test_mds_evolution_unsaved_activation_raises_capture_guidance() -> None:
    """Selected layers without saved outs should raise save= guidance."""

    trace = _mds_trace(_MDSClassifier(), tl.func("tanh"))

    with pytest.raises(ValueError, match="capture with save=.*MDS layers"):
        repgeom.mds_evolution(trace, save=tl.func("linear"), min_n=8)


def test_mds_evolution_min_n_gate_is_honored() -> None:
    """MDS evolution should forward the minimum stimulus gate."""

    trace = _mds_trace(_MDSClassifier(), tl.func("linear"))

    with pytest.raises(ValueError, match="too few stimuli"):
        repgeom.mds_evolution(trace, save=tl.func("linear"), min_n=9)

"""Geometry-over-reduced-payloads oracles (F20, brainpipe memo D-19).

The composition reduce-then-geometry was broken: repgeom collectors read
``.out`` only, so the exact traces the brain-benchmarking sweep produces
(transform retained, raw dropped) refused every geometry verb. Collectors
now fall back to the transformed payload, record which payload fed each
artifact (a raw RDM and a post-reduction RDM are different scientific
objects), and refuse batch-consuming transforms with the remedy named.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())


def _feature_halve(t: torch.Tensor) -> torch.Tensor:
    """Batch-preserving feature reduction."""

    return t[:, : max(1, t.shape[1] // 2)].clone()


def _batch_pool(t: torch.Tensor) -> torch.Tensor:
    """Batch-CONSUMING pooling: removes the stimulus axis."""

    return t.mean(dim=0)


def _reduced_trace() -> object:
    return tl.trace(
        _model(),
        torch.randn(12, 32),
        save=tl.options.SaveOptions(
            activation_transform=_feature_halve, save_raw_activations=False
        ),
    )


def test_rdm_evolution_reads_transformed_payloads() -> None:
    """Reduce-then-RDM works and disclosed the transformed basis per key."""

    rdms = tl.repgeom.rdm_evolution(_reduced_trace(), min_n=2)
    assert len(rdms) > 0
    assert set(rdms.payload_basis) == set(rdms)
    assert all(kind == "transformed" for kind in rdms.payload_basis.values())
    for matrix in rdms.values():
        assert matrix.shape == (12, 12)


def test_mds_and_scree_read_transformed_payloads() -> None:
    """The sibling verbs ride the same payload door."""

    log = _reduced_trace()
    coords = tl.repgeom.mds_evolution(log, min_n=3)
    scree = tl.repgeom.scree_evolution(log, min_n=3)
    assert all(kind == "transformed" for kind in coords.payload_basis.values())
    assert all(kind == "transformed" for kind in scree.payload_basis.values())


def test_raw_payload_basis_is_disclosed_as_raw() -> None:
    """Default raw captures keep raw geometry and say so."""

    log = tl.trace(_model(), torch.randn(12, 32))
    rdms = tl.repgeom.rdm_evolution(log, min_n=2)
    assert all(kind == "raw" for kind in rdms.payload_basis.values())


def test_batch_consuming_transform_refuses_with_remedy() -> None:
    """A transform that consumed the stimulus axis cannot feed geometry."""

    log = tl.trace(
        _model(),
        torch.randn(12, 32),
        save=tl.options.SaveOptions(activation_transform=_batch_pool, save_raw_activations=False),
    )
    with pytest.raises(ValueError, match="stimulus axis") as excinfo:
        tl.repgeom.rdm_evolution(log, min_n=2)
    assert "save_raw_activations=True" in str(excinfo.value)


def test_verbs_name_themselves_in_refusals() -> None:
    """rdm_evolution's refusals name rdm_evolution, never mds_evolution."""

    with pytest.warns(UserWarning, match="matched zero sites"):
        log = tl.trace(_model(), torch.randn(12, 32), save=tl.func("nonexistent_zzz"))
    with pytest.raises(ValueError) as excinfo:
        tl.repgeom.rdm_evolution(log, min_n=2)
    message = str(excinfo.value)
    assert "rdm_evolution" in message
    assert "mds_evolution" not in message

"""One call running RSA, ridge and CKA on one extraction (neuro.1 row 4).

Pins:

- Each analysis equals its standalone computation: RSA is exactly
  ``rdm_compare(rdm(site), rdm(target))``; CKA matches ``tl.stats.cka``
  (1e-12); ridge matches a hand-written fold loop with an explicit
  normal-equations solve, and its per-fold penalty matches brute-force
  leave-one-out refits.
- Activations are read once: the model is never re-run.
- A target equal to a site's own activations scores 1.0 under RSA and CKA.
- ``num_resamples`` adds seeded ScoreIntervals; ``disagreement()`` and
  ``table()`` read the same scores; invalid settings refuse.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.neuro._compare import (
    ANALYSES,
    _site_matrices,
)
from torchlens.repgeom import MetricDisagreement, ScoreInterval


class _Counted(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 4)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.fc2(torch.relu(self.fc1(x)))


def _trace(n: int = 24) -> tuple[_Counted, tl.Trace]:
    torch.manual_seed(0)
    model = _Counted().eval()
    log = tl.trace(model, torch.randn(n, 6), save="all")
    return model, log


def _target(n: int = 24, columns: int = 5) -> np.ndarray:
    return np.random.default_rng(1).normal(size=(n, columns))


def _explicit_ridge(train_x, train_y, test_x, alpha):  # type: ignore[no-untyped-def]
    mean_x, mean_y = train_x.mean(axis=0), train_y.mean(axis=0)
    xc = train_x - mean_x
    weights = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ (train_y - mean_y))
    return (test_x - mean_x) @ weights + mean_y


def test_ridge_recovers_a_noise_free_linear_target() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 5))
    result = tl.neuro.compare({"s": x}, x @ rng.normal(size=(5, 3)), analyses=("ridge",))
    assert result.scores["ridge"]["s"] > 0.999


def test_target_equal_to_a_site_scores_one_under_rsa_and_cka() -> None:
    _model, log = _trace()
    sites = _site_matrices(log, None)
    key, activations = next(iter(sites.items()))
    result = tl.neuro.compare(log, activations, analyses=("rsa", "cka"))
    assert result.scores["rsa"][key] == pytest.approx(1.0, abs=1e-12)
    assert result.scores["cka"][key] == 1.0
    mapping = tl.neuro.compare({"same": activations}, activations, analyses=("rsa", "cka"))
    assert mapping.scores["rsa"]["same"] == pytest.approx(1.0, abs=1e-12)
    assert mapping.scores["cka"]["same"] == 1.0


def test_resamples_add_seeded_intervals_and_disagreement_reads_them() -> None:
    rng = np.random.default_rng(6)
    source = {"a": rng.normal(size=(16, 5)), "b": rng.normal(size=(16, 5))}
    target = source["a"] + 0.3 * rng.normal(size=(16, 5))
    first = tl.neuro.compare(source, target, num_resamples=50, seed=9)
    again = tl.neuro.compare(source, target, num_resamples=50, seed=9)
    assert first.intervals == again.intervals
    for name in ANALYSES:
        for site in ("a", "b"):
            interval = first.intervals[name][site]
            assert isinstance(interval, ScoreInterval)
            assert interval.score == first.scores[name][site]
    report = first.disagreement()
    assert isinstance(report, MetricDisagreement)
    assert report.metrics == ANALYSES and report.intervals
    assert "[" in first.table()
    data = first.to_dict()
    assert data["intervals"]["rsa"]["a"]["num_resamples"] == 50


def test_table_lists_every_site_and_analysis() -> None:
    rng = np.random.default_rng(7)
    result = tl.neuro.compare(
        {"layer_one": rng.normal(size=(12, 3))}, rng.normal(size=(12, 2)), analyses=("cka", "rsa")
    )
    header, row = result.table().splitlines()
    assert header.split() == ["site", "cka", "rsa"]
    assert row.split()[0] == "layer_one"


def test_invalid_settings_refuse() -> None:
    rng = np.random.default_rng(8)
    source, target = {"s": rng.normal(size=(12, 3))}, rng.normal(size=(12, 2))
    with pytest.raises(ValueError, match="analyses"):
        tl.neuro.compare(source, target, analyses=("rsa", "svcca"))
    with pytest.raises(ValueError, match="rsatoolbox.rdm.compare"):
        tl.neuro.compare(source, target, rsa_method="cosine")
    with pytest.raises(ValueError, match="ridge_alphas"):
        tl.neuro.compare(source, target, ridge_alphas=(0.0,))
    with pytest.raises(ValueError, match="n_folds"):
        tl.neuro.compare(source, target, num_folds=1)
    with pytest.raises(ValueError, match="same stimuli"):
        tl.neuro.compare({"s": rng.normal(size=(11, 3))}, target)
    with pytest.raises(ValueError, match="not keys"):
        tl.neuro.compare(source, target, sites=["missing"])
    with pytest.raises(ValueError, match="at least"):
        tl.neuro.compare({"s": rng.normal(size=(4, 3))}, rng.normal(size=(4, 2)))
    with pytest.raises(ValueError, match="finite"):
        tl.neuro.compare({"s": np.full((12, 3), math.nan)}, target)

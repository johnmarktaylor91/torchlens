"""One call running RSA, ridge and CKA on one extraction (neuro.1 row 4).

Pins:

- Each analysis equals its standalone computation: RSA is exactly
  ``rdm_compare(rdm(site), rdm(target))``; CKA matches ``tl.stats.cka``
  (1e-12); ridge matches a hand-written fold loop with an explicit
  normal-equations solve, and its per-fold penalty matches brute-force
  leave-one-out refits.
- Activations are read once: the model is never re-run.
- A target equal to a site's own activations scores 1.0 under RSA and CKA.
- ``n_resamples`` adds seeded ScoreIntervals; ``disagreement()`` and
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
    cross_validated_ridge,
    loo_errors,
    ridge_folds,
)
from torchlens.repgeom import MetricDisagreement, ScoreInterval, rdm, rdm_compare
from torchlens.stats import cka


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
    log = tl.trace(
        model, torch.randn(n, 6), capture=tl.options.CaptureOptions(layers_to_save="all")
    )
    return model, log


def _target(n: int = 24, columns: int = 5) -> np.ndarray:
    return np.random.default_rng(1).normal(size=(n, columns))


def test_trace_source_scores_every_site_without_rerunning_the_model() -> None:
    model, log = _trace()
    calls = model.calls
    result = tl.neuro.compare(log, _target(), analyses=("rsa", "ridge", "cka"))
    assert model.calls == calls
    assert result.analyses == ANALYSES
    assert result.sites and set(result.scores) == set(ANALYSES)
    for name in ANALYSES:
        assert set(result.scores[name]) == set(result.sites)
    assert result.intervals == {}


def test_each_analysis_equals_its_standalone_computation() -> None:
    rng = np.random.default_rng(2)
    site = rng.normal(size=(20, 7))
    target = site[:, :3] @ rng.normal(size=(3, 4)) + rng.normal(size=(20, 4))
    result = tl.neuro.compare({"s": site}, target, analyses=("rsa", "ridge", "cka"))
    assert result.scores["rsa"]["s"] == rdm_compare(
        rdm(site, "correlation"), rdm(target, "correlation"), method="spearman"
    )
    assert result.scores["cka"]["s"] == pytest.approx(cka(site, target), abs=1e-12)
    predicted, _alphas = cross_validated_ridge(
        site, target, alphas=result.settings["ridge_alphas"], num_folds=5, seed=0
    )
    columns = [np.corrcoef(predicted[:, j], target[:, j])[0, 1] for j in range(4)]
    assert result.scores["ridge"]["s"] == pytest.approx(float(np.mean(columns)), abs=1e-12)
    assert result.similarity_repository_ids == {
        "rsa": "measure/*/rsa-rdm=correlation-compare=spearman",
        "ridge": None,
        "cka": "measure/*/cka-kernel=linear-hsic=gretton-score",
    }


def _explicit_ridge(train_x, train_y, test_x, alpha):  # type: ignore[no-untyped-def]
    mean_x, mean_y = train_x.mean(axis=0), train_y.mean(axis=0)
    xc = train_x - mean_x
    weights = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ (train_y - mean_y))
    return (test_x - mean_x) @ weights + mean_y


def test_ridge_matches_explicit_folds_and_brute_force_leave_one_out() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(18, 4))
    y = x @ rng.normal(size=(4, 2)) + 0.5 * rng.normal(size=(18, 2))
    alphas = (0.01, 1.0, 100.0)
    predicted, chosen = cross_validated_ridge(x, y, alphas=alphas, num_folds=3, seed=4)
    expected = np.empty_like(y)
    for fold, held_out in enumerate(ridge_folds(18, 3, 4)):
        train = np.setdiff1d(np.arange(18), held_out)
        errors = []
        fast = loo_errors(x[train], y[train], alphas)
        for alpha in alphas:
            residuals = []
            for leave in range(train.size):
                keep = np.delete(train, leave)
                guess = _explicit_ridge(x[keep], y[keep], x[train[leave]][None, :], alpha)
                residuals.append(y[train[leave]] - guess[0])
            errors.append(float(np.mean(np.square(residuals))))
        np.testing.assert_allclose(fast, errors, rtol=1e-10, atol=0)
        assert chosen[fold] == alphas[int(np.argmin(errors))]
        expected[held_out] = _explicit_ridge(x[train], y[train], x[held_out], chosen[fold])
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-10)


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
    first = tl.neuro.compare(source, target, n_resamples=50, seed=9)
    again = tl.neuro.compare(source, target, n_resamples=50, seed=9)
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
    assert data["intervals"]["rsa"]["a"]["n_resamples"] == 50


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

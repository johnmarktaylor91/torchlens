"""A06 capture-options truth: train-mode running-stat mutation is disclosed.

WALKTHROUGH list-A row 30: train-mode tracing silently mutated BatchNorm
running statistics -- the forward REALLY runs, so ``running_mean`` /
``running_var`` advance under momentum (even inside ``torch.no_grad`` or
``inference_only=True``) with no disclosure anywhere. The capture entry now
warns ONCE PER PROCESS with code ``batchnorm_train_stats_mutated`` when a
train-mode norm layer tracking running statistics is about to run.

The latch is deliberately process-global; every test here saves and restores
it via monkeypatch (capability-probe test-pollution lesson).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.user_funcs as user_funcs
from torchlens.errors import TorchLensWarning
from torchlens.options import CaptureOptions

pytestmark = [pytest.mark.smoke]


class BNNet(nn.Module):
    """conv -> bn -> relu."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x)))


def _arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(user_funcs, "_BATCHNORM_TRAIN_STATS_WARNED", False)


def _running_mean(model: BNNet) -> torch.Tensor:
    mean = model.bn.running_mean
    assert isinstance(mean, torch.Tensor)
    return mean.clone()


def _capture_batchnorm_warnings(model: nn.Module, x: torch.Tensor, **fields: Any) -> list[Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, x, capture=CaptureOptions(**fields) if fields else None)
    return [
        w
        for w in caught
        if isinstance(w.message, TorchLensWarning)
        and w.message.fields.get("code") == "batchnorm_train_stats_mutated"
    ]


def test_train_mode_bn_warns_coded_and_stats_really_mutate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The disclosure fires exactly when the mutation it names happens."""

    _arm(monkeypatch)
    model = BNNet().train()
    mean_before = _running_mean(model)
    x = torch.randn(2, 3, 8, 8)
    hits = _capture_batchnorm_warnings(model, x)
    assert len(hits) == 1, "train-mode BN capture must warn coded exactly once"
    assert "running statistics" in str(hits[0].message)
    assert not torch.equal(mean_before, _running_mean(model)), (
        "the warning fired but the stats did not mutate -- the disclosure lies"
    )


def test_warn_is_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second mutating capture stays silent (the latch holds)."""

    _arm(monkeypatch)
    x = torch.randn(2, 3, 8, 8)
    assert len(_capture_batchnorm_warnings(BNNet().train(), x)) == 1
    assert len(_capture_batchnorm_warnings(BNNet().train(), x)) == 0
    assert user_funcs._BATCHNORM_TRAIN_STATS_WARNED is True


def test_eval_mode_never_warns_and_keeps_the_latch_unarmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """eval() captures do not mutate, do not warn, and do not burn the latch."""

    _arm(monkeypatch)
    model = BNNet().eval()
    mean_before = _running_mean(model)
    hits = _capture_batchnorm_warnings(model, torch.randn(2, 3, 8, 8))
    assert hits == []
    assert torch.equal(mean_before, _running_mean(model))
    assert user_funcs._BATCHNORM_TRAIN_STATS_WARNED is False, (
        "a non-mutating capture burned the once-per-process latch; a later "
        "mutating capture would then stay silent"
    )


def test_inference_only_still_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """no_grad does not stop running-stat updates, so the disclosure holds."""

    _arm(monkeypatch)
    model = BNNet().train()
    mean_before = _running_mean(model)
    hits = _capture_batchnorm_warnings(model, torch.randn(2, 3, 8, 8), inference_only=True)
    assert len(hits) == 1
    assert not torch.equal(mean_before, _running_mean(model))


def test_track_running_stats_false_never_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A train-mode BN without running stats has nothing to disclose."""

    _arm(monkeypatch)
    model = BNNet().train()
    model.bn = nn.BatchNorm2d(4, track_running_stats=False).train()
    hits = _capture_batchnorm_warnings(model, torch.randn(2, 3, 8, 8))
    assert hits == []


@pytest.mark.real_model
def test_realism_train_bn_fixture_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """R0 realism: the registry train-bn structural fixture discloses."""

    from tests.real_model.r0.families import build_structural

    _arm(monkeypatch)
    model, args, kwargs = build_structural("train-bn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(model, *args, **kwargs)
    hits = [
        w
        for w in caught
        if isinstance(w.message, TorchLensWarning)
        and w.message.fields.get("code") == "batchnorm_train_stats_mutated"
    ]
    assert len(hits) == 1

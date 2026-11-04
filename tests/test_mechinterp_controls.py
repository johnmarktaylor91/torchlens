"""Tests for the private controls helper (``torchlens.mechinterp._controls``)."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.mechinterp._controls import BASELINES, run_controls

DEMO = Path(__file__).resolve().parents[1] / "examples" / "intervention"
DEMO = DEMO / "19_controls_wrong_tensor.py"


@pytest.fixture(scope="module")
def demo():
    """The wrong-tensor example's namespace, trained model, rows and metric."""

    namespace = runpy.run_path(str(DEMO))
    model, x, labels = namespace["build"]()
    return model, x, namespace["margin_metric"](labels)


def test_wrong_tensor_claim_fails_the_baseline_family(demo) -> None:
    """Zero-ablating the near-constant gain looks large; mean and resample do not."""

    model, x, metric = demo
    report = run_controls(model, x, tl.module("gain"), metric, axis=0)
    rows = {row.name: row for row in report.rows if row.family == "baseline"}
    assert set(rows) == set(BASELINES)
    assert report.effect < -1.0
    assert rows["mean"].survives is False and rows["resample"].survives is False
    assert abs(rows["mean"].effect) < 0.1 * abs(report.effect)
    assert [family.family for family in report.families] == ["baseline"]
    assert not report.survives
    assert "does not survive" in str(report)


def test_right_tensor_claim_survives_every_family(demo) -> None:
    """The normalized features keep the effect under all three families."""

    model, x, metric = demo
    direction = (model.readout.weight[1] - model.readout.weight[0]).detach()
    report = run_controls(
        model,
        x,
        tl.module("norm"),
        metric,
        axis=0,
        direction=direction,
        feature_axis=-1,
        n_directions=39,
        sites={"pre-norm": tl.module("embed"), "post-norm": tl.module("norm")},
    )
    assert [family.family for family in report.families] == ["baseline", "direction", "site"]
    assert all(family.survives for family in report.families), str(report)
    assert report.direction_p == pytest.approx(1 / 40)
    randoms = [row for row in report.rows if row.name.startswith("random")]
    assert len(randoms) == 39 and all(row.survives is None for row in randoms)
    assert report.survives


def test_controls_are_deterministic(demo) -> None:
    """The same seed gives the same rows."""

    model, x, metric = demo
    first = run_controls(model, x, tl.module("gain"), metric, axis=0, baseline="mean")
    second = run_controls(model, x, tl.module("gain"), metric, axis=0, baseline="mean")
    assert first.rows == second.rows


class _UnusedBranch(nn.Module):
    """A model whose ``side`` branch never reaches the output."""

    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Linear(3, 2)
        self.side = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        self.side(x)
        return self.main(x)


def _sum_metric(trace) -> torch.Tensor:
    """Sum of the output."""

    return trace.output_ops[-1].out.sum()


def test_claim_that_moves_nothing_does_not_survive() -> None:
    """A zero claimed effect is reported as such, never as surviving."""

    torch.manual_seed(0)
    report = run_controls(
        _UnusedBranch(), torch.randn(4, 3), tl.module("side"), _sum_metric, axis=0
    )
    assert report.effect == 0.0
    assert [family.family for family in report.families] == ["claim"]
    assert not report.survives


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"baseline": "noise"}, "baseline must be one of"),
        ({"n_directions": 0}, "n_directions must be at least 1"),
        ({"min_ratio": 0.0}, "min_ratio must be in"),
        ({"alpha": 1.0}, "alpha must be in"),
        ({"direction": torch.ones(2)}, "needs feature_axis"),
        ({"direction": torch.zeros(2), "feature_axis": -1}, "nonzero norm"),
    ],
)
def test_bad_settings_are_refused(kwargs, match) -> None:
    """Each invalid knob is named in the refusal."""

    torch.manual_seed(0)
    with pytest.raises(ValueError, match=match):
        run_controls(
            _UnusedBranch(), torch.randn(4, 3), tl.module("main"), _sum_metric, axis=0, **kwargs
        )


def test_site_firing_more_than_once_is_refused() -> None:
    """A selector matching several ops cannot name one donor tensor."""

    torch.manual_seed(0)
    with pytest.raises(ValueError, match="fired 2 times"):
        run_controls(_UnusedBranch(), torch.randn(4, 3), tl.func("linear"), _sum_metric, axis=0)


def test_non_scalar_metric_is_refused() -> None:
    """The metric must reduce to one number."""

    torch.manual_seed(0)
    with pytest.raises(ValueError, match="must return one number"):
        run_controls(
            _UnusedBranch(),
            torch.randn(4, 3),
            tl.module("main"),
            lambda trace: trace.output_ops[-1].out,
            axis=0,
        )

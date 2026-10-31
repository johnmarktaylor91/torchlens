"""The causal audit measures its single-unit rows through one ``site_sweep``.

Every unit of a layer claim, and (under zero ablation) every unit of a read
table, including one element of an element-grain row, is one candidate of a
``torchlens.experiment.sweep_sites``: its site label or ``tl.units(site,
mask)``. Each unit's measured effect is exactly what ablating it on its own
fork gives (the path the audit took before, kept here as the reference), and
the sweep's effect table holds one member per unit. A read table's "mean" and
baseline edits are each site's own value, so those units keep their own forks.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from test_attribution_eval_causal_audit import SmallCnn, _linear_case
from test_attribution_eval_causal_audit_tables import LinearNet, X, _read, _seed

import torchlens.attribution as attribution
import torchlens.experiment as experiment


class _Spy:
    """Record every ``site_sweep`` call's candidates and effect table."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[dict[str, Any], Any]] = []
        real = experiment.sweep_sites

        def _sweep(baseline: Any, *, candidates: Any, **kwargs: Any) -> Any:
            bundle = real(baseline, candidates=candidates, **kwargs)
            self.calls.append((dict(candidates), bundle.effects()))
            return bundle

        monkeypatch.setattr(experiment, "sweep_sites", _sweep)


def _members(effects: Any) -> list[Any]:
    return [row for row in effects.rows if row.candidate_id != "__baseline__"]


def _assert_each_unit_equals_its_own_fork(audit: Any) -> None:
    plan = audit.units[0]._plan
    for row in audit.units:
        assert row.measured_effect == audit.original_score - plan.ablated_score((row.index,)), row


def test_read_element_units_run_in_one_sweep_and_equal_their_own_fork(monkeypatch) -> None:
    """Every element unit of a read table is one sweep candidate; each equals its fork exactly."""

    spy = _Spy(monkeypatch)
    model = LinearNet().eval().double()
    audit = attribution.causal_audit(model, X, claim=_read(model, reduce=None), target=_seed(), k=3)
    assert audit.extra["grain"] == "element"
    ((candidates, effects),) = spy.calls
    assert len(candidates) == len(audit.units) > len({row.site for row in audit.units})
    assert all(not isinstance(selection, str) for selection in candidates.values())
    members = _members(effects)
    assert len(members) == len(audit.units)
    assert {member.status for member in members} == {"released"}
    _assert_each_unit_equals_its_own_fork(audit)


@pytest.mark.parametrize("ablation", ["mean", 2.0])
def test_read_mean_and_baseline_units_keep_their_own_forks(monkeypatch, ablation: Any) -> None:
    """A read table's "mean" and baseline edits differ per site: no sweep, one fork per unit."""

    spy = _Spy(monkeypatch)
    model = LinearNet().eval().double()
    audit = attribution.causal_audit(
        model, X, claim=_read(model, reduce=None), target=_seed(), k=3, edit=ablation
    )
    assert spy.calls == []
    _assert_each_unit_equals_its_own_fork(audit)


@pytest.mark.parametrize("ablation", ["zero", "mean", 2.0], ids=["zero", "mean", "baseline"])
def test_layer_units_run_in_one_sweep_and_equal_their_own_fork(monkeypatch, ablation: Any) -> None:
    """A layer claim's units share one site and one edit: one sweep for every ablation kind."""

    spy = _Spy(monkeypatch)
    model, x = _linear_case()
    claim = attribution.layer_attribution(model, x, target=1, layer="fc1")
    audit = attribution.causal_audit(model, x, claim=claim, target=1, k=2, edit=ablation)
    ((candidates, effects),) = spy.calls
    assert len(candidates) == len(audit.units) == 6
    assert len(_members(effects)) == 6
    _assert_each_unit_equals_its_own_fork(audit)


def test_layer_units_with_several_element_axes_equal_their_own_fork(monkeypatch) -> None:
    """Units indexed over channels and positions (a conv site) are swept one candidate each."""

    spy = _Spy(monkeypatch)
    torch.manual_seed(0)
    model, x = SmallCnn().eval().double(), torch.randn(2, 1, 4, 4, dtype=torch.float64)
    claim = attribution.layer_attribution(model, x, target=1, layer="conv")
    audit = attribution.causal_audit(model, x, claim=claim, target=1, k=3, num_random_draws=2)
    ((candidates, effects),) = spy.calls
    assert len(candidates) == len(audit.units) == claim.values[0].numel()
    assert len(_members(effects)) == len(audit.units)
    _assert_each_unit_equals_its_own_fork(audit)

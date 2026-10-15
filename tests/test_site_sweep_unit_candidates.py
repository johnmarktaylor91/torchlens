"""``site_sweep`` candidates that select units (element masks) at one site.

A unit selection names a site plus a mask over its output elements. Two
selections of different units at the same site are different candidates: the
sweep measures each one separately, and each member's value equals a direct
``trace.fork().do(...)`` of the same edit exactly. Two selections of the same
units are still the same candidate and refuse as duplicates, however the
units are spelled (a mask or coordinate tuples).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors.episode import BundleExperimentError
from torchlens.experiment import site_sweep
from torchlens.options import CaptureOptions, InterventionOptions


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _setup() -> tuple[nn.Module, torch.Tensor, tl.Trace, str]:
    torch.manual_seed(0)
    model = _Mlp().double().eval()
    x = torch.randn(2, 4, dtype=torch.float64)
    baseline = tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))
    label = next(op.layer_label for op in baseline.layer_list if "relu" in op.layer_label)
    return model, x, baseline, label


def _metric(member: tl.Trace) -> float:
    return float(member[member.output_layers[0]].out.sum().item())


def _unit_mask(units: tuple[int, ...]) -> torch.Tensor:
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, list(units)] = True
    return mask


_UNITS = {"u0": (0,), "u1": (1,), "u23": (2, 3)}


@pytest.mark.parametrize("engine", ["replay", "live_hook"])
def test_distinct_units_at_one_site_are_distinct_candidates(engine: str) -> None:
    model, x, baseline, label = _setup()
    candidates = {name: tl.units(label, _unit_mask(units)) for name, units in _UNITS.items()}
    bundle = site_sweep(
        baseline,
        candidates=candidates,
        edit=tl.zero_ablate(),
        metric=_metric,
        retain="none",
        model=model,
        x=x,
        engine=engine,
    )
    rows = {row.candidate_id: row for row in bundle.effects().rows}
    values = set()
    for name, units in _UNITS.items():
        direct = baseline.fork().do(
            tl.units(label, _unit_mask(units)),
            tl.zero_ablate(),
            intervention=InterventionOptions(engine="replay"),
        )
        assert rows[name].status == "completed"
        assert rows[name].value == _metric(direct)
        values.add(rows[name].value)
    # Each candidate ablated only its own units: the three effects differ, and
    # none equals ablating the whole site.
    whole = baseline.fork().do(
        label, tl.zero_ablate(), intervention=InterventionOptions(engine="replay")
    )
    assert len(values) == len(_UNITS)
    assert _metric(whole) not in values


def test_identical_units_still_refuse_as_duplicates() -> None:
    model, x, baseline, label = _setup()
    with pytest.raises(BundleExperimentError) as excinfo:
        site_sweep(
            baseline,
            candidates={
                "a": tl.units(label, _unit_mask((1,))),
                "b": tl.units(label, _unit_mask((1,))),
            },
            edit=tl.zero_ablate(),
            metric=_metric,
            retain="none",
            model=model,
            x=x,
        )
    assert excinfo.value.fields["code"] == "site_sweep_duplicate_candidates"


def test_same_units_spelled_two_ways_are_one_candidate() -> None:
    model, x, baseline, label = _setup()
    for units in ((1,), tuple(range(6))):
        coordinates = [(row, unit) for row in range(2) for unit in units]
        with pytest.raises(BundleExperimentError) as excinfo:
            site_sweep(
                baseline,
                candidates={
                    "mask": tl.units(label, _unit_mask(units)),
                    "coordinates": tl.units(label, coordinates),
                },
                edit=tl.zero_ablate(),
                metric=_metric,
                retain="none",
                model=model,
                x=x,
            )
        assert excinfo.value.fields["code"] == "site_sweep_duplicate_candidates"

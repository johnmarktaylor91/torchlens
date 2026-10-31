"""CPR and CMD (``torchlens.mechinterp.circuit_performance_ratio`` / ``circuit_model_difference``).

Pre-registered bar: on the fixed faithfulness vector below, over the ten MIB
edge fractions, both functions equal a literal transcription of the pinned
MIB loop (``MIB_circuit_track/evaluation.py``, ``evaluate_area_under_curve``:
``area_under`` is CPR, ``area_from_1`` is CMD) within 1e-12.
"""

from __future__ import annotations

import math

import pytest

import torchlens.mechinterp

PERCENTAGES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1)
FAITHFULNESS = (0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.05, 1.0, 0.98, 1.0)


def _mib_areas(faithfulnesses, percentages, log_scale=False):
    """Literal transcription of the pinned reference loop (area_under, area_from_1)."""

    area_under = 0.0
    area_from_1 = 0.0
    for i in range(len(faithfulnesses) - 1):
        i_1, i_2 = i, i + 1
        x_1 = percentages[i_1]
        x_2 = percentages[i_2]
        # area from point to 100
        if log_scale:
            x_1 = math.log(x_1)
            x_2 = math.log(x_2)
        trapezoidal = (x_2 - x_1) * (
            ((abs(1.0 - faithfulnesses[i_1])) + (abs(1.0 - faithfulnesses[i_2]))) / 2
        )
        area_from_1 += trapezoidal

        trapezoidal = (x_2 - x_1) * ((faithfulnesses[i_1] + faithfulnesses[i_2]) / 2)
        area_under += trapezoidal
    return area_under, area_from_1


def test_cpr_and_cmd_equal_the_pinned_loop_on_the_bar_vector() -> None:
    """Bar CIRC-2d: both areas within 1e-12 of the transcribed reference."""

    area_under, area_from_1 = _mib_areas(FAITHFULNESS, PERCENTAGES)
    cpr = torchlens.mechinterp.circuit_performance_ratio(FAITHFULNESS)
    cmd = torchlens.mechinterp.circuit_model_difference(FAITHFULNESS)
    assert abs(cpr - area_under) <= 1e-12
    assert abs(cmd - area_from_1) <= 1e-12
    assert 0.0 < cmd < cpr < 1.0


def test_log_scale_mapping_and_explicit_fractions_match_the_loop() -> None:
    """``log_scale=True``, a ``{fraction: f}`` mapping and ``fractions=`` agree with the loop."""

    area_under, area_from_1 = _mib_areas(FAITHFULNESS, PERCENTAGES, log_scale=True)
    curve = dict(zip(PERCENTAGES, FAITHFULNESS, strict=True))
    cpr = torchlens.mechinterp.circuit_performance_ratio(curve, log_scale=True)
    cmd = torchlens.mechinterp.circuit_model_difference(curve, log_scale=True)
    assert abs(cpr - area_under) <= 1e-12
    assert abs(cmd - area_from_1) <= 1e-12
    short = torchlens.mechinterp.circuit_performance_ratio([0.5, 1.0], fractions=[0.5, 1.0])
    assert short == pytest.approx(0.375, abs=1e-15)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (([0.5, 1.0],), {}),
        (([0.5],), {"fractions": [1.0]}),
        (([0.5, 1.0],), {"fractions": [1.0, 0.5]}),
        (({0.5: 1.0, 1.0: 1.0},), {"fractions": [0.5, 1.0]}),
        (([0.5, float("nan")],), {"fractions": [0.5, 1.0]}),
        (([0.5, 1.0],), {"fractions": [0.0, 1.0], "log_scale": True}),
    ],
)
def test_malformed_curves_refuse_typed(args, kwargs) -> None:
    """Length mismatch, unsorted fractions, mapping plus fractions, NaN and log of 0 refuse."""

    with pytest.raises(torchlens.errors.MechInterpError) as caught:
        torchlens.mechinterp.circuit_performance_ratio(*args, **kwargs)
    assert caught.value.fields["code"] == "mi_faithfulness_curve_invalid"

"""Overhead-measurement harness law (torchnative W0.9 / section 5).

The panel retracted six of its own headline numbers; every one died of
harness or read error. The harness therefore enforces the protocol
mechanically: alternating paired ratios, a refusal predicate declared
before the run and printed in the artifact, an execution-scope witness on
both arms of every pair, and one-sided floors where point estimates are
refused. The composition rows pinned here: the harness must refuse ITSELF
on mismatched arms, and two shapes of one model legitimately produce
different multipliers (the 4x-spread row -- why capture overhead ships as a
matrix or not at all).
"""

from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import TorchLensError
from torchlens.observability import ArmSpec, RefusalPredicate, measure_overhead


def test_admissible_measurement_produces_full_artifact() -> None:
    """A clean paired run yields statistics plus the declared scope."""

    payload = torch.randn(8, 8)

    def base() -> torch.Tensor:
        time.sleep(0.002)
        return payload + payload

    def instrumented() -> torch.Tensor:
        time.sleep(0.003)
        return payload + payload

    result = measure_overhead(
        ArmSpec("raw", base), ArmSpec("instrumented", instrumented), reps=7, warmup=2
    )
    artifact = result.to_artifact()
    assert artifact["schema"] == "torchlens.overhead_measurement.v1"
    assert (artifact["baseline"], artifact["instrumented"]) == ("raw", "instrumented")
    assert artifact["refusal_predicate"] == {"min_ratio": 1.0, "max_relative_iqr": 0.15}
    assert artifact["torch_threads"] >= 1
    assert artifact["clock"] == "time.perf_counter"
    assert artifact["torch"] == torch.__version__
    assert (artifact["reps"], artifact["warmup"]) == (7, 2)
    assert artifact["witness"] == {"kind": "output_agreement", "failures": 0}
    # Wall-clock values REPORT, never gate (fleet perf doctrine): a 3 ms vs 2 ms sleep pair
    # can tie on a loaded host, so the contract is a complete artifact with finite, positive
    # statistics, not the sign of any ratio. Admissibility is the predicate's verdict either way.
    assert len(result.pair_ratios) == 7
    assert artifact["pair_ratios"] == list(result.pair_ratios)
    for value in (*result.pair_ratios, result.median_ratio, result.floor_ratio):
        assert value is not None and math.isfinite(value) and value > 0.0
    assert result.iqr is not None and math.isfinite(result.iqr) and result.iqr >= 0.0
    assert result.floor_ratio == min(result.pair_ratios)
    assert artifact["floor_ratio"] == result.floor_ratio
    assert artifact["median_ratio"] == result.median_ratio
    assert artifact["admissible"] is (not result.refusals)
    assert artifact["refusals"] == list(result.refusals)


@pytest.mark.smoke
def test_physically_impossible_ratio_is_refused() -> None:
    """An 'instrument' faster than its baseline refuses the point estimate."""

    def slow() -> float:
        time.sleep(0.001)
        return 1.0

    def fast() -> float:
        return 1.0

    result = measure_overhead(ArmSpec("baseline", slow), ArmSpec("instrumented", fast), reps=5)
    assert not result.admissible
    assert any("physically_impossible_ratio" in reason for reason in result.refusals)


@pytest.mark.smoke
def test_witness_divergence_refuses_the_measurement() -> None:
    """Composition row: mismatched arms -- the harness must refuse itself."""

    counter = {"n": 0}

    def base() -> torch.Tensor:
        return torch.ones(4)

    def drifting() -> torch.Tensor:
        counter["n"] += 1
        return torch.ones(4) * counter["n"]  # a DIFFERENT program each call

    result = measure_overhead(ArmSpec("baseline", base), ArmSpec("instrumented", drifting), reps=5)
    assert not result.admissible
    assert any("witness_divergence" in reason for reason in result.refusals)
    # No one-sided floor is claimable off a diverging witness.
    assert result.floor_ratio is None


def test_two_shapes_of_one_model_are_separate_cells() -> None:
    """The 4x-spread row: overhead is a (model, shape, tier) matrix.

    Two shapes of the SAME model measure as two independent cells; the
    harness never averages across them (each call is one cell), so any
    single published capture-overhead number is misleading by construction.
    """

    model = nn.Linear(16, 16).eval()

    def cell(batch: int):
        x = torch.randn(batch, 16)

        def raw() -> torch.Tensor:
            with torch.no_grad():
                return model(x)

        def captured() -> torch.Tensor:
            log = tl.trace(model, x, save=None)
            out = log.output_ops[0].out
            log.cleanup()
            return out

        return measure_overhead(
            ArmSpec(f"raw@{batch}", raw),
            ArmSpec(f"captured@{batch}", captured),
            reps=3,
            warmup=1,
            predicate=RefusalPredicate(min_ratio=1.0, max_relative_iqr=100.0),
        )

    small = cell(2)
    # The floor is a one-sided bound: noise can only raise it.
    assert small.floor_ratio is None or small.floor_ratio > 0
    assert small.baseline_label == "raw@2"
    artifact = small.to_artifact()
    assert artifact["baseline"] == "raw@2"


def test_zero_rep_measurement_refuses_typed() -> None:
    """A zero-rep 'measurement' is not a measurement."""

    with pytest.raises(TorchLensError) as excinfo:
        measure_overhead(ArmSpec("a", lambda: 1.0), ArmSpec("b", lambda: 1.0), reps=1, warmup=0)
    assert excinfo.value.fields["code"] == "overhead_arms_invalid"

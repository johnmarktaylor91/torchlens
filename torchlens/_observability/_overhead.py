"""The overhead-measurement harness (torchnative W0.9; section-5 protocol).

Every published overhead figure -- TorchLens's and competitors' -- goes
through this harness. The protocol is memo law, written in the panel's own
casualty list (every retracted headline died of harness or read error, not
arithmetic):

1. Interleaved A/B with ALTERNATING arm order; the statistic is the median
   of PAIRED ratios (pairing cancels the drift that destroyed the
   block-sequential numbers).
2. The artifact declares warmup, reps, thread count, median AND dispersion,
   and the host load -- RECORDED, never used as a threshold.
3. The refusal predicate is DECLARED BEFORE THE RUN and printed in the
   artifact: no physically impossible ratio (an additive instrument is
   never faster than what it instruments) and relative dispersion below a
   declared bound. The specific constants are tuning; the SHAPE is law.
4. An execution-scope witness runs on BOTH arms of every pair -- outputs
   compared, refusing on divergence. A timing harness without this clause
   produced two wrong numbers in the panel's own file.
5. One-sided bounds are admissible where point estimates are not: the
   minimum paired ratio with a passing witness is a floor noise can only
   raise.
6. Numbers are GENERATED into docs from the artifact, never typed.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from ._errors import ObservabilityError

__tl_layer__ = "L5"


@dataclass(frozen=True)
class ArmSpec:
    """One measurement arm: a label and a zero-argument callable.

    The callable returns the arm's OUTPUT (tensor / scalar / structure);
    the harness compares outputs across arms as its execution-scope
    witness, so both arms must compute the same program.
    """

    label: str
    fn: Callable[[], Any]


@dataclass(frozen=True)
class RefusalPredicate:
    """The refusal predicate, declared BEFORE the run and printed after.

    ``min_ratio`` guards physical possibility (an additive instrument is
    never faster than what it instruments; 1.0 for additive-instrument
    questions, ``None`` to disable for A/B questions with no additivity
    claim). ``max_relative_iqr`` bounds dispersion (IQR / median). The
    constants are tuning ([UI-SPRINT]); the shape is law.
    """

    min_ratio: float | None = 1.0
    max_relative_iqr: float = 0.15


@dataclass(frozen=True)
class OverheadMeasurement:
    """One harness run: statistics, scope, witness, and admissibility.

    ``admissible`` is True only when every pair's witness passed AND the
    declared refusal predicate held. ``floor_ratio`` (the minimum paired
    ratio with a passing witness) is admissible as a ONE-SIDED bound even
    when the point estimate is refused -- a floor that noise can only
    raise.
    """

    baseline_label: str
    instrumented_label: str
    median_ratio: float | None
    iqr: float | None
    floor_ratio: float | None
    pair_ratios: tuple[float, ...]
    admissible: bool
    refusals: tuple[str, ...]
    warmup: int
    reps: int
    torch_threads: int
    host_load: tuple[float, float, float] | None
    predicate: RefusalPredicate
    witness: dict[str, Any] = field(default_factory=dict)

    def to_artifact(self) -> dict[str, Any]:
        """Return the JSON-able artifact (docs numbers GENERATE from this)."""

        return {
            "schema": "torchlens.overhead_measurement.v1",
            "baseline": self.baseline_label,
            "instrumented": self.instrumented_label,
            "median_ratio": self.median_ratio,
            "iqr": self.iqr,
            "floor_ratio": self.floor_ratio,
            "pair_ratios": list(self.pair_ratios),
            "admissible": self.admissible,
            "refusals": list(self.refusals),
            "warmup": self.warmup,
            "reps": self.reps,
            "torch_threads": self.torch_threads,
            "host_load_1_5_15": None if self.host_load is None else list(self.host_load),
            "refusal_predicate": {
                "min_ratio": self.predicate.min_ratio,
                "max_relative_iqr": self.predicate.max_relative_iqr,
            },
            "witness": dict(self.witness),
            "clock": "time.perf_counter",
            "torch": torch.__version__,
        }


def _default_compare(baseline_output: Any, instrumented_output: Any) -> bool:
    """Default execution-scope witness: outputs must agree numerically."""

    if isinstance(baseline_output, torch.Tensor) and isinstance(instrumented_output, torch.Tensor):
        if baseline_output.shape != instrumented_output.shape:
            return False
        return bool(
            torch.allclose(
                baseline_output.detach().float(),
                instrumented_output.detach().float(),
                rtol=1e-4,
                atol=1e-5,
                equal_nan=True,
            )
        )
    if isinstance(baseline_output, (int, float)) and isinstance(instrumented_output, (int, float)):
        return abs(float(baseline_output) - float(instrumented_output)) <= 1e-4 * (
            1.0 + abs(float(baseline_output))
        )
    return type(baseline_output) is type(instrumented_output)


def _timed(fn: Callable[[], Any]) -> tuple[float, Any]:
    """Run one arm once, returning (elapsed_seconds, output)."""

    start = time.perf_counter()
    output = fn()
    return time.perf_counter() - start, output


def _run_pairs(
    baseline: ArmSpec,
    instrumented: ArmSpec,
    reps: int,
    witness_compare: Callable[[Any, Any], bool],
) -> tuple[list[float], int]:
    """Run the alternating pairs; return passing-witness ratios + failures."""

    ratios: list[float] = []
    witness_failures = 0
    for pair_index in range(reps):
        if pair_index % 2 == 0:
            base_seconds, base_out = _timed(baseline.fn)
            inst_seconds, inst_out = _timed(instrumented.fn)
        else:
            inst_seconds, inst_out = _timed(instrumented.fn)
            base_seconds, base_out = _timed(baseline.fn)
        if not witness_compare(base_out, inst_out):
            witness_failures += 1
            continue
        if base_seconds > 0:
            ratios.append(inst_seconds / base_seconds)
    return ratios, witness_failures


def _evaluate(
    ratios: list[float],
    witness_failures: int,
    reps: int,
    predicate: RefusalPredicate,
) -> tuple[float | None, float | None, float | None, tuple[str, ...]]:
    """Apply the DECLARED refusal predicate; return stats + refusal reasons."""

    refusals: list[str] = []
    if witness_failures:
        refusals.append(
            f"witness_divergence: {witness_failures}/{reps} pairs diverged in "
            "output -- the two arms did not execute the same program"
        )
    median_ratio: float | None = None
    iqr: float | None = None
    floor_ratio: float | None = None
    if ratios:
        median_ratio = statistics.median(ratios)
        quartiles = statistics.quantiles(ratios, n=4) if len(ratios) >= 4 else None
        iqr = None if quartiles is None else quartiles[2] - quartiles[0]
        floor_ratio = min(ratios)
        if predicate.min_ratio is not None and floor_ratio < predicate.min_ratio:
            refusals.append(
                f"physically_impossible_ratio: min paired ratio {floor_ratio:.4f} "
                f"< declared floor {predicate.min_ratio} (an additive "
                "instrument is never faster than what it instruments)"
            )
        if iqr is not None and median_ratio and iqr / median_ratio > predicate.max_relative_iqr:
            refusals.append(
                f"dispersion_exceeds_bound: IQR/median {iqr / median_ratio:.3f} > "
                f"declared {predicate.max_relative_iqr}"
            )
    else:
        refusals.append("no_admissible_pairs: every pair was refused before ratio computation")
    return median_ratio, iqr, floor_ratio, tuple(refusals)


def measure_overhead(  # noqa: PLR0913 -- the six knobs ARE the declared section-5 protocol scope (arms, reps, warmup, predicate, witness); packing them would hide the disclosure
    baseline: ArmSpec,
    instrumented: ArmSpec,
    *,
    reps: int = 11,
    warmup: int = 3,
    predicate: RefusalPredicate | None = None,
    compare: Callable[[Any, Any], bool] | None = None,
) -> OverheadMeasurement:
    """Run the interleaved paired-ratio protocol on two arms.

    Parameters
    ----------
    baseline:
        The uninstrumented arm (the ratio's denominator).
    instrumented:
        The instrumented arm (the ratio's numerator).
    reps:
        Number of pairs (each pair runs both arms, order alternating).
    warmup:
        Discarded warmup pairs run before measurement.
    predicate:
        The refusal predicate; defaults to the additive-instrument shape
        (min ratio 1.0, relative IQR < 0.15). Declared before the run and
        printed in the artifact either way.
    compare:
        Execution-scope witness comparing the two arms' outputs per pair;
        defaults to numeric agreement. Divergence REFUSES the measurement
        (clause 4) -- it never silently narrows to the passing pairs.

    Returns
    -------
    OverheadMeasurement
        Statistics, admissibility, refusals, and the full scope disclosure.

    Raises
    ------
    ObservabilityError
        ``overhead_arms_invalid`` when reps/warmup are not positive sane
        values -- a zero-rep "measurement" is not a measurement.
    """

    if reps < 3 or warmup < 0:
        raise ObservabilityError(
            f"reps={reps}, warmup={warmup}: the paired protocol needs at "
            "least 3 measured pairs and non-negative warmup.",
            code="overhead_arms_invalid",
            remedy="Use reps >= 3 (11 is the default) and warmup >= 0.",
        )
    active_predicate = predicate if predicate is not None else RefusalPredicate()
    witness_compare = compare if compare is not None else _default_compare

    for _ in range(warmup):
        baseline.fn()
        instrumented.fn()

    ratios, witness_failures = _run_pairs(baseline, instrumented, reps, witness_compare)
    median_ratio, iqr, floor_ratio, refusals = _evaluate(
        ratios, witness_failures, reps, active_predicate
    )
    try:
        host_load = os.getloadavg()
    except OSError:  # pragma: no cover - platform without loadavg
        host_load = None
    return OverheadMeasurement(
        baseline_label=baseline.label,
        instrumented_label=instrumented.label,
        median_ratio=median_ratio,
        iqr=iqr,
        floor_ratio=floor_ratio if not witness_failures else None,
        pair_ratios=tuple(ratios),
        admissible=not refusals,
        refusals=refusals,
        warmup=warmup,
        reps=reps,
        torch_threads=torch.get_num_threads(),
        host_load=host_load,
        predicate=active_predicate,
        witness={
            "kind": "output_agreement" if compare is None else "caller_supplied",
            "failures": witness_failures,
        },
    )


__all__ = [
    "ArmSpec",
    "OverheadMeasurement",
    "RefusalPredicate",
    "measure_overhead",
]

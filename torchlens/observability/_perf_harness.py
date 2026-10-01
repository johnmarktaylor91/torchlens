"""The pinned interleaved A/B perf harness (explorer D25; lane F25).

Every published watch/explorer performance number comes from THIS harness,
never a casual pair of timings: the panel's shared box showed a 1.77x
noise floor on an IDENTICAL workload, which retired every sub-1.8x number
any lab produced in three rounds. The harness therefore:

- runs A and B INTERLEAVED (drift-cancelling ABBA order), N >= 20 repeats;
- reports median AND spread, never a bare mean;
- measures the box's own noise band from an A/A calibration on the same
  workload shape; and
- REFUSES to publish any row whose effect sits inside that measured band
  (the row still prints -- marked ``within-noise``, with the band -- so a
  non-result is disclosed rather than dropped).

Rows land in a GENERATED markdown table (``format_markdown``); numbers are
never hand-quoted into docs. GPU gate rows additionally require pinned
affinity or a dedicated device (C-EXPLORER cluster row); CPU rows are
reported separately and never stand in for CUDA claims.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

__tl_layer__ = "L5"

#: D25 floor: fewer repeats cannot support a published claim.
MIN_PUBLISHABLE_REPEATS = 20


@dataclass(frozen=True)
class ABMeasurement:
    """One interleaved A/B row: medians, spread, band, publish verdict.

    ``ratio`` is ``median_b / median_a`` (B the candidate, A the baseline).
    ``publishable`` is True only when ``repeats`` meets the D25 floor AND
    the ratio sits outside the measured noise band.
    """

    name: str
    repeats: int
    median_a_s: float
    median_b_s: float
    iqr_a_s: float
    iqr_b_s: float
    ratio: float
    noise_band: float
    publishable: bool
    note: str = ""

    def verdict(self) -> str:
        """One-line human verdict for logs and reports."""

        state = "PUBLISHABLE" if self.publishable else "within-noise (NOT published)"
        return (
            f"{self.name}: ratio {self.ratio:.3f}x "
            f"(A {self.median_a_s * 1e3:.2f} ms, B {self.median_b_s * 1e3:.2f} ms, "
            f"band {self.noise_band:.3f}x, n={self.repeats}) -- {state}"
        )


def _interleaved_times(
    fn_a: Callable[[], object],
    fn_b: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
) -> tuple[list[float], list[float]]:
    """Time A and B interleaved with drift-cancelling ABBA ordering.

    Each repeat pair alternates which side runs first, so a monotone
    background-load drift charges both sides equally instead of the one
    that always ran second.
    """

    for _ in range(warmup):
        fn_a()
        fn_b()
    times_a: list[float] = []
    times_b: list[float] = []
    for i in range(repeats):
        first, second = (fn_a, fn_b) if i % 2 == 0 else (fn_b, fn_a)
        start = time.perf_counter()
        first()
        mid = time.perf_counter()
        second()
        end = time.perf_counter()
        if i % 2 == 0:
            times_a.append(mid - start)
            times_b.append(end - mid)
        else:
            times_b.append(mid - start)
            times_a.append(end - mid)
    return times_a, times_b


def _iqr(values: list[float]) -> float:
    """Interquartile range (the published spread statistic)."""

    if len(values) < 4:
        return max(values) - min(values) if values else 0.0
    quartiles = statistics.quantiles(values, n=4)
    return quartiles[2] - quartiles[0]


def measure_noise_band(
    fn: Callable[[], object],
    *,
    repeats: int = MIN_PUBLISHABLE_REPEATS,
    warmup: int = 3,
) -> float:
    """Measure the box's noise band from an A/A run of the SAME workload.

    Returns the ratio band ``max(r, 1/r) `` of an identical-workload
    interleaved comparison: any A/B effect inside ``[1/band, band]`` is not
    established on this box.
    """

    times_a, times_b = _interleaved_times(fn, fn, repeats=repeats, warmup=warmup)
    ratio = statistics.median(times_b) / statistics.median(times_a)
    return max(ratio, 1.0 / ratio)


def measure_ab(  # noqa: PLR0913 -- the D25 publication rule's inputs are each an explicit named knob by design (bundling them would hide what a published row was measured under)
    name: str,
    fn_a: Callable[[], object],
    fn_b: Callable[[], object],
    *,
    repeats: int = MIN_PUBLISHABLE_REPEATS,
    warmup: int = 3,
    noise_band: float | None = None,
    note: str = "",
) -> ABMeasurement:
    """Run one pinned interleaved A/B comparison and apply the publish rule.

    Parameters
    ----------
    name:
        Row name for the generated table.
    fn_a:
        Baseline workload (zero-argument callable).
    fn_b:
        Candidate workload.
    repeats:
        Interleaved repeats; below :data:`MIN_PUBLISHABLE_REPEATS` the row
        can never publish.
    warmup:
        Untimed warmup runs per side.
    noise_band:
        Pre-measured noise band from :func:`measure_noise_band`; when
        ``None`` the harness measures it here from an A/A run of ``fn_a``.
    note:
        Free-text basis note (device, model, cadence) carried to the table.

    Returns
    -------
    ABMeasurement
        The row, with ``publishable`` decided by the D25 rule.
    """

    band = noise_band if noise_band is not None else measure_noise_band(fn_a, repeats=repeats)
    times_a, times_b = _interleaved_times(fn_a, fn_b, repeats=repeats, warmup=warmup)
    median_a = statistics.median(times_a)
    median_b = statistics.median(times_b)
    ratio = median_b / median_a
    outside_band = ratio > band or ratio < 1.0 / band
    return ABMeasurement(
        name=name,
        repeats=repeats,
        median_a_s=median_a,
        median_b_s=median_b,
        iqr_a_s=_iqr(times_a),
        iqr_b_s=_iqr(times_b),
        ratio=ratio,
        noise_band=band,
        publishable=outside_band and repeats >= MIN_PUBLISHABLE_REPEATS,
        note=note,
    )


def format_markdown(
    rows: list[ABMeasurement],
    *,
    sha: str,
    device: str,
    basis: str,
) -> str:
    """Render the GENERATED perf table; within-noise rows disclose, not drop.

    Parameters
    ----------
    rows:
        Measured rows in presentation order.
    sha:
        Repo SHA the numbers were measured at.
    device:
        Device string (``cpu`` rows never stand in for CUDA claims).
    basis:
        One-line measurement basis (model, batch x seq, cadence, load).

    Returns
    -------
    str
        Markdown document body for the generated numbers file.
    """

    lines = [
        "<!-- generated by torchlens.observability._perf_harness; do not hand-edit -->",
        "",
        f"Measured at SHA `{sha}` on device `{device}`. Basis: {basis}.",
        "",
        "Publication rule (explorer D25): a row publishes only when its effect",
        "sits OUTSIDE the measured same-workload noise band at n >= "
        f"{MIN_PUBLISHABLE_REPEATS} interleaved repeats; within-noise rows are",
        "disclosed as non-results, never quoted as numbers.",
        "",
        "| Row | Median A (ms) | Median B (ms) | Ratio | IQR A/B (ms) | Noise band | Verdict | Note |",
        "|---|---:|---:|---:|---|---:|---|---|",
    ]
    for row in rows:
        verdict = "published" if row.publishable else "WITHIN NOISE -- not published"
        lines.append(
            f"| {row.name} | {row.median_a_s * 1e3:.2f} | {row.median_b_s * 1e3:.2f} "
            f"| {row.ratio:.3f}x | {row.iqr_a_s * 1e3:.2f}/{row.iqr_b_s * 1e3:.2f} "
            f"| {row.noise_band:.3f}x | {verdict} | {row.note} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "ABMeasurement",
    "MIN_PUBLISHABLE_REPEATS",
    "format_markdown",
    "measure_ab",
    "measure_noise_band",
]

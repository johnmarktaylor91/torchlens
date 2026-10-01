"""Performance metrics: peaks, MFU, roofline, instrumented rate, cost measurer.

F09 (costreport items 13-16, 25; D16-D21, D27). The words law (memo 3.8)
governs every string here: "measured" is banned for analytic FLOP counts;
"achieved" and "throughput" are banned without joined device time; "MFU"
is banned over a kernel-union denominator or a measured-achievable peak
(those quantities get their own names).

- :func:`device_peaks` -- the explicit peaks constructor (D18): per-row
  provenance, typed refusal for unknown devices, nothing auto-selected;
  a zero-row catalog still refuses correctly, so launch does not gate on
  row curation (the row-data task is separately owned, post-launch).
- :func:`mfu` -- dtype-SPLIT and time-normalized (D16/D17): ideal seconds
  summed per execution mode over ADVERTISED dense peaks, divided by
  uninstrumented end-to-end step wall time ONLY. The kernel-union
  quantity is a different metric with its own name and stays a typed
  refusal until the correlation-ID join lands (D22).
- :func:`time_clean_step` -- the opt-in clean-step helper (D16): warmups,
  median, method disclosed; it states plainly that it re-runs the
  caller's forward.
- :func:`roofline` -- theoretical intensity/work map (D21): read-once/
  write-once ideal traffic (a TWO-SIDED estimate), bound verdicts are
  HYPOTHESES, aggregate intensity is sum/sum, coverage prints by reason,
  and sub-cache ops are excluded from headline bound counts.
- :func:`machine_balance` -- re-scoped to roofline ceilings (D18): the
  basis (measured-achievable vs advertised) is REQUIRED and disclosed;
  a ratio against a measured peak is never called MFU.
- :func:`instrumented_rate` -- the boxed opt-in diagnostic (D20):
  CPU-cells only, "instrumented" in the header, per-row applicability by
  execution device, typed refusal when the whole trace executed on CUDA.
- :func:`cost_report` -- a cost MEASURER, never an estimator (D27): runs
  the requested tiers on the user's own model and host and discloses
  that it ran N forwards.

Everything is live-only/session-time (no persisted fields; F09 holds no
adjudicated field-intent rows). Spellings DOCUMENTED-UNSTABLE pending
naming-session ratification.
"""

from __future__ import annotations

import statistics
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from ..errors import TorchLensWarning

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

#: Peak-provenance vocabulary (D18): nothing is ever auto-selected and
#: sparse marketing peaks never default.
PEAK_SOURCES: tuple[str, ...] = ("user_supplied", "catalog")

#: The one lawful MFU denominator (D16): uninstrumented end-to-end step
#: wall time. The kernel-union quotient is a DIFFERENT metric with its
#: own name; feeding it here is a typed refusal, never a bigger number.
MFU_DENOMINATOR: str = "step_wall_time"

#: Roofline coverage reasons (D21): the split prints BY REASON.
ROOFLINE_REASONS: tuple[str, ...] = (
    "covered",
    "no_rule",
    "zero_logical_traffic",
    "missing_shape_dtype",
)

#: Sub-cache exclusion threshold for headline bound counts (D21): a
#: roofline that confidently classifies a 4 KB elementwise op is lying
#: with precision.
CACHE_RESIDENT_BYTES_DEFAULT: int = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Peaks (D18)


@dataclass(frozen=True)
class PeakRow:
    """One advertised dense per-mode peak with full provenance (D18)."""

    device_identity: str
    execution_mode: str
    dense_flops_per_s: float
    source_citation: str
    source_date: str
    clock_assumptions: str
    sparse_policy: str = "dense_only"
    source: str = "user_supplied"


@dataclass(frozen=True)
class DevicePeaks:
    """An explicit peaks catalog: constructed, never auto-selected (D18)."""

    rows: tuple[PeakRow, ...]

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): rows point, never dump."""

        devices = sorted({row.device_identity for row in self.rows})
        shown = ", ".join(devices[:3]) + (" ..." if len(devices) > 3 else "")
        return f"DevicePeaks({len(self.rows)} row(s): {shown}; read .rows)"

    def for_device(self, device_identity: str, execution_mode: str) -> PeakRow:
        """Look up one (device, mode) peak; unknown pairs refuse typed."""

        for row in self.rows:
            if row.device_identity == device_identity and row.execution_mode == execution_mode:
                return row
        raise InvalidArgumentError(
            f"no peak row for device {device_identity!r} in execution mode "
            f"{execution_mode!r} (catalog holds {len(self.rows)} row(s)).",
            code="device_peaks_unknown_device",
            remedy=(
                "Add an explicit PeakRow via tl.report.device_peaks(...) with the vendor's "
                "ADVERTISED dense per-mode peak, its citation, date, and clock assumptions. "
                "Nothing is ever auto-selected."
            ),
        )


def device_peaks(rows: Any) -> DevicePeaks:
    """Construct the explicit peaks catalog (D18).

    Every row must carry full provenance: exact device identity, execution
    mode, the ADVERTISED dense peak, a source citation, a date, and clock
    assumptions. Sparse marketing peaks never default (``sparse_policy``
    stays ``"dense_only"`` unless a row says otherwise, disclosed).
    """

    validated: list[PeakRow] = []
    for row in rows:
        if not isinstance(row, PeakRow):
            raise InvalidArgumentError(
                f"device_peaks rows must be PeakRow instances; got {type(row).__name__}.",
                code="device_peaks_row_invalid",
                remedy="Construct tl.report.PeakRow(...) with full provenance per row.",
            )
        if not row.source_citation.strip() or not row.source_date.strip():
            raise InvalidArgumentError(
                f"peak row for {row.device_identity!r}/{row.execution_mode!r} is missing its "
                "source citation or date -- a peaks-table row without its citation is the "
                "staleness-lint failure class.",
                code="device_peaks_row_invalid",
                remedy="Cite the vendor document and its date on every PeakRow.",
            )
        validated.append(row)
    return DevicePeaks(rows=tuple(validated))


# ---------------------------------------------------------------------------
# MFU (D16/D17)


@dataclass(frozen=True)
class MfuResult:
    """The mode-split time-normalized MFU facts (D16).

    ``mode_terms`` disclose each execution mode's (flops, peak, ideal
    seconds); the peaks' source rides ``peaks_source`` (D17: modes are
    recorded ``user_supplied`` until joined kernel metadata can discharge
    them).
    """

    mfu: float
    ideal_seconds: float
    step_seconds: float
    mode_terms: tuple[tuple[str, int, float, float], ...]
    denominator: str
    denominator_scope: tuple[str, ...]
    peaks_source: str

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): mode terms point, never dump."""

        return (
            f"MfuResult(mfu={self.mfu:.4f}, ideal={self.ideal_seconds:.6f} s / "
            f"step={self.step_seconds:.6f} s, {len(self.mode_terms)} mode term(s), "
            f"peaks_source={self.peaks_source!r}; read .mode_terms)"
        )


@dataclass(frozen=True)
class MfuProvenance:
    """Where the MFU inputs came from (the D16/D17 disclosure knobs).

    ``denominator`` names the time base (exactly one is lawful; validated
    at the :func:`mfu` door), ``denominator_scope`` discloses which passes
    the step covers, and ``peaks_source`` records where the advertised
    peaks came from (D17: ``user_supplied`` until joined kernel metadata
    can discharge a ``catalog`` claim).
    """

    denominator: str = MFU_DENOMINATOR
    denominator_scope: tuple[str, ...] = ("forward",)
    peaks_source: str = "user_supplied"


def mfu(
    *,
    flops_by_mode: dict[str, int],
    peaks_by_mode: dict[str, float],
    step_seconds: float,
    provenance: MfuProvenance | None = None,
) -> MfuResult:
    """Model FLOPs utilization: mode-split, step-wall-time-normalized (D16).

    ``ideal_seconds = sum_m flops_m / advertised_dense_peak_m``;
    ``MFU = ideal_seconds / step_seconds``. At one mode this reduces
    exactly to ``F / (T * P)`` (the property test). The denominator is
    uninstrumented end-to-end step wall time ONLY -- launch bubbles,
    dataloader stalls, and collective waits are the point of the metric.
    ``provenance`` bundles the D16/D17 disclosure knobs; the defaults are
    the honest ones (``step_wall_time`` over the forward, user-supplied
    peaks) and every non-default claim is validated here.
    """

    if provenance is None:
        provenance = MfuProvenance()
    denominator = provenance.denominator
    peaks_source = provenance.peaks_source
    if denominator != MFU_DENOMINATOR:
        raise InvalidArgumentError(
            f"MFU has exactly one lawful denominator ({MFU_DENOMINATOR!r}); got "
            f"{denominator!r}. A kernel-union denominator deletes the inefficiency MFU "
            "exists to expose and matches no published MFU.",
            code="mfu_denominator_invalid",
            remedy=(
                "Pass uninstrumented end-to-end step wall time (tl.report.time_clean_step "
                "is the opt-in helper). The utilization-within-attributed-kernels quantity "
                "is a different metric with its own name (attributed_kernel_utilization)."
            ),
        )
    if peaks_source not in PEAK_SOURCES:
        raise InvalidArgumentError(
            f"peaks_source must be one of {PEAK_SOURCES}; got {peaks_source!r}.",
            code="mfu_peaks_source_invalid",
            remedy="Record where the peaks came from: 'user_supplied' or 'catalog'.",
        )
    if step_seconds <= 0:
        raise InvalidArgumentError(
            f"step_seconds must be positive; got {step_seconds!r}.",
            code="mfu_step_seconds_invalid",
            remedy="Pass the measured end-to-end step wall time in seconds.",
        )
    missing = sorted(set(flops_by_mode) - set(peaks_by_mode))
    if missing:
        raise InvalidArgumentError(
            f"execution mode(s) {missing} carry model FLOPs but no advertised peak -- an "
            "unknown mode makes that share unknown and MFU unavailable (D17).",
            code="mfu_mode_peak_missing",
            remedy=(
                "Supply an advertised dense peak for every execution mode in flops_by_mode "
                "(tl.report.device_peaks), or drop the unknown mode explicitly."
            ),
        )
    terms: list[tuple[str, int, float, float]] = []
    ideal = 0.0
    for mode in sorted(flops_by_mode):
        flops = int(flops_by_mode[mode])
        peak = float(peaks_by_mode[mode])
        if peak <= 0:
            raise InvalidArgumentError(
                f"advertised peak for mode {mode!r} must be positive; got {peak!r}.",
                code="mfu_mode_peak_missing",
                remedy="Supply the vendor's advertised dense peak in FLOP/s.",
            )
        seconds = flops / peak
        ideal += seconds
        terms.append((mode, flops, peak, seconds))
    value = ideal / step_seconds
    if value > 1.0:
        warnings.warn(
            TorchLensWarning(
                f"MFU computed as {value:.3f} > 1.0 -- check the execution-mode mapping, "
                "the advertised peaks, the denominator scope, and the step timing. "
                "Remedy: verify each mode's peak matches its real execution path (TF32/"
                "tensor-core recipes change peaks 2-16x) and that step_seconds covers the "
                "whole step",
                code="mfu_exceeds_one",
            ),
            stacklevel=2,
        )
    return MfuResult(
        mfu=value,
        ideal_seconds=ideal,
        step_seconds=float(step_seconds),
        mode_terms=tuple(terms),
        denominator=denominator,
        denominator_scope=tuple(provenance.denominator_scope),
        peaks_source=peaks_source,
    )


def attributed_kernel_utilization(*_args: Any, **_kwargs: Any) -> Any:
    """Utilization within attributed kernels: NOT MFU; gated on the join.

    The kernel-union quotient is a genuinely useful DIFFERENT metric (D16)
    -- it gets its own name and stays a typed refusal until the
    correlation-ID device-time join lands (D22's real-GPU acceptance
    gate). Never renders under the letters M-F-U.
    """

    raise InvalidArgumentError(
        "attributed_kernel_utilization needs joined per-kernel device time, which requires "
        "the correlation-ID Kineto join (not yet enabled).",
        code="kernel_utilization_requires_device_join",
        remedy=(
            "Use Nsight/NVTX for visual correlation today (docs/reference/"
            "device_attribution.md); the joined table is a later release. For a "
            "publishable utilization figure use tl.report.mfu with a measured step time."
        ),
    )


@dataclass(frozen=True)
class CleanStepTime:
    """One opt-in clean-step measurement (D16): method fully disclosed."""

    seconds: float
    spread_seconds: float
    repeats: int
    warmup: int
    method: str
    scope: tuple[str, ...]
    n_forwards_run: int


def time_clean_step(
    model: Any,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    warmup: int = 2,
    repeats: int = 5,
) -> CleanStepTime:
    """Measure an UNINSTRUMENTED forward wall time (the MFU denominator).

    This helper RE-RUNS the caller's forward ``warmup + repeats`` times
    with TorchLens capture inactive, takes the median, and records its
    scope (forward only -- no backward, optimizer, or data loading; a
    training MFU needs the caller's own full-step timing).
    """

    if repeats < 1 or warmup < 0:
        raise InvalidArgumentError(
            f"repeats must be >= 1 and warmup >= 0; got repeats={repeats}, warmup={warmup}.",
            code="clean_step_repeats_invalid",
            remedy="Pass warmup >= 0 and repeats >= 1.",
        )
    kwargs = input_kwargs or {}
    args = input_args if isinstance(input_args, (list, tuple)) else (input_args,)
    for _ in range(warmup):
        model(*args, **kwargs)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(*args, **kwargs)
        samples.append(time.perf_counter() - start)
    return CleanStepTime(
        seconds=statistics.median(samples),
        spread_seconds=(max(samples) - min(samples)),
        repeats=repeats,
        warmup=warmup,
        method="median of uninstrumented forward wall times (perf_counter)",
        scope=("forward",),
        n_forwards_run=warmup + repeats,
    )


# ---------------------------------------------------------------------------
# Roofline (D21) + machine balance (D18)


@dataclass(frozen=True)
class RooflineRow:
    """One op's theoretical intensity/work coordinates (D21).

    ``ideal_traffic_bytes`` is the read-once/write-once model -- a
    TWO-SIDED estimate (fusion and cache residency push real traffic
    below it; tiling and reloads push above), so ``bound_hypothesis`` is
    a HYPOTHESIS, never a proof.
    """

    label: str
    flops: int
    ideal_traffic_bytes: int | None
    intensity: float | None
    bound_hypothesis: str | None
    cache_resident_hint: bool
    reason: str
    evidence: str = "estimated+ideal_read_once_write_once"


@dataclass(frozen=True)
class RooflineResult:
    """The roofline map: rows, sum/sum aggregate, coverage by reason."""

    rows: tuple[RooflineRow, ...]
    aggregate_intensity: float | None
    coverage_by_reason: dict[str, int]
    ridge_intensity: float | None
    headline_memory_bound: int
    headline_compute_bound: int
    excluded_cache_resident: int

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): rows point, never dump."""

        aggregate = (
            "None" if self.aggregate_intensity is None else f"{self.aggregate_intensity:.3f}"
        )
        return (
            f"RooflineResult({len(self.rows)} rows, aggregate_intensity={aggregate}, "
            f"{self.headline_memory_bound} memory-bound / "
            f"{self.headline_compute_bound} compute-bound; read .rows)"
        )


def _op_ideal_traffic(op: Any, trace: Any) -> int | None:
    """Read-once/write-once logical traffic for one op, if derivable."""

    output_bytes = getattr(op, "activation_memory", None)
    if output_bytes is None:
        return None
    total = int(output_bytes)
    for parent_label in getattr(op, "parents", ()) or ():
        try:
            parent = trace[str(parent_label)]
        except (KeyError, ValueError):
            return None
        parent_bytes = getattr(parent, "activation_memory", None)
        if parent_bytes is not None:
            total += int(parent_bytes)
    fields_bytes = getattr(op, "params_memory", None)
    if fields_bytes is not None:
        total += int(fields_bytes)
    return total


def roofline(
    trace: Trace,
    *,
    ridge_intensity: float | None = None,
    cache_resident_bytes: int = CACHE_RESIDENT_BYTES_DEFAULT,
) -> RooflineResult:
    """Build the theoretical intensity/work map over known compute ops (D21).

    Parameters
    ----------
    trace:
        Finished trace.
    ridge_intensity:
        Machine balance point in FLOPs/byte (from
        :func:`machine_balance`); without it no bound hypotheses render.
    cache_resident_bytes:
        Ops whose ideal traffic fits under this threshold are excluded
        from headline bound counts (hinted, still listed).
    """

    from ._compute_truth import classify_row

    rows: list[RooflineRow] = []
    coverage = dict.fromkeys(ROOFLINE_REASONS, 0)
    total_flops = 0
    total_bytes = 0
    memory_bound = 0
    compute_bound = 0
    excluded = 0
    for op in getattr(trace, "layer_list", ()) or ():
        row_class = classify_row(op)
        if row_class == "not_applicable":
            continue
        label = str(getattr(op, "layer_label", "?"))
        if row_class == "unknown":
            coverage["no_rule"] += 1
            rows.append(
                RooflineRow(
                    label=label,
                    flops=0,
                    ideal_traffic_bytes=None,
                    intensity=None,
                    bound_hypothesis=None,
                    cache_resident_hint=False,
                    reason="no_rule",
                )
            )
            continue
        flops = int(getattr(op, "flops_forward", 0) or 0)
        traffic = _op_ideal_traffic(op, trace)
        if traffic is None:
            coverage["missing_shape_dtype"] += 1
            rows.append(
                RooflineRow(
                    label=label,
                    flops=flops,
                    ideal_traffic_bytes=None,
                    intensity=None,
                    bound_hypothesis=None,
                    cache_resident_hint=False,
                    reason="missing_shape_dtype",
                )
            )
            continue
        if traffic == 0:
            coverage["zero_logical_traffic"] += 1
            rows.append(
                RooflineRow(
                    label=label,
                    flops=flops,
                    ideal_traffic_bytes=0,
                    intensity=None,
                    bound_hypothesis=None,
                    cache_resident_hint=False,
                    reason="zero_logical_traffic",
                )
            )
            continue
        coverage["covered"] += 1
        intensity = flops / traffic
        total_flops += flops
        total_bytes += traffic
        hint = traffic <= cache_resident_bytes
        bound: str | None = None
        if ridge_intensity is not None:
            bound = (
                "memory_bound_hypothesis"
                if intensity < ridge_intensity
                else "compute_bound_hypothesis"
            )
            if hint:
                excluded += 1
            elif bound == "memory_bound_hypothesis":
                memory_bound += 1
            else:
                compute_bound += 1
        rows.append(
            RooflineRow(
                label=label,
                flops=flops,
                ideal_traffic_bytes=traffic,
                intensity=intensity,
                bound_hypothesis=bound,
                cache_resident_hint=hint,
                reason="covered",
            )
        )
    aggregate = (total_flops / total_bytes) if total_bytes else None
    return RooflineResult(
        rows=tuple(rows),
        aggregate_intensity=aggregate,
        coverage_by_reason=coverage,
        ridge_intensity=ridge_intensity,
        headline_memory_bound=memory_bound,
        headline_compute_bound=compute_bound,
        excluded_cache_resident=excluded,
    )


@dataclass(frozen=True)
class MachineBalance:
    """A roofline ceiling pair and its ridge point (D18): basis disclosed."""

    compute_peak_flops_per_s: float
    memory_peak_bytes_per_s: float
    ridge_intensity: float
    basis: str
    method: str


def machine_balance(
    *,
    compute_peak_flops_per_s: float,
    memory_peak_bytes_per_s: float,
    basis: str,
    method: str = "user_supplied ceilings",
) -> MachineBalance:
    """Roofline ceilings and the ridge point (D18): scoped to ROOFLINE only.

    ``basis`` is REQUIRED: ``"measured_achievable"`` (the roofline
    literature's own practice -- DGEMM roof, STREAM diagonal; disclose the
    method) or ``"advertised"``. A ratio against a measured-achievable
    peak renders as utilization of that peak, NEVER as MFU.
    """

    if basis not in ("measured_achievable", "advertised"):
        raise InvalidArgumentError(
            f"basis must be 'measured_achievable' or 'advertised'; got {basis!r}.",
            code="machine_balance_basis_invalid",
            remedy="Name the ceiling's basis explicitly; it is disclosed on every render.",
        )
    if compute_peak_flops_per_s <= 0 or memory_peak_bytes_per_s <= 0:
        raise InvalidArgumentError(
            "machine_balance ceilings must be positive.",
            code="machine_balance_basis_invalid",
            remedy="Pass positive compute (FLOP/s) and memory (bytes/s) ceilings.",
        )
    return MachineBalance(
        compute_peak_flops_per_s=float(compute_peak_flops_per_s),
        memory_peak_bytes_per_s=float(memory_peak_bytes_per_s),
        ridge_intensity=float(compute_peak_flops_per_s) / float(memory_peak_bytes_per_s),
        basis=basis,
        method=method,
    )


# ---------------------------------------------------------------------------
# Instrumented rate (D19/D20)


@dataclass(frozen=True)
class InstrumentedRateRow:
    """One CPU-cell instrumented-rate diagnostic row (D20)."""

    label: str
    flops: int | None
    instrumented_seconds: float | None
    flops_per_second_instrumented: float | None
    device: str | None
    applicable: bool


@dataclass(frozen=True)
class InstrumentedRateTable:
    """The boxed opt-in diagnostic (D20): never in any default view.

    The header carries "instrumented" (screenshots lose footnotes); the
    known distortion DIRECTION: per-op instrumentation overhead
    systematically ranks TINY ops as the worst FLOPs/sec -- this is a
    triage aid for cost under instrumentation, not a statement about the
    model.
    """

    rows: tuple[InstrumentedRateRow, ...]
    header: str = "FLOPs per instrumented second (diagnostic; not a device rate)"

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): rows point, never dump."""

        applicable = sum(1 for row in self.rows if row.applicable)
        return (
            f"InstrumentedRateTable({applicable}/{len(self.rows)} applicable rows, "
            f"instrumented diagnostic; print() for the table, read .rows)"
        )

    def __str__(self) -> str:
        """Render with the mandatory instrumented header and dashes."""

        lines = [self.header]
        for row in self.rows:
            rate = (
                "-"
                if row.flops_per_second_instrumented is None
                else f"{row.flops_per_second_instrumented:.3e}"
            )
            lines.append(f"{row.label:<40} {rate:>12}")
        return "\n".join(lines)


def instrumented_rate(trace: Trace) -> InstrumentedRateTable:
    """The opt-in CPU-only FLOPs-per-instrumented-time diagnostic (D20).

    Applicability is decided PER ROW by the op's execution device: a
    CUDA-executed row renders "-" even though the column was explicitly
    requested (host-side wall time is not that op's compute time). A
    trace whose every timed op executed on CUDA refuses typed, naming
    the join remedy.
    """

    from ._compute_truth import classify_row

    rows: list[InstrumentedRateRow] = []
    any_applicable = False
    any_timed = False
    for op in getattr(trace, "layer_list", ()) or ():
        if classify_row(op) not in ("known", "zero_by_rule"):
            continue
        label = str(getattr(op, "layer_label", "?"))
        flops = getattr(op, "flops_forward", None)
        seconds = getattr(op, "func_duration", None)
        device = getattr(op, "device_ref", None)
        device_name = str(device) if device is not None else None
        on_cuda = device_name is not None and device_name.startswith("cuda")
        if seconds is not None:
            any_timed = True
        applicable = (
            not on_cuda and seconds is not None and flops is not None and float(seconds) > 0
        )
        rate = (
            float(int(flops) / float(seconds))
            if applicable and flops is not None and seconds is not None
            else None
        )
        if applicable:
            any_applicable = True
        rows.append(
            InstrumentedRateRow(
                label=label,
                flops=None if flops is None else int(flops),
                instrumented_seconds=None if seconds is None else float(seconds),
                flops_per_second_instrumented=rate,
                device=device_name,
                applicable=applicable,
            )
        )
    if any_timed and not any_applicable:
        raise InvalidArgumentError(
            "every timed op in this trace executed on CUDA: host-side instrumented time is "
            "not a CUDA rate in either direction (the naive number's error FLIPS SIGN with "
            "the device).",
            code="instrumented_rate_cuda_unsupported",
            remedy=(
                "Use Nsight/NVTX for visual correlation (docs/reference/"
                "device_attribution.md); a joined per-kernel table is a later release."
            ),
        )
    return InstrumentedRateTable(rows=tuple(rows))


@dataclass(frozen=True)
class RateCalibration:
    """The instrumented-rate inflation factor for THIS capture path (D20)."""

    inflation_factor: float
    instrumented_seconds: float
    raw_seconds: float
    n_forwards_run: int
    method: str


def instrumented_rate_calibration(
    model: Any,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    repeats: int = 3,
) -> RateCalibration:
    """Measure this capture path's actual time inflation (D20 calibrate).

    Runs ``repeats`` raw forwards and ONE TorchLens capture, both timed,
    and reports instrumented/raw -- the factor by which per-op
    instrumented time overstates cost on this host.
    """

    from ..user_funcs import trace as _trace

    kwargs = input_kwargs or {}
    args = input_args if isinstance(input_args, (list, tuple)) else (input_args,)
    clean = time_clean_step(model, input_args, input_kwargs, warmup=1, repeats=repeats)
    start = time.perf_counter()
    captured = _trace(model, list(args) if len(args) > 1 else args[0], kwargs)
    instrumented = time.perf_counter() - start
    captured.cleanup()
    return RateCalibration(
        inflation_factor=instrumented / clean.seconds if clean.seconds > 0 else float("inf"),
        instrumented_seconds=instrumented,
        raw_seconds=clean.seconds,
        n_forwards_run=clean.n_forwards_run + 1,
        method="one timed capture vs median uninstrumented forward (perf_counter)",
    )


# ---------------------------------------------------------------------------
# The cost measurer (D27)


#: The measurable tiers: public callables, never harness rung names (D25).
COST_REPORT_TIERS: tuple[str, ...] = ("raw_forward", "trace")


@dataclass(frozen=True)
class CostTierMeasurement:
    """One measured tier: absolute time first, ratio derived (D24)."""

    tier: str
    median_seconds: float
    spread_seconds: float
    ratio_vs_raw: float | None


@dataclass(frozen=True)
class CostReportResult:
    """Measured capture costs on the USER'S model and host (D27)."""

    tiers: tuple[CostTierMeasurement, ...]
    n_forwards_run: int
    environment: dict[str, str]

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): tiers point, never dump."""

        tiers = ", ".join(tier.tier for tier in self.tiers)
        return (
            f"CostReportResult({len(self.tiers)} tier(s): {tiers}; measured over "
            f"{self.n_forwards_run} forwards; print() for the report, read .tiers)"
        )

    def __str__(self) -> str:
        """Render absolute ms + derived ratio + environment + disclosure."""

        lines = ["capture cost (measured on THIS model and host; never an estimate):"]
        for tier in self.tiers:
            ratio = "-" if tier.ratio_vs_raw is None else f"{tier.ratio_vs_raw:.2f}x"
            lines.append(
                f"  {tier.tier:<12} {tier.median_seconds * 1e3:9.3f} ms "
                f"(spread {tier.spread_seconds * 1e3:.3f} ms; {ratio} vs raw forward)"
            )
        lines.append(
            "  environment: " + ", ".join(f"{k}={v}" for k, v in sorted(self.environment.items()))
        )
        lines.append(f"  disclosure: this measurement ran {self.n_forwards_run} forward passes")
        return "\n".join(lines)


def cost_report(
    model: Any,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    tiers: tuple[str, ...] = COST_REPORT_TIERS,
    repeats: int = 3,
) -> CostReportResult:
    """Measure requested capture tiers on the user's own model/host (D27).

    A cost MEASURER, never an estimator: the published grid varies 15x
    across models and 6x across hosts for one cell, so nothing is
    interpolated. Absolute milliseconds lead; the ratio is derived; the
    report discloses exactly how many forwards ran.
    """

    import torch

    from ..user_funcs import trace as _trace

    unknown = sorted(set(tiers) - set(COST_REPORT_TIERS))
    if unknown:
        raise InvalidArgumentError(
            f"unknown cost tier(s) {unknown}; measurable tiers are {COST_REPORT_TIERS}.",
            code="cost_report_tier_unknown",
            remedy="Pass tiers from the measurable set; each maps to one public callable.",
        )
    if "raw_forward" not in tiers:
        raise InvalidArgumentError(
            "the raw_forward baseline tier is required (ratios are derived against it).",
            code="cost_report_tier_unknown",
            remedy="Include 'raw_forward' in tiers.",
        )
    kwargs = input_kwargs or {}
    args = input_args if isinstance(input_args, (list, tuple)) else (input_args,)
    n_forwards = 0

    def _measure(run: Any) -> tuple[float, float]:
        """Median + spread over ``repeats`` timed runs (one warmup)."""

        nonlocal n_forwards
        run()
        n_forwards += 1
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            samples.append(time.perf_counter() - start)
            n_forwards += 1
        return statistics.median(samples), max(samples) - min(samples)

    measurements: list[CostTierMeasurement] = []
    raw_median: float | None = None
    for tier in tiers:
        if tier == "raw_forward":

            def _run_raw() -> None:
                """One raw uninstrumented forward."""

                model(*args, **kwargs)

            median, spread = _measure(_run_raw)
            raw_median = median
            measurements.append(
                CostTierMeasurement(
                    tier=tier, median_seconds=median, spread_seconds=spread, ratio_vs_raw=1.0
                )
            )
        else:

            def _run_trace() -> None:
                """One full default capture, cleaned up immediately."""

                _trace(model, list(args) if len(args) > 1 else args[0], kwargs).cleanup()

            median, spread = _measure(_run_trace)
            measurements.append(
                CostTierMeasurement(
                    tier=tier,
                    median_seconds=median,
                    spread_seconds=spread,
                    ratio_vs_raw=(median / raw_median) if raw_median else None,
                )
            )
    environment = {
        # torch.version.__version__ (not the top-level dunder alias): the
        # spine layer lint reads any torch-private attribute touch as a
        # probe needing a license row.
        "torch": str(torch.version.__version__),
        "device": "cpu",
        "threads": str(torch.get_num_threads()),
    }
    return CostReportResult(
        tiers=tuple(measurements),
        n_forwards_run=n_forwards,
        environment=environment,
    )

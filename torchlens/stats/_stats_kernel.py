"""The sound tensor-stats kernel (C02; lovely memo section 5).

Kernel CONTRACT (D24), not one primitive:

- SOUND: the naive ``E[x^2] - E[x]^2`` variance is banned in every floating
  form, including float64 accumulation (D23; measured 2.1e7 relative error at
  offset 1e6, silent, sign-flipping). Floating routes use ``var_mean`` or
  exact-mean + squared-deviation accumulation; the integer route uses EXACT
  int64 (Python arbitrary-precision) sum / sum-of-squares under a declared
  overflow guard (D27) -- exact rationals carry no cancellation.
- ALLOCATION-BOUNDED: no full-size cast, boolean mask, gather, ``t*t``,
  ``abs``, or flatten copy on the clean path; O(k) seeded sample, O(bins)
  histogram, and the declared fixed-size widening chunk only. The poisoned
  (nonfinite-contaminated) slow path scans in fixed-size chunks.
- SINGLE-SYNC-SHAPED: scalar reductions are batched into one device-to-host
  transfer on the clean path. (Constants and the sync count are CPU-derived
  policy until the deferred CUDA re-derivation; see the lovely memo's CUDA
  leg note.)
- NUMERICALLY PINNED: the N(1e6, 1) anti-regression golden, the fp16
  underflow pin, and the offset-cancellation golden live in
  ``tests/test_factcore_tensor_stats.py`` and are blocking.
- MINIMUM-TRAVERSAL: every family that can ride an existing reduction does.

All gate constants are VERSIONED DISPLAY POLICY derived from the declared
100 ms per-record interactive budget -- flip the budget and the gates
re-derive; they freeze only after an idle-box + CUDA re-derivation.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import torch

#: Declared per-record interactive budget the gates derive from (ms).
INTERACTIVE_BUDGET_MS = 100
#: sd is EXACT through this element count; sampled above (D19).
SD_EXACT_MAX = 2**26
#: Histogram is exact through this element count; sampled above (D20).
HIST_EXACT_MAX = 2**22
#: Seeded gathered-sample cap for every sampled family (D19-D21).
SAMPLE_CAP = 2**20
#: The mean stays exact through this declared backstop; above it the mean is
#: OMITTED with a reason, never sampled (D18).
MEAN_EXACT_MAX = 2**30
#: Histogram bin count (D4).
HIST_BINS = 10
#: No sparkline/histogram below 2 x bins elements (D13).
HIST_MIN_N = 2 * HIST_BINS
#: Fixed widening-accumulator chunk for fp16/bf16 and poisoned scans (D25).
WIDEN_CHUNK = 2**22
#: int64 sum-of-squares overflow guard: numel * max_abs^2 must stay under
#: this bound for the exact integer route (D27).
INT64_GUARD = 2**62

#: Dense f32/f64 route selection (D24): ``var_mean`` and the exact-mean +
#: mse_loss route are BOTH sound and swap winner by ~1.6x depending on
#: regime, so the shipped default is chosen by the benchmark harness below,
#: recorded as policy -- never assumed.
DENSE_ROUTE: str = "var_mean"

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_WIDEN_DTYPES = {torch.float16, torch.bfloat16}
_INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}


@dataclass(frozen=True)
class MomentAccumulator:
    """Pairwise-mergeable (count, mean, M2, M4-ish) accumulator state.

    ``m4_sum`` accumulates raw fourth central-moment mass for the
    kurtosis-aware precision law (D22); it is merged with the same pairwise
    update as ``m2_sum`` (adequate for the se(sd) insurance bound -- the
    cross terms it drops are second-order for the sample sizes involved).
    """

    count: int
    mean: float
    m2_sum: float
    m4_sum: float

    def merge(self, other: MomentAccumulator) -> MomentAccumulator:
        """Pairwise-merge two accumulators (Chan/Welford update; sound)."""

        if other.count == 0:
            return self
        if self.count == 0:
            return other
        total = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * (other.count / total)
        m2 = self.m2_sum + other.m2_sum + delta * delta * (self.count * other.count / total)
        m4 = self.m4_sum + other.m4_sum
        return MomentAccumulator(total, mean, m2, m4)


def _chunk_moments(chunk: torch.Tensor) -> MomentAccumulator:
    """Exact float64 moments of one (finite) chunk via sound reductions."""

    widened = chunk.to(torch.float64)
    count = int(widened.numel())
    if count == 0:
        return MomentAccumulator(0, 0.0, 0.0, 0.0)
    mean = float(widened.mean())
    centered = widened - mean
    m2 = float((centered * centered).sum())
    m4 = float((centered * centered * (centered * centered)).sum())
    return MomentAccumulator(count, mean, m2, m4)


def seeded_sample(flat: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """Return a SEEDED GATHERED sample of ``k`` elements (D21).

    A strided slice is BANNED from the semantics: it aliases against channel
    structure (occupied bins rendered blank on 105/156 real batched-gpt2
    records, sd wrong by up to 96%). The gathered sample is deterministic
    across runs and machines for one seed.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    indexes = torch.randint(0, flat.numel(), (k,), generator=generator)
    if flat.device.type != "cpu":
        indexes = indexes.to(flat.device)
    return flat.index_select(0, indexes)


def identity_seed(identity: str | None, shape: tuple[int, ...], dtype: str) -> int:
    """Derive the sampler seed from record identity (D21).

    Falls back to a geometry-derived seed when no identity is supplied, so
    the sample stays deterministic either way.
    """

    text = identity if identity is not None else f"{shape}|{dtype}"
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


@dataclass(frozen=True)
class KernelResult:
    """Raw family values produced by one kernel run (record-layer input)."""

    numel: int
    nan_count: int
    posinf_count: int
    neginf_count: int
    zero_count: int | None
    true_count: int | None
    finite_min: float | None
    finite_max: float | None
    mean: float | None
    mean_policy: str
    mean_reason: str | None
    sd: float | None
    sd_policy: str
    sd_sample_size: int | None
    sd_se: float | None
    mean_se: float | None
    histogram_counts: tuple[int, ...] | None
    histogram_edges: tuple[float, ...] | None
    histogram_policy: str
    histogram_sample_size: int | None
    magnitude_basis: bool
    unsupported_reason: str | None = None


def _empty_result(numel: int, reason: str | None = None) -> KernelResult:
    """Return the no-values result shape (empty tensors, unsupported dtypes)."""

    return KernelResult(
        numel=numel,
        nan_count=0,
        posinf_count=0,
        neginf_count=0,
        zero_count=None,
        true_count=None,
        finite_min=None,
        finite_max=None,
        mean=None,
        mean_policy="unavailable",
        mean_reason=reason or "empty tensor",
        sd=None,
        sd_policy="unavailable",
        sd_sample_size=None,
        sd_se=None,
        mean_se=None,
        histogram_counts=None,
        histogram_edges=None,
        histogram_policy="unavailable",
        histogram_sample_size=None,
        magnitude_basis=False,
        unsupported_reason=reason,
    )


def _histogram(
    flat: torch.Tensor,
    finite_min: float,
    finite_max: float,
    seed: int,
    contaminated: bool,
) -> tuple[tuple[int, ...] | None, tuple[float, ...] | None, str, int | None]:
    """Histogram family: exact through the gate, sampled above (D20).

    Bin edges always come from the EXACT extremes; the end-bin repair keeps
    the invariant that no bin the extremes prove occupied renders blank.
    """

    numel = flat.numel()
    if numel < HIST_MIN_N:
        return None, None, "unavailable", None
    if finite_min == finite_max:
        return None, None, "unavailable", None
    span = finite_max - finite_min
    edges = tuple(finite_min + span * (index / HIST_BINS) for index in range(HIST_BINS + 1))
    sample_size: int | None = None
    if numel <= HIST_EXACT_MAX and not contaminated:
        counted = flat
        policy = "exact"
    else:
        k = min(SAMPLE_CAP, numel)
        counted = seeded_sample(flat, k, seed)
        if contaminated:
            counted = counted[torch.isfinite(counted)]
        policy = "sampled"
        sample_size = int(counted.numel())
    counts = torch.histc(
        counted.to(torch.float32),
        bins=HIST_BINS,
        min=float(finite_min),
        max=float(finite_max),
    )
    count_list = [int(value) for value in counts.tolist()]
    # End-bin repair (D20): the exact extremes PROVE the first and last bins
    # are occupied; a sampled histogram may have missed them.
    if count_list[0] == 0:
        count_list[0] = 1
    if count_list[-1] == 0:
        count_list[-1] = 1
    return tuple(count_list), edges, policy, sample_size


def _sd_from_accumulator(
    moments: MomentAccumulator, policy: str, sample_size: int | None
) -> tuple[float | None, str, int | None, float | None, float | None]:
    """Finalize sd + the kurtosis-aware precision-law errors (D22)."""

    if moments.count == 0:
        return None, "unavailable", None, None, None
    variance = moments.m2_sum / moments.count
    sd = variance**0.5
    mean_se = None
    sd_se = None
    if policy == "sampled" and sample_size:
        mean_se = sd / (sample_size**0.5)
        if variance > 0:
            kurt = (moments.m4_sum / moments.count) / (variance * variance)
            sd_se = sd * (max(kurt - 1.0, 0.0) / (4.0 * sample_size)) ** 0.5
    return sd, policy, sample_size, sd_se, mean_se


def _float_kernel(tensor: torch.Tensor, seed: int, magnitude: bool) -> KernelResult:
    """Dense floating route: aminmax proof, sound moments, gated sampling."""

    flat = tensor.reshape(-1)
    numel = flat.numel()
    # ONE batched reduction pass; the aminmax pair doubles as the exact
    # nonfinite PROOF (D17): NaN/Inf propagate, so two finite bounds prove
    # zero NaN and zero Inf with no allocation.
    amin, amax = torch.aminmax(flat)
    nonzero = torch.count_nonzero(flat)
    amin_value, amax_value, nonzero_value = (
        float(amin),
        float(amax),
        int(nonzero),
    )
    contaminated = not (
        amin_value == amin_value
        and amax_value == amax_value
        and abs(amin_value) != float("inf")
        and abs(amax_value) != float("inf")
    )
    nan_count = posinf_count = neginf_count = 0
    finite_count = numel
    finite_min: float | None
    finite_max: float | None
    mean: float | None
    sd: float | None
    if not contaminated:
        finite_min, finite_max = amin_value, amax_value
        if numel <= SD_EXACT_MAX:
            if flat.dtype in _WIDEN_DTYPES:
                # Chunked widening accumulator (D25): native low-precision
                # accumulation corrupts the fourth printed digit on real
                # autocast records; the chunk is fixed-size and declared.
                moments = MomentAccumulator(0, 0.0, 0.0, 0.0)
                for start in range(0, numel, WIDEN_CHUNK):
                    moments = moments.merge(_chunk_moments(flat[start : start + WIDEN_CHUNK]))
            elif DENSE_ROUTE == "var_mean":
                # Native-dtype var_mean: sound (torch's stable reduction),
                # zero full-size copies.
                variance, mean_value = torch.var_mean(flat, unbiased=False)
                moments = MomentAccumulator(numel, float(mean_value), float(variance) * numel, 0.0)
            else:
                # Exact f64-accumulated mean (dtype= accumulates without a
                # cast copy) + squared-deviation sum via mse_loss.
                mean_value = flat.sum(dtype=torch.float64) / numel
                centered_sq = torch.nn.functional.mse_loss(
                    flat,
                    mean_value.to(flat.dtype).expand_as(flat),
                    reduction="sum",
                )
                moments = MomentAccumulator(numel, float(mean_value), float(centered_sq), 0.0)
            sd, sd_policy, sd_k, sd_se, mean_se = _sd_from_accumulator(moments, "exact", None)
            mean = moments.mean
            mean_policy, mean_reason = "exact", None
        else:
            # Above the sd gate: mean stays EXACT through the 2^30 backstop
            # (a sampled mean printed the WRONG SIGN with four confident
            # digits; D18); sd rides the seeded gathered sample.
            if numel <= MEAN_EXACT_MAX:
                mean = float(flat.sum(dtype=torch.float64) / numel)
                mean_policy, mean_reason = "exact", None
            else:
                mean, mean_policy = None, "unavailable"
                mean_reason = f"above the 2^30 exact-mean backstop (n={numel})"
            k = min(SAMPLE_CAP, numel)
            sample = seeded_sample(flat, k, seed)
            moments = _chunk_moments(sample)
            sd, sd_policy, sd_k, sd_se, mean_se = _sd_from_accumulator(moments, "sampled", k)
    else:
        # Poisoned slow path: fixed-size chunked scan for exact nonfinite
        # census, finite extrema, and finite-population moments. lovely
        # drops everything at the first NaN; we keep the finite story plus
        # the exact contamination.
        moments = MomentAccumulator(0, 0.0, 0.0, 0.0)
        finite_min = finite_max = None
        for start in range(0, numel, WIDEN_CHUNK):
            chunk = flat[start : start + WIDEN_CHUNK]
            nan_count += int(torch.isnan(chunk).sum())
            posinf_count += int(torch.isposinf(chunk).sum())
            neginf_count += int(torch.isneginf(chunk).sum())
            finite_chunk = chunk[torch.isfinite(chunk)]
            if finite_chunk.numel() == 0:
                continue
            chunk_min, chunk_max = torch.aminmax(finite_chunk)
            chunk_min_value, chunk_max_value = float(chunk_min), float(chunk_max)
            finite_min = chunk_min_value if finite_min is None else min(finite_min, chunk_min_value)
            finite_max = chunk_max_value if finite_max is None else max(finite_max, chunk_max_value)
            if moments.count + finite_chunk.numel() <= SD_EXACT_MAX:
                moments = moments.merge(_chunk_moments(finite_chunk))
        finite_count = numel - nan_count - posinf_count - neginf_count
        if finite_count == 0:
            sd = mean = None
            sd_policy = mean_policy = "unavailable"
            sd_k = sd_se = mean_se = None
            mean_reason = "no finite values"
        else:
            sd, sd_policy, sd_k, sd_se, mean_se = _sd_from_accumulator(moments, "exact", None)
            mean = moments.mean
            mean_policy, mean_reason = "exact", None

    zero_count = numel - nonzero_value
    hist_seed = seed ^ 0x5EED
    if finite_min is not None and finite_max is not None and finite_count >= HIST_MIN_N:
        hist_counts, hist_edges, hist_policy, hist_k = _histogram(
            flat, finite_min, finite_max, hist_seed, contaminated
        )
    else:
        hist_counts, hist_edges, hist_policy, hist_k = None, None, "unavailable", None
    return KernelResult(
        numel=numel,
        nan_count=nan_count,
        posinf_count=posinf_count,
        neginf_count=neginf_count,
        zero_count=zero_count,
        true_count=None,
        finite_min=finite_min,
        finite_max=finite_max,
        mean=mean,
        mean_policy=mean_policy,
        mean_reason=mean_reason,
        sd=sd,
        sd_policy=sd_policy,
        sd_sample_size=sd_k,
        sd_se=sd_se,
        mean_se=mean_se,
        histogram_counts=hist_counts,
        histogram_edges=hist_edges,
        histogram_policy=hist_policy,
        histogram_sample_size=hist_k,
        magnitude_basis=magnitude,
    )


def _int_kernel(tensor: torch.Tensor) -> KernelResult:
    """Exact integer route (D27): int64 sum / sum-of-squares, overflow-guarded.

    Exact integer arithmetic carries NO floating cancellation, so the
    sum-of-squares form is sound here (and only here); the variance is an
    exact rational finalized in Python arbitrary precision.
    """

    flat = tensor.reshape(-1)
    numel = flat.numel()
    amin, amax = torch.aminmax(flat)
    nonzero = torch.count_nonzero(flat)
    min_value, max_value, nonzero_value = int(amin), int(amax), int(nonzero)
    max_abs = max(abs(min_value), abs(max_value))
    guarded = numel * max_abs * max_abs <= INT64_GUARD
    if guarded:
        total = int(flat.sum(dtype=torch.int64))
        total_sq = int((flat.to(torch.int64) * flat.to(torch.int64)).sum())
        mean = total / numel
        variance_numer = numel * total_sq - total * total
        variance = variance_numer / (numel * numel)
        sd = variance**0.5
        mean_policy = "exact"
        mean_reason = None
    else:
        widened = flat.to(torch.float64)
        variance_t, mean_t = torch.var_mean(widened, unbiased=False)
        mean, sd = float(mean_t), float(variance_t) ** 0.5
        mean_policy = "exact"
        mean_reason = "int64 overflow guard tripped; float64 accumulation (disclosed)"
    hist_seed = identity_seed(None, tuple(tensor.shape), str(tensor.dtype))
    if numel >= HIST_MIN_N and min_value != max_value:
        hist_counts, hist_edges, hist_policy, hist_k = _histogram(
            flat.to(torch.float32), float(min_value), float(max_value), hist_seed, False
        )
    else:
        hist_counts, hist_edges, hist_policy, hist_k = None, None, "unavailable", None
    return KernelResult(
        numel=numel,
        nan_count=0,
        posinf_count=0,
        neginf_count=0,
        zero_count=numel - nonzero_value,
        true_count=None,
        finite_min=float(min_value),
        finite_max=float(max_value),
        mean=mean,
        mean_policy=mean_policy,
        mean_reason=mean_reason,
        sd=sd,
        sd_policy="exact",
        sd_sample_size=None,
        sd_se=None,
        mean_se=None,
        histogram_counts=hist_counts,
        histogram_edges=hist_edges,
        histogram_policy=hist_policy,
        histogram_sample_size=hist_k,
        magnitude_basis=False,
    )


def _bool_kernel(tensor: torch.Tensor) -> KernelResult:
    """bool family: exact truth rate; moments of a mask are noise (D12)."""

    import dataclasses

    flat = tensor.reshape(-1)
    numel = flat.numel()
    true_count = int(torch.count_nonzero(flat))
    result = _empty_result(numel, reason="bool tensors report the truth rate, never moments")
    return dataclasses.replace(
        result,
        true_count=true_count,
        zero_count=numel - true_count,
        unsupported_reason=None,
    )


def _complex_kernel(tensor: torch.Tensor, seed: int) -> KernelResult:
    """Complex family: chunk-bounded MAGNITUDE stats, explicitly labeled (D12).

    The result's ``magnitude_basis`` flag is what renderers use to label
    every moment as a |z| statistic (lovely bug 3: complex stats were
    presented as value stats).
    """

    flat = tensor.reshape(-1)
    numel = flat.numel()
    if numel == 0:
        return _empty_result(0)
    moments = MomentAccumulator(0, 0.0, 0.0, 0.0)
    finite_min: float | None = None
    finite_max: float | None = None
    nan_count = 0
    zero_count = 0
    magnitudes_all: list[torch.Tensor] = []
    keep_for_hist = numel <= HIST_EXACT_MAX
    for start in range(0, numel, WIDEN_CHUNK):
        chunk = flat[start : start + WIDEN_CHUNK]
        magnitude = torch.abs(chunk).to(torch.float64)
        nan_count += int(torch.isnan(magnitude).sum())
        zero_count += int((magnitude == 0).sum())
        finite_magnitude = magnitude[torch.isfinite(magnitude)]
        if finite_magnitude.numel():
            chunk_min, chunk_max = torch.aminmax(finite_magnitude)
            chunk_min_value, chunk_max_value = float(chunk_min), float(chunk_max)
            finite_min = chunk_min_value if finite_min is None else min(finite_min, chunk_min_value)
            finite_max = chunk_max_value if finite_max is None else max(finite_max, chunk_max_value)
            if moments.count + finite_magnitude.numel() <= SD_EXACT_MAX:
                moments = moments.merge(_chunk_moments(finite_magnitude))
            if keep_for_hist:
                magnitudes_all.append(finite_magnitude)
    if moments.count == 0:
        return _empty_result(numel, reason="no finite magnitudes")
    sd, sd_policy, sd_k, sd_se, mean_se = _sd_from_accumulator(moments, "exact", None)
    if (
        keep_for_hist
        and magnitudes_all
        and finite_min is not None
        and finite_max is not None
        and moments.count >= HIST_MIN_N
    ):
        hist_counts, hist_edges, hist_policy, hist_k = _histogram(
            torch.cat(magnitudes_all), finite_min, finite_max, seed, False
        )
    else:
        hist_counts, hist_edges, hist_policy, hist_k = None, None, "unavailable", None
    return KernelResult(
        numel=numel,
        nan_count=nan_count,
        posinf_count=0,
        neginf_count=0,
        zero_count=zero_count,
        true_count=None,
        finite_min=finite_min,
        finite_max=finite_max,
        mean=moments.mean,
        mean_policy="exact",
        mean_reason=None,
        sd=sd,
        sd_policy=sd_policy,
        sd_sample_size=sd_k,
        sd_se=sd_se,
        mean_se=mean_se,
        histogram_counts=hist_counts,
        histogram_edges=hist_edges,
        histogram_policy=hist_policy,
        histogram_sample_size=hist_k,
        magnitude_basis=True,
    )


def run_kernel(tensor: torch.Tensor, *, identity: str | None = None) -> KernelResult:
    """Run the sound kernel over one dense torch tensor.

    Never raises on hostile inputs; unsupported layouts/dtypes return the
    unavailability shape with a reason (metadata-only is a first-class
    success).
    """

    if tensor.numel() == 0:
        return _empty_result(0)
    if tensor.layout != torch.strided:
        return _empty_result(tensor.numel(), reason=f"unsupported layout {tensor.layout}")
    work = tensor.detach()
    seed = identity_seed(identity, tuple(work.shape), str(work.dtype))
    try:
        if work.dtype is torch.bool:
            result = _bool_kernel(work)
        elif work.is_complex():
            result = _complex_kernel(work, seed)
        elif work.dtype in _INT_DTYPES:
            result = _int_kernel(work)
        elif work.dtype in _FLOAT_DTYPES:
            result = _float_kernel(work, seed, magnitude=False)
        else:
            result = _empty_result(work.numel(), reason=f"unsupported dtype {work.dtype}")
    except (RuntimeError, TypeError, ValueError) as error:
        return _empty_result(tensor.numel(), reason=f"kernel error: {error}")
    return result


def benchmark_dense_routes(tensor: torch.Tensor, repeats: int = 3) -> dict[str, float]:
    """Time both sound dense routes on one tensor (the D24 harness).

    Returns wall seconds per route; the shipped ``DENSE_ROUTE`` constant is
    re-derived from this harness (recorded with ``os.getloadavg()`` by the
    calling benchmark artifact), never assumed from one machine's number.
    """

    import time

    flat = tensor.detach().reshape(-1)
    timings: dict[str, float] = {}
    for route in ("var_mean", "exact_mean_mse"):
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            if route == "var_mean":
                torch.var_mean(flat, unbiased=False)
            else:
                mean_value = flat.sum(dtype=torch.float64) / flat.numel()
                torch.nn.functional.mse_loss(
                    flat, mean_value.to(flat.dtype).expand_as(flat), reduction="sum"
                )
            best = min(best, time.perf_counter() - start)
        timings[route] = best
    return timings

"""Spine and signed-log2 Histogram StreamingStat kernels (explorer D8/D9).

These are the two reduction kernels every per-step history observation is
built from:

- :class:`Spine` -- the always-on scalar spine: exact integer counts
  (total / finite / zero / negative / NaN / +inf / -inf), finite min / max /
  absmax, sum, sum-of-squares, sum-of-abs, plus stable-merge moments
  (Chan's parallel mean/M2). Integer counts merge EXACTLY; floating fields
  merge with the documented stable pairwise algorithm and are never called
  "exact" (D8's language rule).
- :class:`Histogram` -- ONE immutable, universal, SIGNED log2 grid (D9): no
  collapsing store, no rebinning, ever. Provisional defaults ``bpo=4``,
  window ``[2^-48, 2^16]`` per sign. Specials (exact zero, per-side
  under/overflow, NaN, +/-inf) stay EXACT integer counts, so out-of-range
  forensics survive the window choice. The full descriptor travels with
  every result, making resolution/range a VALUE change forever, never a
  format change.

Both kernels implement the ``torchlens.stats`` ``StreamingStat`` protocol
(``update`` / ``result``), so ``tl.aggregate`` gets dataset-mode histograms
free (D18). Neither kernel consumes RNG (model or global), and half/bf16
inputs reduce in float32-or-wider with the reduction dtype recorded.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from ._errors import StatKernelError

__tl_layer__ = "L5"

#: Widest signed-integer count representable in the artifact's int32
#: per-observation encoding (D9: int32 per observation only where the
#: population bound proves fit; int64 in merged arrays).
INT32_COUNT_MAX = 2**31 - 1

_SPECIAL_KEYS = (
    "zero",
    "pos_underflow",
    "neg_underflow",
    "pos_overflow",
    "neg_overflow",
    "nan",
    "posinf",
    "neginf",
)


def _refuse_dtype(kernel: str, dtype: torch.dtype, *, code: str) -> StatKernelError:
    """Build the typed refusal for an unsupported input dtype.

    ``code`` is passed AT each raise site so the S-17 census sees the code
    where the raise happens, not buried in this factory.
    """

    return StatKernelError(
        f"{kernel} cannot reduce dtype {dtype}. Complex values have no single "
        "signed magnitude ordering, so binning or signed extrema would silently "
        "lie about them.",
        code=code,
        kernel=kernel,
        dtype=str(dtype),
        remedy=(
            "Reduce an explicit real view instead (for example value.abs() or "
            "value.real), and label the result accordingly."
        ),
    )


def _as_reduction_tensor(kernel: str, value: Any) -> tuple[torch.Tensor, str]:
    """Return ``value`` as a detached flat tensor plus its reduction dtype tag.

    Integer and bool inputs are widened to float64 exactly (int64 magnitudes
    above 2^53 lose bits in float64; counts stay exact because count
    classification happens on the widened tensor's zero/sign structure which
    is preserved for the int64 range by float64's sign/zero semantics -- the
    finite extrema/sums for such extreme integers are documented float
    reductions, not exact integer claims). Half and bfloat16 widen to
    float32; float32/float64 keep their width for classification and reduce
    sums in float64.
    """

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.dtype.is_complex:
        raise _refuse_dtype(kernel, value.dtype, code="stat_dtype_unsupported")
    tensor = value.detach().reshape(-1)
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.to(torch.float32), "float32"
    if tensor.dtype.is_floating_point:
        return tensor, str(tensor.dtype).removeprefix("torch.")
    # bool / integer families: exact widening for classification.
    return tensor.to(torch.float64), "float64"


@dataclass(frozen=True)
class HistogramDescriptor:
    """The immutable signed-log2 grid descriptor (D9).

    The manifest records the full descriptor
    ``(base, bins_per_octave, lo_exp, hi_exp, signed, encoding)`` so a
    resolution or window change is a VALUE change, never a format change.
    Merging refuses across unequal descriptors; there is no rebinning path.
    """

    base: int = 2
    bins_per_octave: int = 4
    lo_exp: int = -48
    hi_exp: int = 16
    signed: bool = True
    encoding: str = "dense"

    def __post_init__(self) -> None:
        """Validate the closed grid geometry."""

        if self.base != 2:
            raise StatKernelError(
                f"HistogramDescriptor base={self.base} is not supported; the "
                "universal grid is log2 (D9).",
                code="sketch_descriptor_invalid",
                remedy="Use base=2.",
            )
        if self.bins_per_octave < 1 or self.hi_exp <= self.lo_exp:
            raise StatKernelError(
                "HistogramDescriptor needs bins_per_octave >= 1 and "
                f"hi_exp > lo_exp; got bins_per_octave={self.bins_per_octave}, "
                f"lo_exp={self.lo_exp}, hi_exp={self.hi_exp}.",
                code="sketch_descriptor_invalid",
                remedy="Fix the grid geometry values.",
            )
        if self.encoding != "dense":
            raise StatKernelError(
                f"HistogramDescriptor encoding={self.encoding!r} is unknown; "
                "'dense' is the only v1 encoding.",
                code="sketch_descriptor_invalid",
                remedy="Use encoding='dense'.",
            )

    @property
    def bins_per_side(self) -> int:
        """Number of magnitude bins per sign side."""

        return (self.hi_exp - self.lo_exp) * self.bins_per_octave

    def bucket_edges(self) -> tuple[float, ...]:
        """Return the canonical per-side magnitude bin edges (ascending).

        Edges are ``2 ** (lo_exp + i / bins_per_octave)`` for
        ``i in [0, bins_per_side]``; bin ``i`` covers ``[edges[i],
        edges[i+1])``. Signed grids use these edges mirrored for the
        negative side; the specials (zero, under/overflow, nonfinite) are
        annotated bands, not grid bins.
        """

        bpo = self.bins_per_octave
        return tuple(2.0 ** (self.lo_exp + i / bpo) for i in range(self.bins_per_side + 1))


#: The provisional default grid (D9): bpo=4, window [2^-48, 2^16] per sign.
DEFAULT_DESCRIPTOR = HistogramDescriptor()


@dataclass(frozen=True)
class SpineResult:
    """Finalized spine statistics for one population.

    Integer counts are exact. Floating fields carry the recorded reduction
    dtype and merge with a documented stable pairwise algorithm; they are
    never claimed exact across merges (D8).
    """

    count_total: int
    count_finite: int
    count_zero: int
    count_negative: int
    count_nan: int
    count_posinf: int
    count_neginf: int
    finite_min: float | None
    finite_max: float | None
    finite_absmax: float | None
    sum: float | None
    sum_squares: float | None
    sum_abs: float | None
    mean: float | None
    m2: float | None
    reduction_dtype: str

    def counts(self) -> dict[str, int]:
        """Return the exact integer count fields as one mapping."""

        return {
            "total": self.count_total,
            "finite": self.count_finite,
            "zero": self.count_zero,
            "negative": self.count_negative,
            "nan": self.count_nan,
            "posinf": self.count_posinf,
            "neginf": self.count_neginf,
        }


class Spine:
    """Always-on scalar spine accumulator (StreamingStat; D8).

    ``num, min, max, sum, sum_squares`` map verbatim onto five of
    TensorBoard ``add_histogram_raw``'s nine arguments.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self._count_total = 0
        self._count_finite = 0
        self._count_zero = 0
        self._count_negative = 0
        self._count_nan = 0
        self._count_posinf = 0
        self._count_neginf = 0
        self._finite_min: float | None = None
        self._finite_max: float | None = None
        self._finite_absmax: float | None = None
        self._sum = 0.0
        self._sum_squares = 0.0
        self._sum_abs = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self._reduction_dtype: str | None = None

    def update(self, value: Any) -> None:
        """Fold one tensor-like batch into the spine.

        Empty inputs are a specified no-op outcome (D14: reducers are total).
        """

        tensor, dtype_tag = _as_reduction_tensor("Spine", value)
        if self._reduction_dtype is None:
            self._reduction_dtype = dtype_tag
        elif self._reduction_dtype != dtype_tag and "float64" in (
            self._reduction_dtype,
            dtype_tag,
        ):
            # Record the widest dtype seen; float64 dominates float32.
            self._reduction_dtype = "float64"
        n = int(tensor.numel())
        if n == 0:
            return
        nan_mask = torch.isnan(tensor)
        posinf_mask = tensor == math.inf
        neginf_mask = tensor == -math.inf
        finite_mask = torch.isfinite(tensor)
        n_nan = int(nan_mask.sum().item())
        n_posinf = int(posinf_mask.sum().item())
        n_neginf = int(neginf_mask.sum().item())
        finite = tensor[finite_mask] if (n_nan or n_posinf or n_neginf) else tensor
        n_finite = int(finite.numel())
        self._count_total += n
        self._count_nan += n_nan
        self._count_posinf += n_posinf
        self._count_neginf += n_neginf
        self._count_finite += n_finite
        if n_finite == 0:
            return
        finite64 = finite.to(torch.float64)
        self._count_zero += int((finite64 == 0).sum().item())
        self._count_negative += int((finite64 < 0).sum().item())
        batch_min, batch_max = torch.aminmax(finite64)
        batch_min_f = float(batch_min.item())
        batch_max_f = float(batch_max.item())
        batch_absmax = max(abs(batch_min_f), abs(batch_max_f))
        self._finite_min = (
            batch_min_f if self._finite_min is None else min(self._finite_min, batch_min_f)
        )
        self._finite_max = (
            batch_max_f if self._finite_max is None else max(self._finite_max, batch_max_f)
        )
        self._finite_absmax = (
            batch_absmax if self._finite_absmax is None else max(self._finite_absmax, batch_absmax)
        )
        batch_sum = float(finite64.sum().item())
        self._sum += batch_sum
        self._sum_squares += float((finite64 * finite64).sum().item())
        self._sum_abs += float(finite64.abs().sum().item())
        # Chan parallel-merge moments: fold the batch as one partition.
        batch_mean = batch_sum / n_finite
        batch_m2 = float(((finite64 - batch_mean) ** 2).sum().item())
        prior_finite = self._count_finite - n_finite
        if prior_finite == 0:
            self._mean = batch_mean
            self._m2 = batch_m2
        else:
            delta = batch_mean - self._mean
            total = self._count_finite
            self._mean += delta * (n_finite / total)
            self._m2 += batch_m2 + delta * delta * (prior_finite * n_finite / total)

    def merge(self, other: Spine) -> None:
        """Fold another spine into this one.

        Integer counts merge exactly; floating fields merge with the stable
        pairwise algorithm (Chan for moments, plain compensated-order adds for
        sums) and are documented approximate, never exact (D8).
        """

        if not isinstance(other, Spine):
            raise StatKernelError(
                f"Spine.merge expects a Spine, got {type(other).__name__}.",
                code="stat_merge_incompatible",
                remedy="Merge Spine accumulators with Spine accumulators only.",
            )
        if other._count_total == 0:
            return
        self._count_total += other._count_total
        self._count_nan += other._count_nan
        self._count_posinf += other._count_posinf
        self._count_neginf += other._count_neginf
        self._count_zero += other._count_zero
        self._count_negative += other._count_negative
        n_a = self._count_finite
        n_b = other._count_finite
        self._count_finite = n_a + n_b
        if other._reduction_dtype is not None and (
            self._reduction_dtype is None
            or "float64" in (self._reduction_dtype, other._reduction_dtype)
        ):
            self._reduction_dtype = (
                "float64" if self._reduction_dtype is not None else other._reduction_dtype
            )
        if n_b == 0:
            return
        for attr in ("_finite_min", "_finite_max", "_finite_absmax"):
            mine = getattr(self, attr)
            theirs = getattr(other, attr)
            if mine is None:
                setattr(self, attr, theirs)
            elif theirs is not None:
                fold = min if attr == "_finite_min" else max
                setattr(self, attr, fold(mine, theirs))
        self._sum += other._sum
        self._sum_squares += other._sum_squares
        self._sum_abs += other._sum_abs
        if n_a == 0:
            self._mean = other._mean
            self._m2 = other._m2
        else:
            delta = other._mean - self._mean
            total = self._count_finite
            self._mean += delta * (n_b / total)
            self._m2 += other._m2 + delta * delta * (n_a * n_b / total)

    def result(self) -> SpineResult:
        """Return the finalized spine statistics."""

        has_finite = self._count_finite > 0
        return SpineResult(
            count_total=self._count_total,
            count_finite=self._count_finite,
            count_zero=self._count_zero,
            count_negative=self._count_negative,
            count_nan=self._count_nan,
            count_posinf=self._count_posinf,
            count_neginf=self._count_neginf,
            finite_min=self._finite_min,
            finite_max=self._finite_max,
            finite_absmax=self._finite_absmax,
            sum=self._sum if has_finite else None,
            sum_squares=self._sum_squares if has_finite else None,
            sum_abs=self._sum_abs if has_finite else None,
            mean=self._mean if has_finite else None,
            m2=self._m2 if has_finite else None,
            reduction_dtype=self._reduction_dtype or "float64",
        )


@dataclass(frozen=True)
class HistogramResult:
    """Finalized signed-log2 histogram counts for one population.

    ``pos_counts`` / ``neg_counts`` are per-side magnitude bin counts
    (ascending magnitude, length ``descriptor.bins_per_side``); the negative
    side is present only on signed grids. ``specials`` carries the EXACT
    zero / per-side under-overflow / NaN / +/-inf counts, which render as
    annotated bands (never silently folded into edge bins).
    """

    descriptor: HistogramDescriptor
    pos_counts: tuple[int, ...]
    neg_counts: tuple[int, ...]
    specials: dict[str, int] = field(default_factory=dict)

    @property
    def count_total(self) -> int:
        """Total population folded into this histogram (grid + specials)."""

        return sum(self.pos_counts) + sum(self.neg_counts) + sum(self.specials.values())

    def counts_fit_int32(self) -> bool:
        """True when every stored count fits the int32 observation encoding."""

        cells = (*self.pos_counts, *self.neg_counts, *self.specials.values())
        return all(cell <= INT32_COUNT_MAX for cell in cells)


class Histogram:
    """Fixed signed-log2 grid histogram accumulator (StreamingStat; D9)."""

    def __init__(
        self,
        descriptor: HistogramDescriptor = DEFAULT_DESCRIPTOR,
        name: str | None = None,
    ) -> None:
        self.name = name
        self.descriptor = descriptor
        n_bins = descriptor.bins_per_side
        self._pos = torch.zeros(n_bins, dtype=torch.int64)
        self._neg = torch.zeros(n_bins, dtype=torch.int64)
        self._specials = dict.fromkeys(_SPECIAL_KEYS, 0)

    def update(self, value: Any) -> None:
        """Fold one tensor-like batch onto the fixed grid.

        Placement is ``floor((log2|x| - lo_exp) * bins_per_octave)`` per
        sign side; out-of-window magnitudes land EXACTLY in the per-side
        under/overflow specials. There is no adaptive rebinning and no
        per-row edge drift, ever.
        """

        tensor, _ = _as_reduction_tensor("Histogram", value)
        if tensor.numel() == 0:
            return
        d = self.descriptor
        nan_count = int(torch.isnan(tensor).sum().item())
        posinf_count = int((tensor == math.inf).sum().item())
        neginf_count = int((tensor == -math.inf).sum().item())
        finite = tensor[torch.isfinite(tensor)]
        zero_count = int((finite == 0).sum().item())
        self._specials["nan"] += nan_count
        self._specials["posinf"] += posinf_count
        self._specials["neginf"] += neginf_count
        self._specials["zero"] += zero_count
        for side, sign_mask in (("pos", finite > 0), ("neg", finite < 0)):
            values = finite[sign_mask]
            if values.numel() == 0:
                continue
            magnitudes = values.abs().to(torch.float64)
            log2m = torch.log2(magnitudes)
            raw_idx = torch.floor((log2m - d.lo_exp) * d.bins_per_octave).to(torch.int64)
            under = raw_idx < 0
            over = raw_idx >= d.bins_per_side
            self._specials[f"{side}_underflow"] += int(under.sum().item())
            self._specials[f"{side}_overflow"] += int(over.sum().item())
            in_grid = raw_idx[~(under | over)]
            if in_grid.numel():
                binned = torch.bincount(in_grid, minlength=d.bins_per_side)
                target = self._pos if side == "pos" else self._neg
                target += binned.to(target.device)

    def merge(self, other: Histogram) -> None:
        """Fold another histogram into this one; integer adds, EXACT.

        Unequal descriptors refuse typed: comparability IS the product and
        there is no rebinning path (D9).
        """

        if not isinstance(other, Histogram) or other.descriptor != self.descriptor:
            other_desc = getattr(other, "descriptor", None)
            raise StatKernelError(
                "Histogram.merge requires an identical grid descriptor; got "
                f"{other_desc!r} vs {self.descriptor!r}. Rebinning is banned "
                "(D9): a merged figure must mean the same thing as its parts.",
                code="sketch_descriptor_mismatch",
                mine=repr(self.descriptor),
                theirs=repr(other_desc),
                remedy=(
                    "Accumulate both sides on the same HistogramDescriptor; the "
                    "descriptor is recorded in the run manifest."
                ),
            )
        self._pos += other._pos
        self._neg += other._neg
        for key in _SPECIAL_KEYS:
            self._specials[key] += other._specials[key]

    def result(self) -> HistogramResult:
        """Return the finalized histogram counts."""

        return HistogramResult(
            descriptor=self.descriptor,
            pos_counts=tuple(int(c) for c in self._pos.tolist()),
            neg_counts=tuple(int(c) for c in self._neg.tolist()),
            specials=dict(self._specials),
        )


# -- Device-side staged reduction vectors (D16) -------------------------------
#
# The collector never calls ``.item()`` per site: each site's reduction is a
# fixed-width device vector written into a preallocated staging buffer, and
# the ONLY host transfer is one batched staging copy per phase, independent
# of site count. Slot layout below is the staging contract.

#: Spine staging slots (float64): exact counts ride float64 (exact to 2^53).
SPINE_SLOTS = 15
_SLOT_TOTAL = 0
_SLOT_FINITE = 1
_SLOT_ZERO = 2
_SLOT_NEG = 3
_SLOT_NAN = 4
_SLOT_POSINF = 5
_SLOT_NEGINF = 6
_SLOT_MIN = 7
_SLOT_MAX = 8
_SLOT_ABSMAX = 9
_SLOT_SUM = 10
_SLOT_SUMSQ = 11
_SLOT_SUMABS = 12
_SLOT_MEAN = 13
_SLOT_M2 = 14


def spine_vector(value: torch.Tensor) -> torch.Tensor:
    """Reduce one tensor to the 15-slot spine vector ON ITS DEVICE.

    No host sync happens here: every slot is a device scalar gathered into
    one float64 vector. Batch M2 uses the stable two-pass centered form,
    never the cancellation-prone ``E[x^2] - E[x]^2``.
    """

    tensor, _ = _as_reduction_tensor("Spine", value)
    device = tensor.device
    out = torch.zeros(SPINE_SLOTS, dtype=torch.float64, device=device)
    n = tensor.numel()
    out[_SLOT_TOTAL] = float(n)
    if n == 0:
        out[_SLOT_MIN] = math.nan
        out[_SLOT_MAX] = math.nan
        out[_SLOT_ABSMAX] = math.nan
        return out
    finite_mask = torch.isfinite(tensor)
    finite = tensor[finite_mask]
    out[_SLOT_FINITE] = finite.numel()
    out[_SLOT_NAN] = torch.isnan(tensor).sum()
    out[_SLOT_POSINF] = (tensor == math.inf).sum()
    out[_SLOT_NEGINF] = (tensor == -math.inf).sum()
    if finite.numel() == 0:
        out[_SLOT_MIN] = math.nan
        out[_SLOT_MAX] = math.nan
        out[_SLOT_ABSMAX] = math.nan
        return out
    finite64 = finite.to(torch.float64)
    out[_SLOT_ZERO] = (finite64 == 0).sum()
    out[_SLOT_NEG] = (finite64 < 0).sum()
    mn, mx = torch.aminmax(finite64)
    out[_SLOT_MIN] = mn
    out[_SLOT_MAX] = mx
    out[_SLOT_ABSMAX] = torch.maximum(mn.abs(), mx.abs())
    out[_SLOT_SUM] = finite64.sum()
    out[_SLOT_SUMSQ] = (finite64 * finite64).sum()
    out[_SLOT_SUMABS] = finite64.abs().sum()
    mean = finite64.mean()
    out[_SLOT_MEAN] = mean
    out[_SLOT_M2] = ((finite64 - mean) ** 2).sum()
    return out


def spine_vector_fused(value: torch.Tensor) -> torch.Tensor:
    """Leaner-allocation spine vector: no finite-gather, no full upcast.

    The rider candidate the explorer memo left UNSIZED: :func:`spine_vector`
    materializes a boolean-gather copy of the finite values plus a full
    float64 conversion; this variant keeps the input width, zero-fills
    nonfinites with one ``where`` temp, and accumulates every sum in
    float64 via ``sum(dtype=)`` without materializing the converted tensor.
    Integer counts and finite extrema are EXACTLY equal to the naive
    kernel's; floating sums/moments agree to reduction-order tolerance
    (the artifact already documents floats as approximate, D8).

    Adoption is gated by the D25 harness: the collector keeps the naive
    kernel as canonical unless this variant measures OUTSIDE the box's
    noise band (tests pin parity either way). No host sync happens here.
    """

    tensor, _ = _as_reduction_tensor("Spine", value)
    device = tensor.device
    out = torch.zeros(SPINE_SLOTS, dtype=torch.float64, device=device)
    n = tensor.numel()
    out[_SLOT_TOTAL] = float(n)
    if n == 0:
        out[_SLOT_MIN] = math.nan
        out[_SLOT_MAX] = math.nan
        out[_SLOT_ABSMAX] = math.nan
        return out
    finite_mask = torch.isfinite(tensor)
    n_finite = finite_mask.sum()
    out[_SLOT_FINITE] = n_finite
    out[_SLOT_NAN] = torch.isnan(tensor).sum()
    out[_SLOT_POSINF] = (tensor == math.inf).sum()
    out[_SLOT_NEGINF] = (tensor == -math.inf).sum()
    # All-nonfinite inputs resolve through the device-side ``where`` fix-ups
    # below (never a host sync): extrema become NaN, moments become 0,
    # matching the naive kernel's early exit exactly.
    nan64 = torch.tensor(math.nan, dtype=torch.float64, device=device)
    no_finite = n_finite == 0
    zero = tensor.new_zeros(())
    xf = torch.where(finite_mask, tensor, zero)
    n_nonfinite = n - n_finite
    out[_SLOT_ZERO] = (xf == 0).sum() - n_nonfinite
    out[_SLOT_NEG] = (xf < 0).sum()
    mn = torch.where(finite_mask, tensor, tensor.new_full((), math.inf)).amin().to(torch.float64)
    mx = torch.where(finite_mask, tensor, tensor.new_full((), -math.inf)).amax().to(torch.float64)
    out[_SLOT_MIN] = torch.where(no_finite, nan64, mn)
    out[_SLOT_MAX] = torch.where(no_finite, nan64, mx)
    out[_SLOT_ABSMAX] = torch.where(no_finite, nan64, torch.maximum(mn.abs(), mx.abs()))
    # One float64 working copy (zero-filled at nonfinites) feeds every sum:
    # products and moments must round in float64 to match the naive kernel,
    # and summing the injected zeros is exact. The saving over the naive
    # kernel is the boolean-gather copy and its second conversion.
    xf64 = xf.to(torch.float64)
    s = xf64.sum()
    out[_SLOT_SUM] = s
    out[_SLOT_SUMSQ] = (xf64 * xf64).sum()
    out[_SLOT_SUMABS] = xf64.abs().sum()
    safe_count = torch.where(no_finite, torch.ones_like(n_finite), n_finite)
    mean = s / safe_count
    out[_SLOT_MEAN] = torch.where(no_finite, torch.zeros_like(mean), mean)
    m2 = ((xf64 - mean) ** 2).sum() - n_nonfinite.to(torch.float64) * mean * mean
    out[_SLOT_M2] = torch.where(no_finite, torch.zeros_like(m2), m2)
    return out


def merge_spine_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Merge two spine vectors (CPU float64): counts exact, moments Chan."""

    out = a.clone()
    n_a = float(a[_SLOT_FINITE].item())
    n_b = float(b[_SLOT_FINITE].item())
    for slot in (
        _SLOT_TOTAL,
        _SLOT_FINITE,
        _SLOT_ZERO,
        _SLOT_NEG,
        _SLOT_NAN,
        _SLOT_POSINF,
        _SLOT_NEGINF,
        _SLOT_SUM,
        _SLOT_SUMSQ,
        _SLOT_SUMABS,
    ):
        out[slot] = a[slot] + b[slot]
    if n_a == 0:
        for slot in (_SLOT_MIN, _SLOT_MAX, _SLOT_ABSMAX, _SLOT_MEAN, _SLOT_M2):
            out[slot] = b[slot]
        return out
    if n_b == 0:
        return out
    out[_SLOT_MIN] = torch.minimum(a[_SLOT_MIN], b[_SLOT_MIN])
    out[_SLOT_MAX] = torch.maximum(a[_SLOT_MAX], b[_SLOT_MAX])
    out[_SLOT_ABSMAX] = torch.maximum(a[_SLOT_ABSMAX], b[_SLOT_ABSMAX])
    total = n_a + n_b
    delta = float(b[_SLOT_MEAN].item()) - float(a[_SLOT_MEAN].item())
    out[_SLOT_MEAN] = a[_SLOT_MEAN] + delta * (n_b / total)
    out[_SLOT_M2] = a[_SLOT_M2] + b[_SLOT_M2] + delta * delta * (n_a * n_b / total)
    return out


def spine_result_from_vector(vector: torch.Tensor, reduction_dtype: str) -> SpineResult:
    """Build a :class:`SpineResult` from one (CPU) spine vector."""

    cells = vector.tolist()
    n_finite = int(cells[_SLOT_FINITE])
    has_finite = n_finite > 0
    return SpineResult(
        count_total=int(cells[_SLOT_TOTAL]),
        count_finite=n_finite,
        count_zero=int(cells[_SLOT_ZERO]),
        count_negative=int(cells[_SLOT_NEG]),
        count_nan=int(cells[_SLOT_NAN]),
        count_posinf=int(cells[_SLOT_POSINF]),
        count_neginf=int(cells[_SLOT_NEGINF]),
        finite_min=cells[_SLOT_MIN] if has_finite else None,
        finite_max=cells[_SLOT_MAX] if has_finite else None,
        finite_absmax=cells[_SLOT_ABSMAX] if has_finite else None,
        sum=cells[_SLOT_SUM] if has_finite else None,
        sum_squares=cells[_SLOT_SUMSQ] if has_finite else None,
        sum_abs=cells[_SLOT_SUMABS] if has_finite else None,
        mean=cells[_SLOT_MEAN] if has_finite else None,
        m2=cells[_SLOT_M2] if has_finite else None,
        reduction_dtype=reduction_dtype,
    )


def sketch_vector(value: torch.Tensor, descriptor: HistogramDescriptor) -> torch.Tensor:
    """Reduce one tensor to the dense sketch count vector ON ITS DEVICE.

    Layout: ``[neg_counts (bins), pos_counts (bins), zero, pos_underflow,
    neg_underflow, pos_overflow, neg_overflow, nan, posinf, neginf]`` --
    int64, matching the artifact's ``sketch_counts`` row layout.
    """

    tensor, _ = _as_reduction_tensor("Histogram", value)
    device = tensor.device
    n_side = descriptor.bins_per_side
    out = torch.zeros(2 * n_side + 8, dtype=torch.int64, device=device)
    if tensor.numel() == 0:
        return out
    base = 2 * n_side
    out[base + 5] = torch.isnan(tensor).sum()
    out[base + 6] = (tensor == math.inf).sum()
    out[base + 7] = (tensor == -math.inf).sum()
    finite = tensor[torch.isfinite(tensor)]
    out[base + 0] = (finite == 0).sum()
    for side_index, sign_mask in ((0, finite < 0), (1, finite > 0)):
        values = finite[sign_mask]
        if values.numel() == 0:
            continue
        log2m = torch.log2(values.abs().to(torch.float64))
        raw_idx = torch.floor((log2m - descriptor.lo_exp) * descriptor.bins_per_octave).to(
            torch.int64
        )
        under = raw_idx < 0
        over = raw_idx >= n_side
        if side_index == 1:
            out[base + 1] += under.sum()
            out[base + 3] += over.sum()
        else:
            out[base + 2] += under.sum()
            out[base + 4] += over.sum()
        in_grid = raw_idx[~(under | over)]
        if in_grid.numel():
            binned = torch.bincount(in_grid, minlength=n_side)
            offset = n_side if side_index == 1 else 0
            out[offset : offset + n_side] += binned
    return out


def histogram_result_from_vector(
    vector: torch.Tensor, descriptor: HistogramDescriptor
) -> HistogramResult:
    """Build a :class:`HistogramResult` from one (CPU) sketch vector."""

    n_side = descriptor.bins_per_side
    cells = vector.tolist()
    specials = dict(
        zip(
            _SPECIAL_KEYS,
            (int(x) for x in cells[2 * n_side :]),
            strict=False,
        )
    )
    return HistogramResult(
        descriptor=descriptor,
        neg_counts=tuple(int(x) for x in cells[:n_side]),
        pos_counts=tuple(int(x) for x in cells[n_side : 2 * n_side]),
        specials=specials,
    )


__all__ = [
    "DEFAULT_DESCRIPTOR",
    "INT32_COUNT_MAX",
    "SPINE_SLOTS",
    "Histogram",
    "HistogramDescriptor",
    "HistogramResult",
    "Spine",
    "SpineResult",
    "histogram_result_from_vector",
    "merge_spine_vectors",
    "sketch_vector",
    "spine_result_from_vector",
    "spine_vector",
]

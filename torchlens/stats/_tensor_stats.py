"""The frozen TensorStats record (C02; lovely memo item 2).

One backend-neutral record owning the NUMBERS; every stats line, card,
table cell, and seam adapter is a pure formatter over it (D1: the shipped
cost bug existed BECAUSE the helper was a string function). Per-family
evidence carries exact/sampled/unavailable per D16; counts are exact
integers per D17; ``dim``/``dim_names``/``role``/``relations`` are RESERVED
from day one so per-channel tables, semantic roles, and ``= parent`` marks
land as renderers, never rework (memo section 8 plumbing table).

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

import torch

from ._stats_kernel import KernelResult, run_kernel

#: TensorStats record schema version (bump on any field change).
TENSOR_STATS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FamilyEvidence:
    """Per-family computation evidence (D16).

    ``policy`` is one of ``exact`` / ``sampled`` / ``unavailable``; the
    ``~`` display mark belongs to the sampled family ONLY, and
    ``unavailable`` always carries a reason -- never a silent absence.
    """

    policy: str
    population: int
    sample_size: int | None = None
    reason: str | None = None

    @property
    def sampled(self) -> bool:
        """Whether this family rode the seeded gathered sample."""

        return self.policy == "sampled"


@dataclass(frozen=True)
class TensorStats:
    """Frozen, backend-neutral statistics record for ONE tensor payload.

    Every numeric field is a plain int/float/None (the raw-numbers pin);
    formatters never touch tensors and this record never holds one.
    """

    schema_version: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    nbytes: int | None
    #: Exact nonfinite census (D17): silence is a proof on supported paths.
    nan_count: int
    posinf_count: int
    neginf_count: int
    nonfinite_evidence: FamilyEvidence
    zero_count: int | None
    true_count: int | None
    finite_min: float | None
    finite_max: float | None
    extrema_evidence: FamilyEvidence
    mean: float | None
    mean_evidence: FamilyEvidence
    mean_se: float | None
    sd: float | None
    sd_evidence: FamilyEvidence
    sd_se: float | None
    histogram_counts: tuple[int, ...] | None
    histogram_edges: tuple[float, ...] | None
    histogram_evidence: FamilyEvidence
    #: True when moments describe |z| magnitudes (D12: complex stats are
    #: explicitly magnitude-labeled, never presented as value stats).
    magnitude_basis: bool
    #: Live tensor ``_version`` at computation; a mutated tensor invalidates
    #: the cache and the next render says so (D30).
    tensor_version: int | None
    #: Raw values for tiny tensors (D13: at small n the values ARE the
    #: summary); ``None`` above the smallness bound.
    small_values: tuple[float, ...] | None = None
    #: RESERVED plumbing (deferred != forgotten): per-dim reductions,
    #: graph-proven semantic roles, and distribution relations.
    dim: int | None = None
    dim_names: tuple[str, ...] | None = None
    role: str | None = None
    relations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def nonfinite_count(self) -> int:
        """Total nonfinite elements."""

        return self.nan_count + self.posinf_count + self.neginf_count

    @property
    def finite_count(self) -> int:
        """Elements with finite values."""

        return self.numel - self.nonfinite_count

    @property
    def is_empty(self) -> bool:
        """Whether the payload had zero elements."""

        return self.numel == 0

    @property
    def all_zero(self) -> bool:
        """Whether every element is exactly zero (exact claim)."""

        return self.numel > 0 and self.zero_count == self.numel

    @property
    def constant_value(self) -> float | None:
        """The constant value when every finite element is identical."""

        if (
            self.numel > 1
            and self.nonfinite_count == 0
            and self.finite_min is not None
            and self.finite_min == self.finite_max
        ):
            return self.finite_min
        return None

    @property
    def all_true(self) -> bool:
        """bool family: every element true (exact)."""

        return self.true_count is not None and self.true_count == self.numel and self.numel > 0

    @property
    def all_false(self) -> bool:
        """bool family: every element false (exact)."""

        return self.true_count == 0 and self.numel > 0

    @property
    def no_finite_values(self) -> bool:
        """Whether no element is finite (the fully poisoned case)."""

        return self.numel > 0 and self.finite_count == 0 and self.true_count is None


def _dtype_token(dtype: torch.dtype) -> str:
    """Backend-neutral compact dtype token (f32/i64/bf16/c64/bool)."""

    text = str(dtype).replace("torch.", "")
    return {
        "float32": "f32",
        "float64": "f64",
        "float16": "f16",
        "bfloat16": "bf16",
        "int64": "i64",
        "int32": "i32",
        "int16": "i16",
        "int8": "i8",
        "uint8": "u8",
        "complex64": "c64",
        "complex128": "c128",
        "bool": "bool",
    }.get(text, text)


def _record_from_kernel(
    tensor: torch.Tensor, result: KernelResult, version: int | None
) -> TensorStats:
    """Assemble the frozen record from one kernel run."""

    population = result.numel
    supported = result.unsupported_reason is None
    nonfinite_evidence = FamilyEvidence(
        policy="exact" if supported else "unavailable",
        population=population,
        reason=result.unsupported_reason,
    )
    extrema_evidence = FamilyEvidence(
        policy="exact" if result.finite_min is not None else "unavailable",
        population=population,
        reason=None
        if result.finite_min is not None
        else (result.unsupported_reason or "no finite values"),
    )
    mean_evidence = FamilyEvidence(
        policy=result.mean_policy,
        population=population,
        reason=result.mean_reason,
    )
    sd_evidence = FamilyEvidence(
        policy=result.sd_policy,
        population=population,
        sample_size=result.sd_sample_size,
    )
    histogram_evidence = FamilyEvidence(
        policy=result.histogram_policy,
        population=population,
        sample_size=result.histogram_sample_size,
    )
    try:
        nbytes: int | None = tensor.numel() * tensor.element_size()
    except (RuntimeError, AttributeError):
        nbytes = None
    small_values: tuple[float, ...] | None = None
    if 0 < result.numel <= 16 and not tensor.is_complex():
        try:
            small_values = tuple(float(value) for value in tensor.detach().reshape(-1).tolist())
        except (RuntimeError, TypeError, ValueError):
            small_values = None
    return TensorStats(
        schema_version=TENSOR_STATS_SCHEMA_VERSION,
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=_dtype_token(tensor.dtype),
        device=str(tensor.device),
        numel=result.numel,
        nbytes=nbytes,
        nan_count=result.nan_count,
        posinf_count=result.posinf_count,
        neginf_count=result.neginf_count,
        nonfinite_evidence=nonfinite_evidence,
        zero_count=result.zero_count,
        true_count=result.true_count,
        finite_min=result.finite_min,
        finite_max=result.finite_max,
        extrema_evidence=extrema_evidence,
        mean=result.mean,
        mean_evidence=mean_evidence,
        mean_se=result.mean_se,
        sd=result.sd,
        sd_evidence=sd_evidence,
        sd_se=result.sd_se,
        histogram_counts=result.histogram_counts,
        histogram_edges=result.histogram_edges,
        histogram_evidence=histogram_evidence,
        magnitude_basis=result.magnitude_basis,
        tensor_version=version,
        small_values=small_values,
    )


#: ``_version``-keyed record cache (D30): a repr must never be a compute
#: trigger twice, and a cached line that ignores ``_version`` is a stale
#: number presented as current. Keyed by ``id`` with weakref eviction
#: (never key a dict on tensors: ``weakref.ref`` equality calls tensor
#: ``__eq__``, which is elementwise); the cache never extends payload
#: lifetime.
_STATS_CACHE: dict[int, tuple[weakref.ref[torch.Tensor], Any, TensorStats]] = {}


def _cache_key(tensor: torch.Tensor, version: int | None) -> tuple[Any, ...]:
    """Cache identity: geometry + dtype + device + mutation version."""

    return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device), version)


def _tensor_version(tensor: torch.Tensor) -> int | None:
    """Best-effort ``_version`` read (inference tensors may refuse)."""

    try:
        return int(tensor._version)
    except (RuntimeError, AttributeError):
        return None


def tensor_stats(
    tensor: torch.Tensor,
    *,
    identity: str | None = None,
    role: str | None = None,
) -> TensorStats:
    """Compute (or serve cached) TensorStats for one dense torch tensor.

    Parameters
    ----------
    tensor:
        Payload to summarize. Never mutated; autograd state untouched.
    identity:
        Stable record identity (op label / site key) seeding the gathered
        sampler (D21) so sampled families are deterministic per record.
    role:
        Graph-proven semantic role passthrough (``embedding_index`` etc.);
        the kernel never invents one.

    Returns
    -------
    TensorStats
        Frozen record; cached per tensor keyed on geometry + ``_version``.
    """

    version = _tensor_version(tensor)
    key = _cache_key(tensor, version)
    cache_id = id(tensor)
    cached = _STATS_CACHE.get(cache_id)
    if cached is not None and cached[0]() is tensor and cached[1] == key and cached[2].role == role:
        return cached[2]
    record = _record_from_kernel(tensor, run_kernel(tensor, identity=identity), version)
    if role is not None:
        record = _with_role(record, role)

    def _evict(_reference: weakref.ref[torch.Tensor], _id: int = cache_id) -> None:
        """Drop the cache row when the tensor is collected."""

        _STATS_CACHE.pop(_id, None)

    try:
        reference = weakref.ref(tensor, _evict)
    except TypeError:
        return record
    _STATS_CACHE[cache_id] = (reference, key, record)
    return record


def _with_role(record: TensorStats, role: str) -> TensorStats:
    """Return the record with the graph-proven role attached."""

    import dataclasses

    return dataclasses.replace(record, role=role)

"""Echo's stats rungs on the shared C02 tensor-stats kernel (snoop D4).

Four rungs, shaped by two measurements: the shipped whole-tensor stats
helper costs up to ~800 ms PER LINE at an ordinary training activation
(disqualified from the live path), and a sampled statistic CANNOT make a
finiteness claim (a planted NaN in 12.6M elements is invisible to an 8k
subsample -- measured false negative).

- ``off`` (default): metadata only; zero value reads, zero device syncs.
- ``reuse``: only facts another armed feature already paid for
  (``track_nonfinite`` synchronous verdicts; the ``raise_on_nan`` tripwire's
  crash evidence). Exact where present, field ABSENT where not -- never 0%.
- ``sampled``: moments from a bounded seeded-sample element budget, riding
  the kernel's ``seeded_sample`` gather; ``~``-marked with ``sampled=k/N``
  evidence; NO finiteness claim, ever.
- ``exact``: the full C02 kernel record, refused typed above a documented
  numel budget with the remedy naming ``sampled``.

This lane never forks the kernel; the kernel has four consumers whether or
not echo ships (lovely repr, echo, the stats-helper fix, the tripwire).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._errors import EchoStatsError
from ._event import NarrationStats

if TYPE_CHECKING:
    import torch

__tl_layer__ = "L5"

#: Documented element budget for the sampled rung's seeded gather. The
#: sampled kernel is roughly constant in tensor size because it strides to a
#: fixed budget first -- the only property that makes per-line stats viable.
SAMPLED_BUDGET = 4096

#: Documented numel ceiling for the live ``exact`` rung. Full exact stats
#: measured 23 us - 329 ms scaling with bytes; above this budget a narrated
#: line would stall the forward, so the rung refuses typed instead.
EXACT_NUMEL_BUDGET = 2**24

#: Runtime store attribute written by ``track_nonfinite`` capture recording
#: (``data_classes/_nonfinite.py``); the reuse rung READS it, never scans.
_NONFINITE_STORE_ATTR = "_nonfinite_capture"


def sampled_stats(tensor: torch.Tensor, *, identity: str | None = None) -> NarrationStats | None:
    """Compute the sampled rung's moments from a bounded seeded sample.

    Parameters
    ----------
    tensor:
        Live output tensor at the narration seam. Never mutated, never
        retained; the gather allocates O(budget) only.
    identity:
        Stable label seeding the kernel's deterministic sampler.

    Returns
    -------
    NarrationStats | None
        Sampled-policy record with NO finiteness fields, or ``None`` when
        the tensor has no elements or no orderable value family.
    """

    import torch

    from ..stats._stats_kernel import identity_seed, seeded_sample

    numel = tensor.numel()
    if numel == 0:
        return None
    if tensor.dtype is torch.bool or tensor.is_complex():
        # Bool has no moment family; complex moments are magnitude-labeled
        # kernel territory -- the live sampled rung omits rather than mislabels.
        return NarrationStats(policy="sampled", population=numel, sample_size=0)
    probe = tensor.detach().reshape(-1)
    seed = identity_seed(identity, tuple(tensor.shape), str(tensor.dtype))
    sample = seeded_sample(probe, min(SAMPLED_BUDGET, numel), seed)
    values = sample.to(dtype=torch.float64)
    minimum, maximum = values.aminmax()
    sd, mean = torch.var_mean(values, correction=0 if values.numel() < 2 else 1)
    return NarrationStats(
        policy="sampled",
        population=numel,
        sample_size=int(values.numel()),
        mean=float(mean),
        sd=float(sd.sqrt()),
        minimum=float(minimum),
        maximum=float(maximum),
    )


def exact_stats(tensor: torch.Tensor, *, identity: str | None = None) -> NarrationStats:
    """Compute the exact rung via the shared C02 kernel record.

    Parameters
    ----------
    tensor:
        Live output tensor at the narration seam.
    identity:
        Stable record identity for the kernel cache and sampler.

    Returns
    -------
    NarrationStats
        Exact-policy record carrying the exact nonfinite census.

    Raises
    ------
    EchoStatsError
        When ``tensor.numel()`` exceeds :data:`EXACT_NUMEL_BUDGET` -- the
        live exact rung refuses typed instead of running for minutes; the
        remedy names ``stats="sampled"``.
    """

    numel = tensor.numel()
    if numel > EXACT_NUMEL_BUDGET:
        raise EchoStatsError(
            f"echo stats='exact' refuses a {numel}-element tensor: the live exact rung "
            f"is budgeted at {EXACT_NUMEL_BUDGET} elements per line and a larger scan "
            "would stall the forward pass. Remedy: use stats='sampled' (bounded budget, "
            "no finiteness claim) or compute exact stats post-hoc via narrate().",
            code="echo_stats_numel_budget",
            remedy="use stats='sampled', or compute exact stats post-hoc via narrate()",
            numel=numel,
            budget=EXACT_NUMEL_BUDGET,
        )
    # Leaf import (not the ``torchlens.stats`` facade): the facade pulls in
    # ``_aggregate``, whose function-level ``from .. import trace`` would put
    # this module in one import SCC with the capture entries and defer mypy's
    # resolution of the whole cycle.
    from ..stats._tensor_stats import tensor_stats

    record = tensor_stats(tensor.detach(), identity=identity)
    return NarrationStats(
        policy="exact",
        population=record.numel,
        mean=record.mean,
        sd=record.sd,
        minimum=record.finite_min,
        maximum=record.finite_max,
        nan_count=record.nan_count,
        posinf_count=record.posinf_count,
        neginf_count=record.neginf_count,
    )


def reuse_stats(trace: Any, raw_label: str | None, numel: int) -> NarrationStats | None:
    """Read the reuse rung's already-paid-for facts for one op.

    Parameters
    ----------
    trace:
        Active capture trace (either tier).
    raw_label:
        The op's raw capture label (the ``track_nonfinite`` store key).
    numel:
        Element count for the population field.

    Returns
    -------
    NarrationStats | None
        Reuse-policy record when a synchronous verdict exists; ``None``
        otherwise (deferred device flags are batched at finalize and are
        NEVER printed as though known live).
    """

    if raw_label is None:
        return None
    store = getattr(trace, "__dict__", {}).get(_NONFINITE_STORE_ATTR)
    if not isinstance(store, dict):
        return None
    verdict = store.get("events", {}).get(raw_label)
    if not isinstance(verdict, bool):
        return None
    return NarrationStats(policy="reuse", population=numel, has_nonfinite=verdict)


__all__ = ["EXACT_NUMEL_BUDGET", "SAMPLED_BUDGET", "exact_stats", "reuse_stats", "sampled_stats"]

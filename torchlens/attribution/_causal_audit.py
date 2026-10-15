"""The causal audit of an internal attribution: ablate what it ranks, measure the effect.

A layer attribution claims how much each unit of a hidden site matters for a
target. The audit tests the claim causally with unit ablation (Meyes et al.,
"Ablation Studies in Artificial Neural Networks", 2019), the measurement
behind the causal tracing and mean-ablation audits of mechanistic
interpretability (Meng et al., 2022; Wang et al., 2023), at the hidden site
itself: every unit is ablated alone, and the top-k claimed
units are ablated together beside a bottom-k and a seeded random-k control.
Each measured effect is ``score(original) - score(ablated)``, the same sign as
an activation-times-gradient claim (a unit that supports the target has a
positive claim and a positive effect). Rank agreement is the Spearman
correlation of the claimed and measured per-unit effects.

Every ablation runs through the intervention engine on an intervention-ready
trace (``trace.fork().do(tl.units(label, mask), edit)``, the replay engine of
``torchlens.experiment.site_sweep``), one fork per row or set, and every row
and set carries its edit as ``(tl.units(...), edit)`` pairs, so any point
replays exactly.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._audit_claims import (
    _AUDIT_DOCS_POINTER,
    _AuditPlan,
    _AuditRequest,
    _plan_audit,
)
from torchlens.attribution._core import InputKwargs, _AttributionTarget
from torchlens.attribution._eval_maps import _spearman
from torchlens.attribution._result import AttributionResult, AttributionWarning

# Single-unit ablations past this count warn once and name per_unit=False.
_PER_UNIT_WARN_COUNT = 1024


@dataclass(frozen=True)
class CausalAuditUnit:
    """One unit of the audited site, ablated alone.

    Attributes
    ----------
    unit
        Index of the unit after the batch axis (``(3,)`` is unit 3 of a
        linear output, ``(3, 1, 2)`` one position of a convolution channel).
    claimed_effect
        The claim's score for the unit, summed over the batch.
    measured_effect
        ``score(original) - score(ablated)`` with this unit ablated in every
        example.
    """

    unit: tuple[int, ...]
    claimed_effect: float
    measured_effect: float
    _plan: _AuditPlan = field(repr=False, compare=False)

    @property
    def intervention(self) -> tuple[tuple[Any, Any], ...]:
        """The ``(tl.units(label, mask), edit)`` pair that produced ``measured_effect``.

        Replay it with ``fork = trace.fork(); fork.do(*pair)`` on a trace
        captured with ``capture=tl.options.CaptureOptions(intervention_ready=True)``.
        """

        return self._plan.intervention((self.unit,))

    def __repr__(self) -> str:
        """Compact representation."""

        return f"CausalAuditUnit(unit={self.unit}, claimed_effect={self.claimed_effect:.6g}, measured_effect={self.measured_effect:.6g})"


@dataclass(frozen=True)
class CausalAuditSet:
    """A set of units ablated together: the top-k claim or a control.

    Attributes
    ----------
    kind
        ``"top"`` (largest claimed magnitude), ``"bottom"`` (smallest claimed
        magnitude) or ``"random"`` (a seeded draw).
    units
        The ablated units, indexed after the batch axis.
    claimed_effect
        Sum of the claimed scores of ``units`` (what the claim predicts if
        effects add).
    measured_effect
        ``score(original) - score(ablated)`` with every unit in ``units``
        ablated together.
    draw
        Index of the random draw (``0`` for ``"top"`` and ``"bottom"``).
    """

    kind: str
    units: tuple[tuple[int, ...], ...]
    claimed_effect: float
    measured_effect: float
    draw: int
    _plan: _AuditPlan = field(repr=False, compare=False)

    @property
    def intervention(self) -> tuple[tuple[Any, Any], ...]:
        """The ``(tl.units(label, mask), edit)`` pair that produced ``measured_effect``."""

        return self._plan.intervention(self.units)

    def __repr__(self) -> str:
        """Compact representation."""

        return (
            f"CausalAuditSet(kind={self.kind!r}, k={len(self.units)}, "
            f"claimed_effect={self.claimed_effect:.6g}, measured_effect={self.measured_effect:.6g})"
        )


@dataclass(frozen=True)
class CausalAuditResult:
    """The outcome of a causal audit of one internal attribution.

    Attributes
    ----------
    layer
        Module name of the audited site.
    site
        Op label of the site in the audit's trace (the label the recorded
        edits select).
    original_score
        Target score of the unablated model.
    units
        One :class:`CausalAuditUnit` per unit, in index order (empty with
        ``per_unit=False``).
    sets
        The top-k set, then the controls in ``controls`` order (one set per
        random draw).
    rank_agreement
        Spearman correlation of claimed and measured per-unit effects
        (``None`` with ``per_unit=False`` or when either is constant).
    settings
        The resolved audit policy.
    extra
        Method-specific extras: the audited claim's method name
        (``claim_method``), the number of audited units (``n_units``) and the
        mean measured effect of the random draws (``random_measured_mean``).
    """

    layer: str
    site: str
    original_score: float
    units: tuple[CausalAuditUnit, ...]
    sets: tuple[CausalAuditSet, ...]
    rank_agreement: float | None
    settings: dict[str, Any]
    extra: dict[str, Any]

    def set_of(self, kind: str) -> tuple[CausalAuditSet, ...]:
        """Return the sets of one kind (``"top"``, ``"bottom"`` or ``"random"``)."""

        return tuple(item for item in self.sets if item.kind == kind)

    def __repr__(self) -> str:
        """Compact representation without tensors."""

        top = self.set_of("top")
        return (
            f"CausalAuditResult(layer={self.layer!r}, units={len(self.units)}, "
            f"rank_agreement={self.rank_agreement!r}, "
            f"top_measured_effect={top[0].measured_effect if top else None!r})"
        )


def causal_audit(  # noqa: PLR0913 - public door: keyword-only audit settings
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    claim: AttributionResult,
    target: _AttributionTarget,
    k: int = 5,
    ablation: str | float | Tensor = "zero",
    controls: tuple[str, ...] = ("bottom", "random"),
    random_draws: int = 10,
    per_unit: bool = True,
    seed: int = 0,
) -> CausalAuditResult:
    """Ablate the units an internal attribution ranks and measure each effect.

    The claim is a layer attribution (``layer_attribution``,
    ``layer_integrated_gradients``, ``layer_conductance``), possibly computed
    against a :class:`SiteTarget`. Each unit of its site is ablated alone,
    then the top-k claimed units together, beside a bottom-k and a seeded
    random-k control; every effect is ``score(original) - score(ablated)``
    and is measured through ``trace.fork().do(...)`` on an intervention-ready
    trace (unit ablation after Meyes et al., 2019; rank agreement is
    Spearman's correlation).

    Parameters
    ----------
    model
        The PyTorch module the claim explains; never modified.
    inputs
        Bare tensor or tuple/list of positional model arguments (the inputs
        the claim was computed on).
    input_kwargs
        Optional keyword arguments for ``model``.
    claim
        The internal attribution to audit: an ``AttributionResult`` of a
        layer method whose values have the site's output shape.
    target
        The target the claim was computed for: an integer class index, a
        callable ``output -> scalar``, or a :class:`SiteTarget`.
    k
        Units per ablated set (default 5).
    ablation
        ``"zero"`` (default), ``"mean"`` (each unit's mean over the batch), or
        a number or tensor (the baseline activation, broadcast to the site).
    controls
        Control sets beside the top-k set: ``"bottom"`` and ``"random"``
        (default both).
    random_draws
        Number of seeded random-k sets (default 10).
    per_unit
        Ablate every unit alone for the per-unit table and rank agreement
        (default ``True``); past 1024 units the audit warns once and names
        ``per_unit=False``.
    seed
        Seed of the random draws (a local generator; the global random state
        is restored).

    Returns
    -------
    CausalAuditResult
        Per-unit rows, the top-k and control sets, and the rank agreement;
        every row and set carries its replayable edit.

    Raises
    ------
    AttributionError
        On a claim the audit cannot ground at a hidden site, a claim that does
        not fit its site or target, or invalid settings.
    """

    request = _AuditRequest(
        model=model,
        inputs=inputs,
        input_kwargs=input_kwargs,
        claim=claim,
        target=target,
        k=k,
        ablation=ablation,
        controls=controls,
        random_draws=random_draws,
        seed=seed,
    )
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        plan = _plan_audit(request)
        original = plan.score(plan.trace)
        rows = _unit_rows(plan, original) if per_unit else ()
        sets = _ablated_sets(plan, original, request)
    claimed = torch.tensor([row.claimed_effect for row in rows], dtype=torch.float64)
    measured = torch.tensor([row.measured_effect for row in rows], dtype=torch.float64)
    agreement = _spearman(claimed, measured) if len(rows) > 1 else None
    random_effects = [item.measured_effect for item in sets if item.kind == "random"]
    return CausalAuditResult(
        layer=plan.layer,
        site=plan.label,
        original_score=original,
        units=rows,
        sets=sets,
        rank_agreement=agreement,
        settings={
            "k": k,
            "ablation": plan.ablation_name,
            "controls": tuple(controls),
            "random_draws": random_draws,
            "per_unit": per_unit,
            "seed": seed,
            "target": claim.target_repr,
            "ranking": "claimed magnitude, summed over the batch",
        },
        extra={
            "claim_method": claim.method,
            "n_units": len(plan.units),
            "random_measured_mean": (
                sum(random_effects) / len(random_effects) if random_effects else None
            ),
        },
    )


def _unit_rows(plan: _AuditPlan, original: float) -> tuple[CausalAuditUnit, ...]:
    """Ablate every unit alone, one fork each, and record the effects.

    Parameters
    ----------
    plan
        The audit plan.
    original
        Score of the unablated trace.

    Returns
    -------
    tuple[CausalAuditUnit, ...]
        One row per unit, in index order.
    """

    if len(plan.units) > _PER_UNIT_WARN_COUNT:
        warnings.warn(
            AttributionWarning(
                f"causal_audit is ablating each of the {len(plan.units)} units of "
                f"{plan.layer!r} alone ({len(plan.units)} replays). Remedy: pass "
                "per_unit=False to measure only the top-k and control sets. "
                f"{_AUDIT_DOCS_POINTER}",
                code="attribution_audit_units_many",
            ),
            stacklevel=3,
        )
    return tuple(
        CausalAuditUnit(
            unit, float(plan.claimed[unit]), original - plan.ablated_score((unit,)), plan
        )
        for unit in plan.units
    )


def _ablated_sets(
    plan: _AuditPlan, original: float, request: _AuditRequest
) -> tuple[CausalAuditSet, ...]:
    """Ablate the top-k set and the controls jointly, one fork each.

    Parameters
    ----------
    plan
        The audit plan.
    original
        Score of the unablated trace.
    request
        The caller's settings.

    Returns
    -------
    tuple[CausalAuditSet, ...]
        Top first, then the controls in ``controls`` order.
    """

    order = sorted(
        plan.units, key=lambda unit: (-abs(float(plan.claimed[unit])), plan.units.index(unit))
    )
    chosen: list[tuple[str, int, tuple[tuple[int, ...], ...]]] = [
        ("top", 0, tuple(order[: request.k]))
    ]
    for control in request.controls:
        if control == "bottom":
            chosen.append(("bottom", 0, tuple(order[-request.k :])))
            continue
        generator = torch.Generator().manual_seed(request.seed)
        for draw in range(request.random_draws):
            picks = torch.randperm(len(plan.units), generator=generator)[: request.k].tolist()
            chosen.append(("random", draw, tuple(plan.units[index] for index in sorted(picks))))
    return tuple(
        CausalAuditSet(
            kind=kind,
            units=units,
            claimed_effect=math.fsum(float(plan.claimed[unit]) for unit in units),
            measured_effect=original - plan.ablated_score(units),
            draw=draw,
            _plan=plan,
        )
        for kind, draw, units in chosen
    )

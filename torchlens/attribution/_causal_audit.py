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

The same audit grades the internal claims that span sites: a
direct-logit-attribution table (one component per site; see
``_audit_logits``), a one-backward read table (one site, or each element
of it, per row; see ``_audit_reads``) and an EDGE table of edge attribution
patching scores (one component-graph edge per row, measured by patching the
edge rather than ablating a site; see ``_audit_edges``). Their plan
(``_audit_sites``) keeps every unit with its own site.

Every ablation runs through the intervention engine on an intervention-ready
trace: single-unit rows as the members of one
``torchlens.experiment.site_sweep`` (one candidate per unit, replay engine)
where one edit serves every unit, otherwise ``trace.fork().do(tl.units(label,
mask), edit)``, one fork per row; every set on its own fork. Every row and
set carries its edit as ``(tl.units(...), edit)`` pairs, so any point
replays exactly.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._audit_claims import (
    _AUDIT_DOCS_POINTER,
    _AuditPlan,
    _AuditRequest,
    _call_prefix,
    _plan_audit,
    _setting_error,
    _trace_once,
    _validate_settings,
)
from torchlens.attribution._audit_edges import EDGES_REMEDY, _is_edge_table, _plan_edge_audit
from torchlens.attribution._audit_logits import _is_contribution_table, _plan_logit_audit
from torchlens.attribution._audit_reads import _is_read_table, _plan_read_audit
from torchlens.attribution._audit_sites import _SitesPlan
from torchlens.attribution._core import InputKwargs
from torchlens.attribution._eval_maps import _spearman
from torchlens.attribution._result import AttributionWarning

# Single-unit ablations past this count warn once and name per_unit=False.
_PER_UNIT_WARN_COUNT = 1024
_PER_UNIT_REMEDY = "pass per_unit=False to measure only the top-k and control sets"

_Plan = _AuditPlan | _SitesPlan


@dataclass(frozen=True)
class CausalAuditUnit:
    """One unit of the audited site, ablated alone.

    Attributes
    ----------
    unit
        Index of the unit in the claim. For a layer claim, the position after
        the batch axis (``(3,)`` is unit 3 of a linear output, ``(3, 1, 2)``
        one position of a convolution channel); for a direct-logit table, the
        row (``(2,)``); for a read table, the row, then for an element-grain
        read the position after the batch axis (``(1, 0)``); for an EDGE
        table, the row (``(7,)``).
    claimed_effect
        The claim's score for the unit, summed over the batch (and, for a
        direct-logit row, over its positions and tokens; an EDGE row's score
        as the table reduced it over examples).
    measured_effect
        ``score(original) - score(ablated)`` with this unit ablated in every
        example; for an EDGE row ``M(patched) - M(clean)``, the metric with
        the edge patched minus the clean metric, reduced over examples as the
        table was (the quantity an edge score estimates).
    site
        Op label of the site the unit lives at (the label the edit selects;
        for an EDGE row, the op whose value the patch writes into the
        receiver).
    component
        The claim's name for the unit's component: the layer of a layer
        claim, the row label of a direct-logit table (``"blocks.0.mlp"``),
        the site of a read row, the edge of an EDGE row (``"a0.h1->m2"``).
    """

    unit: tuple[int, ...]
    claimed_effect: float
    measured_effect: float
    site: str = ""
    component: str = ""
    _plan: _Plan | None = field(default=None, repr=False, compare=False)

    @property
    def intervention(self) -> tuple[tuple[Any, Any], ...]:
        """The ``(tl.units(label, mask), edit)`` pair that produced ``measured_effect``.

        Replay it with ``fork = trace.fork(); fork.do(*pair)`` on a trace
        captured with ``capture=tl.options.CaptureOptions(intervention_ready=True)``.
        """

        assert self._plan is not None  # noqa: S101 - every audit row carries its plan
        return self._plan.intervention((self.unit,))

    def __repr__(self) -> str:
        """Compact representation."""

        return (
            f"CausalAuditUnit(unit={self.unit}, component={self.component!r}, "
            f"claimed_effect={self.claimed_effect:.6g}, measured_effect={self.measured_effect:.6g})"
        )


@dataclass(frozen=True)
class CausalAuditSet:
    """A set of units ablated together: the top-k claim or a control.

    Attributes
    ----------
    kind
        ``"top"`` (largest claimed magnitude), ``"bottom"`` (smallest claimed
        magnitude) or ``"random"`` (a seeded draw).
    units
        The ablated units, indexed as :attr:`CausalAuditUnit.unit`.
    claimed_effect
        Sum of the claimed scores of ``units`` (what the claim predicts if
        effects add).
    measured_effect
        ``score(original) - score(ablated)`` with every unit in ``units``
        ablated together (for EDGE rows, ``M(patched) - M(clean)`` with every
        edge patched together).
    draw
        Index of the random draw (``0`` for ``"top"`` and ``"bottom"``).
    """

    kind: str
    units: tuple[tuple[int, ...], ...]
    claimed_effect: float
    measured_effect: float
    draw: int
    _plan: _Plan = field(repr=False, compare=False)

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
        Module name of the audited site of a layer claim (``None`` for a
        direct-logit or read table, whose units span sites).
    site
        Op label of that site in the audit's trace (the label the recorded
        edits select); ``None`` for a direct-logit or read table (each unit
        names its own ``site``).
    original_score
        Target score of the unablated model (for a direct-logit table, its
        logit quantity summed over the batch, positions and tokens).
    units
        One :class:`CausalAuditUnit` per unit, in index order (empty with
        ``per_unit=False``).
    sets
        The top-k set, then the controls in ``controls`` order (one set per
        random draw); for an EDGE table each set's edges are patched together
        in one ``tl.mechinterp.patch_edges`` call, into one receiver or
        several.
    rank_agreement
        Spearman correlation of claimed and measured per-unit effects
        (``None`` with ``per_unit=False`` or when either is constant).
    settings
        The resolved audit policy.
    extra
        Method-specific extras: the claim kind (``claim_kind``: ``"layer"``,
        ``"direct_logit"``, ``"read_table"`` or ``"edge_table"``), the audited claim's method
        name (``claim_method``), the number of audited units (``n_units``) and
        the mean measured effect of the random draws (``random_measured_mean``);
        for a direct-logit table also ``constant_row`` (the claimed constant,
        which has no site and is not ablated), ``native_score`` and
        ``logits_site``; for a read table ``target_id``, ``grain`` and
        ``rows_not_audited`` (``(site, status)`` of rows without a score);
        for an EDGE table ``example_reduction`` (``"mean"`` or ``"sum"``),
        ``logits_site``, ``rows_not_audited`` (``(edge, status)``),
        ``sets_not_measured`` (sets the audit could not patch, each with its
        ``kind``, ``draw``, ``units``, ``edges``, ``claimed_effect`` and the
        ``reason``; empty, since every set is patched in one call) and ``trace`` (the clean intervention-ready trace the edges
        were patched on; the recorded patches replay on its forks).
    """

    layer: str | None
    site: str | None
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
    claim: Any,
    target: Any = None,
    corrupted: Any = None,
    metric: Callable[[Tensor], Tensor] | None = None,
    edges: Any = None,
    k: int = 5,
    ablation: str | float | Tensor = "zero",
    controls: tuple[str, ...] = ("bottom", "random"),
    random_draws: int = 10,
    per_unit: bool = True,
    seed: int = 0,
) -> CausalAuditResult:
    """Ablate the units an internal attribution ranks and measure each effect.

    Four claim kinds are audited. A layer attribution (``layer_attribution``,
    ``layer_integrated_gradients``, ``layer_conductance``), possibly computed
    against a :class:`SiteTarget`, scores the units of one hidden site. A
    ``tl.mechinterp.direct_logit_contributions`` table scores components,
    each at its own site, by their direct share of a logit; each component's
    whole site is ablated and the same logit quantity is read from the
    unembedding head (after the model's logits are checked against the
    table's native logits). A ``tl.attribution.read`` table of
    activation-times-gradient ``ACT`` rows scores sites, or their elements,
    for one seeded target; each is ablated and that target is measured. A
    ``tl.mechinterp.edge_scores`` table of ``EDGE`` rows scores
    component-graph edges; each edge is patched alone with
    ``tl.mechinterp.patch_edges`` (its sender's value from the corrupted
    run) and the metric is measured, ``M(patched) - M(clean)``.

    Each unit is ablated alone, then the top-k claimed units together,
    beside a bottom-k and a seeded random-k control; every effect is
    ``score(original) - score(ablated)`` and is measured through
    ``trace.fork().do(...)`` on an intervention-ready trace (unit ablation
    after Meyes et al., 2019; rank agreement is Spearman's correlation).

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
        layer method whose values have the site's output shape, a
        ``ContributionScores`` table, or a ``ReadTable``.
    target
        The target the claim was computed for: for a layer claim an integer
        class index, a callable ``output -> scalar``, or a
        :class:`SiteTarget`; for a read table the ``tl.attribution.seed(...)``
        it was read for. Left out for a direct-logit table, which fixes its
        own logits, and for an EDGE table, whose target is ``metric``.
    corrupted
        For an EDGE table only: the corrupted inputs it was scored with
        (``edge_scores``' ``corrupted=``), the same shape as ``inputs``.
    metric
        For an EDGE table only: the metric it was scored with
        (``edge_scores``' ``metric=``, ``logits -> values``, one value per
        example or one scalar).
    edges
        For an EDGE table only: the edge names (or ``ComponentEdge``
        objects) of the rows to audit; ``None`` (default) audits every ``ok``
        row.
    k
        Units per ablated set (default 5).
    ablation
        ``"zero"`` (default), ``"mean"`` (each unit's mean over the batch), or
        a number or tensor (the baseline activation, broadcast to the site);
        an EDGE table is patched, not ablated, and takes only the default.
    controls
        Control sets beside the top-k set: ``"bottom"`` and ``"random"``
        (default both).
    random_draws
        Number of seeded random-k sets (default 10).
    per_unit
        Ablate every unit alone for the per-unit table and rank agreement
        (default ``True``); past 1024 units the audit warns once and names
        ``per_unit=False`` (and ``edges=`` for an EDGE table).
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
        On a claim the audit cannot ground at its sites, a claim that does
        not fit its sites, target or logits, or invalid settings.
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
        corrupted=corrupted,
        metric=metric,
        edges=edges,
    )
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        _validate_settings(request)
        plan = _plan_any(request)
        original = plan.score(plan.trace)
        rows = _unit_rows(plan, original) if per_unit else ()
        sets, unmeasured = _ablated_sets(plan, original, request)
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
            "target": plan.target_repr,
            "ranking": "claimed magnitude, summed over the batch",
        },
        extra={
            "claim_kind": plan.claim_kind,
            "claim_method": plan.claim_method,
            "n_units": len(plan.units),
            "random_measured_mean": (
                sum(random_effects) / len(random_effects) if random_effects else None
            ),
            **plan.extra,
            **({"sets_not_measured": unmeasured} if plan.claim_kind == "edge_table" else {}),
        },
    )


def _plan_any(request: _AuditRequest) -> _Plan:
    """Ground the claim by its kind; refuse a set size larger than the units."""

    plan: _Plan
    if _is_edge_table(request.claim):
        plan = _plan_edge_audit(request)
    else:
        _refuse_edge_inputs(request)
        if not (_is_contribution_table(request.claim) or _is_read_table(request.claim)):
            return _plan_audit(request)
        trace = _trace_once(request)
        if _is_contribution_table(request.claim):
            plan = _plan_logit_audit(request, trace)
        else:
            plan = _plan_read_audit(request, trace)
    if request.k > len(plan.units):
        raise _setting_error(
            f"k={request.k} exceeds the {len(plan.units)} audited units of the claim",
            f"{_call_prefix(request)}, k={max(1, len(plan.units))})",
            code="attribution_eval_setting_invalid",
        )
    return plan


def _refuse_edge_inputs(request: _AuditRequest) -> None:
    """Refuse ``corrupted=``, ``metric=`` or ``edges=`` for a claim that is not an EDGE table."""

    given = [
        name
        for name, value in (
            ("corrupted", request.corrupted),
            ("metric", request.metric),
            ("edges", request.edges),
        )
        if value is not None
    ]
    if given:
        target = "" if request.target is None else ", target=target"
        raise _setting_error(
            f"got {', '.join(f'{name}=' for name in given)}, which only an EDGE table from "
            "tl.mechinterp.edge_scores takes",
            f"tl.attribution.causal_audit(model, x, claim=claim{target})",
            code="attribution_eval_setting_invalid",
        )


def _unit_rows(plan: _Plan, original: float) -> tuple[CausalAuditUnit, ...]:
    """Ablate every unit alone (one sweep member or fork each) and record the effects.

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
                f"{plan.layer or 'the claim'!r} alone ({len(plan.units)} replays). Remedy: "
                f"{EDGES_REMEDY if plan.claim_kind == 'edge_table' else _PER_UNIT_REMEDY}. "
                f"{_AUDIT_DOCS_POINTER}",
                code="attribution_audit_units_many",
            ),
            stacklevel=3,
        )
    scores = plan.unit_scores()
    return tuple(
        CausalAuditUnit(
            unit=unit,
            claimed_effect=plan.claimed_of(unit),
            measured_effect=plan.effect(original, scores[unit]),
            site=plan.site_of(unit),
            component=plan.component_of(unit),
            _plan=plan,
        )
        for unit in plan.units
    )


def _ablated_sets(
    plan: _Plan, original: float, request: _AuditRequest
) -> tuple[tuple[CausalAuditSet, ...], tuple[dict[str, Any], ...]]:
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
    tuple[tuple[CausalAuditSet, ...], tuple[dict[str, Any], ...]]
        The measured sets, top first, then the controls in ``controls``
        order; and the sets the plan cannot measure, each with its reason,
        never a sum of unit effects (no plan declines a set today).
    """

    order = sorted(
        plan.units, key=lambda unit: (-abs(plan.claimed_of(unit)), plan.units.index(unit))
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
    measured: list[CausalAuditSet] = []
    unmeasured: list[dict[str, Any]] = []
    for kind, draw, units in chosen:
        claimed = math.fsum(plan.claimed_of(unit) for unit in units)
        reason = plan.unmeasured_reason(units)
        if reason is not None:
            unmeasured.append(
                {
                    "kind": kind,
                    "draw": draw,
                    "units": units,
                    "edges": tuple(plan.component_of(unit) for unit in units),
                    "claimed_effect": claimed,
                    "reason": reason,
                }
            )
            continue
        measured.append(
            CausalAuditSet(
                kind=kind,
                units=units,
                claimed_effect=claimed,
                measured_effect=plan.effect(original, plan.ablated_score(units)),
                draw=draw,
                _plan=plan,
            )
        )
    return tuple(measured), tuple(unmeasured)

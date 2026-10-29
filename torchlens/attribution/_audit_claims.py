"""The causal audit's plan: which claims it grounds, the site, the edit and the score.

``_plan_audit`` validates the settings, traces the model once with an
intervention-ready capture, grounds the claim at its hidden site (one module
call with one output op whose shape the claim's values match) and returns the
plan the audit measures through: the site label, the units, the claimed score
per unit, the edit, and the target score of any trace or fork.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._core import (
    InputKwargs,
    _AttributionTarget,
    _scalarize_output,
    _target_repr,
    _temporarily_eval,
)
from torchlens.attribution._result import AttributionError, AttributionResult
from torchlens.attribution._site import SiteTarget, _select_unit, _target_spelling

_AUDIT_DOCS_POINTER = "See docs/reference/attribution.md, section Auditing internal attributions."
_CONTROLS = ("bottom", "random")


@dataclass(frozen=True)
class _AuditRequest:
    """The caller's audit request, before validation."""

    model: Module
    inputs: Any
    input_kwargs: InputKwargs
    claim: Any
    target: _AttributionTarget
    k: int
    ablation: Any
    controls: Any
    random_draws: int
    seed: int
    corrupted: Any = None
    metric: Any = None
    edges: Any = None


@dataclass(frozen=True)
class _AuditPlan:
    """What the audit measures: the site, its units, the claim, the edit and the score."""

    trace: Any
    layer: str
    label: str
    shape: tuple[int, ...]
    units: tuple[tuple[int, ...], ...]
    claimed: Tensor
    ablation_name: str
    edit: Callable[[], Any]
    score: Callable[[Any], float]
    claim_method: str = ""
    target_repr: str = ""
    claim_kind: str = "layer"
    extra: dict[str, Any] = field(default_factory=dict)

    def claimed_of(self, unit: tuple[int, ...]) -> float:
        """Return the claim's score for one unit, summed over the batch."""

        return float(self.claimed[unit])

    def site_of(self, unit: tuple[int, ...]) -> str:
        """Return the op label every unit of a layer claim lives at."""

        return self.label

    def component_of(self, unit: tuple[int, ...]) -> str:
        """Return the claim's layer, the component every unit belongs to."""

        return self.layer

    def effect(self, original: float, scored: float) -> float:
        """Return ``score(original) - score(ablated)``, the measured effect of an ablation."""

        return original - scored

    def unmeasured_reason(self, units: tuple[tuple[int, ...], ...]) -> str | None:
        """Return why a set cannot be measured (``None``: every set of a site can)."""

        return None

    def unit_scores(self) -> dict[tuple[int, ...], float]:
        """Score every unit ablated alone, in one ``torchlens.experiment.site_sweep``.

        One candidate per unit, ``tl.units(label, mask)``, all with the one
        edit of this site (zero, mean or baseline); a candidate the sweep did
        not complete is measured by its own fork.
        """

        from torchlens.attribution._audit_sites import _swept_scores

        swept = _swept_scores(
            self.trace,
            {unit: self.intervention((unit,))[0][0] for unit in self.units},
            self.edit(),
            self.score,
        )
        return {
            unit: swept[unit] if unit in swept else self.ablated_score((unit,))
            for unit in self.units
        }

    def intervention(self, units: tuple[tuple[int, ...], ...]) -> tuple[tuple[Any, Any], ...]:
        """Return the ``(tl.units(label, mask), edit)`` pair ablating ``units``."""

        from torchlens.selection import units as unit_selection

        return ((unit_selection(self.label, _unit_mask(self.shape, units)), self.edit()),)

    def ablated_score(self, units: tuple[tuple[int, ...], ...]) -> float:
        """Score a fork of the trace with ``units`` ablated in every example."""

        fork = self.trace.fork(name="attribution_causal_audit")
        try:
            with torch.no_grad():
                for selection, edit in self.intervention(units):
                    fork.do(selection, edit)
                return self.score(fork)
        finally:
            fork.cleanup()


def _unit_mask(shape: tuple[int, ...], units: tuple[tuple[int, ...], ...]) -> Tensor:
    """Return the bool mask over the site output selecting ``units`` in every example."""

    mask = torch.zeros(shape, dtype=torch.bool)
    for unit in units:
        index: tuple[slice | int, ...] = (slice(None), *unit)
        mask[index] = True
    return mask


def _remedy_call(target: Any, layer: str) -> str:
    """Return a runnable audit spelling with a freshly computed layer claim."""

    spelling = _target_spelling(target)
    return (
        "tl.attribution.causal_audit(model, x, claim=tl.attribution.layer_attribution("
        f"model, x, {spelling}, layer={layer!r}), {spelling}, k=1)"
    )


def _setting_error(why: str, remedy: str, *, code: str) -> AttributionError:
    """Build the coded refusal for an invalid audit setting."""

    return AttributionError(
        f"causal_audit {why}. Remedy: {remedy}. {_AUDIT_DOCS_POINTER}", code=code
    )


def _is_count(value: Any) -> bool:
    """Return whether ``value`` is a positive int (not a bool)."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _call_prefix(request: _AuditRequest) -> str:
    """Return the caller's audit spelling up to its settings (claim, target, edge inputs)."""

    target = "" if request.target is None else f", {_target_spelling(request.target)}"
    edge_inputs = "".join(
        f", {name}={name}"
        for name, value in (("corrupted", request.corrupted), ("metric", request.metric))
        if value is not None
    )
    return f"tl.attribution.causal_audit(model, x, claim=claim{target}{edge_inputs}"


def _validate_settings(request: _AuditRequest) -> None:
    """Refuse an invalid ``k``, ``ablation``, ``controls`` or ``random_draws``."""

    base = _call_prefix(request)
    code = "attribution_eval_setting_invalid"
    if not _is_count(request.k):
        raise _setting_error(
            f"k must be a positive int; got {request.k!r}", f"{base}, k=1)", code=code
        )
    ablation = request.ablation
    numeric = isinstance(ablation, (int, float, Tensor)) and not isinstance(ablation, bool)
    if not numeric and ablation not in ("zero", "mean"):
        raise _setting_error(
            f"ablation must be 'zero', 'mean', or a number or tensor baseline; got {ablation!r}",
            f"{base}, k=1)" if request.metric is not None else f"{base}, ablation='mean')",
            code=code,
        )
    controls = request.controls
    if isinstance(controls, str) or not all(item in _CONTROLS for item in controls):
        raise _setting_error(
            f"controls must be a tuple of {_CONTROLS}; got {controls!r}",
            f"{base}, controls=('bottom',))",
            code=code,
        )
    if not _is_count(request.random_draws):
        raise _setting_error(
            f"random_draws must be a positive int; got {request.random_draws!r}",
            f"{base}, random_draws=10)",
            code=code,
        )


def _trace_once(request: _AuditRequest) -> Any:
    """Trace the model once, eval mode, no grad, intervention-ready."""

    import torchlens as tl

    inputs = request.inputs
    positional = [inputs] if isinstance(inputs, Tensor) else list(inputs)
    with _temporarily_eval(request.model), torch.no_grad():
        return tl.trace(
            request.model,
            positional,
            input_kwargs=request.input_kwargs or None,
            capture=tl.options.CaptureOptions(intervention_ready=True),
        )


def _single_call_layers(request: _AuditRequest, trace: Any) -> list[str]:
    """Return named modules that ran once with one output op, in definition order."""

    return [
        name
        for name, _ in request.model.named_modules()
        if name and len(_output_labels(trace, name) or ()) == 1
    ]


def _output_labels(trace: Any, layer: str) -> list[str] | None:
    """Return the output op labels of ``layer``'s only call, or ``None`` when it did not run once."""

    if layer not in trace.modules:
        return None
    module = trace.modules[layer]
    if module.num_calls != 1:
        return None
    return list(module.calls[0].output_ops)


def _unsupported_claim(
    request: _AuditRequest, trace: Any, what: str, *, code: str
) -> AttributionError:
    """Build the coded refusal for a claim the audit cannot ground at one hidden site."""

    candidates = _single_call_layers(request, trace)
    example = candidates[-1] if candidates else "fc"
    return AttributionError(
        f"causal_audit cannot audit {what}: it audits a layer attribution at one hidden site "
        "(a module that runs once with one output) whose values have that output's shape, a "
        "torchlens.mechinterp.direct_logit_contributions table, or a tl.attribution.read table. "
        f"Remedy: audit a layer attribution, e.g. {_remedy_call(request.target, example)}. "
        f"{_AUDIT_DOCS_POINTER}",
        code=code,
    )


def _misaligned(request: _AuditRequest, layer: str, why: str, *, code: str) -> AttributionError:
    """Build the coded refusal for a claim that does not fit its site or the target."""

    return AttributionError(
        f"causal_audit: {why}, so its ranking cannot be checked against these ablations. "
        f"Remedy: recompute the claim on these inputs for this target, e.g. "
        f"{_remedy_call(request.target, layer)}. {_AUDIT_DOCS_POINTER}",
        code=code,
    )


def _claim_layer(request: _AuditRequest, trace: Any) -> tuple[str, str]:
    """Return the claim's module name and its site's op label, or refuse."""

    claim = request.claim
    layer = (
        getattr(claim, "extra", {}).get("layer") if isinstance(claim, AttributionResult) else None
    )
    if not isinstance(layer, str) or not isinstance(claim.values, Tensor):
        kind = (
            f"an input-level {claim.method!r} result"
            if isinstance(claim, AttributionResult)
            else f"a {type(claim).__name__}"
        )
        raise _unsupported_claim(request, trace, kind, code="attribution_audit_claim_unsupported")
    labels = _output_labels(trace, layer)
    if labels is None or len(labels) != 1:
        raise _unsupported_claim(
            request,
            trace,
            f"a claim at {layer!r}, which does not run exactly once with one output",
            code="attribution_audit_claim_unsupported",
        )
    return layer, labels[0]


def _target_scorer(request: _AuditRequest, trace: Any, layer: str) -> Callable[[Any], float]:
    """Return ``trace -> target score``; a SiteTarget reads its site on the trace or fork."""

    target = request.target
    if isinstance(target, SiteTarget):
        labels = _output_labels(trace, target.layer) if target.call in (None, 0) else None
        if labels is None or len(labels) != 1:
            raise _misaligned(
                request,
                layer,
                f"the target site {target.layer!r} is not one module call here",
                code="attribution_audit_claim_misaligned",
            )
        site = labels[0]
        return lambda run: float(_select_unit(run[site].out, target).detach())

    def _score(run: Any) -> float:
        """Score the model output of a trace or fork."""

        outs = [run[label].out for label in run.output_layers]
        output = outs[0] if len(outs) == 1 else tuple(outs)
        return float(_scalarize_output(output, target).detach())

    return _score


def _edit_factory(request: _AuditRequest, site: Tensor) -> tuple[str, Callable[[], Any]]:
    """Return the ablation's name and a factory of its edit for ``Trace.do``."""

    from torchlens.intervention import zero_ablate
    from torchlens.intervention.predicates import replace_with

    ablation = request.ablation
    if isinstance(ablation, str) and ablation == "zero":
        return "zero", zero_ablate
    if isinstance(ablation, str):
        value = site.detach().mean(dim=0, keepdim=True).expand_as(site).clone()
        return "mean", lambda: replace_with(value.clone())
    baseline = torch.as_tensor(ablation, dtype=site.dtype, device=site.device)
    try:
        value = torch.broadcast_to(baseline, site.shape).clone()
    except RuntimeError as exc:
        raise _setting_error(
            f"the ablation baseline of shape {tuple(baseline.shape)} does not broadcast to "
            f"the site output of shape {tuple(site.shape)}",
            "tl.attribution.causal_audit(model, x, claim=claim, "
            f"{_target_spelling(request.target)}, ablation=0.0)",
            code="attribution_eval_setting_invalid",
        ) from exc
    return "baseline", lambda: replace_with(value.clone())


def _plan_audit(request: _AuditRequest) -> _AuditPlan:
    """Trace once and ground a layer claim at its site (settings already validated)."""

    trace = _trace_once(request)
    layer, label = _claim_layer(request, trace)
    if request.target is None:
        recorded = str(getattr(request.claim, "target_repr", ""))
        index = recorded.removeprefix("index=")
        if recorded.startswith("index=") and index.lstrip("-").isdigit():
            request = dataclasses.replace(request, target=int(index))
        raise _misaligned(
            request,
            layer,
            "a layer attribution needs the target= it was computed for",
            code="attribution_audit_claim_misaligned",
        )
    site = trace[label].out
    values = request.claim.values
    if tuple(values.shape) != tuple(site.shape):
        raise _misaligned(
            request,
            layer,
            f"the claim's values have shape {tuple(values.shape)} but {layer!r} outputs "
            f"{tuple(site.shape)} on these inputs",
            code="attribution_audit_claim_misaligned",
        )
    if request.claim.target_repr != _target_repr(request.target):
        raise _misaligned(
            request,
            layer,
            f"the claim was computed for target {request.claim.target_repr} but the audit "
            f"scores target {_target_repr(request.target)}",
            code="attribution_audit_claim_misaligned",
        )
    units = tuple(itertools.product(*(range(size) for size in site.shape[1:])))
    if request.k > len(units):
        raise _setting_error(
            f"k={request.k} exceeds the {len(units)} units of {layer!r}",
            "tl.attribution.causal_audit(model, x, claim=claim, "
            f"{_target_spelling(request.target)}, k={len(units)})",
            code="attribution_eval_setting_invalid",
        )
    name, edit = _edit_factory(request, site)
    return _AuditPlan(
        trace=trace,
        layer=layer,
        label=label,
        shape=tuple(int(size) for size in site.shape),
        units=units,
        claimed=values.detach().to(torch.float64).sum(dim=0),
        ablation_name=name,
        edit=edit,
        score=_target_scorer(request, trace, layer),
        claim_method=request.claim.method,
        target_repr=request.claim.target_repr,
    )

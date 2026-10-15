"""Deletion and insertion curves with AOPC.

The deletion curve (most relevant first) removes the most-attributed input
features in steps, replacing them with a baseline, and records the target
score after each step (Samek et al., 2017, "Evaluating the visualization of
what a deep neural network has learned"; Petsiuk et al., 2018, "RISE:
Randomized input sampling for explanation of black-box models"). The
insertion curve starts from the baseline and reveals the most-attributed
features first (Petsiuk et al., 2018). Each curve reports its area under the
curve (trapezoid rule over the fraction of features changed) and its AOPC,
the area over the most-relevant-first perturbation curve of Samek et al.
(2017): the mean, over the steps, of the score change from the unperturbed
start.

Each step is a recorded intervention: ``CurveStep.intervention`` holds
``(tl.units(<input layer>, <baseline mask>), tl.replace_with(<baseline>))``
pairs, so ``trace.fork().do(selection, edit)`` on a replay-ready trace of the
original inputs reproduces the step's score. The scores themselves come from
direct no-grad forward passes (the fast engine ``occlusion_map`` uses); the
replay is the same computation through the intervention engine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._core import (
    InputKwargs,
    _AttributionTarget,
    _call_model,
    _Caller,
    _normalize_model_inputs,
    _PreparedInputs,
    _scalarize_output,
    _target_repr,
    _temporarily_eval,
)
from torchlens.attribution._curve_settings import (
    CurveBaseline,
    CurveFeatures,
    _baseline_for,
    _baseline_repr,
    _leaf_groups,
    _patch_sides,
    _resolve_features,
    _validated_steps,
)
from torchlens.attribution._eval_maps import _DOCS_POINTER, _input_space_maps
from torchlens.attribution._result import AttributionError, AttributionResult

_DEFAULT_STEPS = 20
_DEFAULT_PATCH = 8


@dataclass(frozen=True)
class _CurvePlan:
    """Shared state of one curve; each step reads it.

    Attributes
    ----------
    metric
        ``"deletion"`` or ``"insertion"``.
    prepared
        Normalized inputs of the curve call.
    leaves
        Evaluated (detached) input leaves.
    baselines
        Baseline per leaf, each of its leaf's shape.
    groups
        Per leaf, the global feature id of every non-batch element.
    rank
        ``(N, F)`` position of each feature in its example's ranking (0 is the
        most attributed).
    model
        The evaluated model (traced once, lazily, to name the input sites).
    labels
        Cache of the input-layer labels.
    """

    metric: str
    prepared: _PreparedInputs
    leaves: tuple[Tensor, ...]
    baselines: tuple[Tensor, ...]
    groups: tuple[Tensor, ...]
    rank: Tensor
    model: Module
    labels: dict[str, tuple[str, ...]] = field(default_factory=dict, compare=False)
    imputer: Callable[[Tensor, Tensor, int], Tensor] | None = field(default=None, compare=False)

    def input_labels(self) -> tuple[str, ...]:
        """Return the trace label of each evaluated input, traced once on first use.

        Returns
        -------
        tuple[str, ...]
            Input-layer labels (``"input_1"``, ...), one per leaf.

        Raises
        ------
        AttributionError
            If an evaluated input cannot be matched to an input layer.
        """

        if "labels" not in self.labels:
            self.labels["labels"] = _match_input_labels(self)
        return self.labels["labels"]

    def baseline_masks(self, n_features: int) -> tuple[Tensor, ...]:
        """Return, per leaf, the elements holding the baseline at a step.

        Parameters
        ----------
        n_features
            Features removed (deletion) or revealed (insertion) at the step.

        Returns
        -------
        tuple[Tensor, ...]
            Boolean masks of each leaf's shape.
        """

        changed = self.rank < n_features
        masks = []
        for leaf, group in zip(self.leaves, self.groups, strict=True):
            mask = changed[:, group.reshape(-1)].reshape(leaf.shape)
            masks.append(mask if self.metric == "deletion" else ~mask)
        return tuple(masks)

    def step_leaves(self, n_features: int) -> tuple[Tensor, ...]:
        """Return the perturbed input leaves of a step.

        Parameters
        ----------
        n_features
            Features removed (deletion) or revealed (insertion) at the step.

        Returns
        -------
        tuple[Tensor, ...]
            Perturbed leaves in traversal order.
        """

        masks = self.baseline_masks(n_features)
        return tuple(
            torch.where(mask, baseline, leaf)
            for mask, baseline, leaf in zip(
                masks, self.step_baselines(n_features, masks), self.leaves, strict=True
            )
        )

    def step_baselines(self, n_features: int, masks: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Return, per leaf, the values the step's masked elements take.

        Parameters
        ----------
        n_features
            Features removed (deletion) or revealed (insertion) at the step.
        masks
            The step's baseline masks.

        Returns
        -------
        tuple[Tensor, ...]
            The fixed baselines, or with an imputer (ROAD) the values imputed
            for this step's mask.
        """

        if self.imputer is None:
            return self.baselines
        imputer = self.imputer
        return tuple(
            imputer(leaf, mask, n_features) for leaf, mask in zip(self.leaves, masks, strict=True)
        )


@dataclass(frozen=True)
class CurveStep:
    """One point of a deletion or insertion curve.

    Attributes
    ----------
    index
        Step number, from 0 (nothing changed) to ``n_steps``.
    fraction
        Fraction of features removed (deletion) or revealed (insertion).
    n_features
        Number of features removed or revealed.
    score
        Target score after the step: one value per example for an integer
        target, one value for a callable target.
    """

    index: int
    fraction: float
    n_features: int
    score: Tensor
    _plan: _CurvePlan = field(repr=False, compare=False)

    @property
    def inputs(self) -> Any:
        """The perturbed inputs of this step, shaped like the caller's ``inputs``."""

        return _rebuild_inputs(self._plan.prepared, self._plan.step_leaves(self.n_features))[0]

    @property
    def input_kwargs(self) -> dict[str, Any]:
        """The keyword arguments of this step, perturbed where they were evaluated."""

        return _rebuild_inputs(self._plan.prepared, self._plan.step_leaves(self.n_features))[1]

    @property
    def baseline_mask(self) -> Any:
        """Which input elements hold the baseline at this step (``True``)."""

        masks = self._plan.baseline_masks(self.n_features)
        return masks[0] if len(masks) == 1 else masks

    @property
    def intervention(self) -> tuple[tuple[Any, Any], ...]:
        """The recorded edits of this step, as ``(selection, edit)`` pairs.

        One pair per evaluated input that holds any baseline value:
        ``tl.units(<input layer>, <baseline mask>)`` with
        ``tl.replace_with(<baseline>)``, so the selected elements take the
        baseline and the rest keep the input. Replay on a replay-ready trace
        of the original inputs: ``trace = tl.trace(model, x,
        capture=tl.options.CaptureOptions(intervention_ready=True))``, then
        ``fork = trace.fork()`` and ``fork.do(selection, edit)`` for each
        pair; the fork's output scores ``score``. Empty when the step changes
        nothing. The first access traces the model once to name its input
        layers.
        """

        masks = self._plan.baseline_masks(self.n_features)
        return _step_intervention(
            self._plan, masks, self._plan.step_baselines(self.n_features, masks)
        )


@dataclass(frozen=True)
class CurveResult:
    """A deletion or insertion curve, its area and AOPC.

    Attributes
    ----------
    metric
        ``"deletion"`` or ``"insertion"``.
    fractions
        ``(n_steps + 1,)`` fraction of features changed at each step.
    scores
        ``(n_steps + 1, B)`` target score at each step (``B`` examples for an
        integer target, 1 for a callable target).
    area
        Area under the curve (trapezoid rule over ``fractions``), averaged over
        examples. Lower is better for deletion, higher for insertion.
    aopc
        Mean score change from the unperturbed start over all steps (Samek et
        al., 2017), averaged over examples: ``mean_k(s_0 - s_k)`` for deletion,
        ``mean_k(s_k - s_0)`` for insertion; higher is better for both.
    steps
        One :class:`CurveStep` per point, each carrying its recorded edit.
    settings
        The resolved evaluation policy (baseline, step count, features).
    extra
        Per-example areas and AOPC values and the feature count.
    """

    metric: str
    fractions: Tensor
    scores: Tensor
    area: float
    aopc: float
    steps: tuple[CurveStep, ...]
    settings: dict[str, Any]
    extra: dict[str, Any]

    def __repr__(self) -> str:
        """Compact representation without dumping tensors."""

        return (
            f"CurveResult(metric={self.metric!r}, area={self.area:.6g}, aopc={self.aopc:.6g}, "
            f"steps={len(self.steps)}, settings_keys={sorted(self.settings)!r})"
        )


def _rebuild_inputs(prepared: _PreparedInputs, leaves: tuple[Tensor, ...]) -> tuple[Any, Any]:
    """Place replacement leaves back into the caller's input structure.

    Parameters
    ----------
    prepared
        Normalized inputs.
    leaves
        Replacement leaves in traversal order.

    Returns
    -------
    tuple[Any, Any]
        The inputs (a bare tensor for a bare-tensor call, else the positional
        tuple) and the keyword arguments.
    """

    replacement = {
        id(original): leaf
        for original, leaf in zip(prepared.attributed_leaves, leaves, strict=True)
    }

    def _swap(tree: Any) -> Any:
        """Replace evaluated leaves anywhere in a nested input tree."""

        if isinstance(tree, Tensor):
            return replacement.get(id(tree), tree)
        if isinstance(tree, tuple):
            return tuple(_swap(item) for item in tree)
        if isinstance(tree, list):
            return [_swap(item) for item in tree]
        if isinstance(tree, dict):
            return {key: _swap(item) for key, item in tree.items()}
        return tree

    positional = _swap(prepared.positional)
    kwargs = _swap(prepared.kwargs)
    return (positional[0] if prepared.is_single_tensor else positional), kwargs


def _step_intervention(
    plan: _CurvePlan, masks: tuple[Tensor, ...], baselines: tuple[Tensor, ...]
) -> tuple[tuple[Any, Any], ...]:
    """Build the ``(selection, edit)`` pairs that replay one step.

    Parameters
    ----------
    plan
        The curve plan.
    masks
        Per leaf, the elements holding the baseline at the step.
    baselines
        Per leaf, the values those elements take.

    Returns
    -------
    tuple[tuple[Any, Any], ...]
        ``(tl.units(label, mask), tl.replace_with(baseline))`` per leaf whose
        mask selects anything.
    """

    from torchlens.intervention.predicates import replace_with
    from torchlens.selection import units

    if not any(bool(mask.any()) for mask in masks):
        return ()
    return tuple(
        (units(name, mask), replace_with(baseline.clone()))
        for name, mask, baseline in zip(plan.input_labels(), masks, baselines, strict=True)
        if bool(mask.any())
    )


def _match_input_labels(plan: _CurvePlan) -> tuple[str, ...]:
    """Trace the model once and match each evaluated input to its input layer.

    Parameters
    ----------
    plan
        The curve plan.

    Returns
    -------
    tuple[str, ...]
        One input-layer label per leaf, matched by equal value.

    Raises
    ------
    AttributionError
        If a leaf matches no input layer.
    """

    import torchlens as tl

    inputs, kwargs = _rebuild_inputs(plan.prepared, plan.leaves)
    positional = (inputs,) if plan.prepared.is_single_tensor else tuple(inputs)
    with _temporarily_eval(plan.model), torch.no_grad():
        trace = tl.trace(plan.model, list(positional), input_kwargs=kwargs or None)
    candidates = [(name, trace[name].out) for name in trace.input_layers]
    used: set[str] = set()
    labels: list[str] = []
    for leaf in plan.leaves:
        match = next(
            (
                name
                for name, value in candidates
                if name not in used
                and isinstance(value, Tensor)
                and value.shape == leaf.shape
                and torch.equal(value.detach().to(leaf.device), leaf)
            ),
            None,
        )
        if match is None:
            raise AttributionError(
                f"curve step replay could not find the input layer holding an evaluated "
                f"input of shape {tuple(leaf.shape)} (input layers: {list(trace.input_layers)}). "
                "Remedy: replay the step on its perturbed inputs directly, e.g. "
                f"model(step.inputs). {_DOCS_POINTER}",
                code="attribution_curve_input_unmatched",
            )
        used.add(match)
        labels.append(match)
    return tuple(labels)


def _step_score(output: Any, target: _AttributionTarget) -> Tensor:
    """Score one model output: per example for an int target.

    Parameters
    ----------
    output
        Model output.
    target
        Integer class index or callable scalarizer.

    Returns
    -------
    Tensor
        ``(B,)`` float64 scores.
    """

    if isinstance(target, int) and isinstance(output, Tensor) and output.ndim >= 2:
        selected = output[..., target]
        return selected.reshape(selected.shape[0], -1).sum(dim=1).detach().to(torch.float64)
    return _scalarize_output(output, target).detach().reshape(1).to(torch.float64)


@dataclass(frozen=True)
class _CurveRequest:
    """The keyword arguments of one curve call (see :func:`deletion_curve`)."""

    target: _AttributionTarget
    attribution: AttributionResult | Any
    baseline: CurveBaseline
    n_steps: int
    features: CurveFeatures
    patch_size: int | tuple[int, int]
    imputer: Callable[[Tensor, Tensor, int], Tensor] | None = None
    least_relevant_first: bool = False


def _curve(
    kind: str,
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs,
    request: _CurveRequest,
) -> CurveResult:
    """Shared body of :func:`deletion_curve` and :func:`insertion_curve`.

    Parameters
    ----------
    kind
        ``"deletion"`` or ``"insertion"``.
    model, inputs, input_kwargs
        As in :func:`deletion_curve`.
    request
        The keyword arguments of the public call.

    Returns
    -------
    CurveResult
        The curve with its recorded steps.
    """

    target, attribution, baseline = request.target, request.attribution, request.baseline
    n_steps, features, patch_size = request.n_steps, request.features, request.patch_size
    method = f"{kind}_curve"
    prepared = _normalize_model_inputs(inputs, input_kwargs, _Caller(method, target=target))
    leaves = tuple(leaf.detach() for leaf in prepared.attributed_leaves)
    maps = _input_space_maps(attribution, leaves, method)
    if request.least_relevant_first:
        maps = tuple(-values for values in maps)
    resolved = _resolve_features(features, leaves, method)
    patch = _patch_sides(patch_size, method)
    baselines = tuple(_baseline_for(baseline, leaf, method) for leaf in leaves)
    groups: list[Tensor] = []
    feature_scores: list[Tensor] = []
    offset = 0
    for leaf, values in zip(leaves, maps, strict=True):
        local, count = _leaf_groups(leaf, resolved, patch)
        scores = torch.zeros(leaf.shape[0], count, dtype=torch.float64, device=leaf.device)
        scores.index_add_(1, local.reshape(-1), values.reshape(leaf.shape[0], -1))
        groups.append(local + offset)
        feature_scores.append(scores)
        offset += count
    combined = torch.cat(feature_scores, dim=1)
    order = torch.sort(combined, dim=1, descending=True, stable=True).indices
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(offset, device=order.device).expand_as(order))
    plan = _CurvePlan(
        kind, prepared, leaves, baselines, tuple(groups), rank, model, imputer=request.imputer
    )
    counts = _validated_steps(n_steps, offset, method)

    step_scores: list[Tensor] = []
    with _temporarily_eval(model), torch.no_grad():
        for count in counts:
            output = _call_model(model, prepared, plan.step_leaves(count))
            step_scores.append(_step_score(output, target))
    scores_table = torch.stack(step_scores)
    fractions = torch.tensor([count / offset for count in counts], dtype=torch.float64)
    areas = torch.trapezoid(scores_table, fractions, dim=0)
    start = scores_table[0:1]
    change = (start - scores_table) if kind == "deletion" else (scores_table - start)
    aopcs = change.mean(dim=0)
    steps = tuple(
        CurveStep(index, float(fractions[index]), count, scores_table[index], plan)
        for index, count in enumerate(counts)
    )
    return CurveResult(
        metric=kind,
        fractions=fractions,
        scores=scores_table,
        area=float(areas.mean()),
        aopc=float(aopcs.mean()),
        steps=steps,
        settings={
            "baseline": _baseline_repr(baseline),
            "n_steps": n_steps,
            "features": resolved,
            "patch_size": patch if resolved == "patches" else None,
            "ranking": (
                "least attributed first"
                if request.least_relevant_first
                else "most attributed first"
            ),
            "target": _target_repr(target),
        },
        extra={
            "n_features": offset,
            "area_per_example": tuple(float(value) for value in areas),
            "aopc_per_example": tuple(float(value) for value in aopcs),
            "attribution_method": (
                attribution.method if isinstance(attribution, AttributionResult) else None
            ),
        },
    )


def deletion_curve(  # noqa: PLR0913 - public door: keyword-only evaluation settings
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    attribution: AttributionResult | Any,
    baseline: CurveBaseline = "zeros",
    n_steps: int = _DEFAULT_STEPS,
    features: CurveFeatures = "auto",
    patch_size: int | tuple[int, int] = _DEFAULT_PATCH,
) -> CurveResult:
    """Remove the most-attributed features first and record the target score.

    Follows the most-relevant-first perturbation curve and AOPC of Samek et
    al. (2017) and the deletion metric of Petsiuk et al. (2018). Features are
    ranked per example by their summed attribution, highest first (ties keep
    feature order); at step ``k`` the top ``floor(k * F / n_steps)`` features
    hold the baseline. Every step is a recorded intervention
    (``CurveStep.intervention``) that replays to its score.

    Parameters
    ----------
    model
        PyTorch module the attribution explains.
    inputs
        Bare tensor or tuple/list of positional model arguments; every
        floating-point tensor is perturbed (integer and boolean tensors pass
        through unchanged).
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index (scored per example) or callable
        ``output -> scalar tensor``; the target the attribution explains.
    attribution
        ``AttributionResult`` or tensor over the inputs; ``grad_cam`` maps
        (``(N, 1, H, W)`` at input resolution) broadcast across channels.
    baseline
        What a removed feature becomes: ``"zeros"`` (default), ``"blur"`` (an
        11x11 box blur of the input, ``(N, C, H, W)`` inputs only), a number,
        or a tensor of the input's shape.
    n_steps
        Number of steps after the unperturbed start (default 20).
    features
        Feature grouping: ``"auto"`` (default: ``"pixels"`` for
        ``(N, C, H, W)`` inputs, ``"tokens"`` for ``(N, T, D)`` inputs, else
        ``"elements"``), ``"elements"`` (every scalar), ``"pixels"`` (all
        channels at one position), ``"patches"`` (square patches of
        ``patch_size``) or ``"tokens"`` (all values of one sequence position).
    patch_size
        Patch side, or ``(height, width)``, for ``features="patches"``
        (default 8; edge patches may be smaller).

    Returns
    -------
    CurveResult
        Fractions, scores, area, AOPC and the recorded steps.

    Raises
    ------
    AttributionError
        On a misaligned attribution, an invalid setting, or a ``SiteTarget``
        (curves score the model output).
    """

    return _curve(
        "deletion",
        model,
        inputs,
        input_kwargs,
        _CurveRequest(target, attribution, baseline, n_steps, features, patch_size),
    )


def insertion_curve(  # noqa: PLR0913 - mirrors deletion_curve one-to-one
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    attribution: AttributionResult | Any,
    baseline: CurveBaseline = "zeros",
    n_steps: int = _DEFAULT_STEPS,
    features: CurveFeatures = "auto",
    patch_size: int | tuple[int, int] = _DEFAULT_PATCH,
) -> CurveResult:
    """Reveal the most-attributed features first onto a baseline.

    Follows the insertion metric of Petsiuk et al. (2018); AOPC is reported
    as the mean score gain over the baseline start, the insertion mirror of
    Samek et al. (2017). Step ``k`` shows the top ``floor(k * F / n_steps)``
    features of the input and the baseline everywhere else. Parameters,
    defaults and refusals are those of :func:`deletion_curve`.

    Parameters
    ----------
    model, inputs, input_kwargs, target, attribution, baseline, n_steps, features, patch_size
        As in :func:`deletion_curve`; ``baseline`` is the starting input.

    Returns
    -------
    CurveResult
        Fractions, scores, area, AOPC and the recorded steps.
    """

    return _curve(
        "insertion",
        model,
        inputs,
        input_kwargs,
        _CurveRequest(target, attribution, baseline, n_steps, features, patch_size),
    )

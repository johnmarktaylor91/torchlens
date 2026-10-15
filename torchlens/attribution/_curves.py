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

Each step is a recorded intervention: ``CurveStep.intervention`` is the
``tl.when(...)`` spec that replaces the evaluated input with that step's
perturbed input, so ``tl.trace(model, inputs, intervene=step.intervention)``
replays the step and reproduces its score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
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
from torchlens.attribution._eval_maps import _DOCS_POINTER, _input_space_maps
from torchlens.attribution._result import AttributionError, AttributionResult

CurveFeatures: TypeAlias = Literal["auto", "elements", "pixels", "patches", "tokens"]
CurveBaseline: TypeAlias = Literal["zeros", "blur"] | float | Tensor

_DEFAULT_STEPS = 20
_DEFAULT_PATCH = 8
_BLUR_KERNEL = 11


@dataclass(frozen=True)
class _CurvePlan:
    """Shared state of one curve; each step reads it.

    Attributes
    ----------
    kind
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
    """

    kind: str
    prepared: _PreparedInputs
    leaves: tuple[Tensor, ...]
    baselines: tuple[Tensor, ...]
    groups: tuple[Tensor, ...]
    rank: Tensor

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
            masks.append(mask if self.kind == "deletion" else ~mask)
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

        return tuple(
            torch.where(mask, baseline, leaf)
            for mask, baseline, leaf in zip(
                self.baseline_masks(n_features), self.baselines, self.leaves, strict=True
            )
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
    def intervention(self) -> Any:
        """The recorded edit of this step, as an intervention spec.

        ``tl.trace(model, inputs, intervene=step.intervention)`` replays the
        step: it replaces each evaluated input with this step's perturbed
        value, so the replayed output's target score equals ``score``.
        """

        return _step_intervention(self._plan, self._plan.step_leaves(self.n_features))


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


def _step_intervention(plan: _CurvePlan, leaves: tuple[Tensor, ...]) -> Any:
    """Build the intervention spec that replays one step.

    Parameters
    ----------
    plan
        The curve plan.
    leaves
        The step's perturbed leaves.

    Returns
    -------
    Any
        A ``tl.when(...)`` intervention spec.
    """

    from torchlens.intervention.predicates import replace_with
    from torchlens.intervention.selectors import input_at
    from torchlens.intervention.spec import InterventionRule, InterventionSpec

    rules = []
    for path, leaf in zip(_leaf_paths(plan.prepared), leaves, strict=True):
        rules.append(InterventionRule(input_at(*path), replace_with(leaf.clone())))
    return InterventionSpec(rules=tuple(rules))


def _leaf_paths(prepared: _PreparedInputs) -> list[tuple[Any, ...]]:
    """Return the input path of each attributed leaf, traversal order.

    Parameters
    ----------
    prepared
        Normalized inputs.

    Returns
    -------
    list[tuple[Any, ...]]
        Paths into the positional arguments, then keyword arguments.
    """

    originals = {id(leaf) for leaf in prepared.attributed_leaves}
    found: dict[int, tuple[Any, ...]] = {}

    def _walk(tree: Any, path: tuple[Any, ...]) -> None:
        if isinstance(tree, Tensor):
            if id(tree) in originals:
                found.setdefault(id(tree), path)
        elif isinstance(tree, (tuple, list)):
            for index, item in enumerate(tree):
                _walk(item, (*path, index))
        elif isinstance(tree, dict):
            for key, item in tree.items():
                _walk(item, (*path, key))

    for index, item in enumerate(prepared.positional):
        _walk(item, (index,))
    for key, item in prepared.kwargs.items():
        _walk(item, (key,))
    return [found[id(leaf)] for leaf in prepared.attributed_leaves]


def _resolve_features(features: str, leaves: tuple[Tensor, ...], method: str) -> str:
    """Resolve ``features="auto"`` and validate the grouping name.

    Parameters
    ----------
    features
        Requested grouping.
    leaves
        Evaluated inputs.
    method
        Public name used in the refusal.

    Returns
    -------
    str
        ``"elements"``, ``"pixels"``, ``"patches"`` or ``"tokens"``.

    Raises
    ------
    AttributionError
        On an unknown grouping or one the input shape does not support.
    """

    allowed = ("auto", "elements", "pixels", "patches", "tokens")
    if features not in allowed:
        raise AttributionError(
            f"{method} features must be one of {allowed}; got {features!r}. Remedy: "
            f"tl.attribution.{method}(model, x, target=0, attribution=result, "
            f"features='elements'). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    if features == "auto":
        ranks = {leaf.ndim for leaf in leaves}
        if ranks == {4}:
            return "pixels"
        if ranks == {3}:
            return "tokens"
        return "elements"
    needed = {"pixels": 4, "patches": 4, "tokens": 3}.get(features)
    bad = [tuple(leaf.shape) for leaf in leaves if needed is not None and leaf.ndim != needed]
    if bad:
        layout = "(N, T, ...)" if features == "tokens" else "(N, C, H, W)"
        raise AttributionError(
            f"{method} features={features!r} groups inputs laid out {layout}, but got "
            f"inputs of shape {bad}. Remedy: tl.attribution.{method}(model, x, target=0, "
            f"attribution=result, features='elements'). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    return features


def _patch_sides(patch_size: int | tuple[int, int], method: str) -> tuple[int, int]:
    """Validate ``patch_size``.

    Parameters
    ----------
    patch_size
        Square side or ``(height, width)``.
    method
        Public name used in the refusal.

    Returns
    -------
    tuple[int, int]
        Patch height and width.

    Raises
    ------
    AttributionError
        If a side is not a positive integer.
    """

    sides = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
    if len(sides) != 2 or any(
        isinstance(side, bool) or not isinstance(side, int) or side <= 0 for side in sides
    ):
        raise AttributionError(
            f"{method} patch_size must be a positive int or a pair of them; got "
            f"{patch_size!r}. Remedy: tl.attribution.{method}(model, x, target=0, "
            f"attribution=result, features='patches', patch_size=8). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    return (int(sides[0]), int(sides[1]))


def _leaf_groups(leaf: Tensor, features: str, patch: tuple[int, int]) -> tuple[Tensor, int]:
    """Map each non-batch element of a leaf to a local feature id.

    Parameters
    ----------
    leaf
        One evaluated input.
    features
        Resolved grouping.
    patch
        Patch height and width (``"patches"`` only).

    Returns
    -------
    tuple[Tensor, int]
        Long tensor of ``leaf.shape[1:]`` holding feature ids, and the
        number of features.
    """

    shape = tuple(leaf.shape[1:])
    if features == "elements":
        count = math.prod(shape)
        return torch.arange(count, device=leaf.device).reshape(shape), count
    if features == "tokens":
        tokens = shape[0]
        ids = torch.arange(tokens, device=leaf.device).reshape(tokens, *([1] * (len(shape) - 1)))
        return ids.expand(shape).contiguous(), tokens
    channels, height, width = shape
    if features == "pixels":
        ids = torch.arange(height * width, device=leaf.device).reshape(1, height, width)
        return ids.expand(shape).contiguous(), height * width
    rows = torch.arange(height, device=leaf.device) // patch[0]
    cols = torch.arange(width, device=leaf.device) // patch[1]
    n_cols = int(cols.max()) + 1
    ids = (rows[:, None] * n_cols + cols[None, :]).reshape(1, height, width)
    return ids.expand(channels, height, width).contiguous(), (int(rows.max()) + 1) * n_cols


def _baseline_for(baseline: Any, leaf: Tensor, method: str) -> Tensor:
    """Resolve the baseline of one leaf.

    Parameters
    ----------
    baseline
        ``"zeros"``, ``"blur"``, a number or a tensor of the leaf's shape.
    leaf
        The evaluated input.
    method
        Public name used in the refusal.

    Returns
    -------
    Tensor
        Baseline of the leaf's shape, dtype and device.

    Raises
    ------
    AttributionError
        If the baseline is not one of the supported forms.
    """

    if isinstance(baseline, str) and baseline == "zeros":
        return torch.zeros_like(leaf)
    if isinstance(baseline, str) and baseline == "blur" and leaf.ndim == 4:
        pad = _BLUR_KERNEL // 2
        padded = F.pad(leaf, (pad, pad, pad, pad), mode="replicate")
        return F.avg_pool2d(padded, _BLUR_KERNEL, stride=1).to(leaf.dtype)
    if isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
        return torch.full_like(leaf, float(baseline))
    if isinstance(baseline, Tensor) and tuple(baseline.shape) == tuple(leaf.shape):
        return baseline.detach().to(device=leaf.device, dtype=leaf.dtype)
    raise AttributionError(
        f"{method} baseline must be 'zeros', 'blur' (inputs laid out (N, C, H, W)), a "
        f"number, or a tensor of the input's shape {tuple(leaf.shape)}; got "
        f"{_baseline_repr(baseline)}. Remedy: tl.attribution.{method}(model, x, target=0, "
        f"attribution=result, baseline='zeros'). {_DOCS_POINTER}",
        code="attribution_eval_setting_invalid",
    )


def _baseline_repr(baseline: Any) -> str:
    """Short description of a baseline for settings and messages."""

    if isinstance(baseline, Tensor):
        return f"Tensor(shape={tuple(baseline.shape)})"
    return repr(baseline)


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


def _validated_steps(n_steps: int, n_features: int, method: str) -> list[int]:
    """Return the feature count of every step.

    Parameters
    ----------
    n_steps
        Requested number of steps.
    n_features
        Features per example.
    method
        Public name used in the refusal.

    Returns
    -------
    list[int]
        ``n_steps + 1`` non-decreasing counts from 0 to ``n_features``.

    Raises
    ------
    AttributionError
        If ``n_steps`` is not a positive integer.
    """

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise AttributionError(
            f"{method} n_steps must be a positive integer; got {n_steps!r}. Remedy: "
            f"tl.attribution.{method}(model, x, target=0, attribution=result, n_steps=20). "
            f"{_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    return [(step * n_features) // n_steps for step in range(n_steps + 1)]


def _curve(
    kind: str,
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs,
    *,
    target: _AttributionTarget,
    attribution: AttributionResult | Any,
    baseline: CurveBaseline,
    n_steps: int,
    features: CurveFeatures,
    patch_size: int | tuple[int, int],
) -> CurveResult:
    """Shared body of :func:`deletion_curve` and :func:`insertion_curve`.

    Parameters
    ----------
    kind
        ``"deletion"`` or ``"insertion"``.
    model, inputs, input_kwargs, target, attribution, baseline, n_steps, features, patch_size
        As in :func:`deletion_curve`.

    Returns
    -------
    CurveResult
        The curve with its recorded steps.
    """

    method = f"{kind}_curve"
    prepared = _normalize_model_inputs(inputs, input_kwargs, _Caller(method, target=target))
    leaves = tuple(leaf.detach() for leaf in prepared.attributed_leaves)
    maps = _input_space_maps(attribution, leaves, method)
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
    plan = _CurvePlan(kind, prepared, leaves, baselines, tuple(groups), rank)
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
            "ranking": "most attributed first",
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


def deletion_curve(
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
        target=target,
        attribution=attribution,
        baseline=baseline,
        n_steps=n_steps,
        features=features,
        patch_size=patch_size,
    )


def insertion_curve(
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
        target=target,
        attribution=attribution,
        baseline=baseline,
        n_steps=n_steps,
        features=features,
        patch_size=patch_size,
    )


__all__ = ["CurveResult", "CurveStep", "deletion_curve", "insertion_curve"]

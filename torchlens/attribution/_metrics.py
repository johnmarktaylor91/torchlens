"""Attribution quality metrics: infidelity and sensitivity (memo D14-D15).

Infidelity judges the perturbation as much as the attribution. It ships with
a USABLE named default (``perturb="gaussian"``, ``noise_std`` disclosed in
every result) because a no-default perturbation callback is the measured
reason reference implementations of this metric go unused; the named
square-removal and a fully compatible callable escape remain. Both the
normalized and unnormalized values are returned; the unnormalized value is
the reference-parity primary. "Lower is better" holds only under identical
target / baseline / perturbation / radius / norm / normalization / sample
bank -- that qualification rides in every result.

Sensitivity enforces the determinism ladder (D15): estimator noise must never
masquerade as input sensitivity. (1) a native/bound method known stochastic
must carry a fixed seed or stored bank or sensitivity refuses BEFORE work;
(2) an opaque callable whose first result disclosed stochastic-method
provenance without frozen-randomness markers refuses; (3) a fully opaque
callable gets the behavioral probe -- called twice on the unperturbed inputs,
comparing NUMERIC VALUE TREES ONLY under a tight documented RELATIVE
tolerance (1e-6), refusing on excess. Bit-equality is BANNED as the
comparator: GPU backward kernels are nondeterministic by default, so bitwise
comparison would false-refuse fully deterministic methods. The observed
two-call difference is always disclosed; equality records
``determinism_probe="passed_not_proven"`` -- two equal calls do not prove
future determinism.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._binder import _bind_method, _BoundMethod
from torchlens.attribution._core import (
    InputKwargs,
    _AttributionTarget,
    _call_model,
    _interned_by_identity,
    _normalize_model_inputs,
    _PreparedInputs,
    _scalarize_output,
    _target_repr,
    _temporarily_eval,
)
from torchlens.attribution._result import (
    AttributionError,
    AttributionResult,
    AttributionWarning,
)
from torchlens.attribution._sampling import (
    _NoiseSource,
    _perturbed_user_inputs,
    _scalar_distribution,
    _validate_tree_invariance,
)

# The behavioral probe's RELATIVE tolerance (D15). Measured detection margins
# for genuinely stochastic methods run 2.0e-02 .. 3.9e-01 -- about four
# orders of magnitude above float wobble -- which is why the tolerance must
# stay tight.
_PROBE_RTOL = 1e-6
_PROBE_ATOL = 1e-12

# Result methods whose extra disclose stochastic provenance (ladder rung 2).
_STOCHASTIC_METHOD_NAMES = frozenset({"smoothgrad", "noise_tunnel", "gradient_shap"})

_LOWER_IS_BETTER_QUALIFICATION = (
    "lower is better only under identical target, baseline, perturbation, "
    "radius, norm, normalization, and sample bank"
)


@dataclass(frozen=True)
class MetricResult:
    """Frozen container for one attribution quality metric (memo D14).

    Attributes
    ----------
    metric
        Metric name (``"infidelity"`` or ``"sensitivity_max"``).
    value
        The primary scalar (reference-parity convention).
    settings
        The RESOLVED evaluation policy: every value that must match for a
        cross-result comparison to mean anything.
    extra
        Additional disclosure (distributions, probes, qualifications).
    """

    metric: str
    value: float
    settings: dict[str, Any]
    extra: dict[str, Any]

    def __repr__(self) -> str:
        """Compact repr without dumping tensors."""

        return (
            f"MetricResult(metric={self.metric!r}, value={self.value!r}, "
            f"settings_keys={sorted(self.settings)!r}, "
            f"extra_keys={sorted(self.extra)!r})"
        )


def _attribution_leaves(
    attribution: AttributionResult | Any,
    prepared: _PreparedInputs,
) -> tuple[Tensor, ...]:
    """Align attribution values with the attributed input leaves.

    Parameters
    ----------
    attribution
        An ``AttributionResult`` (its ``values`` tree is used) or a raw value
        tree mirroring the inputs.
    prepared
        Normalized inputs of the metric call.

    Returns
    -------
    tuple[Tensor, ...]
        Per-leaf attribution tensors, traversal order.

    Raises
    ------
    AttributionError
        If the attribution does not align with the input leaves -- including
        layer-space values and CAM-shaped maps, whose refusal names the
        input-space expansion recipe.
    """

    values = attribution.values if isinstance(attribution, AttributionResult) else attribution
    leaves = prepared.attributed_leaves
    flat: list[Tensor] = []

    def _collect(tree: Any) -> None:
        """Collect tensor leaves in traversal order (None slots skipped)."""

        if isinstance(tree, Tensor):
            flat.append(tree)
        elif isinstance(tree, (tuple, list)):
            for item in tree:
                _collect(item)
        elif isinstance(tree, dict):
            for item in tree.values():
                _collect(item)

    _collect(values)
    if len(flat) != len(leaves) or any(
        tuple(value.shape) != tuple(leaf.shape) for value, leaf in zip(flat, leaves, strict=False)
    ):
        raise AttributionError(
            "infidelity requires INPUT-SPACE attributions aligned with the "
            "attributed input leaves; got a value tree that does not mirror "
            "the inputs (layer-space values and native-resolution CAMs do "
            "not qualify). Remedy: evaluate an input-space method, or expand "
            "the layer attribution into input space first (for CAMs, "
            "grad_cam already returns input-resolution values via bilinear "
            "upsampling -- pass those only if the map matches the input "
            "shape).",
            code="metric_attribution_shape_mismatch",
        )
    return tuple(flat)


def _square_removal_perturbation(
    leaf: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Draw one named square-removal perturbation for a spatial leaf.

    The square side is ``max(1, round(min(H, W) / 4))`` (disclosed); the
    location is uniform. The perturbation is the removed content, so the
    perturbed input zeroes the square.

    Parameters
    ----------
    leaf
        Spatial input leaf of shape ``(N, C, H, W)`` (or more trailing dims).
    generator
        Seeded CPU generator for the location draw.

    Returns
    -------
    Tensor
        Perturbation tensor (``x - perturbation`` zeroes the square).

    Raises
    ------
    AttributionError
        If the leaf carries no 2-D spatial tail.
    """

    if leaf.ndim < 4:
        raise AttributionError(
            "perturb='square_removal' requires spatial inputs shaped "
            "(N, C, H, W). Remedy: use perturb='gaussian' or a callable.",
            code="metric_perturbation_invalid",
        )
    height, width = int(leaf.shape[-2]), int(leaf.shape[-1])
    side = max(1, round(min(height, width) / 4))
    top = int(torch.randint(0, height - side + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, width - side + 1, (1,), generator=generator).item())
    mask = torch.zeros_like(leaf)
    mask[..., top : top + side, left : left + side] = 1.0
    return leaf.detach() * mask


def _warn_out_of_range(perturbed_leaves: tuple[Tensor, ...], prepared: _PreparedInputs) -> None:
    """Warn (coded, once per call) when perturbed values leave the input hull.

    Valid input ranges are rarely knowable, so out-of-range perturbations WARN
    rather than refuse (memo D14): the metric still computes, but the user is
    told the perturbation left the observed data range.

    Parameters
    ----------
    perturbed_leaves
        Perturbed leaves for one sample.
    prepared
        Normalized inputs whose observed min/max define the hull.
    """

    for perturbed, original in zip(perturbed_leaves, prepared.attributed_leaves, strict=True):
        low, high = original.min(), original.max()
        # Valid ranges are rarely knowable, so the hull is expanded by 10% of
        # its width before the exceedance check: tiny-noise policies on
        # ordinary inputs stay silent, while a degenerate (constant) input or
        # a large perturbation still discloses.
        margin = (high - low) * 0.1
        if bool((perturbed < low - margin).any()) or bool((perturbed > high + margin).any()):
            warnings.warn(
                AttributionWarning(
                    "an infidelity perturbation produced values outside the "
                    "observed input range; the metric judges the perturbation "
                    "as much as the attribution, so consider a perturbation "
                    "matched to the data domain. Remedy: reduce noise_std or "
                    "pass a domain-aware perturb callable.",
                    code="metric_perturbation_out_of_range",
                ),
                stacklevel=3,
            )
            return


def _draw_sample_perturbation(
    perturb: str | Callable[..., tuple[Any, Any]],
    prepared: _PreparedInputs,
    noise: _NoiseSource,
    noise_std: float,
    generator: torch.Generator,
    sample_index: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Draw one sample's (perturbations, perturbed leaves) for infidelity.

    Parameters
    ----------
    perturb
        Resolved policy token or the user's callable escape.
    prepared
        Normalized metric inputs.
    noise
        Seeded per-device noise source (gaussian policy).
    noise_std
        Gaussian scale.
    generator
        Seeded CPU generator (square-removal locations).
    sample_index
        Zero-based sample index.

    Returns
    -------
    tuple[tuple[Tensor, ...], tuple[Tensor, ...]]
        Perturbation tensors and perturbed leaves, traversal order.

    Raises
    ------
    AttributionError
        If a callable escape violates its contract.
    """

    if perturb == "gaussian":
        counter = {"unique": 0}

        def _draw(slot: int) -> Tensor:
            """One gaussian perturbation per unique leaf."""

            leaf = prepared.attributed_leaves[slot]
            unique = counter["unique"]
            counter["unique"] += 1
            return noise_std * noise.draw(leaf, sample_index, unique)

        perturbations = _interned_by_identity(prepared.attributed_leaves, _draw)
    elif perturb == "square_removal":
        perturbations = _interned_by_identity(
            prepared.attributed_leaves,
            lambda slot: _square_removal_perturbation(prepared.attributed_leaves[slot], generator),
        )
    else:
        if not callable(perturb):
            raise AttributionError(
                f"perturb must be 'gaussian', 'square_removal', or a "
                f"callable; got {perturb!r}. Remedy: choose a documented "
                "policy.",
                code="metric_perturbation_invalid",
            )
        returned = perturb(tuple(prepared.attributed_leaves))
        if not isinstance(returned, tuple) or len(returned) != 2:
            raise AttributionError(
                "a perturb callable must return (perturbations, "
                "perturbed_inputs), each mirroring the attributed-leaf "
                "tuple. Remedy: follow the documented callable contract.",
                code="metric_perturbation_invalid",
            )
        perturbations = tuple(returned[0])
        perturbed_from_callable = tuple(returned[1])
        n_leaves = len(prepared.attributed_leaves)
        if len(perturbations) != n_leaves or len(perturbed_from_callable) != n_leaves:
            raise AttributionError(
                "a perturb callable must return one perturbation and one "
                "perturbed input per attributed leaf. Remedy: mirror the "
                "attributed-leaf tuple.",
                code="metric_perturbation_invalid",
            )
        return perturbations, perturbed_from_callable
    perturbed = tuple(
        leaf.detach() - perturbation
        for leaf, perturbation in zip(prepared.attributed_leaves, perturbations, strict=True)
    )
    return perturbations, perturbed


def infidelity(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    attribution: AttributionResult | Any,
    perturb: str | Callable[..., tuple[Any, Any]] = "gaussian",
    noise_std: float = 0.003,
    n_samples: int = 10,
    seed: int | None = None,
) -> MetricResult:
    """Measure how well an attribution predicts perturbation-induced changes.

    ``infidelity = E[(sum(attr * dx) - (f(x) - f(x - dx)))^2]`` over sampled
    perturbations ``dx`` (reference-parity convention). Both the unnormalized
    (primary) and normalized values are returned.

    Parameters
    ----------
    model
        PyTorch module the attribution explains.
    inputs
        Bare tensor or tuple/list of positional model arguments.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index or callable ``output -> scalar tensor``; must be
        the SAME target the attribution was computed for.
    attribution
        The attribution to judge: an ``AttributionResult`` or a value tree
        mirroring the attributed inputs (input space only; layer-space values
        refuse with the expansion recipe).
    perturb
        ``"gaussian"`` (default; ``dx ~ N(0, noise_std^2)``), the named
        ``"square_removal"``, or a fully compatible callable
        ``perturb(inputs_tuple) -> (perturbations, perturbed_inputs)`` where
        both returns mirror the attributed-leaf tuple.
    noise_std
        Gaussian scale, disclosed in every result (default 0.003).
    n_samples
        Number of perturbation samples (default 10, reference parity).
    seed
        Optional deterministic seed.

    Returns
    -------
    MetricResult
        ``value`` is the unnormalized infidelity;
        ``extra["infidelity_normalized"]`` the scale-optimal variant;
        settings carry the full resolved policy.

    Raises
    ------
    AttributionError
        On shape misalignment, invalid perturbation policy, or a callable
        escape returning mismatched trees.
    """

    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise AttributionError(
            "n_samples must be a positive integer. Remedy: keep the default "
            "10 for reference parity, or raise it for tighter estimates.",
            code="metric_perturbation_invalid",
        )
    prepared = _normalize_model_inputs(inputs, input_kwargs)
    attribution_leaves = _attribution_leaves(attribution, prepared)
    noise = _NoiseSource(seed, None)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    perturb_policy: str
    if perturb == "gaussian":
        perturb_policy = f"gaussian(noise_std={noise_std})"
    elif perturb == "square_removal":
        perturb_policy = "square_removal(side=min(H,W)/4, uniform location)"
    elif callable(perturb):
        perturb_policy = f"callable({getattr(perturb, '__name__', repr(perturb))})"
    else:
        raise AttributionError(
            f"perturb must be 'gaussian', 'square_removal', or a callable; "
            f"got {perturb!r}. Remedy: choose a documented policy.",
            code="metric_perturbation_invalid",
        )

    dot_terms: list[Tensor] = []
    delta_terms: list[Tensor] = []
    per_sample_values: list[float] = []
    with _temporarily_eval(model), torch.no_grad():
        original_output = _call_model(model, prepared, prepared.attributed_leaves)
        if isinstance(target, int):
            if not isinstance(original_output, Tensor):
                raise AttributionError(
                    "int targets require tensor model outputs. Remedy: use a callable target.",
                    code="attribution_target_invalid",
                )
            original_scores = original_output[..., target].detach()
        else:
            original_scores = _scalarize_output(original_output, target).detach()
        for sample_index in range(n_samples):
            perturbations, perturbed_leaves = _draw_sample_perturbation(
                perturb, prepared, noise, noise_std, generator, sample_index
            )
            _warn_out_of_range(perturbed_leaves, prepared)
            perturbed_output = _call_model(model, prepared, perturbed_leaves)
            if isinstance(target, int):
                perturbed_scores = perturbed_output[..., target].detach()
            else:
                perturbed_scores = _scalarize_output(perturbed_output, target).detach()
            delta = (original_scores - perturbed_scores).reshape(-1).to(torch.float64)
            per_leaf_dots = []
            for attr, perturbation in zip(attribution_leaves, perturbations, strict=True):
                product = (attr.to(torch.float64) * perturbation.to(torch.float64)).detach()
                if isinstance(target, int) and product.ndim >= 1:
                    per_leaf_dots.append(product.reshape(product.shape[0], -1).sum(dim=-1))
                else:
                    per_leaf_dots.append(product.sum().reshape(1))
            dot = torch.stack(per_leaf_dots, dim=0).sum(dim=0).reshape(-1)
            if dot.shape != delta.shape:
                # Callable targets aggregate over the batch; compare scalars.
                delta = delta.sum().reshape(1)
                dot = dot.sum().reshape(1)
            dot_terms.append(dot)
            delta_terms.append(delta)
            per_sample_values.append(float(((dot - delta) ** 2).mean()))

    dots = torch.stack(dot_terms)
    deltas = torch.stack(delta_terms)
    unnormalized = float(((dots - deltas) ** 2).mean())
    denominator = float((dots * dots).mean())
    beta = float((dots * deltas).mean()) / denominator if denominator > 0 else 0.0
    normalized = float(((beta * dots - deltas) ** 2).mean())
    settings = {
        "perturb": perturb_policy,
        "noise_std": noise_std if perturb == "gaussian" else None,
        "n_samples": n_samples,
        "seed": seed,
        "target": _target_repr(target),
        "normalization_primary": "unnormalized",
    }
    return MetricResult(
        metric="infidelity",
        value=unnormalized,
        settings=settings,
        extra={
            "infidelity_unnormalized": unnormalized,
            "infidelity_normalized": normalized,
            "normalization_beta": beta,
            "per_sample": _scalar_distribution(per_sample_values),
            "qualification": _LOWER_IS_BETTER_QUALIFICATION,
            "note": "infidelity judges the perturbation as much as the attribution.",
        },
    )


def _flatten_result_tree(values: Any) -> Tensor:
    """Flatten a result value tree into one float64 vector for norms.

    Parameters
    ----------
    values
        Attribution value tree.

    Returns
    -------
    Tensor
        Concatenated flattened tensor leaves (empty tensor when none).
    """

    flat: list[Tensor] = []

    def _walk(tree: Any) -> None:
        """Collect tensor leaves depth-first into ``flat``."""

        if isinstance(tree, Tensor):
            flat.append(tree.detach().reshape(-1).to(torch.float64))
        elif isinstance(tree, (tuple, list)):
            for item in tree:
                _walk(item)
        elif isinstance(tree, dict):
            for item in tree.values():
                _walk(item)

    _walk(values)
    if not flat:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(flat)


def _max_relative_difference(first: Any, second: Any) -> float:
    """Return the max elementwise relative difference between two value trees.

    Parameters
    ----------
    first
        First result value tree.
    second
        Second result value tree.

    Returns
    -------
    float
        ``max |a - b| / (|a| + atol)`` over all tensor leaves.
    """

    a = _flatten_result_tree(first)
    b = _flatten_result_tree(second)
    if a.numel() == 0:
        return 0.0
    return float(((a - b).abs() / (a.abs() + _PROBE_ATOL)).max())


def _enforce_determinism_ladder(
    bound: _BoundMethod,
    first_result: AttributionResult,
    probe_result: AttributionResult | None,
) -> dict[str, Any]:
    """Apply the D15 determinism ladder and return the probe disclosure.

    Parameters
    ----------
    bound
        The normalized method binder.
    first_result
        First unperturbed attribution.
    probe_result
        Second unperturbed attribution (opaque callables only).

    Returns
    -------
    dict[str, Any]
        Probe disclosure for the result extra.

    Raises
    ------
    AttributionError
        On rung-2 provenance refusal or rung-3 probe failure.
    """

    if bound.kind != "opaque":
        return {
            "determinism_probe": "structural",
            "reason": (
                "seeded stochastic methods are bit-reproducible by "
                "construction (per-call generator cache seeded on first "
                "use); deterministic methods need no probe"
            ),
        }
    method_names = {
        first_result.method,
        str(first_result.extra.get("child_method", "")),
    }
    if method_names & _STOCHASTIC_METHOD_NAMES:
        frozen = first_result.extra.get("seed") is not None or first_result.extra.get(
            "noise_bank_used"
        )
        if not frozen:
            raise AttributionError(
                "sensitivity refuses an opaque callable whose first result "
                "disclosed STOCHASTIC method provenance "
                f"({sorted(method_names & _STOCHASTIC_METHOD_NAMES)}) without "
                "frozen-randomness markers: estimator noise would "
                "masquerade as input sensitivity. Remedy: fix the wrapped "
                "method's seed (or pass a stored sample bank) inside the "
                "callable.",
                code="metric_determinism_unfrozen",
            )
    # Opaque callables always run the probe's second call before reaching here.
    probe_values = probe_result.values if probe_result is not None else first_result.values
    observed = _max_relative_difference(first_result.values, probe_values)
    if observed > _PROBE_RTOL:
        raise AttributionError(
            "sensitivity's behavioral determinism probe failed: two calls on "
            f"the UNPERTURBED inputs differ by {observed:.3e} relative "
            f"(tolerance {_PROBE_RTOL:.0e}). Estimator noise must never "
            "masquerade as input sensitivity. Remedy: seed the wrapped "
            "method's randomness, or pass a stored sample bank.",
            code="metric_determinism_probe_failed",
            observed_relative_difference=observed,
        )
    return {
        "determinism_probe": "passed_not_proven",
        "observed_two_call_relative_difference": observed,
        "reason": "two equal calls do not prove future determinism",
    }


def sensitivity(
    inputs: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    attribute: Callable[..., AttributionResult] | None = None,
    model: Module | None = None,
    method: Callable[..., AttributionResult] | None = None,
    method_kwargs: dict[str, Any] | None = None,
    target: Any = None,
    n_samples: int = 10,
    radius: float = 0.02,
    seed: int | None = None,
) -> MetricResult:
    """Max-sensitivity: how much the attribution moves under tiny input noise.

    ``sensitivity_max = max_samples ||A(x') - A(x)||_F / ||A(x)||_F`` with
    ``x'`` drawn uniformly within an L-infinity ball of ``radius`` around the
    attributed leaves.

    Parameters
    ----------
    inputs
        Bare tensor or tuple/list of positional model arguments.
    input_kwargs
        Optional keyword arguments for the wrapped call.
    attribute
        Closed callable route (exclusive with ``method``); subject to the
        determinism ladder's provenance and behavioral rungs.
    model
        Model for the ``method=`` sugar route.
    method
        Kit-contract callable; requires ``model`` and ``target``.
    method_kwargs
        Explicit child settings mapping. A known-stochastic method WITHOUT a
        seed or stored bank in here refuses BEFORE any work (ladder rung 1).
    target
        Target for the sugar route.
    n_samples
        Number of perturbed evaluations (default 10).
    radius
        L-infinity perturbation radius (default 0.02).
    seed
        Optional deterministic seed for the perturbation draws.

    Returns
    -------
    MetricResult
        ``value`` is the max relative attribution change; the probe
        disclosure and per-sample distribution ride ``extra``.

    Raises
    ------
    AttributionError
        On ladder refusals or invalid sampling parameters.
    """

    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
        raise AttributionError(
            "n_samples must be a positive integer. Remedy: keep the default "
            "10, or raise it for a tighter max.",
            code="metric_perturbation_invalid",
        )
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius <= 0:
        raise AttributionError(
            "radius must be a positive float (L-infinity perturbation "
            "radius). Remedy: keep the default 0.02.",
            code="metric_perturbation_invalid",
        )
    bound = _bind_method(
        attribute=attribute,
        method=method,
        method_kwargs=method_kwargs,
        model=model,
        target=target,
    )
    if bound.known_stochastic and not bound.frozen_randomness:
        raise AttributionError(
            f"sensitivity refuses the known-stochastic method "
            f"{bound.method_name!r} without a fixed seed or stored sample "
            "bank: estimator noise would masquerade as input sensitivity "
            "(ladder rung 1, refused BEFORE any work). Remedy: pass "
            "method_kwargs={'seed': ...} (or a stored bank).",
            code="metric_determinism_unfrozen",
        )
    prepared = _normalize_model_inputs(inputs, input_kwargs)

    baseline_result = bound.call(*_perturbed_user_inputs(prepared, prepared.attributed_leaves))
    probe_result: AttributionResult | None = None
    if bound.kind == "opaque":
        # The probe's second call supplies the unperturbed-attribution
        # denominator the metric needs anyway, so part of its cost is
        # reclaimed (memo D15).
        probe_result = bound.call(*_perturbed_user_inputs(prepared, prepared.attributed_leaves))
    probe_disclosure = _enforce_determinism_ladder(bound, baseline_result, probe_result)
    reference = probe_result if probe_result is not None else baseline_result
    reference_flat = _flatten_result_tree(reference.values)
    reference_norm = float(reference_flat.norm())
    if reference_norm == 0.0:
        raise AttributionError(
            "sensitivity is undefined for an all-zero unperturbed "
            "attribution (the relative change has no denominator). Remedy: "
            "check the attribution target and method.",
            code="metric_perturbation_invalid",
        )

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    ratios: list[float] = []
    for _sample in range(n_samples):

        def _perturb(slot: int) -> Tensor:
            """Uniform L-infinity perturbation of one unique leaf."""

            leaf = prepared.attributed_leaves[slot]
            offset = (
                torch.rand(leaf.shape, dtype=torch.float64, generator=generator) * 2 - 1
            ) * radius
            return (leaf.detach() + offset.to(leaf.dtype)).detach()

        perturbed_leaves = _interned_by_identity(prepared.attributed_leaves, _perturb)
        perturbed_result = bound.call(*_perturbed_user_inputs(prepared, perturbed_leaves))
        _validate_tree_invariance(reference.values, perturbed_result.values, _sample)
        difference = _flatten_result_tree(perturbed_result.values) - reference_flat
        ratios.append(float(difference.norm()) / reference_norm)

    settings = {
        "method": bound.method_name,
        "method_kwargs": dict(bound.settings),
        "n_samples": n_samples,
        "radius": radius,
        "norm": "frobenius",
        "seed": seed,
    }
    return MetricResult(
        metric="sensitivity_max",
        value=max(ratios),
        settings=settings,
        extra={
            **probe_disclosure,
            "per_sample": _scalar_distribution(ratios),
            "qualification": _LOWER_IS_BETTER_QUALIFICATION,
        },
    )


__all__ = ["MetricResult", "infidelity", "sensitivity"]

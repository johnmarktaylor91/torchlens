"""Noise tunnel: run ANY kit method on noised input copies and aggregate.

Attrib memo D4-D6. The tunnel perturbs floating and complex attributed leaves
only (integer ids, masks, booleans, and strings pass through unchanged),
reuses ONE noised object at every repeated tensor reference, takes ABSOLUTE
noise scales (range-relative noise is a docs recipe, never a hidden
inference), and aggregates with ``"mean"``, ``"mean_square"``, or population
``"variance"``. Sampling is serial behind the replaceable runner seam (D5) --
there is deliberately NO public sample-batch knob.

Composition rulings (D6): ``noise_tunnel(method=smoothgrad)`` refuses with a
pointer to the direct saliency spelling; ``noise_tunnel(method=gradient_shap)``
works with independent per-sample RNG substreams and both sample counts
disclosed; ``noise_tunnel(method=grad_cam)`` disables child rendering and
renders ONCE from the aggregate; trace-bound methods refuse (a static,
already-captured Trace leaves nothing to perturb -- a recapturing closed
callable works because each call re-runs the capture on the noised inputs).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torch import Tensor
from torch.nn import Module

from torchlens.attribution._binder import _bind_method, _BoundMethod
from torchlens.attribution._core import (
    _normalize_model_inputs,
    _validate_positive_int,
)
from torchlens.attribution._result import AttributionError, AttributionResult
from torchlens.attribution._sampling import (
    _aggregate_value_trees,
    _completeness_distributions,
    _noised_sample_leaves,
    _NoiseSource,
    _perturbed_user_inputs,
    _resolve_stdevs,
    _run_serial_samples,
)

_AGGREGATIONS = ("mean", "mean_square", "variance")

# Deterministic substream derivation for stochastic children (NT(gradient_shap)):
# a large odd multiplier keeps per-sample child seeds distinct and reproducible.
_CHILD_SEED_STRIDE = 1_000_003


def _derived_child_seed(seed: int, sample_index: int) -> int:
    """Derive one independent child RNG seed per tunnel sample.

    Parameters
    ----------
    seed
        The tunnel's own seed.
    sample_index
        Zero-based tunnel sample index.

    Returns
    -------
    int
        Deterministic per-sample child seed.
    """

    return (seed * _CHILD_SEED_STRIDE + sample_index) % (2**31 - 1)


def _refuse_composition(bound: _BoundMethod) -> None:
    """Apply the D6 composition refusals for the bound child method.

    Parameters
    ----------
    bound
        The normalized child binder.

    Raises
    ------
    AttributionError
        If the child is SmoothGrad (the tunnel IS SmoothGrad's mechanism) or
        a trace-bound method (a static Trace leaves nothing to perturb).
    """

    if bound.method_name == "smoothgrad":
        raise AttributionError(
            "noise_tunnel(method=smoothgrad) would tunnel a tunnel: smoothgrad "
            "IS noise_tunnel over saliency with per-sample absolute values. "
            "Remedy: call noise_tunnel(method=saliency, ...) for the direct "
            "spelling, or call smoothgrad(...) itself.",
            code="noise_tunnel_composition_unsupported",
        )
    if bound.kind == "trace":
        raise AttributionError(
            f"noise_tunnel cannot wrap {bound.method_name!r}: it attributes a "
            "static, already-captured Trace, so there is no input left to "
            "perturb. Remedy: pass a closed callable that RE-CAPTURES a trace "
            "from the perturbed inputs on every call.",
            code="noise_tunnel_composition_unsupported",
        )


def _grad_cam_aggregate_overlay(
    aggregated: Any,
    inputs_prepared: Any,
    child_results: list[AttributionResult],
) -> Any:
    """Render the ONE aggregate overlay for a grad_cam child (D6).

    Parameters
    ----------
    aggregated
        Aggregated CAM value tree (a 4-D ``N,1,H,W`` tensor for grad_cam).
    inputs_prepared
        Normalized tunnel inputs (the UNNOISED originals).
    child_results
        Per-sample child results (rendering was disabled on every one).

    Returns
    -------
    Any
        A PIL image overlay, or ``None`` when no spatial input exists.
    """

    from torchlens.attribution._layer import _cam_overlay

    if not isinstance(aggregated, Tensor) or aggregated.ndim != 4:
        return None
    spatial_candidates = [leaf for leaf in inputs_prepared.attributed_leaves if leaf.ndim >= 4]
    if not spatial_candidates:
        return None
    return _cam_overlay(
        aggregated,
        spatial_candidates[0],
        image=None,
        alpha=0.6,
        cmap="magma",
    )


def _validate_noise_bank_coverage(
    noise_bank: Sequence[Sequence[Tensor]] | None,
    n_samples: int,
    prepared: Any,
) -> None:
    """Validate a stored bank's coverage up front (attrib memo D4).

    A defective bank is TUNNEL configuration, not a child failure, so
    coverage is checked before any sample runs (per-draw geometry re-checks
    remain in the noise source).

    Parameters
    ----------
    noise_bank
        Optional stored draws, ``noise_bank[sample][unique_leaf]``.
    n_samples
        Planned logical sample count.
    prepared
        Normalized tunnel inputs.

    Raises
    ------
    AttributionError
        If the bank cannot cover every sample's unique-leaf draws.
    """

    if noise_bank is None:
        return
    unique_leaves = len({id(leaf) for leaf in prepared.attributed_leaves})
    if len(noise_bank) < n_samples or any(
        len(sample_draws) < unique_leaves for sample_draws in noise_bank
    ):
        raise AttributionError(
            f"noise_bank must cover {n_samples} samples x {unique_leaves} "
            "unique attributed leaves. Remedy: provide "
            "noise_bank[sample][leaf] tensors for every sample and unique "
            "attributed leaf.",
            code="noise_bank_invalid",
        )


def noise_tunnel(
    inputs: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    attribute: Callable[..., AttributionResult] | None = None,
    model: Module | None = None,
    method: Callable[..., AttributionResult] | None = None,
    method_kwargs: dict[str, Any] | None = None,
    target: Any = None,
    n_samples: int = 25,
    stdevs: float | Sequence[float],
    seed: int | None = None,
    aggregation: str = "mean",
    noise_bank: Sequence[Sequence[Tensor]] | None = None,
) -> AttributionResult:
    """Run an attribution method on ``n_samples`` noised inputs and aggregate.

    Parameters
    ----------
    inputs
        Bare tensor or tuple/list of positional model arguments. Floating and
        complex tensor leaves are perturbed; every other leaf passes through
        unchanged.
    input_kwargs
        Optional keyword arguments for the wrapped call; perturbed by the same
        rule.
    attribute
        Closed callable ``(inputs, input_kwargs) -> AttributionResult``.
        Exclusive with ``method``; ``model``/``target``/``method_kwargs`` are
        refused beside it (nothing is forwarded to a closed callable).
    model
        Model for the ``method=`` sugar route.
    method
        Kit-contract callable; requires ``model`` and ``target``.
    method_kwargs
        Explicit child settings mapping, echoed verbatim into
        ``extra["method_kwargs"]``.
    target
        Target for the ``method=`` sugar route.
    n_samples
        Number of noised copies (logical child calls).
    stdevs
        ABSOLUTE Gaussian noise scale: one nonnegative float for every
        attributed leaf, or a sequence aligned with the leaves in traversal
        order. Complex leaves draw a standard complex normal (real and
        imaginary components each carry variance 1/2). Required -- a hidden
        range-relative default would be an inference, not a disclosure.
    seed
        Optional deterministic seed (per-device generators, seeded on first
        use). For a stochastic child with no explicit child seed, independent
        per-sample child substreams are derived from this seed and disclosed.
    aggregation
        ``"mean"``, ``"mean_square"``, or population ``"variance"``.
    noise_bank
        Optional stored unit-scale draws (``noise_bank[sample][unique_leaf]``)
        replayed instead of fresh draws; oracle rows use this to feed our
        runner and a reference implementation the same bank.

    Returns
    -------
    AttributionResult
        Aggregated values with the full sampling disclosure: resolved noise
        scales, per-sample completeness distributions plus the residual of
        means, planned/completed logical and physical call counts, and the
        names of the child extras that were NOT aggregated.

    Raises
    ------
    AttributionError
        On binding violations, composition refusals, invalid noise scales or
        aggregation tokens, cross-sample invariance violations, or a child
        failure (named by zero-based sample; no partial result).
    """

    _validate_positive_int("n_samples", n_samples)
    if aggregation not in _AGGREGATIONS:
        raise AttributionError(
            f"aggregation must be one of {_AGGREGATIONS}; got {aggregation!r}. "
            "Remedy: choose 'mean', 'mean_square', or 'variance'.",
            code="noise_aggregation_invalid",
        )
    bound = _bind_method(
        attribute=attribute,
        method=method,
        method_kwargs=method_kwargs,
        model=model,
        target=target,
    )
    _refuse_composition(bound)
    prepared = _normalize_model_inputs(inputs, input_kwargs)
    scales = _resolve_stdevs(stdevs, prepared)
    _validate_noise_bank_coverage(noise_bank, n_samples, prepared)
    source = _NoiseSource(seed, noise_bank)

    child_is_grad_cam = bound.method_name == "grad_cam"
    child_is_stochastic_unseeded = bound.known_stochastic and not bound.frozen_randomness
    derived_child_seeds: list[int] = []

    def _call_sample(sample_index: int) -> AttributionResult:
        """Perturb the inputs and invoke the child for one logical sample."""

        noised_leaves = _noised_sample_leaves(prepared, scales, source, sample_index)
        child_inputs, child_kwargs = _perturbed_user_inputs(prepared, noised_leaves)
        if bound.kind == "opaque" or (not child_is_grad_cam and not child_is_stochastic_unseeded):
            return bound.call(child_inputs, child_kwargs)
        # Sugar-route per-sample setting overrides: grad_cam rendering is
        # disabled per sample (the tunnel renders ONCE from the aggregate),
        # and an unseeded stochastic child gets an independent derived
        # substream so tunnel results are reproducible under one seed. The
        # binder guarantees method/model on this route.
        if method is None or model is None:
            raise AttributionError(
                "internal error: sugar-route overrides without a bound "
                "method/model. This is a TorchLens contract breach, not a "
                "user error. Remedy: report this as a bug.",
                code="noise_tunnel_route_unbound",
            )
        settings = dict(bound.settings)
        if child_is_grad_cam:
            settings["overlay"] = False
        if child_is_stochastic_unseeded and seed is not None:
            child_seed = _derived_child_seed(seed, sample_index)
            derived_child_seeds.append(child_seed)
            settings["seed"] = child_seed
        return method(model, child_inputs, child_kwargs, target=target, **settings)

    run = _run_serial_samples(n_samples, _call_sample)
    aggregated = _aggregate_value_trees([result.values for result in run.results], aggregation)

    extra: dict[str, Any] = {
        "child_method": bound.method_name,
        "method_kwargs": dict(bound.settings),
        "aggregation": aggregation,
        "n_samples": n_samples,
        "stdevs_resolved": list(scales),
        "seed": seed,
        "noise_bank_used": noise_bank is not None,
        "complex_noise_rule": (
            "complex leaves draw a standard complex normal; real and imaginary "
            "components each carry variance 1/2"
        ),
        "planned_logical_calls": run.planned_logical_calls,
        "completed_logical_calls": run.completed_logical_calls,
        "child_physical_forward_calls": run.child_physical_forward_calls,
        "completeness_distributions": _completeness_distributions(run.results),
        "non_aggregated_child_extra_keys": sorted(
            {key for result in run.results for key in result.extra}
        ),
    }
    if bound.model_identity is not None:
        extra["model_identity"] = bound.model_identity
    if derived_child_seeds:
        child_n_samples = bound.settings.get("n_samples", 25)
        extra["child_seeds"] = list(derived_child_seeds)
        extra["child_n_samples"] = child_n_samples
        extra["multiplicative_cost_logical"] = n_samples * int(child_n_samples)
    if child_is_grad_cam:
        extra["layer"] = run.results[0].extra.get("layer")
        extra["overlay"] = _grad_cam_aggregate_overlay(aggregated, prepared, run.results)
        extra["overlay_rendered_from"] = "aggregate"
    return AttributionResult(
        method="noise_tunnel",
        values=aggregated,
        target_repr=run.results[0].target_repr,
        extra=extra,
    )


__all__ = ["noise_tunnel"]

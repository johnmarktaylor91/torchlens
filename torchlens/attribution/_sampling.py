"""Serial sample runner shared by noise tunnel, GradientShap, and the metrics.

Attrib memo B1/D4-D5: one substrate owns input-tree perturbation (repeated
tensor references get ONE perturbed object reused at every reference; integer
ids, masks, booleans, and strings pass through unchanged), per-device
generators, sample banks, cross-sample invariance validation, aggregation,
failure naming, and cost counters. Wave-1 sampling is deliberately SERIAL
behind this replaceable executor seam (D5): once a wrapped method arrives as
a closed callable the runner cannot know whether it is batchable, so there is
no public sample-batch knob.

Complex-noise rule (documented, D4): complex leaves draw ``torch.randn`` with
the leaf's complex dtype, i.e. a standard complex normal whose real and
imaginary components each carry variance 1/2 (total variance 1 per element).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from torchlens.attribution._core import (
    _interned_by_identity,
    _PreparedInputs,
    _substitute_inputs,
)
from torchlens.attribution._result import AttributionError, AttributionResult


def _resolve_stdevs(
    stdevs: float | Sequence[float],
    inputs: _PreparedInputs,
) -> tuple[float, ...]:
    """Resolve the absolute noise scale per attributed leaf (attrib memo D4).

    ``stdevs`` is ABSOLUTE and nonnegative -- range-relative noise is a docs
    recipe, never a hidden inference. Repeated references to one tensor must
    resolve to one scale, because they share one noised object.

    Parameters
    ----------
    stdevs
        Scalar applied to every attributed leaf, or a sequence aligned with
        the attributed leaves in traversal order.
    inputs
        Normalized attribution inputs.

    Returns
    -------
    tuple[float, ...]
        One resolved nonnegative scale per attributed leaf, traversal order.

    Raises
    ------
    AttributionError
        If a scale is negative, the alignment is wrong, or repeated
        references carry contradicting scales.
    """

    n_leaves = len(inputs.attributed_leaves)
    if isinstance(stdevs, bool):
        raise AttributionError(
            "stdevs must be a nonnegative float or a sequence of them. "
            "Remedy: pass an absolute noise scale such as stdevs=0.1.",
            code="noise_stdevs_invalid",
        )
    if isinstance(stdevs, (int, float)):
        resolved = (float(stdevs),) * n_leaves
    elif isinstance(stdevs, Sequence) and not isinstance(stdevs, (str, bytes)):
        if len(stdevs) != n_leaves:
            raise AttributionError(
                f"stdevs sequence has {len(stdevs)} entries but the call has "
                f"{n_leaves} attributed leaves. Remedy: align stdevs with the "
                "floating/complex tensor leaves in traversal order, or pass "
                "one scalar.",
                code="noise_stdevs_invalid",
            )
        resolved = tuple(float(value) for value in stdevs)
    else:
        raise AttributionError(
            "stdevs must be a nonnegative float or a sequence of them. "
            "Remedy: pass an absolute noise scale such as stdevs=0.1.",
            code="noise_stdevs_invalid",
        )
    if any(value < 0 for value in resolved):
        raise AttributionError(
            "stdevs must be nonnegative (absolute noise scales). Remedy: pass scales >= 0.",
            code="noise_stdevs_invalid",
        )
    scale_by_id: dict[int, float] = {}
    for leaf, scale in zip(inputs.attributed_leaves, resolved, strict=True):
        seen = scale_by_id.setdefault(id(leaf), scale)
        if seen != scale:
            raise AttributionError(
                "repeated references to the same input tensor must share one "
                "stdevs value: the occurrences share ONE noised object. "
                "Remedy: give both occurrences the same scale.",
                code="noise_stdevs_invalid",
            )
    return resolved


class _NoiseSource:
    """Per-sample noise drawer with per-device generators and an optional bank.

    Draw order is deterministic: one draw per UNIQUE attributed leaf per
    sample, in first-occurrence traversal order -- bit-identical to the
    historical SmoothGrad draw sequence (memo D7).

    Parameters
    ----------
    seed
        Optional deterministic seed; the per-device generator is seeded on
        first use (bit-reproducible by construction).
    noise_bank
        Optional stored draws: ``noise_bank[sample][unique_leaf_index]`` is
        the noise tensor (PRE-scaling) reused instead of drawing. Oracle rows
        use this to feed our runner and captum the same bank.
    """

    def __init__(
        self,
        seed: int | None,
        noise_bank: Sequence[Sequence[Tensor]] | None = None,
    ) -> None:
        """Store the seed and bank; generators are created lazily per device."""

        self.seed = seed
        self.noise_bank = noise_bank
        self._generators: dict[torch.device, torch.Generator] = {}

    def draw(self, leaf: Tensor, sample_index: int, unique_leaf_index: int) -> Tensor:
        """Return one unit-scale noise tensor for ``leaf``.

        Parameters
        ----------
        leaf
            Attributed leaf the noise will perturb.
        sample_index
            Zero-based logical sample index.
        unique_leaf_index
            Index of the unique leaf within the sample's draw order.

        Returns
        -------
        Tensor
            Unit-scale noise matching the leaf's shape/dtype/device.

        Raises
        ------
        AttributionError
            If the bank is present but does not cover the requested draw or
            mismatches the leaf geometry.
        """

        if self.noise_bank is not None:
            try:
                stored = self.noise_bank[sample_index][unique_leaf_index]
            except (IndexError, KeyError, TypeError) as exc:
                raise AttributionError(
                    f"noise_bank does not cover sample {sample_index}, unique "
                    f"leaf {unique_leaf_index}. Remedy: provide "
                    "noise_bank[sample][leaf] tensors for every sample and "
                    "unique attributed leaf.",
                    code="noise_bank_invalid",
                ) from exc
            if (
                not isinstance(stored, Tensor)
                or stored.shape != leaf.shape
                or stored.dtype != leaf.dtype
                or stored.device != leaf.device
            ):
                raise AttributionError(
                    f"noise_bank entry for sample {sample_index}, unique leaf "
                    f"{unique_leaf_index} must be a tensor matching the leaf's "
                    "shape, dtype, and device. Remedy: store draws taken from "
                    "the same inputs.",
                    code="noise_bank_invalid",
                )
            return stored.detach()
        generator = None
        if self.seed is not None:
            generator = self._generators.get(leaf.device)
            if generator is None:
                generator = torch.Generator(device=leaf.device)
                generator.manual_seed(self.seed)
                self._generators[leaf.device] = generator
        return torch.randn(leaf.shape, dtype=leaf.dtype, device=leaf.device, generator=generator)


def _perturbed_user_inputs(
    inputs: _PreparedInputs,
    perturbed_leaves: tuple[Tensor, ...],
) -> tuple[Any, dict[str, Any] | None]:
    """Rebuild user-shaped ``(inputs, input_kwargs)`` around perturbed leaves.

    Parameters
    ----------
    inputs
        Normalized attribution inputs from the wrapper's public call.
    perturbed_leaves
        Identity-interned replacement leaves in traversal order.

    Returns
    -------
    tuple[Any, dict[str, Any] | None]
        The child call's ``inputs`` (bare tensor when the user passed one)
        and ``input_kwargs`` (``None`` when the user passed none).
    """

    positional, kwargs = _substitute_inputs(inputs, perturbed_leaves)
    child_inputs: Any = positional[0] if inputs.is_single_tensor else positional
    child_kwargs = kwargs if inputs.kwargs else None
    return child_inputs, child_kwargs


def _noised_sample_leaves(
    inputs: _PreparedInputs,
    scales: tuple[float, ...],
    source: _NoiseSource,
    sample_index: int,
) -> tuple[Tensor, ...]:
    """Create the identity-interned noised leaves for one logical sample.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    scales
        Resolved absolute noise scale per attributed leaf.
    source
        Noise drawer (generator- or bank-backed).
    sample_index
        Zero-based logical sample index.

    Returns
    -------
    tuple[Tensor, ...]
        Perturbed leaves mirroring the original identity topology.
    """

    counter = {"next_unique": 0}

    def _make(slot: int) -> Tensor:
        """Perturb the slot's unique leaf with its resolved scale."""

        unique_index = counter["next_unique"]
        counter["next_unique"] += 1
        leaf = inputs.attributed_leaves[slot]
        noise = source.draw(leaf, sample_index, unique_index)
        return (leaf.detach() + scales[slot] * noise).detach()

    return _interned_by_identity(inputs.attributed_leaves, _make)


def _validate_tree_invariance(
    reference: Any,
    observed: Any,
    sample_index: int,
    path: str = "values",
) -> None:
    """Require one sample's value tree to mirror the reference sample's.

    Topology, per-tensor shape, and dtype/device family must match across
    samples or aggregation would silently mix incomparable objects.

    Parameters
    ----------
    reference
        Value tree of sample 0.
    observed
        Value tree of the current sample.
    sample_index
        Zero-based index of the current sample (for the refusal message).
    path
        Human-readable tree path for the refusal message.

    Raises
    ------
    AttributionError
        If the trees do not mirror each other.
    """

    def _refuse(detail: str) -> None:
        """Raise the invariance refusal with the tree-path detail."""

        raise AttributionError(
            f"sample {sample_index} (zero-based) produced a result tree that "
            f"does not mirror sample 0 at {path}: {detail}. Aggregation "
            "across samples requires invariant topology, shapes, and "
            "dtype/device. Remedy: fix the wrapped method to be "
            "shape-deterministic, or attribute the samples separately.",
            code="sample_invariance_violated",
        )

    if isinstance(reference, Tensor) or isinstance(observed, Tensor):
        if not isinstance(reference, Tensor) or not isinstance(observed, Tensor):
            _refuse(
                f"tensor vs {type(observed if isinstance(reference, Tensor) else reference).__name__}"
            )
        if reference.shape != observed.shape:
            _refuse(f"shape {tuple(observed.shape)} != {tuple(reference.shape)}")
        if reference.dtype != observed.dtype or reference.device != observed.device:
            _refuse(
                f"dtype/device ({observed.dtype}, {observed.device}) != "
                f"({reference.dtype}, {reference.device})"
            )
        return
    if type(reference) is not type(observed):
        _refuse(f"{type(observed).__name__} != {type(reference).__name__}")
    if isinstance(reference, (tuple, list)):
        if len(reference) != len(observed):
            _refuse(f"length {len(observed)} != {len(reference)}")
        for index, (ref_item, obs_item) in enumerate(zip(reference, observed, strict=True)):
            _validate_tree_invariance(ref_item, obs_item, sample_index, f"{path}[{index}]")
        return
    if isinstance(reference, dict):
        if reference.keys() != observed.keys():
            _refuse(f"keys {sorted(observed)} != {sorted(reference)}")
        for key in reference:
            _validate_tree_invariance(
                reference[key], observed[key], sample_index, f"{path}[{key!r}]"
            )
        return
    if reference is None and observed is None:
        return
    if reference != observed:
        _refuse(f"leaf {observed!r} != {reference!r}")


def _validate_result_invariance(
    reference: AttributionResult,
    observed: AttributionResult,
    sample_index: int,
) -> None:
    """Require child method / target / layer metadata invariance across samples.

    Parameters
    ----------
    reference
        Sample-0 child result.
    observed
        Current sample's child result.
    sample_index
        Zero-based index of the current sample.

    Raises
    ------
    AttributionError
        If the child method, target, or layer metadata changed mid-run.
    """

    if observed.method != reference.method:
        raise AttributionError(
            f"sample {sample_index} (zero-based) came from method "
            f"{observed.method!r} while sample 0 came from "
            f"{reference.method!r}; a wrapper aggregates ONE method. "
            "Remedy: keep the wrapped callable's method fixed across calls.",
            code="sample_invariance_violated",
        )
    if observed.target_repr != reference.target_repr:
        raise AttributionError(
            f"sample {sample_index} (zero-based) scored target "
            f"{observed.target_repr!r} while sample 0 scored "
            f"{reference.target_repr!r}. Remedy: keep the wrapped target "
            "fixed across samples.",
            code="sample_invariance_violated",
        )
    if observed.extra.get("layer") != reference.extra.get("layer"):
        raise AttributionError(
            f"sample {sample_index} (zero-based) attributed layer "
            f"{observed.extra.get('layer')!r} while sample 0 attributed "
            f"{reference.extra.get('layer')!r}. Remedy: keep the wrapped "
            "layer fixed across samples.",
            code="sample_invariance_violated",
        )
    _validate_tree_invariance(reference.values, observed.values, sample_index)


@dataclass(frozen=True)
class _SampleRun:
    """Outcome of one serial sample run.

    Attributes
    ----------
    results
        Per-sample child results, in sample order.
    planned_logical_calls
        Number of child calls the run planned.
    completed_logical_calls
        Number of child calls that completed (equals planned on success; a
        child failure raises and returns no partial result).
    child_physical_forward_calls
        Sum of the children's disclosed physical forward calls, or ``None``
        when any child does not disclose one.
    """

    results: list[AttributionResult]
    planned_logical_calls: int
    completed_logical_calls: int
    child_physical_forward_calls: int | None


def _run_serial_samples(
    n_samples: int,
    call_sample: Callable[[int], AttributionResult],
) -> _SampleRun:
    """Run ``call_sample`` serially for every sample with failure naming.

    Parameters
    ----------
    n_samples
        Number of logical samples to run.
    call_sample
        Callable invoked with the zero-based sample index.

    Returns
    -------
    _SampleRun
        The completed run.

    Raises
    ------
    AttributionError
        If a child call fails; the refusal names the zero-based sample and no
        partial result is returned (memo D4).
    """

    results: list[AttributionResult] = []
    for sample_index in range(n_samples):
        try:
            result = call_sample(sample_index)
        except AttributionError as exc:
            # Tunnel-owned refusals pass through verbatim: cross-sample
            # invariance is the runner's own verdict, and the route-unbound
            # internal tripwire fires in tunnel code BEFORE any child call,
            # so "fix the wrapped method" would be the wrong teaching.
            if exc.fields.get("code") in (
                "sample_invariance_violated",
                "noise_tunnel_route_unbound",
            ):
                raise
            raise AttributionError(
                f"sample {sample_index} (zero-based) of {n_samples} failed: "
                f"{exc}. No partial result is returned. Remedy: fix the "
                "wrapped method for this input, or reduce the perturbation.",
                code="sample_child_failed",
                failed_sample=sample_index,
            ) from exc
        except Exception as exc:
            raise AttributionError(
                f"sample {sample_index} (zero-based) of {n_samples} failed: "
                f"{exc}. No partial result is returned. Remedy: fix the "
                "wrapped method for this input, or reduce the perturbation.",
                code="sample_child_failed",
                failed_sample=sample_index,
            ) from exc
        if results:
            _validate_result_invariance(results[0], result, sample_index)
        results.append(result)
    physical: int | None
    disclosed_counts = [result.extra.get("physical_forward_calls") for result in results]
    if all(isinstance(count, int) for count in disclosed_counts):
        physical = sum(count for count in disclosed_counts if isinstance(count, int))
    else:
        physical = None
    return _SampleRun(
        results=results,
        planned_logical_calls=n_samples,
        completed_logical_calls=len(results),
        child_physical_forward_calls=physical,
    )


def _aggregate_value_trees(trees: list[Any], aggregation: str) -> Any:
    """Aggregate per-sample value trees leafwise.

    Parameters
    ----------
    trees
        Per-sample value trees with validated invariant topology.
    aggregation
        ``"mean"``, ``"mean_square"``, or ``"variance"`` (population).

    Returns
    -------
    Any
        Aggregated tree mirroring the sample topology.
    """

    head = trees[0]
    if isinstance(head, Tensor):
        stacked = torch.stack(list(trees), dim=0)
        if aggregation == "mean":
            return stacked.mean(dim=0)
        if aggregation == "mean_square":
            return (stacked.abs() ** 2).mean(dim=0)
        return stacked.var(dim=0, unbiased=False)
    if isinstance(head, tuple):
        return tuple(
            _aggregate_value_trees([tree[index] for tree in trees], aggregation)
            for index in range(len(head))
        )
    if isinstance(head, list):
        return [
            _aggregate_value_trees([tree[index] for tree in trees], aggregation)
            for index in range(len(head))
        ]
    if isinstance(head, dict):
        return {
            key: _aggregate_value_trees([tree[key] for tree in trees], aggregation) for key in head
        }
    return head


def _scalar_distribution(values: list[float]) -> dict[str, float]:
    """Return the mean/std/min/max disclosure for one per-sample scalar.

    Parameters
    ----------
    values
        Per-sample scalar values.

    Returns
    -------
    dict[str, float]
        Population statistics of the sample scalars.
    """

    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def _completeness_distributions(results: list[AttributionResult]) -> dict[str, Any]:
    """Build the D4 per-sample completeness distributions when disclosed.

    Parameters
    ----------
    results
        Per-sample child results.

    Returns
    -------
    dict[str, Any]
        ``attribution_sum`` / ``target_delta`` / ``completeness_residual``
        distributions plus the residual of means, or a named absence
        disclosure when the child does not carry completeness fields.
    """

    keys = ("attribution_sum", "target_delta", "completeness_residual")
    if not all(all(key in result.extra for key in keys) for result in results):
        return {"available": False, "reason": "child results carry no completeness fields"}
    sums = [float(result.extra["attribution_sum"]) for result in results]
    deltas = [float(result.extra["target_delta"]) for result in results]
    residuals = [float(result.extra["completeness_residual"]) for result in results]
    mean_sum = sum(sums) / len(sums)
    mean_delta = sum(deltas) / len(deltas)
    return {
        "available": True,
        "attribution_sum": _scalar_distribution(sums),
        "target_delta": _scalar_distribution(deltas),
        "completeness_residual": _scalar_distribution(residuals),
        "residual_of_means": mean_sum - mean_delta,
    }


__all__ = [
    "_NoiseSource",
    "_SampleRun",
    "_aggregate_value_trees",
    "_completeness_distributions",
    "_noised_sample_leaves",
    "_perturbed_user_inputs",
    "_resolve_stdevs",
    "_run_serial_samples",
    "_scalar_distribution",
    "_validate_result_invariance",
    "_validate_tree_invariance",
]

"""GradientShap: expected gradients over a baseline pool (attrib memo D10).

The estimator convention is PINNED to captum's ``InputBaselineXGradient``,
verified against captum source before implementation and held by the
stored-draw oracle row: per sample, per example, draw one baseline from the
REQUIRED aligned pool (one pool index shared across every attributed leaf),
optionally noise the input (``stdevs`` absolute, default 0), interpolate
``baseline + alpha * (noised_input - baseline)`` with ``alpha ~ U(0, 1)`` per
example, take the target gradient AT the interpolated point, and multiply by
``(noised_input - baseline)``. The attribution is the sample mean.

The residual disclosure is labeled MONTE-CARLO DIAGNOSTICS, never a
completeness guarantee: SHAP completeness holds only in expectation over the
baseline distribution, so a finite-sample residual measures estimator noise,
not correctness.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._core import (
    InputKwargs,
    _AttributionTarget,
    _call_model,
    _gradient_for_inputs,
    _interned_by_identity,
    _normalize_model_inputs,
    _PreparedInputs,
    _scalarize_output,
    _target_repr,
    _temporarily_eval,
    _validate_positive_int,
    _value_tree_from_leaves,
)
from torchlens.attribution._result import AttributionError, AttributionResult
from torchlens.attribution._sampling import (
    _NoiseSource,
    _scalar_distribution,
)


def _tensor_digest(tensor: Tensor) -> str:
    """Return a short sha256 digest of a tensor's bytes for disclosure.

    Parameters
    ----------
    tensor
        Tensor to digest (moved to CPU, contiguous).

    Returns
    -------
    str
        First 16 hex characters of the sha256 of the raw bytes.
    """

    data = tensor.detach().cpu().contiguous()
    return hashlib.sha256(bytes(data.numpy().data)).hexdigest()[:16]


def _validate_baseline_pool(
    inputs: _PreparedInputs,
    baselines: Any,
) -> tuple[tuple[Tensor, ...], int]:
    """Validate the required aligned baseline pool (leading pool axis).

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    baselines
        The pool is a stack of baseline EXAMPLES: each leaf's leading axis is
        the pool axis and its trailing dimensions match the input leaf's
        trailing (non-batch) dimensions. A bare tensor is accepted for a
        single attributed leaf; otherwise a tree mirroring the input structure
        whose leaves each carry the same leading pool size.

    Returns
    -------
    tuple[tuple[Tensor, ...], int]
        Per-attributed-leaf pool tensors (traversal order) and the shared
        pool size ``P``.

    Raises
    ------
    AttributionError
        If the pool is missing, misaligned, or has inconsistent pool sizes.
    """

    if baselines is None:
        raise AttributionError(
            "gradient_shap requires a baseline pool: a tensor (or tree of "
            "tensors mirroring the attributed inputs) whose LEADING axis "
            "stacks baseline examples. Remedy: pass baselines=<pool>, e.g. a "
            "stack of natural images or zero/blur references.",
            code="gradient_shap_pool_invalid",
        )
    leaves = inputs.attributed_leaves
    if len(leaves) == 1 and isinstance(baselines, Tensor):
        pool_leaves: list[Tensor] = [baselines]
    else:
        # Reuse the baseline-tree walker with pool tensors in leaf positions;
        # shape validation is pool-aware, so bypass the per-tensor check by
        # collecting tensors structurally first.
        pool_leaves = _collect_pool_tree(inputs, baselines)
    if len(pool_leaves) != len(leaves):
        raise AttributionError(
            f"baseline pool has {len(pool_leaves)} leaves but the call has "
            f"{len(leaves)} attributed leaves. Remedy: mirror the attributed "
            "input structure.",
            code="gradient_shap_pool_invalid",
        )
    pool_size: int | None = None
    validated: list[Tensor] = []
    for leaf, pool in zip(leaves, pool_leaves, strict=True):
        if not isinstance(pool, Tensor):
            raise AttributionError(
                "baseline pool leaves must be tensors. Remedy: mirror the "
                "attributed input structure with pool tensors.",
                code="gradient_shap_pool_invalid",
            )
        if pool.ndim != leaf.ndim or tuple(pool.shape[1:]) != tuple(leaf.shape[1:]):
            raise AttributionError(
                f"baseline pool leaf of shape {tuple(pool.shape)} does not "
                f"align with input leaf of shape {tuple(leaf.shape)}: the pool "
                "stacks baseline EXAMPLES on the leading axis, so trailing "
                "dimensions must match the input's. Remedy: build the pool as "
                "torch.stack([...]) of input-shaped examples without the "
                "batch axis collapsed.",
                code="gradient_shap_pool_invalid",
            )
        if pool.dtype != leaf.dtype or pool.device != leaf.device:
            raise AttributionError(
                "baseline pool leaves must match the input leaf dtype and "
                "device. Remedy: cast/move the pool to the input's "
                "dtype/device.",
                code="gradient_shap_pool_invalid",
            )
        if pool.shape[0] < 1:
            raise AttributionError(
                "baseline pool must contain at least one example. Remedy: "
                "stack one or more baseline examples on the leading axis.",
                code="gradient_shap_pool_invalid",
            )
        if pool_size is None:
            pool_size = int(pool.shape[0])
        elif int(pool.shape[0]) != pool_size:
            raise AttributionError(
                "every baseline pool leaf must share ONE pool size: each "
                "example draws one pool index shared across all attributed "
                "leaves. Remedy: give every leaf the same number of pool "
                "entries.",
                code="gradient_shap_pool_invalid",
            )
        validated.append(pool.detach())
    # ``validated`` is non-empty here (the leaf walk refused earlier
    # otherwise), so ``pool_size`` is always resolved.
    return tuple(validated), int(pool_size if pool_size is not None else 0)


def _collect_pool_tree(inputs: _PreparedInputs, baselines: Any) -> list[Tensor]:
    """Collect pool tensors from a tree mirroring the input structure.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    baselines
        Pool tree mirroring the user's input structure.

    Returns
    -------
    list[Tensor]
        Pool tensors in attributed-leaf traversal order.

    Raises
    ------
    AttributionError
        If the container structure does not mirror the inputs.
    """

    original_error: AttributionError | None = None
    try:
        if inputs.kwargs:
            if not isinstance(baselines, dict) or set(baselines) != {"inputs", "input_kwargs"}:
                raise AttributionError(
                    "baselines must be a dict with 'inputs' and 'input_kwargs' "
                    "for kwarg inputs. Remedy: mirror the call structure.",
                    code="gradient_shap_pool_invalid",
                )
            return _walk_pool(
                (inputs.positional, inputs.kwargs),
                (baselines["inputs"], baselines["input_kwargs"]),
            )
        positional = tuple(baselines) if isinstance(baselines, list) else baselines
        return _walk_pool(inputs.positional, positional)
    except AttributionError as exc:
        original_error = exc
        raise AttributionError(
            f"baseline pool tree does not mirror the attributed inputs: "
            f"{original_error}. Remedy: mirror the input containers with pool "
            "tensors in attributed positions.",
            code="gradient_shap_pool_invalid",
        ) from exc


def _walk_pool(input_tree: Any, pool_tree: Any) -> list[Tensor]:
    """Walk mirrored trees collecting pool tensors at attributed positions.

    Parameters
    ----------
    input_tree
        The user's input tree.
    pool_tree
        The candidate pool tree.

    Returns
    -------
    list[Tensor]
        Pool tensors in traversal order.

    Raises
    ------
    AttributionError
        If containers do not mirror.
    """

    from torchlens.attribution._core import _is_attributed_tensor

    if _is_attributed_tensor(input_tree):
        return [pool_tree]
    if isinstance(input_tree, (tuple, list)):
        if not isinstance(pool_tree, (tuple, list)) or len(pool_tree) != len(input_tree):
            raise AttributionError(
                "pool containers must mirror input containers",
                code="gradient_shap_pool_invalid",
            )
        collected: list[Tensor] = []
        for input_item, pool_item in zip(input_tree, pool_tree, strict=True):
            collected.extend(_walk_pool(input_item, pool_item))
        return collected
    if isinstance(input_tree, dict):
        if not isinstance(pool_tree, dict) or pool_tree.keys() != input_tree.keys():
            raise AttributionError(
                "pool containers must mirror input containers",
                code="gradient_shap_pool_invalid",
            )
        collected = []
        for key, input_value in input_tree.items():
            collected.extend(_walk_pool(input_value, pool_tree[key]))
        return collected
    return []


def _validate_draw_bank(
    draw_bank: dict[str, Any] | None,
    n_samples: int,
    batch_size: int,
    pool_size: int,
) -> None:
    """Validate a stored-draw bank's geometry (the oracle-row injection door).

    Parameters
    ----------
    draw_bank
        Optional dict with ``"alphas"`` (``[n_samples, batch]`` in [0, 1]),
        ``"pool_indices"`` (``[n_samples, batch]`` long, in range), and
        optionally ``"noise"`` (``noise[sample][unique_leaf]`` unit draws).
    n_samples
        Number of logical samples.
    batch_size
        Number of input examples.
    pool_size
        Pool size ``P``.

    Raises
    ------
    AttributionError
        If the bank geometry is wrong.
    """

    if draw_bank is None:
        return
    if not isinstance(draw_bank, dict) or not {"alphas", "pool_indices"} <= set(draw_bank):
        raise AttributionError(
            "draw_bank must be a dict with 'alphas' and 'pool_indices' (and "
            "optionally 'noise'). Remedy: store the draws with "
            "store_draws=True and replay that record.",
            code="gradient_shap_draw_bank_invalid",
        )
    alphas = draw_bank["alphas"]
    indices = draw_bank["pool_indices"]
    if (
        not isinstance(alphas, Tensor)
        or tuple(alphas.shape) != (n_samples, batch_size)
        or bool((alphas < 0).any())
        or bool((alphas > 1).any())
    ):
        raise AttributionError(
            f"draw_bank['alphas'] must be a [{n_samples}, {batch_size}] tensor "
            "in [0, 1]. Remedy: replay a bank stored by store_draws=True.",
            code="gradient_shap_draw_bank_invalid",
        )
    if (
        not isinstance(indices, Tensor)
        or tuple(indices.shape) != (n_samples, batch_size)
        or indices.dtype != torch.long
        or bool((indices < 0).any())
        or bool((indices >= pool_size).any())
    ):
        raise AttributionError(
            f"draw_bank['pool_indices'] must be a [{n_samples}, {batch_size}] "
            f"long tensor with values in [0, {pool_size}). Remedy: replay a "
            "bank stored by store_draws=True.",
            code="gradient_shap_draw_bank_invalid",
        )


def _example_axis_view(values: Tensor, leaf: Tensor) -> Tensor:
    """Reshape per-example draw values to broadcast over one leaf.

    Parameters
    ----------
    values
        Per-example values of shape ``[batch]``.
    leaf
        Leaf whose leading axis is the example axis.

    Returns
    -------
    Tensor
        ``values`` viewed as ``[batch, 1, 1, ...]`` on the leaf's device/dtype
        family (float draws cast to the leaf dtype; long indices unchanged).
    """

    shape = (leaf.shape[0],) + (1,) * (leaf.ndim - 1)
    return values.to(device=leaf.device).reshape(shape)


def gradient_shap(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    baselines: Any,
    n_samples: int = 25,
    stdevs: float = 0.0,
    seed: int | None = None,
    draw_bank: dict[str, Any] | None = None,
    store_draws: bool = False,
) -> AttributionResult:
    """Estimate expected gradients against a baseline pool (GradientShap).

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model
        arguments; floating/complex leaves are attributed. The leading axis of
        every attributed leaf is the EXAMPLE axis.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index or callable ``output -> scalar tensor``.
    baselines
        REQUIRED aligned baseline pool: a tensor whose leading axis stacks
        baseline examples (single attributed leaf), or a tree mirroring the
        input structure whose leaves share one pool size. Each example draws
        ONE pool index per sample, shared across every leaf (modality).
    n_samples
        Number of Monte-Carlo samples (default 25, captum parity).
    stdevs
        ABSOLUTE Gaussian input-noise scale (default 0.0 -- zero noise).
    seed
        Optional deterministic seed for alphas, pool draws, and noise.
    draw_bank
        Optional stored draws (``alphas``, ``pool_indices``, optional
        ``noise``) replayed instead of fresh draws; the stored-draw oracle row
        pins the estimator convention through this door.
    store_draws
        Whether to place the full draw record in ``extra["draws"]`` (off by
        default; ordinary results keep only digests and distributions).

    Returns
    -------
    AttributionResult
        Sample-mean attributions with seed, pool/sample digests, per-sample
        diagnostic distributions, and the Monte-Carlo residual disclosure
        (labeled diagnostics, never a completeness guarantee).

    Raises
    ------
    AttributionError
        On pool/draw-bank misalignment or invalid sampling parameters.
    """

    _validate_positive_int("n_samples", n_samples)
    if isinstance(stdevs, bool) or not isinstance(stdevs, (int, float)) or stdevs < 0:
        raise AttributionError(
            "stdevs must be a nonnegative float (absolute noise scale). "
            "Remedy: pass stdevs=0.0 for noiseless draws.",
            code="noise_stdevs_invalid",
        )
    prepared = _normalize_model_inputs(inputs, input_kwargs)
    pool_leaves, pool_size = _validate_baseline_pool(prepared, baselines)
    batch_size = int(prepared.attributed_leaves[0].shape[0])
    for leaf in prepared.attributed_leaves:
        if leaf.ndim == 0 or int(leaf.shape[0]) != batch_size:
            raise AttributionError(
                "gradient_shap requires every attributed leaf to share one "
                "leading example axis. Remedy: batch the inputs consistently.",
                code="gradient_shap_pool_invalid",
            )
    _validate_draw_bank(draw_bank, n_samples, batch_size, pool_size)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    noise_source = _NoiseSource(seed, None)

    contributions_by_sample: list[tuple[Tensor, ...]] = []
    sample_sums: list[float] = []
    sample_deltas: list[float] = []
    drawn_alphas: list[Tensor] = []
    drawn_indices: list[Tensor] = []
    drawn_noise: list[list[Tensor]] = []
    physical_calls = 0

    with _temporarily_eval(model):
        for sample_index in range(n_samples):
            if draw_bank is not None:
                alphas = draw_bank["alphas"][sample_index].to(torch.float64)
                indices = draw_bank["pool_indices"][sample_index]
            else:
                alphas = torch.rand(batch_size, generator=generator, dtype=torch.float64)
                indices = torch.randint(0, pool_size, (batch_size,), generator=generator)
            drawn_alphas.append(alphas)
            drawn_indices.append(indices)

            bank_noise = None
            if draw_bank is not None:
                bank_noise = draw_bank.get("noise")
            noise_record: list[Tensor] = []
            counter = {"next_unique": 0}
            noised_by_id: dict[int, Tensor] = {}
            baseline_by_id: dict[int, Tensor] = {}

            def _make_interpolated(
                slot: int,
                *,
                _counter: dict[str, int] = counter,
                _bank_noise: Any = bank_noise,
                _sample_index: int = sample_index,
                _noise_record: list[Tensor] = noise_record,
                _indices: Tensor = indices,
                _alphas: Tensor = alphas,
                _noised_by_id: dict[int, Tensor] = noised_by_id,
                _baseline_by_id: dict[int, Tensor] = baseline_by_id,
            ) -> Tensor:
                """Build one interpolated leaf for this sample (captum convention).

                The keyword defaults BIND this sample's loop state at definition
                time (the B023 discipline): the closure is handed to the
                identity-interning helper, so late binding would silently mix
                samples.
                """

                leaf = prepared.attributed_leaves[slot]
                unique_index = _counter["next_unique"]
                _counter["next_unique"] += 1
                if stdevs > 0:
                    if _bank_noise is not None:
                        noise = _bank_noise[_sample_index][unique_index].detach()
                    else:
                        noise = noise_source.draw(leaf, _sample_index, unique_index)
                    _noise_record.append(noise)
                    noised = leaf.detach() + stdevs * noise
                else:
                    noised = leaf.detach()
                drawn_baseline = pool_leaves[slot][_indices.to(pool_leaves[slot].device)]
                alpha_view = _example_axis_view(_alphas, leaf).to(leaf.dtype)
                interpolated = drawn_baseline + alpha_view * (noised - drawn_baseline)
                _noised_by_id[id(leaf)] = noised
                _baseline_by_id[id(leaf)] = drawn_baseline
                return interpolated.detach().clone().requires_grad_(True)

            interpolated_leaves = _interned_by_identity(
                prepared.attributed_leaves, _make_interpolated
            )
            if stdevs > 0:
                drawn_noise.append(noise_record)
            gradients, _scalar = _gradient_for_inputs(model, prepared, interpolated_leaves, target)
            physical_calls += 1
            contributions = tuple(
                (gradient * (noised_by_id[id(leaf)] - baseline_by_id[id(leaf)])).detach()
                for gradient, leaf in zip(gradients, prepared.attributed_leaves, strict=True)
            )
            contributions_by_sample.append(contributions)
            sample_sums.append(float(sum(c.sum() for c in contributions)))

            # Monte-Carlo diagnostic endpoints: target at the noised input and
            # at the drawn baseline (no-grad forwards, disclosed in the cost).
            def _noised_leaf(slot: int, _by_id: dict[int, Tensor] = noised_by_id) -> Tensor:
                """Reuse this sample's noised leaf (bound at definition, B023)."""

                return _by_id[id(prepared.attributed_leaves[slot])]

            def _baseline_leaf(slot: int, _by_id: dict[int, Tensor] = baseline_by_id) -> Tensor:
                """Reuse this sample's drawn baseline (bound at definition, B023)."""

                return _by_id[id(prepared.attributed_leaves[slot])]

            noised_leaves = _interned_by_identity(prepared.attributed_leaves, _noised_leaf)
            baseline_sample_leaves = _interned_by_identity(
                prepared.attributed_leaves, _baseline_leaf
            )
            with torch.no_grad():
                noised_scalar = _scalarize_output(
                    _call_model(model, prepared, noised_leaves), target
                )
                baseline_scalar = _scalarize_output(
                    _call_model(model, prepared, baseline_sample_leaves), target
                )
            physical_calls += 2
            sample_deltas.append(float(noised_scalar - baseline_scalar))

    values = tuple(
        torch.stack([sample[index] for sample in contributions_by_sample], dim=0).mean(dim=0)
        for index in range(len(prepared.attributed_leaves))
    )
    mean_sum = sum(sample_sums) / len(sample_sums)
    mean_delta = sum(sample_deltas) / len(sample_deltas)
    extra: dict[str, Any] = {
        "n_samples": n_samples,
        "stdevs": float(stdevs),
        "seed": seed,
        "pool_size": pool_size,
        "pool_digests": [_tensor_digest(pool) for pool in pool_leaves],
        "sample_digest": _tensor_digest(
            torch.cat(
                [
                    torch.stack(drawn_alphas).flatten(),
                    torch.stack(drawn_indices).flatten().to(torch.float64),
                ]
            )
        ),
        "estimator": "captum InputBaselineXGradient convention: grad at "
        "interpolated point times (noised input minus drawn baseline)",
        "monte_carlo_diagnostics": {
            "labeled": "diagnostics, never a completeness guarantee",
            "attribution_sum": _scalar_distribution(sample_sums),
            "target_delta": _scalar_distribution(sample_deltas),
            "residual_of_means": mean_sum - mean_delta,
        },
        "physical_forward_calls": physical_calls,
    }
    if store_draws:
        extra["draws"] = {
            "alphas": torch.stack(drawn_alphas),
            "pool_indices": torch.stack(drawn_indices),
            "noise": drawn_noise if stdevs > 0 else None,
        }
    return AttributionResult(
        method="gradient_shap",
        values=_value_tree_from_leaves(prepared, values),
        target_repr=_target_repr(target),
        extra=extra,
    )


__all__ = ["gradient_shap"]

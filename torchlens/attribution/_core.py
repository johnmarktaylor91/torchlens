"""Native input-attribution methods for TorchLens."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
from torch import Tensor
from torch.nn import Module


TargetSpec: TypeAlias = int | Callable[[Any], Tensor]
AttributionValueTree: TypeAlias = Tensor | tuple[Any, ...] | list[Any] | dict[str, Any]
InputKwargs: TypeAlias = dict[str, Any] | None


class AttributionError(ValueError):
    """Error raised for unsupported or invalid attribution requests."""


@dataclass(frozen=True)
class AttributionResult:
    """Container for provisional input-attribution results.

    Parameters
    ----------
    method
        Name of the attribution method that produced this result.
    values
        Attribution values. Single attributed-leaf calls return a bare tensor for
        v1 compatibility. Multi-leaf calls return a nested structure with
        attribution tensors in attributed positions and ``None`` in non-attributed
        leaf positions.
    target_repr
        Compact representation of the scalarization target.
    extra
        Method-specific metadata. This schema is intentionally minimal for v1.
    """

    method: str
    values: AttributionValueTree
    target_repr: str
    extra: dict[str, Any]


@dataclass(frozen=True)
class _PreparedInputs:
    """Normalized model-call inputs and attributed leaves.

    Attributes
    ----------
    positional
        Positional argument structure used for ``model(*args)``.
    kwargs
        Keyword argument structure used for ``model(**kwargs)``.
    attributed_leaves
        Floating-point or complex tensor leaves selected for attribution.
    is_single_tensor
        Whether the public ``inputs`` argument was a bare tensor.
    """

    positional: tuple[Any, ...]
    kwargs: dict[str, Any]
    attributed_leaves: tuple[Tensor, ...]
    is_single_tensor: bool


@contextmanager
def _temporarily_eval(model: Module) -> Any:
    """Run attribution with ``model`` in eval mode, then restore its prior mode.

    Parameters
    ----------
    model
        Model whose training mode should be temporarily changed.

    Yields
    ------
    None
        Context body executes while the model is in eval mode.
    """

    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _is_attributed_tensor(value: Any) -> bool:
    """Return whether ``value`` is a tensor leaf selected for attribution.

    Parameters
    ----------
    value
        Candidate pytree leaf.

    Returns
    -------
    bool
        ``True`` when ``value`` is a floating-point or complex tensor.
    """

    return isinstance(value, Tensor) and (value.is_floating_point() or value.is_complex())


def _normalize_model_inputs(inputs: Any, input_kwargs: InputKwargs) -> _PreparedInputs:
    """Normalize public attribution inputs into model-call args and kwargs.

    Parameters
    ----------
    inputs
        Bare tensor or tuple/list of positional arguments.
    input_kwargs
        Optional keyword arguments for the model call.

    Returns
    -------
    _PreparedInputs
        Normalized call structure and attributed tensor leaves.

    Raises
    ------
    AttributionError
        If the input contract is invalid or no attributed leaves are present.
    """

    if input_kwargs is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(input_kwargs, dict):
        kwargs = dict(input_kwargs)
    else:
        raise AttributionError("input_kwargs must be a dict when provided")

    is_single_tensor = isinstance(inputs, Tensor)
    if is_single_tensor:
        positional = (inputs,)
    elif isinstance(inputs, tuple | list):
        positional = tuple(inputs)
    else:
        raise AttributionError(
            "inputs must be a tensor or a tuple/list of positional model arguments"
        )

    attributed_leaves = tuple(_iter_attributed_tensors((positional, kwargs)))
    if not attributed_leaves:
        raise AttributionError(
            "input attribution requires at least one floating-point or complex tensor leaf"
        )
    return _PreparedInputs(
        positional=positional,
        kwargs=kwargs,
        attributed_leaves=attributed_leaves,
        is_single_tensor=is_single_tensor,
    )


def _iter_attributed_tensors(tree: Any) -> list[Tensor]:
    """Return attributed tensor leaves in deterministic traversal order.

    Parameters
    ----------
    tree
        Pytree made from tuples, lists, dicts, and leaves.

    Returns
    -------
    list[Tensor]
        Floating-point or complex tensor leaves found in ``tree``.
    """

    if _is_attributed_tensor(tree):
        return [tree]
    if isinstance(tree, tuple | list):
        leaves: list[Tensor] = []
        for item in tree:
            leaves.extend(_iter_attributed_tensors(item))
        return leaves
    if isinstance(tree, dict):
        leaves = []
        for value in tree.values():
            leaves.extend(_iter_attributed_tensors(value))
        return leaves
    return []


def _make_input_leaf(inputs: Tensor) -> Tensor:
    """Create a detached leaf tensor for attribution gradients.

    Parameters
    ----------
    inputs
        User input tensor.

    Returns
    -------
    Tensor
        Detached clone with gradient tracking enabled.
    """

    return inputs.detach().clone().requires_grad_(True)


def _make_input_leaves(inputs: _PreparedInputs) -> tuple[Tensor, ...]:
    """Create detached leaf tensors for all attributed input leaves.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.

    Returns
    -------
    tuple[Tensor, ...]
        Detached clones with gradient tracking enabled.
    """

    return tuple(_make_input_leaf(leaf) for leaf in inputs.attributed_leaves)


def _replace_attributed_tensors(tree: Any, replacements: list[Tensor]) -> Any:
    """Replace attributed tensor leaves in ``tree`` from ``replacements``.

    Parameters
    ----------
    tree
        Pytree made from tuples, lists, dicts, and leaves.
    replacements
        Replacement tensors consumed in traversal order.

    Returns
    -------
    Any
        Tree with attributed leaves replaced.
    """

    if _is_attributed_tensor(tree):
        return replacements.pop(0)
    if isinstance(tree, tuple):
        return tuple(_replace_attributed_tensors(item, replacements) for item in tree)
    if isinstance(tree, list):
        return [_replace_attributed_tensors(item, replacements) for item in tree]
    if isinstance(tree, dict):
        return {
            key: _replace_attributed_tensors(value, replacements) for key, value in tree.items()
        }
    return tree


def _substitute_inputs(
    inputs: _PreparedInputs,
    attributed_replacements: tuple[Tensor, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Substitute attributed leaves into normalized model-call inputs.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    attributed_replacements
        Replacement leaves in the same order as ``inputs.attributed_leaves``.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Positional and keyword arguments ready for ``model(*args, **kwargs)``.
    """

    replacements = list(attributed_replacements)
    positional = _replace_attributed_tensors(inputs.positional, replacements)
    kwargs = _replace_attributed_tensors(inputs.kwargs, replacements)
    if replacements:
        raise AttributionError("internal error: not all attributed replacements were consumed")
    if not isinstance(positional, tuple):
        raise AttributionError("internal error: positional inputs did not remain a tuple")
    if not isinstance(kwargs, dict):
        raise AttributionError("internal error: keyword inputs did not remain a dict")
    return positional, kwargs


def _call_model(
    model: Module,
    inputs: _PreparedInputs,
    attributed_replacements: tuple[Tensor, ...],
) -> Any:
    """Call ``model`` with attributed leaves substituted into their original positions.

    Parameters
    ----------
    model
        PyTorch module to evaluate.
    inputs
        Normalized attribution inputs.
    attributed_replacements
        Replacement leaves in the same order as ``inputs.attributed_leaves``.

    Returns
    -------
    Any
        Model output.
    """

    positional, kwargs = _substitute_inputs(inputs, attributed_replacements)
    return model(*positional, **kwargs)


def _value_tree_from_leaves(
    inputs: _PreparedInputs,
    values: tuple[Tensor, ...],
) -> AttributionValueTree:
    """Build the public attribution value structure from per-leaf tensors.

    Single attributed-leaf calls return a bare tensor for v1 compatibility.
    Multi-leaf positional-only calls return a positional structure. Multi-leaf
    calls with keyword inputs return ``{"inputs": ..., "input_kwargs": ...}``.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    values
        Attribution tensors in attributed-leaf traversal order.

    Returns
    -------
    AttributionValueTree
        Public ``AttributionResult.values`` structure.
    """

    if len(values) == 1:
        return values[0]

    replacements = list(values)
    positional_values = _replace_unattributed_with_none(inputs.positional, replacements)
    kwargs_values = _replace_unattributed_with_none(inputs.kwargs, replacements)
    if replacements:
        raise AttributionError("internal error: not all attribution values were consumed")
    if inputs.kwargs:
        return {"inputs": positional_values, "input_kwargs": kwargs_values}
    if inputs.is_single_tensor:
        return positional_values[0]
    return positional_values


def _replace_unattributed_with_none(tree: Any, replacements: list[Tensor]) -> Any:
    """Replace attributed leaves with values and all other leaves with ``None``.

    Parameters
    ----------
    tree
        Pytree made from tuples, lists, dicts, and leaves.
    replacements
        Replacement tensors consumed in traversal order.

    Returns
    -------
    Any
        Value tree mirroring containers in ``tree``.
    """

    if _is_attributed_tensor(tree):
        return replacements.pop(0)
    if isinstance(tree, tuple):
        return tuple(_replace_unattributed_with_none(item, replacements) for item in tree)
    if isinstance(tree, list):
        return [_replace_unattributed_with_none(item, replacements) for item in tree]
    if isinstance(tree, dict):
        return {
            key: _replace_unattributed_with_none(value, replacements) for key, value in tree.items()
        }
    return None


def _target_repr(target: TargetSpec) -> str:
    """Return a compact target representation for result metadata.

    Parameters
    ----------
    target
        Target scalarization specification.

    Returns
    -------
    str
        Human-readable target representation.
    """

    if isinstance(target, int):
        return f"index={target}"
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        return name
    return repr(target)


def _scalarize_output(output: Any, target: TargetSpec) -> Tensor:
    """Convert a model output to a scalar tensor using ``target``.

    Integer targets select ``output[..., target]`` and sum all selected values to
    produce one scalar. Callable targets must return a scalar tensor and are the
    required form for ambiguous non-classification outputs.

    Parameters
    ----------
    output
        Model output.
    target
        Integer class index or callable scalarizer.

    Returns
    -------
    Tensor
        Scalar tensor suitable for ``torch.autograd.grad``.

    Raises
    ------
    AttributionError
        If scalarization is unsupported or ambiguous.
    """

    if callable(target):
        scalar = target(output)
        if not isinstance(scalar, Tensor):
            raise AttributionError("callable target must return a scalar tensor")
        if scalar.numel() != 1:
            raise AttributionError("callable target must return a scalar tensor")
        return scalar.reshape(())

    if not isinstance(target, int):
        raise AttributionError(
            "target must be an int class index or a callable output -> scalar tensor"
        )
    if not isinstance(output, Tensor):
        raise AttributionError("int targets require tensor model outputs; use a callable target")
    if output.ndim == 0:
        raise AttributionError("int target is ambiguous for scalar outputs; use a callable target")

    try:
        selected = output[..., target]
    except IndexError as exc:
        raise AttributionError(f"target index {target} is out of bounds for output shape") from exc
    return selected.sum()


def _gradient_for_inputs(
    model: Module,
    inputs: _PreparedInputs,
    input_leaves: tuple[Tensor, ...],
    target: TargetSpec,
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Compute gradients of a scalarized model output with respect to input leaves.

    Parameters
    ----------
    model
        Model to evaluate.
    inputs
        Normalized attribution inputs.
    input_leaves
        Differentiable leaves substituted into ``inputs``.
    target
        Integer class index or callable scalarizer.

    Returns
    -------
    tuple[tuple[Tensor, ...], Tensor]
        Gradient tensors and scalarized model output.

    Raises
    ------
    AttributionError
        If the scalar target is not differentiable with respect to the inputs.
    """

    output = _call_model(model, inputs, input_leaves)
    scalar = _scalarize_output(output, target)
    try:
        raw_gradients = torch.autograd.grad(
            scalar,
            input_leaves,
            allow_unused=True,
        )
    except RuntimeError as exc:
        raise AttributionError(
            "target scalar is not differentiable with respect to the attributed inputs"
        ) from exc
    gradients = tuple(
        torch.zeros_like(input_leaf) if gradient is None else gradient
        for input_leaf, gradient in zip(input_leaves, raw_gradients, strict=True)
    )
    return gradients, scalar.detach()


def _validate_positive_int(name: str, value: int) -> None:
    """Validate that a method count parameter is positive.

    Parameters
    ----------
    name
        Parameter name for error reporting.
    value
        Candidate integer value.

    Raises
    ------
    AttributionError
        If ``value`` is not a positive integer.
    """

    if not isinstance(value, int) or value <= 0:
        raise AttributionError(f"{name} must be a positive integer")


def _validate_baseline_tensor(input_tensor: Tensor, baseline: Any) -> Tensor:
    """Validate one baseline tensor against one attributed input tensor.

    The default zero baseline is conventional but not neutral for every problem;
    attribution users should choose a baseline that matches their domain.

    Parameters
    ----------
    input_tensor
        Attributed input tensor.
    baseline
        Candidate baseline tensor.

    Returns
    -------
    Tensor
        Detached baseline tensor matching input shape, dtype, and device.

    Raises
    ------
    AttributionError
        If the baseline does not match ``input_tensor``.
    """

    if not isinstance(baseline, Tensor):
        raise AttributionError("baseline must mirror attributed input leaves")
    if baseline.shape != input_tensor.shape:
        raise AttributionError("baseline must match input shape")
    if baseline.dtype != input_tensor.dtype:
        raise AttributionError("baseline must match input dtype")
    if baseline.device != input_tensor.device:
        raise AttributionError("baseline must match input device")
    return baseline.detach().clone()


def _validate_baseline_tree(input_tree: Any, baseline_tree: Any) -> list[Tensor]:
    """Validate a baseline pytree against an input pytree.

    Parameters
    ----------
    input_tree
        Input tree containing attributed leaves.
    baseline_tree
        Candidate baseline tree.

    Returns
    -------
    list[Tensor]
        Baseline tensors in attributed-leaf traversal order.

    Raises
    ------
    AttributionError
        If containers or attributed leaves do not mirror the input tree.
    """

    if _is_attributed_tensor(input_tree):
        return [_validate_baseline_tensor(input_tree, baseline_tree)]
    if isinstance(input_tree, tuple):
        if not isinstance(baseline_tree, tuple) or len(baseline_tree) != len(input_tree):
            raise AttributionError("baseline must mirror attributed input structure")
        baselines: list[Tensor] = []
        for input_item, baseline_item in zip(input_tree, baseline_tree, strict=True):
            baselines.extend(_validate_baseline_tree(input_item, baseline_item))
        return baselines
    if isinstance(input_tree, list):
        if not isinstance(baseline_tree, list) or len(baseline_tree) != len(input_tree):
            raise AttributionError("baseline must mirror attributed input structure")
        baselines = []
        for input_item, baseline_item in zip(input_tree, baseline_tree, strict=True):
            baselines.extend(_validate_baseline_tree(input_item, baseline_item))
        return baselines
    if isinstance(input_tree, dict):
        if not isinstance(baseline_tree, dict) or baseline_tree.keys() != input_tree.keys():
            raise AttributionError("baseline must mirror attributed input structure")
        baselines = []
        for key, input_value in input_tree.items():
            baselines.extend(_validate_baseline_tree(input_value, baseline_tree[key]))
        return baselines
    return []


def _validate_baselines(inputs: _PreparedInputs, baseline: Any | None) -> tuple[Tensor, ...]:
    """Validate or create Integrated Gradients baselines for attributed leaves.

    The default zero baseline is conventional but not neutral for every problem;
    attribution users should choose baselines that match their domain.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    baseline
        Optional baseline tree. A bare tensor remains valid when there is exactly
        one attributed leaf.

    Returns
    -------
    tuple[Tensor, ...]
        Baseline tensors matching attributed leaves.

    Raises
    ------
    AttributionError
        If the baseline structure or any tensor does not match.
    """

    if baseline is None:
        return tuple(torch.zeros_like(leaf) for leaf in inputs.attributed_leaves)
    if len(inputs.attributed_leaves) == 1 and isinstance(baseline, Tensor):
        return (_validate_baseline_tensor(inputs.attributed_leaves[0], baseline),)
    if inputs.kwargs:
        if not isinstance(baseline, dict):
            raise AttributionError(
                "baseline must be a dict with 'inputs' and 'input_kwargs' for kwarg inputs"
            )
        if set(baseline) != {"inputs", "input_kwargs"}:
            raise AttributionError(
                "baseline must be a dict with 'inputs' and 'input_kwargs' for kwarg inputs"
            )
        baseline_leaves = _validate_baseline_tree(
            (inputs.positional, inputs.kwargs),
            (baseline["inputs"], baseline["input_kwargs"]),
        )
    else:
        positional_baseline = tuple(baseline) if isinstance(baseline, list) else baseline
        baseline_leaves = _validate_baseline_tree(inputs.positional, positional_baseline)

    if len(baseline_leaves) != len(inputs.attributed_leaves):
        raise AttributionError("baseline must mirror attributed input leaves")
    return tuple(baseline_leaves)


def saliency(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
) -> AttributionResult:
    """Compute absolute input gradients for a scalar target.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are attributed.
    input_kwargs
        Optional keyword arguments for ``model``. Floating-point or complex
        tensor leaves are attributed.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.

    Returns
    -------
    AttributionResult
        Bare tensor for one attributed leaf, otherwise a mirrored value tree.
    """

    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    with _temporarily_eval(model):
        input_leaves = _make_input_leaves(prepared_inputs)
        gradients, _scalar = _gradient_for_inputs(model, prepared_inputs, input_leaves, target)
    values = tuple(gradient.detach().abs() for gradient in gradients)
    return AttributionResult(
        method="saliency",
        values=_value_tree_from_leaves(prepared_inputs, values),
        target_repr=_target_repr(target),
        extra={},
    )


def input_x_grad(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
) -> AttributionResult:
    """Compute gradient times input for a scalar target.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are attributed.
    input_kwargs
        Optional keyword arguments for ``model``. Floating-point or complex
        tensor leaves are attributed.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.

    Returns
    -------
    AttributionResult
        Bare tensor for one attributed leaf, otherwise a mirrored value tree.
    """

    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    with _temporarily_eval(model):
        input_leaves = _make_input_leaves(prepared_inputs)
        gradients, _scalar = _gradient_for_inputs(model, prepared_inputs, input_leaves, target)
    values = tuple(
        (gradient * input_leaf).detach()
        for gradient, input_leaf in zip(gradients, input_leaves, strict=True)
    )
    return AttributionResult(
        method="input_x_grad",
        values=_value_tree_from_leaves(prepared_inputs, values),
        target_repr=_target_repr(target),
        extra={},
    )


def integrated_gradients(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    n_steps: int = 50,
    baseline: Any | None = None,
) -> AttributionResult:
    """Compute Integrated Gradients along a straight baseline-to-input path.

    The default baseline is ``torch.zeros_like`` for each attributed leaf.
    Baseline choice changes the interpretation of the result, so callers should
    pass domain-meaningful baselines when zeros are not appropriate.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are attributed.
    input_kwargs
        Optional keyword arguments for ``model``. Floating-point or complex
        tensor leaves are attributed.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    n_steps
        Number of midpoint Riemann samples along the straight path.
    baseline
        Optional baseline tree matching attributed input leaves. A bare tensor is
        accepted when there is exactly one attributed leaf.

    Returns
    -------
    AttributionResult
        Bare tensor for one attributed leaf, otherwise a mirrored value tree.
    """

    _validate_positive_int("n_steps", n_steps)
    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    baseline_tensors = _validate_baselines(prepared_inputs, baseline)
    deltas = tuple(
        input_leaf.detach() - baseline_tensor
        for input_leaf, baseline_tensor in zip(
            prepared_inputs.attributed_leaves, baseline_tensors, strict=True
        )
    )
    gradients_by_step: list[tuple[Tensor, ...]] = []

    with _temporarily_eval(model):
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps
            path_leaves = tuple(
                (baseline_tensor + alpha * delta).detach().clone().requires_grad_(True)
                for baseline_tensor, delta in zip(baseline_tensors, deltas, strict=True)
            )
            gradients, _scalar = _gradient_for_inputs(
                model,
                prepared_inputs,
                path_leaves,
                target,
            )
            gradients_by_step.append(tuple(gradient.detach() for gradient in gradients))

    mean_gradients = tuple(
        torch.stack([step_gradients[index] for step_gradients in gradients_by_step], dim=0).mean(
            dim=0
        )
        for index in range(len(prepared_inputs.attributed_leaves))
    )
    values = tuple(
        (delta * mean_gradient).detach()
        for delta, mean_gradient in zip(deltas, mean_gradients, strict=True)
    )
    return AttributionResult(
        method="integrated_gradients",
        values=_value_tree_from_leaves(prepared_inputs, values),
        target_repr=_target_repr(target),
        extra={
            "n_steps": n_steps,
            "baseline": _value_tree_from_leaves(prepared_inputs, baseline_tensors),
        },
    )


def smoothgrad(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    n_samples: int = 25,
    noise_level: float = 0.1,
    seed: int | None = None,
) -> AttributionResult:
    """Average saliency over Gaussian-noised copies of an input.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are attributed.
    input_kwargs
        Optional keyword arguments for ``model``. Floating-point or complex
        tensor leaves are attributed.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    n_samples
        Number of noised saliency samples to average.
    noise_level
        Standard deviation of Gaussian noise added to each input copy.
    seed
        Optional random seed for deterministic noise samples without mutating
        global torch RNG state.

    Returns
    -------
    AttributionResult
        Bare tensor for one attributed leaf, otherwise a mirrored value tree.
    """

    _validate_positive_int("n_samples", n_samples)
    if noise_level < 0:
        raise AttributionError("noise_level must be non-negative")
    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    generators: dict[torch.device, torch.Generator] = {}

    saliency_samples: list[tuple[Tensor, ...]] = []
    with _temporarily_eval(model):
        for _sample_idx in range(n_samples):
            noised_leaves = tuple(
                _make_noised_leaf(input_leaf, noise_level, seed, generators)
                for input_leaf in prepared_inputs.attributed_leaves
            )
            gradients, _scalar = _gradient_for_inputs(
                model,
                prepared_inputs,
                noised_leaves,
                target,
            )
            saliency_samples.append(tuple(gradient.detach().abs() for gradient in gradients))

    values = tuple(
        torch.stack([sample[index] for sample in saliency_samples], dim=0).mean(dim=0)
        for index in range(len(prepared_inputs.attributed_leaves))
    )
    return AttributionResult(
        method="smoothgrad",
        values=_value_tree_from_leaves(prepared_inputs, values),
        target_repr=_target_repr(target),
        extra={"n_samples": n_samples, "noise_level": noise_level, "seed": seed},
    )


def _make_noised_leaf(
    input_leaf: Tensor,
    noise_level: float,
    seed: int | None,
    generators: dict[torch.device, torch.Generator],
) -> Tensor:
    """Create one noised differentiable SmoothGrad leaf.

    Parameters
    ----------
    input_leaf
        Original attributed input tensor.
    noise_level
        Standard deviation of Gaussian noise.
    seed
        Optional deterministic seed.
    generators
        Per-device generator cache.

    Returns
    -------
    Tensor
        Noised detached clone with gradient tracking enabled.
    """

    generator = None
    if seed is not None:
        generator = generators.get(input_leaf.device)
        if generator is None:
            generator = torch.Generator(device=input_leaf.device)
            generator.manual_seed(seed)
            generators[input_leaf.device] = generator
    noise = torch.randn(
        input_leaf.shape,
        dtype=input_leaf.dtype,
        device=input_leaf.device,
        generator=generator,
    )
    return (input_leaf.detach() + noise_level * noise).detach().clone().requires_grad_(True)

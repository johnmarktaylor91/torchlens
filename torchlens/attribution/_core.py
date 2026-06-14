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
_UNSUPPORTED_INPUTS_MESSAGE = (
    "multi-input / kwarg / pytree inputs unsupported in v1; pass a single tensor"
)


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
        Attribution tensor with the same shape as the input tensor.
    target_repr
        Compact representation of the scalarization target.
    extra
        Method-specific metadata. This schema is intentionally minimal for v1.
    """

    method: str
    values: Tensor
    target_repr: str
    extra: dict[str, Any]


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


def _validate_single_tensor_input(inputs: Tensor) -> Tensor:
    """Validate the v1 input scope and return the tensor input.

    Parameters
    ----------
    inputs
        Candidate user input.

    Returns
    -------
    Tensor
        The validated tensor input.

    Raises
    ------
    AttributionError
        If ``inputs`` is not a single differentiable tensor.
    """

    if not isinstance(inputs, Tensor):
        raise AttributionError(_UNSUPPORTED_INPUTS_MESSAGE)
    if not inputs.is_floating_point() and not inputs.is_complex():
        raise AttributionError(
            "input attribution requires a floating-point or complex tensor input"
        )
    return inputs


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


def _gradient_for_input(model: Module, inputs: Tensor, target: TargetSpec) -> tuple[Tensor, Tensor]:
    """Compute the gradient of a scalarized model output with respect to input.

    Parameters
    ----------
    model
        Model to evaluate.
    inputs
        Input tensor used as the differentiable leaf.
    target
        Integer class index or callable scalarizer.

    Returns
    -------
    tuple[Tensor, Tensor]
        Gradient tensor and scalarized model output.

    Raises
    ------
    AttributionError
        If the scalar target is not differentiable with respect to the input.
    """

    output = model(inputs)
    scalar = _scalarize_output(output, target)
    try:
        gradient = torch.autograd.grad(scalar, inputs)[0]
    except RuntimeError as exc:
        raise AttributionError(
            "target scalar is not differentiable with respect to the input"
        ) from exc
    return gradient, scalar.detach()


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


def _validate_baseline(inputs: Tensor, baseline: Tensor | None) -> Tensor:
    """Validate or create an Integrated Gradients baseline.

    The default zero baseline is conventional but not neutral for every problem;
    attribution users should choose a baseline that matches their domain.

    Parameters
    ----------
    inputs
        Validated input tensor.
    baseline
        Optional user baseline.

    Returns
    -------
    Tensor
        Baseline tensor matching input shape, dtype, and device.

    Raises
    ------
    AttributionError
        If the baseline is not a matching tensor.
    """

    if baseline is None:
        return torch.zeros_like(inputs)
    if not isinstance(baseline, Tensor):
        raise AttributionError("baseline must be a tensor matching the input")
    if baseline.shape != inputs.shape:
        raise AttributionError("baseline must match input shape")
    if baseline.dtype != inputs.dtype:
        raise AttributionError("baseline must match input dtype")
    if baseline.device != inputs.device:
        raise AttributionError("baseline must match input device")
    return baseline.detach().clone()


def saliency(model: Module, inputs: Tensor, *, target: TargetSpec) -> AttributionResult:
    """Compute absolute input gradients for a scalar target.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Single differentiable tensor positional input. Multi-input and pytree
        inputs are unsupported in v1.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.

    Returns
    -------
    AttributionResult
        Saliency values with the same shape as ``inputs``.
    """

    input_tensor = _validate_single_tensor_input(inputs)
    with _temporarily_eval(model):
        input_leaf = _make_input_leaf(input_tensor)
        gradient, _scalar = _gradient_for_input(model, input_leaf, target)
    return AttributionResult(
        method="saliency",
        values=gradient.detach().abs(),
        target_repr=_target_repr(target),
        extra={},
    )


def input_x_grad(model: Module, inputs: Tensor, *, target: TargetSpec) -> AttributionResult:
    """Compute gradient times input for a scalar target.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Single differentiable tensor positional input. Multi-input and pytree
        inputs are unsupported in v1.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.

    Returns
    -------
    AttributionResult
        Input-times-gradient values with the same shape as ``inputs``.
    """

    input_tensor = _validate_single_tensor_input(inputs)
    with _temporarily_eval(model):
        input_leaf = _make_input_leaf(input_tensor)
        gradient, _scalar = _gradient_for_input(model, input_leaf, target)
    return AttributionResult(
        method="input_x_grad",
        values=(gradient * input_leaf).detach(),
        target_repr=_target_repr(target),
        extra={},
    )


def integrated_gradients(
    model: Module,
    inputs: Tensor,
    *,
    target: TargetSpec,
    n_steps: int = 50,
    baseline: Tensor | None = None,
) -> AttributionResult:
    """Compute Integrated Gradients along a straight baseline-to-input path.

    The default baseline is ``torch.zeros_like(inputs)``. Baseline choice changes
    the interpretation of the result, so callers should pass a domain-meaningful
    baseline when zeros are not appropriate.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Single differentiable tensor positional input. Multi-input and pytree
        inputs are unsupported in v1.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    n_steps
        Number of midpoint Riemann samples along the straight path.
    baseline
        Optional baseline tensor matching input shape, dtype, and device.

    Returns
    -------
    AttributionResult
        Integrated Gradients values with the same shape as ``inputs``.
    """

    _validate_positive_int("n_steps", n_steps)
    input_tensor = _validate_single_tensor_input(inputs)
    baseline_tensor = _validate_baseline(input_tensor, baseline)
    delta = input_tensor.detach() - baseline_tensor
    gradients = []

    with _temporarily_eval(model):
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps
            path_input = (baseline_tensor + alpha * delta).detach().clone().requires_grad_(True)
            gradient, _scalar = _gradient_for_input(model, path_input, target)
            gradients.append(gradient.detach())

    mean_gradient = torch.stack(gradients, dim=0).mean(dim=0)
    return AttributionResult(
        method="integrated_gradients",
        values=(delta * mean_gradient).detach(),
        target_repr=_target_repr(target),
        extra={"n_steps": n_steps, "baseline": baseline_tensor},
    )


def smoothgrad(
    model: Module,
    inputs: Tensor,
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
        Single differentiable tensor positional input. Multi-input and pytree
        inputs are unsupported in v1.
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
        SmoothGrad values with the same shape as ``inputs``.
    """

    _validate_positive_int("n_samples", n_samples)
    if noise_level < 0:
        raise AttributionError("noise_level must be non-negative")
    input_tensor = _validate_single_tensor_input(inputs)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=input_tensor.device)
        generator.manual_seed(seed)

    saliency_samples = []
    with _temporarily_eval(model):
        for _sample_idx in range(n_samples):
            noise = torch.randn(
                input_tensor.shape,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
                generator=generator,
            )
            noised_input = (input_tensor.detach() + noise_level * noise).detach()
            input_leaf = noised_input.clone().requires_grad_(True)
            gradient, _scalar = _gradient_for_input(model, input_leaf, target)
            saliency_samples.append(gradient.detach().abs())

    return AttributionResult(
        method="smoothgrad",
        values=torch.stack(saliency_samples, dim=0).mean(dim=0),
        target_repr=_target_repr(target),
        extra={"n_samples": n_samples, "noise_level": noise_level, "seed": seed},
    )

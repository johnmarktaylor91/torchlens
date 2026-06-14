"""Native layer-attribution methods for TorchLens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from torchlens.attribution._core import (
    AttributionError,
    AttributionResult,
    InputKwargs,
    TargetSpec,
    _PreparedInputs,
    _call_model,
    _make_input_leaves,
    _normalize_model_inputs,
    _scalarize_output,
    _target_repr,
    _temporarily_eval,
    _validate_baselines,
    _validate_positive_int,
)

LayerAttributionMethod: TypeAlias = Literal["activation_x_grad", "grad"]


@dataclass
class _LayerCapture:
    """Container for a captured layer activation.

    Attributes
    ----------
    activation
        Forward activation emitted by the target layer.
    """

    activation: Tensor | None = None


def _format_layer_options(model: Module) -> str:
    """Return a compact list of useful layer-name suggestions.

    Parameters
    ----------
    model
        Model whose named modules should be searched.

    Returns
    -------
    str
        Comma-separated module names suitable for an error message.
    """

    named_modules = dict(model.named_modules())
    conv_like = [
        name
        for name, module in named_modules.items()
        if name and ("conv" in name.lower() or "conv" in type(module).__name__.lower())
    ]
    options = conv_like[:5]
    if not options:
        options = [name for name in named_modules if name][:5]
    if not options:
        return "<no named child modules>"
    return ", ".join(options)


def _resolve_named_layer(model: Module, layer: str) -> Module:
    """Resolve a user-specified module name.

    Parameters
    ----------
    model
        Model containing the target layer.
    layer
        Name from ``model.named_modules()``.

    Returns
    -------
    Module
        Resolved PyTorch module.

    Raises
    ------
    AttributionError
        If ``layer`` is not a named module string in ``model``.
    """

    if not isinstance(layer, str):
        raise AttributionError("layer must be a module name string")

    named_modules = dict(model.named_modules())
    if layer not in named_modules:
        options = _format_layer_options(model)
        raise AttributionError(
            f"layer {layer!r} was not found; available conv-like layers include: {options}"
        )
    return named_modules[layer]


def _capture_layer_activation(
    model: Module,
    inputs: _PreparedInputs,
    target: TargetSpec,
    layer: str,
) -> tuple[Tensor, Tensor]:
    """Capture a layer activation and its target gradient.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Normalized attribution inputs.
    target
        Integer class index or callable scalarizer.
    layer
        Name of the module whose activation should be captured.

    Returns
    -------
    tuple[Tensor, Tensor]
        Captured activation and gradient of the scalar target with respect to it.

    Raises
    ------
    AttributionError
        If the target layer does not emit a differentiable tensor activation.
    """

    target_layer = _resolve_named_layer(model, layer)
    capture = _LayerCapture()
    hook_handles: list[RemovableHandle] = []

    def _forward_hook(_module: Module, _args: tuple[Any, ...], output: Any) -> None:
        """Store the target layer output from the forward pass."""

        if not isinstance(output, Tensor):
            raise AttributionError("target layer must return a tensor activation")
        capture.activation = output

    hook_handles.append(target_layer.register_forward_hook(_forward_hook))
    try:
        with _temporarily_eval(model):
            input_leaves = _make_input_leaves(inputs)
            output = _call_model(model, inputs, input_leaves)
            activation = capture.activation
            if activation is None:
                raise AttributionError(f"layer {layer!r} did not run during the forward pass")
            if not activation.requires_grad:
                raise AttributionError(
                    f"layer {layer!r} activation is not differentiable with respect to target"
                )
            scalar = _scalarize_output(output, target)
            try:
                gradient = torch.autograd.grad(scalar, activation)[0]
            except RuntimeError as exc:
                raise AttributionError(
                    f"target scalar is not differentiable with respect to layer {layer!r}"
                ) from exc
    finally:
        for handle in hook_handles:
            handle.remove()

    return activation.detach(), gradient.detach()


def _capture_layer_activation_for_leaves(
    model: Module,
    inputs: _PreparedInputs,
    target: TargetSpec,
    layer: str,
    input_leaves: tuple[Tensor, ...],
    *,
    require_gradient: bool,
) -> tuple[Tensor, Tensor | None]:
    """Capture a layer activation and optionally its target gradient.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Normalized attribution inputs.
    target
        Integer class index or callable scalarizer.
    layer
        Name of the module whose activation should be captured.
    input_leaves
        Differentiable leaves substituted into the model call.
    require_gradient
        Whether to compute ``dTarget / dActivation``.

    Returns
    -------
    tuple[Tensor, Tensor | None]
        Captured activation and optional gradient with respect to it.

    Raises
    ------
    AttributionError
        If the target layer does not emit a differentiable tensor activation.
    """

    target_layer = _resolve_named_layer(model, layer)
    capture = _LayerCapture()
    hook_handles: list[RemovableHandle] = []

    def _forward_hook(_module: Module, _args: tuple[Any, ...], output: Any) -> None:
        """Store the target layer output from the forward pass."""

        if not isinstance(output, Tensor):
            raise AttributionError("target layer must return a tensor activation")
        capture.activation = output

    hook_handles.append(target_layer.register_forward_hook(_forward_hook))
    try:
        output = _call_model(model, inputs, input_leaves)
        activation = capture.activation
        if activation is None:
            raise AttributionError(f"layer {layer!r} did not run during the forward pass")
        if not require_gradient:
            return activation.detach(), None
        if not activation.requires_grad:
            raise AttributionError(
                f"layer {layer!r} activation is not differentiable with respect to target"
            )
        scalar = _scalarize_output(output, target)
        try:
            gradient = torch.autograd.grad(scalar, activation)[0]
        except RuntimeError as exc:
            raise AttributionError(
                f"target scalar is not differentiable with respect to layer {layer!r}"
            ) from exc
    finally:
        for handle in hook_handles:
            handle.remove()

    return activation.detach(), gradient.detach()


def _path_input_leaves(
    baseline_tensors: tuple[Tensor, ...],
    deltas: tuple[Tensor, ...],
    alpha: float,
) -> tuple[Tensor, ...]:
    """Create differentiable input leaves for one baseline-to-input path point.

    Parameters
    ----------
    baseline_tensors
        Baseline leaves in attributed-leaf traversal order.
    deltas
        Input-minus-baseline tensors in the same order.
    alpha
        Interpolation coefficient on ``[0, 1]``.

    Returns
    -------
    tuple[Tensor, ...]
        Detached differentiable path leaves.
    """

    return tuple(
        (baseline_tensor + alpha * delta).detach().clone().requires_grad_(True)
        for baseline_tensor, delta in zip(baseline_tensors, deltas, strict=True)
    )


def _layer_path_basics(
    model: Module,
    inputs: _PreparedInputs,
    target: TargetSpec,
    layer: str,
    baseline_tensors: tuple[Tensor, ...],
    n_steps: int,
) -> tuple[Tensor, Tensor, list[Tensor], tuple[Tensor, ...]]:
    """Capture endpoint activations and midpoint layer gradients along an input path.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Normalized attribution inputs.
    target
        Integer class index or callable scalarizer.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    baseline_tensors
        Baseline leaves matching attributed inputs.
    n_steps
        Number of midpoint Riemann samples.

    Returns
    -------
    tuple[Tensor, Tensor, list[Tensor], tuple[Tensor, ...]]
        Baseline activation, input activation, midpoint gradients, and input deltas.
    """

    deltas = tuple(
        input_leaf.detach() - baseline_tensor
        for input_leaf, baseline_tensor in zip(
            inputs.attributed_leaves, baseline_tensors, strict=True
        )
    )
    with _temporarily_eval(model):
        baseline_activation, _baseline_gradient = _capture_layer_activation_for_leaves(
            model,
            inputs,
            target,
            layer,
            _path_input_leaves(baseline_tensors, deltas, 0.0),
            require_gradient=False,
        )
        input_activation, _input_gradient = _capture_layer_activation_for_leaves(
            model,
            inputs,
            target,
            layer,
            _path_input_leaves(baseline_tensors, deltas, 1.0),
            require_gradient=False,
        )
        gradients_by_step = []
        for step in range(n_steps):
            alpha = (step + 0.5) / n_steps
            _activation, gradient = _capture_layer_activation_for_leaves(
                model,
                inputs,
                target,
                layer,
                _path_input_leaves(baseline_tensors, deltas, alpha),
                require_gradient=True,
            )
            if gradient is None:
                raise AttributionError("internal error: missing layer path gradient")
            gradients_by_step.append(gradient)
    return baseline_activation, input_activation, gradients_by_step, deltas


def _spatial_reference_tensor(inputs: _PreparedInputs) -> Tensor:
    """Return the attributed input tensor used for Grad-CAM spatial upsampling.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.

    Returns
    -------
    Tensor
        First attributed leaf with at least two spatial dimensions.

    Raises
    ------
    AttributionError
        If no attributed tensor leaf has spatial dimensions.
    """

    for input_tensor in inputs.attributed_leaves:
        if input_tensor.ndim >= 4:
            return input_tensor
    raise AttributionError("grad_cam requires an input tensor with spatial dimensions")


def _validate_conv_activation(activation: Tensor, layer: str) -> None:
    """Validate that an activation is a 2D convolution-style feature map.

    Parameters
    ----------
    activation
        Captured target-layer activation.
    layer
        Layer name used for error reporting.

    Raises
    ------
    AttributionError
        If ``activation`` is not shaped ``N, C, H, W``.
    """

    if activation.ndim != 4:
        raise AttributionError(
            f"grad_cam requires layer {layer!r} to produce a 4D N,C,H,W feature map; "
            f"got shape {tuple(activation.shape)}"
        )


def grad_cam(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    layer: str,
    relu: bool = True,
) -> AttributionResult:
    """Compute Grad-CAM for a named convolution-style layer.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    relu
        Whether to apply ReLU to the channel-reduced CAM.

    Returns
    -------
    AttributionResult
        Grad-CAM values upsampled to the input spatial size with shape
        ``N, 1, Hin, Win``.
    """

    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    spatial_reference = _spatial_reference_tensor(prepared_inputs)

    activation, gradient = _capture_layer_activation(model, prepared_inputs, target, layer)
    _validate_conv_activation(activation, layer)
    alpha = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (alpha * activation).sum(dim=1, keepdim=True)
    if relu:
        cam = torch.relu(cam)
    upsampled_cam = F.interpolate(
        cam,
        size=spatial_reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return AttributionResult(
        method="grad_cam",
        values=upsampled_cam.detach(),
        target_repr=_target_repr(target),
        extra={"layer": layer, "relu": relu},
    )


def layer_integrated_gradients(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    layer: str,
    baseline: Any | None = None,
    n_steps: int = 50,
) -> AttributionResult:
    """Compute Layer Integrated Gradients for a named intermediate layer.

    The input path is the straight line from baseline leaves to input leaves.
    The returned values have the same shape as the target layer activation and
    use the Captum-style ``(A(input) - A(baseline)) * mean(dTarget / dA)`` rule.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    baseline
        Optional baseline tree matching attributed input leaves. A bare tensor is
        accepted when there is exactly one attributed leaf.
    n_steps
        Number of midpoint Riemann samples along the straight input path.

    Returns
    -------
    AttributionResult
        Layer attribution values with the same shape as the captured activation.
    """

    _validate_positive_int("n_steps", n_steps)
    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    baseline_tensors = _validate_baselines(prepared_inputs, baseline)
    baseline_activation, input_activation, gradients_by_step, _deltas = _layer_path_basics(
        model,
        prepared_inputs,
        target,
        layer,
        baseline_tensors,
        n_steps,
    )
    mean_gradient = torch.stack(gradients_by_step, dim=0).mean(dim=0)
    values = (input_activation - baseline_activation) * mean_gradient
    return AttributionResult(
        method="layer_integrated_gradients",
        values=values.detach(),
        target_repr=_target_repr(target),
        extra={
            "layer": layer,
            "n_steps": n_steps,
        },
    )


def layer_conductance(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    layer: str,
    baseline: Any | None = None,
    n_steps: int = 50,
) -> AttributionResult:
    """Compute Layer Conductance for a named intermediate layer.

    Conductance decomposes input Integrated Gradients onto hidden units by
    integrating ``(dTarget / dA) * (dA / dalpha)`` along the input path. This
    implementation uses midpoint layer gradients and finite activation
    differences for each path interval.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    baseline
        Optional baseline tree matching attributed input leaves. A bare tensor is
        accepted when there is exactly one attributed leaf.
    n_steps
        Number of midpoint Riemann samples along the straight input path.

    Returns
    -------
    AttributionResult
        Layer conductance values with the same shape as the captured activation.
    """

    _validate_positive_int("n_steps", n_steps)
    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    baseline_tensors = _validate_baselines(prepared_inputs, baseline)
    baseline_activation, _input_activation, gradients_by_step, deltas = _layer_path_basics(
        model,
        prepared_inputs,
        target,
        layer,
        baseline_tensors,
        n_steps,
    )
    activation_left = baseline_activation
    conductance = torch.zeros_like(baseline_activation)
    with _temporarily_eval(model):
        for step, gradient in enumerate(gradients_by_step):
            alpha_right = (step + 1) / n_steps
            activation_right, _right_gradient = _capture_layer_activation_for_leaves(
                model,
                prepared_inputs,
                target,
                layer,
                _path_input_leaves(baseline_tensors, deltas, alpha_right),
                require_gradient=False,
            )
            conductance = conductance + gradient * (activation_right - activation_left)
            activation_left = activation_right
    return AttributionResult(
        method="layer_conductance",
        values=conductance.detach(),
        target_repr=_target_repr(target),
        extra={
            "layer": layer,
            "n_steps": n_steps,
        },
    )


def layer_attribution(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: TargetSpec,
    layer: str,
    method: LayerAttributionMethod = "activation_x_grad",
) -> AttributionResult:
    """Compute attribution for a named intermediate layer.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are threaded through the model call.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, or callable ``output -> scalar tensor``.
    layer
        Name from ``dict(model.named_modules())`` identifying the target layer.
    method
        Layer attribution method. ``"activation_x_grad"`` returns
        ``activation * gradient``. ``"grad"`` returns ``abs(gradient)``.

    Returns
    -------
    AttributionResult
        Layer-attribution values with the same shape as the captured activation.

    Raises
    ------
    AttributionError
        If ``method`` is unsupported.
    """

    if method not in ("activation_x_grad", "grad"):
        raise AttributionError("method must be 'activation_x_grad' or 'grad'")

    prepared_inputs = _normalize_model_inputs(inputs, input_kwargs)
    activation, gradient = _capture_layer_activation(model, prepared_inputs, target, layer)
    if method == "activation_x_grad":
        values = activation * gradient
    else:
        values = gradient.abs()
    return AttributionResult(
        method=f"layer_{method}",
        values=values.detach(),
        target_repr=_target_repr(target),
        extra={"layer": layer},
    )

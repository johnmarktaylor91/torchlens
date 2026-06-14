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

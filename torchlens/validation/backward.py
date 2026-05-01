"""Validation helpers for backward-pass gradient capture."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn


def _sum_tensors(value: Any) -> torch.Tensor:
    """Reduce nested tensor outputs to a scalar loss.

    Parameters
    ----------
    value:
        Tensor or nested container of tensors.

    Returns
    -------
    torch.Tensor
        Scalar sum over all tensors in ``value``.
    """
    if isinstance(value, torch.Tensor):
        return value.sum()
    if isinstance(value, dict):
        tensors = [_sum_tensors(item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        tensors = [_sum_tensors(item) for item in value]
    else:
        tensors = []
    if not tensors:
        raise ValueError("validate_backward_pass requires the model to return at least one tensor.")
    result = tensors[0]
    for tensor in tensors[1:]:
        result = result + tensor
    return result


def _clone_inputs_with_grad(input_args: Any) -> Any:
    """Clone tensor inputs and enable gradients on floating tensors.

    Parameters
    ----------
    input_args:
        User input arguments.

    Returns
    -------
    Any
        Input arguments with cloned floating tensors requiring grad.
    """
    if isinstance(input_args, torch.Tensor):
        cloned = input_args.detach().clone()
        if cloned.is_floating_point() or cloned.is_complex():
            cloned.requires_grad_(True)
        return cloned
    if isinstance(input_args, tuple):
        return tuple(_clone_inputs_with_grad(item) for item in input_args)
    if isinstance(input_args, list):
        return [_clone_inputs_with_grad(item) for item in input_args]
    if isinstance(input_args, dict):
        return {key: _clone_inputs_with_grad(item) for key, item in input_args.items()}
    return input_args


def _param_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect detached parameter gradients by name.

    Parameters
    ----------
    model:
        Model whose parameter gradients should be collected.

    Returns
    -------
    dict[str, torch.Tensor]
        Detached gradient clones for parameters with gradients.
    """
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def validate_backward_pass(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    *,
    perturb_saved_gradients: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> bool:
    """Validate TorchLens backward capture against stock autograd parameter grads.

    Parameters
    ----------
    model:
        Model to validate.
    input_args:
        Positional input arguments for ``model``.
    input_kwargs:
        Keyword input arguments for ``model``.
    loss_fn:
        Optional callable that maps model outputs to a scalar loss. Defaults to
        summing all returned tensors.
    perturb_saved_gradients:
        If True, perturb captured saved gradients before comparison; the
        validation should then return False.
    atol:
        Absolute tolerance for ``torch.allclose``.
    rtol:
        Relative tolerance for ``torch.allclose``.

    Returns
    -------
    bool
        True when captured gradients match stock autograd and perturbation is
        not requested.
    """
    from ..user_funcs import log_forward_pass

    if input_kwargs is None:
        input_kwargs = {}
    if loss_fn is None:
        loss_fn = _sum_tensors

    stock_inputs = _clone_inputs_with_grad(input_args)
    model.zero_grad(set_to_none=True)
    stock_loss = loss_fn(model(stock_inputs, **input_kwargs))
    stock_loss.backward()  # type: ignore[no-untyped-call]
    expected_param_grads = _param_grads(model)

    logged_inputs = _clone_inputs_with_grad(input_args)
    model.zero_grad(set_to_none=True)
    model_log = log_forward_pass(
        model,
        logged_inputs,
        input_kwargs=input_kwargs,
        layers_to_save="all",
        gradients_to_save="all",
    )
    output_layers = [model_log[layer_label].activation for layer_label in model_log.output_layers]
    logged_output: Any = output_layers[0] if len(output_layers) == 1 else output_layers
    logged_loss = loss_fn(logged_output)
    model_log.log_backward(logged_loss)
    if perturb_saved_gradients:
        for layer in model_log.layer_list:
            if layer.has_gradient and isinstance(layer.gradient, torch.Tensor):
                layer.gradient = layer.gradient + torch.randn_like(layer.gradient)
                break
    observed_param_grads = _param_grads(model)
    model_log.cleanup()

    if expected_param_grads.keys() != observed_param_grads.keys():
        return False
    return (
        all(
            torch.allclose(
                observed_param_grads[name], expected_param_grads[name], atol=atol, rtol=rtol
            )
            for name in expected_param_grads
        )
        and not perturb_saved_gradients
    )

"""Validation helpers for backward-pass grad capture."""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, Callable, cast
import warnings

import torch
from torch import nn

from .._robustness import check_model_and_input_variants
from ..intervention.errors import AppendStateValidationWarning
from ..utils.arg_handling import normalize_input_args
from ..utils.display import warn_parallel
from ..utils.rng import set_random_seed


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
    """Clone tensor inputs and enable grads on floating tensors.

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
    """Collect detached parameter grads by name.

    Parameters
    ----------
    model:
        Model whose parameter grads should be collected.

    Returns
    -------
    dict[str, torch.Tensor]
        Detached grad clones for parameters with grads.
    """
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _clone_state_dict_with_metadata(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Clone a module ``state_dict`` while preserving PyTorch metadata.

    Parameters
    ----------
    model:
        Model whose state should be cloned.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Detached tensor clones with PyTorch ``state_dict`` metadata preserved.
    """

    from ..user_funcs import _clone_state_dict_with_metadata as clone_state_dict

    return clone_state_dict(model)


def _move_tensors_to_device(obj: Any, device: torch.device | str) -> Any:
    """Move nested tensors to ``device`` using the public validator helper.

    Parameters
    ----------
    obj:
        Tensor or nested container.
    device:
        Target device.

    Returns
    -------
    Any
        Object with all tensors moved to ``device``.
    """

    from ..user_funcs import _move_tensors_to_device as move_tensors

    return move_tensors(obj, device)


def _prepare_inputs_for_backward(
    input_args: Any,
    input_kwargs: dict[str, Any],
    device: torch.device | None,
) -> tuple[Any, dict[str, Any]]:
    """Clone validation inputs and move them to the model device.

    Parameters
    ----------
    input_args:
        Normalized positional model inputs.
    input_kwargs:
        Keyword model inputs.
    device:
        Target model device, if any parameters exist.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        Cloned positional and keyword inputs.
    """

    cloned_args = _clone_inputs_with_grad(input_args)
    cloned_kwargs = cast(dict[str, Any], _clone_inputs_with_grad(input_kwargs))
    if device is None:
        return cloned_args, cloned_kwargs
    return _move_tensors_to_device(cloned_args, device), _move_tensors_to_device(
        cloned_kwargs, device
    )


def _restore_training_mode(model: nn.Module, training: bool) -> None:
    """Restore a module's train/eval flag.

    Parameters
    ----------
    model:
        Model whose mode should be restored.
    training:
        Original ``model.training`` value.
    """

    model.train(training)


def _is_appended_trace(value: Any) -> bool:
    """Return whether a value is an appended Trace-like object.

    Parameters
    ----------
    value:
        Object passed to the backward validator.

    Returns
    -------
    bool
        True only for objects carrying the appended Trace state marker.
    """

    return bool(getattr(value, "is_appended", False))


def _warn_and_skip_appended_trace_validation(trace: Any) -> bool:
    """Warn that stacked traces cannot be freshly revalidated.

    Parameters
    ----------
    trace:
        Appended Trace-like object supplied to validation.

    Returns
    -------
    bool
        Always True because saved stacked activations are treated as authoritative.
    """

    warnings.warn(
        "validate_backward_pass received a stacked appended trace; fresh backward "
        "re-derivation for appended traces is not supported, so saved grads are "
        "treated as authoritative.",
        AppendStateValidationWarning,
        stacklevel=2,
    )
    return True


def validate_backward_pass(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    *,
    perturb_saved_grads: bool = False,
    validate_metadata: bool = True,
    random_seed: int | None = None,
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
    perturb_saved_grads:
        If True, perturb captured saved grads before comparison; the
        validation should then return False.
    validate_metadata:
        If True, run metadata invariant checks on the captured backward trace.
    random_seed:
        Fixed RNG seed for stock and candidate passes. Auto-generated if None.
    atol:
        Absolute tolerance for ``torch.allclose``.
    rtol:
        Relative tolerance for ``torch.allclose``.

    Returns
    -------
    bool
        True when captured grads match stock autograd and perturbation is
        not requested.
    """
    from ..user_funcs import _reject_opaque_wrappers, _unwrap_data_parallel, trace as trace_fn
    from .invariants import check_metadata_invariants

    if _is_appended_trace(model):
        return _warn_and_skip_appended_trace_validation(model)

    warn_parallel()
    _reject_opaque_wrappers(model)
    model = _unwrap_data_parallel(model)
    check_model_and_input_variants(model, input_args, input_kwargs)
    if input_kwargs is None:
        input_kwargs = {}
    if loss_fn is None:
        loss_fn = _sum_tensors
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    input_args = normalize_input_args(input_args, model)
    model_device = next((parameter.device for parameter in model.parameters()), None)
    state_dict = _clone_state_dict_with_metadata(model)
    original_training = model.training
    trace = None

    try:
        set_random_seed(random_seed)
        stock_inputs, stock_kwargs = _prepare_inputs_for_backward(
            input_args, input_kwargs, model_device
        )
        model.zero_grad(set_to_none=True)
        stock_loss = loss_fn(model(*stock_inputs, **stock_kwargs))
        stock_loss.backward()  # type: ignore[no-untyped-call]
        expected_param_grads = _param_grads(model)

        model.load_state_dict(state_dict)
        _restore_training_mode(model, original_training)
        set_random_seed(random_seed)
        logged_inputs, logged_kwargs = _prepare_inputs_for_backward(
            input_args, input_kwargs, model_device
        )
        model.zero_grad(set_to_none=True)
        trace = trace_fn(
            model,
            logged_inputs,
            input_kwargs=logged_kwargs,
            layers_to_save="all",
            grads_to_save="all",
            random_seed=random_seed,
        )
        output_layers = [trace[layer_label].out for layer_label in trace.output_layers]
        logged_output: Any = output_layers[0] if len(output_layers) == 1 else output_layers
        logged_loss = loss_fn(logged_output)
        trace.log_backward(logged_loss)
        if validate_metadata:
            check_metadata_invariants(trace)
        if perturb_saved_grads:
            for layer in trace.layer_list:
                if layer.has_grad and isinstance(layer.grad, torch.Tensor):
                    layer.grad = layer.grad + torch.randn_like(layer.grad)
                    break
        observed_param_grads = _param_grads(model)

        if expected_param_grads.keys() != observed_param_grads.keys():
            return False
        return (
            all(
                torch.allclose(
                    observed_param_grads[name], expected_param_grads[name], atol=atol, rtol=rtol
                )
                for name in expected_param_grads
            )
            and not perturb_saved_grads
        )
    finally:
        model.load_state_dict(state_dict)
        _restore_training_mode(model, original_training)
        model.zero_grad(set_to_none=True)
        if trace is not None:
            trace.cleanup()

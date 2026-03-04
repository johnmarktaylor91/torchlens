"""Argument copying, normalization, and input validation for model forward calls."""

import inspect
from collections import defaultdict
from typing import Any, Optional

import torch
from torch import nn


def _safe_copy_arg(arg: Any) -> Any:
    """Copy a single argument safely, avoiding deepcopy on arbitrary objects.

    Clones tensors, recurses into standard containers (list, tuple, dict),
    and leaves everything else as-is.  This prevents infinite loops that
    copy.deepcopy can trigger on complex tensor wrappers (e.g. ESCNN
    GeometricTensor) while still protecting user inputs from in-place
    mutations like device moves.

    Note: custom objects containing tensors are passed by reference.  If the
    model is on a different device, _fetch_label_move_input_tensors may
    mutate the wrapper's tensor attribute in-place.  This is acceptable
    because pre-fix such inputs caused an infinite hang.
    """
    if isinstance(arg, torch.Tensor):
        return arg.clone()
    elif isinstance(arg, defaultdict):
        # defaultdict(factory, {k: v, ...}) — preserve the default_factory (#127)
        copied = defaultdict(arg.default_factory, {k: _safe_copy_arg(v) for k, v in arg.items()})
        return copied
    elif isinstance(arg, dict):
        return type(arg)({k: _safe_copy_arg(v) for k, v in arg.items()})
    elif isinstance(arg, (list, tuple)):
        copied = [_safe_copy_arg(item) for item in arg]  # type: ignore[assignment]
        return type(arg)(*copied) if hasattr(type(arg), "_fields") else type(arg)(copied)
    else:
        return arg


def safe_copy_args(args: list) -> list:
    """Safely copy a list of function arguments."""
    return [_safe_copy_arg(arg) for arg in args]


def safe_copy_kwargs(kwargs: dict) -> dict:
    """Safely copy a dict of keyword arguments."""
    return {key: _safe_copy_arg(val) for key, val in kwargs.items()}


def _model_expects_single_arg(model: nn.Module) -> Optional[bool]:
    """Check if the model's forward expects exactly 1 positional arg (excluding self).

    Returns True if exactly 1, False if not, None if introspection fails.
    """
    try:
        spec = inspect.getfullargspec(model.forward)
    except (TypeError, ValueError):
        return None
    named_args = [a for a in spec.args if a != "self"]
    if spec.varargs is not None:
        return False
    return len(named_args) == 1


def normalize_input_args(input_args, model: nn.Module) -> list:
    """Normalize input_args into a list suitable for `model(*input_args)`.

    Handles the ambiguity when the user passes a tuple or list: it could be
    multiple positional args, or a single arg that happens to be a
    tuple/list (issue #43).  Resolves the ambiguity by checking the model's
    forward signature.
    """
    if type(input_args) in (tuple, list):
        single = _model_expects_single_arg(model)
        if single and len(input_args) != 1:
            # Model expects 1 arg; the tuple/list IS the arg.
            input_args = [input_args]
        elif type(input_args) is tuple:
            input_args = list(input_args)
        # If already a list and not wrapping, leave as-is.
    elif input_args is not None:
        input_args = [input_args]
    if not input_args:
        input_args = []
    return input_args

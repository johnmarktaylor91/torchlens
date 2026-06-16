"""Argument copying, normalization, and input validation for model forward calls.

Provides safe argument copying that avoids ``copy.deepcopy`` (which can
trigger infinite loops on complex tensor wrappers with circular references,
e.g. ESCNN GeometricTensor) and normalizes user-supplied ``input_args`` into
the ``list`` form expected by ``model(*input_args)``.
"""

import inspect
from collections import defaultdict
from typing import Any, Optional, cast

import torch
from torch import nn

from .tensor_utils import _clone_tensor_payload, _copy_tensor_payload

INPUT_WAS_PARAMETER_ATTR = "_torchlens_input_was_parameter"


def _clone_input_tensor_payload(arg: torch.Tensor) -> torch.Tensor:
    """Clone a forward-input tensor without preserving ``nn.Parameter`` type.

    Parameters
    ----------
    arg
        Forward input tensor or parameter to clone.

    Returns
    -------
    torch.Tensor
        Plain tensor clone for model input replay. Parameter inputs intentionally
        become tensors so input provenance wins over Python wrapper type.
    """

    if isinstance(arg, torch.nn.Parameter):
        from .._state import pause_logging

        with pause_logging():
            cloned = _copy_tensor_payload(arg, detach_tensor=False, save_mode="copy")
        setattr(cloned, INPUT_WAS_PARAMETER_ATTR, True)
        return cloned
    return cast(torch.Tensor, _clone_tensor_payload(arg, detach_tensor=False, save_mode="copy"))


def copy_arg_tree(arg: Any) -> Any:
    """Copy an input argument tree, cloning tensors and recursing built-in containers.

    Why not ``copy.deepcopy``?  Many third-party tensor wrappers hold
    circular references back to their parent modules.  ``deepcopy`` follows
    these references recursively and either hangs or blows the stack.
    Instead we clone tensors (which copies storage without following Python
    object references) and recurse only into standard containers.

    Clones tensors, recurses into standard containers (list, tuple, dict),
    and leaves everything else as-is.  This prevents infinite loops that
    copy.deepcopy can trigger on complex tensor wrappers (e.g. ESCNN
    GeometricTensor) while still protecting user inputs from in-place
    mutations like device moves.

    Note: custom objects containing tensors are passed by reference.  If the
    model is on a different device, _fetch_label_move_input_tensors may
    mutate the wrapper's tensor attribute in-place.  This is acceptable
    because pre-fix such inputs caused an infinite hang.

    Parameters
    ----------
    arg
        Argument value to copy.

    Returns
    -------
    Any
        Argument copy with nested tensors cloned and custom objects preserved
        by reference.
    """
    if isinstance(arg, torch.Tensor):
        return _clone_input_tensor_payload(arg)
    elif isinstance(arg, defaultdict):
        # defaultdict(factory, {k: v, ...}) — preserve the default_factory (#127).
        # A plain dict() constructor would lose default_factory.
        copied = defaultdict(arg.default_factory, {k: copy_arg_tree(v) for k, v in arg.items()})
        return copied
    elif isinstance(arg, dict):
        # type(arg)({...}) preserves OrderedDict and other dict subclasses.
        return type(arg)({k: copy_arg_tree(v) for k, v in arg.items()})
    elif isinstance(arg, (list, tuple)):
        copied = [copy_arg_tree(item) for item in arg]  # type: ignore[assignment]
        # NamedTuples have _fields and need *args construction;
        # plain tuples/lists take an iterable.
        return type(arg)(*copied) if hasattr(type(arg), "_fields") else type(arg)(copied)
    else:
        # Non-container, non-tensor objects (ints, strings, custom wrappers)
        # are returned by reference — shallow enough to avoid circular ref issues.
        return arg


def _safe_copy_arg(arg: Any) -> Any:
    """Compatibility alias for :func:`copy_arg_tree`.

    Parameters
    ----------
    arg
        Argument value to copy.

    Returns
    -------
    Any
        Recursive input-argument copy result.
    """

    return copy_arg_tree(arg)


def safe_copy_args(args: list[Any]) -> list[Any]:
    """Safely copy a list of positional arguments.

    Each element is copied via :func:`copy_arg_tree`: tensors are cloned,
    containers are recursed into, and everything else is passed by reference.
    """
    return [copy_arg_tree(arg) for arg in args]


def safe_copy_kwargs(kwargs: dict[Any, Any]) -> dict[Any, Any]:
    """Safely copy a dict of keyword arguments.

    Same semantics as :func:`safe_copy_args` but for ``**kwargs``.
    """
    return {key: copy_arg_tree(val) for key, val in kwargs.items()}


def _model_expects_single_arg(model: nn.Module) -> Optional[bool]:
    """Check if the model's forward expects exactly 1 positional arg (excluding self).

    Used by :func:`normalize_input_args` to disambiguate whether a user-supplied
    tuple is "multiple positional args" or "one arg that is a tuple."

    Returns:
        True if exactly 1 positional param, False if more or uses ``*args``,
        None if introspection fails (e.g. C-extension forward).
    """
    try:
        spec = inspect.getfullargspec(model.forward)
    except (TypeError, ValueError):
        # Introspection can fail on C-extension or dynamically-generated forward custom_methods.
        return None
    named_args = [a for a in spec.args if a != "self"]
    if spec.varargs is not None:
        # Has *args — could accept any number of positional args; can't tell.
        return False
    return len(named_args) == 1


def normalize_input_args(input_args: Any, model: nn.Module) -> list[Any]:
    """Normalize ``input_args`` into a list suitable for ``model(*input_args)``.

    Handles the ambiguity when the user ops a tuple or list: it could be
    multiple positional args, or a single arg that happens to be a
    tuple/list (issue #43).  Resolves the ambiguity by inspecting the
    model's ``forward`` signature via :func:`_model_expects_single_arg`:

    * If the model expects exactly one positional param and the user passed
      a multi-element tuple/list, wrap it in a list so it arrives as a
      single argument: ``model(the_tuple)``.
    * Otherwise, treat each element of the tuple/list as a separate
      positional argument: ``model(arg0, arg1, ...)``.
    """
    if type(input_args) in (tuple, list):
        single = _model_expects_single_arg(model)
        if single and len(input_args) != 1:
            # Model expects 1 arg but user passed a multi-element tuple/list.
            # The tuple/list itself IS the single argument.
            input_args = [input_args]
        elif type(input_args) is tuple:
            # Multiple args — convert tuple to mutable list for internal use.
            input_args = list(input_args)
        # If already a list and not wrapping, leave as-is.
    elif input_args is not None:
        # Bare value (single tensor, etc.) — wrap in a list.
        input_args = [input_args]
    if not input_args:
        input_args = []
    return cast(list[Any], input_args)

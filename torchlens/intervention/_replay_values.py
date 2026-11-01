"""Value-shape helpers for replay: in-place mutation detection, tensor cloning and template checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .._data_classes.op import Op


def _looks_like_template_dict(component: tuple[Any, ...]) -> bool:
    """Return whether a tuple encodes a captured dict argument.

    Parameters
    ----------
    component:
        Tuple template component.

    Returns
    -------
    bool
        Whether all items are key/value pairs.
    """

    return all(isinstance(item, tuple) and len(item) == 2 for item in component)


def _is_inplace_none_return(site: Op) -> bool:
    """Return whether a None return should be treated as mutated arg zero.

    Parameters
    ----------
    site:
        Replayed site.

    Returns
    -------
    bool
        Whether to use the first positional argument as output.
    """

    func_name = getattr(site.func, "__name__", "") if site.func is not None else ""
    return bool(site.is_inplace) or func_name in {"__setitem__", "zero_", "__delitem__"}


def _call_mutates_tensor_args(site: Op, kwargs: Mapping[str, Any]) -> bool:
    """Return whether a replayed call may write into its argument tensors.

    Covers the captured in-place flag, the torch trailing-underscore
    convention (dunders excluded), the explicit mutator set from
    :func:`_is_inplace_none_return`, and a tensor ``out=`` destination. A
    false positive only costs one defensive clone; a false negative lets the
    replayed call mutate captured payloads by identity.

    Parameters
    ----------
    site:
        Replayed site.
    kwargs:
        Reconstructed keyword arguments.

    Returns
    -------
    bool
        Whether argument tensors must be cloned before execution.
    """

    if _is_inplace_none_return(site):
        return True
    func_name = getattr(site.func, "__name__", "") if site.func is not None else ""
    if func_name.endswith("_") and not func_name.endswith("__"):
        return True
    return isinstance(kwargs.get("out"), torch.Tensor)


def _clone_tensors_in(value: Any) -> Any:
    """Return ``value`` with every tensor leaf cloned, containers rebuilt.

    Parameters
    ----------
    value:
        Resolved argument value (tensor, container, or opaque object).

    Returns
    -------
    Any
        Structure with cloned tensor leaves; non-tensor leaves unchanged.
    """

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_tensors_in(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensors_in(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tensors_in(item) for key, item in value.items()}
    return value


def _is_namedtuple_instance(value: Any) -> bool:
    """Return whether a value is a namedtuple instance.

    Parameters
    ----------
    value:
        Candidate value.

    Returns
    -------
    bool
        Whether it is a namedtuple instance.
    """

    return isinstance(value, tuple) and hasattr(value, "_fields")

"""Value-shape helpers for replay: in-place mutation detection, tensor cloning and template checks."""

from __future__ import annotations

import hashlib
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


def tensor_content_digest(value: torch.Tensor) -> str:
    """Return the sha256 of a tensor's raw element bytes (dtype-agnostic).

    The bytes are read through a flat ``uint8`` view of the contiguous
    payload, so the digest is byte-identical to the historical
    ``numpy().tobytes()`` spelling for every numpy-representable dtype and
    additionally covers dtypes numpy cannot represent (``bfloat16``,
    ``float8``). Two tensors of the same shape and dtype digest equal iff
    they are bitwise equal (NaN payloads included); the digest carries no
    shape or dtype, so callers compare digests only within one site.
    """

    payload = value.detach().cpu()
    if payload.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    flat = payload.reshape(-1)
    if flat.stride() != (1,):
        # ``contiguous()`` and ``reshape`` leave a one-element payload's stride
        # alone (``x[:, 0]`` of a ``(1, 4)`` tensor is ``(1,)`` with stride 4,
        # and still "contiguous"), and the byte reinterpretation below needs
        # a unit stride. Materialize a dense copy only in that case.
        flat = flat.clone(memory_format=torch.contiguous_format)
    raw = flat.view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def rollback_snapshot(site: Op) -> dict[str, Any]:
    """Return the site fields a replay commit writes, under their live names.

    Every field ``_apply_out_update`` (through ``_set_saved_out_metadata``) or
    the fire-record replacement writes is listed here by its live attribute
    name: a failed commit sets them back verbatim, so a stale key would turn
    the error it unwinds into a bare ``AttributeError``.

    Parameters
    ----------
    site:
        Op record about to be updated.

    Returns
    -------
    dict[str, Any]
        Field name to the value held before the update.
    """

    return {
        "out": site.out,
        "transformed_out": site.transformed_out,
        "shape": site.shape,
        "transformed_out_shape": site.transformed_out_shape,
        "dtype": site.dtype,
        "dtype_ref": site.dtype_ref,
        "device_ref": site.device_ref,
        "transformed_out_dtype": site.transformed_out_dtype,
        "activation_memory": site.cost.activation_memory,
        "transformed_activation_memory": site.cost.transformed_activation_memory,
        "has_saved_activation": site.has_saved_activation,
        "edits": list(site.edits),
        "is_edited": site.is_edited,
    }

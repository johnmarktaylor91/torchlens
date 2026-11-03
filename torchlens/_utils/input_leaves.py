"""Leaf-preserving copies and device moves of grad-tracked model inputs.

Validation copies the caller's inputs for its private forwards (ground truth,
validation capture, re-trace). A grad-tracked leaf input copied through the
graph-connected clone becomes a NON-leaf, so a model that sets
``x.requires_grad = True`` on its input raises inside validation although its
plain forward runs. Validation never promises that gradients reach the
caller's input, so inside :func:`leaf_preserving_input_copies` a leaf input is
copied or moved as a new leaf with the same ``requires_grad``. A plain
``tl.trace`` keeps the graph-connected clone, whose contract is that gradients
still reach the caller's input.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

import torch

from .tensor_utils import _clone_tensor_payload, _copy_tensor_payload

INPUT_WAS_PARAMETER_ATTR = "_torchlens_input_was_parameter"

_PRESERVE_LEAF_INPUT_COPIES: ContextVar[bool] = ContextVar(
    "torchlens_preserve_leaf_input_copies", default=False
)
"""Whether input copies keep grad-tracked leaf inputs as leaves (see the module docstring)."""


@contextmanager
def leaf_preserving_input_copies() -> Iterator[None]:
    """Make every model-input copy and device move inside the block keep leaf inputs as leaves."""

    token = _PRESERVE_LEAF_INPUT_COPIES.set(True)
    try:
        yield
    finally:
        _PRESERVE_LEAF_INPUT_COPIES.reset(token)


def leaf_input_copies_preserved() -> bool:
    """Return whether the current block keeps grad-tracked leaf inputs as leaves."""

    return _PRESERVE_LEAF_INPUT_COPIES.get()


def move_input_tensor(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """Move one input tensor to ``device``, keeping a grad-tracked leaf a leaf in validation.

    ``tensor.to(device)`` of a grad-tracked leaf is a NON-leaf, so a model that
    sets ``x.requires_grad = True`` on its input raised inside validation on a
    device move although its plain forward runs. Inside
    :func:`leaf_preserving_input_copies` the move makes a new leaf with the same
    ``requires_grad``; elsewhere the historical graph-connected move keeps
    gradients reaching the caller's input.

    Parameters
    ----------
    tensor:
        Input tensor on another device.
    device:
        Target device.

    Returns
    -------
    torch.Tensor
        The moved tensor.
    """

    if not (tensor.is_leaf and tensor.requires_grad and _PRESERVE_LEAF_INPUT_COPIES.get()):
        return tensor.to(device)
    moved = tensor.detach().to(device).requires_grad_(True)
    if getattr(tensor, INPUT_WAS_PARAMETER_ATTR, False):
        setattr(moved, INPUT_WAS_PARAMETER_ATTR, True)
    return moved


def leaf_deepcopy_preserves_contract(value: torch.Tensor) -> bool:
    """Return whether the storage-preserving deepcopy may copy this exact-``Tensor`` leaf.

    A leaf without autograd history always may; a grad-tracked leaf only inside
    :func:`leaf_preserving_input_copies` (deepcopy keeps ``requires_grad`` and
    yields a new leaf), elsewhere the graph-connected clone keeps gradients
    reaching the caller's input.
    """

    return (
        value.is_leaf
        and type(value) is torch.Tensor
        and (_PRESERVE_LEAF_INPUT_COPIES.get() or not value.requires_grad)
    )


def _clone_input_tensor_payload(arg: torch.Tensor) -> torch.Tensor:
    """Clone a forward-input tensor without preserving ``nn.Parameter`` type.

    Parameters
    ----------
    arg
        Forward input tensor or parameter to clone. Inside
        :func:`leaf_preserving_input_copies` a grad-tracked leaf Parameter is
        copied as a new leaf with the same ``requires_grad``.

    Returns
    -------
    torch.Tensor
        Plain tensor clone for model input replay. Parameter inputs intentionally
        become tensors so input provenance wins over Python wrapper type.
    """

    if isinstance(arg, torch.nn.Parameter):
        # Deferred as in the pre-split module: ``_state`` sits above ``_utils`` in the
        # layer map, so an eager import here would be an upward inversion.
        from .._state import pause_logging

        detach_leaf = _PRESERVE_LEAF_INPUT_COPIES.get() and arg.is_leaf and arg.requires_grad
        with pause_logging():
            cloned = _copy_tensor_payload(arg, detach_tensor=detach_leaf, save_mode="copy")
            if detach_leaf:
                cloned.requires_grad_(True)
        setattr(cloned, INPUT_WAS_PARAMETER_ATTR, True)
        return cloned
    return cast(torch.Tensor, _clone_tensor_payload(arg, detach_tensor=False, save_mode="copy"))


def flatten_input_tensor_addresses(
    input_arg_tensors: list[list[Any]],
    input_kwarg_tensors: list[list[Any]],
    input_arg_names: list[str],
    input_kwargs: dict[Any, Any],
) -> tuple[list[Any], list[str]]:
    """Flatten the leaf lists into distinct tensors and their source-address labels."""
    # Build flat lists of (tensor, address) for both positional and keyword args.
    # Address format: "input.<argname>" or "input.<argname>.<nested_path>"
    input_tensors = []
    input_tensor_addresses = []
    seen_tensor_ids: set[int] = set()
    for arg_idx, arg_tensors in enumerate(input_arg_tensors):
        for tensor, addr, addr_full in arg_tensors:
            if id(tensor) in seen_tensor_ids:
                continue
            seen_tensor_ids.add(id(tensor))
            input_tensors.append(tensor)
            tensor_addr = f"input.{input_arg_names[arg_idx]}"
            if addr != "":
                tensor_addr += f".{addr}"
            input_tensor_addresses.append(tensor_addr)

    for arg_idx, kwarg_tensors in enumerate(input_kwarg_tensors):
        for tensor, addr, addr_full in kwarg_tensors:
            if id(tensor) in seen_tensor_ids:
                continue
            seen_tensor_ids.add(id(tensor))
            input_tensors.append(tensor)
            tensor_addr = f"input.{list(input_kwargs.keys())[arg_idx]}"
            if addr != "":
                tensor_addr += f".{addr}"
            input_tensor_addresses.append(tensor_addr)

    return input_tensors, input_tensor_addresses

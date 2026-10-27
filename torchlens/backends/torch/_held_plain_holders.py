"""Tensors held on plain (non-Module) objects that a module holds.

A ``torch.distributions`` prior or a diffusers scheduler built in ``__init__``
holds tensors that exist before the forward, exactly like a module tensor
attribute. ``buffer_writes.iter_module_held_plain_tensors`` stamps them as
buffer sources through ``iter_plain_holder_tensors``.
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

_NOT_PLAIN_HOLDERS = (
    torch.Tensor,
    nn.Module,
    type,
    types.ModuleType,
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    list,
    tuple,
    dict,
)


def plain_holder_tensor_attrs(item: Any) -> list[tuple[str, torch.Tensor]]:
    """Return a plain holder object's direct public non-Parameter tensor attributes.

    Modules, tensors, containers, classes, Python modules and functions are not
    plain holders, and neither is an object without an instance ``__dict__``
    (numbers, strings, ``torch.device``). Only the object's own attributes are
    read, never containers inside it, so a tokenizer's vocabulary costs one
    attribute read. Private (``_``) and ``tl_`` attributes are skipped, as on
    modules.

    Parameters
    ----------
    item:
        A module attribute value.

    Returns
    -------
    list[tuple[str, torch.Tensor]]
        ``(attribute name, tensor)`` pairs in attribute order; empty when
        ``item`` is not a plain holder or holds no such tensor.
    """

    if isinstance(item, _NOT_PLAIN_HOLDERS):
        return []
    try:
        attrs = vars(item)
    except TypeError:
        return []
    if not isinstance(attrs, dict):
        return []
    return [
        (name, value)
        for name, value in list(attrs.items())
        if isinstance(name, str)
        and not name.startswith(("_", "tl_"))
        and isinstance(value, torch.Tensor)
        and not isinstance(value, nn.Parameter)
    ]


def iter_plain_holder_tensors(
    scan: Any, attr_name: str, holder: object
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for a plain holder's public tensor attributes.

    Each name is ``<attr_name>[<attr>]``, reserved, admitted and stamped through
    ``scan`` (the module's ``buffer_writes._HeldScan``); the holder's scan stops at
    the first name the object bound refuses.
    """

    for held_attr, item in plain_holder_tensor_attrs(holder):
        name = scan.reserve(f"{attr_name}[{held_attr}]")
        if not scan.admit(name):
            return
        scan.stamp(item)
        yield name, item

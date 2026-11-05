"""Autograd-node facts for in-place view writes: creation order and view classes.

The in-place view-write rebind (``backends/torch/_user_grad_fn.py``) and the
backward-graph invariant read two private torch surfaces: an autograd node's
``_sequence_nr()`` and the alias annotations on aten op schemas. Each is gated by
a capability flag in :mod:`torchlens._utils._torch_compat`
(``HAS_NODE_SEQUENCE_NR``, ``HAS_SCHEMA_ALIAS_INFO``), so ``doctor()`` reports a
torch build that lacks one. Without the capability the helpers answer "unknown"
(``None`` or ``False``) and their callers refuse to rebind.
"""

from __future__ import annotations

import functools
import re
from typing import Any

import torch

from . import _torch_compat

_BACKWARD_CLASS = re.compile(r"([A-Z][A-Za-z0-9]*?)Backward\d+")


def autograd_node_sequence_nr(node: Any) -> int | None:
    """Return an autograd node's creation sequence number, or ``None`` when unknown.

    Parameters
    ----------
    node:
        Autograd node (``grad_fn``) or any object.

    Returns
    -------
    int | None
        The node's ``_sequence_nr()``; ``None`` for a non-node, or when the torch
        build has no ``HAS_NODE_SEQUENCE_NR``.
    """

    if not _torch_compat.HAS_NODE_SEQUENCE_NR:
        return None
    read = getattr(node, "_sequence_nr", None)
    return read() if callable(read) else None


@functools.cache
def is_view_backward_class(class_name: str) -> bool:
    """Return whether ``class_name`` is the backward node of an aten view op.

    The class name (``SliceBackward0``) names its forward op (``aten::slice``); the
    op is a view when one of its schemas takes an argument that its result
    aliases without writing it (``Tensor(a) self``). Read from torch's own
    schemas, so new view ops need no list here; any name that does not resolve,
    or a torch build without ``HAS_SCHEMA_ALIAS_INFO``, is not a view.

    Parameters
    ----------
    class_name:
        Autograd node class name.

    Returns
    -------
    bool
        ``True`` only for a resolvable view op's backward class.
    """

    if not _torch_compat.HAS_SCHEMA_ALIAS_INFO:
        return False
    match = _BACKWARD_CLASS.fullmatch(class_name)
    if match is None:
        return False
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", match.group(1)).lower()
    return _aten_op_is_view(snake) or _aten_op_is_view(f"_{snake}")


def _aten_op_is_view(op_name: str) -> bool:
    """Return whether ``aten::<op_name>`` has a schema whose result aliases an input unwritten."""

    try:
        packet = getattr(torch.ops.aten, op_name)
        schemas = [getattr(packet, overload)._schema for overload in packet.overloads()]
    except (AttributeError, RuntimeError):
        return False
    return any(
        arg.alias_info is not None and not arg.alias_info.is_write
        for schema in schemas
        for arg in schema.arguments
    )

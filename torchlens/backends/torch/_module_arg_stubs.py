"""Module-arg payload stubbing for release-at-emission (F20 W1a).

Split out of ``model_prep.py`` under the R43 file-size ratchet. The
module-arg stash and the ``ModuleEnterEvent`` used to pin every module
call's live input tensors for the whole forward -- a dominant capture-floor
holder (brainpipe memo section 3.3). Tensor leaves are replaced at module
entry with :class:`~torchlens.ir.workspaces.ReleasedTensorStub` surrogates
carrying exactly what the torch trace path's consumers read (shape/dtype
facts for ``format_call_arg`` summaries and quantized-FLOPs estimation;
GC-11 nulls ``ModuleCall.forward_args`` before the trace is returned).
"""

from __future__ import annotations

from typing import Any, cast

import torch

from ...ir.workspaces import ReleasedTensorStub
from ._tl import get_tensor_label

_MODULE_ARG_STUB_MAX_DEPTH = 5


def stub_module_arg_payloads(value: Any, _depth: int = 0) -> Any:
    """Replace tensor leaves in a module-arg tree with payload-free stubs.

    Parameters
    ----------
    value:
        Module-forward argument tree (positional tuple or kwargs dict).
    _depth:
        Internal recursion depth (callers must not supply this).

    Returns
    -------
    Any
        Tree of equal structure with each ``torch.Tensor`` leaf replaced by
        a :class:`ReleasedTensorStub`. Exotic containers beyond plain
        list/tuple/dict are left untouched (a disclosed residual: tensors
        inside them stay pinned).
    """

    if isinstance(value, torch.Tensor):
        return ReleasedTensorStub(
            shape=tuple(value.shape),
            dtype=str(value.dtype).removeprefix("torch."),
            device=str(value.device),
            label_raw=get_tensor_label(value),
        )
    if _depth >= _MODULE_ARG_STUB_MAX_DEPTH:
        return value
    if isinstance(value, tuple):
        rebuilt = tuple(stub_module_arg_payloads(item, _depth + 1) for item in value)
        is_namedtuple = hasattr(value, "_make") and hasattr(value, "_fields")
        return cast(Any, type(value))._make(rebuilt) if is_namedtuple else rebuilt
    if isinstance(value, list):
        return [stub_module_arg_payloads(item, _depth + 1) for item in value]
    if isinstance(value, dict):
        return {key: stub_module_arg_payloads(item, _depth + 1) for key, item in value.items()}
    return value


def first_stub_shape(value: Any, search_depth: int = 5) -> tuple[int, ...] | None:
    """Return the recorded shape of the first stub found in ``value``.

    The stub twin of ``_first_tensor_shape``'s tensor scan, serving the
    quantized-FLOPs estimator on released arg trees.

    Parameters
    ----------
    value:
        Object tree to search.
    search_depth:
        Maximum container nesting inspected.

    Returns
    -------
    tuple[int, ...] | None
        First stub shape, or ``None`` when no stub is present.
    """

    from ...utils.introspection import get_vars_of_type_from_obj

    stubs = get_vars_of_type_from_obj(value, ReleasedTensorStub, search_depth=search_depth)
    if stubs:
        return tuple(stubs[0].shape)
    return None

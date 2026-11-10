"""Module-arg payload stubbing for release-at-emission.

Split out of ``model_prep.py``. The
module-arg stash and the ``ModuleEnterEvent`` used to pin every module
call's live input tensors for the whole forward -- a dominant capture-floor
holder. Tensor leaves are replaced at module
entry with :class:`~torchlens._ir.workspaces.ReleasedTensorStub` surrogates
carrying exactly what the torch trace path's consumers read (shape/dtype
facts for ``format_call_arg`` summaries and quantized-FLOPs estimation;
``ModuleCall.forward_args`` is nulled before the trace is returned).
"""

from __future__ import annotations

from typing import Any, cast

import torch

from ..._ir.workspaces import ReleasedObjectStub, ReleasedTensorStub
from ..._utils._tensor_label import get_tensor_label
from ..._utils.introspection import get_vars_of_type_from_obj

_MODULE_ARG_STUB_MAX_DEPTH = 5
#: ``module_identity_mode`` values whose ``ModuleCall.forward_args`` and
#: ``forward_kwargs`` stay on the returned trace (``_postprocess/finalization.py``);
#: there an opaque argument object stays the live object.
_KEEPS_FORWARD_ARGS = frozenset({"pytree_module", "object_module"})
#: Values ``format_call_arg`` renders by ``repr``; every other opaque value renders
#: as ``<TypeName>``, which a :class:`ReleasedObjectStub` reproduces.
_SUMMARY_LITERALS = (bool, int, float, str, type(None))


def stub_module_arg_payloads(
    value: Any, _depth: int = 0, *, identity_mode: str | None = None
) -> Any:
    """Replace tensor leaves in a module-arg tree with payload-free stubs.

    Parameters
    ----------
    value:
        Module-forward argument tree (positional tuple or kwargs dict).
    _depth:
        Internal recursion depth (callers must not supply this).
    identity_mode:
        The capture's ``module_identity_mode``. When it nulls the module call's
        forward arguments before the trace is returned, every other opaque
        object is replaced by a :class:`ReleasedObjectStub` too.

    Returns
    -------
    Any
        Tree of equal structure with each ``torch.Tensor`` leaf replaced by
        a :class:`ReleasedTensorStub`. Exotic containers beyond plain
        list/tuple/dict are left untouched unless ``identity_mode`` releases
        them (otherwise a disclosed residual: tensors inside them stay pinned).
    """

    release_opaque = identity_mode is not None and identity_mode not in _KEEPS_FORWARD_ARGS
    return _stub(value, _depth, release_opaque)


def _stub(value: Any, depth: int, release_opaque: bool) -> Any:
    """Return ``value`` with tensor leaves (and, if asked, opaque holders) stubbed."""

    if isinstance(value, torch.Tensor):
        return ReleasedTensorStub(
            shape=tuple(value.shape),
            dtype=str(value.dtype).removeprefix("torch."),
            device=str(value.device),
            label_raw=get_tensor_label(value),
        )
    if depth >= _MODULE_ARG_STUB_MAX_DEPTH:
        return value
    if isinstance(value, tuple):
        rebuilt = tuple(_stub(item, depth + 1, release_opaque) for item in value)
        is_namedtuple = hasattr(value, "_make") and hasattr(value, "_fields")
        return cast(Any, type(value))._make(rebuilt) if is_namedtuple else rebuilt
    if isinstance(value, list):
        return [_stub(item, depth + 1, release_opaque) for item in value]
    if isinstance(value, dict):
        return {key: _stub(item, depth + 1, release_opaque) for key, item in value.items()}
    if release_opaque and not isinstance(value, _SUMMARY_LITERALS):
        return ReleasedObjectStub(type_name=type(value).__name__)
    return value


def stub_module_call_args(args: Any, kwargs: Any, trace: Any) -> tuple[Any, Any]:
    """Stub a module call's positional and keyword arguments for the entry record.

    Parameters
    ----------
    args, kwargs:
        The module call's positional tuple and keyword dict.
    trace:
        The capturing trace; its ``module_identity_mode`` option decides whether
        opaque argument objects are released (see :func:`stub_module_arg_payloads`).

    Returns
    -------
    tuple
        ``(stubbed args, stubbed kwargs)``.
    """

    mode = trace._options_module_identity_mode
    return (
        stub_module_arg_payloads(args, identity_mode=mode),
        stub_module_arg_payloads(kwargs, identity_mode=mode),
    )


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

    stubs = get_vars_of_type_from_obj(value, ReleasedTensorStub, search_depth=search_depth)
    if stubs:
        return tuple(stubs[0].shape)
    return None

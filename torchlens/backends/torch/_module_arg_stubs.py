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
from ..._utils._tensor_label import clear_tensor_label, get_tensor_label, raw_tensor_label
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
    value: Any,
    _depth: int = 0,
    labels: dict[int, str | None] | None = None,
    *,
    identity_mode: str | None = None,
) -> Any:
    """Replace tensor leaves in a module-arg tree with payload-free stubs.

    Parameters
    ----------
    value:
        Module-forward argument tree (positional tuple or kwargs dict).
    _depth:
        Internal recursion depth (callers must not supply this).
    labels:
        Labels the caller has just read for some tensors, by ``id``; any other
        tensor's label is read here.
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
    return _stub(value, _depth, release_opaque, labels)


def _stub(
    value: Any, depth: int, release_opaque: bool, labels: dict[int, str | None] | None
) -> Any:
    """Return ``value`` with tensor leaves (and, if asked, opaque holders) stubbed."""

    if isinstance(value, torch.Tensor):
        return ReleasedTensorStub(
            shape=tuple(value.shape),
            dtype=str(value.dtype).removeprefix("torch."),
            device=str(value.device),
            label_raw=(
                labels[id(value)]
                if labels is not None and id(value) in labels
                else get_tensor_label(value)
            ),
        )
    if depth >= _MODULE_ARG_STUB_MAX_DEPTH:
        return value
    if isinstance(value, (tuple, list, dict)):
        return _stub_container(value, depth + 1, release_opaque, labels)
    if release_opaque and not isinstance(value, _SUMMARY_LITERALS):
        return ReleasedObjectStub(type_name=type(value).__name__)
    return value


def _stub_container(
    value: Any, depth: int, release_opaque: bool, labels: dict[int, str | None] | None
) -> Any:
    """Rebuild a tuple (namedtuples keep their type), list or dict with stubbed items."""

    if isinstance(value, dict):
        return {key: _stub(item, depth, release_opaque, labels) for key, item in value.items()}
    if isinstance(value, list):
        return [_stub(item, depth, release_opaque, labels) for item in value]
    rebuilt = tuple(_stub(item, depth, release_opaque, labels) for item in value)
    is_namedtuple = hasattr(value, "_make") and hasattr(value, "_fields")
    return cast(Any, type(value))._make(rebuilt) if is_namedtuple else rebuilt


def stub_module_call_args(
    args: Any, kwargs: Any, trace: Any, labels: dict[int, str | None] | None = None
) -> tuple[Any, Any]:
    """Stub a module call's positional and keyword arguments for the entry record.

    Parameters
    ----------
    args, kwargs:
        The module call's positional tuple and keyword dict.
    trace:
        The capturing trace; its ``module_identity_mode`` option decides whether
        opaque argument objects are released (see :func:`stub_module_arg_payloads`).
    labels:
        Labels read earlier for some argument tensors, by ``id`` (see
        :func:`stub_module_arg_payloads`).

    Returns
    -------
    tuple
        ``(stubbed args, stubbed kwargs)``.
    """

    if type(kwargs) is dict and not kwargs:
        stubbed = _stub_plain_tensor_tuple(args, labels)
        if stubbed is not None:
            return stubbed, {}
    mode = trace._options_module_identity_mode
    return (
        stub_module_arg_payloads(args, labels=labels, identity_mode=mode),
        stub_module_arg_payloads(kwargs, labels=labels, identity_mode=mode),
    )


def _stub_plain_tensor_tuple(
    args: Any, labels: dict[int, str | None] | None
) -> tuple[Any, ...] | None:
    """Return a plain tuple of tensors stubbed as the general walk stubs it, else ``None``.

    The common module call passes only tensors, positionally. The walk would rebuild
    the tuple item by item (no identity mode releases a tensor) and the empty keyword
    dict as a new empty dict; this does the same without the container dispatch.
    """
    if type(args) is not tuple:
        return None
    stubbed = []
    for value in args:
        if not isinstance(value, torch.Tensor):
            return None
        stubbed.append(_stub(value, 1, False, labels))
    return tuple(stubbed)


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


def input_labels_by_id(input_tensors: list[Any]) -> dict[int, str | None]:
    """Return each input tensor's gated label (``get_tensor_label``), keyed by ``id``.

    A module entry reads its input tensors' labels once, for the replay template,
    the input-adoption loop (:func:`entry_live_label`) and the payload stubs.
    """

    return {id(t): get_tensor_label(t) for t in input_tensors}


def entry_live_label(
    t: Any, known: dict[int, str | None] | None, live_labels: Any
) -> tuple[str | None, bool]:
    """Return ``get_live_tensor_label(t, live_labels)`` from a label read earlier.

    Parameters
    ----------
    t:
        Input tensor.
    known:
        :func:`input_labels_by_id` of the call, valid while no stamp has changed
        since it was read; ``None`` reads the label now.
    live_labels:
        Raw labels live in the active trace.

    Returns
    -------
    tuple[str | None, bool]
        The live label, and whether a stale stamp was cleared off ``t``.
    """

    label = get_tensor_label(t) if known is None else known[id(t)]
    if label is not None and label in live_labels:
        return label, False
    if raw_tensor_label(t) is not None:
        clear_tensor_label(t)
        return None, True
    return None, False

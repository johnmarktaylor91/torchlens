"""In-place mutation of Parameters during torch capture (receiver policy).

A prepared model Parameter is a source: every read of it is a parameter edge to
its ``Param``. When forward code mutates one in place (``with torch.no_grad():
self.temp.clamp_(lo, hi)``) the op is logged with the Parameter as its parameter
input and later reads bind to that op. This module holds the receiver policy the
wrapper consults: which Parameters count as receivers, how a mutation result is
turned into a loggable tensor, and when a frozen receiver runs untracked.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch

from ... import _state
from ...utils._torch_compat import HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE
from ...utils.tensor_utils import safe_copy
from ._tl import get_param_meta


def is_unregistered_parameter(trace: Any, value: Any) -> bool:
    """Return whether ``value`` is a Parameter outside the prepared model state.

    Parameters
    ----------
    trace:
        Active capture trace carrying the current-session parameter registry.
    value:
        Candidate operation output.

    Returns
    -------
    bool
        ``True`` for a Parameter that is not the exact prepared parameter object
        recorded at its stamped address in this capture.
    """

    if not isinstance(value, torch.nn.Parameter):
        return False
    meta = get_param_meta(value)
    address = None if meta is None else meta.param_address
    if not address:
        return True
    param_logs = getattr(trace, "param_logs", None)
    if param_logs is None or address not in param_logs:
        return True
    return getattr(param_logs[address], "_param_ref", None) is not value


def is_prepared_parameter_receiver(trace: Any, value: Any) -> bool:
    """Return whether an in-place write to ``value`` is a prepared-state mutation.

    Only a prepared Parameter that was already initialized at prep qualifies. A
    lazy module's ``UninitializedParameter`` has no value when the pass starts:
    its in-forward materialization (``materialize`` then ``reset_parameters``'s
    ``uniform_`` family) creates the Parameter's first value rather than mutating
    a prepared one, so those writes keep the plain-capture treatment (no ops).

    Parameters
    ----------
    trace:
        Active capture trace carrying the current-session parameter registry.
    value:
        Receiver of an in-place call.

    Returns
    -------
    bool
        ``True`` for an initialized-at-prep Parameter of this capture's model.
    """

    if not isinstance(value, torch.nn.Parameter) or is_unregistered_parameter(trace, value):
        return False
    meta = get_param_meta(value)
    if meta is None:
        return False
    return not getattr(trace.param_logs[meta.param_address], "_lazy_at_prep", False)


def _iter_operand_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield the tensors in one call operand, descending into plain containers.

    Parameters
    ----------
    value:
        A positional or keyword argument of a wrapped call.

    Yields
    ------
    torch.Tensor
        Each tensor found directly or inside nested tuples, lists and dicts.
    """

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_operand_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_operand_tensors(item)


def frozen_parameter_receiver_context(
    trace: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    is_inplace_call: bool,
) -> AbstractContextManager[Any]:
    """Run an in-place op on a frozen prepared Parameter untracked, as eager does.

    Capture forces ``requires_grad=True`` on floating prepared Parameters so their
    reads are gradient-capable. A Parameter frozen by the user
    (``requires_grad=False``) may legally be mutated in place with grad mode on
    (EMA weights, fixed tables); under the forced flag autograd would refuse the
    same call. When no other operand requires grad, the eager call records no
    autograd history, so executing it under ``torch.no_grad()`` is value- and
    graph-identical to eager. Operands are scanned positionally, by keyword and
    inside nested containers: with any grad-requiring operand eager would make
    the Parameter a non-leaf, so the call runs unchanged and autograd refuses it
    loudly instead of silently dropping that history.

    Parameters
    ----------
    trace:
        Active capture trace carrying the current-session parameter registry.
    args:
        Positional arguments of the wrapped call; ``args[0]`` is the receiver.
    kwargs:
        Keyword arguments of the wrapped call.
    is_inplace_call:
        Whether the call mutates its receiver (in-place name or ``inplace=``).

    Returns
    -------
    AbstractContextManager[Any]
        ``torch.no_grad()`` for a frozen prepared Parameter receiver, otherwise a
        null context.
    """

    if not is_inplace_call or not args or not torch.is_grad_enabled():
        return nullcontext()
    receiver = args[0]
    if not isinstance(receiver, torch.nn.Parameter) or not receiver.requires_grad:
        return nullcontext()
    if not is_prepared_parameter_receiver(trace, receiver):
        return nullcontext()
    meta = get_param_meta(receiver)
    if meta is None or meta.requires_grad_before_capture is not False:
        return nullcontext()
    for operand in (*args[1:], *kwargs.values()):
        if any(tensor.requires_grad for tensor in _iter_operand_tensors(operand)):
            return nullcontext()
    return torch.no_grad()


def parameter_mutation_output_for_logging(
    trace: Any,
    value: Any,
    *,
    source: Any,
    was_inplace: bool,
    is_storage_rebind: bool = False,
) -> Any:
    """Convert a Parameter mutation result into a loggable Tensor.

    PyTorch constructs module Parameters from ordinary factory tensors, and the
    conversion intentionally drops TorchLens tensor labels. Initializers such as
    ``uniform_`` then return the new Parameter itself. Parameter outputs are normally
    excluded because prepared model state remains a source rather than an op output,
    but that rule also dropped real initialization ops for modules created inside
    ``forward``.

    A prepared model Parameter mutated in place (``with torch.no_grad():
    self.temp.clamp_(lo, hi)``) is converted too: the op is logged with the
    Parameter as its parameter input, and the live Parameter then carries the op's
    label so every later read in the pass consumes the op's output. Two prepared
    cases are NOT converted and keep the plain-capture treatment: a
    storage-rebinding ``param.data = rhs`` setter (still visible to the
    completeness tripwire) and the in-forward initialization of a lazy module's
    Parameter (see ``is_prepared_parameter_receiver``).

    Parameters
    ----------
    trace:
        Active capture trace.
    value:
        Raw callable return value.
    source:
        Live same-object return whose current-session registration is authoritative.
    was_inplace:
        Whether the wrapped callable has an in-place mutation signature.
    is_storage_rebind:
        Whether the call is a storage-rebinding ``.data`` setter.

    Returns
    -------
    Any
        A plain Tensor snapshot for in-place Parameter mutations; otherwise
        ``value`` unchanged.
    """

    if not was_inplace or not isinstance(source, torch.nn.Parameter):
        return value
    if not is_unregistered_parameter(trace, source) and (
        is_storage_rebind or not is_prepared_parameter_receiver(trace, source)
    ):
        return value
    with _state.pause_logging():
        if HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE:
            plain_value = value.as_subclass(torch.Tensor)
        else:
            plain_value = torch.ops.aten.detach.default(value)
        tensor = safe_copy(plain_value, detach_tensor=True)
        tensor.requires_grad_(value.requires_grad)
    return tensor

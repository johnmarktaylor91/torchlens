"""In-place mutation of Parameters during torch capture (receiver policy).

A prepared model Parameter is a source: every read of it is a parameter edge to
its ``Param``. When forward code mutates one in place (``with torch.no_grad():
self.temp.clamp_(lo, hi)``) the op is logged with the Parameter as its parameter
input and later reads bind to that op. This module holds the receiver policy the
wrapper consults: which Parameters count as receivers, how a mutation result is
turned into a loggable tensor, how a frozen receiver runs, and how a receiver
passed by keyword is normalized.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from typing import Any

import torch

from ... import _state
from ..._capture.projections import active_recording_state_or_none
from ..._data_classes._trace_missing_attrs import is_live_trace
from ..._utils._torch_compat import HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE
from ..._utils.tensor_leaves import iter_tensors
from ..._utils.tensor_utils import safe_copy
from ._completeness_origins import internal_scalar_read
from ._tl import (
    NO_USER_GRAD_FN,
    get_param_meta,
    get_tensor_label,
    mutated_parameter_label,
    set_tensor_label,
    storage_alias_index_key,
)


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
    param_logs = trace._param_logs
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
    return not getattr(trace._param_logs[meta.param_address], "_lazy_at_prep", False)


def is_mutated_parameter_receiver(trace: Any, value: Any) -> bool:
    """Return whether an in-place write to ``value`` becomes a Parameter's provenance.

    True for a prepared Parameter initialized at prep (its reads after the write
    bind to the mutation op) and for a Parameter the forward itself created: a
    module built inside ``forward`` initializes its Parameters with in-place
    writes (``uniform_`` family) that are their only origin, so the live Parameter
    takes the op's label and the later read is attributed instead of orphaning
    the initializer ops. A prepared lazy Parameter's in-forward materialization
    stays out (see ``is_prepared_parameter_receiver``).

    Parameters
    ----------
    trace:
        Active capture trace carrying the current-session parameter registry.
    value:
        Receiver of an in-place call.

    Returns
    -------
    bool
        ``True`` for an initialized-at-prep or forward-created Parameter.
    """

    if not isinstance(value, torch.nn.Parameter):
        return False
    lazy_at_prep = _prepared_parameter_lazy_flag(trace, value)
    return lazy_at_prep is None or not lazy_at_prep


def same_object_logging_copy(out_orig: Any) -> Any:
    """Return the distinct tensor a same-object return is logged against.

    Logging against the live receiver would clobber its label with the op's. The
    copy carries the USER op's live autograd node (round-31 M5): the safe copy's
    own ``grad_fn`` is TorchLens's ``CloneBackward`` bookkeeping and must never
    replace the operation's recorded autograd metadata (``grad_fn_class_*`` /
    handle). A leaf receiver (``x.detach().requires_grad_(True)``) has no live
    node: the sentinel makes the record carry ``None`` instead of the copy's
    phantom handle. The ``grad_fn`` read is marked as TorchLens bookkeeping: for
    a same-object in-place return the output IS the user's receiver (a registered
    buffer for BN's ``num_batches_tracked.add_(1)``), and an unmarked read would
    record a phantom declared-state fact (r65 unread-bit).
    """

    with internal_scalar_read():
        live_user_grad_fn = out_orig.grad_fn if isinstance(out_orig, torch.Tensor) else None
    logged = safe_copy(out_orig)
    if isinstance(logged, torch.Tensor):
        with suppress(AttributeError):
            setattr(  # noqa: B010 -- TorchLens stamp on a Tensor, not a declared attribute
                logged,
                "tl_user_grad_fn",
                NO_USER_GRAD_FN if live_user_grad_fn is None else live_user_grad_fn,
            )
    return logged


def adopt_minted_parameter_label(trace: Any, param: Any, sources: tuple[Any, ...]) -> None:
    """Give a forward-created Parameter the label of the tensor it was minted from.

    ``nn.Parameter(data)`` (``Tensor._make_subclass``) returns a new Parameter
    object on ``data``'s storage carrying none of ``data``'s TorchLens metadata,
    and a lazy module's ``materialize`` rebinds its pending slot's ``.data`` to a
    fresh allocation the same way. Inside the captured pass that tensor is a
    logged factory output (``torch.empty`` in ``nn.Linear.__init__``), so the
    Parameter adopts its label: the Parameter's first read or in-place
    initializer binds to the op that produced the storage instead of consuming
    unattributed data. A prepared Parameter never qualifies.

    Parameters
    ----------
    trace:
        Active capture trace.
    param:
        The Parameter the call returned or rebound.
    sources:
        Call arguments; the first labeled plain tensor on ``param``'s storage is
        the origin.
    """

    if not is_unregistered_parameter(trace, param) or mutated_parameter_label(param) is not None:
        return
    for source in sources:
        if isinstance(source, torch.Tensor) and not isinstance(source, torch.nn.Parameter):
            label = get_tensor_label(source)
            if label is not None and _same_storage(source, param):
                set_tensor_label(param, label)
                return


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return whether two tensors are on one storage (unreadable storage fails closed)."""

    try:
        key_a = storage_alias_index_key(a.untyped_storage())
        key_b = storage_alias_index_key(b.untyped_storage())
    except (RuntimeError, ValueError, AttributeError):
        return False
    return key_a is not None and key_a == key_b


def receiver_mutation_label(args: tuple[Any, ...]) -> str | None:
    """Return the mutation label a Parameter receiver carries before a call runs."""

    if not args or not isinstance(args[0], torch.nn.Parameter):
        return None
    return mutated_parameter_label(args[0])


def nested_write_relabeled_receiver(receiver: Any, label_before: str | None) -> bool:
    """Return whether a nested in-place write labeled ``receiver`` during this call.

    Initializers nest: ``kaiming_uniform_`` and ``init.uniform_`` each end in the
    bottom-level ``Tensor.uniform_`` write on the same Parameter. That write is the
    recorded op, and it labels the live Parameter; the enclosing non-bottom-level
    call then returns the same Parameter and must not be logged as a second op
    reading the first (the second op would be insensitive to its parent and
    replay would run the initializer twice).
    """

    if not isinstance(receiver, torch.nn.Parameter):
        return False
    label_now = mutated_parameter_label(receiver)
    return label_now is not None and label_now != label_before


@contextmanager
def _frozen_receiver_flag(receiver: torch.nn.Parameter) -> Iterator[None]:
    """Run one in-place call with a frozen Parameter's own ``requires_grad=False``.

    Capture forced the flag to ``True`` at prep. With the user's flag back, autograd
    accepts the in-place write of a grad-requiring operand exactly as in eager and
    the Parameter becomes a non-leaf, which is the state eager leaves it in, so the
    flag is not re-forced after a successful call. If the call raises, or leaves the
    Parameter a leaf, the forced flag is put back for the rest of the capture.

    Parameters
    ----------
    receiver:
        Frozen prepared Parameter receiving the in-place write.

    Yields
    ------
    None
        Control for the wrapped call.
    """

    from ._completeness_origins import internal_scalar_read

    with _state.pause_logging(), internal_scalar_read():
        receiver.requires_grad_(False)
    try:
        yield
    finally:
        with _state.pause_logging(), internal_scalar_read():
            if receiver.is_leaf:
                receiver.requires_grad_(True)


def frozen_parameter_receiver_context(
    trace: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    is_inplace_call: bool,
) -> AbstractContextManager[Any]:
    """Run an in-place op on a frozen prepared Parameter as eager does.

    Capture forces ``requires_grad=True`` on floating prepared Parameters so their
    reads are gradient-capable. A Parameter frozen by the user
    (``requires_grad=False``) may legally be mutated in place with grad mode on
    (EMA weights, fixed tables); under the forced flag autograd would refuse the
    same call. Operands are scanned positionally, by keyword and inside nested
    containers:

    - no operand requires grad: the eager call records no autograd history, so
      executing it under ``torch.no_grad()`` is value- and graph-identical;
    - some operand requires grad: eager makes the Parameter a non-leaf, so the
      call runs tracked with the receiver's own frozen flag restored
      (``_frozen_receiver_flag``) and the Parameter ends exactly as eager leaves it.

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
        ``torch.no_grad()`` or the frozen-flag context for a frozen prepared
        Parameter receiver, otherwise a null context.
    """

    if not is_inplace_call or not args or not torch.is_grad_enabled():
        return nullcontext()
    receiver = args[0]
    if not isinstance(receiver, torch.nn.Parameter):
        return nullcontext()
    from ._completeness_origins import internal_scalar_read

    with internal_scalar_read():
        is_forced_leaf = receiver.requires_grad and receiver.is_leaf
    if not is_forced_leaf or not is_prepared_parameter_receiver(trace, receiver):
        return nullcontext()
    meta = get_param_meta(receiver)
    if meta is None or meta.requires_grad_before_capture is not False:
        return nullcontext()
    for operand in (*args[1:], *kwargs.values()):
        if any(tensor.requires_grad for tensor in iter_tensors(operand)):
            return _frozen_receiver_flag(receiver)
    return torch.no_grad()


def receiver_kwarg_as_positional(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Move a Parameter receiver passed by keyword into the positional receiver slot.

    ``torch.clamp_(input=p, min=lo)`` mutates ``p`` exactly like
    ``torch.clamp_(p, min=lo)``, but the wrapper recognizes an in-place receiver
    only as ``args[0]``. For an in-place call with no positional argument and a
    Parameter ``input=`` keyword, the receiver is moved to ``args[0]`` (torch's
    in-place functions take ``input`` as their first positional parameter), so
    the call is captured like the positional spelling. Every other call is
    returned unchanged.

    Parameters
    ----------
    args:
        Positional arguments of an in-place-named wrapped call.
    kwargs:
        Keyword arguments of the same call.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        The (possibly normalized) positional and keyword arguments.
    """

    if args or not isinstance(kwargs.get("input"), torch.nn.Parameter):
        return args, kwargs
    rest = {name: value for name, value in kwargs.items() if name != "input"}
    return (kwargs["input"],), rest


def _prepared_parameter_lazy_flag(trace: Any, value: torch.nn.Parameter) -> bool | None:
    """Return whether prepared Parameter ``value`` was lazy at prep; ``None`` if unprepared.

    Exhaustive capture answers from the session ``Param`` registry. Predicate
    capture (``tl.record``) builds no ``Param`` logs, so it answers from the
    recording state's per-pass identity registry instead; without it every
    model Parameter looked unregistered there and a prepared ``param.data =
    rhs`` rebind or a lazy module's in-forward initialization was logged as an
    op that ``tl.trace`` never shows.
    """

    if trace is not None and is_live_trace(trace) and trace._options_mode == "predicate":
        state = active_recording_state_or_none()
        entry = None if state is None else state.prepared_params.get(id(value))
        if entry is None or entry[0]() is not value:
            return None
        return entry[1]
    if is_unregistered_parameter(trace, value):
        return None
    return not is_prepared_parameter_receiver(trace, value)


def _plain_parameter_snapshot(value: torch.nn.Parameter) -> torch.Tensor:
    """Return a detached plain-Tensor copy of a mutated Parameter, logging paused."""

    with _state.pause_logging():
        if HAS_PARAMETER_AS_SUBCLASS_IN_DISPATCH_MODE:
            plain_value = value.as_subclass(torch.Tensor)
        else:
            plain_value = torch.ops.aten.detach.default(value)
        tensor = safe_copy(plain_value, detach_tensor=True)
        tensor.requires_grad_(value.requires_grad)
    return tensor


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
    Parameter (see ``is_prepared_parameter_receiver``). Both capture modes
    classify alike: predicate capture reads its per-pass identity registry
    (``_prepared_parameter_lazy_flag``).

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
    lazy_at_prep = _prepared_parameter_lazy_flag(trace, source)
    if lazy_at_prep is not None and (is_storage_rebind or lazy_at_prep):
        return value
    return _plain_parameter_snapshot(value)

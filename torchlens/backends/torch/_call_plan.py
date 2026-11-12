"""Wrapper fast lane, first version: call plans, per-call admission and the fast commit.

A decorated torch callable gets a :class:`CallPlan` at decoration when its static
facts allow the fast lane: a C-level callable whose every dispatcher schema is
out-of-place and returns exactly one tensor, with no ``inplace`` parameter, no
fused running-statistic write and none of the wrapper's special roles. The
schema facts come from the ``get_jit_all_schemas`` capability probe
(``HAS_JIT_SCHEMA_ENUMERATION``); without it no callable gets a plan.

:func:`build_fast_plan` adds writing plans for two more kinds: an in-place
receiver write (``relu_``, ``__iadd__``) whose every schema writes exactly its
``self`` and returns it, and a fused normalization call (``batch_norm``) with an
out-of-place single-tensor schema, whose running-statistic writes the wrapper
snapshots before the call. Their commit, and the commit of a call that returns
its own receiver, is :func:`._ops_emission.commit_admitted`.

At call time :func:`admit_call` decides, before any bookkeeping side effect,
whether this call may skip the wrapper's general pre-call work. An admitted
call still records every call-time fact and still ends in today's emit
(:func:`commit_fast_call`). A call that is not admitted, or whose output fails
:func:`output_admitted`, runs the wrapper's general body unchanged.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch

from ... import _state
from ..._utils import _torch_compat
from ..._utils._tensor_label import TensorMeta
from ..._utils.rng import log_current_autocast_state
from ..._utils.tensor_utils import arm_deferred_payload_window, disarm_deferred_payload_window
from ._fused_norm_writes import _FUSED_MUTATOR_NAMES
from ._protocol_redispatch import STILL_DECLINED, redispatch_declined

if TYPE_CHECKING:
    from .buffer_writes import BufferWriteTracker

#: Functions of this file that the fast-lane share counter counts: one call per
#: operation the fast lane carried from admission to its emit.
FAST_LANE_COUNTER_NAMES: tuple[str, ...] = ("commit_fast_call",)

#: Shared do-nothing context managers for the wrapper's call scaffold.
NO_PAUSED_MODES = nullcontext(())
NULL_CONTEXT = nullcontext()

#: The admitted pre-call state: the call's tensor arguments in argument order,
#: whether today's buffer first-use and snapshot helpers must run, and the
#: autocast state recorded for the call.
Admission = tuple[list[torch.Tensor], bool, dict[str, dict[str, Any]]]

_TENSOR = torch.Tensor
_PARAMETER = torch.nn.Parameter
_C_CALLABLE_TYPES = (
    types.BuiltinFunctionType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
)
# Argument types that can hold no tensor. Anything outside this set, a plain
# tensor, a Parameter or a flat list or tuple of these sends the call to the
# general body.
_PLAIN_TYPES = frozenset(
    {
        int,
        float,
        bool,
        complex,
        str,
        type(None),
        torch.dtype,
        torch.device,
        torch.Size,
        torch.layout,
        torch.memory_format,
    }
)
_MODEL_OWNED_KEY = "model_owned_tensor_ids_at_entry"
_DECLINED = -1
# A flat list or tuple argument held a tensor (``torch.cat``): the call's
# tensor order then differs from today's argument walk, so it is admitted only
# when the buffer helpers, which consume that order, have nothing to do.
_LISTED = 2
#: Reflected dunders whose dispatcher operator is the forward one.
_REFLECTED_SCHEMA_NAMES = {"__radd__": "add", "__rmul__": "mul"}


class CallPlan:
    """Decoration-time facts of one wrapped callable that may take the fast lane.

    Parameters
    ----------
    name_could_mutate:
        Whether the buffer-write tracker classifies the callable's name as a
        possible mutator, which makes today's pre-call snapshot clear the
        tracker's storage metadata cache. For a planned name that is exactly a
        trailing underscore (every dunder has one): the in-place naming family
        gets no plan.
    writes:
        Whether the call may write an argument: a receiver write (``relu_``) or
        a fused normalization call's running statistics (``batch_norm``), both
        planned by :func:`build_fast_plan`. The wrapper then keeps its
        pre-mutation copy-out of deferred payloads for the call.
    """

    __slots__ = ("name_could_mutate", "writes")

    def __init__(self, name_could_mutate: bool, writes: bool = False) -> None:
        self.name_could_mutate = name_could_mutate
        self.writes = writes


def _schema_suits_fast_lane(schema: Any) -> bool | None:
    """Return whether one dispatcher schema leaves its operator eligible for the fast lane.

    ``None`` for a schema with no tensor in its arguments or returns (the
    scalar, string and list overloads TorchScript registers under names such as
    ``add``): it can never serve a wrapped call that yields a tensor, so it
    counts neither for nor against the name. Any other schema must return
    tensors only (one tensor, a tuple of tensors or one ``Tensor[]``) and write
    no argument. A keyword-only ``out`` destination is allowed here because a
    call that passes ``out=`` is refused per call; a multi-output call's return
    value is checked per call (:func:`output_admitted`).
    """

    returns = schema.returns
    arguments = schema.arguments
    touches_tensor = False
    for value in (*arguments, *returns):
        if "Tensor" in str(value.type):
            touches_tensor = True
            break
    if not touches_tensor:
        return None
    return_types = [str(value.type) for value in returns]
    if not return_types or (
        return_types not in (["Tensor[]"], ["List[Tensor]"])
        and any(kind != "Tensor" for kind in return_types)
    ):
        return False
    for argument in arguments:
        alias = argument.alias_info
        if alias is None or not alias.is_write:
            continue
        if not (argument.kwarg_only and argument.name == "out"):
            return False
    return True


@functools.cache
def single_tensor_schema_names() -> dict[str, bool]:
    """Return, per ``aten`` operator name, whether all its overloads suit the fast lane.

    Read once per process from the dispatcher's registered schemas.

    Returns
    -------
    dict[str, bool]
        Operator name to ``True`` when it has a tensor overload and every
        tensor overload is out-of-place and returns only tensors. Empty
        when the schema enumeration capability is absent, so no callable gets
        a plan.
    """

    names: dict[str, bool] = {}
    for schema in _torch_compat.get_jit_all_schemas() or ():
        namespace, _, name = schema.name.partition("::")
        verdict = _schema_suits_fast_lane(schema) if namespace == "aten" else None
        if verdict is not None:
            names[name] = names.get(name, True) and verdict
    return names


def build_call_plan(
    func: Callable[..., Any],
    func_name: str,
    property_accessor: str | None,
) -> CallPlan | None:
    """Return the fast-lane plan of a wrapped callable, or ``None`` when it has none.

    Parameters
    ----------
    func:
        The original torch callable.
    func_name:
        Name the callable is wrapped under.
    property_accessor:
        Accessor kind when ``func`` belongs to a wrapped property, else ``None``.

    Returns
    -------
    CallPlan | None
        The plan, or ``None`` for a Python-level callable, a property accessor,
        a fused running-statistic writer, a name in the in-place naming family
        or a name without an all-overloads tensor-returning out-of-place schema.
        A reflected dunder (``__radd__``, ``__rmul__``) reads its forward
        operator's schemas.
    """

    if property_accessor is not None or type(func) not in _C_CALLABLE_TYPES:
        return None
    if func_name in _FUSED_MUTATOR_NAMES or func_name.startswith("__i"):
        return None
    schema_names = single_tensor_schema_names()
    schema_name = _REFLECTED_SCHEMA_NAMES.get(func_name, func_name)
    if schema_name not in schema_names and func_name.startswith("__") and func_name.endswith("__"):
        schema_name = func_name[2:-2]
    if not schema_names.get(schema_name, False):
        return None
    return CallPlan(func_name.endswith("_"))


#: Augmented-assignment dunders whose in-place operator has another name.
_AUGMENTED_SCHEMA_NAMES = {"truediv_": "div_", "floordiv_": "floor_divide_", "mod_": "remainder_"}


def _inplace_schema_suits(schema: Any) -> bool | None:
    """Return whether one schema writes exactly its tensor ``self`` and returns it.

    ``None`` for a schema whose first argument is not a single tensor (it cannot
    serve a receiver-writing tensor method and counts neither way).
    """

    arguments = schema.arguments
    if not arguments or str(arguments[0].type) != "Tensor":
        return None
    alias = arguments[0].alias_info
    if alias is None or not alias.is_write:
        return False
    for argument in arguments[1:]:
        if argument.alias_info is not None and argument.alias_info.is_write:
            return False
    returns = schema.returns
    return len(returns) == 1 and str(returns[0].type) == "Tensor"


@functools.cache
def inplace_schema_names() -> dict[str, bool]:
    """Return, per ``aten`` operator name, whether every overload is a pure receiver write.

    Read once per process; empty without the schema enumeration capability.
    """

    names: dict[str, bool] = {}
    for schema in _torch_compat.get_jit_all_schemas() or ():
        namespace, _, name = schema.name.partition("::")
        verdict = _inplace_schema_suits(schema) if namespace == "aten" else None
        if verdict is not None:
            names[name] = names.get(name, True) and verdict
    return names


def _inplace_schema_name(func_name: str) -> str:
    """Return the ``aten`` name of a receiver-writing callable (``__iadd__`` reads ``add_``)."""

    if func_name.startswith("__i") and func_name.endswith("__"):
        name = func_name[3:-2] + "_"
        return _AUGMENTED_SCHEMA_NAMES.get(name, name)
    return func_name


def build_fast_plan(
    hazard: bool,
    func: Callable[..., Any],
    func_name: str,
    property_accessor: str | None,
    mutates_receiver: bool,
) -> CallPlan | None:
    """Return the fast-lane plan of a wrapped callable without a special wrapper role.

    Parameters
    ----------
    hazard:
        Whether the wrapper gives the callable a special role (no plan then).
    func:
        The original torch callable.
    func_name:
        Name the callable is wrapped under.
    property_accessor:
        Accessor kind when ``func`` belongs to a wrapped property, else ``None``.
    mutates_receiver:
        Whether the wrapper classifies the name as writing its first argument.

    Returns
    -------
    CallPlan | None
        A writing plan for a receiver write or a fused normalization call whose
        schemas allow it, else :func:`.build_call_plan`'s verdict.
    """

    if hazard or property_accessor is not None or type(func) not in _C_CALLABLE_TYPES:
        return None
    if mutates_receiver:
        names = inplace_schema_names()
        if names.get(func_name, False) or names.get(_inplace_schema_name(func_name), False):
            return CallPlan(True, writes=True)
        return None
    if func_name in _FUSED_MUTATOR_NAMES:
        if single_tensor_schema_names().get(func_name, False):
            return CallPlan(True, writes=True)
        return None
    return build_call_plan(func, func_name, property_accessor)


def _scan_values(values: Iterable[Any], tensors: list[torch.Tensor]) -> int:
    """Collect the tensors among flat call values and classify the values.

    Returns
    -------
    int
        ``_DECLINED`` when a value is neither plain, a tensor, a Parameter nor a
        flat list or tuple of these, or when a tensor carries foreign metadata
        or is a ``Tensor.data`` alias; otherwise a bit set: ``1`` when a
        non-Parameter tensor carries a buffer address, ``_LISTED`` when a list
        or tuple held a tensor.
    """

    stamped = 0
    for value in values:
        kind = type(value)
        if kind is _TENSOR or kind is _PARAMETER:
            meta = getattr(value, "_tl", None)
            if meta is not None:
                if type(meta) is not TensorMeta or meta.data_alias:
                    return _DECLINED
                if kind is _TENSOR and meta.address is not None:
                    stamped |= 1
            tensors += (value,)
        elif kind is tuple or kind is list:
            for item in value:
                item_kind = type(item)
                if item_kind is _TENSOR or item_kind is _PARAMETER:
                    stamped |= _scan_listed_tensor(item, item_kind, tensors)
                elif item_kind not in _PLAIN_TYPES:
                    return _DECLINED
            if stamped < 0:
                return _DECLINED
        elif kind not in _PLAIN_TYPES:
            return _DECLINED
    return stamped


def _scan_listed_tensor(value: torch.Tensor, kind: type, tensors: list[torch.Tensor]) -> int:
    """Collect one tensor member of a list or tuple argument (see :func:`_scan_values`).

    Returns ``_DECLINED`` for foreign metadata or a ``Tensor.data`` alias, else
    ``_LISTED``, with bit ``1`` set when the tensor carries a buffer address.
    """

    tensors += (value,)
    meta = getattr(value, "_tl", None)
    if meta is None:
        return _LISTED
    if type(meta) is not TensorMeta or meta.data_alias:
        return _DECLINED
    return _LISTED | (1 if kind is _TENSOR and meta.address is not None else 0)


def _buffer_work_needed(plan: CallPlan, trace: Any) -> bool:
    """Return whether today's buffer first-use and snapshot helpers must run.

    They are skipped only when the session holds no model-owned tensor table,
    no registered or pending buffer and no stamped argument, where both helpers
    resolve nothing. The one effect the snapshot helper has in that state, the
    storage metadata cache clear for a possibly mutating name, is done here.
    """

    build_data = trace._module_capture_ws.module_build_data
    if _MODEL_OWNED_KEY in build_data and build_data[_MODEL_OWNED_KEY]:
        return True
    try:
        tracker: BufferWriteTracker | None = trace._buffer_write_tracker
    except AttributeError:
        return False
    if tracker is None:
        return False
    if (
        tracker.address_to_tensor
        or tracker.storage_key_to_addresses
        or tracker._pending_lazy_buffer_ids
    ):
        return True
    # Clearing two empty caches is a no-op, so the clear runs only when one holds an entry.
    if plan.name_could_mutate and (tracker._storage_key_cache or tracker._storage_range_cache):
        tracker.clear_storage_metadata_cache()
    return False


def admit_call(
    plan: CallPlan, trace: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Admission | None:
    """Return the admitted pre-call state of one call, or ``None`` for the general body.

    Runs after the wrapper's region guards; every refusal precedes any side effect.

    Parameters
    ----------
    plan:
        The callable's decoration-time plan.
    trace:
        Active capture trace.
    args:
        Positional arguments of the call.
    kwargs:
        Keyword arguments of the call.

    Returns
    -------
    Admission | None
        ``(tensor arguments, buffer helpers must run, autocast state)``, or
        ``None`` when a session fact or a per-call guard refuses the call.
    """

    # Hot path (CC above 10 on purpose): one frame per admitted call, guards inline.
    if (
        not args
        or trace._options_mode != "exhaustive"
        or trace._options_structure_only
        or (kwargs and ("out" in kwargs or "inplace" in kwargs))
        or _state._active_hook_plan
        or _state.diagnostic_observer_armed()
    ):
        return None
    trace_dict = trace.__dict__
    if "_predicate_save_options" in trace_dict:
        options = trace_dict["_predicate_save_options"]
        if options is not None and options.intervene is not None:
            return None
    # A Parameter receiver matters only to a name that may write it (frozen-receiver context).
    if trace._wrapper_runtime_ws.materialized_lazy_slots is not None or (
        plan.name_could_mutate and type(args[0]) is _PARAMETER
    ):
        return None
    tensors: list[torch.Tensor] = []
    stamped = _scan_values(args, tensors)
    if stamped != _DECLINED and kwargs:
        kwarg_stamped = _scan_values(kwargs.values(), tensors)
        stamped = _DECLINED if kwarg_stamped == _DECLINED else stamped | kwarg_stamped
    if stamped == _DECLINED:
        return None
    autocast_state = log_current_autocast_state(trace)
    for device in autocast_state:
        if autocast_state[device]["enabled"]:
            return None
    buffer_work = (stamped & 1) == 1 or _buffer_work_needed(plan, trace)
    # A listed buffer needs the buffer helpers, which read today's argument order.
    return None if buffer_work and stamped & _LISTED else (tensors, buffer_work, autocast_state)


def call_admitted(
    func: Callable[..., Any],
    wrapped_func: Callable[..., Any],
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Run the original callable of an admitted call, with today's ``TypeError`` redispatch."""

    try:
        return func(*args, **kwargs)
    except TypeError as exc:
        out = redispatch_declined(exc, wrapped_func, func_name, args, kwargs)
        if out is STILL_DECLINED:
            raise
        return out


def output_admitted(out: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Return whether an admitted call's output may take the fast commit.

    The output must be a plain tensor, or a plain tuple or list of distinct
    plain tensors (``chunk``, ``unbind``), where no tensor is one of the call's
    arguments or a member of a list argument and none carries TorchLens
    metadata, hence a label. Anything else (a same-object return, a named
    tuple, a labelled or empty result) is handed to the wrapper's general
    post-call handling.
    """

    if type(out) is not _TENSOR:
        return _multi_output_admitted(out, args, kwargs)
    for value in args:
        if value is out:
            return False
    if kwargs:
        for value in kwargs.values():
            if value is out:
                return False
    return getattr(out, "_tl", None) is None


def _multi_output_admitted(out: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Return whether a tuple or list output may take the fast commit (see :func:`output_admitted`)."""

    kind = type(out)
    if not (kind is tuple or kind is list) or not out:
        return False
    if len({id(member) for member in out}) != len(out):
        return False
    input_ids: set[int] = set()
    for value in (*args, *kwargs.values()):
        input_ids.add(id(value))
        if type(value) is tuple or type(value) is list:
            input_ids.update(id(item) for item in value)
    for member in out:
        if type(member) is not _TENSOR or id(member) in input_ids:
            return False
        if getattr(member, "_tl", None) is not None:
            return False
    return True


def commit_fast_call(
    emit: Callable[..., Any],
    trace: Any,
    call: tuple[Any, ...],
    func_call_id: int,
    window: tuple[Any, list[torch.Tensor]],
    thin: Callable[[Any, tuple[Any, ...], int], bool] | None = None,
) -> Any:
    """Emit one admitted bottom-level call and return its output.

    The thin emit, when given, logs a plain single output in a plain session and
    answers True; every other call goes to today's emit.

    Parameters
    ----------
    emit:
        Today's output emit (``log_function_output_tensors``), passed in by the
        wrapper so this module stays outside the capture import cycle.
    trace:
        Active capture trace.
    call:
        ``(func, func_name, args, kwargs, arg_copies, kwarg_copies, out,
        exec_ctx)`` of the call.
    func_call_id:
        The call's id.
    window:
        ``(cow_state_ptrs, tensors)``: the trace's deferred payload window state
        (or ``None``), and the call's tensor arguments as :func:`admit_call`
        collected them, which the emit's argument scan reuses instead of scanning
        the arguments again.
    thin:
        The bound thin emit (``_thin_emit.make_thin_emit``), passed in by the
        wrapper for the same reason as ``emit``; ``None`` sends every call to ``emit``.

    Returns
    -------
    Any
        The call's output: a tensor, or a tuple or list of tensors.
    """

    cow_state_ptrs, tensors = window
    if cow_state_ptrs is not None:
        arm_deferred_payload_window(cow_state_ptrs)
    runtime = trace._wrapper_runtime_ws
    runtime.fast_scan = (call[2], tensors)
    try:
        if thin is None or not thin(trace, call, func_call_id):
            emit(trace, *call, True, func_call_id)
    finally:
        runtime.fast_scan = None
        if cow_state_ptrs is not None:
            disarm_deferred_payload_window()
    return call[6]

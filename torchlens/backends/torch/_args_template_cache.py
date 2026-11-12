"""Per-capture plans of the replay argument template, keyed on the call's argument layout.

``_build_args_template`` classifies every argument of every call: a tensor becomes a
parent reference or a literal tensor, a plain value a literal, and anything else is
walked by ``_classify_arg_component``. A call that repeats with the same function and
the same argument layout (the exact type of every positional value, the keyword names
and the exact type of every keyword value) takes the same branch for every top-level
value whose exact type alone decides it. The plan records that branch per slot and the
function's portable registry key fields, so a repeat reads each value once in its own
branch instead of walking the classifier.

Every per-value read the template carries (the tensor's label, its capture layout, the
literal value itself, a literal tensor's copy or fingerprint) is still made on every
call, and every object the template holds is new per call. The layout is computed per
call and is part of the plan key, so a call whose layout differs takes its own plan.
The function part of the key is the callable's identity, held so its id cannot be
reused; the registry key fields are reused only where ``_function_registry_key`` itself
memoizes them per capture (permanent C callables and the pure tensor properties).
Plans live in the capture's raw-graph workspace and go away with it.

A template is built in two steps: :func:`observe_template` makes every live read at the
call and returns the recorded facts, and ``_arg_template_resolve.resolve_arg_template``
builds the template objects from those facts and the plan alone, so it may run later.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..._state import pause_logging
from ..._utils._internal_read import internal_scalar_read
from ..._utils._tensor_label import get_live_tensor_label
from ..._utils._tensor_layout import CaptureLayout, layout_of
from ..._utils._torch_compat import get_tensorbase_class
from ._arg_template_resolve import GENERAL, LITERAL, TENSOR, resolve_arg_template
from ._ops_arg_slots import _peek_raw_graph_workspace

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = ("TemplateParts", "capture_layout", "observe_template", "planned_args_template")


@dataclass(frozen=True, slots=True)
class TemplateParts:
    """The argument module's classifier pieces and template types, handed to the plan.

    Passed in rather than imported: the argument module imports this one, and this
    module stays outside the package's import cycle.
    """

    classify: Callable[..., Any]
    capture_layout: Callable[[Any], Any]
    unlabeled_literal: Callable[[Any], Any]
    registry_key: Callable[..., Any]
    is_permanent_c_callable: Callable[[Any], bool]
    safe_property_names: frozenset[str]
    literal_value: type
    parent_ref: type
    template: type
    key_type: type


#: Exact types whose instances take no attributes (so carry no label) and are replay
#: literals: ``_classify_arg_component`` answers ``LiteralValue(value)`` for each.
_LITERAL_TYPES = frozenset({int, float, bool, str, bytes, type(None), slice, type(...)})

_MISSING = object()


def _slot_branch(value_type: type) -> int:
    """Return the plan branch for a top-level argument of exact type ``value_type``."""

    if value_type in _LITERAL_TYPES:
        return LITERAL
    if value_type is torch.Tensor:
        return TENSOR
    return GENERAL


def _new_plan(
    parts: TemplateParts,
    func: Callable[..., Any],
    func_name: str | None,
    trace: "Trace",
    call: tuple[tuple[Any, ...], dict[str, Any]],
) -> tuple[Any, ...]:
    """Return ``(func, registry key fields or None, arg branches, kwarg names, kwarg branches)``."""

    args, kwargs = call
    fields = None
    if func_name in parts.safe_property_names or parts.is_permanent_c_callable(func):
        key = parts.registry_key(func, func_name, trace)
        fields = (key.namespace, key.qualname, key.dispatch_kind, key.version, key.import_path)
    return (
        func,
        fields,
        tuple([_slot_branch(type(value)) for value in args]),
        tuple([str(name) for name in kwargs]),
        tuple([_slot_branch(type(value)) for value in kwargs.values()]),
    )


def _observed_facts(
    parts: TemplateParts,
    branches: tuple[int, ...],
    values: Any,
    notes: list[str],
    scope: tuple["Trace", dict[int, str | None] | None],
) -> tuple[Any, ...]:
    """Read each value's template facts along its planned branch, as the classifier would.

    ``scope`` is the active trace and the op's snapshot labels by tensor id. The facts
    are the resolver's input (``_arg_template_resolve``): every live read happens here.
    """

    trace, known = scope
    out: list[Any] = []
    append = out.append
    for branch, value in zip(branches, values, strict=True):
        if branch == LITERAL:
            append(value)
            continue
        if branch == GENERAL:
            append(parts.classify(value, notes, trace, known))
            continue
        # A plain tensor: the snapshot label when the snapshot read it, else a live read.
        label = _MISSING if known is None else known.get(id(value), _MISSING)
        if label is _MISSING:
            label = get_live_tensor_label(value, trace.capture_events.live_index.by_raw_label)
        if isinstance(label, str):
            append((label, parts.capture_layout(value)))
        else:
            append((None, parts.unlabeled_literal(value)))
    return tuple(out)


def _plan_of(
    parts: TemplateParts,
    call: tuple[Callable[..., Any], str | None, tuple[Any, ...], dict[str, Any]],
    trace: "Trace | None",
    holder: Any,
) -> tuple[Any, ...] | None:
    """Return the call's layout plan, making it on first sight; ``None`` with no workspace.

    A flat call's ``CallLayout`` (``holder``) keeps the plan: its guard is the plan's key.
    """

    func, func_name, args, kwargs = call
    if holder is not None and trace is not None:
        plan = holder.template_plan
        if plan is None:
            plan = holder.template_plan = _new_plan(parts, func, func_name, trace, (args, kwargs))
        return plan
    if (workspace := _peek_raw_graph_workspace(trace)) is None or trace is None:
        return None
    layout: tuple[Any, ...] = (
        (
            id(func),
            func_name,
            tuple(map(type, args)),
            tuple(kwargs),
            tuple(map(type, kwargs.values())),
        )
        if kwargs
        else (id(func), func_name, tuple(map(type, args)))
    )
    plans = workspace._args_template_plans
    plan = plans.get(layout)
    if plan is None or plan[0] is not func:
        plan = plans[layout] = _new_plan(parts, func, func_name, trace, (args, kwargs))
    return plan


def _planned_facts(
    parts: TemplateParts,
    plan: tuple[Any, ...],
    call: tuple[Callable[..., Any], str | None, tuple[Any, ...], dict[str, Any]],
    scope: tuple["Trace", dict[int, str | None] | None],
) -> tuple[Any, ...]:
    """Return the recorded template facts of a call along its layout plan."""

    func, func_name, args, kwargs = call
    _, fields, arg_branches, _, kwarg_branches = plan
    notes: list[str] = []
    arg_facts = _observed_facts(parts, arg_branches, args, notes, scope)
    kwarg_facts = (
        _observed_facts(parts, kwarg_branches, kwargs.values(), notes, scope) if kwargs else ()
    )
    return (
        plan,
        arg_facts,
        kwarg_facts,
        parts.registry_key(func, func_name, scope[0]) if fields is None else None,
        tuple(notes),
    )


def observe_template(
    parts: TemplateParts,
    call: tuple[Callable[..., Any], str | None, tuple[Any, ...], dict[str, Any]],
    trace: "Trace | None",
    known: dict[int, str | None] | None,
    holder: Any = None,
) -> tuple[Any, ...]:
    """Read at the call every fact the call's replay template carries.

    ``call`` is ``(func, func_name, args, kwargs)`` and ``known`` the op's snapshot
    labels by tensor id. Returns the recorded facts ``resolve_arg_template`` builds the
    template from (layout in ``_arg_template_resolve``); with no plan (no raw-graph
    workspace) every value is classified the general way, as ``_build_args_template``.
    """

    plan = _plan_of(parts, call, trace, holder)
    if plan is not None:
        return _planned_facts(parts, plan, call, (trace, known))  # type: ignore[arg-type]
    func, func_name, args, kwargs = call
    notes: list[str] = []
    arg_facts = tuple([parts.classify(arg, notes, trace, known) for arg in args])
    kwarg_facts = tuple(
        [(str(key), parts.classify(value, notes, trace, known)) for key, value in kwargs.items()]
    )
    return (None, arg_facts, kwarg_facts, parts.registry_key(func, func_name, trace), tuple(notes))


def planned_args_template(
    parts: TemplateParts,
    call: tuple[Callable[..., Any], str | None, tuple[Any, ...], dict[str, Any]],
    trace: "Trace | None",
    known: dict[int, str | None] | None,
    holder: Any = None,
) -> Any:
    """Return the call's replay template through its layout's plan, or ``None``.

    ``call`` is ``(func, func_name, args, kwargs)``. ``None`` (build it the general
    way) when the capture holds no raw-graph workspace. The result equals
    ``_build_args_template(func, args, kwargs, trace, func_name=...)``. A flat call's
    ``CallLayout`` (``holder``) keeps the plan: its guard is the plan's key.
    """

    plan = _plan_of(parts, call, trace, holder)
    if plan is None:
        return None
    return resolve_arg_template(parts, _planned_facts(parts, plan, call, (trace, known)))  # type: ignore[arg-type]


def capture_layout(value: Any) -> CaptureLayout | None:
    """Record a parent argument's non-contiguous layout for faithful replay.

    ``stride()`` / ``storage_offset()`` are witnessed metadata surfaces, so this
    bookkeeping read runs under ``pause_logging`` plus ``internal_scalar_read``
    (the sanctioned pattern for TorchLens's own reads during capture). The common
    answer, ``None`` for a plain strided tensor that is empty or row-major
    contiguous, is read first through the C base class's own accessors, which
    TorchLens never patches (its observers shadow them on ``torch.Tensor``), so
    no wrapper runs. Those originals are still reached with logging paused, so the
    escape detector treats them as TorchLens's own read; ``internal_scalar_read`` is
    entered only for a strided read, which goes through the shadowed surfaces.
    """

    if type(value) not in _PLAIN_TENSOR_TYPES:
        return None
    if _BASE_READS is not None:
        layout, numel, is_contiguous = _BASE_READS
        with pause_logging():
            if layout.__get__(value) != torch.strided:
                return None
            if numel(value) == 0 or is_contiguous(value):
                return None
    with pause_logging(), internal_scalar_read():
        return layout_of(value)


_PLAIN_TENSOR_TYPES = frozenset({torch.Tensor, torch.nn.Parameter})


def _base_reads() -> tuple[Any, Any, Any] | None:
    """Return the base class's ``(layout descriptor, numel, is_contiguous)``, if it has one."""

    base = get_tensorbase_class()
    return None if base is None else (base.layout, base.numel, base.is_contiguous)


#: The C tensor base class's ``layout`` descriptor, ``numel`` and ``is_contiguous``:
#: the originals ``layout_of`` reaches through the unwrap ledger, read unwrapped.
_BASE_READS = _base_reads()

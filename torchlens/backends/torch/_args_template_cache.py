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
from ._ops_arg_slots import _peek_raw_graph_workspace

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = ("TemplateParts", "capture_layout", "planned_args_template")


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


#: Slot branches of a plan.
_LITERAL = 0
_TENSOR = 1
_GENERAL = 2

#: Exact types whose instances take no attributes (so carry no label) and are replay
#: literals: ``_classify_arg_component`` answers ``LiteralValue(value)`` for each.
_LITERAL_TYPES = frozenset({int, float, bool, str, bytes, type(None), slice, type(...)})

_MISSING = object()


def _slot_branch(value_type: type) -> int:
    """Return the plan branch for a top-level argument of exact type ``value_type``."""

    if value_type in _LITERAL_TYPES:
        return _LITERAL
    if value_type is torch.Tensor:
        return _TENSOR
    return _GENERAL


def _new_plan(
    parts: TemplateParts,
    func: Callable[..., Any],
    func_name: str | None,
    trace: "Trace",
    call: tuple[tuple[Any, ...], dict[str, Any]],
) -> tuple[Any, ...]:
    """Return ``(func, registry key fields or None, arg branches, kwarg names and branches)``."""

    args, kwargs = call
    fields = None
    if func_name in parts.safe_property_names or parts.is_permanent_c_callable(func):
        key = parts.registry_key(func, func_name, trace)
        fields = (key.namespace, key.qualname, key.dispatch_kind, key.version, key.import_path)
    return (
        func,
        fields,
        tuple([_slot_branch(type(value)) for value in args]),
        tuple([(str(name), _slot_branch(type(value))) for name, value in kwargs.items()]),
    )


def _components(
    parts: TemplateParts,
    branches: tuple[int, ...],
    values: Any,
    notes: list[str],
    scope: tuple["Trace", dict[int, str | None] | None],
) -> tuple[Any, ...]:
    """Classify ``values`` along their planned branches, as ``_classify_arg_component`` does.

    ``scope`` is the active trace and the op's snapshot labels by tensor id.
    """

    trace, known = scope
    out: list[Any] = []
    for branch, value in zip(branches, values, strict=True):
        if branch == _LITERAL:
            out += (parts.literal_value(value),)
            continue
        if branch == _GENERAL:
            out += (parts.classify(value, notes, trace, known),)
            continue
        # A plain tensor: the snapshot label when the snapshot read it, else a live read.
        label = _MISSING if known is None else known.get(id(value), _MISSING)
        if label is _MISSING:
            label = get_live_tensor_label(value, trace.capture_events.live_index.by_raw_label)
        if isinstance(label, str):
            out += (parts.parent_ref(label, _capture_layout=parts.capture_layout(value)),)
        else:
            out += (parts.unlabeled_literal(value),)
    return tuple(out)


def planned_args_template(
    parts: TemplateParts,
    call: tuple[Callable[..., Any], str | None, tuple[Any, ...], dict[str, Any]],
    trace: "Trace | None",
    known: dict[int, str | None] | None,
) -> Any:
    """Return the call's replay template through its layout's plan, or ``None``.

    ``call`` is ``(func, func_name, args, kwargs)``. ``None`` (build it the general
    way) when the capture holds no raw-graph workspace. The result equals
    ``_build_args_template(func, args, kwargs, trace, func_name=...)``.
    """

    func, func_name, args, kwargs = call
    workspace = _peek_raw_graph_workspace(trace)
    if workspace is None or trace is None:
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
        plan = _new_plan(parts, func, func_name, trace, (args, kwargs))
        plans[layout] = plan
    _, fields, arg_branches, kwarg_plan = plan
    notes: list[str] = []
    scope = (trace, known)
    arg_components = _components(parts, arg_branches, args, notes, scope)
    kwarg_components: tuple[tuple[str, Any], ...] = ()
    if kwargs:
        names = tuple([name for name, _ in kwarg_plan])
        branches = tuple([branch for _, branch in kwarg_plan])
        values = _components(parts, branches, kwargs.values(), notes, scope)
        kwarg_components = tuple(zip(names, values, strict=True))
    return parts.template(
        args=arg_components,
        kwargs=kwarg_components,
        func_id=(
            parts.registry_key(func, func_name, trace)
            if fields is None
            else parts.key_type(*fields)
        ),
        notes=tuple(notes),
    )


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

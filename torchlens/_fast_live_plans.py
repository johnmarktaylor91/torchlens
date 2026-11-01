"""Collection plans and direct-call routes for the guarded fast live run.

The fast live session collects saved sites from a native forward: one plan per
saved module CALL (a module the forward calls twice, such as resnet's shared
``BasicBlock.relu``, gets one plan per saved call) and one per scoped functional
call. Each plan names the op labels it refreshes and the shapes, dtypes and
container paths the live output must match.

This module also routes a fast live run around hook-invisible direct
``module.forward(x)`` calls. The fast live session collects module sites, applies staged steering and folds
module-entry fingerprint tokens through torch module hooks, which fire only
inside ``Module.__call__``. A model that reaches a submodule by calling its
``forward`` directly (``return self.embedding.forward(indices)``) skips them,
while capture, whose ``forward`` decoration sees every call, records the module
call. Capture also marks each module it entered that way
(``ModuleMeta.direct_forward_called`` on the live module).

For exactly those modules, and only for the duration of one native forward, the
session installs an instance ``forward`` wrapper. A call through
``Module.__call__`` passes straight to the module's own ``forward`` (the hooks
already ran around it); a direct call runs the session's own hook work around
the forward in torch's hook order: the module-entry fingerprint token (the
token pre-hook), the forward, every staged steer hook for the module (the steer
forward hooks), then the site collection (the collection forward hook). User
hooks are skipped on a direct call, as torch and capture both skip them. Every
other module stays on hooks, and each run restores the prior ``forward`` by
identity.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from . import _state
from ._fast_live_steer import apply_steer_paused, session_is_active
from ._runnable import RunnableErrorCode
from .backends.torch.model_prep import shared_module_addresses
from .backends.torch.wrappers import _MUTATING_TENSOR_PROPERTY_SETTERS
from .errors import RunCapabilityUnavailableError

__all__ = [
    "DirectForwardRoute",
    "_FastOutputPlan",
    "_admitted_collection_labels",
    "_drop_uncollected_payloads",
    "_live_modules_by_address",
    "_module_call_budget",
    "_module_output_paths",
    "_output_tensor_leaves",
    "_pair_output_leaves",
    "_remove_fast_live_hooks",
    "build_function_plans",
    "build_module_plans",
    "collected_function_output",
    "install_collection_hooks",
    "recorded_module_call_counts",
    "repeats_last_function_output",
    "require_distinct_plan_modules",
    "save_payload_paused",
    "unobserved_hook_message",
]


@dataclass(frozen=True, slots=True)
class _FastOutputPlan:
    """Expected tensor leaves for one live module or functional call."""

    address_or_name: str
    op_labels: tuple[str, ...]
    shapes: tuple[tuple[int, ...] | None, ...]
    dtypes: tuple[str | None, ...]
    save_labels: frozenset[str]
    # Each op's container path inside the call's output, when capture tied
    # every op to exactly one path; ``None`` keeps the positional pairing.
    paths: tuple[tuple[Any, ...], ...] | None = None
    # The module call (1-based pass number) this plan collects; ``None`` for
    # functional plans. A module the forward calls more than once (resnet's
    # BasicBlock reuses one ``self.relu``) fires its hook on every call, and
    # only the planned calls are collected.
    call_index: int | None = None


def _pair_output_leaves(
    output: Any, captured_paths: Sequence[tuple[Any, ...]] | None, count: int
) -> tuple[list[Any] | None, str]:
    """Pair ``count`` captured output ops with the tensor leaves of a live output.

    The walk is capture's own output walker (plus its bare-attribute fallback),
    so a typed container capture descended into (a Hugging Face ``ModelOutput``
    carrying a ``DynamicCache`` of per-layer key/value tensors, a namedtuple, a
    dataclass) is seen the same way here. With per-op paths every captured path
    must resolve to exactly one leaf and the output must carry no other leaf;
    without them (a single leaf, an opaque root, a role-hinted family) the
    pairing is positional over the same walk. Returns the paired values, or
    ``None`` with the reason when the structure differs from the capture.
    """

    # Module import: the dotted spelling of this path matches the layer lint's
    # torch-private pattern, but the walker is TorchLens's own, not a torch private.
    from ._utils.introspection import get_vars_of_type_from_obj
    from .backends.torch import _ops_containers

    native = [
        (tuple(path), tensor)
        for tensor, path, _spec in _ops_containers._walk_output_tensors_with_paths(output)
    ]
    if not native:
        native = [
            ((), tensor)
            for tensor in get_vars_of_type_from_obj(output, torch.Tensor, search_depth=4)
        ]
    if len(native) != count:
        return None, f"{len(native)} tensor leaves against {count} captured output ops"
    by_path = dict(native)
    if (
        captured_paths is None
        or len(by_path) != len(native)
        or any(path == () for path in captured_paths)
    ):
        return [tensor for _path, tensor in native], ""
    missing = [path for path in captured_paths if path not in by_path]
    if missing:
        return None, f"unmatched captured output paths {missing}"
    return [by_path[path] for path in captured_paths], ""


def _module_output_paths(
    trace: Any, module_call: Any, address: str, resolved: Sequence[Any]
) -> tuple[tuple[Any, ...], ...] | None:
    """Return each output op's container path in a module call's output.

    ``ModuleCall.output_paths`` is in capture traversal order while
    ``output_ops`` is in op order; the documented tie between them is each
    op's ``multi_output_name``, minted from its path (or a role hint) at
    capture. ``None`` when the call has fewer than two leaves or any op lacks
    a unique path, which keeps the positional pairing.
    """

    from ._data_classes._module_role_hints import (
        multi_output_role_from_path,
        role_hints_for_module_class,
    )

    paths = tuple(tuple(path) for path in (getattr(module_call, "output_paths", None) or ()))
    if len(paths) < 2 or len(paths) != len(resolved):
        return None
    try:
        module = trace.modules[address]
    except (KeyError, IndexError, TypeError, AttributeError):
        module = None
    hints = role_hints_for_module_class(getattr(module, "cls", None))
    by_name: dict[str, tuple[Any, ...]] = {}
    for index, path in enumerate(paths):
        name = multi_output_role_from_path(path, index, hints=hints)
        if name is None or name in by_name:
            return None
        by_name[name] = path
    matched: list[tuple[Any, ...]] = []
    for item in resolved:
        name = getattr(item, "multi_output_name", None)
        if name not in by_name:
            return None
        matched.append(by_name[name])
    if len(set(matched)) != len(matched):
        return None
    return tuple(matched)


def _module_exit_index(trace: Any, module_call: Any, resolved: tuple[Any, ...]) -> int:
    """Return the raw index after which one module call exits.

    A module exits after the LAST op recorded inside its call, its children's
    included (``mc.ops``), not after its last OUTPUT op: a module that returns
    one early tensor of a child's later tuple (``out, _ = self.rnn(x)``) still
    returns after that child.
    """

    contained = (
        trace._layer_dict_all_keys[label]
        for label in getattr(module_call, "ops", ())
        if label in trace._layer_dict_all_keys
    )
    return max(int(getattr(item, "raw_index", 0) or 0) for item in (*contained, *resolved))


def build_module_plans(trace: Any) -> tuple[_FastOutputPlan, ...]:
    """Derive the module-call exit sequence whose outputs the capture saved.

    Every module call (leaf or container) whose output ops carry a saved
    activation gets a plan, including the ``interventionreplacement`` op a
    module-boundary intervention leaves as the module's output. Plans are
    ordered as torch fires forward hooks: by module EXIT, i.e. the raw
    index of the module's last output op, with the deeper module first
    when a nested call hands the same tensor to its parent. The root module is
    the input/output boundary and is refreshed separately.
    """

    keyed: list[tuple[int, int, _FastOutputPlan]] = []
    module_calls = trace.module_calls
    for call_label in list(module_calls.keys()):
        module_call = module_calls[call_label]
        address, _, call_suffix = call_label.rpartition(":")
        if not address:
            address, call_suffix = call_suffix, "1"
        if address == "self":
            continue
        resolved = tuple(
            trace._layer_dict_all_keys[label]
            for label in module_call.output_ops
            if label in trace._layer_dict_all_keys
        )
        if not resolved or not any(
            bool(getattr(item, "has_saved_activation", False)) for item in resolved
        ):
            continue
        labels = tuple(item.label for item in resolved)
        exit_index = _module_exit_index(trace, module_call, resolved)
        keyed.append(
            (
                exit_index,
                -address.count("."),
                _FastOutputPlan(
                    address_or_name=address,
                    op_labels=labels,
                    shapes=tuple(
                        tuple(item.shape) if item.shape is not None else None for item in resolved
                    ),
                    dtypes=tuple(
                        str(item.dtype) if item.dtype is not None else None for item in resolved
                    ),
                    save_labels=frozenset(
                        label
                        for label, item in zip(labels, resolved)
                        if bool(getattr(item, "has_saved_activation", False))
                    ),
                    paths=_module_output_paths(trace, module_call, address, resolved),
                    call_index=int(call_suffix) if call_suffix.isdigit() else 1,
                ),
            )
        )
    keyed.sort(key=lambda item: (item[0], item[1]))
    return tuple(plan for _, _, plan in keyed)


def build_function_plans(
    trace: Any, module_saved_labels: frozenset[str]
) -> tuple[_FastOutputPlan, ...]:
    """Derive scoped functional calls needed by saved non-module ops.

    Both save forms qualify. A functional ``save=`` predicate collects the
    functional ops it saved. The default save-all capture collects only the
    saved functional ops that no module output covers (``out += identity``
    inside a residual block, before the block's last op), so a module
    output such as an intervention replacement is never collected twice.
    Collection runs in capture order under the same per-call path, shape
    and dtype guards.

    Parameters
    ----------
    trace:
        Source trace.
    module_saved_labels:
        Saved op labels the module plans already collect.
    """

    explicit = getattr(trace, "_predicate_save_options", None) is not None
    target_names = {
        str(op.func_name)
        for op in trace._layer_list
        if bool(getattr(op, "has_saved_activation", False))
        and not bool(getattr(op, "is_atomic_module", False))
        and str(getattr(op, "func_name", "none")) not in {"none", "identity"}
        and (explicit or op.label not in module_saved_labels)
    }
    plans: list[_FastOutputPlan] = []
    ops = [op for op in trace._layer_list if str(getattr(op, "func_name", "")) in target_names]
    index = 0
    while index < len(ops):
        first = ops[index]
        group = [first]
        spec = getattr(first, "container_spec", None)
        if spec is not None:
            next_index = index + 1
            while (
                next_index < len(ops)
                and ops[next_index].raw_index == group[-1].raw_index + 1
                and getattr(ops[next_index], "container_spec", None) is spec
            ):
                group.append(ops[next_index])
                next_index += 1
        labels = tuple(item.label for item in group)
        plans.append(
            _FastOutputPlan(
                address_or_name=str(first.func_name),
                op_labels=labels,
                shapes=tuple(
                    tuple(item.shape) if item.shape is not None else None for item in group
                ),
                dtypes=tuple(str(item.dtype) if item.dtype is not None else None for item in group),
                save_labels=frozenset(
                    item.label
                    for item in group
                    if bool(getattr(item, "has_saved_activation", False))
                    and not bool(getattr(item, "is_atomic_module", False))
                    and (explicit or item.label not in module_saved_labels)
                ),
            )
        )
        index += len(group)
    return tuple(plans)


def recorded_module_call_counts(trace: Any, addresses: frozenset[str]) -> dict[str, int]:
    """Count the captured calls of each planned module address.

    Parameters
    ----------
    trace:
        Source trace whose ``module_calls`` keys spell ``address:call``.
    addresses:
        Module addresses the fast session collects from.

    Returns
    -------
    dict[str, int]
        Recorded call count per planned address.
    """

    counts: dict[str, int] = dict.fromkeys(addresses, 0)
    for call_label in list(trace.module_calls.keys()):
        address = str(call_label).rpartition(":")[0] or str(call_label)
        if address in counts:
            counts[address] += 1
    return counts


def _module_call_budget(
    trace: Any, module_plans: Sequence[_FastOutputPlan]
) -> tuple[dict[str, frozenset[int]], dict[str, int]]:
    """Return the planned calls per module address and each address's recorded call count.

    Parameters
    ----------
    trace:
        Source trace whose ``module_calls`` the counts come from.
    module_plans:
        Module output plans the fast session collects.

    Returns
    -------
    tuple[dict[str, frozenset[int]], dict[str, int]]
        The 1-based call numbers planned per address, and the number of calls
        the capture recorded per planned address.
    """

    planned: dict[str, frozenset[int]] = {}
    for plan in module_plans:
        planned[plan.address_or_name] = planned.get(plan.address_or_name, frozenset()) | {
            int(plan.call_index or 1)
        }
    return planned, recorded_module_call_counts(trace, frozenset(planned))


def _admitted_collection_labels(trace: Any, plans: Sequence[_FastOutputPlan]) -> set[str]:
    """Return the op labels the fast live session refreshes, refusing uncollectable saves.

    Parameters
    ----------
    trace:
        Source trace whose saved ops the plans must cover.
    plans:
        Module and function output plans the session collects.

    Returns
    -------
    set[str]
        Labels the plans collect plus the model's boundary ops.

    Raises
    ------
    RunCapabilityUnavailableError
        When a saved functional op is covered by no plan.
    """

    boundary_layer_labels = set(getattr(trace, "input_layers", ())) | set(
        getattr(trace, "output_layers", ())
    )
    supported_labels = {label for plan in plans for label in plan.save_labels} | {
        op.label for op in trace._layer_list if op.layer_label in boundary_layer_labels
    }
    unsupported_function_labels = tuple(
        op.label
        for op in trace._layer_list
        if bool(getattr(op, "has_saved_activation", False))
        and str(getattr(op, "func_name", "none")) not in {"none", "identity"}
        and op.label not in supported_labels
    )
    if unsupported_function_labels:
        raise RunCapabilityUnavailableError(
            "fast=True collects functional operations only when they were explicitly "
            "requested by the capture save predicate. Recapture with save=tl.func(...) "
            f"for: {', '.join(unsupported_function_labels[:8])}.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="fast_live_function_plan",
        )
    return supported_labels


def _drop_uncollected_payloads(trace: Any, supported_labels: set[str]) -> None:
    """Clear saved payloads the fast live session will not refresh.

    Parameters
    ----------
    trace:
        Result trace whose ops are updated (the fast door's fork, or the
        live trace itself on the legacy rerun door).
    supported_labels:
        Op labels the session's module and function plans collect, plus the
        model's boundary ops.
    """

    for op in trace._layer_list:
        if bool(getattr(op, "has_saved_activation", False)) and op.label not in supported_labels:
            op._internal_set("out", None)
            op._internal_set("transformed_out", None)
            op._internal_set("has_saved_activation", False)


def _live_modules_by_address(model: nn.Module) -> dict[str, nn.Module]:
    """Map every registered address of the live model, alias spellings included.

    Capture labels a module registered under several addresses (``self.B =
    self.A[0]``) with the address its own traversal reaches first, which is
    not always the one ``named_modules()`` keeps; the plan lookup accepts every
    registration so such a module is found, not refused as absent.
    """

    modules = dict(model.named_modules())
    by_id = {id(module): module for module in modules.values()}
    for module_id, addresses in shared_module_addresses(model).items():
        module = by_id.get(module_id)
        if module is None:
            continue
        for address in addresses:
            modules.setdefault(address, module)
    return modules


def _output_tensor_leaves(output: Any) -> tuple[Any, ...]:
    """Return the tensor leaves of one live call output (one container level)."""

    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, (tuple, list)):
        return tuple(item for item in output if isinstance(item, torch.Tensor))
    return ()


def repeats_last_function_output(
    last: tuple[str, Any, tuple[Any, ...]] | None, func_name: str, output: Any
) -> bool:
    """Return whether a same-name event hands back the last collected output.

    A wrapped torch function that calls another wrapped spelling of itself
    (``F.relu`` into ``torch.relu``; ``F.adaptive_max_pool1d`` into the
    ``_with_indices`` tuple it unpacks) fires one event per layer for one
    captured op, so the outer event is skipped when its output IS the inner
    output or one of its tensor leaves. Identity is checked on the live
    objects, which the session keeps alive: comparing ``id()`` values let a
    later same-name op whose fresh output reused a freed tensor's address
    be dropped, shifting every later plan by one (a false arm divergence).
    """

    if last is None or last[0] != func_name or output is None:
        return False
    _name, last_output, last_leaves = last
    if output is last_output:
        return True
    if isinstance(output, torch.Tensor):
        return any(output is leaf for leaf in last_leaves)
    leaves = _output_tensor_leaves(output)
    return bool(leaves) and all(any(leaf is kept for kept in last_leaves) for leaf in leaves)


def collected_function_output(
    func_name: str, output: Any, args: tuple[Any, ...], property_accessor: str | None
) -> Any:
    """Return the output a functional event is collected as.

    Capture records a None-returning mutator (``__setitem__``, ``__delitem__``,
    a mutating property setter) with its mutated receiver as the output; the
    collector sees the same.
    """

    if (
        output is None
        and (
            func_name in {"__setitem__", "zero_", "__delitem__"}
            or (property_accessor == "set" and func_name in _MUTATING_TENSOR_PROPERTY_SETTERS)
        )
        and len(args) > 0
        and isinstance(args[0], torch.Tensor)
    ):
        return args[0]
    return output


def save_payload_paused(session: Any, op: Any, value: torch.Tensor) -> None:
    """Save one collected payload with the session's event counting paused.

    The save's own ``safe_copy`` clone and transform calls run through the
    wrapped torch functions and must not count as the model's events.
    """

    session.collection_paused = True
    try:
        op.save_activation(
            value,
            (),
            {},
            False,
            activation_transform=session.trace._options_activation_transform,
        )
    finally:
        session.collection_paused = False


def unobserved_hook_message(plan: _FastOutputPlan, address: str) -> str:
    """Return the refusal text for a planned module call that fired no forward hook."""

    return (
        f"Captured module call {plan.address_or_name!r} call "
        f"{plan.call_index or 1} exited inside {address!r} without firing a "
        "forward hook: the model reaches it by a direct forward() call, which "
        "torch module hooks cannot observe, and the capture recorded no such "
        "call to route (no capture of this model made one, or "
        "HAS_MODULE_CALL_DISPATCH_FRAME is False), so the fast run "
        "cannot collect that site. The full run captures it."
    )


def require_distinct_plan_modules(
    modules: Mapping[str, nn.Module], plan_addresses: tuple[str, ...]
) -> None:
    """Refuse a plan address absent from the live model or sharing another's module.

    Raises
    ------
    RunCapabilityUnavailableError
        When a captured address resolves to no live module, or two captured
        addresses name one live module object.
    """

    planned_objects: dict[int, str] = {}
    for address in plan_addresses:
        module = modules.get("" if address == "self" else address)
        if module is None:
            raise RunCapabilityUnavailableError(
                f"Captured module address {address!r} is absent from the live model.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="fast_live_module_plan",
            )
        other = planned_objects.setdefault(id(module), address)
        if other != address:
            raise RunCapabilityUnavailableError(
                f"Captured module addresses {other!r} and {address!r} name one live module "
                "object; the fast run collects a shared module under one address only.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="fast_live_module_plan",
            )


def install_collection_hooks(
    handles: list[Any],
    modules: Mapping[str, nn.Module],
    plan_addresses: tuple[str, ...],
    session_ref: weakref.ReferenceType[Any],
) -> None:
    """Register one collection forward hook per planned address, appending to ``handles``."""

    for address in plan_addresses:
        module = modules["" if address == "self" else address]

        def hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            module_address: str = address,
            ref: weakref.ReferenceType[Any] = session_ref,
        ) -> None:
            """Forward one module boundary to the active session."""

            session = ref()
            if session is not None:
                session.capture_module(module_address, output)

        handles.append(module.register_forward_hook(hook))


@dataclass(frozen=True)
class DirectForwardRoute:
    """The session hook work one directly called module needs on a direct call."""

    module_ref: weakref.ReferenceType[nn.Module]
    token: int
    steer_addresses: tuple[str, ...]
    collect_address: str | None


def _direct_call(
    route: DirectForwardRoute,
    module: nn.Module,
    original: Callable[..., Any],
    session_ref: weakref.ReferenceType[Any],
) -> Callable[..., Any]:
    """Build the direct-call body: token, forward, steer hooks, site collection."""

    active = session_is_active(session_ref)

    def run_direct(*args: Any, **kwargs: Any) -> Any:
        """Do on a direct call what the session's hooks do around ``module(x)``."""

        session = session_ref()
        if session is None or not active():
            return original(*args, **kwargs)
        _state.note_fingerprint_token(route.token)
        output = original(*args, **kwargs)
        for address in route.steer_addresses:
            replaced = apply_steer_paused(session, address, module, args, output)
            if replaced is not None:
                output = replaced
        if route.collect_address is not None:
            session.capture_module(route.collect_address, output)
        return output

    return run_direct


def _remove_fast_live_hooks(handles: list[Any]) -> None:
    """Remove and discard module-hook handles without retaining their session.

    Every handle gets its own removal attempt: ``weakref.finalize`` pops its
    registry entry BEFORE invoking the callback, so this is the one chance to
    remove these hooks -- a single raising ``remove()`` must never strand the
    remaining handles on the user's modules forever. The first failure
    re-raises only after every handle was attempted and the list cleared.
    """

    first_error: BaseException | None = None
    for handle in handles:
        try:
            handle.remove()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    handles.clear()
    if first_error is not None:
        raise first_error

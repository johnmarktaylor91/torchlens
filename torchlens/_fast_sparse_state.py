"""Cached-state safety analysis for the guarded fast sparse engine.

The fast sparse session (``_fast_run._FastSparseSession``) caches staged
parameter and buffer state across iterations. That cache is safe only when no
recorded call can write it: an in-place call whose target aliases staged state,
or a functional normalization call that may update running statistics. This
module finds those calls from the frozen sparse descriptor alone, and holds the
compiled form of one recorded call that the session replays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch

from ._runnable import (
    RunnableCallDescriptor,
    RunnableErrorCode,
    SparseRunDescriptor,
    is_mode_sensitive_qualname,
)
from ._runnable.call_arguments import _write_argument
from ._runnable.call_outputs import _mutation_target_slot_id
from ._runnable.execution import _VIEW_OP_QUALNAMES
from ._runnable.state_context import _call_execution_context_entered
from ._runnable.verification import _decode_literal
from .errors import RunCapabilityUnavailableError, RuntimeSignatureDriftError


def _state_storage_closure(
    descriptor: SparseRunDescriptor,
    state_slot_ids: set[str],
) -> set[str]:
    """Return state slots plus every slot reachable through alias-producing calls.

    Parameters
    ----------
    descriptor:
        Frozen sparse replay descriptor.
    state_slot_ids:
        Parameter and buffer slot identifiers that seed sparse execution.

    Returns
    -------
    set[str]
        State-derived slots whose storage may alias cached staged state.
    """

    registry_qualnames = {
        entry.registry_id: entry.key.qualname for entry in descriptor.callable_registry
    }
    closure = set(state_slot_ids)
    changed = True
    while changed:
        changed = False
        for slot in descriptor.tensor_slots:
            if slot.version_of in closure and slot.slot_id not in closure:
                closure.add(slot.slot_id)
                changed = True
        for call in descriptor.calls:
            if registry_qualnames.get(call.registry_id) not in _VIEW_OP_QUALNAMES:
                continue
            if not any(argument.slot_id in closure for argument in call.tensor_arguments):
                continue
            for output_slot_id in call.output_slot_ids:
                if output_slot_id not in closure:
                    closure.add(output_slot_id)
                    changed = True
    return closure


def _literal_bool_argument(call: RunnableCallDescriptor, name: str) -> bool | None:
    """Return a recorded boolean call argument by keyword or positional name.

    Parameters
    ----------
    call:
        Frozen sparse call recipe.
    name:
        Callable parameter name to resolve.

    Returns
    -------
    bool | None
        Recorded boolean value, or ``None`` when absent or non-boolean.
    """

    try:
        positional_index = call.argument_names.index(name)
    except ValueError:
        positional_index = None
    for literal in call.literal_arguments:
        path = tuple(literal.argument_path)
        is_keyword = len(path) == 2 and path == ("kwargs", name)
        is_positional = (
            positional_index is not None and len(path) == 2 and path == ("args", positional_index)
        )
        if not (is_keyword or is_positional):
            continue
        value = _decode_literal(literal.value)
        return value if isinstance(value, bool) else None
    return None


def _normalization_call_may_mutate_state(
    call: RunnableCallDescriptor,
    qualname: str | None,
    state_slot_ids: set[str],
) -> bool:
    """Return whether a functional normalization call may update running state.

    Parameters
    ----------
    call:
        Frozen sparse call recipe.
    qualname:
        Resolved callable qualname from the registry.
    state_slot_ids:
        Parameter and buffer slot identifiers that seed sparse execution.

    Returns
    -------
    bool
        ``True`` when cached running statistics could be updated as a hidden
        side effect. Unknown mode flags fail closed.
    """

    if not is_mode_sensitive_qualname(qualname):
        return False
    if not any(argument.slot_id in state_slot_ids for argument in call.tensor_arguments):
        return False
    tail = (qualname or "").rsplit(".", 1)[-1].removesuffix("_")
    mode_argument = "use_input_stats" if tail.endswith("instance_norm") else "training"
    return _literal_bool_argument(call, mode_argument) is not False


def _state_mutating_call_ids(
    descriptor: SparseRunDescriptor,
    state_slot_ids: set[str],
) -> tuple[str, ...]:
    """Return calls that make cached sparse state unsafe across iterations.

    Parameters
    ----------
    descriptor:
        Frozen sparse replay descriptor.
    state_slot_ids:
        Parameter and buffer slot identifiers that seed sparse execution.

    Returns
    -------
    tuple[str, ...]
        Ordered call identifiers that explicitly or implicitly mutate state.
    """

    registry_qualnames = {
        entry.registry_id: entry.key.qualname for entry in descriptor.callable_registry
    }
    state_storage_ids = _state_storage_closure(descriptor, state_slot_ids)
    return tuple(
        call.call_id
        for call in descriptor.calls
        if (call.is_inplace and _mutation_target_slot_id(call) in state_storage_ids)
        or _normalization_call_may_mutate_state(
            call,
            registry_qualnames.get(call.registry_id),
            state_storage_ids,
        )
    )


@dataclass(frozen=True, slots=True)
class _CompiledSparseCall:
    """One sparse call with decoded literals and pre-indexed tensor bindings."""

    descriptor: RunnableCallDescriptor
    func: Callable[..., Any]
    base_args: tuple[Any, ...]
    base_kwargs: Mapping[str, Any]
    top_level_tensor_args: tuple[tuple[str, int | str, str], ...]
    nested_tensor_args: tuple[tuple[tuple[str | int, ...], str], ...]
    requires_context: bool

    def execute(self, slot_values: Mapping[str, torch.Tensor]) -> Any:
        """Bind current tensor slots and invoke the resolved callable.

        Parameters
        ----------
        slot_values:
            Runtime tensor slot store.

        Returns
        -------
        Any
            Raw callable output.
        """

        args = list(self.base_args)
        kwargs = dict(self.base_kwargs)
        for root, key, slot_id in self.top_level_tensor_args:
            try:
                value = slot_values[slot_id]
            except KeyError as exc:
                raise RuntimeSignatureDriftError(
                    f"Fast sparse call {self.descriptor.call_id!r} references unavailable "
                    f"slot {slot_id!r}.",
                    code=RunnableErrorCode.MISSING_TENSOR_SLOT.value,
                    call_id=self.descriptor.call_id,
                    slot_id=slot_id,
                ) from exc
            if root == "args":
                args[cast(int, key)] = value
            else:
                kwargs[cast(str, key)] = value
        for path, slot_id in self.nested_tensor_args:
            try:
                value = slot_values[slot_id]
            except KeyError as exc:
                raise RuntimeSignatureDriftError(
                    f"Fast sparse call {self.descriptor.call_id!r} references unavailable "
                    f"slot {slot_id!r}.",
                    code=RunnableErrorCode.MISSING_TENSOR_SLOT.value,
                    call_id=self.descriptor.call_id,
                    slot_id=slot_id,
                ) from exc
            _write_argument(args, kwargs, path, value)
        try:
            if not self.requires_context:
                return self.func(*args, **kwargs)
            with _call_execution_context_entered(self.descriptor.execution_context):
                return self.func(*args, **kwargs)
        except RunCapabilityUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeSignatureDriftError(
                f"Resolved callable rejected compiled fast recipe for "
                f"{self.descriptor.call_id!r}: {exc}",
                code=RunnableErrorCode.RUNTIME_SIGNATURE_DRIFT.value,
                call_id=self.descriptor.call_id,
                affected_op_labels=self.descriptor.op_labels,
            ) from exc


def _compile_sparse_call(
    call: RunnableCallDescriptor,
    func: Callable[..., Any],
    *,
    ambient_grad_enabled: bool,
    ambient_inference_mode: bool,
) -> _CompiledSparseCall:
    """Compile a sparse call's invariant literal tree and tensor assignments.

    Parameters
    ----------
    call:
        Frozen call recipe.
    func:
        Resolved callable.
    ambient_grad_enabled:
        Grad mode already restored around the compiled loop.
    ambient_inference_mode:
        Inference mode already restored around the compiled loop.

    Returns
    -------
    _CompiledSparseCall
        Reusable per-call binder.
    """

    args: list[Any] = [None] * call.num_positional_args
    kwargs: dict[str, Any] = {}
    for literal in call.literal_arguments:
        _write_argument(args, kwargs, literal.argument_path, _decode_literal(literal.value))
    top_level: list[tuple[str, int | str, str]] = []
    nested: list[tuple[tuple[str | int, ...], str]] = []
    for argument in call.tensor_arguments:
        path = tuple(argument.argument_path)
        if len(path) == 2 and path[0] in {"args", "kwargs"}:
            top_level.append((cast(str, path[0]), path[1], argument.slot_id))
        else:
            nested.append((path, argument.slot_id))
    return _CompiledSparseCall(
        descriptor=call,
        func=func,
        base_args=tuple(args),
        base_kwargs=kwargs,
        top_level_tensor_args=tuple(top_level),
        nested_tensor_args=tuple(nested),
        requires_context=(
            call.execution_context.grad_enabled != ambient_grad_enabled
            or call.execution_context.inference_mode != ambient_inference_mode
            or any(entry.enabled for entry in call.execution_context.autocast)
        ),
    )

"""Pre-hook input state: metadata observation, transition comparison and tensor snapshots.

Split out of ``prehook_provenance`` (which installs the interposition and keeps
the ledger) so each file stays under the size cap; nothing here touches the
ledger or the wrappers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ..._data_classes.prehook import (
    ModuleInputSnapshot,
    TensorInputObservation,
    TypedInputPath,
)
from ._tl import get_tensor_label

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

_INCOMPLETE_VERSION = "tensor_version_unavailable"


_INCOMPLETE_OPAQUE_OBJECT = "opaque_object_untraversed"


_IMMUTABLE_LEAF_TYPES = (str, bytes, int, float, bool, complex, type(None))


@dataclass(frozen=True)
class _TensorState:
    """Cheap live tensor observation used for transition comparison."""

    path: TypedInputPath
    value: torch.Tensor
    object_id: int
    version: int | None
    shape: tuple[int, ...]
    dtype: str
    device: str
    layout: str
    requires_grad: bool
    source_op: str | None


@dataclass(frozen=True)
class _StateObservation:
    """Metadata-only observation of one logical args/kwargs state."""

    structure: tuple[Any, ...]
    tensors: tuple[_TensorState, ...]
    leaves: tuple[tuple[TypedInputPath, str, Any], ...]
    incomplete_reasons: tuple[str, ...]


def _observe_state(args: tuple[Any, ...], kwargs: dict[str, Any]) -> _StateObservation:
    """Capture structure, identity, version, and metadata without tensor copies."""

    acc = _WalkAccumulator()
    _walk_value(args, ("args",), acc)
    _walk_value(kwargs, ("kwargs",), acc)
    return _StateObservation(
        structure=tuple(acc.structure),
        tensors=tuple(acc.tensors),
        leaves=tuple(acc.leaves),
        incomplete_reasons=tuple(sorted(acc.reasons)),
    )


@dataclass
class _WalkAccumulator:
    """Mutable accumulator for one deterministic typed-path input walk."""

    structure: list[Any] = field(default_factory=list)
    tensors: list[_TensorState] = field(default_factory=list)
    leaves: list[tuple[TypedInputPath, str, Any]] = field(default_factory=list)
    reasons: set[str] = field(default_factory=set)


def _walk_value(
    value: Any,
    path: TypedInputPath,
    acc: _WalkAccumulator,
) -> None:
    """Walk supported input containers in deterministic typed-path order."""

    structure, tensors, leaves, reasons = acc.structure, acc.tensors, acc.leaves, acc.reasons

    if isinstance(value, torch.Tensor):
        try:
            version = int(value._version)
        except (AttributeError, RuntimeError):
            version = None
            reasons.add(_INCOMPLETE_VERSION)
        structure.append((path, "tensor"))
        tensors.append(
            _TensorState(
                path=path,
                value=value,
                object_id=id(value),
                version=version,
                shape=tuple(value.shape),
                dtype=str(value.dtype),
                device=str(value.device),
                layout=str(value.layout),
                requires_grad=bool(value.requires_grad),
                source_op=get_tensor_label(value),
            )
        )
        return
    if isinstance(value, tuple):
        structure.append((path, "tuple", len(value)))
        for index, item in enumerate(value):
            _walk_value(item, (*path, index), acc)
        return
    if isinstance(value, list):
        structure.append((path, "list", len(value)))
        for index, item in enumerate(value):
            _walk_value(item, (*path, index), acc)
        return
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        structure.append((path, "mapping", tuple((type(key).__name__, repr(key)) for key in keys)))
        for key in keys:
            _walk_value(value[key], (*path, ("key", key)), acc)
        return
    if isinstance(value, _IMMUTABLE_LEAF_TYPES):
        structure.append((path, "immutable", type(value).__name__))
        leaves.append((path, "immutable", (type(value).__name__, value)))
        return
    reasons.add(_INCOMPLETE_OPAQUE_OBJECT)
    structure.append((path, "object", type(value).__qualname__))
    leaves.append((path, "object", (type(value).__qualname__, id(value))))


def _compare_observations(
    before: _StateObservation,
    after: _StateObservation,
) -> tuple[bool | None, tuple[str, ...], tuple[TypedInputPath, ...], tuple[str, ...]]:
    """Compare two observations using the decided tri-state detection semantics."""

    kinds: set[str] = set()
    paths: set[TypedInputPath] = set()
    if before.structure != after.structure:
        kinds.add("structure")
        before_paths = {entry[0] for entry in before.structure}
        after_paths = {entry[0] for entry in after.structure}
        paths.update(before_paths ^ after_paths)
    before_tensors = {state.path: state for state in before.tensors}
    after_tensors = {state.path: state for state in after.tensors}
    for path in before_tensors.keys() | after_tensors.keys():
        left = before_tensors.get(path)
        right = after_tensors.get(path)
        if left is None or right is None:
            kinds.add("structure")
            paths.add(path)
            continue
        if left.object_id != right.object_id:
            kinds.add("identity")
            paths.add(path)
            continue
        if left.version is not None and right.version is not None and left.version != right.version:
            kinds.add("in_place_tensor")
            paths.add(path)
        left_metadata = (left.shape, left.dtype, left.device, left.layout, left.requires_grad)
        right_metadata = (right.shape, right.dtype, right.device, right.layout, right.requires_grad)
        if left_metadata != right_metadata:
            kinds.add("tensor_metadata")
            paths.add(path)
    before_leaves = {path: value for path, _kind, value in before.leaves}
    after_leaves = {path: value for path, _kind, value in after.leaves}
    for path in before_leaves.keys() | after_leaves.keys():
        if before_leaves.get(path) != after_leaves.get(path):
            kinds.add("value_or_identity")
            paths.add(path)
    reasons = tuple(sorted(set(before.incomplete_reasons) | set(after.incomplete_reasons)))
    changed: bool | None = True if kinds else (None if reasons else False)
    return changed, tuple(sorted(kinds)), tuple(sorted(paths, key=repr)), reasons


def _fingerprint(observation: _StateObservation) -> tuple[Any, ...]:
    """Return a compact metadata-only fingerprint for one observation."""

    tensor_bits = tuple(
        (
            state.path,
            state.object_id,
            state.version,
            state.shape,
            state.dtype,
            state.device,
            state.layout,
            state.requires_grad,
        )
        for state in observation.tensors
    )
    return observation.structure, tensor_bits, observation.leaves


def _clone_tree(value: Any, clone_tensor: Callable[[Any], Any]) -> Any:
    """Copy ``value`` through ``clone_tensor``, preserving tuple, list and dict nesting.

    A module-level walker rather than a self-recursive closure: a closure that
    calls itself holds a cell pointing back at the function, a reference cycle
    that kept every captured name (the trace among them) alive until the cyclic
    collector ran.
    """

    if isinstance(value, torch.Tensor):
        return clone_tensor(value)
    if isinstance(value, tuple):
        return tuple(_clone_tree(item, clone_tensor) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item, clone_tensor) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tree(item, clone_tensor) for key, item in value.items()}
    return value


def _tensor_observations(
    observation: _StateObservation,
    tensor_copies: Mapping[int, Any],
    payload_origins: Mapping[int, str],
    copy_failures: set[int],
) -> tuple[tuple[TensorInputObservation, ...], set[str]]:
    """Return the snapshot's per-tensor observations and its incompleteness reasons."""

    alias_groups: dict[int, int] = {}
    tensor_observations: list[TensorInputObservation] = []
    reasons = set(observation.incomplete_reasons)
    for state in observation.tensors:
        alias_group = alias_groups.setdefault(state.object_id, len(alias_groups) + 1)
        payload = tensor_copies.get(state.object_id)
        if state.object_id in copy_failures:
            reasons.add("tensor_payload_copy_failed")
        tensor_observations.append(
            TensorInputObservation(
                path=state.path,
                source_op=state.source_op,
                python_object_id=state.object_id,
                mutation_version=state.version,
                shape=state.shape,
                dtype=state.dtype,
                device=state.device,
                layout=state.layout,
                requires_grad=state.requires_grad,
                payload=payload,
                payload_status="saved" if payload is not None else "unsaved",
                alias_group=alias_group,
                payload_origin=payload_origins.get(state.object_id, "dedicated_pre_hook_snapshot"),
            )
        )
    return tuple(tensor_observations), reasons


def _snapshot_state(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    observation: _StateObservation,
    *,
    trace: Trace | None = None,
    reuse_source_payload: bool = False,
) -> ModuleInputSnapshot:
    """Create one truthful detached snapshot, deduplicating tensor aliases."""

    tensor_copies: dict[int, Any] = {}
    payload_origins: dict[int, str] = {}
    copy_failures: set[int] = set()
    states_by_id = {state.object_id: state for state in observation.tensors}

    def clone_tensor(value: Any) -> Any:
        """Clone one tensor, reusing an alias copy or the producer's snapshot."""

        tensor_id = id(value)
        if tensor_id in tensor_copies:
            return tensor_copies[tensor_id]
        state = states_by_id[tensor_id]
        if reuse_source_payload and trace is not None and state.source_op is not None:
            event = trace.capture_events.op_event_by_label_raw.get(state.source_op)
            payload = event.output.tensor.payload if event is not None else None
            if isinstance(payload, torch.Tensor) and payload is not value:
                tensor_copies[tensor_id] = payload
                payload_origins[tensor_id] = "immutable_producer_snapshot"
                return payload
        try:
            copied = value.detach().clone()  # detach-ok: provenance snapshot copy
        except Exception:
            copied = None
            copy_failures.add(tensor_id)
        tensor_copies[tensor_id] = copied
        payload_origins[tensor_id] = "dedicated_pre_hook_snapshot"
        return copied

    cloned_args = _clone_tree(args, clone_tensor)
    cloned_kwargs = _clone_tree(kwargs, clone_tensor)
    tensor_observations, reasons = _tensor_observations(
        observation, tensor_copies, payload_origins, copy_failures
    )
    return ModuleInputSnapshot(
        _args=cloned_args,
        _kwargs=cloned_kwargs,
        structure=observation.structure,
        tensor_observations=tensor_observations,
        is_capture_complete=not reasons,
        incomplete_reasons=tuple(sorted(reasons)),
        summary=f"args={len(args)}, kwargs={len(kwargs)}, tensors={len(observation.tensors)}",
    )

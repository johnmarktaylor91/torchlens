"""Gradient influence probes for traces without a saved PyTorch autograd graph.

The PyTorch probes in :mod:`._gradient` and :mod:`._gradient_forward` walk the
autograd graph that ``backward_ready`` capture keeps alive. The other backends
keep no such graph, so this provider computes the same per-unit gradients a
different way: it replays the captured ops between the probed endpoints (the
cone) from their saved parent payloads, through each backend validator's own
per-op replay, and differentiates that replay with the framework's own
autodiff (:mod:`._unit_vjp`).

Before differentiating, the cone is replayed once at the captured endpoint
values and every replayed op with a saved payload must equal it under the
validator's comparison, so the gradient always describes the captured
computation. Source selection, unit normalization, reachability descriptors,
thresholds, batch semantics and the result fields are the ones the PyTorch
probes use: this module reuses their helpers rather than restating them. A
probe touches no backward-capture state (the same tripwire as the PyTorch
probes guards it) and restores the framework's global random state.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import numpy as np
import torch

from . import _engine
from ._engine_forward import solve_projective
from ._errors import ReceptiveFieldUnavailableError
from ._gradient import (
    _build_result,
    _builtin_rules_when_registry_empty,
    _normalize_unit,
    _probe_suppressed,
    _select_inputs,
)
from ._gradient_forward import _build_projective_result, _select_targets, _target_key
from ._path import between_labels, require_path, resolve_graph_point
from ._types import GradReceptiveField, ReceptiveFieldDirection
from ._unit_vjp import ReplayRefusal, backend_module

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from ._unit_vjp import UnitReplayer

__all__: list[str] = []

#: Largest target site a projective probe answers by one reverse probe per element
#: on a framework without forward-mode autodiff.
_ROW_SWEEP_LIMIT = 4096


@dataclass(frozen=True)
class _Cone:
    """The ops a probe replays: endpoints plus the executed sub-DAG between them."""

    framework: ModuleType
    replayer: UnitReplayer
    sources: tuple[Op, ...]
    targets: tuple[Op, ...]
    order: tuple[Op, ...]


_P = ParamSpec("_P")
_R = TypeVar("_R")

#: ``fields["code"]`` of every refusal this provider raises (PROPOSED NAME).
_REFUSAL_CODE = "gradient_replay_unavailable"


def _refusing(probe: Callable[_P, _R]) -> Callable[_P, _R]:
    """Convert the cone replay's internal refusal into the public typed refusal.

    Every reason a probe cannot run here (no cone-replay module or runtime
    records, an op that cannot be replayed or does not reproduce the capture, a
    non-floating endpoint, no reachable input, a row sweep over its limit) is one
    condition for a caller: this trace cannot answer a gradient probe. It surfaces
    as one coded :class:`ReceptiveFieldUnavailableError` with the specific message.
    """

    @functools.wraps(probe)
    def refusing(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return probe(*args, **kwargs)
        except ReplayRefusal as exc:
            raise ReceptiveFieldUnavailableError(str(exc), code=_REFUSAL_CODE) from exc

    return refusing


def _framework(trace: Trace) -> ModuleType:
    """Return the trace backend's cone-replay module (refusing an unknown backend)."""

    return backend_module(str(trace._provenance_backend))


def _alias_labels(op: Op) -> tuple[str, ...]:
    """Every label a child's parent reference may use for ``op``."""

    labels = [getattr(op, "_label_raw", None), getattr(op, "label", None)]
    if getattr(op, "num_passes", 1) == 1:
        labels.append(getattr(op, "layer_label", None))
    return tuple(label for label in labels if isinstance(label, str))


def _bind(env: dict[str, Any], op: Op, value: Any) -> None:
    """Record one op's value under each of its parent-reference labels."""

    for label in _alias_labels(op):
        env[label] = value


def _canonical(trace: Trace, op: Op) -> Op:
    """The trace-owned op behind a handle.

    Trace accessors return lightweight handles that need not carry every
    parent-reference label (the raw label children record), so the cone binds
    and reads the canonical op from ``trace._layer_list`` instead.
    """

    return next((item for item in trace._layer_list if item.label == op.label), op)


def _build_cone(trace: Trace, sources: Sequence[Op], targets: Sequence[Op]) -> _Cone:
    """Collect the executed sub-DAG between any source and any target, in trace order."""

    framework = _framework(trace)
    sources = tuple(_canonical(trace, op) for op in sources)
    targets = tuple(_canonical(trace, op) for op in targets)
    labels: set[str] = set()
    for source in sources:
        for target in targets:
            labels |= between_labels(trace, source, target)
    source_labels = {source.label for source in sources}
    order = tuple(
        op for op in trace._layer_list if op.label in labels and op.label not in source_labels
    )
    return _Cone(framework, framework.replayer(trace), tuple(sources), tuple(targets), order)


def _run_cone(cone: _Cone, source_values: Sequence[Any]) -> tuple[Any, ...]:
    """Replay the cone from the given source values; return the target values."""

    env: dict[str, Any] = {}
    for source, value in zip(cone.sources, source_values, strict=True):
        _bind(env, source, value)
    for op in cone.order:
        _bind(env, op, cone.replayer.replay(op, env))
    return tuple(env[target.label] for target in cone.targets)


def _verify_cone(cone: _Cone, source_values: Sequence[Any]) -> None:
    """Refuse unless replaying the cone at the captured values reproduces the capture."""

    env: dict[str, Any] = {}
    for source, value in zip(cone.sources, source_values, strict=True):
        _bind(env, source, value)
    for op in cone.order:
        replayed = cone.replayer.replay(op, env)
        try:
            saved = cone.replayer.saved(op)
        except ReplayRefusal:
            saved = None
        if saved is not None and not cone.replayer.close(replayed, saved):
            raise ReplayRefusal(
                f"Replaying op {op.label!r} from its saved parents does not reproduce its "
                "captured value, so a gradient through it would describe a different "
                "computation."
            )
        _bind(env, op, replayed)


def _floating_payload(cone: _Cone, op: Op, purpose: str) -> tuple[Any, np.ndarray]:
    """Return an endpoint's native payload and its host copy, refusing non-float dtypes."""

    native = cone.replayer.saved(op)
    host = np.asarray(cone.framework.to_numpy(native))
    if not (
        np.issubdtype(host.dtype, np.floating) or np.issubdtype(host.dtype, np.complexfloating)
    ):
        raise ReplayRefusal(
            f"{purpose.capitalize()} op {op.label!r} has non-differentiable dtype {host.dtype}."
        )
    return native, host


def _one_hot(cone: _Cone, like: Any, host: np.ndarray, unit: tuple[int, ...]) -> Any:
    """A native array shaped and typed like ``like`` with a single one at ``unit``."""

    seed = np.zeros(host.shape, dtype=host.dtype)
    seed[unit] = 1
    return cone.framework.from_numpy(seed, like)


def _to_torch(cone: _Cone, value: Any) -> torch.Tensor:
    """Convert a native gradient to the CPU torch tensor the result fields hold."""

    return torch.from_numpy(np.array(cone.framework.to_numpy(value), order="C", copy=True))


def _differentiate(
    trace: Trace,
    cone: _Cone,
    run: Callable[[], Any],
    verify_at: Sequence[Any] | None = None,
) -> Any:
    """Run one differentiation under the probe tripwire and the framework state guard.

    With ``verify_at``, the cone is first checked against the capture at those
    source values, inside the same guard (replay may touch framework state).
    """

    with _probe_suppressed(trace), cone.framework.state_guard():
        if verify_at is not None:
            _verify_cone(cone, verify_at)
        return run()


def _receptive_grads(trace: Trace, cone: _Cone, unit: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    """Gradients of one target unit with respect to every cone source."""

    target_native, target_host = _floating_payload(cone, cone.targets[0], "target")
    primals = tuple(_floating_payload(cone, source, "source")[0] for source in cone.sources)
    cotangent = _one_hot(cone, target_native, target_host, unit)

    def fn(*values: Any) -> Any:
        return _run_cone(cone, values)[0]

    grads = _differentiate(
        trace, cone, lambda: cone.framework.vjp(fn, primals, cotangent), verify_at=primals
    )
    return tuple(_to_torch(cone, grad) for grad in grads)


def _descriptor_for_source(
    trace: Trace, source_op: Op, target: Op, source_key: str
) -> tuple[Any, Any]:
    """The reachable descriptor and engine state the PyTorch source probe reads."""

    with _builtin_rules_when_registry_empty():
        solution = _engine.solve_from(trace, source_op)
        descriptors = solution.per_op.get(target.label, {})
    descriptor = descriptors.get(source_key) or next(
        (item for item in descriptors.values() if item.input_op_label == source_op.label), None
    )
    return descriptor, solution.states.get((target.label, source_key))


@_refusing
def receptive_for_unit(
    target: Op,
    unit: Sequence[int],
    input_op: Op | str | None = None,
    source: object | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> GradReceptiveField | Mapping[str, GradReceptiveField]:
    """Probe one target unit on a non-torch trace (the receptive gradient probe).

    Arguments and results are those of :func:`._gradient.gradient_for_unit`,
    whose caller has already checked thresholds and normalized ``source``.

    Raises
    ------
    ReceptiveFieldUnavailableError
        With code ``gradient_replay_unavailable`` when the backend has no
        cone-replay module, an endpoint has no floating payload, an op on the
        cone cannot be replayed, or replay does not reproduce the capture.
    """

    trace = target.source_trace
    shape = tuple(int(size) for size in target.shape)
    normalized = _normalize_unit(unit, shape, target.label)
    if source is not None:
        source_op = resolve_graph_point(trace, source)
        require_path(source_op, target, ReceptiveFieldDirection.RECEPTIVE)
        source_key = str(source_op.graph.io_role or source_op.label)
        descriptor, state = _descriptor_for_source(trace, source_op, target, source_key)
        cone = _build_cone(trace, (source_op,), (target,))
        (grad,) = _receptive_grads(trace, cone, normalized)
        return _build_result(
            target=target,
            role=source_key,
            unit=normalized,
            grad=grad,
            descriptor=descriptor,
            state=state,
            atol=atol,
            rtol=rtol,
            unit_shape=shape,
        )
    return _receptive_inputs(trace, target, normalized, input_op, (atol, rtol))


def _receptive_inputs(
    trace: Trace,
    target: Op,
    unit: tuple[int, ...],
    input_op: Op | str | None,
    thresholds: tuple[float, float],
) -> GradReceptiveField | Mapping[str, GradReceptiveField]:
    """Probe one target unit against the model inputs, as the PyTorch input probe does."""

    with _builtin_rules_when_registry_empty():
        solution = _engine.solve(trace)
        descriptors = _engine.lookup(trace, target)
    selected = _select_inputs(trace, descriptors, input_op)
    if not selected:
        raise ReplayRefusal(
            f"Target op {target.label!r} has no geometrically reachable model input."
        )
    framework = _framework(trace)
    connected = tuple(op for _, op, _ in selected if between_labels(trace, op, target))
    grads: dict[str, torch.Tensor] = {}
    if connected:
        cone = _build_cone(trace, connected, (target,))
        grads = {
            op.label: grad
            for op, grad in zip(connected, _receptive_grads(trace, cone, unit), strict=True)
        }
    replayer = framework.replayer(trace)
    results: dict[str, GradReceptiveField] = {}
    for role, selected_op, descriptor in selected:
        grad = grads.get(selected_op.label)
        if grad is None:
            grad = _zeros_like(framework, replayer, selected_op)
        results[role] = _build_result(
            target=target,
            role=role,
            unit=unit,
            grad=grad,
            descriptor=descriptor,
            state=solution.states.get((target.label, role)),
            atol=thresholds[0],
            rtol=thresholds[1],
            unit_shape=tuple(int(size) for size in target.shape),
        )
    if input_op is not None:
        return next(iter(results.values()))
    return MappingProxyType(results)


def _projective_columns(
    trace: Trace, cone: _Cone, unit: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """Jacobian columns of every target with respect to one source unit."""

    source_native, source_host = _floating_payload(cone, cone.sources[0], "source")
    for target in cone.targets:
        _floating_payload(cone, target, "target")
    tangent = _one_hot(cone, source_native, source_host, unit)

    def fn(value: Any) -> tuple[Any, ...]:
        return _run_cone(cone, (value,))

    columns = _differentiate(
        trace,
        cone,
        lambda: cone.framework.jvp(fn, (source_native,), (tangent,)),
        verify_at=(source_native,),
    )
    if columns is None:
        return tuple(
            _row_sweep_column(trace, cone, index, unit) for index in range(len(cone.targets))
        )
    return tuple(_to_torch(cone, column) for column in columns)


def _row_sweep_column(trace: Trace, cone: _Cone, index: int, unit: tuple[int, ...]) -> torch.Tensor:
    """One Jacobian column by a reverse probe per target element (no forward mode)."""

    target = cone.targets[index]
    target_native, target_host = _floating_payload(cone, target, "target")
    if target_host.size > _ROW_SWEEP_LIMIT:
        raise ReplayRefusal(
            f"Projective gradient to {target.label!r} needs one reverse probe per target "
            f"element on this backend ({target_host.size} elements, limit "
            f"{_ROW_SWEEP_LIMIT}). Probe a smaller target site, or use the receptive "
            "direction from each target unit."
        )
    source_native = cone.replayer.saved(cone.sources[0])

    def fn(value: Any) -> Any:
        return _run_cone(cone, (value,))[index]

    column = np.zeros(target_host.shape, dtype=target_host.dtype)
    for flat in range(target_host.size):
        element = np.unravel_index(flat, target_host.shape)
        cotangent = _one_hot(cone, target_native, target_host, tuple(int(i) for i in element))
        (grad,) = _differentiate(
            trace, cone, lambda: cone.framework.vjp(fn, (source_native,), cotangent)
        )
        column[element] = np.asarray(cone.framework.to_numpy(grad))[unit]
    return torch.from_numpy(column)


def _columns_for_targets(
    trace: Trace, source: Op, targets: Sequence[Op], unit: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    """One Jacobian column per target; a target the source cannot reach gets zeros."""

    source = _canonical(trace, source)
    targets = tuple(_canonical(trace, op) for op in targets)
    connected = tuple(op for op in targets if between_labels(trace, source, op))
    columns: dict[str, torch.Tensor] = {}
    if connected:
        cone = _build_cone(trace, (source,), connected)
        found = _projective_columns(trace, cone, unit)
        columns = {op.label: column for op, column in zip(connected, found, strict=True)}
    framework = _framework(trace)
    replayer = framework.replayer(trace) if len(columns) < len(targets) else None
    return tuple(
        columns[op.label] if op.label in columns else _zeros_like(framework, replayer, op)
        for op in targets
    )


def _zeros_like(framework: ModuleType, replayer: Any, op: Op) -> torch.Tensor:
    """A zero gradient shaped and typed like ``op``'s saved payload."""

    host = np.asarray(framework.to_numpy(replayer.saved(op)))
    return torch.from_numpy(np.zeros(host.shape, dtype=host.dtype))


@_refusing
def projective_for_unit(
    source: Op,
    unit: Sequence[int],
    *,
    target: object | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> GradReceptiveField | Mapping[str, GradReceptiveField]:
    """Probe one source unit on a non-torch trace (the projective gradient probe).

    Arguments and results are those of
    :func:`._gradient_forward.projective_gradient_for_unit`, whose caller has
    already checked thresholds.

    Raises
    ------
    ReceptiveFieldUnavailableError
        As :func:`receptive_for_unit`, plus a target too large for the row sweep
        on a framework without forward-mode autodiff.
    """

    trace = source.source_trace
    targets, explicit = _select_targets(source, target)
    normalized = _normalize_unit(unit, tuple(int(size) for size in source.shape), source.label)
    solution = solve_projective(trace, targets)
    columns = _columns_for_targets(trace, source, targets, normalized)
    results = {
        _target_key(target_op): _build_projective_result(
            source=source,
            target=target_op,
            unit=normalized,
            column=column,
            solution=solution,
            atol=atol,
            rtol=rtol,
        )
        for target_op, column in zip(targets, columns, strict=True)
    }
    if explicit:
        return next(iter(results.values()))
    return MappingProxyType(results)

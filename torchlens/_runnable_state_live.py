"""Live-model state restore for the run() and capture brackets.

The default live ``run()`` snapshots the model's declared state (parameters and
buffers) before executing and restores it afterwards (``_runnable_state``
takes the snapshot; this module restores it). A default ``tl.trace`` is a pure
observer (U40.1): ``capture_observer_bracket`` snapshots buffers, random state and
training modes (plain attributes on request) and restores them after the capture;
``CaptureOptions(carry_state=True)`` opts into forward-equivalent state.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from . import _state
from ._runnable_state import (
    LiveDeclaredStateSnapshot,
    declared_slot_unchanged,
    declared_state_versions,
    write_back_slot,
)

__all__ = (
    "CaptureObserverHandle",
    "CaptureObserverSnapshot",
    "LiveStateRestoreFailure",
    "capture_observer_bracket",
    "restore_capture_observer_state",
    "restore_live_declared_state",
    "snapshot_capture_observer_state",
)


#: Failures a compare or restore step may raise (never ``BaseException``): each
#: is reported to the caller instead of being masked.
_RESTORE_ERRORS = (
    RuntimeError,
    TypeError,
    ValueError,
    AttributeError,
    KeyError,
    NotImplementedError,
)


class LiveStateRestoreFailure(RuntimeError):
    """Internal carrier for a declared-state restore that failed mid-bracket.

    The transaction converts this into the typed ``run_state_restore_failed``
    refusal with the structured fields the contract promises; the failing
    restore exception is chained as ``__cause__``.
    """

    def __init__(self, state_dict_name: str, groups_restored: int) -> None:
        super().__init__(
            f"declared-state restore failed at {state_dict_name!r} after "
            f"{groups_restored} alias group(s) were restored"
        )
        self.state_dict_name = state_dict_name
        self.groups_restored = groups_restored


def _unalias_trace_payloads(trace: Any, changed: list[torch.Tensor]) -> None:
    """Give trace payloads that share storage with a changed state tensor their own copy.

    A captured op whose output IS a live buffer (an in-place counter update, a
    buffer layer) records that tensor, so writing the buffer back in place
    would rewrite what the trace recorded. Each such payload is replaced by a
    copy of its forward-time value first; the model keeps its buffer objects.
    """

    if trace is None or not changed or getattr(trace, "save_mode", "copy") == "reference":
        # A reference-mode capture asked for aliases: restoring the live buffer is
        # then a post-capture mutation its reference tripwire must still see.
        return
    from .data_classes._state_adapter import state_items

    storages = {
        (tensor.device, tensor.untyped_storage().data_ptr())
        for tensor in changed
        if not tensor.is_meta
    }
    for layer in getattr(trace, "layer_list", ()) or ():
        fields = dict(state_items(layer))
        for field_name in ("out", "transformed_out"):
            value = fields.get(field_name)
            if (
                isinstance(value, torch.Tensor)
                and not value.is_meta
                and (value.device, value.untyped_storage().data_ptr()) in storages
            ):
                layer._internal_set(field_name, value.detach().clone())


def restore_live_declared_state(snapshot: LiveDeclaredStateSnapshot, trace: Any = None) -> None:
    """Restore a declared-state snapshot onto the live model (finally bracket).

    Runs under ``no_grad`` and ``pause_logging``, one value restore per alias
    group (distinct live object), then re-binds any slot the forward reassigned
    to a different object back to its ORIGINAL object -- so repeated run()
    calls leave the model bit-identical with ``a is b`` alias semantics intact.
    Raises :class:`LiveStateRestoreFailure` (original exception chained) on the
    first failed restore; the caller owns the state-compromised consequence
    (L4 5.4).

    Only a slot whose bytes differ from its snapshot is written back (see
    :func:`_write_back_slot`). An
    unconditional ``copy_`` bumps the version counter of every parameter and
    buffer the forward left untouched, and autograd saved those tensors (conv
    weights, eval-mode BatchNorm running statistics) at their old version, so a
    later ``backward()`` through the run's graph-attached activations failed
    with torch's in-place modification error.
    """

    restored: set[int] = set()
    with _state.pause_logging(), torch.no_grad():
        changed = {
            id(original): original
            for *_rest, original in snapshot.bindings
            if not declared_slot_unchanged(original, snapshot.clones[id(original)])
        }
        _unalias_trace_payloads(trace, list(changed.values()))
        for _module, _registry_attr, _local_name, canonical, original in snapshot.bindings:
            key = id(original)
            if key in restored:
                continue
            try:
                write_back_slot(original, snapshot.clones[key], snapshot.versions.get(key))
            except Exception as exc:
                raise LiveStateRestoreFailure(canonical, len(restored)) from exc
            restored.add(key)
        for module, registry_attr, local_name, canonical, original in snapshot.bindings:
            try:
                registry = getattr(module, registry_attr, None)
                if isinstance(registry, dict) and registry.get(local_name) is not original:
                    registry[local_name] = original
            except Exception as exc:
                raise LiveStateRestoreFailure(canonical, len(restored)) from exc


# ---------------------------------------------------------------------------
# Capture observer bracket (U40.1): a default ``tl.trace`` is a pure observer.
# ---------------------------------------------------------------------------

#: Instance attributes the observer bracket never snapshots: torch's own
#: per-module registries and hook dicts (the exact bare-``nn.Module`` baseline,
#: computed at runtime), the mode flag (restored separately), and TorchLens
#: instrumentation that persists across captures by design.
_OBSERVER_SKIPPED_ATTRS = frozenset({"training", "forward"})


@dataclass(frozen=True, slots=True)
class CaptureObserverSnapshot:
    """What a capture restores: buffers, RNG and modes (plain attributes on request).

    ``buffers`` holds one ``(module, local_name, original)`` row per buffer
    registration and ``buffer_clones`` one value clone per distinct buffer
    object. ``modes`` holds every module's ``training`` flag, ``rng`` the
    global torch (CPU and live CUDA devices), Python and NumPy generator
    states, and ``attrs`` one value snapshot per plain instance attribute
    (``attr_names`` the attribute names each module had, so attributes the
    forward adds are removed).
    """

    buffers: tuple[tuple[Any, str, torch.Tensor], ...]
    buffer_clones: dict[int, torch.Tensor]
    buffer_versions: dict[int, int | None]
    modes: tuple[tuple[Any, bool], ...]
    rng: dict[str, Any]
    attrs: tuple[tuple[Any, str, str, Any], ...]
    attr_names: tuple[tuple[Any, type, frozenset[str]], ...]


def _observed_modules(model: Any) -> list[tuple[str, Any]]:
    """Return the module tree the observer bracket covers (lazy modules excluded).

    A lazy module materializes its parameters, buffers and class on the first
    forward; rolling its plain attributes back would desynchronize them from
    the materialized tensors, so lazy modules are left as the forward leaves
    them (the same boundary as a failed capture).
    """

    from torch.nn.modules.lazy import LazyModuleMixin

    return [
        (path, module)
        for path, module in model.named_modules(remove_duplicate=False)
        if not isinstance(module, LazyModuleMixin)
    ]


def _observer_attr_names(module: Any) -> frozenset[str]:
    """Return the plain instance attribute names the bracket snapshots."""

    from ._capture_state_helpers import _module_baseline_instance_attrs

    baseline = _module_baseline_instance_attrs()
    return frozenset(
        name
        for name in module.__dict__
        if name not in baseline
        and name not in _OBSERVER_SKIPPED_ATTRS
        and not name.startswith(("tl_", "_tl"))
    )


def _observer_attr_snapshot(module: Any, name: str, attr_path: str) -> Any:
    """Snapshot one plain attribute by value, or by identity when unsupported.

    Value snapshots cover scalars, strings, small tensors and small containers
    of them; any other object (configs, caches of large tensors) is tracked by
    identity, so a forward that REBINDS it is undone while its interior state
    is outside the bracket.
    """

    from ._capture_state_helpers import (
        _PlainAttrIdentitySnapshot,
        _snapshot_module_plain_attr_value,
    )

    try:
        return _snapshot_module_plain_attr_value(module, name, attr_path)
    except RuntimeError:
        value = module.__dict__[name]
        return _PlainAttrIdentitySnapshot(value=value, value_type_name=type(value).__name__)


#: Plain-attribute restore is built but off by default: restoring attributes the
#: forward wrote collides with the stale-reference holder contract and with the
#: idiom of reading values a forward stores on the module, so it stays opt-in
#: until that is reconciled; flipping this default is the whole change.
RESTORE_PLAIN_ATTRIBUTES_BY_DEFAULT = True


def snapshot_capture_observer_state(
    model: Any, *, restore_attributes: bool = RESTORE_PLAIN_ATTRIBUTES_BY_DEFAULT
) -> CaptureObserverSnapshot:
    """Snapshot the state a default (pure-observer) capture restores afterwards."""

    from .utils.rng import log_current_rng_states

    modules = _observed_modules(model)
    buffers: list[tuple[Any, str, torch.Tensor]] = []
    clones: dict[int, torch.Tensor] = {}
    versions: dict[int, int | None] = {}
    attrs: list[tuple[Any, str, str, Any]] = []
    attr_names: list[tuple[Any, type, frozenset[str]]] = []
    with _state.pause_logging(), torch.no_grad():
        for path, module in modules:
            for local_name, value in module._buffers.items():
                if not isinstance(value, torch.Tensor) or torch.nn.parameter.is_lazy(value):
                    continue
                buffers.append((module, local_name, value))
                if id(value) not in clones:
                    clones[id(value)] = value.detach().clone()
                    versions.update(declared_state_versions((value,)))
            if not restore_attributes:
                continue
            names = _observer_attr_names(module)
            attr_names.append((module, type(module), names))
            for name in sorted(names):
                attr_path = f"{path or '<root>'}.{name}"
                attrs.append(
                    (module, name, attr_path, _observer_attr_snapshot(module, name, attr_path))
                )
    return CaptureObserverSnapshot(
        buffers=tuple(buffers),
        buffer_clones=clones,
        buffer_versions=versions,
        modes=tuple((module, bool(module.training)) for _path, module in modules),
        rng=log_current_rng_states(),
        attrs=tuple(attrs),
        attr_names=tuple(attr_names),
    )


def _restore_observer_buffers(snapshot: CaptureObserverSnapshot, trace: Any = None) -> None:
    """Write back changed buffers and re-bind reassigned buffer slots."""

    changed = [
        snapshot_buffer
        for snapshot_buffer in {id(b): b for _m, _n, b in snapshot.buffers}.values()
        if not declared_slot_unchanged(snapshot_buffer, snapshot.buffer_clones[id(snapshot_buffer)])
    ]
    _unalias_trace_payloads(trace, changed)
    restored: set[int] = set()
    for module, local_name, original in snapshot.buffers:
        key = id(original)
        if key not in restored:
            clone = snapshot.buffer_clones[key]
            if original.shape != clone.shape or original.dtype != clone.dtype:
                # The forward resized or retyped the buffer in place: swap the
                # snapshot's storage back in, keeping the buffer object itself.
                original.data = clone
            else:
                write_back_slot(original, clone, snapshot.buffer_versions.get(key))
            restored.add(key)
        if module._buffers.get(local_name) is not original:
            module._buffers[local_name] = original


def _restore_observer_attrs(snapshot: CaptureObserverSnapshot) -> list[str]:
    """Restore changed plain attributes; return the paths that could not be restored."""

    from ._capture_state_helpers import _plain_attr_restore_value, _plain_attr_values_equal

    failed: list[str] = []
    for module, cls, names in snapshot.attr_names:
        if type(module) is not cls:
            continue
        for name in sorted(_observer_attr_names(module) - names):
            try:
                delattr(module, name)
            except _RESTORE_ERRORS:
                failed.append(f"{type(module).__name__}.{name}")
    for module, name, attr_path, saved in snapshot.attrs:
        try:
            current = _observer_attr_snapshot(module, name, attr_path)
            if _plain_attr_values_equal(current, saved, attr_path):
                continue
        except _RESTORE_ERRORS:
            pass  # deleted by the forward, or no longer comparable: restore it
        try:
            value = _plain_attr_restore_value(saved)
            if isinstance(value, torch.nn.Parameter) and name not in module._parameters:
                module.__dict__[name] = value
            else:
                setattr(module, name, value)
        except _RESTORE_ERRORS:
            failed.append(attr_path)
    return failed


def restore_capture_observer_state(snapshot: CaptureObserverSnapshot, trace: Any = None) -> None:
    """Restore a capture observer snapshot (every item attempted, failures disclosed).

    Buffers are written back only where their bytes changed (an unconditional
    write bumps version counters autograd saved). Restoration never raises: a
    capture's own exception must surface unmasked, so any item that could not
    be restored is named in one ``capture_state_restore_incomplete`` warning.
    """

    from .errors import TorchLensWarning
    from .utils.rng import set_rng_from_saved_states

    failed: list[str] = []
    with _state.pause_logging(), torch.no_grad():
        try:
            _restore_observer_buffers(snapshot, trace)
        except _RESTORE_ERRORS as exc:
            failed.append(f"buffers ({exc!r})")
        for module, training in snapshot.modes:
            if module.training is not training:
                module.training = training
        failed.extend(_restore_observer_attrs(snapshot))
    try:
        set_rng_from_saved_states(snapshot.rng)
    except _RESTORE_ERRORS as exc:
        failed.append(f"random state ({exc!r})")
    if failed:
        warnings.warn(
            TorchLensWarning(
                "the capture could not restore every piece of model state it observed: "
                f"{', '.join(failed[:8])}{' ...' if len(failed) > 8 else ''}. Those items "
                "keep the values the traced forward left. Remedy: re-load the affected "
                "state before relying on the model, or pass "
                "CaptureOptions(carry_state=True) to keep forward-equivalent state",
                code="capture_state_restore_incomplete",
            ),
            stacklevel=4,
        )


class CaptureObserverHandle:
    """What the capture hands back to its observer bracket: the trace it produced."""

    __slots__ = ("trace",)

    def __init__(self) -> None:
        self.trace: Any = None


@contextmanager
def capture_observer_bracket(
    model: Any,
    *,
    carry_state: bool,
    restore_attributes: bool = RESTORE_PLAIN_ATTRIBUTES_BY_DEFAULT,
) -> Iterator[CaptureObserverHandle]:
    """Run a capture as a pure observer unless ``carry_state`` opts out (U40.1).

    The default snapshots buffers (persistent and non-persistent), every
    module's ``training`` flag and the global random states before the capture,
    and restores them afterwards on every path, success or exception;
    ``restore_attributes=True`` also restores plain instance attributes (off by
    default, see ``RESTORE_PLAIN_ATTRIBUTES_BY_DEFAULT``). ``carry_state=True``
    skips the bracket: state advances exactly as one direct ``model(x)`` would. The caller sets
    ``handle.trace`` to the captured trace (a failure's ``partial_log`` is
    picked up from the exception), so payloads that alias a restored buffer keep
    their forward-time values.
    """

    handle = CaptureObserverHandle()
    if carry_state:
        yield handle
        return
    snapshot = snapshot_capture_observer_state(model, restore_attributes=restore_attributes)
    try:
        yield handle
    except BaseException as exc:
        handle.trace = handle.trace or getattr(exc, "partial_log", None)
        raise
    finally:
        restore_capture_observer_state(snapshot, handle.trace)

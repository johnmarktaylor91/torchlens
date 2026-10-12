"""Live-model state restore for the run() and capture brackets.

The default live ``run()`` snapshots the model's declared state (parameters and
buffers) before executing and restores it afterwards (``_runnable_state``
takes the snapshot; this module restores it). A default ``tl.trace`` is a pure
observer (U40.1): ``capture_observer_bracket`` snapshots buffers, random state,
training modes and plain attributes and restores them after the capture;
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
from ._runnable_state import LiveDeclaredStateSnapshot
from .utils._torch_compat import tensor_version_or_none

__all__ = (
    "CaptureObserverSnapshot",
    "LiveStateRestoreFailure",
    "capture_observer_bracket",
    "restore_capture_observer_state",
    "restore_live_declared_state",
    "snapshot_capture_observer_state",
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


def _declared_slot_unchanged(original: torch.Tensor, clone: torch.Tensor) -> bool:
    """Return whether a live declared-state slot still holds its snapshot bytes.

    Byte-exact and NaN-safe (the shared state-write audit compare). Any compare
    failure answers ``False`` so the caller restores rather than trusts.
    """

    from .backends.torch.rescue import _state_bytes_equal

    try:
        return (
            original.shape == clone.shape
            and original.dtype == clone.dtype
            and original.device == clone.device
            and _state_bytes_equal(original, clone)
        )
    except Exception:
        return False


def _write_back_slot(
    original: torch.Tensor, clone: torch.Tensor, snapshot_version: int | None
) -> None:
    """Write one slot's snapshot back, only if its bytes changed.

    A write that left the version counter where the snapshot found it (a
    kernel-internal update: train-mode BatchNorm advances its running
    statistics without bumping their counter, and autograd saved them at that
    version) is undone the same way, through ``.data``, so a later
    ``backward()`` still finds the version it saved. A tracked in-place write
    is undone with a tracked ``copy_``: autograd then refuses a backward that
    would read the restored values, loudly, instead of computing a wrong
    gradient.
    """

    if _declared_slot_unchanged(original, clone):
        return
    if snapshot_version is not None and tensor_version_or_none(original) == snapshot_version:
        original.data.copy_(clone)
    else:
        original.copy_(clone)


def restore_live_declared_state(snapshot: LiveDeclaredStateSnapshot) -> None:
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
        for _module, _registry_attr, _local_name, canonical, original in snapshot.bindings:
            key = id(original)
            if key in restored:
                continue
            try:
                _write_back_slot(original, snapshot.clones[key], snapshot.versions.get(key))
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
    """What a default capture restores: buffers, RNG, modes and plain attributes.

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


def snapshot_capture_observer_state(model: Any) -> CaptureObserverSnapshot:
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
                    versions[id(value)] = tensor_version_or_none(value)
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


def _restore_observer_buffers(snapshot: CaptureObserverSnapshot) -> None:
    """Write back changed buffers and re-bind reassigned buffer slots."""

    restored: set[int] = set()
    for module, local_name, original in snapshot.buffers:
        key = id(original)
        if key not in restored:
            _write_back_slot(
                original, snapshot.buffer_clones[key], snapshot.buffer_versions.get(key)
            )
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
            except Exception:
                failed.append(f"{type(module).__name__}.{name}")
    for module, name, attr_path, saved in snapshot.attrs:
        try:
            current = _observer_attr_snapshot(module, name, attr_path)
            if _plain_attr_values_equal(current, saved, attr_path):
                continue
        except (AttributeError, KeyError):
            pass  # deleted by the forward: restore it
        try:
            value = _plain_attr_restore_value(saved)
            if isinstance(value, torch.nn.Parameter) and name not in module._parameters:
                module.__dict__[name] = value
            else:
                setattr(module, name, value)
        except Exception:
            failed.append(attr_path)
    return failed


def restore_capture_observer_state(snapshot: CaptureObserverSnapshot) -> None:
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
            _restore_observer_buffers(snapshot)
        except Exception as exc:
            failed.append(f"buffers ({exc!r})")
        for module, training in snapshot.modes:
            if module.training is not training:
                module.training = training
        failed.extend(_restore_observer_attrs(snapshot))
    try:
        set_rng_from_saved_states(snapshot.rng)
    except Exception as exc:
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


@contextmanager
def capture_observer_bracket(model: Any, *, carry_state: bool) -> Iterator[None]:
    """Run a capture as a pure observer unless ``carry_state`` opts out (U40.1).

    The default snapshots buffers (persistent and non-persistent), every
    module's ``training`` flag, the global random states and plain instance
    attributes before the capture, and restores them afterwards on every path,
    success or exception. ``carry_state=True`` skips the bracket: state
    advances exactly as one direct ``model(x)`` would.
    """

    if carry_state:
        yield
        return
    snapshot = snapshot_capture_observer_state(model)
    try:
        yield
    finally:
        restore_capture_observer_state(snapshot)

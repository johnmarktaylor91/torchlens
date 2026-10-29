"""Capture-time parameter fingerprints: replay refuses a parameter changed since capture.

A replay template holds the live ``nn.Parameter`` a call consumed, not a copy, so an edit
made after capture (an optimizer step, ``load_state_dict``, an in-place op under
``torch.no_grad()``, ``param.data = t``) would make every later replay of that trace, and
of its forks, silently compute with the new weight. When a template classifies a
parameter argument it records a :class:`ParamFingerprint` (version counter, storage
pointer, shape); replay compares it before using the value and refuses with code
``replay_param_changed_since_capture`` unless the replay door was given
``allow_changed_params=True``, in which case the run records the changed parameter
names in ``trace.last_run["replayed_with_changed_params"]``.

Not caught: an in-place write through ``param.data`` (its own version counter, same
storage) or through a numpy or raw-storage alias of the parameter. Session-only: the
fingerprint rides ``LiteralTensor`` inside ``Op.args_template``, which never persists.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ..utils._torch_compat import tensor_version_or_none
from .errors import ReplayPreconditionError

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from .types import LiteralTensor

CHANGED_PARAMS_RUN_KEY = "replayed_with_changed_params"
"""``Trace.last_run`` key naming the changed parameters a replay used under the opt-in."""


@dataclass(frozen=True)
class ParamFingerprint:
    """What a parameter looked like when a template recorded it.

    A dataclass, not a tuple, so the template rewriters (labeling's parent-ref rename,
    cleanup's removed-ref sweep) rebuild it with ``dataclasses.replace`` and keep its type.
    """

    address: str | None
    version: int | None
    storage_ptr: int | None
    shape: tuple[int, ...]


@dataclass
class _ChangedParamScope:
    """One replay door's opt-in and the changed parameters its replays used."""

    allow: bool
    used: list[str] = field(default_factory=list)


_SCOPE: ContextVar[_ChangedParamScope | None] = ContextVar(
    "torchlens_changed_param_scope", default=None
)


def _storage_ptr(param: torch.Tensor) -> int | None:
    """Return the storage pointer through the unwrapped C accessors, or ``None``."""

    from ..backends.torch.completeness_witness import (
        _ORIG_TENSORBASE_UNTYPED_STORAGE,
        _ORIG_UNTYPED_STORAGE_DATA_PTR,
    )

    try:
        return int(_ORIG_UNTYPED_STORAGE_DATA_PTR(_ORIG_TENSORBASE_UNTYPED_STORAGE(param)))
    except (RuntimeError, TypeError, NotImplementedError, AttributeError):
        return None


def param_fingerprint(param: torch.Tensor, address: str | None = None) -> ParamFingerprint:
    """Return the current fingerprint of a parameter.

    Parameters
    ----------
    param:
        Parameter to fingerprint.
    address:
        Parameter address for refusal messages, when known.

    Returns
    -------
    ParamFingerprint
        Version counter, storage pointer and shape (no tensor op runs).
    """

    return ParamFingerprint(
        address=address,
        version=tensor_version_or_none(param),
        storage_ptr=_storage_ptr(param),
        shape=tuple(param.shape),
    )


def _changes(recorded: ParamFingerprint, current: ParamFingerprint) -> list[str]:
    """Name every fingerprint component that differs."""

    changes = []
    if recorded.version != current.version:
        changes.append(f"version counter {recorded.version} -> {current.version}")
    if recorded.storage_ptr != current.storage_ptr:
        changes.append("storage replaced")
    if recorded.shape != current.shape:
        changes.append(f"shape {recorded.shape} -> {current.shape}")
    return changes


def checked_param_value(component: LiteralTensor, pass_log: Op) -> Any:
    """Return a template literal's value, refusing a parameter changed since capture.

    Parameters
    ----------
    component:
        Template literal; only ``nn.Parameter`` values carry a fingerprint.
    pass_log:
        Op being replayed, named in the refusal.

    Returns
    -------
    Any
        The literal's value.

    Raises
    ------
    ReplayPreconditionError
        ``replay_param_changed_since_capture`` when the parameter changed and the
        replay door did not opt in with ``allow_changed_params=True``.
    """

    recorded = getattr(component, "capture_fingerprint", None)
    value = component.value
    if not isinstance(recorded, ParamFingerprint) or not isinstance(value, torch.Tensor):
        return value
    changes = _changes(recorded, param_fingerprint(value, recorded.address))
    if not changes:
        return value
    name = recorded.address or "<unnamed parameter>"
    scope = _SCOPE.get()
    if scope is not None and scope.allow:
        if name not in scope.used:
            scope.used.append(name)
        return value
    raise ReplayPreconditionError(
        code="replay_param_changed_since_capture",
        param=name,
        op=str(getattr(pass_log, "label", None) or pass_log.layer_label),
        changed="; ".join(changes),
        reason=(
            "replay reads the live parameter, which changed after capture, so it would "
            "silently compute with the new value"
        ),
        remedy=(
            "re-capture the trace, or pass allow_changed_params=True (InterventionOptions "
            "or ReplayOptions) to replay with the current values; for an as-if weight "
            "edit use fork.do(tl.params(name), edit)"
        ),
    )


@contextmanager
def changed_param_scope(allow: bool) -> Iterator[_ChangedParamScope]:
    """Scope one replay door's ``allow_changed_params`` choice.

    A nested door (``do()`` calling push internally) keeps an enclosing opt-in and
    shares its record of used parameters.

    Parameters
    ----------
    allow:
        Whether this door opted in to replay with changed parameters.

    Yields
    ------
    _ChangedParamScope
        The active scope.
    """

    outer = _SCOPE.get()
    scope = _ChangedParamScope(
        allow=allow or (outer is not None and outer.allow),
        used=outer.used if outer is not None else [],
    )
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def record_changed_params(log: Trace) -> None:
    """Stamp the active scope's used changed parameters on ``log.last_run``.

    Rebinds ``last_run`` (a fork may share the dict) and drops a stale entry when
    this replay used none, so the key always says what the latest replay did.
    """

    run_ctx = dict(log.last_run) if isinstance(log.last_run, dict) else {}
    scope = _SCOPE.get()
    if scope is not None and scope.used:
        run_ctx[CHANGED_PARAMS_RUN_KEY] = tuple(scope.used)
    else:
        run_ctx.pop(CHANGED_PARAMS_RUN_KEY, None)
    log.last_run = run_ctx

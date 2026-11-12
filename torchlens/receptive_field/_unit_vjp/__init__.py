"""Per-backend cone replay for gradient influence probes on non-torch traces.

A gradient probe off PyTorch has no saved autograd graph to walk. Instead it
replays the captured ops between the probed endpoints from their saved parent
payloads, through each backend validator's own per-op replay, and
differentiates that replay with the framework's own autodiff. One module per
backend supplies the framework half of that contract; the backend-neutral half
(cone selection, verification against the capture, seeding, result building)
lives in :mod:`torchlens.receptive_field._gradient_provider`.

Each backend module exposes, as module-level functions (framework imports stay
inside the functions):

``replayer(trace) -> UnitReplayer``
    Bind the trace's runtime replay records, or raise :class:`ReplayRefusal`
    when they are absent (a loaded trace) or unusable.
``vjp(fn, primals, cotangent) -> tuple``
    The framework's reverse-mode derivative of ``fn`` at ``primals`` against one
    output cotangent: one native gradient per primal.
``jvp(fn, primals, tangents) -> tuple | None``
    The forward-mode derivative (one native tangent per output of ``fn``), or
    ``None`` when the framework has no forward mode (the caller then sweeps rows
    of reverse-mode probes).
``to_numpy(value)`` and ``from_numpy(array, like)``
    Host conversion and its inverse in the dtype of ``like``.
``state_guard()``
    Context manager that restores the framework's global random state (and any
    other global the replay could touch) on exit.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, Protocol

__all__: list[str] = []

#: Backend name to the module implementing its cone replay.
_MODULES: Mapping[str, str] = {
    "jax": "jax",
    "tf": "tf",
    "mlx": "mlx",
    "paddle": "paddle",
    "tinygrad": "tinygrad",
}


class ReplayRefusal(LookupError):
    """An op on the probed cone cannot be replayed, or an endpoint cannot be differentiated.

    Internal control flow only: the provider converts it, at one site, into the
    public :class:`~torchlens.receptive_field._errors.ReceptiveFieldUnavailableError`
    carrying the same message. It is deliberately not a ``RuntimeError``, so the
    influence engine does not mistake it for a torch autograd failure.
    """


class UnitReplayer(Protocol):
    """Replay of single captured ops of one trace from supplied parent values."""

    def saved(self, op: Any) -> Any:
        """Return the op's saved payload in its native framework type."""

    def replay(self, op: Any, env: Mapping[str, Any]) -> Any:
        """Re-run one op, reading parents in ``env`` from it and others from the capture.

        Returns the op's output with the structure of its saved payload. Raises
        :class:`ReplayRefusal` naming the op when it cannot be replayed.
        """

    def close(self, replayed: Any, saved: Any) -> bool:
        """Compare a replayed output with the saved one using the validator's own check."""


def backend_module(name: str) -> ModuleType:
    """Return the cone-replay module for a backend name.

    Parameters
    ----------
    name:
        Registered backend name (``trace._provenance_backend``).

    Returns
    -------
    ModuleType
        The implementing module, imported on first use.

    Raises
    ------
    ReplayRefusal
        If the backend has no cone-replay module.
    """

    module = _MODULES.get(name)
    if module is None:
        raise ReplayRefusal(f"Backend {name!r} does not support gradient receptive-field probes.")
    return importlib.import_module(f"{__name__}.{module}")


def single_parent_value(op: Any, env: Mapping[str, Any], saved: Callable[[str], Any]) -> Any:
    """The value an op with exactly one parent forwards unchanged.

    Parameters
    ----------
    op:
        The forwarding op (an output marker or an identity annotation).
    env:
        Cone values replayed so far, keyed by every parent-reference label.
    saved:
        Saved payload of a parent outside the cone, by parent label.

    Returns
    -------
    Any
        The parent's cone value, or its saved payload when it is off the cone.

    Raises
    ------
    ReplayRefusal
        If ``op`` does not have exactly one parent.
    """

    parents = tuple(getattr(op, "parents", ()) or ())
    if len(parents) != 1:
        raise ReplayRefusal(
            f"Op {op.label!r} has no replay record and is not a single-parent pass-through "
            "(a control-flow region or an opaque call), so a gradient probe cannot cross it."
        )
    (parent,) = parents
    return env[parent] if parent in env else saved(parent)

"""Find the replay adapter that drives finished-trace edits on a non-PyTorch trace.

It imports nothing from the intervention engine's import cycle, so the replay
preconditions (:mod:`._replay_sites`), the PyTorch engine (:mod:`._replay`) and
the backend driver (:mod:`._backend_replay`) all import it at module level.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..backends import _replay_adapters
from ..backends._backend_names import TORCH_BACKEND_NAME
from ._replay_cone import cone_of_effect
from .errors import ReplayPreconditionError

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def trace_backend(trace: Trace) -> str:
    """Return the trace's backend name (``torch`` when unset)."""

    return str(getattr(trace, "_provenance_backend", None) or TORCH_BACKEND_NAME)


def _lookup_adapter(backend: str) -> Any | None:
    """Return the replay adapter module for ``backend``, or None when none is built."""

    try:
        return _replay_adapters.get_adapter(backend)
    except _replay_adapters.ReplayRefusal:
        return None


def backend_adapter(trace: Trace) -> Any | None:
    """Return the replay adapter for a non-PyTorch trace, else None.

    PyTorch traces replay through their own engine and get None, as does a
    backend whose adapter module is not built yet (the caller keeps its
    refusal for those).
    """

    backend = trace_backend(trace)
    if backend == TORCH_BACKEND_NAME:
        return None
    return _lookup_adapter(backend)


def differentiable_replay_cone(log: Trace, origins: Sequence[Op]) -> list[Op]:
    """Return a differentiable replay's source cone; refuse it off PyTorch.

    A differentiable replay keeps a PyTorch autograd graph, so a trace that
    replays through a backend adapter is refused before anything is forked.
    """

    if backend_adapter(log) is not None:
        raise ReplayPreconditionError(
            f"a differentiable replay keeps a PyTorch autograd graph; the "
            f"{trace_backend(log)} backend replays a finished trace without one",
            code="backend_replay_not_differentiable",
            remedy="drop differentiable=True; derive gradients on a fresh capture",
        )
    return cone_of_effect(log, origins)

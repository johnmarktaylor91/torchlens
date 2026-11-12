"""The closing writes every replay engine makes once its commit succeeded.

Shared by the PyTorch replay engine (:mod:`._replay`) and the backend replay
engine (:mod:`._backend_replay_run`) so both leave the same evidence: the
replay-propagated state, the ``last_run`` summary, the changed-parameter
record and the ``replay`` operation row.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .._trace_state import TraceState
from ._replay_cone import _disclosure_label
from ._replay_context import _ensure_replay_run_ctx

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def finish_replay(
    log: Trace,
    origins: Sequence[Op],
    hook_names: Sequence[str],
    *,
    strict: bool,
    cone: Sequence[Op],
    started_at: float,
    record_params: Callable[[Trace], None],
    errors_non_fatal: int = 0,
    differentiable_frontier: dict[str, Any] | None = None,
) -> Trace:
    """Write the replay's state, summary and operation row after its commit.

    ``record_params`` is the changed-parameter recorder, passed in by the
    engine: importing it here would pull this module into the intervention
    package's import cycle.
    """

    origin_labels = tuple(_disclosure_label(origin) for origin in origins)
    hook_names = tuple(hook_names)
    cone_labels = tuple(_disclosure_label(site) for site in cone)
    log.state = TraceState.REPLAY_PROPAGATED
    log._out_recipe_revision = getattr(log, "_spec_revision", 0)
    log.last_run = {
        **_ensure_replay_run_ctx(log),
        "engine": "replay",
        "timestamp": started_at,
        "started_at": started_at,
        "origins": origin_labels,
        "hooks": hook_names,
        "strict": strict,
        "errors_non_fatal": errors_non_fatal,
        "cone": cone_labels,
    }
    record_params(log)
    log._record_operation(
        "replay",
        engine="replay",
        origins=origin_labels,
        hooks=hook_names,
        strict=strict,
        cone=cone_labels,
        errors_non_fatal=errors_non_fatal,
    )
    log._has_direct_writes = False
    if differentiable_frontier is not None:
        log._replay_frontier = dict(differentiable_frontier)
    return log

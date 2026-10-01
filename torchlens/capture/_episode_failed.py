"""FAILED-path episode ledger attach (lane F40a; split out of the ledger).

A failed forward's partial product has a settled FAILED outcome but no
finished module-call records, so step truth is recovered from the surviving
module enter/exit event lanes (exact: an entered call that ran zero traced
ops is still a started step; only an exit witnesses a return), falling back
to the raw op module stacks (non-upgrading). Split from
``_episode_ledger.py`` at the R43 size seam: the failed path reads the
settlement writers' building blocks but nothing reads back into it.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from ._episode_ledger import (
    EPISODE_ANNOTATIONS_KEY,
    EpisodeLedger,
    EpisodeLedgerRow,
    RowStatus,
    _build_header,
    _fidelity_basis,
)

if TYPE_CHECKING:
    from ._episode_ledger import ResolvedEpisode

__all__ = ["attach_failed_episode_ledger"]


def attach_failed_episode_ledger(exc: BaseException, resolved: ResolvedEpisode) -> None:
    """Best-effort ledger attach on the FAILED path (exc.partial_log).

    Failures here only warn -- the user's exception is never masked.
    """

    partial = getattr(exc, "partial_log", None)
    trace = getattr(partial, "trace", None)
    if trace is None:
        return
    try:
        from ._episode_join import _live_step_join_envelope

        started, returned = _failed_step_counts(exc, trace, resolved)
        n_total = max(resolved.n_steps or started, started)
        header = _build_header(
            trace,
            resolved,
            fidelity=_fidelity_basis(resolved, None),
            started=started,
            step_join=_live_step_join_envelope(resolved, n_total, started),
        )
        complete_steps = returned if returned is not None else max(started - 1, 0)
        complete_steps = min(complete_steps, started)
        from ._episode_coupling import settlement_fire_counts

        fire_counts = settlement_fire_counts(
            resolved.coupling_session, started=started, n_total=n_total
        )
        rows: list[EpisodeLedgerRow] = []
        for step in range(n_total):
            if step < complete_steps:
                row_status: RowStatus = "complete"
            elif step < started:
                row_status = "interrupted"
            else:
                row_status = "absent"
            rows.append(
                EpisodeLedgerRow(
                    episode_step=step,
                    role="prefill" if step == 0 else "decode",
                    status=row_status,
                    coord={"member_call_index": step + 1, "pass_range": None},
                    fire_count=(fire_counts[step] if fire_counts is not None else None),
                )
            )
        ledger = EpisodeLedger(header, rows)
        trace.annotations[EPISODE_ANNOTATIONS_KEY] = ledger.to_payload()
    except Exception as attach_exc:  # noqa: BLE001 - never mask the user's failure
        from ..errors import TorchLensWarning

        warnings.warn(
            "episode ledger could not be attached to the failed partial "
            f"product: {type(attach_exc).__name__}: {attach_exc}",
            TorchLensWarning,
            stacklevel=2,
        )


def _failed_step_counts(
    exc: BaseException, trace: Any, resolved: ResolvedEpisode
) -> tuple[int, int | None]:
    """Derive ``(started, returned)`` stepped-call counts on the FAILED path.

    Exact where module enter/exit event lanes survive; non-upgrading raw-op
    module-stack fallback otherwise. Strict-arm truth (lane F40c): the join
    hooks fire BEFORE the wrapped forward, so a feed-closed halt at step k's
    entry raises before the module-enter event -- the step's input arrived
    (it ENTERED in the join sense) but no event witnesses it. The typed
    refusal carries ``break_step``; honor it so row k reads interrupted,
    never absent.
    """

    started = 0
    returned: int | None = None
    events = getattr(trace, "_capture_events", None) or getattr(trace, "capture_events", None)
    enter_events = getattr(events, "module_enter_events", None) if events else None
    if enter_events:
        started = sum(
            1 for event in enter_events if getattr(event, "address", None) == resolved.address
        )
        exit_events = getattr(events, "module_exit_events", None) or ()
        returned = sum(
            1 for event in exit_events if getattr(event, "address", None) == resolved.address
        )
    else:
        raw_ws = getattr(trace, "_raw_graph_ws", None)
        raw_layers = getattr(raw_ws, "raw_layer_dict", None) or {}
        for raw_op in raw_layers.values():
            for entry in getattr(raw_op, "modules", ()) or ():
                if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == resolved.address:
                    started = max(started, int(entry[1]))
    break_step = getattr(exc, "fields", {}).get("break_step")
    if isinstance(break_step, int) and not isinstance(break_step, bool):
        started = max(started, break_step + 1)
        returned = min(returned, break_step) if returned is not None else break_step
    return started, returned

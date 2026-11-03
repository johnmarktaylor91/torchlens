"""Call guards shared by the trace edit verbs (``do``, ``set``, hook attachment).

The in-place mutation warning, the ``do`` engine choice, and the refusal of
batch-coherent edits on a chunked-forward capture. Reached from ``_trace_edits``,
``_trace_hooks`` and the record's thin methods through ``_data_classes._doors``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from torch import nn

from .._errors import InvalidArgumentError
from ..options import suppress_mutate_warnings
from .errors import EngineDispatchError, TraceMutationWarning

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def refuse_chunked_batch_coupling(trace: Trace, entries: list[Any]) -> None:
    """Refuse batch-coherent edits on a chunked-forward capture (D34).

    The capture door already refuses ``chunk_size`` with ``hooks=`` /
    ``intervene=``; the surviving hole is a batch-coherent helper applied
    in REPLAY to a trace whose capture ran in forward chunks: the hook
    would see one CHUNK and permute or average it as if it were the whole
    batch -- a silently wrong experiment. Whole-batch visibility is not
    provable from the chunk metadata in v1, so the guard refuses typed.
    An ordinary (unchunked) minibatch is a real batch and stays legal.

    Parameters
    ----------
    trace:
        The trace whose capture metadata says whether the batch ran in chunks.
    entries:
        Normalized hook entries about to attach.
    """

    if not trace._options_chunk_size:
        return
    for entry in entries:
        helper = getattr(entry, "helper_spec", None)
        if helper is None:
            continue
        if dict(helper.metadata).get("batch_coherent"):
            raise InvalidArgumentError(
                f"edit {helper.name!r} reads or exchanges the WHOLE "
                "batch at fire time, but this trace was captured in forward "
                "chunks (chunked_forward=True): each fire would see one "
                "chunk and treat it as the full batch",
                code="chunked_replay_batch_coupling",
                remedy="re-capture without chunk_size for whole-batch "
                "edits, or apply a per-row edit (resample_rows_from) whose "
                "value never couples rows",
                argument="hooks_or_site",
            )


def warn_if_root_mutation(trace: Trace, *, confirm_mutation: bool) -> None:
    """Emit the once-per-root mutate-in-place warning, attributed past the thin method.

    Parameters
    ----------
    trace:
        The root trace about to be mutated in place.
    confirm_mutation:
        Whether the caller explicitly accepted in-place mutation.
    """

    if confirm_mutation or trace._parent_run is not None or trace._warned_mutate_in_place:
        return
    if suppress_mutate_warnings.is_suppressed:
        return
    warnings.warn(
        "TraceMutationWarning: Trace mutators modify root logs in place. "
        "Use log.fork(...) for isolated edits or pass confirm_mutation=True.",
        TraceMutationWarning,
        stacklevel=4,
    )
    trace._warned_mutate_in_place = True


def select_do_engine(trace: Trace, engine: str, *, model: nn.Module | None, x: Any) -> str:
    """Resolve the concrete ``do`` engine from caller arguments.

    Parameters
    ----------
    trace:
        The trace the engine will act on.
    engine:
        Requested engine name.
    model:
        Optional model supplied for rerun.
    x:
        Optional input supplied for rerun.

    Returns
    -------
    str
        Concrete engine name.

    Raises
    ------
    EngineDispatchError
        If the engine cannot be inferred from an incomplete model/input pair.
    """

    if engine != "auto":
        if engine == "rerun" and (model is None or x is None):
            raise EngineDispatchError(
                "do(..., engine='rerun') requires both model= and inputs=. "
                "Pass both, or use engine='replay' if full rerun is not intended."
            )
        return engine
    if (model is None) != (x is None):
        raise EngineDispatchError(
            "do(engine='auto') needs both model= and inputs= for rerun, or neither for "
            "replay. Pass both, or use engine='replay' if rerun is not intended."
        )
    return "rerun" if model is not None else "replay"

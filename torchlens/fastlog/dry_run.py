"""Dry-run tracing API for fastlog predicates."""

from __future__ import annotations

from typing import Any

from torch import nn

from .._input_coerce import _coerce_input_args
from ._recorder import Recorder
from .options import PredicateFn
from .types import RecordingTrace


def dry_run(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    save: PredicateFn | None = None,
    history_size: int = 8,
    include_source_events: bool = False,
    random_seed: int | None = None,
) -> RecordingTrace:
    """Run predicates over one forward pass without retaining tensor payloads.

    Parameters
    ----------
    model:
        PyTorch module to execute.
    input_args:
        Tensor, list, or tuple of positional model inputs.
    input_kwargs:
        Optional keyword arguments for the model call.
    save, history_size, include_source_events, random_seed:
        Fastlog dry-run options. Predicate exceptions propagate immediately.

    Returns
    -------
    RecordingTrace
        Chronological event contexts and any accumulated predicate failures.
    """

    input_args = _coerce_input_args(model, input_args)
    recorder_kwargs: dict[str, Any] = {
        "save": save,
        "history_size": history_size,
        "include_source_events": include_source_events,
        "on_predicate_error": "fail-fast",
        "streaming": None,
        "random_seed": random_seed,
    }
    recorder_cm = Recorder(
        model,
        **recorder_kwargs,
    )

    with recorder_cm as recorder:
        if recorder._state is None:  # noqa: SLF001
            raise RuntimeError("Recorder state was not initialized")
        if recorder._capture_events is None:  # noqa: SLF001
            raise RuntimeError("Recorder capture events were not initialized")
        recorder._state.no_tensor_capture = True  # noqa: SLF001
        recorder.log(input_args, input_kwargs)
        contexts = tuple(recorder._state.all_contexts)  # noqa: SLF001
        decisions = tuple(
            bool(getattr(event, "predicate_matched", False))
            for event in recorder._capture_events.op_events  # noqa: SLF001
        )
        failures = tuple(recorder._state.predicate_failures)  # noqa: SLF001
    return RecordingTrace(contexts=contexts, decisions=decisions, predicate_failures=failures)

"""Dry-run tracing API for fastlog predicates."""

from __future__ import annotations

from typing import Any

from torch import nn

from ._recorder import Recorder
from .options import PredicateFn
from .types import RecordingTrace


def dry_run(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    *,
    keep_op: PredicateFn | None = None,
    keep_module: PredicateFn | None = None,
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
    keep_op, keep_module, history_size, include_source_events, random_seed:
        Fastlog dry-run options. Predicate exceptions propagate immediately.

    Returns
    -------
    RecordingTrace
        Chronological event contexts and any accumulated predicate failures.
    """

    with Recorder(
        model,
        keep_op=keep_op,
        keep_module=keep_module,
        history_size=history_size,
        include_source_events=include_source_events,
        on_predicate_error="fail-fast",
        streaming=None,
        random_seed=random_seed,
    ) as recorder:
        if recorder._state is None:  # noqa: SLF001
            raise RuntimeError("Recorder state was not initialized")
        recorder._state.no_tensor_capture = True  # noqa: SLF001
        recorder.log(input_args, input_kwargs)
        contexts = tuple(recorder._state.all_contexts)  # noqa: SLF001
        failures = tuple(recorder._state.predicate_failures)  # noqa: SLF001
    return RecordingTrace(contexts=contexts, predicate_failures=failures)

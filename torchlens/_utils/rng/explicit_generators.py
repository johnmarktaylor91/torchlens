"""Pre-call state of ``torch.Generator`` objects passed straight to a random op.

``torch.randn(shape, generator=g)`` draws from ``g``, not from the global stream, so
restoring the global torch state (:func:`~.streams.set_rng_from_saved_states`) cannot
reproduce it: by replay time ``g`` has advanced past the draw. Capture therefore records
each such generator's state just before the call, under :data:`EXPLICIT_GENERATORS_KEY`
of the op's ``func_rng_states``, keyed by where the generator sat in the call
(``"arg:<index>"`` or ``"kwarg:<name>"``). Replay sets the generator found at the same
place in the replay arguments to that state for the call, then puts back the state it
had, so the user's generator continues from where the model left it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from .monitor_state import _suppress_active_monitor_marks
from .streams import log_current_rng_states

EXPLICIT_GENERATORS_KEY = "explicit_generators"


def _generator_slots(
    args: tuple[Any, ...] | list[Any], kwargs: dict[str, Any]
) -> Iterator[tuple[str, torch.Generator]]:
    """Yield ``(slot, generator)`` for each generator passed directly as an argument."""

    for index, value in enumerate(args):
        if isinstance(value, torch.Generator):
            yield f"arg:{index}", value
    for name, value in kwargs.items():
        if isinstance(value, torch.Generator):
            yield f"kwarg:{name}", value


def log_op_rng_states(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Snapshot the global torch streams plus every generator passed to this call.

    Parameters
    ----------
    args:
        Positional arguments of the op about to run.
    kwargs:
        Keyword arguments of the op about to run.

    Returns
    -------
    dict[str, Any]
        :func:`~.streams.log_current_rng_states` (torch only), with an
        :data:`EXPLICIT_GENERATORS_KEY` entry mapping each argument slot that holds a
        ``torch.Generator`` to its state before the call, when there is one.
    """

    states = log_current_rng_states(torch_only=True)
    with _suppress_active_monitor_marks():
        generators = {slot: gen.get_state() for slot, gen in _generator_slots(args, kwargs)}
    if generators:
        states[EXPLICIT_GENERATORS_KEY] = generators
    return states


@contextmanager
def explicit_generators_at(
    rng_states: dict[str, Any] | None, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Iterator[None]:
    """Run the body with each recorded argument generator at its pre-call state.

    Parameters
    ----------
    rng_states:
        The op's ``func_rng_states``; only its :data:`EXPLICIT_GENERATORS_KEY` entry
        is read. ``None`` or no entry leaves every generator untouched.
    args:
        Positional replay arguments.
    kwargs:
        Keyword replay arguments.

    Yields
    ------
    None
        Inside the block each generator holds its recorded state; on exit (return or
        raise) each holds the state it had on entry.
    """

    recorded = (rng_states or {}).get(EXPLICIT_GENERATORS_KEY) or {}
    targets = [
        (gen, recorded[slot]) for slot, gen in _generator_slots(args, kwargs) if slot in recorded
    ]
    if not targets:
        yield
        return
    with _suppress_active_monitor_marks():
        entry_states = [(gen, gen.get_state()) for gen, _ in targets]
        for gen, state in targets:
            gen.set_state(state)
    try:
        yield
    finally:
        with _suppress_active_monitor_marks():
            for gen, state in entry_states:
                gen.set_state(state)

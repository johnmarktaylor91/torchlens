"""Early stop (``stop_after=``) for the JAX interpreter.

The interpreter evaluates every equation concretely, so a stop selector, including a
value-dependent one, is evaluated on the real output of each primitive equation. When it
matches, the equation's capture is stored first (save, then halt, as on PyTorch) and the
interpreter unwinds every nested frame (jit calls, cond branches, scan and while bodies)
before any further primitive binds. ``jax.make_jaxpr`` and the eager reference call may
already have run the Python function body; the promise is that the interpreted suffix never
binds.

:class:`JaxHaltSite` wraps the equation session's hook (the session simulates the label
each capture event receives) and supplies the interpreter's halt check.
:func:`stamp_halted_trace` gives a halted trace torch's stop fields: ``is_stopped``,
``stop_reason`` and the halt frontier, the latched stop request and a ``stopped`` output
slot. Only primitive equations are stop sites; region boundaries are not.

This module imports nothing from the backend's import cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from ..._ir.selector_eval import selector_contains_kind

_ALIAS_RETRY_KINDS = ("label", "contains", "regex")


def selector_matches(selector: Any, ctx: Any) -> bool:
    """Evaluate a stop selector on one record context, with the short-label alias retry."""

    if bool(selector(ctx)):
        return True
    if getattr(ctx, "layer_type", None) is None or getattr(ctx, "type_index", None) is None:
        return False
    if not any(selector_contains_kind(selector, kind) for kind in _ALIAS_RETRY_KINDS):
        return False
    return bool(selector(replace(ctx, label=f"{ctx.layer_type}_{ctx.type_index}")))


class JaxHaltSite:
    """Stop selector state for one capture: the session hook wrapper and the halt check.

    Parameters
    ----------
    session
        The capture's ``JaxEquationSession`` (its ``hook`` labels each primitive equation).
    selector
        The stop selector (``stop_after=``).
    """

    def __init__(self, session: Any, selector: Any) -> None:
        """Wrap the session's label reservation so each primitive's label is known."""

        self.session = session
        self.selector = selector
        self.label: str | None = None
        self._frontier_index: int | None = None
        self._last_reserved: Any = None
        reserve = session._reserve

        def remembering_reserve(layer_type: str) -> Any:
            """Reserve as the session does and remember the result."""

            self._last_reserved = reserve(layer_type)
            return self._last_reserved

        session._reserve = remembering_reserve

    def hook(
        self, captures: Sequence[Any], eqn: Any, outputs: tuple[Any, ...], stacks: Any
    ) -> tuple[Any, ...]:
        """Run the session hook, then test the stop selector on the (possibly edited) outputs."""

        outputs = self.session.hook(captures, eqn, outputs, stacks)
        if self.label is None:
            ctx = self.session._context(self._last_reserved, outputs, stacks)
            if selector_matches(self.selector, ctx):
                self.label = self._last_reserved.label_raw
                self._frontier_index = len(captures)
        return outputs

    def check(self, capture: Any) -> bool:
        """Halt check for the interpreter: true once the frontier capture is stored."""

        return self._frontier_index is not None and capture.index == self._frontier_index


def frontier_output(result: Any) -> Any:
    """Return the frontier equation's output (one value, or a tuple for several)."""

    values = tuple(result.halt_frontier.output_values)
    return values[0] if len(values) == 1 else values


def stamp_halted_trace(
    trace: Any,
    site: JaxHaltSite,
    *,
    stopped: Callable[[str | None], Any],
    stop_request: Callable[..., Any],
) -> None:
    """Give a halted trace torch's stop fields.

    Parameters
    ----------
    trace
        Trace under capture.
    site
        The halt site that matched (its ``label`` is the frontier label).
    stopped
        ``torchlens._output_slot.stopped`` (passed in to keep this module a leaf).
    stop_request
        ``torchlens._capture.outcome.StopRequest``.
    """

    label = site.label
    # Latch first, as torch's evaluate_halt does, so the settlement stamp sees the stop.
    trace.__dict__["_stop_requested"] = stop_request(
        kind="halt", reason=label, boundary_label=label
    )
    trace.is_stopped = True
    trace.stop_reason = label
    trace._graph_stop_frontier = label
    trace._output_slot = stopped(label)
    trace.transformed_output = None

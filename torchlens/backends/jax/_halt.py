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

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..._capture._stop_after_policy import settle_unfired_stop_after
from ..._capture.outcome import StopRequest
from ..._ir.selector_eval import module_union_addresses, selector_contains_kind
from ..._output_slot import returned, stopped
from .._options import pop_stop_directive
from ._interp_hooks import InterpreterHooks

_ALIAS_RETRY_KINDS = ("label", "contains", "regex")


def _selectors() -> Any:
    """Return ``torchlens._selection.selectors`` (L4; deferred, this module is L2)."""

    from ..._selection import selectors

    return selectors


def resolve_stop_selector(halt: Any) -> Any:
    """Return the selector for a ``stop_after=`` value (a bare callable becomes ``tl.where``).

    A selector passes through; any other value is returned unchanged for the caller's
    typed refusal, as on MLX.
    """

    selectors = _selectors()
    if isinstance(halt, (selectors.BaseSelector, type)) or not callable(halt):
        return halt
    return selectors.WhereSelector(halt)


def selector_matches(selector: Any, ctx: Any) -> bool:
    """Evaluate a stop selector on one record context, with the short-label alias retry.

    Module selectors match module exits, not equations, so they never match here.
    """

    if module_union_addresses(selector) is not None:
        return False
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


def finish_halted(result: Any, session: Any) -> tuple[Any, Any]:
    """Finish a halted run: settle edits, make the frontier outputs the run's outputs.

    Parameters
    ----------
    result
        Interpreter result with a ``halt_frontier``.
    session
        The capture's ``JaxEquationSession`` (its ``finish`` rewraps edited captures).

    Returns
    -------
    tuple[Any, Any]
        The result whose ``outputs`` are the frontier capture's output values, and the
        frontier output (one value, or a tuple for several).
    """

    result, _ = session.finish(result, None)
    values = tuple(result.captures[-1].output_values)
    output = values[0] if len(values) == 1 else values
    return replace(result, outputs=values), output


def stamp_halted_trace(trace: Any, site: JaxHaltSite) -> None:
    """Give a halted trace torch's stop fields.

    Parameters
    ----------
    trace
        Trace under capture.
    site
        The halt site that matched (its ``label`` is the frontier label).
    """

    label = site.label
    # Latch first, as torch's evaluate_halt does, so the settlement stamp sees the stop.
    trace.__dict__["_stop_requested"] = StopRequest(kind="halt", reason=label, boundary_label=label)
    trace.is_stopped = True
    trace.stop_reason = label
    trace._graph_stop_frontier = label
    trace._output_slot = stopped(label)
    trace.transformed_output = None


class JaxStop:
    """The ``stop_after=`` plan for one JAX capture (inactive when none was requested).

    The backend calls :meth:`pop` before rejecting extra kwargs, passes
    :meth:`interpreter_kwargs` to the interpreter, calls :meth:`finish` on the interpreter
    result and :meth:`stamp_output` where it sets the output slot.

    Parameters
    ----------
    stop_site
        The public ``stop_after=`` value, or ``None``.
    """

    def __init__(self, stop_site: Any) -> None:
        """Resolve the stop selector."""

        self.stop_site = stop_site
        self.selector = resolve_stop_selector(stop_site)
        self.site: JaxHaltSite | None = None
        self.halted = False

    @classmethod
    def pop(cls, kwargs: dict[str, Any]) -> JaxStop:
        """Pop the stop directive from ``kwargs``; leave an unsupported value for rejection."""

        stop = cls(pop_stop_directive(kwargs))
        if stop.selector is not None and not isinstance(stop.selector, _selectors().BaseSelector):
            kwargs["stop_after"] = stop.stop_site
        return stop

    def interpreter_kwargs(self, session: Any) -> dict[str, Any]:
        """Return the interpreter's ``equation_hook`` and ``hooks`` for this capture."""

        if self.selector is None:
            return {"equation_hook": session.hook, "hooks": InterpreterHooks(session=session)}
        self.site = JaxHaltSite(session, self.selector)
        hooks = InterpreterHooks(session=session, halt=self.site.check)
        return {"equation_hook": self.site.hook, "hooks": hooks}

    def finish(self, result: Any, session: Any, output: Any) -> tuple[Any, Any]:
        """Finish the run: a halted run's outputs are the frontier capture's outputs."""

        if result.halt_frontier is None:
            return session.finish(result, output)
        self.halted = True
        return finish_halted(result, session)

    def stamp_output(self, trace: Any, output: Any, output_transform: Any) -> None:
        """Set the output slot (``stopped`` when halted, else ``returned``); settle the stop.

        A selector that never fired refuses (``stop_after_never_fired``) and an unfired
        callable is recorded, as on PyTorch.
        """

        if self.halted and self.site is not None:
            stamp_halted_trace(trace, self.site)
        else:
            trace._output_slot = returned(output)
            trace.transformed_output = (
                output_transform(output) if callable(output_transform) else None
            )
        selector_shaped = self.stop_site is not None and self.stop_site is self.selector
        settle_unfired_stop_after(trace, self.stop_site, selector_shaped=selector_shaped)

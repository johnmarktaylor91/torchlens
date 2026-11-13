"""Derived gradients of the effective edited JAX program.

A capture-time edit (``trace(intervene=...)``) replaces one equation's output
before every downstream equation reads it, so the recorded trace is the
forward of an *edited* program. Derived gradients must differentiate that
program, not rerun the unedited model: this module evaluates the captured
closed jaxpr with each fired edit re-applied to the output of the exact
equation it fired on.

The edited program is traced once into its own closed jaxpr, which a gradient
view evaluates like the captured one. The edited equation is found by object
identity (the interpreter hook saw the equation object itself). Nested pure calls (``pjit``, ``closed_call``) are
evaluated inline so an edit inside a jitted helper is reached; every other
equation, control-flow primitives included, binds whole. An edit is
differentiable only when its equation fired exactly once and the evaluator
reaches it exactly once; an edit inside a loop body, a taken ``cond`` branch
or a helper called more than once has no single position in the program and
refuses typed (``intervention_with_derived_grads_unsupported``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..registry import BackendUnsupportedError

_CODE = "intervention_with_derived_grads_unsupported"
#: Call primitives evaluated inline, mapped to the param holding the callee.
_INLINE_CALL_PARAMS = {"pjit": "jaxpr", "jit": "jaxpr", "closed_call": "call_jaxpr"}


def edited_grads_refusal(message: str, *, code: str) -> BackendUnsupportedError:
    """Return the typed refusal for an edit derived gradients cannot follow."""

    return BackendUnsupportedError(
        f"JAX derived gradients over an edited capture: {message}",
        code=code,
        option="grad_options",
    )


@dataclass
class EditedProgram:
    """The fired edits of one capture, keyed by equation identity.

    ``appliers`` maps ``id(eqn)`` to the output-to-replacement function;
    ``equations`` keeps each edited equation alive (so its id stays unique);
    ``sites`` maps ``id(eqn)`` to the edited site's raw label; ``seen`` counts
    how often the interpreter bound each edited equation.
    """

    appliers: dict[int, Callable[[Any], Any]] = field(default_factory=dict)
    equations: dict[int, Any] = field(default_factory=dict)
    sites: dict[int, str] = field(default_factory=dict)
    seen: Counter[int] = field(default_factory=Counter)

    def note_bound(self, eqn: Any) -> None:
        """Count one interpreter binding of ``eqn`` (only edited ids are read)."""

        self.seen[id(eqn)] += 1

    def add(self, eqn: Any, site: str, apply: Callable[[Any], Any]) -> None:
        """Record one fired edit on ``eqn``."""

        key = id(eqn)
        if key in self.appliers:
            raise edited_grads_refusal(
                f"site {site!r} was edited on an equation that also fired as "
                f"{self.sites[key]!r}; the equation has no single position in the program.",
                code=_CODE,
            )
        self.appliers[key], self.equations[key], self.sites[key] = apply, eqn, site

    def check_positions(self) -> None:
        """Refuse an edited equation the interpreter bound more than once."""

        for key, site in self.sites.items():
            if self.seen[key] != 1:
                raise edited_grads_refusal(
                    f"site {site!r} sits on an equation bound {self.seen[key]} times (a loop "
                    "body, an unrolled branch or a repeated helper call), so the edit has no "
                    "single position to differentiate. Edit an equation outside control flow.",
                    code=_CODE,
                )


def _read(env: Mapping[Any, Any], var: Any) -> Any:
    """Return a jaxpr variable's value (literals carry their own)."""

    from jax.extend import core

    return var.val if isinstance(var, core.Literal) else env[var]


def _inline_callee(eqn: Any, program: EditedProgram) -> Any | None:
    """Return the closed callee of an inline call equation that holds an edit."""

    param = _INLINE_CALL_PARAMS.get(eqn.primitive.name)
    callee = None if param is None else eqn.params.get(param)
    if callee is None or not _holds_edit(getattr(callee, "jaxpr", callee), program):
        return None
    return callee


def _holds_edit(jaxpr: Any, program: EditedProgram) -> bool:
    """Whether ``jaxpr`` (or an inline callee inside it) binds an edited equation."""

    for eqn in jaxpr.eqns:
        if id(eqn) in program.appliers:
            return True
        param = _INLINE_CALL_PARAMS.get(eqn.primitive.name)
        callee = None if param is None else eqn.params.get(param)
        if callee is not None and _holds_edit(getattr(callee, "jaxpr", callee), program):
            return True
    return False


def _bind(eqn: Any, invals: Sequence[Any], program: EditedProgram, applied: Counter[int]) -> Any:
    """Bind one equation (inlining an edited callee) and apply its edit, if any."""

    callee = _inline_callee(eqn, program)
    if callee is not None:
        consts = list(getattr(callee, "consts", ()))
        outs: Any = eval_edited_jaxpr(callee.jaxpr, consts, invals, program, applied)
    else:
        subfuns, params = eqn.primitive.get_bind_params(eqn.params)
        outs = eqn.primitive.bind(*subfuns, *invals, **params)
        outs = list(outs) if eqn.primitive.multiple_results else [outs]
    apply = program.appliers.get(id(eqn))
    if apply is not None:
        applied[id(eqn)] += 1
        outs = [apply(outs[0]), *outs[1:]]
    return outs


def eval_edited_jaxpr(
    jaxpr: Any,
    consts: Sequence[Any],
    args: Sequence[Any],
    program: EditedProgram,
    applied: Counter[int],
) -> list[Any]:
    """Evaluate ``jaxpr`` with ``program``'s edits applied.

    Parameters
    ----------
    jaxpr
        Open jaxpr (``closed_jaxpr.jaxpr``).
    consts, args
        Values for ``jaxpr.constvars`` and ``jaxpr.invars``.
    program
        Fired edits keyed by equation identity.
    applied
        Counter of edit applications per equation id, updated in place.

    Returns
    -------
    list[Any]
        Values of ``jaxpr.outvars``; differentiable by JAX.
    """

    env: dict[Any, Any] = dict(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, args, strict=True))
    for eqn in jaxpr.eqns:
        invals = [_read(env, var) for var in eqn.invars]
        env.update(zip(eqn.outvars, _bind(eqn, invals, program, applied), strict=True))
    return [_read(env, var) for var in jaxpr.outvars]


def edited_program_of(backend: Any, trace: Any) -> EditedProgram | None:
    """Return the fired edits of ``trace``'s capture, or ``None`` when none fired.

    The capture's equation session is pinned on ``backend`` while the trace is
    finalized; it holds an ``EditedProgram`` only when ``grad_options`` was set.
    """

    session = getattr(backend, "_equation_session", None)
    program = getattr(session, "program", None)
    if program is None or getattr(session, "trace", None) is not trace:
        return None
    return program if program.appliers else None


def edited_closed_jaxpr(closed_jaxpr: Any, program: EditedProgram) -> Any:
    """Return the edited program as a closed jaxpr over ``(*consts, *inputs)``.

    The captured jaxpr is traced once (abstractly, from its input avals) through
    the edited evaluator. The result takes the captured constants and inputs as
    explicit arguments, and its own constants are only the arrays the edits
    introduced (a ``tl.add`` delta, a ``tl.replace_with`` value, an array a
    callable hook closes over), so a gradient view can pass them explicitly.

    Raises
    ------
    BackendUnsupportedError
        ``intervention_with_derived_grads_unsupported`` when an edit is not
        reached exactly once (it sits inside a primitive bound whole, such as a
        loop body or a ``cond`` branch).
    """

    import jax

    def avals(values: Sequence[Any]) -> list[Any]:
        """Return shape-and-dtype stand-ins for ``values``."""

        return [
            jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)
            for aval in values
        ]

    def run(consts: list[Any], *args: Any) -> list[Any]:
        """Evaluate the captured jaxpr with edits; refuse an edit reached other than once."""

        applied: Counter[int] = Counter()
        outs = eval_edited_jaxpr(closed_jaxpr.jaxpr, consts, args, program, applied)
        for key, site in program.sites.items():
            if applied[key] != 1:
                raise edited_grads_refusal(
                    f"site {site!r} sits inside a control-flow region the gradient program "
                    "binds whole, so the edit cannot be re-applied there. Edit an equation "
                    "outside control flow.",
                    code=_CODE,
                )
        return outs

    const_avals = avals([var.aval for var in closed_jaxpr.jaxpr.constvars])
    return jax.make_jaxpr(run)(const_avals, *avals(closed_jaxpr.in_avals))

"""Hoist closure constants of nested ``jit`` calls into the top-level jaxpr.

``jax.make_jaxpr`` keeps a concrete array that a nested ``jax.jit`` closes over (for
example an Equinox or NNX module's weights read inside ``jax.jit`` in ``__call__``) as a
constant of the nested closed jaxpr. The backend records closure constants only from the
top-level jaxpr (each is a ``constant`` source op that its readers list as a parent, and
module parameters resolve by value identity), and the inliner refuses a nested call that
carries constants. Moving them to the top level, and passing them to the call as leading
inputs, keeps every read a recorded parent and makes the call inlinable. Equations,
primitives and values are unchanged; a call whose parameters are not the known jit
surface is left as it is, so the inliner's purity gate still refuses it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

_PER_INPUT_PARAMS = ("donated_invars", "in_layouts", "in_shardings")
_PER_INPUT_FILL = {"donated_invars": False, "in_layouts": None}


def hoist_nested_jit_consts(
    closed_jaxpr: Any,
    core: Any,
    *,
    call_primitives: frozenset[str],
    known_params: frozenset[str],
) -> Any:
    """Return ``closed_jaxpr`` with top-level jit calls' constants hoisted.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr from ``jax.make_jaxpr``.
    core
        ``jax.extend.core``.
    call_primitives
        Jit-call primitive names (``jit``, ``pjit``).
    known_params
        The jit-call parameter names the inliner accepts.

    Returns
    -------
    Any
        The same object when nothing was hoisted, else a new closed jaxpr whose extra
        constants are the nested calls' constants (the same array objects).
    """

    jaxpr = closed_jaxpr.jaxpr
    var_by_const = {id(c): v for v, c in zip(jaxpr.constvars, closed_jaxpr.consts, strict=True)}
    hoisted_by_inner: dict[int, tuple[Any, list[Any]]] = {}
    constvars, consts = list(jaxpr.constvars), list(closed_jaxpr.consts)
    eqns = []
    for eqn in jaxpr.eqns:
        rewritten = _hoist_call(eqn, core, call_primitives, known_params, hoisted_by_inner)
        if rewritten is None:
            eqns.append(eqn)
            continue
        new_eqn, inner_consts = rewritten
        lead = []
        for var, value in zip(new_eqn.params["jaxpr"].jaxpr.invars, inner_consts, strict=False):
            existing = var_by_const.get(id(value))
            if existing is None:
                var_by_const[id(value)] = existing = var
                constvars.append(var)
                consts.append(value)
            lead.append(existing)
        eqns.append(new_eqn.replace(invars=[*lead, *new_eqn.invars]))
    if len(consts) == len(closed_jaxpr.consts):
        return closed_jaxpr
    return core.ClosedJaxpr(jaxpr.replace(constvars=constvars, eqns=eqns), consts)


def _hoist_call(
    eqn: Any,
    core: Any,
    call_primitives: frozenset[str],
    known_params: frozenset[str],
    hoisted_by_inner: dict[int, tuple[Any, list[Any]]],
) -> tuple[Any, list[Any]] | None:
    """Return ``(call with const-free body, body constants)``, or ``None`` to keep ``eqn``.

    The returned equation's invars are not yet extended: the caller prepends one
    top-level constvar per constant. A shared body (one jitted function called twice)
    is converted once, so both calls bind the same constvars.
    """

    if eqn.primitive.name not in call_primitives or not set(eqn.params) <= known_params:
        return None
    inner = eqn.params.get("jaxpr")
    if not isinstance(inner, core.ClosedJaxpr):
        return None
    key = id(inner)
    if key not in hoisted_by_inner:
        flat = hoist_nested_jit_consts(
            inner, core, call_primitives=call_primitives, known_params=known_params
        )
        if not tuple(flat.consts):
            return None
        hoisted_by_inner[key] = (_const_free_body(flat, core), list(flat.consts))
    body, inner_consts = hoisted_by_inner[key]
    params = _extend_per_input_params(eqn.params, len(inner_consts))
    if params is None:
        return None
    return eqn.replace(params={**params, "jaxpr": body}), inner_consts


def _const_free_body(closed: Any, core: Any) -> Any:
    """Turn a body's constvars into leading invars (closure conversion)."""

    inner = closed.jaxpr
    count = len(inner.constvars)
    changes: dict[str, Any] = {
        "constvars": [],
        "invars": [*inner.constvars, *inner.invars],
    }
    info = getattr(inner, "debug_info", None)
    arg_names = getattr(info, "arg_names", None)
    if arg_names is not None:
        changes["debug_info"] = _replace_field(info, arg_names=("",) * count + tuple(arg_names))
    return core.ClosedJaxpr(inner.replace(**changes), [])


def _replace_field(record: Any, **fields: Any) -> Any:
    """Replace fields on a ``DebugInfo`` (a named tuple or a dataclass by JAX version)."""

    if hasattr(record, "_replace"):
        return record._replace(**fields)
    return dataclasses.replace(record, **fields)


def _extend_per_input_params(params: Any, count: int) -> dict[str, Any] | None:
    """Prepend ``count`` entries to each per-input jit parameter, or ``None`` if unknown.

    New inputs are not donated, have no layout, and take the call's existing
    unspecified sharding; a call with explicit shardings or no input to copy the
    unspecified marker from keeps its constants (the purity gate refuses it).
    """

    extended = dict(params)
    for name in _PER_INPUT_PARAMS:
        if name not in params:
            continue
        current = tuple(params[name])
        if name in _PER_INPUT_FILL:
            fill = _PER_INPUT_FILL[name]
        elif current and all(type(item).__name__ == "UnspecifiedValue" for item in current):
            fill = current[0]
        else:
            return None
        extended[name] = (fill,) * count + current
    return extended

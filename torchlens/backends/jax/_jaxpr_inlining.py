"""Nested-jaxpr inlining predicates for the JAX interpreter.

Which nested calls (``jit``/``pjit``, ``custom_jvp_call``) the closed-jaxpr
interpreter may inline: the call must close over no constants, carry no
effects, donate nothing and leave its shardings unspecified. Split out of
``jaxpr.py``, which re-exports the primitive sets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

EFFECT_PRIMITIVES = frozenset({"debug_callback", "io_callback", "pure_callback"})
PURE_JIT_CALL_PRIMITIVES = frozenset({"jit", "pjit"})
JIT_CALL_PARAM_NAMES = frozenset(
    {
        "compiler_options_kvs",
        "ctx_mesh",
        "donated_invars",
        "in_layouts",
        "in_shardings",
        "inline",
        "jaxpr",
        "keep_unused",
        "name",
        "out_layouts",
        "out_shardings",
    }
)


def _nested_jaxpr_params(eqn: Any, core: Any) -> tuple[Any, ...]:
    """Return nested closed-jaxpr parameters on an equation.

    Parameters
    ----------
    eqn
        JAX equation to inspect.
    core
        Imported JAX core module.

    Returns
    -------
    tuple[Any, ...]
        Nested closed-jaxpr-like parameters.
    """

    nested: list[Any] = []
    for value in eqn.params.values():
        if isinstance(value, core.ClosedJaxpr):
            nested.append(value)
        elif isinstance(value, (tuple, list)):
            nested.extend(item for item in value if isinstance(item, core.ClosedJaxpr))
    return tuple(nested)


def _closed_jaxpr_param(eqn: Any, core: Any) -> Any | None:
    """Return a nested closed jaxpr parameter for recognized call primitives.

    Parameters
    ----------
    eqn
        Jaxpr equation.
    core
        Imported JAX core module.

    Returns
    -------
    Any | None
        Nested closed jaxpr, if recognized.
    """

    if eqn.primitive.name in PURE_JIT_CALL_PRIMITIVES:
        nested = eqn.params.get("jaxpr")
    elif eqn.primitive.name == "custom_jvp_call":
        nested = eqn.params.get("call_jaxpr")
    else:
        return None
    if isinstance(nested, core.ClosedJaxpr):
        return nested
    return None


def _has_nested_jaxpr(eqn: Any, core: Any) -> bool:
    """Return whether an equation contains any nested jaxpr parameter.

    Parameters
    ----------
    eqn
        Jaxpr equation.
    core
        Imported JAX core module.

    Returns
    -------
    bool
        True when any parameter contains a nested jaxpr.
    """

    def contains(value: Any) -> bool:
        """Recursively inspect a value for jaxpr-like objects.

        Parameters
        ----------
        value
            Candidate parameter value.

        Returns
        -------
        bool
            True when ``value`` contains a jaxpr-like object.
        """

        if isinstance(value, core.ClosedJaxpr | core.Jaxpr):
            return True
        if hasattr(value, "jaxpr") and isinstance(getattr(value, "jaxpr"), core.Jaxpr):
            return True
        if isinstance(value, Mapping):
            return any(contains(child) for child in value.values())
        if isinstance(value, tuple | list):
            return any(contains(child) for child in value)
        return False

    return any(contains(value) for value in eqn.params.values())


def _custom_jvp_call_is_inlinable(eqn: Any, core: Any) -> bool:
    """Return whether a ``custom_jvp_call`` primal is safe to inline.

    A ``jax.custom_jvp`` wrapper customizes DERIVATIVES only; the equation's
    ``call_jaxpr`` is the exact forward primal, so interpreting it inline is
    forward-faithful by construction.  Inlining is accepted when the primal
    frame is recursively effect-free and closes over no constants -- the same
    purity bar nested JIT calls must pass.  This covers both library shapes
    in the wild: a jit/pjit-wrapped primal (``jax.nn.relu``; the former
    name-allowlist check required the literal primitive name ``"jit"``, which
    modern JAX spells ``"pjit"``) and a flat primitive primal
    (``jax.nn.softplus``).  The replay oracle still validates every inlined
    equation numerically, so a primal whose forward disagreed with the
    wrapper would fail validation rather than pass silently.

    Parameters
    ----------
    eqn
        Candidate equation.
    core
        Imported JAX core module.

    Returns
    -------
    bool
        True when the custom-JVP primal is provably pure and const-free.
    """

    nested = _closed_jaxpr_param(eqn, core)
    if eqn.primitive.name != "custom_jvp_call" or nested is None:
        return False
    return _closed_jaxpr_is_effect_free_and_const_free(nested, core)


def _can_inline_call(eqn: Any, core: Any) -> bool:
    """Return whether a call equation is safe to inline.

    Parameters
    ----------
    eqn
        Candidate equation.
    core
        Imported JAX core module.

    Returns
    -------
    bool
        True when the equation matches the pure inlinable call surface.
    """

    if eqn.effects:
        return False
    nested = _closed_jaxpr_param(eqn, core)
    if nested is None:
        return False
    if eqn.primitive.name in PURE_JIT_CALL_PRIMITIVES:
        return _nested_jaxpr_is_pure_inlinable(eqn, nested, core)
    if eqn.primitive.name == "custom_jvp_call":
        return _custom_jvp_call_is_inlinable(eqn, core)
    return False


def _nested_jaxpr_is_pure_inlinable(eqn: Any, closed_jaxpr: Any, core: Any) -> bool:
    """Return whether a nested JIT-shaped call is safe to inline.

    Parameters
    ----------
    eqn
        Candidate call equation carrying ``closed_jaxpr``.
    closed_jaxpr
        Nested closed jaxpr frame to inspect.
    core
        Imported JAX core module.

    Returns
    -------
    bool
        True when the nested call and all nested closed jaxprs have no
        closure constants, effects, donation, explicit shardings, or unknown
        call parameters.
    """

    if not _jit_call_params_are_known(eqn):
        return False
    if _has_donated_invars(eqn.params.get("donated_invars", ())):
        return False
    if not _jit_shardings_are_unspecified(eqn.params.get("in_shardings", ())):
        return False
    if not _jit_shardings_are_unspecified(eqn.params.get("out_shardings", ())):
        return False
    return _closed_jaxpr_is_effect_free_and_const_free(closed_jaxpr, core)


def _jit_call_params_are_known(eqn: Any) -> bool:
    """Return whether a JIT-shaped call uses only audited parameters.

    Parameters
    ----------
    eqn
        Candidate call equation.

    Returns
    -------
    bool
        True when all parameter names are part of the audited JIT-call frame.
    """

    return set(eqn.params) <= JIT_CALL_PARAM_NAMES


def _has_donated_invars(donated_invars: Any) -> bool:
    """Return whether any nested JIT input is donated.

    Parameters
    ----------
    donated_invars
        Donation flags from the JAX call equation.

    Returns
    -------
    bool
        True when any donation flag is truthy.
    """

    if isinstance(donated_invars, bool):
        return donated_invars
    if isinstance(donated_invars, Sequence):
        return any(bool(flag) for flag in donated_invars)
    return bool(donated_invars)


def _jit_shardings_are_unspecified(shardings: Any) -> bool:
    """Return whether a JIT sharding parameter is fully unspecified.

    Parameters
    ----------
    shardings
        JAX ``in_shardings`` or ``out_shardings`` parameter value.

    Returns
    -------
    bool
        True when every entry is JAX's unspecified sharding sentinel.
    """

    if shardings is None:
        return True
    if not isinstance(shardings, Sequence) or isinstance(shardings, str):
        shardings = (shardings,)
    return all(_is_unspecified_jax_sharding(sharding) for sharding in shardings)


def _is_unspecified_jax_sharding(sharding: Any) -> bool:
    """Return whether a value is JAX's unspecified sharding sentinel.

    Parameters
    ----------
    sharding
        Candidate sharding value.

    Returns
    -------
    bool
        True for the JAX ``UnspecifiedValue`` sentinel.
    """

    return type(sharding).__name__ == "UnspecifiedValue"


def _closed_jaxpr_is_effect_free_and_const_free(closed_jaxpr: Any, core: Any) -> bool:
    """Return whether a closed jaxpr tree has no constants or effects.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr frame to inspect recursively.
    core
        Imported JAX core module.

    Returns
    -------
    bool
        True when the closed jaxpr and all nested closed jaxprs are pure and
        free of closed-over constants.
    """

    if tuple(getattr(closed_jaxpr, "consts", ())):
        return False
    if tuple(getattr(closed_jaxpr, "effects", ())):
        return False
    for child_eqn in closed_jaxpr.jaxpr.eqns:
        if child_eqn.effects:
            return False
        if child_eqn.primitive.name in EFFECT_PRIMITIVES:
            return False
        child_nested = _closed_jaxpr_param(child_eqn, core)
        if child_nested is not None:
            if child_eqn.primitive.name in PURE_JIT_CALL_PRIMITIVES:
                if not _nested_jaxpr_is_pure_inlinable(child_eqn, child_nested, core):
                    return False
            elif not _closed_jaxpr_is_effect_free_and_const_free(child_nested, core):
                return False
            continue
        for child_closed_jaxpr in _nested_jaxpr_params(child_eqn, core):
            if not _closed_jaxpr_is_effect_free_and_const_free(child_closed_jaxpr, core):
                return False
    return True

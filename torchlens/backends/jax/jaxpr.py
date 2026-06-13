"""Closed-jaxpr derivation and interpretation for the JAX backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SAFE_JIT_NAMES = frozenset(
    {
        "_bernoulli",
        "_normal",
        "_normal_real",
        "_one_hot",
        "_randint",
        "_uniform",
        "_where",
        "clip",
        "relu",
    }
)
REJECTED_NESTED_PRIMITIVES = frozenset(
    {
        "cond",
        "custom_vjp_call",
        "remat2",
        "scan",
        "while",
        "while_loop",
    }
)
EFFECT_PRIMITIVES = frozenset({"debug_callback", "io_callback", "pure_callback"})


@dataclass(frozen=True)
class JaxEquationCapture:
    """Captured primitive equation after safe pure-call inlining.

    Parameters
    ----------
    index
        Sequential equation index in flattened capture order.
    primitive
        JAX primitive name.
    primitive_obj
        Primitive object used for replay validation.
    input_values
        Concrete inputs read from the interpreter environment.
    output_values
        Concrete primitive outputs.
    params
        Primitive bind parameters.
    source_path
        Nested source path through the outer jaxpr and inlined calls.
    invars
        String representations of equation input variables.
    outvars
        String representations of equation output variables.
    input_avals
        String representations of input abstract values.
    output_avals
        String representations of output abstract values.
    inlined
        Whether this equation came from an accepted nested pure call.
    """

    index: int
    primitive: str
    primitive_obj: Any
    input_values: tuple[Any, ...]
    output_values: tuple[Any, ...]
    params: Mapping[str, Any]
    source_path: tuple[str, ...]
    invars: tuple[str, ...]
    outvars: tuple[str, ...]
    input_avals: tuple[str | None, ...]
    output_avals: tuple[str | None, ...]
    inlined: bool


@dataclass(frozen=True)
class JaxCaptureResult:
    """Concrete result for an interpreted closed jaxpr.

    Parameters
    ----------
    outputs
        Flat dynamic jaxpr outputs.
    equations
        Captured primitive equations.
    inlined_call_primitives
        Names of pure call primitives expanded during interpretation.
    """

    outputs: tuple[Any, ...]
    equations: tuple[JaxEquationCapture, ...]
    inlined_call_primitives: tuple[str, ...]


def derive_closed_jaxpr(fn: Any, args: Sequence[Any]) -> Any:
    """Derive a closed jaxpr for ``fn(*args)``.

    Parameters
    ----------
    fn
        JAX-compatible callable.
    args
        Dynamic positional arguments.

    Returns
    -------
    Any
        ``jax.core.ClosedJaxpr`` on the installed JAX runtime.
    """

    import jax

    return jax.make_jaxpr(fn)(*args)


def flatten_dynamic_args(args: Sequence[Any]) -> tuple[list[Any], Any]:
    """Flatten positional dynamic arguments in JAX pytree order.

    Parameters
    ----------
    args
        Positional arguments passed to the captured function.

    Returns
    -------
    tuple[list[Any], Any]
        Flat leaves and the corresponding treedef.
    """

    import jax

    return jax.tree.flatten(tuple(args))


def reject_undeclared_consts(closed_jaxpr: Any) -> None:
    """Reject hidden closure constants in the closed jaxpr.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr to inspect.

    Raises
    ------
    ValueError
        If the jaxpr contains closure constants.
    """

    consts = tuple(getattr(closed_jaxpr, "consts", ()))
    if consts:
        raise ValueError(
            "JAX backend found closed-jaxpr constants. Pass captured values as explicit "
            "pytree leaves to fn(params, *inputs); hidden closure constants are unsupported "
            "in this preview."
        )


def interpret_closed_jaxpr_with_inlining(
    closed_jaxpr: Any, flat_args: Sequence[Any]
) -> JaxCaptureResult:
    """Interpret a closed jaxpr while inlining safe pure call primitives.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr to interpret.
    flat_args
        Dynamic argument leaves in jaxpr input order.

    Returns
    -------
    JaxCaptureResult
        Final outputs and flattened equation captures.
    """

    from jax._src import core

    _assert_no_rejected_effects(closed_jaxpr)
    captures: list[JaxEquationCapture] = []
    inlined_calls: list[str] = []

    def interpret_inner(
        inner: Any,
        args: Sequence[Any],
        path: tuple[str, ...],
        inlined_depth: int,
    ) -> tuple[Any, ...]:
        """Interpret one closed jaxpr frame.

        Parameters
        ----------
        inner
            Closed jaxpr frame.
        args
            Concrete frame inputs.
        path
            Nested source path.
        inlined_depth
            Non-zero when interpreting an accepted call primitive.

        Returns
        -------
        tuple[Any, ...]
            Concrete frame outputs.
        """

        env: dict[Any, Any] = {}
        for var, const in zip(inner.jaxpr.constvars, inner.consts):
            _write_env(env, var, const, core)
        for var, arg in zip(inner.jaxpr.invars, args):
            _write_env(env, var, arg, core)

        for eqn_index, eqn in enumerate(inner.jaxpr.eqns):
            primitive_name = eqn.primitive.name
            eqn_path = (*path, f"{eqn_index}:{primitive_name}")
            if primitive_name in REJECTED_NESTED_PRIMITIVES:
                raise ValueError(f"unsupported nested primitive: {primitive_name}")
            if eqn.effects:
                raise ValueError(f"unsupported equation effects: {primitive_name} {eqn.effects}")
            if primitive_name in EFFECT_PRIMITIVES:
                raise ValueError(f"unsupported jaxpr effect primitive: {primitive_name}")
            nested = _closed_jaxpr_param(eqn, core)
            inputs = tuple(_read_env(env, var, core) for var in eqn.invars)
            if nested is not None:
                if not _can_inline_call(eqn, core):
                    raise ValueError(
                        f"unsupported nested call primitive: {primitive_name} "
                        f"name={eqn.params.get('name')!r}"
                    )
                inlined_calls.append(primitive_name)
                outputs = interpret_inner(nested, inputs, eqn_path, inlined_depth + 1)
                for var, value in zip(eqn.outvars, outputs):
                    _write_env(env, var, value, core)
                continue
            if _has_nested_jaxpr(eqn, core):
                raise ValueError(f"unsupported nested jaxpr in primitive: {primitive_name}")
            result = eqn.primitive.bind(*inputs, **eqn.params)
            outputs = tuple(result if eqn.primitive.multiple_results else (result,))
            for var, value in zip(eqn.outvars, outputs):
                _write_env(env, var, value, core)
            captures.append(
                JaxEquationCapture(
                    index=len(captures),
                    primitive=primitive_name,
                    primitive_obj=eqn.primitive,
                    input_values=inputs,
                    output_values=outputs,
                    params=eqn.params,
                    source_path=eqn_path,
                    invars=tuple(str(var) for var in eqn.invars),
                    outvars=tuple(str(var) for var in eqn.outvars),
                    input_avals=tuple(_aval_repr(var) for var in eqn.invars),
                    output_avals=tuple(_aval_repr(var) for var in eqn.outvars),
                    inlined=inlined_depth > 0,
                )
            )
        return tuple(_read_env(env, var, core) for var in inner.jaxpr.outvars)

    outputs = interpret_inner(closed_jaxpr, flat_args, ("root",), 0)
    return JaxCaptureResult(outputs, tuple(captures), tuple(inlined_calls))


def replay_equation(
    capture: JaxEquationCapture, inputs: Sequence[Any] | None = None
) -> tuple[Any, ...]:
    """Replay a captured primitive equation on saved inputs.

    Parameters
    ----------
    capture
        Captured equation.
    inputs
        Optional replacement inputs. When omitted, saved inputs are used.

    Returns
    -------
    tuple[Any, ...]
        Primitive outputs.
    """

    replay_inputs = tuple(capture.input_values if inputs is None else inputs)
    result = capture.primitive_obj.bind(*replay_inputs, **capture.params)
    return tuple(result if capture.primitive_obj.multiple_results else (result,))


def _read_env(env: Mapping[Any, Any], atom: Any, core: Any) -> Any:
    """Read a jaxpr atom from the interpreter environment.

    Parameters
    ----------
    env
        Variable environment.
    atom
        Jaxpr atom.
    core
        Imported JAX core module.

    Returns
    -------
    Any
        Concrete atom value.
    """

    if isinstance(atom, core.Literal):
        return atom.val
    return env[atom]


def _write_env(env: dict[Any, Any], var: Any, value: Any, core: Any) -> None:
    """Write a jaxpr variable unless it is a dropped output.

    Parameters
    ----------
    env
        Mutable variable environment.
    var
        Jaxpr variable.
    value
        Concrete value.
    core
        Imported JAX core module.

    Returns
    -------
    None
        The environment is updated in place.
    """

    if not isinstance(var, core.DropVar):
        env[var] = value


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

    if eqn.primitive.name == "jit":
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


def _is_library_custom_jvp_call(eqn: Any, core: Any) -> bool:
    """Return whether ``custom_jvp_call`` is a recognized library wrapper.

    Parameters
    ----------
    eqn
        Candidate equation.
    core
        Imported JAX core module.

    Returns
    -------
    bool
        True for allowlisted library-internal custom JVP calls.
    """

    nested = _closed_jaxpr_param(eqn, core)
    if eqn.primitive.name != "custom_jvp_call" or nested is None:
        return False
    nested_eqns = nested.jaxpr.eqns
    return bool(nested_eqns) and all(
        child.primitive.name == "jit" and child.params.get("name") in SAFE_JIT_NAMES
        for child in nested_eqns
    )


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
        True when the equation matches the pure library allowlist.
    """

    if eqn.effects:
        return False
    if eqn.primitive.name == "jit":
        return eqn.params.get("name") in SAFE_JIT_NAMES
    if eqn.primitive.name == "custom_jvp_call":
        return _is_library_custom_jvp_call(eqn, core)
    return False


def _assert_no_rejected_effects(closed_jaxpr: Any) -> None:
    """Reject jaxprs containing effects or callback primitives.

    Parameters
    ----------
    closed_jaxpr
        Jaxpr to scan.

    Raises
    ------
    ValueError
        If unsupported effects are present.
    """

    effects = tuple(getattr(closed_jaxpr, "effects", ()))
    if effects:
        raise ValueError(f"unsupported jaxpr effects: {effects}")
    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive.name in EFFECT_PRIMITIVES:
            raise ValueError(f"unsupported jaxpr effect primitive: {eqn.primitive.name}")


def _aval_repr(var: Any) -> str | None:
    """Return a stable string representation of a variable aval.

    Parameters
    ----------
    var
        Jaxpr variable-like object.

    Returns
    -------
    str | None
        Abstract-value representation, when present.
    """

    aval = getattr(var, "aval", None)
    return None if aval is None else str(aval)

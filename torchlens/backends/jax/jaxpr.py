"""Closed-jaxpr derivation and interpretation for the JAX backend."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, cast

from ...ir.events import JaxEquationKind

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
_ITERATION_COMPONENT_RE = re.compile(
    r"^(?P<prefix>(?:scan|while)[_\-:]?)?(?:iter|iteration)(?P<sep>[:=_-])\d+$"
)
_BRACKETED_ITERATION_RE = re.compile(r"(?P<prefix>\b(?:scan|while))\[(?:iter=)?\d+\]")


@dataclass(frozen=True)
class JaxEquationCapture:
    """Captured primitive equation after safe pure-call inlining.

    Parameters
    ----------
    index
        Sequential equation index in flattened capture order.
    kind
        Replay kind used to dispatch validation.
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
    kind: JaxEquationKind
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


def derive_closed_jaxpr(fn: Any, args: Sequence[Any], static_argnums: Sequence[int] = ()) -> Any:
    """Derive a closed jaxpr for ``fn(*args)``.

    Parameters
    ----------
    fn
        JAX-compatible callable.
    args
        Dynamic positional arguments.
    static_argnums
        Positional argument indexes to treat as declared static values.

    Returns
    -------
    Any
        ``jax.core.ClosedJaxpr`` on the installed JAX runtime.
    """

    import jax

    return jax.make_jaxpr(fn, static_argnums=tuple(static_argnums))(*args)


def flatten_dynamic_args(
    args: Sequence[Any], static_argnums: Sequence[int] = ()
) -> tuple[list[Any], Any]:
    """Flatten positional dynamic arguments in JAX pytree order.

    Parameters
    ----------
    args
        Positional arguments passed to the captured function.
    static_argnums
        Positional argument indexes excluded from dynamic jaxpr inputs.

    Returns
    -------
    tuple[list[Any], Any]
        Flat leaves and the corresponding treedef.
    """

    import jax

    static_indexes = set(static_argnums)
    dynamic_args = tuple(arg for index, arg in enumerate(args) if index not in static_indexes)
    return jax.tree.flatten(dynamic_args)


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
    closed_jaxpr: Any,
    flat_args: Sequence[Any],
    *,
    jax_control_flow: str = "unroll",
    jax_max_control_flow_unroll: int = 64,
) -> JaxCaptureResult:
    """Interpret a closed jaxpr while inlining safe pure call primitives.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr to interpret.
    flat_args
        Dynamic argument leaves in jaxpr input order.
    jax_control_flow
        Control-flow handling policy. ``"reject"`` preserves the historical
        nested-primitive error; ``"unroll"`` expands supported control-flow
        primitives into graph-visible equations.
    jax_max_control_flow_unroll
        Maximum supported static unroll length for one control-flow primitive.

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
            if primitive_name == "scan":
                if jax_control_flow == "reject":
                    raise ValueError(f"unsupported nested primitive: {primitive_name}")
                outputs = _interpret_scan(
                    eqn=eqn,
                    env=env,
                    core=core,
                    path=eqn_path,
                    inlined_depth=inlined_depth,
                    captures=captures,
                    interpret_inner=interpret_inner,
                    jax_max_control_flow_unroll=jax_max_control_flow_unroll,
                )
                for var, value in zip(eqn.outvars, outputs):
                    _write_env(env, var, value, core)
                continue
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
                    kind="primitive",
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


def _interpret_scan(
    *,
    eqn: Any,
    env: dict[Any, Any],
    core: Any,
    path: tuple[str, ...],
    inlined_depth: int,
    captures: list[JaxEquationCapture],
    interpret_inner: Callable[[Any, Sequence[Any], tuple[str, ...], int], tuple[Any, ...]],
    jax_max_control_flow_unroll: int,
) -> tuple[Any, ...]:
    """Unroll one ``lax.scan`` equation into synthetic and body captures.

    Parameters
    ----------
    eqn
        Outer scan equation.
    env
        Current interpreter environment.
    core
        Imported JAX core module.
    path
        Source path for the scan equation.
    inlined_depth
        Current accepted-call inlining depth.
    captures
        Mutable capture list receiving synthetic and body equations.
    interpret_inner
        Recursive closed-jaxpr interpreter.
    jax_max_control_flow_unroll
        Maximum supported static scan length.

    Returns
    -------
    tuple[Any, ...]
        Concrete outputs matching the outer scan equation's outvars.
    """

    num_consts = int(eqn.params["num_consts"])
    num_carry = int(eqn.params["num_carry"])
    length = int(eqn.params["length"])
    reverse = bool(eqn.params.get("reverse", False))
    body_jaxpr = eqn.params["jaxpr"]
    if length < 1:
        raise ValueError("unsupported lax.scan length: zero-length scans are not yet supported.")
    if length > jax_max_control_flow_unroll:
        raise ValueError(
            "JAX lax.scan length exceeds jax_max_control_flow_unroll: "
            f"length={length}, max={jax_max_control_flow_unroll}."
        )

    inputs = tuple(_read_env(env, var, core) for var in eqn.invars)
    consts = inputs[:num_consts]
    carry = inputs[num_consts : num_consts + num_carry]
    xs_values = inputs[num_consts + num_carry :]
    ys_by_output_index: list[list[Any]] | None = None
    body_invars = tuple(body_jaxpr.jaxpr.invars)
    body_outvars = tuple(body_jaxpr.jaxpr.outvars)

    for logical_index in range(length):
        physical_index = length - 1 - logical_index if reverse else logical_index
        x_slices = tuple(_slice_scan_leaf(value, physical_index) for value in xs_values)
        for xs_index, (xs_value, x_slice) in enumerate(zip(xs_values, x_slices)):
            captures.append(
                JaxEquationCapture(
                    index=len(captures),
                    kind="scan_read",
                    primitive="scan_read",
                    primitive_obj=None,
                    input_values=(xs_value,),
                    output_values=(x_slice,),
                    params={"index": physical_index, "axis": 0},
                    source_path=(
                        *path,
                        f"iter={logical_index}",
                        f"x={xs_index}:scan_read",
                    ),
                    invars=(str(eqn.invars[num_consts + num_carry + xs_index]),),
                    outvars=(f"{path[-1]}/iter={logical_index}/x={xs_index}",),
                    input_avals=(_aval_repr(eqn.invars[num_consts + num_carry + xs_index]),),
                    output_avals=(_aval_repr(body_invars[num_consts + num_carry + xs_index]),),
                    inlined=inlined_depth > 0,
                )
            )
        body_outputs = interpret_inner(
            body_jaxpr,
            (*consts, *carry, *x_slices),
            (*path, f"iter={logical_index}", "body"),
            inlined_depth,
        )
        carry = body_outputs[:num_carry]
        ys = body_outputs[num_carry:]
        if ys_by_output_index is None:
            ys_by_output_index = [[] for _value in ys]
        for ys_index, y_value in enumerate(ys):
            ys_by_output_index[ys_index].append((physical_index, y_value))

    stack_outputs: list[Any] = []
    for ys_index, indexed_values in enumerate(ys_by_output_index or []):
        sorted_indexed_values = tuple(sorted(indexed_values, key=lambda item: item[0]))
        ordered_values = tuple(value for _index, value in sorted_indexed_values)
        output = _stack_scan_outputs(ordered_values)
        stack_outputs.append(output)
        captures.append(
            JaxEquationCapture(
                index=len(captures),
                kind="scan_stack",
                primitive="scan_stack",
                primitive_obj=None,
                input_values=ordered_values,
                output_values=(output,),
                params={"axis": 0},
                source_path=(*path, f"ys={ys_index}:scan_stack"),
                invars=tuple(
                    f"{path[-1]}/ys={ys_index}/index={index}"
                    for index, _value in sorted_indexed_values
                ),
                outvars=(str(eqn.outvars[num_carry + ys_index]),),
                input_avals=tuple(
                    _aval_repr(body_outvars[num_carry + ys_index]) for _ in ordered_values
                ),
                output_avals=(_aval_repr(eqn.outvars[num_carry + ys_index]),),
                inlined=inlined_depth > 0,
            )
        )
    return (*carry, *stack_outputs)


def replay_equation(
    capture: JaxEquationCapture, inputs: Sequence[Any] | None = None
) -> tuple[Any, ...]:
    """Replay a captured equation on saved inputs.

    Parameters
    ----------
    capture
        Captured equation.
    inputs
        Optional replacement inputs. When omitted, saved inputs are used.

    Returns
    -------
    tuple[Any, ...]
        Captured outputs.
    """

    try:
        handler = JAX_REPLAY_HANDLERS[cast(JaxEquationKind, capture.kind)]
    except KeyError as exc:
        expected = ", ".join(ALL_JAX_EQUATION_KINDS)
        raise NotImplementedError(
            f"JAX replay kind {capture.kind!r} is not registered; expected one of: {expected}."
        ) from exc
    return handler(capture, inputs)


def jax_equivalence_key(capture: JaxEquationCapture) -> str:
    """Return the backend-neutral structural key for a JAX equation.

    Parameters
    ----------
    capture
        Captured JAX equation metadata.

    Returns
    -------
    str
        Stable key composed from primitive name, input/output avals, and the
        normalized source path.
    """

    payload = {
        "primitive": capture.primitive,
        "input_avals": tuple(capture.input_avals),
        "output_avals": tuple(capture.output_avals),
        "source_path": normalize_jax_source_path(capture.source_path),
    }
    return "jax:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


def normalize_jax_source_path(source_path: Sequence[str]) -> tuple[str, ...]:
    """Normalize a JAX source path for cross-iteration grouping.

    Parameters
    ----------
    source_path
        Nested source path emitted by the jaxpr interpreter.

    Returns
    -------
    tuple[str, ...]
        Path with scan/while iteration indexes removed and the leaf equation
        ordinal stripped, while retaining non-leaf call/control-site identity.
    """

    normalized: list[str] = []
    last_index = len(source_path) - 1
    for index, component in enumerate(source_path):
        normalized_component = _normalize_jax_source_path_component(
            str(component),
            is_leaf=index == last_index,
        )
        if normalized_component is not None:
            normalized.append(normalized_component)
    return tuple(normalized)


def _normalize_jax_source_path_component(component: str, *, is_leaf: bool) -> str | None:
    """Normalize one source-path component.

    Parameters
    ----------
    component
        Source-path component.
    is_leaf
        Whether this component names the captured equation itself.

    Returns
    -------
    str | None
        Normalized component, or ``None`` when the component is only an
        iteration marker.
    """

    iteration_match = _ITERATION_COMPONENT_RE.match(component)
    if iteration_match is not None:
        return None
    without_bracketed_iteration = _BRACKETED_ITERATION_RE.sub(
        lambda match: match.group("prefix"),
        component,
    )
    if is_leaf:
        _ordinal, separator, primitive_name = without_bracketed_iteration.partition(":")
        if separator and _ordinal.isdigit() and primitive_name:
            return primitive_name
    return without_bracketed_iteration


JaxReplayHandler = Callable[[JaxEquationCapture, Sequence[Any] | None], tuple[Any, ...]]
ALL_JAX_EQUATION_KINDS: tuple[JaxEquationKind, ...] = (
    "primitive",
    "scan_read",
    "scan_stack",
    "cond_decision",
    "while_decision",
)


def _replay_primitive(
    capture: JaxEquationCapture, inputs: Sequence[Any] | None = None
) -> tuple[Any, ...]:
    """Replay a captured JAX primitive bind.

    Parameters
    ----------
    capture
        Captured primitive equation.
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


def _replay_scan_read(
    capture: JaxEquationCapture, inputs: Sequence[Any] | None = None
) -> tuple[Any, ...]:
    """Replay a synthetic scan leaf read.

    Parameters
    ----------
    capture
        Captured synthetic scan-read equation.
    inputs
        Optional replacement parent array. When omitted, saved inputs are used.

    Returns
    -------
    tuple[Any, ...]
        Single sliced scan leaf.
    """

    replay_inputs = tuple(capture.input_values if inputs is None else inputs)
    if len(replay_inputs) != 1:
        raise ValueError("scan_read replay expects exactly one scanned input leaf.")
    return (_slice_scan_leaf(replay_inputs[0], int(capture.params["index"])),)


def _replay_scan_stack(
    capture: JaxEquationCapture, inputs: Sequence[Any] | None = None
) -> tuple[Any, ...]:
    """Replay a synthetic scan output stack.

    Parameters
    ----------
    capture
        Captured synthetic scan-stack equation.
    inputs
        Optional replacement per-iteration y leaves. When omitted, saved inputs
        are used.

    Returns
    -------
    tuple[Any, ...]
        Single stacked scan output leaf.
    """

    replay_inputs = tuple(capture.input_values if inputs is None else inputs)
    return (_stack_scan_outputs(replay_inputs),)


def _replay_bcf_stub(
    capture: JaxEquationCapture, _inputs: Sequence[Any] | None = None
) -> tuple[Any, ...]:
    """Raise for control-flow replay kinds reserved for B-CF.

    Parameters
    ----------
    capture
        Captured synthetic control-flow equation.
    _inputs
        Optional replacement inputs, unused until B-CF implements the kind.

    Returns
    -------
    tuple[Any, ...]
        Never returned.

    Raises
    ------
    NotImplementedError
        Always raised for B-CF-reserved replay kinds.
    """

    raise NotImplementedError(f"kind {capture.kind} lands in B-CF")


JAX_REPLAY_HANDLERS: Mapping[JaxEquationKind, JaxReplayHandler] = {
    "primitive": _replay_primitive,
    "scan_read": _replay_scan_read,
    "scan_stack": _replay_scan_stack,
    "cond_decision": _replay_bcf_stub,
    "while_decision": _replay_bcf_stub,
}


def _slice_scan_leaf(value: Any, index: int) -> Any:
    """Return one leading-axis leaf from a scanned input.

    Parameters
    ----------
    value
        Scanned input leaf.
    index
        Leading-axis index to read.

    Returns
    -------
    Any
        Sliced scan input leaf.
    """

    return value[index]


def _stack_scan_outputs(values: Sequence[Any]) -> Any:
    """Stack per-iteration scan y leaves on the leading axis.

    Parameters
    ----------
    values
        Per-iteration scan output leaves in final scan output order.

    Returns
    -------
    Any
        Leading-axis stacked output leaf.
    """

    import jax.numpy as jnp

    return jnp.stack(tuple(values), axis=0)


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

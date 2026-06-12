"""S0.J-slim round 1 probes for JAX jaxpr-first capture feasibility."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax._src import core


ArrayTree = Any
StaticArgnums = tuple[int, ...]


@dataclass(frozen=True)
class DeclaredCall:
    """Declared raw-JAX function contract for this spike.

    Parameters
    ----------
    name
        Human-readable case name.
    fn
        Callable shaped as ``fn(params, *inputs)``.
    params
        Parameter pytree passed as the first dynamic argument unless index 0 is static.
    inputs
        Dynamic input pytrees.
    static_argnums
        Positional argument indices, including params at index 0, treated as declared statics.
    """

    name: str
    fn: Callable[..., ArrayTree]
    params: ArrayTree
    inputs: tuple[ArrayTree, ...]
    static_argnums: StaticArgnums = ()


@dataclass(frozen=True)
class EquationCapture:
    """Captured inputs and outputs for one jaxpr equation."""

    index: int
    primitive: str
    input_values: tuple[Any, ...]
    output_values: tuple[Any, ...]
    params: Mapping[str, Any]


@dataclass(frozen=True)
class CaptureResult:
    """Concrete interpretation result for a closed jaxpr."""

    outputs: tuple[Any, ...]
    equations: tuple[EquationCapture, ...]


@dataclass(frozen=True)
class ScanResult:
    """Unsupported jaxpr features found before interpretation."""

    nested_primitives: tuple[str, ...]
    effect_primitives: tuple[str, ...]
    effects: tuple[str, ...]


def tree_allclose(left: ArrayTree, right: ArrayTree, atol: float = 1e-5) -> bool:
    """Return whether two pytrees have numerically equal leaves.

    Parameters
    ----------
    left
        First pytree.
    right
        Second pytree.
    atol
        Absolute tolerance.

    Returns
    -------
    bool
        True when all leaves are close.
    """

    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.allclose(np.asarray(a), np.asarray(b), atol=atol)
        for a, b in zip(left_leaves, right_leaves)
    )


def block_until_ready(tree: ArrayTree) -> ArrayTree:
    """Synchronize any JAX arrays in a pytree.

    Parameters
    ----------
    tree
        Pytree with possible JAX array leaves.

    Returns
    -------
    ArrayTree
        The same pytree after readiness barriers on array leaves.
    """

    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, tree
    )


def derive_closed_jaxpr(call: DeclaredCall) -> core.ClosedJaxpr:
    """Derive the closed jaxpr for a declared call.

    Parameters
    ----------
    call
        Declared call contract.

    Returns
    -------
    core.ClosedJaxpr
        Closed jaxpr produced by ``jax.make_jaxpr``.
    """

    return jax.make_jaxpr(call.fn, static_argnums=call.static_argnums)(call.params, *call.inputs)


def flatten_dynamic_args(call: DeclaredCall) -> tuple[Any, ...]:
    """Flatten non-static declared-call arguments into jaxpr input order.

    Parameters
    ----------
    call
        Declared call contract.

    Returns
    -------
    tuple[Any, ...]
        Dynamic leaves in the order expected by ``ClosedJaxpr.jaxpr.invars``.
    """

    args = (call.params, *call.inputs)
    flat_args: list[Any] = []
    for index, arg in enumerate(args):
        if index not in call.static_argnums:
            flat_args.extend(jax.tree.leaves(arg))
    return tuple(flat_args)


def summarize_closed_jaxpr(closed_jaxpr: core.ClosedJaxpr) -> dict[str, Any]:
    """Summarize consts, inputs, outputs, and equations for the findings file.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr to summarize.

    Returns
    -------
    dict[str, Any]
        JSON-serializable jaxpr inventory.
    """

    return {
        "const_count": len(closed_jaxpr.consts),
        "const_shapes": [str(getattr(const, "shape", None)) for const in closed_jaxpr.consts],
        "invar_count": len(closed_jaxpr.jaxpr.invars),
        "outvar_count": len(closed_jaxpr.jaxpr.outvars),
        "equation_count": len(closed_jaxpr.jaxpr.eqns),
        "primitives": [eqn.primitive.name for eqn in closed_jaxpr.jaxpr.eqns],
    }


def _safe_param_values(value: Any) -> Iterable[Any]:
    """Yield nested values from primitive params for recursive scanning.

    Parameters
    ----------
    value
        Primitive parameter value.

    Yields
    ------
    Any
        Nested values.
    """

    if isinstance(value, Mapping):
        yield from value.values()
    elif isinstance(value, tuple | list):
        yield from value
    else:
        yield value


def _contains_nested_jaxpr(value: Any) -> bool:
    """Return whether a primitive param contains a nested jaxpr.

    Parameters
    ----------
    value
        Primitive parameter value.

    Returns
    -------
    bool
        True when a nested jaxpr-like object is present.
    """

    if isinstance(value, core.ClosedJaxpr | core.Jaxpr):
        return True
    if hasattr(value, "jaxpr") and isinstance(getattr(value, "jaxpr"), core.Jaxpr):
        return True
    return any(
        _contains_nested_jaxpr(child) for child in _safe_param_values(value) if child is not value
    )


def scan_unsupported(closed_jaxpr: core.ClosedJaxpr) -> ScanResult:
    """Scan a closed jaxpr for nested jaxprs and effects.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr to scan.

    Returns
    -------
    ScanResult
        Unsupported nested/effect inventory.
    """

    effect_primitive_names = {"pure_callback", "io_callback", "debug_callback"}
    nested: list[str] = []
    effect_primitives: list[str] = []
    effects = [str(effect) for effect in closed_jaxpr.effects]
    for eqn in closed_jaxpr.jaxpr.eqns:
        if any(_contains_nested_jaxpr(value) for value in eqn.params.values()):
            nested.append(eqn.primitive.name)
        if eqn.effects or eqn.primitive.name in effect_primitive_names:
            effect_primitives.append(eqn.primitive.name)
            effects.extend(str(effect) for effect in eqn.effects)
    return ScanResult(tuple(nested), tuple(effect_primitives), tuple(sorted(set(effects))))


def _read_env(env: Mapping[core.Var, Any], atom: core.Atom) -> Any:
    """Read a jaxpr atom from the interpreter environment.

    Parameters
    ----------
    env
        Variable environment.
    atom
        Jaxpr atom.

    Returns
    -------
    Any
        Concrete value for the atom.
    """

    if isinstance(atom, core.Literal):
        return atom.val
    return env[atom]


def _write_env(env: dict[core.Var, Any], var: core.Var, value: Any) -> None:
    """Write a jaxpr variable into the interpreter environment.

    Parameters
    ----------
    env
        Mutable variable environment.
    var
        Jaxpr variable.
    value
        Concrete value.
    """

    if not isinstance(var, core.DropVar):
        env[var] = value


def interpret_closed_jaxpr(
    closed_jaxpr: core.ClosedJaxpr, flat_args: Sequence[Any]
) -> CaptureResult:
    """Interpret a flat closed jaxpr and capture each equation output.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr without unsupported nested jaxprs or effects.
    flat_args
        Dynamic argument leaves in jaxpr order.

    Returns
    -------
    CaptureResult
        Final outputs and per-equation captures.

    Raises
    ------
    ValueError
        If unsupported features are present.
    """

    scan = scan_unsupported(closed_jaxpr)
    if scan.nested_primitives or scan.effect_primitives or scan.effects:
        raise ValueError(f"unsupported jaxpr features: {scan}")
    env: dict[core.Var, Any] = {}
    for var, const in zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts):
        _write_env(env, var, const)
    for var, arg in zip(closed_jaxpr.jaxpr.invars, flat_args):
        _write_env(env, var, arg)

    captures: list[EquationCapture] = []
    for index, eqn in enumerate(closed_jaxpr.jaxpr.eqns):
        inputs = tuple(_read_env(env, var) for var in eqn.invars)
        result = eqn.primitive.bind(*inputs, **eqn.params)
        outputs = result if eqn.primitive.multiple_results else (result,)
        outputs = tuple(outputs)
        for var, value in zip(eqn.outvars, outputs):
            _write_env(env, var, value)
        captures.append(EquationCapture(index, eqn.primitive.name, inputs, outputs, eqn.params))
    outputs = tuple(_read_env(env, var) for var in closed_jaxpr.jaxpr.outvars)
    return CaptureResult(outputs, tuple(captures))


def replay_equation_from_jaxpr(
    closed_jaxpr: core.ClosedJaxpr,
    capture: EquationCapture,
    inputs: Sequence[Any] | None = None,
) -> tuple[Any, ...]:
    """Replay a captured equation using its primitive object from a closed jaxpr.

    Parameters
    ----------
    closed_jaxpr
        Source closed jaxpr containing the captured equation.
    capture
        Captured equation.
    inputs
        Optional replacement inputs.

    Returns
    -------
    tuple[Any, ...]
        Replayed primitive outputs.
    """

    eqn = closed_jaxpr.jaxpr.eqns[capture.index]
    replay_inputs = tuple(capture.input_values if inputs is None else inputs)
    result = eqn.primitive.bind(*replay_inputs, **eqn.params)
    return tuple(result if eqn.primitive.multiple_results else (result,))


def perturb_value(value: Any) -> Any:
    """Perturb a concrete parent value while preserving shape and dtype.

    Parameters
    ----------
    value
        Concrete value.

    Returns
    -------
    Any
        Perturbed value.
    """

    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        return array + jnp.ones_like(array)
    return array + jnp.full_like(array, 0.125)


def find_replay_candidate(captures: Sequence[EquationCapture]) -> EquationCapture:
    """Find a single-output, nontrivial equation suitable for replay validation.

    Parameters
    ----------
    captures
        Equation captures.

    Returns
    -------
    EquationCapture
        Candidate capture.

    Raises
    ------
    AssertionError
        If no candidate exists.
    """

    for capture in captures:
        if len(capture.input_values) >= 2 and len(capture.output_values) == 1:
            first = np.asarray(capture.input_values[0])
            second = np.asarray(capture.input_values[1])
            if first.shape == second.shape and first.dtype == second.dtype:
                return capture
    raise AssertionError("no replay candidate with same-shaped parents")


def probe_declared_call_and_capture() -> dict[str, Any]:
    """Probe closed-jaxpr derivation, interpretation, replay, and perturbation.

    Returns
    -------
    dict[str, Any]
        Probe evidence.
    """

    params = {
        "w": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10.0,
        "b": jnp.tile(jnp.linspace(-0.2, 0.2, 4, dtype=jnp.float32), (2, 1)),
    }

    def declared_fn(params: Mapping[str, Any], x: Any, scale: float) -> Any:
        """Declared-call fixture with one closed-over const and one static."""

        hidden_bias = jnp.asarray(
            [[0.5, -0.25, 0.125, 0.0], [0.25, 0.0, -0.125, -0.5]],
            dtype=jnp.float32,
        )
        hidden = jnp.dot(x, params["w"]) + params["b"]
        return jnp.tanh(hidden + hidden_bias) * scale

    call = DeclaredCall(
        "declared_call_static_and_const",
        declared_fn,
        params,
        (jnp.ones((2, 3), dtype=jnp.float32), 2.0),
        static_argnums=(2,),
    )
    closed_jaxpr = derive_closed_jaxpr(call)
    flat_args = flatten_dynamic_args(call)
    interpreted = interpret_closed_jaxpr(closed_jaxpr, flat_args)
    direct = block_until_ready(call.fn(call.params, *call.inputs))
    interpreted_out = block_until_ready(interpreted.outputs[0])
    assert tree_allclose(interpreted_out, direct)
    assert len(closed_jaxpr.consts) == 1
    assert len(interpreted.equations) == len(closed_jaxpr.jaxpr.eqns)

    candidate = find_replay_candidate(interpreted.equations)
    replayed = replay_equation_from_jaxpr(closed_jaxpr, candidate)
    assert tree_allclose(replayed, candidate.output_values)

    perturbed_inputs = (perturb_value(candidate.input_values[0]), *candidate.input_values[1:])
    perturbed = replay_equation_from_jaxpr(closed_jaxpr, candidate, perturbed_inputs)
    assert not tree_allclose(perturbed, candidate.output_values)

    wrong_inputs = (
        candidate.input_values[1],
        candidate.input_values[1],
        *candidate.input_values[2:],
    )
    wrong = replay_equation_from_jaxpr(closed_jaxpr, candidate, wrong_inputs)
    assert not tree_allclose(wrong, candidate.output_values)

    return {
        "summary": summarize_closed_jaxpr(closed_jaxpr),
        "captured_equations": len(interpreted.equations),
        "replay_candidate": {
            "index": candidate.index,
            "primitive": candidate.primitive,
            "input_shapes": [
                str(getattr(value, "shape", None)) for value in candidate.input_values
            ],
        },
        "wrong_parent_wiring_failed": True,
        "parent_perturbation_detected": True,
    }


def make_nested_probe_cases() -> dict[str, Callable[[Any], Any]]:
    """Build nested-jaxpr rejection fixtures.

    Returns
    -------
    dict[str, Callable[[Any], Any]]
        Probe names mapped to functions.
    """

    @jax.custom_jvp
    def custom_jvp_fn(x: Any) -> Any:
        """Custom JVP fixture."""

        return x * x

    @custom_jvp_fn.defjvp
    def custom_jvp_rule(primals: tuple[Any], tangents: tuple[Any]) -> tuple[Any, Any]:
        """Custom JVP rule fixture."""

        (x,) = primals
        (t,) = tangents
        return custom_jvp_fn(x), 2.0 * x * t

    @jax.custom_vjp
    def custom_vjp_fn(x: Any) -> Any:
        """Custom VJP fixture."""

        return x * x

    def custom_vjp_fwd(x: Any) -> tuple[Any, Any]:
        """Custom VJP forward rule fixture."""

        return x * x, x

    def custom_vjp_bwd(saved: Any, g: Any) -> tuple[Any]:
        """Custom VJP backward rule fixture."""

        return (2.0 * saved * g,)

    custom_vjp_fn.defvjp(custom_vjp_fwd, custom_vjp_bwd)

    def scan_fn(x: Any) -> Any:
        """Scan fixture."""

        def body(carry: Any, item: Any) -> tuple[Any, Any]:
            """Scan body."""

            new_carry = carry + item
            return new_carry, new_carry * item

        return lax.scan(body, x[0], x)[1]

    def cond_fn(x: Any) -> Any:
        """Cond fixture."""

        return lax.cond(x[0] > 0.0, lambda y: y + 1.0, lambda y: y - 1.0, x)

    def while_fn(x: Any) -> Any:
        """While fixture."""

        def cond_fun(state: tuple[Any, Any]) -> Any:
            """While condition."""

            i, _ = state
            return i < 2

        def body_fun(state: tuple[Any, Any]) -> tuple[Any, Any]:
            """While body."""

            i, value = state
            return i + 1, value + x

        return lax.while_loop(cond_fun, body_fun, (0, x))[1]

    def remat_fn(x: Any) -> Any:
        """Remat fixture."""

        return jax.checkpoint(lambda y: jnp.sin(y) + 1.0)(x)

    jitted_inner = jax.jit(lambda y: jnp.cos(y) + 1.0)

    def pjit_fn(x: Any) -> Any:
        """Jit/pjit-like nested call fixture."""

        return jitted_inner(x)

    return {
        "scan": scan_fn,
        "cond": cond_fn,
        "while": while_fn,
        "remat": remat_fn,
        "pjit": pjit_fn,
        "custom_jvp": custom_jvp_fn,
        "custom_vjp": custom_vjp_fn,
    }


def probe_nested_rejections() -> dict[str, Any]:
    """Probe nested-jaxpr detection and rejection.

    Returns
    -------
    dict[str, Any]
        Rejection evidence by fixture.
    """

    results: dict[str, Any] = {}
    for name, fn in make_nested_probe_cases().items():
        closed_jaxpr = jax.make_jaxpr(fn)(jnp.ones((3,), dtype=jnp.float32))
        scan = scan_unsupported(closed_jaxpr)
        rejected = bool(scan.nested_primitives)
        if name in {"custom_jvp", "custom_vjp"} and not rejected:
            rejected = any(
                primitive.startswith("custom_")
                for primitive in summarize_closed_jaxpr(closed_jaxpr)["primitives"]
            )
        assert rejected, f"{name} did not expose a nested/custom primitive"
        try:
            interpret_closed_jaxpr(
                closed_jaxpr, jax.tree.leaves((jnp.ones((3,), dtype=jnp.float32),))
            )
        except ValueError:
            interpreter_rejected = True
        else:
            interpreter_rejected = False
        assert interpreter_rejected or name in {"custom_jvp", "custom_vjp"}
        results[name] = {
            "nested_primitives": scan.nested_primitives,
            "all_primitives": summarize_closed_jaxpr(closed_jaxpr)["primitives"],
            "interpreter_rejected": interpreter_rejected,
        }
    return results


def probe_effect_scanning() -> dict[str, Any]:
    """Probe callback/effect detection.

    Returns
    -------
    dict[str, Any]
        Effect detection evidence.
    """

    def pure_callback_fn(x: Any) -> Any:
        """Pure callback fixture."""

        return jax.pure_callback(lambda y: y, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    def io_callback_fn(x: Any) -> Any:
        """IO callback fixture."""

        return jax.experimental.io_callback(lambda y: y, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    def debug_callback_fn(x: Any) -> Any:
        """Debug callback fixture."""

        jax.debug.callback(lambda y: None, x)
        return x + 1.0

    results: dict[str, Any] = {}
    for name, fn in {
        "pure_callback": pure_callback_fn,
        "io_callback": io_callback_fn,
        "debug_callback": debug_callback_fn,
    }.items():
        closed_jaxpr = jax.make_jaxpr(fn)(jnp.ones((2,), dtype=jnp.float32))
        scan = scan_unsupported(closed_jaxpr)
        assert scan.effect_primitives or scan.effects, f"{name} was not detected"
        try:
            interpret_closed_jaxpr(
                closed_jaxpr, jax.tree.leaves((jnp.ones((2,), dtype=jnp.float32),))
            )
        except ValueError:
            rejected = True
        else:
            rejected = False
        assert rejected
        results[name] = {
            "effect_primitives": scan.effect_primitives,
            "effects": scan.effects,
            "primitives": summarize_closed_jaxpr(closed_jaxpr)["primitives"],
        }
    return results


def build_corpus() -> tuple[DeclaredCall, ...]:
    """Build representative raw-JAX corpus functions for accepted capture measurement.

    Returns
    -------
    tuple[DeclaredCall, ...]
        Corpus cases.
    """

    key = random.key(0)
    legacy_key = random.PRNGKey(1)
    mlp_params = {
        "w1": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 7.0,
        "b1": jnp.zeros((4,), dtype=jnp.float32),
        "w2": jnp.arange(8, dtype=jnp.float32).reshape(4, 2) / 9.0,
        "b2": jnp.ones((2,), dtype=jnp.float32) * 0.1,
    }
    conv_params = {"kernel": jnp.ones((3, 3, 1, 2), dtype=jnp.float32) / 9.0}
    attn_params = {
        "wq": jnp.eye(4, dtype=jnp.float32),
        "wk": jnp.eye(4, dtype=jnp.float32) * 0.5,
        "wv": jnp.eye(4, dtype=jnp.float32) * 0.25,
    }

    def mlp(params: Mapping[str, Any], x: Any) -> Any:
        """Two-layer MLP."""

        hidden = jnp.maximum(jnp.dot(x, params["w1"]) + params["b1"], 0.0)
        return jnp.dot(hidden, params["w2"]) + params["b2"]

    def cnn(params: Mapping[str, Any], x: Any) -> Any:
        """Small NHWC convolution block."""

        y = lax.conv_general_dilated(
            x, params["kernel"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        return jax.nn.relu(y).mean(axis=(1, 2))

    def attention(params: Mapping[str, Any], x: Any) -> Any:
        """Single-head attention block."""

        q = x @ params["wq"]
        k = x @ params["wk"]
        v = x @ params["wv"]
        weights = jax.nn.softmax(
            q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(jnp.asarray(4.0, dtype=jnp.float32)), axis=-1
        )
        return weights @ v

    def operator_heavy(_: None, x: Any) -> Any:
        """Operator-heavy spelling fixture."""

        return ((x + 2.0) * (x - 0.5) / 3.0) ** 2

    def method_spellings(_: None, x: Any) -> Any:
        """Array method spelling fixture."""

        return x.reshape((2, 3)).T.astype(jnp.float32).sum(axis=0)

    def stochastic_typed(_: None, key_arg: Any, x: Any) -> Any:
        """Stochastic fixture with explicit typed key."""

        return x + random.normal(key_arg, x.shape)

    def stochastic_legacy(_: None, key_arg: Any, x: Any) -> Any:
        """Stochastic fixture with explicit legacy key."""

        return x + random.uniform(key_arg, x.shape)

    def pytree_multi(_: None, x: Any) -> Mapping[str, Any]:
        """Pytree multi-output fixture."""

        return {"sum": x.sum(axis=0), "parts": (x[:, :2], x[:, 2:])}

    def reductions(_: None, x: Any) -> Any:
        """Reduction-heavy fixture."""

        return jnp.stack([x.mean(), x.max(), x.min()])

    def broadcasting(_: None, x: Any) -> Any:
        """Broadcasting fixture."""

        return x + jnp.arange(x.shape[-1], dtype=x.dtype)

    def slicing(_: None, x: Any) -> Any:
        """Slicing and concatenate fixture."""

        return jnp.concatenate([x[:, :2], x[:, -2:]], axis=1)

    def einsum_case(params: Mapping[str, Any], x: Any) -> Any:
        """Einsum fixture."""

        return jnp.einsum("bi,ij->bj", x, params["w1"])

    def nn_funcs(_: None, x: Any) -> Any:
        """JAX NN function fixture."""

        return jax.nn.gelu(x) + jax.nn.sigmoid(x)

    def dtype_cast(_: None, x: Any) -> Any:
        """Dtype conversion fixture."""

        return x.astype(jnp.float32).astype(jnp.bfloat16).astype(jnp.float32)

    def static_scale(_: None, x: Any, scale: float) -> Any:
        """Declared static argument fixture."""

        return x * scale + 1.0

    def const_capture(_: None, x: Any) -> Any:
        """Closed-over const fixture."""

        const = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
        return x + const

    return (
        DeclaredCall("mlp", mlp, mlp_params, (jnp.ones((2, 3), dtype=jnp.float32),)),
        DeclaredCall("cnn", cnn, conv_params, (jnp.ones((1, 5, 5, 1), dtype=jnp.float32),)),
        DeclaredCall(
            "attention", attention, attn_params, (jnp.ones((2, 3, 4), dtype=jnp.float32),)
        ),
        DeclaredCall(
            "operator_heavy", operator_heavy, None, (jnp.ones((2, 3), dtype=jnp.float32),)
        ),
        DeclaredCall(
            "method_spellings", method_spellings, None, (jnp.arange(6, dtype=jnp.float32),)
        ),
        DeclaredCall(
            "stochastic_typed_key",
            stochastic_typed,
            None,
            (key, jnp.ones((2, 3), dtype=jnp.float32)),
        ),
        DeclaredCall(
            "stochastic_legacy_key",
            stochastic_legacy,
            None,
            (legacy_key, jnp.ones((2, 3), dtype=jnp.float32)),
        ),
        DeclaredCall(
            "pytree_multi_output", pytree_multi, None, (jnp.ones((2, 4), dtype=jnp.float32),)
        ),
        DeclaredCall(
            "reductions", reductions, None, (jnp.arange(6, dtype=jnp.float32).reshape(2, 3),)
        ),
        DeclaredCall("broadcasting", broadcasting, None, (jnp.ones((2, 3), dtype=jnp.float32),)),
        DeclaredCall("slicing", slicing, None, (jnp.arange(10, dtype=jnp.float32).reshape(2, 5),)),
        DeclaredCall("einsum", einsum_case, mlp_params, (jnp.ones((2, 3), dtype=jnp.float32),)),
        DeclaredCall("nn_funcs", nn_funcs, None, (jnp.ones((2, 3), dtype=jnp.float32),)),
        DeclaredCall("dtype_cast", dtype_cast, None, (jnp.ones((2, 3), dtype=jnp.float32),)),
        DeclaredCall(
            "static_scale",
            static_scale,
            None,
            (jnp.ones((2, 3), dtype=jnp.float32), 3.0),
            static_argnums=(2,),
        ),
        DeclaredCall("const_capture", const_capture, None, (jnp.ones((3,), dtype=jnp.float32),)),
    )


def measure_corpus() -> dict[str, Any]:
    """Measure useful accepted capture rate over the raw-JAX corpus.

    Returns
    -------
    dict[str, Any]
        Corpus acceptance evidence.
    """

    case_results: list[dict[str, Any]] = []
    accepted = 0
    for call in build_corpus():
        try:
            closed_jaxpr = derive_closed_jaxpr(call)
            flat_args = flatten_dynamic_args(call)
            interpreted = interpret_closed_jaxpr(closed_jaxpr, flat_args)
            direct = block_until_ready(call.fn(call.params, *call.inputs))
            interpreted_outputs = block_until_ready(
                jax.tree.unflatten(jax.tree.structure(direct), interpreted.outputs)
            )
            output_match = tree_allclose(interpreted_outputs, direct)
            accepted_case = output_match
            assert output_match
        except Exception as exc:  # noqa: BLE001 - spike records rejection cause.
            case_results.append(
                {"name": call.name, "accepted": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        accepted += 1
        case_results.append(
            {
                "name": call.name,
                "accepted": accepted_case,
                "equations": len(interpreted.equations),
                "consts": len(closed_jaxpr.consts),
                "primitives": summarize_closed_jaxpr(closed_jaxpr)["primitives"],
            }
        )
    total = len(case_results)
    return {
        "accepted": accepted,
        "total": total,
        "accepted_capture_rate": accepted / total,
        "cases": case_results,
    }


def time_call(fn: Callable[[], Any], repeats: int) -> list[float]:
    """Time repeated synchronized calls.

    Parameters
    ----------
    fn
        Zero-argument callable.
    repeats
        Number of repeats.

    Returns
    -------
    list[float]
        Elapsed seconds per call.
    """

    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        block_until_ready(fn())
        durations.append(time.perf_counter() - start)
    return durations


def measure_overhead() -> dict[str, Any]:
    """Measure interpreted-vs-direct overhead on toy and MLP-scale cases.

    Returns
    -------
    dict[str, Any]
        Timing evidence.
    """

    toy_call = DeclaredCall(
        "toy",
        lambda _params, x: jnp.sin(x) + jnp.cos(x * 2.0),
        None,
        (jnp.ones((8,), dtype=jnp.float32),),
    )
    mlp_call = next(call for call in build_corpus() if call.name == "mlp")
    results: dict[str, Any] = {}
    for call, repeats in ((toy_call, 200), (mlp_call, 100)):
        closed_jaxpr = derive_closed_jaxpr(call)
        flat_args = flatten_dynamic_args(call)
        interpret_closed_jaxpr(closed_jaxpr, flat_args)
        block_until_ready(call.fn(call.params, *call.inputs))
        direct_times = time_call(lambda c=call: c.fn(c.params, *c.inputs), repeats)
        interpreted_times = time_call(
            lambda cj=closed_jaxpr, args=flat_args: interpret_closed_jaxpr(cj, args).outputs,
            repeats,
        )
        direct_median = statistics.median(direct_times)
        interpreted_median = statistics.median(interpreted_times)
        results[call.name] = {
            "repeats": repeats,
            "equations": len(closed_jaxpr.jaxpr.eqns),
            "direct_median_s": direct_median,
            "interpreted_median_s": interpreted_median,
            "overhead_x": interpreted_median / direct_median,
        }
    return results


def run_all() -> dict[str, Any]:
    """Run all S0.J round 1 probes.

    Returns
    -------
    dict[str, Any]
        Full probe report.
    """

    return {
        "versions": {
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "numpy": np.__version__,
            "python": ".".join(str(part) for part in __import__("sys").version_info[:3]),
        },
        "declared_call_capture": probe_declared_call_and_capture(),
        "nested_rejections": probe_nested_rejections(),
        "effect_scanning": probe_effect_scanning(),
        "corpus": measure_corpus(),
        "overhead": measure_overhead(),
    }


def main() -> None:
    """Run probes and print JSON evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args()
    report = run_all()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

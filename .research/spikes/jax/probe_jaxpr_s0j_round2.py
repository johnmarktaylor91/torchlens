"""S0.J-slim round 2 probes for safe pure-call jaxpr inlining."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax._src import core

from probe_jaxpr_s0j_round1 import (
    ArrayTree,
    DeclaredCall,
    block_until_ready,
    build_corpus,
    derive_closed_jaxpr,
    flatten_dynamic_args,
    perturb_value,
    scan_unsupported,
    tree_allclose,
)


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


@dataclass(frozen=True)
class InlinedEquationCapture:
    """Captured primitive equation after safe call-primitive inlining.

    Parameters
    ----------
    index
        Sequential capture index in flattened graph-construction order.
    primitive
        Primitive name.
    primitive_obj
        JAX primitive object used for replay.
    input_values
        Concrete values read from parent variables.
    output_values
        Concrete primitive outputs.
    params
        Primitive params passed to ``bind`` during replay.
    source_path
        Nested source path through outer and inlined equations.
    inlined
        Whether the equation came from inside an accepted call primitive.
    """

    index: int
    primitive: str
    primitive_obj: Any
    input_values: tuple[Any, ...]
    output_values: tuple[Any, ...]
    params: Mapping[str, Any]
    source_path: tuple[str, ...]
    inlined: bool


@dataclass(frozen=True)
class InlinedCaptureResult:
    """Concrete result for an interpreted jaxpr with safe inlining."""

    outputs: tuple[Any, ...]
    equations: tuple[InlinedEquationCapture, ...]
    inlined_call_primitives: tuple[str, ...]


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
        Concrete atom value.
    """

    if isinstance(atom, core.Literal):
        return atom.val
    return env[atom]


def _write_env(env: dict[core.Var, Any], var: core.Var, value: Any) -> None:
    """Write a jaxpr variable unless it is a dropped output.

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


def _closed_jaxpr_param(eqn: core.JaxprEqn) -> core.ClosedJaxpr | None:
    """Return a nested closed jaxpr parameter for supported call primitives.

    Parameters
    ----------
    eqn
        Jaxpr equation.

    Returns
    -------
    core.ClosedJaxpr | None
        Nested closed jaxpr when the equation is a recognized call primitive.
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


def _has_nested_jaxpr(eqn: core.JaxprEqn) -> bool:
    """Return whether an equation contains a nested jaxpr parameter.

    Parameters
    ----------
    eqn
        Jaxpr equation.

    Returns
    -------
    bool
        True when any parameter contains a nested jaxpr.
    """

    def contains(value: Any) -> bool:
        """Recursively inspect a value for jaxpr-like objects."""

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


def _is_library_custom_jvp_call(eqn: core.JaxprEqn) -> bool:
    """Return whether a ``custom_jvp_call`` is a recognized library wrapper.

    Parameters
    ----------
    eqn
        Candidate custom JVP equation.

    Returns
    -------
    bool
        True for the round-2 allowlisted library-internal pattern.
    """

    nested = _closed_jaxpr_param(eqn)
    if eqn.primitive.name != "custom_jvp_call" or nested is None:
        return False
    nested_eqns = nested.jaxpr.eqns
    return bool(nested_eqns) and all(
        child.primitive.name == "jit" and child.params.get("name") in SAFE_JIT_NAMES
        for child in nested_eqns
    )


def _can_inline_call(eqn: core.JaxprEqn) -> bool:
    """Return whether an equation is safe to inline in this spike.

    Parameters
    ----------
    eqn
        Candidate call equation.

    Returns
    -------
    bool
        True when the equation matches a pure library-internal allowlist.
    """

    if eqn.effects:
        return False
    if eqn.primitive.name == "jit":
        return eqn.params.get("name") in SAFE_JIT_NAMES
    if eqn.primitive.name == "custom_jvp_call":
        return _is_library_custom_jvp_call(eqn)
    return False


def _assert_no_rejected_effects(closed_jaxpr: core.ClosedJaxpr) -> None:
    """Reject effects before inlined interpretation.

    Parameters
    ----------
    closed_jaxpr
        Jaxpr to scan.

    Raises
    ------
    ValueError
        If effects or callback primitives are present.
    """

    scan = scan_unsupported(closed_jaxpr)
    if scan.effect_primitives or scan.effects:
        raise ValueError(f"unsupported jaxpr effects: {scan}")


def interpret_closed_jaxpr_with_inlining(
    closed_jaxpr: core.ClosedJaxpr, flat_args: Sequence[Any]
) -> InlinedCaptureResult:
    """Interpret a closed jaxpr while inlining safe pure call primitives.

    Parameters
    ----------
    closed_jaxpr
        Closed jaxpr to interpret.
    flat_args
        Dynamic argument leaves in jaxpr input order.

    Returns
    -------
    InlinedCaptureResult
        Final outputs and flattened equation captures.
    """

    _assert_no_rejected_effects(closed_jaxpr)
    captures: list[InlinedEquationCapture] = []
    inlined_calls: list[str] = []

    def interpret_inner(
        inner: core.ClosedJaxpr,
        args: Sequence[Any],
        path: tuple[str, ...],
        inlined_depth: int,
    ) -> tuple[Any, ...]:
        """Interpret one closed jaxpr frame and append captures."""

        env: dict[core.Var, Any] = {}
        for var, const in zip(inner.jaxpr.constvars, inner.consts):
            _write_env(env, var, const)
        for var, arg in zip(inner.jaxpr.invars, args):
            _write_env(env, var, arg)

        for eqn_index, eqn in enumerate(inner.jaxpr.eqns):
            primitive_name = eqn.primitive.name
            eqn_path = (*path, f"{eqn_index}:{primitive_name}")
            if primitive_name in REJECTED_NESTED_PRIMITIVES:
                raise ValueError(f"unsupported nested primitive: {primitive_name}")
            if eqn.effects:
                raise ValueError(f"unsupported equation effects: {primitive_name} {eqn.effects}")
            nested = _closed_jaxpr_param(eqn)
            inputs = tuple(_read_env(env, var) for var in eqn.invars)
            if nested is not None:
                if not _can_inline_call(eqn):
                    raise ValueError(
                        f"unsupported nested call primitive: {primitive_name} "
                        f"name={eqn.params.get('name')!r}"
                    )
                inlined_calls.append(primitive_name)
                outputs = interpret_inner(nested, inputs, eqn_path, inlined_depth + 1)
                for var, value in zip(eqn.outvars, outputs):
                    _write_env(env, var, value)
                continue
            if _has_nested_jaxpr(eqn):
                raise ValueError(f"unsupported nested jaxpr in primitive: {primitive_name}")
            result = eqn.primitive.bind(*inputs, **eqn.params)
            outputs = tuple(result if eqn.primitive.multiple_results else (result,))
            for var, value in zip(eqn.outvars, outputs):
                _write_env(env, var, value)
            captures.append(
                InlinedEquationCapture(
                    index=len(captures),
                    primitive=primitive_name,
                    primitive_obj=eqn.primitive,
                    input_values=inputs,
                    output_values=outputs,
                    params=eqn.params,
                    source_path=eqn_path,
                    inlined=inlined_depth > 0,
                )
            )
        return tuple(_read_env(env, var) for var in inner.jaxpr.outvars)

    outputs = interpret_inner(closed_jaxpr, flat_args, ("root",), 0)
    return InlinedCaptureResult(outputs, tuple(captures), tuple(inlined_calls))


def replay_inlined_equation(
    capture: InlinedEquationCapture, inputs: Sequence[Any] | None = None
) -> tuple[Any, ...]:
    """Replay a captured primitive equation.

    Parameters
    ----------
    capture
        Captured primitive equation.
    inputs
        Optional replacement inputs.

    Returns
    -------
    tuple[Any, ...]
        Replayed primitive outputs.
    """

    replay_inputs = tuple(capture.input_values if inputs is None else inputs)
    result = capture.primitive_obj.bind(*replay_inputs, **capture.params)
    return tuple(result if capture.primitive_obj.multiple_results else (result,))


def _find_inlined_replay_candidate(
    captures: Sequence[InlinedEquationCapture],
) -> InlinedEquationCapture:
    """Find an inlined equation suitable for replay and wrong-wiring probes.

    Parameters
    ----------
    captures
        Captured equations.

    Returns
    -------
    InlinedEquationCapture
        Candidate inlined equation.
    """

    for capture in captures:
        if capture.inlined and len(capture.input_values) >= 2 and len(capture.output_values) == 1:
            wrong_inputs = (
                capture.input_values[1],
                capture.input_values[1],
                *capture.input_values[2:],
            )
            try:
                wrong = replay_inlined_equation(capture, wrong_inputs)
            except Exception:  # noqa: BLE001 - incompatible wrong wiring is still a failure.
                return capture
            if not tree_allclose(wrong, capture.output_values):
                return capture
    raise AssertionError("no inlined replay candidate")


def probe_inlined_replay_and_perturbation() -> dict[str, Any]:
    """Prove replay and perturbation work on an inlined equation.

    Returns
    -------
    dict[str, Any]
        Replay evidence for an inlined primitive.
    """

    params = {"kernel": jnp.ones((3, 3, 1, 2), dtype=jnp.float32) / 9.0}

    def cnn(params: Mapping[str, Any], x: Any) -> Any:
        """Small convolution block with library custom-JVP ReLU."""

        y = lax.conv_general_dilated(
            x, params["kernel"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        return jax.nn.relu(y).mean(axis=(1, 2))

    call = DeclaredCall("cnn_replay", cnn, params, (jnp.ones((1, 5, 5, 1), dtype=jnp.float32),))
    closed_jaxpr = derive_closed_jaxpr(call)
    captured = interpret_closed_jaxpr_with_inlining(closed_jaxpr, flatten_dynamic_args(call))
    direct = block_until_ready(call.fn(call.params, *call.inputs))
    interpreted = block_until_ready(captured.outputs[0])
    assert tree_allclose(interpreted, direct)

    candidate = _find_inlined_replay_candidate(captured.equations)
    replayed = replay_inlined_equation(candidate)
    assert tree_allclose(replayed, candidate.output_values)

    perturbed_inputs = (perturb_value(candidate.input_values[0]), *candidate.input_values[1:])
    perturbed = replay_inlined_equation(candidate, perturbed_inputs)
    assert not tree_allclose(perturbed, candidate.output_values)

    wrong_inputs = (
        candidate.input_values[1],
        candidate.input_values[1],
        *candidate.input_values[2:],
    )
    try:
        wrong = replay_inlined_equation(candidate, wrong_inputs)
    except Exception as exc:  # noqa: BLE001 - wrong parent wiring may be shape-invalid.
        wrong_parent_failed = True
        wrong_parent_error = f"{type(exc).__name__}: {exc}"
    else:
        wrong_parent_failed = not tree_allclose(wrong, candidate.output_values)
        wrong_parent_error = ""
    assert wrong_parent_failed

    return {
        "outer_primitives": [eqn.primitive.name for eqn in closed_jaxpr.jaxpr.eqns],
        "inlined_call_primitives": captured.inlined_call_primitives,
        "captured_equations": len(captured.equations),
        "candidate": {
            "index": candidate.index,
            "primitive": candidate.primitive,
            "source_path": candidate.source_path,
            "input_shapes": [
                str(getattr(value, "shape", None)) for value in candidate.input_values
            ],
        },
        "parent_perturbation_detected": True,
        "wrong_parent_wiring_failed": wrong_parent_failed,
        "wrong_parent_error": wrong_parent_error,
    }


def build_extended_corpus() -> tuple[DeclaredCall, ...]:
    """Build the round-2 corpus including conv/random inlining targets.

    Returns
    -------
    tuple[DeclaredCall, ...]
        Representative raw-JAX corpus.
    """

    base_cases = list(build_corpus())
    key = random.key(42)
    legacy_key = random.PRNGKey(7)
    depthwise_kernel = jnp.ones((3, 3, 1, 2), dtype=jnp.float32) / 9.0
    point_kernel = jnp.ones((1, 1, 2, 3), dtype=jnp.float32) / 2.0
    layer_params = {
        "scale": jnp.ones((4,), dtype=jnp.float32),
        "bias": jnp.zeros((4,), dtype=jnp.float32),
    }
    nested_params = {
        "block": {
            "w": jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 11.0,
            "b": jnp.linspace(-0.1, 0.2, 4, dtype=jnp.float32),
        }
    }

    def depthwise_conv(params: Mapping[str, Any], x: Any) -> Any:
        """Depthwise convolution fixture."""

        return lax.conv_general_dilated(
            x,
            params["kernel"],
            (1, 1),
            "SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=2,
        )

    def pointwise_relu(params: Mapping[str, Any], x: Any) -> Any:
        """Pointwise convolution plus ReLU fixture."""

        y = lax.conv_general_dilated(
            x, params["kernel"], (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
        return jax.nn.relu(y)

    def dropout_like(_: None, key_arg: Any, x: Any) -> Any:
        """Explicit-key dropout-like fixture."""

        keep = random.bernoulli(key_arg, 0.75, x.shape)
        return jnp.where(keep, x / 0.75, 0.0)

    def randint_index(_: None, key_arg: Any, x: Any) -> Any:
        """Explicit-key integer random indexing fixture."""

        index = random.randint(key_arg, (), 0, x.shape[0])
        return x[index]

    def layer_norm(params: Mapping[str, Any], x: Any) -> Any:
        """Layer-normalization spelling fixture."""

        centered = x - x.mean(axis=-1, keepdims=True)
        variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
        return centered * lax.rsqrt(variance + 1e-5) * params["scale"] + params["bias"]

    def one_hot_take(_: None, x: Any) -> Any:
        """One-hot and take fixture."""

        indices = jnp.asarray([0, 2, 1], dtype=jnp.int32)
        return jax.nn.one_hot(indices, x.shape[-1]) @ x.T

    def nested_param_mlp(params: Mapping[str, Any], x: Any) -> Any:
        """Nested pytree parameter fixture."""

        return jnp.tanh(x @ params["block"]["w"] + params["block"]["b"])

    return (
        *base_cases,
        DeclaredCall(
            "depthwise_conv",
            depthwise_conv,
            {"kernel": depthwise_kernel},
            (jnp.ones((1, 4, 4, 2), dtype=jnp.float32),),
        ),
        DeclaredCall(
            "pointwise_conv_relu",
            pointwise_relu,
            {"kernel": point_kernel},
            (jnp.ones((1, 4, 4, 2), dtype=jnp.float32),),
        ),
        DeclaredCall(
            "dropout_like_explicit_key",
            dropout_like,
            None,
            (key, jnp.ones((2, 3), dtype=jnp.float32)),
        ),
        DeclaredCall(
            "randint_index_explicit_key",
            randint_index,
            None,
            (legacy_key, jnp.arange(12, dtype=jnp.float32).reshape(4, 3)),
        ),
        DeclaredCall(
            "layer_norm",
            layer_norm,
            layer_params,
            (jnp.arange(8, dtype=jnp.float32).reshape(2, 4),),
        ),
        DeclaredCall("one_hot_take", one_hot_take, None, (jnp.eye(4, dtype=jnp.float32),)),
        DeclaredCall(
            "nested_param_mlp",
            nested_param_mlp,
            nested_params,
            (jnp.ones((2, 4), dtype=jnp.float32),),
        ),
    )


def probe_rejection_boundaries() -> dict[str, Any]:
    """Assert control-flow and custom-VJP boundaries remain rejected.

    Returns
    -------
    dict[str, Any]
        Rejection evidence by case name.
    """

    @jax.custom_jvp
    def user_custom_jvp(x: Any) -> Any:
        """User custom-JVP fixture that should not match the library allowlist."""

        return x * x

    @user_custom_jvp.defjvp
    def user_custom_jvp_rule(primals: tuple[Any], tangents: tuple[Any]) -> tuple[Any, Any]:
        """User custom-JVP rule."""

        (x,) = primals
        (t,) = tangents
        return user_custom_jvp(x), 2.0 * x * t

    @jax.custom_vjp
    def user_custom_vjp(x: Any) -> Any:
        """User custom-VJP fixture."""

        return x * x

    def custom_vjp_fwd(x: Any) -> tuple[Any, Any]:
        """User custom-VJP forward rule."""

        return x * x, x

    def custom_vjp_bwd(saved: Any, g: Any) -> tuple[Any]:
        """User custom-VJP backward rule."""

        return (2.0 * saved * g,)

    user_custom_vjp.defvjp(custom_vjp_fwd, custom_vjp_bwd)

    def scan_fn(x: Any) -> Any:
        """Scan rejection fixture."""

        def body(carry: Any, item: Any) -> tuple[Any, Any]:
            """Scan body."""

            new_carry = carry + item
            return new_carry, new_carry

        return lax.scan(body, x[0], x)[1]

    def cond_fn(x: Any) -> Any:
        """Cond rejection fixture."""

        def true_fun(y: Any) -> Any:
            """True branch."""

            return y + 1

        def false_fun(y: Any) -> Any:
            """False branch."""

            return y - 1

        return lax.cond(x[0] > 0, true_fun, false_fun, x)

    def while_fn(x: Any) -> Any:
        """While-loop rejection fixture."""

        def cond_fun(state: tuple[Any, Any]) -> Any:
            """Loop condition."""

            i, _ = state
            return i < 2

        def body_fun(state: tuple[Any, Any]) -> tuple[Any, Any]:
            """Loop body."""

            i, y = state
            return i + 1, y + x

        return lax.while_loop(cond_fun, body_fun, (0, x))[1]

    cases: Mapping[str, Callable[[Any], Any]] = {
        "scan": scan_fn,
        "cond": cond_fn,
        "while": while_fn,
        "custom_jvp_user": user_custom_jvp,
        "custom_vjp": user_custom_vjp,
    }
    results: dict[str, Any] = {}
    for name, fn in cases.items():
        closed_jaxpr = jax.make_jaxpr(fn)(jnp.ones((3,), dtype=jnp.float32))
        try:
            interpret_closed_jaxpr_with_inlining(
                closed_jaxpr, jax.tree.leaves((jnp.ones((3,), dtype=jnp.float32),))
            )
        except ValueError as exc:
            rejected = True
            error = str(exc)
        else:
            rejected = False
            error = ""
        assert rejected, f"{name} was unexpectedly accepted"
        results[name] = {
            "rejected": rejected,
            "error": error,
            "outer_primitives": [eqn.primitive.name for eqn in closed_jaxpr.jaxpr.eqns],
        }
    return results


def measure_corpus_with_inlining() -> dict[str, Any]:
    """Measure accepted capture rate using the inlining interpreter.

    Returns
    -------
    dict[str, Any]
        Corpus acceptance evidence.
    """

    case_results: list[dict[str, Any]] = []
    accepted = 0
    for call in build_extended_corpus():
        try:
            closed_jaxpr = derive_closed_jaxpr(call)
            captured = interpret_closed_jaxpr_with_inlining(
                closed_jaxpr, flatten_dynamic_args(call)
            )
            direct = block_until_ready(call.fn(call.params, *call.inputs))
            interpreted_outputs = block_until_ready(
                jax.tree.unflatten(jax.tree.structure(direct), captured.outputs)
            )
            assert tree_allclose(interpreted_outputs, direct)
        except Exception as exc:  # noqa: BLE001 - spike records rejection cause.
            case_results.append(
                {"name": call.name, "accepted": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        accepted += 1
        case_results.append(
            {
                "name": call.name,
                "accepted": True,
                "equations": len(captured.equations),
                "inlined_equations": sum(eqn.inlined for eqn in captured.equations),
                "inlined_call_primitives": sorted(set(captured.inlined_call_primitives)),
                "outer_primitives": [eqn.primitive.name for eqn in closed_jaxpr.jaxpr.eqns],
                "consts": len(closed_jaxpr.consts),
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


def measure_overhead_with_inlining() -> dict[str, Any]:
    """Measure interpreted-vs-direct overhead with the inlining interpreter.

    Returns
    -------
    dict[str, Any]
        Timing evidence.
    """

    calls_by_name = {call.name: call for call in build_extended_corpus()}

    def toy_fn(_params: None, x: Any) -> Any:
        """Toy elementwise timing fixture."""

        return jnp.sin(x) + jnp.cos(x * 2.0)

    toy_call = DeclaredCall("toy", toy_fn, None, (jnp.ones((8,), dtype=jnp.float32),))
    timed_calls = (
        (toy_call, 200),
        (calls_by_name["mlp"], 100),
        (calls_by_name["cnn"], 50),
    )
    results: dict[str, Any] = {}
    for call, repeats in timed_calls:
        closed_jaxpr = derive_closed_jaxpr(call)
        flat_args = flatten_dynamic_args(call)
        interpret_closed_jaxpr_with_inlining(closed_jaxpr, flat_args)
        block_until_ready(call.fn(call.params, *call.inputs))

        def direct_fn(captured_call: DeclaredCall = call) -> Any:
            """Run the direct JAX fixture for timing."""

            return captured_call.fn(captured_call.params, *captured_call.inputs)

        def interpreted_fn(
            captured_jaxpr: core.ClosedJaxpr = closed_jaxpr,
            captured_args: Sequence[Any] = flat_args,
        ) -> Any:
            """Run the interpreted JAX fixture for timing."""

            return interpret_closed_jaxpr_with_inlining(captured_jaxpr, captured_args).outputs

        direct_times = time_call(direct_fn, repeats)
        interpreted_times = time_call(interpreted_fn, repeats)
        direct_median = statistics.median(direct_times)
        interpreted_median = statistics.median(interpreted_times)
        results[call.name] = {
            "repeats": repeats,
            "outer_equations": len(closed_jaxpr.jaxpr.eqns),
            "captured_equations": len(
                interpret_closed_jaxpr_with_inlining(closed_jaxpr, flat_args).equations
            ),
            "direct_median_s": direct_median,
            "interpreted_median_s": interpreted_median,
            "overhead_x": interpreted_median / direct_median,
        }
    return results


def run_all() -> dict[str, Any]:
    """Run all round-2 inlining probes.

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
        "safe_inline_policy": {
            "safe_jit_names": sorted(SAFE_JIT_NAMES),
            "custom_jvp_rule": "only custom_jvp_call whose call_jaxpr contains allowlisted jit calls",
            "rejected_nested_primitives": sorted(REJECTED_NESTED_PRIMITIVES),
        },
        "inlined_replay": probe_inlined_replay_and_perturbation(),
        "rejection_boundaries": probe_rejection_boundaries(),
        "corpus": measure_corpus_with_inlining(),
        "overhead": measure_overhead_with_inlining(),
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

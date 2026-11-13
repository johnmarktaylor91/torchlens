"""Native JAX backward capture: records, validator, leaf gradients, refusals.

Each case captures a JAX program, runs ``trace.backward(loss_fn)`` through the
native surface, and proves three things: the backward validator in
``validation/_backward_neutral.py`` passes on the recorded pass, the recorded
leaf gradients equal ``jax.grad`` of the original program, and the forward
records are unchanged.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("jax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import torchlens as tl  # noqa: E402
from torchlens._utils._record_group_table import record_member  # noqa: E402
from torchlens.backends import registry  # noqa: E402
from torchlens.backends._backward_native import (
    jax as native_jax,  # noqa: E402
    native_backward_surface,  # noqa: E402
)
from torchlens.backends._replay_adapters import get_adapter  # noqa: E402
from torchlens.errors import RunCapabilityUnavailableError  # noqa: E402
from torchlens.validation._backward_neutral import (  # noqa: E402
    RecordedBackward,
    check_backward_records,
    recorded_backward_from_trace,
)


@pytest.fixture(autouse=True)
def _jax_backward_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry hunk pp-merge applies: JAX's backward_capture through the native surface."""

    spec = registry.get_backend_spec("jax")
    replaced = dataclasses.replace(
        spec,
        capabilities=dataclasses.replace(spec.capabilities, backward_capture=True),
        capability_implementations={
            **(spec.capability_implementations or {}),
            "backward_capture": lambda: native_backward_surface("jax"),
        },
    )
    monkeypatch.setitem(registry._REGISTRY, "jax", replaced)


def _x(*shape: int, seed: int = 1) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32)


def _capture(net: Callable[..., Any], *args: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        example = args[0] if len(args) == 1 else args
        return tl.trace(net, example, backend="jax")


def _input_ops(trace: Any) -> list[Any]:
    return [op for op in trace._layer_list if record_member(op, "func_name", "") == "input"]


def _oracle(net: Callable[..., Any], args: tuple[Any, ...], loss: Callable[[Any], Any]) -> Any:
    """``jax.grad`` of ``loss(net(*args))`` for every argument leaf, flattened in order."""

    grads = jax.grad(lambda *a: loss(net(*a)), argnums=tuple(range(len(args))))(*args)
    return jax.tree.leaves(grads)


def _snapshot(trace: Any) -> dict[str, Any]:
    return {op.label: op.out for op in trace._layer_list}


def _assert_recorded(
    net: Callable[..., Any], args: tuple[Any, ...], loss: Callable[[Any], Any]
) -> Any:
    """Capture, run backward, and check validator, leaf gradients and forward records."""

    adapter = get_adapter("jax")
    trace = _capture(net, *args)
    before = _snapshot(trace)
    output_before = trace.output
    assert trace.backward(loss) is trace
    passes = trace.autograd.backward_passes
    assert len(passes) == 1
    grad_fns = list(trace._grad_fn_logs.values())
    assert grad_fns and all(g.class_name.startswith("transpose(jvp(") for g in grad_fns)
    assert all(g.op_label is not None for g in grad_fns)

    leaves = _input_ops(trace)
    expected = _oracle(net, args, loss)
    assert len(expected) == len(leaves)
    leaf_grads = {op.label: want for op, want in zip(leaves, expected, strict=True)}
    for op in leaves:
        assert adapter.close(leaf_grads[op.label], op.grad), op.label

    recorded = recorded_backward_from_trace(trace, 1)
    graph, calls, checker = native_jax.validation_inputs(trace, recorded)
    result = check_backward_records(
        trace, checker, graph, RecordedBackward(calls=calls, grads=recorded.grads), leaf_grads
    )
    assert result.passed, result.findings
    assert result.leaves_checked == len(leaves)

    assert trace.output is output_before
    after = _snapshot(trace)
    assert after.keys() == before.keys()
    for label, value in before.items():
        assert after[label] is value, label
    return trace


def test_diamond_sums_three_consumers() -> None:
    def diamond(x: Any) -> Any:
        y = jnp.tanh(x)
        return jnp.sum(y * y + jnp.exp(y))

    trace = _assert_recorded(diamond, (_x(4, 3),), lambda out: out)
    tanh = [g for g in trace._grad_fn_logs.values() if g.class_name == "transpose(jvp(tanh))"]
    assert len(tanh) == 1
    (call,) = list(getattr(tanh[0].calls, "_dict", tanh[0].calls).values())
    assert call.backward_pass_number == 1


def test_shared_dense_layer_has_one_node_per_call() -> None:
    def dense(w: Any, b: Any, x: Any) -> Any:
        return jax.nn.relu(x @ w + b)

    def shared(params: Any, x: Any) -> Any:
        return dense(params["w"], params["b"], dense(params["w"], params["b"], x))

    params = {"w": _x(3, 3, seed=2) * 0.5, "b": _x(3, seed=3) * 0.1}
    x = _x(4, 3)
    loss = lambda out: jnp.sum(out**2)  # noqa: E731
    trace = _assert_recorded(shared, (params, x), loss)
    dots = [g for g in trace._grad_fn_logs.values() if "dot_general" in g.class_name]
    assert len(dots) == 2
    expected = jax.grad(lambda p: loss(shared(p, x)))(params)
    for address in ("w", "b"):
        assert get_adapter("jax").close(expected[address], trace._param_logs[address].grad)


def test_multi_output_split_and_top_k() -> None:
    def multi(x: Any) -> Any:
        a, b = jnp.split(x, 2, axis=0)
        v, _ = jax.lax.top_k(x[0], 2)
        return jnp.sum(a * 2.0 + jnp.sin(b)) + jnp.sum(v * v)

    _assert_recorded(multi, (_x(4, 3),), lambda out: out)


def test_pass_through_and_zero_derivative_rules_are_recorded() -> None:
    def program(x: Any) -> Any:
        return jnp.floor(x) * x + (x + 1.0)

    trace = _assert_recorded(program, (_x(5),), lambda out: jnp.sum(out))
    names = {g.class_name for g in trace._grad_fn_logs.values()}
    assert "transpose(jvp(add))" in names
    assert "transpose(jvp(floor))" in names


def test_output_also_consumed_inside_gets_the_loss_cotangent_plus_consumers() -> None:
    def two_outputs(x: Any) -> Any:
        h = jnp.sin(x)
        return h, jnp.sum(h * 3.0)

    _assert_recorded(two_outputs, (_x(3),), lambda out: out[1] + jnp.sum(out[0] ** 2))


def test_ignored_output_is_not_reached() -> None:
    def two_outputs(x: Any) -> Any:
        return jnp.cos(x), jnp.sum(jnp.sin(x))

    trace = _assert_recorded(two_outputs, (_x(3),), lambda out: out[1])
    cos = [g for g in trace._grad_fn_logs.values() if "cos" in g.class_name]
    assert cos == []


def test_unrolled_scan_records_one_node_per_iteration() -> None:
    def scan_fn(w: Any, xs: Any) -> Any:
        def body(c: Any, x: Any) -> Any:
            c2 = jnp.tanh(c * w + x)
            return c2, c2 * 2.0

        c_t, ys = jax.lax.scan(body, jnp.zeros((3,)), xs)
        return jnp.sum(ys) + jnp.sum(c_t)

    trace = _assert_recorded(scan_fn, (_x(3, seed=4), _x(4, 3)), lambda out: out)
    tanh = [g for g in trace._grad_fn_logs.values() if g.class_name == "transpose(jvp(tanh))"]
    assert len(tanh) == 4


def test_second_backward_adds_a_second_pass() -> None:
    def program(x: Any) -> Any:
        return jnp.sum(jnp.tanh(x) * 2.0)

    trace = _capture(program, _x(3))
    trace.backward(lambda out: out)
    trace.backward(lambda out: out * 3.0)
    assert sorted(trace._backward_pass_logs) == [1, 2]
    (op,) = _input_ops(trace)
    expected = jax.grad(lambda v: program(v) * 3.0)(_x(3))
    assert get_adapter("jax").close(expected, op.autograd.grad_for(bwd=2))


def test_non_callable_loss_is_refused() -> None:
    trace = _capture(lambda v: jnp.sum(v * 2.0), _x(3))
    with pytest.raises(RunCapabilityUnavailableError, match="callable"):
        trace.backward(trace.output)


def test_keyword_arguments_are_refused() -> None:
    trace = _capture(lambda v: jnp.sum(v * 2.0), _x(3))
    with pytest.raises(RunCapabilityUnavailableError, match="keyword"):
        trace.backward(lambda out: out, retain_graph=True)


def test_non_scalar_loss_is_refused() -> None:
    trace = _capture(lambda v: v * 2.0, _x(3))
    with pytest.raises(RunCapabilityUnavailableError, match="scalar"):
        trace.backward(lambda out: out)


def test_custom_vjp_region_is_refused_not_recorded() -> None:
    @jax.custom_vjp
    def odd_sin(x: Any) -> Any:
        return jnp.sin(x)

    odd_sin.defvjp(lambda x: (jnp.sin(x), jnp.cos(x)), lambda c, g: (3.0 * g * c,))
    trace = _capture(lambda v: jnp.sum(odd_sin(v) * v), _x(3))
    with pytest.raises(RunCapabilityUnavailableError, match="unavailable"):
        trace.backward(lambda out: out)
    assert not trace._grad_fn_logs

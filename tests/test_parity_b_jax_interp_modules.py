"""Module attribution inside nested jaxprs (lane b-jax-interp, gap X8).

Equinox and Flax NNX modules that call submodules inside ``jax.jit``, ``lax.cond`` and
``lax.scan`` keep each op on the module that ran it: the enclosing equation's module scope
is threaded into the nested jaxpr. Two same-shaped submodules stay distinct, a shared
submodule keeps one module with two calls, the unroll cap still falls back to a region,
``jax_control_flow="reject"`` still refuses, and rematerialization refuses with a typed error.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import torchlens as tl
from torchlens.errors import BackendUnsupportedError, CaptureGapWarning
from torchlens.validation._invariants import check_metadata_invariants

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")
nnx = pytest.importorskip("flax.nnx")

pytestmark = [pytest.mark.backend_jax, pytest.mark.optional]

_WIDTH = 3


class _Lin(eqx.Module):
    """Square linear layer."""

    weight: Any
    bias: Any

    def __init__(self, seed: int) -> None:
        """Draw weights from ``seed``."""

        self.weight = jax.random.normal(jax.random.PRNGKey(seed), (_WIDTH, _WIDTH)) / _WIDTH
        self.bias = jnp.full((_WIDTH,), 0.1 * seed)

    def __call__(self, value: Any) -> Any:
        """Return ``tanh(weight @ value + bias)``."""

        return jnp.tanh(self.weight @ value + self.bias)


class _Twin(eqx.Module):
    """Two same-shaped submodules ``a`` and ``b`` called inside one transform."""

    a: _Lin
    b: _Lin
    kind: str = eqx.field(static=True)
    length: int = eqx.field(static=True, default=2)

    def __call__(self, value: Any) -> Any:
        """Route ``a`` then ``b`` through scan, cond or jit."""

        if self.kind == "scan":

            def body(carry: Any, _item: Any) -> tuple[Any, None]:
                """One step through both submodules."""

                return self.b(self.a(carry)), None

            return jax.lax.scan(body, value, None, length=self.length)[0]
        if self.kind == "cond":
            return jax.lax.cond(
                jnp.sum(value) > 0, lambda v: self.b(self.a(v)), lambda v: self.a(v) * 2.0, value
            )
        return jax.jit(lambda v: self.b(self.a(v)))(value)


class _JitTwice(eqx.Module):
    """One jitted closure over a submodule, called twice (one cached nested jaxpr)."""

    a: _Lin

    def __call__(self, value: Any) -> Any:
        """Apply the jitted submodule call twice."""

        step = jax.jit(lambda v: self.a(v))
        return step(step(value))


class _SharedInScan(eqx.Module):
    """One submodule instance called twice per scan step."""

    shared: _Lin

    def __call__(self, value: Any) -> Any:
        """Call ``shared`` twice inside each of two scan steps."""

        def body(carry: Any, _item: Any) -> tuple[Any, None]:
            """Two calls of the shared submodule."""

            return self.shared(self.shared(carry)), None

        return jax.lax.scan(body, value, None, length=2)[0]


class _Remat(eqx.Module):
    """Submodule under ``jax.checkpoint`` (rematerialization)."""

    a: _Lin

    def __call__(self, value: Any) -> Any:
        """Return the checkpointed submodule output."""

        return jax.checkpoint(lambda v: self.a(v))(value)


class _NnxTwin(nnx.Module):
    """NNX module with two same-shaped children called inside scan, cond or jit."""

    def __init__(self, kind: str) -> None:
        """Build two ``nnx.Linear`` children."""

        self.kind = kind
        self.a = nnx.Linear(_WIDTH, _WIDTH, rngs=nnx.Rngs(31))
        self.b = nnx.Linear(_WIDTH, _WIDTH, rngs=nnx.Rngs(32))

    def __call__(self, value: Any) -> Any:
        """Route ``a`` then ``b`` through the chosen transform."""

        def both(v: Any) -> Any:
            """Apply both children."""

            return jnp.tanh(self.b(jnp.tanh(self.a(v))))

        if self.kind == "scan":
            return jax.lax.scan(lambda c, _x: (both(c), None), value, None, length=2)[0]
        if self.kind == "cond":
            return jax.lax.cond(jnp.sum(value) > 0, both, lambda v: v + 1.0, value)
        return jax.jit(both)(value)


def _input() -> Any:
    """Return a deterministic input with a positive sum (the cond takes its true branch)."""

    return jnp.linspace(0.1, 0.6, _WIDTH)


def _labels(trace: Any, address: str) -> set[str]:
    """Return the labels of ops whose module stack includes a call of ``address``."""

    prefix = f"{address}:"
    return {op.label for op in trace._layer_list if any(m.startswith(prefix) for m in op.modules)}


def _trace(model: Any, value: Any, **options: Any) -> Any:
    """Trace ``model``; submodule calls inside transform bodies are disclosed as unrecorded.

    Their returns are values of the body's inner trace (one trace-time call stands for
    every run), so the return ledger leaves them unrecorded with a ``CaptureGapWarning``.
    """

    with pytest.warns(CaptureGapWarning, match="returned arrays of an inner trace"):
        return tl.trace(model, value, backend="jax", **options)


def _assert_sound(trace: Any, model: Any) -> None:
    """Assert invariants, validation and the output value."""

    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True
    np.testing.assert_allclose(
        np.asarray(trace.ops[trace.output_layers[0]].out), np.asarray(model(_input())), rtol=1e-5
    )


def _assert_twins_distinct(trace: Any, steps: int) -> None:
    """Assert ``a`` and ``b`` own disjoint, non-empty op sets and their own parameters."""

    a_ops, b_ops = _labels(trace, "a"), _labels(trace, "b")
    assert a_ops and b_ops
    assert not a_ops & b_ops
    assert len(a_ops) == len(b_ops)
    assert len(a_ops) % steps == 0
    assert all(trace[label].modules[-1].startswith("a:") for label in a_ops)
    assert all(trace[label].modules[-1].startswith("b:") for label in b_ops)
    assert {p.module_address for p in trace.modules["a"].params} == {"a"}
    assert {p.module_address for p in trace.modules["b"].params} == {"b"}


@pytest.mark.parametrize(("kind", "steps"), [("scan", 2), ("cond", 1), ("jit", 1)])
def test_equinox_same_shaped_submodules_stay_distinct(kind: str, steps: int) -> None:
    """Same-shaped Equinox submodules inside scan, cond and jit keep their own ops."""

    model = _Twin(_Lin(1), _Lin(2), kind)
    trace = _trace(model, _input())

    _assert_sound(trace, model)
    _assert_twins_distinct(trace, steps)


@pytest.mark.parametrize(("kind", "steps"), [("scan", 2), ("cond", 1), ("jit", 1)])
def test_nnx_same_shaped_children_stay_distinct(kind: str, steps: int) -> None:
    """Same-shaped NNX children inside scan, cond and jit keep their own ops."""

    model = _NnxTwin(kind)
    trace = _trace(model, _input())

    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True
    a_ops, b_ops = _labels(trace, "a"), _labels(trace, "b")
    assert a_ops and b_ops
    assert not a_ops & b_ops
    assert len(a_ops) % steps == 0


def test_cond_untaken_branch_has_no_ops() -> None:
    """Only the taken branch runs: ``b`` is called only in the true branch."""

    model = _Twin(_Lin(1), _Lin(2), "cond")
    negative = -_input()
    trace = _trace(model, negative)

    assert trace.validate_forward_pass([]) is True
    assert _labels(trace, "a")
    assert not _labels(trace, "b")


def test_shared_submodule_in_scan_gets_one_call_per_run() -> None:
    """A shared submodule called twice per scan step is one module with a call per run.

    As on PyTorch, each run of a module inside an unrolled loop is its own call: two
    calls per step over two steps are four calls with disjoint, same-sized op sets.
    """

    model = _SharedInScan(_Lin(3))
    trace = _trace(model, _input())

    _assert_sound(trace, model)
    shared = trace.modules["shared"]
    assert shared.num_calls == 4
    calls = [shared.ops.get(index) for index in range(1, 5)]
    assert all(call is not None and call.ops for call in calls)
    op_sets = [set(call.ops) for call in calls]
    assert all(not op_sets[i] & op_sets[j] for i in range(4) for j in range(i + 1, 4))
    assert len({len(ops) for ops in op_sets}) == 1


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"jax_max_control_flow_unroll": 2}, "scan_length_exceeds_unroll_cap"),
        ({"jax_control_flow": "region"}, "scan_region_requested"),
    ],
)
def test_scan_imported_as_region_inside_module(options: dict[str, Any], reason: str) -> None:
    """A scan imported as a region inside an attributed module becomes one region op."""

    model = _Twin(_Lin(1), _Lin(2), "scan", length=5)
    trace = _trace(model, _input(), capture=tl.options.CaptureOptions(**options))

    regions = [op for op in trace._layer_list if op.annotations.get("jax_capture_kind") == "region"]
    assert len(regions) == 1
    assert regions[0].annotations["jax_region_reason"] == reason
    assert not _labels(trace, "a")
    assert trace.validate_forward_pass([]).state == "unverified"


def test_reject_mode_still_refuses_control_flow_inside_module() -> None:
    """``jax_control_flow='reject'`` refuses a scan inside an attributed module."""

    model = _Twin(_Lin(1), _Lin(2), "scan")
    with pytest.raises(BackendUnsupportedError, match="scan"):
        tl.trace(
            model,
            _input(),
            backend="jax",
            capture=tl.options.CaptureOptions(jax_control_flow="reject"),
        )


def test_rematerialization_inside_module_is_a_typed_refusal() -> None:
    """``jax.checkpoint`` inside an attributed module refuses with a typed, coded error."""

    with pytest.raises(BackendUnsupportedError) as excinfo:
        tl.trace(_Remat(_Lin(1)), _input(), backend="jax")

    assert excinfo.value.code == "jax_nested_primitive_unsupported"
    notes = " ".join(getattr(excinfo.value, "__notes__", []))
    assert "inside attributed module" in notes


@pytest.mark.parametrize(
    "factory", [lambda: _Twin(_Lin(1), _Lin(2), "jit"), lambda: _NnxTwin("jit")]
)
def test_jit_closing_over_module_weights_records_params(factory: Any) -> None:
    """Weights a nested ``jax.jit`` closes over are recorded as module params."""

    model = factory()
    trace = _trace(model, _input())

    assert check_metadata_invariants(trace) is True
    assert trace.validate_forward_pass([]) is True
    for address in ("a", "b"):
        assert any(trace[label].num_params for label in _labels(trace, address))


def test_jitted_submodule_called_twice_stays_sound() -> None:
    """A cached nested jaxpr bound twice is two calls that both read the module weights."""

    model = _JitTwice(_Lin(4))
    trace = _trace(model, _input())

    _assert_sound(trace, model)
    module = trace.modules["a"]
    assert module.num_calls == 2
    for call in (1, 2):
        readers = [trace[label] for label in module.ops.get(call).ops]
        assert any("a.weight" in [p.address for p in op._param_logs] for op in readers)


def test_hoisted_jaxpr_evaluates_like_the_function() -> None:
    """Hoisting moves the jit body's constants to the top level without changing values."""

    from jax.extend import core

    from torchlens.backends.jax.jaxpr import derive_closed_jaxpr

    weight = jnp.arange(9.0).reshape(3, 3) / 9.0

    def fn(x: Any) -> Any:
        """Close a nested jit over ``weight``, twice."""

        step = jax.jit(lambda v: jnp.tanh(weight @ v))
        return step(step(x)) + weight[0]

    closed = derive_closed_jaxpr(fn, (_input(),))
    jits = [e for e in closed.jaxpr.eqns if e.primitive.name in {"jit", "pjit"}]

    assert jits and all(not tuple(e.params["jaxpr"].consts) for e in jits)
    assert sum(c is weight for c in closed.consts) == 1
    got = core.jaxpr_as_fun(closed)(_input())[0]
    np.testing.assert_allclose(np.asarray(got), np.asarray(fn(_input())), rtol=1e-6)

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
from torchlens.errors import BackendUnsupportedError
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

        return jax.checkpoint(self.a)(value)


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
    trace = tl.trace(model, _input(), backend="jax")

    _assert_sound(trace, model)
    _assert_twins_distinct(trace, steps)


@pytest.mark.parametrize(("kind", "steps"), [("scan", 2), ("cond", 1), ("jit", 1)])
def test_nnx_same_shaped_children_stay_distinct(kind: str, steps: int) -> None:
    """Same-shaped NNX children inside scan, cond and jit keep their own ops."""

    model = _NnxTwin(kind)
    trace = tl.trace(model, _input(), backend="jax")

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
    trace = tl.trace(model, negative, backend="jax")

    assert trace.validate_forward_pass([]) is True
    assert _labels(trace, "a")
    assert not _labels(trace, "b")


def test_shared_submodule_in_scan_keeps_two_calls() -> None:
    """A shared submodule called twice per scan step is one module with two calls."""

    model = _SharedInScan(_Lin(3))
    trace = tl.trace(model, _input(), backend="jax")

    _assert_sound(trace, model)
    shared = trace.modules["shared"]
    assert shared.num_calls == 2
    first, second = shared.ops.get(1), shared.ops.get(2)
    assert first is not None and second is not None
    assert first.ops and second.ops
    assert not set(first.ops) & set(second.ops)
    assert len(first.ops) == len(second.ops)


def test_scan_over_unroll_cap_falls_back_to_region_inside_module() -> None:
    """Over the unroll cap, the scan inside an attributed module becomes one region."""

    model = _Twin(_Lin(1), _Lin(2), "scan", length=5)
    trace = tl.trace(
        model,
        _input(),
        backend="jax",
        capture=tl.options.CaptureOptions(jax_max_control_flow_unroll=2),
    )

    regions = [op for op in trace._layer_list if op.annotations.get("jax_capture_kind") == "region"]
    assert len(regions) == 1
    assert regions[0].annotations["jax_region_reason"] == "scan_length_exceeds_unroll_cap"
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

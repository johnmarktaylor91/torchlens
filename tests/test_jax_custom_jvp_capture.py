"""JAX ``custom_jvp_call`` capture: inline pure primals, refuse impure typed.

``jax.custom_jvp`` customizes DERIVATIVES only; the equation's ``call_jaxpr``
is the exact forward primal, so inlining a recursively pure, const-free primal
is forward-faithful by construction.  Two library shapes exist in the wild:

* ``jax.nn.relu`` -- a jit/pjit-wrapped primal inside the custom-JVP frame.
* ``jax.nn.softplus`` -- a flat primitive primal with no jit wrapper.

Both previously died with the bare ``ValueError: unsupported nested call
primitive: custom_jvp_call name=None``.  A primal that is NOT provably pure
still refuses, now with the typed ``BackendUnsupportedError`` naming the
capability gap instead of the bare ValueError.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

import torchlens as tl
from torchlens.errors import BackendUnsupportedError

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = pytest.mark.backend_jax


def _params() -> dict[str, Any]:
    """Return a small parameter pytree."""

    return {"w": jnp.ones((3, 2), dtype=jnp.float32)}


def _x() -> Any:
    """Return a small input batch with both signs represented."""

    return jnp.array([[1.0, -2.0, 0.5], [-0.25, 3.0, -1.0]], dtype=jnp.float32)


def _final_payload(trace: Any) -> Any:
    """Return the saved payload of the last captured op."""

    return trace.layer_list[-1].out


def test_jax_nn_relu_captures_faithfully() -> None:
    """``jax.nn.relu`` inlines through its custom-JVP wrapper with exact values.

    ``trace.validate_forward_pass`` is deliberately NOT asserted here: on
    jax 0.6 the preview's postprocess currently leaves raw ``:pass`` labels on
    EVERY capture (plain ``tanh`` captures fail the same graph_ordering
    invariant on main), which is a pre-existing issue independent of the
    custom-JVP inlining under test.
    """

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return a dense block through the library ReLU wrapper."""

        return jax.nn.relu(x @ params["w"])

    params, x = _params(), _x()
    trace = tl.trace(cast(Any, model), (params, x), backend="jax")

    assert "custom_jvp_call" in trace.provenance.jax_inlined_call_primitives
    assert "max" in {op.func_name for op in trace.layer_list}
    expected = jax.nn.relu(x @ params["w"])
    assert jnp.array_equal(_final_payload(trace), expected)


def test_jax_nn_softplus_captures_faithfully() -> None:
    """``jax.nn.softplus`` (flat custom-JVP primal) inlines with exact values."""

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return a dense block through the library softplus wrapper."""

        return jax.nn.softplus(x @ params["w"])

    params, x = _params(), _x()
    trace = tl.trace(cast(Any, model), (params, x), backend="jax")

    assert "custom_jvp_call" in trace.provenance.jax_inlined_call_primitives
    captured_primitives = {op.func_name for op in trace.layer_list}
    assert "log1p" in captured_primitives or "exp" in captured_primitives
    expected = jax.nn.softplus(x @ params["w"])
    assert jnp.array_equal(_final_payload(trace), expected)


def test_user_custom_jvp_with_pure_primal_captures() -> None:
    """A user ``jax.custom_jvp`` with a pure primal inlines like the library's."""

    @jax.custom_jvp
    def cubed(value: Any) -> Any:
        """Return the elementwise cube."""

        return value * value * value

    @cubed.defjvp
    def cubed_jvp(primals: Any, tangents: Any) -> Any:
        """Return the custom JVP for :func:`cubed`."""

        (value,), (tangent,) = primals, tangents
        return cubed(value), 3.0 * value * value * tangent

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return a dense block through the user custom-JVP wrapper."""

        return cubed(x @ params["w"])

    params, x = _params(), _x()
    trace = tl.trace(cast(Any, model), (params, x), backend="jax")

    assert "custom_jvp_call" in trace.provenance.jax_inlined_call_primitives
    hidden = x @ params["w"]
    assert jnp.array_equal(_final_payload(trace), hidden * hidden * hidden)


def test_non_inlinable_nested_call_refuses_typed() -> None:
    """A nested call failing the purity bar refuses TYPED, naming the gap.

    Donated inputs make a nested JIT frame non-inlinable (the buffer may be
    reused in place, which the replay contract cannot represent).  The
    refusal must be the typed ``BackendUnsupportedError`` naming the nested
    call capability gap, never the historical bare ``ValueError``.
    """

    import functools

    @functools.partial(jax.jit, donate_argnums=0)
    def donating(value: Any) -> Any:
        """Return a scaled value behind a donating JIT boundary."""

        return value * 2.0

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return a dense block through the donating JIT boundary."""

        return donating(x @ params["w"])

    with pytest.raises(BackendUnsupportedError, match="nested call"):
        tl.trace(cast(Any, model), (_params(), _x()), backend="jax")

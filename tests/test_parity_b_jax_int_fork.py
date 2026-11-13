"""JAX parity: finished-trace edits (``trace.fork().do``) with the six helpers.

A fork of a clean JAX capture, edited with any of the six JAX helpers or a
callable, equals a live ``intervene=`` capture with the same edit and passes
the backend replay check. An op downstream of the site that the replay adapter
cannot replay refuses typed and names the op.
"""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
import pytest

import torchlens as tl
from torchlens.intervention._backend_replay import verify_backend_replay

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax, pytest.mark.rare]

_W = np.asarray([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype=np.float32)
_X = np.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=np.float32)

_EDITS = [
    tl.zero_ablate(),
    tl.scale(0.5),
    tl.add(1.0),
    tl.clamp(low=-0.1, high=0.1),
    tl.mean_ablate(dim=None),
    tl.mean_ablate(dim=1),
    tl.replace_with(jnp.ones((2, 2), jnp.float32)),
    lambda out, *, ctx: out * 3.0,
]


def _model(x: Any) -> Any:
    """Return ``(tanh(x @ W) + 1) * 2``: the edit propagates through two ops."""

    return (jnp.tanh(x @ jnp.asarray(_W)) + 1.0) * 2.0


def _output(trace: Any) -> np.ndarray:
    """Return the trace's single output as NumPy."""

    return np.asarray(trace[trace.output_layers[0]].out)


def _clean() -> Any:
    """Return a clean JAX capture of :func:`_model`."""

    return tl.trace(cast(Any, _model), jnp.asarray(_X), backend="jax")


@pytest.mark.parametrize("edit", _EDITS)
def test_fork_helper_edit_equals_live_intervene(edit: Any) -> None:
    """A fork edit equals the live capture-time edit and passes the replay check."""

    clean = _clean()
    live = tl.trace(
        cast(Any, _model),
        jnp.asarray(_X),
        backend="jax",
        intervene=tl.when(tl.func("tanh"), edit),
    )
    edited = clean.fork().do(tl.func("tanh"), edit)
    np.testing.assert_allclose(_output(edited), _output(live), rtol=1e-6, atol=1e-7)
    assert not np.allclose(_output(edited), _output(clean))
    assert verify_backend_replay(edited) == []
    assert edited.is_pristine is False
    np.testing.assert_array_equal(_output(clean), np.asarray(_model(jnp.asarray(_X))))


def test_fork_refuses_unreplayable_downstream_op_by_name() -> None:
    """An op the adapter cannot replay (a custom_vjp region) refuses typed and names the op."""

    @jax.custom_vjp
    def opaque(x: Any) -> Any:
        """Return ``sin(x)`` behind a custom VJP."""

        return jnp.sin(x)

    opaque.defvjp(lambda x: (jnp.sin(x), x), lambda x, g: (g * jnp.cos(x),))

    def model(x: Any) -> Any:
        """Return a custom_vjp call downstream of a tanh."""

        return opaque(jnp.tanh(x @ jnp.asarray(_W)))

    clean = tl.trace(cast(Any, model), jnp.asarray(_X), backend="jax")
    with pytest.raises(Exception) as caught:
        clean.fork().do(tl.func("tanh"), tl.scale(0.5))
    error = cast(Any, caught.value)
    code = getattr(error, "code", None) or getattr(error, "fields", {}).get("code")
    assert code == "backend_replay_op_unreplayable"
    named = re.findall(r"'([^']+)'", str(error))
    assert any(name in clean._layer_dict_all_keys for name in named), str(error)

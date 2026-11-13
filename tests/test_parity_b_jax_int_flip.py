"""JAX parity: ``can_intervene`` is declared and live ``intervene=`` dispatches.

The JAX spec declares ``can_intervene=True`` with the capture-time edit runtime
bound, so ``tl.trace(..., backend="jax", intervene=...)`` reaches the jaxpr
interpreter without any test-side flag flip. ``stop_after=`` shares the
capability's options and halts the interpreter at the first matching equation.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends.registry import require_capability_implementation

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax, pytest.mark.rare]

_W = np.asarray([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype=np.float32)
_X = np.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=np.float32)


def _model(x: Any) -> Any:
    """Return ``tanh(x @ W) * 2``."""

    return jnp.tanh(x @ jnp.asarray(_W)) * 2.0


def test_jax_spec_declares_intervene_with_runtime_binding() -> None:
    """The flag is declared on the shipped spec and resolves the edit runtime."""

    from torchlens.backends.jax._edits import JaxEditRuntime

    spec = tl.backends.spec("jax")
    assert spec.capabilities.can_intervene is True
    assert require_capability_implementation(spec, "can_intervene") is JaxEditRuntime


@pytest.mark.parametrize(
    "action",
    [
        tl.zero_ablate(),
        tl.scale(0.5),
        tl.add(1.0),
        tl.clamp(low=-0.1, high=0.1),
        tl.mean_ablate(dim=None),
        tl.replace_with(jnp.ones((2, 2), jnp.float32)),
    ],
)
def test_jax_live_helper_capture_validates(action: Any) -> None:
    """Each of the six helpers edits a live capture that then validates."""

    clean = tl.trace(cast(Any, _model), jnp.asarray(_X), backend="jax")
    trace = tl.trace(
        cast(Any, _model),
        jnp.asarray(_X),
        backend="jax",
        intervene=tl.when(tl.func("tanh"), action),
    )
    edited = [op for op in trace.ops if op.is_edited]
    assert len(edited) == 1
    assert not np.allclose(
        np.asarray(trace[trace.output_layers[0]].out),
        np.asarray(clean[clean.output_layers[0]].out),
    )
    assert bool(trace.validate_forward_pass([]))


def test_jax_stop_after_halts_in_backend() -> None:
    """``stop_after=`` reaches the backend and stops the trace at the first tanh."""

    trace = tl.trace(cast(Any, _model), jnp.asarray(_X), backend="jax", stop_after=tl.func("tanh"))
    assert trace.is_stopped is True
    frontier = trace[trace.stop_reason]
    assert frontier.type == "tanh"
    np.testing.assert_allclose(np.asarray(frontier.out), np.tanh(_X @ _W), rtol=1e-6, atol=1e-6)

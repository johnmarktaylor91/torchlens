"""End-to-end dispatch-consumption conformance (sol probe, live runtime).

A capability flag flipped True with a bound-but-undispatched implementation
must refuse at ``tl.trace()``, never capture while silently ignoring the
requested option. The policy-level biconditional lives in
``test_capability_truth.py``; these tests prove the refusal reaches the real
public entry on a live preview runtime (sol's original probe registered a JAX
spec with ``interventions=True`` and a dummy binding, passed ``intervene=``,
and got a capture whose output showed the intervention was ignored). A flag
a backend really lifts is proved the other way: the bound runtime is the one
its capture path consumes and the option takes effect.
"""

from __future__ import annotations

import dataclasses

import pytest

pytestmark = [pytest.mark.backend_parity, pytest.mark.rare]


def _spec_flipped_true(name: str, flag: str):
    import torchlens as tl

    original = tl.backends.spec(name)
    implementations = dict(original.capability_implementations or {})
    implementations[flag] = lambda: object()
    return original, dataclasses.replace(
        original,
        capabilities=dataclasses.replace(original.capabilities, **{flag: True}),
        capability_implementations=implementations,
    )


@pytest.mark.backend_jax
def test_jax_interventions_flip_is_dispatched_at_trace() -> None:
    """JAX declares can_intervene=True: the bound runtime is the one the capture
    path consumes, an edit reaches the output, and nothing is silently ignored."""

    jnp = pytest.importorskip("jax.numpy")
    import torchlens as tl
    from torchlens.backends.jax._edits import JaxEditRuntime
    from torchlens.backends.registry import require_capability_implementation
    from torchlens.errors import BackendUnsupportedError

    spec = tl.backends.spec("jax")
    assert spec.capabilities.can_intervene is True
    assert require_capability_implementation(spec, "can_intervene") is JaxEditRuntime

    def model(x):
        return jnp.tanh(x + 1.0)

    x = jnp.ones((2,))
    clean = tl.trace(model, x, backend="jax")
    edited = tl.trace(model, x, backend="jax", intervene=tl.when(tl.func("tanh"), tl.zero_ablate()))
    assert float(jnp.abs(clean[clean.output_layers[0]].out).sum()) > 0.0
    assert float(jnp.abs(edited[edited.output_layers[0]].out).sum()) == 0.0
    # A malformed intervene value refuses typed; it is never accepted and ignored.
    with pytest.raises(BackendUnsupportedError, match="callable predicate"):
        tl.trace(model, x, backend="jax", intervene=object())


@pytest.mark.backend_tinygrad
def test_tinygrad_backward_flip_with_binding_refuses_at_trace() -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    import torchlens as tl
    from torchlens.errors import BackendCapabilityConformanceError

    original, opened = _spec_flipped_true("tinygrad", "backward_capture")
    tl.backends.register(opened, replace=True)
    try:

        def model(x):
            return (x + 1.0).relu()

        with pytest.raises(BackendCapabilityConformanceError, match="never\\s+dispatches"):
            tl.trace(
                model,
                Tensor([1.0, -2.0]),
                backend="tinygrad",
                capture=tl.options.CaptureOptions(backward_ready=True),
            )
    finally:
        tl.backends.register(original, replace=True)

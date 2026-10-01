"""End-to-end dispatch-consumption conformance (sol probe, live runtime).

A capability flag flipped True with a bound-but-undispatched implementation
must refuse at ``tl.trace()``, never capture while silently ignoring the
requested option. The policy-level biconditional lives in
``test_capability_truth.py``; these tests prove the refusal reaches the real
public entry on a live preview runtime (sol's original probe registered a JAX
spec with ``interventions=True`` and a dummy binding, passed ``intervene=``,
and got a capture whose output showed the intervention was ignored).
"""

from __future__ import annotations

import dataclasses

import pytest

pytestmark = pytest.mark.backend_parity


def _spec_flipped_true(name: str, flag: str):
    from torchlens.backends import get_backend_spec

    original = get_backend_spec(name)
    implementations = dict(original.capability_implementations or {})
    implementations[flag] = lambda: object()
    return original, dataclasses.replace(
        original,
        capabilities=dataclasses.replace(original.capabilities, **{flag: True}),
        capability_implementations=implementations,
    )


@pytest.mark.backend_jax
def test_jax_interventions_flip_with_binding_refuses_at_trace() -> None:
    jnp = pytest.importorskip("jax.numpy")
    import torchlens as tl
    from torchlens.backends import BackendCapabilityConformanceError, register_backend_spec

    original, opened = _spec_flipped_true("jax", "interventions")
    register_backend_spec(opened, replace=True)
    try:

        def model(x):
            return x + 1.0

        with pytest.raises(BackendCapabilityConformanceError, match="never\\s+dispatches"):
            tl.trace(model, jnp.ones((2,)), backend="jax", intervene=object())
    finally:
        register_backend_spec(original, replace=True)


@pytest.mark.backend_tinygrad
def test_tinygrad_backward_flip_with_binding_refuses_at_trace() -> None:
    pytest.importorskip("tinygrad")
    from tinygrad import Tensor

    import torchlens as tl
    from torchlens.backends import BackendCapabilityConformanceError, register_backend_spec

    original, opened = _spec_flipped_true("tinygrad", "backward_capture")
    register_backend_spec(opened, replace=True)
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
        register_backend_spec(original, replace=True)

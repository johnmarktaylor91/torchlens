"""Purity/state harness core (M(oracles) build item 7; F36 oracle wave 1).

Every invocation template runs under the four postconditions: parameters and
buffers byte-identical, torch global RNG untouched, no module hooks left
attached, and the model still forward-computes identically. Each template's
POSITIVE CONTROL runs first (D7: no verdict off a dead channel), and the
exception-stage arm injects a mid-forward failure and re-checks the same
postconditions (read-after-failure).

Instance-``__dict__`` contamination (the tl_* forward attrs) is FORK-A
territory (state contract UNSET) and is pinned separately by the compo
sweep's SG#33 cell -- this harness asserts the contracts all three FORK-A
branches share.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
import torch

from tests.oracles._invocation_templates import SEED_TEMPLATES, InvocationTemplate

pytestmark = [pytest.mark.smoke]


def _state_digest(model: torch.nn.Module) -> str:
    """Byte digest of every parameter and buffer, name-keyed."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _hook_count(model: torch.nn.Module) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks) + len(module._backward_hooks)
        for module in model.modules()
    )


def _shared_fixture_model() -> torch.nn.Module:
    """The templates' own module-level fixture recipe (CF-022: picklable)."""

    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()


@pytest.mark.parametrize("template", SEED_TEMPLATES, ids=lambda t: t.template_id)
def test_template_positive_control_channel_is_alive(template: InvocationTemplate) -> None:
    """D7: the measurement channel must prove itself before any verdict."""

    template.positive_control()


@pytest.mark.parametrize("template", SEED_TEMPLATES, ids=lambda t: t.template_id)
def test_template_invocation_restores_model_and_global_state(
    template: InvocationTemplate,
) -> None:
    """One real invocation; params/buffers/RNG/hooks/forward all restored."""

    reference_model = _shared_fixture_model()
    state_before = _state_digest(reference_model)
    torch.manual_seed(1)
    example = torch.randn(2, 4)
    with torch.no_grad():
        forward_before = reference_model(example).clone()
    rng_before = torch.get_rng_state().clone()
    grad_before = torch.is_grad_enabled()

    template.invoke()

    assert torch.is_grad_enabled() == grad_before, f"{template.door} flipped grad mode"
    assert torch.equal(torch.get_rng_state(), rng_before), (
        f"{template.door} consumed/replaced the GLOBAL torch RNG state"
    )
    # The template builds its own fixture instance; the reference instance
    # proves no cross-instance/global leakage, and its own state is exact.
    assert _state_digest(reference_model) == state_before
    assert _hook_count(reference_model) == 0
    with torch.no_grad():
        assert torch.equal(reference_model(example), forward_before)


def test_exception_stage_injection_restores_state_and_reads_stay_sane() -> None:
    """A mid-forward failure leaves the model restored AND still usable
    (read-after-failure: the next capture of the same model is COMPLETE)."""

    import torchlens as tl

    class _FailsOnFlag(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)
            self.fail = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.lin(x)
            if self.fail:
                raise RuntimeError("planted mid-forward failure (purity harness)")
            return torch.relu(hidden)

    torch.manual_seed(0)
    model = _FailsOnFlag().eval()
    example = torch.randn(2, 4)
    state_before = _state_digest(model)
    hooks_before = _hook_count(model)

    model.fail = True
    with pytest.raises(RuntimeError, match="planted mid-forward"):
        tl.trace(model, example)

    assert _state_digest(model) == state_before, (
        "a FAILED capture left the model's parameters/buffers mutated"
    )
    assert _hook_count(model) == hooks_before, "a FAILED capture leaked module hooks"

    model.fail = False
    recovery = tl.trace(model, example)
    try:
        assert recovery.outcome.status.name == "COMPLETE", (
            "the model cannot be re-captured after a failed capture --"
            " failure-path state restoration is incomplete"
        )
    finally:
        recovery.cleanup()
        tl.release_model(model)


def test_repeat_invocation_is_digest_stable() -> None:
    """Two identical captures produce identical structural digests (the
    warm-vs-cold history cell H-CLEAN x H-CLEAN, cheapest pair)."""

    import torchlens as tl

    def capture_digest() -> str:
        torch.manual_seed(0)
        model = _shared_fixture_model()
        torch.manual_seed(1)
        trace = tl.trace(model, torch.randn(2, 4))
        try:
            digest = hashlib.sha256()
            for label in trace.layer_labels:
                digest.update(label.encode())
                payload = trace[label].out
                if isinstance(payload, torch.Tensor):
                    digest.update(payload.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        finally:
            trace.cleanup()

    assert capture_digest() == capture_digest(), (
        "identical fixture captures differ -- a hidden state axis is leaking between invocations"
    )


EXTRA_TEMPLATE_DOORS: tuple[tuple[str, Any], ...] = (
    ("torchlens.summary", lambda tl, model, x: tl.summary(model, x)),
    (
        "torchlens.sweep",
        lambda tl, model, x: tl.sweep(
            model, x, at=tl.func("relu"), values=[0.0], include_baseline=True
        ),
    ),
    (
        "torchlens.validate",
        lambda tl, model, x: tl.validate(model, x, scope="forward"),
    ),
)


@pytest.mark.parametrize(
    "door_name,invoke", EXTRA_TEMPLATE_DOORS, ids=[name for name, _ in EXTRA_TEMPLATE_DOORS]
)
def test_wave1_extra_doors_restore_model_state(door_name: str, invoke: Any) -> None:
    """Wave-1 template growth (item 7): summary/sweep/validate doors under
    the same postconditions the seed templates carry."""

    import torchlens as tl

    model = _shared_fixture_model()
    torch.manual_seed(1)
    example = torch.randn(2, 4)
    state_before = _state_digest(model)
    with torch.no_grad():
        forward_before = model(example).clone()

    invoke(tl, model, example)

    assert _state_digest(model) == state_before, f"{door_name} mutated model state"
    assert _hook_count(model) == 0, f"{door_name} leaked hooks"
    with torch.no_grad():
        assert torch.equal(model(example), forward_before), (
            f"{door_name} changed what the model computes"
        )
    tl.release_model(model)

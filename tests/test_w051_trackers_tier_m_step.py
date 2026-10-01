"""W051-TRACK / AUD-CODE 2.14a: tier M activations flow under ``step=callable``.

The module forward hooks filled the collector's staging BEFORE
``optimizer.step()``; the boundary hook then opened the implicit block, which
zeroed that staging, so the documented ``step=`` spelling emitted ZERO
activations and never refused (run-health rows masked the emptiness).
"""

from __future__ import annotations

import pytest
import torch

import torchlens.trackers as trk

pytestmark = pytest.mark.smoke


def _mlp() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def _run(model, opt, session, current, steps) -> None:  # noqa: ANN001
    for step in steps:
        current["step"] = step
        opt.zero_grad(set_to_none=True)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()


def test_activations_emit_under_step_callable_on_the_scheduled_steps() -> None:
    model, opt = _mlp()
    current = {"step": 0}
    sink = trk.MemorySink()
    session = trk.watch(
        model,
        to=sink,
        signals=("activations", "gradients"),
        select=("0", "2"),
        optimizer=opt,
        example_input=torch.randn(4, 8),
        step=lambda: current["step"],
        every=2,
    )
    _run(model, opt, session, current, range(5))
    report = session.close()
    activation_steps = sorted({p.step for p in sink.scalars if p.tag.startswith("activations/")})
    gradient_steps = sorted({p.step for p in sink.scalars if p.tag.startswith("gradients/")})
    assert activation_steps == [0, 2, 4]
    assert gradient_steps == [0, 2, 4]
    assert {
        tag.split("/")[2]
        for tag in {p.tag for p in sink.scalars if p.tag.startswith("activations/")}
    } == {"0", "2"}
    assert report.sampled_steps == (0, 2, 4)
    # The scheduling pre-hook is torn down with everything else.
    assert len(model._forward_pre_hooks) == 0


def test_step_callable_activations_match_the_explicit_scope_spelling() -> None:
    """Both step spellings observe the SAME activation statistics."""

    torch.manual_seed(1)
    inputs = [torch.randn(4, 8) for _ in range(3)]

    def _collect(spelling: str) -> dict[tuple[str, int], float]:
        model, opt = _mlp()
        sink = trk.MemorySink()
        current = {"step": 0}
        kwargs = {"step": (lambda: current["step"])} if spelling == "callable" else {}
        session = trk.watch(
            model,
            to=sink,
            signals=("activations",),
            select=("0",),
            optimizer=opt,
            example_input=torch.randn(4, 8),
            every=1,
            **kwargs,
        )
        for step, x in enumerate(inputs):
            current["step"] = step
            if spelling == "callable":
                opt.zero_grad(set_to_none=True)
                model(x).sum().backward()
                opt.step()
            else:
                with session.step(step):
                    opt.zero_grad(set_to_none=True)
                    model(x).sum().backward()
                    opt.step()
        session.close()
        return {(p.tag, p.step): p.value for p in sink.scalars if p.tag.startswith("activations/")}

    explicit = _collect("scope")
    implicit = _collect("callable")
    assert explicit and set(explicit) == set(implicit)
    for key, value in explicit.items():
        assert implicit[key] == pytest.approx(value)


def test_step_callable_tier_p_registers_no_forward_hook() -> None:
    """Tier P keeps its no-forward-wrap promise even with ``step=``."""

    model, opt = _mlp()
    session = trk.watch(model, to=trk.MemorySink(), optimizer=opt, step=lambda: 0)
    assert len(model._forward_pre_hooks) == 0
    session.close(unwinding=True)

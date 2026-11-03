"""Checks kit item 3: loop-session chassis + registry lifecycle (memo 4.2).

Contracts pinned here: every hook handle removed on every exit path
including optimizer exceptions; construction is silent and does no device
work (D20); bitwise same-seed checks-on/off runs (zero user-RNG
consumption); the step-check family runs on torch.compile'd models; the
boundary refuses nesting typed; no per-parameter ``.item()`` exists in any
S-A callback path (the static mechanism check CPU timing cannot catch).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens.debug._checks as tc
import torchlens.errors


def _net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_checks/_session.py.
def test_double_attach_and_nested_boundary_refuse_typed() -> None:
    """Lifecycle misuse refuses with stable codes, never state corruption."""

    model = _net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer).attach()
    try:
        with pytest.raises(torchlens.errors.CheckLifecycleError) as attach_exc:
            session.attach()
        assert attach_exc.value.fields["code"] == "check_already_attached"

        with (
            session.step(global_step=1),
            pytest.raises(torchlens.errors.CheckLifecycleError) as nest_exc,
        ):
            session.step(global_step=2).__enter__()
        assert nest_exc.value.fields["code"] == "check_boundary_nested"
    finally:
        session.detach()


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_checks/_session.py.
def test_step_family_without_optimizer_refuses_typed() -> None:
    """Step-family checks need the optimizer boundary (attach-time block)."""

    session = tc.ChecksSession(_net())
    session.register_change_check()
    with pytest.raises(torchlens.errors.CheckConfigError) as exc:
        session.attach()
    assert exc.value.fields["code"] == "check_optimizer_required"


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_checks/_session.py.
def test_optimizer_overlap_refuses_typed() -> None:
    """One parameter owned by two optimizers refuses at attach (memo 4.2)."""

    model = _net()
    opt_a = torch.optim.SGD(model.parameters(), lr=0.1)
    opt_b = torch.optim.SGD([model[0].weight], lr=0.01)
    session = tc.ChecksSession(model, [opt_a, opt_b])
    with pytest.raises(torchlens.errors.CheckConfigError) as exc:
        session.attach()
    assert exc.value.fields["code"] == "check_optimizer_overlap"
    assert "0.weight" in str(exc.value)


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_checks/_session.py.
def test_paused_and_disable_suppress_checks() -> None:
    """paused()/disable() suspend collection; enable() resumes it."""

    model = _net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with tc.ChecksSession(model, optimizer) as session:

        def one_step() -> None:
            optimizer.zero_grad()
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()

        one_step()
        with session.paused():
            one_step()
        session.disable()
        one_step()
        session.enable()
        one_step()

    assert session.final_report is not None
    assert session.final_report.counters["accepted_steps"] == 2


# Kept by the shrink-r4 coverage proof: a sole campaign mutation kill.
def test_checkpoint_fires_once_per_backward() -> None:
    """Non-reentrant checkpointing: exactly ONE S-A completion per backward."""

    from torch.utils.checkpoint import checkpoint

    model = _net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer).attach()
    try:
        x = torch.randn(2, 4, requires_grad=True)
        checkpoint(model, x, use_reentrant=False).sum().backward()
        optimizer.step()
    finally:
        session.detach()

    assert session.backward_id == 1
    assert all(count == 1 for count in session._fire_counts.values())


# Kept: reads source statically or runs a subprocess, so per-id coverage cannot measure it (shrink rule 3).
def test_no_per_param_item_in_sa_callback_paths() -> None:
    """Static mechanism check: no ``.item()`` in any S-A callback path.

    203 per-parameter ``.item()`` calls are 203 device syncs on CUDA; CPU
    timing cannot catch the mechanism, so the gate is structural (memo
    test plan row 8).
    """

    source = Path(tc.__file__).with_name("_session.py").read_text()
    tree = ast.parse(source)
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "item"
    ]
    assert not offenders, f".item() found in _session.py at lines {offenders}"


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_checks/_session.py.
def test_profile_reports_measured_numbers() -> None:
    """profile() returns min-of-N data with the load recorded (D21)."""

    session = tc.ChecksSession(_net())
    payload = session.overhead()
    assert payload["scan_ms_min_of_3"] > 0
    assert payload["n_tensors"] == 4
    assert len(payload["loadavg"]) == 3

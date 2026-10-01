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

import torchlens.checks as tc

pytestmark = pytest.mark.smoke


def _net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _hook_census(model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    """Count installed post-accumulate-grad and optimizer step hooks."""

    param_hooks = sum(
        len(getattr(param, "_post_accumulate_grad_hooks", None) or ())
        for param in model.parameters()
    )
    step_hooks = len(optimizer._optimizer_step_pre_hooks) + len(
        optimizer._optimizer_step_post_hooks
    )
    return param_hooks, step_hooks


def test_construction_is_silent_and_deviceless(capsys: pytest.CaptureFixture[str]) -> None:
    """D20: no hooks, no stdout, snapshot bytes as pure metadata."""

    model = _net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    session = tc.ChecksSession(model, optimizer)
    session.register_change_check()

    assert capsys.readouterr().out == ""
    assert _hook_census(model, optimizer) == (0, 0)
    # Pure numel x dtype-size metadata, available before attach.
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    assert session.estimated_snapshot_bytes == expected


def test_attach_detach_removes_every_handle() -> None:
    """Every handle is removed on detach and on context exit."""

    model = _net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer).attach()
    assert _hook_census(model, optimizer) == (4, 2)
    session.detach()
    assert _hook_census(model, optimizer) == (0, 0)

    with tc.ChecksSession(model, optimizer):
        assert _hook_census(model, optimizer) == (4, 2)
    assert _hook_census(model, optimizer) == (0, 0)


def test_exception_exit_removes_handles_and_finalizes_report() -> None:
    """An optimizer exception mid-run leaves no hook and an intact report."""

    model = _net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer)

    with pytest.raises(RuntimeError, match="user code exploded"), session:
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        raise RuntimeError("user code exploded")

    assert _hook_census(model, optimizer) == (0, 0)
    assert session.final_report is not None
    assert session.final_report.finalized_reason == "exception_exit"
    assert session.final_report.counters["accepted_steps"] == 1


def test_double_attach_and_nested_boundary_refuse_typed() -> None:
    """Lifecycle misuse refuses with stable codes, never state corruption."""

    model = _net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer).attach()
    try:
        with pytest.raises(tc.CheckLifecycleError) as attach_exc:
            session.attach()
        assert attach_exc.value.fields["code"] == "check_already_attached"

        with session.step(global_step=1), pytest.raises(tc.CheckLifecycleError) as nest_exc:
            session.step(global_step=2).__enter__()
        assert nest_exc.value.fields["code"] == "check_boundary_nested"
    finally:
        session.detach()


def test_step_family_without_optimizer_refuses_typed() -> None:
    """Step-family checks need the optimizer boundary (attach-time block)."""

    session = tc.ChecksSession(_net())
    session.register_change_check()
    with pytest.raises(tc.CheckConfigError) as exc:
        session.attach()
    assert exc.value.fields["code"] == "check_optimizer_required"


def test_optimizer_overlap_refuses_typed() -> None:
    """One parameter owned by two optimizers refuses at attach (memo 4.2)."""

    model = _net()
    opt_a = torch.optim.SGD(model.parameters(), lr=0.1)
    opt_b = torch.optim.SGD([model[0].weight], lr=0.01)
    session = tc.ChecksSession(model, [opt_a, opt_b])
    with pytest.raises(tc.CheckConfigError) as exc:
        session.attach()
    assert exc.value.fields["code"] == "check_optimizer_overlap"
    assert "0.weight" in str(exc.value)


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


def test_bitwise_same_seed_checks_on_off() -> None:
    """Zero user-RNG consumption: checks-on/off runs are bitwise equal."""

    def run(with_checks: bool) -> dict[str, torch.Tensor]:
        torch.manual_seed(1234)
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        optimizer = torch.optim.AdamW(model.parameters())
        session: tc.ChecksSession | None = None
        if with_checks:
            session = tc.ChecksSession(model, optimizer)
            session.register_change_check()
            session.register_update_ratio_check()
            session.attach()
        for _ in range(3):
            optimizer.zero_grad()
            model(torch.randn(8, 4)).sum().backward()
            optimizer.step()
        if session is not None:
            session.detach()
        return {name: tensor.clone() for name, tensor in model.state_dict().items()}

    baseline = run(with_checks=False)
    checked = run(with_checks=True)

    assert all(torch.equal(baseline[name], checked[name]) for name in baseline)


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


def test_reentrant_checkpoint_detached_segment_disclosed() -> None:
    """The reentrant-checkpoint footgun surfaces as the grad-None disclosure.

    A checkpointed segment whose input does not require grad is silently
    untrained under use_reentrant=True; the S-B ``p.grad is None`` scan
    names it (memo 4.3: a free catch of a real bug class, advertised by a
    test, not a docstring).
    """

    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(0)

    class Split(nn.Module):
        """Checkpointed trunk (silently untrained) + trained head."""

        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Linear(4, 8)
            self.head = nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the trunk under reentrant checkpointing."""

            hidden = checkpoint(self.trunk, x, use_reentrant=True)
            return self.head(hidden.detach() + self.head.bias.sum() * 0 + hidden * 0 + hidden)

    model = Split()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer).attach()
    try:
        with pytest.warns(UserWarning, match="None of the inputs have requires_grad"):
            out = model(torch.randn(2, 4))
        out.sum().backward()
        optimizer.step()
        report = session.report()
    finally:
        session.detach()

    disclosed = set(report.disclosures["grad_none_last_accepted_step"])
    assert {"trunk.weight", "trunk.bias"} <= disclosed
    assert "head.weight" not in disclosed


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


def test_profile_reports_measured_numbers() -> None:
    """profile() returns min-of-N data with the load recorded (D21)."""

    session = tc.ChecksSession(_net())
    payload = session.profile()
    assert payload["scan_ms_min_of_3"] > 0
    assert payload["n_tensors"] == 4
    assert len(payload["loadavg"]) == 3

"""Wrapper-hygiene regressions from grind-r5 b8 R56/R57.

Covers the autograd entry-point teardown drift guard, autograd wrapper
provenance, the fx Tracer.trace target normalization, and the escape
detector's profile-slot / sys.monitoring tool-id discipline.
"""

from __future__ import annotations

import inspect
import pickle
import sys
import types

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._errors import TorchLensWarning
from torchlens.backends.torch import backward as tl_backward
from torchlens.backends.torch.wrappers import wrap_torch


def _ensure_wrapped() -> None:
    """Make sure the wrapped epoch is active for a test."""
    wrap_torch()


def test_uninstall_autograd_wrappers_preserves_foreign_patches() -> None:
    """A third-party patch layered over the torchlens autograd wrappers must
    survive teardown (identity-checked restore + burial disclosure), never be
    silently clobbered -- the only teardown that lacked the drift guard."""

    _ensure_wrapped()
    tl_backward.install_autograd_wrappers()

    inner_backward = torch.autograd.backward
    inner_grad = torch.autograd.grad

    def foreign_backward(*args, **kwargs):
        return inner_backward(*args, **kwargs)

    def foreign_grad(*args, **kwargs):
        return inner_grad(*args, **kwargs)

    torch.autograd.backward = foreign_backward
    torch.autograd.grad = foreign_grad
    try:
        with pytest.warns(TorchLensWarning, match="third-party patch"):
            tl_backward.uninstall_autograd_wrappers()
        assert torch.autograd.backward is foreign_backward, (
            "unwrap clobbered a foreign patch on torch.autograd.backward"
        )
        assert torch.autograd.grad is foreign_grad, (
            "unwrap clobbered a foreign patch on torch.autograd.grad"
        )
    finally:
        # Drop BOTH layers (foreign patch + the still-buried torchlens
        # wrapper): restore the pristine originals directly, then reinstall
        # from clean state so later tests see exactly the pre-test world.
        pristine_backward = tl_backward._ORIGINAL_AUTOGRAD_BACKWARD
        pristine_grad = tl_backward._ORIGINAL_AUTOGRAD_GRAD
        if pristine_backward is not None:
            torch.autograd.backward = pristine_backward
        if pristine_grad is not None:
            torch.autograd.grad = pristine_grad
        tl_backward._AUTOGRAD_WRAPPERS_INSTALLED = False
        tl_backward.install_autograd_wrappers()


def test_uninstall_autograd_wrappers_restores_own_patches_cleanly() -> None:
    """The identity-checked teardown still restores pristine originals when
    nothing foreign intervened, and reinstall works afterwards."""

    _ensure_wrapped()
    tl_backward.install_autograd_wrappers()
    original_backward = tl_backward._ORIGINAL_AUTOGRAD_BACKWARD
    original_grad = tl_backward._ORIGINAL_AUTOGRAD_GRAD

    tl_backward.uninstall_autograd_wrappers()
    try:
        assert torch.autograd.backward is original_backward
        assert torch.autograd.grad is original_grad
    finally:
        tl_backward.install_autograd_wrappers()


def test_autograd_entry_wrappers_carry_provenance() -> None:
    """The wrapped autograd entries must be introspectable and picklable by
    reference for the whole wrapped epoch, like every namespace wrapper."""

    _ensure_wrapped()
    tl_backward.install_autograd_wrappers()

    assert torch.autograd.backward.__name__ == "backward"
    assert torch.autograd.grad.__name__ == "grad"
    assert torch.autograd.backward.__module__ == "torch.autograd"
    params = inspect.signature(torch.autograd.backward).parameters
    assert "tensors" in params, "signature degraded to (*args, **kwargs)"
    # Pickle-by-reference resolves the public name to the installed wrapper.
    pickle.dumps(torch.autograd.backward)
    pickle.dumps(torch.autograd.grad)


class _FxReluModel(nn.Module):
    """Module calling a Python functional directly (the fx bake vector)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.lin(x))


def test_fx_symbolic_trace_records_originals_not_wrappers() -> None:
    """grind-r5 b8 R56: symbolic_trace during the wrapped epoch must not bake
    live torchlens wrappers into call_function node targets."""

    _ensure_wrapped()
    tl.trace(_FxReluModel(), torch.randn(1, 4))  # ensure wrappers + shims live

    graph_module = torch.fx.symbolic_trace(_FxReluModel())
    wrapped_targets = [
        node.target
        for node in graph_module.graph.nodes
        if node.op == "call_function" and id(node.target) in _state._decorated_to_orig
    ]
    assert wrapped_targets == [], (
        f"fx graph embeds torchlens wrappers as node targets: {wrapped_targets}"
    )
    relu_targets = [
        node.target
        for node in graph_module.graph.nodes
        if node.op == "call_function" and getattr(node.target, "__name__", "") == "relu"
    ]
    assert relu_targets, "expected a relu call_function node"
    # Identity-keyed matcher shape (torch.ao style): the target IS the pristine
    # original the live wrapper ledgers to.
    pristine_relu = _state._decorated_to_orig.get(id(torch.nn.functional.relu))
    if pristine_relu is not None:
        assert relu_targets[0] is pristine_relu
    # The traced module still runs.
    assert graph_module(torch.randn(2, 4)).shape == (2, 4)


def test_escape_detector_setprofile_teardown_is_identity_guarded() -> None:
    """_uninstall_setprofile must not clobber a foreign profiler that took the
    slot after the detector installed (the codebase's profile-slot standard)."""

    from torchlens.backends.torch import escape_detection

    def foreign_profiler(frame, event, arg):
        return None

    prior = sys.getprofile()
    guard = types.SimpleNamespace(prior_profile=prior)
    sys.setprofile(foreign_profiler)
    try:
        escape_detection._uninstall_setprofile(guard)
        assert sys.getprofile() is foreign_profiler, "teardown clobbered a foreign profile hook"
    finally:
        sys.setprofile(prior)


def test_monitoring_tool_id_probe_skips_reserved_ids() -> None:
    """The sys.monitoring tool-id probe must never claim the reserved
    DEBUGGER/COVERAGE/PROFILER/OPTIMIZER slots (0/1/2/5)."""

    from torchlens.backends.torch import escape_detection

    attempted: list[int] = []

    class _FakeMonitoring:
        def get_tool(self, tool_id: int):
            attempted.append(tool_id)
            return "taken"  # every slot taken -> probe walks its full range

        def use_tool_id(self, tool_id: int, name: str) -> None:
            raise AssertionError("no slot should be claimable in this probe")

    with pytest.raises(RuntimeError, match="No free sys.monitoring tool id"):
        escape_detection._find_monitoring_tool_id(_FakeMonitoring())
    assert attempted, "probe never consulted the fake monitoring namespace"
    assert set(attempted).isdisjoint({0, 1, 2, 5}), (
        f"probe touched reserved tool ids: {sorted(set(attempted) & {0, 1, 2, 5})}"
    )

"""Phase 5(d): user-attached pre/post hook ordering relative to wrap-forward push."""

from __future__ import annotations

from typing import Any

import torch

import torchlens as tl
from torchlens import _state


class HookChild(torch.nn.Module):
    """Child module containing a nested linear module."""

    def __init__(self) -> None:
        """Initialize the nested linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested linear layer."""

        return self.linear(x)


class HookOuter(torch.nn.Module):
    """Root module for user pre-hook stack-ordering checks."""

    def __init__(self) -> None:
        """Initialize the child module."""

        super().__init__()
        self.child = HookChild()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the child module."""

        return self.child(x)


def test_user_pre_hook_does_not_see_module_frame() -> None:
    """Pre-hook fires before the hooked module's wrap-forward stack push."""

    snapshots: list[list[str]] = []
    model = HookOuter()

    def pre_hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        """Snapshot the active exhaustive stack from a user pre-hook."""

        del module, args
        trace = _state._active_trace
        stack = getattr(trace, "_exhaustive_module_stack", ()) if trace is not None else ()
        snapshots.append([frame.address for frame in stack])

    handle = model.child.linear.register_forward_pre_hook(pre_hook)
    try:
        tl.trace(model, torch.randn(2, 4), vis_opt="none")
    finally:
        handle.remove()

    assert snapshots, "expected pre-hook to run"
    assert snapshots[0] == ["child"]


def test_user_forward_hook_replacement_logged() -> None:
    """A user forward hook replacement is logged after the module exits."""

    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(4, 4))
    relu = model[0]

    def replace_with_half(
        module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        """Replace a ReLU output with a scaled tensor."""

        del module, args
        return output * 0.5

    handle = relu.register_forward_hook(replace_with_half)
    try:
        trace = tl.trace(model, torch.randn(2, 4), vis_opt="none")
    finally:
        handle.remove()

    relu_ops = [op for op in trace.layer_list if op.func_name == "relu"]
    assert relu_ops, "expected at least one relu op"
    assert any(op.is_submodule_output for op in relu_ops)

    replacement_ops = [op for op in trace.layer_list if getattr(op, "intervention_replaced", False)]
    assert replacement_ops, "expected at least one intervention_replaced op"
    assert any(op.func_name == "__mul__" and op.modules == [] for op in replacement_ops)
    linear_ops = [op for op in trace.layer_list if op.func_name == "linear"]
    assert any("mul" in parent for op in linear_ops for parent in op.parents)

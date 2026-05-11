"""Phase 5(c): exception safety -- pop on exception."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens import _state


class RaisingChild(torch.nn.Module):
    """Child that raises after entering a nested module."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one op, then raise intentionally."""

        _ = self.linear(x)
        raise RuntimeError("intentional test failure")


class Outer(torch.nn.Module):
    """Root module that delegates to a raising child."""

    def __init__(self, child: torch.nn.Module | None = None) -> None:
        """Initialize the child module."""

        super().__init__()
        self.child = child if child is not None else RaisingChild()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate forward to the child."""

        return self.child(x)


class NonRaisingChild(torch.nn.Module):
    """Child used to verify a fresh capture after failure."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear layer normally."""

        return self.linear(x)


def test_exception_unwinds_module_stack() -> None:
    """A failed forward must not corrupt the next capture."""

    model = Outer()
    with pytest.raises(RuntimeError, match="intentional"):
        tl.trace(model, torch.randn(2, 4), vis_opt="none")

    assert _state._logging_enabled is False
    assert _state._active_trace is None

    trace = tl.trace(Outer(NonRaisingChild()), torch.randn(2, 4), vis_opt="none")
    linear_ops = [op for op in trace.layer_list if "linear" in op.layer_label]
    assert linear_ops, "expected at least one linear op"
    module_calls = {str(getattr(op, "module", "")) for op in linear_ops}
    assert any(module_call.endswith(":1") for module_call in module_calls), (
        f"expected first module pass, got {module_calls}"
    )

"""Phase 5(b): _tag_untagged_buffers ordering relative to stack push."""

from __future__ import annotations

import torch

import torchlens as tl


class BufferConsumer(torch.nn.Module):
    """Child module that consumes a dynamically registered parent buffer."""

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Scale input by the provided buffer tensor."""

        return x * scale


class DynamicBufferUser(torch.nn.Module):
    """Register a buffer during forward before calling a child module."""

    def __init__(self) -> None:
        """Initialize child modules."""

        super().__init__()
        self.consumer = BufferConsumer()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create ``scale`` once, then use it through a child module."""

        if not hasattr(self, "scale"):
            self.register_buffer("scale", torch.ones(4))
        return self.proj(self.consumer(x, self.scale))


class BufferParent(torch.nn.Module):
    """Root wrapper so the dynamic buffer module has a stack frame."""

    def __init__(self) -> None:
        """Initialize the dynamic buffer user."""

        super().__init__()
        self.dynamic = DynamicBufferUser()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the dynamic buffer user."""

        return self.dynamic(x)


def test_dynamic_buffer_module_attribution() -> None:
    """Dynamic buffer creation inside forward attributes to the registering module."""

    torch.manual_seed(0)
    model = BufferParent()
    trace = tl.trace(model, torch.randn(2, 4))

    creation_ops = [op for op in trace.layer_list if op.func_name == "ones"]
    assert creation_ops, "expected the dynamic buffer creation op to be captured"
    assert any("dynamic" in str(module) for op in creation_ops for module in op.modules)
    assert any("consumer" in str(module) for op in trace.layer_list for module in op.modules)
    assert any("mul" in op.layer_label for op in trace.layer_list), "expected child buffer use"

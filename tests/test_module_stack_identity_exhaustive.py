"""Phase 5(a): identity / pass-through ordering in EXHAUSTIVE mode."""

from __future__ import annotations

import torch

import torchlens as tl


class IdentityChild(torch.nn.Module):
    """Module with an identity child followed by a parameterized child."""

    def __init__(self) -> None:
        """Initialize child modules."""

        super().__init__()
        self.identity = torch.nn.Identity()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the identity pass-through before projection."""

        y = self.identity(x)
        return self.proj(y)


def test_identity_module_stack_frame_visible() -> None:
    """The identity-synthesized op must record identity's module frame."""

    torch.manual_seed(0)
    model = IdentityChild()
    trace = tl.trace(model, torch.randn(2, 4), vis_opt="none")
    identity_ops = [op for op in trace.layer_list if "identity" in op.layer_label]
    assert identity_ops, "expected at least one identity-tagged op"
    for op in identity_ops:
        modules = list(getattr(op, "modules", []))
        assert any("identity" in str(module) for module in modules), (
            f"identity op {op.layer_label} should include identity module in modules: got {modules}"
        )

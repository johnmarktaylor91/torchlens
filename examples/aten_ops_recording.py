"""Record the ATen operations behind each captured op with ``CaptureOptions(aten_ops=True)``.

The recorder is opt-in: a default capture installs no dispatch mode and its
accessors refuse typed, so the script shows both sides.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class NormAttend(nn.Module):
    """Layer norm, self-attention over the sequence, then GELU."""

    def __init__(self) -> None:
        """Initialize the norm."""

        super().__init__()
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize, attend, activate."""

        h = self.norm(x)
        attended = torch.nn.functional.scaled_dot_product_attention(h, h, h)
        return torch.nn.functional.gelu(attended)


def main() -> None:
    """Capture once with the switch on, once off, and print what each serves."""

    torch.manual_seed(0)
    model = NormAttend()
    x = torch.randn(2, 3, 8)

    trace = tl.trace(model, x, capture=CaptureOptions(aten_ops=True))
    print("has_aten_ops:", trace.has_aten_ops)
    print("rows recorded:", len(trace.aten_ops))
    for op in trace.ops:
        operators = [f"{row.operator}.{row.overload}" for row in op.aten_ops]
        print(f"  {op.layer_label:<28} -> {operators}")
    print("gaps:", trace.aten_gaps)

    plain = tl.trace(model, x)
    print("default capture has_aten_ops:", plain.has_aten_ops)
    try:
        plain.aten_ops
    except tl.errors.AtenOpsUnavailableError as exc:
        print("refused:", exc.fields["code"])
        print(exc)


if __name__ == "__main__":
    main()

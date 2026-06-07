"""Minimal unified capture-path examples."""

from __future__ import annotations

from pathlib import Path
import tempfile
import warnings

import torch
from torch import nn

import torchlens as tl


class Encoder(nn.Module):
    """Small encoder module with visible convolution and ReLU ops."""

    def __init__(self) -> None:
        """Initialize encoder layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder."""

        return self.relu(self.conv(x))


class TinyNet(nn.Module):
    """Small model with convolution, module, and ReLU sites."""

    def __init__(self) -> None:
        """Initialize deterministic layers."""

        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(2 * 4 * 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        hidden = self.encoder(x)
        flat = hidden.flatten(1)
        return torch.relu(self.head(flat))


def main() -> None:
    """Run unified capture examples."""

    torch.manual_seed(0)
    warnings.filterwarnings(
        "ignore",
        message="TorchLens intervention-ready output traversal does not support UntypedStorage.*",
        category=UserWarning,
    )
    model = TinyNet().eval()
    x = torch.randn(1, 1, 4, 4)

    relu_trace = tl.trace(model, x, save=tl.func("relu"))
    print("saved relu ops:", [op.label for op in relu_trace.layer_list if op.has_saved_activation])

    windowed = tl.trace(
        model,
        x,
        save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
        lookback=4,
        lookback_payload_policy="detached_raw",
    )
    print(
        "windowed saves:",
        [op.label for op in windowed.layer_list if op.has_saved_activation],
    )

    patched = tl.trace(
        model,
        x,
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.scale(0.5)),
    )
    print("intervened relu count:", sum(op.intervention_replaced for op in patched.layer_list))

    recording = tl.record(model, x, save=tl.func("conv2d"))
    print("recorded ops:", len(recording.records))
    full_structure = recording.to_trace()
    print("to_trace layers:", len(full_structure.layer_list))

    with tempfile.TemporaryDirectory(prefix="torchlens_selective_") as tmpdir:
        streamed = tl.trace(
            model,
            x,
            save=tl.in_module("encoder"),
            storage=tl.to_disk(Path(tmpdir) / "streamed.tlspec"),
        )
        print(
            "streamed saves:", [op.label for op in streamed.layer_list if op.has_saved_activation]
        )


if __name__ == "__main__":
    main()

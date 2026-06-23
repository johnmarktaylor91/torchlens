"""Render a small smart-collapse visual review set to /tmp."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torchlens as tl


OUT_DIR = Path("/tmp/autocollapse/sample_renders")


class ConvBlock(torch.nn.Module):
    """Small convolutional block."""

    def __init__(self, channels: int) -> None:
        """Initialize layers."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.net(x)


class TinyCNN(torch.nn.Module):
    """Compact CNN."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.stem = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.blocks = torch.nn.Sequential(ConvBlock(8), ConvBlock(8), ConvBlock(8))
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        x = self.blocks(self.stem(x)).mean(dim=(2, 3))
        return self.head(x)


class ResidualBlock(torch.nn.Module):
    """Residual MLP block."""

    def __init__(self, width: int) -> None:
        """Initialize layers."""

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return self.net(x) + x


class TinyResNet(torch.nn.Module):
    """Residual stack."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.blocks = torch.nn.ModuleList([ResidualBlock(16) for _ in range(5)])
        self.head = torch.nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        for block in self.blocks:
            x = block(x)
        return self.head(x)


class TinyUNet(torch.nn.Module):
    """Small U-Net-style encoder/decoder."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.enc1 = ConvBlock(4)
        self.enc2 = ConvBlock(4)
        self.dec1 = ConvBlock(8)
        self.out = torch.nn.Conv2d(8, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        skip = self.enc1(x)
        deep = torch.nn.functional.avg_pool2d(self.enc2(skip), 2)
        up = torch.nn.functional.interpolate(deep, scale_factor=2, mode="nearest")
        return self.out(self.dec1(torch.cat([skip, up], dim=1)))


class TinyInception(torch.nn.Module):
    """Small multi-branch block."""

    def __init__(self) -> None:
        """Initialize branches."""

        super().__init__()
        self.a = torch.nn.Conv2d(4, 4, 1)
        self.b = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 1), torch.nn.Conv2d(4, 4, 3, padding=1))
        self.c = torch.nn.Sequential(
            torch.nn.AvgPool2d(3, stride=1, padding=1), torch.nn.Conv2d(4, 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return torch.cat([self.a(x), self.b(x), self.c(x)], dim=1)


class TinyTransformer(torch.nn.Module):
    """Small transformer encoder."""

    def __init__(self) -> None:
        """Initialize encoder."""

        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(16, 4, dim_feedforward=32, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder."""

        return self.encoder(x)


class TinyRNN(torch.nn.Module):
    """Small recurrent model."""

    def __init__(self) -> None:
        """Initialize recurrent layers."""

        super().__init__()
        self.rnn = torch.nn.GRU(8, 8, batch_first=True)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run recurrent model."""

        y, _ = self.rnn(x)
        return self.head(y[:, -1])


def _cases() -> list[tuple[str, torch.nn.Module, torch.Tensor]]:
    """Return sample models and inputs."""

    return [
        ("cnn", TinyCNN(), torch.randn(1, 3, 16, 16)),
        ("resnet", TinyResNet(), torch.randn(1, 16)),
        ("unet", TinyUNet(), torch.randn(1, 4, 16, 16)),
        ("inception", TinyInception(), torch.randn(1, 4, 16, 16)),
        ("transformer", TinyTransformer(), torch.randn(1, 6, 16)),
        ("rnn", TinyRNN(), torch.randn(1, 6, 8)),
    ]


def main() -> None:
    """Render none/auto/max SVGs for the sample set."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, model, example in _cases():
        with torch.no_grad():
            trace = tl.trace(model.eval(), example)
        try:
            for collapse in ("none", "auto", "max"):
                trace.draw(
                    vis_outpath=str(OUT_DIR / f"{name}_{collapse}"),
                    vis_save_only=True,
                    vis_fileformat="svg",
                    collapse=collapse,
                    show_containers=False,
                )
        finally:
            trace.cleanup()


if __name__ == "__main__":
    main()

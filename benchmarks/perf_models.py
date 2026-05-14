"""Model fixtures for the TorchLens performance benchmark suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

DeviceName = Literal["cpu", "cuda"]
ModelName = Literal["tinynet", "resnet18", "gpt2_hf", "gpt2_hooked", "small_lstm"]

CACHE_DIR = Path(
    os.environ.get("TORCHLENS_BENCH_CACHE", Path.home() / ".cache" / "torchlens-bench")
)


class TinyNet(nn.Module):
    """Small convolutional network used as the signal-floor benchmark."""

    def __init__(self) -> None:
        """Initialize TinyNet layers."""

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a TinyNet forward pass.

        Parameters
        ----------
        x:
            Image batch with shape ``(batch, 3, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch, 10)``.
        """

        return self.classifier(self.features(x))


class SmallLSTM(nn.Module):
    """Compact LSTM fixture adapted from the multi-output module tests."""

    def __init__(self) -> None:
        """Initialize the recurrent layers."""

        super().__init__()
        self.lstm = nn.LSTM(5, 10)
        self.label = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LSTM fixture.

        Parameters
        ----------
        x:
            Sequence-major input with shape ``(seq, batch, features)``.

        Returns
        -------
        torch.Tensor
            Projection of the final hidden state.
        """

        batch_size = x.shape[1]
        h_0 = torch.zeros(1, batch_size, 10, device=x.device, dtype=x.dtype)
        c_0 = torch.zeros(1, batch_size, 10, device=x.device, dtype=x.dtype)
        _output, (h_n, _c_n) = self.lstm(x, (h_0, c_0))
        return self.label(h_n[-1])


def available_devices() -> list[DeviceName]:
    """Return benchmark devices available on this host.

    Returns
    -------
    list[DeviceName]
        ``["cpu"]`` plus ``"cuda"`` when CUDA is available.
    """

    devices: list[DeviceName] = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def input_summary(model_name: str) -> str:
    """Return a compact input-shape summary for a benchmark model.

    Parameters
    ----------
    model_name:
        Benchmark model identifier.

    Returns
    -------
    str
        Human-readable input shape.
    """

    summaries = {
        "tinynet": "(4, 3, 32, 32)",
        "resnet18": "(8, 3, 224, 224)",
        "gpt2_hf": "(4, 64) input_ids",
        "gpt2_hooked": "(4, 64) tokens",
        "small_lstm": "(7, 4, 5)",
    }
    return summaries.get(model_name, "unknown")


def build_tiny_dummy(device: str) -> tuple[nn.Module, torch.Tensor]:
    """Build a tiny model used only to prime global TorchLens wrappers.

    Parameters
    ----------
    device:
        Device name.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        Dummy model and input tensor.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).to(device).eval()
    x = torch.randn(2, 4, device=device)
    return model, x


def load_model_and_input(model_name: str, device: str) -> tuple[nn.Module, Any]:
    """Load a benchmark model and deterministic input.

    Parameters
    ----------
    model_name:
        Benchmark model identifier.
    device:
        Device name, either ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    tuple[nn.Module, Any]
        Model in eval mode and forward input.
    """

    if model_name == "tinynet":
        model = TinyNet().to(device).eval()
        x = torch.randn(4, 3, 32, 32, device=device)
        return model, x
    if model_name == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(weights=None).to(device).eval()
        x = torch.randn(8, 3, 224, 224, device=device)
        return model, x
    if model_name == "gpt2_hf":
        from transformers import GPT2LMHeadModel

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=str(CACHE_DIR)).to(device).eval()
        x = torch.randint(0, int(model.config.vocab_size), (4, 64), device=device)
        return model, x
    if model_name == "gpt2_hooked":
        from transformer_lens import HookedTransformer

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model = HookedTransformer.from_pretrained(
            "gpt2",
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        ).eval()
        vocab_size = int(getattr(model.cfg, "d_vocab", 50257))
        x = torch.randint(0, vocab_size, (4, 64), device=device)
        return model, x
    if model_name == "small_lstm":
        model = SmallLSTM().to(device).eval()
        x = torch.randn(7, 4, 5, device=device)
        return model, x
    raise ValueError(f"Unknown benchmark model: {model_name}")

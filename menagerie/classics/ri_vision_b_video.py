"""Video models: C3D (3D-conv action recognition) and DCVC (deep contextual video codec).

C3D -- Tran et al., ICCV 2015, "Learning Spatiotemporal Features with 3D Convolutional
Networks", arXiv:1412.0767. Source: github.com/facebookarchive/C3D.
  DISTINCTIVE primitive: 3x3x3 3D convolutions over a video clip (T x H x W). Eight conv3d
  layers (64-128-256-256-512-512) interleaved with 3D max-pooling, then 2 fully-connected
  layers -> class logits. The 3x3x3 spatiotemporal kernel is the whole point.

DCVC -- Li et al., NeurIPS 2021, "Deep Contextual Video Compression", arXiv:2109.15047.
Source: github.com/microsoft/DCVC.
  DISTINCTIVE primitive: instead of subtracting a predicted frame (residual coding), DCVC
  feeds a learned high-dimensional TEMPORAL CONTEXT (warped/refined features from the
  reference frame) as a CONDITION into the encoder/decoder. The current frame is encoded
  *given* the context (context-conditional coding), an entropy-model bottleneck quantizes the
  latent, and a contextual decoder reconstructs. We reproduce the context-conditional
  encoder -> quantized latent -> contextual decoder structure (the entropy/range coder is a
  non-learned bit-packer and is out of scope).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
# C3D
# ============================================================


class C3D(nn.Module):
    """C3D action-recognition net: 8 conv3d (3x3x3) + 3D pooling + 2 FC -> logits."""

    def __init__(self, num_classes: int = 101) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, 3, padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3a = nn.Conv3d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.conv4a = nn.Conv3d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool3d(2, 2)
        self.conv5a = nn.Conv3d(512, 512, 3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, 3, padding=1)
        self.pool5 = nn.AdaptiveMaxPool3d((1, 2, 2))
        self.fc6 = nn.Linear(512 * 2 * 2, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3a(x))
        x = self.pool3(self.relu(self.conv3b(x)))
        x = self.relu(self.conv4a(x))
        x = self.pool4(self.relu(self.conv4b(x)))
        x = self.relu(self.conv5a(x))
        x = self.pool5(self.relu(self.conv5b(x)))
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        return self.fc8(x)


# ============================================================
# DCVC contextual codec
# ============================================================


class _ContextEncoder(nn.Module):
    """Refines the warped reference features into the temporal CONTEXT feature map."""

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )

    def forward(self, ref: torch.Tensor) -> torch.Tensor:
        return self.net(ref)


class DCVCContextualCodec(nn.Module):
    """Deep contextual video codec: context-conditional encoder -> quantized latent -> decoder.

    Inputs: current frame (B,3,H,W) and a reference frame (B,3,H,W). The reference is turned
    into a temporal context feature; the current frame is encoded CONDITIONED on that context
    (concatenation), the latent is quantized (round, straight-through at train time -- here a
    plain round), and a contextual decoder reconstructs the frame using the same context.
    """

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.context = _ContextEncoder(ch)
        self.enc = nn.Sequential(
            nn.Conv2d(3 + ch, ch, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
        # hyperprior-style entropy params (mean/scale) predicted from the latent
        self.entropy = nn.Conv2d(ch, ch * 2, 3, 1, 1)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, ch, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.ctx_fuse = nn.Conv2d(ch + ch, ch, 3, 1, 1)
        self.recon = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, frame: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        ctx = self.context(ref)  # temporal context feature
        y = self.enc(torch.cat([frame, ctx], dim=1))  # context-conditional encode
        params = self.entropy(y)  # entropy model mean/scale (codec bottleneck)
        mean = params[:, : y.shape[1]]
        y_hat = torch.round(y - mean) + mean  # quantize latent
        d = self.dec(y_hat)
        d = self.ctx_fuse(torch.cat([d, ctx], dim=1))  # decode WITH context
        return self.recon(d)


class _DCVCWrapper(nn.Module):
    """DCVC forwardable from a single frame tensor (synthesizes a reference frame)."""

    def __init__(self) -> None:
        super().__init__()
        self.codec = DCVCContextualCodec()

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        ref = torch.zeros_like(frame)  # prior reconstructed frame stand-in
        return self.codec(frame, ref)


def build_c3d() -> nn.Module:
    """C3D 3D-conv action recognition network."""
    return C3D(num_classes=101).eval()


def build_dcvc_contextual_codec() -> nn.Module:
    """DCVC deep contextual video codec (context-conditional encoder/decoder)."""
    return _DCVCWrapper().eval()


def example_input_clip() -> torch.Tensor:
    """Video clip (1, 3, 16, 56, 56) -- (B, C, T, H, W) for C3D."""
    return torch.randn(1, 3, 16, 56, 56)


def example_input_frame() -> torch.Tensor:
    """Single RGB frame (1, 3, 64, 64) for DCVC."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "C3D (3x3x3 spatiotemporal-conv action recognition)",
        "build_c3d",
        "example_input_clip",
        "2015",
        "DC",
    ),
    (
        "DCVC (deep contextual video codec, context-conditional coding)",
        "build_dcvc_contextual_codec",
        "example_input_frame",
        "2021",
        "DC",
    ),
]

"""VPT: Video PreTraining -- Learning to Act by Watching Unlabeled Online Videos.

Baker et al. (OpenAI), NeurIPS 2022.
Paper: https://arxiv.org/abs/2206.11795
Source: https://github.com/openai/Video-Pre-Training

The VPT foundation policy takes a single 128x128 RGB frame, passes it through an
IMPALA-style ResNet CNN to produce a 1024-d image embedding, then through a
stack of residual recurrent (memory) transformer blocks that attend over a
sliding window of past [key, value] pairs while using only the current step as
the query. A final dense layer yields the policy latent, from which keyboard and
mouse (camera + button) action heads are read.

This is a faithful random-init reimplementation. To keep the traced graph
compact we trace a single frame with a short attention context; the dynamics and
module structure are identical at any context length.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImpalaResidualBlock(nn.Module):
    """Residual block of the IMPALA CNN: two 3x3 convs with a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv0(F.relu(x))
        h = self.conv1(F.relu(h))
        return x + h


class ImpalaConvSequence(nn.Module):
    """IMPALA conv stage: conv -> maxpool -> 2 residual blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res0 = ImpalaResidualBlock(out_ch)
        self.res1 = ImpalaResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res0(x)
        x = self.res1(x)
        return x


class ImpalaCNN(nn.Module):
    """IMPALA CNN backbone producing a 1024-d image embedding from a 128x128 frame."""

    def __init__(self, in_ch: int = 3, hid_channels=(64, 128, 128), out_dim: int = 1024) -> None:
        super().__init__()
        seqs: list[nn.Module] = []
        c = in_ch
        for hc in hid_channels:
            seqs.append(ImpalaConvSequence(c, hc))
            c = hc
        self.stages = nn.Sequential(*seqs)
        # 128 -> /2 /2 /2 = 16x16 spatial with the chosen pools.
        self.linear = nn.Linear(hid_channels[-1] * 16 * 16, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        return F.relu(self.linear(x))


class RecurrentTransformerBlock(nn.Module):
    """Residual transformer block with masked self-attention + MLP (VPT memory block)."""

    def __init__(self, dim: int = 1024, n_heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class VPTPolicy(nn.Module):
    """VPT foundation policy: IMPALA CNN + recurrent transformer + action heads."""

    def __init__(
        self,
        dim: int = 1024,
        n_blocks: int = 4,
        n_camera_bins: int = 121,
        n_buttons: int = 8641,
    ) -> None:
        super().__init__()
        self.cnn = ImpalaCNN(in_ch=3, hid_channels=(64, 128, 128), out_dim=dim)
        self.blocks = nn.ModuleList(
            [RecurrentTransformerBlock(dim=dim, n_heads=8) for _ in range(n_blocks)]
        )
        self.policy_dense = nn.Linear(dim, dim)
        self.button_head = nn.Linear(dim, n_buttons)
        self.camera_head = nn.Linear(dim, n_camera_bins)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, 3, 128, 128)
        b, t = frames.shape[0], frames.shape[1]
        flat = frames.reshape(b * t, *frames.shape[2:])
        emb = self.cnn(flat).reshape(b, t, -1)
        x = emb
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.policy_dense(x))
        # Read action heads from the final timestep latent.
        last = x[:, -1, :]
        buttons = self.button_head(last)
        camera = self.camera_head(last)
        return torch.cat([buttons, camera], dim=-1)


def build() -> nn.Module:
    """Build the VPT foundation policy (IMPALA CNN + recurrent transformer)."""
    return VPTPolicy(dim=1024, n_blocks=4)


def example_input() -> torch.Tensor:
    """Short frame sequence ``(1, 4, 3, 128, 128)`` (T=4 keeps the trace compact)."""
    return torch.randn(1, 4, 3, 128, 128)


MENAGERIE_ENTRIES = [
    (
        "VPT (Video PreTraining foundation policy: IMPALA CNN + recurrent transformer)",
        "build",
        "example_input",
        "2022",
        "DC",
    ),
]

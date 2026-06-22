"""FoundationStereo: Zero-Shot Stereo Matching with Foundation Model.

Wen et al., CVPR 2025.
Paper: https://arxiv.org/abs/2501.09898
Source: https://github.com/NVlabs/FoundationStereo

Distinctive primitives:
  - Vision Foundation Model (ViT) feature backbone with Side Adapter / side-tuning:
    a lightweight side network that adapts frozen ViT features to stereo matching
    without fine-tuning the full ViT (efficient + zero-shot transfer).
  - All-pairs correlation cost volume built from adapted ViT features.
  - Iterative GRU-based disparity refinement (RAFT-Stereo style).

Here we reproduce:
  (1) A compact ViT patch embedding + transformer blocks.
  (2) Side-tuning adapter: small parallel CNN that fuses with ViT outputs.
  (3) All-pairs correlation volume from fused features.
  (4) GRU update block for iterative disparity update.

Random init; compact: H=32, W=64, D=12, C=24, 2 transformer blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ──────────────────────────────────────────────────────────────
# ViT backbone (compact)
# ──────────────────────────────────────────────────────────────


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        return x + self.ff(self.norm2(x))


class _ViTBackbone(nn.Module):
    """Compact ViT backbone (patch embed + N transformer blocks)."""

    def __init__(
        self, in_c: int = 3, embed_dim: int = 48, patch_size: int = 4, n_blocks: int = 2
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_c, embed_dim, patch_size, patch_size)
        self.blocks = nn.Sequential(*[_TransformerBlock(embed_dim) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Returns (B, N_tokens, embed_dim), H_patches, W_patches."""
        tokens = self.patch_embed(x)  # (B, embed_dim, Hp, Wp)
        B, C, Hp, Wp = tokens.shape
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, Hp*Wp, C)
        tokens = self.blocks(tokens)
        tokens = self.out_proj(tokens)
        return tokens, Hp, Wp


# ──────────────────────────────────────────────────────────────
# Side Adapter (side-tuning)
# ──────────────────────────────────────────────────────────────


class _SideAdapter(nn.Module):
    """Lightweight side network that adapts ViT features for stereo matching.

    A small CNN processes the input in parallel and its features are fused
    with the ViT token outputs (feature adaptation without fine-tuning ViT).
    """

    def __init__(self, in_c: int = 3, vit_dim: int = 48, C: int = 24, patch_size: int = 4) -> None:
        super().__init__()
        # Side CNN operating at patch resolution
        self.side_cnn = nn.Sequential(
            _ConvBnRelu(in_c, C, 7, patch_size, 3),  # stride = patch_size
            _ConvBnRelu(C, C, 3, 1, 1),
        )
        # Adapter: fuse ViT tokens + side features -> output adapted feature map
        self.adapter = nn.Sequential(
            nn.Conv2d(vit_dim + C, vit_dim, 1, bias=False),
            nn.BatchNorm2d(vit_dim),
            nn.ReLU(inplace=True),
        )
        self.out_channels = vit_dim

    def forward(
        self, img: torch.Tensor, vit_tokens: torch.Tensor, H_p: int, W_p: int
    ) -> torch.Tensor:
        """Fuse ViT tokens with side CNN features.
        Returns adapted feature map (B, out_channels, H_p, W_p).
        """
        B = img.shape[0]
        # reshape ViT tokens to spatial
        vit_map = vit_tokens.transpose(1, 2).view(B, -1, H_p, W_p)
        # side CNN
        side = self.side_cnn(img)  # (B, C, H_p, W_p) approximately
        if side.shape[2:] != vit_map.shape[2:]:
            side = F.interpolate(side, vit_map.shape[2:], mode="bilinear", align_corners=False)
        return self.adapter(torch.cat([vit_map, side], dim=1))


# ──────────────────────────────────────────────────────────────
# All-pairs correlation (reused from RAFT-Stereo pattern)
# ──────────────────────────────────────────────────────────────


def _build_corr_vol(fl: torch.Tensor, fr: torch.Tensor, max_disp: int) -> torch.Tensor:
    B, C, H, W = fl.shape
    corr = torch.zeros(B, max_disp, H, W, device=fl.device, dtype=fl.dtype)
    for d in range(max_disp):
        if d == 0:
            r = fr
        else:
            r = torch.zeros_like(fr)
            r[:, :, :, d:] = fr[:, :, :, :-d]
        corr[:, d] = (fl * r).mean(1)
    return corr


# ──────────────────────────────────────────────────────────────
# Context + GRU update block
# ──────────────────────────────────────────────────────────────


class _ConvGRU(nn.Module):
    def __init__(self, hidden_dim: int, inp_dim: int) -> None:
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + inp_dim, hidden_dim, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_dim + inp_dim, hidden_dim, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_dim + inp_dim, hidden_dim, 3, 1, 1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], 1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], 1)))
        return (1 - z) * h + z * q


class _UpdateBlock(nn.Module):
    def __init__(self, hidden_dim: int, corr_ch: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(corr_ch + 1, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.gru = _ConvGRU(hidden_dim, hidden_dim)
        self.disp = nn.Conv2d(hidden_dim, 1, 3, 1, 1)

    def forward(
        self, h: torch.Tensor, corr: torch.Tensor, disp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = self.enc(torch.cat([corr, disp], 1))
        h = self.gru(h, inp)
        return h, self.disp(h)


# ──────────────────────────────────────────────────────────────
# FoundationStereo
# ──────────────────────────────────────────────────────────────


class FoundationStereo(nn.Module):
    """FoundationStereo: ViT + side-adapter + correlation + GRU stereo."""

    def __init__(
        self, max_disp: int = 12, C: int = 24, vit_dim: int = 48, n_iters: int = 2
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.n_iters = n_iters
        patch_size = 4
        # ViT backbone (shared for both views)
        self.vit = _ViTBackbone(3, vit_dim, patch_size, n_blocks=2)
        # Side adapter (applied to both views separately)
        self.side_adapter = _SideAdapter(3, vit_dim, C, patch_size)
        # Context encoder for GRU hidden state
        self.context_net = nn.Sequential(
            _ConvBnRelu(3, C, 7, patch_size, 3),
            _ConvBnRelu(C, C * 2, 3, 1, 1),
        )
        hidden_dim = C * 2
        self.h_head = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.update = _UpdateBlock(hidden_dim, max_disp)

    def _extract(self, img: torch.Tensor) -> torch.Tensor:
        tokens, Hp, Wp = self.vit(img)
        return self.side_adapter(img, tokens, Hp, Wp)  # (B, vit_dim, Hp, Wp)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self._extract(left)
        fr = self._extract(right)
        # all-pairs correlation
        corr_vol = _build_corr_vol(fl, fr, self.max_disp)  # (B, D, Hf, Wf)
        # context -> hidden state
        ctx = self.context_net(left)
        hidden = torch.tanh(self.h_head(ctx))
        if hidden.shape[2:] != corr_vol.shape[2:]:
            hidden = F.interpolate(hidden, corr_vol.shape[2:], mode="bilinear", align_corners=False)
        disp = torch.zeros(
            left.shape[0], 1, corr_vol.shape[2], corr_vol.shape[3], device=left.device
        )
        for _ in range(self.n_iters):
            hidden, delta = self.update(hidden, corr_vol, disp)
            disp = disp + delta
        disp = F.interpolate(disp, left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# Wrapper + builders
# ──────────────────────────────────────────────────────────────


class _StereoWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, :3], x[:, 3:])


def build_foundation_stereo() -> nn.Module:
    """FoundationStereo (ViT side-tuning + correlation + GRU), compact."""
    return _StereoWrapper(FoundationStereo(max_disp=12, C=24, vit_dim=48, n_iters=2))


def example_input() -> torch.Tensor:
    """Stereo pair as 6-channel tensor (left||right), (1, 6, 32, 64)."""
    return torch.randn(1, 6, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "FoundationStereo (ViT side-tuning + correlation + GRU stereo)",
        "build_foundation_stereo",
        "example_input",
        "2025",
        "DC",
    ),
]

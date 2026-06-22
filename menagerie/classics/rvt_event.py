"""RVT: Recurrent Vision Transformer for event cameras (object detection).

Gehrig et al., ICCV 2023.
Paper: https://arxiv.org/abs/2212.05074
Source: https://github.com/uzh-rpg/RVT

NOTE: This is a DIFFERENT paper from menagerie/classics/rvt.py (which implements
"RVT: Robotic View Transformer" for robot manipulation).  This RVT is for
event-camera-based object detection with recurrent state.

RVT's distinctive primitive:

  Multi-stage backbone where each stage alternates:
    1. Conv downsample (stride 2)
    2. Local-window attention block (multi-head self-attention within non-overlapping windows)
    3. Dilated global attention block (attention with dilated stride for long-range context)
  PLUS a ConvLSTM recurrent cell per stage that fuses the current feature map
  with the hidden state from the previous time step.

  The interleaved (local-window attention + dilated global attention + ConvLSTM)
  per-stage topology over a sequence of event frames is RVT's defining structure.

  The model is unrolled for T=3 time steps.  A detection head (CenterNet-style:
  heatmap + wh + offset) is applied to the final stage feature maps at the last
  time step.

Architecture notes / simplifications:
  - 2 stages (paper uses 4).
  - Hidden dims: 32 and 64 (paper: 64/128/256/512).
  - Window size: 4x4 (paper: 8x8).
  - Attention heads: 2 per block.
  - ConvLSTM kernel: 3x3.
  - Input: event frame represented as (N, T, C, H, W) stacked time steps,
    with C=2 (positive/negative event polarity channels).
  - Random init, CPU, forward-only.
  - trace+draw verified 2026-06-21.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Local-window multi-head self-attention
# ============================================================


class WindowAttention(nn.Module):
    """Multi-head self-attention within non-overlapping windows.

    Splits the feature map into windows of size (win_h, win_w),
    applies self-attention within each window independently,
    then reassembles the output.
    """

    def __init__(self, dim: int, nhead: int, win_size: int = 4) -> None:
        super().__init__()
        self.win_size = win_size
        self.nhead = nhead
        self.scale = (dim // nhead) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        ws = self.win_size
        # Pad if needed
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        nh, nw = Hp // ws, Wp // ws

        # Reshape: (N, C, Hp, Wp) -> (N*nh*nw, ws*ws, C)
        x = x.permute(0, 2, 3, 1)  # (N, Hp, Wp, C)
        x = x.view(N, nh, ws, nw, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(N * nh * nw, ws * ws, C)

        # Self-attention
        qkv = self.qkv(x).reshape(x.shape[0], ws * ws, 3, self.nhead, C // self.nhead)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], ws * ws, C)
        x = self.proj(x)

        # Reassemble: (N*nh*nw, ws*ws, C) -> (N, C, Hp, Wp)
        x = x.view(N, nh, nw, ws, ws, C).permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(N, C, Hp, Wp)
        # Unpad
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


# ============================================================
# Dilated global attention
# ============================================================


class DilatedAttention(nn.Module):
    """Global dilated attention: subsample tokens at stride d, apply attention.

    Faithfully reproduces RVT's dilated-grid global attention:
    take every d-th pixel in both H and W dimensions, apply self-attention
    across those tokens, scatter back.
    """

    def __init__(self, dim: int, nhead: int, dilation: int = 2) -> None:
        super().__init__()
        self.dilation = dilation
        self.nhead = nhead
        self.scale = (dim // nhead) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        d = self.dilation
        # Subsample at stride d
        x_sub = x[:, :, ::d, ::d]  # (N, C, H/d, W/d)
        Hs, Ws = x_sub.shape[2], x_sub.shape[3]
        tokens = x_sub.flatten(2).transpose(1, 2)  # (N, Hs*Ws, C)

        qkv = self.qkv(tokens).reshape(N, Hs * Ws, 3, self.nhead, C // self.nhead)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N, Hs * Ws, C)
        out = self.proj(out).transpose(1, 2).view(N, C, Hs, Ws)

        # Scatter back: residual add at subsampled positions
        result = x.clone()
        result[:, :, ::d, ::d] = result[:, :, ::d, ::d] + out
        return result


# ============================================================
# ConvLSTM recurrent cell
# ============================================================


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for spatial recurrent state across time steps.

    Implements the standard ConvLSTM gating equations using conv2d.
    """

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        # All gates in one conv: [i, f, g, o] -> 4 * hidden_ch outputs
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        N, C, H, W = x.shape
        if state is None:
            h = torch.zeros(N, self.hidden_ch, H, W, device=x.device)
            c = torch.zeros(N, self.hidden_ch, H, W, device=x.device)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)


# ============================================================
# RVT Stage: conv downsample + window attn + dilated attn + ConvLSTM
# ============================================================


class RVTStage(nn.Module):
    """One RVT stage: downsample -> window attention -> dilated attention -> ConvLSTM.

    This interleaved (local + global attention + recurrence) per-stage
    structure is RVT's distinctive primitive.
    """

    def __init__(
        self, in_ch: int, out_ch: int, nhead: int = 2, win_size: int = 4, dilation: int = 2
    ) -> None:
        super().__init__()
        # Strided conv downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.norm1 = nn.GroupNorm(1, out_ch)  # LayerNorm-equivalent for spatial feats
        self.win_attn = WindowAttention(out_ch, nhead, win_size)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.dil_attn = DilatedAttention(out_ch, nhead, dilation)
        self.norm3 = nn.GroupNorm(1, out_ch)
        # Feed-forward (pointwise)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 2, 1), nn.GELU(), nn.Conv2d(out_ch * 2, out_ch, 1)
        )
        # Recurrent cell
        self.conv_lstm = ConvLSTMCell(out_ch, out_ch)

    def forward(
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.downsample(x)
        # Window attention with pre-norm
        x = x + self.win_attn(self.norm1(x))
        # Dilated global attention with pre-norm
        x = x + self.dil_attn(self.norm2(x))
        # FFN
        x = x + self.ffn(self.norm3(x))
        # Recurrent update
        h, new_state = self.conv_lstm(x, state)
        return h, new_state


# ============================================================
# Full RVT model: multi-stage backbone + CenterNet detection head
# ============================================================


class RVTEventModel(nn.Module):
    """RVT (Recurrent Vision Transformer) for event camera object detection.

    Processes T time steps of event frames through a multi-stage backbone.
    Each stage has ConvLSTM recurrent state that persists across time steps.
    At the final time step, CenterNet-style detection heads are applied.

    Input: (N, T, C_in, H, W) -- T event frames, C_in channels (e.g., 2 polarities).
    """

    def __init__(
        self,
        in_ch: int = 2,
        stage_chs: Tuple[int, ...] = (32, 64),
        num_classes: int = 2,
        nhead: int = 2,
        win_size: int = 4,
        t_steps: int = 3,
    ) -> None:
        super().__init__()
        self.t_steps = t_steps
        # Initial feature extractor (before stages)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stage_chs[0] // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(stage_chs[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stage_chs[0] // 2, stage_chs[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(stage_chs[0]),
            nn.ReLU(inplace=True),
        )

        # Multi-stage RVT backbone
        self.stages = nn.ModuleList()
        prev_ch = stage_chs[0]
        for ch in stage_chs:
            self.stages.append(RVTStage(prev_ch, ch, nhead=nhead, win_size=win_size))
            prev_ch = ch

        # CenterNet detection head on the deepest feature map
        final_ch = stage_chs[-1]
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(final_ch, num_classes, 1),
            nn.Sigmoid(),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, 3, padding=1), nn.ReLU(), nn.Conv2d(final_ch, 2, 1)
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(final_ch, final_ch, 3, padding=1), nn.ReLU(), nn.Conv2d(final_ch, 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process T event frames sequentially with recurrent state.

        Args:
            x: (N, T, C, H, W)
        Returns:
            detections: concatenated heatmap + wh + offset at last time step.
        """
        N, T, C, H, W = x.shape
        states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.stages)

        feat = None
        for t in range(T):
            frame = x[:, t]  # (N, C, H, W)
            feat = self.stem(frame)

            for i, stage in enumerate(self.stages):
                feat, states[i] = stage(feat, states[i])

        # Detection heads on final time step's feature map
        hm = self.heatmap_head(feat)
        wh = self.wh_head(feat)
        off = self.offset_head(feat)
        return torch.cat([hm, wh, off], dim=1)


# ============================================================
# Zero-arg builder + example input
# ============================================================


def build_rvt_event() -> nn.Module:
    """Build RVT-event: recurrent ViT backbone for event-camera object detection."""
    return RVTEventModel(in_ch=2, stage_chs=(32, 64), num_classes=2, nhead=2, win_size=4, t_steps=3)


def example_input_rvt_event() -> torch.Tensor:
    """Event frames (1, 3, 2, 32, 32): batch=1, T=3 time steps, 2 polarity channels, 32x32."""
    return torch.randn(1, 3, 2, 32, 32)


# ============================================================
# Menagerie entries
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "RVT-event (Recurrent Vision Transformer for event cameras)",
        "build_rvt_event",
        "example_input_rvt_event",
        "2023",
        "DC",
    ),
]

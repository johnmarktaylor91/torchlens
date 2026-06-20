"""Searth (Shifted Earth) Transformer: geospheric-prior window attention.

Li, Liu, Li, Chen, Cheng, Zheng, Xia et al. (Institute of Atmospheric Physics,
Chinese Academy of Sciences; Fudan University), 2026, arXiv:2601.09467
("Searth Transformer: A Transformer Architecture Incorporating Earth's
Geospheric Physical Priors for Global Mid-Range Weather Forecasting").  The
Searth Transformer is the backbone of the **YanTian** global weather model.

The distinctive primitive is an Earth-aware variant of Swin window attention.
Each Searth Transformer block is two successive sub-blocks:
  - **E-MSA** (Earth-aware MSA): algorithmically identical to Swin's
    window-based multi-head self-attention (W-MSA) on a regular partition.
  - **SE-MSA** (Shifted-Earth MSA): like Swin's shifted-window attention, but
    with an ASYMMETRIC shift-and-mask that encodes Earth's spherical topology:
      * **zonal periodicity (east-west / longitude)**: the cyclic shift wraps
        around the dateline and the east-west boundary masks are REMOVED, so
        windows straddling longitude 0/360 exchange information seamlessly;
      * **meridional boundaries (north-south / latitude)**: the north-south
        (polar) masks are RETAINED, so attention never mixes features across the
        poles (no unphysical pole wrap-around).
    This asymmetric "shift-and-mask" is the paper's core contribution.

The full YanTian model is an Encoder(x6)-Core(x20)-Decoder(x6) hierarchy with
3D-conv embedding, patch-merging/-expanding, and an ``X_t`` skip connection that
makes the network predict the weather TENDENCY (state difference) which is added
to the persisted current state.  This faithful COMPACT random-init reimpl keeps
that whole pipeline -- embedding -> Searth encoder -> patch-merge -> Searth core
-> patch-expand -> Searth decoder -> unembedding + X_t skip -- but with a tiny
lat-lon grid (H=16, W=32), few channels, small ``d_model``, and 1 block per
stage so the unrolled graph stays renderable.  Pure torch, CPU, forward-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, wh: int, ww: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, wh*ww, C)."""
    B, H, W, C = x.shape
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, wh * ww, C)
    return windows


def window_reverse(windows: torch.Tensor, wh: int, ww: int, H: int, W: int) -> torch.Tensor:
    """(num_windows*B, wh*ww, C) -> (B, H, W, C)."""
    C = windows.shape[-1]
    B = windows.shape[0] // ((H // wh) * (W // ww))
    x = windows.view(B, H // wh, W // ww, wh, ww, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


class WindowAttention(nn.Module):
    """Multi-head self-attention within a window, with optional additive mask."""

    def __init__(self, dim: int, n_head: int = 4) -> None:
        super().__init__()
        self.h = n_head
        self.hd = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (nW*B, L, C); mask: (nW, L, L) additive (-inf to block) or None
        BnW, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(BnW, L, self.h, self.hd).transpose(1, 2)
        k = k.view(BnW, L, self.h, self.hd).transpose(1, 2)
        v = v.view(BnW, L, self.h, self.hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hd**0.5)
        if mask is not None:
            nW = mask.shape[0]
            scores = scores.view(BnW // nW, nW, self.h, L, L) + mask[None, :, None]
            scores = scores.view(BnW, self.h, L, L)
        att = torch.softmax(scores, dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(BnW, L, C)
        return self.proj(out)


class SearthBlock(nn.Module):
    """Two successive sub-blocks: E-MSA (regular) then SE-MSA (Earth-shifted).

    SE-MSA performs a cyclic shift of (wh//2, ww//2). The roll is periodic in
    BOTH axes (torch.roll wraps), but the attention mask is asymmetric: the
    east-west (longitude / W) wrap-around is ALLOWED (zonal periodicity), while
    the north-south (latitude / H) wrap-around across the poles is MASKED.
    """

    def __init__(self, dim: int, grid_h: int, grid_w: int, wh: int = 4, ww: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.H, self.W = grid_h, grid_w
        self.wh, self.ww = wh, ww
        self.shift_h, self.shift_w = wh // 2, ww // 2

        self.norm1 = nn.LayerNorm(dim)
        self.emsa = WindowAttention(dim)
        self.norm1b = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

        self.norm2 = nn.LayerNorm(dim)
        self.semsa = WindowAttention(dim)
        self.norm2b = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

        # Precompute the asymmetric Earth shift-mask (north-south masked, east-west open).
        self.register_buffer("earth_mask", self._build_earth_mask(), persistent=False)

    def _build_earth_mask(self) -> torch.Tensor:
        """Block attention between window members that are non-adjacent ONLY because
        of the north-south (pole) wrap; allow the east-west (dateline) wrap.

        Mirrors Swin's SW-MSA mask construction, but the H-axis region id changes
        across the polar seam while the W-axis seam shares an id (periodic).
        """
        H, W, wh, ww = self.H, self.W, self.wh, self.ww
        sh = self.shift_h
        img = torch.zeros(H, W)
        # latitude (H): three regions split by the shifted-window seam -> distinct ids
        h_slices = (slice(0, -wh), slice(-wh, -sh), slice(-sh, None))
        cnt = 0
        for hsl in h_slices:
            img[hsl, :] = cnt
            cnt += 1
        # NOTE: deliberately NO west-east region split: longitude is periodic, so the
        # east-west wrapped windows keep a SHARED region id and are NOT masked.
        windows = window_partition(img.unsqueeze(0).unsqueeze(-1), wh, ww)  # (nW,L,1)
        windows = windows.squeeze(-1)  # (nW, L)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)  # (nW, L, L)
        mask = mask.masked_fill(mask != 0, float("-inf")).masked_fill(mask == 0, 0.0)
        return mask

    def _attn(self, x, norm, attn, normb, mlp, shifted):
        B, H, W, C = x.shape
        shortcut = x
        h = norm(x)
        if shifted:
            h = torch.roll(h, shifts=(-self.shift_h, -self.shift_w), dims=(1, 2))
            mask = self.earth_mask
        else:
            mask = None
        win = window_partition(h, self.wh, self.ww)
        win = attn(win, mask)
        h = window_reverse(win, self.wh, self.ww, H, W)
        if shifted:
            h = torch.roll(h, shifts=(self.shift_h, self.shift_w), dims=(1, 2))
        x = shortcut + h
        x = x + mlp(normb(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        x = self._attn(x, self.norm1, self.emsa, self.norm1b, self.mlp1, shifted=False)  # E-MSA
        x = self._attn(x, self.norm2, self.semsa, self.norm2b, self.mlp2, shifted=True)  # SE-MSA
        return x


class PatchMerging(nn.Module):
    """Downsample 2x in H,W and double channels (Swin patch-merging)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduce = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        return self.reduce(self.norm(x))


class PatchExpanding(nn.Module):
    """Upsample 2x in H,W and halve channels (inverse of patch-merging)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.expand(x)  # (B, H, W, 2C)
        x = x.view(B, H, W, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H * 2, W * 2, C // 2)
        return self.norm(x)


class SearthTransformer(nn.Module):
    """Compact YanTian: 3D-conv embed -> Searth encoder -> core -> decoder -> X_t skip.

    Input ``x``: (B, T=2, C, H, W) -- two consecutive weather states (X_{t-1}, X_t).
    Output: predicted next state (B, C, H, W) = X_t + predicted tendency.
    """

    def __init__(
        self,
        in_ch: int = 6,
        d_model: int = 32,
        grid_h: int = 16,
        grid_w: int = 32,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.H, self.W = grid_h, grid_w
        # 3D-conv embedding jointly encoding the 2 time steps + spatial (kernel over T).
        self.embed = nn.Conv3d(in_ch, d_model, kernel_size=(2, 3, 3), padding=(0, 1, 1))

        # Encoder: Searth block at full res, then patch-merge.
        self.enc = SearthBlock(d_model, grid_h, grid_w)
        self.merge = PatchMerging(d_model)
        # Core: Searth blocks at half res / 2x channels.
        self.core = SearthBlock(2 * d_model, grid_h // 2, grid_w // 2)
        # Decoder: patch-expand back to full res, then Searth block.
        self.expand = PatchExpanding(2 * d_model)
        self.dec = SearthBlock(d_model, grid_h, grid_w)

        # Unembedding: project channels back to weather variables (the tendency).
        self.unembed = nn.Linear(d_model, in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=2, C, H, W). Persist X_t for the residual skip.
        x_t = x[:, -1]  # (B, C, H, W) current state baseline
        # 3D-conv embed: move channels to dim 1, time to dim 2.
        emb = self.embed(x.transpose(1, 2))  # (B, d_model, 1, H, W)
        emb = emb.squeeze(2).permute(0, 2, 3, 1)  # (B, H, W, d_model)

        h = self.enc(emb)
        h = self.merge(h)
        h = self.core(h)
        h = self.expand(h)
        h = self.dec(h)

        tendency = self.unembed(h).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x_t + tendency  # skip-connection: predict the increment


def build() -> nn.Module:
    return SearthTransformer()


def example_input() -> torch.Tensor:
    """Two consecutive weather states ``(1, T=2, C=6, H=16, W=32)`` on a tiny lat-lon grid."""
    return torch.randn(1, 2, 6, 16, 32)


MENAGERIE_ENTRIES = [
    (
        "Searth-Transformer",
        "build",
        "example_input",
        "2026",
        "DC",
    ),
]

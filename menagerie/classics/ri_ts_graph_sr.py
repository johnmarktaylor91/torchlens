"""Image / video super-resolution + compression transformers and CNNs.

DITN: "Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for SR"
  Liu et al., ACM MM 2023.  Paper: https://arxiv.org/abs/2308.02794
  Source: https://github.com/yongliuy/DITN
  Distinctive primitive: the UFONE block = (ITL) Intra-patch Transformer Layer doing
  self-attention WITHIN local patches, followed by (SAL) Spatial-Aware Layer -- a
  depthwise-conv FFN that mixes spatial context across patches.  Reshape into patches,
  attend intra-patch, then restore + spatial conv.  PixelShuffle upsampler.

DRCT: "DRCT: Saving Image Super-Resolution away from Information Bottleneck"
  Hsu et al., CVPRW 2024.  Paper: https://arxiv.org/abs/2404.00722
  Source: https://github.com/ming053l/DRCT
  Distinctive primitive: SwinIR's window-attention Swin Transformer blocks, but each
  Residual Dense Group (RDG) adds DENSE connections (concatenation across the group's
  Swin blocks, RDB-style) so feature/gradient information does not collapse through the
  network's information bottleneck.  l / s / sr variants differ only in depth/width.

DUF: "Deep Video Super-Resolution Network Using Dynamic Upsampling Filters..."
  Jo et al., CVPR 2018.  Source: https://github.com/yhjo09/VSR-DUF
  Distinctive primitive: from a clip of T frames a 3D-conv network predicts, for the
  centre frame, a set of per-pixel DYNAMIC UPSAMPLING FILTERS (r^2 filters of size 5x5)
  AND a residual; the HR output is the locally-filtered (dynamic conv) input plus the
  residual -- no fixed upsampling kernel.

DVC: "DVC: An End-to-end Deep Video Compression Framework"
  Lu et al., CVPR 2019.  Paper: https://arxiv.org/abs/1812.00101
  Source: https://github.com/ZhihaoHu/PyTorchVideoCompression
  Distinctive primitive: optical-flow motion estimation between current/reference frame,
  a motion (flow) autoencoder that compresses the flow, motion-compensated warping to a
  predicted frame, then a residual autoencoder compressing (current - predicted).  An
  end-to-end learned analogue of the classic predictive video codec.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DITN -- UFONE (ITL intra-patch attention + SAL spatial-aware)
# ============================================================


class _ITL(nn.Module):
    """Intra-patch Transformer Layer: self-attention within non-overlapping patches."""

    def __init__(self, dim: int, num_heads: int, patch: int) -> None:
        super().__init__()
        self.patch = patch
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> partition into patch x patch windows, attend intra-window.
        b, c, h, w = x.shape
        p = self.patch
        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, p * p, c)  # (B*nW, p*p, C)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(b, h // p, w // p, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(b, c, h, w)


class _SAL(nn.Module):
    """Spatial-Aware Layer: depthwise-conv FFN mixing context across patches."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pw1 = nn.Conv2d(dim, dim * 2, 1)
        self.pw2 = nn.Conv2d(dim * 2, dim, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dwconv(y)
        y = self.pw2(self.act(self.pw1(y)))
        return x + y


class _UFONE(nn.Module):
    """UFONE block: ITL intra-patch attention then SAL spatial-aware mixing."""

    def __init__(self, dim: int, num_heads: int, patch: int) -> None:
        super().__init__()
        self.itl = _ITL(dim, num_heads, patch)
        self.sal = _SAL(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sal(self.itl(x))


class DITN(nn.Module):
    """DITN: shallow conv -> stacked UFONE blocks -> PixelShuffle upsampler."""

    def __init__(
        self,
        upscale: int = 4,
        in_ch: int = 3,
        dim: int = 60,
        num_blocks: int = 4,
        num_heads: int = 6,
        patch: int = 8,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.body = nn.Sequential(*[_UFONE(dim, num_heads, patch) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(dim, dim, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, in_ch * upscale * upscale, 3, padding=1),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = feat + self.body_tail(self.body(feat))
        return self.upsample(feat)


# ============================================================
# DRCT -- Swin window attention + dense residual groups
# ============================================================


def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)


def _window_reverse(win: torch.Tensor, ws: int, h: int, w: int) -> torch.Tensor:
    b = win.shape[0] // ((h // ws) * (w // ws))
    x = win.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class _SwinBlock(nn.Module):
    """Window-based multi-head self-attention block (SwinIR style)."""

    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.ws = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)
        win = _window_partition(x, self.ws)
        win = self.attn(win, win, win)[0]
        x = _window_reverse(win, self.ws, h, w)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class _RDG(nn.Module):
    """Residual Dense Group: Swin blocks with DENSE concat connections (DRCT)."""

    def __init__(self, dim: int, num_heads: int, window_size: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_SwinBlock(dim, num_heads, window_size) for _ in range(depth)])
        # Dense transition convs reduce concatenated channels back to `dim`.
        self.transitions = nn.ModuleList([nn.Conv2d(dim * (i + 2), dim, 1) for i in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C). Maintain a growing list of features, densely concatenated.
        feats = [x]
        cur = x
        for blk, trans in zip(self.blocks, self.transitions):
            out = blk(cur)
            feats.append(out)
            cat = torch.cat(feats, dim=-1).permute(0, 3, 1, 2)  # (B, C*, H, W)
            cur = trans(cat).permute(0, 2, 3, 1)  # back to (B,H,W,C)
        return cur + x


class DRCT(nn.Module):
    """DRCT: shallow conv -> dense residual Swin groups -> PixelShuffle upsampler."""

    def __init__(
        self,
        upscale: int = 4,
        in_ch: int = 3,
        dim: int = 60,
        num_groups: int = 4,
        depth: int = 3,
        num_heads: int = 6,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.groups = nn.ModuleList(
            [_RDG(dim, num_heads, window_size, depth) for _ in range(num_groups)]
        )
        self.body_tail = nn.Conv2d(dim, dim, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, in_ch * upscale * upscale, 3, padding=1),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        h = feat.permute(0, 2, 3, 1)  # (B,H,W,C)
        for g in self.groups:
            h = g(h)
        feat = feat + self.body_tail(h.permute(0, 3, 1, 2))
        return self.upsample(feat)


# ============================================================
# DUF -- dynamic upsampling filters video SR
# ============================================================


class DUF(nn.Module):
    """DUF: 3D-conv net predicts per-pixel dynamic upsampling filters + residual."""

    def __init__(self, scale: int = 4, num_frames: int = 7, filt: int = 5) -> None:
        super().__init__()
        self.scale = scale
        self.filt = filt
        # 3D conv trunk over the (T, H, W) clip.
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, (num_frames, 3, 3), padding=(0, 1, 1)),  # collapse time
            nn.ReLU(inplace=True),
        )
        # Two branches off the shared 2D features: dynamic filters + residual.
        self.filter_branch = nn.Conv2d(64, scale * scale * filt * filt, 3, padding=1)
        self.residual_branch = nn.Conv2d(64, scale * scale * 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W) -> 3D conv wants (B, C, T, H, W)
        b, t, c, h, w = x.shape
        feat = self.conv3d(x.permute(0, 2, 1, 3, 4))  # (B, 64, 1, H, W)
        feat = feat.squeeze(2)  # (B, 64, H, W)
        # Predict per-pixel dynamic upsampling filters (softmax-normalized).
        filters = self.filter_branch(feat)
        filters = filters.view(b, self.scale * self.scale, self.filt * self.filt, h, w)
        filters = torch.softmax(filters, dim=2)
        # Apply dynamic filters: unfold centre frame, weight by per-pixel filters.
        centre = x[:, t // 2]  # (B, C, H, W)
        patches = F.unfold(centre, kernel_size=self.filt, padding=self.filt // 2)
        patches = patches.view(b, c, self.filt * self.filt, h, w)
        # (B, scale^2, filt^2, H, W) x (B, C, filt^2, H, W) -> (B, scale^2, C, H, W)
        dyn = torch.einsum("bkfhw,bcfhw->bkchw", filters, patches)
        dyn = dyn.reshape(b, self.scale * self.scale * c, h, w)
        residual = self.residual_branch(feat)
        # Concatenate dynamic-filtered output and residual, PixelShuffle to HR.
        out = torch.cat(
            [
                dyn.view(b, self.scale * self.scale, c, h, w),
                residual.view(b, self.scale * self.scale, c, h, w),
            ],
            dim=1,
        )
        out = out.sum(dim=1)  # combine filtered + residual (B, C, H, W)
        out = out.repeat_interleave(self.scale, 2).repeat_interleave(self.scale, 3)
        return out


# ============================================================
# DVC -- end-to-end deep video compression
# ============================================================


class _FlowNet(nn.Module):
    """Tiny optical-flow estimator between two RGB frames."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),  # (dx, dy) flow
        )

    def forward(self, cur: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cur, ref], dim=1))


class _AutoEncoder(nn.Module):
    """Analysis/synthesis transform with a quantization bottleneck (codec stage)."""

    def __init__(self, in_ch: int, latent: int = 32) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, latent, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent, latent, 5, stride=2, padding=2),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent, latent, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(latent, in_ch, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.enc(x)
        y = torch.round(y)  # uniform-quantization bottleneck (codec)
        return self.dec(y)


def _warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp `img` by `flow` (motion compensation)."""
    b, _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=img.dtype), torch.arange(w, dtype=img.dtype), indexing="ij"
    )
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1] / max(h - 1, 1) - 1.0
    vgrid = torch.stack([vgrid_x, vgrid_y], dim=-1)
    return F.grid_sample(img, vgrid, align_corners=True)


class DVC(nn.Module):
    """DVC: flow estimation -> motion codec -> warp -> residual codec -> reconstruct."""

    def __init__(self) -> None:
        super().__init__()
        self.flownet = _FlowNet()
        self.motion_codec = _AutoEncoder(2, latent=32)
        self.residual_codec = _AutoEncoder(3, latent=32)
        self.refine = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, cur: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        flow = self.flownet(cur, ref)
        flow_hat = self.motion_codec(flow)  # compressed motion
        predicted = _warp(ref, flow_hat)  # motion-compensated prediction
        residual = cur - predicted
        residual_hat = self.residual_codec(residual)  # compressed residual
        recon = predicted + residual_hat
        return self.refine(recon)


# ============================================================
# Wrappers + menagerie wiring (single-tensor forward for the atlas)
# ============================================================


class _DVCWrapper(nn.Module):
    """DVC wrapper: synthesizes a reference frame internally from one input frame."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, cur: torch.Tensor) -> torch.Tensor:
        ref = torch.roll(cur, shifts=1, dims=2)  # shifted copy as the reference frame
        return self.model(cur, ref)


def build_ditn() -> nn.Module:
    """Build DITN x4 (UFONE: ITL intra-patch attention + SAL spatial-aware)."""
    return DITN(upscale=4, in_ch=3, dim=48, num_blocks=4, num_heads=6, patch=8).eval()


def example_input_ditn() -> torch.Tensor:
    """Example LR image ``(1, 3, 64, 64)`` for DITN x4."""
    return torch.randn(1, 3, 64, 64)


def build_drct_l() -> nn.Module:
    """Build DRCT-L x4 (deep: 6 dense Swin groups)."""
    return DRCT(upscale=4, dim=48, num_groups=6, depth=3, num_heads=6, window_size=8).eval()


def build_drct_s() -> nn.Module:
    """Build DRCT-S x4 (small: 4 dense Swin groups)."""
    return DRCT(upscale=4, dim=48, num_groups=4, depth=2, num_heads=6, window_size=8).eval()


def build_drct_sr() -> nn.Module:
    """Build DRCT x4 (base: 5 dense Swin groups)."""
    return DRCT(upscale=4, dim=48, num_groups=5, depth=3, num_heads=6, window_size=8).eval()


def example_input_drct() -> torch.Tensor:
    """Example LR image ``(1, 3, 64, 64)`` for DRCT x4 (window-aligned)."""
    return torch.randn(1, 3, 64, 64)


def build_duf() -> nn.Module:
    """Build DUF x4 (dynamic-upsampling-filter video SR, 7-frame clip)."""
    return DUF(scale=4, num_frames=7, filt=5).eval()


def example_input_duf() -> torch.Tensor:
    """Example clip ``(1, 7, 3, 32, 32)`` (B, T, C, H, W) for DUF."""
    return torch.randn(1, 7, 3, 32, 32)


def build_dvc() -> nn.Module:
    """Build DVC (end-to-end video compressor; wrapper supplies the reference frame)."""
    return _DVCWrapper(DVC()).eval()


def example_input_dvc() -> torch.Tensor:
    """Example RGB frame ``(1, 3, 64, 64)`` for DVC (reference synthesized internally)."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "DITN (UFONE: ITL intra-patch attention + SAL spatial-aware, x4 SR)",
        "build_ditn",
        "example_input_ditn",
        "2023",
        "DC",
    ),
    (
        "DRCT-L (dense-residual Swin-Transformer SR, deep)",
        "build_drct_l",
        "example_input_drct",
        "2024",
        "DC",
    ),
    (
        "DRCT-S (dense-residual Swin-Transformer SR, small)",
        "build_drct_s",
        "example_input_drct",
        "2024",
        "DC",
    ),
    (
        "DRCT (dense-residual Swin-Transformer SR, base x4)",
        "build_drct_sr",
        "example_input_drct",
        "2024",
        "DC",
    ),
    (
        "DUF (dynamic upsampling filters video SR, x4)",
        "build_duf",
        "example_input_duf",
        "2018",
        "DC",
    ),
    (
        "DVC (end-to-end deep video compression: flow + motion/residual codecs)",
        "build_dvc",
        "example_input_dvc",
        "2019",
        "DC",
    ),
]

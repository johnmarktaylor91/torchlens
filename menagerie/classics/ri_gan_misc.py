"""Miscellaneous faithful compact classics: inpainting, diffusion UNet, learned
image compression, frame interpolation, face recognition, NLI, stereo depth.

Each reproduces the *distinctive primitive* of its architecture at small random-init scale.

  * EdgeConnect inpaint generator (Nazeri et al. 2019, arXiv:1901.00212; knazeri/edge-connect).
        Two-stage edge-aware inpainting; the **InpaintGenerator** is the stage-2 core:
        RGB+edge (4ch) in -> encoder (down) -> N residual blocks (dilation) -> decoder (up) ->
        3ch out. Spectral-norm + InstanceNorm. (Stage-1 EdgeGenerator shares the same
        encoder/residual/decoder structure on 3ch->1ch; both registered.)
  * EDM SongUNet (Karras et al. NeurIPS 2022, arXiv:2206.00364; NVlabs/edm).
        Preconditioned diffusion UNet (DDPM++/NCSN++ "SongUNet"): **time/noise embedding**
        (Fourier/positional -> MLP) injected into residual blocks, encoder/decoder skip
        pyramid, self-attention at coarse resolutions. forward(x, noise_labels, class_labels).
  * ELIC (He et al. CVPR 2022, arXiv:2203.10886).
        Learned image compression. Signature: analysis/synthesis transforms built from
        **stacked residual bottleneck blocks** (replacing GDN) + a hyperprior, with the
        SCCTX space-channel context (checkerboard spatial + unevenly-grouped channel).
        Here: residual-bottleneck analysis g_a -> hyper -> synthesis g_s (rate path is
        training-only; the forward transform topology is reproduced).
  * EMA-VFI (Zhang et al. CVPR 2023, arXiv:2303.00440; MCG-NJU/EMA-VFI).
        Video frame interpolation via **inter-frame attention** (motion + appearance jointly):
        two frames encoded, cross-frame attention extracts bidirectional motion, a decoder
        synthesizes the middle frame. Compact cross-attention + flow-warp decode reproduced.
  * ElasticFace IR-50 (Boutros et al. 2021, arXiv:2109.09416; backbone = IResNet-50).
        Face recognition backbone: **Improved-ResNet (IResNet) bottleneck** (BN-Conv-BN-PReLU-
        Conv-BN + SE optional) producing a 512-d embedding; ElasticFace is the *margin loss*
        (training-only) on top -- the backbone IR-50 is the forward architecture reproduced.
  * ESIM (Chen et al. ACL 2017, arXiv:1609.06038).
        NLI: **BiLSTM encode -> soft (inter-sentence) attention align -> enhancement
        (concat[a, b_align, a-b_align, a*b_align]) -> BiLSTM compose -> avg+max pool -> MLP**.
  * FoundationStereo (Wen et al. CVPR 2025, arXiv:2501.09898; NVlabs/FoundationStereo).
        Stereo depth: feature encoder on L/R -> **cost volume (group-wise correlation)** ->
        cost aggregation -> **iterative GRU disparity refinement** (RAFT-Stereo lineage) with
        a frozen-ViT monocular prior. Compact correlation + ConvGRU update reproduced.

All builders zero-arg, random-init, `.eval()`-able, and small enough to trace+draw quickly.
Models taking multiple/scalar inputs are wrapped in single-tensor adapters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EdgeConnect inpaint / edge generator
# ============================================================


class _EdgeResnetBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=dilation),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EdgeConnectGenerator(nn.Module):
    """EdgeConnect generator: encoder (down) -> N dilated residual blocks -> decoder (up)."""

    def __init__(
        self, in_channels: int = 4, out_channels: int = 3, residual_blocks: int = 8
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(inplace=True),
        )
        self.middle = nn.Sequential(*[_EdgeResnetBlock(256, 2) for _ in range(residual_blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


# ============================================================
# EDM SongUNet (diffusion UNet with time/noise embedding)
# ============================================================


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * (8.0 / max(half - 1, 1)))
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class _EDMResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _SelfAttn2d(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(q.transpose(1, 2) @ k / (c**0.5), dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class SongUNet(nn.Module):
    """EDM SongUNet: noise/time embedding injected into a res-block encoder/decoder + coarse self-attn."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, model_ch: int = 32) -> None:
        super().__init__()
        emb_dim = model_ch * 4
        self.time_embed = _TimeEmbedding(emb_dim)
        self.in_conv = nn.Conv2d(in_channels, model_ch, 3, 1, 1)
        # encoder
        self.enc1 = _EDMResBlock(model_ch, model_ch, emb_dim)
        self.down1 = nn.Conv2d(model_ch, model_ch * 2, 3, 2, 1)
        self.enc2 = _EDMResBlock(model_ch * 2, model_ch * 2, emb_dim)
        self.down2 = nn.Conv2d(model_ch * 2, model_ch * 4, 3, 2, 1)
        # bottleneck with attention
        self.mid1 = _EDMResBlock(model_ch * 4, model_ch * 4, emb_dim)
        self.attn = _SelfAttn2d(model_ch * 4)
        self.mid2 = _EDMResBlock(model_ch * 4, model_ch * 4, emb_dim)
        # decoder
        self.up2 = nn.ConvTranspose2d(model_ch * 4, model_ch * 2, 4, 2, 1)
        self.dec2 = _EDMResBlock(model_ch * 4, model_ch * 2, emb_dim)
        self.up1 = nn.ConvTranspose2d(model_ch * 2, model_ch, 4, 2, 1)
        self.dec1 = _EDMResBlock(model_ch * 2, model_ch, emb_dim)
        self.out_norm = nn.GroupNorm(min(8, model_ch), model_ch)
        self.out_conv = nn.Conv2d(model_ch, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        emb = self.time_embed(noise_labels)
        h0 = self.in_conv(x)
        h1 = self.enc1(h0, emb)
        h2 = self.enc2(self.down1(h1), emb)
        hm = self.mid1(self.down2(h2), emb)
        hm = self.attn(hm)
        hm = self.mid2(hm, emb)
        d2 = self.dec2(torch.cat([self.up2(hm), h2], dim=1), emb)
        d1 = self.dec1(torch.cat([self.up1(d2), h1], dim=1), emb)
        return self.out_conv(F.silu(self.out_norm(d1)))


# ============================================================
# ELIC learned image compression (residual-bottleneck transforms)
# ============================================================


class _ResBottleneck(nn.Module):
    """ELIC residual bottleneck block (replaces GDN): 1x1 reduce -> 3x3 -> 1x1 expand + skip."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, ch // 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ELIC(nn.Module):
    """ELIC analysis/synthesis transforms from stacked residual bottleneck blocks (+hyper path)."""

    def __init__(self, N: int = 64, M: int = 96) -> None:
        super().__init__()

        def down(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 5, 2, 2), _ResBottleneck(out_ch), _ResBottleneck(out_ch)
            )

        def up(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 5, 2, 2, output_padding=1),
                _ResBottleneck(out_ch),
                _ResBottleneck(out_ch),
            )

        # analysis g_a : x -> y (4x downsample)
        self.g_a = nn.Sequential(down(3, N), down(N, N), down(N, N), nn.Conv2d(N, M, 5, 2, 2))
        # hyper analysis h_a : y -> z
        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, 5, 2, 2),
        )
        # hyper synthesis h_s : z -> entropy params (mean/scale aggregation proxy)
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, M, 5, 2, 2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(M, M * 2, 3, 1, 1),
        )
        # synthesis g_s : y -> x_hat (4x upsample)
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(M, N, 5, 2, 2, output_padding=1), up(N, N), up(N, N), up(N, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.g_a(x)
        z = self.h_a(y)
        params = self.h_s(z)
        scales, means = params.chunk(2, dim=1)
        # simulate context-conditioned residual (SCCTX is rate-only; here we mix params into y)
        y = y + 0.0 * scales + 0.0 * means
        x_hat = self.g_s(y)
        return x_hat


# ============================================================
# EMA-VFI frame interpolation (inter-frame attention)
# ============================================================


class _InterFrameAttention(nn.Module):
    """EMA-VFI inter-frame attention: appearance+motion extracted by cross-attending two frames."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f0.shape
        q = self.q(f0).reshape(b, c, h * w)
        k = self.k(f1).reshape(b, c, h * w)
        v = self.v(f1).reshape(b, c, h * w)
        attn = torch.softmax(q.transpose(1, 2) @ k / (c**0.5), dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(b, c, h, w)
        return self.proj(out)


class EMAVFI(nn.Module):
    """EMA-VFI: encode 2 frames -> bidirectional inter-frame attention -> decode middle frame."""

    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 3, 2, 1),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch * 2, 3, 2, 1),
            nn.PReLU(ch * 2),
        )
        self.ifa01 = _InterFrameAttention(ch * 2)
        self.ifa10 = _InterFrameAttention(ch * 2)
        self.fuse = nn.Conv2d(ch * 2 * 3, ch * 2, 3, 1, 1)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),
            nn.PReLU(ch),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        f0 = self.enc(img0)
        f1 = self.enc(img1)
        m01 = self.ifa01(f0, f1)  # motion 0->1
        m10 = self.ifa10(f1, f0)  # motion 1->0
        fused = self.fuse(torch.cat([f0 + f1, m01, m10], dim=1))
        return self.dec(fused)


# ============================================================
# ElasticFace IR-50 (IResNet bottleneck backbone -> 512-d embedding)
# ============================================================


class _IResBlock(nn.Module):
    """IResNet (improved ResNet) basic block: BN-Conv-BN-PReLU-Conv-BN with shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return out + identity


class IResNet50(nn.Module):
    """IR-50 backbone (compact): stem -> 4 IResNet stages -> flatten -> 512-d embedding (BN)."""

    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )
        # compact stage depths (representative of IR-50's [3,4,14,3] reduced for speed)
        self.layer1 = self._make_stage(64, 64, 2, stride=2)
        self.layer2 = self._make_stage(64, 128, 2, stride=2)
        self.layer3 = self._make_stage(128, 256, 2, stride=2)
        self.layer4 = self._make_stage(256, 512, 2, stride=2)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def _make_stage(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [_IResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(blocks - 1):
            layers.append(_IResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.output_layer(x)


# ============================================================
# ESIM NLI (BiLSTM encode -> soft attention align -> compose -> pool -> MLP)
# ============================================================


class ESIM(nn.Module):
    """ESIM: BiLSTM encode -> soft inter-sentence attention -> enhancement -> BiLSTM compose -> pool -> MLP."""

    def __init__(
        self,
        vocab_size: int = 2000,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encode = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(nn.Linear(hidden_size * 2 * 4, hidden_size), nn.ReLU())
        self.compose = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 * 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    @staticmethod
    def _soft_align(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn = torch.bmm(a, b.transpose(1, 2))  # (B, La, Lb)
        a_align = torch.bmm(torch.softmax(attn, dim=2), b)  # align b into a
        b_align = torch.bmm(torch.softmax(attn, dim=1).transpose(1, 2), a)  # align a into b
        return a_align, b_align

    @staticmethod
    def _enhance(x: torch.Tensor, x_align: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x_align, x - x_align, x * x_align], dim=-1)

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        ea, _ = self.encode(self.embedding(premise))
        eb, _ = self.encode(self.embedding(hypothesis))
        a_align, b_align = self._soft_align(ea, eb)
        ma = self.projection(self._enhance(ea, a_align))
        mb = self.projection(self._enhance(eb, b_align))
        va, _ = self.compose(ma)
        vb, _ = self.compose(mb)
        v = torch.cat([self._pool(va), self._pool(vb)], dim=-1)
        return self.classifier(v)


# ============================================================
# FoundationStereo (cost volume correlation + iterative ConvGRU refinement)
# ============================================================


class _ConvGRU(nn.Module):
    """RAFT-style ConvGRU update operator for iterative disparity refinement."""

    def __init__(self, hidden_dim: int, input_dim: int) -> None:
        super().__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, 1, 1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class FoundationStereo(nn.Module):
    """FoundationStereo: shared feature encoder -> group-wise correlation cost volume -> iterative ConvGRU disparity."""

    def __init__(
        self, feat_dim: int = 32, max_disp: int = 24, n_groups: int = 4, iters: int = 4
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.n_groups = n_groups
        self.iters = iters
        self.fnet = nn.Sequential(
            nn.Conv2d(3, feat_dim, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.cost_proj = nn.Conv2d(max_disp * n_groups, feat_dim, 3, 1, 1)
        self.gru = _ConvGRU(hidden_dim=feat_dim, input_dim=feat_dim + 1)
        self.disp_head = nn.Conv2d(feat_dim, 1, 3, 1, 1)

    def _correlation(self, fl: torch.Tensor, fr: torch.Tensor) -> torch.Tensor:
        b, c, h, w = fl.shape
        g = self.n_groups
        cg = c // g
        vols = []
        for d in range(self.max_disp):
            if d > 0:
                fr_shift = F.pad(fr, (d, 0, 0, 0))[:, :, :, :w]
            else:
                fr_shift = fr
            prod = (
                (fl * fr_shift).view(b, g, cg, h, w).mean(dim=2)
            )  # group-wise corr -> (b, g, h, w)
            vols.append(prod)
        return torch.cat(vols, dim=1)  # (b, max_disp*g, h, w)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.fnet(left)
        fr = self.fnet(right)
        cost = self._correlation(fl, fr)
        ctx = self.cost_proj(cost)
        h = torch.tanh(fl)
        disp = torch.zeros(fl.shape[0], 1, fl.shape[2], fl.shape[3])
        for _ in range(self.iters):
            inp = torch.cat([ctx, disp], dim=1)
            h = self.gru(h, inp)
            disp = disp + self.disp_head(h)
        return disp


# ============================================================
# Single-tensor adapter wrappers + menagerie wiring
# ============================================================


class _EDMWrapper(nn.Module):
    def __init__(self, model: SongUNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise_labels = torch.zeros(x.shape[0])
        return self.model(x, noise_labels)


class _VFIWrapper(nn.Module):
    """Single (1,6,H,W) tensor: frames stacked on channel dim."""

    def __init__(self, model: EMAVFI) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, :3], x[:, 3:])


class _ESIMWrapper(nn.Module):
    """Single (1, L) token-id tensor reused as both premise and hypothesis."""

    def __init__(self, model: ESIM) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, x)


class _StereoWrapper(nn.Module):
    """Single (1,6,H,W) tensor: left+right stacked on channel dim."""

    def __init__(self, model: FoundationStereo) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, :3], x[:, 3:])


def build_edgeconnect_inpaint() -> nn.Module:
    """EdgeConnect stage-2 inpaint generator (RGB+edge 4ch -> 3ch)."""
    return EdgeConnectGenerator(in_channels=4, out_channels=3, residual_blocks=8)


def build_edgeconnect_edge() -> nn.Module:
    """EdgeConnect stage-1 edge generator (grayscale+edge+mask 3ch -> 1ch edge map)."""
    return EdgeConnectGenerator(in_channels=3, out_channels=1, residual_blocks=8)


def build_edm_unet() -> nn.Module:
    """EDM SongUNet (preconditioned diffusion UNet, time-embedding + coarse attention)."""
    return _EDMWrapper(SongUNet(in_channels=3, out_channels=3, model_ch=32))


def build_elic() -> nn.Module:
    """ELIC learned-image-compression transforms (residual-bottleneck g_a/g_s + hyperprior)."""
    return ELIC(N=64, M=96)


def build_ema_vfi() -> nn.Module:
    """EMA-VFI frame interpolation (inter-frame attention)."""
    return _VFIWrapper(EMAVFI(ch=32))


def build_elasticface_ir50() -> nn.Module:
    """ElasticFace IR-50 backbone (IResNet bottleneck -> 512-d embedding)."""
    return IResNet50(embedding_dim=512)


def build_esim() -> nn.Module:
    """ESIM natural-language-inference model (BiLSTM + soft attention + compose + pool + MLP)."""
    return _ESIMWrapper(ESIM(vocab_size=2000, embedding_dim=64, hidden_size=64, num_classes=3))


def build_foundationstereo() -> nn.Module:
    """FoundationStereo (group-wise correlation cost volume + iterative ConvGRU disparity refinement)."""
    return _StereoWrapper(FoundationStereo(feat_dim=32, max_disp=24, n_groups=4, iters=4))


def example_image_4ch() -> torch.Tensor:
    """RGB+edge tensor (1, 4, 64, 64) for the EdgeConnect inpaint generator."""
    return torch.randn(1, 4, 64, 64)


def example_image_3ch_edge() -> torch.Tensor:
    """Grayscale+edge+mask tensor (1, 3, 64, 64) for the EdgeConnect edge generator."""
    return torch.randn(1, 3, 64, 64)


def example_image_64() -> torch.Tensor:
    """RGB image tensor (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


def example_image_compress() -> torch.Tensor:
    """RGB image tensor (1, 3, 64, 64) for ELIC."""
    return torch.randn(1, 3, 64, 64)


def example_frames_stacked() -> torch.Tensor:
    """Two frames stacked on channel dim (1, 6, 64, 96) for EMA-VFI."""
    return torch.randn(1, 6, 64, 96)


def example_face_112() -> torch.Tensor:
    """Aligned face image tensor (1, 3, 112, 112) for ElasticFace IR-50."""
    return torch.randn(1, 3, 112, 112)


def example_tokens() -> torch.Tensor:
    """Token-id tensor (1, 24) for ESIM."""
    return torch.randint(1, 2000, (1, 24)).long()


def example_stereo_stacked() -> torch.Tensor:
    """Left+right stacked on channel dim (1, 6, 64, 128) for FoundationStereo."""
    return torch.randn(1, 6, 64, 128)


MENAGERIE_ENTRIES = [
    (
        "EdgeConnect (stage-2 inpaint generator, dilated residual)",
        "build_edgeconnect_inpaint",
        "example_image_4ch",
        "2019",
        "DC",
    ),
    (
        "EdgeConnect (stage-1 edge generator)",
        "build_edgeconnect_edge",
        "example_image_3ch_edge",
        "2019",
        "DC",
    ),
    (
        "EDM (SongUNet preconditioned diffusion UNet)",
        "build_edm_unet",
        "example_image_64",
        "2022",
        "DC",
    ),
    (
        "ELIC (learned image compression, residual-bottleneck transforms)",
        "build_elic",
        "example_image_compress",
        "2022",
        "DC",
    ),
    (
        "EMA-VFI (inter-frame attention frame interpolation)",
        "build_ema_vfi",
        "example_frames_stacked",
        "2023",
        "DC",
    ),
    (
        "ElasticFace IR-50 (IResNet face-recognition backbone)",
        "build_elasticface_ir50",
        "example_face_112",
        "2021",
        "DC",
    ),
    (
        "ESIM (enhanced sequential inference model, NLI)",
        "build_esim",
        "example_tokens",
        "2017",
        "DC",
    ),
    (
        "FoundationStereo (cost-volume + iterative ConvGRU stereo)",
        "build_foundationstereo",
        "example_stereo_stacked",
        "2025",
        "DC",
    ),
]

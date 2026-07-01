"""Menagerie batch w2a13: compact, faithful reimplementations of 5 architectures.

Candidates covered (rows 115-120 of build_queue.tsv):
  - DeepUPE            (build_deepupe / example_input_deepupe)
  - DeepVecFont-v2      (build_deepvecfont_v2 / example_input_deepvecfont_v2)
  - DFNet Inpainting    (build_dfnet_inpainting / example_input_dfnet_inpainting)
  - DRUNet              (build_drunet / example_input_drunet)
  - DeepSketch2Face     (build_deepsketch2face / example_input_deepsketch2face)

DiffSketcher (cand_00175) is SKIPPED -- see rationale at the bottom of this file.

Sources checked (paper + official/reference code, read via GitHub, not cloned):
  - DeepUPE: Wang et al., "Underexposed Photo Enhancement Using Deep Illumination
    Estimation", CVPR 2019. https://arxiv.org/abs/1902.04249
    Official repo (TF/HDRNet-based): https://github.com/wangruixing/DeepUPE
    (mirrored at JIA-Lab-research/DeepUPE) -- read main/models.py (HDRNetCurves:
    splat/local/global towers, bilateral-grid coefficient prediction, guide-map
    curve, bilateral slicing) and main/hdrnet_ops.py (bilateral_slice_apply).
  - DeepVecFont-v2: Wang & Lian, "DeepVecFont: Synthesizing High-quality Vector
    Fonts via Dual-modality Learning", SIGGRAPH Asia 2021 (arXiv:2110.06688), and
    the CVPR 2023 DeepVecFont-v2 follow-up. Official repo
    https://github.com/yizhiwang96/deepvecfont-v2 -- read models/image_encoder.py,
    models/image_decoder.py, models/modality_fusion.py, models/model_main.py
    (dual CNN image encoder/decoder + sequence transformer + modality-fusion VAE
    bottleneck that concatenates image and sequence features into mu/log-sigma).
  - DFNet Inpainting: Hong et al., "Deep Fusion Network for Image Completion",
    ACM MM 2019 (arXiv:1904.08060). Official repo
    https://github.com/hughplay/DFNet -- read model.py directly (EncodeBlock /
    DecodeBlock U-Net with per-scale FusionBlock: image + raw prediction blended
    via a learned per-pixel alpha map, at multiple decoder depths).
  - DRUNet: Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser
    Prior", TPAMI 2021 (arXiv:2008.13751). Official repo
    https://github.com/cszn/DPIR -- architecture confirmed via paper text and
    DeepWiki summary (4-scale residual U-Net, strided-conv down / transposed-conv
    up, noise-level map concatenated as an extra input channel).
  - DeepSketch2Face: Han et al., "DeepSketch2Face: A Deep Learning Based
    Sketching System for 3D Face and Caricature Modeling", SIGGRAPH 2017
    (ACM TOG 36(4), arXiv:1706.02042). Demo/DB repo
    https://github.com/irsisyphus/deepsketch2face -- architecture (CNN sketch
    encoder feeding two independent FC branches that regress identity and
    expression coefficients of a bilinear face model) taken from the paper
    abstract/description; reimplemented compactly with a small bilinear-model
    blend head standing in for the FaceWarehouse bilinear tensor.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DeepUPE -- HDRNet-style bilateral-grid illumination estimation
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv2d + optional BatchNorm + ReLU, used throughout the HDRNet towers."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, kernel_size // 2, bias=not norm)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepUPE(nn.Module):
    """HDRNet-style bilateral-grid illumination-map estimator (DeepUPE).

    Predicts a low-resolution 3D bilateral grid of per-pixel affine color
    transforms from a downsampled input, using a splat tower (shared low-level
    features) that fans out into a global tower (image-wide context, fully
    connected) and a local tower (spatially-resolved), fused and projected
    into ``luma_bins`` affine-coefficient slices.  A full-resolution guide map
    (a small pointwise network) then determines which luma slice each
    full-resolution pixel reads from (bilateral slicing), and the resulting
    per-pixel affine map is applied to the full-resolution input to produce
    an estimated illumination map; the enhanced image is input / illumination.
    """

    def __init__(
        self,
        lowres_size: int = 64,
        luma_bins: int = 4,
        channel_multiplier: int = 4,
        spatial_bin: int = 8,
    ) -> None:
        super().__init__()
        self.luma_bins = luma_bins
        self.spatial_bin = spatial_bin
        gd = luma_bins
        cm = channel_multiplier
        n_in, n_out = 4, 3  # affine: 3 in-channels + bias, 3 out-channels

        # --- splat tower: shared low-level downsampling to spatial_bin ---
        n_ds = int(math.log2(lowres_size / spatial_bin))
        splat_layers = []
        c_in = 3
        for i in range(n_ds):
            c_out = cm * (2**i) * gd
            splat_layers.append(ConvBlock(c_in, c_out, 3, stride=2, norm=(i > 0)))
            c_in = c_out
        self.splat = nn.Sequential(*splat_layers)
        c_splat = c_in

        # --- global tower: further downsample to 4x4, then FC to a vector ---
        n_global_ds = int(math.log2(spatial_bin / 4))
        global_layers = []
        c_in = c_splat
        c_glob = 8 * cm * gd
        for i in range(n_global_ds):
            global_layers.append(ConvBlock(c_in, c_glob, 3, stride=2))
            c_in = c_glob
        global_layers.append(ConvBlock(c_in, c_glob, 3, stride=1))
        self.global_conv = nn.Sequential(*global_layers)
        flat_dim = c_glob * 4 * 4
        self.global_fc = nn.Sequential(
            nn.Linear(flat_dim, 8 * cm * gd),
            nn.ReLU(inplace=True),
            nn.Linear(8 * cm * gd, 4 * cm * gd),
            nn.ReLU(inplace=True),
            nn.Linear(4 * cm * gd, c_glob),
        )

        # --- local tower: keep spatial_bin resolution, no downsampling ---
        self.local = nn.Sequential(
            ConvBlock(c_splat, c_glob, 3, stride=1),
            ConvBlock(c_glob, c_glob, 3, stride=1, norm=False, act=False),
        )

        # --- prediction: fuse global + local, project to bilateral grid ---
        self.predict = nn.Conv2d(c_glob, gd * n_out * n_in, 1)

        # --- guide map: small pointwise network producing per-pixel index ---
        self.guide = nn.Sequential(
            ConvBlock(3, 16, 1, stride=1),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.n_in = n_in
        self.n_out = n_out

    def _bilateral_slice_apply(
        self, grid: torch.Tensor, guide: torch.Tensor, full_img: torch.Tensor
    ) -> torch.Tensor:
        """Trilinear bilateral slicing + affine apply via grid_sample.

        ``grid``: (B, gd*n_out*n_in, Hs, Ws) bilateral grid of affine coeffs.
        ``guide``: (B, 1, H, W) per-pixel luma index in [0, 1].
        ``full_img``: (B, 3, H, W) full-resolution input image.
        """
        b, _, h, w = full_img.shape
        gd, n_out, n_in = self.luma_bins, self.n_out, self.n_in
        grid_3d = grid.view(b, n_out * n_in, gd, grid.shape[-2], grid.shape[-1])
        # 3D grid_sample over (luma, y, x) using guide as the depth coordinate;
        # a single depth slab (D_out=1) is sampled per pixel, and grid_sample
        # gathers ALL channels (the full affine-coefficient grid) at once.
        ys = torch.linspace(-1, 1, h, device=full_img.device)
        xs = torch.linspace(-1, 1, w, device=full_img.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        gy = gy.unsqueeze(0).expand(b, h, w)
        gx = gx.unsqueeze(0).expand(b, h, w)
        gz = guide[:, 0] * 2 - 1  # (B, H, W) in [-1, 1]
        sample_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1)  # (B,1,H,W,3)
        sliced = F.grid_sample(
            grid_3d, sample_grid, mode="bilinear", align_corners=True
        )  # (B, n_out*n_in, 1, H, W)
        coeffs = sliced.squeeze(2).view(b, n_out, n_in, h, w)
        affine = coeffs[:, :, :3, :, :]
        bias = coeffs[:, :, 3, :, :]
        out = torch.einsum("boihw,bihw->bohw", affine, full_img) + bias
        return out

    def forward(self, full_img: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, 3, H, W) illumination map estimate."""
        lowres = F.interpolate(full_img, size=(64, 64), mode="bilinear", align_corners=False)
        splat = self.splat(lowres)
        glob = self.global_conv(splat)
        glob_vec = self.global_fc(glob.flatten(1))
        glob_vec = glob_vec.view(glob_vec.shape[0], -1, 1, 1)
        local = self.local(splat)
        fused = F.relu(local + glob_vec)
        grid = self.predict(fused)
        guide = self.guide(full_img)
        illum = self._bilateral_slice_apply(grid, guide, full_img)
        return torch.sigmoid(illum)


def build_deepupe() -> nn.Module:
    """Build a compact DeepUPE illumination-estimation network."""
    return DeepUPE(lowres_size=64, luma_bins=4, channel_multiplier=2, spatial_bin=8).eval()


def example_input_deepupe() -> torch.Tensor:
    """(1, 3, 64, 64) RGB underexposed photo."""
    return torch.rand(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# DeepVecFont-v2 -- dual-modality (image + sequence) font VAE
# ---------------------------------------------------------------------------


class VFImageEncoder(nn.Module):
    """CNN glyph-image encoder (mirrors deepvecfont-v2's ImageEncoder)."""

    def __init__(self, img_size: int = 32, in_ch: int = 1, ngf: int = 8) -> None:
        super().__init__()
        n_down = int(math.log2(img_size))
        layers: list[nn.Module] = [nn.Conv2d(in_ch, ngf, 7, padding=3), nn.ReLU(True)]
        c_prev = ngf
        for i in range(n_down):
            c_next = ngf * (2 ** (i + 1))
            layers += [nn.Conv2d(c_prev, c_next, 3, stride=2, padding=1), nn.ReLU(True)]
            c_prev = c_next
        self.encode = nn.Sequential(*layers)
        self.out_ch = c_prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x).flatten(1)


class VFImageDecoder(nn.Module):
    """CNN glyph-image decoder from a fused latent + char one-hot code."""

    def __init__(self, img_size: int = 32, in_dim: int = 64, ngf: int = 8) -> None:
        super().__init__()
        n_up = int(math.log2(img_size))
        layers: list[nn.Module] = []
        c_prev = in_dim
        for i in range(n_up):
            c_next = max(ngf, in_dim // (2 ** (i + 1)))
            layers += [
                nn.ConvTranspose2d(c_prev, c_next, 4, stride=2, padding=1),
                nn.ReLU(True),
            ]
            c_prev = c_next
        layers += [nn.Conv2d(c_prev, 1, 3, padding=1), nn.Sigmoid()]
        self.decode = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        return self.decode(z)


class VFSeqEncoder(nn.Module):
    """Small transformer encoder over the SVG command/argument sequence."""

    def __init__(self, dim_seq: int = 9, d_model: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_seq, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=4 * d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, dim_seq) -> (B, d_model) pooled sequence feature (CLS)."""
        x = self.proj(seq)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]


class DeepVecFontV2(nn.Module):
    """Dual-modality (image + vector-sequence) font-glyph VAE.

    An image CNN encoder and a sequence transformer encoder each produce a
    feature vector; these are concatenated and mapped to VAE mu/log-sigma
    parameters (the modality-fusion bottleneck).  The reparameterized latent
    is concatenated with a target-character one-hot and decoded back into a
    glyph raster image by a transposed-conv CNN decoder -- the distinctive
    DeepVecFont mechanism of fusing raster and vector-outline modalities into
    one shared latent code.
    """

    def __init__(
        self,
        img_size: int = 32,
        char_num: int = 26,
        dim_seq: int = 9,
        d_model: int = 64,
        bottleneck_bits: int = 32,
    ) -> None:
        super().__init__()
        self.bottleneck_bits = bottleneck_bits
        self.img_encoder = VFImageEncoder(img_size=img_size, in_ch=1, ngf=8)
        self.seq_encoder = VFSeqEncoder(dim_seq=dim_seq, d_model=d_model, n_layers=2)
        self.fc_fusion = nn.Linear(self.img_encoder.out_ch + d_model, bottleneck_bits * 2)
        self.img_decoder = VFImageDecoder(
            img_size=img_size, in_dim=bottleneck_bits + char_num, ngf=8
        )

    def forward(
        self, ref_img: torch.Tensor, ref_seq: torch.Tensor, trg_char_onehot: torch.Tensor
    ) -> torch.Tensor:
        """ref_img: (B,1,H,W); ref_seq: (B,T,dim_seq); trg_char_onehot: (B,char_num).

        Returns the reconstructed target glyph image (B, 1, H, W).
        """
        img_feat = self.img_encoder(ref_img)
        seq_feat = self.seq_encoder(ref_seq)
        fused = torch.cat([img_feat, seq_feat], dim=-1)
        dist_param = self.fc_fusion(fused)
        mu, log_sigma = dist_param.chunk(2, dim=-1)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * log_sigma) * eps
        dec_in = torch.cat([z, trg_char_onehot], dim=-1)
        return self.img_decoder(dec_in)


def build_deepvecfont_v2() -> nn.Module:
    """Build a compact DeepVecFont-v2 dual-modality glyph VAE."""
    return DeepVecFontV2(img_size=32, char_num=26, dim_seq=9, d_model=64, bottleneck_bits=32).eval()


def example_input_deepvecfont_v2() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(ref_img (1,1,32,32), ref_seq (1,16,9), trg_char_onehot (1,26))."""
    ref_img = torch.rand(1, 1, 32, 32)
    ref_seq = torch.randn(1, 16, 9)
    trg_char_onehot = F.one_hot(torch.tensor([3]), num_classes=26).float()
    return ref_img, ref_seq, trg_char_onehot


# ---------------------------------------------------------------------------
# DFNet Inpainting -- fusion-block U-Net with multi-scale alpha blending
# ---------------------------------------------------------------------------


class DFEncodeBlock(nn.Module):
    """Strided conv encoder stage."""

    def __init__(self, c_in: int, c_out: int, k: int, norm: bool, act: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(c_in, c_out, k, stride=2, padding=k // 2)]
        if norm:
            layers.append(nn.BatchNorm2d(c_out))
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.encode = nn.Sequential(*layers)
        self.c_in, self.c_out = c_in, c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class DFDecodeBlock(nn.Module):
    """Nearest-upsample + concat-skip + conv decoder stage."""

    def __init__(self, c_up: int, c_down: int, c_out: int, k: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_up + c_down, c_out, k, stride=1, padding=k // 2),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.c_down = c_down
        self.c_out = c_out

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.c_down > 0:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DFFusionBlock(nn.Module):
    """Projects decoder features to RGB and blends with the masked input via
    a learned per-pixel alpha map (DFNet's namesake fusion mechanism)."""

    def __init__(self, c_feat: int) -> None:
        super().__init__()
        self.map2img = nn.Sequential(nn.Conv2d(c_feat, 3, 1), nn.Sigmoid())
        c_mid = max((3 * 2) // 2, 16)
        self.blend = nn.Sequential(
            nn.Conv2d(6, c_mid, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_mid, 3, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, img_miss: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        img_miss = F.interpolate(
            img_miss, size=feat.shape[-2:], mode="bilinear", align_corners=False
        )
        raw = self.map2img(feat)
        alpha = self.blend(torch.cat([img_miss, raw], dim=1))
        return alpha * raw + (1 - alpha) * img_miss


class DFNetInpainting(nn.Module):
    """Compact DFNet: encoder-decoder U-Net with FusionBlocks at every scale.

    Each decoder stage produces its own RGB completion at that resolution via
    a FusionBlock (raw-prediction + learned alpha-blend against the masked
    input) -- the distinctive multi-scale fusion mechanism from the paper.
    Only the finest-scale (full-resolution) result is returned here.
    """

    def __init__(self, base_ch: int = 24, n_layers: int = 4) -> None:
        super().__init__()
        en_k = [7, 5, 3, 3][:n_layers]
        de_k = [3] * n_layers

        self.en = nn.ModuleList()
        c_in = 4  # RGB + mask
        for i, k in enumerate(en_k):
            c_out = min(base_ch * (2**i), 8 * base_ch)
            self.en.append(DFEncodeBlock(c_in, c_out, k, norm=(i > 0), act=True))
            c_in = c_out

        self.de = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i, k in enumerate(de_k):
            c_up = self.en[-1].c_out if i == 0 else self.de[-1].c_out
            en_match = self.en[-i - 1]
            c_down = en_match.c_in
            c_out = c_down
            self.de.append(DFDecodeBlock(c_up, c_down, c_out, k))
            self.fuse.append(DFFusionBlock(c_out))

    def forward(self, img_miss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """img_miss: (B,3,H,W) masked RGB; mask: (B,1,H,W). Returns (B,3,H,W)."""
        x = torch.cat([img_miss, mask], dim=1)
        feats = [x]
        for enc in self.en:
            x = enc(x)
            feats.append(x)

        result = None
        for i, (dec, fuse) in enumerate(zip(self.de, self.fuse)):
            skip = feats[-i - 2]
            x = dec(x, skip)
            result = fuse(img_miss, x)
        return result


def build_dfnet_inpainting() -> nn.Module:
    """Build a compact 4-scale DFNet inpainting model."""
    return DFNetInpainting(base_ch=16, n_layers=4).eval()


def example_input_dfnet_inpainting() -> tuple[torch.Tensor, torch.Tensor]:
    """(masked RGB (1,3,64,64), binary mask (1,1,64,64))."""
    img_miss = torch.rand(1, 3, 64, 64)
    mask = (torch.rand(1, 1, 64, 64) > 0.5).float()
    return img_miss, mask


# ---------------------------------------------------------------------------
# DRUNet -- 4-scale residual U-Net denoiser with a noise-level-map input
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """3x3 conv - ReLU - 3x3 conv residual block (no norm, per DRUNet)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class DRUNet(nn.Module):
    """Deep Residual U-Net plug-and-play denoiser (DPIR).

    4 scales; each scale applies ``n_resblocks`` residual blocks then
    downsamples via a strided conv (encoder) / upsamples via a transposed
    conv (decoder) with additive skip connections, mirroring cszn/DPIR's
    UNetRes.  The noisy image is concatenated with a uniform noise-level map
    as an extra input channel, letting one network denoise at any sigma.
    """

    def __init__(
        self, in_ch: int = 3, nc: tuple[int, ...] = (16, 32, 64, 128), n_resblocks: int = 2
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_ch + 1, nc[0], 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(len(nc) - 1):
            self.down_blocks.append(nn.Sequential(*[ResBlock(nc[i]) for _ in range(n_resblocks)]))
            self.downsample.append(nn.Conv2d(nc[i], nc[i + 1], 2, stride=2))

        self.body = nn.Sequential(*[ResBlock(nc[-1]) for _ in range(n_resblocks)])

        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(len(nc) - 1, 0, -1):
            self.upsample.append(nn.ConvTranspose2d(nc[i], nc[i - 1], 2, stride=2))
            self.up_blocks.append(nn.Sequential(*[ResBlock(nc[i - 1]) for _ in range(n_resblocks)]))

        self.tail = nn.Conv2d(nc[0], in_ch, 3, padding=1)

    def forward(self, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """noisy: (B,3,H,W); sigma: (B,1,1,1) uniform noise-level map value."""
        noise_map = sigma.expand(-1, 1, noisy.shape[-2], noisy.shape[-1])
        x = self.head(torch.cat([noisy, noise_map], dim=1))

        skips = []
        for down_block, down in zip(self.down_blocks, self.downsample):
            x = down_block(x)
            skips.append(x)
            x = down(x)

        x = self.body(x)

        for up, up_block, skip in zip(self.upsample, self.up_blocks, reversed(skips)):
            x = up(x)
            x = x + skip
            x = up_block(x)

        return noisy - self.tail(x)


def build_drunet() -> nn.Module:
    """Build a compact 4-scale DRUNet denoiser."""
    return DRUNet(in_ch=3, nc=(16, 32, 64, 128), n_resblocks=2).eval()


def example_input_drunet() -> tuple[torch.Tensor, torch.Tensor]:
    """(noisy RGB (1,3,64,64), uniform sigma map value (1,1,1,1))."""
    noisy = torch.rand(1, 3, 64, 64)
    sigma = torch.full((1, 1, 1, 1), 0.1)
    return noisy, sigma


# ---------------------------------------------------------------------------
# DeepSketch2Face -- dual-branch CNN regression to bilinear-face coefficients
# ---------------------------------------------------------------------------


class DeepSketch2Face(nn.Module):
    """Sketch -> 3D face regression network (dual-branch bilinear model).

    A shared CNN trunk extracts features from a binary sketch image; two
    independent fully-connected branches then regress the identity and
    expression coefficient vectors of a bilinear face model separately (the
    paper's key design: decoupling identity/expression regression instead of
    predicting one flat mesh-coefficient vector).  The predicted
    identity/expression coefficients are finally combined through a small
    learned bilinear-blend head that stands in for the (fixed,
    dataset-derived) FaceWarehouse core tensor contraction, producing the
    final per-vertex-basis face coefficient vector.
    """

    def __init__(
        self,
        img_size: int = 64,
        n_id: int = 30,
        n_exp: int = 20,
        n_verts_basis: int = 64,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        flat_dim = 64 * (img_size // 16) * (img_size // 16)

        self.id_branch = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_id),
        )
        self.exp_branch = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_exp),
        )

        # Bilinear blend: outer product of (identity, expression) coefficients
        # contracted with a learned core tensor, standing in for the fixed
        # FaceWarehouse bilinear core tensor used in the original paper.
        self.core_tensor = nn.Parameter(torch.randn(n_verts_basis, n_id, n_exp) * 0.02)

    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        """sketch: (B, 1, H, W) binary sketch image.

        Returns (B, n_verts_basis) bilinear face-coefficient vector.
        """
        feat = self.trunk(sketch).flatten(1)
        id_coef = self.id_branch(feat)
        exp_coef = self.exp_branch(feat)
        face_coef = torch.einsum("vij,bi,bj->bv", self.core_tensor, id_coef, exp_coef)
        return face_coef


def build_deepsketch2face() -> nn.Module:
    """Build a compact DeepSketch2Face sketch-to-face-coefficients network."""
    return DeepSketch2Face(img_size=64, n_id=30, n_exp=20, n_verts_basis=64).eval()


def example_input_deepsketch2face() -> torch.Tensor:
    """(1, 1, 64, 64) binary face-sketch image."""
    return (torch.rand(1, 1, 64, 64) > 0.7).float()


# ---------------------------------------------------------------------------
# SKIPPED: DiffSketcher (cand_00175)
# ---------------------------------------------------------------------------
#
# DiffSketcher (Xing et al., NeurIPS 2023) is a *test-time optimization*
# method, not a trainable nn.Module architecture: it directly optimizes the
# control-point / width / opacity tensors of a fixed number of Bezier curves
# per input prompt using a score-distillation-sampling (SDS) loss computed
# through a *frozen*, pretrained Stable Diffusion U-Net (via diffusers) and a
# differentiable rasterizer (diffvg). Its only "learned" components (the
# attention-aware stroke-initialization heuristic, and the differentiable
# rasterizer itself) are not neural networks -- there is no distinct,
# trainable nn.Module whose weights are learned end-to-end for this task; the
# curve parameters ARE the optimization variables, updated directly by an
# optimizer against a frozen backbone. This matches the "pure
# prompting/optimization loop over a frozen model with no trained component"
# skip criterion, so it is skipped rather than stubbed into a fake network.


MENAGERIE_ENTRIES = [
    (
        "DeepUPE",
        "build_deepupe",
        "example_input_deepupe",
        "2019",
        "DC",
    ),
    (
        "DeepVecFont",
        "build_deepvecfont_v2",
        "example_input_deepvecfont_v2",
        "2021",
        "GEN",
    ),
    (
        "DFNet Inpainting",
        "build_dfnet_inpainting",
        "example_input_dfnet_inpainting",
        "2019",
        "DC",
    ),
    (
        "DRUNet",
        "build_drunet",
        "example_input_drunet",
        "2021",
        "DC",
    ),
    (
        "DeepSketch2Face",
        "build_deepsketch2face",
        "example_input_deepsketch2face",
        "2017",
        "DC",
    ),
]

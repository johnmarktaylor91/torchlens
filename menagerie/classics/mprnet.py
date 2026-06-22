"""MPRNet: Multi-Stage Progressive Image Restoration.

Zamir et al., CVPR 2021.
Paper: https://arxiv.org/abs/2102.02808
Source: https://github.com/swz30/MPRNet

MPRNet decomposes restoration into three progressively-coupled stages:

  Stage 1 & Stage 2  -- encoder-decoder subnetworks built from Channel
                        Attention Blocks (CAB).  Stage 1 operates on
                        non-overlapping spatial patches (top/bottom), Stage 2
                        on left/right patches.  Cross-Stage Feature Fusion
                        (CSFF) carries Stage-1 encoder/decoder features into
                        Stage 2's encoder.
  Supervised Attention Module (SAM) -- between stages, produces a restored
                        image at the current stage and an attention-gated
                        feature map that is passed forward to the next stage.
  Stage 3            -- an Original-Resolution Subnetwork (ORSNet) of stacked
                        Original-Resolution Blocks (ORB).  No down/upsampling;
                        it refines the full-resolution feature stream after
                        fusing the previous stages' multi-scale features.

The three catalog variants (deblurring / denoising / deraining) share ONE
parametric MPRNet module and differ only by configuration (widths, ORB depth):
deblurring / deraining use the full encoder-decoder + ORSNet at the published
config; denoising uses a similar lighter config.  All widths here are reduced
(n_feat=40, num_cab=4) so the captured graph stays small and ``draw`` is fast,
while the architecture (3 stages, CAB/ORB/SAM/ORSNet, CSFF, patch supervision)
is reproduced faithfully.

MPRNet.forward returns a list ``[stage3_img, stage2_img, stage1_img]``; the
menagerie wrapper returns the full-resolution Stage-3 output as one tensor.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# ============================================================
# Basic blocks (faithful to the MPRNet source)
# ============================================================


def conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = False, stride: int = 1
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


class CALayer(nn.Module):
    """Channel attention: squeeze (global avg-pool) -> excite (1x1 bottleneck)."""

    def __init__(self, channel: int, reduction: int = 16, bias: bool = False) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    """Channel Attention Block: conv-act-conv with channel attention + residual."""

    def __init__(
        self, n_feat: int, kernel_size: int, reduction: int, bias: bool, act: nn.Module
    ) -> None:
        super().__init__()
        modules_body = [
            conv(n_feat, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias),
        ]
        self.body = nn.Sequential(*modules_body)
        self.CA = CALayer(n_feat, reduction, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = self.CA(res)
        res = res + x
        return res


class SAM(nn.Module):
    """Supervised Attention Module: emit a restored image and an attention-gated feature map."""

    def __init__(self, n_feat: int, kernel_size: int, bias: bool) -> None:
        super().__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x: torch.Tensor, x_img: torch.Tensor):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


# ============================================================
# U-Net (Encoder + Decoder) used in Stage 1 and Stage 2
# ============================================================


class DownSample(nn.Module):
    def __init__(self, in_channels: int, s_factor: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, s_factor: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class SkipUpSample(nn.Module):
    def __init__(self, in_channels: int, s_factor: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        bias: bool,
        scale_unetfeats: int,
        csff: bool,
        num_cab: int = 2,
    ) -> None:
        super().__init__()
        self.encoder_level1 = nn.Sequential(
            *[CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        )
        self.encoder_level2 = nn.Sequential(
            *[
                CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
                for _ in range(num_cab)
            ]
        )
        self.encoder_level3 = nn.Sequential(
            *[
                CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act)
                for _ in range(num_cab)
            ]
        )
        self.down12 = DownSample(n_feat, scale_unetfeats)
        self.down23 = DownSample(n_feat + scale_unetfeats, scale_unetfeats)

        self.csff = csff
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(
                n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias
            )
            self.csff_enc3 = nn.Conv2d(
                n_feat + (scale_unetfeats * 2),
                n_feat + (scale_unetfeats * 2),
                kernel_size=1,
                bias=bias,
            )
            self.csff_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(
                n_feat + scale_unetfeats, n_feat + scale_unetfeats, kernel_size=1, bias=bias
            )
            self.csff_dec3 = nn.Conv2d(
                n_feat + (scale_unetfeats * 2),
                n_feat + (scale_unetfeats * 2),
                kernel_size=1,
                bias=bias,
            )

    def forward(self, x: torch.Tensor, encoder_outs=None, decoder_outs=None) -> List[torch.Tensor]:
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)
        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)
        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]


class Decoder(nn.Module):
    def __init__(
        self,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        bias: bool,
        scale_unetfeats: int,
        num_cab: int = 2,
    ) -> None:
        super().__init__()
        self.decoder_level1 = nn.Sequential(
            *[CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        )
        self.decoder_level2 = nn.Sequential(
            *[
                CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
                for _ in range(num_cab)
            ]
        )
        self.decoder_level3 = nn.Sequential(
            *[
                CAB(n_feat + (scale_unetfeats * 2), kernel_size, reduction, bias=bias, act=act)
                for _ in range(num_cab)
            ]
        )
        self.skip_attn1 = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat + scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.up21 = SkipUpSample(n_feat, scale_unetfeats)
        self.up32 = SkipUpSample(n_feat + scale_unetfeats, scale_unetfeats)

    def forward(self, outs: List[torch.Tensor]) -> List[torch.Tensor]:
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)
        return [dec1, dec2, dec3]


# ============================================================
# Original-Resolution Subnetwork (Stage 3)
# ============================================================


class ORB(nn.Module):
    """Original-Resolution Block: stacked CABs at full resolution + residual."""

    def __init__(
        self,
        n_feat: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        bias: bool,
        num_cab: int,
    ) -> None:
        super().__init__()
        modules_body = [
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = res + x
        return res


class ORSNet(nn.Module):
    def __init__(
        self,
        n_feat: int,
        scale_orsnetfeats: int,
        kernel_size: int,
        reduction: int,
        act: nn.Module,
        bias: bool,
        scale_unetfeats: int,
        num_cab: int,
    ) -> None:
        super().__init__()
        self.orb1 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat + scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)
        self.up_enc2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats)
        )
        self.up_dec2 = nn.Sequential(
            UpSample(n_feat + scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats)
        )

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat + scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(
        self, x: torch.Tensor, encoder_outs: List[torch.Tensor], decoder_outs: List[torch.Tensor]
    ) -> torch.Tensor:
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = (
            x
            + self.conv_enc2(self.up_enc1(encoder_outs[1]))
            + self.conv_dec2(self.up_dec1(decoder_outs[1]))
        )

        x = self.orb3(x)
        x = (
            x
            + self.conv_enc3(self.up_enc2(encoder_outs[2]))
            + self.conv_dec3(self.up_dec2(decoder_outs[2]))
        )
        return x


# ============================================================
# Full MPRNet
# ============================================================


class MPRNet(nn.Module):
    """3-stage progressive image restoration network (random-init reimplementation)."""

    def __init__(
        self,
        in_c: int = 3,
        out_c: int = 3,
        n_feat: int = 40,
        scale_unetfeats: int = 20,
        scale_orsnetfeats: int = 16,
        num_cab: int = 4,
        kernel_size: int = 3,
        reduction: int = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(
            conv(in_c, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
        )
        self.shallow_feat2 = nn.Sequential(
            conv(in_c, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
        )
        self.shallow_feat3 = nn.Sequential(
            conv(in_c, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
        )

        # Stage 1 and Stage 2 share encoder/decoder topology (CSFF only in stage 2's encoder).
        self.stage1_encoder = Encoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False, num_cab=2
        )
        self.stage1_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, num_cab=2
        )

        self.stage2_encoder = Encoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True, num_cab=2
        )
        self.stage2_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, num_cab=2
        )

        self.stage3_orsnet = ORSNet(
            n_feat,
            scale_orsnetfeats,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            num_cab,
        )

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)

        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.concat23 = conv(n_feat * 2, n_feat + scale_orsnetfeats, kernel_size, bias=bias)
        self.tail = conv(n_feat + scale_orsnetfeats, out_c, kernel_size, bias=bias)

    def forward(self, x3_img: torch.Tensor) -> List[torch.Tensor]:
        # Original-resolution image
        H = x3_img.size(2)

        # ----- Stage 1: split image into top/bottom patches -----
        x1top_img = x3_img[:, :, 0 : int(H / 2), :]
        x1bot_img = x3_img[:, :, int(H / 2) : H, :]

        x1top = self.shallow_feat1(x1top_img)
        x1bot = self.shallow_feat1(x1bot_img)

        feat1_top = self.stage1_encoder(x1top)
        feat1_bot = self.stage1_encoder(x1bot)

        res1_top = self.stage1_decoder(feat1_top)
        res1_bot = self.stage1_decoder(feat1_bot)

        # Recombine top/bottom patch features into full-resolution maps (for CSFF
        # into Stage 2), exactly as the MPRNet source does.
        feat1 = [torch.cat((k, v), 2) for k, v in zip(feat1_top, feat1_bot)]
        res1 = [torch.cat((k, v), 2) for k, v in zip(res1_top, res1_bot)]

        # Apply SAM per patch, then recombine into a full-resolution feature map.
        x2top_samfeats, stage1_img_top = self.sam12(res1_top[0], x1top_img)
        x2bot_samfeats, stage1_img_bot = self.sam12(res1_bot[0], x1bot_img)

        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)

        # ----- Stage 2: operate on the recombined full-resolution image -----
        x2top_sam = torch.cat([x2top_samfeats, x2bot_samfeats], 2)  # full-res SAM features
        x2_cat = self.concat12(torch.cat([self.shallow_feat2(x3_img), x2top_sam], 1))

        feat2 = self.stage2_encoder(x2_cat, feat1, res1)
        res2 = self.stage2_decoder(feat2)

        x3_samfeats, stage2_img = self.sam23(res2[0], x3_img)

        # ----- Stage 3: original-resolution subnetwork -----
        x3 = self.shallow_feat3(x3_img)
        x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))
        x3_cat = self.stage3_orsnet(x3_cat, feat2, res2)
        stage3_img = self.tail(x3_cat)

        return [stage3_img + x3_img, stage2_img, stage1_img]


class _MPRNetWrapper(nn.Module):
    """Return only the full-resolution Stage-3 restored image for tracing."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[0]


def build(variant: str = "deblurring") -> nn.Module:
    """Build an MPRNet variant wrapped to return one full-resolution tensor.

    All three published task variants share the MPRNet topology and differ only
    by configuration; deblurring/deraining use the full config, denoising a
    slightly lighter ORB depth.  Widths are reduced here for a compact graph.
    """
    if variant == "denoising":
        model = MPRNet(n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=3)
    else:  # deblurring / deraining
        model = MPRNet(n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=4)
    return _MPRNetWrapper(model)


# ============================================================
# Menagerie wiring: zero-arg builders + example input + entries.
# ============================================================


def build_mprnet_deblurring() -> nn.Module:
    """Build MPRNet configured for image deblurring (full encoder-decoder + ORSNet)."""
    return build("deblurring")


def build_mprnet_denoising() -> nn.Module:
    """Build MPRNet configured for image denoising (lighter ORB depth)."""
    return build("denoising")


def build_mprnet_deraining() -> nn.Module:
    """Build MPRNet configured for image deraining (full encoder-decoder + ORSNet)."""
    return build("deraining")


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 128, 128)`` for MPRNet."""
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ENTRIES = [
    (
        "MPRNet (3-stage progressive image deblurring)",
        "build_mprnet_deblurring",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "MPRNet (3-stage progressive image denoising)",
        "build_mprnet_denoising",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "MPRNet (3-stage progressive image deraining)",
        "build_mprnet_deraining",
        "example_input",
        "2021",
        "DC",
    ),
]

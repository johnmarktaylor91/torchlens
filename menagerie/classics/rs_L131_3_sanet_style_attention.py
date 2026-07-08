# FAITHFUL REIMPLEMENTATION from Park & Lee, "Arbitrary Style Transfer with
# Style-Attentional Networks", CVPR 2019 (arXiv:1812.02342) -- no public code.
#
# The paper's own repo (github.com/dypark86/SANET, linked from the queue
# candidate) is a GitHub Pages *project site* only (README.md, _config.yml,
# _layouts/, images/, index.md) -- it contains zero model source code, so
# Rungs 2/3 (vendor / port real repo code) are unavailable. This module
# transcribes the architecture exactly as specified in the paper:
#
#  - Encoder: fixed, pretrained-style VGG-19 (relu1_1..relu5_1 taps); here we
#    build the real torchvision.models.vgg19 features stack (random-init,
#    frozen, eval-mode) and slice it at the standard AdaIN/SANet layer
#    boundaries, Sec 3.1/3.3.
#  - SANet module (Fig. 3, Eq. 5): mean-variance channel-wise normalize F_c
#    and F_s, project both through learned 1x1 convs f(.)=W_f, g(.)=W_g
#    (query/key) and h(F_s)=W_h*F_s (value, on the *unnormalized* style
#    feature per the paper), compute softmax attention over all style
#    positions for each content position, and produce F_cs as the attention-
#    weighted sum of style values.
#  - Eq. 2: F_csc = F_c + conv1x1(F_cs)  (residual add of a learned 1x1 conv
#    over the raw SANet output).
#  - Eq. 3: two SANets run on relu4_1 and relu5_1 features respectively; the
#    relu5_1 output is upsampled (nearest, factor 2) to relu4_1 resolution,
#    summed with the relu4_1 output, and passed through a 3x3 conv to produce
#    the merged feature map F_csc^m fed to the decoder.
#  - Decoder (Sec 3.1): "Our decoder follows the settings of [AdaIN]" -- a
#    symmetric mirror of the VGG-19 encoder up to relu4_1, using reflection
#    padding, 3x3 convs, ReLU, and nearest-neighbor 2x upsampling in place of
#    the encoder's max-pools (the standard AdaIN decoder architecture the
#    paper explicitly cites and reuses verbatim).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

MENAGERIE_ZOO = "reimpl-pytorch"

_EPS = 1e-5


def _mean_std(feat: torch.Tensor):
    # channel-wise mean/std over the spatial dims, per-sample.
    n, c = feat.shape[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + _EPS
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def _mean_variance_norm(feat: torch.Tensor) -> torch.Tensor:
    mean, std = _mean_std(feat)
    return (feat - mean) / std


class SANet(nn.Module):
    """Style-Attentional Network module (Fig. 3 / Eq. 5 of the paper)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.f = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.h = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        b, c, h_c, w_c = content_feat.shape
        _, _, h_s, w_s = style_feat.shape

        f_c = self.f(_mean_variance_norm(content_feat)).view(b, c, h_c * w_c).permute(0, 2, 1)
        g_s = self.g(_mean_variance_norm(style_feat)).view(b, c, h_s * w_s)
        attn = self.softmax(torch.bmm(f_c, g_s))  # (b, h_c*w_c, h_s*w_s)

        h_s_val = self.h(style_feat).view(b, c, h_s * w_s).permute(0, 2, 1)
        out = torch.bmm(attn, h_s_val)  # (b, h_c*w_c, c)
        out = out.permute(0, 2, 1).view(b, c, h_c, w_c)
        return out


class VGGEncoder(nn.Module):
    """Slices of a (random-init) VGG-19 feature stack at the relu1_1,
    relu2_1, relu3_1, relu4_1, and relu5_1 boundaries used by the SANet
    style/content losses and the two SANet modules."""

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=None).features
        # torchvision vgg19 layer indices (0-based, matching the standard
        # AdaIN/SANet slicing convention):
        #   relu1_1 -> layer 1, relu2_1 -> layer 6,
        #   relu3_1 -> layer 11, relu4_1 -> layer 20, relu5_1 -> layer 29
        self.slice1 = nn.Sequential(*list(vgg.children())[0:2])  # -> relu1_1
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])  # -> relu2_1
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12])  # -> relu3_1
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21])  # -> relu4_1
        self.slice5 = nn.Sequential(*list(vgg.children())[21:30])  # -> relu5_1
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return h1, h2, h3, h4, h5


class Decoder(nn.Module):
    """Symmetric VGG-19-mirroring decoder from relu4_1 back to RGB,
    "following the settings of AdaIN" per Sec 3.1 -- reflection-padded 3x3
    convs, ReLU, and nearest-neighbor 2x upsampling replacing the encoder's
    max-pool downsamples."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SANetStyleTransfer(nn.Module):
    """Full SANet style-transfer network (Fig. 2a): fixed VGG-19 encoder,
    two SANet modules (on relu4_1 and relu5_1 features), multi-level SANet
    output fusion (Eq. 3), and a symmetric decoder producing the stylized
    image I_cs."""

    def __init__(self):
        super().__init__()
        self.encoder = VGGEncoder()
        self.sanet4_1 = SANet(512)
        self.sanet5_1 = SANet(512)
        self.merge_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.decoder = Decoder()

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        _, _, _, c4, c5 = self.encoder(content)
        _, _, _, s4, s5 = self.encoder(style)

        # Eq. 1-2: SANet output + residual 1x1-conv add, per level.
        f_cs_4 = self.sanet4_1(c4, s4)
        f_csc_4 = c4 + f_cs_4

        f_cs_5 = self.sanet5_1(c5, s5)
        f_csc_5 = c5 + f_cs_5

        # Eq. 3: multi-level fusion -- upsample relu5_1 branch to relu4_1
        # resolution, add, and merge with a 3x3 conv.
        f_csc_5_up = F.interpolate(f_csc_5, size=f_csc_4.shape[-2:], mode="nearest")
        f_csc_m = self.merge_conv(f_csc_4 + f_csc_5_up)

        return self.decoder(f_csc_m)


# --- menagerie glue (not part of the reimplemented architecture) -----------


def build_sanet_style_attention():
    model = SANetStyleTransfer()
    model.eval()
    return model


def example_input_sanet_style_attention():
    # relu4_1 sits at 1/8 spatial resolution (post 3 max-pools); 64x64 input
    # keeps relu4_1/relu5_1 feature maps at a tiny but valid (8x8 / 4x4) size.
    content = torch.randn(1, 3, 64, 64)
    style = torch.randn(1, 3, 64, 64)
    return (content, style)


MENAGERIE_ENTRIES = [
    (
        "SANet (style-attentional network)",
        "build_sanet_style_attention",
        "example_input_sanet_style_attention",
        2019,
        MENAGERIE_ZOO,
    ),
]

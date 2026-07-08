# SOURCE: vendored from diyiiyiii/MCCNet @ (main branch HEAD as fetched 2026-07-01)
# https://raw.githubusercontent.com/diyiiyiii/MCCNet/main/net.py
# https://raw.githubusercontent.com/diyiiyiii/MCCNet/main/function.py
#
# Deng, Tang, Dong et al. 2021 (AAAI) "Arbitrary Video Style Transfer via Multi-Channel
# Correlation" (MCCNet) -- a VGG19-style fixed encoder feeding an "MCC_Module" that
# computes a per-channel content<->style correlation matrix (`MCCNet.forward`: builds a
# style channel-correlation matrix `FC_S` via `torch.bmm` + a learned `nn.Linear` "fc",
# then modulates the normalized content feature by it) and decodes the fused feature
# back to an image with a mirrored decoder. This channel-correlation mechanism (vs.
# MAST's spatial cross-attention `CA`) is the architectural novelty distinguishing MCCNet.
#
# `decoder`, `vgg` (the encoder backbone), `MCCNet`, `MCC_Module` are copied verbatim
# from the source `net.py`. `normal` and `calc_mean_std` are copied verbatim from
# `function.py` (the only two symbols `net.py` imports from it).
#
# No architectural changes were made; only mechanical fixes for import isolation and
# inference wiring:
#   - The upstream `net.py` also defines a `Net` class whose `forward` is a *training*
#     forward: it calls `.cuda()` unconditionally on a noise tensor
#     (`noise = torch.nn.init.normal(t, ...).cuda()`) and computes 5 losses. That is not
#     the architecture under study for capture purposes and is device-nonportable by
#     construction. The paper's actual style-transfer *inference* path -- what
#     `test_video.py::style_transfer()` calls at eval time -- is a thinner, CUDA-free
#     pipeline: encode content+style through the frozen VGG stages, run
#     `mcc_module(content_feats, style_feats)`, and decode.
#     `MCCNetStyleTransfer` below reproduces exactly that inference wiring (`enc_1..5`
#     slices identical to `Net`, `mcc_module` identical to `Net.mcc_module`, `decoder`
#     identical to `Net.decoder`) instead of instantiating the training-only, CUDA-hard-
#     coded `Net.forward`, since that is the real code path this repo ships for using the
#     network, not a re-derivation.
#   - `import scipy.stats as stats`, `from torchvision.utils import save_image` were
#     unused by any function kept here (they supported `Net`'s training losses / other
#     file-scope code in `net.py` that this vendor drops) and are omitted.

import torch
import torch.nn as nn

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def normal(feat, eps=1e-5):
    feat_mean, feat_std = calc_mean_std(feat, eps)
    normalized = (feat - feat_mean) / feat_std
    return normalized


class MCCNet(nn.Module):
    def __init__(self, in_dim):
        super(MCCNet, self).__init__()
        self.f = nn.Conv2d(in_dim, int(in_dim), (1, 1))
        self.g = nn.Conv2d(in_dim, int(in_dim), (1, 1))
        self.h = nn.Conv2d(in_dim, int(in_dim), (1, 1))
        self.softmax = nn.Softmax(dim=-2)  # 17
        self.out_conv = nn.Conv2d(int(in_dim), in_dim, (1, 1))
        self.fc = nn.Linear(in_dim, in_dim)

    def forward(self, content_feat, style_feat):
        B, C, H, W = content_feat.size()

        F_Fc_norm = self.f(normal(content_feat))

        B, C, H, W = style_feat.size()
        G_Fs_norm = self.g(normal(style_feat)).view(-1, 1, H * W)
        G_Fs_sum = G_Fs_norm.view(B, C, H * W).sum(-1)
        FC_S = torch.bmm(G_Fs_norm, G_Fs_norm.permute(0, 2, 1)).view(B, C) / G_Fs_sum  # 14
        FC_S = self.fc(FC_S).view(B, C, 1, 1)

        out = F_Fc_norm * FC_S
        B, C, H, W = content_feat.size()
        out = out.contiguous().view(B, -1, H, W)
        out = self.out_conv(out)
        out = content_feat + out

        return out


class MCC_Module(nn.Module):
    def __init__(self, in_dim):
        super(MCC_Module, self).__init__()
        self.MCCN = MCCNet(in_dim)

    def forward(self, content_feats, style_feats):
        content_feat_4 = content_feats[-2]
        style_feat_4 = style_feats[-2]
        Fcsc = self.MCCN(content_feat_4, style_feat_4)

        return Fcsc


class MCCNetStyleTransfer(nn.Module):
    """Real MCCNet inference wiring (mirrors test_video.py::style_transfer + feat_extractor)."""

    def __init__(self, encoder, decoder):
        super(MCCNetStyleTransfer, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        self.mcc_module = MCC_Module(512)
        self.decoder = decoder
        for name in ["enc_1", "enc_2", "enc_3", "enc_4", "enc_5"]:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def forward(self, content, style, alpha=1.0):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        Fccc = self.mcc_module(content_feats, content_feats)
        feat = self.mcc_module(content_feats, style_feats)
        feat = feat * alpha + Fccc * (1 - alpha)
        return self.decoder(feat)


def build_mccnet():
    return MCCNetStyleTransfer(vgg, decoder)


def example_input_mccnet():
    content = torch.randn(1, 3, 64, 64)
    style = torch.randn(1, 3, 64, 64)
    return (content, style)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MCCNet", "build_mccnet", "example_input_mccnet", 2021, "vendored"),
]

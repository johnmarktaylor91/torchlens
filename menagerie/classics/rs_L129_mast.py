# SOURCE: vendored from diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network @ e98469dd7cbd618f3f978fe6d4df92d9b7b406b6
# https://raw.githubusercontent.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/e98469dd7cbd618f3f978fe6d4df92d9b7b406b6/net.py
# https://raw.githubusercontent.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network/e98469dd7cbd618f3f978fe6d4df92d9b7b406b6/function.py
#
# Deng, Tang, Dong et al. 2020 (ACM MM) "Arbitrary Style Transfer via Multi-Adaptation
# Network" (MAST) -- a VGG19-style fixed encoder feeding a "Multi-Adaptation Module"
# with three attention-based sub-modules: content self-attention (Content_SA), style
# self-attention (Style_SA), and cross content<->style attention (CA), whose fused
# feature is decoded back to an image by a mirrored decoder.
#
# `decoder`, `vgg` (the encoder backbone), `CA`, `Style_SA`, `Content_SA`,
# `Multi_Adaptation_Module` are copied verbatim from the source `net.py`. `normal` and
# `calc_mean_std` are copied verbatim from `function.py` (the only two symbols `net.py`
# imports from it).
#
# No architectural changes were made; only mechanical fixes for import isolation and
# inference wiring:
#   - The upstream `net.py` also defines a `Net` class whose `forward` is a *training*
#     forward (computes 5 decoded reconstructions + identity losses via the `ma_module`
#     across independently-sampled content/style pairs). The paper's actual style-transfer
#     *inference* path -- what `test.py::style_transfer()` calls at eval time -- is a
#     thinner pipeline: encode content+style through the frozen VGG stages, run
#     `ma_module(content_feats, style_feats)`, and decode. `MASTStyleTransfer` below
#     reproduces exactly that inference wiring (`enc_1..5` slices identical to `Net`,
#     `ma_module` identical to `Net.ma_module`, `decoder` identical to `Net.decoder`)
#     instead of instantiating the training-only `Net.forward`, since that is the real
#     code path this repo ships for using the network, not a re-derivation.
#   - `import scipy.stats as stats`, `from torchvision.utils import save_image`, and
#     `random` were unused by any function kept here (they supported `Net`'s training
#     losses / other file-scope code in `net.py` that this vendor drops) and are omitted.

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


class CA(nn.Module):
    def __init__(self, in_dim):
        super(CA, self).__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content_feat, style_feat):
        B, C, H, W = content_feat.size()
        F_Fc_norm = self.f(normal(content_feat)).view(B, -1, H * W).permute(0, 2, 1)

        B, C, H, W = style_feat.size()
        G_Fs_norm = self.g(normal(style_feat)).view(B, -1, H * W)

        energy = torch.bmm(F_Fc_norm, G_Fs_norm)
        attention = self.softmax(energy)

        H_Fs = self.h(style_feat).view(B, -1, H * W)
        out = torch.bmm(H_Fs, attention.permute(0, 2, 1))
        B, C, H, W = content_feat.size()
        out = out.view(B, C, H, W)
        out = self.out_conv(out)

        out += content_feat

        return out


class Style_SA(nn.Module):
    def __init__(self, in_dim):
        super(Style_SA, self).__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, style_feat):
        B, C, H, W = style_feat.size()
        F_Fc_norm = self.f(style_feat).view(B, -1, H * W)

        B, C, H, W = style_feat.size()
        G_Fs_norm = self.g(style_feat).view(B, -1, H * W).permute(0, 2, 1)

        energy = torch.bmm(F_Fc_norm, G_Fs_norm)
        attention = self.softmax(energy)

        H_Fs = self.h(normal(style_feat)).view(B, -1, H * W)
        out = torch.bmm(attention.permute(0, 2, 1), H_Fs)

        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        out += style_feat
        return out


class Content_SA(nn.Module):
    def __init__(self, in_dim):
        super(Content_SA, self).__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content_feat):
        B, C, H, W = content_feat.size()
        F_Fc_norm = self.f(normal(content_feat)).view(B, -1, H * W).permute(0, 2, 1)

        B, C, H, W = content_feat.size()
        G_Fs_norm = self.g(normal(content_feat)).view(B, -1, H * W)

        energy = torch.bmm(F_Fc_norm, G_Fs_norm)
        attention = self.softmax(energy)

        H_Fs = self.h(content_feat).view(B, -1, H * W)
        out = torch.bmm(H_Fs, attention.permute(0, 2, 1))
        B, C, H, W = content_feat.size()
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        out += content_feat

        return out


class Multi_Adaptation_Module(nn.Module):
    def __init__(self, in_dim):
        super(Multi_Adaptation_Module, self).__init__()

        self.CA = CA(in_dim)
        self.CSA = Content_SA(in_dim)
        self.SSA = Style_SA(in_dim)

    def forward(self, content_feats, style_feats):
        content_feat = self.CSA(content_feats[-2])
        style_feat = self.SSA(style_feats[-2])
        Fcsc = self.CA(content_feat, style_feat)

        return Fcsc


class MASTStyleTransfer(nn.Module):
    """Real MAST inference wiring (mirrors test.py::style_transfer + feat_extractor)."""

    def __init__(self, encoder, decoder):
        super(MASTStyleTransfer, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        self.ma_module = Multi_Adaptation_Module(512)
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
        feat = self.ma_module(content_feats, style_feats)
        Fccc = self.ma_module(content_feats, content_feats)
        feat = feat * alpha + Fccc * (1 - alpha)
        return self.decoder(feat)


def build_mast():
    return MASTStyleTransfer(vgg, decoder)


def example_input_mast():
    content = torch.randn(1, 3, 64, 64)
    style = torch.randn(1, 3, 64, 64)
    return (content, style)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MAST", "build_mast", "example_input_mast", 2020, "vendored"),
]

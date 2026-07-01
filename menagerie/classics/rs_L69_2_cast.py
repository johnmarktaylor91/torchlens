# SOURCE: vendored from zyxElsa/CAST_pytorch @ main
# https://github.com/zyxElsa/CAST_pytorch
# https://raw.githubusercontent.com/zyxElsa/CAST_pytorch/main/models/net.py
#
# Zhang et al. 2022 (SIGGRAPH) "Domain Enhanced Arbitrary Image Style
# Transfer via Contrastive Learning" (CAST) -- the official repo's core
# image-to-image generator (used at inference/test time, see `test.py` ->
# `CASTModel.forward()` in `models/cast_model.py`) is the pair
# `ADAIN_Encoder` + `Decoder` from `models/net.py`: `ADAIN_Encoder` wraps a
# fixed (frozen at train time via `requires_grad = False`) VGG19-style
# `vgg` feature stack (also defined in this file) sliced into 4 stages
# (`enc_1..enc_4`, splitting at the standard relu1_1/relu2_1/relu3_1/relu4_1
# breakpoints), extracts intermediate content/style features via
# `encode_with_intermediate`, and performs Adaptive Instance Normalization
# (`adain`: match content-feature per-channel mean/std to the style
# feature's) on the deepest (relu4_1) content/style feature pair. `Decoder`
# is a mirrored ReflectionPad/Conv/ReLU/Upsample stack that reconstructs an
# RGB image from the AdaIN-normalized feature map. `CASTModel.forward()`
# (test-time path) is exactly `fake_B = Dec_B(AE(real_A, real_B))` -- the
# stylization pass exercised by `build_cast()`/`example_input_cast()` below
# via a thin `CASTGenerator` wrapper.
#
# `vgg`, `ADAIN_Encoder`, `Decoder` are copied VERBATIM from `models/net.py`
# above (architecture completely unchanged; the module docstring/comments
# are the only textual difference). The real `CASTModel.__init__` loads a
# pretrained-ImageNet `vgg_normalised.pth` checkpoint into `vgg` before
# wrapping it in `ADAIN_Encoder`
# (`vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))`,
# `models/cast_model.py:84`); this harness constructs the SAME random-init
# `vgg` Sequential (no checkpoint download) and wraps it exactly as
# `CASTModel.__init__` does (`net.ADAIN_Encoder(vgg, gpu_ids)`,
# `net.Decoder(gpu_ids)`) -- tiny-random-init construction only, no
# architecture change.
#
# `CASTGenerator` below is a thin harness `nn.Module`
# (`forward(content, style) = self.decoder(self.encoder(content, style))`)
# that reproduces `CASTModel.forward()`'s test-time computation graph
# (`models/cast_model.py:175-179`, `isTrain=False` branch:
# `self.real_A_feat = self.netAE(self.real_A, self.real_B); self.fake_B =
# self.netDec_B(self.real_A_feat)`) as a single top-level module TorchLens
# can trace end to end; it adds no new architecture, only plumbing the two
# real vendored classes together the same way the real model script does.

import torch.nn as nn
import torch

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


class ADAIN_Encoder(nn.Module):
    def __init__(self, encoder, gpu_ids=[]):
        super(ADAIN_Encoder, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1 64
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1 128
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1 256
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1 512

        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ["enc_1", "enc_2", "enc_3", "enc_4"]:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert len(size) == 4
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adain(self, content_feat, style_feat):
        assert content_feat.size()[:2] == style_feat.size()[:2]
        size = content_feat.size()
        style_mean, style_std = self.calc_mean_std(style_feat)
        content_mean, content_std = self.calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, content, style, encoded_only=False):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        if encoded_only:
            return content_feats[-1], style_feats[-1]
        else:
            adain_feat = self.adain(content_feats[-1], style_feats[-1])
            return adain_feat


class Decoder(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(Decoder, self).__init__()
        decoder = [
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),  # 256
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
            nn.ReLU(),  # 128
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),  # 64
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        ]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, adain_feat):
        fake_image = self.decoder(adain_feat)

        return fake_image


class CASTGenerator(nn.Module):
    """Harness wrapper reproducing CASTModel.forward()'s test-time graph:
    fake_B = Dec_B(AE(real_A, real_B)) -- see models/cast_model.py:175-179.
    """

    def __init__(self):
        super().__init__()
        self.encoder = ADAIN_Encoder(vgg)
        self.decoder = Decoder()

    def forward(self, content, style):
        adain_feat = self.encoder(content, style)
        return self.decoder(adain_feat)


def build_cast():
    return CASTGenerator().eval()


def example_input_cast():
    content = torch.randn(1, 3, 64, 64)
    style = torch.randn(1, 3, 64, 64)
    return (content, style)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("CAST", "build_cast", "example_input_cast", 2022, "vendored"),
]

# FAITHFUL PORT of mattpoggi/pydnet @ master (original framework: TensorFlow 1.x)
# https://raw.githubusercontent.com/mattpoggi/pydnet/master/pydnet.py
# https://raw.githubusercontent.com/mattpoggi/pydnet/master/layers.py
#
# Poggi, Aleotti, Tosi, Mattoccia 2018 (IROS) "Towards real-time unsupervised monocular
# depth estimation on CPU". The original repo defines `pydnet` (6-level pyramid, IROS18
# checkpoint) and `pydnet2` (4-level pyramid, ITS checkpoint) as TensorFlow-1.x graph
# builders (tf.variable_scope + manual conv2d/deconv2d wrappers, no torch equivalent
# runnable in this env). Transcribed here layer-for-layer into torch: `build_pyramid`
# (conv1a/conv1b .. conv6a/conv6b strided-then-plain leaky-conv pairs), `build_estimator`
# (disp-3..disp-6 4-layer leaky-conv stack, no activation on the last), and
# `bilinear_upsampling_by_deconvolution` (stride-2 ConvTranspose2d + leaky relu, matching
# the original's [2,2,f,f] deconv kernel and 'SAME' padding via output_padding=0). Both
# `PyDNet` (IROS18, levels 1-6, sigmoid*0.3 disparity head) and `PyDNet2` (ITS, levels 1-4,
# relu disparity head) are ported; only the multi-scale bilinear-resize-to-input-size final
# step is dropped since TorchLens traces a single deterministic forward (the 3 returned
# disparity maps stay at their native pyramid resolution instead of being upsampled to the
# input resolution -- purely a post-processing step, not part of the learned architecture).
"""PyD-Net: real-time pyramidal CNN for monocular depth estimation (CPU-friendly)."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class ConvLeaky(nn.Module):
    """Port of conv2d_leaky (layers.py): conv + optional leaky-relu(0.2), 'SAME' padding."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, relu=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(0.2) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DeconvLeaky(nn.Module):
    """Port of deconv2d_leaky (layers.py): stride-2 transposed conv + leaky-relu(0.2)."""

    def __init__(self, channels, kernel_size=2, stride=2, relu=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels, channels, kernel_size, stride=stride)
        self.relu = nn.LeakyReLU(0.2) if relu else None

    def forward(self, x):
        x = self.deconv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class PyramidEncoder6(nn.Module):
    """Port of build_pyramid (pydnet.py, pydnet class): conv1a/b .. conv6a/b."""

    def __init__(self):
        super().__init__()
        chans = [(3, 16), (16, 32), (32, 64), (64, 96), (96, 128), (128, 192)]
        self.stages = nn.ModuleList()
        for in_ch, out_ch in chans:
            self.stages.append(
                nn.ModuleDict(
                    {
                        "a": ConvLeaky(in_ch, out_ch, 3, stride=2, relu=True),
                        "b": ConvLeaky(out_ch, out_ch, 3, stride=1, relu=True),
                    }
                )
            )

    def forward(self, x):
        features = [x]
        for stage in self.stages:
            x = stage["a"](x)
            x = stage["b"](x)
            features.append(x)
        return features


class PyramidEncoder4(nn.Module):
    """Port of build_pyramid (pydnet.py, pydnet2 class): conv1a/b .. conv4a/b."""

    def __init__(self):
        super().__init__()
        chans = [(3, 16), (16, 32), (32, 64), (64, 96)]
        self.stages = nn.ModuleList()
        for in_ch, out_ch in chans:
            self.stages.append(
                nn.ModuleDict(
                    {
                        "a": ConvLeaky(in_ch, out_ch, 3, stride=2, relu=True),
                        "b": ConvLeaky(out_ch, out_ch, 3, stride=1, relu=True),
                    }
                )
            )

    def forward(self, x):
        features = [x]
        for stage in self.stages:
            x = stage["a"](x)
            x = stage["b"](x)
            features.append(x)
        return features


class Estimator(nn.Module):
    """Port of build_estimator (pydnet.py): disp-3 .. disp-6, 4 leaky convs, last has no relu."""

    def __init__(self, in_ch):
        super().__init__()
        self.disp3 = ConvLeaky(in_ch, 96, 3, stride=1, relu=True)
        self.disp4 = ConvLeaky(96, 64, 3, stride=1, relu=True)
        self.disp5 = ConvLeaky(64, 32, 3, stride=1, relu=True)
        self.disp6 = ConvLeaky(32, 8, 3, stride=1, relu=False)

    def forward(self, features, upsampled_disp=None):
        x = torch.cat([features, upsampled_disp], dim=1) if upsampled_disp is not None else features
        x = self.disp3(x)
        x = self.disp4(x)
        x = self.disp5(x)
        x = self.disp6(x)
        return x


class PyDNet(nn.Module):
    """Port of the `pydnet` class (pydnet.py): 6-level pyramid, IROS18 checkpoint config."""

    def __init__(self, level=1):
        super().__init__()
        self.level = level
        self.pyramid = PyramidEncoder6()

        # estimator input channels: feature channels (+8 upsampled-disp channels if present)
        self.est6 = Estimator(192)
        self.up6 = DeconvLeaky(8)

        self.est5 = Estimator(128 + 8)
        self.up5 = DeconvLeaky(8)

        self.est4 = Estimator(96 + 8)
        self.up4 = DeconvLeaky(8)

        self.est3 = Estimator(64 + 8)
        self.up3 = DeconvLeaky(8)

        self.est2 = Estimator(32 + 8)

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _get_disp(x):
        # Port of get_disp: 0.3 * sigmoid(x[:, :2])
        return 0.3 * torch.sigmoid(x[:, :2])

    def forward(self, x):
        pyramid = self.pyramid(x)  # [im0, f1..f6]

        conv6 = self.est6(pyramid[6])
        disp7 = self._get_disp(conv6)
        upconv6 = self.up6(conv6)

        conv5 = self.est5(pyramid[5], upconv6)
        disp6 = self._get_disp(conv5)
        upconv5 = self.up5(conv5)

        conv4 = self.est4(pyramid[4], upconv5)
        disp5 = self._get_disp(conv4)
        upconv4 = self.up4(conv4)

        conv3 = self.est3(pyramid[3], upconv4)
        disp4 = self._get_disp(conv3)
        upconv3 = self.up3(conv3)

        conv2 = self.est2(pyramid[2], upconv3)
        disp3 = self._get_disp(conv2)

        return disp7, disp6, disp5, disp4, disp3


class PyDNet2(nn.Module):
    """Port of the `pydnet2` class (pydnet.py): 4-level pyramid, ITS checkpoint config."""

    def __init__(self, level=1):
        super().__init__()
        self.level = level
        self.pyramid = PyramidEncoder4()

        self.est4 = Estimator(96)
        self.up4 = DeconvLeaky(8)

        self.est3 = Estimator(64 + 8)
        self.up3 = DeconvLeaky(8)

        self.est2 = Estimator(32 + 8)
        self.up2 = DeconvLeaky(8)

        self.est1 = Estimator(16 + 8)

    @staticmethod
    def _get_disp(x):
        # Port of get_disp (pydnet2): relu(x[:, :2])
        return torch.relu(x[:, :2])

    def forward(self, x):
        pyramid = self.pyramid(x)  # [im0, f1..f4]

        conv4 = self.est4(pyramid[4])
        disp5 = self._get_disp(conv4)
        upconv4 = self.up4(conv4)

        conv3 = self.est3(pyramid[3], upconv4)
        disp4 = self._get_disp(conv3)
        upconv3 = self.up3(conv3)

        conv2 = self.est2(pyramid[2], upconv3)
        disp3 = self._get_disp(conv2)
        upconv2 = self.up2(conv2)

        conv1 = self.est1(pyramid[1], upconv2)
        disp2 = self._get_disp(conv1)

        return disp5, disp4, disp3, disp2


def build_pydnet():
    torch.manual_seed(0)
    return PyDNet(level=1)


def example_input_pydnet():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 256, 256),)


def build_pydnet2():
    torch.manual_seed(0)
    return PyDNet2(level=1)


def example_input_pydnet2():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 256, 256),)


MENAGERIE_ENTRIES = [
    ("PyD-Net", "build_pydnet", "example_input_pydnet", 2018, "ported"),
    ("PyD-Net2", "build_pydnet2", "example_input_pydnet2", 2019, "ported"),
]

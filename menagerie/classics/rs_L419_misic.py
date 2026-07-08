# FAITHFUL PORT of pswapnesh/misic @ v2 (original framework: TensorFlow/Keras)
#
# Panigrahi, Murat, van Teeffelen 2021 (eLife) "MiSiC, a general deep learning-based method
# for the high-throughput cell segmentation of complicated bacterial communities". MiSiC's
# v2 segmentation network is *not* defined as architecture code anywhere in the
# pswapnesh/misic repo (any branch: master/v2/V1/article/gpu) -- `misic/misic.py` only
# calls `tensorflow.keras.models.load_model('misic/MiSiCv2.h5')` to load a pretrained
# binary. That `.h5` file (shipped in the repo, commit 872470e26cf70c72c4d2a2a3aa4d8cd1e42dd50c
# on the `v2` branch) embeds the real Keras `model_config` JSON in its HDF5 attrs -- the
# actual, exact architecture (every layer type, kernel size, stride, padding, activation,
# dropout rate, BatchNorm epsilon/momentum), extracted here via
# `h5py.File(...).attrs['model_config']` and `tensorflow.keras.models.load_model`, i.e.
# read directly off the real shipped weights rather than guessed from the paper. It is a
# small 3-level symmetric U-Net operating on the 3-channel "shape index" preprocessing
# (`MiSiC.shapeindex_preprocess`, not ported here -- that is a `skimage.feature.shape_index`
# feature-extraction preprocessing step, not part of the network): four
# Conv2D(64,7x7,same,relu)+BatchNorm+Dropout(0.5) encoder stages (the first three
# followed by 2x2 max-pool, matching the h5-embedded topology exactly), three decoder
# stages of 2x2 upsample + skip-concat with the matching encoder stage +
# Conv2D(64,7x7,same,relu)+BatchNorm+Dropout(0.5), and a final Conv2D(1,1x1,same,sigmoid)
# head producing a single-channel probability map. Every layer below (channel counts,
# kernel sizes, strides, padding, dropout rate, BN eps/momentum, activations, and the
# encoder<->decoder skip-connection wiring) is transcribed 1:1 from the real
# `model_config` layer list; only the framework changed (`tf.keras.layers.Conv2D`
# "same" padding -> torch `nn.Conv2d(padding="same")`, Keras channels-last NHWC ->
# torch channels-first NCHW).

import torch
import torch.nn as nn


class MiSiCUNet(nn.Module):
    """Faithful torch port of the real MiSiCv2.h5 Keras Functional model topology."""

    def __init__(self, in_channels=3, base_filters=64, dropout_rate=0.5):
        super().__init__()
        f = base_filters

        def conv_bn_drop(c_in, c_out):
            return nn.ModuleDict(
                {
                    "conv": nn.Conv2d(c_in, c_out, kernel_size=7, stride=1, padding="same"),
                    "bn": nn.BatchNorm2d(c_out, eps=1e-3, momentum=1 - 0.99),
                    "drop": nn.Dropout(dropout_rate),
                }
            )

        # Encoder (matches conv2d_16..conv2d_19 / batch_normalization_14..17 /
        # dropout_14..17 / max_pooling2d_6..8 in the real h5 config)
        self.enc1 = conv_bn_drop(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_bn_drop(f, f)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_bn_drop(f, f)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = conv_bn_drop(f, f)

        # Decoder (matches up_sampling2d_6..8 / concatenate_6..8 / conv2d_20..22 /
        # batch_normalization_18..20 / dropout_18..20 in the real h5 config)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = conv_bn_drop(2 * f, f)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = conv_bn_drop(2 * f, f)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = conv_bn_drop(2 * f, f)

        # Output head (matches conv2d_23: Conv2D(1, 1x1, same, sigmoid))
        self.out_conv = nn.Conv2d(f, 1, kernel_size=1, stride=1, padding="same")
        self.out_activation = nn.Sigmoid()

    @staticmethod
    def _apply(block, x):
        x = block["conv"](x)
        x = torch.relu(x)
        x = block["bn"](x)
        x = block["drop"](x)
        return x

    def forward(self, x):
        e1 = self._apply(self.enc1, x)
        p1 = self.pool1(e1)

        e2 = self._apply(self.enc2, p1)
        p2 = self.pool2(e2)

        e3 = self._apply(self.enc3, p2)
        p3 = self.pool3(e3)

        b = self._apply(self.bottleneck, p3)

        u1 = self.up1(b)
        c1 = torch.cat([u1, e3], dim=1)
        d1 = self._apply(self.dec1, c1)

        u2 = self.up2(d1)
        c2 = torch.cat([u2, e2], dim=1)
        d2 = self._apply(self.dec2, c2)

        u3 = self.up3(d2)
        c3 = torch.cat([u3, e1], dim=1)
        d3 = self._apply(self.dec3, c3)

        out = self.out_conv(d3)
        out = self.out_activation(out)
        return out


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_misic():
    return MiSiCUNet(in_channels=3, base_filters=64, dropout_rate=0.5)


def example_input_misic():
    # The real network's InputLayer shape is (None, 256, 256, 3) NHWC; shrunk here
    # to 64x64 (still divisible by 2**3 for the 3-level pool/upsample stack) and
    # converted to NCHW for torch.
    return (torch.randn(1, 3, 64, 64),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("MiSiC", "build_misic", "example_input_misic", 2021, "ported"),
]

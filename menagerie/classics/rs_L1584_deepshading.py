# FAITHFUL PORT of marcelsan/DeepShading @ master (original framework: TensorFlow/Keras)
# https://raw.githubusercontent.com/marcelsan/DeepShading/master/models/shading_net.py
#
# Deep Shading: Convolutional Neural Networks for Screen-Space Shading (Nalbach et
# al., EGSR 2017 / marcelsan's DeepShading reimplementation). The official repo's
# `ShadingNet` (models/shading_net.py) is a `tf.keras.Model` built with the
# functional API -- TF1-era `tf.contrib`/legacy `tf.keras` graph mode that will not
# run against our installed torch/timm/transformers stack (no TF/Keras dependency
# in this env, and the code itself is written against a pre-2.x Keras functional
# API). No PyTorch port exists in the repo or elsewhere; base env can't run TF1.x
# Keras. This is a faithful line-for-line port to torch: a 6-level encoder-decoder
# (U-Net) over a 6-channel input (RGB normal + RGB world-position G-buffer, per
# `input_shape=(None, None, 6)`), with the exact channel widths (8/16/32/64/128/256),
# LeakyReLU(0.01) activations, symmetric skip-connection concatenation at every
# level, Dropout(0.5) at levels 4 and 5, and the transposed-conv upsampling path --
# every layer of the original Keras graph is reproduced 1:1 (only the "same"-padding
# Conv2D/Conv2DTranspose stride bookkeeping is expressed via explicit torch padding
# arithmetic instead of Keras' implicit `padding='SAME'`).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class ShadingNet(nn.Module):
    """Faithful torch port of the Keras `ShadingNet` functional model.

    Original Keras graph (verbatim structure, `models/shading_net.py`):
      down0: Conv2D(8,3x3) -> LeakyReLU -> MaxPool(2x2)
      down1: Conv2D(16,3x3) -> LeakyReLU -> MaxPool(2x2)
      down2: Conv2D(32,3x3) -> LeakyReLU -> MaxPool(2x2)
      down3: Conv2D(64,3x3) -> LeakyReLU -> MaxPool(2x2)
      down4: Conv2D(128,3x3) -> LeakyReLU -> MaxPool(2x2) -> Dropout(0.5)
      down5: Conv2D(256,3x3) -> LeakyReLU -> Dropout(0.5)
      up4: ConvTranspose(256,4x4,s=2) -> concat(down4) -> Conv2D(128) -> LeakyReLU -> Dropout(0.5)
      up3: ConvTranspose(128,4x4,s=2) -> concat(down3) -> Conv2D(64)  -> LeakyReLU -> Dropout(0.5)
      up2: ConvTranspose(64,4x4,s=2)  -> concat(down2) -> Conv2D(32)  -> LeakyReLU
      up1: ConvTranspose(32,4x4,s=2)  -> concat(down1) -> Conv2D(16)  -> LeakyReLU
      up0: ConvTranspose(16,4x4,s=2)  -> concat(down0) -> Conv2D(1)   -> LeakyReLU
    """

    def __init__(self, in_channels=6):
        super().__init__()
        self.leaky = nn.LeakyReLU(0.01)

        # Down branch (Conv2D(k=3, stride=1, padding='SAME') -> same spatial size).
        self.down0_conv = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.down1_conv = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.down2_conv = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.down3_conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.down4_conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.down5_conv = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.5)

        # Up branch. Keras `Conv2DTranspose(filters, 4, strides=2, padding='SAME')`
        # on an even input size doubles spatial resolution exactly; the torch
        # equivalent is kernel_size=4, stride=2, padding=1.
        self.up5_to_4 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.up4_conv = nn.Conv2d(
            128 + 256, 128, kernel_size=3, stride=1, padding=1
        )  # relu, then LeakyReLU

        self.up4_to_3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.up3_conv = nn.Conv2d(64 + 128, 64, kernel_size=3, stride=1, padding=1)

        self.up3_to_2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.up2_conv = nn.Conv2d(32 + 64, 32, kernel_size=3, stride=1, padding=1)

        self.up2_to_1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.up1_conv = nn.Conv2d(16 + 32, 16, kernel_size=3, stride=1, padding=1)

        self.up1_to_0 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        self.up0_conv = nn.Conv2d(8 + 16, 1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (B, in_channels, H, W), H and W divisible by 32 (5 poolings).
        down0 = self.leaky(self.down0_conv(x))
        down0_to_1 = self.pool(down0)

        down1 = self.leaky(self.down1_conv(down0_to_1))
        down1_to_2 = self.pool(down1)

        down2 = self.leaky(self.down2_conv(down1_to_2))
        down2_to_3 = self.pool(down2)

        down3 = self.leaky(self.down3_conv(down2_to_3))
        down3_to_4 = self.pool(down3)

        down4 = self.leaky(self.down4_conv(down3_to_4))
        down4_to_5 = self.drop(self.pool(down4))

        down5 = self.leaky(self.down5_conv(down4_to_5))
        down5 = self.drop(down5)

        up5_to_4 = self.up5_to_4(down5)
        up4 = torch.cat([down4, up5_to_4], dim=1)
        # Original Keras layer sets activation='relu' on the Conv2D call itself,
        # then applies a second LeakyReLU(0.01) on top of the ReLU output.
        up4 = self.leaky(self.relu(self.up4_conv(up4)))
        up4 = self.drop(up4)

        up4_to_3 = self.up4_to_3(up4)
        up3 = torch.cat([down3, up4_to_3], dim=1)
        up3 = self.leaky(self.relu(self.up3_conv(up3)))
        up3 = self.drop(up3)

        up3_to_2 = self.up3_to_2(up3)
        up2 = torch.cat([down2, up3_to_2], dim=1)
        up2 = self.leaky(self.relu(self.up2_conv(up2)))

        up2_to_1 = self.up2_to_1(up2)
        up1 = torch.cat([down1, up2_to_1], dim=1)
        up1 = self.leaky(self.relu(self.up1_conv(up1)))

        up1_to_0 = self.up1_to_0(up1)
        up0 = torch.cat([down0, up1_to_0], dim=1)
        up0 = self.leaky(self.relu(self.up0_conv(up0)))

        return up0


def build_deepshading():
    model = ShadingNet(in_channels=6)
    model.eval()
    return model


def example_input_deepshading():
    torch.manual_seed(0)
    # Spatial size divisible by 32 (5 poolings in the down branch); kept tiny for
    # a fast trace. Real usage: 6-channel screen-space G-buffer (RGB normals +
    # RGB world position), per `input_shape=(None, None, 6)`.
    return torch.randn(1, 6, 32, 32)


MENAGERIE_ENTRIES = [
    ("DeepShading", "build_deepshading", "example_input_deepshading", 2017, "ported-pytorch"),
]

# SOURCE: vendored from joeylitalien/noise2noise-pytorch @ master
# https://raw.githubusercontent.com/joeylitalien/noise2noise-pytorch/master/src/unet.py
#
# Noise2Noise (Lehtinen et al., ICML 2018, "Noise2Noise: Learning Image Restoration
# without Clean Data") -- unofficial PyTorch reimplementation by Joey Litalien. `UNet`
# is the custom encoder/decoder architecture described in the paper's Appendix Table 2:
# five downsampling conv blocks (48-channel), a bottleneck upsample, then five
# concat-skip upsampling decoder blocks (96-channel, sharing block5 across the deepest
# three stages exactly as in the original code), finishing with a 3-conv RGB head.
#
# `UNet` is transcribed verbatim from `src/unet.py`. No architectural changes were
# made -- every Conv2d/ConvTranspose2d/ReLU/LeakyReLU layer, its arguments, and the
# skip-connection concat order in forward() are unchanged. Only the module docstring
# formatting is preserved as-is.

import torch
import torch.nn as nn


class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1),
        )
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1),
        )
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1),
        )
        # nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections."""

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)


def build_noise2noise():
    torch.manual_seed(0)
    model = UNet(in_channels=3, out_channels=3)
    model.eval()
    return model


def example_input_noise2noise():
    torch.manual_seed(0)
    # UNet uses 4 MaxPool2d(2) stages, so H/W must be divisible by 16.
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Noise2Noise", "build_noise2noise", "example_input_noise2noise", 2018, MENAGERIE_ZOO),
]

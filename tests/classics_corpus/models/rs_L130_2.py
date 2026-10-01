# SOURCE: vendored from cubeyoung/Noise2Score @ main (models/networks.py, class UNet)
"""Noise2Score (NeurIPS 2021) -- Tweedie-formula score-based denoiser.

Noise2Score estimates the score function (gradient of log-density) of noisy observations
via amortized residual learning (AR-DAE), then applies Tweedie's formula to recover a
denoised estimate without ever seeing clean images. The official PyTorch repo
(cubeyoung/Noise2Score) is a pix2pix/CycleGAN-style (Zhu et al.) codebase; models/networks.py
defines several candidate generators (ResNet, DnCNN, UNet) behind a define_G() factory. The
Gaussian/Poisson/Gamma model wrappers (models/Gaussian_model.py etc.) all instantiate
netG='unet' -- i.e. the `UNet` class in networks.py -- as the score network `f`. Its
forward() bakes in the AR-DAE mechanism directly: it injects Gaussian noise of scale `std`
into the input, runs the encoder/decoder, and returns both the estimated log-density
gradient and the amortized denoising-autoencoder loss term used during training. This file
vendors that `UNet` class verbatim (only the unused `loss` bookkeeping via
`torch.nn.functional.mse_loss` is inlined; no architecture change).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/networks.py :: UNet (verbatim; this is the network instantiated by
# define_G(netG='unet') and used by every Gaussian/Poisson/Gamma model variant)
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2)
        )

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1),
        )

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1),
        )

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def add_noise(self, input, std):
        mu = torch.randn_like(input)

        return input + std * mu, mu

    def forward(self, x, std):
        """Through encoder, then decoder by adding U-skip connections."""
        x_bar, mu = self.add_noise(x, std)
        # Encoder
        pool1 = self._block1(x_bar)
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
        concat1 = torch.cat((upsample1, x_bar), dim=1)
        # Final activation
        log_prob = self._block6(concat1)
        loss = F.mse_loss(std * log_prob, -mu)
        return log_prob, loss


# ---------------------------------------------------------------------------
# Menagerie build/example helpers
# ---------------------------------------------------------------------------
class Noise2ScoreWrapper(nn.Module):
    """Thin wrapper fixing the noise scale `std` so the traced model takes a single
    tensor input, matching how Gaussian_model.py calls `self.netf(self.lr, self.sigma)`."""

    def __init__(self, in_channels=3, out_channels=3, std=0.02):
        super().__init__()
        self.net = UNet(in_channels=in_channels, out_channels=out_channels)
        self.std = std

    def forward(self, x):
        log_prob, loss = self.net(x, self.std)
        return log_prob


def build_noise2score():
    return Noise2ScoreWrapper(in_channels=3, out_channels=3, std=0.02)


def example_input_noise2score():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("Noise2Score", build_noise2score, example_input_noise2score, 2021, MENAGERIE_ZOO),
]

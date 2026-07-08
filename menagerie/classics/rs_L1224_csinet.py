# SOURCE: vendored from mdelrosa/csinet-lstm @ master
# https://github.com/mdelrosa/csinet-lstm/blob/master/torch/csinet_torch.py
#
# CsiNet: "Deep Learning for Massive MIMO CSI Feedback" (Wen, Shih, Jin --
# IEEE Wireless Commun. Lett. 2018, arXiv:1712.08919). The original repo
# (sydney222/Python_CsiNet) is TensorFlow/Keras; this genuine PyTorch port of
# the CsiNet encoder-decoder architecture lives in the companion CsiNet-LSTM
# repo's `torch/csinet_torch.py` (by the same maintainer lineage, mdelrosa,
# who built the follow-on CsiNet-LSTM). The architecture: a convolutional
# `Encoder` (3 Conv2d+BatchNorm+LeakyReLU layers over a 2-channel real/
# imaginary CSI image, flattened to a dense bottleneck of `latent_dim`) and a
# `Decoder` (dense expansion from `latent_dim + aux_dim` back to the image
# grid, followed by 7 Conv2d+BatchNorm+LeakyReLU residual-style refinement
# layers with a final Tanh head) -- this is CsiNet-Pro's auxiliary-input
# variant (`aux` channel-state side input concatenated at the decoder's dense
# layer), the variant this torch port implements; setting `aux_dim=0` would
# recover a plain CsiNet, but the port always wires the aux tensor through
# `CsiNet.forward`, so `aux` is kept as a required second input here to match
# the actual traced code path.
#
# `Encoder`, `Decoder`, and the composing `CsiNet` module are transcribed
# verbatim, with ONLY the unused training-script imports/CLI (`torch.optim`,
# `Variable`, `torchvision`, `matplotlib`, `tqdm`, and the `if __name__ ==
# "__main__":` training/eval driver that depends on external `utils.*`
# modules not present in this repo) dropped -- no architectural line was
# altered.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(torch.nn.Module):
    """encoder for CsiNet"""

    def __init__(self, n_chan, H, W, latent_dim):
        super().__init__()
        self.img_total = H * W
        self.n_chan = n_chan
        self.latent_dim = latent_dim
        self.enc_conv1 = nn.Conv2d(2, 8, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(8)
        self.enc_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(16)
        self.enc_conv3 = nn.Conv2d(16, 2, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(2)
        self.enc_dense = nn.Linear(H * W * n_chan, latent_dim)

        self.activ = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.activ(self.bn_1(self.enc_conv1(x)))
        x = self.activ(self.bn_2(self.enc_conv2(x)))
        x = self.activ(self.bn_3(self.enc_conv3(x)))
        x = torch.reshape(x, (x.size(0), -1))
        x = self.enc_dense(x)
        return x


class Decoder(torch.nn.Module):
    """decoder for CsiNet"""

    def __init__(self, n_chan, H, W, latent_dim, aux_dim=512):
        super().__init__()
        self.H = H
        self.W = W
        self.img_total = H * W
        self.n_chan = n_chan
        self.dec_dense = nn.Linear(latent_dim + aux_dim, self.img_total * self.n_chan)
        self.dec_conv1 = nn.Conv2d(2, 128, 1)
        self.bn_1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.Conv2d(128, 64, 1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.dec_conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(32)
        self.dec_conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(32)
        self.dec_conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(16)
        self.dec_conv6 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn_6 = nn.BatchNorm2d(16)
        self.dec_conv7 = nn.Conv2d(16, 2, 3, padding=1)

        self.activ = nn.LeakyReLU(0.1)
        self.out_activ = nn.Tanh()

    def forward(self, x):
        """x = aux, input"""
        aux, H_in = x
        x = self.dec_dense(torch.cat((aux, H_in), 1))
        x = torch.reshape(x, (x.size(0), self.n_chan, self.H, self.W))
        x = self.activ(self.bn_1(self.dec_conv1(x)))
        x = self.activ(self.bn_2(self.dec_conv2(x)))
        x = self.activ(self.bn_3(self.dec_conv3(x)))
        x = self.activ(self.bn_4(self.dec_conv4(x)))
        x = self.activ(self.bn_5(self.dec_conv5(x)))
        x = self.activ(self.bn_6(self.dec_conv6(x)))
        x = self.out_activ(self.dec_conv7(x))
        return x


class CsiNet(nn.Module):
    """CsiNet for csi estimation"""

    def __init__(self, encoder, decoder, latent_dim, device=None):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.device = device
        self.training = True

    def forward(self, x):
        """forward call for CsiNet"""
        aux, H_in = x
        h_enc = self.encoder(H_in)
        return self.decoder((aux, h_enc))

    def latent_loss(self, z_mean, z_stddev):
        """if we want to do semi-supervised learning, then we could define the loss here"""
        pass


# ============================================================================
# staging build/example functions
# ============================================================================


def build_csinet():
    """Tiny-config CsiNet-Pro (n_chan=2, 32x32 CSI image, latent_dim=32, aux_dim=64)."""
    n_chan, H, W = 2, 32, 32
    latent_dim = 32
    aux_dim = 64
    encoder = Encoder(n_chan, H, W, latent_dim)
    decoder = Decoder(n_chan, H, W, latent_dim, aux_dim=aux_dim)
    return CsiNet(encoder, decoder, latent_dim)


def example_input_csinet():
    batch_size = 2
    aux = torch.randn(batch_size, 64)
    H_in = torch.randn(batch_size, 2, 32, 32)
    return (aux, H_in)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("CsiNet", "build_csinet", "example_input_csinet", 2018, MENAGERIE_ZOO),
]

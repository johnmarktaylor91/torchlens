# SOURCE: vendored from https://github.com/imbingox/BeatGAN @ master
# (experiments/ecg/network.py::Encoder/Decoder + experiments/ecg/model.py::Generator/Discriminator)
#
# BeatGAN (Zhou et al., IJCAI 2019, "BeatGAN: Anomalous Rhythm Detection using
# Adversarial Generated Time Series"): an adversarial autoencoder for ECG/motion
# beat anomaly detection. A 1D-convolutional Encoder-Decoder Generator
# reconstructs an input beat through a low-dimensional latent code; a
# convolutional Discriminator (an Encoder variant with a sigmoid classifier
# head, feature layer split off) is trained adversarially against it. Anomaly
# score = reconstruction error (+ optional latent/feature discrepancy).
#
# Vendored real repo code (Encoder, Decoder, Generator, Discriminator nn.Module
# classes). Only the AD_MODEL training/plotting base class, weights_init,
# print_network, and the `opt`-namespace argparse plumbing were dropped --
# none of that is architecture; Generator/Discriminator here take explicit
# constructor args (nc, ndf, ngf, nz, isize, ngpu) instead of an `opt` object,
# but every Conv1d/BatchNorm1d/LeakyReLU/ConvTranspose1d/Tanh layer, channel
# width, kernel size, and stride is unchanged from the original.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class Encoder(nn.Module):
    """Real repo `Encoder` (experiments/ecg/network.py): 5-stage strided
    Conv1d downsampling stack (320 -> 160 -> 80 -> 40 -> 20 -> 10) collapsed
    to a `out_z`-dim latent code by a final kernel-10 Conv1d."""

    def __init__(self, ngpu, nc, ndf, out_z):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 320
            nn.Conv1d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 160
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 80
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 40
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 20
            nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 10
            nn.Conv1d(ndf * 16, out_z, 10, 1, 0, bias=False),
            # state size. (nz) x 1
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Decoder(nn.Module):
    """Real repo `Decoder`: mirror-image ConvTranspose1d upsampling stack
    (1 -> 10 -> 20 -> 40 -> 80 -> 160 -> 320), Tanh output."""

    def __init__(self, ngpu, nz, ngf, nc):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(nz, ngf * 16, 10, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x10
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 20
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 40
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 80
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 160
            nn.ConvTranspose1d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 320
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Generator(nn.Module):
    """Real repo `Generator` (experiments/ecg/model.py): Encoder-Decoder
    autoencoder. Returns (reconstruction, latent_code)."""

    def __init__(self, ngpu, nc, ndf, ngf, nz):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(ngpu, nc, ndf, nz)
        self.decoder = Decoder(ngpu, nz, ngf, nc)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_x = self.decoder(latent_i)
        return gen_x, latent_i


class Discriminator(nn.Module):
    """Real repo `Discriminator`: an `Encoder(..., out_z=1)` with its final
    Conv1d split off as the sigmoid classifier head; the penultimate feature
    map is also returned (used for feature-matching loss in training)."""

    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        model = Encoder(ngpu, nc, ndf, 1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class BeatGAN(nn.Module):
    """Thin wrapper bundling the real repo's Generator + Discriminator into a
    single traceable module (the real repo's `BeatGAN(AD_MODEL)` class holds
    G/D as attributes and drives them from a hand-rolled adversarial training
    loop rather than a single forward -- that training-loop orchestration is
    non-architectural). forward() runs the same G(x) -> D(G(x)) path used in
    both the generator-loss and discriminator-fake-loss computations."""

    def __init__(self, ngpu=1, nc=1, ndf=32, ngf=32, nz=50):
        super(BeatGAN, self).__init__()
        self.G = Generator(ngpu, nc, ndf, ngf, nz)
        self.D = Discriminator(ngpu, nc, ndf)

    def forward(self, x):
        gen_x, latent_i = self.G(x)
        classifier, features = self.D(gen_x)
        return gen_x, latent_i, classifier, features


def build_beatgan():
    # Tiny config matching the real repo's ECG defaults' *shape* (nc=1 channel
    # ECG beat, isize=320) but with reduced ndf/ngf/nz for a fast, small trace.
    return BeatGAN(ngpu=1, nc=1, ndf=8, ngf=8, nz=10)


def example_input_beatgan():
    # (batch, nc=1, isize=320) matches the real repo's fixed 320-sample ECG
    # beat window (required: 5 stride-2 halvings of 320 land exactly on the
    # kernel-10 bottleneck conv, i.e. 320/32=10).
    return torch.randn(2, 1, 320)


MENAGERIE_ENTRIES = [
    (
        "BeatGAN",
        build_beatgan,
        example_input_beatgan,
        2019,
        MENAGERIE_ZOO,
    ),
]

# SOURCE: vendored from https://github.com/arunppsg/TadGAN @ master
# (model.py + anomaly_detection.py test-time pipeline)
#
# TadGAN (Time-series Anomaly Detection GAN, Geiger et al. 2020) -- real repo
# code, vendored verbatim. A bidirectional-LSTM Encoder maps a raw signal
# window to a latent code, a bidirectional-LSTM Decoder reconstructs the
# signal from the latent code, and two Wasserstein critics (CriticX over the
# signal domain, CriticZ over the latent domain) provide the adversarial
# training signal (cycle-consistent GAN, i.e. a CycleGAN-style architecture
# applied to time series). None of the four `nn.Module` classes (Encoder,
# Decoder, CriticX, CriticZ) were changed -- same LSTM/Linear layer stack,
# same shapes, same forward() bodies as the real repo. The real repo's
# `anomaly_detection.test()` test-time pipeline (`critic_x(x)` and
# `decoder(encoder(x))`) is wrapped here in a thin `TadGANPipeline` module
# so the whole real GAN traces through one forward call; the pipeline
# wrapper itself contributes no new layers, only calls into the four real
# submodules. The dropped code is training-only (critic_x_iteration /
# critic_z_iteration / encoder_iteration / decoder_iteration Wasserstein-GP
# loss functions and the file-path/`.pt`-checkpoint bookkeeping constructor
# args) -- none of it is architecture.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class Encoder(nn.Module):
    def __init__(self, encoder_path, signal_shape=100):
        super(Encoder, self).__init__()
        self.signal_shape = signal_shape
        self.lstm = nn.LSTM(
            input_size=self.signal_shape, hidden_size=20, num_layers=1, bidirectional=True
        )
        self.dense = nn.Linear(in_features=40, out_features=20)
        self.encoder_path = encoder_path

    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_path, signal_shape=100):
        super(Decoder, self).__init__()
        self.signal_shape = signal_shape
        self.lstm = nn.LSTM(input_size=20, hidden_size=64, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(in_features=128, out_features=self.signal_shape)
        self.decoder_path = decoder_path

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = self.dense(x)
        return x


class CriticX(nn.Module):
    def __init__(self, critic_x_path, signal_shape=100):
        super(CriticX, self).__init__()
        self.signal_shape = signal_shape
        self.dense1 = nn.Linear(in_features=self.signal_shape, out_features=20)
        self.dense2 = nn.Linear(in_features=20, out_features=1)
        self.critic_x_path = critic_x_path

    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class CriticZ(nn.Module):
    def __init__(self, critic_z_path):
        super(CriticZ, self).__init__()
        self.dense1 = nn.Linear(in_features=20, out_features=1)
        self.critic_z_path = critic_z_path

    def forward(self, x):
        x = self.dense1(x)
        return x


class TadGANPipeline(nn.Module):
    """Thin wrapper reproducing the real repo's test-time pipeline
    (anomaly_detection.test): reconstructed_signal = decoder(encoder(x)),
    critic_score = critic_x(x). Adds no layers of its own -- only calls
    into the four real vendored submodules above."""

    def __init__(self, signal_shape=100):
        super().__init__()
        self.encoder = Encoder(encoder_path=None, signal_shape=signal_shape)
        self.decoder = Decoder(decoder_path=None, signal_shape=signal_shape)
        self.critic_x = CriticX(critic_x_path=None, signal_shape=signal_shape)
        self.critic_z = CriticZ(critic_z_path=None)

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        cx_score = self.critic_x(x)
        cz_score = self.critic_z(z)
        return reconstructed, cx_score, cz_score


def build_tadgan():
    # signal_shape=20 (real repo default is 100; Encoder/Decoder/CriticX all
    # hardcode batch_size=64 in their forward() view() calls, matching the
    # real repo's `batch_size = 64` training constant, so example_input_
    # below must supply exactly 64*signal_shape elements).
    return TadGANPipeline(signal_shape=20)


def example_input_tadgan():
    # Real Encoder.forward does x.view(1, 64, signal_shape) -- the input
    # must reshape cleanly to (1, 64, 20), matching the real repo's
    # `sample['signal'].view(1, batch_size, signal_shape)` call.
    return torch.randn(64, 20)


MENAGERIE_ENTRIES = [
    (
        "TadGAN",
        build_tadgan,
        example_input_tadgan,
        2020,
        MENAGERIE_ZOO,
    ),
]

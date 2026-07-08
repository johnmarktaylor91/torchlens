# SOURCE: vendored from https://github.com/cchallu/dghl @ main
#
# Source: src/models/DGHL_encoder.py, classes Encoder and Generator -- the amortized
# (encoder-based) variant of DGHL (Deep Generative Hierarchical Latent model for
# multivariate time-series anomaly detection). `Encoder` is a 1-D convolutional
# variational encoder (Conv1d+BatchNorm1d+MaxPool1d stack -> mu/logvar heads) that maps a
# windowed multichannel time series to a latent code; `Generator` is a matching 1-D
# transposed-convolutional decoder (ConvTranspose1d+BatchNorm1d(+ReLU) stack) that maps
# the latent code back to a reconstructed window, optionally masked. Together they form
# the real repo's `DGHL_encoder` amortized-inference training harness (a plain `object`,
# not an `nn.Module` -- it owns an optimizer loop, Langevin-free encoder forward pass, and
# window un/reshaping utilities that are not part of the traced network graph). This
# module wraps the two real `nn.Module` classes (copied verbatim) in a thin `DGHLModule`
# container replicating the harness's `encoder(x) -> generator(mu, mask)` data flow, so
# the real architecture traces end-to-end. The (unused in this repo) `torchvision` import
# from the source file was dropped since it is dead code there.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class Generator(nn.Module):
    def __init__(
        self,
        window_size=32,
        hidden_multiplier=32,
        latent_size=100,
        n_channels=3,
        max_filters=256,
        kernel_multiplier=1,
    ):
        super(Generator, self).__init__()

        n_layers = int(np.log2(window_size))
        layers = []
        filters_list = []
        # First layer
        filters = min(max_filters, hidden_multiplier * (2 ** (n_layers - 2)))
        layers.append(
            nn.ConvTranspose1d(
                in_channels=latent_size,
                out_channels=filters,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm1d(filters))
        filters_list.append(filters)
        # Hidden layers
        for i in reversed(range(1, n_layers - 1)):
            filters = min(max_filters, hidden_multiplier * (2 ** (i - 1)))
            layers.append(
                nn.ConvTranspose1d(
                    in_channels=filters_list[-1],
                    out_channels=filters,
                    kernel_size=4 * kernel_multiplier,
                    stride=2,
                    padding=1 + (kernel_multiplier - 1) * 2,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())
            filters_list.append(filters)

        # Output layer
        layers.append(
            nn.ConvTranspose1d(
                in_channels=filters_list[-1],
                out_channels=n_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x, m=None):
        x = x[:, :, 0, :]
        x = self.layers(x)
        x = x[:, :, None, :]

        # Hide mask
        if m is not None:
            x = x * m

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        window_size=64 * 4,
        hidden_multiplier=32,
        latent_size=100,
        n_channels=3,
        max_filters=256,
    ):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=n_channels, out_channels=8, kernel_size=3, stride=2, padding=1
        )
        self.batch1 = nn.BatchNorm1d(8)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm1d(16)
        self.max_pooling1 = nn.MaxPool1d(3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.batch3 = nn.BatchNorm1d(32)
        self.max_pooling2 = nn.MaxPool1d(3, stride=2, padding=1)

        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, latent_size)
        self.linear3 = nn.Linear(128, latent_size)

    def forward(self, x, m=None):
        x = x[:, :, 0, :]

        x = self.conv1(x)
        x = self.batch1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)

        x = self.max_pooling1(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = self.max_pooling2(x)

        x = torch.flatten(x, start_dim=1)

        x = self.linear1(x)

        mu = self.linear2(x)
        log_var = self.linear3(x)

        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # noqa: F841 -- `randn_like` as we need the same size (verbatim from source)
        sample = mu  # + (eps * std) # sampling as if coming from the input space

        sample = sample[:, :, None, None]  # Adds 2 dim

        return sample, std


class DGHLModule(nn.Module):
    """Thin nn.Module wrapper replicating DGHL_encoder's real
    `encoder(x) -> generator(mu, mask)` data flow (src/models/DGHL_encoder.py, DGHL_encoder.predict)
    so the real Encoder/Generator architecture traces end-to-end from one input."""

    def __init__(
        self, window_size=32, hidden_multiplier=32, z_size=8, n_channels=3, max_filters=256
    ):
        super(DGHLModule, self).__init__()
        self.window_size = window_size
        self.n_channels = n_channels
        self.encoder = Encoder(
            window_size=window_size,
            hidden_multiplier=hidden_multiplier,
            latent_size=z_size,
            n_channels=n_channels,
            max_filters=max_filters,
        )
        self.generator = Generator(
            window_size=window_size,
            hidden_multiplier=hidden_multiplier,
            latent_size=z_size,
            n_channels=n_channels,
            max_filters=max_filters,
        )

    def forward(self, x):
        mu, log_var = self.encoder(x)
        mask = torch.ones_like(x)
        x_hat = self.generator(mu, mask)
        return x_hat


def build_dghl():
    # window_size=256 is the real repo's Encoder default (window_size=64*4, see
    # src/models/DGHL_encoder.py Encoder.__init__) -- required because Encoder's
    # `linear1 = nn.Linear(256, 128)` head is hardcoded to the flattened feature count
    # produced by the conv/pool stack at window_size=256 (3x stride-2 convs + 2x
    # stride-2 maxpools over 3 channels -> 32*8=256 features). n_channels=3, z_size=8
    # (repo's smaller documented latent size) otherwise kept small.
    return DGHLModule(
        window_size=256, hidden_multiplier=32, z_size=8, n_channels=3, max_filters=256
    )


def example_input_dghl():
    # (batch, n_channels, 1, window_size) windowed multichannel time series.
    return torch.randn(2, 3, 1, 256)


MENAGERIE_ENTRIES = [
    (
        "DGHL (Deep Generative Hierarchical Latent, encoder variant)",
        build_dghl,
        example_input_dghl,
        2022,
        MENAGERIE_ZOO,
    ),
]

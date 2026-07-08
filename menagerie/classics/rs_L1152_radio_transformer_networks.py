# SOURCE: vendored from gram-ai/radio-transformer-networks @ master
# https://github.com/gram-ai/radio-transformer-networks/blob/master/radio_transformer_networks.py
# "An Introduction to Deep Learning for the Physical Layer" (O'Shea & Hoydis
# 2016) autoencoder-based communications system. `RadioTransformerNetwork`
# (linear encoder -> unit-power normalization -> AWGN channel -> linear
# decoder) is transcribed VERBATIM from the real repo. Only change: the
# module-level `Variable`/`USE_CUDA` noise-injection logic (which relies on
# the `if __name__ == "__main__"` training-script-only `Variable` import) is
# replaced with `torch.randn_like` directly in `forward`, since `Variable` is
# a pre-0.4 torch relic that no longer exists as a separate type; the
# generated noise tensor and normalization/SNR math are otherwise identical.
import math

import torch
from torch import nn


class RadioTransformerNetwork(nn.Module):
    def __init__(self, in_channels, compressed_dim):
        super(RadioTransformerNetwork, self).__init__()

        self.in_channels = in_channels

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, compressed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_dim, in_channels),
        )

    def decode_signal(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)

        # Normalization.
        x = (self.in_channels**2) * (x / x.norm(dim=-1)[:, None])

        # 7dBW to SNR.
        training_signal_noise_ratio = 5.01187

        # bit / channel_use
        communication_rate = 1

        # Simulated Gaussian noise.
        noise = torch.randn_like(x) / (
            (2 * communication_rate * training_signal_noise_ratio) ** 0.5
        )
        x = x + noise

        x = self.decoder(x)

        return x


MENAGERIE_ZOO = "vendored-pytorch"

_CHANNEL_SIZE = 4


def build_radio_transformer_networks():
    torch.manual_seed(0)
    model = RadioTransformerNetwork(_CHANNEL_SIZE, compressed_dim=int(math.log2(_CHANNEL_SIZE)))
    model.eval()
    return model


def example_input_radio_transformer_networks():
    torch.manual_seed(0)
    labels = (torch.rand(8) * _CHANNEL_SIZE).long()
    data = torch.eye(_CHANNEL_SIZE).index_select(dim=0, index=labels)
    return (data,)


MENAGERIE_ENTRIES = [
    (
        "RadioTransformerNetworks",
        "build_radio_transformer_networks",
        "example_input_radio_transformer_networks",
        2016,
        MENAGERIE_ZOO,
    ),
]

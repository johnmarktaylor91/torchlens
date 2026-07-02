# SOURCE: vendored from gram-ai/radio-transformer-networks @ master
# https://github.com/gram-ai/radio-transformer-networks/blob/master/radio_transformer_networks.py
# PyTorch implementation of O'Shea & Hoydis, "An Introduction to Deep Learning for the
# Physical Layer" (IEEE TCCN 2017): an autoencoder communications system where the
# encoder acts as transmitter (message -> normalized channel symbols) and the decoder
# acts as receiver (noisy channel symbols -> recovered message), with a simulated
# AWGN channel injected between them at a fixed training SNR. Transcribed verbatim
# from the real `RadioTransformerNetwork` class; only the `__main__` training driver
# (which pulls in `tqdm`/`torchnet`, non-base packages) is dropped -- the model class
# itself uses only `torch`/`torch.nn`.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


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
        noise = torch.randn(*x.size(), device=x.device) / (
            (2 * communication_rate * training_signal_noise_ratio) ** 0.5
        )
        x = x + noise

        x = self.decoder(x)

        return x


def build_radio_transformer_networks():
    # Source __main__ smoke config: CHANNEL_SIZE=4, compressed_dim=log2(CHANNEL_SIZE).
    import math

    channel_size = 4
    return RadioTransformerNetwork(channel_size, compressed_dim=int(math.log2(channel_size)))


def example_input_radio_transformer_networks():
    # Source trains on one-hot message vectors of width CHANNEL_SIZE.
    torch.manual_seed(0)
    channel_size = 4
    labels = (torch.rand(8) * channel_size).long()
    return (torch.eye(channel_size).index_select(dim=0, index=labels),)


MENAGERIE_ENTRIES = [
    (
        "Radio-Transformer-Networks",
        "build_radio_transformer_networks",
        "example_input_radio_transformer_networks",
        2017,
        "vendored-pytorch",
    ),
]

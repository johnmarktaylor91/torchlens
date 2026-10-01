# SOURCE: vendored from NVlabs/sionna @ 04ddb9312116b408093b9d3ad363a3df355093a6
# (tutorials/phy/Neural_Receiver.ipynb, classes ResidualBlock / NeuralReceiver). The queue's
# "Sionna Neural Demapper" candidate refers to this component: Sionna's official convolutional
# neural-network OFDM receiver, which replaces classical channel-estimation + LMMSE-equalization
# + demapping with an end-to-end learned CNN that maps post-DFT received resource-grid samples
# (plus noise-variance side information) directly to bit LLRs. Sionna 2.x's PHY layer (`Block`)
# is itself torch-based, but this architecture is self-contained plain `torch.nn` code with no
# dependency on the `sionna` package at all -- an input Conv2d, 4 residual conv blocks (each
# LayerNorm -> ReLU -> Conv2d -> LayerNorm -> ReLU -> Conv2d -> skip-add), and an output Conv2d.
# Only imports/docstring framing changed from the source; class bodies are verbatim.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# Tiny stand-ins for the notebook's real OFDM resource-grid dimensions
# (num_ofdm_symbols=14, fft_size=128 in the source configuration).
num_ofdm_symbols = 4
fft_size = 8


class ResidualBlock(nn.Module):
    """
    Convolutional residual block with two convolutional layers, ReLU activation,
    layer normalization, and a skip connection.

    The number of convolutional channels of the input must match num_conv_channels
    for the skip connection to work.

    Input shape: [batch_size, num_conv_channels, num_ofdm_symbols, num_subcarriers]
    Output shape: [batch_size, num_conv_channels, num_ofdm_symbols, num_subcarriers]
    """

    def __init__(self, num_conv_channels: int):
        super().__init__()
        # Layer normalization over the last three dimensions (C, H, W)
        self._layer_norm_1 = nn.LayerNorm([num_conv_channels, num_ofdm_symbols, fft_size])
        self._conv_1 = nn.Conv2d(
            in_channels=num_conv_channels,
            out_channels=num_conv_channels,
            kernel_size=3,
            padding=1,  # 'same' padding
        )
        self._layer_norm_2 = nn.LayerNorm([num_conv_channels, num_ofdm_symbols, fft_size])
        self._conv_2 = nn.Conv2d(
            in_channels=num_conv_channels,
            out_channels=num_conv_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self._layer_norm_1(inputs)
        z = F.relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = F.relu(z)
        z = self._conv_2(z)
        # Skip connection
        z = z + inputs
        return z


class NeuralReceiver(nn.Module):
    """
    Residual convolutional neural receiver.

    This neural receiver is fed with the post-DFT received samples, forming a
    resource grid of size num_ofdm_symbols x fft_size, and computes LLRs on
    the transmitted coded bits.

    Input
    -----
    y : [batch_size, num_rx_antenna, num_ofdm_symbols, num_subcarriers], complex
        Received post-DFT samples.
    no : [batch_size], float
        Noise variance.

    Output
    ------
    llr : [batch_size, num_ofdm_symbols, num_subcarriers, num_bits_per_symbol], float
        LLRs on the transmitted bits.
    """

    def __init__(self, num_conv_channels: int, num_bits_per_symbol: int):
        super().__init__()
        self._num_bits_per_symbol = num_bits_per_symbol

        # Input convolution: 2*num_rx_antenna + 1 input channels (real, imag, noise)
        # For dual polarization: 2*2 + 1 = 5 input channels
        num_input_channels = 2 * 2 + 1  # 2 antennas, real+imag, plus noise
        self._input_conv = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=num_conv_channels,
            kernel_size=3,
            padding=1,
        )
        # Residual blocks
        self._res_block_1 = ResidualBlock(num_conv_channels)
        self._res_block_2 = ResidualBlock(num_conv_channels)
        self._res_block_3 = ResidualBlock(num_conv_channels)
        self._res_block_4 = ResidualBlock(num_conv_channels)
        # Output convolution
        self._output_conv = nn.Conv2d(
            in_channels=num_conv_channels,
            out_channels=num_bits_per_symbol,
            kernel_size=3,
            padding=1,
        )

    def forward(self, y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
        # y: [batch, num_rx_ant, num_ofdm_symbols, num_subcarriers]
        # no: [batch]

        # Feeding the noise power in log10 scale helps with the performance
        no = torch.log10(no)

        # Stack real and imaginary components
        y_real = y.real  # [batch, num_rx_ant, time, freq]
        y_imag = y.imag  # [batch, num_rx_ant, time, freq]

        # Reshape noise to [batch, 1, 1, 1] and broadcast to match y's batch size
        batch_size = y.shape[0]
        no = no.view(-1, 1, 1, 1)
        no = no.expand(batch_size, 1, y.shape[2], y.shape[3])  # [batch, 1, time, freq]

        # Concatenate: [batch, 2*num_rx_ant + 1, time, freq]
        z = torch.cat(
            [
                y_real[:, 0:1],  # real part of antenna 0
                y_real[:, 1:2],  # real part of antenna 1
                y_imag[:, 0:1],  # imag part of antenna 0
                y_imag[:, 1:2],  # imag part of antenna 1
                no,
            ],
            dim=1,
        )

        # Input conv
        z = self._input_conv(z)
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv: [batch, num_bits_per_symbol, time, freq]
        z = self._output_conv(z)

        # Transpose to [batch, time, freq, num_bits_per_symbol]
        z = z.permute(0, 2, 3, 1)

        return z


def build_sionna_neural_receiver():
    model = NeuralReceiver(num_conv_channels=8, num_bits_per_symbol=2)
    model.eval()
    return model


def example_input_sionna_neural_receiver():
    y = torch.randn(2, 2, num_ofdm_symbols, fft_size, dtype=torch.complex64)
    no = torch.full((2,), 0.1)
    return (y, no)


MENAGERIE_ENTRIES = [
    (
        "Sionna-NeuralReceiver",
        "build_sionna_neural_receiver",
        "example_input_sionna_neural_receiver",
        2024,
        MENAGERIE_ZOO,
    ),
]

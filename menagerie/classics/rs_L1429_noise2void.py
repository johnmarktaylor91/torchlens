# FAITHFUL PORT of juglab/n2v @ main (original framework: TensorFlow/Keras via csbdeep)
#
# Noise2Void (Krull, Buchholz & Jug, CVPR 2019) -- a self-supervised blind-spot
# denoising CNN. The N2V *contribution* is the blind-spot masking training
# scheme (N2V_DataWrapper / n2v_utils manipulate_val_data), not a new
# architecture: the network itself is the standard CARE/CSBDeep configurable
# U-Net (`n2v.nets.unet.build_unet` -> `n2v.nets.unet_blocks.unet_block`,
# which composes `csbdeep.internals.blocks.conv_block2`). Real source (TF1/
# Keras, `tensorflow.keras.layers.Conv2D/MaxPooling2D/UpSampling2D/Concatenate/
# Add/Activation`) is not runnable in our torch-only base env (would need
# `tensorflow` + `csbdeep`, neither installed and not base libs here), so the
# architecture below is TRANSCRIBED faithfully layer-for-layer into torch:
#
#   n2v/nets/unet.py:build_unet(...)                     -> Noise2VoidUNet.forward
#   n2v/nets/unet_blocks.py:unet_block(...)               -> Noise2VoidUNet down/mid/up loops
#   csbdeep/internals/blocks.py:conv_block2(...)          -> ConvBlock2D
#
# Default config values (n2v.models.n2v_config.N2VConfig defaults, 2D case):
#   unet_n_depth=2, unet_kern_size=5, unet_n_first=32, n_conv_per_depth=2,
#   activation='relu', unet_last_activation='linear', pool=(2,2),
#   batch_norm=False, blurpool=False, skip_skipone=False, residual=False,
#   prob_out=False. `n_channel_in == n_channel_out` (single-channel grayscale
#   is the common N2V use case; kept here as in/out channels = 1).
#
# Only base-lib deps used: torch, torch.nn, torch.nn.functional.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"

# Tiny-init constants (originals: unet_n_depth=2, unet_n_first=32, kern_size=5;
# shrunk here only for a fast trace -- same architecture/mechanism, smaller width).
_N_DEPTH = 2
_N_FILTER_BASE = 4
_KERN = 5
_N_CONV_PER_DEPTH = 2
_POOL = 2
_N_CHANNELS = 1


class ConvBlock2D(nn.Module):
    """Port of csbdeep.internals.blocks.conv_block2 (Conv2D 'same' + activation
    [+ optional BatchNorm, disabled by default config] + optional Dropout,
    disabled by default config)."""

    def __init__(self, in_channels, n_filter, kernel_size, activation="relu"):
        super().__init__()
        pad = kernel_size // 2  # Keras padding="same" for odd kernel, stride=1
        self.conv = nn.Conv2d(in_channels, n_filter, kernel_size, padding=pad)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "linear":
            pass
        else:
            raise ValueError(f"unsupported activation {self.activation}")
        return x


class Noise2VoidUNet(nn.Module):
    """Port of n2v.nets.unet_blocks.unet_block + n2v.nets.unet.build_unet
    (blurpool=False, skip_skipone=False, batch_norm=False, residual=False,
    prob_out=False -- the N2V library defaults)."""

    def __init__(
        self,
        n_channels=_N_CHANNELS,
        n_depth=_N_DEPTH,
        n_filter_base=_N_FILTER_BASE,
        kernel_size=_KERN,
        n_conv_per_depth=_N_CONV_PER_DEPTH,
        pool=_POOL,
        last_activation="linear",
    ):
        super().__init__()
        self.n_depth = n_depth
        self.n_conv_per_depth = n_conv_per_depth
        self.pool = pool
        self.last_activation = last_activation

        # down path: n_depth stages, each n_conv_per_depth ConvBlock2D, then maxpool
        self.down_blocks = nn.ModuleList()
        in_ch = n_channels
        for n in range(n_depth):
            stage = nn.ModuleList()
            n_filter = n_filter_base * (2**n)
            for i in range(n_conv_per_depth):
                stage.append(ConvBlock2D(in_ch, n_filter, kernel_size, activation="relu"))
                in_ch = n_filter
            self.down_blocks.append(stage)
        self.pool_op = nn.MaxPool2d(pool)

        # middle: n_conv_per_depth - 1 blocks at n_filter_base * 2**n_depth,
        # then 1 block at n_filter_base * 2**max(0, n_depth - 1)
        mid_filter = n_filter_base * (2**n_depth)
        self.middle_blocks = nn.ModuleList()
        for i in range(n_conv_per_depth - 1):
            self.middle_blocks.append(
                ConvBlock2D(in_ch, mid_filter, kernel_size, activation="relu")
            )
            in_ch = mid_filter
        last_mid_filter = n_filter_base * (2 ** max(0, n_depth - 1))
        self.middle_last = ConvBlock2D(in_ch, last_mid_filter, kernel_size, activation="relu")
        in_ch = last_mid_filter

        # up path: for n in reversed(range(n_depth)): concat with skip[n], then
        # (n_conv_per_depth - 1) blocks at n_filter_base*2**n, then 1 block at
        # n_filter_base*2**max(0, n-1) (or last_activation on the final stage)
        self.up_blocks = nn.ModuleList()
        skip_channels = [n_filter_base * (2**n) for n in range(n_depth)]
        for n in reversed(range(n_depth)):
            stage = nn.ModuleList()
            concat_ch = in_ch + skip_channels[n]
            n_filter = n_filter_base * (2**n)
            cur_in = concat_ch
            for i in range(n_conv_per_depth - 1):
                stage.append(ConvBlock2D(cur_in, n_filter, kernel_size, activation="relu"))
                cur_in = n_filter
            final_filter = n_filter_base * (2 ** max(0, n - 1))
            final_activation = "relu" if n > 0 else last_activation
            stage.append(
                ConvBlock2D(cur_in, final_filter, kernel_size, activation=final_activation)
            )
            cur_in = final_filter
            self.up_blocks.append(stage)
            in_ch = cur_in

        # final 1x1 conv back to n_channels ("linear" activation, matches
        # build_unet's `conv(num_channels, (1,)*n_dim, activation='linear')`)
        self.final_conv = nn.Conv2d(in_ch, n_channels, kernel_size=1)

    def forward(self, x):
        skip_layers = []
        layer = x
        for stage in self.down_blocks:
            for block in stage:
                layer = block(layer)
            skip_layers.append(layer)
            layer = self.pool_op(layer)

        for block in self.middle_blocks:
            layer = block(layer)
        layer = self.middle_last(layer)

        for idx, n in enumerate(reversed(range(self.n_depth))):
            layer = F.interpolate(layer, scale_factor=self.pool, mode="nearest")
            layer = torch.cat([layer, skip_layers[n]], dim=1)
            for block in self.up_blocks[idx]:
                layer = block(layer)

        return self.final_conv(layer)


def build_noise2void():
    torch.manual_seed(0)
    model = Noise2VoidUNet()
    model.eval()
    return model


def example_input_noise2void():
    torch.manual_seed(0)
    # 2 downsampling stages (n_depth=2) -> H/W must be divisible by 2**2=4.
    return torch.randn(1, _N_CHANNELS, 32, 32)


MENAGERIE_ENTRIES = [
    ("Noise2Void", "build_noise2void", "example_input_noise2void", 2019, MENAGERIE_ZOO),
]

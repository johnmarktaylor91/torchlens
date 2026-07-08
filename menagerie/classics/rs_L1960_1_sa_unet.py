# FAITHFUL PORT of clguo/SA-UNet @ master (original framework: Keras/TensorFlow)
#
# SA-UNet (Guo, Peng, Zhang, Han, "SA-UNet: Spatial Attention U-Net for Retinal Vessel
# Segmentation", ICPR 2020) is a compact U-Net variant for retinal-vessel segmentation
# that adds (1) DropBlock2D (structured spatial dropout, in place of ordinary
# elementwise dropout, applied after every conv) and (2) a single spatial-attention
# gate at the bottleneck between the two bottleneck conv blocks.
#
# The official repo (clguo/SA-UNet) ships the model in Keras/TensorFlow 1.x
# (`SA_UNet.py` builds the functional-API graph via `keras.layers`, `Dropblock.py`
# implements `DropBlock2D` as a `keras.layers.Layer` using `keras.backend` ops,
# `Spatial_Attention.py` implements the CBAM-style spatial-attention gate). Neither
# `keras` (the standalone/legacy package these files import, pre-`keras.engine.base_layer`
# API) nor TF1.x is an installed base lib here, and installing them is out of scope per
# the RUNG-2 "no new package installs" rule, so this is a faithful architectural
# transcription into base-env torch:
#   - `SA_UNet.py::SA_UNet(...)`: the 4-level encoder / bottleneck+spatial-attention /
#     4-level decoder graph -- every Conv2D/Conv2DTranspose/MaxPooling2D/BatchNorm/
#     concatenate/Activation call, in the same order, with the same channel widths
#     (`start_neurons * {1,2,4,8}`), same kernel sizes (3x3 convs, 2x2 pool/deconv
#     stride), and the same "conv -> DropBlock2D -> BatchNorm -> ReLU" block repeated
#     twice per level (bottleneck additionally inserts spatial_attention between its
#     two blocks) is reproduced 1:1 as `SAUNet` below.
#   - `Dropblock.py::DropBlock2D`: the real gamma-based block-mask-dropout mechanism
#     (compute drop probability per unit from block_size/keep_prob/spatial size, sample
#     a Bernoulli seed mask restricted to a valid interior region, dilate it with a
#     max-pool of size `block_size`, invert, and rescale the surviving activations by
#     `numel(mask) / mask.sum()`) is reproduced as `DropBlock2D` below using
#     `torch.nn.functional.max_pool2d` in place of Keras' `MaxPool2D` layer call. Like
#     the original (`K.in_train_phase`), it is training-mode-only: at `eval()` it is an
#     exact identity, matching Keras' `in_train_phase(dropped_inputs, inputs,
#     training=training)` fallthrough.
#   - `Spatial_Attention.py::spatial_attention`: channel-wise avg-pool and max-pool
#     (each collapsing to 1 channel), concatenated to 2 channels, fed through a bias-free
#     7x7 `Conv2D(filters=1, ..., activation='sigmoid')`, and used to gate
#     (elementwise-multiply) the input feature map. Reproduced as `SpatialAttention` below
#     with `nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)` + `sigmoid` (same
#     "same"-padding semantics as Keras' `padding='same'` for an odd kernel/stride=1 conv).
#   - Only `channels_last`/default `K.image_data_format()=="channels_last"` branches are
#     ported (torch is channels-first natively, so no permute plumbing is needed); the
#     `channels_first` Permute branches in both Keras files are dead code paths under the
#     library's own default and are omitted.
#   - `Model(input=..., output=...)`/`.compile(...)` (Keras graph-construction/training
#     boilerplate, not architecture) is replaced by a plain `nn.Module` with an explicit
#     `forward`.

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class DropBlock2D(nn.Module):
    """Faithful port of Dropblock.py::DropBlock2D (structured spatial dropout).

    Identity at eval() (matches Keras' `K.in_train_phase(dropped_inputs, inputs,
    training=training)`); active block-mask dropout only in `.train()` mode.
    """

    def __init__(self, block_size: int, keep_prob: float):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def _gamma(self, height: int, width: int, device, dtype) -> torch.Tensor:
        height_f = torch.tensor(float(height), device=device, dtype=dtype)
        width_f = torch.tensor(float(width), device=device, dtype=dtype)
        block_size = torch.tensor(float(self.block_size), device=device, dtype=dtype)
        return ((1.0 - self.keep_prob) / (block_size**2)) * (
            (height_f * width_f) / ((height_f - block_size + 1.0) * (width_f - block_size + 1.0))
        )

    def _valid_seed_region(self, height: int, width: int, device, dtype) -> torch.Tensor:
        ys = torch.arange(height, device=device).unsqueeze(1).expand(height, width)
        xs = torch.arange(width, device=device).unsqueeze(0).expand(height, width)
        half = self.block_size // 2
        valid = (ys >= half) & (ys < height - half) & (xs >= half) & (xs < width - half)
        return (
            valid.to(dtype).unsqueeze(0).unsqueeze(-1)
        )  # [1, H, W, 1] (channels-last layout, as in Keras)

    def _drop_mask(self, x_chlast: torch.Tensor) -> torch.Tensor:
        b, height, width, c = x_chlast.shape
        gamma = self._gamma(height, width, x_chlast.device, x_chlast.dtype)
        seed = (
            torch.rand(b, height, width, c, device=x_chlast.device, dtype=x_chlast.dtype) < gamma
        ).to(x_chlast.dtype)
        seed = seed * self._valid_seed_region(height, width, x_chlast.device, x_chlast.dtype)
        # dilate the seed mask with a `block_size` max-pool (matches Keras MaxPool2D(padding='same', strides=1))
        seed_chfirst = seed.permute(0, 3, 1, 2)
        pad = self.block_size // 2
        # 'same' padding for even block_size (e.g. 7 is odd -> symmetric; kept general for any block_size)
        pad_l = pad
        pad_r = self.block_size - 1 - pad
        dilated = F.max_pool2d(
            F.pad(seed_chfirst, (pad_l, pad_r, pad_l, pad_r)), kernel_size=self.block_size, stride=1
        )
        dilated = dilated.permute(0, 2, 3, 1)
        return 1.0 - dilated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        x_chlast = x.permute(0, 2, 3, 1)  # NCHW -> NHWC (Keras' native layout)
        mask = self._drop_mask(x_chlast)
        numel = mask.numel()
        out = x_chlast * mask * (numel / mask.sum().clamp(min=1e-8))
        return out.permute(0, 3, 1, 2)  # NHWC -> NCHW


class SpatialAttention(nn.Module):
    """Faithful port of Spatial_Attention.py::spatial_attention (CBAM-style spatial gate,
    channels_last default branch only)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        gate = torch.sigmoid(self.conv(concat))
        return x * gate


class ConvBlock(nn.Module):
    """Faithful port of the repeated "Conv2D(..., activation=None) -> DropBlock2D ->
    BatchNormalization -> Activation('relu')" pattern used throughout SA_UNet.py."""

    def __init__(self, in_ch: int, out_ch: int, block_size: int, keep_prob: float):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.dropblock = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.dropblock(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SAUNet(nn.Module):
    """Faithful port of SA_UNet.py::SA_UNet(input_size, block_size, keep_prob,
    start_neurons, lr) -- the `lr`/optimizer/`.compile(...)` args are training
    boilerplate, not architecture, and are dropped; the graph itself (channel widths,
    conv/pool/deconv/concat/spatial-attention topology) is unchanged."""

    def __init__(
        self,
        in_channels: int = 3,
        block_size: int = 7,
        keep_prob: float = 0.9,
        start_neurons: int = 16,
    ):
        super().__init__()
        n = start_neurons

        # Encoder level 1
        self.conv1a = ConvBlock(in_channels, n * 1, block_size, keep_prob)
        self.conv1b = ConvBlock(n * 1, n * 1, block_size, keep_prob)
        self.pool1 = nn.MaxPool2d(2)

        # Encoder level 2
        self.conv2a = ConvBlock(n * 1, n * 2, block_size, keep_prob)
        self.conv2b = ConvBlock(n * 2, n * 2, block_size, keep_prob)
        self.pool2 = nn.MaxPool2d(2)

        # Encoder level 3
        self.conv3a = ConvBlock(n * 2, n * 4, block_size, keep_prob)
        self.conv3b = ConvBlock(n * 4, n * 4, block_size, keep_prob)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck (with spatial attention between the two conv blocks)
        self.convma = ConvBlock(n * 4, n * 8, block_size, keep_prob)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.convmb = ConvBlock(n * 8, n * 8, block_size, keep_prob)

        # Decoder level 3
        self.deconv3 = nn.ConvTranspose2d(
            n * 8, n * 4, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.uconv3a = ConvBlock(n * 4 + n * 4, n * 4, block_size, keep_prob)
        self.uconv3b = ConvBlock(n * 4, n * 4, block_size, keep_prob)

        # Decoder level 2
        self.deconv2 = nn.ConvTranspose2d(
            n * 4, n * 2, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.uconv2a = ConvBlock(n * 2 + n * 2, n * 2, block_size, keep_prob)
        self.uconv2b = ConvBlock(n * 2, n * 2, block_size, keep_prob)

        # Decoder level 1
        self.deconv1 = nn.ConvTranspose2d(
            n * 2, n * 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.uconv1a = ConvBlock(n * 1 + n * 1, n * 1, block_size, keep_prob)
        self.uconv1b = ConvBlock(n * 1, n * 1, block_size, keep_prob)

        self.output_conv = nn.Conv2d(n * 1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1b(self.conv1a(x))
        pool1 = self.pool1(conv1)

        conv2 = self.conv2b(self.conv2a(pool1))
        pool2 = self.pool2(conv2)

        conv3 = self.conv3b(self.conv3a(pool2))
        pool3 = self.pool3(conv3)

        convm = self.convma(pool3)
        convm = self.spatial_attention(convm)
        convm = self.convmb(convm)

        deconv3 = self.deconv3(convm)
        uconv3 = torch.cat([deconv3, conv3], dim=1)
        uconv3 = self.uconv3b(self.uconv3a(uconv3))

        deconv2 = self.deconv2(uconv3)
        uconv2 = torch.cat([deconv2, conv2], dim=1)
        uconv2 = self.uconv2b(self.uconv2a(uconv2))

        deconv1 = self.deconv1(uconv2)
        uconv1 = torch.cat([deconv1, conv1], dim=1)
        uconv1 = self.uconv1b(self.uconv1a(uconv1))

        output = torch.sigmoid(self.output_conv(uconv1))
        return output


def build_sa_unet():
    torch.manual_seed(0)
    net = SAUNet(in_channels=3, block_size=7, keep_prob=0.9, start_neurons=4)
    net.eval()
    return net


def example_input_sa_unet():
    torch.manual_seed(0)
    # Real usage is 512x512 fundus-image patches; shrunk spatially (must stay divisible
    # by 2**3=8 for the three pool/deconv stages to round-trip cleanly) and shrunk
    # channel-wise (start_neurons=4 vs the paper's 16) for a tiny trace target.
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("SA-UNet", build_sa_unet, example_input_sa_unet, 2020, MENAGERIE_ZOO),
]

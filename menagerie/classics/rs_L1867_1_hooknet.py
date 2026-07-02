# SOURCE: vendored from DIAGNijmegen/pathology-hooknet @ 3f2a28a4e91c
# (hooknet/models/torchmodel.py). HookNet is a dual/multi-resolution encoder-decoder
# for histopathology segmentation with "hook" skip connections carrying decoder
# features from a coarser (low-magnification) branch into a finer (higher-
# magnification) branch's decoder at a matched depth. The class code below is
# copied verbatim from the real repo file (only the relative import of
# `center_crop` is unchanged -- torchvision already provides it in the base env);
# no architectural line was rewritten.
"""Vendored HookNet (DIAGNijmegen/pathology-hooknet, torch model variant)."""

import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop

MENAGERIE_ZOO = "vendored-pytorch"


class HookNet(nn.Module):
    def __init__(
        self,
        n_classes,
        depth=4,
        n_convs=2,
        n_filters=2,
        batch_norm=True,
    ):
        super().__init__()

        self.low_mag_branch = Branch(
            n_classes=n_classes,
            n_filters=n_filters,
            depth=depth,
            n_convs=n_convs,
            batch_norm=batch_norm,
            hook_from_index=0,
        )

        low_channels = self.low_mag_branch.decoder._out_channels[0]
        self.mid_mag_branch = Branch(
            n_classes=n_classes,
            n_filters=n_filters,
            depth=depth,
            n_convs=n_convs,
            batch_norm=batch_norm,
            hook_channels=low_channels,
            hook_to_index=1,
            hook_from_index=2,
        )

        mid_channels = self.mid_mag_branch.decoder._out_channels[2]
        self.high_mag_branch = Branch(
            n_classes=n_classes,
            n_filters=n_filters,
            depth=depth,
            n_convs=n_convs,
            batch_norm=batch_norm,
            hook_channels=mid_channels,
            hook_to_index=3,
        )

        self.high_last_conv = nn.Conv2d(self.high_mag_branch.decoder._out_channels[0], n_classes, 1)
        self.mid_last_conv = nn.Conv2d(self.mid_mag_branch.decoder._out_channels[0], n_classes, 1)
        self.low_last_conv = nn.Conv2d(self.low_mag_branch.decoder._out_channels[0], n_classes, 1)

    def forward(self, high_input, mid_input, low_input):
        low_out, low_hook_out = self.low_mag_branch(low_input)
        mid_out, mid_hook_out = self.mid_mag_branch(mid_input, low_hook_out)
        high_out, high_hook_out = self.high_mag_branch(high_input, mid_hook_out)
        return {
            "high_out": self.high_last_conv(high_out),
            "mid_out": self.mid_last_conv(mid_out),
            "low_out": self.low_last_conv(low_out),
        }


class Branch(nn.Module):
    def __init__(
        self,
        n_classes,
        n_filters,
        depth,
        n_convs,
        batch_norm,
        hook_channels=0,
        hook_from_index=None,
        hook_to_index=None,
    ):
        super().__init__()
        self.encoder = Encoder(3, n_filters, depth, n_convs, batch_norm)
        self.mid_conv_block = ConvBlock(
            self.encoder._out_channels[depth - 1],
            n_filters * 2 * (depth + 1),
            n_convs,
            batch_norm,
        )
        self.decoder = Decoder(
            n_filters * 2 * (depth + 1),
            hook_channels,
            self.encoder._out_channels,
            n_filters,
            depth,
            n_convs,
            batch_norm,
            hook_from_index,
            hook_to_index,
        )

    def forward(self, x, hook_in=None):
        out, residuals = self.encoder(x)
        out = self.mid_conv_block(out)
        out, hook_out = self.decoder(out, residuals, hook_in)
        return out, hook_out


class Encoder(nn.Module):
    def __init__(self, in_channels, n_filters, depth, n_convs, batch_norm):
        super().__init__()
        self._out_channels = {}
        self._encode_path = nn.ModuleDict()
        self._depth = depth
        for d in range(self._depth):
            self._out_channels[d] = n_filters + in_channels
            self._encode_path[f"convblock{d}"] = ConvBlock(
                in_channels, n_filters, n_convs, batch_norm, residual=True
            )
            self._encode_path[f"pool{d}"] = nn.MaxPool2d((2, 2))
            in_channels += n_filters
            n_filters *= 2

    def forward(self, x):
        residuals = []
        for d in range(self._depth):
            x = self._encode_path[f"convblock{d}"](x)
            residuals.append(x)
            x = self._encode_path[f"pool{d}"](x)
        return x, residuals


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hook_channels,
        encoder_channels,
        n_filters,
        depth,
        n_convs,
        batch_norm,
        hook_from_index,
        hook_to_index,
    ):
        super().__init__()

        self._depth = depth
        self._hook_channels = hook_channels
        self._hook_from_index = hook_from_index
        self._hook_to_index = hook_to_index
        self._decode_path = nn.ModuleDict()
        self._out_channels = {}
        n_filters = n_filters * 2 * depth
        for d in reversed(range(self._depth)):
            self._out_channels[d] = n_filters
            if d == self._hook_to_index:
                in_channels += self._hook_channels

            self._decode_path[f"upsample{d}"] = UpSample(in_channels, n_filters)
            self._decode_path[f"convblock{d}"] = ConvBlock(
                n_filters + encoder_channels[d], n_filters, n_convs, batch_norm
            )

            in_channels = n_filters
            n_filters = n_filters // 2

    def forward(self, x, residuals, hook_in=None):
        out = x
        hook_true = False
        for d in reversed(range(self._depth)):
            if hook_in is not None and d == self._hook_to_index:
                out = concatenator(out, hook_in)

            out = self._decode_path[f"upsample{d}"](out)
            out = concatenator(out, residuals[d])
            out = self._decode_path[f"convblock{d}"](out)

            if self._hook_from_index is not None and d == self._hook_from_index:
                hook_out = out
                hook_true = True

        if not hook_true:
            hook_out = out
        return out, hook_out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs, batch_norm, residual=False):
        super().__init__()
        self._residual = residual
        block = nn.ModuleList()
        for _ in range(n_convs):
            block.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
            )
            block.append(nn.LeakyReLU(inplace=True))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
        self._block = nn.Sequential(*block)

    def forward(self, x):
        out = self._block(x)
        if self._residual:
            return concatenator(out, x)
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sampler = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        out = self.up_sampler(x)
        out = self.conv(out)
        return self.activation(out)


def concatenator(x, x2):
    x2_cropped = center_crop(x2, x.shape[-1])
    conc = torch.cat([x, x2_cropped], dim=1)
    return conc


# ---------------------------------------------------------------------------
# Staging build/example helpers. Real constructor with default kwargs
# (n_classes, depth=4, n_convs=2, n_filters=2, batch_norm=True) matching the
# repo's own `create_hooknet` defaults; input size 220 is the smallest square
# patch (found empirically, matching the "valid" conv/pool geometry the real
# repo also observes -- production inference uses 284x284 -> 70x70 per
# `hooknet/inference/apply_torch.py`) that survives all four downsample/
# upsample stages across all three magnification branches without a negative
# spatial dim.
# ---------------------------------------------------------------------------


def build_hooknet():
    return HookNet(n_classes=3, depth=4, n_convs=2, n_filters=2, batch_norm=True)


def example_input_hooknet():
    high_input = torch.rand(1, 3, 220, 220)
    mid_input = torch.rand(1, 3, 220, 220)
    low_input = torch.rand(1, 3, 220, 220)
    return (high_input, mid_input, low_input)


MENAGERIE_ENTRIES = [
    ("HookNet", "build_hooknet", "example_input_hooknet", 2021, "vendored-pytorch"),
]

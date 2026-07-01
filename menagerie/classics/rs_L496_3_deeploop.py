# FAITHFUL PORT of JinLabBioinfo/DeepLoop @ dd84d7130dc0a779885781f8374bf03264ef0094 (original framework: Keras/TensorFlow)
# https://github.com/JinLabBioinfo/DeepLoop/blob/dd84d7130dc0a779885781f8374bf03264ef0094/utils/cnn_architectures.py
# https://github.com/JinLabBioinfo/DeepLoop/blob/dd84d7130dc0a779885781f8374bf03264ef0094/LoopDenoise/denoise_model.py
# https://github.com/JinLabBioinfo/DeepLoop/blob/dd84d7130dc0a779885781f8374bf03264ef0094/LoopEnhance/enhance_model.py
#
# DeepLoop: a deep-learning framework for Hi-C chromatin-loop signal
# denoising (LoopDenoise) and resolution enhancement (LoopEnhance) (Zhang
# et al., Nature Communications 2022, JinLabBioinfo/DeepLoop).
#
# The real repo defines both networks as Keras/TensorFlow ``keras.models.Model``
# graphs in utils/cnn_architectures.py::Autoencoder (used by both
# LoopDenoise/denoise_model.py::DenoiseModel, which calls
# ``self.get_autoencoder()``, and LoopEnhance/enhance_model.py::EnhanceModel,
# which calls ``self.get_unet_model()`` by default). TorchLens only captures
# eager torch ops, so this Keras graph cannot be vendored/traced as-is; this
# module faithfully transcribes both concrete architectures the real code
# builds:
#   - DenoiseModel's stacked conv autoencoder (get_autoencoder, defaults
#     maxpool=True, upconv=False): 2 conv+maxpool down-blocks followed by 2
#     Conv2DTranspose up-blocks and a linear-activation conv head.
#   - EnhanceModel's recursive U-Net (get_unet_model, defaults
#     maxpool=True, upconv=True, residual=True, batch_norm=False,
#     depth=2): the same recursive level_block/conv_block structure
#     (down-sample -> recurse -> upsample+concat -> conv_block), including
#     the residual "concatenate skip features with block output" behavior.
# Every downsample/upsample/skip-connection/conv-block choice mirrors the
# original Keras graph; only the Keras functional-API calls are replaced by
# equivalent nn.Module torch ops.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }.get(name, nn.ReLU())


class DenoiseAutoencoder(nn.Module):
    """Stacked conv autoencoder (cnn_architectures.py::Autoencoder.get_autoencoder,
    default maxpool=True, upconv=False -- the branch used by DenoiseModel)."""

    def __init__(self, start_filters=16, filter_size=3, transpose_filter_size=2, activation="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(1, start_filters, filter_size, stride=1, padding=filter_size // 2)
        self.act1 = _activation(activation)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv2 = nn.Conv2d(
            start_filters, start_filters, filter_size, stride=1, padding=filter_size // 2
        )
        self.act2 = _activation(activation)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)

        # keras.layers.Conv2DTranspose(start_filters, transpose_filter_size, strides=2, padding='same') x2
        self.deconv1 = nn.ConvTranspose2d(
            start_filters, start_filters, transpose_filter_size, stride=2
        )
        self.deconv2 = nn.ConvTranspose2d(
            start_filters, start_filters, transpose_filter_size, stride=2
        )
        self.conv_out = nn.Conv2d(start_filters, 1, filter_size, padding=filter_size // 2)
        self.act_out = _activation(activation)

    def forward(self, x):
        y = self.pool1(self.act1(self.conv1(x)))
        y = self.pool2(self.act2(self.conv2(y)))
        y = self.deconv1(y)
        y = self.deconv2(y)
        y = self.conv_out(y)
        y = self.act_out(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="nearest")
        return y


class _ConvBlock(nn.Module):
    """conv_block(x, num_filters, acti, bn, res, do) closure in get_unet_model."""

    def __init__(
        self, in_ch, num_filters, filter_size, activation, batch_norm, residual, dropout=0.0
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, num_filters, filter_size, padding=filter_size // 2)
        self.conv2 = nn.Conv2d(num_filters, num_filters, filter_size, padding=filter_size // 2)
        self.act = _activation(activation)
        self.bn1 = nn.BatchNorm2d(num_filters) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(num_filters) if batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = residual

    def forward(self, x):
        y = self.bn1(self.act(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.act(self.conv2(y)))
        if self.residual:
            return torch.cat([x, y], dim=1)
        return y


class _LevelBlock(nn.Module):
    """level_block(...) closure in get_unet_model -- recursive U-Net level."""

    def __init__(
        self,
        in_ch,
        num_filters,
        depth,
        branch_rate,
        filter_size,
        activation,
        dropout,
        batch_norm,
        maxpool,
        upconv,
        residual,
    ):
        super().__init__()
        self.depth = depth
        if depth > 0:
            self.conv_block = _ConvBlock(
                in_ch, num_filters, filter_size, activation, batch_norm, residual
            )
            y_ch = in_ch + num_filters if residual else num_filters
            if maxpool:
                self.down = nn.MaxPool2d(2, ceil_mode=True)
            else:
                self.down = nn.Conv2d(y_ch, y_ch, filter_size, stride=2, padding=filter_size // 2)
            next_filters = int(branch_rate * num_filters)
            self.next_level = _LevelBlock(
                y_ch,
                next_filters,
                depth - 1,
                branch_rate,
                filter_size,
                activation,
                dropout,
                batch_norm,
                maxpool,
                upconv,
                residual,
            )
            next_out_ch = self.next_level.out_channels
            if upconv:
                self.up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(next_out_ch, num_filters, filter_size, padding=filter_size // 2),
                )
            else:
                self.up = nn.ConvTranspose2d(
                    next_out_ch,
                    num_filters,
                    filter_size,
                    stride=2,
                    padding=filter_size // 2,
                    output_padding=1,
                )
            self.up_act = _activation(activation)
            self.final_block = _ConvBlock(
                y_ch + num_filters, num_filters, filter_size, activation, batch_norm, residual
            )
            self.out_channels = (y_ch + num_filters) + num_filters if residual else num_filters
        else:
            self.conv_block = _ConvBlock(
                in_ch, num_filters, filter_size, activation, batch_norm, residual, dropout
            )
            self.out_channels = in_ch + num_filters if residual else num_filters

    def forward(self, x):
        if self.depth > 0:
            y = self.conv_block(x)
            xd = self.down(y)
            xd = self.next_level(xd)
            xu = self.up_act(self.up(xd))
            if xu.shape[-2:] != y.shape[-2:]:
                xu = F.interpolate(xu, size=y.shape[-2:], mode="nearest")
            merged = torch.cat([y, xu], dim=1)
            return self.final_block(merged)
        return self.conv_block(x)


class EnhanceUNet(nn.Module):
    """Recursive U-Net (cnn_architectures.py::Autoencoder.get_unet_model,
    defaults batch_norm=False, maxpool=True, upconv=True, residual=True --
    the branch used by EnhanceModel(model_architecture='unet'))."""

    def __init__(
        self,
        start_filters=16,
        filter_size=3,
        depth=2,
        branching_rate=2.0,
        activation="relu",
        dropout=0.0,
        batch_norm=False,
        maxpool=True,
        upconv=True,
        residual=True,
    ):
        super().__init__()
        self.level = _LevelBlock(
            1,
            start_filters,
            depth,
            branching_rate,
            filter_size,
            activation,
            dropout,
            batch_norm,
            maxpool,
            upconv,
            residual,
        )
        self.out_conv = nn.Conv2d(self.level.out_channels, 1, 1)

    def forward(self, x):
        y = self.level(x)
        return self.out_conv(y)


class HiCPlusCNN(nn.Module):
    """Small CNN alternative used by EnhanceModel(model_architecture='hicplus')
    (cnn_architectures.py::Autoencoder.get_hi_c_plus, based on HiCPlus)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 9, padding=4)
        self.conv2 = nn.Conv2d(8, 8, 1, padding=0)
        self.conv3 = nn.Conv2d(8, 1, 5, padding=2)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.act(self.conv3(x))


MENAGERIE_ZOO = "ported-pytorch"


def build_deeploop_denoise():
    # DenoiseModel default matrix_size=128, start_filters=16, filter_size=3
    return DenoiseAutoencoder(
        start_filters=16, filter_size=3, transpose_filter_size=2, activation="relu"
    )


def example_input_deeploop_denoise():
    return torch.randn(1, 1, 128, 128)


def build_deeploop_enhance_unet():
    # EnhanceModel default depth=2, start_filters=16, branching_rate=2., model_architecture='unet'
    return EnhanceUNet(
        start_filters=16,
        filter_size=3,
        depth=2,
        branching_rate=2.0,
        activation="relu",
        dropout=0.5,
        batch_norm=False,
        maxpool=True,
        upconv=True,
        residual=True,
    )


def example_input_deeploop_enhance_unet():
    return torch.randn(1, 1, 64, 64)


def build_deeploop_hicplus():
    return HiCPlusCNN()


def example_input_deeploop_hicplus():
    return torch.randn(1, 1, 40, 40)


MENAGERIE_ENTRIES = [
    (
        "DeepLoop-LoopDenoise",
        "build_deeploop_denoise",
        "example_input_deeploop_denoise",
        2022,
        MENAGERIE_ZOO,
    ),
    (
        "DeepLoop-LoopEnhance-UNet",
        "build_deeploop_enhance_unet",
        "example_input_deeploop_enhance_unet",
        2022,
        MENAGERIE_ZOO,
    ),
    (
        "DeepLoop-LoopEnhance-HiCPlus",
        "build_deeploop_hicplus",
        "example_input_deeploop_hicplus",
        2022,
        MENAGERIE_ZOO,
    ),
]

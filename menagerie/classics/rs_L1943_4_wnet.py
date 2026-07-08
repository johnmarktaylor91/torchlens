# SOURCE: vendored from gr-b/W-Net-Pytorch @ master
# https://raw.githubusercontent.com/gr-b/W-Net-Pytorch/master/model.py
# https://raw.githubusercontent.com/gr-b/W-Net-Pytorch/master/config.py
#
# "W-Net: A Deep Model for Fully Unsupervised Image Segmentation" (Xia & Kulis, 2017).
# Two stacked U-Nets (`BaseNet`) forming an autoencoder: `U_encoder` maps a 3-channel image
# to a k-channel soft segmentation (softmax), `U_decoder` maps that segmentation back to a
# 3-channel reconstruction (sigmoid) for the unsupervised soft-N-cut + reconstruction loss.
# `ConvModule`/`BaseNet`/`WNet` are copied verbatim from the real repo's model.py. The
# repo's global `config = Config()` (imported from a top-level `config.py` module,
# instantiated once at model.py's module scope) is inlined here as a local
# `_DefaultConfig` dataclass-like class with the exact same field values from the repo's
# own `config.py`, since this build has no package-relative `config` module to import from
# -- this changes only where the config values live (still real repo constants: k=64,
# encoderLayerSizes=[64,128,256], decoderLayerSizes=[512,256], useInstanceNorm=True,
# useBatchNorm=False, useDropout=True, drop=0.2), not the architecture. `output_channels`
# in `BaseNet.__init__`'s default arg (`config.k`) is likewise evaluated against the same
# inlined config object.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class _DefaultConfig:
    """Inlined equivalent of the real repo's config.py Config() defaults."""

    def __init__(self):
        self.k = 64  # Number of classes
        self.useInstanceNorm = True  # Instance Normalization
        self.useBatchNorm = False  # Only use one of either instance or batch norm
        self.useDropout = True
        self.drop = 0.2
        self.encoderLayerSizes = [64, 128, 256]
        self.decoderLayerSizes = [512, 256]


config = _DefaultConfig()


class ConvModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvModule, self).__init__()

        layers = [
            nn.Conv2d(input_dim, output_dim, 1),  # Pointwise (1x1) through all channels
            nn.Conv2d(
                output_dim, output_dim, 3, padding=1, groups=output_dim
            ),  # Depthwise (3x3) through each channel
            nn.InstanceNorm2d(output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(config.drop),
            nn.Conv2d(output_dim, output_dim, 1),
            nn.Conv2d(output_dim, output_dim, 3, padding=1, groups=output_dim),
            nn.InstanceNorm2d(output_dim),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Dropout(config.drop),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class BaseNet(nn.Module):  # 1 U-net
    def __init__(
        self,
        input_channels=3,
        encoder=[64, 128, 256, 512],
        decoder=[1024, 512, 256],
        output_channels=config.k,
    ):
        super(BaseNet, self).__init__()

        layers = [
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.first_module = nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(2, 2)
        self.enc_modules = nn.ModuleList(
            [ConvModule(channels, 2 * channels) for channels in encoder]
        )

        decoder_out_sizes = [int(x / 2) for x in decoder]
        self.dec_transpose_layers = nn.ModuleList(
            [nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder]
        )  # Stride of 2 makes it right size
        self.dec_modules = nn.ModuleList(
            [ConvModule(3 * channels_out, channels_out) for channels_out in decoder_out_sizes]
        )
        self.last_dec_transpose_layer = nn.ConvTranspose2d(128, 128, 2, stride=2)

        layers = [
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(config.drop),
            nn.Conv2d(64, output_channels, 1),  # No padding on pointwise
            nn.ReLU(),
        ]

        if not config.useInstanceNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if not config.useBatchNorm:
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if not config.useDropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.last_module = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.first_module(x)
        activations = [x1]
        for module in self.enc_modules:
            activations.append(module(self.pool(activations[-1])))

        x_ = activations.pop(-1)

        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = activations.pop(-1)
            x_ = conv(torch.cat((skip_connection, upconv(x_)), 1))

        segmentations = self.last_module(
            torch.cat((activations[-1], self.last_dec_transpose_layer(x_)), 1)
        )
        return segmentations


class WNet(nn.Module):
    def __init__(self):
        super(WNet, self).__init__()

        self.U_encoder = BaseNet(
            input_channels=3,
            encoder=config.encoderLayerSizes,
            decoder=config.decoderLayerSizes,
            output_channels=config.k,
        )
        self.softmax = nn.Softmax2d()
        self.U_decoder = BaseNet(
            input_channels=config.k,
            encoder=config.encoderLayerSizes,
            decoder=config.decoderLayerSizes,
            output_channels=3,
        )
        self.sigmoid = nn.Sigmoid()

    def forward_encoder(self, x):
        x9 = self.U_encoder(x)
        segmentations = self.softmax(x9)
        return segmentations

    def forward_decoder(self, segmentations):
        x18 = self.U_decoder(segmentations)
        reconstructions = self.sigmoid(x18)
        return reconstructions

    def forward(self, x):  # x is (3 channels 224x224)
        segmentations = self.forward_encoder(x)
        x_prime = self.forward_decoder(segmentations)
        return segmentations, x_prime


def build_wnet():
    return WNet()


def example_input_wnet():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("WNet", "build_wnet", "example_input_wnet", 2017, "vendored"),
]

# SOURCE: vendored from cy-xu/cosmic-conn @ main (cosmic_conn/dl_framework/unet.py)
# SOURCE: vendored from roggirg/count-ception_mbm @ master (model.py)
# SOURCE: vendored from yzspku/CQTNet @ master (models/CQTNet.py)
# SOURCE: vendored from clovaai/CRAFT-pytorch @ master (craft.py, basenet/vgg16_bn.py)
# SOURCE: vendored from leeyeehoo/CSRNet-pytorch @ master (model.py)
# SOURCE: vendored from LilitYolyan/CutPaste @ main (model.py)
# SOURCE: vendored from LouisSerrano/coral @ main (coral/mlp.py)
from __future__ import annotations

from collections import OrderedDict, namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

MENAGERIE_ZOO = "vendored-pytorch"


class CoralSwish(nn.Module):
    """Swish activation from CORAL."""

    def __init__(self) -> None:
        """Initialize CORAL Swish."""
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CORAL Swish."""
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class CoralResBlock(nn.Module):
    """Residual MLP block from CORAL."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        drop_rate: float = 0.0,
    ) -> None:
        """Initialize a CORAL residual block."""
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.activation1 = CoralSwish()
        self.activation2 = CoralSwish()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CORAL residual block."""
        eta = self.linear1(x)
        eta = self.linear2(self.activation1(eta))
        return x + self.activation2(self.dropout(eta))


class CoralResNet(nn.Module):
    """Residual MLP from CORAL."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 64,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialize a CORAL residual MLP."""
        super().__init__()
        blocks = [CoralResBlock(input_dim, hidden_dim, dropout)]
        for _ in range(depth - 1):
            blocks.append(CoralResBlock(input_dim, hidden_dim, dropout))
        self.net = nn.Sequential(*blocks)
        self.project_map = nn.Linear(input_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Run the CORAL residual MLP."""
        return self.project_map(self.net(z))


class CosmicDoubleConv(nn.Module):
    """Double convolution block from Cosmic-CoNN."""

    def __init__(
        self, in_ch: int, out_ch: int, norm: str, norm_setting: tuple[int, int, bool]
    ) -> None:
        """Initialize the Cosmic-CoNN convolution block."""
        super().__init__()
        group, channel, no_affine = norm_setting
        affine = not no_affine
        n_group = max(1, int(out_ch // channel)) if group == 0 and channel > 0 else group
        norms = nn.ModuleDict(
            {
                "batch": nn.BatchNorm2d(
                    out_ch, momentum=0.005, affine=True, track_running_stats=True
                ),
                "group": nn.GroupNorm(n_group, out_ch, affine=affine),
                "instance": nn.InstanceNorm2d(out_ch, affine=affine),
            }
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
            norms[norm],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
            norms[norm],
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the double convolution block."""
        return self.conv(x)


class CosmicInConv(nn.Module):
    """Input block from Cosmic-CoNN."""

    def __init__(
        self, in_ch: int, out_ch: int, norm: str, norm_setting: tuple[int, int, bool]
    ) -> None:
        """Initialize the input block."""
        super().__init__()
        self.conv = CosmicDoubleConv(in_ch, out_ch, norm, norm_setting)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input block."""
        return self.conv(x)


class CosmicDown(nn.Module):
    """Downsampling block from Cosmic-CoNN."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: str,
        norm_setting: tuple[int, int, bool],
        down_type: str,
    ) -> None:
        """Initialize the downsampling block."""
        super().__init__()
        downs = nn.ModuleDict(
            {
                "maxpool": nn.MaxPool2d(2),
                "avgpool": nn.AvgPool2d(2),
                "stride": nn.Conv2d(in_ch, in_ch, 3, 2, padding=1),
            }
        )
        self.down_cov = nn.Sequential(
            downs[down_type], CosmicDoubleConv(in_ch, out_ch, norm, norm_setting)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run downsampling."""
        return self.down_cov(x)


class CosmicUp(nn.Module):
    """Upsampling block from Cosmic-CoNN."""

    def __init__(
        self, in_ch: int, out_ch: int, norm: str, norm_setting: tuple[int, int, bool]
    ) -> None:
        """Initialize the upsampling block."""
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = CosmicDoubleConv(in_ch, out_ch, norm, norm_setting)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Run upsampling and skip concatenation."""
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CosmicOutConv(nn.Module):
    """Output convolution from Cosmic-CoNN."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize the output projection."""
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the output projection."""
        return self.conv(x)


class CosmicConnUNet(nn.Module):
    """Cosmic-CoNN U-Net module."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden: int,
        norm: str,
        norm_setting: tuple[int, int, bool],
        down_type: str,
        deeper: bool,
    ) -> None:
        """Initialize the Cosmic-CoNN U-Net."""
        super().__init__()
        self.deeper = deeper
        self.inc = CosmicInConv(n_channels, hidden, norm, norm_setting)
        self.down1 = CosmicDown(hidden, hidden * 2, norm, norm_setting, down_type)
        self.down2 = CosmicDown(hidden * 2, hidden * 4, norm, norm_setting, down_type)
        if deeper:
            self.down3 = CosmicDown(hidden * 4, hidden * 8, norm, norm_setting, down_type)
            self.up3 = CosmicUp(hidden * 8, hidden * 4, norm, norm_setting)
        self.up2 = CosmicUp(hidden * 4, hidden * 2, norm, norm_setting)
        self.up1 = CosmicUp(hidden * 2, hidden, norm, norm_setting)
        self.outc = CosmicOutConv(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Cosmic-CoNN segmentation."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if self.deeper:
            x4 = self.down3(x3)
            x = self.up3(x4, x3)
            x = self.up2(x, x2)
            x = self.up1(x, x1)
        else:
            x = self.up2(x3, x2)
            x = self.up1(x, x1)
        return torch.sigmoid(self.outc(x))


class CountceptionConvBlock(nn.Module):
    """Count-ception convolution block."""

    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        ksize: int = 3,
        stride: int = 1,
        pad: int = 0,
    ) -> None:
        """Initialize the Count-ception convolution block."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)
        self.activation = nn.LeakyReLU(0.01)
        self.batch_norm = nn.BatchNorm2d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run convolution, batch normalization, and activation."""
        return self.activation(self.batch_norm(self.conv1(x)))


class CountceptionSimpleBlock(nn.Module):
    """Parallel 1x1 and 3x3 Count-ception block."""

    def __init__(self, in_chan: int, out_chan_1x1: int, out_chan_3x3: int) -> None:
        """Initialize the parallel Count-ception block."""
        super().__init__()
        self.conv1 = CountceptionConvBlock(in_chan, out_chan_1x1, ksize=1, pad=0)
        self.conv2 = CountceptionConvBlock(in_chan, out_chan_3x3, ksize=3, pad=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both branches and concatenate their outputs."""
        return torch.cat([self.conv1(x), self.conv2(x)], 1)


class ModelCountception(nn.Module):
    """Count-ception model from the PyTorch port."""

    def __init__(self, inplanes: int = 3, outplanes: int = 1) -> None:
        """Initialize Count-ception."""
        super().__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.patch_size = 32
        self.conv1 = CountceptionConvBlock(self.inplanes, 64, ksize=3, pad=self.patch_size)
        self.simple1 = CountceptionSimpleBlock(64, 16, 16)
        self.simple2 = CountceptionSimpleBlock(32, 16, 32)
        self.conv2 = CountceptionConvBlock(48, 16, ksize=14)
        self.simple3 = CountceptionSimpleBlock(16, 112, 48)
        self.simple4 = CountceptionSimpleBlock(160, 64, 32)
        self.simple5 = CountceptionSimpleBlock(96, 40, 40)
        self.simple6 = CountceptionSimpleBlock(80, 32, 96)
        self.conv3 = CountceptionConvBlock(128, 32, ksize=18)
        self.conv4 = CountceptionConvBlock(32, 64, ksize=1)
        self.conv5 = CountceptionConvBlock(64, 64, ksize=1)
        self.conv6 = CountceptionConvBlock(64, self.outplanes, ksize=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Count-ception."""
        net = self.conv1(x)
        net = self.simple1(net)
        net = self.simple2(net)
        net = self.conv2(net)
        net = self.simple3(net)
        net = self.simple4(net)
        net = self.simple5(net)
        net = self.simple6(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net)
        return self.conv6(net)


class CQTNet(nn.Module):
    """CQTNet for cover song identification."""

    def __init__(self) -> None:
        """Initialize CQTNet."""
        super().__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            1, 32, kernel_size=(12, 3), dilation=(1, 1), padding=(6, 0), bias=False
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(32)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("conv1", nn.Conv2d(32, 64, kernel_size=(13, 3), dilation=(1, 2), bias=False)),
                    ("norm1", nn.BatchNorm2d(64)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("pool1", nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
                    ("conv2", nn.Conv2d(64, 64, kernel_size=(13, 3), dilation=(1, 1), bias=False)),
                    ("norm2", nn.BatchNorm2d(64)),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("conv3", nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
                    ("norm3", nn.BatchNorm2d(64)),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("pool3", nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
                    ("conv4", nn.Conv2d(64, 128, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
                    ("norm4", nn.BatchNorm2d(128)),
                    ("relu4", nn.ReLU(inplace=True)),
                    ("conv5", nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
                    ("norm5", nn.BatchNorm2d(128)),
                    ("relu5", nn.ReLU(inplace=True)),
                    ("pool5", nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
                    ("conv6", nn.Conv2d(128, 256, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
                    ("norm6", nn.BatchNorm2d(256)),
                    ("relu6", nn.ReLU(inplace=True)),
                    ("conv7", nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
                    ("norm7", nn.BatchNorm2d(256)),
                    ("relu7", nn.ReLU(inplace=True)),
                    ("pool7", nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),
                    ("conv8", nn.Conv2d(256, 512, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
                    ("norm8", nn.BatchNorm2d(512)),
                    ("relu8", nn.ReLU(inplace=True)),
                    ("conv9", nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
                    ("norm9", nn.BatchNorm2d(512)),
                    ("relu9", nn.ReLU(inplace=True)),
                ]
            )
        )
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc0 = nn.Linear(512, 300)
        self.fc1 = nn.Linear(300, 10000)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CQTNet and return logits plus embedding."""
        batch_size = x.size()[0]
        x = self.features(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        feature = self.fc0(x)
        return self.fc1(feature), feature


def init_craft_weights(modules: nn.Module | list[nn.Module]) -> None:
    """Initialize CRAFT modules with the source initialization scheme."""
    for module in modules:
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0, 0.01)
            module.bias.data.zero_()


class CraftVGG16BN(nn.Module):
    """VGG16-BN feature extractor from CRAFT."""

    def __init__(self) -> None:
        """Initialize the CRAFT VGG feature slices without pretrained downloads."""
        super().__init__()
        vgg_features = models.vgg16_bn(weights=None).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(12):
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_features[x])
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )
        init_craft_weights(self.slice5.modules())

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return CRAFT VGG skip features."""
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        vgg_outputs = namedtuple("VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])
        return vgg_outputs(h, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)


class CraftDoubleConv(nn.Module):
    """CRAFT decoder convolution block."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int) -> None:
        """Initialize the CRAFT decoder block."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CRAFT decoder block."""
        return self.conv(x)


class CRAFT(nn.Module):
    """CRAFT text detector."""

    def __init__(self) -> None:
        """Initialize CRAFT with random VGG weights."""
        super().__init__()
        self.basenet = CraftVGG16BN()
        self.upconv1 = CraftDoubleConv(1024, 512, 256)
        self.upconv2 = CraftDoubleConv(512, 256, 128)
        self.upconv3 = CraftDoubleConv(256, 128, 64)
        self.upconv4 = CraftDoubleConv(128, 64, 32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )
        init_craft_weights(self.upconv1.modules())
        init_craft_weights(self.upconv2.modules())
        init_craft_weights(self.upconv3.modules())
        init_craft_weights(self.upconv4.modules())
        init_craft_weights(self.conv_cls.modules())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CRAFT and return score maps plus decoder features."""
        sources = self.basenet(x)
        y = self.upconv1(torch.cat([sources[0], sources[1]], dim=1))
        y = F.interpolate(y, size=sources[2].size()[2:], mode="bilinear", align_corners=False)
        y = self.upconv2(torch.cat([y, sources[2]], dim=1))
        y = F.interpolate(y, size=sources[3].size()[2:], mode="bilinear", align_corners=False)
        y = self.upconv3(torch.cat([y, sources[3]], dim=1))
        y = F.interpolate(y, size=sources[4].size()[2:], mode="bilinear", align_corners=False)
        feature = self.upconv4(torch.cat([y, sources[4]], dim=1))
        y = self.conv_cls(feature)
        return y.permute(0, 2, 3, 1), feature


class CSRNet(nn.Module):
    """CSRNet crowd density model."""

    def __init__(self) -> None:
        """Initialize CSRNet without pretrained VGG transfer."""
        super().__init__()
        self.seen = 0
        self.frontend_feat: list[int | str] = [
            64,
            64,
            "M",
            128,
            128,
            "M",
            256,
            256,
            256,
            "M",
            512,
            512,
            512,
        ]
        self.backend_feat: list[int | str] = [512, 512, 512, 256, 128, 64]
        self.frontend = make_csr_layers(self.frontend_feat)
        self.backend = make_csr_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run CSRNet density regression."""
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

    def _initialize_weights(self) -> None:
        """Initialize CSRNet convolution and normalization weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def make_csr_layers(
    cfg: list[int | str],
    in_channels: int = 3,
    batch_norm: bool = False,
    dilation: bool = False,
) -> nn.Sequential:
    """Construct CSRNet VGG-style layers."""
    d_rate = 2 if dilation else 1
    layers: list[nn.Module] = []
    for value in cfg:
        if value == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            out_channels = int(value)
            conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=d_rate, dilation=d_rate
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = out_channels
    return nn.Sequential(*layers)


class CutPasteNet(nn.Module):
    """CutPaste classifier with ResNet encoder and projection head."""

    def __init__(self) -> None:
        """Initialize CutPasteNet using the source ResNet18 setting."""
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        dims = [512, 512, 512, 512, 512, 512, 512, 512, 128]
        proj_layers: list[nn.Module] = []
        for dim in dims[:-1]:
            proj_layers.append(nn.Linear(dim, dim, bias=False))
            proj_layers.append(nn.BatchNorm1d(dim))
            proj_layers.append(nn.ReLU(inplace=True))
        proj_layers.append(nn.Linear(dims[-2], dims[-1], bias=True))
        self.head = nn.Sequential(*proj_layers)
        self.out = nn.Linear(dims[-1], 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CutPaste classification and return logits plus embedding."""
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return logits, embeds


def build_cosmic_conn() -> CosmicConnUNet:
    """Build a tiny Cosmic-CoNN U-Net."""
    return CosmicConnUNet(
        1, 1, hidden=4, norm="group", norm_setting=(1, 0, False), down_type="maxpool", deeper=False
    )


def example_input_cosmic_conn() -> torch.Tensor:
    """Return an example Cosmic-CoNN image tensor."""
    return torch.randn(1, 1, 32, 32)


def build_countception() -> ModelCountception:
    """Build Count-ception."""
    return ModelCountception(inplanes=3, outplanes=1)


def example_input_countception() -> torch.Tensor:
    """Return an example Count-ception image tensor."""
    return torch.randn(1, 3, 64, 64)


def build_cqtnet() -> CQTNet:
    """Build CQTNet."""
    return CQTNet()


def example_input_cqtnet() -> torch.Tensor:
    """Return an example CQT spectrogram tensor."""
    return torch.randn(1, 1, 84, 192)


def build_craft() -> CRAFT:
    """Build CRAFT."""
    return CRAFT()


def example_input_craft() -> torch.Tensor:
    """Return an example CRAFT image tensor."""
    return torch.randn(1, 3, 64, 64)


def build_csrnet() -> CSRNet:
    """Build CSRNet."""
    return CSRNet()


def example_input_csrnet() -> torch.Tensor:
    """Return an example CSRNet image tensor."""
    return torch.randn(1, 3, 64, 64)


def build_cutpaste() -> CutPasteNet:
    """Build CutPasteNet."""
    return CutPasteNet()


def example_input_cutpaste() -> torch.Tensor:
    """Return an example CutPaste image tensor."""
    return torch.randn(2, 3, 64, 64)


def build_coral() -> CoralResNet:
    """Build the CORAL residual MLP."""
    return CoralResNet(input_dim=8, hidden_dim=16, output_dim=4, depth=2)


def example_input_coral() -> torch.Tensor:
    """Return an example CORAL coordinate/code tensor."""
    return torch.randn(2, 5, 8)


MENAGERIE_ENTRIES = [
    ("CORAL", "build_coral", "example_input_coral", 2023, "CV10-296"),
    ("Cosmic-CoNN", "build_cosmic_conn", "example_input_cosmic_conn", 2022, "CV10-302"),
    ("Count-ception", "build_countception", "example_input_countception", 2017, "CV10-307"),
    (
        "CQTNet for Cover Song Identification",
        "build_cqtnet",
        "example_input_cqtnet",
        2020,
        "CV10-313",
    ),
    ("CRAFT text detection", "build_craft", "example_input_craft", 2019, "CV10-318"),
    ("CSRNet", "build_csrnet", "example_input_csrnet", 2018, "CV10-324"),
    ("CSRNet-agri usage", "build_csrnet", "example_input_csrnet", 2018, "CV10-325"),
    ("CutPaste", "build_cutpaste", "example_input_cutpaste", 2021, "CV10-330"),
]

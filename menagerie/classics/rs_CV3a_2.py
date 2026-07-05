# SOURCE: vendored from ReaFly/ACSNet @ 9762736
"""CV3a vendored ACSNet staging module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

MENAGERIE_ZOO = "vendored-pytorch"


class ConvBlock(nn.Module):
    """Convolution, batch normalization, and ReLU block from ACSNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        """Initialize the convolution block.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size
            Convolution kernel size.
        stride
            Convolution stride.
        padding
            Convolution padding.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class DecoderBlock(nn.Module):
    """ACSNet decoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        """Initialize the decoder block.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size
            Convolution kernel size.
        stride
            Convolution stride.
        padding
            Convolution padding.
        """
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size, stride, padding)
        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size, stride, padding)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode and upsample a feature map.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        torch.Tensor
            Upsampled output feature map.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return self.upsample(x)


class SideoutBlock(nn.Module):
    """ACSNet side-output block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        """Initialize a side-output block.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size
            Convolution kernel size.
        stride
            Convolution stride.
        padding
            Convolution padding.
        """
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size, stride, padding)
        self.dropout = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply side-output projection.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        torch.Tensor
            Side-output logits.
        """
        x = self.conv1(x)
        x = self.dropout(x)
        return self.conv2(x)


class LCA(nn.Module):
    """Local Context Attention module."""

    def __init__(self) -> None:
        """Initialize the attention module."""
        super().__init__()

    def forward(self, x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Apply local context attention.

        Parameters
        ----------
        x
            Encoder feature map.
        pred
            Side-output prediction map.

        Returns
        -------
        torch.Tensor
            Locally attended feature map.
        """
        residual = x
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        att = 1 - (dist / 0.5)
        att_x = x * att
        return att_x + residual


class GCM(nn.Module):
    """Global Context Module."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize global context branches.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of channels in each context branch.
        """
        super().__init__()
        pool_size = [1, 3, 5]
        out_channel_list = [256, 128, 64, 64]
        upsample_scale = [2, 4, 8, 16]
        gc_list = []
        gc_out_list = []
        for ps in pool_size:
            gc_list.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_channels, out_channels, 1, 1),
                    nn.ReLU(inplace=True),
                )
            )
        gc_list.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True),
                NonLocalBlock(out_channels),
            )
        )
        self.GCmodule = nn.ModuleList(gc_list)
        for i in range(4):
            gc_out_list.append(
                nn.Sequential(
                    nn.Conv2d(out_channels * 4, out_channel_list[i], 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=upsample_scale[i], mode="bilinear"),
                )
            )
        self.GCoutmodel = nn.ModuleList(gc_out_list)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Compute global context tensors for each decoder scale.

        Parameters
        ----------
        x
            Deep encoder feature map.

        Returns
        -------
        list[torch.Tensor]
            Context tensors ordered from deepest to shallowest decoder scale.
        """
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
            global_context.append(
                F.interpolate(self.GCmodule[i](x), xsize, mode="bilinear", align_corners=True)
            )
        global_context.append(self.GCmodule[-1](x))
        global_context_tensor = torch.cat(global_context, dim=1)
        return [module(global_context_tensor) for module in self.GCoutmodel]


class ASM(nn.Module):
    """Adaptive Selection Module."""

    def __init__(self, in_channels: int, all_channels: int) -> None:
        """Initialize the adaptive selection block.

        Parameters
        ----------
        in_channels
            Feature channels entering the non-local block.
        all_channels
            Concatenated channels for squeeze-excitation.
        """
        super().__init__()
        self.non_local = NonLocalBlock(in_channels)
        self.selayer = SELayer(all_channels)

    def forward(self, lc: torch.Tensor, fuse: torch.Tensor, gc: torch.Tensor) -> torch.Tensor:
        """Fuse local, decoder, and global context tensors.

        Parameters
        ----------
        lc
            Local-context tensor.
        fuse
            Decoder tensor.
        gc
            Global-context tensor.

        Returns
        -------
        torch.Tensor
            Fused tensor.
        """
        fuse = self.non_local(fuse)
        fuse = torch.cat([lc, fuse, gc], dim=1)
        return self.selayer(fuse)


class SELayer(nn.Module):
    """Squeeze-and-excitation layer."""

    def __init__(self, channel: int, reduction: int = 16) -> None:
        """Initialize squeeze-and-excitation projection.

        Parameters
        ----------
        channel
            Number of input channels.
        reduction
            Channel reduction factor.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel recalibration.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        torch.Tensor
            Recalibrated feature map.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class NonLocalBlock(nn.Module):
    """Non-local block used by ACSNet."""

    def __init__(
        self,
        in_channels: int,
        inter_channels: int | None = None,
        sub_sample: bool = True,
        bn_layer: bool = True,
    ) -> None:
        """Initialize a non-local attention block.

        Parameters
        ----------
        in_channels
            Number of input channels.
        inter_channels
            Internal attention width.
        sub_sample
            Whether to max-pool key/value projections.
        bn_layer
            Whether to use batch normalization in the output projection.
        """
        super().__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        self.g = nn.Conv2d(
            self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0
        )
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(
                    self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0
                ),
                nn.BatchNorm2d(self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(
                self.inter_channels,
                self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(
            self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0
        )
        self.phi = nn.Conv2d(
            self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0
        )
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply non-local attention.

        Parameters
        ----------
        x
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map with residual non-local response.
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_c = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_c, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        w_y = self.W(y)
        return w_y + x


class ACSNet(nn.Module):
    """Adaptive Context Selection Network."""

    def __init__(self, num_classes: int) -> None:
        """Initialize ACSNet.

        Parameters
        ----------
        num_classes
            Number of output segmentation classes.
        """
        super().__init__()
        resnet = models.resnet34(weights=None)
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=192, out_channels=64)
        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, 1),
        )
        self.sideout2 = SideoutBlock(64, 1)
        self.sideout3 = SideoutBlock(128, 1)
        self.sideout4 = SideoutBlock(256, 1)
        self.sideout5 = SideoutBlock(512, 1)
        self.lca1 = LCA()
        self.lca2 = LCA()
        self.lca3 = LCA()
        self.lca4 = LCA()
        self.gcm = GCM(512, 64)
        self.asm4 = ASM(512, 1024)
        self.asm3 = ASM(256, 512)
        self.asm2 = ASM(128, 256)
        self.asm1 = ASM(64, 192)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run ACSNet segmentation.

        Parameters
        ----------
        x
            Input image tensor.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Main and side-output sigmoid predictions.
        """
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        global_contexts = self.gcm(e5)
        d5 = self.decoder5(e5)
        out5 = self.sideout5(d5)
        lc4 = self.lca4(e4, out5)
        comb4 = self.asm4(lc4, d5, global_contexts[0])
        d4 = self.decoder4(comb4)
        out4 = self.sideout4(d4)
        lc3 = self.lca3(e3, out4)
        comb3 = self.asm3(lc3, d4, global_contexts[1])
        d3 = self.decoder3(comb3)
        out3 = self.sideout3(d3)
        lc2 = self.lca2(e2, out3)
        comb2 = self.asm2(lc2, d3, global_contexts[2])
        d2 = self.decoder2(comb2)
        out2 = self.sideout2(d2)
        lc1 = self.lca1(e1, out2)
        comb1 = self.asm1(lc1, d2, global_contexts[3])
        d1 = self.decoder1(comb1)
        out1 = self.outconv(d1)
        return (
            torch.sigmoid(out1),
            torch.sigmoid(out2),
            torch.sigmoid(out3),
            torch.sigmoid(out4),
            torch.sigmoid(out5),
        )


def build_acsnet() -> ACSNet:
    """Build ACSNet with one output class.

    Returns
    -------
    ACSNet
        Randomly initialized ACSNet architecture.
    """
    return ACSNet(num_classes=1)


def example_input_acsnet() -> torch.Tensor:
    """Create an example ACSNet image input.

    Returns
    -------
    torch.Tensor
        Example RGB image tensor.
    """
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [("ACSNet", "build_acsnet", "example_input_acsnet", 2020, "CV3a-11")]

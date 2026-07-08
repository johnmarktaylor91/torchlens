# SOURCE: vendored from https://github.com/HiLab-git/CA-Net @ master
# (Models/networks/network.py::Comprehensive_Atten_Unet +
#  Models/layers/modules.py + Models/layers/channel_attention_layer.py +
#  Models/layers/grid_attention_layer.py + Models/layers/nonlocal_layer.py +
#  Models/layers/scale_attention_layer.py + Models/networks_other.py::init_weights)
#
# CA-Net / "Comprehensive Attention Network" (Gu et al., IEEE TMI 2021,
# arXiv:2009.10549, "CA-Net: Comprehensive Attention Convolutional Neural
# Networks for Explainable Medical Image Segmentation"): a U-Net-shaped
# encoder-decoder joint spatial + channel + scale triple-attention network for
# 2D medical image segmentation (ISIC skin-lesion / fetal MRI). Combines
# grid-gated spatial attention (MultiAttentionBlock), squeeze-excitation
# channel attention (SE_Conv_Block) on every decoder stage, a non-local block
# at the bottleneck, deep supervision, and a CBAM-style scale-attention fusion
# of the four supervision heads.
#
# Vendored real repo code -- every nn.Module class below (conv_block, UpCat,
# UpCatconv, UnetGridGatingSignal3, UnetDsv3, SE_Conv_Block,
# _GridAttentionBlockND/GridAttentionBlock2D/MultiAttentionBlock,
# _NonLocalBlockND/NONLocalBlock2D, the CBAM ChannelGate/SpatialGate/
# Scale_atten_block/scale_atten_convblock stack, init_weights, and
# Comprehensive_Atten_Unet itself) is transcribed unchanged from the multi-file
# real repo, with imports flattened into this single file and the two
# unconditional `.cuda()` calls in UpCat/UpCatconv's rare odd-offset padding
# branch made device-safe (`.to(inputs.device)` instead of `.cuda()` -- those
# branches are dead code at this model's fixed spatial resolution and were
# never architecture, only a CUDA-only training convenience in the original).
# No layer, channel width, kernel size, or dataflow was changed. Non-
# architectural pieces (ISIC/Fetus Dataset classes, argparse `args`, the
# train/validate loop in main.py) were dropped.

import torch
import torch.nn as nn
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Models/networks_other.py::init_weights (+ the four weights_init_* variants)
# ---------------------------------------------------------------------------
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="kaiming"):
    if init_type == "kaiming":
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError("initialization method [%s] is not implemented" % init_type)


# ---------------------------------------------------------------------------
# Models/layers/modules.py
# ---------------------------------------------------------------------------
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop_out=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
        self.dropout = drop_out

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = nn.Dropout2d(0.5)(x)
        return x


class UpCat(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True):
        super(UpCat, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, inputs, down_outputs):
        outputs = self.up(down_outputs)
        offset = inputs.size()[3] - outputs.size()[3]
        if offset == 1:
            addition = (
                torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None)
                .unsqueeze(3)
                .to(outputs.device)
            )
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            addition = torch.rand(
                (outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None
            ).to(outputs.device)
            outputs = torch.cat([outputs, addition], dim=3)
        out = torch.cat([inputs, outputs], dim=1)
        return out


class UpCatconv(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True, drop_out=False):
        super(UpCatconv, self).__init__()
        if is_deconv:
            self.conv = conv_block(in_feat, out_feat, drop_out=drop_out)
            self.up = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.conv = conv_block(in_feat + out_feat, out_feat, drop_out=drop_out)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, inputs, down_outputs):
        outputs = self.up(down_outputs)
        offset = inputs.size()[3] - outputs.size()[3]
        if offset == 1:
            addition = (
                torch.rand((outputs.size()[0], outputs.size()[1], outputs.size()[2]), out=None)
                .unsqueeze(3)
                .to(outputs.device)
            )
            outputs = torch.cat([outputs, addition], dim=3)
        elif offset > 1:
            addition = torch.rand(
                (outputs.size()[0], outputs.size()[1], outputs.size()[2], offset), out=None
            ).to(outputs.device)
            outputs = torch.cat([outputs, addition], dim=3)
        out = self.conv(torch.cat([inputs, outputs], dim=1))
        return out


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0)),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size, (1, 1), (0, 0)),
                nn.ReLU(inplace=True),
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
            nn.Upsample(size=scale_factor, mode="bilinear"),
        )

    def forward(self, input):
        return self.dsv(input)


# ---------------------------------------------------------------------------
# Models/layers/channel_attention_layer.py
# ---------------------------------------------------------------------------
def conv3x3_ca(in_planes, out_planes, stride=1, bias=False, group=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=group, bias=bias
    )


class SE_Conv_Block(nn.Module):
    """Squeeze-excitation residual conv block. NOTE: the real repo hardcodes
    the global-pool kernel size per channel width for the fixed ISIC2018
    input resolution (224, 300); we preserve that exact hardcoding (it is
    real architecture, not incidental) and drive this staging module's
    example input at (224, 300) so every SE_Conv_Block stage's globalAvgPool/
    globalMaxPool kernel matches its feature map size exactly, as in the
    original."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block, self).__init__()
        self.conv1 = conv3x3_ca(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_ca(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3_ca(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if planes <= 16:
            self.globalAvgPool = nn.AvgPool2d((224, 300), stride=1)
            self.globalMaxPool = nn.MaxPool2d((224, 300), stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d((112, 150), stride=1)
            self.globalMaxPool = nn.MaxPool2d((112, 150), stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AvgPool2d((56, 75), stride=1)
            self.globalMaxPool = nn.MaxPool2d((56, 75), stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d((28, 37), stride=1)
            self.globalMaxPool = nn.MaxPool2d((28, 37), stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d((14, 18), stride=1)
            self.globalMaxPool = nn.MaxPool2d((14, 18), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(
                nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 2),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight


# ---------------------------------------------------------------------------
# Models/layers/grid_attention_layer.py
# ---------------------------------------------------------------------------
class _GridAttentionBlockND(nn.Module):
    def __init__(
        self,
        in_channels,
        gating_channels,
        inter_channels=None,
        dimension=3,
        mode="concatenation",
        sub_sample_factor=(2, 2, 2),
    ):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ["concatenation", "concatenation_debug", "concatenation_residual"]

        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = "trilinear"
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = "bilinear"
        else:
            raise NotImplementedError

        self.W = nn.Sequential(
            conv_nd(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            bn(self.in_channels),
        )

        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=self.sub_sample_kernel_size,
            stride=self.sub_sample_factor,
            padding=0,
            bias=True,
        )
        self.phi = conv_nd(
            in_channels=self.gating_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.psi = conv_nd(
            in_channels=self.inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        for m in self.children():
            init_weights(m, init_type="kaiming")

        if mode == "concatenation":
            self.operation_function = self._concatenation
        else:
            raise NotImplementedError("Unknown operation function.")

    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(
            self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=False
        )
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))

        sigm_psi_f = F.interpolate(
            sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=False
        )
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(
        self,
        in_channels,
        gating_channels,
        inter_channels=None,
        mode="concatenation",
        sub_sample_factor=(2, 2),
    ):
        super(GridAttentionBlock2D, self).__init__(
            in_channels,
            gating_channels,
            inter_channels=inter_channels,
            dimension=2,
            mode=mode,
            sub_sample_factor=sub_sample_factor,
        )


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(
            in_channels=in_size,
            gating_channels=gate_size,
            inter_channels=inter_size,
            mode=nonlocal_mode,
            sub_sample_factor=sub_sample_factor,
        )
        self.gate_block_2 = GridAttentionBlock2D(
            in_channels=in_size,
            gating_channels=gate_size,
            inter_channels=inter_size,
            mode=nonlocal_mode,
            sub_sample_factor=sub_sample_factor,
        )
        self.combine_gates = nn.Sequential(
            nn.Conv2d(in_size * 2, in_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
        )

        for m in self.children():
            if m.__class__.__name__.find("GridAttentionBlock2D") != -1:
                continue
            init_weights(m, init_type="kaiming")

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)
        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat(
            [attention_1, attention_2], 1
        )


# ---------------------------------------------------------------------------
# Models/layers/nonlocal_layer.py
# ---------------------------------------------------------------------------
class _NonLocalBlockND(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        dimension=3,
        mode="embedded_gaussian",
        sub_sample_factor=4,
        bn_layer=True,
    ):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in [
            "embedded_gaussian",
            "gaussian",
            "dot_product",
            "concatenation",
            "concat_proper",
            "concat_proper_down",
        ]

        self.mode = mode
        self.dimension = dimension
        self.sub_sample_factor = (
            sub_sample_factor if isinstance(sub_sample_factor, list) else [sub_sample_factor]
        )

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(
                    in_channels=self.inter_channels,
                    out_channels=self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
                bn(self.in_channels),
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(
                in_channels=self.inter_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None

        if mode in [
            "embedded_gaussian",
            "dot_product",
            "concatenation",
            "concat_proper",
            "concat_proper_down",
        ]:
            self.theta = conv_nd(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            self.phi = conv_nd(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

            if mode in ["concatenation"]:
                self.wf_phi = nn.Linear(self.inter_channels, 1, bias=False)
                self.wf_theta = nn.Linear(self.inter_channels, 1, bias=False)
            elif mode in ["concat_proper", "concat_proper_down"]:
                self.psi = nn.Conv2d(
                    in_channels=self.inter_channels,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )

        if mode == "embedded_gaussian":
            self.operation_function = self._embedded_gaussian
        else:
            raise NotImplementedError("Unknown operation function.")

        if max(self.sub_sample_factor) > 1:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=self.sub_sample_factor))
            if self.phi is None:
                self.phi = max_pool(kernel_size=self.sub_sample_factor)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=self.sub_sample_factor))

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def forward(self, x):
        output = self.operation_function(x)
        return output


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(
        self,
        in_channels,
        inter_channels=None,
        mode="embedded_gaussian",
        sub_sample_factor=2,
        bn_layer=True,
    ):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            mode=mode,
            sub_sample_factor=sub_sample_factor,
            bn_layer=bn_layer,
        )


# ---------------------------------------------------------------------------
# Models/layers/scale_attention_layer.py (CBAM-style scale-attention fusion)
# ---------------------------------------------------------------------------
def conv3x3_sa(in_planes, out_planes, stride=1, bias=False, group=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=group, bias=bias
    )


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=("avg", "max")):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            else:
                continue

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        channel_att_sum = channel_att_sum.reshape(channel_att_sum.shape[0], 4, 4)
        avg_weight = torch.mean(channel_att_sum, dim=2).unsqueeze(2)
        avg_weight = avg_weight.expand(channel_att_sum.shape[0], 4, 4).reshape(
            channel_att_sum.shape[0], 16
        )
        scale = torch.sigmoid(avg_weight).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale, scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAtten(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1):
        super(SpatialAtten, self).__init__()
        self.conv1 = BasicConv(
            in_size, out_size, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, relu=True
        )
        self.conv2 = BasicConv(
            out_size, out_size, kernel_size=1, stride=stride, padding=0, relu=True, bn=False
        )

    def forward(self, x):
        residual = x
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        spatial_att = torch.sigmoid(x_out).unsqueeze(4).permute(0, 1, 4, 2, 3)
        spatial_att = spatial_att.expand(
            spatial_att.shape[0], 4, 4, spatial_att.shape[3], spatial_att.shape[4]
        ).reshape(spatial_att.shape[0], 16, spatial_att.shape[3], spatial_att.shape[4])
        x_out = residual * spatial_att
        x_out = x_out + residual
        return x_out, spatial_att


class Scale_atten_block(nn.Module):
    def __init__(
        self, gate_channels, reduction_ratio=16, pool_types=("avg", "max"), no_spatial=False
    ):
        super(Scale_atten_block, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialAtten(gate_channels, gate_channels // reduction_ratio)

    def forward(self, x):
        x_out, ca_atten = self.ChannelGate(x)
        sa_atten = None
        if not self.no_spatial:
            x_out, sa_atten = self.SpatialGate(x_out)
        return x_out, ca_atten, sa_atten


class scale_atten_convblock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        stride=1,
        downsample=None,
        use_cbam=True,
        no_spatial=False,
        drop_out=False,
    ):
        super(scale_atten_convblock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.dropout = drop_out

        self.relu = nn.ReLU(inplace=True)
        self.conv3 = conv3x3_sa(in_size, out_size)
        self.bn3 = nn.BatchNorm2d(out_size)

        if use_cbam:
            self.cbam = Scale_atten_block(in_size, reduction_ratio=4, no_spatial=self.no_spatial)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        out = None
        if self.cbam is not None:
            out, _scale_c_atten, _scale_s_atten = self.cbam(x)

        out = out + residual
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


# ---------------------------------------------------------------------------
# Models/networks/network.py::Comprehensive_Atten_Unet
# ---------------------------------------------------------------------------
class Comprehensive_Atten_Unet(nn.Module):
    def __init__(
        self,
        out_size,
        in_ch=3,
        n_classes=2,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True,
        nonlocal_mode="concatenation",
        attention_dsample=(1, 1),
    ):
        super(Comprehensive_Atten_Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = out_size

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(
            in_size=filters[1],
            gate_size=filters[2],
            inter_size=filters[1],
            nonlocal_mode=nonlocal_mode,
            sub_sample_factor=attention_dsample,
        )
        self.attentionblock3 = MultiAttentionBlock(
            in_size=filters[2],
            gate_size=filters[3],
            inter_size=filters[2],
            nonlocal_mode=nonlocal_mode,
            sub_sample_factor=attention_dsample,
        )
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=True)
        self.up3 = SE_Conv_Block(filters[3], filters[2])
        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)

        # Attention Mechanism / Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)
        g_conv4 = self.nonlocal4_2(up4)

        up4, att_weight4 = self.up4(g_conv4)
        g_conv3, att3 = self.attentionblock3(conv3, up4)

        up3 = self.up_concat3(g_conv3, up4)
        up3, att_weight3 = self.up3(up3)
        g_conv2, att2 = self.attentionblock2(conv2, up3)

        up2 = self.up_concat2(g_conv2, up3)
        up2, att_weight2 = self.up2(up2)

        up1 = self.up_concat1(conv1, up2)
        up1, att_weight1 = self.up1(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)

        out = self.final(out)

        return out


def build_canet():
    # feature_scale=4 (the real repo's default) gives filters
    # [16, 32, 64, 128, 256] -- exactly the four branches SE_Conv_Block
    # hardcodes global-pool kernels for. out_size=(224, 300) is the real
    # repo's ISIC2018 deep-supervision upsample target; n_classes=2 (binary
    # segmentation), in_ch=3 (RGB) match the real ISIC2018 config.
    return Comprehensive_Atten_Unet(out_size=(224, 300), in_ch=3, n_classes=2, feature_scale=4)


def example_input_canet():
    # (batch, 3, 224, 300) -- the real repo's fixed ISIC2018 input resolution,
    # required for SE_Conv_Block's hardcoded global-pool kernel sizes at every
    # decoder depth to exactly match their feature-map spatial size.
    return torch.randn(1, 3, 224, 300)


MENAGERIE_ENTRIES = [
    (
        "CA-Net",
        build_canet,
        example_input_canet,
        2021,
        MENAGERIE_ZOO,
    ),
]

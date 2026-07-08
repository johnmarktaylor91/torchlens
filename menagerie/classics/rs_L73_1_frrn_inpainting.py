# SOURCE: vendored from ZongyuGuo/Inpainting_FRRN @ master
# Files: src/networks.py, src/partialconv2d.py
# https://github.com/ZongyuGuo/Inpainting_FRRN
#
# Minimal changes from the original source:
#   - Merged networks.py and partialconv2d.py into one file (relative
#     `from partialconv2d import PartialConv2d` import resolved by inlining
#     PartialConv2d directly above the classes that use it).
#   - Dropped `src/models.py` (training-loop wrapper: losses/optimizers/
#     checkpoint I/O) and `src/loss.py` (AdversarialLoss/StyleLoss) since
#     they are training-only scaffolding, not architecture.
#   - No changes to any nn.Module architecture: FRRNet, FRRBlock, PConvLayer,
#     Discriminator, and PartialConv2d (itself a modified port of NVIDIA's
#     official partialconv, as declared in the source file's own header)
#     are reproduced verbatim from the source.
#
# Architecture (unmodified from source): "Progressive Image Inpainting with
# Full-Resolution Residual Network" (FRRN). The generator (FRRNet) stacks
# `block_num=16` FRRBlocks. Each FRRBlock has two parallel processing
# streams over a *full-resolution* branch (3 stacked partial convolutions
# at the input's native H x W, contributing fine detail) and a
# *downsample-then-upsample* branch (3 stride-2 partial convolutions down
# to 1/8 resolution, then 3 nearest-neighbor-upsample + partial-conv steps
# back to full resolution, contributing large receptive-field context).
# Every conv is `PartialConv2d` (renormalizes convolution output by the
# fraction of valid/unmasked pixels in each receptive field and updates
# the mask alongside the features, following Liu et al. "Image Inpainting
# for Irregular Holes Using Partial Convolutions"). The two branch outputs
# are mask-gated and averaged, then residually combined with the block's
# input. Blocks are chained end-to-end (mask_new from block 2i+1 feeds
# block 2i+2), progressively refining the inpainted region and its
# validity mask over depth_num=8 macro-iterations (16 FRRBlocks total).
# A PatchGAN-style Discriminator (5 spectral-normalized strided convs) is
# included for completeness (used only during adversarial training).

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        # whether the mask is multi-channel or not
        if "multi_channel" in kwargs:
            self.multi_channel = kwargs["multi_channel"]
            kwargs.pop("multi_channel")
        else:
            self.multi_channel = False

        if "return_mask" in kwargs:
            self.return_mask = kwargs["return_mask"]
            kwargs.pop("return_mask")
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(
                self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
            )
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = (
            self.weight_maskUpdater.shape[1]
            * self.weight_maskUpdater.shape[2]
            * self.weight_maskUpdater.shape[3]
        )

        self.last_size = (None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        if mask_in is not None or self.last_size != (input.data.shape[2], input.data.shape[3]):
            self.last_size = (input.data.shape[2], input.data.shape[3])

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(
                            input.data.shape[0],
                            input.data.shape[1],
                            input.data.shape[2],
                            input.data.shape[3],
                        ).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=1,
                )

                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return torch.mul(output, self.update_mask), self.update_mask
        else:
            return output


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type="normal", gain=0.02):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find("BatchNorm2d") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class FRRNet(BaseNetwork):
    def __init__(self, block_num=16, init_weights=True):
        super(FRRNet, self).__init__()
        self.block_num = block_num
        self.dilation_num = block_num // 2
        blocks = []
        for _ in range(self.block_num):
            blocks.append(FRRBlock())
        self.blocks = nn.ModuleList(blocks)

        if init_weights:
            self.init_weights()

    def forward(self, x, mask):
        mid_x = []
        mid_m = []

        mask_new = mask
        for index in range(self.dilation_num):
            x, _ = self.blocks[index * 2](x, mask_new, mask)
            x, mask_new = self.blocks[index * 2 + 1](x, mask_new, mask)
            mid_x.append(x)
            mid_m.append(mask_new)

        return x, mid_x, mid_m


class FRRBlock(nn.Module):
    def __init__(self):
        super(FRRBlock, self).__init__()
        self.full_conv1 = PConvLayer(3, 32, kernel_size=5, stride=1, padding=2, use_norm=False)
        self.full_conv2 = PConvLayer(32, 32, kernel_size=5, stride=1, padding=2, use_norm=False)
        self.full_conv3 = PConvLayer(32, 3, kernel_size=5, stride=1, padding=2, use_norm=False)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.branch_conv1 = PConvLayer(3, 64, kernel_size=3, stride=2, padding=1, use_norm=False)
        self.branch_conv2 = PConvLayer(64, 96, kernel_size=3, stride=2, padding=1)
        self.branch_conv3 = PConvLayer(96, 128, kernel_size=3, stride=2, padding=1)
        self.branch_conv4 = PConvLayer(128, 96, kernel_size=3, stride=1, padding=1, act="LeakyReLU")
        self.branch_conv5 = PConvLayer(96, 64, kernel_size=3, stride=1, padding=1, act="LeakyReLU")
        self.branch_conv6 = PConvLayer(64, 3, kernel_size=3, stride=1, padding=1, act="Tanh")

    def forward(self, input, mask, mask_ori):
        x = input
        out_f, mask_f = self.full_conv1(x, mask)
        out_f, mask_f = self.full_conv2(out_f, mask_f)
        out_f, mask_f = self.full_conv3(out_f, mask_f)

        out_b, mask_b = self.branch_conv1(x, mask)
        out_b, mask_b = self.branch_conv2(out_b, mask_b)
        out_b, mask_b = self.branch_conv3(out_b, mask_b)

        out_b = self.upsample(out_b)
        mask_b = self.upsample(mask_b)
        out_b, mask_b = self.branch_conv4(out_b, mask_b)
        out_b = self.upsample(out_b)
        mask_b = self.upsample(mask_b)
        out_b, mask_b = self.branch_conv5(out_b, mask_b)
        out_b = self.upsample(out_b)
        mask_b = self.upsample(mask_b)
        out_b, mask_b = self.branch_conv6(out_b, mask_b)

        mask_new = mask_f * mask_b
        out = (out_f * mask_new + out_b * mask_new) / 2 * (1 - mask_ori) + input
        return out, mask_new


class PConvLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, act="ReLU", use_norm=True
    ):
        super(PConvLayer, self).__init__()
        self.conv = PartialConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_mask=True,
        )
        self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        self.use_norm = use_norm
        if act == "ReLU":
            self.act = nn.ReLU(True)
        elif act == "LeakyReLU":
            self.act = nn.LeakyReLU(0.2, True)
        elif act == "Tanh":
            self.act = nn.Tanh()

    def forward(self, x, mask):
        x, mask_update = self.conv(x, mask)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)
        return x, mask_update


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1,
                    kernel_size=4,
                    stride=1,
                    padding=1,
                    bias=not use_spectral_norm,
                ),
                use_spectral_norm,
            ),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


def build_frrn_inpainting():
    # Small block_num for a fast trace; architecture per block is unchanged.
    return FRRNet(block_num=4, init_weights=True)


def example_input_frrn_inpainting():
    image = torch.rand(1, 3, 64, 64)
    mask = torch.ones(1, 1, 64, 64)
    mask[:, :, 20:40, 20:40] = 0.0
    return (image, mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "FRRN Inpainting",
        "build_frrn_inpainting",
        "example_input_frrn_inpainting",
        2019,
        "vendored",
    ),
]

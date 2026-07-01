# SOURCE: vendored from VITA-Group/EnlightenGAN @ b0349848f0cd1e52317baa04e09ac32a2ae771d6
# https://raw.githubusercontent.com/VITA-Group/EnlightenGAN/b0349848f0cd1e52317baa04e09ac32a2ae771d6/models/networks.py
#
# Yifan Jiang et al. 2021 "EnlightenGAN: Deep Light Enhancement without Paired
# Supervision" (IEEE TIP) -- official PyTorch implementation. `Unet_resize_conv`
# is the paper's actual generator (`which_model_netG='sid_unet_resize'` in
# `scripts/script.py`, the repo's own train/predict entrypoint): a 5-level
# attention-guided U-Net that (1) concatenates a grayscale illumination map
# onto the RGB input (`self_attention=True`), (2) multiplies each encoder
# skip-connection feature map by a downsampled copy of that grayscale map
# ("self-regularized attention" -- the paper's core mechanism for focusing
# enhancement on dark regions), and (3) adds a global residual skip from the
# input image to the decoded output scaled by `--skip` ("skip connection" in
# the paper's ablation). Only plain `nn.Conv2d`/bilinear upsampling is used
# (no ConvTranspose), matching the maintainers' actual training script.
#
# `Unet_resize_conv` plus its two free-function dependencies (`pad_tensor`,
# `pad_tensor_back`) are copied verbatim. No architectural changes were made;
# only mechanical fixes for import isolation and instantiation:
#   - `from lib.nn import SynchronizedBatchNorm2d as SynBN2d` -> dropped;
#     the real training/predict invocation (`scripts/script.py`) never passes
#     `--syn_norm`, so `self.opt.syn_norm` is always falsy at runtime and the
#     `SynBN2d` branch is dead code on the actual code path used here (plain
#     `nn.BatchNorm2d` is used instead, exactly as the unmodified source does).
#   - the constructor takes an argparse `Namespace`-like `opt` object in the
#     original code; `_DFLT_OPT` below is a tiny stand-in carrying exactly the
#     flag values the real `scripts/script.py` passes for train/predict
#     (`self_attention=True`, `use_norm=1`, `use_avgpool=0`, `syn_norm=False`,
#     `tanh=False`, `linear=False`, `skip=1`, `times_residual=True`,
#     `linear_add=False`, `latent_threshold=False`, `latent_norm=False`).
# All imports (`torch`, `torch.nn`, `torch.nn.functional`) are base libs
# already installed.

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:
        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, "width cant divided by stride"
    assert height % divide == 0, "height cant divided by stride"

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top : height - pad_bottom, pad_left : width - pad_right]


class _DfltOpt:
    """Stand-in for the argparse Namespace `opt`, carrying exactly the flag
    values the repo's own `scripts/script.py` passes to `predict.py` /
    `train.py` (--which_model_netG sid_unet_resize --skip 1 --self_attention
    --use_norm 1 --times_residual), i.e. the actual EnlightenGAN configuration.
    `skip` is included because the real `Unet_resize_conv.forward` reads
    `self.opt.skip` (a float scale factor) in addition to the boolean
    `self.skip` passed separately to `__init__` -- both come from the same
    parsed `opt` Namespace in the original `define_G(..., skip=opt.skip, opt=opt)`
    call, so this stand-in mirrors that by carrying both."""

    self_attention = True
    use_norm = 1
    use_avgpool = 0
    syn_norm = False
    tanh = False
    linear = False
    skip = 1.0
    times_residual = True
    linear_add = False
    latent_threshold = False
    latent_norm = False


class Unet_resize_conv(nn.Module):
    def __init__(self, opt, skip):
        super(Unet_resize_conv, self).__init__()

        self.opt = opt
        self.skip = skip
        p = 1
        if opt.self_attention:
            self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
            self.downsample_1 = nn.MaxPool2d(2)
            self.downsample_2 = nn.MaxPool2d(2)
            self.downsample_3 = nn.MaxPool2d(2)
            self.downsample_4 = nn.MaxPool2d(2)
        else:
            self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_2 = nn.BatchNorm2d(512)

        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_1 = nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_2 = nn.BatchNorm2d(256)

        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_2 = nn.BatchNorm2d(128)

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_2 = nn.BatchNorm2d(64)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 3, 1)
        if self.opt.tanh:
            self.tanh = nn.Tanh()

    def depth_to_space(self, input, block_size):
        block_size_sq = block_size * block_size
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / block_size_sq)
        s_width = int(d_width * block_size)
        s_height = int(d_height * block_size)
        t_1 = output.resize(batch_size, d_height, d_width, block_size_sq, s_depth)
        spl = t_1.split(block_size, 3)
        stack = [t_t.resize(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = (
            torch.stack(stack, 0)
            .transpose(0, 1)
            .permute(0, 2, 1, 3, 4)
            .resize(batch_size, s_height, s_width, s_depth)
        )
        output = output.permute(0, 3, 1, 2)
        return output

    def forward(self, input, gray):
        flag = 0
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)
        if self.opt.self_attention:
            gray_2 = self.downsample_1(gray)
            gray_3 = self.downsample_2(gray_2)
            gray_4 = self.downsample_3(gray_3)
            gray_5 = self.downsample_4(gray_4)
        if self.opt.use_norm == 1:
            if self.opt.self_attention:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1))))
            else:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
            x = self.max_pool1(conv1)

            x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
            conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
            x = self.max_pool2(conv2)

            x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
            conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
            x = self.max_pool3(conv3)

            x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
            conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
            x = self.max_pool4(conv4)

            x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
            x = x * gray_5 if self.opt.self_attention else x
            conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))

            conv5 = F.interpolate(conv5, scale_factor=2, mode="bilinear")
            conv4 = conv4 * gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
            conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

            conv6 = F.interpolate(conv6, scale_factor=2, mode="bilinear")
            conv3 = conv3 * gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
            conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

            conv7 = F.interpolate(conv7, scale_factor=2, mode="bilinear")
            conv2 = conv2 * gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
            conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

            conv8 = F.interpolate(conv8, scale_factor=2, mode="bilinear")
            conv1 = conv1 * gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent * gray

            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (
                            torch.max(latent) - torch.min(latent)
                        )
                    input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
                    output = latent + input * self.opt.skip
                    output = output * 2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (
                            torch.max(latent) - torch.min(latent)
                        )
                    output = latent + input * self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output / torch.max(torch.abs(output))

        elif self.opt.use_norm == 0:
            if self.opt.self_attention:
                x = self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1)))
            else:
                x = self.LReLU1_1(self.conv1_1(input))
            conv1 = self.LReLU1_2(self.conv1_2(x))
            x = self.max_pool1(conv1)

            x = self.LReLU2_1(self.conv2_1(x))
            conv2 = self.LReLU2_2(self.conv2_2(x))
            x = self.max_pool2(conv2)

            x = self.LReLU3_1(self.conv3_1(x))
            conv3 = self.LReLU3_2(self.conv3_2(x))
            x = self.max_pool3(conv3)

            x = self.LReLU4_1(self.conv4_1(x))
            conv4 = self.LReLU4_2(self.conv4_2(x))
            x = self.max_pool4(conv4)

            x = self.LReLU5_1(self.conv5_1(x))
            x = x * gray_5 if self.opt.self_attention else x
            conv5 = self.LReLU5_2(self.conv5_2(x))

            conv5 = F.interpolate(conv5, scale_factor=2, mode="bilinear")
            conv4 = conv4 * gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.LReLU6_1(self.conv6_1(up6))
            conv6 = self.LReLU6_2(self.conv6_2(x))

            conv6 = F.interpolate(conv6, scale_factor=2, mode="bilinear")
            conv3 = conv3 * gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.LReLU7_1(self.conv7_1(up7))
            conv7 = self.LReLU7_2(self.conv7_2(x))

            conv7 = F.interpolate(conv7, scale_factor=2, mode="bilinear")
            conv2 = conv2 * gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.LReLU8_1(self.conv8_1(up8))
            conv8 = self.LReLU8_2(self.conv8_2(x))

            conv8 = F.interpolate(conv8, scale_factor=2, mode="bilinear")
            conv1 = conv1 * gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.LReLU9_1(self.conv9_1(up9))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent * gray

            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (
                            torch.max(latent) - torch.min(latent)
                        )
                    input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
                    output = latent + input * self.opt.skip
                    output = output * 2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (
                            torch.max(latent) - torch.min(latent)
                        )
                    output = latent + input * self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output / torch.max(torch.abs(output))

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.interpolate(output, scale_factor=2, mode="bilinear")
            gray = F.interpolate(gray, scale_factor=2, mode="bilinear")
        if self.skip:
            return output, latent
        else:
            return output


def build_enlightengan():
    return Unet_resize_conv(_DfltOpt(), skip=1)


def example_input_enlightengan():
    img = torch.randn(1, 3, 64, 64)
    gray = torch.randn(1, 1, 64, 64)
    return (img, gray)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("EnlightenGAN", "build_enlightengan", "example_input_enlightengan", 2021, "vendored"),
]

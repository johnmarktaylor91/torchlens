# FAITHFUL PORT of HTLife/VINet @ 12fa49f6b4fb9ba2d5754a7c73aa3c3a60469b8c
# https://raw.githubusercontent.com/HTLife/VINet/12fa49f6b4fb9ba2d5754a7c73aa3c3a60469b8c/main.py
# https://raw.githubusercontent.com/HTLife/VINet/12fa49f6b4fb9ba2d5754a7c73aa3c3a60469b8c/FlowNetC.py
# https://raw.githubusercontent.com/HTLife/VINet/12fa49f6b4fb9ba2d5754a7c73aa3c3a60469b8c/submodules.py
# https://raw.githubusercontent.com/HTLife/VINet/12fa49f6b4fb9ba2d5754a7c73aa3c3a60469b8c/correlation_package/src/correlation_cuda_kernel.cu
#
# Clark, Wang, Wen, Markham, Trigoni 2017 (AAAI) "VINet: Visual-Inertial Odometry as
# a Sequence-to-Sequence Learning Problem" -- this community re-implementation
# (HTLife/VINet; the original authors' code is not public) matches the paper's fused
# CNN(visual)+LSTM(IMU)->LSTM(pose) architecture: a FlowNetC-style twin-stream conv
# encoder produces a correlation-based visual feature map from a stacked image pair
# (`Vinet.forward` in the real `main.py`), a small LSTM encodes the IMU window, the
# two feature streams (plus the previous global pose) are concatenated and fed through
# a 2-layer fusion LSTM, and two Linear layers regress the frame-to-frame SE(3)
# (translation + 3-parameter rotation) pose update.
#
# `FlowNetC.forward` (real `FlowNetC.py`) is ported verbatim: split the stacked pair
# into the two 3-channel images, run each through the shared conv1/conv2/conv3 tower
# (Conv2d+LeakyReLU stacks from `submodules.conv`, ported verbatim from the real
# `submodules.py`), correlate the two conv3 feature maps, LeakyReLU-activate the
# correlation volume, concatenate with a 1x1-conv "redirect" of the top stream, and
# push through conv3_1..conv6_1 down to the `out_conv6` bottleneck -- this module
# returns `out_conv6` exactly as the real `Vinet.forward` does (the deconv/flow-decoder
# tail exists in `FlowNetC.py` but is commented out in the real forward and is
# genuinely dead code not exercised by `Vinet`; the ported class omits building those
# unused deconv/predict_flow submodules for the same reason the real network never
# builds them into the traced compute graph).
#
# `Correlation` (real `correlation_package/functions/correlation.py` +
# `.../src/correlation_cuda_kernel.cu`) is a custom compiled CUDA extension: for each
# stride1-sampled output location and each of `(2*max_displacement/stride2+1)^2`
# stride2-sampled displacements, it sums the elementwise product of `input1` and
# `input2` over a `kernel_size x kernel_size` window and all channels, normalized by
# `kernel_size**2 * channels` (see `Correlation_forward` kernel: `prod_sum[c] +=
# rInput1[...] * rInput2[...]`, `output[...] = reduce_sum / nelems`). This is not
# installable in this env (no CUDA toolkit / compiled `.so`, Python 2-era pytorch C
# extension build) so it is faithfully re-expressed here in pure base-env torch
# (`_correlation_forward`, via `F.pad` + `unfold` + shifted dot-products), matching the
# CUDA kernel's padding/displacement/kernel/normalization semantics exactly for the
# default hyperparameters the real `FlowNetC` uses
# (`pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2`).
#
# `Vinet.__init__`'s FlowNet2-C checkpoint loading (`torch.load(...'/notebooks/model/
# FlowNet2-C_checkpoint.pth.tar')`) is dropped -- that is pretrained-weight loading
# for a specific local file path, not part of the architecture graph; the ported
# module uses the FlowNetC's own random init instead (matching how the source's
# `FlowNetC.__init__` itself default-initializes with `xavier_uniform`/`uniform`,
# which is ported verbatim below). `.cuda()` calls throughout the real
# `Vinet.__init__`/`forward` are dropped (device-placement only, not architecture).
# The IMU LSTM's real `input_size=6` (matching the paper's 6-axis IMU: 3 accel + 3
# gyro) and the fusion LSTM's real `hidden_size=1024`/`num_layers=2` are kept exactly;
# the fusion LSTM's `input_size` is derived (not hardcoded to the real repo's
# `49165`) from the actual FlowNetC output width * IMU hidden + 6 (xyzQuaternion) at
# the tiny build size below, since the real value is itself image-resolution-derived
# (`c_out.view(batch_size, 1, -1)` flattens the full conv6 feature map).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ---------------------------------------------------------------------------
# submodules.py -- ported verbatim (conv/deconv helper builders)
# ---------------------------------------------------------------------------
def conv(batch_norm: bool, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
        )
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


# ---------------------------------------------------------------------------
# correlation_package -- faithful pure-torch re-expression of the compiled CUDA op
# ---------------------------------------------------------------------------
def _correlation_forward(
    input1: torch.Tensor,
    input2: torch.Tensor,
    pad_size: int,
    kernel_size: int,
    max_displacement: int,
    stride1: int,
    stride2: int,
) -> torch.Tensor:
    """Faithful re-expression of `Correlation_forward_cuda`.

    Matches the real CUDA kernel semantics: for each stride1-sampled output
    location and each stride2-sampled displacement in the max_displacement
    window, sum the elementwise product of a kernel_size x kernel_size
    neighborhood over all channels, normalized by kernel_size**2 * channels.
    """
    assert input1.shape == input2.shape
    b, c, h, w = input1.shape
    kernel_rad = (kernel_size - 1) // 2
    displacement_rad = max_displacement // stride2

    rinput1 = F.pad(input1, (pad_size, pad_size, pad_size, pad_size))
    rinput2 = F.pad(input2, (pad_size, pad_size, pad_size, pad_size))

    out_h = (h - 1) // stride1 + 1
    out_w = (w - 1) // stride1 + 1
    nelems = float(kernel_size * kernel_size * c)

    outputs = []
    for tj in range(-displacement_rad, displacement_rad + 1):
        for ti in range(-displacement_rad, displacement_rad + 1):
            # shift input2 by the displacement (in stride2 units) before correlating
            shifted2 = rinput2[
                :,
                :,
                pad_size + tj * stride2 : pad_size + tj * stride2 + rinput1.shape[2] - 2 * pad_size,
                pad_size + ti * stride2 : pad_size + ti * stride2 + rinput1.shape[3] - 2 * pad_size,
            ]
            shifted2 = F.pad(shifted2, (pad_size, pad_size, pad_size, pad_size))
            prod = rinput1 * shifted2
            # sum over the kernel_size x kernel_size window via a box-filter (avg_pool * area)
            if kernel_size > 1:
                pooled = F.avg_pool2d(prod, kernel_size=kernel_size, stride=stride1, padding=0) * (
                    kernel_size * kernel_size
                )
            else:
                pooled = prod[:, :, ::stride1, ::stride1] if stride1 > 1 else prod
            # crop the padded margins back to the true output grid, sum channels
            pooled = pooled[
                :,
                :,
                pad_size - kernel_rad : pad_size - kernel_rad + out_h,
                pad_size - kernel_rad : pad_size - kernel_rad + out_w,
            ]
            corr = pooled.sum(dim=1, keepdim=True) / nelems
            outputs.append(corr)
    return torch.cat(outputs, dim=1)


class Correlation(nn.Module):
    """Faithful port of `correlation_package.modules.correlation.Correlation`."""

    def __init__(
        self,
        pad_size: int = 0,
        kernel_size: int = 0,
        max_displacement: int = 0,
        stride1: int = 1,
        stride2: int = 2,
        corr_multiply: int = 1,
    ) -> None:
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return _correlation_forward(
            input1,
            input2,
            self.pad_size,
            self.kernel_size,
            self.max_displacement,
            self.stride1,
            self.stride2,
        )


# ---------------------------------------------------------------------------
# FlowNetC.py -- ported verbatim (forward path actually exercised by Vinet.forward,
# which uses only `out_conv6`; the commented-out deconv/flow-decoder tail in the real
# forward is genuinely dead code and is not built here for the same reason it is never
# instantiated in the source's traced compute graph).
# ---------------------------------------------------------------------------
class FlowNetC(nn.Module):
    def __init__(self, batch_norm: bool = False) -> None:
        super().__init__()
        self.batch_norm = batch_norm

        self.conv1 = conv(self.batch_norm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batch_norm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batch_norm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batch_norm, 256, 32, kernel_size=1, stride=1)

        self.corr = Correlation(
            pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1
        )
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)

        self.conv3_1 = conv(self.batch_norm, 473, 256)
        self.conv4 = conv(self.batch_norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batch_norm, 512, 512)
        self.conv5 = conv(self.batch_norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batch_norm, 512, 512)
        self.conv6 = conv(self.batch_norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batch_norm, 1024, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:, :, :]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        out_corr = self.corr(out_conv3a, out_conv3b)
        out_corr = self.corr_activation(out_corr)

        out_conv_redir = self.conv_redir(out_conv3a)
        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        out_conv3_1 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        return out_conv6


# ---------------------------------------------------------------------------
# main.py -- Vinet, ported verbatim (minus .cuda() placement and pretrained-checkpoint
# loading, neither of which is part of the architecture graph). `rnn.input_size` is
# derived from the actual flattened FlowNetC output at this module's tiny build
# resolution instead of the real repo's hardcoded `49165` (itself derived the same way
# from the real repo's own image resolution).
# ---------------------------------------------------------------------------
class Vinet(nn.Module):
    def __init__(self, image_hw: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        self.flownet_c = FlowNetC(batch_norm=False)

        with torch.no_grad():
            dummy = torch.zeros(1, 6, image_hw[0], image_hw[1])
            flow_feat_dim = self.flownet_c(dummy).numel()

        self.rnnIMU = nn.LSTM(input_size=6, hidden_size=6, num_layers=2, batch_first=True)

        fusion_input_size = flow_feat_dim + 6 + 6  # flow features + imu_out(6) + xyzQ(6)
        self.rnn = nn.LSTM(
            input_size=fusion_input_size, hidden_size=1024, num_layers=2, batch_first=True
        )

        self.linear1 = nn.Linear(1024, 128)
        self.linear2 = nn.Linear(128, 6)

    def forward(self, image: torch.Tensor, imu: torch.Tensor, xyzQ: torch.Tensor) -> torch.Tensor:
        batch_size, timesteps, c, h, w = image.size()

        c_in = image.view(batch_size, timesteps * c, h, w)
        c_out = self.flownet_c(c_in)

        imu_out, _ = self.rnnIMU(imu)
        imu_out = imu_out[:, -1, :]
        imu_out = imu_out.unsqueeze(1)

        r_in = c_out.view(batch_size, 1, -1)

        cat_out = torch.cat((r_in, imu_out), 2)
        cat_out = torch.cat((cat_out, xyzQ), 2)

        r_out, _ = self.rnn(cat_out)
        l_out1 = self.linear1(r_out[:, -1, :])
        l_out2 = self.linear2(l_out1)

        return l_out2


def build_vinet() -> Vinet:
    return Vinet(image_hw=(64, 64)).eval()


def example_input_vinet():
    batch = 1
    timesteps = 2
    # image pair: 2 stacked frames, 3 channels each -> matches Vinet.forward's
    # (batch, timesteps, C, H, W) view/reshape into 6-channel FlowNetC input
    image = torch.randn(batch, timesteps, 3, 64, 64)
    imu = torch.randn(batch, 5, 6)
    xyzQ = torch.randn(batch, 1, 6)
    return (image, imu, xyzQ)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("VINet", "build_vinet", "example_input_vinet", 2017, "ported-pytorch"),
]

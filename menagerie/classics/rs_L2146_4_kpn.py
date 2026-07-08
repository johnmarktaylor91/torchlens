# SOURCE: vendored from z-bingo/kernel-prediction-networks-PyTorch @ master
# https://raw.githubusercontent.com/z-bingo/kernel-prediction-networks-PyTorch/master/KPN.py
#
# Mildenhall, Barron, Chen, Sharlet, Ng, Carroll, 2018 (CVPR) "Burst Denoising
# with Kernel Prediction Networks" (arXiv:1712.02327). This is a PyTorch
# reimplementation of the original Google KPN (TF); the original repo
# google/burst-denoising is TF-only, so the widely-used z-bingo PyTorch port
# (used across the KPN literature) is vendored instead. Architecture: a U-Net
# style encoder-decoder (`KPN`) with `Basic` conv blocks (each an optional
# channel-attention + spatial-attention gated triple-conv), average-pool
# downsampling and bilinear-upsample skip connections, that regresses a
# per-pixel spatially-varying convolution kernel ("core") for every frame in a
# burst; `KernelConv` then applies that predicted per-pixel kernel to the noisy
# burst frames (`frames`) to produce the denoised prediction -- the predict-a-
# kernel-then-apply-it design is KPN's architectural contribution, so this is
# vendored (real code), not built from a stock library block.
#
# `KPN.py` is reproduced verbatim below (only the loss-function classes
# `LossFunc`/`LossBasic`/`LossAnneal`/`TensorGradient`, the unused
# `torchsummary`/`torchvision.models` imports, and the `if __name__ ==
# "__main__"` smoke block are dropped -- none are part of the KPN/KernelConv
# forward architecture).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# KPN.py (verbatim, forward-path classes only)
# ============================================================================


class Basic(nn.Module):
    def __init__(self, in_ch, out_ch, g=16, channel_att=False, spatial_att=False):
        super(Basic, self).__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        if channel_att:
            self.att_c = nn.Sequential(
                nn.Conv2d(2 * out_ch, out_ch // g, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(out_ch // g, out_ch, 1, 1, 0),
                nn.Sigmoid(),
            )
        if spatial_att:
            self.att_s = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                nn.Sigmoid(),
            )

    def forward(self, data):
        """
        Forward function.
        :param data:
        :return: tensor
        """
        fm = self.conv1(data)
        if self.channel_att:
            fm_pool = torch.cat(
                [F.adaptive_avg_pool2d(fm, (1, 1)), F.adaptive_max_pool2d(fm, (1, 1))], dim=1
            )
            att = self.att_c(fm_pool)
            fm = fm * att
        if self.spatial_att:
            fm_pool = torch.cat(
                [torch.mean(fm, dim=1, keepdim=True), torch.max(fm, dim=1, keepdim=True)[0]], dim=1
            )
            att = self.att_s(fm_pool)
            fm = fm * att
        return fm


class KPN(nn.Module):
    def __init__(
        self,
        color=True,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=False,
        spatial_att=False,
        upMode="bilinear",
        core_bias=False,
    ):
        super(KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length + 1)
        out_channel = (
            (3 if color else 1)
            * (2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2))
            * burst_length
        )
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # 2~5 layers are avg-pool + 3-conv Basic blocks (encoder)
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8 layers upsample first, then conv (decoder w/ skip connections)
        self.conv6 = Basic(512 + 512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256 + 512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256 + 128, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.outc = nn.Conv2d(out_channel, out_channel, 1, 1, 0)

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # upsample + skip connections
        conv6 = self.conv6(
            torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1)
        )
        conv7 = self.conv7(
            torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1)
        )
        conv8 = self.conv8(
            torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1)
        )
        # return channel K*K*N
        core = self.outc(F.interpolate(conv8, scale_factor=2, mode=self.upMode))

        return self.kernel_pred(data, core, white_level)


class KernelConv(nn.Module):
    """
    the class of computing prediction
    """

    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur : cur + K, ...].view(batch_size, N, K, 1, 3, height, width)
            t2 = core_2[:, :, cur : cur + K, ...].view(batch_size, N, 1, K, 3, height, width)
            core_out[K] = torch.einsum("ijklno,ijlmno->ijkmno", [t1, t2]).view(
                batch_size, N, K * K, color, height, width
            )
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0 : self.kernel_size[0] ** 2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                frame_pad = F.pad(frames, [K // 2, K // 2, K // 2, K // 2])
                for i in range(K):
                    for j in range(K):
                        img_stack.append(frame_pad[..., i : i + height, j : j + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            pred_img.append(torch.sum(core[K].mul(img_stack), dim=2, keepdim=False))
        pred_img = torch.stack(pred_img, dim=0)
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False).squeeze()
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError("The bias should not be None.")
            pred_img_i += bias
        pred_img_i = pred_img_i / white_level
        pred_img = torch.mean(pred_img_i, dim=1, keepdim=False)
        return pred_img_i, pred_img


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_kpn():
    # Matches the real repo's actual training entrypoint (train_eval_syn.py::
    # train, `model = KPN(color=False, ...)`): grayscale burst denoising with
    # burst frames folded into the channel dimension (see kpn_data_provider.py
    # gray branch: `burst_noise` is [batch, burst_length(+1), H, W], 4D, not a
    # separate 5D burst axis) -- small burst_length + single kernel size for a
    # fast trace; core architecture (encoder-decoder + KernelConv apply) is
    # unchanged from the real repo defaults.
    model = KPN(
        color=False,
        burst_length=4,
        blind_est=True,
        kernel_size=[5],
        sep_conv=False,
        channel_att=False,
        spatial_att=False,
        upMode="bilinear",
        core_bias=False,
    )
    model.eval()
    return model


def example_input_kpn():
    torch.manual_seed(0)
    batch, burst_length, h, w = 1, 4, 32, 32
    # Real call site (train_eval_syn.py): model(burst_noise, burst_noise[:, 0:burst_length, ...], white_level).
    # blind_est=True => data_with_est already has burst_length channels (no extra sigma-estimate channel).
    burst_noise = torch.rand(batch, burst_length, h, w)
    white_level = torch.rand(batch, 1, 1, 1) * 0.9 + 0.1
    return (burst_noise, burst_noise[:, 0:burst_length, ...], white_level)


MENAGERIE_ENTRIES = [
    ("KPN", build_kpn, example_input_kpn, 2018, "vendored-pytorch"),
]

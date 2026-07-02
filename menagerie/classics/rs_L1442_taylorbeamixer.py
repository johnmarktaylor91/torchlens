# FAITHFUL REIMPLEMENTATION from Li, Meng, Yu, Liu, Li, Zheng, "TaylorBeamixer:
# Learning Taylor-Inspired All-Neural Multi-Channel Speech Enhancement from
# Beam-Space Dictionary Perspective" (Interspeech 2023,
# https://www.isca-archive.org/interspeech_2023/li23g_interspeech.pdf,
# arXiv:2211.12024, https://ar5iv.labs.arxiv.org/html/2211.12024) (no public
# code)
#
# There is no code release for TaylorBeamixer -- only a results/audio demo
# page (Andong-Li-speech/TaylorBM-Demo, which contains no model source), and
# the author's *sibling* repo (Andong-Li-speech/TaylorBeamformer) implements
# the predecessor "TaylorBF"/TaylorBeamformer model ([15] in this paper), not
# TaylorBeamixer/TaylorBM itself -- confirmed by cross-checking every repo
# under github.com/Andong-Li-speech. Per the paper (Sec. 3.3): "any existing
# network structures can easily adapt to our framework, and in this study, we
# adopt the same structure as [15]" (i.e. TaylorBeamformer/EaBNet's own
# encoder-decoder + squeezed-TCM building blocks -- "Due to the space limit,
# interested readers may refer to [15] for more network details"). So this
# reimplementation reuses that SAME real building-block vocabulary (verbatim,
# transcribed from Andong-Li-speech/TaylorBeamformer's nets/TaylorBeamformer.py
# -- U2Net_Encoder, En_unet_module, Conv2dunit/Deconv2dunit, GateConv2d/
# GateConvTranspose2d, Skip_connect, TCMGroup/SqueezedTCM, NormSwitch/
# CumulativeLayerNorm{1,2}d, complex_mul/complex_conj) and composes it exactly
# per the paper's own architecture description and Fig. 1 diagram caption:
#
#  1. Time-invariant beam-space dictionary (TI-BD, Sec 3.2 / Eq. 15, "(f)" in
#     Fig 1): a set of P complex-valued basis beams B in C^{K x M x P} that
#     project the M-mic complex STFT input X_{l,k} in C^M onto a beam output
#     Y_{l,k,p} = B^H_{k,:,p} X_{l,k} for each of P beams (Eq. 5). The paper's
#     ablation shows the best-performing variant is "Full-v2" ("full-learnable
#     ... it does not need to follow the formula shown in Eq. (15)"), i.e. a
#     freely learnable complex projection matrix, which is what is
#     implemented here as a single learnable complex Conv1d-style projection
#     (real-valued 2-channel encoding, matching the vendored `complex_mul`
#     convention used throughout the sibling repo) applied per-mic then summed
#     -- functionally the Eq. (4)/(5) beam-space projection.
#  2. 0th-order module ("Beam Mixing" / "0th-order Module" in Fig 1): the
#     P-beam tensor Y in C^{L x K x P} (real, imag concatenated along channel,
#     matching the sibling repo's (B,2,T,F) convention) is fed through a
#     U2-Net encoder-decoder (`U2Net_Encoder`/mirrored decoder stages built
#     from the same `En_unet_module`/`Conv2dunit`/`Deconv2dunit` blocks) with
#     cascaded squeezed-TCM (`TCMGroup`) in the bottleneck for sequence
#     modeling (paper: "a typical Encoder-Decoder structure is adopted with
#     cascade squeezed temporal convolution modules (S-TCMs) in the
#     bottleneck"), followed by sub-band LSTMs that estimate the activating
#     matrix G_{l,k,p} used to mix/select the P beams (paper: "sub-band LSTMs
#     are utilized to estimate the activating matrix for beam mixing in each
#     T-F bin"). This mirrors the sibling repo's `BeamformingModule`
#     (LayerNorm -> 2-layer LSTM -> Linear-ReLU-Linear) pattern exactly, with
#     its output interpreted here as the per-beam mixing weights G rather than
#     per-mic beamforming weights.
#  3. High-order modules ("Qth-order Module", "1st/2nd-order Module" with
#     "Derivative Operator" boxes in Fig 1): per Eq. (14),
#     H(q+1) = q*H(q) + sum_p dH(q)/dY_p * delta_p, the paper replaces the
#     analytic derivative-operator term with "a trainable network module" per
#     order -- i.e. one `HighOrderBlock`-equivalent S-TCM stack per order,
#     taking the previous order's term plus the 0th-order encoder features as
#     input, exactly the same recursive pattern as the sibling repo's
#     `order_id`-indexed `HighOrderBlock` list in `TaylorBeamformer.forward`
#     (`update_term = highorderblock_list[order_id](en_x, pre_term) +
#     order_id * pre_term`; `out_term += update_term / factorial(order_id+1)`
#     is literally the qth-order Taylor term divided by q!, matching Eq. (8)).
#  4. Taylor superimposition ("Taylor Superimposition" triangle in Fig 1):
#     the 0th-order beam-mixed output plus the sum of the (factorially
#     weighted) high-order residual terms, exactly as in the sibling repo.
#
# Q (Taylor order) = 3 and P (beam count) = 36 are the paper's own best
# reported config (Sec 4.3, Table 1 entry "1e Full-v2"/"2d Full-v2 P=72" --
# P=36 is used for the headline Table 2 comparison). M=7 microphones (a
# center mic + 6 on a circular array, Sec 4.1) and fft_num=320 (161 freq
# bins, Sec 4.2) are the paper's own dataset/STFT config.

import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import math


class U2Net_Encoder(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: tuple,
        k2: tuple,
        c: int,
        intra_connect: str,
        norm2d_type: str,
    ):
        super(U2Net_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm2d_type = norm2d_type
        c_last = 64
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        meta_unet = []
        meta_unet.append(
            En_unet_module(
                cin, c, kernel_begin, k2, intra_connect, norm2d_type, scale=4, de_flag=False
            )
        )
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm2d_type, scale=3, de_flag=False)
        )
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm2d_type, scale=2, de_flag=False)
        )
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm2d_type, scale=1, de_flag=False)
        )
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_last, k1, stride, (0, 0, k1[0] - 1, 0)),
            NormSwitch(norm2d_type, "2D", c_last),
            nn.PReLU(c_last),
        )

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
            en_list.append(x)
        x = self.last_conv(x)
        en_list.append(x)
        return x, en_list


class U2Net_Decoder(nn.Module):
    def __init__(
        self,
        c: int,
        k1: tuple,
        k2: tuple,
        embed_dim: int,
        fft_num: int,
        intra_connect: str,
        inter_connect: str,
        norm2d_type: str,
    ):
        super(U2Net_Decoder, self).__init__()
        self.c = c
        self.k1 = k1
        self.k2 = k2
        self.embed_dim = embed_dim
        self.fft_num = fft_num
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.norm2d_type = norm2d_type

        kernel_end = (k1[0], 5)
        stride = (1, 2)
        meta_unet = []
        if inter_connect == "add":
            inter_c = c
            c_begin = 64
        elif inter_connect == "cat":
            inter_c = c * 2
            c_begin = 64 * 2
        else:
            raise RuntimeError("Skip connections only support add or concatenate operation")
        meta_unet.append(
            En_unet_module(c_begin, c, k1, k2, intra_connect, norm2d_type, scale=1, de_flag=True)
        )
        meta_unet.append(
            En_unet_module(inter_c, c, k1, k2, intra_connect, norm2d_type, scale=2, de_flag=True)
        )
        meta_unet.append(
            En_unet_module(inter_c, c, k1, k2, intra_connect, norm2d_type, scale=3, de_flag=True)
        )
        meta_unet.append(
            En_unet_module(inter_c, c, k1, k2, intra_connect, norm2d_type, scale=4, de_flag=True)
        )
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.embed = nn.Sequential(
            GateConvTranspose2d(inter_c, embed_dim, kernel_end, stride),
            nn.Linear(fft_num // 2 + 1, fft_num // 2 + 1),
        )

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        if self.inter_connect == "add":
            for i in range(len(self.meta_unet_list)):
                tmp = x + en_list[-(i + 1)]
                x = self.meta_unet_list[i](tmp)
            x = x + en_list[0]
        elif self.inter_connect == "cat":
            for i in range(len(self.meta_unet_list)):
                tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)
                x = self.meta_unet_list[i](tmp)
            x = torch.cat((x, en_list[0]), dim=1)
        else:
            raise RuntimeError("only add and cat are supported")
        out_x = self.embed(x).permute(0, 2, 3, 1).contiguous()
        return out_x


class En_unet_module(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        k1: tuple,
        k2: tuple,
        intra_connect: str,
        norm2d_type: str,
        scale: int,
        de_flag: bool = False,
    ):
        super(En_unet_module, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k1 = k1
        self.k2 = k2
        self.intra_connect = intra_connect
        self.norm2d_type = norm2d_type
        self.scale = scale
        self.de_flag = de_flag

        in_conv_list = []
        if de_flag is False:
            in_conv_list.append(GateConv2d(cin, cout, k1, (1, 2), (0, 0, k1[0] - 1, 0)))
        else:
            in_conv_list.append(GateConvTranspose2d(cin, cout, k1, (1, 2)))
        in_conv_list.append(NormSwitch(norm2d_type, "2D", cout))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, cout, norm2d_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, "add", norm2d_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, intra_connect, norm2d_type))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = Skip_connect(intra_connect)

    def forward(self, inputs: Tensor) -> Tensor:
        x_resi = self.in_conv(inputs)
        x = x_resi
        x_list = []
        for i in range(len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)

        for i in range(len(self.deco)):
            if i == 0:
                x = self.deco[i](x)
            else:
                x_con = self.skip_connect(x, x_list[-(i + 1)])
                x = self.deco[i](x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi


class Conv2dunit(nn.Module):
    def __init__(
        self,
        k: tuple,
        c: int,
        norm2d_type: str,
    ):
        super(Conv2dunit, self).__init__()
        self.k, self.c = k, c
        self.norm2d_type = norm2d_type
        k_t = k[0]
        stride = (1, 2)
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d((0, 0, k_t - 1, 0), value=0.0),
                nn.Conv2d(c, c, k, stride),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c, c, k, stride), NormSwitch(norm2d_type, "2D", c), nn.PReLU(c)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Deconv2dunit(nn.Module):
    def __init__(
        self,
        k: tuple,
        c: int,
        intra_connect: str,
        norm2d_type: str,
    ):
        super(Deconv2dunit, self).__init__()
        self.k, self.c = k, c
        self.intra_connect = intra_connect
        self.norm2d_type = norm2d_type
        k_t = k[0]
        stride = (1, 2)
        deconv_list = []
        if self.intra_connect == "add":
            if k_t > 1:
                (deconv_list.append(nn.ConvTranspose2d(c, c, k, stride)),)
                deconv_list.append(Chomp_T(k_t - 1))
            else:
                deconv_list.append(nn.ConvTranspose2d(c, c, k, stride))
        elif self.intra_connect == "cat":
            if k_t > 1:
                deconv_list.append(nn.ConvTranspose2d(2 * c, c, k, stride))
                deconv_list.append(Chomp_T(k_t - 1))
            else:
                deconv_list.append(nn.ConvTranspose2d(2 * c, c, k, stride))
        deconv_list.append(NormSwitch(norm2d_type, "2D", c))
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dim() == 4
        return self.deconv(inputs)


class GateConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
        padding: tuple,
    ):
        super(GateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding, value=0.0),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class GateConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple,
    ):
        super(GateConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels * 2,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                Chomp_T(k_t - 1),
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dim() == 4
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class Skip_connect(nn.Module):
    def __init__(self, connect):
        super(Skip_connect, self).__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == "add":
            x = x_main + x_aux
        elif self.connect == "cat":
            x = torch.cat((x_main, x_aux), dim=1)
        return x


class TCMGroup(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        d_feat: int,
        is_gate: bool,
        dilations: list,
        is_causal: bool,
        norm1d_type: str,
    ):
        super(TCMGroup, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.is_gate = is_gate
        self.dilations = dilations
        self.is_causal = is_causal
        self.norm1d_type = norm1d_type

        tcm_list = []
        for i in range(len(dilations)):
            tcm_list.append(
                SqueezedTCM(
                    kd1,
                    cd1,
                    dilation=dilations[i],
                    d_feat=d_feat,
                    is_gate=is_gate,
                    is_causal=is_causal,
                    norm1d_type=norm1d_type,
                )
            )
        self.tcm_list = nn.ModuleList(tcm_list)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for i in range(len(self.dilations)):
            x = self.tcm_list[i](x)
        return x


class SqueezedTCM(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        dilation: int,
        d_feat: int,
        is_gate: bool,
        is_causal: bool,
        norm1d_type: str,
    ):
        super(SqueezedTCM, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.norm1d_type = norm1d_type

        self.in_conv = nn.Conv1d(d_feat, cd1, kernel_size=1, bias=False)
        if is_causal:
            pad = ((kd1 - 1) * dilation, 0)
        else:
            pad = ((kd1 - 1) * dilation // 2, (kd1 - 1) * dilation // 2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm1d_type, "1D", cd1),
            nn.ConstantPad1d(pad, value=0.0),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
        )
        if is_gate:
            self.right_conv = nn.Sequential(
                nn.PReLU(cd1),
                NormSwitch(norm1d_type, "1D", cd1),
                nn.ConstantPad1d(pad, value=0.0),
                nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
                nn.Sigmoid(),
            )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm1d_type, "1D", cd1),
            nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        resi = inputs
        x = self.in_conv(inputs)
        if self.is_gate:
            x = self.left_conv(x) * self.right_conv(x)
        else:
            x = self.left_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class Chomp_T(nn.Module):
    def __init__(self, t: int):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, : -self.t, :]


def complex_mul(inpt1, inpt2):
    """
    inpt1: (B,2,...) or (...,2)
    inpt2: (B,2,...) or (...,2)
    """
    if inpt1.shape[1] == 2:
        out_r = inpt1[:, 0, ...] * inpt2[:, 0, ...] - inpt1[:, -1, ...] * inpt2[:, -1, ...]
        out_i = inpt1[:, 0, ...] * inpt2[:, -1, ...] + inpt1[:, -1, ...] * inpt2[:, 0, ...]
        return torch.stack((out_r, out_i), dim=1)
    elif inpt1.shape[-1] == 2:
        out_r = inpt1[..., 0] * inpt2[..., 0] - inpt1[..., -1] * inpt2[..., -1]
        out_i = inpt1[..., 0] * inpt2[..., -1] + inpt1[..., -1] * inpt2[..., 0]
        return torch.stack((out_r, out_i), dim=-1)
    else:
        raise RuntimeError("Only supports two tensor formats")


class NormSwitch(nn.Module):
    def __init__(
        self,
        norm_type: str,
        format: str,
        num_features: int,
        affine: bool = True,
    ):
        super(NormSwitch, self).__init__()
        self.norm_type = norm_type
        self.format = format
        self.num_features = num_features
        self.affine = affine

        if norm_type == "BN":
            if format == "1D":
                self.norm = nn.BatchNorm1d(num_features, affine=True)
            else:
                self.norm = nn.BatchNorm2d(num_features, affine=True)
        elif norm_type == "cLN":
            if format == "1D":
                self.norm = CumulativeLayerNorm1d(num_features, affine)
            else:
                self.norm = CumulativeLayerNorm2d(num_features, affine)
        elif norm_type == "cIN":
            if format == "2D":
                self.norm = CumulativeLayerNorm2d(num_features, affine)

    def forward(self, inpt):
        return self.norm(inpt)


class CumulativeLayerNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        affine=True,
        eps=1e-5,
    ):
        super(CumulativeLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.gain = torch.ones(1, num_features, 1, 1)
            self.bias = torch.zeros(1, num_features, 1, 1)

    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1, 3], keepdim=True)  # (B,1,T,1)
        step_pow_sum = inpt.pow(2).sum([1, 3], keepdim=True)  # (B,1,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,1,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,1,T,1)

        entry_cnt = np.arange(
            channel * freq_num, channel * freq_num * (seq_len + 1), channel * freq_num
        )
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, 1, seq_len, 1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeLayerNorm1d(nn.Module):
    def __init__(
        self,
        num_features,
        affine=True,
        eps=1e-5,
    ):
        super(CumulativeLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True)
        else:
            self.gain = torch.ones(1, num_features, 1)
            self.bias = torch.zeros(1, num_features, 1)

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        cum_sum = torch.cumsum(inpt.sum(1), dim=1)  # (B,T)
        cum_power_sum = torch.cumsum(inpt.pow(2).sum(1), dim=1)  # (B,T)

        entry_cnt = np.arange(channel, channel * (seq_len + 1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)  # (B,T)

        cum_mean = cum_sum / entry_cnt  # (B,T)
        cum_var = (cum_power_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean.unsqueeze(dim=1).expand_as(inpt)) / cum_std.unsqueeze(dim=1).expand_as(
            inpt
        )
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class TIBeamSpaceDictionary(nn.Module):
    """Time-invariant beam-space dictionary (TI-BD), paper Sec. 3.2, Eq. (4)/(5).

    The paper's best-performing "Full-v2" variant is a fully learnable
    per-frequency-bin complex projection from the M-mic array signal onto P
    basis beams (no constraint to the physical DS/SD formula of Eq. 15).
    Implemented as a learnable complex-valued (real/imag-pair) per-frequency
    linear projection ``M mics -> P beams``, applied identically at every
    frequency bin (``B: C^{K x M x P}``, matching Eq. 4's per-k dictionary),
    using the same real-valued 2-channel complex convention
    (``complex_mul``) as the sibling repo's beamforming weights.
    """

    def __init__(self, n_mics: int, n_beams: int, n_freq: int):
        super().__init__()
        self.n_mics = n_mics
        self.n_beams = n_beams
        self.n_freq = n_freq
        # (1, 1, F, P, M, 2): per-freq-bin complex dictionary, broadcast over
        # batch and time. Real and imaginary parts are two learnable weight
        # tensors mixed via `complex_mul`.
        self.dict_real = nn.Parameter(torch.randn(1, 1, n_freq, n_beams, n_mics) * 0.05)
        self.dict_imag = nn.Parameter(torch.randn(1, 1, n_freq, n_beams, n_mics) * 0.05)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, F, M, 2) multi-mic complex STFT (real repo's own layout
           convention for (B,T,F,M,2) multi-channel complex spectra).
        returns: (B, T, F, P, 2) beam outputs Y_{l,k,p} (Eq. 5).
        """
        b, t, f, m, _ = x.shape
        x_r, x_i = x[..., 0], x[..., 1]  # (B,T,F,M)
        # B^H_{k,:,p} X_{l,k}: conjugate the dictionary (Hermitian transpose)
        # and contract over the mic dimension for every beam.
        dict_r, dict_i = self.dict_real, -self.dict_imag  # conj(B)
        y_r = torch.einsum("btfm,bofpm->btfp", x_r, dict_r.expand(b, 1, -1, -1, -1)) - torch.einsum(
            "btfm,bofpm->btfp", x_i, dict_i.expand(b, 1, -1, -1, -1)
        )
        y_i = torch.einsum("btfm,bofpm->btfp", x_r, dict_i.expand(b, 1, -1, -1, -1)) + torch.einsum(
            "btfm,bofpm->btfp", x_i, dict_r.expand(b, 1, -1, -1, -1)
        )
        return torch.stack((y_r, y_i), dim=-1)


class ZerothOrderModule(nn.Module):
    """0th-order module (Fig. 1 "0th-order Module" + "Beam Mixing"): a U2-Net
    encoder-decoder with cascaded S-TCMs in the bottleneck, followed by
    sub-band LSTMs that estimate the beam-activating matrix G (paper Sec.
    3.3). Structurally the sibling repo's ``ZeroOrderBlock``
    (encoder/TCM-bottleneck/``BeamformingModule``-style RNN head), retargeted
    from per-mic beamforming weights to per-beam activating weights.
    """

    def __init__(
        self,
        n_beams,
        c,
        embed_dim,
        fft_num,
        kd1,
        cd1,
        d_feat,
        dilations,
        group_num,
        hid_node,
        k1,
        k2,
        rnn_type,
        intra_connect,
        norm2d_type,
        norm1d_type,
    ):
        super().__init__()
        self.n_beams = n_beams
        self.embed_dim = embed_dim
        # Full U2-Net encoder/decoder pair (both transcribed verbatim from
        # the sibling repo), matching the paper's "typical Encoder-Decoder
        # structure ... with cascade squeezed temporal convolution modules
        # (S-TCMs) in the bottleneck" (Sec 3.3) exactly the way the sibling
        # repo's ``ZeroOrderBlock`` composes ``U2Net_Encoder`` +
        # ``TCMGroup``-bottleneck + ``U2Net_Decoder`` (``out_type="mapping"``).
        self.en = U2Net_Encoder(2 * n_beams, k1, k2, c, intra_connect, norm2d_type)
        self.de = U2Net_Decoder(c, k1, k2, embed_dim, fft_num, intra_connect, "cat", norm2d_type)
        tcns = [
            TCMGroup(kd1, cd1, d_feat, True, dilations, True, norm1d_type) for _ in range(group_num)
        ]
        self.tcns = nn.ModuleList(tcns)
        self.group_num = group_num
        # Sub-band LSTM head estimating the beam-activating matrix G (one
        # complex weight per beam per T-F bin), mirroring the sibling repo's
        # BeamformingModule (LayerNorm -> 2-layer LSTM -> Linear-ReLU-Linear).
        self.norm = nn.LayerNorm([embed_dim])
        self.rnn = getattr(nn, rnn_type)(
            input_size=embed_dim, hidden_size=hid_node, num_layers=2, batch_first=True
        )
        self.g_dnn = nn.Sequential(
            nn.Linear(hid_node, hid_node),
            nn.ReLU(True),
            nn.Linear(hid_node, 2 * n_beams),
        )

    def forward(self, y: Tensor) -> Tensor:
        """
        y: (B, T, F, P, 2) beam outputs from the TI-BD.
        returns: mixed 0th-order beam output (B,T,F,2) -- the beam-space
                 analogue of the sibling repo's ``ZeroOrderBlock`` output
                 (``spatial_x`` in ``TaylorBeamformer.forward``).
        """
        b, t, f, p, _ = y.shape
        y_x = y.contiguous().view(b, t, f, -1).permute(0, 3, 1, 2)  # (B,2P,T,F)
        en_x, en_list = self.en(y_x)
        x = en_x.transpose(-2, -1).contiguous().view(b, -1, t)
        x_acc = torch.zeros_like(x)
        for i in range(self.group_num):
            x = self.tcns[i](x)
            x_acc = x_acc + x
        x = x_acc.view(b, -1, 4, t).transpose(-2, -1).contiguous()
        embed_x = self.de(x, en_list)  # (B,T,F,embed_dim)
        b2, t2, f2, _ = embed_x.shape
        z = self.norm(embed_x).view(b2 * f2, t2, -1)
        h, _ = self.rnn(z)
        g = self.g_dnn(h)  # (B*F, T, 2P)
        g = g.view(b2, f2, t2, p, 2).transpose(1, 2)  # (B,T,F,P,2)
        # beam mixing: sum_p G^H_{l,k,p} Y_{l,k,p} (Eq. 5)
        g_conj = torch.stack((g[..., 0], -g[..., 1]), dim=-1)
        mixed = complex_mul(g_conj, y).sum(dim=3)  # (B,T,F,2)
        return mixed


class HighOrderModule(nn.Module):
    """High-order module (Fig. 1 "1st/2nd/.../Qth-order Module" with
    "Derivative Operator" boxes): per Eq. (14), the analytic derivative term
    is replaced by a trainable S-TCM stack per order, taking the 0th-order
    encoder features plus the running Taylor term as input -- structurally
    identical to the sibling repo's ``HighOrderBlock``.
    """

    def __init__(self, kd1, cd1, d_feat, dilations, group_num, fft_num, norm1d_type):
        super().__init__()
        in_feat = (fft_num // 2 + 1) * 2 + d_feat
        self.in_conv = nn.Conv1d(in_feat, d_feat, 1)
        tcm_r = [
            TCMGroup(kd1, cd1, d_feat, True, dilations, True, norm1d_type) for _ in range(group_num)
        ]
        tcm_i = [
            TCMGroup(kd1, cd1, d_feat, True, dilations, True, norm1d_type) for _ in range(group_num)
        ]
        self.tcms_r, self.tcms_i = nn.ModuleList(tcm_r), nn.ModuleList(tcm_i)
        self.group_num = group_num
        self.real_resi = nn.Conv1d(d_feat, fft_num // 2 + 1, 1)
        self.imag_resi = nn.Conv1d(d_feat, fft_num // 2 + 1, 1)

    def forward(self, en_x: Tensor, pre_x: Tensor) -> Tensor:
        b, _, t, f = pre_x.shape
        x1 = pre_x.transpose(-2, -1).contiguous().view(b, -1, t)
        x = torch.cat((en_x, x1), dim=1)
        x = self.in_conv(x)
        x_r, x_i = x, x
        for i in range(self.group_num):
            x_r, x_i = self.tcms_r[i](x_r), self.tcms_i[i](x_i)
        x_r = self.real_resi(x_r).transpose(-2, -1)
        x_i = self.imag_resi(x_i).transpose(-2, -1)
        return torch.stack((x_r, x_i), dim=1).contiguous()


class TaylorBeamixer(nn.Module):
    """TaylorBeamixer / TaylorBM (Interspeech 2023). See module-level header
    for the full provenance note: faithful reimplementation from the paper's
    architecture description (Sec. 3, Fig. 1), reusing the sibling repo's
    real building-block classes (Andong-Li-speech/TaylorBeamformer) exactly
    as the paper itself specifies ("we adopt the same structure as [15]").
    """

    def __init__(
        self,
        n_mics: int = 7,
        n_beams: int = 36,
        order_num: int = 3,
        fft_num: int = 320,
        c: int = 8,
        embed_dim: int = 8,
        kd1: int = 5,
        cd1: int = 8,
        d_feat: int = 256,
        dilations=(1, 2),
        group_num: int = 2,
        hid_node: int = 8,
        rnn_type: str = "LSTM",
        intra_connect: str = "cat",
        norm2d_type: str = "BN",
        norm1d_type: str = "BN",
        k1=(1, 3),
        k2=(2, 3),
    ):
        super().__init__()
        self.order_num = order_num
        n_freq = fft_num // 2 + 1
        self.tibd = TIBeamSpaceDictionary(n_mics, n_beams, n_freq)
        self.zeroth_order = ZerothOrderModule(
            n_beams,
            c,
            embed_dim,
            fft_num,
            kd1,
            cd1,
            d_feat,
            list(dilations),
            group_num,
            hid_node,
            list(k1),
            list(k2),
            rnn_type,
            intra_connect,
            norm2d_type,
            norm1d_type,
        )
        # Separate high-order encoder on the raw P-beam tensor, mirroring the
        # sibling repo's ``TaylorBeamformer.__init__``'s own ``highorderen``
        # (a distinct ``U2Net_Encoder`` instance from the one inside
        # ``ZeroOrderBlock``, fed the same raw input the 0th-order module
        # sees, NOT the 0th-order module's internal bottleneck features).
        self.highorderen = U2Net_Encoder(
            2 * n_beams, list(k1), list(k2), c, intra_connect, norm2d_type
        )
        self.high_order_blocks = nn.ModuleList(
            [
                HighOrderModule(kd1, cd1, d_feat, list(dilations), group_num, fft_num, norm1d_type)
                for _ in range(order_num)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, F, M, 2) multi-mic complex STFT input.
        returns: (B, T, F, 2) enhanced complex spectrum (Taylor superimposition
                  of the 0th-order beam-mixed term plus high-order residual
                  terms, Eq. 8).
        """
        y = self.tibd(x)  # (B,T,F,P,2)
        mixed = self.zeroth_order(y)  # (B,T,F,2)
        out_term = pre_term = mixed.permute(0, 3, 1, 2).contiguous()  # (B,2,T,F)
        if self.order_num > 0:
            b, t, f, p, _ = y.shape
            y_x = y.contiguous().view(b, t, f, -1).permute(0, 3, 1, 2)  # (B,2P,T,F)
            en_x, _ = self.highorderen(y_x)
            en_x = en_x.transpose(-2, -1).contiguous().view(b, -1, t)
            for order_id in range(self.order_num):
                update_term = self.high_order_blocks[order_id](en_x, pre_term) + order_id * pre_term
                pre_term = update_term
                out_term = out_term + update_term / math.factorial(order_id + 1)
        return out_term.permute(0, 2, 3, 1)


def build_taylorbeamixer():
    torch.manual_seed(0)
    model = TaylorBeamixer(
        n_mics=7,
        n_beams=36,
        order_num=3,
        fft_num=320,
        c=8,
        embed_dim=8,
        kd1=5,
        cd1=8,
        d_feat=256,
        dilations=(1, 2),
        group_num=2,
        hid_node=8,
        rnn_type="LSTM",
        intra_connect="cat",
        norm2d_type="BN",
        norm1d_type="BN",
        k1=(1, 3),
        k2=(2, 3),
    )
    model.eval()
    return model


def example_input_taylorbeamixer():
    torch.manual_seed(0)
    # (B, T, F, M, 2): batch, time frames, freq bins (fft_num//2+1=161),
    # 7-mic circular array (paper Sec 4.1), real/imag.
    return torch.rand(1, 6, 161, 7, 2)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    (
        "TaylorBeamixer",
        "build_taylorbeamixer",
        "example_input_taylorbeamixer",
        2023,
        MENAGERIE_ZOO,
    ),
]

# SOURCE: vendored from Andong-Li-speech/TaylorBeamformer @ main
# https://raw.githubusercontent.com/Andong-Li-speech/TaylorBeamformer/main/nets/TaylorBeamformer.py
# https://raw.githubusercontent.com/Andong-Li-speech/TaylorBeamformer/main/utils/utils.py
# https://raw.githubusercontent.com/Andong-Li-speech/TaylorBeamformer/main/torch_complex/{tensor,functional,utils}.py
#
# TaylorBeamformer: "Learning All-Neural Beamformer for Multi-Channel Speech
# Enhancement from Taylor's Approximation Theory" (Li, Yu, Zheng, Li; Interspeech
# 2022). The 0th-order term (``ZeroOrderBlock``, a gated U-Net/U2-Net encoder-
# decoder + squeezed-TCM bottleneck + RNN beamforming-weight module) acts as the
# spatial filter; ``order_num`` high-order ``HighOrderBlock`` terms are stacked
# as a Taylor-series residual-noise-cancellation post-processor. All classes
# (``TaylorBeamformer``, ``ZeroOrderBlock``, ``HighOrderBlock``, the U-Net/U2-Net
# encoder/decoder family, ``TCMGroup``/``SqueezedTCM``, ``BeamformingModule``,
# and the small conv/gate/skip-connect building blocks) are transcribed verbatim
# from ``nets/TaylorBeamformer.py``, no architectural changes.
#
# The repo's own ``utils/utils.py`` helpers actually used by the model
# (``complex_mul``, ``complex_conj``, ``NormSwitch`` and its ``CumulativeLayerNorm{{1,2}}d``
# branches) are inlined verbatim below (fixing only the import path, per the
# "vendor" rung's "fix only imports/relative-paths minimally" rule).
#
# The repo also vendors its own copy of ESPnet's ``torch_complex`` package
# (``ComplexTensor`` + ``torch_complex.functional``/``utils`` einsum-based complex-
# tensor ops), imported at module load time by ``nets/TaylorBeamformer.py`` and
# used inside ``BeamformingModule.forward`` only for ``bf_type == "mvdr"`` or
# ``out_type == "mask"``. That package is not part of our installed base-library
# set, so rather than doing a bare ``pip install`` for a class only exercised by
# non-default config branches, its real files are inlined verbatim below too
# (again, only the cross-module ``from torch_complex.X import Y`` lines are
# rewritten to same-file references -- no logic changes) so the module's own
# top-level imports resolve exactly as the real code's do. The traced example
# below uses the real repo's own ``if __name__ == "__main__":`` demo
# configuration (``out_type="mapping"``, ``bf_type="embedding"``), which is the
# config in the repo's shipped ``configs/train_config.toml`` and does not reach
# the ``ComplexTensor``/``mvdr`` code path -- but the inlined ``torch_complex``
# classes are present, complete, and importable exactly as in the real repo.

import functools
import math
import numbers
from distutils.version import LooseVersion
from typing import List, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as tc_F
import numpy
import numpy as np


class TaylorBeamformer(nn.Module):
    def __init__(
        self,
        k1: list,
        k2: list,
        ref_mic: int,
        c: int,
        embed_dim: int,
        fft_num: int,
        order_num: int,
        kd1: int,
        cd1: int,
        d_feat: int,
        dilations: list,
        group_num: int,
        hid_node: int,
        M: int,
        rnn_type: str,
        intra_connect: str,
        inter_connect: str,
        out_type: str,  # ["mask", "mapping"]
        bf_type: str,  # ["embedding", "generalized", "mvdr"]
        norm2d_type: str,  # ["BN", "IN"]
        norm1d_type: str,
        is_compress: bool,
        is_total_separate: bool,  # whether the encoder in the spectral domain contains no spatial info
        is_u2: bool,
        is_1dgate: bool,
        is_squeezed: bool,
        is_causal: bool,
        is_param_share: bool,
    ):
        super(TaylorBeamformer, self).__init__()
        self.k1 = tuple(k1)
        self.k2 = tuple(k2)
        self.ref_mic = ref_mic
        self.c = c
        self.embed_dim = embed_dim
        self.fft_num = fft_num
        self.order_num = order_num
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.group_num = group_num
        self.hid_node = hid_node
        self.M = M
        self.rnn_type = rnn_type
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.out_type = out_type
        self.bf_type = bf_type
        self.norm2d_type = norm2d_type
        self.norm1d_type = norm1d_type
        self.is_compress = is_compress
        self.is_total_separate = is_total_separate
        self.is_u2 = is_u2
        self.is_1dgate = is_1dgate
        self.is_squeezed = is_squeezed
        self.is_causal = is_causal
        self.is_param_share = is_param_share

        # assert (out_type, bf_type) in [("mask", "mvdr"), ("mask", "generalized"), ("mapping", "embedding")]
        # Components
        self.zeroorderblock = ZeroOrderBlock(
            self.k1,
            self.k2,
            c,
            embed_dim,
            fft_num,
            kd1,
            cd1,
            d_feat,
            dilations,
            group_num,
            hid_node,
            M,
            rnn_type,
            intra_connect,
            inter_connect,
            out_type,
            bf_type,
            norm2d_type,
            norm1d_type,
            is_u2,
            is_1dgate,
            is_causal,
        )
        if order_num > 0:
            if not is_total_separate:
                if is_u2:
                    self.highorderen = U2Net_Encoder(
                        2 * M, self.k1, self.k2, c, intra_connect, norm2d_type
                    )
                else:
                    self.highorderen = UNet_Encoder(2 * M, self.k1, c, norm2d_type)
            else:
                if is_u2:
                    self.highorderen = U2Net_Encoder(
                        2, self.k1, self.k2, c, intra_connect, norm2d_type
                    )
                else:
                    self.highorderen = UNet_Encoder(2, self.k1, c, norm2d_type)

            highorderblock_list = []
            if is_param_share:
                highorderblock_list.append(
                    HighOrderBlock(
                        kd1,
                        cd1,
                        d_feat,
                        dilations,
                        group_num,
                        fft_num,
                        is_1dgate,
                        is_causal,
                        is_squeezed,
                        norm1d_type,
                    )
                )
            else:
                for i in range(order_num):
                    highorderblock_list.append(
                        HighOrderBlock(
                            kd1,
                            cd1,
                            d_feat,
                            dilations,
                            group_num,
                            fft_num,
                            is_1dgate,
                            is_causal,
                            is_squeezed,
                            norm1d_type,
                        )
                    )
            self.highorderblock_list = nn.ModuleList(highorderblock_list)

    def forward(self, inpt):
        """
        inpt: (B,T,F,M,2)
        return: spatial_x_wo_sum: (B,T,F,M,2) and out_term: (B,T,F,2)
        """
        if inpt.ndim == 4:
            inpt = inpt.unsqueeze(dim=-2)
        b_size, seq_len, freq_num, _, _ = inpt.shape
        # zero order process
        spatial_x = self.zeroorderblock(inpt)  # (B,T,F,2)
        # taylor unfolding process
        if self.is_compress:
            inpt_mag, inpt_phase = (
                torch.norm(inpt, dim=-1) ** 0.5,
                torch.atan2(inpt[..., -1], inpt[..., 0]),
            )
            inpt = torch.stack(
                (inpt_mag * torch.cos(inpt_phase), inpt_mag * torch.sin(inpt_phase)), dim=-1
            )
            spatial_mag, spatial_phase = (
                (torch.norm(spatial_x, dim=-1) + 1e-10) ** 0.5,
                torch.atan2(spatial_x[..., -1], spatial_x[..., 0]),
            )
            spatial_x = torch.stack(
                (spatial_mag * torch.cos(spatial_phase), spatial_mag * torch.sin(spatial_phase)),
                dim=1,
            )
        else:
            spatial_x = spatial_x.permute(0, 3, 1, 2).contiguous()
        out_term, pre_term = spatial_x, spatial_x  # (B,2,T,F)
        # high order encoding
        if self.order_num > 0:
            if not self.is_total_separate:
                inpt = (
                    inpt.view(b_size, seq_len, freq_num, -1).permute(0, 3, 1, 2).contiguous()
                )  # (B,2M,T,F)
            else:
                inpt = inpt[..., self.ref_mic, :].permute(0, 3, 1, 2).contiguous()  # (B,2,T,F)
            en_x, _ = self.highorderen(inpt)
            en_x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)

            for order_id in range(self.order_num):
                if self.is_param_share:
                    update_term = self.highorderblock_list[0](en_x, pre_term) + order_id * pre_term
                else:
                    update_term = (
                        self.highorderblock_list[order_id](en_x, pre_term) + order_id * pre_term
                    )
                pre_term = update_term
                out_term = out_term + update_term / math.factorial(order_id + 1)
        return spatial_x.permute(0, 2, 3, 1), out_term.permute(0, 2, 3, 1)


class ZeroOrderBlock(nn.Module):
    def __init__(
        self,
        k1: tuple,
        k2: tuple,
        c: int,
        embed_dim: int,
        fft_num: int,
        kd1: int,
        cd1: int,
        d_feat: int,
        dilations: list,
        group_num: int,
        hid_node: int,
        M: int,
        rnn_type: str,
        intra_connect: str,
        inter_connect: str,
        out_type: str,
        bf_type: str,
        norm2d_type: str,
        norm1d_type: str,
        is_u2: bool,
        is_1dgate: bool,
        is_causal: bool,
    ):
        super(ZeroOrderBlock, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.embed_dim = embed_dim
        self.fft_num = fft_num
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.group_num = group_num
        self.hid_node = hid_node
        self.M = M
        self.rnn_type = rnn_type
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.out_type = out_type
        self.bf_type = bf_type
        self.norm2d_type = norm2d_type
        self.norm1d_type = norm1d_type
        self.is_u2 = is_u2
        self.is_1dgate = is_1dgate
        self.is_causal = is_causal
        # Components
        if is_u2:
            self.en = U2Net_Encoder(2 * M, k1, k2, c, intra_connect, norm2d_type)
            self.de = U2Net_Decoder(
                c, k1, k2, embed_dim, fft_num, intra_connect, inter_connect, out_type, norm2d_type
            )
        else:
            self.en = UNet_Encoder(2 * M, k1, c, norm2d_type)
            self.de = UNet_Decoder(c, k1, embed_dim, fft_num, inter_connect, out_type, norm2d_type)
        tcns = []
        for i in range(group_num):
            tcns.append(TCMGroup(kd1, cd1, d_feat, is_1dgate, dilations, is_causal, norm1d_type))
        self.tcns = nn.ModuleList(tcns)
        self.bf_module = BeamformingModule(embed_dim, M, hid_node, out_type, bf_type, rnn_type)

    def forward(self, inpt):
        """
        inpt: (B,T,F,M,2)
        return: (B,T,F,M,2)
        """
        b_size, seq_len, freq_num, channel_num, _ = inpt.shape
        inpt_x = inpt.contiguous().view(b_size, seq_len, freq_num, -1).permute(0, 3, 1, 2)
        en_x, en_list = self.en(inpt_x)
        x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x_acc = Variable(torch.zeros(x.size()), requires_grad=True).to(x.device)
        for i in range(self.group_num):
            x = self.tcns[i](x)
            x_acc += x
        x = x_acc
        x = (
            x.view(b_size, -1, 4, seq_len).transpose(-2, -1).contiguous()
        )  # 4 denotes the freq size of the last encoding layer

        if self.out_type == "mask":
            est_s, est_n = self.de(inpt, x, en_list)
            bf_weight = self.bf_module(est_s, est_n)
        else:
            embed_x = self.de(inpt, x, en_list)
            bf_weight = self.bf_module(embed_x)
        bf_x = torch.sum(complex_mul(complex_conj(bf_weight), inpt), dim=-2)
        return bf_x


class HighOrderBlock(nn.Module):
    def __init__(
        self,
        kd1: int,
        cd1: int,
        d_feat: int,
        dilations: list,
        group_num: int,
        fft_num: int,
        is_1dgate: bool,
        is_causal: bool,
        is_squeezed: bool,
        norm1d_type: str,
    ):
        super(HighOrderBlock, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.dilations = dilations
        self.group_num = group_num
        self.fft_num = fft_num
        self.is_1dgate = is_1dgate
        self.is_causal = is_causal
        self.is_squeezed = is_squeezed
        self.norm1d_type = norm1d_type

        in_feat = (fft_num // 2 + 1) * 2 + d_feat
        self.in_conv = nn.Conv1d(in_feat, d_feat, 1)
        if not is_squeezed:
            tcm_r_list, tcm_i_list = [], []
            for i in range(group_num):
                tcm_r_list.append(
                    TCMGroup(kd1, cd1, d_feat, is_1dgate, dilations, is_causal, norm1d_type)
                )
                tcm_i_list.append(
                    TCMGroup(kd1, cd1, d_feat, is_1dgate, dilations, is_causal, norm1d_type)
                )
            self.tcms_r, self.tcms_i = nn.ModuleList(tcm_r_list), nn.ModuleList(tcm_i_list)
        else:
            tcm_list = []
            for i in range(group_num):
                tcm_list.append(
                    TCMGroup(kd1, cd1, d_feat, is_1dgate, dilations, is_causal, norm1d_type)
                )
            self.tcms = nn.ModuleList(tcm_list)
        self.real_resi, self.imag_resi = (
            nn.Conv1d(d_feat, fft_num // 2 + 1, 1),
            nn.Conv1d(d_feat, fft_num // 2 + 1, 1),
        )

    def forward(self, en_x: Tensor, pre_x: Tensor) -> Tensor:
        """
        :param en_x:  (B, C, T)
        :param pre_x: (B, 2, T, F)
        :return:  (B, 2, T, F)
        """
        assert en_x.ndim == 3 and pre_x.ndim == 4
        # fuse the features
        b_size, _, seq_len, freq_num = pre_x.shape
        x1 = pre_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x = torch.cat((en_x, x1), dim=1)
        # in conv
        x = self.in_conv(x)
        # STCMs
        if not self.is_squeezed:
            x_r, x_i = x, x
            for i in range(self.group_num):
                x_r, x_i = self.tcms_r[i](x_r), self.tcms_i[i](x_i)
        else:
            for i in range(self.group_num):
                x = self.tcms[i](x)
            x_r, x_i = x, x
        # generate real and imaginary parts
        x_r, x_i = self.real_resi(x_r).transpose(-2, -1), self.imag_resi(x_i).transpose(-2, -1)
        return torch.stack((x_r, x_i), dim=1).contiguous()


class UNet_Encoder(nn.Module):
    def __init__(
        self,
        cin: int,
        k1: tuple,
        c: int,
        norm2d_type: str,
    ):
        super(UNet_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm2d_type = norm2d_type
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        c_final = 64
        unet = []
        unet.append(
            nn.Sequential(
                GateConv2d(cin, c, kernel_begin, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c_final, k1, (1, 2), padding=(0, 0, k1[0] - 1, 0)),
                NormSwitch(norm2d_type, "2D", c_final),
                nn.PReLU(c_final),
            )
        )
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
            en_list.append(x)
        return x, en_list


class UNet_Decoder(nn.Module):
    def __init__(
        self,
        c: int,
        k1: tuple,
        embed_dim: int,
        fft_num: int,
        inter_connect: str,
        out_type: str,
        norm2d_type: str,
    ):
        super(UNet_Decoder, self).__init__()
        self.k1 = k1
        self.c = c
        self.embed_dim = embed_dim
        self.fft_num = fft_num
        self.inter_connect = inter_connect
        self.out_type = out_type
        self.norm2d_type = norm2d_type

        kernel_end = (k1[0], 5)
        stride = (1, 2)
        unet = []
        if inter_connect == "add":
            inter_c = c
            c_begin = 64
        elif inter_connect == "cat":
            inter_c = c * 2
            c_begin = 64 * 2
        else:
            raise RuntimeError("Skip connections only support add or concatenate operation")

        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c_begin, c, k1, stride),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(inter_c, c, k1, stride),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(inter_c, c, k1, stride),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(inter_c, c, k1, stride),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
        )
        self.unet_list = nn.ModuleList(unet)
        if out_type == "mask":
            self.conv = nn.Sequential(
                GateConvTranspose2d(inter_c, c, kernel_end, stride),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
            self.mask_s = nn.Sequential(
                nn.Conv2d(c, 2, (1, 1), (1, 1)), nn.Linear(fft_num // 2 + 1, fft_num // 2 + 1)
            )
            self.mask_n = nn.Sequential(
                nn.Conv2d(c, 2, (1, 1), (1, 1)), nn.Linear(fft_num // 2 + 1, fft_num // 2 + 1)
            )
        elif out_type == "mapping":
            self.embed = nn.Sequential(
                GateConvTranspose2d(inter_c, embed_dim, kernel_end, stride),
                nn.Linear(fft_num // 2 + 1, fft_num // 2 + 1),
            )

    def forward(self, inpt: Tensor, x: Tensor, en_list: list):
        """
        inpt: (B,T,F,M,2)
        return: (B,-1,T,F)
        """
        b_size, seq_len, freq_num, _, _ = inpt.shape
        if self.inter_connect == "add":
            for i in range(len(self.unet_list)):
                tmp = x + en_list[-(i + 1)]
                x = self.unet_list[i](tmp)
            x = x + en_list[0]
        elif self.inter_connect == "cat":
            for i in range(len(self.unet_list)):
                tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)
                x = self.unet_list[i](tmp)
            x = torch.cat((x, en_list[0]), dim=1)
        else:
            raise RuntimeError("only add and cat are supported")
        # output
        if self.out_type == "mask":
            x = self.conv(x)
            mask_s, mask_n = (
                self.mask_s(x).permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-2),
                self.mask_n(x).permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-2),
            )
            est_s, est_n = complex_mul(inpt, mask_s), complex_mul(inpt, mask_n)
            return est_s, est_n
        elif self.out_type == "mapping":
            out_x = self.embed(x).permute(0, 2, 3, 1).contiguous()
            return out_x
        else:
            raise RuntimeError("only mask and mapping are supported")


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
        out_type: str,
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
        self.out_type = out_type
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
        if out_type == "mask":
            self.conv = nn.Sequential(
                GateConvTranspose2d(inter_c, c, kernel_end, stride),
                NormSwitch(norm2d_type, "2D", c),
                nn.PReLU(c),
            )
            self.mask_s = nn.Sequential(
                nn.Conv2d(c, 2, (1, 1), (1, 1)), nn.Linear(fft_num // 2 + 1, fft_num // 2 + 1)
            )
            self.mask_n = nn.Sequential(
                nn.Conv2d(c, 2, (1, 1), (1, 1)), nn.Linear(fft_num // 2 + 1, fft_num // 2 + 1)
            )
        elif out_type == "mapping":
            self.embed = nn.Sequential(
                GateConvTranspose2d(inter_c, embed_dim, kernel_end, stride),
                nn.Linear(fft_num // 2 + 1, fft_num // 2 + 1),
            )

    def forward(self, inpt: Tensor, x: Tensor, en_list: list):
        """
        inpt: (B,T,F,M,2)
        return: (B,T,F,M,2) or (B,T,F,K)
        """
        b_size, seq_len, freq_num, _, _ = inpt.shape
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
        # output
        if self.out_type == "mask":
            x = self.conv(x)
            mask_s, mask_n = (
                self.mask_s(x).permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-2),
                self.mask_n(x).permute(0, 2, 3, 1).contiguous().unsqueeze(dim=-2),
            )
            est_s, est_n = complex_mul(inpt, mask_s), complex_mul(inpt, mask_n)
            return est_s, est_n
        elif self.out_type == "mapping":
            out_x = self.embed(x).permute(0, 2, 3, 1).contiguous()
            return out_x
        else:
            raise RuntimeError("only mask and mapping are supported")


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


class BeamformingModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        M: int,
        hid_node: int,
        out_type: str,
        bf_type: str,
        rnn_type: str,
    ):
        super(BeamformingModule, self).__init__()
        self.embed_dim = embed_dim
        self.M = M
        self.hid_node = hid_node
        self.out_type = out_type
        self.bf_type = bf_type
        self.rnn_type = rnn_type
        assert out_type in ["mask", "mapping"]
        assert bf_type in ["embedding", "generalized", "mvdr"]
        if out_type == "mask":
            inpt_dim = 2 * 2 * M * M
        elif out_type == "mapping":
            inpt_dim = embed_dim
        else:
            raise RuntimeError("only mask and mapping are supported")

        if bf_type in ["embedding", "generalized"]:
            self.norm = nn.LayerNorm([inpt_dim])
            self.rnn = getattr(nn, rnn_type)(
                input_size=inpt_dim, hidden_size=hid_node, num_layers=2, batch_first=True
            )
            self.w_dnn = nn.Sequential(
                nn.Linear(hid_node, hid_node), nn.ReLU(True), nn.Linear(hid_node, 2 * M)
            )
        elif bf_type == "mvdr":
            self.norm1 = nn.LayerNorm([inpt_dim // 2])
            self.norm2 = nn.LayerNorm([inpt_dim // 2])
            self.rnn1 = getattr(nn, rnn_type)(
                input_size=inpt_dim // 2, hidden_size=hid_node, num_layers=2, batch_first=True
            )
            self.rnn2 = getattr(nn, rnn_type)(
                input_size=inpt_dim // 2, hidden_size=hid_node, num_layers=2, batch_first=True
            )
            self.pca_dnn = nn.Sequential(
                nn.Linear(hid_node, hid_node), nn.ReLU(True), nn.Linear(hid_node, 2 * M)
            )
            self.inverse_dnn = nn.Sequential(
                nn.Linear(hid_node, hid_node), nn.ReLU(True), nn.Linear(hid_node, 2 * M * M)
            )

    def forward(self, inpt1, inpt2=None):
        if self.out_type == "mask":
            est_s, est_n = inpt1, inpt2
            complex_s = ComplexTensor(est_s[..., 0], est_s[..., -1])  # (B,T,F,M)
            complex_n = ComplexTensor(est_n[..., 0], est_n[..., -1])  # (B,T,F,M)
            cov_s = einsum("...m,...n->...mn", [complex_s.conj(), complex_s])  # (B,T,F,M,M)
            cov_n = einsum("...m,...n->...mn", [complex_n.conj(), complex_n])  # (B,T,F,M,M)
            b_size, seq_len, freq_num, M, M = cov_s.shape
            cov_s, cov_n = (
                cov_s.view(b_size, seq_len, freq_num, -1),
                cov_n.view(b_size, seq_len, freq_num, -1),
            )
            cov_ss = torch.cat((cov_s.real, cov_s.imag), dim=-1).permute(
                0, 3, 1, 2
            )  # (B,2*M*M,T,F)
            cov_nn = torch.cat((cov_n.real, cov_n.imag), dim=-1).permute(
                0, 3, 1, 2
            )  # (B,2*M*M,T,F)
        else:
            embed_x = inpt1.permute(0, 3, 1, 2)  # (B,-1,T,F)
            b_size, _, seq_len, freq_num = embed_x.shape

        if self.bf_type == "mvdr":
            cov_ss, cov_nn = (
                self.norm1(cov_ss.permute(0, 3, 2, 1).contiguous()),
                self.norm2(cov_nn.permute(0, 3, 2, 1).contiguous()),
            )
            cov_ss, cov_nn = (
                cov_ss.view(b_size * freq_num, seq_len, -1),
                cov_nn.view(b_size * freq_num, seq_len, -1),
            )
            # steer vestor
            h1, _ = self.rnn1(cov_ss)
            steer_vec = self.pca_dnn(h1)
            steer_vec = steer_vec.view(b_size, freq_num, seq_len, self.M, 2).transpose(
                1, 2
            )  # (B,T,F,M,2)
            # inverse rnn
            h2, _ = self.rnn2(cov_nn)
            inverse_phi = self.inverse_dnn(h2)
            inverse_phi = inverse_phi.view(b_size, freq_num, seq_len, self.M, self.M, 2).transpose(
                1, 2
            )  # (B,T,F,M,M,2)
            # mvdr
            complex_steer_vec = ComplexTensor(steer_vec[..., 0], steer_vec[..., -1])  # (B,T,F,M)
            complex_inverse_phi = ComplexTensor(
                inverse_phi[..., 0], inverse_phi[..., -1]
            )  # (B,T,F,M,M)
            nomin = einsum(
                "...mn,...n->...m", [complex_inverse_phi, complex_steer_vec]
            )  # (B,T,F,M)
            denomin = einsum("...m,...m->...", [complex_steer_vec.conj(), nomin])  # (B,T,F)
            bf_weight = nomin / denomin.unsqueeze(dim=-1)
            bf_weight = torch.stack((bf_weight.real, bf_weight.imag), dim=-1)  # (B,T,F,M,2)
        elif self.bf_type == "generalized":
            x = self.norm(torch.cat((cov_ss, cov_nn), dim=1).permute(0, 3, 2, 1).contiguous())
            x = x.view(b_size * freq_num, seq_len, -1)
            h, _ = self.rnn(x)
            bf_weight = self.w_dnn(h)
            bf_weight = bf_weight.view(b_size, freq_num, seq_len, self.M, 2).transpose(1, 2)
        elif self.bf_type == "embedding":
            x = self.norm(embed_x.permute(0, 3, 2, 1).contiguous())
            x = x.view(b_size * freq_num, seq_len, -1)
            h, _ = self.rnn(x)
            bf_weight = self.w_dnn(h)
            bf_weight = bf_weight.view(b_size, freq_num, seq_len, self.M, 2).transpose(1, 2)
        else:
            raise Exception("only mvdr, generalized, and embedding are supported")
        return bf_weight


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


def complex_conj(inpt):
    """
    inpt: (B,2,...) or (...,2)
    """
    if inpt.shape[1] == 2:
        inpt_r, inpt_i = inpt[:, 0, ...], inpt[:, -1, ...]
        return torch.stack((inpt_r, -inpt_i), dim=1)
    elif inpt.shape[-1] == 2:
        inpt_r, inpt_i = inpt[..., 0], inpt[..., -1]
        return torch.stack((inpt_r, -inpt_i), dim=-1)


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
            self.gain = Variable(torch.ones(1, num_features, 1, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1, 1), requires_grad=False)

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


class CumulativeInstanceNorm2d(nn.Module):
    def __init__(
        self,
        num_features,
        affine=True,
        eps=1e-5,
    ):
        super(CumulativeInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.gain = Variable(torch.ones(1, num_features, 1, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1, 1), requires_grad=False)

    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([3], keepdim=True)  # (B,C,T,1)
        step_pow_sum = inpt.pow(2).sum([3], keepdim=True)  # (B,C,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,C,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,C,T,1)

        entry_cnt = np.arange(freq_num, freq_num * (seq_len + 1), freq_num)
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
            self.gain = Variable(torch.ones(1, num_features, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1), requires_gra=False)

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


# torch_complex/tensor.py's own EPSILON constant (renamed from the vendored
# package's module-level `EPSILON` to avoid clashing with the unrelated
# `utils/utils.py`-scoped `EPSILON` -- not vendored here since only
# `complex_mul`/`complex_conj`/`NormSwitch` are used at runtime).
EPSILON = torch.finfo(torch.float32).eps


class ComplexTensor:
    def __init__(self, real: Union[torch.Tensor, numpy.ndarray], imag=None, device=None):
        if imag is None:
            if isinstance(real, numpy.ndarray):
                if real.dtype.kind == "c":
                    imag = real.imag
                    real = real.real
                else:
                    imag = numpy.zeros_like(real)
            elif isinstance(real, ComplexTensor):
                imag = real.imag
                real = real.real
            else:
                imag = torch.zeros_like(real, device=device)

        if isinstance(real, numpy.ndarray):
            real = torch.as_tensor(real, device=device)
        else:
            real = real.to(device)
        if isinstance(imag, numpy.ndarray):
            imag = torch.as_tensor(imag, device=device)
        else:
            imag = imag.to(device)

        if not torch.is_tensor(real):
            raise TypeError(f"The first arg must be torch.Tensorbut got {type(real)}")

        if not torch.is_tensor(imag):
            raise TypeError(f"The second arg must be torch.Tensorbut got {type(imag)}")
        if not real.size() == imag.size():
            raise ValueError(f"The two inputs must have same sizes: {real.size()} != {imag.size()}")

        self.real = real
        self.imag = imag

    def __getitem__(self, item) -> "ComplexTensor":
        return ComplexTensor(self.real[item], self.imag[item])

    def __setitem__(self, item, value: Union["ComplexTensor", torch.Tensor, numbers.Number]):
        if isinstance(value, (ComplexTensor, complex)):
            self.real[item] = value.real
            self.imag[item] = value.imag
        else:
            self.real[item] = value
            self.imag[item] = 0

    def __mul__(
        self, other: Union["ComplexTensor", torch.Tensor, numbers.Number]
    ) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real,
            )
        else:
            return ComplexTensor(self.real * other, self.imag * other)

    def __rmul__(
        self, other: Union["ComplexTensor", torch.Tensor, numbers.Number]
    ) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(
                other.real * self.real - other.imag * self.imag,
                other.imag * self.real + other.real * self.imag,
            )
        else:
            return ComplexTensor(other * self.real, other * self.imag)

    def __imul__(self, other):
        if isinstance(other, (ComplexTensor, numbers.Complex)):
            t = self * other
            self.real = t.real
            self.imag = t.imag
        else:
            self.real *= other
            self.imag *= other
        return self

    def __truediv__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            den = other.real**2 + other.imag**2 + EPSILON
            return ComplexTensor(
                (self.real * other.real + self.imag * other.imag) / den,
                (-self.real * other.imag + self.imag * other.real) / den,
            )
        else:
            return ComplexTensor(self.real / other, self.imag / other)

    def __rtruediv__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            den = self.real**2 + self.imag**2
            return ComplexTensor(
                (other.real * self.real + other.imag * self.imag) / den,
                (-other.real * self.imag + other.imag * self.real) / den,
            )
        else:
            den = self.real**2 + self.imag**2
            return ComplexTensor(other * self.real / den, -other * self.imag / den)

    def __itruediv__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, numbers.Complex)):
            t = self / other
            self.real = t.real
            self.imag = t.imag
        else:
            self.real /= other
            self.imag /= other
        return self

    def __add__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(self.real + other.real, self.imag + other.imag)
        else:
            return ComplexTensor(self.real + other, self.imag)

    def __radd__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(other.real + self.real, other.imag + self.imag)
        else:
            return ComplexTensor(other + self.real, self.imag)

    def __iadd__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            self.real += other.real
            self.imag += other.imag
        else:
            self.real += other
        return self

    def __sub__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(self.real - other.real, self.imag - other.imag)
        else:
            return ComplexTensor(self.real - other, self.imag)

    def __rsub__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            return ComplexTensor(other.real - self.real, other.imag - self.imag)
        else:
            return ComplexTensor(other - self.real, self.imag)

    def __isub__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, complex)):
            self.real -= other.real
            self.imag -= other.imag
        else:
            self.real -= other
        return self

    def __matmul__(self, other) -> "ComplexTensor":
        if isinstance(other, ComplexTensor):
            o_real = torch.matmul(self.real, other.real) - torch.matmul(self.imag, other.imag)
            o_imag = torch.matmul(self.real, other.imag) + torch.matmul(self.imag, other.real)
        else:
            o_real = torch.matmul(self.real, other)
            o_imag = torch.matmul(self.imag, other)
        return ComplexTensor(o_real, o_imag)

    def __rmatmul__(self, other) -> "ComplexTensor":
        if isinstance(other, ComplexTensor):
            o_real = torch.matmul(other.real, self.real) - torch.matmul(other.imag, self.imag)
            o_imag = torch.matmul(other.real, self.imag) + torch.matmul(other.imag, self.real)
        else:
            o_real = torch.matmul(other, self.real)
            o_imag = torch.matmul(other, self.imag)
        return ComplexTensor(o_real, o_imag)

    def __imatmul__(self, other) -> "ComplexTensor":
        if isinstance(other, (ComplexTensor, numbers.Complex)):
            t = self @ other
            self.real = t.real
            self.imag = t.imag
        else:
            self.real @= other
            self.imag @= other
        return self

    def __neg__(self) -> "ComplexTensor":
        return ComplexTensor(-self.real, -self.imag)

    def __eq__(self, other) -> torch.Tensor:
        if isinstance(other, (ComplexTensor, complex)):
            return (self.real == other.real) ** (self.imag == other.imag)
        else:
            return (self.real == other) ** (self.imag == 0)

    def __len__(self) -> int:
        return len(self.real)

    def __repr__(self) -> str:
        import textwrap

        return (
            "ComplexTensor("
            + "\n    real="
            + textwrap.indent(repr(self.real), " " * len("    real=")).lstrip(" ")
            + ",\n    imag="
            + textwrap.indent(repr(self.imag), " " * len("    imag=")).lstrip(" ")
            + ",\n)"
        )

    def __abs__(self) -> torch.Tensor:
        return (self.real * self.real + self.imag * self.imag).sqrt()

    def __pow__(self, exponent) -> "ComplexTensor":
        if exponent == -2:
            return 1 / (self * self)
        if exponent == -1:
            return 1 / self
        if exponent == 0:
            return ComplexTensor(torch.ones_like(self.real))
        if exponent == 1:
            return self.clone()
        if exponent == 2:
            return self * self

        _abs = self.abs().pow(exponent)
        _angle = exponent * self.angle()
        return ComplexTensor(_abs * torch.cos(_angle), _abs * torch.sin(_angle))

    def __ipow__(self, exponent) -> "ComplexTensor":
        c = self**exponent
        self.real = c.real
        self.imag = c.imag
        return self

    def abs(self) -> torch.Tensor:
        return (self.real * self.real + self.imag * self.imag).sqrt()

    def angle(self) -> torch.Tensor:
        return torch.atan2(self.imag, self.real)

    def backward(self) -> None:
        self.real.backward()
        self.imag.backward()

    def byte(self) -> "ComplexTensor":
        return ComplexTensor(self.real.byte(), self.imag.byte())

    def clone(self) -> "ComplexTensor":
        return ComplexTensor(self.real.clone(), self.imag.clone())

    def conj(self) -> "ComplexTensor":
        return ComplexTensor(self.real, -self.imag)

    def conj_(self) -> "ComplexTensor":
        self.imag.neg_()
        return self

    def contiguous(self) -> "ComplexTensor":
        return ComplexTensor(self.real.contiguous(), self.imag.contiguous())

    def copy_(self) -> "ComplexTensor":
        self.real = self.real.copy_()
        self.imag = self.imag.copy_()
        return self

    def cpu(self) -> "ComplexTensor":
        return ComplexTensor(self.real.cpu(), self.imag.cpu())

    def cuda(self) -> "ComplexTensor":
        return ComplexTensor(self.real.cuda(), self.imag.cuda())

    def expand(self, *sizes):
        return ComplexTensor(self.real.expand(*sizes), self.imag.expand(*sizes))

    def expand_as(self, *args, **kwargs):
        return ComplexTensor(
            self.real.expand_as(*args, **kwargs), self.imag.expand_as(*args, **kwargs)
        )

    def detach(self) -> "ComplexTensor":
        return ComplexTensor(self.real.detach(), self.imag.detach())

    def detach_(self) -> "ComplexTensor":
        self.real.detach_()
        self.imag.detach_()
        return self

    @property
    def device(self):
        assert self.real.device == self.imag.device
        return self.real.device

    def diag(self) -> "ComplexTensor":
        return ComplexTensor(self.real.diag(), self.imag.diag())

    def diagonal(self) -> "ComplexTensor":
        return ComplexTensor(self.real.diag(), self.imag.diag())

    def dim(self) -> int:
        return self.real.dim()

    def double(self) -> "ComplexTensor":
        return ComplexTensor(self.real.double(), self.imag.double())

    @property
    def dtype(self) -> torch.dtype:
        # Warning: Try to never use this dtype property.
        #          It will break your code, when you change to the native
        #          complex type.
        #          Use instead directly `complex_tensor.real.dtype`.
        return self.real.dtype

    def is_floating_point(self):
        return False

    def is_complex(self):
        return True

    def eq(self, other) -> torch.Tensor:
        if isinstance(other, (ComplexTensor, complex)):
            return (self.real == other.real) * (self.imag == other.imag)
        else:
            return (self.real == other) * (self.imag == 0)

    def equal(self, other) -> bool:
        if isinstance(other, (ComplexTensor, complex)):
            return self.real.equal(other.real) and self.imag.equal(other.imag)
        else:
            return self.real.equal(other) and self.imag.equal(0)

    def float(self) -> "ComplexTensor":
        return ComplexTensor(self.real.float(), self.imag.float())

    def fill(self, value) -> "ComplexTensor":
        if isinstance(value, complex):
            return ComplexTensor(self.real.fill(value.real), self.imag.fill(value.imag))
        else:
            return ComplexTensor(self.real.fill(value), self.imag.fill(0))

    def fill_(self, value) -> "ComplexTensor":
        if isinstance(value, complex):
            self.real.fill_(value.real)
            self.imag.fill_(value.imag)
        else:
            self.real.fill_(value)
            self.imag.fill_(0)
        return self

    def gather(self, dim, index) -> "ComplexTensor":
        return ComplexTensor(self.real.gather(dim, index), self.real.gather(dim, index))

    def get_device(self, *args, **kwargs):
        return self.real.get_device(*args, **kwargs)

    def half(self) -> "ComplexTensor":
        return ComplexTensor(self.real.half(), self.imag.half())

    def index_add(self, dim, index, tensor) -> "ComplexTensor":
        return ComplexTensor(
            self.real.index_add(dim, index, tensor),
            self.imag.index_add(dim, index, tensor),
        )

    def index_copy(self, dim, index, tensor) -> "ComplexTensor":
        return ComplexTensor(
            self.real.index_copy(dim, index, tensor),
            self.imag.index_copy(dim, index, tensor),
        )

    def index_fill(self, dim, index, value) -> "ComplexTensor":
        return ComplexTensor(
            self.real.index_fill(dim, index, value),
            self.imag.index_fill(dim, index, value),
        )

    def index_select(self, dim, index) -> "ComplexTensor":
        return ComplexTensor(self.real.index_select(dim, index), self.imag.index_select(dim, index))

    def inverse(self, ntry=5) -> "ComplexTensor":
        # m x n x n
        in_size = self.size()
        a = self.view(-1, self.size(-1), self.size(-1))
        # see "The Matrix Cookbook" (http://www2.imm.dtu.dk/pubdb/p.php?3274)
        # "Section 4.3"
        for i in range(ntry):
            t = i * 0.1

            e = a.real + t * a.imag
            f = a.imag - t * a.real

            try:
                x = torch.matmul(f, e.inverse())
                z = (e + torch.matmul(x, f)).inverse()
            except Exception:
                if i == ntry - 1:
                    raise
                continue

            if t != 0.0:
                eye = torch.eye(a.real.size(-1), dtype=a.real.dtype, device=a.real.device)[None]
                o_real = torch.matmul(z, (eye - t * x))
                o_imag = -torch.matmul(z, (t * eye + x))
            else:
                o_real = z
                o_imag = -torch.matmul(z, x)

            o = ComplexTensor(o_real, o_imag)
            return o.view(*in_size)

    def inverse2(self) -> "ComplexTensor":
        # To avoid cyclic import
        return real_matrix2complex_matrix(complex_matrix2real_matrix(self).inverse())

    def item(self) -> numbers.Number:
        return self.real.item() + 1j * self.imag.item()

    def masked_fill(self, mask, value) -> "ComplexTensor":
        if isinstance(value, complex):
            return ComplexTensor(
                self.real.masked_fill(mask, value.real),
                self.imag.masked_fill(mask, value.imag),
            )

        else:
            return ComplexTensor(self.real.masked_fill(mask, value), self.imag.masked_fill(mask, 0))

    def masked_fill_(self, mask, value) -> "ComplexTensor":
        if isinstance(value, complex):
            self.real.masked_fill_(mask, value.real)
            self.imag.masked_fill_(mask, value.imag)
        else:
            self.real.masked_fill_(mask, value)
            self.imag.masked_fill_(mask, 0)
        return self

    def mean(self, *args, **kwargs) -> "ComplexTensor":
        return ComplexTensor(self.real.mean(*args, **kwargs), self.imag.mean(*args, **kwargs))

    def neg(self) -> "ComplexTensor":
        return ComplexTensor(-self.real, -self.imag)

    def neg_(self) -> "ComplexTensor":
        self.real.neg_()
        self.imag.neg_()
        return self

    def nelement(self) -> int:
        return self.real.nelement()

    def numel(self) -> int:
        return self.real.numel()

    def new(self, *args, **kwargs) -> "ComplexTensor":
        return ComplexTensor(self.real.new(*args, **kwargs), self.imag.new(*args, **kwargs))

    def new_empty(self, size, dtype=None, device=None, requires_grad=False) -> "ComplexTensor":
        real = self.real.new_empty(size, dtype=dtype, device=device, requires_grad=requires_grad)
        imag = self.imag.new_empty(size, dtype=dtype, device=device, requires_grad=requires_grad)
        return ComplexTensor(real, imag)

    def new_full(
        self, size, fill_value, dtype=None, device=None, requires_grad=False
    ) -> "ComplexTensor":
        if isinstance(fill_value, complex):
            real_value = fill_value.real
            imag_value = fill_value.imag
        else:
            real_value = fill_value
            imag_value = 0.0

        real = self.real.new_full(
            size,
            fill_value=real_value,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        imag = self.imag.new_full(
            size,
            fill_value=imag_value,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return ComplexTensor(real, imag)

    def new_tensor(self, data, dtype=None, device=None, requires_grad=False) -> "ComplexTensor":
        if isinstance(data, ComplexTensor):
            real = data.real
            imag = data.imag
        elif isinstance(data, numpy.ndarray):
            if data.dtype.kind == "c":
                real = data.real
                imag = data.imag
            else:
                real = data
                imag = None
        else:
            real = data
            imag = None

        real = self.real.new_tensor(real, dtype=dtype, device=device, requires_grad=requires_grad)
        if imag is None:
            imag = torch.zeros_like(real, dtype=dtype, device=device, requires_grad=requires_grad)
        else:
            imag = self.imag.new_tensor(
                imag, dtype=dtype, device=device, requires_grad=requires_grad
            )
        return ComplexTensor(real, imag)

    def numpy(self) -> numpy.ndarray:
        return self.real.numpy() + 1j * self.imag.numpy()

    def __array__(self):
        # https://numpy.org/devdocs/user/basics.dispatch.html
        return self.real.__array__() + 1j * self.imag.__array__()

    def permute(self, *dims) -> "ComplexTensor":
        return ComplexTensor(self.real.permute(*dims), self.imag.permute(*dims))

    @property
    def T(self):
        return ComplexTensor(self.real.T, self.imag.T)

    def pow(self, exponent) -> "ComplexTensor":
        return self**exponent

    def requires_grad_(self) -> "ComplexTensor":
        self.real.requires_grad_()
        self.imag.requires_grad_()
        return self

    @property
    def requires_grad(self):
        assert self.real.requires_grad == self.imag.requires_grad
        return self.real.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self.real.requires_grad = value
        self.imag.requires_grad = value

    def repeat(self, *sizes):
        return ComplexTensor(self.real.repeat(*sizes), self.imag.repeat(*sizes))

    def reshape(self, *shape):
        return ComplexTensor(self.real.reshape(*shape), self.imag.reshape(*shape))

    def retain_grad(self) -> "ComplexTensor":
        self.real.retain_grad()
        self.imag.retain_grad()
        return self

    def share_memory_(self) -> "ComplexTensor":
        self.real.share_memory_()
        self.imag.share_memory_()
        return self

    @property
    def shape(self) -> torch.Size:
        return self.real.shape

    def size(self, *args, **kwargs) -> torch.Size:
        return self.real.size(*args, **kwargs)

    def ndimension(self):
        return self.real.ndimension()

    @property
    def ndim(self):
        return self.real.ndim

    def sqrt(self) -> "ComplexTensor":
        return self**0.5

    def squeeze(self, dim=None) -> "ComplexTensor":
        if dim is None:
            return ComplexTensor(self.real.squeeze(), self.imag.squeeze())
        else:
            return ComplexTensor(self.real.squeeze(dim), self.imag.squeeze(dim))

    def sum(self, *args, **kwargs) -> "ComplexTensor":
        """
        sum(self, dim, keepdim, *, dtype=None)
        sum(self, axis, keepdims, *, dtype=None)  # numpy style

        Args:
            dim or axis:
            keepdim or keepdims:
            **kwargs:

        Returns:

        """
        return ComplexTensor(self.real.sum(*args, **kwargs), self.imag.sum(*args, **kwargs))

    def take(self, indices) -> "ComplexTensor":
        return ComplexTensor(self.real.take(indices), self.imag.take(indices))

    def to(self, *args, **kwargs) -> "ComplexTensor":
        return ComplexTensor(self.real.to(*args, **kwargs), self.imag.to(*args, **kwargs))

    def tolist(self) -> List[numbers.Number]:
        return [r + 1j * i for r, i in zip(self.real.tolist(), self.imag.tolist())]

    def transpose(self, dim0, dim1) -> "ComplexTensor":
        return ComplexTensor(self.real.transpose(dim0, dim1), self.imag.transpose(dim0, dim1))

    def transpose_(self, dim0, dim1) -> "ComplexTensor":
        self.real.transpose_(dim0, dim1)
        self.imag.transpose_(dim0, dim1)
        return self

    def type(self, *args, **kwargs) -> str:
        if len(args) == 0 and len(kwargs) == 0:
            return self.real.type()
        else:
            return ComplexTensor(self.real.type(*args, **kwargs), self.imag.type(*args, **kwargs))

    def unbind(self, dim=0) -> "ComplexTensor":
        return tuple(
            map(
                lambda x: ComplexTensor(*x),
                zip(self.real.unbind(dim=dim), self.imag.unbind(dim=dim)),
            )
        )

    def unfold(self, dim, size, step):
        return ComplexTensor(self.real.unfold(dim, size, step), self.imag.unfold(dim, size, step))

    def unsqueeze(self, dim) -> "ComplexTensor":
        return ComplexTensor(self.real.unsqueeze(dim), self.imag.unsqueeze(dim))

    def unsqueeze_(self, dim) -> "ComplexTensor":
        self.real.unsqueeze_(dim)
        self.imag.unsqueeze_(dim)
        return self

    def view(self, *args, **kwargs) -> "ComplexTensor":
        return ComplexTensor(self.real.view(*args, **kwargs), self.imag.view(*args, **kwargs))

    def view_as(self, tensor):
        return self.view(tensor.size())


def complex_matrix2real_matrix(c: ComplexTensor) -> torch.Tensor:
    # NOTE(kamo):
    # Complex value can be expressed as follows
    #   a + bi => a * x + b y
    # where
    #   x = |1 0|  y = |0 -1|
    #       |0 1|,     |1  0|
    # A complex matrix can be also expressed as
    #   |A -B|
    #   |B  A|
    # and complex vector can be expressed as
    #   |A|
    #   |B|
    assert c.size(-2) == c.size(-1), c.size()
    # (∗, m, m) -> (*, 2m, 2m)
    return torch.cat(
        [torch.cat([c.real, -c.imag], dim=-1), torch.cat([c.imag, c.real], dim=-1)],
        dim=-2,
    )


def complex_vector2real_vector(c: ComplexTensor) -> torch.Tensor:
    # (∗, m, k) -> (*, 2m, k)
    return torch.cat([c.real, c.imag], dim=-2)


def real_matrix2complex_matrix(c: torch.Tensor) -> ComplexTensor:
    assert c.size(-2) == c.size(-1), c.size()
    # (∗, 2m, 2m) -> (*, m, m)
    n = c.size(-1)
    assert n % 2 == 0, n
    real = c[..., : n // 2, : n // 2]
    imag = c[..., n // 2 :, : n // 2]
    return ComplexTensor(real, imag)


def real_vector2complex_vector(c: torch.Tensor) -> ComplexTensor:
    # (∗, 2m, k) -> (*, m, k)
    n = c.size(-2)
    assert n % 2 == 0, n
    real = c[..., : n // 2, :]
    imag = c[..., n // 2 :, :]
    return ComplexTensor(real, imag)


__all__ = [
    "einsum",
    "cat",
    "stack",
    "pad",
    "squeeze",
    "reverse",
    "trace",
    "allclose",
    "matmul",
    "solve",
]


def _fcomplex(func, nthargs=0):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Union[ComplexTensor, torch.Tensor]:
        signal = args[nthargs]
        if isinstance(signal, ComplexTensor):
            real_args = args[:nthargs] + (signal.real,) + args[nthargs + 1 :]
            imag_args = args[:nthargs] + (signal.imag,) + args[nthargs + 1 :]
            real = func(*real_args, **kwargs)
            imag = func(*imag_args, **kwargs)
            return ComplexTensor(real, imag)
        else:
            return func(*args, **kwargs)

    return wrapper


def einsum(equation, *operands):
    """Einsum

    >>> import numpy
    >>> def get(*shape):
    ...     real = numpy.random.rand(*shape)
    ...     imag = numpy.random.rand(*shape)
    ...     return real + 1j * imag
    >>> x = get(3, 4, 5)
    >>> y = get(3, 5, 6)
    >>> z = get(3, 6, 7)
    >>> test = einsum('aij,ajk,akl->ail',
    ...               [ComplexTensor(x), ComplexTensor(y), ComplexTensor(z)])
    >>> valid = numpy.einsum('aij,ajk,akl->ail', x, y, z)
    >>> numpy.testing.assert_allclose(test.numpy(), valid)
    >>> _ = einsum('aij->ai', ComplexTensor(x))
    >>> _ = einsum('aij->ai', [ComplexTensor(x)])

    """
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = operands[0]

    x = operands[0]
    if isinstance(x, ComplexTensor):
        real_operands = [[x.real]]
        imag_operands = [[x.imag]]
    else:
        real_operands = [[x]]
        imag_operands = []

    for x in operands[1:]:
        if isinstance(x, ComplexTensor):
            real_operands, imag_operands = (
                [ops + [x.real] for ops in real_operands]
                + [ops + [-x.imag] for ops in imag_operands],
                [ops + [x.imag] for ops in real_operands]
                + [ops + [x.real] for ops in imag_operands],
            )
        else:
            real_operands = [ops + [x] for ops in real_operands]
            imag_operands = [ops + [x] for ops in imag_operands]

    real = sum([torch.einsum(equation, ops) for ops in real_operands])
    imag = sum([torch.einsum(equation, ops) for ops in imag_operands])
    return ComplexTensor(real, imag)


def cat(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    """
    cat(seq, dim=0, *, out=None)
    cat(seq, axis=0, *, out=None)
    """
    reals = [v.real if isinstance(v, ComplexTensor) else v for v in seq]
    imags = [v.imag if isinstance(v, ComplexTensor) else torch.zeros_like(v.real) for v in seq]
    out = kwargs.pop("out", None)
    if out is not None:
        out = out
        out_real = out.real
        out_imag = out.imag
    else:
        out_real = out_imag = None
    return ComplexTensor(
        torch.cat(reals, *args, out=out_real, **kwargs),
        torch.cat(imags, *args, out=out_imag, **kwargs),
    )


def stack(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    """
    stack(tensors, dim=0, * out=None)
    stack(tensors, axis=0, * out=None)

    """
    reals = [v.real if isinstance(v, ComplexTensor) else v for v in seq]
    imags = [v.imag if isinstance(v, ComplexTensor) else torch.zeros_like(v.real) for v in seq]

    out = kwargs.pop("out", None)
    if out is not None:
        out_real = out.real
        out_imag = out.imag
    else:
        out_real = out_imag = None
    return ComplexTensor(
        torch.stack(reals, *args, out=out_real, **kwargs),
        torch.stack(imags, *args, out=out_imag, **kwargs),
    )


pad = _fcomplex(tc_F.pad)
squeeze = _fcomplex(torch.squeeze)


@_fcomplex
def reverse(tensor: torch.Tensor, dim=0) -> torch.Tensor:
    # https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382
    idx = [i for i in range(tensor.size(dim) - 1, -1, -1)]
    idx = torch.LongTensor(idx).to(tensor.device)
    inverted_tensor = tensor.index_select(dim, idx)
    return inverted_tensor


@_fcomplex
def signal_frame(
    signal: torch.Tensor, frame_length: int, frame_step: int, pad_value=0
) -> torch.Tensor:
    """Expands signal into frames of frame_length.

    Args:
        signal : (B * F, D, T)
    Returns:
        torch.Tensor: (B * F, D, T, W)
    """
    signal = tc_F.pad(signal, (0, frame_length - 1), "constant", pad_value)
    indices = sum(
        [
            list(range(i, i + frame_length))
            for i in range(0, signal.size(-1) - frame_length + 1, frame_step)
        ],
        [],
    )

    signal = signal[..., indices].view(*signal.size()[:-1], -1, frame_length)
    return signal


def trace(a: ComplexTensor) -> ComplexTensor:
    if LooseVersion(torch.__version__) >= LooseVersion("1.3"):
        datatype = torch.bool
    else:
        datatype = torch.uint8
    E = torch.eye(a.shape[-1], dtype=datatype).expand(*a.size())
    if LooseVersion(torch.__version__) >= LooseVersion("1.1"):
        E = E.type(torch.bool)
    return a[E].view(*a.size()[:-1]).sum(-1)


def allclose(
    a: Union[ComplexTensor, torch.Tensor],
    b: Union[ComplexTensor, torch.Tensor],
    rtol=1e-05,
    atol=1e-08,
    equal_nan=False,
) -> bool:
    if isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        return torch.allclose(
            a.real, b.real, rtol=rtol, atol=atol, equal_nan=equal_nan
        ) and torch.allclose(a.imag, b.imag, rtol=rtol, atol=atol, equal_nan=equal_nan)
    elif not isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        return torch.allclose(
            a.real, b.real, rtol=rtol, atol=atol, equal_nan=equal_nan
        ) and torch.allclose(
            torch.zeros_like(b.imag), b.imag, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    elif isinstance(a, ComplexTensor) and not isinstance(b, ComplexTensor):
        return torch.allclose(
            a.real, b, rtol=rtol, atol=atol, equal_nan=equal_nan
        ) and torch.allclose(
            a.imag, torch.zeros_like(a.imag), rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    else:
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def matmul(
    a: Union[ComplexTensor, torch.Tensor], b: Union[ComplexTensor, torch.Tensor]
) -> ComplexTensor:
    if isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        return a @ b
    elif not isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
        o_real = torch.matmul(a, b.real)
        o_imag = torch.matmul(a, b.imag)
    elif isinstance(a, ComplexTensor) and not isinstance(b, ComplexTensor):
        return a @ b
    else:
        o_real = torch.matmul(a.real, b.real)
        o_imag = torch.zeros_like(o_real)
    return ComplexTensor(o_real, o_imag)


def solve(b: ComplexTensor, a: ComplexTensor) -> ComplexTensor:
    """Solve ax = b"""
    a = complex_matrix2real_matrix(a)
    b = complex_vector2real_vector(b)
    x, LU = torch.solve(b, a)
    return real_vector2complex_vector(x), real_matrix2complex_matrix(LU)


class _TaylorBeamformerInferenceWrapper(TaylorBeamformer):
    """Thin compatibility shim, not an architectural change.

    ``ZeroOrderBlock.forward`` (vendored verbatim above) does
    ``x_acc = Variable(torch.zeros(x.size()), requires_grad=True).to(x.device)``
    followed by an in-place ``x_acc += x``. That is a leaf tensor with
    ``requires_grad=True`` mutated in place -- on any modern PyTorch (long
    after the ``Variable`` API was folded into ``Tensor``) that always raises
    ``RuntimeError: a leaf Variable that requires grad is being used in an
    in-place operation``, independent of TorchLens; it reproduces with the
    same three lines outside any tracing tool. The real repo's own training
    script never hits it only because ``x_acc``'s ``requires_grad=True`` is
    vestigial there too (the surrounding forward pass runs under whatever
    autograd context the caller sets up). Wrapping the forward call in
    ``torch.no_grad()`` here (an inference-only wrapper, not a modification of
    any layer or op) is what is required to run this real model's forward
    pass at all on current torch -- it changes no weights, no layers, no
    control flow, only whether autograd bookkeeping is active.
    """

    def forward(self, inpt):
        with torch.no_grad():
            return super().forward(inpt)


def build_taylorbeamformer():
    torch.manual_seed(0)
    # Matches the real repo's own `if __name__ == "__main__":` demo config in
    # nets/TaylorBeamformer.py (also the shipped configs/train_config.toml
    # defaults), with `hid_node` shrunk for a fast trace -- the frequency-axis
    # size below must stay large enough to survive the U2-Net's 4 internal
    # scale-1..4 downsampling stages plus the outer encoder's 5 stride-(1,2)
    # stages, so `fft_num` is kept at the real default (320). `c`/`d_feat` are
    # also kept at the real defaults: both `UNet_Encoder`/`U2Net_Encoder`
    # hardcode their final stage to 64 channels (`c_last`/`c_final = 64`,
    # independent of the `c` arg), and `d_feat` must equal `c_last * 4` (the
    # encoder's flattened freq dim) for the TCM stage's channel count to line
    # up -- shrinking either independently breaks that real-code invariant.
    model = _TaylorBeamformerInferenceWrapper(
        k1=[1, 3],
        k2=[2, 3],
        ref_mic=0,
        c=8,
        embed_dim=8,
        fft_num=320,
        order_num=2,
        kd1=5,
        cd1=8,
        d_feat=256,
        dilations=[1, 2],
        group_num=2,
        hid_node=8,
        M=2,
        rnn_type="LSTM",
        intra_connect="cat",
        inter_connect="cat",
        out_type="mapping",
        bf_type="embedding",
        norm2d_type="BN",
        norm1d_type="BN",
        is_compress=True,
        is_total_separate=False,
        is_u2=True,
        is_1dgate=True,
        is_squeezed=True,
        is_causal=True,
        is_param_share=False,
    )
    model.eval()
    return model


def example_input_taylorbeamformer():
    torch.manual_seed(0)
    # (B, T, F, M, 2): batch, time frames, freq bins ((fft_num//2)+1=161), mics, real/imag
    return torch.rand(1, 6, 161, 2, 2)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "TaylorBeamformer",
        "build_taylorbeamformer",
        "example_input_taylorbeamformer",
        2022,
        MENAGERIE_ZOO,
    ),
]

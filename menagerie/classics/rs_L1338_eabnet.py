# SOURCE: vendored from Andong-Li-speech/EaBNet @ 719f764bbc5128df8cf5eaf59d91e7a2a577939d
# (EaBNet.py -- full file: EaBNet, U2Net_Encoder/Decoder, UNet_Encoder/Decoder, En_unet_module,
#  Conv2dunit, Deconv2dunit, GateConv2d, GateConvTranspose2d, Skip_connect, SqueezedTCNGroup,
#  SqueezedTCM, LSTM_BF, Chomp_T, NormSwitch, CumulativeLayerNorm1d/2d)
"""Embedding and Beamforming Network (EaBNet) for multichannel speech enhancement
(Li, Liang, Yu & Wang, "Embedding and Beamforming: All-neural Causal Beamformer for
Multichannel Speech Enhancement", ICASSP 2022, arXiv:2109.00265). Official repo:
https://github.com/Andong-Li-speech/EaBNet (``EaBNet.py`` @ main).

Two-stage architecture operating on multichannel STFT input ``(B, T, F, M, 2)``: (1) an
embedding stage -- a U^2-Net-style (or plain U-Net) encoder/decoder over stacked
Squeezed Temporal Convolution Modules (TCN groups with dilated causal convs) producing a
per-time-frequency embedding, then (2) a beamforming stage (an LSTM-based or pointwise-CNN
beamforming-weight predictor) that maps the embedding to complex beamforming weights,
applied to the raw multichannel input to produce the enhanced single-channel complex
spectrogram estimate. Every module (gated 2-D convs, the nested En_unet_module U-Net
blocks, squeezed TCN groups with cumulative/instance/batch norm switches, and the LSTM
beamformer) is vendored verbatim from the official single-file repo; no layer, channel
count, or forward-pass control-flow was changed. Only the CLI-only ``main()``/``argparse``
harness at the bottom of the original file was dropped (not part of the model).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable


class EaBNet(nn.Module):
    def __init__(
        self,
        k1: tuple = (2, 3),
        k2: tuple = (1, 3),
        c: int = 64,
        M: int = 9,
        embed_dim: int = 64,
        kd1: int = 5,
        cd1: int = 64,
        d_feat: int = 256,
        p: int = 6,
        q: int = 3,
        is_causal: bool = True,
        is_u2: bool = True,
        bf_type: str = "lstm",
        topo_type: str = "mimo",
        intra_connect: str = "cat",
        norm_type: str = "IN",
    ):
        """
        :param k1: kernel size in the 2-D GLU, (2, 3) by default
        :param k2: kernel size in the UNet-blok, (1, 3) by defauly
        :param c: channel number in the 2-D Convs, 64 by default
        :param M: mic number, 9 by default
        :param embed_dim: embedded dimension, 64 by default
        :param kd1: kernel size in the Squeezed-TCM (dilation-part), 5 by default
        :param cd1: channel number in the Squeezed-TCM (dilation-part), 64 by default
        :param d_feat: channel number in the Squeezed-TCM(pointwise-part), 256 by default
        :param p: the number of Squeezed-TCMs within a group, 6 by default
        :param q: group numbers, 3 by default
        :param is_causal: causal flag, True by default
        :param is_u2: whether U^{2} is set, True by default
        :param bf_type: beamformer type, "lstm" by default
        :param topo_type: topology type, "mimo" and "miso", "mimo" by default
        :param intra_connect: intra connection type, "cat" by default
        :param norm_type: "IN" by default.
        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.M = M
        self.embed_dim = embed_dim
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.q = q
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.bf_type = bf_type
        self.intra_connect = intra_connect
        self.topo_type = topo_type
        self.norm_type = norm_type

        if is_u2:
            self.en = U2Net_Encoder(M * 2, k1, k2, c, intra_connect, norm_type)
            self.de = U2Net_Decoder(embed_dim, c, k1, k2, intra_connect, norm_type)
        else:
            self.en = UNet_Encoder(M * 2, k1, c, norm_type)
            self.de = UNet_Decoder(embed_dim, k1, c, norm_type)

        if topo_type == "mimo":
            if bf_type == "lstm":
                self.bf_map = LSTM_BF(embed_dim, M)
            elif bf_type == "cnn":
                self.bf_map = nn.Conv2d(embed_dim, M * 2, (1, 1), (1, 1))  # pointwise
        elif topo_type == "miso":
            self.bf_map = nn.Conv2d(embed_dim, 2, (1, 1), (1, 1))  # pointwise

        stcn_list = []
        for _ in range(q):
            stcn_list.append(SqueezedTCNGroup(kd1, cd1, d_feat, p, is_causal, norm_type))
        self.stcns = nn.ModuleList(stcn_list)

    def forward(self, inpt: Tensor) -> Tensor:
        """
        :param inpt: (B, T, F, M, 2) -> (batchsize, seqlen, freqsize, mics, 2)
        :return: beamformed estimation: (B, 2, T, F)
        """
        if inpt.ndim == 4:
            inpt = inpt.unsqueeze(dim=-2)
        b_size, seq_len, freq_len, M, _ = inpt.shape
        x = inpt.transpose(-2, -1).contiguous()
        x = x.view(b_size, seq_len, freq_len, -1).permute(0, 3, 1, 2)
        x, en_list = self.en(x)
        c = x.shape[1]
        x = x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x_acc = Variable(torch.zeros(x.size()), requires_grad=True).to(x.device)
        for i in range(len(self.stcns)):
            x = self.stcns[i](x)
            x_acc = x_acc + x
        x = x_acc
        x = x.view(b_size, c, -1, seq_len).transpose(-2, -1).contiguous()
        x = self.de(x, en_list)
        if self.topo_type == "mimo":
            if self.bf_type == "lstm":
                bf_w = self.bf_map(x)  # (B, T, F, M, 2)
            elif self.bf_type == "cnn":
                bf_w = self.bf_map(x)
                bf_w = bf_w.view(b_size, M, -1, seq_len, freq_len).permute(
                    0, 3, 4, 1, 2
                )  # (B,T,F,M,2)
            bf_w_r, bf_w_i = bf_w[..., 0], bf_w[..., -1]
            esti_x_r, esti_x_i = (
                (bf_w_r * inpt[..., 0] - bf_w_i * inpt[..., -1]).sum(dim=-1),
                (bf_w_r * inpt[..., -1] + bf_w_i * inpt[..., 0]).sum(dim=-1),
            )
            return torch.stack((esti_x_r, esti_x_i), dim=1)
        elif self.topo_type == "miso":
            bf_w = self.bf_map(x)  # (B,2,T,F)
            bf_w = bf_w.permute(0, 2, 3, 1)  # (B,T,F,2)
            bf_w_r, bf_w_i = bf_w[..., 0], bf_w[..., -1]
            # mic-0 is selected as the target mic herein
            esti_x_r, esti_x_i = (
                (bf_w_r * inpt[..., 0, 0] - bf_w_i * inpt[..., 0, -1]).sum(dim=-1),
                (bf_w_r * inpt[..., 0, -1] + bf_w_i * inpt[..., 0, 0]).sum(dim=-1),
            )
            return torch.stack((esti_x_r, esti_x_i), dim=1)


class U2Net_Encoder(nn.Module):
    def __init__(self, cin: int, k1: tuple, k2: tuple, c: int, intra_connect: str, norm_type: str):
        super().__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        k_beg = (2, 5)
        c_end = 64
        meta_unet = []
        meta_unet.append(
            En_unet_module(cin, c, k_beg, k2, intra_connect, norm_type, scale=4, is_deconv=False)
        )
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=3, is_deconv=False)
        )
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=2, is_deconv=False)
        )
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type, scale=1, is_deconv=False)
        )
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_end, k1, (1, 2)),
            NormSwitch(norm_type, "2D", c_end),
            nn.PReLU(c_end),
        )

    def forward(self, x: Tensor):
        en_list = []
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
            en_list.append(x)
        x = self.last_conv(x)
        en_list.append(x)
        return x, en_list


class UNet_Encoder(nn.Module):
    def __init__(self, cin: int, k1: tuple, c: int, norm_type: str):
        super().__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        k_beg = (2, 5)
        c_end = 64
        unet = []
        unet.append(
            nn.Sequential(
                GateConv2d(cin, c, k_beg, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c)
            )
        )
        unet.append(nn.Sequential(GateConv2d(c, c, k1, (1, 2)), nn.PReLU(c)))
        unet.append(nn.Sequential(GateConv2d(c, c, k1, (1, 2)), nn.PReLU(c)))
        unet.append(
            nn.Sequential(GateConv2d(c, c, k1, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c))
        )
        unet.append(
            nn.Sequential(
                GateConv2d(c, c_end, k1, (1, 2)),
                NormSwitch(norm_type, "2D", c_end),
                nn.PReLU(c_end),
            )
        )
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor):
        en_list = []
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
            en_list.append(x)
        return x, en_list


class U2Net_Decoder(nn.Module):
    def __init__(self, embed_dim, c, k1, k2, intra_connect, norm_type):
        super().__init__()
        self.embed_dim = embed_dim
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        c_beg = 64
        k_end = (2, 5)

        meta_unet = []
        meta_unet.append(
            En_unet_module(c_beg * 2, c, k1, k2, intra_connect, norm_type, scale=1, is_deconv=True)
        )
        meta_unet.append(
            En_unet_module(c * 2, c, k1, k2, intra_connect, norm_type, scale=2, is_deconv=True)
        )
        meta_unet.append(
            En_unet_module(c * 2, c, k1, k2, intra_connect, norm_type, scale=3, is_deconv=True)
        )
        meta_unet.append(
            En_unet_module(c * 2, c, k1, k2, intra_connect, norm_type, scale=4, is_deconv=True)
        )
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConvTranspose2d(c * 2, embed_dim, k_end, (1, 2)),
            NormSwitch(norm_type, "2D", embed_dim),
            nn.PReLU(embed_dim),
        )

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        for i in range(len(self.meta_unet_list)):
            tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)
            x = self.meta_unet_list[i](tmp)
        x = torch.cat((x, en_list[0]), dim=1)
        x = self.last_conv(x)
        return x


class UNet_Decoder(nn.Module):
    def __init__(self, embed_dim: int, k1: tuple, c: int, norm_type: str):
        super().__init__()
        self.embed_dim = embed_dim
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        c_beg = (
            64  # the channels of the last encoder and the first decoder are fixed at 64 by default
        )
        k_end = (2, 5)
        unet = []
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c_beg * 2, c, k1, (1, 2)),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * 2, c, k1, (1, 2)),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * 2, c, k1, (1, 2)),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * 2, c, k1, (1, 2)),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c),
            )
        )
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c * 2, embed_dim, k_end, (1, 2)),
                NormSwitch(norm_type, "2D", embed_dim),
                nn.PReLU(embed_dim),
            )
        )
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        for i in range(len(self.unet_list)):
            tmp = torch.cat((x, en_list[-(i + 1)]), dim=1)  # skip connections
            x = self.unet_list[i](tmp)
        return x


class En_unet_module(nn.Module):
    def __init__(
        self,
        cin: int,
        cout: int,
        k1: tuple,
        k2: tuple,
        intra_connect: str,
        norm_type: str,
        scale: int,
        is_deconv: bool,
    ):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.cin = cin
        self.cout = cout
        self.intra_connect = intra_connect
        self.scale = scale
        self.is_deconv = is_deconv

        in_conv_list = []
        if not is_deconv:
            in_conv_list.append(GateConv2d(cin, cout, k1, (1, 2)))
        else:
            in_conv_list.append(GateConvTranspose2d(cin, cout, k1, (1, 2)))
        in_conv_list.append(NormSwitch(norm_type, "2D", cout))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, cout, norm_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, "add", norm_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, intra_connect, norm_type))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = Skip_connect(intra_connect)

    def forward(self, x):
        x_resi = self.in_conv(x)
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
    def __init__(self, k: tuple, c: int, norm_type: str):
        super().__init__()
        self.k = k
        self.c = c
        self.norm_type = norm_type
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, k, (1, 2)), NormSwitch(norm_type, "2D", c), nn.PReLU(c)
        )

    def forward(self, x):
        return self.conv(x)


class Deconv2dunit(nn.Module):
    def __init__(self, k: tuple, c: int, intra_connect: str, norm_type: str):
        super().__init__()
        self.k, self.c = k, c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        deconv_list = []
        if self.intra_connect == "add":
            deconv_list.append(nn.ConvTranspose2d(c, c, k, (1, 2)))
        elif self.intra_connect == "cat":
            deconv_list.append(nn.ConvTranspose2d(2 * c, c, k, (1, 2)))
        (deconv_list.append(NormSwitch(norm_type, "2D", c)),)
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)


class GateConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d((0, 0, k_t - 1, 0), value=0.0),  # for causal-setting
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
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class GateConvTranspose2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple):
        super().__init__()
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
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class Skip_connect(nn.Module):
    def __init__(self, connect):
        super().__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == "add":
            x = x_main + x_aux
        elif self.connect == "cat":
            x = torch.cat((x_main, x_aux), dim=1)
        return x


class SqueezedTCNGroup(nn.Module):
    def __init__(self, kd1: int, cd1: int, d_feat: int, p: int, is_causal: bool, norm_type: str):
        super().__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.is_causal = is_causal
        self.norm_type = norm_type

        # Components
        self.tcm_list = nn.ModuleList(
            [SqueezedTCM(kd1, cd1, 2**i, d_feat, is_causal, norm_type) for i in range(p)]
        )

    def forward(self, x):
        for i in range(self.p):
            x = self.tcm_list[i](x)
        return x


class SqueezedTCM(nn.Module):
    def __init__(
        self, kd1: int, cd1: int, dilation: int, d_feat: int, is_causal: bool, norm_type: str
    ):
        super().__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.is_causal = is_causal
        self.norm_type = norm_type

        self.in_conv = nn.Conv1d(d_feat, cd1, 1, bias=False)
        if is_causal:
            pad = ((kd1 - 1) * dilation, 0)
        else:
            pad = ((kd1 - 1) * dilation // 2, (kd1 - 1) * dilation // 2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.ConstantPad1d(pad, value=0.0),
            nn.Conv1d(cd1, cd1, kd1, dilation=dilation, bias=False),
        )
        self.right_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.ConstantPad1d(pad, value=0.0),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False),
        )

    def forward(self, x):
        resi = x
        x = self.in_conv(x)
        x = self.left_conv(x) * self.right_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class LSTM_BF(nn.Module):
    def __init__(self, embed_dim: int, M: int, hid_node: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.M = M
        self.hid_node = hid_node
        # Components
        self.rnn1 = nn.LSTM(input_size=embed_dim, hidden_size=hid_node, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=hid_node, hidden_size=hid_node, batch_first=True)
        self.w_dnn = nn.Sequential(
            nn.Linear(hid_node, hid_node), nn.ReLU(True), nn.Linear(hid_node, 2 * M)
        )
        self.norm = nn.LayerNorm([embed_dim])

    def forward(self, embed_x: Tensor) -> Tensor:
        """
        formulate the bf operation
        :param embed_x: (B, C, T, F)
        :return: (B, T, F, M, 2)
        """
        # norm
        B, _, T, F = embed_x.shape
        x = self.norm(embed_x.permute(0, 3, 2, 1).contiguous())
        x = x.view(B * F, T, -1)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x.view(B, F, T, -1).transpose(1, 2).contiguous()
        bf_w = self.w_dnn(x).view(B, T, F, self.M, 2)
        return bf_w


class Chomp_T(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x):
        return x[:, :, : -self.t, :]


class NormSwitch(nn.Module):
    """
    Currently, BN, IN, and cLN are considered
    """

    def __init__(self, norm_type: str, dim_size: str, c: int):
        super().__init__()
        self.norm_type = norm_type
        self.dim_size = dim_size
        self.c = c

        assert norm_type in ["BN", "IN", "cLN"] and dim_size in ["1D", "2D"]
        if norm_type == "BN":
            if dim_size == "1D":
                self.norm = nn.BatchNorm1d(c)
            else:
                self.norm = nn.BatchNorm2d(c)
        elif norm_type == "IN":
            if dim_size == "1D":
                self.norm = nn.InstanceNorm1d(c, affine=True)
            else:
                self.norm = nn.InstanceNorm2d(c, affine=True)
        elif norm_type == "cLN":
            if dim_size == "1D":
                self.norm = CumulativeLayerNorm1d(dim_size, affine=True)
            else:
                self.norm = CumulativeLayerNorm2d(dim_size, affine=True)

    def forward(self, x):
        return self.norm(x)


class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, num_features, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, num_features, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, num_features, 1), requires_grad=False)

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


class CumulativeLayerNorm2d(nn.Module):
    def __init__(self, num_features, affine=True, eps=1e-5):
        super().__init__()
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


MENAGERIE_ZOO = "vendored-pytorch"


def build_eabnet():
    # Small config for a fast trace: 2 mics, tiny embed/beamformer channel counts,
    # is_u2=True (default), single squeezed-TCN group with 2 blocks, causal + IN norm
    # (repo defaults for the architectural switches). `d_feat` must equal
    # `c_end * F_reduced` where `c_end=64` is the encoder's (hardcoded in the repo)
    # final channel count and `F_reduced` is the frequency dim after the encoder's
    # stride-2 downsampling -- with the example input's F=65 that reduces to 1, so
    # d_feat=64 (see example_input_eabnet).
    return EaBNet(
        k1=(2, 3),
        k2=(1, 3),
        c=4,
        M=2,
        embed_dim=4,
        kd1=3,
        cd1=4,
        d_feat=64,
        p=2,
        q=1,
        is_causal=True,
        is_u2=True,
        bf_type="lstm",
        topo_type="mimo",
        intra_connect="cat",
        norm_type="IN",
    )


def example_input_eabnet():
    # (B, T, F, M, 2): batch, time frames, freq bins, mics, real/imag. F must be large
    # enough to survive the encoder's repeated freq-axis stride-2 downsampling (outer
    # U-Net stages + each En_unet_module's internal `scale` downsamples); F=65 reduces
    # to F_reduced=1 through the encoder, matching d_feat=64 above.
    return (torch.randn(1, 6, 65, 2, 2),)


MENAGERIE_ENTRIES = [
    ("EaBNet", "build_eabnet", "example_input_eabnet", 2022, "vendored-pytorch"),
]

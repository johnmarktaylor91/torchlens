# SOURCE: vendored from https://github.com/aispeech-lab/LiMuSE @ main (0e9c3480)
# (LiMuSE.py :: LiMuSE and its building blocks; utils.py :: TAC/TAC_Q/split_feature/
#  merge_feature/pad_segment; min_max_quantization.py :: min_max_quantize/RoundFunction)
#
# LiMuSE: Lightweight Multi-modal Speaker Extraction (Liu et al., 2021).
# https://github.com/aispeech-lab/LiMuSE
#
# LiMuSE is a real, self-contained PyTorch model: a group-communication (TAC)
# temporal-convolutional separator that fuses an audio mixture with a voiceprint
# (auxiliary speaker embedding) and lip-motion (visual) embedding to extract a
# single target speaker's waveform. The classes below are copied verbatim from
# the real repo's LiMuSE.py, plus the handful of helper classes/functions from
# utils.py that LiMuSE.py's `from utils import *` actually pulls in for the model
# forward pass (TAC, TAC_Q, pad_segment, split_feature, merge_feature), plus the
# quantization helpers from min_max_quantization.py. The rest of the real utils.py
# (config/checkpoint I/O, bss_eval scoring, clustering, logging -- needing
# soundfile/sklearn/scipy/yaml) is training/eval scaffolding unrelated to the
# model definition and was dropped; no architecture code was altered.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

MENAGERIE_ZOO = "vendored-pytorch"


# --------------------------------------------------------------------------
# from min_max_quantization.py (verbatim)
# --------------------------------------------------------------------------
class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def min_max_quantize(x, k):
    n = 2**k
    a = torch.min(x)
    b = torch.max(x)
    s = (b - a) / (n - 1)

    x = torch.clamp(x, float(a), float(b))
    x = (x - a) / s
    x = RoundFunction.apply(x)
    x = x * s + a
    return x


# --------------------------------------------------------------------------
# from utils.py (verbatim, model-only subset)
# --------------------------------------------------------------------------
class TAC(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TAC, self).__init__()

        self.TAC_input = nn.Sequential(nn.Linear(input_size, hidden_size), nn.PReLU())
        self.TAC_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.PReLU())
        self.TAC_output = nn.Sequential(nn.Linear(hidden_size * 2, input_size), nn.PReLU())
        self.TAC_norm = nn.GroupNorm(1, input_size)

    def forward(self, input):
        # input shape: batch, group, N, seq_length

        batch_size, G, N, T = input.shape
        output = input

        # transform
        group_input = output  # B, G, N, T
        group_input = output.permute(0, 3, 1, 2).contiguous().view(-1, N)  # B*T*G, N
        group_output = self.TAC_input(group_input).view(batch_size, T, G, -1)  # B, T, G, H

        # mean pooling
        group_mean = group_output.mean(2).view(batch_size * T, -1)  # B*T, H

        # concate
        group_output = group_output.view(batch_size * T, G, -1)  # B*T, G, H
        group_mean = (
            self.TAC_mean(group_mean).unsqueeze(1).expand_as(group_output).contiguous()
        )  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H
        group_output = self.TAC_output(group_output.view(-1, group_output.shape[-1]))  # B*T*G, N
        group_output = (
            group_output.view(batch_size, T, G, -1).permute(0, 2, 3, 1).contiguous()
        )  # B, G, N, T
        group_output = self.TAC_norm(group_output.view(batch_size * G, N, T))  # B*G, N, T
        output = output + group_output.view(input.shape)

        return output


class TAC_Q(nn.Module):
    def __init__(self, input_size, hidden_size, QA_flag=False, ak=8):
        super(TAC_Q, self).__init__()

        self.TAC_input = nn.Sequential(nn.Linear(input_size, hidden_size), nn.PReLU())
        self.TAC_mean = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.PReLU())
        self.TAC_output = nn.Sequential(nn.Linear(hidden_size * 2, input_size), nn.PReLU())
        self.TAC_norm = nn.GroupNorm(1, input_size)

        self.QA_flag = QA_flag
        self.ak = ak

    def forward(self, input):
        # input shape: batch, group, N, seq_length

        batch_size, G, N, T = input.shape
        output = input

        # transform
        group_input = output  # B, G, N, T
        group_input = output.permute(0, 3, 1, 2).contiguous().view(-1, N)  # B*T*G, N

        if self.QA_flag:
            group_input = min_max_quantize(group_input, self.ak)
        group_output = self.TAC_input(group_input).view(batch_size, T, G, -1)  # B, T, G, H

        # mean pooling
        group_mean = group_output.mean(2).view(batch_size * T, -1)  # B*T, H

        # concate
        group_output = group_output.view(batch_size * T, G, -1)  # B*T, G, H

        if self.QA_flag:
            group_output = min_max_quantize(group_output, self.ak)
        group_mean = (
            self.TAC_mean(group_mean).unsqueeze(1).expand_as(group_output).contiguous()
        )  # B*T, G, H
        group_output = torch.cat([group_output, group_mean], 2)  # B*T, G, 2H

        if self.QA_flag:
            group_output = min_max_quantize(group_output, self.ak)
        group_output = self.TAC_output(group_output.view(-1, group_output.shape[-1]))  # B*T*G, N
        group_output = (
            group_output.view(batch_size, T, G, -1).permute(0, 2, 3, 1).contiguous()
        )  # B, G, N, T
        group_output = self.TAC_norm(group_output.view(batch_size * G, N, T))  # B*G, N, T
        output = output + group_output.view(input.shape)

        return output


def pad_segment(input, block_size):
    # input is the features: (B, N, T)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    rest = block_size - (block_stride + seq_len % block_size) % block_size
    if rest > 0:
        pad = Variable(torch.zeros(batch_size, dim, rest)).type(input.type()).to(input.device)
        input = torch.cat([input, pad], 2)

    pad_aux = (
        Variable(torch.zeros(batch_size, dim, block_stride)).type(input.type()).to(input.device)
    )
    input = torch.cat([pad_aux, input, pad_aux], 2)

    return input, rest


def split_feature(input, block_size):
    # split the feature into chunks of segment size
    # input is the features: (B, N, T)

    input, rest = pad_segment(input, block_size)
    batch_size, dim, seq_len = input.shape
    block_stride = block_size // 2

    block1 = input[:, :, :-block_stride].contiguous().view(batch_size, dim, -1, block_size)
    block2 = input[:, :, block_stride:].contiguous().view(batch_size, dim, -1, block_size)
    block = torch.cat([block1, block2], 3).view(batch_size, dim, -1, block_size).transpose(2, 3)

    return block.contiguous(), rest


def merge_feature(input, rest):
    # merge the splitted features into full utterance
    # input is the features: (B, N, L, K)

    batch_size, dim, block_size, _ = input.shape
    block_stride = block_size // 2
    input = (
        input.transpose(2, 3).contiguous().view(batch_size, dim, -1, block_size * 2)
    )  # B, N, K, L

    input1 = input[:, :, :, :block_size].contiguous().view(batch_size, dim, -1)[:, :, block_stride:]
    input2 = (
        input[:, :, :, block_size:].contiguous().view(batch_size, dim, -1)[:, :, :-block_stride]
    )

    output = input1 + input2
    if rest > 0:
        output = output[:, :, :-rest]

    return output.contiguous()  # B, N, T


# --------------------------------------------------------------------------
# from LiMuSE.py (verbatim)
# --------------------------------------------------------------------------
class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = super().forward(x)
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm2d(dim)


class Conv1D_Q(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, QA_flag=False, ak=8):
        super(Conv1D_Q, self).__init__()

        self.QA_flag = QA_flag
        self.ak = ak
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, kernel)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))

        if self.QA_flag:
            x = min_max_quantize(x, self.ak)

        output = self.conv1d(x)
        return output


class Conv1D(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x, dim=1)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x, dim=1)
        return x


class DepthConv1d(nn.Module):
    def __init__(
        self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=False, causal=False
    ):
        super(DepthConv1d, self).__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=self.padding,
        )
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = select_norm(norm="cln", dim=hidden_channel, shape=3)
            self.reg2 = select_norm(norm="cln", dim=hidden_channel, shape=3)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, : -self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class DepthConv1d_Q(nn.Module):
    def __init__(
        self,
        input_channel,
        hidden_channel,
        kernel,
        padding,
        dilation=1,
        skip=False,
        causal=False,
        QA_flag=False,
        ak=8,
    ):
        super(DepthConv1d_Q, self).__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(
            hidden_channel,
            hidden_channel,
            kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=self.padding,
        )
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = select_norm(norm="cln", dim=hidden_channel, shape=3)
            self.reg2 = select_norm(norm="cln", dim=hidden_channel, shape=3)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

        self.QA_flag = QA_flag
        self.ak = ak

    def forward(self, input):
        if self.QA_flag:
            input = min_max_quantize(input, self.ak)

        output = self.reg1(self.nonlinearity1(self.conv1d(input)))

        if self.QA_flag:
            output = min_max_quantize(output, self.ak)
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, : -self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))

        if self.QA_flag:
            output = min_max_quantize(output, self.ak)
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


# GC-equipped TCN
class GC_TCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer,
        stack,
        kernel=3,
        skip=False,
        causal=False,
        dilated=True,
        num_group=2,
    ):
        super(GC_TCN, self).__init__()

        self.receptive_field = 0
        self.dilated = dilated
        self.num_group = num_group

        self.TAC = nn.ModuleList([])
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(
                        DepthConv1d(
                            input_dim // num_group,
                            hidden_dim // num_group,
                            kernel,
                            dilation=2**i,
                            padding=2**i,
                            skip=skip,
                            causal=causal,
                        )
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(
                            input_dim // num_group,
                            hidden_dim // num_group,
                            kernel,
                            dilation=1,
                            padding=1,
                            skip=skip,
                            causal=causal,
                        )
                    )
                self.TAC.append(TAC(input_dim // num_group, hidden_dim * 3 // num_group))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += kernel - 1
        # output layer
        self.skip = skip

    def forward(self, input):
        batch_size, N, L = input.shape
        output = input.view(batch_size, self.num_group, -1, L)

        if self.skip:
            skip_connection = 0.0
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)
                output = output.view(batch_size * self.num_group, -1, L)
                residual, skip = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L)
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)
                output = output.view(batch_size * self.num_group, -1, L)
                residual = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L)

        output = output.view(batch_size, -1, L)

        return output


class GC_TCN_Q(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer,
        stack,
        kernel=3,
        skip=False,
        causal=False,
        dilated=True,
        num_group=2,
        QA_flag=False,
        ak=8,
    ):
        super(GC_TCN_Q, self).__init__()

        self.receptive_field = 0
        self.dilated = dilated
        self.num_group = num_group
        self.skip = skip

        self.QA_flag = QA_flag
        self.ak = ak

        self.TAC = nn.ModuleList([])
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(
                        DepthConv1d_Q(
                            input_dim // num_group,
                            hidden_dim // num_group,
                            kernel,
                            dilation=2**i,
                            padding=2**i,
                            skip=skip,
                            causal=causal,
                            QA_flag=QA_flag,
                            ak=ak,
                        )
                    )
                else:
                    self.TCN.append(
                        DepthConv1d_Q(
                            input_dim // num_group,
                            hidden_dim // num_group,
                            kernel,
                            dilation=1,
                            padding=1,
                            skip=skip,
                            causal=causal,
                            QA_flag=QA_flag,
                            ak=ak,
                        )
                    )
                self.TAC.append(
                    TAC_Q(
                        input_dim // num_group, hidden_dim * 3 // num_group, QA_flag=QA_flag, ak=ak
                    )
                )
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += kernel - 1

    def forward(self, input):
        batch_size, N, L = input.shape  # B, context*L, N
        output = input.view(batch_size, self.num_group, -1, L)  # B, context, L, N

        if self.skip:
            skip_connection = 0.0
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)
                output = output.view(batch_size * self.num_group, -1, L)
                residual, skip = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L)
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                output = self.TAC[i](output)
                output = output.view(batch_size * self.num_group, -1, L)
                residual = self.TCN[i](output)
                output = (output + residual).view(batch_size, self.num_group, -1, L)

        output = output.view(batch_size, -1, L)  # B, N, L

        return output


class LiMuSE(nn.Module):
    def __init__(
        self,
        N=128,
        hidden_dim=256,
        K=32,
        E=50,
        layer=24,
        num_spks=2,
        context_size=32,
        group_size=16,
        activate="relu",
        causal=False,
        QA_flag=False,
        ak=8,
    ):
        super(LiMuSE, self).__init__()
        self.E = E
        self.N = N
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        self.num_spks = num_spks
        self.context_size = context_size
        self.layer = layer
        self.encoder = Conv1D(2, N, K, stride=K // 2, padding=0)
        self.voiceprint_encoder = nn.Conv1d(
            in_channels=512, out_channels=N, kernel_size=1, stride=1, padding=0
        )
        self.visual_encoder = nn.Linear(256, N)
        self.Normal_S = select_norm("gln", N, 3)

        # context encoder/decoder
        self.context_enc_1 = GC_TCN_Q(
            self.N,
            self.hidden_dim,
            layer=2,
            stack=1,
            kernel=3,
            skip=False,
            causal=causal,
            num_group=self.group_size,
            QA_flag=QA_flag,
            ak=ak,
        )
        self.context_dec_1 = GC_TCN_Q(
            self.N,
            self.hidden_dim,
            layer=2,
            stack=1,
            kernel=3,
            skip=False,
            causal=causal,
            num_group=self.group_size,
            QA_flag=QA_flag,
            ak=ak,
        )
        self.context_enc_2 = GC_TCN_Q(
            3 * self.N,
            3 * self.hidden_dim,
            layer=2,
            stack=1,
            kernel=3,
            skip=False,
            causal=causal,
            num_group=self.group_size,
            QA_flag=QA_flag,
            ak=ak,
        )
        self.context_dec_2 = GC_TCN_Q(
            3 * self.N,
            3 * self.hidden_dim,
            layer=2,
            stack=1,
            kernel=3,
            skip=False,
            causal=causal,
            num_group=self.group_size,
            QA_flag=QA_flag,
            ak=ak,
        )

        # Separation block
        self.audio_block = GC_TCN_Q(
            self.N,
            self.N * 4,
            layer=6,
            stack=2,
            kernel=3,
            skip=False,
            causal=causal,
            num_group=self.group_size,
            QA_flag=QA_flag,
            ak=ak,
        )
        self.fusion_block = GC_TCN_Q(
            3 * self.N,
            self.N * 12,
            layer=6,
            stack=1,
            kernel=3,
            skip=False,
            causal=causal,
            num_group=self.group_size,
            QA_flag=QA_flag,
            ak=ak,
        )

        self.gen_masks = Conv1D_Q(3 * N, N, 1, QA_flag=QA_flag, ak=ak)
        self.decoder = ConvTrans1D(N, 1, K, stride=K // 2)

        # activation function
        active_f = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "softmax": nn.Softmax(dim=0)}
        self.activation_type = activate
        self.activation = active_f[activate]

    def forward(self, mix, aux, visual):
        enc_out = self.encoder(mix)  # B x N x T
        batch_size, num_channel, T = enc_out.shape
        aux = aux.transpose(1, 2)
        aux = self.voiceprint_encoder(aux)  # B x N x 1
        aux = aux.repeat(1, 1, T)  # B x N x T
        visual = self.visual_encoder(visual)  # B x max_video_len x T
        visual = F.interpolate(
            visual.transpose(1, 2), T, mode="linear", align_corners=False
        )  # B x N x T

        audio = self.Normal_S(enc_out)  # B, N, T
        aux = self.Normal_S(aux)
        visual = self.Normal_S(visual)

        ###########  Part 1  ###########
        # context encoding
        squeeze_block, squeeze_rest = split_feature(audio, self.context_size)  # B, N, context, L
        squeeze_frame = squeeze_block.shape[-1]  # L

        squeeze_input = (
            squeeze_block.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * squeeze_frame, self.N, self.context_size)
        )  # B*L, N, context

        squeeze_output = self.context_enc_1(squeeze_input)  # B*L, N, context

        squeeze_mean = (
            squeeze_output.mean(2)
            .view(batch_size, squeeze_frame, self.N)
            .transpose(1, 2)
            .contiguous()
        )  # B, N, L
        # sequence modeling
        feature_output = self.audio_block(squeeze_mean).view(
            batch_size, -1, squeeze_frame
        )  # B, N, L
        # context decoding
        feature_output = feature_output.unsqueeze(2) + squeeze_block  # B, N, context, L

        feature_output = (
            feature_output.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * squeeze_frame, self.N, self.context_size)
        )  # B*L, N, context

        unsqueeze_output = self.context_dec_1(feature_output).view(
            batch_size, squeeze_frame, self.N, -1
        )  # B, L, N, context
        unsqueeze_output = unsqueeze_output.permute(0, 2, 3, 1).contiguous()  # B, N, context, L
        unsqueeze_output = merge_feature(unsqueeze_output, squeeze_rest)  # B, N, T

        ###########  Fusion  ###########
        feature_fusion = torch.cat((unsqueeze_output, aux, visual), dim=2)
        feature_fusion = feature_fusion.reshape(batch_size, -1, T)

        ###########  Part 2  ###########
        # context encoding
        squeeze_block_2, squeeze_rest_2 = split_feature(
            feature_fusion, self.context_size
        )  # B, N, context, L
        squeeze_frame_2 = squeeze_block_2.shape[-1]

        squeeze_input_2 = (
            squeeze_block_2.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * squeeze_frame_2, -1, self.context_size)
        )  # B*L, N, context

        squeeze_output_2 = self.context_enc_2(squeeze_input_2)  # B*L, N, context
        squeeze_mean_2 = (
            squeeze_output_2.mean(2)
            .view(batch_size, squeeze_frame_2, -1)
            .transpose(1, 2)
            .contiguous()
        )  # B, N, L
        # Fusion Block
        fusion_output = self.fusion_block(squeeze_mean_2).view(
            batch_size, -1, squeeze_frame_2
        )  # B, 3*N, T

        # context decoding
        fusion_output = fusion_output.unsqueeze(2) + squeeze_block_2  # B, 3N, context, L

        fusion_output = (
            fusion_output.permute(0, 3, 1, 2)
            .contiguous()
            .view(batch_size * squeeze_frame_2, -1, self.context_size)
        )  # B*L, N, context

        unsqueeze_output_2 = self.context_dec_2(fusion_output).view(
            batch_size, squeeze_frame_2, -1, self.context_size
        )  # B, L, N, context
        unsqueeze_output_2 = unsqueeze_output_2.permute(0, 2, 3, 1).contiguous()  # B, N, context, L
        unsqueeze_output_2 = merge_feature(unsqueeze_output_2, squeeze_rest_2)  # B, N, T

        # Mask Generation
        masks = self.gen_masks(unsqueeze_output_2)
        mask_output = masks * enc_out  # B, N, T

        # Waveform Decoder
        output = self.decoder(mask_output, squeeze=False)  # B, 1, T_wav
        return output


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing.
#
# The real repo's own `test_limuse()` (in LiMuSE.py, __main__ block) exercises the
# full model with default hyperparameters: mix (B,2,48000), aux voiceprint
# (B,1,512), visual lip embedding (B,75,256). Those default hyperparameters
# (N=128, hidden_dim=256, layer=24, group_size=16, context_size=32) produce a
# very deep/wide network; for a tiny tracing instance we shrink N, hidden_dim,
# layer counts and group_size (must still divide N/hidden_dim as in the repo)
# while keeping the exact same architecture and the exact same 3-input
# (mix, aux, visual) calling convention as test_limuse().
# ---------------------------------------------------------------------------
_N = 16
_HIDDEN_DIM = 32
_K = 8
_GROUP_SIZE = 4
_BATCH = 2
_WAV_LEN = 512
_VIDEO_LEN = 6


def build_limuse():
    torch.manual_seed(0)
    model = LiMuSE(
        N=_N, hidden_dim=_HIDDEN_DIM, K=_K, layer=2, group_size=_GROUP_SIZE, context_size=8
    )
    model.eval()
    return model


def example_input_limuse():
    torch.manual_seed(0)
    mix = torch.randn(_BATCH, 2, _WAV_LEN)
    aux = torch.randn(_BATCH, 1, 512)
    visual = torch.randn(_BATCH, _VIDEO_LEN, 256)
    return (mix, aux, visual)


MENAGERIE_ENTRIES = [
    ("LiMuSE", "build_limuse", "example_input_limuse", 2021, MENAGERIE_ZOO),
]

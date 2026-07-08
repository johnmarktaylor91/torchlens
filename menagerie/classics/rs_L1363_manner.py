# SOURCE: vendored from https://github.com/winddori2002/MANNER @ main
# (src/models.py, src/attention.py, src/conv_modules.py, src/chunk.py,
#  and the two pure-torch helper functions `rescale_module`/`rescale_conv`
#  from src/utils.py)
#
# MANNER: "Multi-View Attention Network for Noise ERasure" (Park et al.,
# ICASSP 2022) -- a U-Net-style time-domain speech-enhancement model with
# down/up Conv1d encoder-decoder stages, residual Conformer-style conv
# blocks (ResConBlock), and a three-branch "Multiview Attention" block per
# encoder/decoder stage that mixes channel attention, chunk-based global
# (scaled dot-product) attention, and chunk-based local (depthwise-conv)
# attention. This file vendors the real classes verbatim from the repo's
# `src/models.py` + its real dependencies `src/attention.py`,
# `src/conv_modules.py`, `src/chunk.py` -- all of which import only base
# libs already installed here (torch, numpy, math). `src/utils.py` also
# unconditionally imports `neptune` and a local `.dataset` module (both
# training-pipeline-only, not installed / not part of the model
# architecture); rather than importing that whole file, the two pure-torch
# weight-init helper functions it defines that the real `MANNER.__init__`
# calls (`rescale_module`, `rescale_conv`) are vendored verbatim below --
# this is a minimal import-path fix (the "fix only imports/relative-paths
# minimally" vendoring rule), not an architectural change; no other code in
# this file was rewritten from the original.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# vendored from src/utils.py (pure-torch weight-rescaling helpers only;
# neptune-dependent training utilities in the original file are not vendored)
# ---------------------------------------------------------------------------
def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


# ---------------------------------------------------------------------------
# vendored from src/conv_modules.py
# ---------------------------------------------------------------------------
class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class ResConBlock(nn.Module):
    """Residual Conformer block."""

    def __init__(self, in_channels, kernel_size=31, growth1=2, growth2=2):
        super().__init__()

        out_channels1 = int(in_channels * growth1)
        out_channels2 = int(in_channels * growth2)

        self.point_conv1 = nn.Sequential(
            PointwiseConv(in_channels, out_channels1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(out_channels1),
            nn.GLU(dim=1),
        )
        self.depth_conv = nn.Sequential(
            DepthwiseConv(
                in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2
            ),
            nn.BatchNorm1d(in_channels),
            Swish(),
        )
        self.point_conv2 = nn.Sequential(
            PointwiseConv(in_channels, out_channels2, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(out_channels2),
            Swish(),
        )
        self.conv = BasicConv(out_channels2, out_channels2, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_channels, out_channels2, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        out = self.point_conv1(x)
        out = self.depth_conv(out)
        out = self.point_conv2(out)
        out = self.conv(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ---------------------------------------------------------------------------
# vendored from src/chunk.py
# ---------------------------------------------------------------------------
class DualPathProcessing(nn.Module):
    """Overlapped chunking (originally adapted in MANNER from Asteroid,
    https://github.com/asteroid-team/asteroid)."""

    def __init__(self, chunk_size, hop_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_orig_frames = None

    def unfold(self, x):
        # x is (batch, chan, frames)
        batch, chan, frames = x.size()
        assert x.ndim == 3
        self.n_orig_frames = x.shape[-1]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        return unfolded.reshape(batch, chan, self.chunk_size, -1)

    def fold(self, x, output_size=None):
        output_size = output_size if output_size is not None else self.n_orig_frames
        batch, chan, chunk_size, n_chunks = x.size()
        to_unfold = x.reshape(batch, chan * self.chunk_size, n_chunks)
        x = torch.nn.functional.fold(
            to_unfold,
            (output_size, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        x /= float(self.chunk_size) / self.hop_size
        return x.reshape(batch, chan, self.n_orig_frames)


# ---------------------------------------------------------------------------
# vendored from src/attention.py
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.ReLU(), nn.Linear(channels // 2, channels)
        )

    def forward(self, x):
        # [B,N,T] -> [B,N,1]
        attn_max = F.adaptive_max_pool1d(x, 1)
        attn_avg = F.adaptive_avg_pool1d(x, 1)

        attn_max = self.fc(attn_max.squeeze())
        attn_avg = self.fc(attn_avg.squeeze())

        attn = attn_max + attn_avg
        attn = torch.sigmoid(attn).unsqueeze(-1)

        x = x * attn
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn


class GlobalAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

    def forward(self, q, k, v):
        # [B*N,P,C]
        b, p, c = q.shape
        d_k, n_head = self.d_k, self.n_head

        q = self.w_qs(q).view(b, p, n_head, d_k)
        k = self.w_ks(k).view(b, p, n_head, d_k)
        v = self.w_vs(v).view(b, p, n_head, d_k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(b, p, -1)
        q = self.fc(q)
        return q


class LocalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel_size1 = 31
        kernel_size2 = 7
        self.depth_conv = nn.Sequential(
            DepthwiseConv(
                channels, channels, kernel_size1, stride=1, padding=(kernel_size1 - 1) // 2
            ),
            nn.BatchNorm1d(channels),
            Swish(),
        )
        self.conv = BasicConv(
            2, 1, kernel_size2, stride=1, padding=(kernel_size2 - 1) // 2, relu=False
        )

    def forward(self, x):
        b, n, p, c = x.size()
        attn = x.permute(0, 2, 1, 3).contiguous().view(b * p, n, c)
        attn = self.depth_conv(attn)
        attn = torch.cat(
            [torch.max(attn, dim=1)[0].unsqueeze(1), torch.mean(attn, dim=1).unsqueeze(1)], dim=1
        )
        attn = self.conv(attn)
        attn = torch.sigmoid(attn)
        attn = attn.view(b, p, 1, c).permute(0, 2, 1, 3).contiguous()
        x = x * attn
        return x


class MultiviewAttentionBlock(nn.Module):
    """Multiview Attention block: channel / global (chunked) / local (chunked) branches."""

    def __init__(self, channels, segment_len, head):
        super().__init__()

        self.inter = int(channels / 3)
        d_k = int(segment_len * head)

        self.dsp = DualPathProcessing(segment_len, segment_len // 2)

        self.in_branch0 = BasicConv(channels, self.inter, kernel_size=1, stride=1)
        self.in_branch1 = BasicConv(channels, self.inter, kernel_size=1, stride=1)
        self.in_branch2 = BasicConv(channels, self.inter, kernel_size=1, stride=1)

        self.channel_attn = ChannelAttention(self.inter)
        self.global_attn = GlobalAttention(head, segment_len, d_k, d_k)
        self.local_attn = LocalAttention(self.inter)

        self.out_branch0 = BasicConv(self.inter, self.inter, kernel_size=3, stride=1, padding=1)
        self.out_branch1 = BasicConv(self.inter, self.inter, kernel_size=3, stride=1, padding=1)
        self.out_branch2 = BasicConv(self.inter, self.inter, kernel_size=3, stride=1, padding=1)

        self.conv = BasicConv(self.inter * 3, channels, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(channels, channels, kernel_size=1, stride=1, relu=False)

        self.output_tanh = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Tanh())
        self.output_sigmoid = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid()
        )
        self.gate_conv = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.ReLU())

    def forward(self, x):
        # [B,N,T] -> [B,N/3,T]
        x0 = self.in_branch0(x)
        x1 = self.in_branch1(x)
        x2 = self.in_branch2(x)

        x1 = self.dsp.unfold(x1).transpose(2, 3)
        x2 = self.dsp.unfold(x2).transpose(2, 3)

        b, n, p, c = x1.size()

        x1 = x1.view(b * n, p, c)

        x0 = self.channel_attn(x0)
        x1 = self.global_attn(x1, x1, x1)
        x2 = self.local_attn(x2)

        x1 = x1.view(b, n, p, c)

        x1 = self.dsp.fold(x1.transpose(2, 3))
        x2 = self.dsp.fold(x2.transpose(2, 3))

        x0 = self.out_branch0(x0)
        x1 = self.out_branch1(x1)
        x2 = self.out_branch2(x2)

        out = torch.cat([x0, x1, x2], dim=1)
        out = self.conv(out)
        short = self.shortcut(x)

        gated_tanh = self.output_tanh(out)
        gated_sig = self.output_sigmoid(out)
        gated = gated_tanh * gated_sig
        out = self.gate_conv(gated)

        x = short + out
        return x


# ---------------------------------------------------------------------------
# vendored from src/models.py
# ---------------------------------------------------------------------------
class MaskGate(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.output = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.mask = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.ReLU())

    def forward(self, x):
        mask = self.output(x) * self.output_gate(x)
        mask = self.mask(mask)
        return mask


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, segment_len, head):
        super().__init__()

        self.down_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size, stride),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
        )
        self.conv_block = ResConBlock(in_channels, growth1=2, growth2=2)
        self.attn_block = MultiviewAttentionBlock(out_channels, segment_len, head)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.conv_block(x)
        x = self.attn_block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, segment_len, head):
        super().__init__()

        self.conv_block = ResConBlock(in_channels, growth1=2, growth2=1 / 2)
        self.up_conv = nn.Sequential(
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.attn_block = MultiviewAttentionBlock(out_channels, segment_len, head)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.up_conv(x)
        x = self.attn_block(x)
        return x


class MANNER_Block(nn.Module):
    """
    MANNER block.
        in_channels  :  inital in channel.
        out_channels :  inital out channel.
        hidden       :  hidden channel size.
        depth        :  depth of encoder and decoder layer.
        kernel_size  :  kernel size for down and up convolution layer.
        stride       :  stride for down and up convolution layer.
        growth       :  growth rate of the channel size.
        head         :  number of heads in global attention.
        segment_len  :  chunk size for overlapped chunking in global and local attention.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden,
        depth,
        kernel_size,
        stride,
        growth,
        head,
        segment_len,
    ):
        super().__init__()

        self.depth = depth
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(hidden, in_channels, kernel_size=3, stride=1, padding=1)
        )
        in_channels = in_channels * hidden
        out_channels = out_channels * growth

        encoder = []
        decoder = []
        for layer in range(depth):
            encoder.append(
                Encoder(in_channels, out_channels * hidden, kernel_size, stride, segment_len, head)
            )
            decoder.append(
                Decoder(out_channels * hidden, in_channels, kernel_size, stride, segment_len, head)
            )

            in_channels = hidden * (2 ** (layer + 1))
            out_channels *= growth

        decoder.reverse()

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

        hdim = hidden * growth ** (layer + 1)
        self.linear = nn.Sequential(nn.Linear(hdim, hdim, bias=False), nn.ReLU())
        self.mask_gate = MaskGate(hidden)

    def forward(self, x):
        """
        input X : [B, 1, T]
        output X: [B, 1, T]
        """
        x = self.in_conv(x)  # [B,1,T] -> [B,N,T]
        enc_out = x  # for applying mask

        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)

        x = x.permute(0, 2, 1)  # [B,N,T] -> [B,T,N]
        x = self.linear(x)
        x = x.permute(0, 2, 1)  # [B,N,T]

        for decoder in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., : x.shape[-1]]
            x = decoder(x)

        mask = self.mask_gate(x)  # [B,N,T]
        x = enc_out * mask
        x = self.out_conv(x)  # [B,1,T]

        return x


class MANNER(nn.Module):
    """
    MANNER for speech enhancement in time-domain.
        in_channels  :  inital in channel.
        out_channels :  inital out channel.
        hidden       :  hidden channel size.
        depth        :  depth of encoder and decoder layer.
        kernel_size  :  kernel size for down and up convolution layer.
        stride       :  stride for down and up convolution layer.
        growth       :  growth rate of the channel size.
        head         :  number of heads in global attention.
        segment_len  :  chunk size for overlapped chunking in global and local attention.
    """

    eps = 1e-3
    rescale = 0.1

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden,
        depth,
        kernel_size,
        stride,
        growth,
        head,
        segment_len,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.depth = depth

        self.manner_block = MANNER_Block(
            in_channels, out_channels, hidden, depth, kernel_size, stride, growth, head, segment_len
        )
        rescale_module(self, reference=self.rescale)

    def padding(self, length):
        length = math.ceil(length)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        return int(length)

    def forward(self, x):
        """
        Input X: [B, 1, T]
        B: batch size
        1: channel size
        T: signal length
        """
        # input normalization by its standardiviation
        x2 = x.mean(dim=1, keepdim=True)
        std = x2.std(dim=-1, keepdim=True)
        x = x / (self.eps + std)

        # estimate enhanced speech
        length = x.shape[-1]
        x = F.pad(x, (0, self.padding(length) - length))
        x = self.manner_block(x)
        x = x[..., :length]

        return std * x


# ---------------------------------------------------------------------------
# Tiny build/example for TorchLens tracing. The repo's default config
# (config/train.yaml) uses hidden=60, depth=4, kernel_size=8, stride=4,
# growth=2, head=8, segment_len=32, in_channels=out_channels=1, on 16kHz
# audio segments (segment=4 sec = 64000 samples). depth=4 with kernel=8/
# stride=4 needs a moderately long input to stay valid after 4 downsampling
# stages, so a shorter clip is used here for a fast CPU trace while keeping
# every other architectural hyperparameter at the real default.
# ---------------------------------------------------------------------------
def build_manner():
    torch.manual_seed(0)
    model = MANNER(
        in_channels=1,
        out_channels=1,
        hidden=60,
        depth=4,
        kernel_size=8,
        stride=4,
        growth=2,
        head=8,
        segment_len=32,
    )
    model.eval()
    return model


def example_input_manner():
    torch.manual_seed(0)
    return torch.randn(1, 1, 8000)


MENAGERIE_ENTRIES = [
    ("MANNER", "build_manner", "example_input_manner", 2022, MENAGERIE_ZOO),
]

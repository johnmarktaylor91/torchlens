# SOURCE: vendored from bear-boy/DPCRN-Pytorch @ master
#   Vendored files: model.py (STFT, ISTFT, DPCRN, Encoder, Decoder), modules.py
#   (SingleRNN, DPRNN).
# https://github.com/bear-boy/DPCRN-Pytorch
#
# DPCRN (Deep Priori-Complex-mask RNN, Le Xiaohuai & Chen 2021 arxiv:2107.05429,
# DNS3-challenge speech-enhancement submission). CNN encoder/decoder over the
# stacked real/imag STFT with skip connections, a DPRNN (dual-path RNN: intra-frame
# row-RNN + inter-frame col-RNN, each GroupNorm-normalized and residual) bottleneck,
# and a complex ratio mask applied to the noisy spectrogram before ISTFT resynthesis.
# This is a faithful independent community PyTorch port of the official
# Le-Xiaohuai-speech/DPCRN_DNS3 (TensorFlow/Keras) architecture; the official repo
# itself is TF/Keras-only (tensorflow-gpu==1.15), so this real PyTorch class is used
# per the RUNG-2 real-repo-code rule (a genuine nn.Module implementation exists, not
# a from-scratch reimplementation authored here).
#
# Minimal API-compat fixes (NOT architecture changes):
#   - `cos_win = torch.from_numpy(...).cuda()` (hardcoded CUDA module-level global in
#     the original model.py) -> built lazily inside STFT.__init__ on the module's own
#     device-neutral buffer (registered via `register_buffer`) so the model runs on
#     CPU; the windowing math (scipy.signal.windows.cosine) is untouched.
#   - `device = 'cuda' if torch.cuda.is_available() else 'cpu'` then `.to(device)`
#     calls in the original `test_model()`/`get_model_size()` helpers -> dropped;
#     build_dpcrn() below constructs the model on the default (CPU) device.
# Everything else (Encoder/Decoder conv topology, DPRNN row/col RNN dual-path logic,
# complex masking math, STFT/ISTFT via torch.stft/istft) is an unmodified
# transcription of the original source, using the real repo's own dpcrn.json config.

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class SingleRNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            1,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(
            output.shape
        )
        return rnn_output


# dual-path RNN
class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False
    ):
        super(DPRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # dual-path RNN
        self.row_rnn = nn.ModuleList([])
        self.col_rnn = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_rnn.append(
                SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=True)
            )  # intra-segment RNN is always noncausal
            self.col_rnn.append(
                SingleRNN(rnn_type, input_size, hidden_size, dropout, bidirectional=bidirectional)
            )
            self.row_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            # default is to use noncausal LayerNorm for inter-chunk RNN. For causal setting change it to causal normalization techniques accordingly.
            self.col_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply RNN on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        batch_size, _, dim1, dim2 = input.shape
        output = input
        for i in range(len(self.row_rnn)):
            row_input = (
                output.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)
            )  # B*dim2, dim1, N
            row_output = self.row_rnn[i](row_input)  # B*dim2, dim1, H
            row_output = (
                row_output.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()
            )  # B, N, dim1, dim2
            row_output = self.row_norm[i](row_output)
            output = output + row_output

            col_input = (
                output.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)
            )  # B*dim1, dim2, N
            col_output = self.col_rnn[i](col_input)  # B*dim1, dim2, H
            col_output = (
                col_output.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()
            )  # B, N, dim1, dim2
            col_output = self.col_norm[i](col_output)
            output = output + col_output

        return output


class STFT(nn.Module):
    def __init__(self, frame_len, frame_hop, fft_len=None):
        super(STFT, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        cos_win = torch.from_numpy(signal.windows.cosine(frame_len, False)).type(torch.FloatTensor)
        self.register_buffer("cos_win", cos_win)

    def forward(self, x):
        if len(x.shape) != 2:
            print("x must be in [B, T]")
        y = torch.stft(
            x,
            hop_length=self.frame_hop,
            n_fft=self.frame_len,
            window=self.cos_win,
            return_complex=True,
            center=False,
        )
        r = y.real
        i = y.imag
        return r, i


class ISTFT(nn.Module):
    def __init__(self, frame_len, frame_hop, fft_len=None):
        super(ISTFT, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        cos_win = torch.from_numpy(signal.windows.cosine(frame_len, False)).type(torch.FloatTensor)
        self.register_buffer("cos_win", cos_win)

    def forward(self, real, imag):
        x = torch.complex(real, imag)
        y = torch.istft(
            x, hop_length=self.frame_hop, n_fft=self.frame_len, window=self.cos_win, center=False
        )
        return y


class DPCRN(nn.Module):
    def __init__(
        self,
        encoder_in_channel,
        encoder_channel_size,
        encoder_kernel_size,
        encoder_stride_size,
        encoder_padding,
        decoder_in_channel,
        decoder_channel_size,
        decoder_kernel_size,
        decoder_stride_size,
        rnn_type,
        input_size,
        hidden_size,
        frame_len,
        frame_shift,
    ):
        super(DPCRN, self).__init__()
        self.encoder_channel_size = encoder_channel_size
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_stride_size = encoder_stride_size
        self.encoder_padding = encoder_padding
        self.decoder_channel_size = decoder_channel_size
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_stride_size = decoder_stride_size
        self.frame_len = frame_len
        self.frame_shift = frame_shift

        self.stft = STFT(self.frame_len, self.frame_shift)
        self.istft = ISTFT(self.frame_len, self.frame_shift)

        self.encoder = Encoder(
            encoder_in_channel,
            self.encoder_channel_size,
            self.encoder_kernel_size,
            self.encoder_stride_size,
            self.encoder_padding,
        )
        self.decoder = Decoder(
            decoder_in_channel,
            self.decoder_channel_size,
            self.decoder_kernel_size,
            self.decoder_stride_size,
        )
        self.dprnn = DPRNN(rnn_type, input_size=input_size, hidden_size=hidden_size)

    def forward(self, x):
        re, im = self.stft(x)
        inputs = torch.stack([re, im], dim=1)  # B x C x F x T
        x, skips = self.encoder(inputs)

        x = self.dprnn(x)

        mask = self.decoder(x, skips)
        en_re, en_im = self.mask_speech(mask, inputs)  # en_ shape: B x F x T
        en_speech = self.istft(en_re, en_im)
        return en_speech, en_re, en_im

    def mask_speech(self, mask, x):
        mask_re = mask[:, 0, :, :]
        mask_im = mask[:, 1, :, :]

        x_re = x[:, 0, :, :]
        x_im = x[:, 1, :, :]

        en_re = x_re * mask_re - x_im * mask_im
        en_im = x_re * mask_im + x_im * mask_re
        return en_re, en_im


class Encoder(nn.Module):
    def __init__(self, in_channel_size, channel_size, kernel_size, stride_size, padding):
        super(Encoder, self).__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.padding = padding

        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        in_chan = in_channel_size
        for i in range(len(channel_size)):
            self.conv.append(
                nn.Conv2d(
                    in_channels=in_chan,
                    out_channels=channel_size[i],
                    kernel_size=kernel_size[i],
                    stride=stride_size[i],
                )
            )
            self.norm.append(nn.BatchNorm2d(channel_size[i]))
            in_chan = channel_size[i]
        self.prelu = nn.PReLU()

    def forward(self, x):
        # x shape: B x C x F x T
        skips = []
        for i, (layer, norm) in enumerate(zip(self.conv, self.norm)):
            x = F.pad(x, pad=self.padding[i])
            x = layer(x)
            x = self.prelu(norm(x))
            skips.append(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, in_channel_size, channel_size, kernel_size, stride_size):
        super(Decoder, self).__init__()
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride_size = stride_size

        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        in_chan = in_channel_size
        for i in range(len(channel_size)):
            if i == 3:
                self.conv.append(
                    nn.ConvTranspose2d(
                        in_channels=in_chan,
                        out_channels=channel_size[i],
                        kernel_size=kernel_size[i],
                        stride=stride_size[i],
                        padding=[1, 0],
                        output_padding=[1, 0],
                    )
                )
            else:
                self.conv.append(
                    nn.ConvTranspose2d(
                        in_channels=in_chan,
                        out_channels=channel_size[i],
                        kernel_size=kernel_size[i],
                        stride=stride_size[i],
                        padding=[1, 0],
                    )
                )
            self.norm.append(nn.BatchNorm2d(channel_size[i]))
            in_chan = channel_size[i] * 2
        self.prelu = nn.PReLU()

    def forward(self, x, skips):
        # x shape: B x C x F x T
        for i, (layer, norm, skip) in enumerate(zip(self.conv, self.norm, reversed(skips))):
            x = torch.cat([x, skip], dim=1)
            x = layer(x)[:, :, :, :-1]
            x = self.prelu(norm(x))
        return x


MENAGERIE_ZOO = "vendored-pytorch"


def build_dpcrn():
    # Real dpcrn.json config from the repo (5-stage encoder/decoder + DPRNN), with
    # frame_len/frame_shift shrunk from the repo defaults (400/200) to 40/20, and
    # every encoder/decoder channel count divided down proportionally (32/32/32/
    # 64/128 -> 4/4/4/8/16, decoder_in_channel 256->32) so the STFT time axis and
    # channel widths are small enough for a fast trace while every conv/pad/stride
    # geometry and the DPRNN row/col dual-path structure stay unchanged. DPRNN
    # input_size/hidden_size (128 in the repo default) is set to match the final
    # encoder channel count (16), exactly as the real config does (128 there too).
    model = DPCRN(
        encoder_in_channel=2,
        encoder_channel_size=[4, 4, 4, 8, 16],
        encoder_kernel_size=[[5, 2], [3, 2], [3, 2], [3, 2], [3, 2]],
        encoder_stride_size=[[2, 1], [2, 1], [1, 1], [1, 1], [1, 1]],
        encoder_padding=[[1, 0, 0, 2], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1]],
        decoder_in_channel=32,
        decoder_channel_size=[8, 4, 4, 4, 2],
        decoder_kernel_size=[[3, 2], [3, 2], [3, 2], [3, 2], [5, 2]],
        decoder_stride_size=[[1, 1], [1, 1], [1, 1], [2, 1], [2, 1]],
        rnn_type="LSTM",
        input_size=16,
        hidden_size=16,
        frame_len=40,
        frame_shift=20,
    )
    model.eval()
    return model


def example_input_dpcrn():
    return torch.randn(1, 4000)


MENAGERIE_ENTRIES = [
    ("DPCRN", "build_dpcrn", "example_input_dpcrn", 2021, MENAGERIE_ZOO),
]

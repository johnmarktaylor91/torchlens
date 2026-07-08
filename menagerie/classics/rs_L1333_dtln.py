# SOURCE: vendored from lhwcv/DTLN_pytorch @ main
#   Vendored file: DTLN_model.py (Simple_STFT_Layer, Pytorch_InstantLayerNormalization,
#   SeperationBlock, Pytorch_DTLN classes).
# https://github.com/lhwcv/DTLN_pytorch
#
# DTLN (Dual-signal Transformation LSTM Network, Westhausen & Meyer, Interspeech 2020)
# real-time speech-enhancement model. This is a faithful PyTorch reimplementation of
# the official breizhn/DTLN TensorFlow-2 model, vendored here rather than the TF repo
# itself since it is real (not from-scratch) PyTorch code with no non-base deps:
# stage 1 is an STFT-magnitude LSTM mask predictor, stage 2 is a learned-basis
# (1x1-conv) encoder/decoder LSTM mask predictor operating on the stage-1 output
# waveform, with an instant (per-frame) layer normalization between the encoder and
# the second separation block.
#
# No API-compat fixes needed: the vendored forward pass already uses modern
# `torch.stft(..., return_complex=True)` / `torch.fft.irfft2` and unfold/fold ops.

import torch
import torch.nn as nn


class Simple_STFT_Layer(nn.Module):
    def __init__(self, frame_len, frame_hop):
        super(Simple_STFT_Layer, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop

    def forward(self, x):
        if len(x.shape) != 2:
            print("x must be in [B, T]")
        y = torch.stft(
            x,
            n_fft=self.frame_len,
            hop_length=self.frame_hop,
            win_length=self.frame_len,
            return_complex=True,
            center=False,
        )
        r = y.real
        i = y.imag
        mag = torch.clamp(r**2 + i**2, self.eps) ** 0.5
        phase = torch.atan2(i + self.eps, r + self.eps)
        return mag, phase


class Pytorch_InstantLayerNormalization(nn.Module):
    """
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    """

    def __init__(self, channels):
        """
        Constructor
        """
        super(Pytorch_InstantLayerNormalization, self).__init__()
        self.epsilon = 1e-7
        self.gamma = nn.Parameter(torch.ones(1, 1, channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, 1, channels), requires_grad=True)
        self.register_parameter("gamma", self.gamma)
        self.register_parameter("beta", self.beta)

    def forward(self, inputs):
        # calculate mean of each frame
        mean = torch.mean(inputs, dim=-1, keepdim=True)

        # calculate variance of each frame
        variance = torch.mean(torch.square(inputs - mean), dim=-1, keepdim=True)
        # calculate standard deviation
        std = torch.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs


class SeperationBlock(nn.Module):
    def __init__(self, input_size=257, hidden_size=128, dropout=0.25):
        super(SeperationBlock, self).__init__()
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.rnn2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, (h, c) = self.rnn1(x)
        x1 = self.drop(x1)
        x2, _ = self.rnn2(x1)
        x2 = self.drop(x2)

        mask = self.dense(x2)
        mask = self.sigmoid(mask)
        return mask


class Pytorch_DTLN(nn.Module):
    def __init__(self, frame_len=512, frame_hop=128, window="rect"):
        super(Pytorch_DTLN, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.stft = Simple_STFT_Layer(frame_len, frame_hop)

        self.sep1 = SeperationBlock(input_size=(frame_len // 2 + 1), hidden_size=128, dropout=0.25)

        self.encoder_size = 256
        self.encoder_conv1 = nn.Conv1d(
            in_channels=frame_len,
            out_channels=self.encoder_size,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.encoder_norm1 = Pytorch_InstantLayerNormalization(channels=self.encoder_size)

        self.sep2 = SeperationBlock(input_size=self.encoder_size, hidden_size=128, dropout=0.25)

        self.decoder_conv1 = nn.Conv1d(
            in_channels=self.encoder_size,
            out_channels=frame_len,
            kernel_size=1,
            stride=1,
            bias=False,
        )

    def forward(self, x):
        """
        :param x:  [N, T]
        :return:
        """
        batch, n_frames = x.shape

        mag, phase = self.stft(x)
        mag = mag.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)

        # N, T, hidden_size
        mask = self.sep1(mag)
        estimated_mag = mask * mag

        s1_stft = estimated_mag * torch.exp((1j * phase))
        y1 = torch.fft.irfft2(s1_stft, dim=-1)
        y1 = y1.permute(0, 2, 1)

        encoded_f = self.encoder_conv1(y1)
        encoded_f = encoded_f.permute(0, 2, 1)
        encoded_f_norm = self.encoder_norm1(encoded_f)

        mask_2 = self.sep2(encoded_f_norm)
        estimated = mask_2 * encoded_f
        estimated = estimated.permute(0, 2, 1)

        decoded_frame = self.decoder_conv1(estimated)

        # overlap and add
        out = torch.nn.functional.fold(
            decoded_frame,
            (n_frames, 1),
            kernel_size=(self.frame_len, 1),
            padding=(0, 0),
            stride=(self.frame_hop, 1),
        )
        out = out.reshape(batch, -1)

        return out


MENAGERIE_ZOO = "vendored-pytorch"


def build_dtln():
    # Small frame_len/frame_hop (repo defaults 512/128 -> 64/16) to keep the STFT
    # window count and trace/render fast; architecture (2-stage LSTM separation +
    # learned encoder/decoder) is unchanged.
    model = Pytorch_DTLN(frame_len=64, frame_hop=16)
    model.eval()
    return model


def example_input_dtln():
    # Needs at least 2 STFT frames (frame_len + frame_hop samples) for the fold/
    # overlap-add reconstruction to run with a nontrivial number of frames.
    return torch.randn(1, 64 * 3)


MENAGERIE_ENTRIES = [
    ("DTLN", "build_dtln", "example_input_dtln", 2020, MENAGERIE_ZOO),
]

# SOURCE: vendored from wangtianrui/DCCRN @ master
#   Vendored files: model.py (DCCRN class), utils/complexnn.py (ComplexConv2d,
#   ComplexConvTranspose2d, ComplexBatchNorm, NavieComplexLSTM, complex_cat, cPReLU),
#   utils/conv_stft.py (ConvSTFT, ConviSTFT).
# https://github.com/wangtianrui/DCCRN
#
# DCCRN (Deep Complex Convolution Recurrent Network for Phase-Aware Speech
# Enhancement, Hu et al., Interspeech 2020 -- winner of the Interspeech 2020 Deep
# Noise Suppression challenge). Complex-valued convolutional encoder/decoder with a
# complex LSTM bottleneck, operating on the complex STFT of the input waveform and
# producing a complex ratio mask; ConvSTFT/ConviSTFT implement the (I)STFT as fixed
# (non-trainable) complex convolutions so the whole pipeline is waveform-in,
# waveform-out and traceable end to end.
#
# Minimal API-compat fixes (NOT architecture changes):
#   - `np.int` (removed from modern numpy) -> Python builtin `int`.
#   - `torch.addcmul(a, -1, b, c)` (pre-1.0 positional-scalar signature, removed in
#     modern torch) -> `torch.addcmul(a, b, c, value=-1)`.
# Everything else (conv topology, complex arithmetic, channel counts, STFT kernel
# construction) is untouched from the original source.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window

# ---------------------------------------------------------------------------
# utils/conv_stft.py
# ---------------------------------------------------------------------------


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == "None" or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)  # **0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(
        window[None, :, None].astype(np.float32)
    )


class ConvSTFT(nn.Module):
    def __init__(
        self, win_len, win_inc, fft_len=None, win_type="hamming", feature_type="real", fix=True
    ):
        super(ConvSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        self.register_buffer("weight", kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)
        inputs = F.pad(inputs, [self.win_len - self.stride, self.win_len - self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == "complex":
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase


class ConviSTFT(nn.Module):
    def __init__(
        self, win_len, win_inc, fft_len=None, win_type="hamming", feature_type="real", fix=True
    ):
        super(ConviSTFT, self).__init__()
        if fft_len is None:
            self.fft_len = int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        self.register_buffer("weight", kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer("window", window)
        self.register_buffer("enframe", torch.eye(win_len)[:, None, :])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """
        if phase is not None:
            real = inputs * torch.cos(phase)
            imag = inputs * torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        outputs = outputs[..., self.win_len - self.stride : -(self.win_len - self.stride)]

        return outputs


# ---------------------------------------------------------------------------
# utils/complexnn.py
# ---------------------------------------------------------------------------


class cPReLU(nn.Module):
    def __init__(self, complex_axis=1):
        super(cPReLU, self).__init__()
        self.r_prelu = nn.PReLU()
        self.i_prelu = nn.PReLU()
        self.complex_axis = complex_axis

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        real = self.r_prelu(real)
        imag = self.i_prelu(imag)
        return torch.cat([real, imag], self.complex_axis)


class NavieComplexLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, projection_dim=None, bidirectional=False, batch_first=False
    ):
        super(NavieComplexLSTM, self).__init__()

        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.real_lstm = nn.LSTM(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.imag_lstm = nn.LSTM(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=False,
        )
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        if projection_dim is not None:
            self.projection_dim = projection_dim // 2
            self.r_trans = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
            self.i_trans = nn.Linear(self.rnn_units * bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs, list):
            real, imag = inputs
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        return [real_out, imag_out]

    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()


def complex_cat(inputs, axis):
    real, imag = [], []
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, axis)
        real.append(r)
        imag.append(i)
    real = torch.cat(real, axis)
    imag = torch.cat(imag, axis)
    outputs = torch.cat([real, imag], axis)
    return outputs


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
    ):
        """
        in_channels: real+imag
        out_channels: real+imag
        kernel_size : input [B,C,D,T] kernel size in [D,T]
        padding : input [B,C,D,T] padding in [D,T]
        causal: if causal, will padding time dimension's left side,
                otherwise both
        """
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=[self.padding[0], 0],
            dilation=self.dilation,
            groups=self.groups,
        )
        self.imag_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=[self.padding[0], 0],
            dilation=self.dilation,
            groups=self.groups,
        )

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.0)
        nn.init.constant_(self.imag_conv.bias, 0.0)

    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)
        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real)
            imag2imag = self.imag_conv(imag)

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        causal=False,
        complex_axis=1,
        groups=1,
    ):
        """
        in_channels: real+imag
        out_channels: real+imag
        """
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.real_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
        )
        self.imag_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
        )
        self.complex_axis = complex_axis

        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.0)
        nn.init.constant_(self.imag_conv.bias, 0.0)

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, 2, self.complex_axis)
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            real = inputs[0]
            imag = inputs[1]
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)
        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(real)
            imag2imag = self.imag_conv(imag)

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


# Source: https://github.com/ChihebTrabelsi/deep_complex_networks/tree/pytorch
# from https://github.com/IMLHF/SE_DCUNet/blob/f28bf1661121c8901ad38149ea827693f1830715/models/layers/complexnn.py#L55


class ComplexBatchNorm(torch.nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        complex_axis=1,
    ):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter("Wrr", None)
            self.register_parameter("Wri", None)
            self.register_parameter("Wii", None)
            self.register_parameter("Br", None)
            self.register_parameter("Bi", None)

        if self.track_running_stats:
            self.register_buffer("RMr", torch.zeros(self.num_features))
            self.register_buffer("RMi", torch.zeros(self.num_features))
            self.register_buffer("RVrr", torch.ones(self.num_features))
            self.register_buffer("RVri", torch.zeros(self.num_features))
            self.register_buffer("RVii", torch.ones(self.num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter("RMr", None)
            self.register_parameter("RMi", None)
            self.register_parameter("RVrr", None)
            self.register_parameter("RVri", None)
            self.register_parameter("RVii", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-0.9, +0.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert xr.shape == xi.shape
        assert xr.size(1) == self.num_features

    def forward(self, inputs):
        xr, xi = torch.chunk(inputs, 2, axis=self.complex_axis)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr - Mr, xi - Mi

        #
        # Variance Matrix V Computation
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value=-1)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (-Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------


class DCCRN(nn.Module):
    def __init__(
        self,
        rnn_layer=2,
        rnn_hidden=256,
        win_len=400,
        hop_len=100,
        fft_len=512,
        win_type="hanning",
        use_clstm=True,
        use_cbn=False,
        masking_mode="E",
        kernel_size=5,
        kernel_num=(32, 64, 128, 256, 256, 256),
    ):
        super(DCCRN, self).__init__()
        self.rnn_layer = rnn_layer
        self.rnn_hidden = rnn_hidden

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.win_type = win_type

        self.use_clstm = use_clstm
        self.use_cbn = use_cbn
        self.masking_mode = masking_mode

        self.kernel_size = kernel_size
        self.kernel_num = (2,) + kernel_num

        self.stft = ConvSTFT(
            self.win_len, self.hop_len, self.fft_len, self.win_type, "complex", fix=True
        )
        self.istft = ConviSTFT(
            self.win_len, self.hop_len, self.fft_len, self.win_type, "complex", fix=True
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1])
                    if not use_cbn
                    else ComplexBatchNorm(self.kernel_num[idx + 1]),
                    nn.PReLU(),
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layer):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1]
                        if idx == 0
                        else self.rnn_hidden,
                        hidden_size=self.rnn_hidden,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1]
                        if idx == rnn_layer - 1
                        else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_hidden,
                num_layers=2,
                dropout=0.0,
                batch_first=False,
            )
            self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1])
                        if not use_cbn
                        else ComplexBatchNorm(self.kernel_num[idx - 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                        )
                    )
                )
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, x):
        stft = self.stft(x)
        real = stft[:, : self.fft_len // 2 + 1]
        imag = stft[:, self.fft_len // 2 + 1 :]
        spec_mags = torch.sqrt(real**2 + imag**2 + 1e-8)
        spec_phase = torch.atan2(imag, real)
        spec_complex = torch.stack([real, imag], dim=1)[:, :, 1:]  # B,2,256

        out = spec_complex
        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            encoder_out.append(out)
        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = out[:, :, : C // 2]
            i_rnn_in = out[:, :, C // 2 :]
            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2 * D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2 * D])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2, D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2, D])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
        else:
            out = torch.reshape(out, [T, B, C * D])
            out, _ = self.enhance(out)
            out = self.transform(out)
            out = torch.reshape(out, [T, B, C, D])
        out = out.permute(1, 2, 3, 0)
        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])
        if self.masking_mode == "E":
            mask_mags = (mask_real**2 + mask_imag**2) ** 0.5
            real_phase = mask_real / (mask_mags + 1e-8)
            imag_phase = mask_imag / (mask_mags + 1e-8)
            mask_phase = torch.atan2(imag_phase, real_phase)
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags * spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags * torch.cos(est_phase)
            imag = est_mags * torch.sin(est_phase)
        elif self.masking_mode == "C":
            real = real * mask_real - imag * mask_imag
            imag = real * mask_imag + imag * mask_real
        elif self.masking_mode == "R":
            real = real * mask_real
            imag = imag * mask_imag

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = out_wav.clamp_(-1, 1)
        return out_wav


MENAGERIE_ZOO = "vendored-pytorch"


def build_dccrn():
    # Small (2-stage) kernel_num to keep the trace/render fast, with fft_len/win_len
    # shrunk from the repo defaults (512/400) to 64 while keeping the real
    # kernel_size=5 (the repo default) so the per-stage frequency-axis downsampling
    # (ComplexConv2d stride=(2,1), padding=(2,1)) still divides out cleanly, matching
    # the shapes DCCRN.__init__'s `hidden_dim` bottleneck-LSTM sizing assumes.
    # win_type="hann" (scipy's modern canonical name; the original repo's default
    # "hanning" is a deprecated scipy.signal.get_window alias for the same window).
    model = DCCRN(
        rnn_layer=1,
        rnn_hidden=16,
        win_len=64,
        hop_len=16,
        fft_len=64,
        win_type="hann",
        use_clstm=True,
        use_cbn=False,
        masking_mode="E",
        kernel_size=5,
        kernel_num=(4, 8),
    )
    model.eval()
    return model


def example_input_dccrn():
    return torch.randn(1, 16000)


MENAGERIE_ENTRIES = [
    ("DCCRN", "build_dccrn", "example_input_dccrn", 2020, MENAGERIE_ZOO),
]

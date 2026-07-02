# SOURCE: vendored from https://github.com/axeber01/ngcc @ main
# (model.py: GCC, NGCCPHAT, PGCCPHAT + dnn_models.py: SincConv_fast, act_fun,
#  SincNet -- Axel Berg et al., "Extending GCC-PHAT using Shift Equivariant
#  Neural Networks", Interspeech 2022, arXiv:2208.04654)
# https://github.com/axeber01/ngcc
#
# The classes below are the REAL NGCC-PHAT model code, copied verbatim from
# the official repo's model.py and the SincNet backbone it imports from
# dnn_models.py (dnn_models.py itself is a lightly-adapted copy of Mirco
# Ravanelli's SincNet, per that file's own header comment). Only the classes
# on the traced forward path (GCC, NGCCPHAT, PGCCPHAT, SincConv_fast,
# act_fun, SincNet) were kept; the repo's legacy/unused `sinc`/`sinc_conv`/
# `flip`/`LayerNorm`/`MLP` helpers (dead code not reached by SincNet.forward,
# and in the case of `sinc_conv`/`sinc` hardcoded to `.cuda()` and thus not
# CPU-portable) were dropped. The repo's own `get_pad` helper comes from the
# tiny (35-line) third-party `torch_same_pad` package
# (https://github.com/CyberZHG/torch-same-pad, MIT); since that package is
# not on PyPI (only a source-level GitHub utility), its `get_pad` function is
# inlined verbatim below rather than adding a pip dependency. No architecture
# was altered.

import math

import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# torch_same_pad (https://github.com/CyberZHG/torch-same-pad) :: get_pad
# Inlined verbatim (MIT-licensed, ~20-line pure-torch utility, not on PyPI).
# ---------------------------------------------------------------------------
def _calc_pad(size: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1):
    pad = (((size + stride - 1) // stride - 1) * stride + kernel_size - size) * dilation
    return pad // 2, pad - pad // 2


def get_pad(size, kernel_size=3, stride=1, dilation=1):
    len_size = 1
    if isinstance(size, (list, tuple)):
        len_size = len(size)

    def _compressed(item, index):
        if isinstance(item, (list, tuple)):
            return item[index]
        return item

    pad = ()
    for i in range(len_size):
        pad = (
            _calc_pad(
                size=_compressed(size, i),
                kernel_size=_compressed(kernel_size, i),
                stride=_compressed(stride, i),
                dilation=_compressed(dilation, i),
            )
            + pad
        )
    return pad


# ---------------------------------------------------------------------------
# dnn_models.py :: act_fun, SincConv_fast, SincNet
# ---------------------------------------------------------------------------
def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=16000,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # computing only half of the window
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        # Due to symmetry, I only need half of the time axes
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class SincNet(nn.Module):
    def __init__(self, options):
        super(SincNet, self).__init__()

        self.cnn_N_filt = options["cnn_N_filt"]
        self.cnn_len_filt = options["cnn_len_filt"]
        self.cnn_max_pool_len = options["cnn_max_pool_len"]

        self.cnn_act = options["cnn_act"]
        self.cnn_drop = options["cnn_drop"]

        self.cnn_use_laynorm = options["cnn_use_laynorm"]
        self.cnn_use_batchnorm = options["cnn_use_batchnorm"]
        self.cnn_use_laynorm_inp = options["cnn_use_laynorm_inp"]
        self.cnn_use_batchnorm_inp = options["cnn_use_batchnorm_inp"]

        self.input_dim = int(options["input_dim"])

        self.fs = options["fs"]

        self.N_cnn_lay = len(options["cnn_N_filt"])
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
        self.use_sinc = options["use_sinc"]

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):
            N_filt = int(self.cnn_N_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            self.bn.append(nn.BatchNorm1d(N_filt, momentum=0.05))

            if i == 0:
                if self.use_sinc:
                    self.conv.append(
                        SincConv_fast(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs)
                    )
                else:
                    self.conv.append(nn.Conv1d(1, self.cnn_N_filt[i], self.cnn_len_filt[i]))

            else:
                self.conv.append(
                    nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i])
                )

            current_input = int(
                (current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]
            )

        self.out_dim = current_input * N_filt

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[-1]

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):
            s = x.shape[2]
            padding = get_pad(size=s, kernel_size=self.cnn_len_filt[i], stride=1, dilation=1)
            x = F.pad(x, pad=padding, mode="circular")

            if self.cnn_use_laynorm[i]:
                if i == 0:
                    x = self.drop[i](
                        self.act[i](
                            self.ln[i](
                                F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i])
                            )
                        )
                    )
                else:
                    x = self.drop[i](
                        self.act[i](
                            self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))
                        )
                    )

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](
                    self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))
                )

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:  # noqa: E712
                x = self.drop[i](
                    self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))
                )

        return x


# ---------------------------------------------------------------------------
# model.py :: GCC, NGCCPHAT, PGCCPHAT
# ---------------------------------------------------------------------------
class GCC(nn.Module):
    def __init__(self, max_tau=None, dim=2, filt="phat", epsilon=0.001, beta=None):
        super().__init__()

        """ GCC implementation based on Knapp and Carter,
        "The Generalized Correlation Method for Estimation of Time Delay",
        IEEE Trans. Acoust., Speech, Signal Processing, August, 1976 """

        self.max_tau = max_tau
        self.dim = dim
        self.filt = filt
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, x, y):
        n = x.shape[-1] + y.shape[-1]

        # Generalized Cross Correlation Phase Transform
        X = torch.fft.rfft(x, n=n)
        Y = torch.fft.rfft(y, n=n)
        Gxy = X * torch.conj(Y)

        if self.filt == "phat":
            phi = 1 / (torch.abs(Gxy) + self.epsilon)

        elif self.filt == "roth":
            phi = 1 / (X * torch.conj(X) + self.epsilon)

        elif self.filt == "scot":
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            phi = 1 / (torch.sqrt(Gxx * Gyy) + self.epsilon)

        elif self.filt == "ht":
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            gamma = Gxy / torch.sqrt(Gxx * Gxy)
            phi = torch.abs(gamma) ** 2 / (torch.abs(Gxy) * (1 - gamma) ** 2 + self.epsilon)

        elif self.filt == "cc":
            phi = 1.0

        else:
            raise ValueError("Unsupported filter function")

        if self.beta is not None:
            cc = []
            for i in range(self.beta.shape[0]):
                cc.append(torch.fft.irfft(Gxy * torch.pow(phi, self.beta[i]), n))

            cc = torch.cat(cc, dim=1)

        else:
            cc = torch.fft.irfft(Gxy * phi, n)

        max_shift = int(n / 2)
        if self.max_tau:
            max_shift = np.minimum(self.max_tau, int(max_shift))

        if self.dim == 2:
            cc = torch.cat((cc[:, -max_shift:], cc[:, : max_shift + 1]), dim=-1)
        elif self.dim == 3:
            cc = torch.cat((cc[:, :, -max_shift:], cc[:, :, : max_shift + 1]), dim=-1)

        return cc


class NGCCPHAT(nn.Module):
    def __init__(
        self, max_tau=42, head="classifier", use_sinc=True, sig_len=2048, num_channels=128, fs=16000
    ):
        super().__init__()

        """
        Neural GCC-PHAT with SincNet backbone

        arguments:
        max_tau - the maximum possible delay considered
        head - classifier or regression
        use_sinc - use sincnet backbone if True, otherwise use regular conv layers
        sig_len - length of input signal
        n_channel - number of gcc correlation channels to use
        fs - sampling frequency
        """

        self.max_tau = max_tau
        self.head = head

        sincnet_params = {
            "input_dim": sig_len,
            "fs": fs,
            "cnn_N_filt": [128, 128, 128, num_channels],
            "cnn_len_filt": [1023, 11, 9, 7],
            "cnn_max_pool_len": [1, 1, 1, 1],
            "cnn_use_laynorm_inp": False,
            "cnn_use_batchnorm_inp": False,
            "cnn_use_laynorm": [False, False, False, False],
            "cnn_use_batchnorm": [True, True, True, True],
            "cnn_act": ["leaky_relu", "leaky_relu", "leaky_relu", "linear"],
            "cnn_drop": [0.0, 0.0, 0.0, 0.0],
            "use_sinc": use_sinc,
        }

        self.backbone = SincNet(sincnet_params)
        self.mlp_kernels = [11, 9, 7]
        self.channels = [num_channels, 128, 128, 128]
        self.final_kernel = [5]

        self.gcc = GCC(max_tau=self.max_tau, dim=3, filt="phat")

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.channels[i], self.channels[i + 1], kernel_size=k),
                    nn.BatchNorm1d(self.channels[i + 1]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5),
                )
                for i, k in enumerate(self.mlp_kernels)
            ]
        )

        self.final_conv = nn.Conv1d(128, 1, kernel_size=self.final_kernel)

        if head == "regression":
            self.reg = nn.Sequential(
                nn.BatchNorm1d(2 * self.max_tau + 1),
                nn.LeakyReLU(0.2),
                nn.Linear(2 * self.max_tau + 1, 1),
            )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]

        y1 = self.backbone(x1)
        y2 = self.backbone(x2)

        cc = self.gcc(y1, y2)

        for k, layer in enumerate(self.mlp):
            s = cc.shape[2]
            padding = get_pad(size=s, kernel_size=self.mlp_kernels[k], stride=1, dilation=1)
            cc = F.pad(cc, pad=padding, mode="constant")
            cc = layer(cc)

        s = cc.shape[2]
        padding = get_pad(size=s, kernel_size=self.final_kernel, stride=1, dilation=1)
        cc = F.pad(cc, pad=padding, mode="constant")
        cc = self.final_conv(cc).reshape([batch_size, -1])
        if self.head == "regression":
            cc = self.reg(cc).squeeze()

        return cc


class PGCCPHAT(nn.Module):
    def __init__(self, beta=np.arange(0, 1.1, 0.1), max_tau=42, head="regression"):
        super().__init__()

        """
        Implementation of CNN-Based Parametrized GCC-PHAT by Salvati et al.
        https://www.isca-speech.org/archive/pdfs/interspeech_2021/salvati21_interspeech.pdf
        """

        self.beta = beta
        self.gcc = GCC(max_tau=max_tau, dim=3, filt="phat", beta=beta)
        self.head = head
        self.max_tau = max_tau

        if head == "regression":
            n_out = 1
        else:
            n_out = 2 * self.max_tau + 1

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3))
        self.bn5 = nn.BatchNorm2d(512)

        self.mlp = nn.Sequential(
            nn.Linear(512 * (2 * max_tau + 1 - 10), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_out),
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]

        x = self.gcc(x1, x2).unsqueeze(1)

        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.mlp(x.reshape([batch_size, -1])).squeeze()

        return x


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing. The repo's cfg.py
# defaults are max_delay=23, num_channels=128, head='classifier', fs=16000,
# sig_len=2048 for NGCCPHAT (which takes two mic-signal tensors x1, x2). We
# keep max_tau/fs/head at the real defaults (the SincConv_fast filterbank
# math depends on fs; the classifier head's output width is 2*max_tau+1) but
# shrink sig_len and num_channels for a fast CPU trace -- the SincNet
# backbone's `cnn_len_filt=[1023, 11, 9, 7]` kernels each get circularly
# same-padded via get_pad; circular padding requires the pad amount on each
# side to stay below the current input length, so sig_len must comfortably
# exceed the largest kernel (1023) for the very first SincConv_fast layer.
# ---------------------------------------------------------------------------
_SIG_LEN = 1200
_NUM_CHANNELS = 8


def build_ngccphat():
    torch.manual_seed(0)
    model = NGCCPHAT(
        max_tau=23,
        head="classifier",
        use_sinc=True,
        sig_len=_SIG_LEN,
        num_channels=_NUM_CHANNELS,
        fs=16000,
    )
    model.eval()
    return model


def example_input_ngccphat():
    torch.manual_seed(0)
    x1 = torch.randn(2, _SIG_LEN)
    x2 = torch.randn(2, _SIG_LEN)
    return (x1, x2)


MENAGERIE_ENTRIES = [
    ("NGCC-PHAT", "build_ngccphat", "example_input_ngccphat", 2022, MENAGERIE_ZOO),
]

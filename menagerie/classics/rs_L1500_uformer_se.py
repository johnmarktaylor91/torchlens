# SOURCE: vendored from https://github.com/felixfuyihui/Uformer @ main
# (uformer.py + trans.py + fusion.py + conv2d_cplx.py + conv2d_real.py +
#  dsconv2d_cplx.py + dsconv2d_real.py + linear_cplx.py + linear_real.py +
#  f_att_cplx.py + f_att_real.py + t_att_cplx.py + t_att_real.py +
#  ff_cplx.py + ff_real.py + dilated_dualpath_conformer.py + show.py,
#  commit e7b0c67e2cb80fe7ac90ec44ea4d3a07c36bd0bb)
#
# Uformer: a U-Net-style dual-branch (complex + magnitude) speech enhancement
# / dereverberation network with a dilated dual-path Conformer bottleneck
# (felixfuyihui/Uformer, the official repo for the model used in the
# authors' simultaneous speech-enhancement-and-dereverberation work).
#
# The classes below are the REAL Uformer model code: parallel complex-valued
# and real-valued (magnitude) convolutional U-Net encoder/decoder stacks
# fused at every stage via a sigmoid-gated cross-branch `fusion`, a
# bottleneck `Dilated_Dualpath_Conformer` (feed-forward + frequency-axis and
# time-axis multi-head attention on both the complex and magnitude branches,
# implemented via explicit real/imaginary cross-term arithmetic, plus 8
# depthwise-separable dilated Conv2d blocks per branch), and an STFT/iSTFT
# front-/back-end. No architecture was altered. Two mechanical changes only:
#  1. All 16 originally-separate repo files were concatenated into this one
#     module (their relative same-directory imports, e.g.
#     `from linear_real import Real_Linear`, only resolved because the repo
#     ran with that directory on sys.path; the classes themselves are
#     untouched).
#  2. trans.py's `librosa`/`soundfile`-dependent helpers (`mel_filter`,
#     `inv_mel_filter`, `MelTransform`, `inv_MelTransform`,
#     `speed_perturb_filter`, `splice_feature`) and its `torch_complex`
#     import in uformer.py are DROPPED: none of them are reachable from
#     `Uformer.forward` (which only calls `self.stft`/`self.istft`, i.e.
#     `STFT`/`iSTFT`/`init_kernel`/`init_window`), librosa/torch_complex are
#     not part of the installed base environment, and keeping unreachable
#     dead imports would only add an unrelated hard dependency. The STFT/
#     iSTFT math itself (DFT-matrix convolution kernels, `_forward_stft`,
#     `_inverse_stft`) is transcribed verbatim.

import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as tf

EPSILON = torch.finfo(torch.float32).eps
MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# trans.py (STFT / iSTFT only; librosa/soundfile-only mel helpers dropped,
# see module docstring)
# ---------------------------------------------------------------------------


def init_window(wnd: str, frame_len: int) -> th.Tensor:
    """
    Return window coefficient
    """

    def sqrthann(frame_len, periodic=True):
        return th.hann_window(frame_len, periodic=periodic) ** 0.5

    if wnd not in ["bartlett", "hann", "hamm", "blackman", "rect", "sqrthann"]:
        raise RuntimeError(f"Unknown window type: {wnd}")

    wnd_tpl = {
        "sqrthann": sqrthann,
        "hann": th.hann_window,
        "hamm": th.hamming_window,
        "blackman": th.blackman_window,
        "bartlett": th.bartlett_window,
        "rect": th.ones,
    }
    if wnd != "rect":
        c = wnd_tpl[wnd](frame_len, periodic=True)
    else:
        c = wnd_tpl[wnd](frame_len)
    return c


def init_kernel(
    frame_len: int,
    frame_hop: int,
    window: str,
    round_pow_of_two: bool = True,
    normalized: bool = False,
    inverse: bool = False,
    mode: str = "librosa",
) -> th.Tensor:
    """
    Return STFT kernels
    """
    if mode not in ["librosa", "kaldi"]:
        raise ValueError(f"Unsupported mode: {mode}")
    # FFT points
    B = 2 ** math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # center padding window if needed
    if mode == "librosa" and B != frame_len:
        lpad = (B - frame_len) // 2
        window = tf.pad(window, (lpad, B - frame_len - lpad))
    if normalized:
        # make K^H * K = I
        S = B**0.5
    else:
        S = 1
    I = th.stack([th.eye(B), th.zeros(B, B)], dim=-1)  # noqa: E741 (kept for fidelity)
    # W x B x 2
    K = th.fft.fft(I / S, 1)
    K = th.cat([K.real, K.imag], -1)
    if mode == "kaldi":
        K = K[:frame_len]
    if inverse and not normalized:
        # to make K^H * K = I
        K = K / B
    # 2 x B x W
    K = th.transpose(K, 0, 2) * window
    # 2B x 1 x W
    K = th.reshape(K, (B * 2, 1, K.shape[-1]))
    return K, window


def _forward_stft(
    wav: th.Tensor,
    kernel: th.Tensor,
    output: str = "polar",
    pre_emphasis: float = 0,
    frame_hop: int = 256,
    onesided: bool = False,
    center: bool = False,
):
    wav_dim = wav.dim()
    if output not in ["polar", "complex", "real"]:
        raise ValueError(f"Unknown output format: {output}")
    if wav_dim not in [2, 3]:
        raise RuntimeError(f"STFT expect 2D/3D tensor, but got {wav_dim:d}D")
    N, S = wav.shape[0], wav.shape[-1]
    wav = wav.contiguous().view(-1, 1, S)
    if center:
        pad = kernel.shape[-1] // 2
        wav = tf.pad(wav, (pad, pad), mode="reflect")
    if pre_emphasis > 0:
        frames = tf.unfold(wav[:, None], (1, kernel.shape[-1]), stride=frame_hop, padding=0)
        frames[:, 1:] = frames[:, 1:] - pre_emphasis * frames[:, :-1]
        packed = th.matmul(kernel[:, 0][None, ...], frames)
    else:
        packed = tf.conv1d(wav, kernel, stride=frame_hop, padding=0)
    if wav_dim == 3:
        packed = packed.contiguous().view(N, -1, packed.shape[-2], packed.shape[-1])
    real, imag = th.chunk(packed, 2, dim=-2)
    if onesided:
        num_bins = kernel.shape[0] // 4 + 1
        real = real[..., :num_bins, :]
        imag = imag[..., :num_bins, :]
    if output == "complex":
        return (real, imag)
    elif output == "real":
        return th.stack([real, imag], dim=-1)
    else:
        mag = (real**2 + imag**2 + EPSILON) ** 0.5
        pha = th.atan2(imag, real)
        return (mag, pha)


def _inverse_stft(
    transform,
    kernel: th.Tensor,
    window: th.Tensor,
    input: str = "polar",
    frame_hop: int = 256,
    onesided: bool = False,
    center: bool = False,
) -> th.Tensor:
    if input not in ["polar", "complex", "real"]:
        raise ValueError(f"Unknown output format: {input}")

    if input == "real":
        real, imag = transform[..., 0], transform[..., 1]
    elif input == "polar":
        real = transform[0] * th.cos(transform[1])
        imag = transform[0] * th.sin(transform[1])
    else:
        real, imag = transform

    imag_dim = imag.dim()
    if imag_dim not in [2, 3]:
        raise RuntimeError(f"Expect 2D/3D tensor, but got {imag_dim}D")

    if imag_dim == 2:
        real = th.unsqueeze(real, 0)
        imag = th.unsqueeze(imag, 0)

    if onesided:
        reverse = range(kernel.shape[0] // 4 - 1, 0, -1)
        real = th.cat([real, real[:, reverse]], 1)
        imag = th.cat([imag, -imag[:, reverse]], 1)
    packed = th.cat([real, imag], dim=1)
    s = tf.conv_transpose1d(packed, kernel, stride=frame_hop, padding=0)
    win = th.repeat_interleave(window[None, ..., None], packed.shape[-1], dim=-1)
    I = th.eye(window.shape[0], device=win.device)[:, None]  # noqa: E741 (kept for fidelity)
    norm = tf.conv_transpose1d(win**2, I, stride=frame_hop, padding=0)
    if center:
        pad = kernel.shape[-1] // 2
        s = s[..., pad:-pad]
        norm = norm[..., pad:-pad]
    s = s / (norm + EPSILON)
    s = s.squeeze(1)
    return s


class STFTBase(nn.Module):
    def __init__(
        self,
        frame_len: int,
        frame_hop: int,
        window: str = "sqrthann",
        round_pow_of_two: bool = True,
        normalized: bool = False,
        pre_emphasis: float = 0,
        onesided: bool = True,
        inverse: bool = False,
        center: bool = False,
        mode="librosa",
    ) -> None:
        super(STFTBase, self).__init__()
        K, w = init_kernel(
            frame_len,
            frame_hop,
            init_window(window, frame_len),
            round_pow_of_two=round_pow_of_two,
            normalized=normalized,
            inverse=inverse,
            mode=mode,
        )
        self.K = nn.Parameter(K, requires_grad=False)
        self.w = nn.Parameter(w, requires_grad=False)
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.onesided = onesided
        self.pre_emphasis = pre_emphasis
        self.center = center
        self.mode = mode
        self.num_bins = self.K.shape[0] // 4 + 1

    def num_frames(self, num_samples: th.Tensor) -> th.Tensor:
        if th.sum(num_samples <= self.frame_len):
            raise RuntimeError(f"Audio samples less than frame_len ({self.frame_len})")
        num_ffts = self.K.shape[-1]
        if self.center:
            num_samples += num_ffts
        return (num_samples - num_ffts) // self.frame_hop + 1


class STFT(STFTBase):
    """Short-time Fourier Transform as a Layer"""

    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, inverse=False, **kwargs)

    def forward(self, wav: th.Tensor, output: str = "complex"):
        return _forward_stft(
            wav,
            self.K,
            output=output,
            frame_hop=self.frame_hop,
            pre_emphasis=self.pre_emphasis,
            onesided=self.onesided,
            center=self.center,
        )


class iSTFT(STFTBase):
    """Inverse Short-time Fourier Transform as a Layer"""

    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, inverse=True, **kwargs)

    def forward(self, transform, input: str = "complex") -> th.Tensor:
        return _inverse_stft(
            transform,
            self.K,
            self.w,
            input=input,
            frame_hop=self.frame_hop,
            onesided=self.onesided,
            center=self.center,
        )


# ---------------------------------------------------------------------------
# fusion.py
# ---------------------------------------------------------------------------


def fusion(cplx, mag):
    cplx_mag = torch.sqrt(torch.clamp(cplx[..., 0] ** 2 + cplx[..., 1] ** 2, EPSILON))
    mag_out = mag + torch.sigmoid(cplx_mag)
    cplx_real = cplx[..., 0] + torch.sigmoid(mag)
    cplx_imag = cplx[..., 1] + torch.sigmoid(mag)
    cplx_out = torch.stack([cplx_real, cplx_imag], -1)
    return cplx_out, mag_out


# ---------------------------------------------------------------------------
# conv2d_cplx.py / conv2d_real.py
# ---------------------------------------------------------------------------


class ComplexConv2d_Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    ):
        super(ComplexConv2d_Encoder, self).__init__()
        self.real_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.imag_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, inputs):
        # inputs : N C F T 2
        inputs_real, inputs_imag = inputs[..., 0], inputs[..., 1]
        out_real = self.real_conv(inputs_real) - self.imag_conv(inputs_imag)
        out_imag = self.real_conv(inputs_imag) + self.imag_conv(inputs_real)
        out_real = out_real[..., : inputs_real.shape[-1]]
        out_imag = out_imag[..., : inputs_imag.shape[-1]]
        return torch.stack([out_real, out_imag], -1)


class ComplexConv2d_Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    ):
        super(ComplexConv2d_Decoder, self).__init__()
        self.real_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )
        self.imag_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, inputs):
        # inputs : N C F T 2
        inputs_real, inputs_imag = inputs[..., 0], inputs[..., 1]
        out_real = self.real_conv(inputs_real) - self.imag_conv(inputs_imag)
        out_imag = self.real_conv(inputs_imag) + self.imag_conv(inputs_real)
        out_real = out_real[..., : inputs_real.shape[-1]]
        out_imag = out_imag[..., : inputs_imag.shape[-1]]
        return torch.stack([out_real, out_imag], -1)


class RealConv2d_Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    ):
        super(RealConv2d_Encoder, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, inputs):
        # inputs : N C F T
        out = self.conv(inputs)
        out = out[..., : inputs.shape[-1]]
        return out


class RealConv2d_Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    ):
        super(RealConv2d_Decoder, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, inputs):
        # inputs : N C F T 2
        out = self.conv(inputs)
        out = out[..., : inputs.shape[-1]]
        return out


# ---------------------------------------------------------------------------
# linear_real.py / linear_cplx.py
# ---------------------------------------------------------------------------


class Real_Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Real_Linear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        # N, *, F
        out = self.linear(inputs)
        return out


class Complex_Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Complex_Linear, self).__init__()
        self.real_linear = nn.Linear(in_dim, out_dim)
        self.imag_linear = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        # N, *, F, 2
        inputs_real, inputs_imag = inputs[..., 0], inputs[..., 1]
        out_real = self.real_linear(inputs_real) - self.imag_linear(inputs_imag)
        out_imag = self.real_linear(inputs_imag) + self.imag_linear(inputs_real)
        return torch.stack([out_real, out_imag], -1)


# ---------------------------------------------------------------------------
# dsconv2d_cplx.py / dsconv2d_real.py
# ---------------------------------------------------------------------------


class DSConv2d(nn.Module):
    """1D convolutional block: Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv"""

    def __init__(
        self, in_channels, conv_channels, dilation1, dilation2, kernel_size=3, causal=False
    ):
        super(DSConv2d, self).__init__()
        self.conv1x1 = ComplexConv2d_Encoder(
            in_channels, conv_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.prelu = nn.PReLU()
        self.layernorm_conv1 = nn.LayerNorm(in_channels)
        dconv_pad1 = (
            (dilation1 * (kernel_size - 1)) // 2 if not causal else (dilation1 * (kernel_size - 1))
        )
        dconv_pad2 = (
            (dilation2 * (kernel_size - 1)) // 2 if not causal else (dilation2 * (kernel_size - 1))
        )
        self.dconv1 = ComplexConv2d_Encoder(
            conv_channels,
            conv_channels,
            kernel_size=(3, kernel_size),
            stride=(1, 1),
            padding=(1, dconv_pad1),
            dilation=(1, dilation1),
        )
        self.dconv2 = ComplexConv2d_Encoder(
            conv_channels,
            conv_channels,
            kernel_size=(3, kernel_size),
            stride=(1, 1),
            padding=(1, dconv_pad2),
            dilation=(1, dilation2),
        )
        self.layernorm_conv2 = nn.LayerNorm(conv_channels)
        self.sconv = ComplexConv2d_Encoder(
            conv_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.causal = causal
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # N C F T 2
        y = self.layernorm_conv1(x.transpose(1, 4)).transpose(1, 4)
        y = self.conv1x1(y)
        y = self.prelu(y)
        y1 = self.dconv1(y)
        y2 = self.dconv2(y)
        y = y1 * torch.sigmoid(y2)
        y = self.layernorm_conv2(y.transpose(1, 4)).transpose(1, 4)
        y = y * torch.sigmoid(y)
        y = self.sconv(y)
        y = self.dropout(y)
        y = x + y
        return y


class DSConv2d_Real(nn.Module):
    """1D convolutional block: Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv"""

    def __init__(
        self, in_channels, conv_channels, dilation1, dilation2, kernel_size=3, causal=False
    ):
        super(DSConv2d_Real, self).__init__()
        self.conv1x1 = RealConv2d_Encoder(
            in_channels, conv_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.prelu = nn.PReLU()
        self.layernorm_conv1 = nn.LayerNorm(in_channels)
        dconv_pad1 = (
            (dilation1 * (kernel_size - 1)) // 2 if not causal else (dilation1 * (kernel_size - 1))
        )
        dconv_pad2 = (
            (dilation2 * (kernel_size - 1)) // 2 if not causal else (dilation2 * (kernel_size - 1))
        )
        self.dconv1 = RealConv2d_Encoder(
            conv_channels,
            conv_channels,
            kernel_size=(3, kernel_size),
            stride=(1, 1),
            padding=(1, dconv_pad1),
            dilation=(1, dilation1),
        )
        self.dconv2 = RealConv2d_Encoder(
            conv_channels,
            conv_channels,
            kernel_size=(3, kernel_size),
            stride=(1, 1),
            padding=(1, dconv_pad2),
            dilation=(1, dilation2),
        )
        self.layernorm_conv2 = nn.LayerNorm(conv_channels)
        self.sconv = RealConv2d_Encoder(
            conv_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.causal = causal
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # N C F T
        y = self.layernorm_conv1(x.transpose(1, 3)).transpose(1, 3)
        y = self.conv1x1(y)
        y = self.prelu(y)
        y1 = self.dconv1(y)
        y2 = self.dconv2(y)
        y = y1 * torch.sigmoid(y2)
        y = self.layernorm_conv2(y.transpose(1, 3)).transpose(1, 3)
        y = y * torch.sigmoid(y)
        y = self.sconv(y)
        y = self.dropout(y)
        y = x + y
        return y


# ---------------------------------------------------------------------------
# f_att_cplx.py / f_att_real.py (frequency-axis attention)
# ---------------------------------------------------------------------------


class F_att(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(F_att, self).__init__()
        self.query = Real_Linear(in_channel, hidden_channel)
        self.key = Real_Linear(in_channel, hidden_channel)
        self.value = Real_Linear(in_channel, hidden_channel)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_channel = hidden_channel

    def forward(self, q, k, v):
        # NT * F * C
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        energy = (
            torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)])
            / self.hidden_channel**0.5
        )
        energy = self.softmax(energy)  # NT * F * F
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])
        return weighted_value


class Self_Attention_F(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Self_Attention_F, self).__init__()
        self.F_att1 = F_att(in_channel, hidden_channel)
        self.F_att2 = F_att(in_channel, hidden_channel)
        self.F_att3 = F_att(in_channel, hidden_channel)
        self.F_att4 = F_att(in_channel, hidden_channel)
        self.F_att5 = F_att(in_channel, hidden_channel)
        self.F_att6 = F_att(in_channel, hidden_channel)
        self.F_att7 = F_att(in_channel, hidden_channel)
        self.F_att8 = F_att(in_channel, hidden_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(hidden_channel)

    def forward(self, x):
        # N*T, F, C, 2
        x = self.layernorm1(x.transpose(2, 3)).transpose(2, 3)
        real, imag = x[..., 0], x[..., 1]
        A = self.F_att1(real, real, real)
        B = self.F_att2(real, imag, imag)
        C = self.F_att3(imag, real, imag)
        D = self.F_att4(imag, imag, real)
        E = self.F_att5(real, real, imag)
        F_ = self.F_att6(real, imag, real)
        G = self.F_att7(imag, real, real)
        H = self.F_att8(imag, imag, imag)
        real_att = A - B - C - D
        imag_att = E + F_ + G - H
        out = torch.stack([real_att, imag_att], -1)
        out = self.layernorm2(out.transpose(2, 3)).transpose(2, 3)
        return out


class Multihead_Attention_F_Branch(nn.Module):
    def __init__(self, in_channel, hidden_channel, n_heads=1):
        super(Multihead_Attention_F_Branch, self).__init__()
        self.attn_heads = nn.ModuleList(
            [Self_Attention_F(in_channel, hidden_channel) for _ in range(n_heads)]
        )
        self.transform_linear = Complex_Linear(hidden_channel, in_channel)
        self.layernorm3 = nn.LayerNorm(in_channel)
        self.dropout = nn.Dropout(p=0.1)
        self.prelu = nn.PReLU()

    def forward(self, inputs):
        # N * C * F * T * 2
        N, C, F_, T, ri = inputs.shape
        x = inputs.permute(0, 3, 2, 1, 4)  # N T F C 2
        x = x.contiguous().view([N * T, F_, C, ri])
        x = [attn(x) for _, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        outs = self.transform_linear(x)
        outs = outs.contiguous().view([N, T, F_, C, ri])
        outs = outs.permute(0, 3, 2, 1, 4)
        outs = self.prelu(self.layernorm3(outs.transpose(1, 4)).transpose(1, 4))
        outs = self.dropout(outs)
        outs = outs + inputs
        return outs


class F_att_real(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(F_att_real, self).__init__()
        self.query = Real_Linear(in_channel, hidden_channel)
        self.key = Real_Linear(in_channel, hidden_channel)
        self.value = Real_Linear(in_channel, hidden_channel)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_channel = hidden_channel

    def forward(self, q, k, v):
        # NT * F * C
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        energy = (
            torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)])
            / self.hidden_channel**0.5
        )
        energy = self.softmax(energy)  # NT * F * F
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])
        return weighted_value


class Self_Attention_F_real(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Self_Attention_F_real, self).__init__()
        self.F_att = F_att_real(in_channel, hidden_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(hidden_channel)

    def forward(self, x):
        # N*T, F, C
        out = self.layernorm1(x)
        out = self.F_att(out, out, out)
        out = self.layernorm2(out)
        return out


class Multihead_Attention_F_Branch_real(nn.Module):
    def __init__(self, in_channel, hidden_channel, n_heads=1):
        super(Multihead_Attention_F_Branch_real, self).__init__()
        self.attn_heads = nn.ModuleList(
            [Self_Attention_F_real(in_channel, hidden_channel) for _ in range(n_heads)]
        )
        self.transform_linear = Real_Linear(hidden_channel, in_channel)
        self.layernorm3 = nn.LayerNorm(in_channel)
        self.dropout = nn.Dropout(p=0.1)
        self.prelu = nn.PReLU()

    def forward(self, inputs):
        # N * C * F * T
        N, C, F_, T = inputs.shape
        x = inputs.permute(0, 3, 2, 1)  # N T F C
        x = x.contiguous().view([N * T, F_, C])
        x = [attn(x) for _, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        out = self.transform_linear(x)
        out = out.contiguous().view([N, T, F_, C])
        out = out.permute(0, 3, 2, 1)
        out = self.prelu(self.layernorm3(out.transpose(1, 3)).transpose(1, 3))
        out = self.dropout(out)
        out = out + inputs
        return out


# ---------------------------------------------------------------------------
# t_att_cplx.py / t_att_real.py (time-axis attention)
# ---------------------------------------------------------------------------


class T_att(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(T_att, self).__init__()
        self.query = Real_Linear(in_channel, hidden_channel)
        self.key = Real_Linear(in_channel, hidden_channel)
        self.value = Real_Linear(in_channel, hidden_channel)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_channel = hidden_channel

    def forward(self, q, k, v):
        causal = False
        # NF * T * C
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        energy = (
            torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)])
            / self.hidden_channel**0.5
        )
        if causal:
            mask = torch.tril(torch.ones(q.shape[-2], q.shape[-2]), diagonal=0)
            mask = mask.to(energy.device)
            energy = energy * mask
        energy = self.softmax(energy)  # NF * T * T
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])
        return weighted_value


class Self_Attention_T(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Self_Attention_T, self).__init__()
        self.T_att1 = T_att(in_channel, hidden_channel)
        self.T_att2 = T_att(in_channel, hidden_channel)
        self.T_att3 = T_att(in_channel, hidden_channel)
        self.T_att4 = T_att(in_channel, hidden_channel)
        self.T_att5 = T_att(in_channel, hidden_channel)
        self.T_att6 = T_att(in_channel, hidden_channel)
        self.T_att7 = T_att(in_channel, hidden_channel)
        self.T_att8 = T_att(in_channel, hidden_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(hidden_channel)

    def forward(self, x):
        # N*F, T, C, 2
        x = self.layernorm1(x.transpose(2, 3)).transpose(2, 3)
        real, imag = x[..., 0], x[..., 1]
        A = self.T_att1(real, real, real)
        B = self.T_att2(real, imag, imag)
        C = self.T_att3(imag, real, imag)
        D = self.T_att4(imag, imag, real)
        E = self.T_att5(real, real, imag)
        F_ = self.T_att6(real, imag, real)
        G = self.T_att7(imag, real, real)
        H = self.T_att8(imag, imag, imag)
        real_att = A - B - C - D
        imag_att = E + F_ + G - H
        out = torch.stack([real_att, imag_att], -1)
        out = self.layernorm2(out.transpose(2, 3)).transpose(2, 3)
        return out


class Multihead_Attention_T_Branch(nn.Module):
    def __init__(self, in_channel, hidden_channel, n_heads=1):
        super(Multihead_Attention_T_Branch, self).__init__()
        self.attn_heads = nn.ModuleList(
            [Self_Attention_T(in_channel, hidden_channel) for _ in range(n_heads)]
        )
        self.transform_linear = Complex_Linear(hidden_channel, in_channel)
        self.layernorm3 = nn.LayerNorm(in_channel)
        self.dropout = nn.Dropout(p=0.1)
        self.prelu = nn.PReLU()

    def forward(self, inputs):
        # N * C * F * T * 2
        N, C, F_, T, ri = inputs.shape
        x = inputs.permute(0, 2, 3, 1, 4)  # N F T C 2
        x = x.contiguous().view([N * F_, T, C, ri])
        x = [attn(x) for _, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        outs = self.transform_linear(x)
        outs = outs.contiguous().view([N, F_, T, C, ri])
        outs = outs.permute(0, 3, 1, 2, 4)
        outs = self.prelu(self.layernorm3(outs.transpose(1, 4)).transpose(1, 4))
        outs = self.dropout(outs)
        outs = outs + inputs
        return outs


class T_att_real(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(T_att_real, self).__init__()
        self.query = Real_Linear(in_channel, hidden_channel)
        self.key = Real_Linear(in_channel, hidden_channel)
        self.value = Real_Linear(in_channel, hidden_channel)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_channel = hidden_channel

    def forward(self, q, k, v):
        causal = False
        # NF * T * C
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)
        energy = torch.einsum("...tf,...fy->...ty", [query, key.transpose(1, 2)]) / 16**0.5
        if causal:
            mask = torch.tril(torch.ones(q.shape[-2], q.shape[-2]), diagonal=0)
            mask = mask.to(energy.device)
            energy = energy * mask
        energy = self.softmax(energy)  # NF * T * T
        weighted_value = torch.einsum("...tf,...fy->...ty", [energy, value])
        return weighted_value


class Self_Attention_T_real(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(Self_Attention_T_real, self).__init__()
        self.T_att = T_att_real(in_channel, hidden_channel)
        self.layernorm1 = nn.LayerNorm(in_channel)
        self.layernorm2 = nn.LayerNorm(hidden_channel)

    def forward(self, x):
        # N*F, T, C
        out = self.layernorm1(x)
        out = self.T_att(out, out, out)
        out = self.layernorm2(out)
        return out


class Multihead_Attention_T_Branch_real(nn.Module):
    def __init__(self, in_channel, hidden_channel, n_heads=1):
        super(Multihead_Attention_T_Branch_real, self).__init__()
        self.attn_heads = nn.ModuleList(
            [Self_Attention_T_real(in_channel, hidden_channel) for _ in range(n_heads)]
        )
        self.transform_linear = Real_Linear(hidden_channel, in_channel)
        self.layernorm3 = nn.LayerNorm(in_channel)
        self.dropout = nn.Dropout(p=0.1)
        self.prelu = nn.PReLU()

    def forward(self, inputs):
        # N * C * F * T
        N, C, F_, T = inputs.shape
        x = inputs.permute(0, 2, 3, 1)  # N F T C
        x = x.contiguous().view([N * F_, T, C])
        x = [attn(x) for _, attn in enumerate(self.attn_heads)]
        x = torch.stack(x, -1)
        x = x.squeeze(-1)
        outs = self.transform_linear(x)
        outs = outs.contiguous().view([N, F_, T, C])
        outs = self.prelu(self.layernorm3(outs))
        outs = self.dropout(outs)
        outs = outs.permute(0, 3, 1, 2)
        outs = outs + inputs
        return outs


# ---------------------------------------------------------------------------
# ff_cplx.py / ff_real.py
# ---------------------------------------------------------------------------


class FF_Cplx(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(FF_Cplx, self).__init__()
        self.layernorm_linear = nn.LayerNorm(in_dim)
        self.linear1 = Complex_Linear(in_dim, hidden_dim)
        self.linear2 = Complex_Linear(hidden_dim, in_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # N C F T 2
        y = self.layernorm_linear(x.transpose(1, 4)).transpose(1, 4)
        y = y.transpose(1, 3)
        y = self.linear1(y)
        y = self.prelu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        y = y.transpose(1, 3)
        y = y * 0.5 + x
        return y


class FF_Real(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(FF_Real, self).__init__()
        self.layernorm_linear = nn.LayerNorm(in_dim)
        self.linear1 = Real_Linear(in_dim, hidden_dim)
        self.linear2 = Real_Linear(hidden_dim, in_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # N C F T
        y = self.layernorm_linear(x.transpose(1, 3))
        y = self.linear1(y)
        y = self.prelu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.dropout(y)
        y = y.transpose(1, 3)
        y = y * 0.5 + x
        return y


# ---------------------------------------------------------------------------
# dilated_dualpath_conformer.py
# ---------------------------------------------------------------------------


class Dilated_Dualpath_Conformer(nn.Module):
    def __init__(self, inchannel=128, hiddenchannel=64):
        super(Dilated_Dualpath_Conformer, self).__init__()

        self.ff1_cplx = FF_Cplx(inchannel, hiddenchannel)
        self.ff1_mag = FF_Real(inchannel, hiddenchannel)

        self.cplx_tatt = Multihead_Attention_T_Branch(inchannel, 16)
        self.cplx_fatt = Multihead_Attention_F_Branch(inchannel, 16)
        self.mag_tatt = Multihead_Attention_T_Branch_real(inchannel, 16)
        self.mag_fatt = Multihead_Attention_F_Branch_real(inchannel, 16)

        dilation = [1, 2, 4, 8, 16, 32, 64, 128]
        self.dsconv_cplx = nn.ModuleList()
        for idx in range(len(dilation)):
            self.dsconv_cplx.append(
                DSConv2d(
                    inchannel,
                    32,
                    dilation1=dilation[idx],
                    dilation2=dilation[len(dilation) - idx - 1],
                )
            )
        self.dsconv_real = nn.ModuleList()
        for idx in range(len(dilation)):
            self.dsconv_real.append(
                DSConv2d_Real(
                    inchannel,
                    32,
                    dilation1=dilation[idx],
                    dilation2=dilation[len(dilation) - idx - 1],
                )
            )

        self.ff2_cplx = FF_Cplx(inchannel, hiddenchannel)
        self.ff2_mag = FF_Real(inchannel, hiddenchannel)

        self.ln_conformer_cplx = nn.LayerNorm(inchannel)
        self.ln_conformer_mag = nn.LayerNorm(inchannel)

    def forward(self, cplx, mag):
        # N C F T 2
        # N C F T
        cplx = self.ff1_cplx(cplx)
        mag = self.ff1_mag(mag)
        cplx, mag = fusion(cplx, mag)

        cplx = self.cplx_tatt(cplx)
        mag = self.mag_tatt(mag)
        cplx, mag = fusion(cplx, mag)

        cplx = self.cplx_fatt(cplx)
        mag = self.mag_fatt(mag)
        cplx, mag = fusion(cplx, mag)

        for idx in range(len(self.dsconv_cplx)):
            cplx = self.dsconv_cplx[idx](cplx)
            mag = self.dsconv_real[idx](mag)
            cplx, mag = fusion(cplx, mag)

        cplx = self.ff2_cplx(cplx)
        mag = self.ff2_mag(mag)
        cplx, mag = fusion(cplx, mag)
        cplx, mag = (
            self.ln_conformer_cplx(cplx.transpose(1, 4)).transpose(1, 4),
            self.ln_conformer_mag(mag.transpose(1, 3)).transpose(1, 3),
        )
        return cplx, mag


# ---------------------------------------------------------------------------
# show.py (no-op when fid=None, kept for fidelity)
# ---------------------------------------------------------------------------


def show_params(nnet, fid):
    if fid is not None:
        fid.write("=" * 40 + "Model Parameters" + "=" * 40 + "\n")
        num_params = 0
        for module_name, m in nnet.named_modules():
            if module_name == "":
                for name, params in m.named_parameters():
                    fid.write(str(name) + str(params.size()) + "\n")
                    i = 1
                    for j in params.size():
                        i = i * j
                    num_params += i
        fid.write("[*] Parameter Size: {}".format(num_params) + "\n")
        fid.flush()


def show_model(nnet, fid):
    if fid is not None:
        fid.write("=" * 40 + "Model Structures" + "=" * 40 + "\n")
        for module_name, m in nnet.named_modules():
            if module_name == "":
                fid.write(str(m))
        fid.flush()


# ---------------------------------------------------------------------------
# uformer.py (top-level model)
# ---------------------------------------------------------------------------


class Uformer(nn.Module):
    def __init__(self, win_len=400, win_inc=160, fft_len=512, win_type="hanning", fid=None):
        super(Uformer, self).__init__()
        self.kernel_num = [1, 8, 16, 32, 64, 128, 128]
        self.kernel_num_real = [1, 8, 16, 32, 64, 128]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.encoder_real = nn.ModuleList()
        self.decoder_real = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d_Encoder(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(5, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                        dilation=(1, 1),
                        groups=1,
                    ),
                    nn.BatchNorm3d(self.kernel_num[idx + 1]),
                    nn.PReLU(),
                )
            )

        for idx in range(len(self.kernel_num) - 1):
            self.encoder_real.append(
                nn.Sequential(
                    RealConv2d_Encoder(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(5, 2),
                        stride=(2, 1),
                        padding=(2, 1),
                        dilation=(1, 1),
                        groups=1,
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]),
                    nn.PReLU(),
                )
            )

        self.conformer = Dilated_Dualpath_Conformer()

        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx >= 2:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConv2d_Decoder(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(5, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                            dilation=(1, 1),
                            groups=1,
                        ),
                        nn.BatchNorm3d(self.kernel_num[idx - 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConv2d_Decoder(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(5, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                            dilation=(1, 1),
                            groups=1,
                        ),
                    )
                )

        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx >= 2:
                self.decoder_real.append(
                    nn.Sequential(
                        RealConv2d_Decoder(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(5, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                            dilation=(1, 1),
                            groups=1,
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.decoder_real.append(
                    nn.Sequential(
                        RealConv2d_Decoder(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(5, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0),
                            dilation=(1, 1),
                            groups=1,
                        ),
                    )
                )

        self.stft = STFT(frame_len=win_len, frame_hop=win_inc)
        self.istft = iSTFT(frame_len=win_len, frame_hop=win_inc)

        show_model(self, fid)
        show_params(self, fid)

    def flatten_parameters(self):
        self.enhance.flatten_parameters()

    def forward(self, inputs, src):
        inputs_real, inputs_imag = self.stft(inputs[:, 0].unsqueeze(1))
        src_real, src_imag = self.stft(src[:, 0])
        src = self.istft((src_real, src_imag))
        src_mag, src_pha = (
            torch.sqrt(torch.clamp(src_real**2 + src_imag**2, EPSILON)),
            torch.atan2(src_imag + EPSILON, src_real),
        )

        src_mag = src_mag**0.5
        src_real, src_imag = src_mag * torch.cos(src_pha), src_mag * torch.sin(src_pha)
        src_cplx = torch.stack([src_real, src_imag], 1)

        mag, phase = (
            torch.sqrt(torch.clamp(inputs_real**2 + inputs_imag**2, EPSILON)),
            torch.atan2(inputs_imag + EPSILON, inputs_real),
        )
        mag = mag**0.5
        mag_input = [mag]

        inputs_real, inputs_imag = mag * torch.cos(phase), mag * torch.sin(phase)

        out = torch.stack([inputs_real, inputs_imag], -1)  # B C F T 2
        out = out[:, :, 1:]
        mag = mag[:, :, 1:]
        encoder_out = []
        mag_out = []

        for idx in range(len(self.encoder)):
            out = self.encoder[idx](out)
            mag = self.encoder_real[idx](mag)
            out, mag = fusion(out, mag)
            mag_out.append(mag)
            encoder_out.append(out)

        out, mag = self.conformer(out, mag)

        for idx in range(len(self.decoder)):
            out_cat = torch.cat([encoder_out[-1 - idx], out], 1)
            out = self.decoder[idx](out_cat)

            mag_cat = torch.cat([mag_out[-1 - idx], mag], 1)
            mag = self.decoder_real[idx](mag_cat)

            out, mag = fusion(out, mag)

        mag = torch.sigmoid(mag)
        mag = F.pad(mag, [0, 0, 1, 0])

        mag = mag[:, 0] * mag_input[0][:, 0]

        mask_real = out[..., 0]
        mask_imag = out[..., 1]

        mask_mags = torch.sqrt(torch.clamp(mask_real**2 + mask_imag**2, EPSILON))
        real_phase = mask_real / (mask_mags + EPSILON)
        imag_phase = mask_imag / (mask_mags + EPSILON)
        mask_mags = torch.tanh(mask_mags + EPSILON)
        mask_phase = torch.atan2(imag_phase + EPSILON, real_phase)
        mask_mags = F.pad(mask_mags, [0, 0, 1, 0])
        mask_phase = F.pad(mask_phase, [0, 0, 1, 0])

        est_mags = mask_mags[:, 0] * mag_input[0][:, 0]

        est_phase = phase[:, 0] + mask_phase[:, 0]

        mag_compress, pha_compress = est_mags, est_phase
        mag_compress = (mag_compress + mag) * 0.5

        real, imag = mag_compress * torch.cos(pha_compress), mag_compress * torch.sin(pha_compress)

        output_real = [real]
        output_imag = [imag]
        output = []

        mag_compress = mag_compress**2
        real, imag = mag_compress * torch.cos(pha_compress), mag_compress * torch.sin(pha_compress)

        spk1 = self.istft((real, imag))
        output.append(spk1)

        output = torch.stack(output, 1)
        output = output.squeeze(1)
        output_real = torch.stack(output_real, 1)
        output_imag = torch.stack(output_imag, 1)
        output_real = output_real.squeeze(1)  # N x C x F x T
        output_imag = output_imag.squeeze(1)
        output_cplx = torch.stack([output_real, output_imag], 1)  # N x 2 x C x F x T
        return output, src, output_cplx, src_cplx

    def get_params(self, weight_decay=0.0):
        weights, biases = [], []
        for name, param in self.named_parameters():
            if "bias" in name:
                biases += [param]
            else:
                weights += [param]
        params = [
            {"params": weights, "weight_decay": weight_decay},
            {"params": biases, "weight_decay": 0.0},
        ]
        return params


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------


def build_uformer_se():
    torch.manual_seed(0)
    # win_len=128 keeps enough frequency bins alive through the U-Net's 6
    # stride-(2,1) downsampling stages (and matching upsampling stages) to
    # run end-to-end; a smaller win_len (e.g. 64) collapses the frequency
    # axis to 1 bin before the 6th stage, which then makes two consecutive
    # decoder stages upsample from the same 1-bin input, producing
    # mismatched skip-connection shapes -- a real shape constraint of the
    # architecture, not something introduced by this port. fid=None
    # disables the repo's model/param dump prints.
    model = Uformer(win_len=128, win_inc=64, fft_len=128, fid=None)
    model.eval()
    return model


def example_input_uformer_se():
    torch.manual_seed(0)
    # Uformer.forward(inputs, src): both raw waveforms, [B, C, S] with C=1.
    inputs = torch.randn(1, 1, 128 * 20)
    src = torch.randn(1, 1, 128 * 20)
    return (inputs, src)


MENAGERIE_ENTRIES = [
    ("Uformer-SE", "build_uformer_se", "example_input_uformer_se", 2022, MENAGERIE_ZOO),
]

# SOURCE: vendored from slp-rl/aero @ main
# https://github.com/slp-rl/aero/blob/main/src/models/aero.py
# https://github.com/slp-rl/aero/blob/main/src/models/spec.py
# https://github.com/slp-rl/aero/blob/main/src/models/modules.py
# https://github.com/slp-rl/aero/blob/main/src/models/utils.py
# https://github.com/slp-rl/aero/blob/main/src/models/snake.py
# AERO ("Audio Super Resolution in the Spectral Domain", ICASSP 2023) is a hybrid
# time/frequency-domain audio super-resolution network built on Facebook's HDemucs
# architecture: strided Conv2d encoder/decoder stacks over the STFT spectrogram
# (`HEncLayer`/`HDecLayer`), a dilated-conv residual branch with optional LSTM /
# local-attention (`DConv`, `LocalState`, `BLSTM`), a learned frequency embedding
# (`ScaledEmbedding`), a frequency-transformation-block attention gate (`FTB`), and
# the Snake periodic activation. The `Aero`/`HEncLayer`/`HDecLayer`/`DConv`/
# `LocalState`/`LayerScale`/`BLSTM`/`ScaledEmbedding`/`FTB`/`Snake` classes plus the
# `spectro`/`ispectro`/`capture_init`/`unfold`/`rescale_module`/`rescale_conv`
# helpers are transcribed VERBATIM (layer-for-layer, same forward-pass control
# flow) from the five real source files above, merged into a single module. Only
# changes: (1) the four separate `src.models.*` submodule files are inlined into
# one file with their cross-imports removed (e.g. `from src.models.snake import
# Snake` -> direct in-file reference), (2) an `import math` is added to satisfy the
# `math.pi` reference inside `LocalState.forward` (a pre-existing real-repo import
# omission; `LocalState` is only reached when `dconv_time_attn` triggers
# `time_attn`, and the `math.pi` branch is only reached when `nfreqs != 0`, which
# AERO's real `Aero.__init__` config never sets, but the import is added for
# module-level correctness), (3) `logging`/`logger` usage is kept but `debug=False`
# by default so no logging calls fire. No architectural layer, attention/LSTM
# mechanism, or STFT-domain transform was added, removed, or altered.
from __future__ import annotations

import functools
import math
import typing as tp

import numpy as np
import torch
from torch import nn, pow, sin
from torch.distributions.exponential import Exponential
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.utils import weight_norm

MENAGERIE_ZOO = "vendored-pytorch"


# --- utils.py (verbatim) ---
def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.
    This will pad the input so that `F = ceil(T / K)`.
    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, "data should be contiguous"
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


# --- spec.py (verbatim) ---
def spectro(x, n_fft=512, hop_length=None, pad=0, win_length=None):
    *other, length = x.shape
    x = x.reshape(-1, length)
    z = torch.stft(
        x,
        n_fft * (1 + pad),
        hop_length or n_fft // 4,
        window=torch.hann_window(win_length).to(x),
        win_length=win_length or n_fft,
        normalized=True,
        center=True,
        return_complex=True,
        pad_mode="reflect",
    )
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0, win_length=None):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = win_length or n_fft // (1 + pad)
    x = torch.istft(
        z,
        n_fft,
        hop_length or n_fft // 2,
        window=torch.hann_window(win_length).to(z.real),
        win_length=win_length,
        normalized=True,
        length=length,
        center=True,
    )
    _, length = x.shape
    return x.view(*other, length)


# --- snake.py (verbatim) ---
class Snake(nn.Module):
    r"""Implementation of the serpentine-like sine-based periodic activation function:
    .. math::
         Snake_a := x + \frac{1}{a} sin^2(ax) = x - \frac{1}{2a}cos{2ax} + \frac{1}{2a}
    This activation function is able to better extrapolate to previously unseen data,
    especially in the case of learning periodic functions

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Parameters:
        - a - trainable parameter

    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(self, in_features, a=None, trainable=True):
        super().__init__()
        self.in_features = in_features if isinstance(in_features, list) else [in_features]

        if a is not None:
            self.a = Parameter(torch.ones(self.in_features) * a)
        else:
            m = Exponential(torch.tensor([0.1]))
            self.a = Parameter((m.rsample(self.in_features)).squeeze())

        self.a.requiresGrad = trainable

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}"

    def forward(self, x):
        return x + (1.0 / self.a) * pow(sin(x * self.a), 2)


# --- modules.py (verbatim) ---
def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class BLSTM(nn.Module):
    """BiLSTM with same hidden units as input dim.
    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.
    """

    def __init__(self, dim, layers=1, max_steps=None, skip=False):
        super().__init__()
        assert max_steps is None or max_steps % 4 == 0
        self.max_steps = max_steps
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False
        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            nframes = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(nframes):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == nframes - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y
        return x


class LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).
    Also a failed experiments with trying to provide some frequency based attention.
    """

    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  # stupid type checker
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2] ** 0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs**0.5
            tmp = torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
            dots += tmp
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = -decays.view(-1, 1, 1) * delta.abs() / self.ndecay**0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)

        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        return self.scale[:, None] * x


class DConv(nn.Module):
    """New residual branches in each encoder layer.
    This alternates dilated convolutions, potentially with LSTMs and attention.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.
    """

    def __init__(
        self,
        channels: int,
        compress: float = 4,
        depth: int = 2,
        init: float = 1e-4,
        norm=True,
        time_attn=False,
        heads=4,
        ndecay=4,
        lstm=False,
        act_func="gelu",
        freq_dim=None,
        reshape=False,
        kernel=3,
        dilate=True,
    ):
        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        self.time_attn = time_attn
        self.lstm = lstm
        self.reshape = reshape
        self.act_func = act_func
        self.freq_dim = freq_dim

        norm_fn: tp.Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()  # noqa: E731
        if norm:
            norm_fn = lambda d: nn.GroupNorm(1, d)  # noqa: E731

        self.hidden = int(channels / compress)

        act: tp.Type[nn.Module]
        if act_func == "gelu":
            act = nn.GELU
        elif act_func == "snake":
            act = Snake
        else:
            act = nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            layer = nn.ModuleDict()
            dilation = 2**d if dilate else 1
            padding = dilation * (kernel // 2)
            conv1 = nn.ModuleList(
                [
                    nn.Conv1d(channels, self.hidden, kernel, dilation=dilation, padding=padding),
                    norm_fn(self.hidden),
                ]
            )
            act_layer = act(freq_dim) if act_func == "snake" else act()
            conv2 = nn.ModuleList(
                [
                    nn.Conv1d(self.hidden, 2 * channels, 1),
                    norm_fn(2 * channels),
                    nn.GLU(1),
                    LayerScale(channels, init),
                ]
            )

            layer.update(
                {"conv1": nn.Sequential(*conv1), "act": act_layer, "conv2": nn.Sequential(*conv2)}
            )
            if lstm:
                layer.update({"lstm": BLSTM(self.hidden, layers=2, max_steps=200, skip=True)})
            if time_attn:
                layer.update({"time_attn": LocalState(self.hidden, heads=heads, ndecay=ndecay)})

            self.layers.append(layer)

    def forward(self, x):
        if self.reshape:
            B, C, Fr, T = x.shape
            x = x.permute(0, 2, 1, 3).reshape(-1, C, T)

        for layer in self.layers:
            skip = x

            x = layer["conv1"](x)

            if self.act_func == "snake" and self.reshape:
                x = x.view(B, Fr, self.hidden, T).permute(0, 2, 3, 1)
            x = layer["act"](x)
            if self.act_func == "snake" and self.reshape:
                x = x.permute(0, 3, 1, 2).reshape(-1, self.hidden, T)

            if self.lstm:
                x = layer["lstm"](x)
            if self.time_attn:
                x = layer["time_attn"](x)

            x = layer["conv2"](x)
            x = skip + x

        if self.reshape:
            x = x.view(B, Fr, C, T).permute(0, 2, 1, 3)

        return x


class ScaledEmbedding(nn.Module):
    """Boost learning rate for embeddings (with `scale`).
    Also, can make embeddings continuous with `smooth`.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 10.0, smooth=False):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        if smooth:
            weight = torch.cumsum(self.embedding.weight.data, dim=0)
            # when summing gaussian, overscale raises as sqrt(n), so we nornalize by that.
            weight = weight / torch.arange(1, num_embeddings + 1).to(weight).sqrt()[:, None]
            self.embedding.weight.data[:] = weight
        self.embedding.weight.data /= scale
        self.scale = scale

    @property
    def weight(self):
        return self.embedding.weight * self.scale

    def forward(self, x):
        out = self.embedding(x) * self.scale
        return out


class FTB(nn.Module):
    def __init__(self, input_dim=257, in_channel=9, r_channel=5):
        super().__init__()
        self.input_dim = input_dim
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, r_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(r_channel),
            nn.ReLU(),
        )

        self.conv1d = nn.Sequential(
            nn.Conv1d(r_channel * input_dim, in_channel, kernel_size=9, padding=4),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(),
        )
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """inputs should be [Batch, Ca, Dim, Time]"""
        # T-F attention
        conv1_out = self.conv1(inputs)
        B, C, D, T = conv1_out.size()
        reshape1_out = torch.reshape(conv1_out, [B, C * D, T])
        conv1d_out = self.conv1d(reshape1_out)
        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel, 1, T])

        # now is also [B,C,D,T]
        att_out = conv1d_out * inputs

        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)
        return outputs


# --- aero.py (verbatim) ---
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


class HEncLayer(nn.Module):
    def __init__(
        self,
        chin,
        chout,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        is_first=False,
        freq_attn=False,
        freq_dim=None,
        norm=True,
        context=0,
        dconv_kw={},
        pad=True,
        rewrite=True,
    ):
        """Encoder layer. This used both by the time and the frequency branch."""
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa: E731
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa: E731
        if stride == 1 and kernel_size % 2 == 0 and kernel_size > 1:
            kernel_size -= 1
        if pad:
            pad = (kernel_size - stride) // 2
        else:
            pad = 0
        klass = nn.Conv2d
        self.chin = chin
        self.chout = chout
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.freq_attn = freq_attn
        self.freq_dim = freq_dim
        self.norm = norm
        self.pad = pad
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            if pad != 0:
                pad = [pad, 0]
        else:
            kernel_size = [1, kernel_size]
            stride = [1, stride]
            if pad != 0:
                pad = [0, pad]

        self.is_first = is_first

        if is_first:
            self.pre_conv = nn.Conv2d(chin, chout, [1, 1])
            chin = chout

        if self.freq_attn:
            self.freq_attn_block = FTB(input_dim=freq_dim, in_channel=chin)

        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
        """`inject` is used to inject the result from the time branch into the frequency branch,
        when both have the same stride.
        """
        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))

        if self.is_first:
            x = self.pre_conv(x)

        if self.freq_attn:
            x = self.freq_attn_block(x)

        x = self.conv(x)

        x = F.gelu(self.norm1(x))
        if self.dconv:
            x = self.dconv(x)

        if self.rewrite:
            x = self.norm2(self.rewrite(x))
            x = F.glu(x, dim=1)

        return x


class HDecLayer(nn.Module):
    def __init__(
        self,
        chin,
        chout,
        last=False,
        kernel_size=8,
        stride=4,
        norm_groups=1,
        empty=False,
        freq=True,
        dconv=True,
        norm=True,
        context=1,
        dconv_kw={},
        pad=True,
        context_freq=True,
        rewrite=True,
    ):
        """Same as HEncLayer but for decoder. See `HEncLayer` for documentation."""
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa: E731
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa: E731
        if stride == 1 and kernel_size % 2 == 0 and kernel_size > 1:
            kernel_size -= 1
        if pad:
            pad = (kernel_size - stride) // 2
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        klass = nn.Conv2d
        klass_tr = nn.ConvTranspose2d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
        else:
            kernel_size = [1, kernel_size]
            stride = [1, stride]
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1, [0, context])
            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = torch.cat([x, skip], dim=1)

            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv:
                y = self.dconv(y)
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad : -self.pad, :]
        else:
            z = z[..., self.pad : self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)
        if not self.last:
            z = F.gelu(z)
        return z


class Aero(nn.Module):
    """Deep model for Audio Super Resolution."""

    @capture_init
    def __init__(
        self,
        # Channels
        in_channels=1,
        out_channels=1,
        audio_channels=2,
        channels=48,
        growth=2,
        # STFT
        nfft=512,
        hop_length=64,
        end_iters=0,
        cac=True,
        # Main structure
        rewrite=True,
        hybrid=False,
        hybrid_old=False,
        # Frequency branch
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        strides=[4, 4, 2, 2],
        context=1,
        context_enc=0,
        freq_ends=4,
        enc_freq_attn=4,
        # Normalization
        norm_starts=2,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=4,
        dconv_time_attn=2,
        dconv_lstm=2,
        dconv_init=1e-3,
        # Weight init
        rescale=0.1,
        # Metadata
        lr_sr=4000,
        hr_sr=16000,
        spec_upsample=True,
        act_func="snake",
        debug=False,
    ):
        super().__init__()
        self.cac = cac
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.audio_channels = audio_channels
        self.kernel_size = kernel_size
        self.context = context
        self.strides = strides
        self.depth = len(strides)
        self.channels = channels
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.spec_upsample = spec_upsample

        self.scale = hr_sr / lr_sr if self.spec_upsample else 1

        self.nfft = nfft
        self.hop_length = int(hop_length // self.scale)  # this is for the input signal
        self.win_length = int(self.nfft // self.scale)  # this is for the input signal
        self.end_iters = end_iters
        self.freq_emb = None
        self.hybrid = hybrid
        self.hybrid_old = hybrid_old
        self.debug = debug

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        chin_z = self.in_channels
        if self.cac:
            chin_z *= 2
        chout_z = channels
        freqs = nfft // 2

        for index in range(self.depth):
            freq_attn = index >= enc_freq_attn
            lstm = index >= dconv_lstm
            time_attn = index >= dconv_time_attn
            norm = index >= norm_starts
            freq = index <= freq_ends
            stri = strides[index]
            ker = kernel_size

            pad = True
            if freq and freqs < kernel_size:
                ker = freqs

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "lstm": lstm,
                    "time_attn": time_attn,
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "act_func": act_func,
                    "reshape": True,
                    "freq_dim": freqs // strides[index] if freq else freqs,
                },
            }

            kw_dec = dict(kw)

            enc = HEncLayer(
                chin_z,
                chout_z,
                dconv=dconv_mode & 1,
                context=context_enc,
                is_first=index == 0,
                freq_attn=freq_attn,
                freq_dim=freqs,
                **kw,
            )

            self.encoder.append(enc)
            if index == 0:
                chin = self.out_channels
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = HDecLayer(
                2 * chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec,
            )

            self.decoder.insert(0, dec)

            chin_z = chout_z
            chout_z = int(growth * chout_z)

            if freq:
                freqs //= strides[index]

            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

    def _spec(self, x, scale=False):
        if np.mod(x.shape[-1], self.hop_length):
            x = F.pad(x, (0, self.hop_length - np.mod(x.shape[-1], self.hop_length)))
        hl = self.hop_length
        nfft = self.nfft
        win_length = self.win_length

        if scale:
            hl = int(hl * self.scale)
            win_length = int(win_length * self.scale)

        z = spectro(x, nfft, hl, win_length=win_length)[..., :-1, :]
        return z

    def _ispec(self, z):
        hl = int(self.hop_length * self.scale)
        win_length = int(self.win_length * self.scale)
        z = F.pad(z, (0, 0, 0, 1))
        x = ispectro(z, hl, win_length=win_length)
        return x

    def _move_complex_to_channels_dim(self, z):
        B, C, Fr, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, Fr, T)
        return m

    def _convert_to_complex(self, x):
        """
        :param x: signal of shape [Batch, Channels, 2, Freq, TimeFrames]
        :return: complex signal of shape [Batch, Channels, Freq, TimeFrames]
        """
        out = x.permute(0, 1, 3, 4, 2)
        out = torch.view_as_complex(out.contiguous())
        return out

    def forward(self, mix, return_spec=False, return_lr_spec=False):
        x = mix
        length = x.shape[-1]

        z = self._spec(x)
        x = self._move_complex_to_channels_dim(z)

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        x = torch.zeros_like(x)
        # initialize everything to zero (signal will go through u-net skips).

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x = decode(x, skip, lengths.pop(-1))

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0

        x = x.view(B, self.out_channels, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        x_spec_complex = self._convert_to_complex(x)

        x = self._ispec(x_spec_complex)

        x = x[..., : int(length * self.scale)]

        if return_spec:
            if return_lr_spec:
                return x, x_spec_complex, z
            else:
                return x, x_spec_complex

        return x


# --- staging entry points ---
def build_aero():
    # Small config: fewer channels/depth than the released checkpoints
    # (channels=48, strides len 4) so the traced module is tiny while keeping the
    # real architecture (same layer types, same control flow) intact.
    model = Aero(
        in_channels=1,
        out_channels=1,
        channels=16,
        growth=2,
        nfft=64,
        hop_length=16,
        strides=[4, 2],
        kernel_size=8,
        dconv_depth=1,
        dconv_comp=2,
        dconv_time_attn=1,
        dconv_lstm=1,
        norm_starts=1,
        enc_freq_attn=1,
        rescale=0.1,
        lr_sr=4000,
        hr_sr=8000,
        act_func="snake",
    )
    model.eval()
    return model


def example_input_aero():
    # 1 second of mono low-resolution audio at 4kHz.
    torch.manual_seed(0)
    return torch.randn(1, 1, 4000)


MENAGERIE_ENTRIES = [
    (
        "AERO (audio super-resolution, spectral domain)",
        "build_aero",
        "example_input_aero",
        2023,
        MENAGERIE_ZOO,
    ),
]

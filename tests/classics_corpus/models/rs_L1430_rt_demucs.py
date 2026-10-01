# SOURCE: vendored from facebookresearch/denoiser @ main
# https://raw.githubusercontent.com/facebookresearch/denoiser/main/denoiser/demucs.py
# https://raw.githubusercontent.com/facebookresearch/denoiser/main/denoiser/resample.py
# https://raw.githubusercontent.com/facebookresearch/denoiser/main/denoiser/utils.py (capture_init only)
#
# "Real Time Speech Enhancement in the Waveform Domain" (Defossez, Synnaeve & Adi,
# Interspeech 2020) -- Facebook's real-time streaming Demucs speech-denoiser. `Demucs`
# is a causal (by default) waveform-domain U-Net: a stack of strided Conv1d
# encoder blocks (each Conv1d -> ReLU -> 1x1 Conv1d -> GLU) with matching
# ConvTranspose1d decoder blocks and additive skip connections, a `BLSTM` bottleneck,
# and learned sinc-based `upsample2`/`downsample2` resampling (from `resample.py`)
# applied before/after the encoder-decoder stack. Every layer, its channel-growth
# schedule, the `rescale_module` weight-init pass, and the encoder/decoder skip-add
# topology are transcribed verbatim from `demucs.py` and `resample.py`. The
# `DemucsStreamer` real-time streaming wrapper, the `test()` CLI benchmark harness,
# and the `fast_conv` low-latency conv path are dropped since they are inference/
# deployment plumbing around the same `Demucs.forward`, not part of the traced
# architecture; the plain (offline) `Demucs.forward` used here is architecturally
# identical to what `DemucsStreamer` calls incrementally. `capture_init` is inlined
# from `denoiser/utils.py` verbatim (it only records constructor args/kwargs on the
# instance; no architectural effect).

import functools
import math

import torch as th
from torch import nn
from torch.nn import functional as F


# ---- utils.py (capture_init only) ----
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


# ---- resample.py ----
def sinc(t):
    """sinc.

    :param t: the input tensor
    """
    return th.where(t == 0, th.tensor(1.0, device=t.device, dtype=t.dtype), th.sin(t) / t)


def kernel_upsample2(zeros=56):
    """kernel_upsample2."""
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape
    kernel = kernel_upsample2(zeros).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = th.stack([x, out], dim=-1)
    return y.view(*other, -1)


def kernel_downsample2(zeros=56):
    """kernel_downsample2."""
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time
    )
    return out.view(*other, -1).mul(0.5)


# ---- demucs.py ----
class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


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


class Demucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.

    """

    @capture_init
    def __init__(
        self,
        chin=1,
        chout=1,
        hidden=48,
        depth=5,
        kernel_size=8,
        stride=4,
        causal=True,
        resample=4,
        growth=2,
        max_hidden=10_000,
        normalize=True,
        glu=True,
        rescale=0.1,
        floor=1e-3,
        sample_rate=16_000,
    ):
        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1),
                activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1),
                activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride**self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., : x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x


def build_rt_demucs():
    th.manual_seed(0)
    # Small hidden/depth/resample to keep the trace fast; architecture (encoder/
    # decoder depth, GLU activations, BLSTM bottleneck, sinc resampling) unchanged.
    model = Demucs(
        chin=1,
        chout=1,
        hidden=8,
        depth=3,
        kernel_size=8,
        stride=4,
        causal=True,
        resample=4,
        growth=2,
        normalize=True,
        glu=True,
        rescale=0.1,
        sample_rate=16_000,
    )
    model.eval()
    return model


def example_input_rt_demucs():
    th.manual_seed(0)
    # Raw waveform, (batch, channels, time); a few thousand samples is enough to
    # clear valid_length() for depth=3/resample=4/kernel=8/stride=4.
    return th.randn(1, 1, 4000)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Real-Time Demucs (denoiser)",
        "build_rt_demucs",
        "example_input_rt_demucs",
        2020,
        MENAGERIE_ZOO,
    ),
]

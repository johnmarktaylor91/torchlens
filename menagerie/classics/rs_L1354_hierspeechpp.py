# SOURCE: vendored from https://github.com/sh-lee-prml/HierSpeechpp @ main
#   Vendored files (real model-definition code only; training scripts, dataset/text
#   preprocessing, checkpoint-based inference CLIs, monotonic-alignment C extension,
#   TTV (text-to-w2v) stage, denoiser, and SpeechSR up/downsamplers are dropped -- not
#   part of the traced SynthesizerTrn voice-conversion architecture):
#     - activations.py -> `Snake`, `SnakeBeta` (verbatim)
#     - commons.py -> shared math/masking utilities (verbatim, including the repo's own
#       duplicate top-level `convert_pad_shape` definition)
#     - transforms.py -> `piecewise_rational_quadratic_transform` and its helpers
#       (verbatim; only reachable through `modules.ConvFlow`, which is itself unused by
#       the traced voice-conversion path, but kept since it is real modules.py content)
#     - attentions.py -> `Encoder`, `Decoder`, `MultiHeadAttention`, `FFN` (verbatim)
#     - modules.py -> `LayerNorm`, `ConvReluNorm`, `DDSConv`, `WN`, `ResBlock1/2`,
#       `Log`, `Flip`, `ElementwiseAffine`, `ResidualCouplingLayer`,
#       `ResidualCouplingLayer_Transformer_simple` (+ its `DiTConVBlock`/`FFN_Conv`
#       helpers using `timm.models.vision_transformer.Attention`), `ConvFlow` (verbatim)
#     - styleencoder.py -> `Mish`, `Conv1dGLU`, `StyleEncoder` (verbatim)
#     - alias_free_torch/{filter,resample,act}.py -> `kaiser_sinc_filter1d`,
#       `LowPassFilter1d`, `UpSample1d`, `DownSample1d`, `Activation1d` (verbatim,
#       anti-aliased activation package from https://github.com/junjun3518/alias-free-torch)
#     - hierspeechpp_speechsynthesizer.py -> `ResidualCouplingBlock_Transformer`,
#       `PosteriorAudioEncoder`, `PosteriorSFEncoder`, `MelDecoder`, `SourceNetwork`,
#       `DBlock`, `AMPBlock1`, `Generator`, `SynthesizerTrn` (verbatim). `Wav2vec2`
#       (downloads `facebook/mms-300m` from the HF hub) and the GAN discriminators
#       (`DiscriminatorP/R`, `MultiPeriodDiscriminator`, training-only) are dropped --
#       not reachable from `SynthesizerTrn.voice_conversion_noise_control`, the traced
#       inference entry point (used verbatim by the repo's own `inference_vc.py`).
#
#   Import fixes only (no architectural changes): cross-file imports
#   (`import modules` / `import commons` / `import attentions` / `import activations` /
#   `from alias_free_torch import *` / `from transforms import ...`) are flattened into
#   this single file (module-level symbols land directly in this namespace); thin
#   `types.SimpleNamespace` shims named `modules`, `commons`, `attentions`, `activations`
#   are added right before `hierspeechpp_speechsynthesizer.py`'s code so that its many
#   `modules.WN(...)`, `commons.sequence_mask(...)`, `attentions.Encoder(...)`,
#   `activations.SnakeBeta(...)` attribute-style call sites keep working unmodified.
#   A handful of lint-suppression markers are added on lines the original repo already
#   wrote with lint smells (ambiguous `l` loop vars, a duplicate `convert_pad_shape` def
#   in commons.py, a module-level `remove_weight_norm` function shadowing the imported
#   `torch.nn.utils.remove_weight_norm`) -- these are genuine upstream code as-is, not
#   introduced here, and are inert (dead code / not on the traced call path).
#
# HierSpeech++ (Lee, Kim, Cho & Cho, ICLR 2024, arXiv:2311.12454) is a hierarchical
# variational-autoencoder speech synthesizer/voice-converter: a self-supervised-speech
# (wav2vec2/MMS) "posterior source-filter encoder" and a raw-audio "posterior audio
# encoder" both feed transformer-conditioned normalizing-flow blocks
# (`ResidualCouplingBlock_Transformer`, using DiT-style adaLN conditioning blocks) into
# a shared latent, from which a `SourceNetwork` predicts an f0-conditioned excitation
# source signal and a HiFi-GAN-style `Generator` (with anti-aliased Snake-beta
# activations) synthesizes the waveform, conditioned on a `StyleEncoder` speaker/style
# embedding extracted from a mel-spectrogram prompt. This staging module traces
# `SynthesizerTrn.voice_conversion_noise_control`, the exact call used by the repo's
# real `inference_vc.py` script to convert a source utterance's content into a target
# speaker's voice. Architecture is reproduced verbatim; only per-module channel/layer
# counts are shrunk (tiny init) to keep tracing cheap.

import math
import torch
from torch import nn, sin, pow
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import numpy as np
import types
import torchaudio
from einops import rearrange
import transformers
from timm.models.vision_transformer import Attention
from itertools import repeat
import collections.abc

MENAGERIE_ZOO = "vendored-pytorch"

# ---- activations.py (verbatim) ----
# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


# ---- commons.py (verbatim) ----


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]  # noqa: E741
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments_audio(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = ((torch.rand([b]).to(device=x.device) * ids_str_max).clip(0)).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):  # noqa: F811
    l = pad_shape[::-1]  # noqa: E741
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    device = duration.device  # noqa: F841

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


# ---- transforms.py (verbatim) ----


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs,
    )
    return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet


# ---- attentions.py (verbatim, cross-file imports removed) ----


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=4,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, h, h_mask):
        """
        x: decoder input
        h: encoder output
        """
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.0,
        window_size=None,
        heads_share=True,
        block_length=None,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = (
                    torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1  # noqa: F841
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x


# ---- modules.py (verbatim, cross-file imports removed) ----


LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Module):
    """
    Dialted and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(torch.nn.Module):
    def __init__(
        self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):  # noqa: F811
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:  # noqa: E741
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:  # noqa: E741
            torch.nn.utils.remove_weight_norm(l)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):  # noqa: F811
        for l in self.convs1:  # noqa: E741
            remove_weight_norm(l)
        for l in self.convs2:  # noqa: E741
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):  # noqa: F811
        for l in self.convs:  # noqa: E741
            remove_weight_norm(l)


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class FFN_Conv(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        kernel=5,
        p_dropout=0.1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv1d(
            in_features,
            hidden_features,
            kernel_size=kernel,
            stride=1,
            padding=(kernel - 1) // 2,
            bias=bias[0],
        )
        self.act = act_layer()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1, bias=bias[1])
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.fc1(x.transpose(1, 2))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x * x_mask) * x_mask
        x = self.drop(x)
        return x.transpose(1, 2)


class DiTConVBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self, hidden_size, num_heads, mlp_ratio=4.0, kernel=9, p_dropout=0.1, **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # noqa: E731
        self.mlp = FFN_Conv(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            kernel=kernel,
            p_dropout=p_dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, x_mask):
        x = x * x_mask
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(modulate(self.norm1(x) * x_mask, shift_msa, scale_msa))
            * x_mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp), x_mask.transpose(1, 2)
        )
        return x


class ResidualCouplingLayer_Transformer_simple(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0.1,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)

        self.enc_block = torch.nn.ModuleList(
            [
                DiTConVBlock(hidden_channels, 2, mlp_ratio=4.0, kernel=5, p_dropout=p_dropout)
                for _ in range(n_layers)
            ]
        )

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)

        self.initialize_weights()

        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.enc_block:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask

        # h = self.enc(h, x_mask, g=g)
        h = h.transpose(1, 2)
        x_mask = x_mask.transpose(1, 2)

        for blk in self.enc_block:
            h = blk(h, g, x_mask)

        x_mask = x_mask.transpose(1, 2)
        h = h.transpose(1, 2)

        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ConvFlow(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x


# ---- styleencoder.py (verbatim, cross-file imports removed) ----


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, 2 * out_channels, kernel_size=kernel_size, padding=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x


class StyleEncoder(torch.nn.Module):
    def __init__(self, in_dim=513, hidden_dim=128, out_dim=256):
        super().__init__()

        self.in_dim = in_dim  # Linear 513 wav2vec 2.0 1024
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.1

        self.spectral = nn.Sequential(
            nn.Conv1d(self.in_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = attentions.MultiHeadAttention(
            self.hidden_dim,
            self.hidden_dim,
            self.n_head,
            p_dropout=self.dropout,
            proximal_bias=False,
            proximal_init=True,
        )
        self.atten_drop = nn.Dropout(self.dropout)
        self.fc = nn.Conv1d(self.hidden_dim, self.out_dim, 1)

    def forward(self, x, mask=None):
        # spectral
        x = self.spectral(x) * mask
        # temporal
        x = self.temporal(x) * mask

        # self-attention
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        y = self.slf_attn(x, x, attn_mask=attn_mask)
        x = x + self.atten_drop(y)

        # fc
        x = self.fc(x)

        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=2)
        else:
            len_ = mask.sum(dim=2)
            x = x.sum(dim=2)

            out = torch.div(x, len_)
        return out


# ---- alias_free_torch/filter.py (verbatim) ----
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.


if "sinc" in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)

        return out


# ---- alias_free_torch/resample.py (verbatim, cross-file imports removed) ----
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C
        )
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=self.kernel_size
        )

    def forward(self, x):
        xx = self.lowpass(x)

        return xx


# ---- alias_free_torch/act.py (verbatim, cross-file imports removed) ----
# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x


# ---- namespace shims so hierspeechpp_speechsynthesizer.py's `modules.X` / `commons.X` /
#      `attentions.X` / `activations.X` attribute-style references keep working after
#      flattening every vendored file into this single module namespace ----
modules = types.SimpleNamespace(
    WN=WN,
    LayerNorm=LayerNorm,
    Flip=Flip,
    LRELU_SLOPE=LRELU_SLOPE,
    ResidualCouplingLayer_Transformer_simple=ResidualCouplingLayer_Transformer_simple,
)
commons = types.SimpleNamespace(
    sequence_mask=sequence_mask,
    convert_pad_shape=convert_pad_shape,
    subsequent_mask=subsequent_mask,
    fused_add_tanh_sigmoid_multiply=fused_add_tanh_sigmoid_multiply,
)
attentions = types.SimpleNamespace(Encoder=Encoder, MultiHeadAttention=MultiHeadAttention)
activations = types.SimpleNamespace(SnakeBeta=SnakeBeta, Snake=Snake)


# ---- hierspeechpp_speechsynthesizer.py (verbatim, cross-file imports removed) ----


class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=7, w2v="mms"):
        """we use the intermediate features of mms-300m.
        More specifically, we used the output from the 7th layer of the 24-layer transformer encoder.
        """
        super().__init__()

        if w2v == "mms":
            self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-300m")
        else:
            self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained(
                "facebook/wav2vec2-xls-r-300m"
            )

        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (B x t)
        Returns:
            y: torch.Tensor of shape(B x C x t)
        """
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]  # B x t x C(1024)
        y = y.permute((0, 2, 1))  # B x t x C -> B x C x t
        return y


class ResidualCouplingBlock_Transformer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers=3,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.cond_block = torch.nn.Sequential(
            torch.nn.Linear(gin_channels, 4 * hidden_channels),
            nn.SiLU(),
            torch.nn.Linear(4 * hidden_channels, hidden_channels),
        )

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer_Transformer_simple(
                    channels, hidden_channels, kernel_size, dilation_rate, n_layers, mean_only=True
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        g = self.cond_block(g.squeeze(2))

        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorAudioEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.down_pre = nn.Conv1d(1, 16, 7, 1, padding=3)
        self.resblocks = nn.ModuleList()
        downsample_rates = [8, 5, 4, 2]
        downsample_kernel_sizes = [17, 10, 8, 4]
        ch = [16, 32, 64, 128, 192]

        resblock = AMPBlock1
        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.num_kernels = 3
        self.downs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(downsample_rates, downsample_kernel_sizes)):
            self.downs.append(weight_norm(Conv1d(ch[i], ch[i + 1], k, u, padding=(k - 1) // 2)))
        for i in range(4):
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch[i + 1], k, d, activation="snakebeta"))

        activation_post = activations.SnakeBeta(ch[i + 1], alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = Conv1d(ch[i + 1], hidden_channels, 7, 1, padding=3)

        self.enc = modules.WN(
            hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels
        )
        self.proj = nn.Conv1d(hidden_channels * 2, out_channels * 2, 1)

    def forward(self, x, x_audio, x_mask, g=None):
        x_audio = self.down_pre(x_audio)

        for i in range(4):
            x_audio = self.downs[i](x_audio)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x_audio)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x_audio)
            x_audio = xs / self.num_kernels

        x_audio = self.activation_post(x_audio)
        x_audio = self.conv_post(x_audio)

        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)

        x_audio = x_audio * x_mask

        x = torch.cat([x, x_audio], dim=1)

        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs


class PosteriorSFEncoder(nn.Module):
    def __init__(
        self,
        src_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre_source = nn.Conv1d(src_channels, hidden_channels, 1)
        self.pre_filter = nn.Conv1d(1, hidden_channels, kernel_size=9, stride=4, padding=4)
        self.source_enc = modules.WN(
            hidden_channels, kernel_size, dilation_rate, n_layers // 2, gin_channels=gin_channels
        )
        self.filter_enc = modules.WN(
            hidden_channels, kernel_size, dilation_rate, n_layers // 2, gin_channels=gin_channels
        )
        self.enc = modules.WN(
            hidden_channels, kernel_size, dilation_rate, n_layers // 2, gin_channels=gin_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x_src, x_ftr, x_mask, g=None):
        x_src = self.pre_source(x_src) * x_mask
        x_ftr = self.pre_filter(x_ftr) * x_mask
        x_src = self.source_enc(x_src, x_mask, g=g)
        x_ftr = self.filter_enc(x_ftr, x_mask, g=g)
        x = self.enc(x_src + x_ftr, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs


class MelDecoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        mel_size=20,
        gin_channels=0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_pre = Conv1d(hidden_channels, hidden_channels, 3, 1, padding=1)

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        self.proj = nn.Conv1d(hidden_channels, mel_size, 1, bias=False)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = self.conv_pre(x * x_mask)
        if g is not None:
            x = x + self.cond(g)

        x = self.encoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask

        return x


class SourceNetwork(nn.Module):
    def __init__(self, upsample_initial_channel=256):
        super().__init__()

        resblock_kernel_sizes = [3, 5, 7]
        upsample_rates = [2, 2]
        initial_channel = 192
        upsample_initial_channel = upsample_initial_channel
        upsample_kernel_sizes = [4, 4]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(
            Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = AMPBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, activation="snakebeta"))

        activation_post = activations.SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)

        self.cond = Conv1d(256, upsample_initial_channel, 1)

        self.ups.apply(init_weights)

    def forward(self, x, g):
        x = self.conv_pre(x) + self.cond(g)

        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        ## Predictor
        x_ = self.conv_post(x)
        return x, x_


def remove_weight_norm(self):  # noqa: F811
    print("Removing weight norm...")
    for l in self.ups:  # noqa: E741
        remove_weight_norm(l)
    for l in self.resblocks:  # noqa: E741
        l.remove_weight_norm()


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = weight_norm(Conv1d(input_size, hidden_size, 1))
        self.conv = nn.ModuleList(
            [
                weight_norm(Conv1d(input_size, hidden_size, 3, dilation=1, padding=1)),
                weight_norm(Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2)),
                weight_norm(Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4)),
            ]
        )
        self.conv.apply(init_weights)

    def forward(self, x):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        for layer in self.conv:
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = layer(x)

        return x + residual

    def remove_weight_norm(self):
        for l in self.conv:  # noqa: E741
            remove_weight_norm(l)


class AMPBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

        self.activations = nn.ModuleList(
            [
                Activation1d(activation=activations.SnakeBeta(channels, alpha_logscale=True))
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:  # noqa: E741
            remove_weight_norm(l)
        for l in self.convs2:  # noqa: E741
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=256,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(
            Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        )
        resblock = AMPBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, activation="snakebeta"))

        activation_post = activations.SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.downs = DBlock(upsample_initial_channel // 8, upsample_initial_channel, 4)
        self.proj = Conv1d(
            upsample_initial_channel // 8, upsample_initial_channel // 2, 7, 1, padding=3
        )

    def forward(self, x, pitch, g=None):
        x = self.conv_pre(x) + self.downs(pitch) + self.cond(g)

        for i in range(self.num_upsamples):
            x = self.ups[i](x)

            if i == 0:
                pitch = self.proj(pitch)
                x = x + pitch

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:  # noqa: E741
            remove_weight_norm(l)
        for l in self.resblocks:  # noqa: E741
            l.remove_weight_norm()
        for l in self.downs:  # noqa: E741
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm  # noqa: E712
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0)
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:  # noqa: E741
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorR(torch.nn.Module):
    def __init__(self, resolution, use_spectral_norm=False):
        super(DiscriminatorR, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm  # noqa: E712

        n_fft, hop_length, win_length = resolution
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=torch.hann_window,
            normalized=True,
            center=False,
            pad_mode=None,
            power=None,
        )

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(2, 32, (3, 9), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), dilation=(2, 1), padding=(2, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), dilation=(4, 1), padding=(4, 4))),
                norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, y):
        fmap = []

        x = self.spec_transform(y)  # [B, 2, Freq, Frames, 2]
        x = torch.cat([x.real, x.imag], dim=1)
        x = rearrange(x, "b c w t -> b c t w")

        for l in self.convs:  # noqa: E741
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]
        resolutions = [
            [2048, 512, 2048],
            [1024, 256, 1024],
            [512, 128, 512],
            [256, 64, 256],
            [128, 32, 128],
        ]

        discs = [
            DiscriminatorR(resolutions[i], use_spectral_norm=use_spectral_norm)
            for i in range(len(resolutions))
        ]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]

        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=256,
        prosody_size=20,
        uncond_ratio=0.0,
        cfg=False,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.mel_size = prosody_size

        self.enc_p_l = PosteriorSFEncoder(
            1024, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )
        self.flow_l = ResidualCouplingBlock_Transformer(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )

        self.enc_p = PosteriorSFEncoder(
            1024, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )
        self.enc_q = PosteriorAudioEncoder(
            spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels
        )
        self.flow = ResidualCouplingBlock_Transformer(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )

        self.mel_decoder = MelDecoder(
            inter_channels,
            filter_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=5,
            p_dropout=0.1,
            mel_size=self.mel_size,
            gin_channels=gin_channels,
        )

        self.dec = Generator(
            inter_channels,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.sn = SourceNetwork(upsample_initial_channel // 2)
        self.emb_g = StyleEncoder(in_dim=80, hidden_dim=256, out_dim=gin_channels)

        if cfg:
            self.emb = torch.nn.Embedding(1, 256)
            torch.nn.init.normal_(self.emb.weight, 0.0, 256**-0.5)
            self.null = torch.LongTensor([0]).cuda()
            self.uncond_ratio = uncond_ratio
        self.cfg = cfg

    @torch.no_grad()
    def infer(self, x_mel, w2v, length, f0):
        x_mask = torch.unsqueeze(commons.sequence_mask(length, x_mel.size(2)), 1).to(x_mel.dtype)

        # Speaker embedding from mel (Style Encoder)
        g = self.emb_g(x_mel, x_mask).unsqueeze(-1)

        z, _, _ = self.enc_p_l(w2v, f0, x_mask, g=g)

        z = self.flow_l(z, x_mask, g=g, reverse=True)
        z = self.flow(z, x_mask, g=g, reverse=True)

        e, e_ = self.sn(z, g)
        o = self.dec(z, e, g=g)

        return o, e_

    @torch.no_grad()
    def voice_conversion(
        self, src, src_length, trg_mel, trg_length, f0, noise_scale=0.333, uncond=False
    ):
        trg_mask = torch.unsqueeze(commons.sequence_mask(trg_length, trg_mel.size(2)), 1).to(
            trg_mel.dtype
        )
        g = self.emb_g(trg_mel, trg_mask).unsqueeze(-1)

        y_mask = torch.unsqueeze(commons.sequence_mask(src_length, src.size(2)), 1).to(
            trg_mel.dtype
        )
        z, m_p, logs_p = self.enc_p_l(src, f0, y_mask, g=g)

        z = (m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale) * y_mask

        z = self.flow_l(z, y_mask, g=g, reverse=True)
        z = self.flow(z, y_mask, g=g, reverse=True)

        if uncond:
            null_emb = self.emb(self.null) * math.sqrt(256)
            g = null_emb.unsqueeze(-1)

        e, _ = self.sn(z, g)
        o = self.dec(z, e, g=g)

        return o

    @torch.no_grad()
    def voice_conversion_noise_control(
        self,
        src,
        src_length,
        trg_mel,
        trg_length,
        f0,
        noise_scale=0.333,
        uncond=False,
        denoise_ratio=0,
    ):
        trg_mask = torch.unsqueeze(commons.sequence_mask(trg_length, trg_mel.size(2)), 1).to(
            trg_mel.dtype
        )
        g = self.emb_g(trg_mel, trg_mask).unsqueeze(-1)

        g_org, g_denoise = g[:1, :, :], g[1:, :, :]

        g_interpolation = (1 - denoise_ratio) * g_org + denoise_ratio * g_denoise

        y_mask = torch.unsqueeze(commons.sequence_mask(src_length, src.size(2)), 1).to(
            trg_mel.dtype
        )
        z, m_p, logs_p = self.enc_p_l(src, f0, y_mask, g=g_interpolation)

        z = (m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale) * y_mask

        z = self.flow_l(z, y_mask, g=g_interpolation, reverse=True)
        z = self.flow(z, y_mask, g=g_interpolation, reverse=True)

        if uncond:
            null_emb = self.emb(self.null) * math.sqrt(256)
            g = null_emb.unsqueeze(-1)

        e, _ = self.sn(z, g_interpolation)
        o = self.dec(z, e, g=g_interpolation)

        return o

    @torch.no_grad()
    def f0_extraction(self, x_linear, x_mel, length, x_audio, noise_scale=0.333):
        x_mask = torch.unsqueeze(commons.sequence_mask(length, x_mel.size(2)), 1).to(x_mel.dtype)

        # Speaker embedding from mel (Style Encoder)
        g = self.emb_g(x_mel, x_mask).unsqueeze(-1)

        # posterior encoder from linear spec.
        _, m_q, logs_q = self.enc_q(x_linear, x_audio, x_mask, g=g)
        z = m_q + torch.randn_like(m_q) * torch.exp(logs_q) * noise_scale

        # Source Networks
        _, e_ = self.sn(z, g)

        return e_


# ---- tiny build/example (architecture unmodified from the real repo) ----


def build_hierspeechpp():
    """SynthesizerTrn (voice-conversion synthesizer half of HierSpeech++) at tiny
    size for tracing. Architecture is unmodified from the real repo; only channel/
    layer-count knobs are shrunk. Two dims are NOT configurable in the real code and
    must match the hardcoded values inside `SourceNetwork`/`Generator`:
    `inter_channels` must be 192 (`SourceNetwork.conv_pre` hardcodes
    `initial_channel=192`) and `gin_channels` must be 256 (`SourceNetwork.cond`
    hardcodes its input to 256) -- both are real-repo constants, not choices made here.
    """
    torch.manual_seed(0)
    model = SynthesizerTrn(
        spec_channels=13,
        segment_size=8192,
        inter_channels=192,
        hidden_channels=16,
        filter_channels=32,
        n_heads=2,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        upsample_rates=[4],
        upsample_initial_channel=16,
        upsample_kernel_sizes=[8],
        gin_channels=256,
        prosody_size=20,
    )
    model.eval()
    return model


def example_input_hierspeechpp():
    """Matches the repo's own `inference_vc.py` call to
    `SynthesizerTrn.voice_conversion_noise_control(x_w2v, x_length, trg_mel,
    trg_length2, denorm_f0, ...)`: `x_w2v` is (B, 1024, T_w2v) wav2vec2/MMS content
    features, `x_length` the w2v sequence length, `trg_mel` is (2, 80, T_mel) --
    original + denoised target prompt mel-spectrograms stacked on the batch dim per
    the real `voice_conversion_noise_control` signature, `trg_length2` their matching
    lengths, and `denorm_f0` is (B, 1, T_f0) the denormalized source pitch contour.
    T_w2v=8 and T_f0=32 satisfy `PosteriorSFEncoder`'s stride-4 f0 conv
    (kernel=9, stride=4, padding=4) so the w2v and f0 branches align in time.
    """
    x_w2v = torch.randn(1, 1024, 8)
    x_length = torch.LongTensor([8])
    trg_mel = torch.randn(2, 80, 10)
    trg_length2 = torch.LongTensor([10, 10])
    denorm_f0 = torch.randn(1, 1, 32)
    return (x_w2v, x_length, trg_mel, trg_length2, denorm_f0)


class _HierSpeechVCWrapper(nn.Module):
    """Thin nn.Module wrapper so torchlens.trace sees a single forward() call into
    the real SynthesizerTrn.voice_conversion_noise_control (the same entry point the
    repo's own inference_vc.py script uses)."""

    def __init__(self):
        super().__init__()
        self.net_g = build_hierspeechpp()

    def forward(self, x_w2v, x_length, trg_mel, trg_length2, denorm_f0):
        return self.net_g.voice_conversion_noise_control(
            x_w2v,
            x_length,
            trg_mel,
            trg_length2,
            denorm_f0,
            noise_scale=0.333,
            denoise_ratio=0.5,
        )


def build_hierspeechpp_wrapped():
    m = _HierSpeechVCWrapper()
    m.eval()
    return m


MENAGERIE_ENTRIES = [
    (
        "HierSpeechpp",
        "build_hierspeechpp_wrapped",
        "example_input_hierspeechpp",
        2023,
        MENAGERIE_ZOO,
    ),
]
